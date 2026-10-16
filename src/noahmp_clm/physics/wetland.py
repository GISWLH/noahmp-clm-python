"""Conservative bulk wetland methane: carbon partition, oxidation and export.

This reduced single-column model uses one persistent CH4-C pool. The donor
is actual decomposed organic carbon, already removed from soil C/N stores;
it must not also be emitted as CO2. Constant-environment linear oxidation
and diffusion are integrated exactly with uniform production over the step.
An end-step pressure threshold releases excess gas as ebullition. See
docs/CLM_WETLAND.md for the explicitly reduced chemistry/transport scope.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class WetlandParameters:
    active_depth_m: float=.3
    anaerobic_saturation: float=.8
    methane_co2_ratio: float=.2
    methane_q10: float=2.
    donor_q10: float=1.5
    production_reference_k: float=295.15
    henry_water_air: float=.035
    aqueous_diffusivity_m2s: float=2e-9
    gaseous_diffusivity_m2s: float=2e-5
    oxidation_rate_s: float=1e-5
    oxidation_q10: float=2.
    oxidation_reference_k: float=285.15
    ebullition_pressure_fraction: float=.15

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in vars(self).values()) or self.active_depth_m<=0.
            or not 0.<=self.anaerobic_saturation<1. or not 0.<=self.methane_co2_ratio<=1.
            or not all(1.<=v<=5. for v in (self.methane_q10,self.donor_q10,self.oxidation_q10))
            or not all(250.<=v<=330. for v in (self.production_reference_k,self.oxidation_reference_k))
            or self.henry_water_air<=0. or min(self.aqueous_diffusivity_m2s,self.gaseous_diffusivity_m2s,self.oxidation_rate_s)<0.
            or not 0.<self.ebullition_pressure_fraction<=1.):
            raise ValueError('Invalid bulk wetland parameters')


@dataclass(frozen=True)
class WetlandState:
    methane_gc_m2: float=0.

    def __post_init__(self):
        if not math.isfinite(self.methane_gc_m2) or self.methane_gc_m2<0.:
            raise ValueError('Invalid stored methane carbon')


def wetland_rates(temperature,liquid_theta,ice_theta,porosity,pressure,parameters=WetlandParameters()):
    """Bulk phase capacity and rates; changing volume never deletes CH4 mass.

    Henry's coefficient is aqueous/gas concentration, dimensionless. Aqueous
    and gaseous pathways act in parallel, with a zero-CH4 atmospheric boundary
    and half-column diffusion distance. Ice occupies pore space. The linear
    oxidation law represents an explicit reduced kinetics assumption.
    """
    if (not all(math.isfinite(v) for v in (temperature,liquid_theta,ice_theta,porosity,pressure))
        or not 150.<=temperature<=350. or min(liquid_theta,ice_theta)<0. or not 0.<porosity<=1.
        or liquid_theta+ice_theta>porosity+8.*np.finfo(float).eps or not 20000.<=pressure<=120000.):
        raise ValueError('Invalid wetland soil phases, temperature or pressure')
    p=parameters;air=max(porosity-liquid_theta-ice_theta,0.)
    saturation=min(liquid_theta/porosity,1.)
    anaerobic=max(0.,(saturation-p.anaerobic_saturation)/(1.-p.anaerobic_saturation))
    # RH already has its donor Q10. Only the relative methane response is added.
    ratio=p.methane_co2_ratio*(p.methane_q10/p.donor_q10)**((temperature-p.production_reference_k)/10.)
    production_fraction=anaerobic*ratio/(1.+ratio) if temperature>273.15 else 0.
    capacity=(liquid_theta+air/p.henry_water_air)*p.active_depth_m
    aqueous=p.aqueous_diffusivity_m2s*liquid_theta**2
    gaseous=p.gaseous_diffusivity_m2s*air**(10./3.)/porosity**2
    conductance=(aqueous+gaseous/p.henry_water_air)/(.5*p.active_depth_m)
    diffusion=conductance/capacity if capacity>0. else 0.
    oxidation=(p.oxidation_rate_s*(1.-anaerobic)*p.oxidation_q10**((temperature-p.oxidation_reference_k)/10.)
        if temperature>273.15 and liquid_theta>0. else 0.)
    # Ideal-gas pressure threshold, converted to dissolved gC/m3 using Henry.
    dissolved_limit=12.*p.henry_water_air*p.ebullition_pressure_fraction*pressure/(8.31446261815324*temperature)
    return dict(anaerobic_fraction=anaerobic,production_fraction=production_fraction,
        diffusion_rate_s=diffusion,oxidation_rate_s=oxidation,
        methane_capacity_gc_m2=dissolved_limit*capacity,
        ebullition_active=bool(temperature>273.15 and anaerobic>0. and liquid_theta>0.))


def step_wetland(state,respired_carbon_gc_m2,temperature,liquid_theta,ice_theta,porosity,pressure,dt,*,parameters=WetlandParameters()):
    """Partition an already funded decomposition C loss into CO2/CH4/storage."""
    if (not isinstance(state,WetlandState) or not isinstance(parameters,WetlandParameters)
        or not math.isfinite(respired_carbon_gc_m2) or respired_carbon_gc_m2<0.
        or not math.isfinite(dt) or not 0.<dt<=86400.):
        raise ValueError('Invalid wetland state, carbon donor or time step')
    rates=wetland_rates(temperature,liquid_theta,ice_theta,porosity,pressure,parameters)
    produced=respired_carbon_gc_m2*rates['production_fraction']
    ko=rates['oxidation_rate_s'];kd=rates['diffusion_rate_s'];k=ko+kd;x=k*dt
    survival=math.exp(-x);lost_old=-math.expm1(-x)
    if x<1e-4:
        lost_new=x*(.5+x*(-1./6.+x*(1./24.-x/120.)))
    else:lost_new=1.-lost_old/x
    end=state.methane_gc_m2*survival+produced*(1.-lost_new)
    loss=state.methane_gc_m2*lost_old+produced*lost_new
    oxidized=loss*ko/k if k>0. else 0.;diffused=loss-oxidized
    ebullition=max(end-rates['methane_capacity_gc_m2'],0.) if rates['ebullition_active'] else 0.
    end-=ebullition;emitted=diffused+ebullition
    co2=respired_carbon_gc_m2-produced+oxidized
    residual=end-state.methane_gc_m2+co2+emitted-respired_carbon_gc_m2
    methane_residual=end-state.methane_gc_m2+oxidized+emitted-produced
    bound=128.*np.finfo(float).eps*max(end+state.methane_gc_m2+co2+emitted+respired_carbon_gc_m2,1.)
    if abs(residual)>bound or abs(methane_residual)>bound:raise RuntimeError('Wetland carbon balance exceeds numerical bound')
    return WetlandState(end),dict(**rates,methane_production_gc_m2=produced,oxidized_gc_m2=oxidized,
        diffused_gc_m2=diffused,ebullition_gc_m2=ebullition,co2_gc_m2=co2,
        methane_emission_gc_m2=emitted,methane_emission_mgch4_m2=emitted*(16000./12.),
        carbon_residual_gc_m2=residual,methane_residual_gc_m2=methane_residual,carbon_bound_gc_m2=bound)

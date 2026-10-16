"""Finite glacier ice stores, snow/ice conduction and conservative melt runoff.

This is a single-column simplification of the CLM glacier land unit: no CISM,
elevation classes, ice flow, or invisible replacement-ice reservoir. Ice and
snow share the same finite-volume heat flux at their interface. See
docs/CLM_GLACIER.md for the finite-reservoir convention and limitations.
"""
from dataclasses import dataclass,replace
import math
import numpy as np
from .snowpack import Snowpack,TF,CI,LF,RHO_ICE,drain_liquid,equilibrium
from .snow_heat import heat_step,conductivity as snow_conductivity


@dataclass(frozen=True)
class GlacierParameters:
    thermal_conductivity_wmk: float=2.29
    geothermal_wm2: float=.05
    ice_albedo: float=.55
    emissivity: float=.97

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in vars(self).values())
            or self.thermal_conductivity_wmk<=0. or self.geothermal_wm2<0.
            or not 0.<=self.ice_albedo<=1. or not 0.<self.emissivity<=1.):
            raise ValueError('Invalid glacier thermal or optical parameters')


@dataclass(frozen=True)
class GlacierIce:
    """Up to four pure-ice layers after drainage; extensive kg/m2 and J/m2.

    Density is 917 kg/m3; geometry contracts/expands with actual ice mass.
    Meltwater is exported, so stored H is nonpositive relative to ice at Tf.
    The five impurity stores are conserved, without bare-ice optical feedback.
    An empty result is an exhausted finite glacier, not replenished ice.
    """
    mass: np.ndarray
    enthalpy: np.ndarray
    impurities: np.ndarray | None=None

    def __post_init__(self):
        m=np.array(self.mass,float,copy=True);h=np.array(self.enthalpy,float,copy=True);n=m.size
        a=np.zeros((n,5)) if self.impurities is None else np.array(self.impurities,float,copy=True)
        if (m.ndim!=1 or n>4 or h.shape!=(n,) or a.shape!=(n,5)
            or not all(np.all(np.isfinite(v)) for v in (m,h,a)) or np.any(m<=0.)
            or np.any(h>0.) or np.any(h<CI*m*(150.-TF)) or np.any(a<0.)):
            raise ValueError('Glacier ice requires zero to four finite cold-ice layers and five impurity stores')
        for name,value in (('mass',m),('enthalpy',h),('impurities',a)):
            value.setflags(write=False);object.__setattr__(self,name,value)

    @classmethod
    def empty(cls):return cls(np.empty(0),np.empty(0))

    @classmethod
    def from_thickness(cls,thickness,temperature=263.15):
        z,t=np.broadcast_arrays(np.atleast_1d(thickness).astype(float),temperature)
        if np.any(z<=0.) or np.any(t>TF) or np.any(t<150.):raise ValueError('Invalid initial glacier thickness or temperature')
        mass=RHO_ICE*z
        return cls(mass,mass*CI*(t-TF))

    @property
    def thickness(self):return self.mass/RHO_ICE

    @property
    def temperature(self):return TF+self.enthalpy/(CI*self.mass)


@dataclass(frozen=True)
class _ThermalStack:
    mass: np.ndarray
    enthalpy: np.ndarray
    thickness: np.ndarray

    def with_enthalpy(self,enthalpy):return replace(self,enthalpy=enthalpy)


def _ice_after_melt(mass,enthalpy,impurities):
    """Export liquid and its exact positive H; retain all cold-ice energy."""
    liquid=np.minimum(mass,np.maximum(enthalpy,0.)/LF)
    keep=mass>liquid;exported=impurities*(liquid/mass)[:,None]
    new=GlacierIce((mass-liquid)[keep],np.minimum(enthalpy,0.)[keep],(impurities-exported)[keep])
    return new,dict(runoff_kgm2=float(liquid.sum()),runoff_enthalpy_jm2=float(np.maximum(enthalpy,0.).sum()),
        runoff_impurities_kgm2=exported.sum(axis=0))


def add_glacier_surface_water(ice,mass_kgm2,enthalpy_jm2,impurities=None):
    """Mix an incoming liquid parcel at exposed ice; freeze or export it.

    Water cannot infiltrate solid ice. If the exposed layer melts entirely,
    its water/heat reach the next newly exposed layer. Any remaining liquid
    over surviving ice runs off, carrying positive enthalpy and impurities.
    """
    a=np.zeros(5) if impurities is None else np.asarray(impurities,float)
    if (not isinstance(ice,GlacierIce) or not math.isfinite(mass_kgm2) or not math.isfinite(enthalpy_jm2)
        or mass_kgm2<0. or enthalpy_jm2<LF*mass_kgm2 or (mass_kgm2==0. and enthalpy_jm2!=0.)
        or a.shape!=(5,) or not np.all(np.isfinite(a)) or np.any(a<0.)):
        raise ValueError('Invalid glacier surface liquid parcel')
    m=ice.mass.copy();h=ice.enthalpy.copy();tracer=ice.impurities.copy()
    incoming=float(mass_kgm2);energy=float(enthalpy_jm2);carried=a.copy();first=0
    while first<len(m) and (incoming>0. or np.any(carried>0.)):
        m[first]+=incoming;h[first]+=energy;tracer[first]+=carried
        new,out=_ice_after_melt(m[first:first+1],h[first:first+1],tracer[first:first+1])
        incoming=out['runoff_kgm2'];energy=out['runoff_enthalpy_jm2'];carried=out['runoff_impurities_kgm2']
        if len(new.mass):
            m[first]=new.mass[0];h[first]=new.enthalpy[0];tracer[first]=new.impurities[0]
            break
        first+=1
    final=GlacierIce(m[first:],h[first:],tracer[first:])
    return final,dict(runoff_kgm2=incoming,runoff_enthalpy_jm2=energy,runoff_impurities_kgm2=carried,
        net_ice_formation_kgm2=float(final.mass.sum()-ice.mass.sum()))


def glacier_heat_step(snow,ice,dt,top_flux,*,snow_absorbed=None,ice_absorbed=None,
                      top_temperature=None,top_conductance=0.,parameters=GlacierParameters()):
    """Joint implicit conduction, phase change, drainage and refreezing.

    The caller supplies the atmospheric boundary; this is not a meteorology
    driver. Geothermal heat is an explicit external bottom input. Snow pore
    storage and its two liquid outlets retain their existing definitions.
    Internal ice melt exits directly; snow bottom drainage contacts exposed
    ice and may refreeze. No through-ice porous flow is represented.
    """
    if not isinstance(snow,Snowpack) or not isinstance(ice,GlacierIce) or not len(ice.mass):
        raise ValueError('Glacier heat step requires snow state and non-exhausted glacier ice')
    ns=len(snow.mass);ni=len(ice.mass);p=parameters
    qs=np.zeros(ns) if snow_absorbed is None else np.asarray(snow_absorbed,float)
    qi=np.zeros(ni) if ice_absorbed is None else np.asarray(ice_absorbed,float)
    if qs.shape!=(ns,) or qi.shape!=(ni,):raise ValueError('Glacier absorbed radiation must match active snow/ice layers')
    old=_ThermalStack(np.r_[snow.mass,ice.mass],np.r_[snow.enthalpy,ice.enthalpy],np.r_[snow.thickness,ice.thickness])
    absorbed=np.r_[qs,qi];absorbed[-1]+=p.geothermal_wm2
    k=np.r_[snow_conductivity(snow),np.full(ni,p.thermal_conductivity_wmk)]
    thermal,td=heat_step(old,dt,top_flux,absorbed=absorbed,thermal_conductivity=k,
        top_temperature=top_temperature,top_conductance=top_conductance)
    heated_snow=snow.with_enthalpy(thermal.enthalpy[:ns])
    ice,im=_ice_after_melt(ice.mass,thermal.enthalpy[ns:],ice.impurities)
    snow,sd=drain_liquid(heated_snow,return_ledger=True)
    ice,refrozen=add_glacier_surface_water(ice,sd['drainage_kgm2'],sd['drainage_enthalpy_jm2'],sd['drainage_impurities_kgm2'])
    runoff=im['runoff_kgm2']+sd['runoff_kgm2']+refrozen['runoff_kgm2']
    heat=im['runoff_enthalpy_jm2']+sd['runoff_enthalpy_jm2']+refrozen['runoff_enthalpy_jm2']
    tracer=im['runoff_impurities_kgm2']+sd['runoff_impurities_kgm2']+refrozen['runoff_impurities_kgm2']
    water_res=float(snow.mass.sum()+ice.mass.sum()-old.mass.sum())+runoff
    heat_res=float(snow.enthalpy.sum()+ice.enthalpy.sum()-old.enthalpy.sum())+heat-td['input_energy_jm2']
    eps=np.finfo(float).eps;wm=64.*eps*max(float(old.mass.sum())+runoff,1.)
    wh=64.*eps*max(float(abs(old.enthalpy).sum())+abs(heat)+abs(td['input_energy_jm2']),1.)+len(old.mass)*td['convergence_tolerance_jm2']
    if abs(water_res)>wm or abs(heat_res)>wh:raise RuntimeError('Glacier water/heat ledger exceeds numerical bound')
    return snow,ice,dict(thermal=td,thermal_enthalpy_jm2=thermal.enthalpy.copy(),
        thermal_temperature_k=equilibrium(old.mass,thermal.enthalpy)[0],
        runoff_kgm2=runoff,runoff_enthalpy_jm2=heat,runoff_impurities_kgm2=tracer,
        ice_melt_kgm2=im['runoff_kgm2'],snow_drainage_kgm2=sd['drainage_kgm2'],
        net_refreezing_kgm2=refrozen['net_ice_formation_kgm2'],geothermal_input_jm2=p.geothermal_wm2*dt,
        water_residual_kgm2=water_res,energy_residual_jm2=heat_res,water_bound_kgm2=wm,energy_bound_jm2=wh)

"""Conservative layer condensate for CLM rain/snow interception and unloading.

CLM5 hydrology 2.7.2--25 supplies bulk interception, capacity and unloading
relations. Intercepted precipitation is distributed by exposed leaf+stem area;
each layer then retains its own equilibrium liquid/ice mass and enthalpy.
This is an original extensive formulation, not a translation of CTSM code.
The canopy energy/column drivers do not yet consume this new state.
"""
from dataclasses import dataclass
import numpy as np
from .snowpack import equilibrium,TF,CI,CW,LF

LV=2.5e6


@dataclass(frozen=True)
class CanopyCondensate:
    """Layer water-equivalent mass kg/m2 ground and enthalpy J/m2 ground.

    The heat origin is ice at freezing. Mixtures remain exactly at freezing
    while H varies through [0, LF*mass]. No supercooled-liquid-only shortcut.
    """
    mass: np.ndarray
    enthalpy: np.ndarray

    def __post_init__(self):
        mass=np.array(self.mass,float,copy=True);heat=np.array(self.enthalpy,float,copy=True)
        if (mass.ndim!=1 or not len(mass) or heat.shape!=mass.shape or
            not np.all(np.isfinite(mass)) or not np.all(np.isfinite(heat)) or
            np.any(mass<0.) or np.any((mass==0.)&(heat!=0.))):
            raise ValueError('Invalid canopy condensate mass/enthalpy')
        temperature=equilibrium(mass,heat)[0]
        if np.any((temperature<150.)|(temperature>400.)):
            raise ValueError('Canopy condensate outside 150--400 K')
        for name,value in (('mass',mass),('enthalpy',heat)):
            value.setflags(write=False);object.__setattr__(self,name,value)

    @classmethod
    def empty(cls,layers=10):
        return cls(np.zeros(layers),np.zeros(layers))

    @classmethod
    def from_temperature(cls,mass,temperature,*,liquid_fraction=None):
        m,t=np.broadcast_arrays(np.atleast_1d(mass).astype(float),temperature)
        fraction=np.where(t>TF,1.,0.) if liquid_fraction is None else np.broadcast_to(liquid_fraction,m.shape)
        if (not np.all(np.isfinite(t)) or not np.all(np.isfinite(fraction)) or
            np.any((fraction<0.)|(fraction>1.)) or np.any((t<TF)&(fraction!=0.)) or np.any((t>TF)&(fraction!=1.))):
            raise ValueError('Inconsistent canopy temperature and phase fraction')
        heat=m*((CI*(1.-fraction)+CW*fraction)*(t-TF)+LF*fraction)
        return cls(m,heat)

    @property
    def temperature(self):return equilibrium(self.mass,self.enthalpy)[0]
    @property
    def ice(self):return equilibrium(self.mass,self.enthalpy)[1]
    @property
    def liquid(self):return equilibrium(self.mass,self.enthalpy)[2]

    def with_enthalpy(self,heat):
        """Replace extensive H; the caller owns the complete input heat ledger."""
        return CanopyCondensate(self.mass,heat)


def _geometry(state,leaf_area,stem_area,vegetated_fraction):
    leaves=np.broadcast_to(leaf_area,state.mass.shape).astype(float)
    stems=np.broadcast_to(stem_area,state.mass.shape).astype(float)
    if (not np.all(np.isfinite(leaves)) or not np.all(np.isfinite(stems)) or
        np.any(leaves<0.) or np.any(stems<0.) or not np.isfinite(vegetated_fraction) or not 0.<=vegetated_fraction<=1.):
        raise ValueError('Invalid exposed canopy area or vegetated fraction')
    return leaves,stems,leaves+stems


def coverage(state,leaf_area,*,stem_area=0.,vegetated_fraction=1.):
    """Layer wet, transpiring-dry and snow-covered fractions, dimensionless."""
    leaves,stems,area=_geometry(state,leaf_area,stem_area,vegetated_fraction)
    cap=.1*vegetated_fraction*area;snowcap=6.*vegetated_fraction*area
    wet=np.where(cap>0.,np.minimum(1.,(state.mass/np.maximum(cap,1e-300))**(2./3.)),0.)
    dry=(1.-wet)*np.divide(leaves,area,out=np.zeros_like(area),where=area>0.)
    snow=np.where(snowcap>0.,np.minimum(1.,(state.ice/np.maximum(snowcap,1e-300))**.15),0.)
    return dict(wet_fraction=wet,dry_fraction=dry,snow_fraction=snow)


def intercept_precipitation(state,rain_kgm2,snow_kgm2,air_temperature,dt,leaf_area,*,
                            stem_area=0.,vegetated_fraction=1.,wind=0.,unloading_threshold=270.15):
    """Add precipitation, mix phases, shed capacity excess and unload snow.

    Rain/snow are whole-column step totals, not rates. Ground liquid/ice and
    their individual transported H are returned explicitly. Temperature and
    wind unloading are bounded by remaining ice divided by dt (the printed
    technical note omits /dt in that dimensional bound). 270.15 K follows
    CTSM's atmospheric-temperature threshold; the note rounds it to 270 K.
    Leaf/stem area is per vegetated area; all masses are per ground area.
    """
    scalars=[rain_kgm2,snow_kgm2,air_temperature,dt,wind,unloading_threshold]
    if (not np.all(np.isfinite(scalars)) or min(rain_kgm2,snow_kgm2,wind)<0. or dt<=0.
        or not 200.<=air_temperature<=350. or not 200.<=unloading_threshold<=TF):
        raise ValueError('Invalid canopy precipitation or unloading forcing')
    _,_,area=_geometry(state,leaf_area,stem_area,vegetated_fraction);total_area=float(area.sum())
    weights=area/total_area if total_area>0. else np.zeros_like(area)
    rain_fraction=vegetated_fraction*np.tanh(total_area)
    snow_fraction=vegetated_fraction*(-np.expm1(-.5*total_area))
    intercepted_rain=rain_fraction*rain_kgm2;intercepted_snow=snow_fraction*snow_kgm2
    rain_specific=LF+CW*(air_temperature-TF);snow_specific=CI*(min(air_temperature,TF)-TF)
    precipitation_heat=rain_kgm2*rain_specific+snow_kgm2*snow_specific
    added_mass=weights*(intercepted_rain+intercepted_snow)
    added_heat=weights*(intercepted_rain*rain_specific+intercepted_snow*snow_specific)
    mixed=CanopyCondensate(state.mass+added_mass,state.enthalpy+added_heat)
    ice,liquid=mixed.ice,mixed.liquid
    # In equilibrium, only the ice phase carries H below freezing; otherwise
    # only liquid carries H. This preserves the supplied H without a T->H
    # reconstruction and gives the correct phase-specific parcel heat.
    liquid_heat=np.where(mixed.enthalpy>0.,mixed.enthalpy,0.)
    ice_heat=mixed.enthalpy-liquid_heat
    capacity_liquid=.1*vegetated_fraction*area;capacity_ice=6.*vegetated_fraction*area
    kept_liquid=np.minimum(liquid,capacity_liquid);kept_ice=np.minimum(ice,capacity_ice)
    drip=liquid-kept_liquid;snowfall=ice-kept_ice
    rate=wind/1.56e5+max(0.,air_temperature-unloading_threshold)/1.87e5
    unload=kept_ice*min(1.,dt*rate)
    ice_out=snowfall+unload
    remaining_ice=kept_ice-unload
    # Compute the small retained parcel directly. Subtracting two large
    # nearly equal enthalpies amplified rounding in thin wet layers.
    retained_liquid_heat=np.divide(kept_liquid,liquid,out=np.zeros_like(liquid),where=liquid>0.)*liquid_heat
    retained_ice_heat=np.divide(remaining_ice,ice,out=np.zeros_like(ice),where=ice>0.)*ice_heat
    drip_heat=liquid_heat-retained_liquid_heat;snow_heat=ice_heat-retained_ice_heat
    mass=kept_liquid+remaining_ice;heat=retained_liquid_heat+retained_ice_heat
    final=CanopyCondensate(mass,heat)
    through_rain=rain_kgm2-intercepted_rain;through_snow=snow_kgm2-intercepted_snow
    ground_liquid=through_rain+float(drip.sum());ground_ice=through_snow+float(ice_out.sum())
    ground_liquid_heat=through_rain*rain_specific+float(drip_heat.sum())
    ground_ice_heat=through_snow*snow_specific+float(snow_heat.sum())
    ledger=dict(intercepted_rain_kgm2=intercepted_rain,intercepted_snow_kgm2=intercepted_snow,
        through_rain_kgm2=through_rain,through_snow_kgm2=through_snow,
        canopy_drip_kgm2=drip,canopy_snowfall_kgm2=snowfall,canopy_unloading_kgm2=unload,
        ground_liquid_kgm2=ground_liquid,ground_ice_kgm2=ground_ice,
        ground_liquid_enthalpy_jm2=ground_liquid_heat,ground_ice_enthalpy_jm2=ground_ice_heat,
        precipitation_enthalpy_jm2=precipitation_heat,
        water_residual_kgm2=float(final.mass.sum()-state.mass.sum()-rain_kgm2-snow_kgm2+ground_liquid+ground_ice),
        energy_residual_jm2=float(final.enthalpy.sum()-state.enthalpy.sum()-precipitation_heat+ground_liquid_heat+ground_ice_heat))
    ledger.update(coverage(final,leaf_area,stem_area=stem_area,vegetated_fraction=vegetated_fraction))
    return final,ledger


def exchange_vapor(state,amount_kgm2,*,leaf_temperature=None):
    """Layer outward vapour totals; negative amounts deposit frost/dew.

    CLM5 2.7.19--20 selects liquid above freezing, ice at/below freezing.
    Loss is limited by the selected phase, not by total canopy water. The
    caller can supply the iterated leaf temperature; the source parcel keeps
    the state phase's authoritative H. Transported condensed-phase H and
    latent energy are separate boundary ledger terms.
    """
    requested=np.broadcast_to(amount_kgm2,state.mass.shape).astype(float)
    temperature=state.temperature if leaf_temperature is None else np.broadcast_to(leaf_temperature,state.mass.shape).astype(float)
    if (not np.all(np.isfinite(requested)) or not np.all(np.isfinite(temperature)) or np.any((temperature<200.)|(temperature>350.))):
        raise ValueError('Invalid canopy vapour amount or deposition temperature')
    use_ice=temperature<=TF;available=np.where(use_ice,state.ice,state.liquid)
    loss=np.minimum(np.maximum(requested,0.),available)
    remaining=available-loss
    deposit=np.maximum(-requested,0.);deposit_ice=np.where(temperature<=TF,deposit,0.)
    deposit_liquid=deposit-deposit_ice
    out_ice=np.where(use_ice,loss,0.)-deposit_ice;out_liquid=np.where(use_ice,0.,loss)-deposit_liquid
    source_heat=np.where(use_ice,np.minimum(state.enthalpy,0.),np.maximum(state.enthalpy,0.))
    retained_heat=np.divide(remaining,available,out=np.zeros_like(loss),where=available>0.)*source_heat
    deposit_heat=deposit_ice*CI*(temperature-TF)+deposit_liquid*(LF+CW*(temperature-TF))
    heat=source_heat-retained_heat-deposit_heat
    actual=loss-deposit
    final=CanopyCondensate(remaining+np.where(use_ice,state.liquid,state.ice)+deposit,
        (state.enthalpy-source_heat)+retained_heat+deposit_heat)
    latent=LV*(out_ice+out_liquid)+LF*out_ice
    return final,dict(mass_kgm2=actual,ice_kgm2=out_ice,liquid_kgm2=out_liquid,
        advected_enthalpy_jm2=heat,latent_energy_jm2=latent,
        unmet_evaporation_kgm2=np.maximum(requested-loss,0.))

"""Simplified C3 annual-crop thermal phenology and conservative harvest.

Planting is an explicit management event. Soil heat units trigger emergence;
air heat units or maximum age trigger harvest. Carbon comes from imported
seed and the existing BGC photosynthesis/allocation, never prescribed LAI.
The harvest-index approximation exports a fraction of aboveground biomass;
it is not CLM's explicit reproductive pool. See docs/CLM_CROP.md.
"""
from dataclasses import dataclass
from enum import IntEnum
import math
import numpy as np
from .plant_cn import PlantState,PlantParameters

DAY=86400.


class CropPhase(IntEnum):
    FALLOW=0
    SOWN=1
    VEGETATIVE=2
    GRAIN_FILL=3


@dataclass(frozen=True)
class CropManagement:
    """Explicit one-step events; absent management means no new event."""
    planting: bool=False
    force_harvest: bool=False

    def __post_init__(self):
        if (not isinstance(self.planting,(bool,np.bool_))
            or not isinstance(self.force_harvest,(bool,np.bool_))
            or (self.planting and self.force_harvest)):
            raise ValueError('Crop management requires mutually exclusive boolean events')


@dataclass(frozen=True)
class CropParameters:
    base_temperature_k: float=273.15
    maximum_daily_heat: float=26.
    maturity_degree_days: float=1700.
    emergence_fraction: float=.05
    grain_fill_fraction: float=.6
    maximum_age_days: float=150.
    seed_carbon_gc_m2: float=3.
    harvest_index: float=.45
    maximum_lai: float=5.

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in vars(self).values())
            or not 250.<=self.base_temperature_k<=310. or self.maximum_daily_heat<=0.
            or self.maturity_degree_days<=0. or not 0.<self.emergence_fraction<self.grain_fill_fraction<1.
            or self.maximum_age_days<=0. or self.seed_carbon_gc_m2<=0.
            or not 0.<=self.harvest_index<=1. or self.maximum_lai<=0.):
            raise ValueError('Invalid crop thermal or management parameters')


@dataclass(frozen=True)
class CropState:
    air_degree_days: float=0.
    soil_degree_days: float=0.
    age_days: float=0.
    phase: CropPhase=CropPhase.FALLOW
    seed_carbon: float=0.
    seed_nitrogen: float=0.

    def __post_init__(self):
        values=(self.air_degree_days,self.soil_degree_days,self.age_days,self.seed_carbon,self.seed_nitrogen)
        if (not all(math.isfinite(v) and v>=0. for v in values)
            or not isinstance(self.phase,(CropPhase,int,np.integer)) or self.phase not in tuple(CropPhase)
            or (self.seed_carbon==0. and self.seed_nitrogen!=0.)
            or (self.phase!=CropPhase.SOWN and (self.seed_carbon!=0. or self.seed_nitrogen!=0.))):
            raise ValueError('Invalid crop phase, thermal time or seed stores')
        object.__setattr__(self,'phase',CropPhase(self.phase))


def step_crop(state,plant,air_temperature,soil_temperature,dt,*,planting=False,force_harvest=False,
              vegetated_fraction=1.,parameters=CropParameters(),plant_parameters=PlantParameters()):
    """Return crop state, actual plant state and explicit C/N transfers.

    Temperatures are K, dt seconds, C/N g/m2 total column area. Planting is
    a one-step pulse, not a persistent growing-season switch. Imported seed
    remains a conserved store until emergence. Harvest removes every live
    and reserve pool; non-exported material enters three nonwoody litter
    pools. The caller must add those inputs to soil and book seed/product
    as external fluxes. This operator does not calculate photosynthesis.
    """
    if (not isinstance(state,CropState) or not isinstance(plant,PlantState)
        or not all(math.isfinite(v) for v in (air_temperature,soil_temperature,dt,vegetated_fraction))
        or not 150.<=air_temperature<=350. or not 150.<=soil_temperature<=350.
        or not 0.<dt<=DAY or not 0.<vegetated_fraction<=1.
        or not isinstance(planting,(bool,np.bool_)) or not isinstance(force_harvest,(bool,np.bool_))):
        raise ValueError('Invalid crop state, weather, timestep or management event')
    p=parameters;pp=plant_parameters
    old_c=plant.total_carbon+state.seed_carbon;old_n=plant.total_nitrogen+state.seed_nitrogen
    phase=state.phase;air=state.air_degree_days;soil=state.soil_degree_days;age=state.age_days
    seed_c=state.seed_carbon;seed_n=state.seed_nitrogen;import_c=import_n=0.
    if phase==CropPhase.FALLOW and (plant.total_carbon>0. or plant.total_nitrogen>0.):
        raise ValueError('Fallow annual crop cannot retain living or reserve biomass')
    if phase==CropPhase.SOWN and (plant.total_carbon>0. or plant.total_nitrogen>0.):
        raise ValueError('Unemerged annual crop cannot carry displayed biomass')
    if planting:
        if phase!=CropPhase.FALLOW:raise ValueError('Planting requires a fallow crop')
        if force_harvest:raise ValueError('Planting and forced harvest cannot coincide')
        phase=CropPhase.SOWN;air=soil=age=0.
        seed_c=import_c=p.seed_carbon_gc_m2;seed_n=import_n=seed_c/pp.target_cn[0]
    if force_harvest and phase==CropPhase.FALLOW:raise ValueError('Forced harvest requires a planted crop')
    emerged=False;harvested=False;reason=0;export_c=export_n=litter_c=litter_n=0.
    if phase!=CropPhase.FALLOW:
        air+=min(max(air_temperature-p.base_temperature_k,0.),p.maximum_daily_heat)*dt/DAY
        soil+=min(max(soil_temperature-p.base_temperature_k,0.),p.maximum_daily_heat)*dt/DAY
        age+=dt/DAY
        # Priority makes an over-age/cold non-emerged crop return its seed
        # to soil. It cannot acquire or export phantom aboveground biomass.
        reason=3 if force_harvest else (1 if air>=p.maturity_degree_days else (2 if age>=p.maximum_age_days else 0))
        if reason:
            harvested=True
            export_c=p.harvest_index*float(plant.carbon[0]+plant.carbon[1])
            export_n=p.harvest_index*float(plant.nitrogen[0]+plant.nitrogen[1])
            litter_c=plant.total_carbon+seed_c-export_c;litter_n=plant.total_nitrogen+seed_n-export_n
            plant=PlantState(np.zeros(3),np.zeros(3));seed_c=seed_n=0.;phase=CropPhase.FALLOW
        else:
            if phase==CropPhase.SOWN and soil>=p.emergence_fraction*p.maturity_degree_days:
                c=plant.carbon.copy();n=plant.nitrogen.copy();c[0]+=seed_c;n[0]+=seed_n
                plant=PlantState(c,n,plant.reserve_carbon,plant.reserve_nitrogen)
                seed_c=seed_n=0.;phase=CropPhase.VEGETATIVE;emerged=True
            if phase==CropPhase.VEGETATIVE and (air>=p.grain_fill_fraction*p.maturity_degree_days
                    or plant.carbon[0]*pp.specific_leaf_area/vegetated_fraction>=p.maximum_lai):
                phase=CropPhase.GRAIN_FILL
    final=CropState(air,soil,age,phase,seed_c,seed_n)
    ci=np.zeros(7);ni=np.zeros(7);ci[:3]=litter_c*np.asarray(pp.litter_fractions);ni[:3]=litter_n*np.asarray(pp.litter_fractions)
    new_c=plant.total_carbon+seed_c;new_n=plant.total_nitrogen+seed_n
    cr=new_c-old_c+export_c+float(ci.sum())-import_c;nr=new_n-old_n+export_n+float(ni.sum())-import_n
    cb=128.*np.finfo(float).eps*max(new_c+old_c+export_c+float(ci.sum())+import_c,1.)
    nb=128.*np.finfo(float).eps*max(new_n+old_n+export_n+float(ni.sum())+import_n,1.)
    if abs(cr)>cb or abs(nr)>nb:raise RuntimeError('Crop management C/N budget exceeds numerical bound')
    diagnostics=dict(planted=bool(planting),emerged=emerged,harvested=harvested,harvest_reason=reason,
        seed_input_gc_m2=import_c,seed_input_gn_m2=import_n,harvest_gc_m2=export_c,harvest_gn_m2=export_n,
        litter_carbon_gc_m2=ci,litter_nitrogen_gn_m2=ni,carbon_residual_gc_m2=cr,nitrogen_residual_gn_m2=nr,
        carbon_bound_gc_m2=cb,nitrogen_bound_gn_m2=nb,growing=phase in (CropPhase.VEGETATIVE,CropPhase.GRAIN_FILL),
        target_lai=p.maximum_lai if phase in (CropPhase.VEGETATIVE,CropPhase.GRAIN_FILL) else 0.)
    return final,plant,diagnostics

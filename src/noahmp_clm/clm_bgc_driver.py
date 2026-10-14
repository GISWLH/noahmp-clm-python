"""NumPy whole-column plant/soil C-N feedback with a conservative ledger.

GDD target -> real leaf loss -> actual LAI/N capacity -> physical column ->
gross assimilation and leaf Rd -> N competition -> funded growth and NEE.
New biomass affects the following physical step (explicit operator split).
"""
from dataclasses import replace
import math
import numpy as np
from .physics.biogeochemistry import BGCState,soil_environment
from .physics.plant_cn import (turnover_plant,plant_lai,leaf_nitrogen_factor,plan_plant_allocation,
    apply_plant_allocation,carbon_exchange_diagnostics,GC_PER_UMOL)
from .physics.decomposition import step_decomposition
from .physics.phenology import PhenologyEnvironment,PhenologyPhase,step_phenology


def step_clm_bgc_column(state,forcing,*,phenology_environment=None,nitrogen_deposition_gn_m2=0.,
                        oxygen_factor=1.,nitrogen_factor=1.,**physical_options):
    """Advance actual water/heat/C/N; N deposition is an amount for this step.

    Nitrogen capacity belongs to the plant state. An independent caller
    multiplier would bypass that feedback and is therefore rejected.
    """
    from .clm_enthalpy_driver import step_clm_enthalpy_column
    if state.bgc_state is None:raise ValueError('A BGC column requires explicit C/N pools')
    if np.ndim(nitrogen_factor)!=0 or nitrogen_factor!=1.:
        raise ValueError('BGC derives nitrogen_factor from actual leaf C/N')
    if (not math.isfinite(nitrogen_deposition_gn_m2) or nitrogen_deposition_gn_m2<0.
        or not math.isfinite(oxygen_factor) or not 0.<=oxygen_factor<=1.):
        raise ValueError('Invalid BGC nitrogen deposition or oxygen factor')
    old=state.bgc_state;p=state.bgc_parameters;f=forcing;season=state.phenology_state;pd=None;retention=1.;growing=True
    if state.options.OptDynamicVeg==5:
        if not isinstance(phenology_environment,PhenologyEnvironment):raise ValueError('BGC dynamic vegetation requires PhenologyEnvironment')
        e=phenology_environment
        season,pd=step_phenology(season,float(state.ground.temperature[2]),e.daylength_seconds,
            e.increasing_daylength,e.winter_solstice,f.dt,state.phenology_parameters)
        before=state.phenology_state.leaf_fraction;after=season.leaf_fraction
        retention=(min(1.,after/before) if before>0. else (1. if after>0. else 0.))
        target_lai=pd['lai'];growing=season.phase in (PhenologyPhase.ONSET,PhenologyPhase.ACTIVE)
    else:
        if phenology_environment is not None:raise ValueError('Phenology environment requires OptDynamicVeg=5')
        target_lai=p.maximum_lai
    plant,fall=turnover_plant(old.plant,f.dt,leaf_retention=retention,parameters=p.plant)
    lai=plant_lai(plant,state.fveg,p.plant);nf=leaf_nitrogen_factor(plant,p.plant)
    # This temporary physical state has no unadvanced seasonal or C/N state.
    # The outer step retains and returns both; M/H is copied without loss.
    physical=replace(state,bgc_state=None,bgc_parameters=None,phenology_state=None,phenology_parameters=None,
        options=replace(state.options,OptDynamicVeg=4),lai=lai)
    next_physical,d=step_clm_enthalpy_column(physical,f,nitrogen_factor=nf,**physical_options)
    environment=soil_environment(next_physical.ground,state.root_frac,p)
    plan=plan_plant_allocation(plant,d['gpp_gcm2'],d['leaf_respiration']*GC_PER_UMOL*f.dt,f.tair,
        environment['root_temperature'],f.dt,target_leaf_carbon=target_lai*state.fveg/p.plant.specific_leaf_area,
        growing=growing,parameters=p.plant)
    soil,sd=step_decomposition(old.soil,environment['soil_temperature'],environment['water_potential_mpa'],f.dt,
        carbon_input=fall['litter_carbon_gc_m2'],nitrogen_input=fall['litter_nitrogen_gn_m2'],
        mineral_n_input=nitrogen_deposition_gn_m2,plant_n_demand=plan.soil_n_demand,oxygen_factor=oxygen_factor,parameters=p.decomposition)
    plant,ad=apply_plant_allocation(plan,sd['plant_uptake_gn_m2'])
    new=BGCState(plant,soil)
    exchange=carbon_exchange_diagnostics(d['gpp_gcm2'],ad['autotrophic_respiration_gc_m2'],
        sd['heterotrophic_respiration_gc_m2'],ad['root_respiration_gc_m2'],f.dt)
    cr=new.total_carbon-old.total_carbon+exchange['nee_gc_m2']
    nr=new.total_nitrogen-old.total_nitrogen-nitrogen_deposition_gn_m2
    eps=np.finfo(float).eps
    cb=128.*eps*max(new.total_carbon+old.total_carbon+exchange['gpp_gc_m2']+exchange['ecosystem_respiration_gc_m2'],1.)
    nb=128.*eps*max(new.total_nitrogen+old.total_nitrogen+nitrogen_deposition_gn_m2,1.)
    if abs(cr)>cb or abs(nr)>nb:raise RuntimeError('Whole-column C/N budget exceeds numerical bound')
    final=replace(next_physical,options=state.options,phenology_state=season,phenology_parameters=state.phenology_parameters,
        bgc_state=new,bgc_parameters=p)
    d.update(exchange)
    d['biogeochemistry']=dict(turnover=fall,allocation=ad,decomposition=sd,environment=environment,
        lai_before=state.lai,lai_used=lai,lai_after=final.lai,target_lai=target_lai,nitrogen_factor_used=nf,
        nitrogen_factor_after=leaf_nitrogen_factor(plant,p.plant),carbon_residual_gc_m2=cr,nitrogen_residual_gn_m2=nr,
        carbon_bound_gc_m2=cb,nitrogen_bound_gn_m2=nb,nitrogen_deposition_gn_m2=nitrogen_deposition_gn_m2)
    if pd is not None:d['phenology']=dict(pd,lai_before=state.lai,lai_used=lai,lai_after=final.lai)
    return final,d

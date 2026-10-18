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
from .physics.crop import CropManagement,CropPhase,step_crop
from .physics.vegetation_structure import StemAreaScheme,step_stem_area


def step_clm_bgc_column(state,forcing,*,phenology_environment=None,nitrogen_deposition_gn_m2=0.,
                        oxygen_factor=1.,nitrogen_factor=1.,crop_management=None,**physical_options):
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
    if state.wetland_state is not None and oxygen_factor!=1.:
        raise ValueError('Wetland derives oxygen limitation from actual soil phases')
    old=state.bgc_state;p=state.bgc_parameters;f=forcing;season=state.phenology_state;pd=None;retention=1.;growing=True
    crop=state.crop_state;cd=None;managed_plant=old.plant
    if crop is not None:
        if phenology_environment is not None:raise ValueError('Crop thermal phenology does not use a seasonal environment')
        management=CropManagement() if crop_management is None else crop_management
        if not isinstance(management,CropManagement):raise ValueError('Expected CropManagement')
        crop,managed_plant,cd=step_crop(crop,old.plant,f.tair,float(state.ground.temperature[0]),f.dt,
            planting=management.planting,force_harvest=management.force_harvest,vegetated_fraction=state.fveg,
            parameters=state.crop_parameters,plant_parameters=p.plant)
        target_lai=cd['target_lai'];growing=cd['growing']
    elif crop_management is not None:
        raise ValueError('Crop management requires OptCropModel=1')
    elif state.options.OptDynamicVeg==5:
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
    growing=bool(growing and state.fveg>0.)
    plant,fall=turnover_plant(managed_plant,f.dt,leaf_retention=retention,parameters=p.plant)
    lai=plant_lai(plant,state.fveg,p.plant);nf=leaf_nitrogen_factor(plant,p.plant)
    target_leaf_carbon=target_lai*state.fveg/p.plant.specific_leaf_area
    if crop is not None and crop.phase==CropPhase.GRAIN_FILL:
        # No new displayed leaf during grain fill. Fixed stem/root allocation
        # continues; the unused leaf share stays in the actual reserve pool.
        target_leaf_carbon=min(target_leaf_carbon,float(plant.carbon[0]))
        target_lai=target_leaf_carbon*p.plant.specific_leaf_area/state.fveg
    litter_c=fall['litter_carbon_gc_m2'];litter_n=fall['litter_nitrogen_gn_m2']
    if cd is not None:
        litter_c=litter_c+cd['litter_carbon_gc_m2'];litter_n=litter_n+cd['litter_nitrogen_gn_m2']
    structure_parameters=state.canopy_structure_parameters
    structure_options={};structure_used=None
    if structure_parameters is not None:
        if structure_parameters.scheme!=StemAreaScheme.PRESCRIBED:
            if physical_options.get('prescribed_sai') is not None:
                raise ValueError('Prescribed SAI cannot bypass dynamic C/N structure')
            flags={}
            if structure_parameters.scheme==StemAreaScheme.ANNUAL_C3_CROP:
                flags=dict(harvested=bool(cd['harvested']),fallow=crop.phase==CropPhase.FALLOW,
                    newly_planted=bool(cd['planted']))
            sai_used,structure_used=step_stem_area(state.sai,state.lai,lai,f.dt,structure_parameters,**flags)
        else:sai_used=state.sai
        structure_options=dict(sai=sai_used,
            canopy_structure_parameters=replace(structure_parameters,scheme=StemAreaScheme.PRESCRIBED))
    # This temporary physical state has no unadvanced seasonal or C/N state.
    # The outer step retains and returns both; M/H is copied without loss.
    physical=replace(state,bgc_state=None,bgc_parameters=None,phenology_state=None,phenology_parameters=None,
        crop_state=None,crop_parameters=None,wetland_state=None,wetland_parameters=None,
        options=replace(state.options,OptDynamicVeg=4,OptCropModel=0,OptWetlandModel=0),lai=lai,**structure_options)
    next_physical,d=step_clm_enthalpy_column(physical,f,nitrogen_factor=nf,**physical_options)
    environment=soil_environment(next_physical.ground,state.root_frac,p)
    methane=state.wetland_state;wetland_environment=wd=None;ch4_emission=0.
    if methane is not None:
        from .physics.wetland import column_wetland_environment,step_wetland
        wetland_environment=column_wetland_environment(next_physical.ground,f.ps,state.wetland_parameters)
        oxygen_factor=wetland_environment['oxygen_factor']
    plan=plan_plant_allocation(plant,d['gpp_gcm2'],d['leaf_respiration']*GC_PER_UMOL*f.dt,f.tair,
        environment['root_temperature'],f.dt,target_leaf_carbon=target_leaf_carbon,
        growing=growing,parameters=p.plant)
    soil,sd=step_decomposition(old.soil,environment['soil_temperature'],environment['water_potential_mpa'],f.dt,
        carbon_input=litter_c,nitrogen_input=litter_n,
        mineral_n_input=nitrogen_deposition_gn_m2,plant_n_demand=plan.soil_n_demand,oxygen_factor=oxygen_factor,parameters=p.decomposition)
    plant,ad=apply_plant_allocation(plan,sd['plant_uptake_gn_m2'])
    new=BGCState(plant,soil)
    respired_co2=sd['heterotrophic_respiration_gc_m2']
    if methane is not None:
        w=wetland_environment
        methane,wd=step_wetland(methane,respired_co2,w['temperature_k'],w['liquid_theta'],w['ice_theta'],
            w['porosity'],w['pressure_pa'],f.dt,parameters=state.wetland_parameters)
        respired_co2=wd['co2_gc_m2'];ch4_emission=wd['methane_emission_gc_m2']
    exchange=carbon_exchange_diagnostics(d['gpp_gcm2'],ad['autotrophic_respiration_gc_m2'],
        respired_co2,ad['root_respiration_gc_m2'],f.dt)
    new_c=new.total_carbon+(crop.seed_carbon if crop is not None else 0.)+(methane.methane_gc_m2 if methane is not None else 0.)
    new_n=new.total_nitrogen+(crop.seed_nitrogen if crop is not None else 0.)
    seed_c=cd['seed_input_gc_m2'] if cd is not None else 0.;seed_n=cd['seed_input_gn_m2'] if cd is not None else 0.
    export_c=cd['harvest_gc_m2'] if cd is not None else 0.;export_n=cd['harvest_gn_m2'] if cd is not None else 0.
    cr=new_c-state.carbon_gc_m2+exchange['nee_gc_m2']+ch4_emission+export_c-seed_c
    nr=new_n-state.nitrogen_gn_m2-nitrogen_deposition_gn_m2+export_n-seed_n
    eps=np.finfo(float).eps
    cb=128.*eps*max(new_c+state.carbon_gc_m2+exchange['gpp_gc_m2']+exchange['ecosystem_respiration_gc_m2']+ch4_emission+seed_c+export_c,1.)
    nb=128.*eps*max(new_n+state.nitrogen_gn_m2+nitrogen_deposition_gn_m2+seed_n+export_n,1.)
    if abs(cr)>cb or abs(nr)>nb:raise RuntimeError('Whole-column C/N budget exceeds numerical bound')
    final_structure={};structure_after=None
    if structure_parameters is not None:
        final_sai=next_physical.sai
        if structure_parameters.scheme!=StemAreaScheme.PRESCRIBED:
            # Re-evaluate from the SAME old state with funded end-of-step LAI;
            # do not apply the monthly decay twice. The current physical solve
            # used the post-turnover geometry, just like its displayed LAI.
            final_sai,structure_after=step_stem_area(state.sai,state.lai,plant_lai(new.plant,state.fveg,p.plant),
                f.dt,structure_parameters,**flags)
        final_structure=dict(sai=final_sai,canopy_structure_parameters=structure_parameters)
    final=replace(next_physical,options=state.options,phenology_state=season,phenology_parameters=state.phenology_parameters,
        bgc_state=new,bgc_parameters=p,crop_state=crop,crop_parameters=state.crop_parameters,
        wetland_state=methane,wetland_parameters=state.wetland_parameters,**final_structure)
    d.update(exchange)
    d['biogeochemistry']=dict(turnover=fall,allocation=ad,decomposition=sd,environment=environment,
        lai_before=state.lai,lai_used=lai,lai_after=final.lai,target_lai=target_lai,nitrogen_factor_used=nf,
        nitrogen_factor_after=leaf_nitrogen_factor(plant,p.plant),carbon_residual_gc_m2=cr,nitrogen_residual_gn_m2=nr,
        carbon_bound_gc_m2=cb,nitrogen_bound_gn_m2=nb,nitrogen_deposition_gn_m2=nitrogen_deposition_gn_m2)
    if structure_parameters is not None:
        d['vegetation_structure']=dict(used=structure_used,after=structure_after,sai_before=state.sai,
            sai_used=next_physical.sai,sai_after=final.sai,scheme=int(structure_parameters.scheme))
    if pd is not None:d['phenology']=dict(pd,lai_before=state.lai,lai_used=lai,lai_after=final.lai)
    if cd is not None:d['crop']=cd
    if wd is not None:
        d['wetland']=dict(wd,environment=wetland_environment)
        d.update(decomposed_carbon_gc_m2=sd['heterotrophic_respiration_gc_m2'],methane_emission_gc_m2=ch4_emission,
            methane_emission_mgch4_m2=wd['methane_emission_mgch4_m2'],total_atmospheric_carbon_gc_m2=exchange['nee_gc_m2']+ch4_emission)
    return final,d

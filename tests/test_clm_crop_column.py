"""Actual crop/soil/canopy transfers, including seed/product and harvested M/H."""
from dataclasses import replace
import math
import numpy as np
import pytest
from noahmp_clm.clm_driver import step_clm_column
from noahmp_clm.physics.crop import CropState,CropParameters,CropPhase as Phase,CropManagement
from noahmp_clm.physics.plant_cn import PlantState,turnover_plant
from test_clm_bgc_column import case as bgc_case


def case(kind='vegetative',temperature=285.):
    state,f,_=bgc_case(None,temperature)
    f=replace(f,tair=temperature)
    crop=dict(fallow=CropState(),sown=CropState(20.,20.,2.,Phase.SOWN,3.,.12),
        emergence=CropState(84.95,84.95,4.,Phase.SOWN,3.,.12),
        vegetative=CropState(200.,200.,15.,Phase.VEGETATIVE),
        grain=CropState(1100.,1100.,70.,Phase.GRAIN_FILL),
        mature=CropState(1699.99,1699.99,90.,Phase.GRAIN_FILL),
        cold_seed=CropState(20.,20.,149.99,Phase.SOWN,3.,.12))[kind]
    if crop.phase in (Phase.FALLOW,Phase.SOWN):
        state=replace(state,bgc_state=replace(state.bgc_state,plant=PlantState(np.zeros(3),np.zeros(3))))
    return state.with_crop(initial=crop),f


def stores(state):
    b=state.bgc_state;c=state.crop_state
    return (math.fsum([*b.plant.carbon,b.plant.reserve_carbon,*b.soil.carbon,c.seed_carbon]),
        math.fsum([*b.plant.nitrogen,b.plant.reserve_nitrogen,*b.soil.nitrogen,b.soil.mineral_nitrogen,c.seed_nitrogen]))


def budgets(old,new,f,d,deposition):
    a=stores(old);b=stores(new);crop=d['crop'];bgc=d['biogeochemistry']
    c=b[0]-a[0]+d['nee_gc_m2']+crop['harvest_gc_m2']-crop['seed_input_gc_m2']
    n=b[1]-a[1]-deposition+crop['harvest_gn_m2']-crop['seed_input_gn_m2']
    assert abs(c)<=bgc['carbon_bound_gc_m2'] and abs(n)<=bgc['nitrogen_bound_gn_m2']
    water=new.water_kgm2-old.water_kgm2-f.prcp*f.dt*1000.+d['et_mm']+d['runoff_mm']
    heat=new.enthalpy_jm2-old.enthalpy_jm2-d['precipitation_enthalpy_jm2']+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    heat-=(f.swdown+f.lwdown-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*f.dt
    assert abs(water)<1e-9 and abs(heat)<.02
    return dict(carbon=c,nitrogen=n,water=water,heat=heat)


@pytest.mark.parametrize('kind,temperature,event',[
    ('fallow',285.,CropManagement(planting=True)),('sown',285.,None),('emergence',285.,None),
    ('vegetative',285.,None),('grain',285.,None),('mature',285.,None),
    ('vegetative',270.,CropManagement(force_harvest=True)),('cold_seed',270.,None)])
def test_crop_manages_real_biomass_and_all_four_conserved_stores(kind,temperature,event):
    old,f=case(kind,temperature);dep=.00001
    new,d=step_clm_column(old,f,crop_management=event,nitrogen_deposition_gn_m2=dep)
    crop=d['crop'];bgc=d['biogeochemistry']
    assert new.options.OptCropModel==1 and new.crop_parameters==old.crop_parameters
    assert new.lai==pytest.approx(new.bgc_state.plant.carbon[0]*old.bgc_parameters.plant.specific_leaf_area/old.fveg,abs=1e-15)
    assert d['nee_gc_m2']==d['autotrophic_respiration_gc_m2']+d['heterotrophic_respiration_gc_m2']-d['gpp_gcm2']
    if kind=='fallow':
        assert crop['seed_input_gc_m2']==new.crop_state.seed_carbon==3.
        assert new.lai==d['gpp_gcm2']==0. and new.crop_state.phase==Phase.SOWN
    if kind=='emergence':
        assert crop['emerged'] and new.crop_state.seed_carbon==0. and d['gpp_gcm2']>0.
        assert new.bgc_state.plant.total_carbon>0. and bgc['lai_used']>0.
    if kind=='grain':
        turned,_=turnover_plant(old.bgc_state.plant,f.dt,parameters=old.bgc_parameters.plant)
        assert new.bgc_state.plant.carbon[0]==turned.carbon[0]
        assert bgc['allocation']['carbon_growth_gc_m2'][0]==0.
        assert new.bgc_state.plant.carbon[1]>turned.carbon[1]
    if crop['harvested']:
        assert new.crop_state.phase==Phase.FALLOW and new.bgc_state.plant.total_carbon==new.bgc_state.plant.total_nitrogen==0.
        assert new.lai==new.canopy_water_kgm2==d['gpp_gcm2']==0.
        flow=d['canopy_interception']
        assert flow['ground_liquid_kgm2']+flow['ground_ice_kgm2']==pytest.approx(old.canopy_water_kgm2,abs=1e-12)
        assert flow['ground_liquid_enthalpy_jm2']+flow['ground_ice_enthalpy_jm2']==pytest.approx(old.canopy_enthalpy_jm2,abs=1e-7)
        if kind=='cold_seed':assert crop['harvest_gc_m2']==0. and sum(crop['litter_carbon_gc_m2'])==pytest.approx(3.)
        else:assert crop['harvest_gc_m2']==pytest.approx(.45*sum(old.bgc_state.plant.carbon[:2]))
    print('crop full NumPy',kind,temperature,dict(**budgets(old,new,f,d,dep),
        gpp=d['gpp_gcm2'],nee=d['nee_gc_m2'],seed_input=crop['seed_input_gc_m2'],harvest=crop['harvest_gc_m2'],lai=new.lai),flush=True)
    # The second real step must preserve the crop stores rather than reset
    # management or repeat harvest/seed import.
    after,dd=step_clm_column(new,f)
    assert dd['crop']['seed_input_gc_m2']==dd['crop']['harvest_gc_m2']==0.
    budgets(new,after,f,dd,0.)


def test_crop_contract_rejects_competing_phenology_or_lost_stores():
    from noahmp_clm.gpu.clm_bgc_column import pack_clm_bgc_columns
    state,f=case()
    with pytest.raises(NotImplementedError,match='Crop'):pack_clm_bgc_columns([state])
    with pytest.raises(ValueError,match='Crop'):replace(state,crop_parameters=None)
    with pytest.raises(ValueError,match='crop'):replace(state,crop_state=CropState())
    with pytest.raises(ValueError,match='Bare ground'):replace(state,fveg=0.)
    seasonal,_,_=bgc_case()
    with pytest.raises(ValueError,match='OptDynamicVeg'):seasonal.with_crop(initial=state.crop_state)
    plain,_,_=bgc_case(None)
    with pytest.raises(ValueError,match='Crop management'):step_clm_column(plain,f,crop_management=CropManagement())
    with pytest.raises(ValueError,match='CropManagement'):step_clm_column(state,f,crop_management=True)
    with pytest.raises(ValueError,match='boolean'):CropManagement(.5)
    with pytest.raises(ValueError,match='boolean'):CropManagement(True,True)


def test_packing_crop_does_not_truncate_fractional_events():
    from noahmp_clm.gpu.crop import pack_crop
    state,f=case('fallow')
    with pytest.raises(ValueError,match='boolean'):
        pack_crop(state.crop_state,state.bgc_state.plant,f.tair,f.tair,f.dt,planting=.5)

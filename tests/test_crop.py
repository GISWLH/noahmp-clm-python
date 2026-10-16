"""Independent heat-unit references and explicit seed/harvest C/N accounts."""
from dataclasses import replace
from decimal import Decimal,localcontext
import numpy as np
import pytest
from noahmp_clm.physics.crop import CropParameters,CropState,CropPhase,step_crop,DAY
from noahmp_clm.physics.plant_cn import PlantState,PlantParameters


def empty():return PlantState(np.zeros(3),np.zeros(3))


def audit(old_crop,old_plant,crop,plant,d):
    c=(plant.total_carbon+crop.seed_carbon)-(old_plant.total_carbon+old_crop.seed_carbon)
    n=(plant.total_nitrogen+crop.seed_nitrogen)-(old_plant.total_nitrogen+old_crop.seed_nitrogen)
    c+=d['harvest_gc_m2']+sum(d['litter_carbon_gc_m2'])-d['seed_input_gc_m2']
    n+=d['harvest_gn_m2']+sum(d['litter_nitrogen_gn_m2'])-d['seed_input_gn_m2']
    assert abs(c)<=d['carbon_bound_gc_m2'] and abs(n)<=d['nitrogen_bound_gn_m2']
    return abs(c),abs(n)


def test_heat_units_match_independent_decimal_capped_integral():
    # CLM5 crop eq. 2.26.3: clipped positive heat integrated in days.
    p=CropParameters();errors=[]
    with localcontext() as ctx:
        ctx.prec=45
        for tair in (250.,273.15,273.16,280.,299.15,330.):
            for tsoil in (260.,273.15,284.35,310.):
                for dt in (900.,1800.,3600.,DAY):
                    old=CropState();before=empty()
                    new,plant,d=step_crop(old,before,tair,tsoil,dt,planting=True)
                    expected=[float(min(max(Decimal(str(t))-Decimal('273.15'),Decimal(0)),Decimal(26))
                                    *Decimal(str(dt))/Decimal(86400)) for t in (tair,tsoil)]
                    errors.extend(abs(np.array([new.air_degree_days,new.soil_degree_days])-expected))
                    audit(old,before,new,plant,d)
    assert max(errors)<1e-12
    print('crop independent heat-unit maximum absolute error',max(errors),flush=True)


def test_constant_weather_phase_dates_seed_and_harvest_are_analytical():
    crop=CropState();plant=empty();events={};imports=exports=litter=0.;residual=np.zeros(2)
    for day in range(1,91):
        old,before=crop,plant
        crop,plant,d=step_crop(crop,plant,293.15,293.15,DAY,planting=day==1)
        imports+=d['seed_input_gc_m2'];exports+=d['harvest_gc_m2'];litter+=sum(d['litter_carbon_gc_m2'])
        residual=np.maximum(residual,audit(old,before,crop,plant,d))
        if d['emerged']:events['emergence']=day
        if crop.phase==CropPhase.GRAIN_FILL:events.setdefault('grain_fill',day)
        if d['harvested']:events['harvest']=day
        if 1<=day<5:assert crop.seed_carbon==3. and plant.total_carbon==0.
    # 5% * 1700 / 20, 60% * 1700 / 20, 1700 / 20, rounded to step end.
    assert events==dict(emergence=5,grain_fill=51,harvest=85)
    assert imports==3. and exports==pytest.approx(1.35,abs=1e-14) and litter==pytest.approx(1.65,abs=1e-14)
    assert crop.phase==CropPhase.FALLOW and plant.total_carbon==crop.seed_carbon==0.
    print('crop analytic seasonal dates',events,'C/N residuals',residual.tolist(),flush=True)


@pytest.mark.parametrize('cause',[1,2,3])
def test_harvest_routes_all_actual_tissue_and_reserves(cause):
    plant=PlantState(np.array([100.,200.,50.]),np.array([4.,2.,1.]),10.,.5)
    state=CropState(1699. if cause==1 else 500.,200.,149.9 if cause==2 else 30.,CropPhase.GRAIN_FILL)
    new,after,d=step_crop(state,plant,293.15,283.15,DAY,force_harvest=cause==3)
    assert d['harvest_reason']==cause and d['harvested']
    assert new.phase==CropPhase.FALLOW and after.total_carbon==after.total_nitrogen==0.
    assert d['harvest_gc_m2']==135. and d['harvest_gn_m2']==2.7
    np.testing.assert_allclose(d['litter_carbon_gc_m2'],[67.5,101.25,56.25,0,0,0,0],rtol=0,atol=1e-13)
    np.testing.assert_allclose(d['litter_nitrogen_gn_m2'],[1.44,2.16,1.2,0,0,0,0],rtol=0,atol=1e-14)
    audit(state,plant,new,after,d)


def test_failed_emergence_returns_seed_and_never_invents_yield():
    p=replace(CropParameters(),maximum_age_days=2.)
    crop,plant,_=step_crop(CropState(),empty(),270.,270.,DAY,planting=True,parameters=p)
    new,after,d=step_crop(crop,plant,270.,270.,DAY,parameters=p)
    assert d['harvested'] and not d['emerged'] and d['harvest_gc_m2']==d['harvest_gn_m2']==0.
    assert sum(d['litter_carbon_gc_m2'])==3. and sum(d['litter_nitrogen_gn_m2'])==pytest.approx(.12)
    audit(crop,plant,new,after,d)


def test_lai_can_trigger_grain_fill_without_changing_carbon():
    p=CropParameters();pp=PlantParameters();plant=PlantState(np.array([250.,20.,10.]),np.array([10.,1.,.25]))
    old=CropState(100.,100.,10.,CropPhase.VEGETATIVE)
    new,after,d=step_crop(old,plant,273.15,273.15,1800.,parameters=p,plant_parameters=pp)
    assert new.phase==CropPhase.GRAIN_FILL and d['growing'] and d['target_lai']==5.
    np.testing.assert_array_equal(after.carbon,plant.carbon);audit(old,plant,new,after,d)


def test_events_and_invalid_states_are_rejected():
    live=PlantState(np.ones(3),np.ones(3))
    with pytest.raises(ValueError,match='Fallow'):step_crop(CropState(),live,280.,280.,1800.)
    with pytest.raises(ValueError,match='planted'):step_crop(CropState(),empty(),280.,280.,1800.,force_harvest=True)
    old=CropState(phase=CropPhase.SOWN,seed_carbon=3.,seed_nitrogen=.12)
    with pytest.raises(ValueError,match='fallow'):step_crop(old,empty(),280.,280.,1800.,planting=True)
    with pytest.raises(ValueError,match='cannot coincide'):step_crop(CropState(),empty(),280.,280.,1800.,planting=True,force_harvest=True)
    with pytest.raises(ValueError):CropState(phase=1.5)
    with pytest.raises(ValueError):CropState(seed_carbon=1.)
    with pytest.raises(ValueError):replace(CropParameters(),emergence_fraction=.7)
    for dt in (0.,-1.,DAY+1.):
        with pytest.raises(ValueError):step_crop(CropState(),empty(),280.,280.,dt)

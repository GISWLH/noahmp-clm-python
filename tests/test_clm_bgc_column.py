"""Actual canopy assimilation, leaf carbon/LAI feedback and whole-column C/N."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.physics.biogeochemistry import BGCState,BGCParameters,soil_environment
from noahmp_clm.physics.plant_cn import PlantState,turnover_plant,plant_lai,leaf_nitrogen_factor
from noahmp_clm.physics.decomposition import DecompositionState
from noahmp_clm.physics.phenology import PhenologyPhase as Phase,step_phenology
from test_clm_phenology_column import case as seasonal_case


def case(phase=Phase.ACTIVE,temperature=285.,poor=False):
    state,f,e=seasonal_case(Phase.ACTIVE if phase is None else phase,temperature,.01 if temperature<273.15 else 0.)
    if phase is None:
        state=replace(state,options=replace(state.options,OptDynamicVeg=4),phenology_state=None,phenology_parameters=None)
        e=None
    p=BGCParameters();c=np.array([120.,600.,150.]);n=c/np.array(p.plant.target_cn)*(.25 if poor else 1.)
    plant=PlantState(c,n,30.,0. if poor else .05)
    c=np.array([50.,100.,100.,1000.,500.,5000.,10000.])
    soil=DecompositionState(c,c/25.,0. if poor else 1.)
    return state.with_biogeochemistry(BGCState(plant,soil),p),f,e


def budgets(old,new,f,d,deposition):
    bgc=d['biogeochemistry'];c=new.bgc_state.total_carbon-old.bgc_state.total_carbon+d['nee_gc_m2']
    n=new.bgc_state.total_nitrogen-old.bgc_state.total_nitrogen-deposition
    assert abs(c)<=bgc['carbon_bound_gc_m2'] and abs(n)<=bgc['nitrogen_bound_gn_m2']
    water=new.water_kgm2-old.water_kgm2-f.prcp*f.dt*1000.+d['et_mm']+d['runoff_mm']
    heat=new.enthalpy_jm2-old.enthalpy_jm2-d['precipitation_enthalpy_jm2']+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    heat-=(f.swdown+f.lwdown-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*f.dt
    assert abs(water)<1e-9 and abs(heat)<.02
    return dict(carbon=c,nitrogen=n,water=water,heat=heat)


@pytest.mark.parametrize('phase,temperature,poor',[(None,285.,False),(Phase.ACTIVE,285.,False),
    (Phase.ACTIVE,285.,True),(Phase.ONSET,285.,False),(Phase.OFFSET,270.,False),(Phase.DORMANT,270.,True)])
def test_bgc_couples_actual_leaf_carbon_nitrogen_and_conserves_all_stores(phase,temperature,poor):
    old,f,e=case(phase,temperature,poor);p=old.bgc_parameters;retention=1.
    if e is not None:
        season,sd=step_phenology(old.phenology_state,old.tsoil[2],e.daylength_seconds,e.increasing_daylength,e.winter_solstice,f.dt,old.phenology_parameters)
        before=old.phenology_state.leaf_fraction;after=season.leaf_fraction
        retention=min(1.,after/before) if before>0. else (1. if after>0. else 0.)
    turned,_=turnover_plant(old.bgc_state.plant,f.dt,leaf_retention=retention,parameters=p.plant)
    lai=plant_lai(turned,old.fveg,p.plant);nf=leaf_nitrogen_factor(turned,p.plant)
    physical=replace(old,bgc_state=None,bgc_parameters=None,phenology_state=None,phenology_parameters=None,
        options=replace(old.options,OptDynamicVeg=4),lai=lai)
    manual,md=step_clm_enthalpy_column(physical,f,nitrogen_factor=nf)
    new,d=step_clm_enthalpy_column(old,f,phenology_environment=e,nitrogen_deposition_gn_m2=.00001)
    for key in ('gpp_gcm2','h','le','runoff_mm'):assert d[key]==md[key]
    np.testing.assert_array_equal(new.ground.water,manual.ground.water)
    np.testing.assert_array_equal(new.ground.enthalpy,manual.ground.enthalpy)
    assert new.lai==plant_lai(new.bgc_state.plant,new.fveg,p.plant)
    assert d['biogeochemistry']['lai_used']==lai
    assert d['nee_gc_m2']==d['autotrophic_respiration_gc_m2']+d['heterotrophic_respiration_gc_m2']-d['gpp_gcm2']
    allocation=d['biogeochemistry']['allocation']
    assert allocation['maintenance_respiration_gc_m2'][0]==pytest.approx(d['leaf_respiration']*12.0107e-6*f.dt,abs=1e-14)
    assert d['soil_respiration_gc_m2']==allocation['root_respiration_gc_m2']+d['heterotrophic_respiration_gc_m2']
    if phase in (Phase.OFFSET,Phase.DORMANT):
        assert new.lai==new.canopy_water_kgm2==d['gpp_gcm2']==0.
        flow=d['canopy_interception']
        assert flow['ground_liquid_kgm2']+flow['ground_ice_kgm2']==pytest.approx(old.canopy_water_kgm2,abs=1e-12)
    print('BGC full NumPy',str(phase),temperature,poor,dict(**budgets(old,new,f,d,.00001),lai_used=lai,lai_after=new.lai,
        gpp=d['gpp_gcm2'],nee=d['nee_gc_m2'],nitrogen_factor=nf),flush=True)


def test_actual_leaf_nitrogen_reduces_whole_canopy_gpp():
    rich,f,_=case(None,285.,False);poor,_,_=case(None,285.,True)
    _,a=step_clm_enthalpy_column(rich,f);_,b=step_clm_enthalpy_column(poor,f)
    assert 0.<b['gpp_gcm2']<a['gpp_gcm2']
    assert b['biogeochemistry']['nitrogen_factor_used']==pytest.approx(.25,abs=1e-15)
    print('BGC real canopy N feedback',a['gpp_gcm2'],b['gpp_gcm2'],flush=True)


def test_root_maintenance_uses_weighted_q10_not_mean_temperature():
    old,_,_=case();g=old.ground
    from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
    from noahmp_clm.column import ColumnState
    g=SoilEnthalpyState.from_column(ColumnState(tsoil=np.array([280.,285.,295.,310.]),smc=np.full(4,.3)))
    env=soil_environment(g,old.root_frac,old.bgc_parameters);q=old.bgc_parameters.plant.maintenance_q10
    factor=sum(float(w)*q**((float(t)-293.15)/10.) for w,t in zip(old.root_frac,g.temperature))
    assert env['root_q10_factor']==pytest.approx(factor,abs=1e-14)
    assert q**((env['root_temperature']-293.15)/10.)==pytest.approx(factor,abs=1e-14)
    assert abs(factor-q**((old.root_frac@g.temperature-293.15)/10.))>.001
    expected=-.0001*(.3/.45)**-5.5
    assert env['water_potential_mpa']==pytest.approx(expected,abs=1e-13)


def test_bgc_cannot_be_silently_lost_in_old_gpu_abi_or_overridden():
    from noahmp_clm.gpu.clm_layered_column import pack_clm_layered_columns
    from noahmp_clm.gpu.clm_phenology_column import pack_clm_phenology_columns
    old,f,e=case()
    with pytest.raises(NotImplementedError,match='BGC'):pack_clm_phenology_columns([old])
    with pytest.raises(NotImplementedError,match='BGC'):pack_clm_layered_columns([old])
    with pytest.raises(ValueError,match='nitrogen_factor'):step_clm_enthalpy_column(old,f,nitrogen_factor=.5,phenology_environment=e)
    with pytest.raises(ValueError,match='BGC'):replace(old,bgc_parameters=None)


@pytest.mark.parametrize('fraction_type',[float,np.float64])
def test_bare_ground_cannot_grow_tissue_from_a_carried_reserve(fraction_type):
    old,f,_=case(None)
    plant=PlantState(np.zeros(3),np.zeros(3),20.,.1)
    old=replace(old,fveg=fraction_type(0.),bgc_state=replace(old.bgc_state,plant=plant))
    new,d=step_clm_enthalpy_column(old,f)
    np.testing.assert_array_equal(new.bgc_state.plant.carbon,np.zeros(3))
    assert new.bgc_state.plant.reserve_carbon==20. and new.lai==0. and d['gpp_gcm2']==0.
    budgets(old,new,f,d,0.)


def test_numpy_scalar_site_fraction_can_grow():
    old,f,_=case(None)
    old=replace(old,fveg=np.float64(old.fveg))
    new,d=step_clm_enthalpy_column(old,f)
    assert d['gpp_gcm2']>0. and new.bgc_state.plant.carbon[0]>old.bgc_state.plant.carbon[0]
    budgets(old,new,f,d,0.)

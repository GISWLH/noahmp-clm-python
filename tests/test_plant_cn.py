"""Independent survival, allocation and closed ecosystem carbon/nitrogen ledgers."""
from dataclasses import replace
from decimal import Decimal,localcontext
import numpy as np
import pytest
from noahmp_clm.physics.plant_cn import (PlantState as State,PlantParameters as Parameters,
    turnover_plant,plan_plant_allocation,apply_plant_allocation,leaf_nitrogen_factor,
    plant_lai,carbon_exchange_diagnostics,DAY,GC_PER_UMOL)


def example():
    p=Parameters();c=np.array([100.,1000.,200.])
    return State(c,c/np.array(p.target_cn),20.,.2),p


def test_turnover_against_exponential_survival_and_free_n_resorption():
    state,p=example();old=state;carbon=np.zeros(7);nitrogen=np.zeros(7)
    max_c=max_n=0.
    for _ in range(365):
        state,d=turnover_plant(state,DAY,parameters=p)
        carbon+=d['litter_carbon_gc_m2'];nitrogen+=d['litter_nitrogen_gn_m2']
        max_c=max(max_c,abs(d['carbon_residual_gc_m2']));max_n=max(max_n,abs(d['nitrogen_residual_gn_m2']))
    expected=old.carbon*np.exp(-365./np.array(p.turnover_days))
    relative=float(np.max(abs(state.carbon-expected)/expected))
    assert relative<.1
    np.testing.assert_allclose(state.carbon,expected,rtol=2e-13,atol=1e-12)
    assert state.reserve_nitrogen-old.reserve_nitrogen==pytest.approx((old.carbon[0]-expected[0])/75.,abs=1e-13)
    assert abs(state.total_carbon-old.total_carbon+carbon.sum())<1e-10
    assert abs(state.total_nitrogen-old.total_nitrogen+nitrogen.sum())<1e-12
    print('Plant annual survival',dict(relative=relative,step_carbon=max_c,step_nitrogen=max_n),flush=True)


def test_complete_leaf_fall_routes_all_carbon_and_resorbs_only_available_n():
    old,p=example();new,d=turnover_plant(old,1800.,leaf_retention=0.,parameters=p)
    assert new.carbon[0]==new.nitrogen[0]==0.
    assert d['retranslocated_nitrogen_gn_m2']==pytest.approx(4.-100./37.5)
    assert new.reserve_carbon==old.reserve_carbon
    poor=replace(old,nitrogen=old.nitrogen*np.array([.1,1.,1.]))
    _,d=turnover_plant(poor,1800.,leaf_retention=0.,parameters=p)
    assert d['retranslocated_nitrogen_gn_m2']==0.


def test_funded_allocation_matches_high_precision_independent_cash_and_n_ledger():
    old,p=example();dt=1800.;gpp=1.8;leaf=.12
    plan=plan_plant_allocation(old,gpp,leaf,303.15,283.15,dt,target_leaf_carbon=200.,parameters=p)
    new,d=apply_plant_allocation(plan,plan.soil_n_demand)
    with localcontext() as context:
        context.prec=60;D=lambda x:Decimal(str(x))
        # Independent scalar ledger at exactly +/-10 K, avoiding a duplicated
        # implementation of the production planner or its floating operations.
        m=D(leaf)+D(old.nitrogen[1])*D(p.stem_live_n_fraction)*D('0.000002525')*D(dt)*D('1.5')
        m+=D(old.nitrogen[2])*D('0.000002525')*D(dt)/D('1.5')
        released=D(20)*(1-(-D(dt)/(D(30)*D(DAY))).exp())
        total=((D(gpp)-m)*D('.9')+released)/D('1.11')
        growth=np.array([float(total*D(v)) for v in p.allocation])
        expected_reserve=float(D(20)+D(gpp)-m-total*D('1.11'))
    err=float(np.max(abs(d['carbon_growth_gc_m2']-growth)))
    np.testing.assert_allclose(d['carbon_growth_gc_m2'],growth,rtol=1e-13,atol=1e-14)
    assert new.reserve_carbon==pytest.approx(expected_reserve,abs=1e-13)
    assert d['autotrophic_respiration_gc_m2']==pytest.approx(float(m)+.11*growth.sum(),abs=1e-13)
    np.testing.assert_allclose(new.nitrogen,new.carbon/np.array(p.target_cn),rtol=1e-13)
    assert d['unmet_maintenance_gc_m2'].sum()==0.
    print('Plant Decimal allocation',dict(growth_max_absolute=err,carbon=d['carbon_residual_gc_m2'],nitrogen=d['nitrogen_residual_gn_m2']),flush=True)


def test_n_shortage_limits_growth_and_next_leaf_photosynthesis():
    from noahmp_clm.physics.photosynthesis import farquhar_rates
    old,p=example();old=replace(old,reserve_nitrogen=0.)
    plan=plan_plant_allocation(old,10.,.1,293.15,293.15,DAY,target_leaf_carbon=300.,parameters=p)
    rich,dr=apply_plant_allocation(plan,plan.soil_n_demand)
    poor,dp=apply_plant_allocation(plan,.25*plan.soil_n_demand)
    zero,dz=apply_plant_allocation(plan,0.)
    np.testing.assert_allclose(dp['carbon_growth_gc_m2'],.5*dr['carbon_growth_gc_m2'],rtol=1e-12)
    assert np.all(dz['carbon_growth_gc_m2']==0.) and zero.reserve_carbon>rich.reserve_carbon
    factor=leaf_nitrogen_factor(poor,p)
    assert 0.<factor<1.
    a=farquhar_rates(280.,2000.,298.15,nitrogen_factor=factor)
    b=farquhar_rates(280.,2000.,298.15,nitrogen_factor=1.)
    assert a['gross']<b['gross']
    assert poor.total_nitrogen-old.total_nitrogen==pytest.approx(.25*plan.soil_n_demand,abs=1e-13)


def test_starvation_never_creates_negative_carbon_and_reports_unmet_maintenance():
    old,p=example();old=replace(old,reserve_carbon=0.,reserve_nitrogen=0.)
    plan=plan_plant_allocation(old,.001,.2,303.15,303.15,DAY,target_leaf_carbon=300.,parameters=p)
    new,d=apply_plant_allocation(plan,0.)
    assert d['autotrophic_respiration_gc_m2']==pytest.approx(.001,abs=1e-15)
    assert np.all(d['carbon_growth_gc_m2']==0.) and new.reserve_carbon<1e-15
    assert d['unmet_maintenance_gc_m2'].sum()>.2
    np.testing.assert_array_equal(new.carbon,old.carbon)


def test_n_limited_repair_cannot_create_leaves_from_rounded_remainders():
    old=State(np.array([0.,1344.1875,563.083]),np.array([0.,1.234567,2.765431]),10.,0.)
    plan=plan_plant_allocation(old,1.,0.,293.15,293.15,1800.,target_leaf_carbon=200.)
    new,d=apply_plant_allocation(plan,.5*plan.tissue_n_deficit.sum())
    assert new.carbon[0]==new.nitrogen[0]==d['nitrogen_factor']==0.
    assert np.all(d['carbon_growth_gc_m2']==0.) and new.reserve_nitrogen==0.


def test_dormant_and_leaf_ceiling_preserve_unallocated_carbon():
    old,p=example()
    for growing,ceiling in ((False,300.),(True,old.carbon[0])):
        plan=plan_plant_allocation(old,1.,.1,293.15,293.15,1800.,target_leaf_carbon=ceiling,growing=growing,parameters=p)
        new,d=apply_plant_allocation(plan,plan.soil_n_demand)
        assert d['carbon_growth_gc_m2'][0]==0.
        if not growing:np.testing.assert_array_equal(new.carbon,old.carbon)
        assert new.reserve_carbon>old.reserve_carbon


def test_lai_exchange_units_and_contract():
    old,p=example()
    assert plant_lai(old,.5,p)==4.
    d=carbon_exchange_diagnostics(3.,1.,.5,.2,1800.)
    assert d['nee_gc_m2']==-1.5 and d['soil_respiration_gc_m2']==.7
    assert d['gpp_umol_m2s']==pytest.approx(3./(12.0107e-6*1800.))
    with pytest.raises(ValueError):old.carbon[0]=0.
    with pytest.raises(ValueError):plant_lai(old,0.)
    with pytest.raises(ValueError):Parameters(allocation=(.3,.3,.3))
    with pytest.raises(ValueError):turnover_plant(old,0.)
    with pytest.raises(ValueError):turnover_plant(old,1800.,leaf_retention=1.1)
    with pytest.raises(ValueError):State(np.zeros(3),np.ones(3))


def test_two_year_plant_litter_soil_exchange_has_closed_total_cn():
    from noahmp_clm.physics.decomposition import DecompositionState,step_decomposition
    plant,p=example();c=np.array([50.,100.,100.,1000.,500.,5000.,10000.])
    soil=DecompositionState(c,c/25.,1.)
    c0=plant.total_carbon+soil.carbon.sum();n0=plant.total_nitrogen+soil.nitrogen.sum()+soil.mineral_nitrogen
    total_gpp=total_resp=total_deposition=0.;max_c=max_n=0.;c_bound=n_bound=0.
    for day in range(730):
        plant,fall=turnover_plant(plant,DAY,leaf_retention=.95 if day%365>300 else 1.,parameters=p)
        temp=288.15+10.*np.sin(2*np.pi*day/365.)
        # Prescribed gross carbon input is an idealized component check, not
        # a substitute for canopy physics or field/official-output validation.
        gpp=max(0.,3.*np.sin(np.pi*(day%365)/365.))
        plan=plan_plant_allocation(plant,gpp,.1,temp,temp,DAY,target_leaf_carbon=200.,growing=day%365<300,parameters=p)
        soil,sd=step_decomposition(soil,temp,-.03,DAY,carbon_input=fall['litter_carbon_gc_m2'],
            nitrogen_input=fall['litter_nitrogen_gn_m2'],mineral_n_input=.001,plant_n_demand=plan.soil_n_demand)
        plant,pd=apply_plant_allocation(plan,sd['plant_uptake_gn_m2'])
        exchange=carbon_exchange_diagnostics(gpp,pd['autotrophic_respiration_gc_m2'],sd['heterotrophic_respiration_gc_m2'],pd['root_respiration_gc_m2'],DAY)
        total_gpp+=gpp;total_resp+=exchange['ecosystem_respiration_gc_m2'];total_deposition+=.001
        cr=plant.total_carbon+soil.carbon.sum()-c0-total_gpp+total_resp
        nr=plant.total_nitrogen+soil.nitrogen.sum()+soil.mineral_nitrogen-n0-total_deposition
        for d in (fall,sd,pd):c_bound+=d['carbon_bound_gc_m2'];n_bound+=d['nitrogen_bound_gn_m2']
        assert abs(cr)<=c_bound and abs(nr)<=n_bound
        max_c=max(max_c,abs(cr));max_n=max(max_n,abs(nr))
    print('Plant soil two-year C/N ledger',dict(carbon=max_c,nitrogen=max_n,carbon_bound=c_bound,nitrogen_bound=n_bound,
        cumulative_gpp=total_gpp,cumulative_respiration=total_resp,leaf_n_factor=leaf_nitrogen_factor(plant,p)),flush=True)

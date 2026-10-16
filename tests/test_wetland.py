"""Independent ODE, pressure threshold and actual decomposition C/N ledgers."""
from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from noahmp_clm.physics.wetland import WetlandState,WetlandParameters,step_wetland
from noahmp_clm.physics.decomposition import DecompositionState,step_decomposition


def test_saturated_partition_and_pressure_threshold_have_physical_units():
    p=replace(WetlandParameters(),aqueous_diffusivity_m2s=0.,gaseous_diffusivity_m2s=0.,oxidation_rate_s=0.)
    after,d=step_wetland(WetlandState(),6.,295.15,.45,0.,.45,101325.,1800.,parameters=p)
    # CH4:CO2 = 0.2:1 in the fully anaerobic donor; 6 gC gives 1 gC CH4.
    assert d['methane_production_gc_m2']==pytest.approx(1.,abs=2e-15)
    threshold=.45*.3*.035*.15*101325./(8.31446261815324*295.15)*12.
    assert after.methane_gc_m2==pytest.approx(threshold,abs=2e-15)
    assert d['ebullition_gc_m2']==pytest.approx(1.-threshold,abs=2e-15)
    assert d['co2_gc_m2']==5. and d['diffused_gc_m2']==d['oxidized_gc_m2']==0.
    assert d['methane_emission_mgch4_m2']==pytest.approx(d['ebullition_gc_m2']*16./12.*1000.)


def numerical_reference(initial,donor,t,theta,porosity,pressure,dt,p):
    # Independent concentration/flux construction and DOP853 integration.
    air=porosity-theta;water_volume=theta*p.active_depth_m
    gas_volume=air*p.active_depth_m
    aqueous_capacity=water_volume+gas_volume/p.henry_water_air
    anoxic=max(0.,(theta/porosity-p.anaerobic_saturation)/(1.-p.anaerobic_saturation))
    methane_ratio=p.methane_co2_ratio*math.exp(math.log(p.methane_q10/p.donor_q10)*(t-p.production_reference_k)/10.)
    produced=donor*anoxic*methane_ratio/(1.+methane_ratio)
    production_rate=produced/dt
    oxidation=p.oxidation_rate_s*(1.-anoxic)*math.exp(math.log(p.oxidation_q10)*(t-p.oxidation_reference_k)/10.)
    liquid_transport=p.aqueous_diffusivity_m2s*theta**2
    gas_transport=p.gaseous_diffusivity_m2s*air**(10./3.)/porosity**2
    def rhs(time,y):
        water_concentration=y[0]/aqueous_capacity
        gas_concentration=water_concentration/p.henry_water_air
        surface_flux=(liquid_transport*water_concentration+gas_transport*gas_concentration)/(.5*p.active_depth_m)
        reaction=oxidation*y[0]
        return [production_rate-reaction-surface_flux,reaction,surface_flux]
    answer=solve_ivp(rhs,[0.,dt],[initial,0.,0.],method='DOP853',rtol=2e-12,atol=1e-13)
    assert answer.success
    end,oxidized,diffused=answer.y[:,-1]
    max_concentration=12.*p.henry_water_air*p.ebullition_pressure_fraction*pressure/(8.31446261815324*t)
    bubble=max(end-max_concentration*aqueous_capacity,0.) if anoxic>0. else 0.
    return np.array([end-bubble,oxidized,diffused,bubble,donor-produced+oxidized])


def test_bulk_reaction_transport_matches_independent_high_accuracy_ode():
    errors=[];relative=[];p=WetlandParameters()
    for t in (275.,285.,295.15):
        for saturation in (.5,.85,1.):
            for initial in (.001,1.,100.):
                for dt in (900.,86400.):
                    after,d=step_wetland(WetlandState(initial),.7,t,.45*saturation,0.,.45,85000.,dt,parameters=p)
                    ref=numerical_reference(initial,.7,t,.45*saturation,.45,85000.,dt,p)
                    got=np.array([after.methane_gc_m2,d['oxidized_gc_m2'],d['diffused_gc_m2'],d['ebullition_gc_m2'],d['co2_gc_m2']])
                    delta=abs(got-ref);errors.append(float(delta.max()))
                    relative.extend((delta[abs(ref)>1e-6]/abs(ref[abs(ref)>1e-6])).tolist())
                    np.testing.assert_allclose(got,ref,rtol=1e-8,atol=1e-10)
    print('wetland 54 independent ODE cases max gC/m2 / relative',max(errors),max(relative),flush=True)


def test_small_rate_limit_keeps_small_export_and_preserves_carbon():
    p=replace(WetlandParameters(),aqueous_diffusivity_m2s=1e-25,gaseous_diffusivity_m2s=0.,oxidation_rate_s=0.)
    after,d=step_wetland(WetlandState(.01),1e-8,295.15,.45,0.,.45,101325.,1.,parameters=p)
    assert 0.<d['diffused_gc_m2']<1e-20
    assert after.methane_gc_m2==pytest.approx(.01+1e-8/6.,abs=2e-18)
    assert abs(d['carbon_residual_gc_m2'])<=d['carbon_bound_gc_m2']


def test_freeze_retains_trapped_mass_and_thaw_releases_it_without_new_carbon():
    frozen,d=step_wetland(WetlandState(10.),1.,270.,0.,.45,.45,101325.,86400.)
    assert frozen.methane_gc_m2==10. and d['co2_gc_m2']==1.
    assert d['methane_production_gc_m2']==d['methane_emission_gc_m2']==0.
    thawed,d=step_wetland(frozen,0.,285.,.45,0.,.45,101325.,86400.)
    assert d['ebullition_gc_m2']>9.
    assert thawed.methane_gc_m2+d['methane_emission_gc_m2']+d['co2_gc_m2']==pytest.approx(10.,abs=1e-14)


def test_decomposition_donor_is_not_emitted_twice_over_a_seasonal_year():
    c=np.array([100.,200.,200.,500.,1000.,5000.,10000.]);n=c/np.array([30.,50.,100.,150.,8.,11.,11.])
    soil=DecompositionState(c,n,10.);methane=WetlandState();initial=float(c.sum());nitrogen=float(n.sum()+10.)
    co2=ch4=0.;max_step=0.;max_cumulative=0.
    for day in range(365):
        t=285.+12.*math.sin(2.*math.pi*day/365.);saturation=.75+.25*math.sin(math.pi*day/365.)
        soil,d=step_decomposition(soil,t,-.001,86400.,oxygen_factor=.2)
        donor=d['heterotrophic_respiration_gc_m2']
        before=methane.methane_gc_m2
        methane,w=step_wetland(methane,donor,t,.45*saturation,0.,.45,101325.,86400.)
        step=methane.methane_gc_m2-before+w['co2_gc_m2']+w['methane_emission_gc_m2']-donor
        max_step=max(max_step,abs(step));co2+=w['co2_gc_m2'];ch4+=w['methane_emission_gc_m2']
        residual=float(soil.carbon.sum())+methane.methane_gc_m2+co2+ch4-initial
        max_cumulative=max(max_cumulative,abs(residual))
        assert abs(step)<=w['carbon_bound_gc_m2']
        assert abs(residual)<=128.*np.finfo(float).eps*(initial+co2+ch4)*365.
        assert abs(float(soil.nitrogen.sum()+soil.mineral_nitrogen)-nitrogen)<1e-10
    assert ch4>0. and co2>0.
    print('wetland decomposition year',dict(co2_gc_m2=co2,ch4_gc_m2=ch4,stored_gc_m2=methane.methane_gc_m2,
        max_step_residual=max_step,max_cumulative_residual=max_cumulative),flush=True)


def test_invalid_wetland_inputs_fail_explicitly():
    with pytest.raises(ValueError):WetlandState(-1.)
    with pytest.raises(ValueError):WetlandParameters(henry_water_air=0.)
    with pytest.raises(ValueError):WetlandParameters(anaerobic_saturation=1.)
    args=(WetlandState(),1.,285.,.3,0.,.45,101325.,1800.)
    with pytest.raises(ValueError):step_wetland(*args[:3],.5,*args[4:])
    with pytest.raises(ValueError):step_wetland(args[0],-1.,*args[2:])

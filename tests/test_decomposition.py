"""Independent continuous-time and analytic C/N checks for the simplified cascade."""
from dataclasses import replace
from decimal import Decimal,localcontext
import numpy as np
import pytest
from scipy.linalg import expm
from noahmp_clm.physics.decomposition import (DecompositionState as State,DecompositionParameters as Parameters,
    step_decomposition,environment_scalars,YEAR)


def independent_generator(p):
    # Independently assemble the continuous first-order ODE from pathway
    # fractions in the published CENTURY scheme, not production cascade().
    matrix=-np.diag(1./(np.array(p.turnover_years)*YEAR))
    weights={(4,0):.45,(4,1):.5,(5,2):.5,(1,3):p.wood_cellulose_fraction,
        (2,3):1.-p.wood_cellulose_fraction,(5,4):.826-.68*p.sand_fraction,
        (6,4):.004,(4,5):.42,(6,5):.03,(4,6):.45}
    for (receiver,donor),value in weights.items():matrix[receiver,donor]+=value/(p.turnover_years[donor]*YEAR)
    return matrix


def test_environment_matches_decimal_and_clapeyron_bounds():
    maximum=0.
    with localcontext() as ctx:
        ctx.prec=60
        for t in (250.,268.,273.15,280.,288.15,298.15,308.15,320.):
            for psi in (-10.,-2.5,-.5,-.05,-.002,0.):
                a,b,c=environment_scalars(t,psi)
                td=Decimal(str(t));pd=min(Decimal(str(psi)),-Decimal(334000)*max(Decimal('273.15')-td,Decimal(0))/(1000*td))
                temp=Decimal('1.5')**((td-Decimal('298.15'))/10)
                moisture=Decimal(0) if pd<=Decimal('-2.5') else Decimal(1) if pd>=Decimal('-.002') else (Decimal('-2.5')/pd).ln()/Decimal(1250).ln()
                err=max(abs(a-float(temp)),abs(b-float(moisture)),abs(c-float(temp*moisture)));maximum=max(maximum,err)
        assert maximum<1e-13
    assert environment_scalars(250.,0.)[1]==0.
    print('Decomposition environmental max absolute',maximum,flush=True)


@pytest.mark.parametrize('dt',[1800.,3600.,86400.])
def test_carbon_pools_and_respiration_against_independent_matrix_exponential(dt):
    rng=np.random.default_rng(4103);worst_pool=0.;worst_rh=0.
    for sand in (0.,.4,1.):
        p=Parameters(sand_fraction=sand);matrix=independent_generator(p)
        for _ in range(12):
            old=rng.uniform(100.,5000.,7);source=rng.uniform(0.,1.,7)
            state=State(old,old/25.,1000.)
            new,d=step_decomposition(state,298.15,-.002,dt,carbon_input=source,nitrogen_input=source/25.,parameters=p)
            expected=expm(matrix*dt)@(old+source)
            rh=(old+source).sum()-expected.sum()
            bias=float(np.max(abs(new.carbon-expected)/expected));rh_bias=abs(d['heterotrophic_respiration_gc_m2']-rh)/rh
            worst_pool=max(worst_pool,bias);worst_rh=max(worst_rh,rh_bias)
            assert bias<.1 and rh_bias<.1
            assert abs(new.carbon.sum()-old.sum()-source.sum()+d['heterotrophic_respiration_gc_m2'])<=d['carbon_bound_gc_m2']
            assert abs(new.nitrogen.sum()+new.mineral_nitrogen-state.nitrogen.sum()-state.mineral_nitrogen-(source/25.).sum())<=d['nitrogen_bound_gn_m2']
    print('Decomposition matrix-exponential',dt,dict(pool_relative=worst_pool,respiration_relative=worst_rh),flush=True)


def test_nitrogen_limited_litter_matches_analytic_budget_and_competition():
    carbon=np.zeros(7);carbon[0]=100.;nitrogen=carbon/100.
    dt=1800.;decay=100.*(-np.expm1(-dt/(.066*YEAR)))
    potential_n=decay*(.45/8.-.01)
    supply=.2*potential_n;plant_demand=.5*potential_n;fraction=supply/(potential_n+plant_demand)
    state=State(carbon,nitrogen,supply)
    result,d=step_decomposition(state,298.15,-.002,dt,plant_n_demand=plant_demand)
    assert d['nitrogen_limitation']==pytest.approx(fraction,abs=1e-15)
    assert result.carbon[0]==pytest.approx(100.-decay*fraction,abs=1e-13)
    assert result.carbon[4]==pytest.approx(.45*decay*fraction,abs=1e-14)
    assert result.nitrogen[4]==pytest.approx(result.carbon[4]/8.,abs=1e-15)
    assert result.mineral_nitrogen<1e-17
    assert d['plant_uptake_gn_m2']==pytest.approx(plant_demand*fraction,abs=1e-17)
    zero,_=step_decomposition(State(carbon,nitrogen,0.),298.15,-.002,dt)
    np.testing.assert_array_equal(zero.carbon,carbon)
    print('Decomposition N competition',d['nitrogen_limitation'],d['carbon_residual_gc_m2'],d['nitrogen_residual_gn_m2'],flush=True)


def test_mineralization_needs_no_external_n_and_new_n_waits_for_next_step():
    carbon=np.array([100.,100.,100.,100.,100.,100.,100.]);nitrogen=carbon/2.
    state=State(carbon,nitrogen,0.)
    new,d=step_decomposition(state,298.15,-.002,1800.,plant_n_demand=1.)
    assert d['gross_mineralization_gn_m2']>0. and d['immobilization_gn_m2']==0.
    assert d['plant_uptake_gn_m2']==0.
    assert new.mineral_nitrogen==d['gross_mineralization_gn_m2']
    assert abs(new.nitrogen.sum()+new.mineral_nitrogen-nitrogen.sum())<1e-12


def test_dry_or_anoxic_zero_rate_keeps_organic_stores_and_tracks_inputs():
    c=np.arange(1.,8.)*100.;old=State(c,c/30.,.3)
    for psi,oxygen in ((-3.,1.),(-.002,0.)):
        new,d=step_decomposition(old,298.15,psi,1800.,oxygen_factor=oxygen,mineral_n_input=.1,plant_n_demand=.2)
        np.testing.assert_array_equal(new.carbon,c);np.testing.assert_array_equal(new.nitrogen,c/30.)
        assert new.mineral_nitrogen==pytest.approx(.2) and d['heterotrophic_respiration_gc_m2']==0.
    with pytest.raises(ValueError):old.carbon[0]=0.
    with pytest.raises(ValueError):step_decomposition(old,298.15,.1,1800.)
    with pytest.raises(ValueError):step_decomposition(old,298.15,-.1,0.)
    with pytest.raises(ValueError):Parameters(soil_cn=(0.,11.,11.))


def test_long_spinup_daily_operator_has_equilibrium_and_small_century_drift():
    # At unlimited N and constant climate, the implemented daily C map is
    # linear. Construct its exact operator by advancing basis stocks through
    # the actual model, then propagate 5000 years by matrix powers. This tests
    # the model's discrete equilibrium, not a fabricated replacement tendency.
    p=Parameters();dt=86400.;source=np.array([.2,.2,.1,.1,0.,0.,0.])
    columns=[]
    for k in range(7):
        basis=np.eye(7)[k];next_state,_=step_decomposition(State(basis,basis/25.,100.),298.15,-.002,dt,parameters=p)
        columns.append(next_state.carbon)
    transition=np.stack(columns,axis=1);input_term=transition@source
    equilibrium=np.linalg.solve(np.eye(7)-transition,input_term)
    augmented=np.eye(8);augmented[:7,:7]=transition;augmented[:7,7]=input_term
    x=np.linalg.matrix_power(augmented,365*5000)@np.r_[np.zeros(7),1.]
    later=np.linalg.matrix_power(augmented,365*100)@x
    drift=float(np.max(abs(later[:7]-x[:7])/x[:7]));error=float(np.max(abs(x[:7]-equilibrium)/equilibrium))
    assert drift<.05 and error<.01
    continuous=np.linalg.solve(-independent_generator(p),source/dt)
    assert np.max(abs(equilibrium-continuous)/continuous)<.1
    print('Decomposition 5000-year constant-climate spinup',dict(century_drift=drift,equilibrium_relative=error,
        continuous_equilibrium_relative=float(np.max(abs(equilibrium-continuous)/continuous)),pools_gc_m2=x[:7].tolist()),flush=True)

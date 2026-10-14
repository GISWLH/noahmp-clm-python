"""Five-layer state checks against extensive budgets and analytic phase states."""
import numpy as np
import pytest

from noahmp_clm.physics.snowpack import (Snowpack, equilibrium, target_thickness,
    regrid, remap_layers, add_snowfall, add_rain, drain_liquid, remove_sublimation,
    TF, CI, CW, LF)


def assert_budget(before, after, mass_out=0., heat_out=0., impurity_out=0.):
    assert after.mass.sum()+mass_out == pytest.approx(before.mass.sum(),abs=2e-12)
    assert after.enthalpy.sum()+heat_out == pytest.approx(before.enthalpy.sum(),abs=2e-7)
    np.testing.assert_allclose(after.impurities.sum(axis=0)+impurity_out,
        before.impurities.sum(axis=0),rtol=2e-13,atol=1e-20)


@pytest.mark.parametrize('depth,n',[ (0.,0),(.0001,1),(.03,1),(.04,2),(.07,2),
    (.12,3),(.18,3),(.29,4),(.41,4),(.64,5),(2.,5)])
def test_depth_grid(depth,n):
    z=target_thickness(depth)
    assert z.size==n
    assert z.sum()==pytest.approx(depth,abs=1e-15)
    assert np.all(z>0.)


def test_analytic_mixed_phase_and_warm_water():
    mass=np.array([2.,2.,2.,2.,2.,0.])
    heat=np.array([-2*CI*10,0.,LF,2*LF,2*LF+2*CW*3,0.])
    t,ice,water=equilibrium(mass,heat)
    np.testing.assert_allclose(t,[TF-10,TF,TF,TF,TF+3,TF],rtol=0,atol=1e-13)
    np.testing.assert_array_equal(ice,[2.,2.,1.,0.,0.,0.])
    np.testing.assert_array_equal(water,[0.,0.,1.,2.,2.,0.])


def test_remap_preserves_all_extensive_contents_and_optical_area():
    rng=np.random.default_rng(2815)
    for _ in range(100):
        n=rng.integers(1,15)
        z=rng.uniform(.001,.3,n);m=z*rng.uniform(60,700,n)
        h=m*rng.uniform(-CI*30,LF*1.1,n)
        r=rng.uniform(30,1500,n);a=m[:,None]*rng.uniform(0.,1e-5,(n,5))
        p=remap_layers(m,h,z,r,a,target_thickness(z.sum()))
        np.testing.assert_allclose([p.mass.sum(),p.enthalpy.sum(),p.thickness.sum(),
            np.sum(p.mass/p.radius_um)], [m.sum(),h.sum(),z.sum(),np.sum(m/r)],rtol=2e-13,atol=1e-7)
        np.testing.assert_allclose(p.impurities.sum(0),a.sum(0),rtol=2e-13,atol=1e-19)


def test_snowfall_cold_content_and_immutability():
    old=regrid(Snowpack.from_ice([120.],[TF-15],300.))
    new=add_snowfall(old,3.,TF-2,100.,100.,[1e-8,2e-8,3e-8,4e-8,5e-8])
    assert new.mass.sum()==pytest.approx(123.)
    assert new.enthalpy.sum()==pytest.approx(old.enthalpy.sum()-3*CI*2)
    assert new.temperature[0]>new.temperature[-1]
    assert not np.shares_memory(new.mass,old.mass)
    with pytest.raises(ValueError): new.mass[0]=0.


def test_rain_freezes_using_cold_content():
    old=Snowpack.from_ice([10.],TF-10.,300.)
    mixed=add_rain(old,.2,TF)
    expected_h=-10*CI*10+.2*LF
    assert mixed.enthalpy[0]==pytest.approx(expected_h)
    assert mixed.liquid[0]==0.
    assert mixed.temperature[0]==pytest.approx(TF+expected_h/(10.2*CI))
    new,water,heat,tracer=drain_liquid(mixed)
    assert water==0.
    assert_budget(mixed,new,water,heat,tracer)


def test_full_melt_exports_all_excess_energy_and_tracers():
    m=np.array([1.,2.,3.]);a=np.arange(15).reshape(3,5)*1e-9
    old=Snowpack(m,m*(LF+CW*4.),[.02,.04,.06],impurities=a)
    new,water,heat,tracer=drain_liquid(old)
    assert new.mass.size==0
    assert water==6.
    assert heat==pytest.approx(6*(LF+CW*4.))
    assert_budget(old,new,water,heat,tracer)


def test_meltwater_refreezes_in_lower_cold_layer():
    old=Snowpack([1.,10.],[LF+CW*2,-10*CI*20],[.01,.1])
    new,water,heat,tracer=drain_liquid(old)
    assert water==0.
    assert new.liquid.sum()==0.
    assert np.max(new.temperature)<TF
    assert_budget(old,new,water,heat,tracer)


def test_drain_sublimation_budgets_random():
    rng=np.random.default_rng(819)
    for _ in range(100):
        m=rng.uniform(.1,50.,5);z=m/rng.uniform(50.,700.,5)
        h=m*rng.uniform(-CI*30,LF*1.1,5)
        a=m[:,None]*rng.uniform(0.,1e-6,(5,5))
        old=Snowpack(m,h,z,impurities=a)
        p,w,e,imp=drain_liquid(old)
        assert_budget(old,p,w,e,imp)
        demand=rng.uniform()*p.ice.sum()
        sub,energy,residue=remove_sublimation(p,demand)
        assert_budget(p,sub,demand,energy,residue)


def test_full_sublimation_returns_deposited_residue():
    old=Snowpack([1.,2.],[-CI*10,-2*CI*20],[.01,.02],impurities=np.ones((2,5))*1e-8)
    new,heat,tracer=remove_sublimation(old,3.)
    assert not new.mass.size
    assert_budget(old,new,3.,heat,tracer)


def test_invalid_state_and_excess_sublimation():
    with pytest.raises(ValueError): Snowpack([-1.],[0.],[.1])
    with pytest.raises(ValueError): Snowpack(np.ones(6),np.zeros(6),np.ones(6))
    with pytest.raises(ValueError): add_rain(Snowpack.empty(),1.,TF)
    with pytest.raises(ValueError): remove_sublimation(Snowpack.from_ice([1.],TF),2.)


def test_ice_barrier_perching_and_overflow_are_not_soil_drainage():
    # Upper layer: 1 kg ice + 10 kg liquid. Lower layer: impermeable ice.
    old=Snowpack([11.,9.17],[10*LF,0.],[.003,.01],impurities=np.ones((2,5))*1e-8)
    new,d=drain_liquid(old,return_ledger=True)
    assert d['drainage_kgm2']==0.
    assert d['runoff_kgm2']>8.
    assert new.liquid.sum()>0.
    assert_budget(old,new,d['runoff_kgm2'],d['runoff_enthalpy_jm2'],d['runoff_impurities_kgm2'])


def test_receiving_cell_capacity_accounts_for_freezing_expansion():
    old=Snowpack([10.,5.],[10*LF,-5*CI*20],[.02,.008])
    new,d=drain_liquid(old,return_ledger=True)
    # The melted upper cell vanishes; some of its water cannot fit below.
    assert d['runoff_kgm2']>0.
    assert np.all(new.ice/917.+new.liquid/1000.<=new.thickness+1e-13)
    assert_budget(old,new,d['runoff_kgm2']+d['drainage_kgm2'],
        d['runoff_enthalpy_jm2']+d['drainage_enthalpy_jm2'],
        d['runoff_impurities_kgm2']+d['drainage_impurities_kgm2'])

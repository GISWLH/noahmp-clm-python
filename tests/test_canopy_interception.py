"""Phase, parcel-energy and CLM interception reference checks."""
import numpy as np
import pytest
from noahmp_clm.physics.canopy_interception import CanopyCondensate,intercept_precipitation,exchange_vapor,coverage
from noahmp_clm.physics.snowpack import TF,CI,CW,LF


def test_phase_plateau_freeze_and_complete_melt_preserve_all_heat():
    original=CanopyCondensate.from_temperature([2.,2.,2.],[TF-10.,TF,TF+3.],liquid_fraction=[0.,.5,1.])
    np.testing.assert_allclose(original.enthalpy,[-42000.,334000.,693080.],rtol=0.,atol=1e-9)
    np.testing.assert_allclose(original.ice,[2.,1.,0.],rtol=0.,atol=0.)
    # Same total heat in three independent reservoirs crosses both latent
    # plateau endpoints and leaves the fully melted cell warm.
    heated=original.with_enthalpy(original.enthalpy+np.array([209000.,334000.,-735080.]))
    np.testing.assert_allclose(heated.temperature,[TF,TF,TF-10.],rtol=0.,atol=1e-12)
    np.testing.assert_allclose(heated.liquid,[.5,2.,0.],rtol=0.,atol=1e-14)
    with pytest.raises(ValueError):original.mass[0]=0.
    with pytest.raises(ValueError):CanopyCondensate([0.],[1.])
    with pytest.raises(ValueError):CanopyCondensate.from_temperature([1.],TF-1.,liquid_fraction=1.)


def test_clm_bulk_snow_interception_and_wind_unloading_reference():
    # CLM5 2.7.5, 2.7.14: L+S=3, fv=.8, 2 kg snow, T=265 K,
    # wind=3 m/s, dt=1800 s. Independent 50-digit Decimal evaluation.
    final,d=intercept_precipitation(CanopyCondensate.empty(),0.,2.,265.,1800.,[.3]*10,vegetated_fraction=.8,wind=3.)
    assert d['intercepted_snow_kgm2']==pytest.approx(1.2429917437625123,rel=2e-14)
    assert d['canopy_unloading_kgm2'].sum()==pytest.approx(.0430266372840870,rel=2e-14)
    assert final.mass.sum()==pytest.approx(1.1999651064784253,rel=2e-14)
    assert d['ground_ice_kgm2']==pytest.approx(.8000348935215747,rel=2e-14)
    assert abs(d['energy_residual_jm2'])<1e-9
    np.testing.assert_allclose(final.temperature,265.,rtol=0.,atol=1e-12)


def test_liquid_capacity_throughfall_and_warm_drip_heat():
    final,d=intercept_precipitation(CanopyCondensate.empty(1),10.,0.,TF+5.,1800.,[3.],vegetated_fraction=.8)
    # Capacity .1*3*.8 = .24 kg, all other rain must reach the soil. There
    # is no hidden capacity loss, and exported water keeps its 5 K warmth.
    assert final.mass[0]==pytest.approx(.24)
    assert d['ground_liquid_kgm2']==pytest.approx(9.76)
    assert final.enthalpy[0]==pytest.approx(85176.)
    assert d['ground_liquid_enthalpy_jm2']==pytest.approx(3463824.)
    assert abs(d['water_residual_kgm2'])<1e-13
    assert abs(d['energy_residual_jm2'])<1e-8


def test_cold_content_freezes_intercepted_rain_without_unaccounted_heat():
    cold=CanopyCondensate.from_temperature([10.],TF-10.)
    result,d=intercept_precipitation(cold,.1,0.,TF,1800.,[3.],wind=0.)
    assert result.liquid[0]==0.
    expected=10.+d['intercepted_rain_kgm2']-d['canopy_unloading_kgm2'][0]
    assert result.mass[0]==pytest.approx(expected)
    assert result.temperature[0]<TF
    assert abs(d['energy_residual_jm2'])<1e-9


def test_zero_vegetation_sheds_old_phases_and_all_new_precipitation():
    old=CanopyCondensate.from_temperature([1.,2.,3.],[TF-5.,TF,TF+2.],liquid_fraction=[0.,.5,1.])
    final,d=intercept_precipitation(old,4.,5.,270.,1800.,[1.,1.,1.],vegetated_fraction=0.,wind=8.)
    np.testing.assert_array_equal(final.mass,0.)
    np.testing.assert_array_equal(final.enthalpy,0.)
    assert d['ground_liquid_kgm2']==8.
    assert d['ground_ice_kgm2']==7.
    assert abs(d['water_residual_kgm2'])<1e-13
    assert abs(d['energy_residual_jm2'])<1e-8


def test_layer_partition_invariance_for_uniform_water_per_exposed_area():
    area=np.array([.05,.1,.15,.2,.25,.3,.35,.4,.5,.7]);mass=area*2.
    layered=CanopyCondensate.from_temperature(mass,TF-5.)
    bulk=CanopyCondensate.from_temperature([mass.sum()],TF-5.)
    for rain,snow in ((0.,3.),(.2,0.),(5.,3.)):
        many,dm=intercept_precipitation(layered,rain,snow,274.,1800.,area,vegetated_fraction=.7,wind=4.)
        one,do=intercept_precipitation(bulk,rain,snow,274.,1800.,[area.sum()],vegetated_fraction=.7,wind=4.)
        np.testing.assert_allclose([many.mass.sum(),many.enthalpy.sum()],[one.mass.sum(),one.enthalpy.sum()],rtol=3e-14,atol=1e-9)
        for key in ('ground_liquid_kgm2','ground_ice_kgm2','ground_liquid_enthalpy_jm2','ground_ice_enthalpy_jm2'):
            assert dm[key]==pytest.approx(do[key],rel=2e-13,abs=1e-9)


def test_unloading_cannot_exceed_ice_even_at_extreme_wind_or_long_step():
    old=CanopyCondensate.from_temperature([3.],260.)
    final,d=intercept_precipitation(old,0.,0.,280.,86400.,[1.],wind=80.)
    assert final.mass[0]==0.
    assert d['ground_ice_kgm2']==3.
    assert d['ground_ice_enthalpy_jm2']==old.enthalpy[0]
    assert d['canopy_unloading_kgm2'][0]==3.


def test_phase_vapour_parcels_and_deposition_at_the_canopy_freezing_point():
    old=CanopyCondensate.from_temperature([2.,2.,2.,0.],[TF-4.,TF,TF+3.,TF],liquid_fraction=[0.,.5,1.,0.])
    final,d=exchange_vapor(old,[.5,.5,3.,-.1],leaf_temperature=[TF-4.,TF,TF+3.,TF])
    np.testing.assert_allclose(d['ice_kgm2'],[.5,.5,0.,-.1],rtol=0.,atol=1e-14)
    np.testing.assert_allclose(d['liquid_kgm2'],[0.,0.,2.,0.],rtol=0.,atol=1e-14)
    np.testing.assert_allclose(d['latent_energy_jm2'],[1417000.,1417000.,5000000.,-283400.],rtol=0.,atol=1e-9)
    np.testing.assert_allclose(d['advected_enthalpy_jm2'],[-4200.,0.,693080.,0.],rtol=0.,atol=1e-8)
    np.testing.assert_allclose(final.mass+d['mass_kgm2'],old.mass,rtol=0.,atol=1e-14)
    np.testing.assert_allclose(final.enthalpy+d['advected_enthalpy_jm2'],old.enthalpy,rtol=0.,atol=1e-9)
    assert final.mass[2]==0. and final.enthalpy[2]==0.
    assert final.ice[3]==.1 and d['unmet_evaporation_kgm2'][2]==1.


def test_wet_dry_and_snow_coverage_use_exposed_area_and_phase():
    state=CanopyCondensate.from_temperature([.1,0.,6.],[TF,TF,TF-1.],liquid_fraction=[1.,0.,0.])
    f=coverage(state,[.5,.5,1.],stem_area=[.5,.5,0.])
    np.testing.assert_allclose(f['wet_fraction'],[.05,0.,.05],rtol=0.,atol=1e-14)
    np.testing.assert_allclose(f['dry_fraction'],[.475,.5,.95],rtol=0.,atol=1e-14)
    np.testing.assert_allclose(f['snow_fraction'],[0.,0.,1.],rtol=0.,atol=1e-14)


def test_random_transfers_keep_mass_heat_and_phase_capacity():
    rng=np.random.default_rng(80817);maximum=np.zeros(2)
    for _ in range(200):
        area=rng.uniform(0.,1.,10);mass=rng.uniform(0.,1.,10);t=rng.uniform(255.,290.,10)
        state=CanopyCondensate.from_temperature(mass,t);fv=rng.uniform(0.,1.)
        final,d=intercept_precipitation(state,*rng.uniform(0.,10.,2),rng.uniform(255.,290.),1800.,area,vegetated_fraction=fv,wind=rng.uniform(0.,20.))
        assert np.all(final.liquid<=.1*fv*area+1e-12)
        assert np.all(final.ice<=6.*fv*area+1e-12)
        maximum=np.maximum(maximum,[abs(d['water_residual_kgm2']),abs(d['energy_residual_jm2'])])
        vapor,ledger=exchange_vapor(final,rng.uniform(-.01,.1,10),leaf_temperature=t)
        np.testing.assert_allclose(vapor.mass+ledger['mass_kgm2'],final.mass,rtol=1e-13,atol=1e-14)
        np.testing.assert_allclose(vapor.enthalpy+ledger['advected_enthalpy_jm2'],final.enthalpy,rtol=1e-13,atol=1e-8)
    assert maximum[0]<2e-13 and maximum[1]<2e-8
    print('canopy interception random budgets',maximum.tolist())

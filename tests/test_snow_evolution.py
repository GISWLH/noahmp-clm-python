import numpy as np
import pytest
from noahmp_clm.physics.snowpack import Snowpack, TF, LF
from noahmp_clm.physics.snow_evolution import (fresh_snow_properties, compact,
    aging_parameters, grain_growth, deposit_impurities)
from noahmp_clm.physics.snicar import snow_tables


def test_new_snow_and_load_above_not_below():
    density,radius=fresh_snow_properties([TF-30,TF-15,TF,TF+2],0.)
    np.testing.assert_allclose(radius,[54.5,129.5,204.5,204.5])
    assert np.all((density>=30)&(density<300))
    assert fresh_snow_properties(TF-10,10.)[0]>fresh_snow_properties(TF-10,0.)[0]
    p=Snowpack.from_ice([10.,10.,10.],TF-10.,300.)
    q=compact(p,1800.,remap=False)
    assert q.thickness[0]>q.thickness[1]>q.thickness[2]
    np.testing.assert_array_equal(p.mass,q.mass)
    np.testing.assert_array_equal(p.enthalpy,q.enthalpy)


def test_compaction_melt_and_pore_volume_long_step():
    p=Snowpack([10.,50.],[LF*2.,LF*20.],[.03,.08])
    q=compact(p,1e8,ice_before=[10.,50.],remap=False)
    np.testing.assert_allclose(q.mass,p.mass,rtol=0,atol=0)
    np.testing.assert_allclose(q.enthalpy,p.enthalpy,rtol=0,atol=0)
    assert np.all(q.thickness>=q.ice/917.+q.liquid/1000.)
    assert np.all(q.thickness<=p.thickness)


def test_lookup_against_scipy_interpolator():
    interpolate=pytest.importorskip('scipy.interpolate').RegularGridInterpolator
    rng=np.random.default_rng(55);table=snow_tables()
    point=np.column_stack((rng.uniform(223.15,273.15,100),rng.uniform(0,300,100),rng.uniform(50,400,100)))
    got=aging_parameters(*point.T)
    grid=tuple(table['aging_'+x] for x in ('T','dTdz','sno_dns'))
    for values,key in zip(got,('tau','kappa','drdsdt0')):
        ref=interpolate(grid,table['aging_'+key])(point)
        np.testing.assert_allclose(values,ref,rtol=1e-12,atol=1e-10)


def test_grain_growth_independent_time_integral():
    ivp=pytest.importorskip('scipy.integrate').solve_ivp
    # Fixed physical conditions; compare independent adaptive ODE integration
    # of simultaneous wet+dry growth to the production split exact operators.
    for temperature,gradient,density,radius,liquid_fraction in (
        (TF-15,0.,300.,54.5,0.),(TF-3,100.,100.,100.,0.),
        (TF,50.,200.,54.5,.1),(TF,150.,350.,500.,.033)):
        h=10*LF*liquid_fraction if liquid_fraction else 10*2100*(temperature-TF)
        p=Snowpack([10.],[h],[10/density],[radius])
        tau,kappa,rate=(float(x[0]) for x in aging_parameters([temperature],[gradient],[density]))
        wet=1e18*4.22e-13*liquid_fraction**3/(4*np.pi)
        def rhs(_,r):
            return rate/3600*(tau/(r-54.5+tau))**(1/kappa)+wet/r**2
        ref=ivp(rhs,(0,1800),[radius],rtol=1e-11,atol=1e-11).y[0,-1]
        q=grain_growth(p,1800.,temperature_gradient=gradient)
        assert abs(q.radius_um[0]-ref)/ref<.01
        np.testing.assert_array_equal(q.enthalpy,p.enthalpy)
        np.testing.assert_array_equal(q.mass,p.mass)


def test_aging_full_table_domain_stays_finite():
    table=snow_tables()
    for temp in table['aging_T']:
        for gradient in (0.,150.,300.):
            for density in (50.,200.,400.):
                p=Snowpack.from_ice([10.],temp,density,54.5)
                q=grain_growth(p,86400.,temperature_gradient=gradient)
                assert 54.5<=q.radius_um[0]<=1500.


def test_deposition_and_refreezing_optical_area():
    p=Snowpack.from_ice([10.,20.],TF,300.,100.)
    q=grain_growth(p,0.,refrozen_mass=[5.,0.])
    assert q.radius_um[0]==pytest.approx(1/(.5/100+.5/1000))
    rates=np.arange(1,6)*1e-12
    r,bare=deposit_impurities(q,rates,1800.)
    np.testing.assert_allclose(r.impurities.sum(0),rates*1800,rtol=1e-15)
    assert not bare.any()
    _,bare=deposit_impurities(Snowpack.empty(),rates,1800.)
    np.testing.assert_allclose(bare,rates*1800,rtol=1e-15)

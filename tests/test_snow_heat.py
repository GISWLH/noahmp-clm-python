"""Independent analytic/SciPy references for the snow enthalpy solver."""
import numpy as np
import pytest

from noahmp_clm.physics.snowpack import Snowpack, TF, CI, CW, LF, drain_liquid
from noahmp_clm.physics.snow_heat import heat_step


@pytest.mark.parametrize('flux',[-300.,0.,100.,1000.])
def test_one_cell_analytic_flux_across_melt(flux):
    p=Snowpack.from_ice([2.],TF-5.,100.)
    q,diag=heat_step(p,1800.,flux)
    expect=-2*CI*5+1800*flux
    assert q.enthalpy[0]==pytest.approx(expect,abs=2e-8)
    assert abs(diag['residual_jm2'])<2e-8
    expected_t=(TF+expect/(2*CI) if expect<0 else TF if expect<=2*LF
        else TF+(expect-2*LF)/(2*CW))
    assert q.temperature[0]==pytest.approx(expected_t,abs=1e-12)
    if flux==1000.:
        snow,melt,outheat,_=drain_liquid(q)
        assert snow.mass.size==0 and melt==2.
        assert outheat>2*LF
        assert outheat-p.enthalpy.sum()==pytest.approx(1800*flux,abs=2e-8)


def test_heat_capacity_depends_on_mass_not_density():
    temperatures=[]
    for density in (80.,300.,650.):
        p=Snowpack.from_ice([3.],TF-10,density)
        new,_=heat_step(p,60.,100.)
        temperatures.append(new.temperature[0])
    np.testing.assert_allclose(temperatures,TF-10+6000/(3*CI),rtol=0,atol=1e-12)


def test_implicit_top_exchange_analytic_solution():
    p=Snowpack.from_ice([.2],TF-20.,100.)
    q,d=heat_step(p,1800.,10.,top_temperature=TF-10.,top_conductance=8.)
    expected_offset=(.2*CI*(-20.)+1800*(10.+8.*(-10.)))/(.2*CI+1800*8.)
    assert q.temperature[0]==pytest.approx(TF+expected_offset,abs=1e-12)
    assert d['top_flux_wm2']==pytest.approx(10.+8.*(-10.-expected_offset),abs=1e-11)
    assert abs(d['residual_jm2'])<1e-7


def test_two_cells_exact_backward_euler():
    # Different thickness and conductivity: equal harmonic averaging is wrong.
    m=np.array([3.,20.]);z=np.array([.03,.08]);k=np.array([.1,.6])
    old=Snowpack(m,m*CI*np.array([-15.,-5.]),z)
    dt=600.;g=1/(.015/.1+.04/.6)
    matrix=np.array([[3*CI+dt*g,-dt*g],[-dt*g,20*CI+dt*g]])
    expected=np.linalg.solve(matrix,m*CI*np.array([-15.,-5.])+[dt*10.,0.])
    new,d=heat_step(old,dt,10.,thermal_conductivity=k)
    np.testing.assert_allclose(new.temperature-TF,expected,rtol=0,atol=1e-12)
    assert abs(d['residual_jm2'])<1e-7


def independent_scipy_reference(pack,dt,top,absorbed,bottom_t,bottom_g,k):
    """MINPACK solves independently assembled balances with finite differences."""
    root=pytest.importorskip('scipy.optimize').root
    n=len(pack.mass)
    def resid(specific_h):
        t=[]
        for v in specific_h:
            t.append(v/2100. if v<0 else 0. if v<334000. else (v-334000.)/4180.)
        flux=[top]
        for j in range(n-1):
            resistance=pack.thickness[j]/(2*k[j])+pack.thickness[j+1]/(2*k[j+1])
            flux.append((t[j]-t[j+1])/resistance)
        flux.append(bottom_g*(t[-1]-(bottom_t-273.15)))
        return np.array([(specific_h[j]*pack.mass[j]-pack.enthalpy[j])/dt-
            (flux[j]-flux[j+1]+absorbed[j]) for j in range(n)])
    # Scaling to MJ/kg avoids a Kelvin/melting-plateau singular temperature solve.
    sol=root(lambda x:resid(x*1e6),pack.enthalpy/pack.mass/1e6,tol=1e-10)
    assert np.max(np.abs(resid(sol.x*1e6)))<1e-6,(sol.message,resid(sol.x*1e6))
    return sol.x*1e6*pack.mass


def test_mixed_phase_scipy_reference():
    rng=np.random.default_rng(22046)
    for _ in range(80):
        n=rng.integers(1,6)
        m=rng.uniform(.5,30,n);z=m/rng.uniform(80,600,n)
        h=m*rng.uniform(-CI*25,LF*1.1,n)
        p=Snowpack(m,h,z)
        dt=rng.uniform(30,3600);top=rng.uniform(-300,600)
        sw=rng.uniform(0,100,n);tb=rng.uniform(TF-15,TF+10);gb=rng.uniform(0,20)
        k=rng.uniform(.05,1.,n)
        new,d=heat_step(p,dt,top,absorbed=sw,bottom_temperature=tb,
            bottom_conductance=gb,thermal_conductivity=k)
        ref=independent_scipy_reference(p,dt,top,sw,tb,gb,k)
        np.testing.assert_allclose(new.enthalpy,ref,rtol=1e-9,atol=.003)
        assert abs(d['residual_jm2'])<2e-6
        assert np.max(np.abs(d['layer_residual_jm2']))<2e-6


def test_closed_column_no_spurious_uniform_drift():
    p=Snowpack.from_ice([2.,5.,11.,23.,80.],TF-23,[100.,180.,300.,410.,510.])
    for _ in range(100):
        p,d=heat_step(p,1800.,0.)
    np.testing.assert_allclose(p.temperature,TF-23,rtol=0,atol=5e-13)
    assert abs(d['residual_jm2'])<1e-8


def test_invalid_heat_inputs():
    p=Snowpack.from_ice([1.],TF)
    for kwargs in ({'dt':0.},{'dt':1800.,'thermal_conductivity':0.},
                   {'dt':1800.,'bottom_conductance':-1.}):
        with pytest.raises(ValueError): heat_step(p,top_flux=0.,**kwargs)

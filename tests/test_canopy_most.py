"""Independent differential-profile quadrature and manufactured flux checks."""
import math
import numpy as np
import pytest
from scipy.integrate import quad
from noahmp_clm.physics.canopy_most import (MOSTGeometry,profile_integral,
    flux_stability,canopy_exchange,CP,RD)


def test_integrated_profiles_against_published_differential_equations():
    # Integrate phi over log-height, independently of the analytic primitive.
    # Published phi constants differ slightly from rounded 1.14 and 0.8 in
    # published integrated forms. Record this approximation, not an exact fit.
    maximum=0.
    for momentum in (False,True):
        for zeta in (-100.,-10.,-1.574,-.465,-.01,0.,.01,.5,1.,2.):
            for ratio in (1.01,10.,1000.):
                height=10.;roughness=height/ratio;inv=zeta/height
                edge=-1.574 if momentum else -.465
                def phi(logz):
                    x=math.exp(logz)*inv
                    if x<edge:return (.7*.4**(2/3)*(-x)**(1/3) if momentum else .9*.4**(4/3)*(-x)**(-1/3))
                    if x<0:return (1.-16.*x)**(-.25 if momentum else -.5)
                    return 1.+5.*x if x<=1 else 5.+x
                points=[math.log(e/inv) for e in (edge,1.) if inv!=0 and roughness<e/inv<height]
                expected=quad(phi,math.log(roughness),math.log(height),points=points,epsabs=1e-11)[0]
                actual=profile_integral(height,roughness,inv,momentum=momentum)
                relative=abs(actual-expected)/expected;maximum=max(maximum,relative)
                assert relative<.01
    print('MOST 60 differential-profile cases max relative bias percent',100.*maximum,flush=True)


def test_neutral_distinct_measurement_heights_and_displacement():
    g=MOSTGeometry(wind_height=30.,temperature_height=25.,humidity_height=20.,displacement=12.,
                   momentum_roughness=1.,heat_roughness=.1,water_roughness=.2)
    result=canopy_exchange(0.,3.,g)
    expected=.4*3./math.log(18.)
    assert result['friction_velocity']==pytest.approx(expected,rel=1e-14)
    assert result['heat_velocity']==pytest.approx(.4*expected/math.log(130.),rel=1e-14)
    assert result['water_velocity']==pytest.approx(.4*expected/math.log(40.),rel=1e-14)
    assert result['water_velocity']>result['heat_velocity']


def test_actual_water_mass_buoyancy_and_stability_bounds():
    g=MOSTGeometry();t,q,p=280.,.003,90000.
    neutral=flux_stability(t,q,p,0.,0.,0.,.2,g)
    assert neutral['zeta']==0. and neutral['effective_wind']==1.
    moist=flux_stability(t,q,p,3.,0.,1e-4,.2,g)
    rho=p/(RD*t*(1.+(.378/.622)*q))
    expected=9.81*.61*1e-4/(rho*(1.+.61*q))
    assert moist['buoyancy']==pytest.approx(expected,rel=1e-14)
    assert moist['zeta']<0 and moist['convective_wind']>0
    # H and E cancel virtual buoyancy exactly, despite neither being zero.
    cancel=-.61*(t+.0098*g.temperature_height)*1e-4*CP/(1.+.61*q)
    balanced=flux_stability(t,q,p,3.,cancel,1e-4,.2,g)
    assert abs(balanced['raw_zeta'])<1e-14
    stable=flux_stability(t,q,p,3.,-100.,0.,.01,g)
    assert stable['raw_zeta']>2. and stable['zeta']==2. and stable['convective_wind']==0.
    unstable=flux_stability(t,q,p,0.,100.,0.,.01,g)
    assert unstable['raw_zeta']<-100. and unstable['zeta']==-100.


def test_invalid_geometry_is_not_silently_clipped():
    for kwargs in (dict(displacement=10.),dict(heat_roughness=10.),dict(water_roughness=0.),
                   dict(wind_height=np.nan),dict(boundary_layer_height=0.)):
        with pytest.raises(ValueError):MOSTGeometry(**kwargs)
    with pytest.raises(ValueError):profile_integral(1.,2.,0.)


def test_stable_feedback_brackets_a_nearly_noncontracting_map():
    from noahmp_clm.physics.canopy_energy import CanopyExchange
    from noahmp_clm.physics.canopy_most import solve_surface_most
    geometry=MOSTGeometry();wind=2.5;t=280.;q=.003;p=101325.
    exchange=CanopyExchange(t,q,p,.02,.003,.03,278.,10.)
    theta=t+.0098*geometry.temperature_height
    rho=p/(RD*t*(1.+(.378/.622)*q))
    # In this weakly stable range the momentum integral is ln(z/z0)
    # + 5*zeta*(1-z0/z). Construct F(zeta)=.98*zeta+.016-zeta,
    # whose independent exact root is .8. Half-Picard needs hundreds of
    # steps; the bounded root should retain the original 50-step budget.
    def solve(boundary):
        ustar=.04*(boundary.leaf_velocity/.01)**2
        zeta=(.4*wind/ustar-math.log(10./.05))/(5.*(1.-.05/10.))
        target=.98*zeta+.016
        buoyancy=-target*ustar**3/(10.*.4)
        h=buoyancy*rho*theta*CP/9.81
        return None,None,None,dict(h=h,et_kgm2=0.,leaf_temperature=np.array([t]))
    result=solve_surface_most(solve,exchange,wind,geometry,time_step=1800.)[-1]['most']
    assert abs(result['used_zeta']-.8)<.005
    assert result['coefficient_residual']<=1e-4
    assert result['iterations']<25

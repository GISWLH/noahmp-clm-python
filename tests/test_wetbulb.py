"""Independent root, published sigmoid and atmospheric physical constraints."""
import math
import numpy as np
import pytest
from scipy.optimize import brentq
from noahmp_clm.physics.wetbulb import wetbulb_temperature,wetbulb_snow_fraction


def humidity(tc,rh,p):
    e=rh*610.8*np.exp(17.27*tc/(237.3+tc))
    return .622*e/(p-.378*e)


def test_depression_equation_against_independent_brent_roots():
    maximum_t=maximum_fraction=maximum_residual=0.;count=0
    for tc in (-70.,-50.,-25.,-10.,-1.,0.,.5,1.,2.,5.,8.,15.,30.,50.):
        for rh in (0.,.1,.3,.5,.7,.9,1.,1.05):
            for p in (40000.,70000.,101325.):
                q=humidity(tc,rh,p)
                if q>=.2:continue
                gamma=1004.*p/(.622*(2500000. if tc>0 else 2834000.))
                ea=rh*610.8*math.exp(17.27*tc/(237.3+tc))
                def balance(tw):
                    return 610.8*math.exp(17.27*tw/(237.3+tw))-ea+gamma*(tw-tc)
                ref=brentq(balance,-150.,100.,xtol=1e-12)
                got=wetbulb_temperature(tc+273.15,q,p)-273.15
                expected=1./(1.+6.99e-5*math.exp(2.*(ref+3.97)))
                fraction=wetbulb_snow_fraction(tc+273.15,q,p)
                maximum_t=max(maximum_t,abs(got-ref))
                maximum_fraction=max(maximum_fraction,abs(fraction-expected))
                maximum_residual=max(maximum_residual,abs(balance(got)))
                count+=1
    print('wetbulb independent',dict(cases=count,temperature_k=maximum_t,
        fraction_absolute=maximum_fraction,equation_residual_pa=maximum_residual),flush=True)
    assert maximum_t<1e-9 and maximum_fraction<1e-10 and maximum_residual<1e-7


def test_wetbulb_saturation_dryness_pressure_and_supersaturation():
    tc=np.array([-30.,-5.,0.,2.,8.,30.]);p=101325.
    saturated=humidity(tc,1.,p)
    np.testing.assert_allclose(wetbulb_temperature(tc+273.15,saturated,p),tc+273.15,rtol=0,atol=1e-12)
    dry=humidity(tc,.5,p)
    assert np.all(wetbulb_temperature(tc+273.15,dry,p)<tc+273.15)
    assert np.all(wetbulb_temperature(tc+273.15,humidity(tc,1.05,p),p)>tc+273.15)
    assert wetbulb_snow_fraction(278.15,humidity(5.,.5,70000.),70000.)>wetbulb_snow_fraction(278.15,humidity(5.,.5,p),p)
    assert wetbulb_snow_fraction(278.15,humidity(5.,.5,p),p)>wetbulb_snow_fraction(278.15,humidity(5.,.9,p),p)


def test_sigmoid_midpoint_and_warm_dry_tail_are_continuous():
    midpoint=-math.log(6.99e-5)/2.-3.97
    fraction=wetbulb_snow_fraction(midpoint+273.15,humidity(midpoint,1.,101325.),101325.)
    assert fraction==pytest.approx(.5,abs=2e-14)
    near=np.array([8.-1e-5,8.,8.+1e-5])
    values=wetbulb_snow_fraction(near+273.15,humidity(near,.2,101325.),101325.)
    assert np.all(values>0.) and np.max(np.abs(np.diff(values)))<1e-4


def test_input_contract():
    for values in ((199.,.005,101325.),(280.,-.1,101325.),(280.,.005,0.),(np.nan,.005,101325.)):
        with pytest.raises(ValueError):wetbulb_temperature(*values)

"""Physical phase-curve consistency and open-system soil energy budgets."""
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,default_soil_params
from clm_soil_oracle import reference_liquid as unfrozen_water_content
from noahmp_clm.physics.soil_enthalpy import (liquid_and_derivative,enthalpy_and_capacity,
    temperature_from_enthalpy,SoilEnthalpyState,infiltrate,withdraw_liquid)
from noahmp_clm.physics.snowpack import TF,LF,CW

@pytest.mark.parametrize('total,t,sat,b,psi,official',[
    (.28,270.259,.439,5.25,.355,.11723104119300842),
    (.382,260.1010416666666,.465,8.17,.263,.15881063044071198),
])
def test_extensive_soil_state_uses_independent_official_suction(total,t,sat,b,psi,official):
    from noahmp_clm.column import SoilParams
    p=SoilParams(smcmax=sat,bexp=b,saturated_suction_m=psi)
    old=SoilEnthalpyState.from_column(ColumnState(soil=p,tsoil=np.full(4,t),smc=np.full(4,total)))
    np.testing.assert_allclose(old.liquid/(1000*old.dz),official,rtol=.01,atol=2e-5)
    np.testing.assert_allclose(old.temperature,t,rtol=0.,atol=2e-10)
    # Changing a constitutive parameter changes latent storage at the same T;
    # this catches a parameter that is serialized but never reaches the curve.
    other=SoilEnthalpyState.from_column(ColumnState(soil=SoilParams(smcmax=sat,bexp=b,saturated_suction_m=psi*2),
        tsoil=np.full(4,t),smc=np.full(4,total)))
    assert np.all(other.liquid>old.liquid) and np.all(other.enthalpy>old.enthalpy)


@pytest.mark.parametrize('texture',['sand','loam','clay'])
@pytest.mark.parametrize('option',[1,2])
def test_liquid_curve_matches_pressure_head_reference_and_derivative(texture,option):
    p=default_soil_params(texture);rng=np.random.default_rng(151)
    t=np.r_[np.linspace(243.,275.,400),TF-np.logspace(-5,1,400)]
    theta=rng.uniform(.001,p.smcmax,t.size)
    got,d=liquid_and_derivative(theta,t,p,option)
    expected=unfrozen_water_content(theta,t,p,option)
    np.testing.assert_allclose(got,expected,rtol=3e-14,atol=1e-15)
    # Narrow enough not to straddle successive option-2 smooth-cap regions.
    eps=1e-7
    reference=(unfrozen_water_content(theta,t+eps,p,option)-unfrozen_water_content(theta,t-eps,p,option))/(2*eps)
    np.testing.assert_allclose(d,reference,rtol=2e-5,atol=1e-7)
    assert np.all(d>=0.)


@pytest.mark.parametrize('texture',['sand','loam','clay'])
@pytest.mark.parametrize('option',[1,2])
def test_temperature_inverse_against_brent(texture,option):
    brent=pytest.importorskip('scipy.optimize').brentq
    p=default_soil_params(texture);rng=np.random.default_rng(920)
    temp=np.r_[rng.uniform(240.,310.,50),TF-np.logspace(-4,1,50)]
    dz=rng.uniform(.01,1.,len(temp));water=rng.uniform(0.,p.smcmax,len(temp))*1000*dz
    h,c=enthalpy_and_capacity(water,temp,dz,p,option)
    got=temperature_from_enthalpy(water,h,dz,p,option)
    np.testing.assert_allclose(got,temp,rtol=0.,atol=2e-10)
    assert np.all(c>0.)
    for j in range(0,len(temp),10):
        def reference(t):
            liquid=float(unfrozen_water_content(np.array([water[j]/(1000*dz[j])]),np.array([t]),p,option)[0])*1000*dz[j]
            return ((1-p.smcmax)*2e6*dz[j]+2100*(water[j]-liquid)+4180*liquid)*(t-273.15)+334000*liquid-h[j]
        expected=brent(reference,150.,400.,xtol=1e-12)
        assert got[j]==pytest.approx(expected,abs=2e-10)


def test_water_carries_fusion_and_sensible_heat():
    c=ColumnState(smc=np.full(4,.1),tsoil=np.full(4,TF-5.))
    old=SoilEnthalpyState.from_column(c)
    supplied=20.*(LF+CW*10.)
    new,runoff,runoff_h=infiltrate(old,20.,supplied)
    assert new.water.sum()+runoff-old.water.sum()==pytest.approx(20.,abs=1e-12)
    assert new.enthalpy.sum()+runoff_h-old.enthalpy.sum()==pytest.approx(supplied,abs=2e-7)
    assert new.temperature[0]>old.temperature[0]
    demand=.1*new.liquid
    withdrawn,out=withdraw_liquid(new,demand)
    assert withdrawn.water.sum()+demand.sum()==pytest.approx(new.water.sum(),abs=1e-12)
    assert withdrawn.enthalpy.sum()+out.sum()==pytest.approx(new.enthalpy.sum(),abs=2e-7)
    np.testing.assert_array_equal(c.smc,np.full(4,.1))


def test_full_soil_overflow_exports_all_input_heat():
    c=ColumnState(smc=np.full(4,.45),tsoil=np.full(4,TF))
    old=SoilEnthalpyState.from_column(c)
    new,w,h=infiltrate(old,100.,100*LF)
    assert w==100. and h==100*LF
    np.testing.assert_array_equal(new.enthalpy,old.enthalpy)
    with pytest.raises(ValueError): infiltrate(old,0.,1.)
    with pytest.raises(ValueError): withdraw_liquid(old,old.water+1.)

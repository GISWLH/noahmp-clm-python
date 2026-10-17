"""Independent official point outputs and invariants for the explicit-suction law."""
import numpy as np
import pytest
from noahmp_clm.physics.clm_freezing import equilibrium_liquid_water as liquid

@pytest.mark.parametrize('total,t,sat,b,psi,official',[
    (.28,278.722,.439,5.25,.355,.2800000011920929),
    (.28,270.259,.439,5.25,.355,.11723104119300842),
    (.382,260.1010416666666,.465,8.17,.263,.15881063044071198),
])
def test_independent_official_hrldas_initial_water(total,t,sat,b,psi,official):
    # Unchanged official binary f27deb..., PBS37; actual soil-table6/9 points.
    # Noah273.16K/constants and CLM273.15K differ, so no exact-bit claim.
    assert float(liquid(total,t,sat,b,psi)['liquid'])==pytest.approx(official,rel=.01,abs=2e-5)

def test_conserves_bounds_and_increases_with_temperature():
    t=np.linspace(180.,300.,2001)
    for sat,b,psi in [(.339,2.79,.069),(.439,5.25,.355),(.465,8.17,.263),(.468,11.55,.468)]:
        for total in [0.,.1*sat,.6*sat,sat]:
            out=liquid(total,t,sat,b,psi)
            assert np.all(out['liquid']>=0.) and np.all(out['liquid']<=total)
            assert np.all(np.diff(out['liquid'])>=-1e-14)
            assert np.all(out['d_liquid_d_temperature']>=0.)

def test_temperature_and_available_water_derivatives_at_smooth_cap():
    sat=.439;b=5.25;psi=.355;total=.2
    center=273.15/(1.+9.80616*psi/334000.*(sat/total)**b)
    t=np.array([250.,265.,center-.00001,center,center+.00001,280.])
    out=liquid(total,t,sat,b,psi);step=1e-7
    dt=(liquid(total,t+step,sat,b,psi)['liquid']-liquid(total,t-step,sat,b,psi)['liquid'])/(2*step)
    dw=(liquid(total+step,t,sat,b,psi)['liquid']-liquid(total-step,t,sat,b,psi)['liquid'])/(2*step)
    np.testing.assert_allclose(out['d_liquid_d_temperature'],dt,rtol=5e-5,atol=1e-8)
    np.testing.assert_allclose(out['d_liquid_d_total'],dw,rtol=5e-5,atol=1e-8)

@pytest.mark.parametrize('change',[dict(total_water=-.1),dict(temperature=0.),dict(porosity=1.),dict(bexp=0.),dict(saturated_suction_m=-.1)])
def test_reject_nonphysical_parameters(change):
    arguments=dict(total_water=.2,temperature=260.,porosity=.45,bexp=5.,saturated_suction_m=.355)
    arguments.update(change)
    with pytest.raises(ValueError):liquid(**arguments)

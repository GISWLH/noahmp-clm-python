import numpy as np
import pytest
from noahmp_clm.physics.snowpack import Snowpack,TF,LF,CI,CW
from noahmp_clm.physics.snow_vapor import exchange_vapor


@pytest.mark.parametrize('temperature,liquid_fraction',[(260.,0.),(TF,0.),(TF,.3),(TF,.9),(278.,1.)])
def test_phase_partition_and_total_heat(temperature,liquid_fraction):
    mass=3.;h=mass*((CI*(1-liquid_fraction)+CW*liquid_fraction)*(temperature-TF)+LF*liquid_fraction)
    pack=Snowpack([mass],[h],[.02],impurities=np.full((1,5),1e-6))
    result,d=exchange_vapor(pack,.6)
    assert abs(d['liquid_kgm2']-.6*liquid_fraction)<1e-14
    assert abs(d['ice_kgm2']-.6*(1-liquid_fraction))<1e-14
    assert abs(result.mass.sum()+d['mass_kgm2']-mass)<1e-14
    assert abs(result.enthalpy.sum()+d['advected_enthalpy_jm2']-h)<1e-9
    np.testing.assert_allclose(result.temperature,temperature,atol=1e-12)
    assert abs(d['latent_energy_jm2']-(2.5e6*.6+LF*.6*(1-liquid_fraction)))<1e-9
    np.testing.assert_array_equal(result.impurities,pack.impurities)


@pytest.mark.parametrize('temperature',[260.,TF,280.])
def test_frost_and_dew_mass_heat(temperature):
    pack=Snowpack.from_ice([3.],270.)
    result,d=exchange_vapor(pack,-.4,deposition_temperature=temperature)
    expected=.4*(CI*(temperature-TF) if temperature<TF else LF+CW*(temperature-TF))
    np.testing.assert_allclose(result.enthalpy.sum()-pack.enthalpy.sum(),expected,atol=1e-9)
    assert abs(result.mass.sum()-pack.mass.sum()-.4)<1e-14
    assert d['ice_kgm2']==(-.4 if temperature<TF else 0.)
    assert d['liquid_kgm2']==(0. if temperature<TF else -.4)
    assert abs(d['latent_energy_jm2']+.4*(2.5e6+(LF if temperature<TF else 0.)))<1e-9


def test_exhaustion_retains_or_exports_tracers_and_rejects_overdraw():
    pack=Snowpack.from_ice([2.,5.],260.)
    pack=Snowpack(pack.mass,pack.enthalpy,pack.thickness,impurities=np.full((2,5),1e-6))
    result,d=exchange_vapor(pack,2.)
    np.testing.assert_allclose(result.impurities.sum(0),pack.impurities.sum(0),atol=1e-20)
    assert result.mass.sum()==5.
    single=Snowpack.from_ice([2.],260.);single=Snowpack(single.mass,single.enthalpy,single.thickness,impurities=np.full((1,5),1e-6))
    empty,d=exchange_vapor(single,2.)
    assert len(empty.mass)==0
    np.testing.assert_array_equal(d['residue_kgm2'],single.impurities[0])
    with pytest.raises(ValueError): exchange_vapor(pack,2.1)
    frost,d=exchange_vapor(Snowpack.empty(),-.3,deposition_temperature=260.)
    assert frost.mass.sum()==.3 and frost.temperature[0]==260.
    near=Snowpack.from_ice([.05],TF)
    empty,d=exchange_vapor(near,np.nextafter(.05,0.))
    assert len(empty.mass)==0 and d['mass_kgm2']==.05

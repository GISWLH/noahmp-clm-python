import numpy as np
import pytest
from noahmp_clm.physics.snowpack import Snowpack
from noahmp_clm.physics.snicar import snow_shortwave
from noahmp_clm.physics.canopy_radiation import canopy_shortwave
from noahmp_clm.physics.canopy_snow_radiation import canopy_snow_shortwave


@pytest.mark.parametrize('layers',[0,1,5])
@pytest.mark.parametrize('fraction',[0.,.6,1.])
@pytest.mark.parametrize('diffuse',[0.,.4,1.])
def test_every_reflection_and_absorption_is_accounted(layers,fraction,diffuse):
    pack=Snowpack.from_ice(np.full(layers,10.),265.) if layers else Snowpack.empty()
    result=canopy_snow_shortwave([.3]*10,pack,700.,vegetated_fraction=fraction,
        diffuse_fraction=diffuse,cosine_zenith=.4)
    assert abs(result['residual'])<2e-10
    assert np.min(result['canopy_absorbed'])>=0
    assert np.min(result['snow_absorbed'],initial=0.)>=-1e-10
    assert result['soil_absorbed']>=0
    if fraction==0:
        reference=snow_shortwave(pack,[350*(1-diffuse)]*2,[350*diffuse]*2,cosine_zenith=.4)
        np.testing.assert_allclose(result['snow_absorbed'],reference['absorbed'],atol=1e-10)
        np.testing.assert_allclose(result['reflected'],reference['reflected'],atol=1e-10)
    if layers==0 and fraction==1:
        reference=canopy_shortwave([.3]*10,700.,diffuse_fraction=diffuse,cosine_zenith=.4)
        np.testing.assert_allclose(result['canopy_absorbed'],reference['absorbed'],atol=1e-10)
        np.testing.assert_allclose(result['soil_absorbed'],reference['ground_absorbed'],atol=1e-10)


def test_night_and_leafless_pack():
    pack=Snowpack.from_ice([20.,30.],260.)
    night=canopy_snow_shortwave([.3]*10,pack,0.)
    assert night['reflected']==night['soil_absorbed']==night['residual']==0.
    bare=canopy_snow_shortwave([0.]*10,pack,500.,vegetated_fraction=.7)
    reference=snow_shortwave(pack,[125.,125.],[125.,125.])
    np.testing.assert_allclose(bare['snow_absorbed'],reference['absorbed'],atol=1e-11)
    np.testing.assert_allclose(bare['reflected'],reference['reflected'],atol=1e-11)

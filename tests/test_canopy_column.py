import numpy as np
import pytest
from noahmp_clm.column import ColumnState
from noahmp_clm.physics.snowpack import Snowpack,TF,LF
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_column import surface_step


@pytest.mark.parametrize('snow_mass,air,humidity,solar',[
    (0.,295.,.008,700.),(0.,268.,.003,0.),(0.,270.,.0001,200.),
    (20.,260.,.0002,300.),(20.,270.,.004,0.),(20.,280.,.004,800.),
    (.05,285.,.001,900.),(.001,295.,.0001,800.),(0.,295.,.02,0.)])
def test_coupled_surface_heat_water_and_phase(snow_mass,air,humidity,solar):
    snow=Snowpack.from_ice([snow_mass],min(air,TF)) if snow_mass else Snowpack.empty()
    soil=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,min(air+2.,290.)),smc=np.full(4,.3)))
    exchange=CanopyExchange(air,humidity,101325.,.02,.004,.03,float(soil.temperature[0]),15.)
    pack,ground,canopy,d=surface_step(snow,soil,1800.,[.3]*10,solar,300.,exchange,
        canopy_water_kgm2=.3,wet_fraction=.3,root_weights=[.6,.25,.1,.05],vegetated_fraction=.8)
    assert abs(d['water_residual_kgm2'])<2e-10
    assert abs(d['energy_residual_jm2'])<.01
    assert np.all(np.isfinite(ground.temperature)) and np.all(pack.mass>0.)
    assert np.all(ground.water>=0.) and canopy>=0.
    # Keep the original thermal-root budget and report all extra event work.
    # Bisection over 200--350 K to 2e-7 K needs at most 30 trials + endpoint.
    assert d['surface_evaluations']-d['surface_branch_evaluations']<40
    assert d['surface_branch_evaluations']<=31
    print('surface evaluation costs',snow_mass,d['surface_evaluations'],d['surface_branch_evaluations'],flush=True)


def test_surface_complete_melt_energy_enters_real_soil():
    snow=Snowpack.from_ice([.1],TF)
    soil=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,TF-1.),smc=np.full(4,.25)))
    e=CanopyExchange(285.,.01,101325.,.02,.004,.03,272.,15.)
    pack,ground,canopy,d=surface_step(snow,soil,1800.,[0.]*10,800.,380.,e)
    assert not len(pack.mass)
    assert d['snow_drainage_kgm2']>0.
    assert d['snow_drainage_enthalpy_jm2']>0.
    assert abs(d['energy_residual_jm2'])<.01
    assert abs(d['water_residual_kgm2'])<2e-10

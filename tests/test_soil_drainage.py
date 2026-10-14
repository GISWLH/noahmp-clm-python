import numpy as np
import pytest
from noahmp_clm.column import ColumnState
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.soil_drainage import drain_soil


@pytest.mark.parametrize('temperature',[260.,273.1,285.])
@pytest.mark.parametrize('option',[1,2])
def test_gravity_transfers_mass_heat_not_ice(temperature,option):
    ground=SoilEnthalpyState.from_column(ColumnState(smc=np.array([.4,.4,.4,.4]),tsoil=np.full(4,temperature)))
    result,d=drain_soil(ground,1800.,permeability_option=option)
    assert abs(d['water_residual_kgm2'])<1e-10
    assert abs(d['energy_residual_jm2'])<1e-6
    assert np.all(result.water>=0.) and np.all(result.water<=1000.*ground.params.smcmax*ground.dz)
    assert d['interface_mass_kgm2'][0]<=ground.liquid[0]
    if temperature==285.:
        expected=min(1800.*.01*(.4/.45)**14,12.,ground.liquid[0])
        assert abs(d['interface_mass_kgm2'][0]-expected)<1e-12
        assert abs(d['interface_enthalpy_jm2'][0]-expected*(334000.+4180.*(285.-273.15)))<1e-8


def test_impermeable_receiver_retains_source_water():
    ground=SoilEnthalpyState.from_column(ColumnState(smc=np.full(4,.45),tsoil=np.full(4,260.)))
    result,d=drain_soil(ground,1800.,permeability_option=2)
    assert d['interface_mass_kgm2'][0]==0.
    np.testing.assert_array_equal(result.water[0],ground.water[0])

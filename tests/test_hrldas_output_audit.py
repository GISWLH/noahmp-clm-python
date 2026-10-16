"""Independent interval accounting and invalid-reference guards."""
from datetime import datetime, timedelta
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

spec = importlib.util.spec_from_file_location('hrldas_audit', Path(__file__).parents[1] / 'benchmarks/audit_hrldas_point_output.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def write_fixture(folder, index, **changes):
    timestamp = datetime(2000, 1, 1) + timedelta(hours=12 * index)
    path = folder / (timestamp.strftime('%Y%m%d%H%M') + '.LDASOUT_DOMAIN1')
    with netcdf_file(path, 'w') as data:
        for name, size in [('Time', 1), ('DateStrLen', 19), ('west_east', 1), ('south_north', 1), ('soil_layers_stag', 2)]:
            data.createDimension(name, size)
        v = data.createVariable('Times', 'c', ('Time', 'DateStrLen'))
        v[:] = np.frombuffer(timestamp.strftime('%Y-%m-%d_%H:%M:%S').encode(), dtype='S1')[None]
        for key, units in audit_module.UNITS.items():
            dims = ('Time', 'south_north', 'soil_layers_stag', 'west_east') if key.startswith('SOIL_') else ('Time', 'south_north', 'west_east')
            v = data.createVariable(key, 'd', dims)
            v.units = changes.get('units', {}).get(key, units)
            value = 280. + index if key == 'SOIL_T' else .3
            if key in audit_module.COUNTERS: value = 10. + index
            if key in ('HFX', 'LH'): value = -2. + index
            if key in ('ECAN', 'ETRAN', 'EDIR'): value = float('nan') if index == 0 else .001
            v[:] = changes.get(key, value)
    return path


def test_right_end_midnight_initial_flux_and_cumulative_counters(tmp_path):
    for i in range(3): write_fixture(tmp_path, i)
    report = audit_module.audit(tmp_path, initial_time=datetime(2000, 1, 1), timestep_seconds=43200)
    assert report['physical_intervals'] == 2
    assert list(report['daily']) == ['2000-01-01']
    day = report['daily']['2000-01-01']
    assert day['complete_day']
    assert day['et_mm'] == pytest.approx(.003 * 86400)
    assert day['runoff_mm'] == 4.  # both counters +2, not sum of their running totals
    assert day['h_jm2'] == -43200.
    assert day['soil_temperature_k_mean_of_interval_ends'] == [281.5, 281.5]
    assert day['soil_temperature_k_end'] == [282., 282.]


@pytest.mark.parametrize('invalid,match', [({'ECAN': np.nan}, 'nonfinite'),
    ({'units': {'ECAN': 'mm/timestep'}}, 'units'), ({'SFCRNOFF': 2.}, 'decreased')])
def test_bad_reference_is_rejected(tmp_path, invalid, match):
    write_fixture(tmp_path, 0)
    write_fixture(tmp_path, 1, **invalid)
    with pytest.raises(ValueError, match=match):
        audit_module.audit(tmp_path, initial_time=datetime(2000, 1, 1), timestep_seconds=43200)


def test_missing_interval_cannot_be_silently_truncated(tmp_path):
    write_fixture(tmp_path, 0)
    write_fixture(tmp_path, 2)
    with pytest.raises(ValueError, match='missing/duplicate/off-grid'):
        audit_module.audit(tmp_path, initial_time=datetime(2000, 1, 1), timestep_seconds=43200)

"""Read official HRLDAS classic-NetCDF point output with explicit time/units.

The official driver writes initial state at ITIME=0 and then fluxes/states at
interval ends. This reader excludes initial flux placeholders, differences
cumulative runoff counters, and never silently truncates or fills a series.
Requires scipy (the project's dev extra); does not run either model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file


UNITS = {
    'HFX': 'W/m2', 'LH': 'W/m2', 'ECAN': 'mm/s', 'ETRAN': 'mm/s',
    'EDIR': 'mm/s', 'SOIL_T': 'K', 'SOIL_M': 'm3/m3', 'SOIL_W': 'm3/m3',
    'SNEQV': 'mm', 'SFCRNOFF': 'mm', 'UGDRNOFF': 'mm',
}
COUNTERS = ('SFCRNOFF', 'UGDRNOFF')
STATES = ('SOIL_T', 'SOIL_M', 'SOIL_W', 'SNEQV')


def _text(value):
    return value.decode('ascii').strip() if isinstance(value, bytes) else str(value).strip()


def read_point(path: Path, *, initial: bool = False):
    """Validate a single-time/single-point file; retain all soil layers."""
    with netcdf_file(path, 'r', mmap=False) as data:
        time = data.variables['Times'].data
        if time.shape != (1, 19):
            raise ValueError(f'{path}: expected one 19-character Times record')
        timestamp = datetime.strptime(time.tobytes().decode('ascii'), '%Y-%m-%d_%H:%M:%S')
        if path.name.split('.')[0] != timestamp.strftime('%Y%m%d%H%M'):
            raise ValueError(f'{path}: filename/internal time mismatch')
        values = {}
        for key, units in UNITS.items():
            # Initial flux diagnostics are undefined in the upstream driver.
            if initial and key not in (*COUNTERS, *STATES):
                continue
            var = data.variables[key]
            if _text(getattr(var, 'units', '')) != units:
                raise ValueError(f'{path}: {key} units must be {units}')
            for dim in ('Time', 'west_east', 'south_north'):
                if dim not in var.dimensions or var.shape[var.dimensions.index(dim)] != 1:
                    raise ValueError(f'{path}: {key} requires one Time and one horizontal point')
            array = np.asarray(var.data, dtype=float).copy().ravel()
            expected_size = data.dimensions['soil_layers_stag'] if key.startswith('SOIL_') else 1
            if array.size != expected_size:
                raise ValueError(f'{path}: unexpected {key} shape {var.shape}')
            invalid = ~np.isfinite(array) | (abs(array) >= 1e30)
            for attr in ('_FillValue', 'missing_value'):
                if hasattr(var, attr):
                    invalid |= np.isin(array, np.asarray(getattr(var, attr)).ravel())
            if invalid.any():
                raise ValueError(f'{path}: {key} contains missing/nonfinite values')
            values[key] = array.tolist() if key.startswith('SOIL_') else float(array[0])
    return timestamp, values


def audit(directory: Path, *, initial_time: datetime, timestep_seconds: int):
    if timestep_seconds <= 0 or 86400 % timestep_seconds:
        raise ValueError('timestep must be a positive integer divisor of one day')
    if (initial_time - initial_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() % timestep_seconds:
        raise ValueError('initial time must align with the daily timestep grid')
    files = sorted(directory.glob('*.LDASOUT_DOMAIN1'))
    if len(files) < 2:
        raise ValueError('initial state and at least one physical output are required')
    manifest, records = {}, []
    previous = None
    for index, path in enumerate(files):
        timestamp, values = read_point(path, initial=index == 0)
        expected = initial_time + timedelta(seconds=index * timestep_seconds)
        if timestamp != expected:
            raise ValueError(f'{path}: missing/duplicate/off-grid time, expected {expected}')
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if previous is not None:
            increments = [values[k] - previous[k] for k in COUNTERS]
            if min(increments) < 0:
                raise ValueError(f'{path}: accumulated runoff decreased; explicit restart handling required')
            records.append({
                'time_start': (timestamp - timedelta(seconds=timestep_seconds)).isoformat(),
                'time_end': timestamp.isoformat(),
                'et_mm': sum(values[k] for k in ('ECAN', 'ETRAN', 'EDIR')) * timestep_seconds,
                'h_jm2': values['HFX'] * timestep_seconds,
                'le_jm2': values['LH'] * timestep_seconds,
                'h_wm2': values['HFX'], 'le_wm2': values['LH'],
                'runoff_mm': sum(increments),
                'soil_temperature_k': values['SOIL_T'],
                'soil_total_water_m3m3': values['SOIL_M'],
                'soil_liquid_water_m3m3': values['SOIL_W'],
                'swe_mm': values['SNEQV'],
            })
        else:
            initial_values = values
        previous = values
    days = {}
    for row in records:
        day = row['time_start'][:10]
        group = days.setdefault(day, {'duration_seconds': 0, 'samples': 0, 'et_mm': 0.,
                                     'h_jm2': 0., 'le_jm2': 0., 'runoff_mm': 0.})
        group['duration_seconds'] += timestep_seconds
        group['samples'] += 1
        for key in ('et_mm', 'h_jm2', 'le_jm2', 'runoff_mm'):
            group[key] += row[key]
        for key in ('soil_temperature_k', 'soil_total_water_m3m3', 'soil_liquid_water_m3m3', 'swe_mm'):
            accumulator = key + '_weighted_sum'
            group[accumulator] = np.asarray(group.get(accumulator, 0.)) + np.asarray(row[key]) * timestep_seconds
            group[key + '_end'] = row[key]
    for group in days.values():
        duration = group['duration_seconds']
        group['complete_day'] = duration == 86400
        group['h_mean_wm2'] = group['h_jm2'] / duration
        group['le_mean_wm2'] = group['le_jm2'] / duration
        for key in list(group):
            if key.endswith('_weighted_sum'):
                group[key.replace('_weighted_sum', '_mean_of_interval_ends')] = (group.pop(key) / duration).tolist()
    return {
        'scope': 'Official point-output integrity and aggregation; not model-comparison acceptance',
        'time_convention': 'Initial state excluded from flux sums; remaining Times label interval ends; calendar/timezone inherited from forcing',
        'timestep_seconds': timestep_seconds, 'initial_time': initial_time.isoformat(),
        'file_count': len(files), 'physical_intervals': len(records),
        'source_sha256': manifest, 'raw_units': UNITS, 'initial_state': initial_values,
        'daily': days, 'intervals': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--initial-time', type=datetime.fromisoformat, required=True)
    parser.add_argument('--timestep-seconds', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, initial_time=args.initial_time, timestep_seconds=args.timestep_seconds)
    result['instrument_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('file_count', 'physical_intervals', 'daily')}))


if __name__ == '__main__':
    main()

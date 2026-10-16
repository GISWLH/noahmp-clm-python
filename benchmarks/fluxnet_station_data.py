"""Auditable FLUXNET station inputs for the v1 extensive column.

Reads an exact local-standard-time window directly from the original archive.
No observation is used to alter forcing. QC remains field-specific. Humidity
repair is opt-in and leaves a per-row ledger; strict is the default.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from fluxnet_solar import solar_geometry, diffuse_fraction_erbs
from inventory_fluxnet_archive import digest, number

FORCING = ('TA_F', 'PA_F', 'VPD_F', 'WS_F', 'SW_IN_F', 'LW_IN_F', 'P_F')
OBSERVATIONS = ('H_F_MDS', 'LE_F_MDS', 'H_CORR', 'LE_CORR', 'NEE_VUT_REF',
                'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF', 'RECO_NT_VUT_REF', 'RECO_DT_VUT_REF',
                'TS_F_MDS_1', 'SWC_F_MDS_1', 'CO2_F_MDS')
QC_FIELD = {'H_CORR': 'H_F_MDS_QC', 'LE_CORR': 'LE_F_MDS_QC'}


def humidity(temperature_c, pressure_pa, vpd_hpa, timestamps, dt, policy='strict'):
    """Tetens over water; optional interpolation of RH inferred from VPD.

    Only finite nonnegative VPD with e<0 can be repaired. Both valid bracketing
    rows must exist inside the requested window, at most three hours apart.
    This is an explicit preprocessing estimate, not original tower RH.
    """
    t, p, v = (np.asarray(a, float) for a in (temperature_c, pressure_pa, vpd_hpa))
    if policy not in ('strict', 'rh-interpolate-3h'):
        raise ValueError('Unknown humidity policy')
    if t.ndim != 1 or t.shape != p.shape or t.shape != v.shape or len(t) != len(timestamps):
        raise ValueError('Humidity arrays/time must match')
    if (not all(np.all(np.isfinite(a)) for a in (t, p, v)) or np.any(v < 0) or np.any(p <= 0)
            or np.any((t < -80.) | (t > 65.)) or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Missing/invalid humidity inputs cannot be silently filled')
    es = 611.2 * np.exp(17.67 * t / (t + 243.5))
    e = es - 100. * v
    if np.any(e >= p) or np.any(~np.isfinite(es)):
        raise ValueError('Invalid humidity/pressure')
    bad = e < 0
    ledger = []
    if bad.any() and policy == 'strict':
        raise ValueError(f'Invalid vapor pressure at {timestamps[int(np.flatnonzero(bad)[0])]}: strict policy')
    start = 0
    while start < len(t):
        if not bad[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(t) and bad[stop]: stop += 1
        left, right = start - 1, stop
        if left < 0 or right >= len(t) or (right - left) * dt > 10800:
            raise ValueError('Invalid humidity run lacks valid brackets within three hours')
        rh_left, rh_right = e[left] / es[left], e[right] / es[right]
        for index in range(start, stop):
            weight = (index - left) / (right - left)
            relative = (1 - weight) * rh_left + weight * rh_right
            updated = relative * es[index]
            if not 0 <= updated < p[index]: raise ValueError('Interpolated humidity outside physical bounds')
            ledger.append({'timestamp_start': timestamps[index], 'raw_vapor_pressure_pa': float(e[index]),
                           'estimated_vapor_pressure_pa': float(updated), 'estimated_relative_humidity_fraction': float(relative),
                           'left_timestamp': timestamps[left], 'right_timestamp': timestamps[right],
                           'method': 'Linear interpolation of inferred RH between valid bracket rows; raw QC retained'})
            e[index] = updated
        start = stop
    return .622 * e / (p - .378 * e), bad, ledger


def prepare_rows(reader, *, start, end, latitude, longitude, utc_offset_hours, co2_monthly, humidity_policy='strict'):
    begin, finish = datetime.fromisoformat(start), datetime.fromisoformat(end)
    if begin.tzinfo is not None or finish.tzinfo is not None or begin >= finish:
        raise ValueError('Require increasing naive local-standard start/end')
    header = reader.fieldnames or []
    if len(header) != len(set(header)) or not {'TIMESTAMP_START', 'TIMESTAMP_END', *FORCING}.issubset(header):
        raise ValueError('Missing or duplicate forcing columns')
    stamps, stops, raw = [], [], []
    observation, qc = {}, {}
    for field in OBSERVATIONS: observation[field] = []
    for field in (*FORCING, *OBSERVATIONS): qc[field] = []
    cursor, interval = begin, None
    for row in reader:
        stamp = datetime.strptime(row['TIMESTAMP_START'], '%Y%m%d%H%M')
        if not begin <= stamp < finish: continue
        stop = datetime.strptime(row['TIMESTAMP_END'], '%Y%m%d%H%M')
        step = (stop - stamp).total_seconds()
        if stamp != cursor or step not in (1800., 3600.) or stop > finish or (interval is not None and step != interval):
            raise ValueError(f'Noncontiguous/mixed/invalid interval at {stamp}')
        values = [number(row.get(key)) for key in FORCING]
        if any(value is None for value in values): raise ValueError(f'Missing forcing at {stamp}')
        tc, pa, vpd, wind, sw, lw, rain = values
        if not -80 <= tc <= 65 or not 20 <= pa <= 120 or min(vpd, wind, sw, lw, rain) < 0:
            raise ValueError(f'Forcing outside input range at {stamp}')
        raw.append(values); stamps.append(row['TIMESTAMP_START']); stops.append(row['TIMESTAMP_END'])
        for field in OBSERVATIONS:
            value = number(row.get(field)); observation[field].append(np.nan if value is None else value)
        for field in qc:
            value = number(row.get(QC_FIELD.get(field, field + '_QC')))
            qc[field].append(np.nan if value is None else value)
        cursor, interval = stop, step
    if cursor != finish or not stamps: raise ValueError('Incomplete requested window')
    raw = np.asarray(raw, float)
    q, repaired, ledger = humidity(raw[:,0], raw[:,1] * 1000., raw[:,2], stamps, interval, humidity_policy)
    forcing = np.column_stack((np.full(len(raw), interval), raw[:,0]+273.15, q, raw[:,3], raw[:,1]*1000.,
                               raw[:,4], raw[:,5], raw[:,6]/(1000.*interval)))
    environment, geometry, flags = [], [], []
    for stamp, values in zip(stamps, raw):
        left = datetime.strptime(stamp, '%Y%m%d%H%M')
        middle = left + timedelta(seconds=interval/2)
        geo = solar_geometry(middle, latitude, longitude, utc_offset_hours)
        shortwave = diffuse_fraction_erbs(values[4], geo['cosine_zenith'], geo['extraterrestrial_normal_wm2'])
        co2 = co2_monthly.get((left.year, left.month))
        if co2 is None or not np.isfinite(co2) or not 200 <= co2 <= 1000:
            raise ValueError(f'Missing/invalid explicit background CO2 for {left:%Y-%m}')
        # Radiation APIs require positive mu even for entirely diffuse input.
        # Keep true signed mu separately; do not discard any shortwave energy.
        environment.append([co2, max(.001, geo['cosine_zenith']), shortwave['diffuse_fraction']])
        geometry.append([geo['cosine_zenith'], geo['daylength_seconds'], shortwave['clearness_index_raw']])
        flags.append(shortwave['low_sun_all_diffuse'])
    arrays = {'time_start': np.asarray(stamps), 'time_end': np.asarray(stops), 'forcing': forcing,
              'environment': np.asarray(environment), 'solar_diagnostics': np.asarray(geometry),
              'low_sun_all_diffuse': np.asarray(flags), 'humidity_repaired': repaired}
    arrays.update({'obs_' + k: np.asarray(v) for k,v in observation.items()})
    arrays.update({'qc_' + k: np.asarray(v) for k,v in qc.items()})
    counts = {k: dict(Counter('absent_or_missing' if not np.isfinite(v) else str(v) for v in values)) for k,values in qc.items()}
    report = {'scope': 'Prepared forcing/QC only; no land-model score or calibration', 'start': start, 'end_exclusive': end,
              'timezone': 'Local standard time, fixed offset, no DST', 'utc_offset_hours': utc_offset_hours,
              'latitude': latitude, 'longitude': longitude, 'steps': len(raw), 'dt_seconds': interval,
              'forcing_columns': ['dt_s','tair_k','qair_kgkg','wind_ms','pressure_pa','swdown_wm2','lwdown_wm2','precipitation_m_s'],
              'environment_columns': ['background_co2_ppm','model_cosine_zenith','estimated_diffuse_fraction'],
              'solar_diagnostics_columns': ['true_signed_cosine_zenith','geometric_daylength_seconds','raw_clearness_index'],
              'humidity_policy': humidity_policy, 'humidity_repairs': ledger, 'qc_counts': counts,
              'observation_note': 'Original units/values and own QC retained; no GPP QC inferred from NEE; no observation filling',
              'total_precipitation_mm': float(raw[:,6].sum()),
              'low_sun_positive_shortwave_rows': int(np.count_nonzero(np.asarray(flags) & (raw[:,4]>0.))),
              'low_sun_positive_shortwave_jm2': float(raw[np.asarray(flags),4].sum()*interval)}
    return arrays, report


def prepare_archive(path, *, co2_path, member=None, **kwargs):
    path, co2_path = Path(path), Path(co2_path)
    source = json.loads(co2_path.read_text(encoding='utf-8'))
    co2 = {(r['year'], r['month']): r['average_ppm'] for r in source['rows']}
    if len(co2) != len(source['rows']): raise ValueError('Duplicate background CO2 months')
    with path.open('rb') as stream: archive_sha = digest(stream)
    with zipfile.ZipFile(path) as archive:
        candidates = [n for n in archive.namelist() if n.endswith('.csv') and ('_FULLSET_HH_' in n or '_FLUXMET_HH_' in n)]
        if member is None:
            if len(candidates) != 1: raise ValueError('Select exactly one original HH member')
            member = candidates[0]
        if member not in candidates: raise ValueError('Expected original FULLSET_HH or FLUXMET_HH')
        with archive.open(member) as stream: member_sha = digest(stream)
        with archive.open(member) as stream:
            arrays, report = prepare_rows(csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig', newline='')), co2_monthly=co2, **kwargs)
    report.update(archive_name=path.name, archive_sha256=archive_sha, member=member, member_sha256=member_sha,
                  co2_excerpt_sha256=hashlib.sha256(co2_path.read_bytes()).hexdigest(), co2_source=source['url'],
                  instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return arrays, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    for key in ('start','end'): parser.add_argument('--'+key, required=True)
    for key in ('latitude','longitude','utc-offset-hours'): parser.add_argument('--'+key, required=True, type=float)
    parser.add_argument('--co2-path', required=True, type=Path)
    parser.add_argument('--humidity-policy', choices=('strict','rh-interpolate-3h'), default='strict')
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args()); output = args.pop('output'); archive = args.pop('archive')
    arrays, report = prepare_archive(archive, **args)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    report['prepared_npz_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('steps','dt_seconds','total_precipitation_mm','humidity_repairs')}))


if __name__ == '__main__': main()

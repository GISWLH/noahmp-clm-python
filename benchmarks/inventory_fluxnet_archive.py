"""Read-only FULLSET/FLUXMET HH inventory before fixing a validation protocol.

Reports the original time axis, missing values and each field's own QC. This
does not convert, fill, clip or select data, and it does not score a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

FORCING = ('TA_F', 'PA_F', 'P_F', 'WS_F', 'SW_IN_F', 'LW_IN_F', 'VPD_F')
FIELDS = (*FORCING, 'CO2_F_MDS', 'H_F_MDS', 'LE_F_MDS', 'H_CORR', 'LE_CORR',
          'NEE_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF', 'RECO_NT_VUT_REF',
          'RECO_DT_VUT_REF', 'TS_F_MDS_1', 'SWC_F_MDS_1')
# Corrected fluxes have the same QC flag as their uncorrected MDS product.
QC_FIELD = {'H_CORR': 'H_F_MDS_QC', 'LE_CORR': 'LE_F_MDS_QC'}


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def number(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x != -9999. else None


def empty_year():
    return dict(rows=0, first=None, last_end=None, duration_seconds=Counter(),
                forcing_complete_rows=0, forcing_and_co2_complete_rows=0,
                physical_input_flags=Counter(), fields={name: dict(valid=0, missing=0,
                valid_qc_zero=0, qc=Counter(), minimum=None, maximum=None) for name in FIELDS})


def inventory_rows(reader):
    header = reader.fieldnames or []
    if len(header) != len(set(header)):
        raise ValueError('Duplicate CSV column names')
    if not {'TIMESTAMP_START', 'TIMESTAMP_END'} <= set(header):
        raise ValueError('HH timestamps are required')
    header_set = set(header)
    report = dict(rows=0, first=None, last_end=None, missing_columns=sorted(set(FIELDS)-set(header)),
                  time_axis_issues=[], time_axis_issue_count=0, years={})
    previous = None
    expected_dt = None
    for row_number, row in enumerate(reader, start=2):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f'Malformed CSV row {row_number}')
        try:
            start, end = (datetime.strptime(row[k], '%Y%m%d%H%M') for k in ('TIMESTAMP_START', 'TIMESTAMP_END'))
        except ValueError as exc:
            raise ValueError(f'Invalid timestamp at row {row_number}') from exc
        dt = int((end-start).total_seconds())
        issues = []
        if dt not in (1800, 3600):
            issues.append('invalid interval')
        if expected_dt is None and dt in (1800, 3600):
            expected_dt = dt
        if expected_dt is not None and dt != expected_dt:
            issues.append('mixed interval lengths')
        if previous is not None and start != previous:
            issues.append('gap or overlap')
        if issues:
            report['time_axis_issue_count'] += 1
            if len(report['time_axis_issues']) < 20:
                report['time_axis_issues'].append(dict(row=row_number, start=row['TIMESTAMP_START'],
                    end=row['TIMESTAMP_END'], previous_end=previous.isoformat() if previous else None, issues=issues))
        previous = end
        year = str(start.year)
        if year not in report['years']:
            report['years'][year] = empty_year()
        y = report['years'][year]
        y['rows'] += 1
        y['first'] = y['first'] or row['TIMESTAMP_START']
        y['last_end'] = row['TIMESTAMP_END']
        y['duration_seconds'][str(dt)] += 1
        report['rows'] += 1
        report['first'] = report['first'] or row['TIMESTAMP_START']
        report['last_end'] = row['TIMESTAMP_END']
        values = {name: number(row.get(name)) for name in FIELDS}
        complete = all(values[name] is not None for name in FORCING)
        y['forcing_complete_rows'] += int(complete)
        y['forcing_and_co2_complete_rows'] += int(complete and values['CO2_F_MDS'] is not None)
        for name, value in values.items():
            field = y['fields'][name]
            qc_name = QC_FIELD.get(name, name+'_QC')
            qc = number(row.get(qc_name))
            label = 'absent_column' if qc_name not in header_set else 'missing' if qc is None else format(qc, 'g')
            field['qc'][label] += 1
            if value is None:
                field['missing'] += 1
                continue
            field['valid'] += 1
            field['valid_qc_zero'] += int(qc == 0.)
            field['minimum'] = value if field['minimum'] is None else min(field['minimum'], value)
            field['maximum'] = value if field['maximum'] is None else max(field['maximum'], value)
        # Flags are evidence for preprocessing, never a data repair.
        for name in ('P_F', 'WS_F', 'SW_IN_F', 'LW_IN_F', 'VPD_F'):
            if values[name] is not None and values[name] < 0.:
                y['physical_input_flags'][name+'_negative'] += 1
        if values['PA_F'] is not None and values['PA_F'] <= 0.:
            y['physical_input_flags']['PA_F_nonpositive'] += 1
        if complete:
            tc, pressure, vpd = values['TA_F'], values['PA_F']*1000., values['VPD_F']
            try:
                vapor = 611.2*math.exp(17.67*tc/(tc+243.5))-100.*vpd
                if not 0. <= vapor < pressure:
                    y['physical_input_flags']['vpd_implied_vapor_out_of_range'] += 1
            except (OverflowError, ZeroDivisionError):
                y['physical_input_flags']['vpd_conversion_not_finite'] += 1
    return report


def inventory_archive(path, member=None):
    path = Path(path)
    with path.open('rb') as stream:
        archive_sha = digest(stream)
    with zipfile.ZipFile(path) as archive:
        candidates = [n for n in archive.namelist() if n.endswith('.csv') and
                      ('_FULLSET_HH_' in n or '_FLUXMET_HH_' in n)]
        if member is None:
            if len(candidates) != 1:
                raise ValueError('Select exactly one HH member explicitly')
            member = candidates[0]
        if member not in candidates:
            raise ValueError('Expected an original FULLSET_HH or FLUXMET_HH member')
        with archive.open(member) as stream:
            member_sha = digest(stream)
        with archive.open(member) as stream:
            report = inventory_rows(csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig', newline='')))
        report.update(archive_name=path.name, archive_bytes=path.stat().st_size,
                      archive_sha256=archive_sha, member=member,
                      member_bytes=archive.getinfo(member).file_size, member_sha256=member_sha)
    report['instrument_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['scope'] = 'Original input inventory only; no model run, calibration or observation acceptance.'
    report['interpretation'] = [
        'Local standard time as archived; rows assigned by TIMESTAMP_START year.',
        'valid_qc_zero requires both a finite value and its own QC=0; an absent QC is never zero.',
        'GPP and RECO are partitioned products, not direct observations. No GPP QC is invented from NEE QC.',
        'H_CORR/LE_CORR use their corresponding H_F_MDS_QC/LE_F_MDS_QC flags.',
        'QC counts cover all rows including missing values; valid_qc_zero counts only nonmissing values.',
        'Input flags use the existing Tetens-over-water VPD convention; no humidity or radiation is clipped.']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--member')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inventory_archive(args.archive, args.member)
    data = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(data, mtime=0) if args.output.suffix == '.gz' else data)
    print(json.dumps(dict(rows=report['rows'], years=list(report['years']),
                         time_axis_issues=report['time_axis_issue_count'], output=str(args.output))))


if __name__ == '__main__':
    main()

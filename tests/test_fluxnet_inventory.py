"""Inventory must not promote missing/QC-filled products to observations."""
import csv
import io
import importlib.util
from pathlib import Path
import zipfile
import pytest

spec = importlib.util.spec_from_file_location('fluxnet_inventory', Path(__file__).parents[1]/'benchmarks/inventory_fluxnet_archive.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_qc_is_field_specific_and_requires_finite_value():
    rows = ('TIMESTAMP_START,TIMESTAMP_END,LE_F_MDS,LE_F_MDS_QC,LE_CORR,NEE_VUT_REF,NEE_VUT_REF_QC,GPP_NT_VUT_REF\n'
            '200812312330,200901010000,12,1,15,2,0,4\n'
            '200901010000,200901010030,-9999,0,-9999,3,2,5\n')
    report = module.inventory_rows(csv.DictReader(io.StringIO(rows)))
    assert report['time_axis_issue_count'] == 0
    first, second = (report['years'][y]['fields'] for y in ('2008', '2009'))
    assert first['LE_F_MDS']['valid_qc_zero'] == 0
    assert first['LE_CORR']['qc'] == {'1': 1}
    assert first['NEE_VUT_REF']['valid_qc_zero'] == 1
    assert first['GPP_NT_VUT_REF']['valid_qc_zero'] == 0
    assert first['GPP_NT_VUT_REF']['qc'] == {'absent_column': 1}
    assert second['LE_F_MDS']['valid_qc_zero'] == 0
    assert second['LE_F_MDS']['missing'] == 1
    assert report['years']['2008']['forcing_complete_rows'] == 0


def test_raw_time_gaps_and_nonfinite_values_are_reported():
    rows = ('TIMESTAMP_START,TIMESTAMP_END,TA_F,TA_F_QC\n'
            '200701010000,200701010030,nan,0\n'
            '200701010100,200701010200,20,-9999\n')
    report = module.inventory_rows(csv.DictReader(io.StringIO(rows)))
    assert report['time_axis_issue_count'] == 1
    assert report['time_axis_issues'][0]['issues'] == ['mixed interval lengths', 'gap or overlap']
    field = report['years']['2007']['fields']['TA_F']
    assert field['missing'] == 1 and field['valid_qc_zero'] == 0
    assert field['qc'] == {'0': 1, 'missing': 1}


def test_duplicate_columns_and_ambiguous_archive_fail(tmp_path):
    with pytest.raises(ValueError, match='Duplicate'):
        module.inventory_rows(csv.DictReader(io.StringIO('TIMESTAMP_START,TIMESTAMP_END,TA_F,TA_F\n')))
    path = tmp_path/'ambiguous.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for n in ('FLX_A_FULLSET_HH_2001.csv', 'JPF_B_FLUXMET_HH_2001.csv'):
            archive.writestr(n, 'TIMESTAMP_START,TIMESTAMP_END\n')
    with pytest.raises(ValueError, match='exactly one'):
        module.inventory_archive(path)

import csv
from datetime import datetime, timedelta
import importlib.util
import io
from pathlib import Path
import sys

import numpy as np
import pytest

directory = Path(__file__).parents[1] / 'benchmarks'
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location('fluxnet_station_data', directory / 'fluxnet_station_data.py')
station = importlib.util.module_from_spec(spec)
spec.loader.exec_module(station)


def rows():
    return csv.DictReader(io.StringIO(
        'TIMESTAMP_START,TIMESTAMP_END,TA_F,PA_F,VPD_F,WS_F,SW_IN_F,LW_IN_F,P_F,H_F_MDS,H_F_MDS_QC,GPP_NT_VUT_REF,NEE_VUT_REF_QC\n'
        '200812312330,200901010000,10,100,5,2,0,280,1,20,1,2,0\n'
        '200901010000,200901010030,10,100,5,2,0,280,0,-9999,0,3,0\n'))


def test_units_co2_month_boundary_and_qc_not_silently_relabelled():
    a, report = station.prepare_rows(rows(), start='2008-12-31T23:30:00', end='2009-01-01T00:30:00',
        latitude=37., longitude=101., utc_offset_hours=8, co2_monthly={(2008,12):385., (2009,1):386.})
    assert a['forcing'][0,1] == 283.15
    assert a['forcing'][0,4] == 100000.
    assert a['forcing'][0,7] * 1800 * 1000 == pytest.approx(1.)
    np.testing.assert_array_equal(a['environment'][:,0], [385.,386.])
    assert a['obs_H_F_MDS'][0] == 20 and a['qc_H_F_MDS'][0] == 1
    assert np.isnan(a['obs_H_F_MDS'][1]) and a['qc_H_F_MDS'][1] == 0
    assert np.all(np.isnan(a['qc_GPP_NT_VUT_REF']))
    assert report['total_precipitation_mm'] == 1.
    assert report['humidity_repairs'] == []


def test_explicit_humidity_repair_keeps_raw_error_and_valid_rows():
    t = np.zeros(7); p = np.full(7,100000.)
    v = np.array([3.056, 8., 8., 8., 8., 8., 1.2224])
    stamps = [(datetime(2000,1,1)+timedelta(minutes=30*i)).isoformat() for i in range(7)]
    with pytest.raises(ValueError, match='strict'):
        station.humidity(t,p,v,stamps,1800.)
    q, mask, ledger = station.humidity(t,p,v,stamps,1800.,'rh-interpolate-3h')
    expected_rh = np.linspace(.5,.8,7)
    e = expected_rh*611.2
    np.testing.assert_allclose(q,.622*e/(100000.-.378*e),rtol=1e-14)
    np.testing.assert_array_equal(mask,[False,True,True,True,True,True,False])
    assert len(ledger)==5 and ledger[0]['raw_vapor_pressure_pa'] == pytest.approx(-188.8)
    assert ledger[-1]['right_timestamp']==stamps[-1]


@pytest.mark.parametrize('vpd', [[8.,1.,1.],[1.,1.,8.],[1.,8.,8.,8.,8.,8.,8.,1.]])
def test_humidity_cannot_extrapolate_or_bridge_more_than_three_hours(vpd):
    with pytest.raises(ValueError,match='brackets'):
        station.humidity(np.zeros(len(vpd)),np.full(len(vpd),100000.),vpd,
                         [str(i) for i in range(len(vpd))],1800.,'rh-interpolate-3h')


def test_missing_co2_and_truncated_window_rejected():
    options=dict(start='2008-12-31T23:30:00',end='2009-01-01T00:30:00',latitude=37.,longitude=101.,utc_offset_hours=8)
    with pytest.raises(ValueError,match='CO2'): station.prepare_rows(rows(),co2_monthly={(2008,12):385.},**options)
    options['end']='2009-01-01T01:00:00'
    with pytest.raises(ValueError,match='Incomplete'):
        station.prepare_rows(rows(),co2_monthly={(2008,12):385.,(2009,1):386.},**options)

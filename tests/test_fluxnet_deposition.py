"""Calendar, units and explicit configuration for station deposition."""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from fluxnet_deposition import DepositionSchedule


def config(monthly=True):
    key = 'aerosol_deposition_monthly_kg_m2_s' if monthly else 'aerosol_deposition_kg_m2_s'
    values = np.arange(1, 6)*1e-12
    if monthly: values = np.arange(1, 13)[:, None]*values
    return {key: values.tolist(), 'nitrogen_deposition_gn_m2_s': 2./(365*86400)}


@pytest.mark.parametrize('year,midpoint', [(2001, datetime(2001, 2, 15)),
                                         (2004, datetime(2004, 2, 15, 12))])
def test_interval_midpoint_and_leap_month_centers(year, midpoint):
    d = DepositionSchedule(config())
    np.testing.assert_allclose(d.aerosol(midpoint-timedelta(minutes=15), 1800),
                               np.arange(1, 6)*2e-12, rtol=1e-15, atol=0)
    # At a quarter of the interval from February center to March center,
    # linear interpolation must give 2.25 times the first month's rates.
    march = datetime(year, 3, 16, 12)
    quarter = midpoint + (march-midpoint)/4
    np.testing.assert_allclose(d.aerosol(quarter-timedelta(minutes=15), 1800),
                               np.arange(1, 6)*2.25e-12, rtol=1e-15, atol=0)


def test_year_wrap_and_source_nitrogen_noleap_units():
    d = DepositionSchedule(config())
    expected = np.arange(1, 6)*6.5e-12
    for stamp in (datetime(2000, 12, 31, 23, 45), datetime(2001, 12, 31, 23, 45)):
        np.testing.assert_allclose(d.aerosol(stamp, 1800), expected, rtol=1e-15, atol=0)
    assert d.nitrogen_rate*365*86400 == pytest.approx(2.)
    assert d.nitrogen_rate*366*86400 == pytest.approx(2.*366/365)


def test_constant_compatibility_and_species_separation():
    d = DepositionSchedule(config(False))
    for stamp in (datetime(2001, 1, 1), datetime(2004, 7, 31)):
        np.testing.assert_array_equal(d.aerosol(stamp, 1800), np.arange(1, 6)*1e-12)
    with pytest.raises(ValueError): d.rates[0] = 10.


@pytest.mark.parametrize('problem', ['both', 'neither', 'shape', 'negative', 'nan', 'nnegative', 'nnan'])
def test_ambiguous_or_invalid_rates_fail_before_running(problem):
    c = config()
    if problem == 'both': c['aerosol_deposition_kg_m2_s'] = [0.]*5
    elif problem == 'neither': del c['aerosol_deposition_monthly_kg_m2_s']
    elif problem == 'shape': c['aerosol_deposition_monthly_kg_m2_s'] = [[0.]*5]*11
    elif problem in ('negative', 'nan'): c['aerosol_deposition_monthly_kg_m2_s'][0][0] = -1. if problem == 'negative' else np.nan
    else: c['nitrogen_deposition_gn_m2_s'] = -1. if problem == 'nnegative' else np.nan
    with pytest.raises(ValueError): DepositionSchedule(c)

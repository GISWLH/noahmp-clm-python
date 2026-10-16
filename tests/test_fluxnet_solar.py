from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('fluxnet_solar', Path(__file__).parents[1] / 'benchmarks/fluxnet_solar.py')
solar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solar)


def test_equivalent_instants_across_local_time_and_year_boundary():
    # Same physical instant represented in UTC and UTC+8, across a leap year.
    local = datetime(2000, 1, 1, 2, 15)
    a = solar.solar_geometry(local, 37.607432, 101.332, 8)
    b = solar.solar_geometry(local - timedelta(hours=8), 37.607432, 101.332, 0)
    assert a == pytest.approx(b, abs=1e-12)


def test_equatorial_daylength_and_polar_seasons():
    for month in range(1, 13):
        assert solar.solar_geometry(datetime(2004, month, 15, 12), 0, 0, 0)['daylength_seconds'] == pytest.approx(43200., abs=1e-9, rel=0.)
    north = solar.solar_geometry(datetime(2004, 6, 21, 12), 90, 0, 0)
    south = solar.solar_geometry(datetime(2004, 6, 21, 12), -90, 0, 0)
    assert north['daylength_seconds'] == 86400.
    assert south['daylength_seconds'] == 0.


def test_erbs_independent_published_points_and_energy():
    # Hand-evaluated polynomial at kt=.5; linear and clear-sky branches.
    for k, fraction in [(.1, .991), (.5, .65915), (.9, .165)]:
        result = solar.diffuse_fraction_erbs(500 * k, .5, 1000.)
        assert result['diffuse_fraction'] == pytest.approx(fraction, abs=1e-12)
        assert result['diffuse_horizontal_wm2'] + result['direct_horizontal_wm2'] == pytest.approx(500 * k)
    # Nighttime input is flagged and retained, not silently zeroed.
    result = solar.diffuse_fraction_erbs(4., -.1, 1366.1)
    assert result['low_sun_all_diffuse']
    assert result['diffuse_horizontal_wm2'] == 4.
    assert result['direct_horizontal_wm2'] == 0.


def test_bad_radiation_or_geometry_is_rejected():
    with pytest.raises(ValueError): solar.diffuse_fraction_erbs(-.1, .5, 1366.1)
    with pytest.raises(ValueError): solar.solar_geometry(datetime(2000, 1, 1), 91., 0, 0)

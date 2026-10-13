"""Input conversion and data-integrity tests; synthetic rows are not site evidence."""
import csv
from datetime import datetime, timedelta
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("fluxnet_smoke", Path(__file__).parents[1] / "examples/fluxnet_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def write_day(path, mutate=None):
    rows = []
    start = datetime(2007, 7, 1)
    for i in range(48):
        row = dict(TIMESTAMP_START=(start + timedelta(minutes=30*i)).strftime("%Y%m%d%H%M"),
                   TIMESTAMP_END=(start + timedelta(minutes=30*(i+1))).strftime("%Y%m%d%H%M"),
                   TA_F=20, WS_F=2, PA_F=100, SW_IN_F=500, LW_IN_F=350, P_F=1.8,
                   VPD_F=10, RH=50, LE_F_MDS=100, LE_F_MDS_QC=0, H_F_MDS=-10, H_F_MDS_QC=0)
        rows.append(row)
    if mutate:
        mutate(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_standard_units_precipitation_total_and_measured_only(tmp_path):
    path = tmp_path / "site.csv"
    write_day(path, lambda rows: rows[0].update(LE_F_MDS_QC=1, H_F_MDS=-9999))
    forcing, obs, stamps, meta = smoke.load_fluxnet_csv(path, "2007-07-01", 1)
    assert len(forcing) == 48 and len(stamps) == 48
    f = forcing[0]
    assert f.tair == 293.15 and f.ps == 100000 and f.prcp == pytest.approx(1e-6)
    es = 611.2 * np.exp(17.67 * 20 / (20 + 243.5))
    e = es - 1000
    assert f.qair == pytest.approx(0.622 * e / (100000 - 0.378 * e))
    assert meta["precipitation_mm"] == pytest.approx(48 * 1.8)
    assert np.isnan(obs["le"][0]) and np.isnan(obs["h"][0])
    assert meta["forcing_qc_counts"]["TA_F"] == {"unavailable": 48}


def test_rh_route_and_no_implicit_vpd_fallback(tmp_path):
    path = tmp_path / "site.csv"
    write_day(path, lambda rows: rows[0].update(VPD_F=-9999))
    with pytest.raises(ValueError, match="missing VPD_F"):
        smoke.load_fluxnet_csv(path, "2007-07-01", 1)
    forcing, _, _, _ = smoke.load_fluxnet_csv(path, "2007-07-01", 1, "rh")
    assert forcing[0].qair == pytest.approx(smoke.specific_humidity(20, 100000, rh_percent=50))
    with pytest.raises(ValueError, match="invalid vapor"):
        smoke.specific_humidity(20, 100000, vpd_hpa=100)


@pytest.mark.parametrize("mutate,match", [
    (lambda rows: rows.pop(4), "noncontiguous"),
    (lambda rows: rows[3].update(P_F=-9999), "missing P_F"),
    (lambda rows: rows[3].update(P_F=-1), "negative"),
    (lambda rows: rows.pop(), "incomplete"),
])
def test_bad_forcing_or_time_fails_without_filling(tmp_path, mutate, match):
    path = tmp_path / "site.csv"
    write_day(path, mutate)
    with pytest.raises(ValueError, match=match):
        smoke.load_fluxnet_csv(path, "2007-07-01", 1)


def test_observation_metrics_mask_sign_and_constant_correlation():
    result = smoke.observation_metrics(np.array([2., 4., 100.]), np.array([1., 2., np.nan]))
    assert result["n"] == 2 and result["r"] == pytest.approx(1)
    assert result["bias_W_m2"] == 1.5
    assert result["rmse_W_m2"] == pytest.approx(np.sqrt(2.5))
    assert smoke.observation_metrics(np.ones(3), np.ones(3))["r"] is None

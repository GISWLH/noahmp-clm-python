"""硬约束测试：能量闭合、水平衡、格点独立性。

这三条是仓库的底线——任何阶段的改动都不能破坏它们。
"""
import numpy as np

from noahmp_clm import ColumnState, default_soil_params, synthetic_forcing
from noahmp_clm.driver import run_column, run_columns, step_column


def _storage(s: ColumnState) -> float:
    return float(np.sum(s.smc * s.dz) + s.swe + s.canw)


def test_energy_closure():
    """每步 |Rnet − H − LE − G| < 0.5 W/m2。"""
    s = ColumnState(soil=default_soil_params("loam"))
    diags, _ = run_column(s, synthetic_forcing(n_days=3))
    worst = max(abs(d["residual"]) for d in diags)
    assert worst < 0.5, f"energy residual too large: {worst}"


def test_water_conservation():
    """降水 = ET + 径流 + Δ蓄水，误差 < 1e-9 m。"""
    s0 = ColumnState(soil=default_soil_params("loam"))
    forcings = synthetic_forcing(n_days=3)
    s = s0.copy()
    prcp = et = runoff = 0.0
    s_start = _storage(s)
    for f in forcings:
        s, d = step_column(s, f)
        prcp += d["prcp_mm"] / 1000.0
        et += d["et_mm"] / 1000.0
        runoff += d["runoff_mm"] / 1000.0
    imbalance = prcp - et - runoff - (_storage(s) - s_start)
    assert abs(imbalance) < 1e-9, f"water imbalance: {imbalance} m"


def test_batch_independence():
    """批量跑 N 个柱 ≡ 逐个单柱跑（bit 级一致）。"""
    forcings = synthetic_forcing(n_days=2)
    s1 = ColumnState(soil=default_soil_params("sand"), lai=2.0)
    s2 = ColumnState(soil=default_soil_params("clay"), lai=5.0, albedo=0.25)
    (d1, f1), (d2, f2) = run_columns([s1, s2], [forcings, forcings])
    d1s, f1s = run_column(s1, forcings)
    d2s, f2s = run_column(s2, forcings)
    for a, b in zip(d1, d1s):
        assert abs(a["residual"] - b["residual"]) == 0.0
        assert abs(a["et_mm"] - b["et_mm"]) == 0.0
    for a, b in zip(d2, d2s):
        assert abs(a["runoff_mm"] - b["runoff_mm"]) == 0.0
    assert np.array_equal(f1.smc, f1s.smc) and np.array_equal(f2.tsoil, f2s.tsoil)


def test_snow_accumulation_and_melt():
    """冷期积雪、暖期消融：swe 先增后减。"""
    cold = synthetic_forcing(n_days=4, tmean=268.0, tamp=3.0, seed=1)
    warm = synthetic_forcing(n_days=4, tmean=288.0, tamp=6.0, seed=2)
    s = ColumnState(soil=default_soil_params("loam"))
    _, s = run_column(s, cold)
    swe_after_cold = s.swe
    assert swe_after_cold > 1e-3, f"no snow accumulated: {swe_after_cold}"
    _, s = run_column(s, warm)
    assert s.swe < swe_after_cold, "snow did not melt in warm period"

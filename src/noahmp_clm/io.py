"""IO：合成强迫（测试/示例用）与 CSV 输出。"""
from __future__ import annotations

import csv
import numpy as np

from .column import Forcing


def synthetic_forcing(n_days: int = 5, dt: float = 1800.0,
                      tmean: float = 285.0, tamp: float = 7.0,
                      seed: int = 0) -> list[Forcing]:
    """生成 n_days 的合成日循环强迫（30 分钟步长）。

    气温/短波为正弦日循环；第 2 天 10–14 时安排一场 5 mm/h 的降水，
    用于检验入渗—径流路径。
    """
    rng = np.random.default_rng(seed)
    n = int(n_days * 86400 / dt)
    t = np.arange(n) * dt
    hour = (t / 3600.0) % 24.0
    day = (t / 86400.0).astype(int)

    tair = tmean + tamp * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    swdown = np.maximum(0.0, 850.0 * np.sin(2 * np.pi * (hour - 6.0) / 24.0))
    lwdown = 280.0 + 0.6 * (tair - 273.15)
    rh = 0.65
    es = 611.2 * np.exp(17.67 * (tair - 273.15) / (tair - 29.65))
    ea = rh * es
    qair = 0.622 * ea / (1.0e5 - 0.378 * ea)
    wind = np.clip(3.0 + 0.8 * rng.standard_normal(n), 0.5, None)
    prcp = np.zeros(n)
    storm = (day == 1) & (hour >= 10) & (hour < 14)
    prcp[storm] = 0.005 / 3600.0  # 5 mm/h → m/s

    return [Forcing(dt=dt, tair=float(tair[i]), qair=float(qair[i]),
                    wind=float(wind[i]), ps=1.0e5,
                    swdown=float(swdown[i]), lwdown=float(lwdown[i]),
                    prcp=float(prcp[i]))
            for i in range(n)]


def write_csv(diags: list[dict], path: str) -> None:
    """把逐步诊断写成 CSV（smc/tsoil 展平成列）。"""
    rows = []
    for d in diags:
        row = {k: v for k, v in d.items() if k not in ("smc", "tsoil")}
        for i, v in enumerate(d["smc"]):
            row[f"smc_L{i+1}"] = float(v)
        for i, v in enumerate(d["tsoil"]):
            row[f"tsoil_L{i+1}"] = float(v)
        rows.append(row)
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

"""与 Noah-MP 官方单柱输出对标的轻量脚本。

脚本使用仓库合成强迫运行同一套 Python 单柱，并把日 ET、感热和表层地温
写成 CSV。若 ``--reference`` 指向官方 HRLDAS/Noah-MP 单柱导出的 CSV，
脚本会按 ``day,et_mm,h_wm2,tsoil1_k`` 列计算日偏差；官方 CSV 的生成仍
由用户在 NCAR/noahmp + HRLDAS 环境中完成，本脚本不调用 Fortran 二进制。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from noahmp_clm import ColumnState, default_soil_params, synthetic_forcing
from noahmp_clm.driver import run_column


def _local_daily(days: int, dt: float) -> list[dict[str, float]]:
    state = ColumnState(soil=default_soil_params("loam"))
    forcings = synthetic_forcing(n_days=days, dt=dt)
    diags, _ = run_column(state, forcings)
    steps = int(round(86400.0 / dt))
    daily: list[dict[str, float]] = []
    for day in range(days):
        chunk = diags[day * steps:(day + 1) * steps]
        daily.append({
            "day": float(day + 1),
            "et_mm": float(sum(d["et_mm"] for d in chunk)),
            "h_wm2": float(np.mean([d["h"] for d in chunk])),
            "tsoil1_k": float(np.mean([d["tsoil"][0] for d in chunk])),
        })
    return daily


def _write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["day", "et_mm", "h_wm2", "tsoil1_k"])
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _report_bias(local: list[dict[str, float]], reference: list[dict[str, float]]) -> None:
    count = min(len(local), len(reference))
    if count == 0:
        raise ValueError("reference CSV 没有可比较的日记录")
    print("daily mean bias (Python - Noah-MP):")
    for key in ("et_mm", "h_wm2", "tsoil1_k"):
        bias = float(np.mean([local[i][key] - reference[i][key] for i in range(count)]))
        rmse = float(np.sqrt(np.mean([(local[i][key] - reference[i][key]) ** 2 for i in range(count)])))
        print(f"  {key:10s} bias={bias: .4f}  rmse={rmse: .4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--dt", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/parity_python.csv"))
    parser.add_argument("--reference", type=Path, help="官方 Noah-MP 日 CSV")
    args = parser.parse_args()
    local = _local_daily(args.days, args.dt)
    _write_csv(local, args.output)
    print(f"wrote Python daily diagnostics: {args.output}")
    if args.reference is None:
        print("No official reference supplied; pass --reference <noahmp.csv> to report bias.")
    else:
        _report_bias(local, _read_csv(args.reference))


if __name__ == "__main__":
    main()


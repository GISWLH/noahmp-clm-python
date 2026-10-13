"""One-site FLUXNET2015 readiness smoke, not calibrated physical validation.

Run with PYTHONPATH=src. Inputs stay outside git; JSON contains provenance,
QC counts, original-tolerance parity, and observed LE/H r, bias and RMSE.
No forcing gaps are filled and no observations enter model parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np

from noahmp_clm.column import ColumnState, Forcing
from noahmp_clm.driver import run_column
from noahmp_clm.gpu.pack import (
    DZ_DEFAULT, column_state_to_vec, forcing_to_vec, forcing_vec_to_forcing,
    vec_to_column_state,
)

DIAGNOSTICS = ("ts", "rnet", "h", "le", "g", "et_mm", "prcp_mm", "runoff_mm",
               "melt_mm", "swe", "canw", "residual")
FORCING_FIELDS = ("TA_F", "WS_F", "PA_F", "SW_IN_F", "LW_IN_F", "P_F")
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(row, name, required=True):
    """FLUXNET missing value is -9999, never a physical value."""
    try:
        value = float(row[name])
    except (KeyError, ValueError, TypeError):
        value = float("nan")
    if not math.isfinite(value) or value == -9999:
        if required:
            raise ValueError(f"missing {name} at {row.get('TIMESTAMP_START')}")
        return float("nan")
    return value


def qc_label(value, missing):
    if not math.isfinite(value):
        return missing
    return str(int(value)) if value.is_integer() else str(value)


def specific_humidity(tair_c, pressure_pa, *, vpd_hpa=None, rh_percent=None):
    """Tetens e_s over water, then q = epsilon e / (p - (1-epsilon)e).

    This matches the baseline saturation-vapor-pressure convention; VPD and
    RH are alternative inputs, not quantities that may be silently mixed.
    """
    if not math.isfinite(tair_c) or pressure_pa <= 0:
        raise ValueError("invalid temperature/pressure for humidity")
    es = 611.2 * math.exp(17.67 * tair_c / (tair_c + 243.5))
    if (vpd_hpa is None) == (rh_percent is None):
        raise ValueError("provide exactly one of VPD or RH")
    if vpd_hpa is not None:
        if not math.isfinite(vpd_hpa) or vpd_hpa < 0:
            raise ValueError("VPD must be finite and nonnegative")
        e = es - vpd_hpa * 100.0
    else:
        if not math.isfinite(rh_percent) or not 0 <= rh_percent <= 100:
            raise ValueError("RH must be in [0, 100] percent")
        e = es * rh_percent / 100.0
    if not 0 <= e < pressure_pa:
        raise ValueError("VPD/RH implies invalid vapor pressure; no clipping applied")
    return 0.622 * e / (pressure_pa - 0.378 * e)


def load_fluxnet_csv(path, start, days=30, humidity="vpd"):
    """Read labeled HH/HR columns for an exact, contiguous local-standard window.

    Consolidated *_F forcings may include gapfills/ERA (reported in QC counts).
    Missing/inconsistent forcing or timestamps fail the run; observations are
    kept only when their own LE/H_F_MDS_QC equals 0 (measured).
    """
    path = Path(path)
    begin = datetime.fromisoformat(start)
    if begin.tzinfo is not None or begin.time() != datetime.min.time() or days < 1:
        raise ValueError("start must be a local-standard midnight; days must be positive")
    end = begin + timedelta(days=days)
    humidity_field = {"vpd": "VPD_F", "rh": "RH"}[humidity]
    forcings, stamps, observations = [], [], {"le": [], "h": []}
    qc = {name: Counter() for name in (*FORCING_FIELDS, humidity_field)}
    observation_qc = {key: Counter() for key in observations}
    cursor, interval = begin, None
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"TIMESTAMP_START", "TIMESTAMP_END", *FORCING_FIELDS,
                    humidity_field, "LE_F_MDS", "LE_F_MDS_QC", "H_F_MDS", "H_F_MDS_QC"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing FLUXNET columns: {sorted(missing)}")
        for row in reader:
            stamp = datetime.strptime(row["TIMESTAMP_START"], "%Y%m%d%H%M")
            if not begin <= stamp < end:
                continue
            stop = datetime.strptime(row["TIMESTAMP_END"], "%Y%m%d%H%M")
            dt = (stop - stamp).total_seconds()
            if stamp != cursor or dt not in (1800.0, 3600.0) or stop > end:
                raise ValueError(f"noncontiguous or invalid interval at {stamp.isoformat()}")
            if interval is not None and dt != interval:
                raise ValueError("mixed HH/HR interval lengths")
            interval, cursor = dt, stop
            values = {name: number(row, name) for name in (*FORCING_FIELDS, humidity_field)}
            tc, ps = values["TA_F"], values["PA_F"] * 1000.0
            if not -80 <= tc <= 65 or not 20000 <= ps <= 120000:
                raise ValueError(f"TA/PA outside physical input range at {stamp}")
            if any(values[k] < 0 for k in ("WS_F", "SW_IN_F", "LW_IN_F", "P_F")):
                raise ValueError(f"negative wind/radiation/precipitation at {stamp}")
            q = specific_humidity(tc, ps, **{
                "vpd_hpa" if humidity == "vpd" else "rh_percent": values[humidity_field]})
            forcings.append(Forcing(dt, tc + 273.15, q, values["WS_F"], ps,
                                    values["SW_IN_F"], values["LW_IN_F"],
                                    values["P_F"] / (1000.0 * dt)))
            stamps.append(stamp.strftime("%Y%m%d%H%M"))
            for name in qc:
                flag = number(row, name + "_QC", required=False)
                qc[name][qc_label(flag, "unavailable")] += 1
            for key in observations:
                field = key.upper() + "_F_MDS"
                flag = number(row, field + "_QC", required=False)
                observation_qc[key][qc_label(flag, "missing")] += 1
                value = number(row, field, required=False)
                observations[key].append(value if flag == 0 else float("nan"))
    if cursor != end:
        raise ValueError(f"incomplete requested window: ended {cursor}, expected {end}")
    metadata = {
        "csv_name": path.name, "csv_sha256": sha256(path), "start_local_standard": start,
        "end_exclusive_local_standard": end.isoformat(), "timezone_convention": "local standard, no DST; no UTC conversion",
        "days": days, "steps": len(forcings), "dt_seconds": interval,
        "humidity_source": humidity_field, "forcing_qc_counts": {k: dict(v) for k, v in qc.items()},
        "observation_qc_counts": {k: dict(v) for k, v in observation_qc.items()},
        "observation_policy": "uncorrected LE_F_MDS/H_F_MDS, finite and QC=0 only; no gapfill scoring",
        "precipitation_mm": sum(f.prcp * f.dt * 1000.0 for f in forcings),
    }
    return forcings, {k: np.asarray(v) for k, v in observations.items()}, stamps, metadata


def observation_metrics(model, observed):
    valid = np.isfinite(model) & np.isfinite(observed)
    a, b = np.asarray(model)[valid], np.asarray(observed)[valid]
    if not len(a):
        return {"n": 0, "r": None, "bias_W_m2": None, "rmse_W_m2": None}
    r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else None
    return {"n": len(a), "r": r, "bias_W_m2": float(np.mean(a - b)),
            "rmse_W_m2": float(np.sqrt(np.mean((a - b) ** 2)))}


def compare(actual, expected, rtol, atol=1e-6):
    actual, expected = np.asarray(actual), np.asarray(expected)
    delta = np.abs(actual - expected)
    good = np.isfinite(actual) & np.isfinite(expected) & (delta <= atol + rtol * np.abs(expected))
    finite_delta = delta[np.isfinite(delta)]
    return {"passed": bool(np.all(good)), "failed_values": int(np.count_nonzero(~good)),
            "values": actual.size, "max_abs": float(np.max(finite_delta)) if len(finite_delta) else None,
            "rtol": rtol, "atol": atol}


def numpy_run(f0, i0, forcing):
    state = vec_to_column_state(f0.astype(np.float64), i0, DZ_DEFAULT)
    ds, final = run_column(state, [forcing_vec_to_forcing(row.astype(np.float64)) for row in forcing])
    ff, ii = column_state_to_vec(final)
    return ff, ii, np.asarray([[d[key] for key in DIAGNOSTICS] for d in ds])


def gpu_run(backend, f0, i0, forcing):
    if backend.startswith("jax"):
        import jax
        jax.config.update("jax_enable_x64", True)
        from noahmp_clm.gpu.jax_core import run_columns_jax
        result = run_columns_jax(f0[None, :], i0[None, :], forcing)
        if not all(device.platform == "gpu" for device in result[2].devices()):
            raise RuntimeError("GPU smoke cannot use JAX CPU fallback")
        device = str(result[2].devices())
        host = tuple(np.asarray(x)[0] for x in result)
    else:
        import cupy as cp
        from noahmp_clm.gpu.cuda.loader import run_columns_cuda
        result = run_columns_cuda(f0[None, :], i0[None, :], forcing)
        device = str(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"])
        host = tuple(cp.asnumpy(x)[0] for x in result)
    return host, device


def write_report(path, report):
    """Persist partial results without presenting an in-flight run as complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD local standard time")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--humidity", choices=("vpd", "rh"), default="vpd")
    parser.add_argument("--backends", nargs="*", choices=("jax64", "jax32", "cuda"), default=["jax64", "jax32", "cuda"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--series", type=Path, help="optional local model/observation CSV; do not commit raw observations")
    args = parser.parse_args()
    forcing, obs, stamps, metadata = load_fluxnet_csv(args.csv, args.start, args.days, args.humidity)
    f0, i0 = column_state_to_vec(ColumnState())
    u = np.asarray([forcing_to_vec(f) for f in forcing])
    source_paths = [Path(__file__)] + sorted((ROOT / "src/noahmp_clm").rglob("*.py")) + sorted((ROOT / "src/noahmp_clm/gpu/cuda").glob("*.cu"))
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in source_paths}
    report = {"site": args.site, "input": metadata, "utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "numpy": np.__version__,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "source_sha256": sources, "physical_validation": False,
              "parameters": "unchanged ColumnState defaults: loam, LAI=3, fveg=0.8, z0m=0.05m; uniform 285K; no spinup/calibration; tower-height correction not applied",
              "observed_comparisons": {}, "backend_parity": {}, "run_completed": False,
              "completed_backends": []}
    models = {}
    started = time.perf_counter()
    base = numpy_run(f0, i0, u)
    models["numpy64"] = base[2]
    report["numpy_elapsed_s"] = time.perf_counter() - started
    report["numpy_finite"] = bool(np.all(np.isfinite(base[0])) and np.all(np.isfinite(base[2])))
    report["observed_comparisons"]["numpy64"] = {
        key: observation_metrics(base[2][:, DIAGNOSTICS.index(key)], obs[key]) for key in ("le", "h")}
    report["completed_backends"].append("numpy64")
    write_report(args.output, report)
    print(json.dumps({"completed": "numpy64", "seconds": report["numpy_elapsed_s"]}), flush=True)
    references = {"float64": base}
    for backend in args.backends:
        print(json.dumps({"started": backend, "steps": len(u)}), flush=True)
        dtype = np.float64 if backend == "jax64" else np.float32
        fi, ui = f0.astype(dtype), u.astype(dtype)
        key = np.dtype(dtype).name
        if key not in references:
            references[key] = numpy_run(fi, i0, ui)
        expected_f, expected_i, expected_d = references[key]
        (actual_f, actual_i, actual_d), device = gpu_run(backend, fi, i0, ui)
        models[backend] = actual_d
        rtol = 1e-6 if backend == "jax64" else 1e-4
        active = np.ones(45, dtype=bool)
        for offset in (16, 19, 22):
            active[offset + int(expected_i[32]):offset + 3] = False
        diagnostics = {k: compare(actual_d[:, j], expected_d[:, j], rtol)
                       for j, k in enumerate(DIAGNOSTICS)}
        final = compare(actual_f[:45][active], expected_f[:45][active], rtol)
        integers = bool(np.array_equal(actual_i, expected_i))
        report["backend_parity"][backend] = {
            "device": device, "matched_input_dtype": key,
            "reference": "NumPy64 using identical packed/quantized inputs; ordinary physics unchanged",
            "finite": bool(np.all(np.isfinite(actual_f)) and np.all(np.isfinite(actual_d))),
            "diagnostics": diagnostics, "active_final_state": final,
            "inactive_snow_padding": compare(actual_f[:45][~active], expected_f[:45][~active], rtol),
            "integer_equal": integers,
            "passed": integers and final["passed"] and all(x["passed"] for x in diagnostics.values()),
        }
        report["observed_comparisons"][backend] = {
            key: observation_metrics(actual_d[:, DIAGNOSTICS.index(key)], obs[key]) for key in ("le", "h")}
        report["completed_backends"].append(backend)
        write_report(args.output, report)
        print(json.dumps({"completed": backend, "parity_passed": report["backend_parity"][backend]["passed"]}), flush=True)
    report["source_unchanged"] = all(sha256(ROOT / name) == digest for name, digest in sources.items())
    report["all_parity_passed"] = bool(args.backends) and all(x["passed"] for x in report["backend_parity"].values())
    report["run_completed"] = True
    write_report(args.output, report)
    if args.series:
        args.series.parent.mkdir(parents=True, exist_ok=True)
        with args.series.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["TIMESTAMP_START", "observed_LE_QC0", "observed_H_QC0"] +
                            [f"{name}_{key}" for name in models for key in DIAGNOSTICS])
            for index, stamp in enumerate(stamps):
                writer.writerow([stamp, *[obs[k][index] if math.isfinite(obs[k][index]) else "" for k in ("le", "h")],
                                 *[float(models[name][index, j]) for name in models for j in range(len(DIAGNOSTICS))]])
    print(json.dumps({"output": str(args.output), "steps": len(u), "numpy_finite": report["numpy_finite"],
                      "all_parity_passed": report["all_parity_passed"], "source_unchanged": report["source_unchanged"]}))
    return 0 if report["numpy_finite"] and report["source_unchanged"] and (not args.backends or report["all_parity_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())

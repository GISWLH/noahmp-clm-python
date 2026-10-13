"""CPU-only local temperature-quantization audit of the ordinary energy closure.

Wraps secant_solve transiently, records F(T), F(float32(T)), adjacent float32
temperatures and numerical derivatives, then returns the untouched ordinary root.
The 156 unique H4 input combinations run 48 continuous steps. This isolates one
local source of error; it does not simulate float32 arithmetic or prove a universal
residual floor. No production file or acceptance tolerance is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from noahmp_clm.column import ColumnState, Forcing, default_soil_params
from noahmp_clm.driver import run_column
from noahmp_clm.gpu.pack import (
    column_state_to_vec, forcing_to_vec, vec_to_column_state, forcing_vec_to_forcing,
)
from noahmp_clm.physics import energy


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("case", "step", "T64", "T32", "ulp", "rounding_delta_K", "F_T64",
          "F_T32", "F_lower", "F_upper", "delta_F_rounding", "dF_dT_ulp",
          "dF_dT_2ulp", "linear_delta_F", "linearization_error",
          "slope_half_ulp", "local_neighbor_min_abs_F", "local_neighbor_max_abs_F")


def cases(steps):
    for index in range(156):
        kind = index % 4
        state = ColumnState(
            soil=default_soil_params(("sand", "loam", "clay")[index % 3]),
            lai=1.0 + index % 13 / 4,
            tsoil=np.full(4, 268.0 if kind == 0 else 285.0),
            swe=0.2 if kind == 0 else (0.008 if kind == 1 else 0.),
            snow_t=268.0 if kind == 0 else 273.15,
            canw=0.0004 if kind == 2 else 0.,
        )
        f, i = column_state_to_vec(state)
        f = f.astype(np.float32)
        state = vec_to_column_state(f, i, np.array([.1, .3, .6, 1.]))
        tair, sw, lw, rain = ((268., 100., 280., 1e-5), (295., 500., 350., 0.),
                            (285., 200., 330., 2e-5), (280., 300., 320., 0.))[kind]
        forcings = [Forcing(1800., tair, .004, 2., 1e5, sw, lw, rain if step < 12 else 0.)
                    for step in range(steps)]
        u = np.array([forcing_to_vec(row) for row in forcings], dtype=np.float32)
        yield index, state, [forcing_vec_to_forcing(row) for row in u], f, i, u


def summary(data):
    columns = {name: data[:, index] for index, name in enumerate(FIELDS)}
    result = {"rows": len(data), "finite": bool(np.all(np.isfinite(data))),
              "root_rounding_within_half_ulp": bool(np.all(np.abs(columns["rounding_delta_K"]) <= columns["ulp"] / 2)),
              "ordinary_F_at_root_abs_max": float(np.max(np.abs(columns["F_T64"]))),
              "rounded_temperature_F_abs_max": float(np.max(np.abs(columns["F_T32"]))),
              "temperature_rounding_only_delta_F_abs_max": float(np.max(np.abs(columns["delta_F_rounding"]))),
              "derivative_abs_max_W_m2_K": float(np.max(np.abs(columns["dF_dT_ulp"]))),
              "slope_times_half_ulp_max_W_m2": float(np.max(columns["slope_half_ulp"])),
              "linearization_error_abs_max_W_m2": float(np.max(np.abs(columns["linearization_error"]))),
              "neighbor_best_residual_abs_max_W_m2": float(np.max(columns["local_neighbor_min_abs_F"])),
              "rounded_F_above_2e_minus_3": int(np.count_nonzero(np.abs(columns["F_T32"]) > .002)),
              "ordinary_root_above_stop_1e_minus_3": int(np.count_nonzero(np.abs(columns["F_T64"]) >= .001)),
              "worst": {}}
    for name in ("F_T64", "F_T32", "delta_F_rounding", "slope_half_ulp", "linearization_error"):
        row = data[int(np.argmax(np.abs(columns[name])))]
        result["worst"][name] = {key: float(value) for key, value in zip(FIELDS, row)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--cpu", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=Path, help="Optional compact NPZ of every probe row")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.output.exists() or (args.rows and args.rows.exists()):
        parser.error("Use new output paths")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.cpu})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.rows:
        args.rows.parent.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "src/noahmp_clm/driver.py", ROOT / "src/noahmp_clm/column.py"]
    sources += sorted((ROOT / "src/noahmp_clm/physics").glob("*.py"))
    sources += [ROOT / "src/noahmp_clm/gpu/pack.py", Path(__file__).resolve()]

    def emit(event, **values):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}
        line = json.dumps(row, allow_nan=False)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    emit("start", python=platform.python_version(), numpy=np.__version__,
         affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
         cases=156, steps=args.steps, ordinary_compute_dtype="float64",
         initial_state_forcing_quantized_dtype="float32", gpu_used=False,
         source_sha256={str(p.relative_to(ROOT)).replace("\\", "/"):
                        hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    all_rows = []
    original = energy.secant_solve
    started = time.monotonic()
    input_hash = hashlib.sha256()
    for case, state, forcing, packed_f, packed_i, packed_u in cases(args.steps):
        for array in (packed_f, packed_i, packed_u):
            input_hash.update(array.tobytes())
        rows = []

        def probe(func, x0, x1, *positional, **keywords):
            root = original(func, x0, x1, *positional, **keywords)
            rounded = np.float32(root)
            lower = float(np.nextafter(rounded, np.float32(-np.inf)))
            upper = float(np.nextafter(rounded, np.float32(np.inf)))
            rounded = float(rounded)
            ulp = upper - rounded
            froot, frounded = float(func(root)), float(func(rounded))
            flo, fhi = float(func(lower)), float(func(upper))
            derivative = (float(func(root + ulp)) - float(func(root - ulp))) / (2 * ulp)
            derivative2 = (float(func(root + 2 * ulp)) - float(func(root - 2 * ulp))) / (4 * ulp)
            delta = rounded - root
            difference = frounded - froot
            linear = derivative * delta
            rows.append((case, len(rows), root, rounded, ulp, delta, froot, frounded,
                         flo, fhi, difference, derivative, derivative2, linear,
                         difference - linear, abs(derivative) * ulp / 2,
                         min(abs(flo), abs(frounded), abs(fhi)),
                         max(abs(flo), abs(frounded), abs(fhi))))
            return root

        energy.secant_solve = probe
        try:
            diagnostics, _ = run_column(state, forcing)
        finally:
            energy.secant_solve = original
        if len(rows) != args.steps:
            raise AssertionError("Expected exactly one secant solve per column step")
        array = np.asarray(rows, dtype=np.float64)
        all_rows.extend(rows)
        emit("case", case=case, soil=("sand", "loam", "clay")[case % 3],
             regime=("cold_snow", "melting", "rain", "dry")[case % 4],
             lai=1.0 + case % 13 / 4, summary=summary(array),
             diagnostic_residual_abs_max=float(max(abs(d["residual"]) for d in diagnostics)))
    data = np.asarray(all_rows, dtype=np.float64)
    if args.rows:
        np.savez_compressed(args.rows, fields=np.array(FIELDS), values=data)
    emit("summary", **summary(data), input_sha256=input_hash.hexdigest(),
         elapsed_seconds=time.monotonic() - started,
         rows_file=str(args.rows) if args.rows else None,
         scope="local Ts rounding with float64 frozen-state residual; no float32 arithmetic or trajectory error")


if __name__ == "__main__":
    main()

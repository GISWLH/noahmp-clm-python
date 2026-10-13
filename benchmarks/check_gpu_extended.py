"""J4 synthetic annual/domain integration audit; never a real-continent claim.

Examples (requires an actual CUDA GPU)::

    PYTHONPATH=src python benchmarks/check_gpu_extended.py --mode annual \
        --backend jax64 --days 365 --output outputs/annual-jax64.jsonl
    PYTHONPATH=src python benchmarks/check_gpu_extended.py --mode domain \
        --backend cuda --columns 64800 --output outputs/domain-cuda.jsonl

Annual runs carry all state between daily chunks, save an atomic checkpoint after
each completed day, and may resume into a new JSONL output. Checkpoints carry a
completed-day counter; a resumed day is never counted twice. Water balance is
reported without introducing a new acceptance tolerance. Existing energy and
sample-parity criteria remain strict, including the near-zero residual field.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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
from noahmp_clm.gpu.pack import column_state_to_vec


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ("ts", "rnet", "h", "le", "g", "et_mm", "prcp_mm", "runoff_mm",
               "melt_mm", "swe", "canw", "residual")
SCENARIOS = ("seasonal_freeze_sand", "seasonal_melt_loam", "warm_wet_clay",
             "dry_sand", "cold_snow_loam", "seasonal_melt_clay",
             "transition_sand", "seasonal_loam")
DZ = np.array([0.1, 0.3, 0.6, 1.0])


def source_hashes():
    paths = sorted((ROOT / "src/noahmp_clm").rglob("*.py"))
    paths += sorted((ROOT / "src/noahmp_clm/gpu/cuda").glob("*.cu"))
    paths.append(Path(__file__).resolve())
    return {str(path.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def digest_arrays(*arrays):
    value = hashlib.sha256()
    for array in arrays:
        value.update(str(array.dtype).encode())
        value.update(str(array.shape).encode())
        value.update(np.ascontiguousarray(array).view(np.uint8))
    return value.hexdigest()


def initial_columns(columns, dtype):
    """Eight named regimes plus deterministic within-regime heterogeneity."""
    packed = []
    for group in range(8):
        soil = default_soil_params(("sand", "loam", "clay")[group % 3])
        state = ColumnState(soil=soil, lai=1.0 + group % 3)
        if group in (0, 4):
            state = replace(state, tsoil=np.full(4, 268.0))
        if group == 4:
            state = replace(state, swe=0.2, snow_t=268.0,
                            snow_swe_layers=np.full(3, 0.2 / 3),
                            snow_t_layers=np.full(3, 268.0),
                            snow_density_layers=np.full(3, 300.0))
        elif group in (1, 5):
            state = replace(state, swe=0.008, snow_t=273.15,
                            snow_swe_layers=np.array([0.008]),
                            snow_t_layers=np.array([273.15]),
                            snow_density_layers=np.array([300.0]))
        elif group == 3:
            state = replace(state, smc=np.full(4, soil.smcwlt + 0.001))
        packed.append(column_state_to_vec(state))
    ids = np.arange(columns)
    f = np.asarray([item[0] for item in packed], dtype=dtype)[ids % 8].copy()
    i = np.asarray([item[1] for item in packed], dtype=np.int32)[ids % 8].copy()
    # Differences within each regime make copied/omitted columns observable.
    f[:, 31] *= 0.8 + 0.02 * (ids % 23)
    f[:, 32] = 0.25 + 0.025 * (ids % 23)
    f[:, 33] *= 0.8 + 0.04 * (ids % 11)
    f[:, 4:8] = np.clip(f[:, 4:8] + (0.003 * np.sin(ids))[:, None],
                        0.0, f[:, 39, None])
    return f, i


def daily_forcing(ids, day, dtype, seed, steps=48):
    """Analytic seasonal+diurnal forcing; fixed ID-based perturbations and storms."""
    ids = np.asarray(ids, dtype=np.int64)
    group = ids % 8
    means = np.array([271., 279., 295., 288., 267., 280., 274., 283.])[group, None]
    seasons = np.array([14., 16., 4., 8., 5., 12., 10., 12.])[group, None]
    amplitudes = np.array([5., 7., 4., 8., 3., 6., 4., 6.])[group, None]
    step = np.arange(steps)[None, :]
    hour = (step * 0.5) % 24
    phase = 0.001 * seed + ids[:, None] * 0.017
    tair = means + seasons * np.sin(2 * np.pi * (day - 80) / 365.)
    tair = tair + amplitudes * np.sin(2 * np.pi * (hour - 9) / 24.) + 0.15 * np.sin(phase)
    sw = np.maximum(0., 700. * np.sin(2 * np.pi * (hour - 6) / 24.))
    sw = sw * (0.85 + 0.01 * (ids[:, None] % 23))
    lw = 280. + 0.6 * (tair - 273.15)
    rh = 0.50 + 0.05 * (ids[:, None] % 7)
    es = 611.2 * np.exp(17.67 * (tair - 273.15) / (tair - 29.65))
    ea = rh * es
    qair = 0.622 * ea / (1.e5 - 0.378 * ea)
    wind = 2.5 + 0.75 * np.sin(phase + step * 0.3 + day * 0.1)
    storm = ((day % 7) == (group[:, None] % 7)) & (step >= 10) & (step < 18)
    rates = np.array([2e-6, 1e-6, 4e-6, 0., 1e-6, 2e-6, 1e-6, 2e-6])[group, None]
    u = np.empty((len(ids), steps, 8), dtype=dtype)
    for index, value in enumerate((1800., tair, qair, wind, 1.e5, sw, lw, storm * rates)):
        u[:, :, index] = value
    return u


class Backend:
    def __init__(self, name):
        self.name = name
        if name.startswith("jax"):
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            from noahmp_clm.gpu.jax_core import run_columns_jax
            self.xp, self.run = jnp, run_columns_jax
            devices = jax.devices()
            if jax.default_backend() != "gpu" or not devices or any(
                d.platform != "gpu" or "cuda" not in str(d.client.platform_version).lower()
                for d in devices
            ):
                raise RuntimeError(f"Actual CUDA backend required: {devices}")
            self.device = str(devices[0])
            self.host = lambda x: np.asarray(jax.device_get(x))
            self.sync = lambda result: jax.block_until_ready(result)
            self.version = jax.__version__
        else:
            import cupy as cp
            from noahmp_clm.gpu.cuda.loader import run_columns_cuda
            if cp.cuda.runtime.getDeviceCount() < 1 or cp.cuda.runtime.is_hip:
                raise RuntimeError("Actual NVIDIA CUDA device required")
            self.xp, self.run = cp, run_columns_cuda
            self.host = cp.asnumpy
            self.sync = lambda result: cp.cuda.get_current_stream().synchronize()
            self.device = str(cp.cuda.Device())
            self.version = cp.__version__

    def scalar(self, value):
        return self.host(value).item()

    def assert_device(self, arrays):
        for array in arrays:
            if self.name.startswith("jax"):
                if not array.devices() or any(d.platform != "gpu" for d in array.devices()):
                    raise RuntimeError("Output is not GPU resident")
            elif not isinstance(array, self.xp.ndarray):
                raise RuntimeError("Output is not a CuPy device array")

    def storage(self, state):
        xp = self.xp
        return xp.sum(state[:, 4:8].astype(xp.float64) * xp.asarray(DZ), axis=1) + state[:, 26] + state[:, 30]

    def metrics(self, f, i, d, previous_storage):
        xp = self.xp
        d64 = d.astype(xp.float64)
        sums = xp.sum(d64, axis=(0, 1))
        f64 = f.astype(xp.float64)
        weights = (1.0 + (xp.arange(f.shape[0], dtype=xp.float64) % 1009) / 1009.)
        delta = self.storage(f) - previous_storage
        water_error = delta - xp.sum(d64[:, :, 6] - d64[:, :, 5] - d64[:, :, 7], axis=1) / 1000.
        finite = bool(self.scalar(xp.all(xp.isfinite(f)) & xp.all(xp.isfinite(d))))
        smc_ok = bool(self.scalar(xp.all((f[:, 4:8] >= 0.) & (f[:, 4:8] <= f[:, 39, None]))))
        inactive = xp.arange(3)[None, :] >= i[:, 32, None]
        snow_ok = bool(self.scalar(xp.all(f[:, 16:19] >= 0.) & xp.all(f[:, 26] >= 0.) &
                                   xp.all((i[:, 32] >= 1) & (i[:, 32] <= 3)) &
                                   xp.all(xp.where(inactive, f[:, 16:19] == 0., True))))
        ice_ok = bool(self.scalar(xp.all((f[:, 8:12] >= 0.) & (f[:, 8:12] <= f[:, 4:8]))))
        canopy_ok = bool(self.scalar(xp.all(f[:, 30] >= 0.)))
        residual = float(self.scalar(xp.max(xp.abs(d[:, :, 11]))))
        result = {"finite": finite, "smc_physical_bounds": smc_ok, "snow_nonnegative_count": snow_ok,
                  "soil_ice_physical_bounds": ice_ok, "canopy_nonnegative": canopy_ok,
                  "snow_sum_vs_total_max_abs_m": float(self.scalar(xp.max(xp.abs(
                      xp.sum(f[:, 16:19].astype(xp.float64), axis=1) - f[:, 26])))),
                  "energy_abs_residual_max_W_m2": residual,
                  "energy_limit_W_m2": 0.5, "energy_passed": finite and residual < 0.5,
                  "state_min": self.host(xp.min(f, axis=0)).tolist(),
                  "state_max": self.host(xp.max(f, axis=0)).tolist(),
                  "diagnostic_min": self.host(xp.min(d, axis=(0, 1))).tolist(),
                  "diagnostic_max": self.host(xp.max(d, axis=(0, 1))).tolist(),
                  "checksum": {"state_sum": self.host(xp.sum(f64, axis=0)).tolist(),
                               "diagnostic_sum": self.host(sums).tolist(),
                               "state_weighted_sum": self.host(xp.sum(f64 * weights[:, None], axis=0)).tolist(),
                               "diagnostic_weighted_sum": self.host(xp.sum(d64 * weights[:, None, None], axis=(0, 1))).tolist(),
                               "diagnostic_sumsq": self.host(xp.sum(d64 * d64, axis=(0, 1))).tolist()},
                  "total_ET_mm_column_sum": float(self.scalar(sums[5])),
                  "total_H_MJ_m2_column_sum": float(self.scalar(sums[2])) * 1800. / 1.e6,
                  "water_daily_abs_max_m": float(self.scalar(xp.max(xp.abs(water_error)))),
                  "water_daily_signed_sum_m": float(self.scalar(xp.sum(water_error))),
                  "water_balance_acceptance": "measured_only_no_new_tolerance"}
        result["passed"] = finite and smc_ok and snow_ok and ice_ok and canopy_ok and result["energy_passed"]
        failures = (("state_nonfinite", ~xp.isfinite(f)),
                    ("diagnostic_nonfinite", ~xp.isfinite(d)),
                    ("smc_bounds", (f[:, 4:8] < 0.) | (f[:, 4:8] > f[:, 39, None])),
                    ("snow_negative", f[:, 16:19] < 0.),
                    ("snow_count", (i[:, 32] < 1) | (i[:, 32] > 3)),
                    ("inactive_snow_layer_nonzero", inactive & (f[:, 16:19] != 0.)),
                    ("soil_ice_bounds", (f[:, 8:12] < 0.) | (f[:, 8:12] > f[:, 4:8])),
                    ("canopy_negative", f[:, 30] < 0.),
                    ("energy_residual", xp.abs(d[:, :, 11]) >= 0.5))
        for label, mask in failures:
            if bool(self.scalar(xp.any(mask))):
                coordinate = np.unravel_index(int(self.scalar(xp.argmax(mask))), mask.shape)
                result["first_failure"] = {"kind": label, "coordinate": [int(v) for v in coordinate]}
                break
        if f.shape[0] <= 8:
            result["per_column_ET_mm"] = self.host(xp.sum(d64[:, :, 5], axis=1)).tolist()
            result["per_column_H_MJ_m2"] = self.host(xp.sum(d64[:, :, 2], axis=1) * 1800. / 1.e6).tolist()
            result["per_column_water_error_m"] = self.host(water_error).tolist()
        return result, water_error


def compare_arrays(actual, expected, rtol):
    finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)))
    error = np.abs(actual.astype(np.float64) - expected)
    ratio = error / (1.e-6 + rtol * np.abs(expected))
    result = {"passed": finite and bool(np.all(ratio <= 1.)), "rtol": rtol, "atol": 1.e-6,
            "max_abs": float(np.max(error, initial=0.)), "max_tolerance_ratio": float(np.max(ratio, initial=0.)),
            "mismatch_count": int(np.count_nonzero(ratio > 1.))}
    if ratio.size:
        worst = np.unravel_index(np.argmax(ratio), ratio.shape)
        result.update(worst_index=[int(v) for v in worst],
                      actual_at_worst=float(actual[worst]), expected_at_worst=float(expected[worst]))
        bad = ratio > 1.
        if bad.ndim > 1:
            bad = np.any(bad, axis=tuple(range(bad.ndim - 1)))
        result["failed_last_axis_indices"] = np.flatnonzero(bad).tolist()
        if result["mismatch_count"]:
            result["first_mismatch_index"] = [int(v) for v in np.argwhere(ratio > 1.)[0]]
            result["field_failures"] = {}
            for field in result["failed_last_axis_indices"]:
                locations = np.argwhere(ratio[..., field] > 1.)
                prefix = tuple(int(v) for v in locations[0]) if ratio.ndim > 1 else ()
                first = prefix + (field,)
                result["field_failures"][str(field)] = {
                    "first_index": list(first), "actual": float(actual[first]),
                    "expected": float(expected[first]),
                    "mismatch_count": int(np.count_nonzero(ratio[..., field] > 1.)),
                    "max_abs": float(np.max(error[..., field])),
                    "max_tolerance_ratio": float(np.max(ratio[..., field])),
                }
    return result


def domain_samples(backend, f0, i0, forcing, result, sample_count):
    from noahmp_clm.gpu.pack import vec_to_column_state
    n = len(f0)
    ids = np.unique(np.r_[np.arange(min(8, n)), np.linspace(0, n - 1, sample_count, dtype=int)])
    actual_f, actual_i, actual_d = (backend.host(x[ids]) for x in result)
    rows = []
    rtol = 1.e-6 if backend.name == "jax64" else 1.e-4
    for position, index in enumerate(ids):
        state = vec_to_column_state(f0[index], i0[index], DZ)
        fs = [Forcing(*map(float, row)) for row in forcing[index]]
        diagnostics, final = run_column(state, fs)
        expected_f, expected_i = column_state_to_vec(final)
        expected_d = np.array([[row[key] for key in DIAGNOSTICS] for row in diagnostics])
        single = backend.run(backend.xp.asarray(f0[index:index+1]),
                             backend.xp.asarray(i0[index:index+1]),
                             backend.xp.asarray(forcing[index:index+1]))
        backend.sync(single)
        single_f, single_i, single_d = (backend.host(x)[0] for x in single)
        single_match = (np.array_equal(single_f, actual_f[position]) and
                        np.array_equal(single_i, actual_i[position]) and
                        np.array_equal(single_d, actual_d[position]))
        # Report exact equality separately; apply the requested same-precision
        # numerical gate rather than inventing a bitwise-equality requirement.
        single_state_check = compare_arrays(actual_f[position], single_f, 1.e-6)
        single_diag_check = compare_arrays(actual_d[position], single_d, 1.e-6)
        diag_check = compare_arrays(actual_d[position], expected_d, rtol)
        # Match H2's active snow mask. NumPy's pack() fills inactive snow
        # temperatures with bulk snow_t, whereas JAX uses the freezing point.
        # Those unused slots are not physical layers; report their representation
        # difference separately rather than treating it as state-physics failure.
        active = np.ones(45, dtype=bool)
        for offset in (16, 19, 22):
            active[offset + int(expected_i[32]):offset + 3] = False
        state_check = compare_arrays(actual_f[position, :45][active], expected_f[:45][active], rtol)
        padding_check = compare_arrays(actual_f[position, :45][~active], expected_f[:45][~active], rtol)
        state_check["compared_abi_fields"] = np.flatnonzero(active).tolist()
        state_check["failed_abi_fields"] = np.flatnonzero(active)[state_check.get("failed_last_axis_indices", [])].tolist()
        diag_check["failed_fields"] = [DIAGNOSTICS[k] for k in diag_check.get("failed_last_axis_indices", [])]
        single_diag_check["failed_fields"] = [DIAGNOSTICS[k] for k in single_diag_check.get("failed_last_axis_indices", [])]
        rows.append({"column": int(index), "scenario": SCENARIOS[index % 8],
                     "numpy_diagnostics": diag_check, "numpy_state": state_check,
                     "numpy_state_scope": "physical_fields_0_to_44_with_H2_active_snow_mask",
                     "inactive_snow_padding_representation": padding_check,
                     "numpy_integer_equal": bool(np.array_equal(actual_i[position], expected_i)),
                     "single_column_exact_equal": single_match,
                     "single_column_state": single_state_check,
                     "single_column_diagnostics": single_diag_check,
                     "single_column_diagnostic_max_abs": float(np.max(np.abs(single_d - actual_d[position]))),
                     "passed": diag_check["passed"] and state_check["passed"] and
                               single_state_check["passed"] and single_diag_check["passed"] and
                               bool(np.array_equal(single_i, actual_i[position])) and
                               bool(np.array_equal(actual_i[position], expected_i))})
    # Unique Ts is an observation, not a universal scientific acceptance criterion.
    return {"sample_count": len(rows), "rows": rows,
            "sample_unique_final_ts": int(len(np.unique(actual_f[:, 25]))),
            "passed": all(row["passed"] for row in rows)}


def domain_sensitivity(backend, f0, i0, forcing, original):
    """Perturb every column's air temperature and check observable response."""
    perturbed = forcing.copy()
    perturbed[:, :, 1] += 0.125
    result = backend.run(backend.xp.asarray(f0), backend.xp.asarray(i0),
                         backend.xp.asarray(perturbed))
    backend.sync(result)
    backend.assert_device(result)
    xp = backend.xp
    change = xp.max(xp.abs(result[2] - original[2]), axis=(1, 2))
    unchanged = int(backend.scalar(xp.count_nonzero(change == 0.)))
    finite = bool(backend.scalar(xp.all(xp.isfinite(result[0])) & xp.all(xp.isfinite(result[2]))))
    return {"perturbation": "tair_plus_0.125_K_all_columns_all_steps",
            "unchanged_columns": unchanged, "columns": len(f0), "finite": finite,
            "minimum_column_max_diagnostic_change": float(backend.scalar(xp.min(change))),
            "maximum_column_max_diagnostic_change": float(backend.scalar(xp.max(change))),
            "passed": finite and unchanged == 0}


def save_checkpoint(path, metadata, f, i, cumulative, cumulative_absolute, backend):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        arrays = dict(f=backend.host(f), i=backend.host(i), cumulative=backend.host(cumulative),
                      cumulative_absolute=backend.host(cumulative_absolute))
        metadata = {**metadata, "checkpoint_schema": 2,
                    "payload_sha256": digest_arrays(*arrays.values())}
        np.savez_compressed(stream, metadata=json.dumps(metadata), **arrays)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("annual", "domain"), required=True)
    parser.add_argument("--backend", choices=("jax64", "jax32", "cuda"), required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--columns", type=int, default=64800)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.days, args.columns, args.steps, args.samples) < 1 or args.max_seconds <= 0:
        parser.error("Counts and time budget must be positive")
    if args.mode == "annual" and args.steps != 48:
        parser.error("Annual days must contain exactly 48 half-hour steps")
    if args.resume and args.mode != "annual":
        parser.error("Only annual mode supports checkpoint resume")
    if args.output.exists():
        parser.error("Use a fresh JSONL output; previous evidence is never overwritten")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or args.output.with_suffix(".checkpoint.npz")
    if checkpoint.exists() and not (args.resume and checkpoint.resolve() == args.resume.resolve()):
        parser.error("Existing checkpoint requires --resume or a fresh --checkpoint path")

    def emit(event, **values):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}
        # Nonfinite results are explicitly labelled failures; encode them as strings
        # rather than nonstandard JSON NaN, retaining diagnostic evidence.
        def clean(value):
            if isinstance(value, float) and not np.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            return value
        line = json.dumps(clean(row), allow_nan=False)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        print(line, flush=True)

    started = time.monotonic()
    hashes = source_hashes()
    columns = 8 if args.mode == "annual" else args.columns
    dtype = np.dtype("float64" if args.backend == "jax64" else "float32")
    metadata = {"backend": args.backend, "dtype": str(dtype), "seed": args.seed,
                "columns": columns, "source_hashes": hashes, "completed_days": 0,
                "scenario_names": SCENARIOS, "input_kind": "synthetic_no_geographic_observations"}
    f0, i0 = initial_columns(columns, dtype)
    cumulative0 = np.zeros(columns, dtype=np.float64)
    cumulative_absolute0 = np.zeros(columns, dtype=np.float64)
    start_day = 0
    if args.resume:
        with np.load(args.resume, allow_pickle=False) as saved:
            old = json.loads(str(saved["metadata"]))
            if old.get("checkpoint_schema") != 2:
                raise ValueError("Unsupported checkpoint schema; cumulative absolute error must be preserved")
            for key in ("backend", "dtype", "seed", "columns", "source_hashes"):
                if old[key] != metadata[key]:
                    raise ValueError(f"Checkpoint mismatch: {key}")
            if not old.get("last_day_passed", False):
                raise ValueError("A failed-day checkpoint is evidence, not an accepted resume point")
            start_day = int(old["completed_days"])
            f0, i0, cumulative0, cumulative_absolute0 = (
                saved[name].copy() for name in ("f", "i", "cumulative", "cumulative_absolute"))
            if digest_arrays(f0, i0, cumulative0, cumulative_absolute0) != old["payload_sha256"]:
                raise ValueError("Checkpoint payload hash mismatch")
        if f0.shape != (columns, 70) or i0.shape != (columns, 33) or f0.dtype != dtype or i0.dtype != np.int32:
            raise ValueError("Checkpoint ABI/dtype mismatch")
        if cumulative0.shape != (columns,) or cumulative_absolute0.shape != (columns,):
            raise ValueError("Checkpoint water-balance shape mismatch")
        if start_day < 0 or any(not np.isfinite(array).all() for array in (f0, cumulative0, cumulative_absolute0)):
            raise ValueError("Checkpoint contains invalid completed day or nonfinite values")
        if start_day >= args.days:
            raise ValueError("Checkpoint already completed the requested days")
    metadata["completed_days"] = start_day
    emit("start", mode=args.mode, requested_days=args.days if args.mode == "annual" else None,
         start_day=start_day, initial_state_sha256=digest_arrays(f0, i0),
         resume=str(args.resume) if args.resume else None, python=platform.python_version(),
         max_seconds=args.max_seconds, **metadata)
    try:
        backend = Backend(args.backend)
        emit("backend", device=backend.device, library_version=backend.version,
             production_dtype=str(dtype), reduction_dtype="float64", actual_cuda_required=True)
        f, i = backend.xp.asarray(f0), backend.xp.asarray(i0)
        cumulative = backend.xp.asarray(cumulative0)
        cumulative_absolute = backend.xp.asarray(cumulative_absolute0)
        failed = False
        completed = start_day
        target_days = args.days if args.mode == "annual" else 1
        for day in range(start_day, target_days):
            # One budget origin for the invocation, including backend setup.
            # A running GPU chunk is non-preemptible; check before scheduling
            # another day and report any one-chunk overshoot explicitly.
            if time.monotonic() - started >= args.max_seconds:
                break
            forcing = daily_forcing(np.arange(columns), day, dtype, args.seed, args.steps)
            previous_storage = backend.storage(f)
            previous_f, previous_i = f, i
            begin = time.monotonic()
            result = backend.run(f, i, backend.xp.asarray(forcing))
            backend.sync(result)
            backend.assert_device(result)
            f, i, d = result
            if f.dtype != dtype or d.dtype != dtype or i.dtype != np.int32:
                raise AssertionError(f"Unexpected production dtype: {f.dtype}, {i.dtype}, {d.dtype}")
            if f.shape != (columns, 70) or i.shape != (columns, 33) or d.shape != (columns, args.steps, 12):
                raise AssertionError("Output does not match requested ABI/column/step shape")
            if not bool(backend.scalar(backend.xp.all(i[:, :32] == previous_i[:, :32]))):
                raise AssertionError("A physics step modified immutable option fields")
            metrics, water_error = backend.metrics(f, i, d, previous_storage)
            cumulative += water_error
            cumulative_absolute += backend.xp.abs(water_error)
            completed = day + 1
            failed |= not metrics["passed"]
            emit("day" if args.mode == "annual" else "domain", day_index_zero_based=day,
                 completed_days=completed, columns=columns, steps=args.steps,
                 forcing_sha256=digest_arrays(forcing),
                 integration_and_reduce_seconds=time.monotonic() - begin,
                 water_cumulative_abs_max_m=float(backend.scalar(backend.xp.max(backend.xp.abs(cumulative)))),
                 water_sum_daily_absolute_error_max_m=float(backend.scalar(backend.xp.max(cumulative_absolute))),
                 water_sum_daily_absolute_error_column_sum_m=float(backend.scalar(backend.xp.sum(cumulative_absolute))),
                 water_cumulative_signed_per_column_m=backend.host(cumulative).tolist() if columns <= 8 else None,
                 water_cumulative_absolute_per_column_m=backend.host(cumulative_absolute).tolist() if columns <= 8 else None,
                 **metrics)
            if args.mode == "annual":
                save_checkpoint(checkpoint, {**metadata, "completed_days": completed,
                                             "last_day_passed": metrics["passed"]}, f, i, cumulative,
                                cumulative_absolute, backend)
            else:
                samples = domain_samples(backend, f0, i0, forcing, result, args.samples)
                emit("domain_samples", **samples)
                failed |= not samples["passed"]
                sensitivity = domain_sensitivity(backend, f0, i0, forcing, result)
                emit("domain_sensitivity", **sensitivity)
                failed |= not sensitivity["passed"]
            if "first_failure" in metrics:
                column = metrics["first_failure"]["coordinate"][0]
                failure_path = args.output.with_suffix(f".failure-day{day}-column{column}.npz")
                np.savez_compressed(failure_path, f=backend.host(previous_f[column]),
                                    i=backend.host(previous_i[column]), forcing=forcing[column],
                                    final_f=backend.host(f[column]), diagnostics=backend.host(d[column]))
                emit("failure_replay", path=str(failure_path), day_index_zero_based=day,
                     column=column, detail=metrics["first_failure"])
            if failed or time.monotonic() - started > args.max_seconds:
                break
        complete = completed == target_days
        emit("summary", status="failed" if failed else "passed" if complete else "budget_exhausted",
             complete=complete, completed_days=completed, requested_days=target_days,
             continuous_steps_completed=completed * args.steps,
             this_invocation_steps=(completed - start_day) * args.steps,
             wall_seconds=time.monotonic() - started,
             budget_seconds=args.max_seconds,
             budget_overshoot_seconds=max(0., time.monotonic() - started - args.max_seconds),
             budget_enforcement="invocation_wall_clock_checked_between_daily_chunks",
             checkpoint=str(checkpoint) if args.mode == "annual" else None,
             water_balance_status="reported_without_new_acceptance_tolerance")
        return 1 if failed else 0 if complete else 2
    except Exception as exc:
        emit("error", exception_type=type(exc).__name__, message=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())

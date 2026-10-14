"""H6: implemented-option parity and daily JAX float32/float64 audit on CUDA.

Run from the repository root with its Linux GPU virtual environment::

    PYTHONPATH=src python benchmarks/check_gpu_options.py \
        --output outputs/gpu_options.jsonl --summary outputs/gpu_options.json

Every accepted value of every PhysicsOptions field is inventoried. Implemented
and ABI-only values run eight columns for 24 steps; unsupported physics values
must raise NotImplementedError in the ordinary implementation and are never
claimed to pass GPU parity. Singleton configuration fields are reported as such.
The eight 48-step precision scenarios use default options and integrate a full
86400 s day. Precision differences are measurements, without an invented pass
threshold. A failed assertion/exception yields a failure record and nonzero exit;
an XLA process abort leaves an unfinished JSONL log and no success summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import traceback

import numpy as np

from noahmp_clm.column import ColumnState, Forcing, default_soil_params
from noahmp_clm.driver import run_column
from noahmp_clm.gpu.pack import (
    OPTION_FIELDS, SNOW_COUNT_INDEX, column_state_to_vec, forcing_to_vec,
)
from noahmp_clm.options import PhysicsOptions


DIAGNOSTICS = (
    "ts", "rnet", "h", "le", "g", "et_mm", "prcp_mm", "runoff_mm",
    "melt_mm", "swe", "canw", "residual",
)
STEPS = 24
DAILY_STEPS = 48


@dataclass(frozen=True)
class OptionSpec:
    category: str
    values: tuple[int, ...]
    reason: str
    unsupported: tuple[int, ...] = ()


# Reviewed against options.py, driver.py, physics/* and gpu/jax_core.py.
# A new field fails inventory validation until it has an explicit classification.
SPECS = {
    "OptDynamicVeg": OptionSpec("singleton", (4,), "Input LAI only; no dynamic vegetation selector."),
    "OptRainSnowPartition": OptionSpec("physics", (1, 2, 3, 4, 5), "physics.precip.partition_precipitation", (4,)),
    "OptSoilWaterTranspiration": OptionSpec("physics", (1, 2, 3), "physics.stress.soil_water_stress"),
    "OptGroundResistanceEvap": OptionSpec("physics", (1, 2, 3, 4), "physics.ground_evap.ground_resistance"),
    "OptSurfaceDrag": OptionSpec("physics", (1, 2), "physics.surface_drag.exchange_velocity"),
    "OptStomataResistance": OptionSpec("physics", (1, 2), "physics.canopy.stomatal_resistance"),
    "OptSnowAlbedo": OptionSpec("physics", (1, 2, 3), "physics.snow.age_albedo; SNICAR unimplemented.", (3,)),
    "OptCanopyRadiationTransfer": OptionSpec("abi_only", (1, 2, 3), "Fixed big-leaf radiation; selector is not consumed."),
    "OptSnowSoilTempTime": OptionSpec("abi_only", (1, 2, 3), "Fixed time integration; selector is not consumed."),
    "OptSnowThermConduct": OptionSpec("physics", (1, 2, 3, 4, 5), "physics.snow.snow_thermal_conductivity"),
    "OptSoilTemperatureBottom": OptionSpec("abi_only", (1, 2), "Fixed zero-flux bottom in step_soil_heat; selector is not consumed."),
    "OptSoilSupercoolWater": OptionSpec("physics", (1, 2), "physics.frozen_soil.unfrozen_water_content"),
    "OptRunoffSurface": OptionSpec("physics", tuple(range(1, 9)), "physics.runoff.route_runoff; diagnostics expose only summed runoff.", (5, 6, 7, 8)),
    "OptRunoffSubsurface": OptionSpec("physics", tuple(range(1, 9)), "physics.runoff.route_runoff; diagnostics expose only summed runoff.", (5, 6, 7, 8)),
    "OptSoilPermeabilityFrozen": OptionSpec("physics", (1, 2), "physics.frozen_soil.permeability_factor"),
    "OptDynVicInfiltration": OptionSpec("abi_only", (1, 2, 3), "Dynamic VIC is unimplemented; selector is not consumed."),
    "OptTileDrainage": OptionSpec("abi_only", (0, 1, 2), "Tile drainage is unimplemented; selector is not consumed."),
    "OptIrrigation": OptionSpec("abi_only", (0, 1, 2, 3), "Irrigation is unimplemented; selector is not consumed."),
    "OptIrrigationMethod": OptionSpec("abi_only", (0, 1, 2, 3), "Irrigation is unimplemented; selector is not consumed."),
    "OptCropModel": OptionSpec("abi_only", (0, 1), "Crop model is unimplemented; selector is not consumed."),
    "OptSoilProperty": OptionSpec("abi_only", (1, 2, 3, 4), "Explicit SoilParams are used; selector is not consumed."),
    "OptPedotransfer": OptionSpec("singleton", (1,), "Explicit SoilParams are used; no pedotransfer selection."),
    "OptGlacierTreatment": OptionSpec("abi_only", (1, 2), "Glacier treatment is unimplemented; selector is not consumed."),
    "OptSnowCompaction": OptionSpec("physics", (1, 2), "physics.snow.compact_snow"),
    "OptSnowCoverGround": OptionSpec("abi_only", (1, 2), "Fixed snow presence threshold; selector is not consumed."),
    "OptWetlandModel": OptionSpec("abi_only", (0, 1, 2), "Wetland model is unimplemented; selector is not consumed."),
    "OptSnicarSnowShape": OptionSpec("abi_only", (1, 2, 3, 4), "SNICAR is unimplemented; selector is not consumed."),
    "OptSnicarRTSolver": OptionSpec("abi_only", (1, 2), "SNICAR is unimplemented; selector is not consumed."),
    "OptSnicarBandNum": OptionSpec("abi_only", (1, 2), "SNICAR is unimplemented; selector is not consumed."),
    "OptSnicarSolarSpec": OptionSpec("abi_only", (1, 2, 3, 4, 5, 6), "SNICAR is unimplemented; selector is not consumed."),
    "OptSnicarSnwOptic": OptionSpec("abi_only", (1, 2, 3), "SNICAR is unimplemented; selector is not consumed."),
    "OptSnicarDustOptic": OptionSpec("abi_only", (1, 2, 3), "SNICAR is unimplemented; selector is not consumed."),
}


def scenarios(steps: int, options: PhysicsOptions):
    """Independent deterministic scenarios, with no dependency on pytest helpers."""
    result = []
    names = (
        "freezing_sand", "melting_loam", "heavy_rain_clay", "drought_sand",
        "three_layer_snow_loam", "melting_clay", "rain_snow_transition_sand",
        "drought_loam",
    )
    for index, name in enumerate(names):
        soil = default_soil_params(("sand", "loam", "clay")[index % 3])
        state = ColumnState(soil=soil, lai=1.0 + index % 3, options=options)
        if index == 0:
            state = replace(state, tsoil=np.full(4, 268.0))
        elif index == 4:
            state = replace(
                state, tsoil=np.full(4, 268.0), swe=0.2, snow_t=268.0,
                snow_swe_layers=np.full(3, 0.2 / 3),
                snow_t_layers=np.full(3, 268.0),
                snow_density_layers=np.full(3, 300.0),
            )
        elif index in (1, 5):
            state = replace(
                state, swe=0.008, snow_t=273.15,
                snow_swe_layers=np.array([0.008]),
                snow_t_layers=np.array([273.15]),
                snow_density_layers=np.array([300.0]),
            )
        elif index in (3, 7):
            state = replace(state, smc=np.full(4, soil.smcwlt + 0.001))
        forcings = []
        for step in range(steps):
            if index in (0, 4):
                tair, sw, lw = 268.0, 100.0, 280.0
                prcp = 1.0e-5 if step < 12 else 0.0
            elif index in (1, 5):
                tair, sw, lw, prcp = 295.0, 500.0, 350.0, 0.0
            elif index == 2:
                tair, sw, lw = 285.0, 200.0, 330.0
                prcp = 2.0e-5 if 10 <= step < 20 else 0.0
            elif index == 6:
                # Jordan/BATS/freezing-threshold produce distinct precipitation.
                tair, sw, lw = 274.2, 200.0, 310.0
                prcp = 1.0e-6 if step < 12 else 0.0
            else:
                tair, sw, lw, prcp = 280.0, 300.0, 320.0, 0.0
            forcings.append(Forcing(1800.0, tair, 0.004, 2.0, 1.0e5, sw, lw, prcp))
        result.append((name, state, forcings))
    return result


def reference(cases):
    result = []
    for name, state, forcings in cases:
        diagnostics, final = run_column(state, forcings)
        f, i = column_state_to_vec(final)
        diag = np.asarray([[row[key] for key in DIAGNOSTICS] for row in diagnostics])
        result.append((name, f, i, diag))
    return result


def gpu_run(jax, runner, cases, dtype):
    packed = [column_state_to_vec(state) for _, state, _ in cases]
    f = np.asarray([pair[0] for pair in packed], dtype=dtype)
    i = np.asarray([pair[1] for pair in packed], dtype=np.int32)
    u = np.asarray([[forcing_to_vec(x) for x in fs] for _, _, fs in cases], dtype=dtype)
    result = runner(*jax.device_put((f, i, u)))
    jax.block_until_ready(result)
    for array in result:
        if not array.devices() or any(device.platform != "gpu" for device in array.devices()):
            raise AssertionError(f"Result is not on GPU: {array.devices()}")
    if result[0].dtype != dtype or result[2].dtype != dtype:
        raise AssertionError(f"Requested {dtype}, got {result[0].dtype}/{result[2].dtype}")
    host = tuple(np.asarray(array) for array in result)
    if any(not np.all(np.isfinite(array)) for array in host):
        raise AssertionError("Nonfinite GPU result")
    return host


def compare(actual, expected, rtol, atol):
    """Report the H2 allclose criterion, including near-zero absolute tolerance."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)))
    if not finite:
        return {"passed": False, "nonfinite": True, "rtol": rtol, "atol": atol}
    absolute = np.abs(actual - expected)
    ratio = absolute / (atol + rtol * np.abs(expected))
    return {
        "passed": bool(np.all(ratio <= 1.0)), "rtol": rtol, "atol": atol,
        "max_abs_error": float(np.max(absolute, initial=0.0)),
        "max_tolerance_ratio": float(np.max(ratio, initial=0.0)),
    }


def parity_rows(gpu, refs):
    """Match the existing mixed H2 diagnostic/state groups and tolerances."""
    rows = []
    for index, (name, f, i, diag) in enumerate(refs):
        actual_f, actual_i, actual_diag = (array[index] for array in gpu)
        count = int(i[SNOW_COUNT_INDEX])
        groups = {
            "diagnostics_ts_to_prcp": compare(actual_diag[:, :7], diag[:, :7], 1e-6, 1e-6),
            "diagnostics_runoff_melt_swe": compare(actual_diag[:, 7:10], diag[:, 7:10], 1e-4, 1e-4),
            "diagnostics_canw_residual": compare(actual_diag[:, 10:], diag[:, 10:], 1e-6, 1e-6),
            "state_soil_root": compare(actual_f[:16], f[:16], 1e-6, 1e-6),
            "state_scalars_parameters": compare(actual_f[25:45], f[25:45], 1e-6, 1e-6),
        }
        for offset, label in ((16, "snow_swe"), (19, "snow_t"), (22, "snow_density")):
            groups[label] = compare(actual_f[offset:offset + count], f[offset:offset + count], 1e-4, 1e-4)
        ints_equal = bool(np.array_equal(actual_i, i))
        residual = float(np.max(np.abs(actual_diag[:, 11])))
        rows.append({
            "scenario": name, "passed": all(g["passed"] for g in groups.values()) and ints_equal and residual < 0.5,
            "groups": groups, "integer_state_equal": ints_equal,
            "max_abs_residual_W_m2": residual, "residual_limit_W_m2": 0.5,
            "melt_occurred": bool(np.any(diag[:, 8] > 0.0)),
        })
    return rows


def inventory():
    names = tuple(field.name for field in fields(PhysicsOptions))
    if names != OPTION_FIELDS or set(names) != set(SPECS):
        raise AssertionError("PhysicsOptions/ABI/audit inventory differ; update the audit classification")
    defaults = PhysicsOptions()
    for name, spec in SPECS.items():
        for value in spec.values:
            replace(defaults, **{name: value})
        if not set(spec.unsupported).issubset(spec.values):
            raise AssertionError(f"Invalid unsupported-value inventory for {name}")
        supported = tuple(value for value in spec.values if value not in spec.unsupported)
        if len(supported) < (1 if spec.category == "singleton" else 2):
            raise AssertionError(f"Insufficient supported audit values for {name}")
    return names


def precision_rows(gpu32, gpu64, cases):
    rows = []
    for index, (name, _, fs) in enumerate(cases):
        dt = np.asarray([forcing.dt for forcing in fs], dtype=np.float64)
        if len(fs) != DAILY_STEPS or float(np.sum(dt)) != 86400.0:
            raise AssertionError("Daily precision requires 48 steps covering exactly 86400 seconds")
        # Both sums are accumulated in float64; this isolates model precision
        # from a separate low-precision reduction error.
        d32 = gpu32[2][index].astype(np.float64)
        d64 = gpu64[2][index]
        quantities = {}
        for key, a32, a64 in (
            ("ET_mm", np.sum(d32[:, 5]), np.sum(d64[:, 5])),
            ("sensible_H_MJ_m2", np.sum(d32[:, 2] * dt) / 1e6, np.sum(d64[:, 2] * dt) / 1e6),
        ):
            quantities[key] = {
                "float32": float(a32), "float64": float(a64),
                "bias_32_minus_64": float(a32 - a64),
                "absolute_bias": float(abs(a32 - a64)),
                "relative_bias": float((a32 - a64) / abs(a64)) if a64 != 0 else None,
            }
        rows.append({"scenario": name, "steps": DAILY_STEPS, "seconds": 86400,
                     "status": "measured", "accumulation_dtype": "float64", **quantities})
    return rows


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parent.parent, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/gpu_options.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/gpu_options.json"))
    args = parser.parse_args()
    if args.output.resolve() == args.summary.resolve():
        parser.error("JSONL output and JSON summary must be different files")
    for path in (args.output, args.summary):
        if path.exists():
            parser.error(f"Refusing to overwrite an existing audit: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    def emit(row):
        row = {"utc": datetime.now(timezone.utc).isoformat(), **row}
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)
        records.append(row)

    failed = 0
    try:
        names = inventory()
        emit({"event": "start", "git_revision": git_revision(), "python": platform.python_version(),
              "numpy": np.__version__, "matrix_steps": STEPS, "daily_steps": DAILY_STEPS,
              "option_field_count": len(names), "default_options": vars(PhysicsOptions())})
        import jax
        jax.config.update("jax_enable_x64", True)
        from noahmp_clm.gpu.jax_core import run_columns_jax
        devices = jax.devices()
        if jax.default_backend() != "gpu" or not devices or any(d.platform != "gpu" for d in devices):
            raise AssertionError(f"H6 requires JAX-CUDA; backend={jax.default_backend()}, devices={devices}")
        # JAX uses platform='gpu' for CUDA and ROCm. Require its CUDA plugin too.
        if not any("cuda" in str(d.client.platform_version).lower() for d in devices):
            raise AssertionError(f"Expected CUDA platform, got {[d.client.platform_version for d in devices]}")
        emit({"event": "environment", "jax": jax.__version__, "x64_enabled": bool(jax.config.x64_enabled),
              "backend": jax.default_backend(), "devices": [str(d) for d in devices],
              "platform_versions": [str(d.client.platform_version) for d in devices]})
        defaults = PhysicsOptions()
        base_cases = scenarios(STEPS, defaults)
        base_refs = reference(base_cases)
        base_gpu = gpu_run(jax, run_columns_jax, base_cases, np.dtype("float64"))
        baseline_rows = parity_rows(base_gpu, base_refs)
        baseline_passed = all(row["passed"] for row in baseline_rows)
        failed += not baseline_passed
        emit({"event": "baseline", "passed": baseline_passed, "scenarios": baseline_rows})
        for name in names:
            spec = SPECS[name]
            supported = [value for value in spec.values if value not in spec.unsupported]
            emit({"event": "option_inventory", "option": name, "category": spec.category,
                  "accepted_values": spec.values, "audited_values": supported,
                  "unsupported_physics_values": spec.unsupported, "reason": spec.reason,
                  "two_values_possible": spec.category != "singleton"})
            for value in spec.values:
                base = {"option": name, "value": value, "category": spec.category}
                emit({"event": "case_start", **base})
                try:
                    cases = scenarios(STEPS, replace(defaults, **{name: value}))
                    if value in spec.unsupported:
                        try:
                            reference(cases)
                        except NotImplementedError as exc:
                            emit({"event": "option_result", **base, "status": "unsupported_expected",
                                  "reference_error": str(exc), "gpu_executed": False,
                                  "note": "JAX switch clamps out-of-range values; no parity/support claim is made."})
                            continue
                        raise AssertionError("Expected ordinary implementation to reject this unsupported value")
                    refs = reference(cases)
                    gpu = gpu_run(jax, run_columns_jax, cases, np.dtype("float64"))
                    rows = parity_rows(gpu, refs)
                    same_numpy = all(np.array_equal(item[1], base_item[1]) and np.array_equal(item[3], base_item[3])
                                     for item, base_item in zip(refs, base_refs))
                    same_jax = bool(np.array_equal(gpu[0], base_gpu[0]) and np.array_equal(gpu[2], base_gpu[2]))
                    noop_verified = same_numpy and same_jax if spec.category in ("abi_only", "singleton") else None
                    passed = all(row["passed"] for row in rows) and noop_verified is not False
                    failed += not passed
                    emit({"event": "option_result", **base, "status": "passed" if passed else "failed",
                          "steps": STEPS, "columns": len(cases), "scenarios": rows,
                          "same_observable_output_as_default_numpy": same_numpy,
                          "same_observable_output_as_default_jax": same_jax,
                          "abi_only_noop_verified": noop_verified})
                except Exception as exc:
                    failed += 1
                    emit({"event": "option_result", **base, "status": "failed",
                          "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        emit({"event": "precision_start", "steps": DAILY_STEPS, "options": "defaults"})
        daily_cases = scenarios(DAILY_STEPS, defaults)
        daily64 = gpu_run(jax, run_columns_jax, daily_cases, np.dtype("float64"))
        daily32 = gpu_run(jax, run_columns_jax, daily_cases, np.dtype("float32"))
        for row in precision_rows(daily32, daily64, daily_cases):
            emit({"event": "daily_precision", **row})
    except Exception as exc:
        failed += 1
        emit({"event": "fatal", "status": "failed", "error": f"{type(exc).__name__}: {exc}",
              "traceback": traceback.format_exc()})
    summary = {
        "event": "summary", "status": "passed" if failed == 0 else "failed", "failures": int(failed),
        "scope": "Implemented physics and explicitly identified ABI-only selectors; not full Noah-MP option support.",
        "precision_policy": "Report daily float32 minus float64 bias; no pass threshold was specified.",
        "jsonl": str(args.output), "json": str(args.summary),
    }
    emit(summary)
    args.summary.write_text(json.dumps({**summary, "records": records}, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

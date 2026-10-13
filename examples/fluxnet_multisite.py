"""Bounded real-data check; observations never select parameters or tune the model.

Manifest entries require site/csv/start/days. Run with PYTHONPATH=src.
All step states and diagnostics are retained in NPZ beside the JSON report.
The M2 criteria are read from benchmarks/closeout_criteria.json, never redefined.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from fluxnet_smoke import (DIAGNOSTICS, ROOT, load_fluxnet_csv,
                           observation_metrics, sha256, write_report)
from noahmp_clm.column import ColumnState
from noahmp_clm.driver import step_column
from noahmp_clm.gpu.pack import (DZ_DEFAULT, column_state_to_vec,
                                forcing_to_vec, forcing_vec_to_forcing,
                                vec_to_column_state)


def numpy_trace(f0, i0, forcing):
    states, integers, diagnostics = [], [], []
    for f, i, u in zip(f0, i0, forcing):
        state = vec_to_column_state(f.astype(np.float64), i, DZ_DEFAULT)
        fs, ix, ds = [f.astype(np.float64)], [i.copy()], []
        for row in u:
            state, diagnostic = step_column(state, forcing_vec_to_forcing(row.astype(np.float64)))
            ff, ii = column_state_to_vec(state)
            fs.append(ff)
            ix.append(ii)
            ds.append([diagnostic[key] for key in DIAGNOSTICS])
        states.append(fs)
        integers.append(ix)
        diagnostics.append(ds)
    return tuple(np.asarray(x) for x in (states, integers, diagnostics))


def jax_trace(f0, i0, forcing):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from noahmp_clm.gpu.jax_core import run_columns_jax, step_column_jax

    def one(f, i, u):
        def body(carry, row):
            ff, ii, d = step_column_jax(*carry, row)
            return (ff, ii), (ff, ii, d)
        return jax.lax.scan(body, (f, i), u)[1]

    raw = jax.jit(jax.vmap(one))(jnp.asarray(f0), jnp.asarray(i0), jnp.asarray(forcing))
    if not all(d.platform == "gpu" for d in raw[2].devices()):
        raise RuntimeError("JAX real-data check requires an actual GPU")
    fs, ix, ds = (np.asarray(x) for x in raw)
    # Trace instrumentation must reproduce the production trajectory exactly.
    production = tuple(np.asarray(x) for x in run_columns_jax(f0, i0, forcing))
    checked = bool(np.array_equal(fs[:, -1], production[0]) and
                   np.array_equal(ix[:, -1], production[1]) and
                   np.array_equal(ds, production[2]))
    if not checked:
        raise RuntimeError("JAX traced trajectory differs from production")
    return (np.concatenate((f0[:, None], fs), axis=1),
            np.concatenate((i0[:, None], ix), axis=1), ds), str(raw[2].devices()), checked


def cuda_trace(f0, i0, forcing):
    import cupy as cp
    from noahmp_clm.gpu.cuda.loader import (_check_options, _launch_unchecked,
                                          run_columns_cuda)
    state = cp.asarray(f0.T).copy(order="C")
    integer = cp.asarray(i0.T).copy(order="C")
    _check_options(integer)
    state_out, integer_out = cp.empty_like(state), cp.empty_like(integer)
    u = cp.ascontiguousarray(cp.asarray(forcing).transpose(1, 2, 0))
    ds = cp.empty((forcing.shape[1], 12, len(f0)), dtype=cp.float32)
    fs = cp.empty((forcing.shape[1] + 1, 70, len(f0)), dtype=cp.float32)
    ix = cp.empty((forcing.shape[1] + 1, 33, len(f0)), dtype=cp.int32)
    fs[0], ix[0] = state, integer
    for step in range(forcing.shape[1]):
        _launch_unchecked(state, integer, u[step], state_out, integer_out, ds[step])
        state, state_out = state_out, state
        integer, integer_out = integer_out, integer
        fs[step + 1], ix[step + 1] = state, integer
    host = tuple(cp.asnumpy(x.transpose(2, 0, 1)) for x in (fs, ix, ds))
    production = tuple(cp.asnumpy(x) for x in run_columns_cuda(f0, i0, forcing))
    checked = bool(np.array_equal(host[0][:, -1], production[0]) and
                   np.array_equal(host[1][:, -1], production[1]) and
                   np.array_equal(host[2], production[2]))
    if not checked:
        raise RuntimeError("CUDA traced trajectory differs from production")
    return host, str(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]), checked


def provenance():
    paths = sorted((ROOT / "src/noahmp_clm").rglob("*.py"))
    paths += sorted((ROOT / "src/noahmp_clm/gpu/cuda").glob("*.cu"))
    paths += [Path(__file__), ROOT / "examples/fluxnet_smoke.py",
              ROOT / "benchmarks/closeout_criteria.json"]
    check = ROOT / "benchmarks/check_closeout.py"
    if check.exists():
        paths.append(check)
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in paths}


def automatic_checks_pass(report):
    """A completed calculation must not turn a failed M2 gate into exit success."""
    return bool(report["source_unchanged"] and report["run_completed"]
                and all(record["finite"] and record["production_trace_exact"]
                        and record["conservation"]["surface_accounting_passed"]
                        and record["conservation"]["water_step_passed"]
                        for record in report["backends"].values())
                and report.get("jax64_parity", {"passed": True})["passed"]
                and report.get("same_precision", {"passed": True})["passed"]
                and all(record["numeric_thresholds_passed"] and record["integers_equal"]
                        and all(site["numeric_thresholds_passed"] for site in record["per_site"].values())
                        for record in report["m2"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="*", choices=("jax64", "jax32", "cuda"),
                        default=["jax64", "jax32", "cuda"])
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    report = {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "numpy": np.__version__, "physical_validation": False, "calibrated": False,
              "parameters": "unchanged ColumnState defaults; no spinup; no tower-height correction",
              "source_sha256": provenance(), "inputs": [], "backends": {},
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "run_completed": False,
              "criteria": json.loads((ROOT / "benchmarks/closeout_criteria.json").read_text())}
    packed, observations = [], []
    for entry in entries:
        path = args.manifest.parent / entry["csv"]
        forcing, obs, stamps, metadata = load_fluxnet_csv(
            path, entry["start"], entry.get("days", 30), entry.get("humidity", "vpd"))
        packed.append([forcing_to_vec(row) for row in forcing])
        observations.append(obs)
        report["inputs"].append({"site": entry["site"], **metadata,
                                 "source": entry.get("source"),
                                 "source_sha256": entry.get("source_sha256")})
    u = np.asarray(packed)
    if u.ndim != 3:
        raise ValueError("batch requires equal-length windows; run differing lengths separately")
    f, i = column_state_to_vec(ColumnState())
    f0, i0 = np.repeat(f[None], len(entries), axis=0), np.repeat(i[None], len(entries), axis=0)
    arrays = {"forcing": u, "initial_states": f0, "initial_integers": i0}
    results = {}
    backend_order = ["numpy64"]
    if any(backend in args.backends for backend in ("jax32", "cuda")):
        backend_order.append("numpy64_matched32")
    for backend in [*backend_order, *args.backends]:
        print(json.dumps({"started": backend, "sites": len(entries), "steps": u.shape[1]}), flush=True)
        started = time.perf_counter()
        if backend.startswith("numpy64"):
            dtype = np.float32 if backend == "numpy64_matched32" else np.float64
            result, device, traced = numpy_trace(f0.astype(dtype), i0, u.astype(dtype)), "CPU", True
        else:
            dtype = np.float64 if backend == "jax64" else np.float32
            fi, ui = f0.astype(dtype), u.astype(dtype)
            result, device, traced = (jax_trace(fi, i0, ui) if backend.startswith("jax")
                                      else cuda_trace(fi, i0, ui))
        results[backend] = result
        record = {"elapsed_s": time.perf_counter() - started, "device": device,
                  "production_trace_exact": traced,
                  "finite": bool(all(np.isfinite(x).all() for x in result)),
                  "observations": {},
                  "checksum_sha256": hashlib.sha256(b"".join(np.ascontiguousarray(x).tobytes()
                                                               for x in result)).hexdigest()}
        for site_index, entry in enumerate(entries):
            record["observations"][entry["site"]] = {
                key: observation_metrics(result[2][site_index, :, DIAGNOSTICS.index(key)],
                                         observations[site_index][key]) for key in ("le", "h")}
        for name, value in zip(("states", "integers", "diagnostics"), result):
            arrays[backend + "_" + name] = value
        report["backends"][backend] = record
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
        write_report(args.output, report)
        print(json.dumps({"completed": backend, "elapsed_s": record["elapsed_s"],
                          "finite": record["finite"]}), flush=True)
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from check_closeout import compare_cross, conservation, error_summary
    reference = results["numpy64"]
    labels = [e["site"] for e in entries]
    report["m2"] = {}
    report["end_to_end_cross_precision"] = {}
    for backend, (states, integers, diagnostics) in results.items():
        report["backends"][backend]["conservation"] = conservation(states, diagnostics)
        if backend in ("jax32", "cuda"):
            matched = results["numpy64_matched32"]
            summary, rows = compare_cross(states, diagnostics, matched[0], matched[2], u.astype(np.float32),
                                          labels=labels)
            summary["reference"] = "NumPy64 physics starting from identical float32-quantized state and forcing"
            report["end_to_end_cross_precision"][backend] = compare_cross(
                states, diagnostics, reference[0], reference[2], u, labels=labels)[0]
            summary["per_site"] = {}
            for col, site in enumerate(labels):
                summary["per_site"][site] = compare_cross(
                    states[col:col+1], diagnostics[col:col+1], matched[0][col:col+1],
                    matched[2][col:col+1], u[col:col+1].astype(np.float32), labels=[site])[0]
            summary["integers_equal"] = bool(np.array_equal(integers, matched[1]))
            outliers_path = args.output.with_name(args.output.stem + "-" + backend + "-outliers.jsonl.gz")
            with gzip.open(outliers_path, "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            summary["outliers_file"] = outliers_path.name
            report["m2"][backend] = summary
        elif backend == "jax64":
            mask = np.ones(reference[0].shape, dtype=bool)
            mask[:, :, 45:] = False
            for offset in (16, 19, 22):
                for layer in range(3):
                    mask[:, :, offset+layer] = layer < reference[1][:, :, 32]
            state_check, _ = error_summary(states[mask], reference[0][mask],
                                           1e-6+1e-6*abs(reference[0][mask]))
            diag_check, _ = error_summary(diagnostics, reference[2], 1e-6+1e-6*abs(reference[2]))
            same_int = bool(np.array_equal(integers, reference[1]))
            report["jax64_parity"] = {"active_states": state_check, "diagnostics": diag_check,
                                      "integers_equal": same_int,
                                      "passed": same_int and not state_check["failed_elements"]
                                      and not diag_check["failed_elements"]}
    if "jax32" in results and "cuda" in results:
        same = {}
        for index, key in ((0, "states"), (2, "diagnostics")):
            a, b = results["cuda"][index], results["jax32"][index]
            same[key], _ = error_summary(a, b, 1e-6+1e-6*abs(b.astype(float)))
        same["integers_equal"] = bool(np.array_equal(results["cuda"][1], results["jax32"][1]))
        same["passed"] = same["integers_equal"] and all(same[k]["failed_elements"] == 0
                                                        for k in ("states", "diagnostics"))
        report["same_precision"] = same
    report["source_unchanged"] = provenance() == report["source_sha256"]
    report["arrays"] = args.output.with_suffix(".npz").name
    report["arrays_sha256"] = sha256(args.output.with_suffix(".npz"))
    report["run_completed"] = True
    report["automated_checks_passed"] = automatic_checks_pass(report)
    report["acceptance_status"] = ("numeric_checks_passed_requires_outlier_review"
                                   if report["automated_checks_passed"] else "failed_numeric_gate")
    write_report(args.output, report)
    return 0 if report["automated_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

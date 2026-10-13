"""Wang M2 closeout audit. Original H4 tests and their thresholds are untouched.

Retains every step's state, all diagnostics, all failure coordinates, source and
input hashes. A completed report is not necessarily a passed acceptance gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CRITERIA_PATH = Path(__file__).with_name("closeout_criteria.json")
CRITERIA = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
NAMES = ("ts", "rnet", "h", "le", "g", "et_mm", "prcp_mm", "runoff_mm",
         "melt_mm", "swe", "canw", "residual")
STATE = {"tsoil": slice(0, 4), "smc": slice(4, 8), "soil_ice": slice(8, 12),
         "swe": slice(26, 27), "canw": slice(30, 31)}


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "src").rglob("*.cu"))
    paths += [Path(__file__), CRITERIA_PATH]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def error_summary(actual, reference, allowed):
    a, b = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    delta = np.abs(a - b)
    finite = np.isfinite(a) & np.isfinite(b)
    bad = ~finite | (delta > allowed)
    values = delta[np.isfinite(delta)]
    return {"elements": int(a.size), "failed_elements": int(bad.sum()),
            "fraction_passed": float(1 - bad.sum() / a.size),
            "finite": bool(finite.all()),
            "max_abs": float(values.max()) if values.size else None,
            "mean_signed_error": float(np.mean(a-b)) if finite.all() else None,
            "p99_abs_error": float(np.quantile(values, .99)) if values.size else None}, bad


def compare_cross(states, diagnostics, reference_states, reference_diagnostics, forcing,
                  *, labels=None):
    """N,T+1,70 states; N,T,12 diagnostics; N,T,8 forcing, all on host.

    Returns a summary and every failed coordinate/value for explicit attribution.
    A group label only locates failure; it does not establish its physical cause.
    """
    c = CRITERIA["cross_precision"]
    rows, fields = [], {}
    labels = labels if labels is not None else ["unspecified"] * len(states)
    pairs = [(name, states[:, 1:, sl], reference_states[:, 1:, sl], False)
             for name, sl in STATE.items()]
    for name, k in (("h", 2), ("le", 3), ("g", 4), ("et_mm", 5)):
        pairs.append((name, diagnostics[:, :, k], reference_diagnostics[:, :, k], True))
    for name, actual, expected, flux in pairs:
        factor = (CRITERIA["latent_heat_J_kg"] / forcing[:, :, 0]
                  if name == "et_mm" else 1.)
        a, b = np.asarray(actual, np.float64) * factor, np.asarray(expected, np.float64) * factor
        allowed = c["rtol"] * abs(b)
        if flux:
            allowed = np.maximum(allowed, c["flux_atol_W_m2"])
        result, bad = error_summary(a, b, allowed)
        result["comparison_unit"] = "W/m2" if flux else ("K" if name == "tsoil" else "m" if name in ("swe", "canw") else "m3/m3")
        result["passed"] = result["finite"] and result["fraction_passed"] >= c["minimum_fraction_per_field"]
        result["failure_groups"] = dict(Counter(labels[int(v[0])] for v in np.argwhere(bad)))
        if name == "et_mm":
            result["max_abs_mm_per_step"] = float(np.max(np.abs(actual-expected)))
        fields[name] = result
        for idx in np.argwhere(bad):
            key = tuple(idx)
            rows.append({"field": name, "column": int(idx[0]), "step_1based": int(idx[1])+1,
                         "layer": int(idx[2]) if len(idx) == 3 else None,
                         "group": labels[int(idx[0])], "actual": float(a[key]),
                         "reference": float(b[key]), "allowed": float(allowed[key])})
    residual = np.abs(np.asarray(diagnostics[:, :, 11], np.float64))
    rd = np.abs(diagnostics[:, :, 11].astype(float)-reference_diagnostics[:, :, 11])
    residual_metrics = {"median_abs_W_m2": float(np.median(residual)),
                        "p99_abs_W_m2": float(np.quantile(residual, .99)),
                        "max_abs_W_m2": float(residual.max()),
                        "median_abs_difference_W_m2": float(np.median(rd)),
                        "p99_abs_difference_W_m2": float(np.quantile(rd, .99)),
                        "max_abs_difference_W_m2": float(rd.max())}
    residual_metrics["passed"] = bool(np.all(np.isfinite(residual)) and
        residual_metrics["median_abs_W_m2"] < CRITERIA["residual"]["absolute_median_lt_W_m2"] and
        residual_metrics["p99_abs_W_m2"] < CRITERIA["residual"]["absolute_p99_lt_W_m2"])
    return {"fields": fields, "residual": residual_metrics,
            "numeric_thresholds_passed": all(v["passed"] for v in fields.values()) and residual_metrics["passed"],
            "outlier_attribution": "not_needed" if not rows else "requires_evidence_review",
            "outlier_count": len(rows)}, rows


def conservation(states, diagnostics):
    d = np.asarray(diagnostics, np.float64)
    s = np.asarray(states, np.float64)
    eps = np.finfo(np.float32).eps
    mult = CRITERIA["roundoff_multiplier"]
    surface = d[:, :, 1]-d[:, :, 2]-d[:, :, 3]-d[:, :, 4]-d[:, :, 11]
    surface_allowed = mult*eps*(1+np.sum(abs(d[:, :, 1:5]), axis=2))
    storage = np.sum(s[:, :, 4:8]*np.array([.1, .3, .6, 1.]), axis=2)+s[:, :, 26]+s[:, :, 30]
    water = np.diff(storage, axis=1)-(d[:, :, 6]-d[:, :, 5]-d[:, :, 7])/1000.
    water_allowed = mult*eps*(abs(storage[:, :-1])+abs(storage[:, 1:])+
                            (abs(d[:, :, 6])+abs(d[:, :, 5])+abs(d[:, :, 7]))/1000.)
    return {"surface_accounting_max_abs_W_m2": float(np.max(abs(surface))),
            "surface_accounting_max_bound_ratio": float(np.max(abs(surface)/surface_allowed)),
            "surface_accounting_passed": bool(np.all(np.isfinite(surface)) and np.all(abs(surface) <= surface_allowed)),
            "water_step_max_abs_m": float(np.max(abs(water))),
            "water_step_max_bound_ratio": float(np.max(abs(water)/np.maximum(water_allowed, np.finfo(float).tiny))),
            "water_step_passed": bool(np.all(np.isfinite(water)) and np.all(abs(water) <= water_allowed)),
            "water_cumulative_per_column_m": np.sum(water, axis=1).tolist(),
            "water_cumulative_max_abs_m": float(np.max(abs(np.sum(water, axis=1)))),
            "whole_column_energy_passed": False,
            "whole_column_energy_status": "known melt-out routing gap; explicitly accepted limitation"}


def numpy_trajectory(f, i, u):
    from noahmp_clm.column import ColumnState
    from noahmp_clm.driver import step_column
    from noahmp_clm.gpu.pack import column_state_to_vec, vec_to_column_state, forcing_vec_to_forcing
    cache, states, integers, diagnostics = {}, [], [], []
    for col in range(len(f)):
        key = digest(f[col], i[col], u[col])
        if key not in cache:
            state = vec_to_column_state(f[col], i[col], ColumnState().dz)
            ss, ii, dd = [f[col].astype(float)], [i[col]], []
            for forcing in u[col]:
                state, d = step_column(state, forcing_vec_to_forcing(forcing))
                a, b = column_state_to_vec(state)
                ss.append(a); ii.append(b); dd.append([d[k] for k in NAMES])
            cache[key] = np.asarray(ss), np.asarray(ii), np.asarray(dd)
        a, b, d = cache[key]
        states.append(a); integers.append(b); diagnostics.append(d)
    return np.asarray(states), np.asarray(integers), np.asarray(diagnostics)


def gpu_trajectory(f, i, u, backend):
    import jax
    import cupy as cp
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("Actual CUDA device required")
    if backend == "cuda":
        from noahmp_clm.gpu.cuda.loader import run_columns_cuda as run
        host = cp.asnumpy
        sf, si = cp.asarray(f), cp.asarray(i)
    else:
        from noahmp_clm.gpu.jax_core import run_columns_jax as run
        host = np.asarray
        sf, si = jax.device_put((f, i))
    final = tuple(host(x) for x in run(sf, si, u))
    ss, ii, dd = [f.copy()], [i.copy()], []
    for t in range(u.shape[1]):
        sf, si, d = run(sf, si, u[:, t:t+1])
        ss.append(host(sf)); ii.append(host(si)); dd.append(host(d[:, 0]))
    states, integers, diagnostics = np.stack(ss, 1), np.stack(ii, 1), np.stack(dd, 1)
    replay = []
    for a, b in ((states[:, -1], final[0]), (diagnostics, final[2])):
        item, _ = error_summary(a, b, 1e-6+1e-6*abs(b.astype(float)))
        replay.append(item)
    exact_integer = bool(np.array_equal(integers[:, -1], final[1]))
    return states, integers, diagnostics, {"groups": replay, "integers_equal": exact_integer,
         "passed": exact_integer and all(x["failed_elements"] == 0 for x in replay)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    from probe_gpu_numerics import inputs
    from noahmp_clm.column import ColumnState
    from noahmp_clm.gpu.pack import column_state_to_vec
    f, one, _ = inputs(1024)
    i = np.broadcast_to(column_state_to_vec(ColumnState())[1], (1024, 33)).copy()
    u = np.broadcast_to(one[:, None], (1024, 48, 8)).copy()
    u[:, 12:, 7] = 0
    labels = [("cold_snow", "melting_initial", "rain", "dry")[c % 4] for c in range(1024)]
    initial_hashes = source_hashes()
    report = {"utc": datetime.now(timezone.utc).isoformat(), "N": 1024, "steps": 48,
              "criteria": CRITERIA, "source_sha256": initial_hashes, "input_sha256": digest(f, i, u),
              "results": {}}
    arrays = {"initial": f, "options": i, "forcing": u}
    start = time.perf_counter()
    ns, ni, nd = numpy_trajectory(f, i, u)
    print(json.dumps({"numpy_seconds": time.perf_counter()-start}), flush=True)
    arrays.update(numpy_states=ns, numpy_options=ni, numpy_diagnostics=nd)
    for backend in ("cuda", "jax32"):
        start = time.perf_counter()
        s, ints, d, replay = gpu_trajectory(f, i, u, backend)
        summary, rows = compare_cross(s, d, ns, nd, u, labels=labels)
        summary.update(conservation=conservation(s, d), replay=replay,
                       seconds_including_compile=time.perf_counter()-start,
                       checksum_sha256=digest(s, ints, d), integers_equal_numpy=bool(np.array_equal(ints, ni)))
        with gzip.open(args.directory / (backend+"-outliers.jsonl.gz"), "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        report["results"][backend] = summary
        arrays.update({backend+"_states": s, backend+"_options": ints, backend+"_diagnostics": d})
        print(json.dumps({"backend": backend, "numeric_thresholds_passed": summary["numeric_thresholds_passed"],
                          "outlier_count": summary["outlier_count"], "seconds": summary["seconds_including_compile"]}), flush=True)
    same = {}
    for name in ("states", "diagnostics"):
        a, b = arrays["cuda_"+name], arrays["jax32_"+name]
        same[name], _ = error_summary(a, b, 1e-6+1e-6*abs(b.astype(float)))
    same["integers_equal"] = bool(np.array_equal(arrays["cuda_options"], arrays["jax32_options"]))
    same["passed"] = same["integers_equal"] and all(same[k]["failed_elements"] == 0 for k in ("states", "diagnostics"))
    report["same_precision"] = same
    report["source_unchanged"] = initial_hashes == source_hashes()
    report["automated_checks_passed"] = (report["source_unchanged"] and same["passed"] and all(
        r["numeric_thresholds_passed"] and r["integers_equal_numpy"] and r["replay"]["passed"] and
        r["conservation"]["surface_accounting_passed"] and r["conservation"]["water_step_passed"]
        for r in report["results"].values()))
    report["acceptance_status"] = "requires annual/domain/environment/real-data and outlier review"
    np.savez_compressed(args.directory / "trajectories.npz", **arrays)
    (args.directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"same_precision_passed": same["passed"], "automated_checks_passed": report["automated_checks_passed"]}), flush=True)
    return 0 if report["automated_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

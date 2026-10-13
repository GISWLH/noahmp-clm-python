"""K3 full-domain NumPy comparison, with no geographic claim or new tolerance.

Run gpu, cpu, then summarize with the same fresh --directory. Intermediate NPY
files preserve every output, and CPU workers write disjoint column slices.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from check_gpu_extended import (Backend, DIAGNOSTICS, DZ, SCENARIOS,
                                daily_forcing, digest_arrays, initial_columns,
                                source_hashes)
from noahmp_clm.column import Forcing
from noahmp_clm.driver import run_column
from noahmp_clm.gpu.pack import column_state_to_vec, vec_to_column_state


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu(args):
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=False)
    f0, i0 = initial_columns(args.columns, np.float32)
    forcing = daily_forcing(np.arange(args.columns), 0, np.float32, 20261013, 48)
    begin = time.monotonic()
    backend = Backend(args.backend)
    result = backend.run(*(backend.xp.asarray(x) for x in (f0, i0, forcing)))
    backend.sync(result)
    backend.assert_device(result)
    metrics, _ = backend.metrics(*result, backend.storage(backend.xp.asarray(f0)))
    arrays = dict(f0=f0, i0=i0, forcing=forcing)
    arrays.update(zip(("actual_f", "actual_i", "actual_d"), map(backend.host, result)))
    for name, value in arrays.items():
        np.save(directory / (name + ".npy"), value)
    save_json(directory / "gpu.json", {
        "backend": args.backend, "columns": args.columns, "steps": 48,
        "seed": 20261013, "source_hashes": source_hashes(),
        "forensic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": digest_arrays(f0, i0, forcing),
        "device": backend.device, "version": backend.version,
        "elapsed_seconds": time.monotonic() - begin, "physical_metrics": metrics,
        "coordinates": "unavailable: synthetic input contains no latitude or longitude",
        "rtol": 1e-4, "atol": 1e-6,
    })
    print(json.dumps({"gpu_saved": str(directory), "physical_gate": metrics["passed"]}), flush=True)
    return 0 if metrics["passed"] else 1


def worker_init(directory, cpus):
    global ARRAYS
    identity = multiprocessing.current_process()._identity[-1] - 1
    os.sched_setaffinity(0, {cpus[identity % len(cpus)]})
    directory = Path(directory)
    ARRAYS = {name: np.load(directory / (name + ".npy"), mmap_mode="r+" if name.startswith("expected") else "r")
              for name in ("f0", "i0", "forcing", "expected_f", "expected_i", "expected_d")}


def reference_chunk(bounds):
    start, end = bounds
    for index in range(start, end):
        state = vec_to_column_state(ARRAYS["f0"][index], ARRAYS["i0"][index], DZ)
        forcing = [Forcing(*map(float, row)) for row in ARRAYS["forcing"][index]]
        rows, final = run_column(state, forcing)
        ARRAYS["expected_f"][index], ARRAYS["expected_i"][index] = column_state_to_vec(final)
        ARRAYS["expected_d"][index] = [[row[name] for name in DIAGNOSTICS] for row in rows]
    return end - start


def cpu(args):
    directory = args.directory
    n = np.load(directory / "f0.npy", mmap_mode="r").shape[0]
    for name, shape, dtype in (("expected_f", (n, 70), np.float64),
                               ("expected_i", (n, 33), np.int32),
                               ("expected_d", (n, 48, 12), np.float64)):
        if (directory / (name + ".npy")).exists():
            raise ValueError("Reference already exists; use a fresh directory")
        output = np.lib.format.open_memmap(directory / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
        output[:] = np.nan if dtype == np.float64 else -999
        del output
    completed = 0
    begin = time.monotonic()
    with ProcessPoolExecutor(max_workers=len(args.cpus), initializer=worker_init,
                             initargs=(str(directory), args.cpus)) as pool:
        futures = [pool.submit(reference_chunk, (start, min(start + 128, n))) for start in range(0, n, 128)]
        for future in as_completed(futures):
            completed += future.result()
            print(json.dumps({"completed": completed, "columns": n, "seconds": time.monotonic() - begin}), flush=True)
    save_json(directory / "cpu.json", {"columns_completed": completed, "elapsed_seconds": time.monotonic() - begin,
                                       "cpus": args.cpus, "source_hashes": source_hashes()})


def summarize(args):
    directory = args.directory
    arrays = {name: np.load(directory / (name + ".npy"), mmap_mode="r") for name in
              ("f0", "actual_f", "actual_i", "actual_d", "expected_f", "expected_i", "expected_d")}
    if not (directory / "cpu.json").exists():
        raise ValueError("Full NumPy reference has not completed")
    gpu_metadata = json.loads((directory / "gpu.json").read_text())
    cpu_metadata = json.loads((directory / "cpu.json").read_text())
    if gpu_metadata["source_hashes"] != cpu_metadata["source_hashes"]:
        raise ValueError("GPU and CPU source snapshots differ; do not combine evidence")
    n = len(arrays["f0"])
    if cpu_metadata["columns_completed"] != n:
        raise ValueError("NumPy reference is incomplete")
    counts = np.zeros((n, 12), np.int32)
    state_bad = np.zeros((n, 45), bool)
    max_abs = np.zeros((n, 12), np.float64)
    state_error = np.zeros((n, 45), np.float64)
    finite = np.ones(n, bool)
    for start in range(0, n, 1024):
        end = min(start + 1024, n)
        actual, expected = arrays["actual_d"][start:end], arrays["expected_d"][start:end]
        error = np.abs(actual.astype(np.float64) - expected)
        bad = error > 1e-6 + 1e-4 * np.abs(expected)
        counts[start:end] = bad.sum(axis=1)
        max_abs[start:end] = error.max(axis=1)
        af, ef = arrays["actual_f"][start:end, :45], arrays["expected_f"][start:end, :45]
        active = np.ones(af.shape, bool)
        snow_count = arrays["expected_i"][start:end, 32]
        for offset in (16, 19, 22):
            active[:, offset:offset+3] = np.arange(3)[None] < snow_count[:, None]
        state_error[start:end] = np.abs(af.astype(np.float64) - ef)
        state_bad[start:end] = active & (state_error[start:end] > 1e-6 + 1e-4 * np.abs(ef))
        finite[start:end] = np.isfinite(actual).all((1, 2)) & np.isfinite(expected).all((1, 2)) & np.isfinite(af).all(1) & np.isfinite(ef).all(1)
    integer_bad = np.any(arrays["actual_i"] != arrays["expected_i"], axis=1)
    diag_bad = np.any(counts > 0, axis=1)
    cols = np.arange(n)
    groups = {}
    def group_row(mask):
        return {"columns": int(mask.sum()), "diagnostic_failed_columns": int(diag_bad[mask].sum()),
                "state_failed_columns": int(np.any(state_bad[mask], axis=1).sum()),
                "integer_failed_columns": int(integer_bad[mask].sum()),
                "diagnostic_failed_elements": dict(zip(DIAGNOSTICS, counts[mask].sum(0).tolist())),
                "diagnostic_max_abs": dict(zip(DIAGNOSTICS, max_abs[mask].max(0).tolist())),
                "state_failed_elements_by_abi": {str(k): int(state_bad[mask, k].sum()) for k in range(45) if state_bad[mask, k].any()}}
    groups["scenario"] = {SCENARIOS[k]: group_row(cols % 8 == k) for k in range(8)}
    groups["soil"] = {name: group_row((cols % 8) % 3 == k) for k, name in enumerate(("sand", "loam", "clay"))}
    groups["initial_snow"] = {str(flag): group_row((arrays["f0"][:, 26] > 0) == flag) for flag in (False, True)}
    groups["vegetation_lai_bin"] = {f"{lo:g}–{hi:g}": group_row((arrays["f0"][:, 31] >= lo) & (arrays["f0"][:, 31] < hi)) for lo, hi in ((0., 1.5), (1.5, 2.5), (2.5, 4.))}
    groups["column_index_octile"] = {str(k): group_row(cols // (n // 8) == k) for k in range(8)}
    summary = {"columns": n, "steps": 48, "all_finite": bool(finite.all()),
               "passed": bool(finite.all() and not diag_bad.any() and not state_bad.any() and not integer_bad.any()),
               "coordinate_scope": "synthetic, no latitude/longitude or vegetation class supplied",
               "all": group_row(np.ones(n, bool)), "groups": groups,
               "state_max_abs_by_abi": {str(k): float(state_error[:, k].max()) for k in range(45)},
               "rtol": 1e-4, "atol": 1e-6,
               "gpu": gpu_metadata, "cpu": cpu_metadata}
    save_json(directory / "summary.json", summary)
    np.savez_compressed(directory / "column_errors.npz", diagnostic_counts=counts, diagnostic_max_abs=max_abs,
                        state_bad=state_bad, state_error=state_error, integer_bad=integer_bad, finite=finite)
    with gzip.open(directory / "columns.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["column", "scenario", "soil", "lai", "initial_swe", "final_swe_numpy", "diag_failed_count", "state_failed_abi", "numpy_ice0", "gpu_ice0", "numpy_tsoil0", "gpu_tsoil0", "max_residual_difference"])
        for k in cols:
            writer.writerow([int(k), SCENARIOS[k % 8], ("sand", "loam", "clay")[(k % 8) % 3],
                             float(arrays["f0"][k, 31]), float(arrays["f0"][k, 26]), float(arrays["expected_f"][k, 26]),
                             int(counts[k].sum()), ";".join(map(str, np.flatnonzero(state_bad[k]))),
                             *[float(arrays[name][k, field]) for field in (8, 0) for name in ("expected_f", "actual_f")],
                             float(max_abs[k, 11])])
    print(json.dumps({"summary": str(directory / "summary.json"), "passed": summary["passed"], "all": summary["all"]}), flush=True)
    return 0 if summary["passed"] else 1


def decompose(args):
    """Exact telescoping ice-error decomposition evaluated with NumPy float64."""
    directory = args.directory
    af, ef = (np.load(directory / (name + ".npy"), mmap_mode="r") for name in ("actual_f", "expected_f"))
    i0 = np.load(directory / "i0.npy", mmap_mode="r")
    from noahmp_clm.gpu.pack import OPTION_FIELDS
    if np.any(i0[:, OPTION_FIELDS.index("OptSoilSupercoolWater")] != 1):
        raise ValueError("Analytic decomposition applies to Niu--Yang option 1 only")
    def ice(smc, temperature):
        smc, temperature = np.asarray(smc, dtype=np.float64), np.asarray(temperature, dtype=np.float64)
        undercooling = np.maximum(0., 273.15 - temperature)
        clausius = 3.34e5 / 461.5 * undercooling / 273.15**2
        potential = ef[:, 39, None] * (np.maximum(clausius, 1e-8) / 5e-5)**(-1. / ef[:, 43, None])
        liquid = np.where(temperature >= 273.15, np.maximum(0., smc), np.minimum(np.maximum(0., smc), potential))
        return np.maximum(0., smc - liquid)
    matched = ice(af[:, 4:8], af[:, :4])
    same_t = ice(af[:, 4:8], ef[:, :4])
    reference = ice(ef[:, 4:8], ef[:, :4])
    formula = af[:, 8:12].astype(np.float64) - matched
    temperature = matched - same_t
    water = same_t - reference
    total = af[:, 8:12].astype(np.float64) - ef[:, 8:12]
    remainder = total - formula - temperature - water
    bad = np.abs(total) > 1e-6 + 1e-4 * np.abs(ef[:, 8:12])
    rows = []
    for col, layer in np.argwhere(bad):
        rows.append({"column": int(col), "layer": int(layer), "scenario": SCENARIOS[col % 8],
                     "total": float(total[col, layer]), "formula": float(formula[col, layer]),
                     "temperature": float(temperature[col, layer]), "water": float(water[col, layer]),
                     "tsoil_numpy": float(ef[col, layer]), "tsoil_gpu": float(af[col, layer]),
                     "ice_numpy": float(ef[col, layer + 8]), "ice_gpu": float(af[col, layer + 8]),
                     "final_swe_numpy": float(ef[col, 26])})
    result = {"identity_max_abs_remainder": float(np.max(np.abs(remainder))),
              "reference_formula_max_abs_error": float(np.max(np.abs(reference - ef[:, 8:12]))),
              "ice_failures": len(rows),
              "failed_cell_term_max_abs": {name: float(np.max(np.abs(value[bad]), initial=0.)) for name, value in
                                            (("formula", formula), ("temperature", temperature), ("water", water))},
              "failed_cell_temperature_dominates_formula_and_water": int(np.sum(np.abs(temperature[bad]) > np.abs(formula[bad]) + np.abs(water[bad]))),
              "failed_temperature_minmax": [float(ef[:, :4][bad].min()), float(ef[:, :4][bad].max())] if bad.any() else None,
              "worst_20": sorted(rows, key=lambda row: abs(row["total"]), reverse=True)[:20]}
    save_json(directory / "ice-decomposition.json", result)
    with gzip.open(directory / "ice-decomposition-rows.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    np.savez_compressed(directory / "ice-decomposition.npz", total=total, formula=formula, temperature=temperature,
                        water=water, bad=bad, numpy_temperature=ef[:, :4], initial_lai=af[:, 31], final_swe=ef[:, 26])
    print(json.dumps(result), flush=True)


def trace(args):
    """Trace selected tail columns; check the trace against saved full-batch data."""
    directory = args.directory
    a, e = (np.load(directory / (name + ".npy"), mmap_mode="r") for name in ("actual_d", "expected_d"))
    max_ts_error = np.abs(a[:, :, 0].astype(np.float64) - e[:, :, 0]).max(axis=1)
    worst = np.argsort(max_ts_error)[-10:]
    ids = np.unique(np.r_[6, 15414, 55542, worst])
    inputs = [np.asarray(np.load(directory / (name + ".npy"), mmap_mode="r")[ids]) for name in ("f0", "i0", "forcing")]
    backend = Backend("jax32")
    import jax
    from noahmp_clm.gpu.jax_core import step_column_jax
    from noahmp_clm.driver import step_column
    def run(f, i, forcing):
        def body(carry, u):
            fn, inn, d = step_column_jax(*carry, u)
            return (fn, inn), (fn, inn, d)
        return jax.lax.scan(body, (f, i), forcing)[1]
    traced = jax.jit(jax.vmap(run))(*(backend.xp.asarray(value) for value in inputs))
    backend.sync(traced)
    af, ai, ad = map(backend.host, traced)
    ef, ei, ed = [], [], []
    for index in range(len(ids)):
        state = vec_to_column_state(inputs[0][index], inputs[1][index], DZ)
        states, integers, diagnostics = [], [], []
        for forcing in inputs[2][index]:
            state, diag = step_column(state, Forcing(*map(float, forcing)))
            f, i = column_state_to_vec(state)
            states.append(f); integers.append(i); diagnostics.append([diag[name] for name in DIAGNOSTICS])
        ef.append(states); ei.append(integers); ed.append(diagnostics)
    ef, ei, ed = map(np.asarray, (ef, ei, ed))
    np.savez_compressed(directory / "tail-trace.npz", columns=ids, initial=inputs[0], options=inputs[1],
                        forcing=inputs[2], gpu_state=af, gpu_int=ai, gpu_diag=ad,
                        numpy_state=ef, numpy_int=ei, numpy_diag=ed)
    saved = a[ids]
    ratio = np.abs(ad.astype(np.float64)-saved)/(1e-6+1e-6*np.abs(saved))
    report = {"columns": ids.tolist(), "trace_vs_full_batch_gate": bool(np.all(ratio<=1)),
              "trace_vs_full_batch_mismatches": int(np.sum(ratio>1)),
              "trace_vs_full_batch_max_abs": float(np.max(np.abs(ad-saved))),
              "tail_columns": [{"column": int(index), "max_ts_abs": float(max_ts_error[index]),
                                "max_ts_step": int(np.argmax(np.abs(a[index,:,0]-e[index,:,0])))} for index in ids],
              "source_hashes": source_hashes()}
    save_json(directory / "tail-trace.json", report)
    print(json.dumps(report), flush=True)


def plot(args):
    """Render only measured column-index/physical-regime clusters, not a map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = args.directory
    summary = json.loads((directory / "summary.json").read_text())
    data = np.load(directory / "column_errors.npz")
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw={"width_ratios": [1.2, 1.2, 1.5]})
    scenarios = list(summary["groups"]["scenario"])
    rows = [summary["groups"]["scenario"][name] for name in scenarios]
    failed = [row["state_failed_columns"] for row in rows]
    axes[0].barh(np.arange(8), failed, color=["#b43b3b" if value else "#c7cdd3" for value in failed])
    axes[0].set_yticks(np.arange(8), [name.replace("_", " ") for name in scenarios], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 8500)
    for k, value in enumerate(failed):
        axes[0].text(value + 100, k, f"{value:,}/8,100", va="center", fontsize=8)
    axes[0].set_title("Final physical-state failures")
    axes[0].set_xlabel("Columns (original strict tolerance)")
    ids = np.arange(summary["columns"])
    bad = data["state_bad"][:, 8]
    axes[1].scatter(ids[bad], data["state_error"][bad, 8], s=3, alpha=0.5, color="#b43b3b", rasterized=True)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Synthetic column index; no latitude supplied")
    axes[1].set_ylabel("Top-layer ice absolute difference (m3/m3)")
    axes[1].set_title(f"{int(bad.sum()):,} top-layer ice failures")
    rates = np.array([[row["diagnostic_failed_elements"][name] / (row["columns"] * 48) for name in DIAGNOSTICS] for row in rows])
    im = axes[2].imshow(rates, aspect="auto", vmin=0, vmax=1, cmap="magma")
    axes[2].set_xticks(np.arange(12), DIAGNOSTICS, rotation=60, ha="right", fontsize=8)
    axes[2].set_yticks(np.arange(8), [str(k) for k in range(8)])
    axes[2].set_ylabel("Scenario index (same order as left)")
    axes[2].set_title("Fraction of column-steps failing per diagnostic")
    fig.colorbar(im, ax=axes[2], shrink=0.7, label="Failure fraction")
    fig.suptitle(f"K3: complete {summary['columns']:,}-column x 48-step NumPy64 comparison | frozen source", fontsize=13)
    fig.tight_layout()
    fig.savefig(directory / "clusters.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, group, title in zip(axes, ("soil", "vegetation_lai_bin", "initial_snow"),
                                 ("Soil", "LAI (not vegetation type)", "Initial snow")):
        values = summary["groups"][group]
        labels = list(values)
        fail = np.array([value["state_failed_columns"] for value in values.values()])
        totals = np.array([value["columns"] for value in values.values()])
        axis.bar(np.arange(len(labels)), 100 * fail / totals, color="#b43b3b")
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_ylim(0, max(26., float((100*fail/totals).max())*1.15))
        for k in range(len(labels)):
            axis.text(k, 100*fail[k]/totals[k]+0.7, f"{fail[k]:,}/{totals[k]:,}", ha="center", fontsize=8)
        axis.set_ylabel("Columns failing final-state gate (%)")
        axis.set_title(title)
    fig.suptitle("Complete synthetic domain; soil/LAI/scenario are correlated by input design")
    fig.tight_layout()
    fig.savefig(directory / "physical-groups.png", dpi=160)
    plt.close(fig)


def causal(args):
    """Frozen-state CPU probes of heat-capacity amplification and snow roots."""
    from noahmp_clm.driver import step_column
    from noahmp_clm.physics import energy, soil
    from noahmp_clm.physics.frozen_soil import apparent_heat_capacity
    data = np.load(args.directory / "tail-trace.npz")
    results = {"heat": [], "snow_root": []}
    for column, step in ((49974, 7), (49974, 20), (15414, 21), (15414, 42)):
        position = data["columns"].tolist().index(column)
        for backend in ("numpy", "gpu"):
            state = vec_to_column_state(data[backend + "_state"][position, step-1], data[backend + "_int"][position, step-1], DZ)
            forcing = Forcing(*map(float, data["forcing"][position, step]))
            original = soil.step_soil_heat
            def capture(s, dt, g):
                row = {"column": column, "step": step, "input_backend": backend,
                       "old_temperature": s.tsoil.tolist(), "smc": s.smc.tolist(), "g": g,
                       "capacity": apparent_heat_capacity(s.smc, s.tsoil, s.soil, s.options.OptSoilSupercoolWater).tolist()}
                original(s, dt, g)
                row["new_temperature_float64"] = s.tsoil.tolist()
                results["heat"].append(row)
            soil.step_soil_heat = capture
            try:
                step_column(state, forcing)
            finally:
                soil.step_soil_heat = original
    column, step = 5063, 44
    position = data["columns"].tolist().index(column)
    for backend in ("numpy", "gpu"):
        state = vec_to_column_state(data[backend + "_state"][position, step-1], data[backend + "_int"][position, step-1], DZ)
        forcing = Forcing(*map(float, data["forcing"][position, step]))
        original = energy.secant_solve
        def capture(fun, *positional, **kwargs):
            chosen = original(fun, *positional, **kwargs)
            temperatures = np.linspace(260., 273., 2601)
            values = np.array([fun(value) for value in temperatures])
            candidates = []
            for k in np.flatnonzero(values[:-1] * values[1:] < 0):
                lo, hi = temperatures[k:k+2]
                for _ in range(45):
                    mid = (lo+hi)/2
                    if fun(lo)*fun(mid)<=0:
                        hi=mid
                    else:
                        lo=mid
                root=(lo+hi)/2
                candidates.append({"temperature": root, "residual": float(fun(root))})
            results["snow_root"].append({"column": column, "step": step, "input_backend": backend,
                                         "ordinary_secant_root": chosen, "ordinary_residual": fun(chosen),
                                         "sign_change_candidates": candidates,
                                         "at_gpu_root": float(fun(float(data["gpu_diag"][position,step,0]))),
                                         "at_numpy_root": float(fun(float(data["numpy_diag"][position,step,0])))})
            np.savez_compressed(args.directory / ("snow-root-"+backend+"-input.npz"), temperature=temperatures, residual=values)
            return chosen
        energy.secant_solve = capture
        try:
            step_column(state, forcing)
        finally:
            energy.secant_solve = original
    save_json(args.directory / "causal.json", results)
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gpu", "cpu", "summarize", "decompose", "trace", "plot", "causal"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=64800)
    parser.add_argument("--backend", choices=("jax32", "cuda"), default="jax32")
    parser.add_argument("--cpus", type=int, nargs="+", default=list(range(24, 40)))
    args = parser.parse_args()
    return globals()[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())

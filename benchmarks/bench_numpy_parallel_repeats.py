"""Time full NumPy single-core runs using three concurrent independent repeats.

Each sample runs every requested column and time step via bench_gpu._numpy_runner.
Only the independent repeats overlap: columns are never partitioned among cores.
For each N, all three full warmups finish before three full measured runs begin.
Choose three different physical CPU cores, not sibling SMT threads, for example::

    PYTHONPATH=src python benchmarks/bench_numpy_parallel_repeats.py \
        --sizes 100 --cores 0 1 2 --steps 48 --output outputs/numpy-check.jsonl

Linux CPU affinity and sysfs topology are required. Existing output files are
preserved: choose a fresh output path for a new run. No extrapolation is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import platform
import statistics
import time
import traceback


REPEATS = 3
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _physical_core(cpu: int) -> tuple[int, int]:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return (
        int((topology / "physical_package_id").read_text().strip()),
        int((topology / "core_id").read_text().strip()),
    )


def _validate_cores(cores: list[int]) -> list[dict]:
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("This benchmark requires Linux CPU-affinity support")
    if len(cores) != REPEATS or len(set(cores)) != REPEATS:
        raise ValueError("Exactly three distinct CPU IDs are required")
    available = os.sched_getaffinity(0)
    if any(cpu not in available for cpu in cores):
        raise ValueError(f"Requested cores {cores} are not all in allowed affinity {sorted(available)}")
    identities = [_physical_core(cpu) for cpu in cores]
    if len(set(identities)) != REPEATS:
        raise ValueError(f"CPU IDs must name distinct physical cores, not SMT siblings: {identities}")
    return [
        {"cpu_id": cpu, "physical_package_id": identity[0], "physical_core_id": identity[1]}
        for cpu, identity in zip(cores, identities)
    ]


def _run_sample(n: int, steps: int, cpu: int, phase: str, index: int) -> dict:
    # Pin before importing NumPy: any library worker threads inherit this mask.
    os.sched_setaffinity(0, {cpu})
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != [cpu]:
        raise AssertionError(f"Could not bind sample to CPU {cpu}: affinity={affinity}")

    import numpy as np
    from noahmp_clm.io import synthetic_forcing
    if __package__:
        from .bench_gpu import _numpy_runner
    else:
        from bench_gpu import _numpy_runner

    forcings = synthetic_forcing(n_days=steps / 48.0, dt=1800.0)
    if len(forcings) != steps:
        raise ValueError(f"Requested {steps} forcing steps, generated {len(forcings)}")
    runner = _numpy_runner(n, forcings)
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    checksum = runner()
    process_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    ended = datetime.now(timezone.utc).isoformat()
    final_affinity = sorted(os.sched_getaffinity(0))
    if final_affinity != [cpu]:
        raise AssertionError(f"Sample CPU affinity changed during execution: {final_affinity}")
    if not np.isfinite(checksum):
        raise AssertionError(f"Nonfinite checksum from full NumPy run: {checksum}")
    if wall_seconds <= 0.0 or process_seconds < 0.0:
        raise AssertionError("Invalid sample timing")
    package_id, physical_id = _physical_core(cpu)
    return {
        "event": phase, "status": "measured", "backend": "numpy", "n": n,
        "steps": steps, "index": index, "worker_pid": os.getpid(),
        "started_utc": started, "ended_utc": ended,
        "wall_seconds": wall_seconds, "process_time": process_seconds,
        "affinity": final_affinity, "physical_package_id": package_id,
        "physical_core_id": physical_id, "checksum": float(checksum),
        "numpy": np.__version__, "full_column_steps": n * steps,
    }


def _run_phase(pool, n, steps, cores, phase, emit):
    futures = {}
    for index, cpu in enumerate(cores):
        emit({"event": "sample_submitted", "backend": "numpy", "n": n,
              "steps": steps, "phase": phase, "index": index, "cpu_id": cpu})
        future = pool.submit(_run_sample, n, steps, cpu, phase, index)
        futures[future] = (index, cpu)
    results = []
    failed = False
    # Every sample completion is flushed immediately, irrespective of submit order.
    for future in as_completed(futures):
        index, cpu = futures[future]
        try:
            result = future.result()
            emit(result)
            results.append(result)
        except Exception as exc:
            failed = True
            emit({"event": "sample_error", "status": "failed", "backend": "numpy",
                  "n": n, "steps": steps, "phase": phase, "index": index,
                  "cpu_id": cpu, "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()})
    if failed or len(results) != REPEATS:
        raise RuntimeError(f"{phase}: not all three full independent samples completed successfully")
    return sorted(results, key=lambda row: row["index"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--cores", nargs=REPEATS, type=int, required=True,
                        metavar=("CPU0", "CPU1", "CPU2"))
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or any(n <= 0 for n in args.sizes):
        parser.error("sizes and steps must be positive")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("Each size must be listed only once")
    try:
        topology = _validate_cores(args.cores)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error(f"Refusing to overwrite an existing benchmark: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"

    def emit(row):
        line = json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **row}, allow_nan=False)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    emit({
        "event": "environment", "backend": "numpy", "python": platform.python_version(),
        "platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
        "sizes": args.sizes, "steps": args.steps, "warmups": REPEATS, "repeats": REPEATS,
        "workers": REPEATS, "cpu_topology": topology,
        "execution": "three concurrent independent repeats, each pinned to one distinct physical core",
        "columns_partitioned": False, "full_size_per_sample": True,
        "warmups_completed_before_measurements": True,
        "timing_excludes_imports_forcing_and_runner_setup": True,
        "thread_environment": {variable: os.environ[variable] for variable in THREAD_VARIABLES},
        "median_basis": "three full measured wall_seconds samples; no extrapolation",
        "process_time_unit": "seconds",
    })
    completed = []
    try:
        # Spawn avoids inherited numerical-library thread pools and worker state.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=REPEATS, mp_context=context) as pool:
            for n in args.sizes:
                emit({"event": "start", "backend": "numpy", "n": n, "steps": args.steps})
                warmups = _run_phase(pool, n, args.steps, args.cores, "warmup", emit)
                samples = _run_phase(pool, n, args.steps, args.cores, "measured_run", emit)
                values = [row["wall_seconds"] for row in samples]
                median = statistics.median(values)
                # Independent repeats must calculate the identical deterministic run.
                if len({row["checksum"] for row in warmups + samples}) != 1:
                    raise AssertionError(f"Independent NumPy samples disagree for N={n}")
                emit({
                    "event": "result", "status": "measured", "backend": "numpy",
                    "n": n, "steps": args.steps, "seconds": median,
                    "median_wall_seconds": median, "samples_seconds": values,
                    "samples_process_time": [row["process_time"] for row in samples],
                    "median_process_time": statistics.median(row["process_time"] for row in samples),
                    "us_per_column_step": median / (n * args.steps) * 1e6,
                    "column_steps_per_second": n * args.steps / median,
                    "sample_affinities": [row["affinity"] for row in samples],
                    "parallel_independent_repeats": True, "single_core_per_sample": True,
                    "warmups": REPEATS, "repeats": REPEATS, "checksum": samples[0]["checksum"],
                })
                completed.append(n)
    except Exception as exc:
        emit({"event": "summary", "status": "failed", "completed_sizes": completed,
              "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return 1
    emit({"event": "summary", "status": "completed", "completed_sizes": completed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

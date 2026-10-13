"""CPU column smoke with serial, process-pool and PBS/Open MPI launch modes.

MPI ranks own disjoint global column IDs. They exchange no model state; after
mpirun exits, a separate gather validates and assembles the shared-file outputs.
No mpi4py dependency is needed for this embarrassingly parallel workload.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "src"))
from check_gpu_extended import DIAGNOSTICS, DZ, daily_forcing, digest_arrays, initial_columns
from noahmp_clm.column import Forcing
from noahmp_clm.driver import run_column
from noahmp_clm.gpu.pack import column_state_to_vec, vec_to_column_state


def run_one(arg):
    column_id, f, i, forcing = arg
    state = vec_to_column_state(f, i, DZ)
    diag, state = run_column(state, [Forcing(*row) for row in forcing])
    ff, ii = column_state_to_vec(state)
    return column_id, ff, ii, np.array([[row[k] for k in DIAGNOSTICS] for row in diag])


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "src").rglob("*.cu"))
    paths += [Path(__file__).resolve(), ROOT / "benchmarks/check_gpu_extended.py"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def emit(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


def compare(actual, reference):
    result = {}
    for key in ("f", "i", "d"):
        a, b = actual[key], reference[key]
        close = a == b if key == "i" else np.isclose(a, b, rtol=1e-6, atol=1e-6)
        result[key] = {"passed": bool(np.all(close)), "elements": int(a.size),
                       "failed_elements": int(np.count_nonzero(~close)),
                       "max_abs_difference": float(np.max(np.abs(a.astype(float)-b.astype(float)))),
                       "bitwise_equal": bool(np.array_equal(a, b))}
    result["passed"] = all(v["passed"] for v in result.values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("serial", "pool", "mpi", "gather"), required=True)
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
              "columns": args.columns, "steps": args.steps, "dtype": "float64",
              "host": socket.gethostname(), "python": platform.python_version(),
              "numpy": np.__version__, "platform": platform.platform(),
              "pbs_jobid": os.environ.get("PBS_JOBID"), "source_hashes": source_hashes(),
              "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "gather":
        chunks = [dict(np.load(str(output)+f".rank{r}.npz")) for r in range(args.ranks)]
        ids = np.concatenate([c["ids"] for c in chunks])
        order = np.argsort(ids)
        if not np.array_equal(ids[order], np.arange(args.columns)):
            raise RuntimeError("MPI ranks did not cover each global column exactly once")
        arrays = {k: np.concatenate([c[k] for c in chunks])[order] for k in ("f", "i", "d")}
        rank_metadata = [json.loads(Path(str(output)+f".rank{r}.json").read_text()) for r in range(args.ranks)]
        result["rank_metadata"] = [{k: d[k] for k in ("host", "rank", "columns_local", "compute_seconds", "checksum_sha256")} for d in rank_metadata]
        result["distinct_hosts"] = sorted({d["host"] for d in rank_metadata})
        result["rank_source_hashes_equal"] = all(d["source_hashes"] == result["source_hashes"] for d in rank_metadata)
        if not result["rank_source_hashes_equal"] or len(result["distinct_hosts"]) != args.ranks:
            raise RuntimeError("MPI must execute on distinct hosts with identical source hashes")
        result["compute_seconds"] = max(d["compute_seconds"] for d in rank_metadata)
    else:
        f, i = initial_columns(args.columns, np.float64)
        forcing = daily_forcing(np.arange(args.columns), 0, np.float64, 20261013, args.steps)
        result["input_sha256"] = digest_arrays(f, i, forcing)
        ids = np.arange(args.columns)
        if args.mode == "mpi":
            rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
            if size != args.ranks:
                raise RuntimeError("MPI size differs from requested rank count")
            ids = ids[rank::size]
            output = Path(str(output)+f".rank{rank}")
            result.update(rank=rank, ranks=size, columns_local=len(ids))
        tasks = [(int(k), f[k], i[k], forcing[k]) for k in ids]
        tick = time.perf_counter()
        if args.mode == "pool":
            with ProcessPoolExecutor(args.workers) as pool:
                rows = list(pool.map(run_one, tasks))
            result["workers"] = args.workers
        else:
            rows = list(map(run_one, tasks))
        result["compute_seconds"] = time.perf_counter()-tick
        arrays = {k: np.array([row[j] for row in rows]) for j, k in enumerate(("ids", "f", "i", "d"))}
    result["finite"] = bool(np.isfinite(arrays["f"]).all() and np.isfinite(arrays["d"]).all())
    result["checksum_sha256"] = digest_arrays(arrays["f"], arrays["i"], arrays["d"])
    result["diagnostic_sum"] = arrays["d"].sum(axis=(0, 1)).tolist()
    result["state_sum"] = arrays["f"].sum(axis=0).tolist()
    np.savez_compressed(str(output)+".npz", **arrays)
    result["passed"] = result["finite"]
    if args.reference:
        with np.load(args.reference) as reference:
            result["parity"] = compare(arrays, reference)
        result["passed"] = result["passed"] and result["parity"]["passed"]
    result["wall_seconds_including_output_validation"] = time.perf_counter()-started
    emit(Path(str(output)+".json"), result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Strict single-GPU versus two-GPU scaling of the same heterogeneous columns.

Preparation, full-size JAX reference generation, .npy storage and per-device H2D
are outside timing. Each measured worker starts at a process barrier and consumes
every output on GPU, copies checksums to host and compares every element to its
slice of the full-size reference. Three complete warmups precede three samples.
CUDA always uses JAX32 reference; this timing gate cannot establish H4 parity.

    env -u LD_LIBRARY_PATH PYTHONPATH=src python benchmarks/bench_gpu_scaling.py --backend jax32 \
        --sizes 100000 64800 --devices 0 1 --steps 48 \
        --output outputs/scaling.jsonl --work-dir profile_output/scaling-reference

Launch only when both requested physical GPUs have been allocated to this job.
The script preserves intermediate .npy files and refuses existing output/work
paths. No GPU is initialized for --help or module import.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import statistics
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPEATS = 3
NAMES = ("float_state", "integer_state", "diagnostics")


def _imports():
    if __package__:
        from . import gpu_checksum as checksum
        from . import check_gpu_extended as workload
    else:
        import gpu_checksum as checksum
        import check_gpu_extended as workload
    return checksum, workload


def _source_hashes():
    paths = sorted((ROOT / "src/noahmp_clm").rglob("*.py"))
    paths += sorted((ROOT / "src/noahmp_clm/gpu/cuda").glob("*.cu"))
    paths += [Path(__file__).resolve(), ROOT / "benchmarks/gpu_checksum.py",
              ROOT / "benchmarks/check_gpu_extended.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _configure(device):
    # Executed before importing any CUDA library in each fresh spawned process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_PLATFORMS"] = "cuda"
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    import cupy as cp
    if cp.cuda.runtime.is_hip or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError(f"Physical GPU {device} must expose exactly one NVIDIA CUDA device")
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    info = {"physical_device": device, "logical_device": 0, "name": name,
            "pci_identity": [int(props[key]) for key in ("pciDomainID", "pciBusID", "pciDeviceID")],
            "total_memory_bytes": int(props["totalGlobalMem"]), "cupy": cp.__version__}
    return cp, info


def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu" or any(d.platform != "gpu" for d in jax.devices()):
        raise RuntimeError("JAX CUDA backend required; CPU fallback is forbidden")
    from noahmp_clm.gpu.jax_core import run_columns_jax
    return jax, run_columns_jax


def _once_worker(kind, config, result_queue):
    try:
        cp, device = _configure(config["device"])
        if kind == "probe":
            result_queue.put({"event": "device_probe", "status": "ready", "device": device})
            return
        checksum, _ = _imports()
        if _source_hashes() != config["sourcehash"]:
            raise AssertionError("Source changed before reference generation")
        jax, run = _jax()
        folder = Path(config["folder"])
        host = [np.load(folder / f"input_{key}.npy", mmap_mode="r") for key in ("f", "i", "u")]
        started = time.perf_counter()
        result = run(*jax.device_put(tuple(host)))
        jax.block_until_ready(result)
        digest, gate = checksum.consume_device_result(result, config["n"], config["steps"],
                                                     expected_dtype=config["dtype"])
        checksum.assert_gate(gate)
        for name, array in zip(NAMES, result):
            np.save(folder / f"reference_{name}.npy", np.asarray(jax.device_get(array)), allow_pickle=False)
        if _source_hashes() != config["sourcehash"]:
            raise AssertionError("Source changed during reference generation")
        result_queue.put({"event": "reference_ready", "status": "ready", "device": device,
                          "reference_backend": "jax64" if config["dtype"] == "float64" else "jax32",
                          "preparation_seconds": time.perf_counter() - started,
                          "digest": digest, "gate": gate, "n": config["n"], "steps": config["steps"]})
    except Exception as exc:
        result_queue.put({"event": "worker_error", "status": "failed", "device": config["device"],
                          "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def _once(context, kind, config):
    results = context.Queue()
    process = context.Process(target=_once_worker, args=(kind, config, results))
    process.start()
    try:
        record = results.get(timeout=config["timeout"])
        process.join(timeout=10)
        if process.is_alive() or process.exitcode != 0 or record["status"] != "ready":
            raise RuntimeError(f"{kind} did not finish successfully: {record}")
        return record
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        results.close()


def _timing_worker(config, commands, results, barrier):
    try:
        cp, device = _configure(config["device"])
        checksum, _ = _imports()
        if _source_hashes() != config["sourcehash"]:
            raise AssertionError("Source changed before worker preparation")
        start, stop = config["start"], config["stop"]
        count = stop - start
        folder = Path(config["folder"])
        preparation_start = time.perf_counter()
        host = [np.load(folder / f"input_{key}.npy", mmap_mode="r")[start:stop]
                for key in ("f", "i", "u")]
        reference = tuple(cp.asarray(np.load(folder / f"reference_{name}.npy", mmap_mode="r")[start:stop])
                          for name in NAMES)
        reference_digest, gate = checksum.consume_device_result(reference, count, config["steps"],
                                                               expected_dtype=config["dtype"])
        checksum.assert_gate(gate)
        if config["backend"] == "cuda":
            from noahmp_clm.gpu.cuda.loader import run_columns_cuda, KERNEL_SOURCE
            kernel_path = ROOT / "src/noahmp_clm/gpu/cuda/column_kernel.cu"
            if hashlib.sha256(KERNEL_SOURCE.encode()).hexdigest() != hashlib.sha256(kernel_path.read_text(encoding="utf-8").encode()).hexdigest():
                raise AssertionError("Loaded kernel source differs from recorded disk source")
            inputs = tuple(cp.asarray(array) for array in host)

            def run():
                value = run_columns_cuda(*inputs)
                cp.cuda.get_current_stream().synchronize()
                return value
        else:
            jax, run_columns_jax = _jax()
            inputs = jax.device_put(tuple(host))
            jax.block_until_ready(inputs)

            def run():
                value = run_columns_jax(*inputs)
                jax.block_until_ready(value)
                return value
        cp.cuda.runtime.deviceSynchronize()
        results.put({"event": "worker_ready", "status": "ready", "device": device,
                     "start": start, "stop": stop, "columns": count,
                     "preparation_and_h2d_seconds": time.perf_counter() - preparation_start})
        while True:
            command = commands.get()
            if command["phase"] == "stop":
                break
            cp.cuda.runtime.deviceSynchronize()
            event_start = cp.cuda.Event() if config["backend"] == "cuda" else None
            event_stop = cp.cuda.Event() if config["backend"] == "cuda" else None
            barrier.wait(timeout=config["timeout"])
            worker_started = time.perf_counter_ns()
            if event_start is not None:
                event_start.record()
            result = run()
            if event_stop is not None:
                event_stop.record()
                event_stop.synchronize()
            compute_finished = time.perf_counter_ns()
            digest, gate = checksum.consume_device_result(result, count, config["steps"],
                reference, reference_digest, expected_dtype=config["dtype"])
            consumed_finished = time.perf_counter_ns()
            results.put({
                "event": "worker_sample", "status": "validated" if gate["passed"] else "rejected",
                "physical_device": config["device"], "start": start, "stop": stop,
                "phase": command["phase"], "index": command["index"],
                "compute_finished_perf_ns": compute_finished,
                "consume_finished_perf_ns": consumed_finished,
                "compute_wall_seconds": (compute_finished - worker_started) / 1e9,
                "inclusive_worker_seconds": (consumed_finished - worker_started) / 1e9,
                "cuda_event_span_seconds": (cp.cuda.get_elapsed_time(event_start, event_stop) / 1000.
                                            if event_start is not None else None),
                "digest": digest, "gate": gate,
            })
            del result
    except Exception as exc:
        results.put({"event": "worker_error", "status": "failed", "physical_device": config["device"],
                     "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def _collect(results, count, timeout):
    records = []
    deadline = time.monotonic() + timeout
    while len(records) < count:
        row = results.get(timeout=max(0.001, deadline - time.monotonic()))
        if row["event"] == "worker_error":
            raise RuntimeError(json.dumps(row))
        records.append(row)
    return records


def _run_mode(context, config, devices, emit):
    count = len(devices)
    bounds = np.linspace(0, config["n"], count + 1, dtype=int)
    barrier = context.Barrier(count + 1)
    results = context.Queue()
    commands = [context.Queue() for _ in devices]
    workers = []
    mode = "single_gpu" if count == 1 else "dual_gpu"
    metadata = {"mode": mode, "backend": config["backend"], "n": config["n"], "steps": config["steps"]}
    try:
        for index, device in enumerate(devices):
            worker_config = {**config, "device": device, "start": int(bounds[index]), "stop": int(bounds[index + 1])}
            process = context.Process(target=_timing_worker,
                                      args=(worker_config, commands[index], results, barrier))
            process.start()
            workers.append(process)
        for row in _collect(results, count, config["timeout"]):
            emit({**metadata, **row})
        samples = []
        for phase in ("warmup", "measured_run"):
            for index in range(REPEATS):
                emit({**metadata, "event": "sample_start", "phase": phase, "index": index})
                for stream in commands:
                    stream.put({"phase": phase, "index": index})
                started = time.perf_counter_ns()
                barrier.wait(timeout=config["timeout"])
                rows = _collect(results, count, config["timeout"])
                finished = time.perf_counter_ns()
                rows.sort(key=lambda row: row["start"])
                if any(row["phase"] != phase or row["index"] != index for row in rows):
                    raise AssertionError("Worker returned a result for the wrong sample")
                coverage = [(row["start"], row["stop"]) for row in rows]
                if coverage != list(zip(map(int, bounds[:-1]), map(int, bounds[1:]))):
                    raise AssertionError("Worker partition coverage is incomplete or overlapping")
                effective = sum(row["digest"]["counts"]["effective_column_steps"] for row in rows)
                passed = all(row["gate"]["passed"] for row in rows) and effective == config["n"] * config["steps"]
                timing = {
                    "inclusive_coordinator_seconds": (finished - started) / 1e9,
                    "compute_coordinator_seconds": (max(row["compute_finished_perf_ns"] for row in rows) - started) / 1e9,
                    "compute_worker_max_seconds": max(row["compute_wall_seconds"] for row in rows),
                    "inclusive_worker_max_seconds": max(row["inclusive_worker_seconds"] for row in rows),
                }
                emit({**metadata, "event": phase if passed else "validation_failed",
                      "status": "validated" if passed else "rejected", "index": index,
                      "timing_is_diagnostic_only": not passed, **timing,
                      "effective_column_steps": effective, "workers": rows})
                if not passed:
                    raise AssertionError(f"{mode} {phase} {index}: strict full-output gate failed")
                if phase == "measured_run":
                    samples.append(timing)
        if _source_hashes() != config["sourcehash"]:
            raise AssertionError("Source changed during scaling run; reject all timing")
        summary = {key: statistics.median(row[key] for row in samples) for key in samples[0]}
        summary.update({**metadata, "event": "mode_result", "status": "timing_validated",
                        "samples": samples, "devices": devices, "full_reference_gate_passed": True})
        emit(summary)
        return summary
    finally:
        barrier.abort()
        for stream in commands:
            stream.put({"phase": "stop"})
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        for stream in commands:
            stream.close()
        results.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax32", "jax64", "cuda"), default="jax32")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100000, 64800])
    parser.add_argument("--devices", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--sample-timeout", type=float, default=1800.)
    parser.add_argument("--cuda-parity-passed", action="store_true",
                        help="Independent H4 assertion only; cannot bypass the scaling checksum gate")
    args = parser.parse_args()
    if any(n < 2 or n % 2 for n in args.sizes) or len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must be distinct positive even integers, allowing exact N/2 splits")
    if args.steps != 48 or args.sample_timeout <= 0 or min(args.devices) < 0 or len(set(args.devices)) != 2:
        parser.error("Require steps=48, a positive timeout, and two distinct nonnegative physical GPU IDs")
    if args.output.exists() or args.work_dir.exists():
        parser.error("Use fresh output and work-dir paths to preserve all prior evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=False)

    def emit(row):
        checksum, _ = _imports()
        line = json.dumps(checksum.json_safe({"utc": datetime.now(timezone.utc).isoformat(), **row}), allow_nan=False)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    context = mp.get_context("spawn")
    hashes = _source_hashes()
    dtype = "float64" if args.backend == "jax64" else "float32"
    base = {"backend": args.backend, "dtype": dtype, "steps": args.steps,
            "sourcehash": hashes, "timeout": args.sample_timeout}
    emit({"event": "start", **base, "sizes": args.sizes, "devices": args.devices,
          "warmups": REPEATS, "repeats": REPEATS,
          "input_kind": "deterministic heterogeneous synthetic columns; no geographic observations",
          "preparation_h2d_reference_outside_timing": True,
          "inclusive_boundary": "barrier start through all device reductions, host readback and result IPC",
          "compute_boundary": "coordinator before barrier wait through final worker synchronized compute completion; includes barrier waiting and scheduling",
          "worker_compute_boundary": "each worker after barrier release through synchronized compute completion",
          "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
          "strict_gate": "all float outputs pointwise rtol=atol=1e-6; all integers exact; all-field reductions",
          "h4_independently_asserted": bool(args.cuda_parity_passed),
          "reference_cache": str(args.work_dir.resolve())})
    try:
        devices = [_once(context, "probe", {**base, "device": device}) for device in args.devices]
        for row in devices:
            emit(row)
        if len({tuple(row["device"]["pci_identity"]) for row in devices}) != 2:
            raise AssertionError("Two distinct physical NVIDIA CUDA GPUs are required")
        _, workload = _imports()
        for n in args.sizes:
            folder = args.work_dir.resolve() / f"n{n}"
            folder.mkdir(exist_ok=False)
            preparation_started = time.perf_counter()
            f, i = workload.initial_columns(n, np.dtype(dtype))
            u = workload.daily_forcing(np.arange(n), 0, np.dtype(dtype), args.seed, args.steps)
            for key, array in zip(("f", "i", "u"), (f, i, u)):
                np.save(folder / f"input_{key}.npy", array, allow_pickle=False)
            emit({"event": "inputs_prepared", "n": n, "steps": args.steps,
                  "seed": args.seed, "input_sha256": workload.digest_arrays(f, i, u),
                  "preparation_seconds": time.perf_counter() - preparation_started})
            del f, i, u
            config = {**base, "n": n, "folder": str(folder)}
            emit(_once(context, "reference", {**config, "device": args.devices[0]}))
            single = _run_mode(context, config, args.devices[:1], emit)
            dual = _run_mode(context, config, args.devices, emit)
            independent_h4 = args.backend != "cuda" or args.cuda_parity_passed
            emit({"event": "scaling_result", "status": "measured" if independent_h4 else "timing_validated_h4_unverified",
                  "backend": args.backend, "n": n, "steps": args.steps,
                  "accepted_speedup": bool(independent_h4), "full_reference_gate_passed": True,
                  "h4_independently_passed": bool(args.cuda_parity_passed) if args.backend == "cuda" else None,
                  "inclusive_speedup": single["inclusive_coordinator_seconds"] / dual["inclusive_coordinator_seconds"],
                  "compute_speedup": single["compute_coordinator_seconds"] / dual["compute_coordinator_seconds"],
                  "single_gpu": single, "dual_gpu": dual})
        emit({"event": "summary", "status": "completed"})
        return 0
    except (Exception, queue.Empty) as exc:
        emit({"event": "summary", "status": "rejected", "accepted_speedup": False,
              "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

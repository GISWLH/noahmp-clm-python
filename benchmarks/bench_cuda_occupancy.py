"""Bounded launch-size experiment with the full J2 output-consumption gate.

Only the launch configuration changes inside this process. Production source is
not edited. Times include three warmups and three measured complete trajectories;
all returned state and step diagnostics must agree with the same-N JAX reference.
The experiment cannot turn an unpassed NumPy H4 gate into physics validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from bench_gpu import _audited_gpu_timed
from gpu_checksum import json_safe
from noahmp_clm.io import synthetic_forcing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--blocks", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 1 or args.steps < 1 or args.output.exists():
        parser.error("Positive counts and a fresh output path are required")
    if any(size < 32 or size > 1024 or size % 32 for size in args.blocks):
        parser.error("Block sizes must be warp multiples between 32 and 1024")
    if len(set(args.blocks)) != len(args.blocks):
        parser.error("Duplicate block sizes would distort this bounded experiment")
    import cupy as cp
    from noahmp_clm.gpu.cuda import loader
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py"))
    paths += sorted((root / "src").rglob("*.cu"))
    paths += [Path(__file__).resolve(), root / "benchmarks/bench_gpu.py",
              root / "benchmarks/gpu_checksum.py"]
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    forcing = synthetic_forcing(n_days=args.steps / 48., dt=1800.)
    if len(forcing) != args.steps:
        raise ValueError("Forcing length differs from requested steps")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original_launch = loader._launch_unchecked
    results = []
    with args.output.open("x", encoding="utf-8") as stream:
        def emit(row):
            value = json.dumps(json_safe({"utc": datetime.now(timezone.utc).isoformat(),
                                         **row}), allow_nan=False)
            print(value, file=stream, flush=True)
            print(value, flush=True)

        emit({"event": "environment", "n": args.n, "steps": args.steps,
              "blocks": args.blocks, "device": str(cp.cuda.Device()),
              "device_pci_bus_id": cp.cuda.Device().pci_bus_id, "cupy": cp.__version__,
              "sourcehash": hashes, "source_modified": False,
              "launch_override": "this process only; grid=min(ceil(N/block),4096)",
              "order_randomized": False, "h4_passed": False})
        try:
            for block_size in args.blocks:
                def launch(state, options, forcing_soa, state_out, options_out, diag,
                           size=block_size):
                    count = state.shape[1]
                    if count:
                        grid = min((count + size - 1) // size, 4096)
                        loader.load_column_kernel()((grid,), (size,),
                            (state, options, forcing_soa, state_out, options_out, diag,
                             np.int32(count)))

                loader._launch_unchecked = launch
                base = {"n": args.n, "steps": args.steps, "block_size": block_size}
                try:
                    elapsed, samples = _audited_gpu_timed("cuda", args.n, forcing, 3,
                        lambda event, index, details: emit({**base, "event": event,
                                                           "index": index, **details}))
                    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[
                           path.relative_to(root).as_posix()] for path in paths):
                        raise AssertionError("Source changed during launch-size experiment")
                    row = {**base, "event": "block_result",
                           "status": "timing_validated_h4_unverified",
                           "inclusive_seconds": elapsed,
                           "compute_wall_seconds": statistics.median(
                               value["synchronized_compute_wall_seconds"] for value in samples),
                           "cuda_event_span_seconds": statistics.median(
                               value["cuda_event_span_seconds"] for value in samples),
                           "samples": samples,
                           "kernel_attributes": loader.load_column_kernel().attributes,
                           "h4_passed": False}
                except Exception as exc:
                    row = {**base, "event": "block_result", "status": "rejected",
                           "error": f"{type(exc).__name__}: {exc}"}
                results.append(row)
                emit(row)
        finally:
            loader._launch_unchecked = original_launch
        final_hashes = {path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()).hexdigest() for path in paths}
        source_unchanged = hashes == final_hashes
        passed = source_unchanged and all(row["status"] != "rejected" for row in results)
        emit({"event": "experiment_end", "all_output_gates_passed": passed,
              "source_unchanged": source_unchanged, "final_sourcehash": final_hashes,
              "production_launch_changed": False, "h4_passed": False,
              "selection_rule": "Report inclusive and compute medians separately; do not auto-select"})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

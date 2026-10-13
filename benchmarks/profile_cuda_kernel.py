"""Bounded CUDA counter probe; diagnostic only, never an accepted benchmark.

Use ncu --profile-from-start off --kernel-name regex:step_columns --launch-count 1.
Three fully validated warmups precede the captured call. The profiler selects its first
column step, so counters cannot be extrapolated to all 48 steps without evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from bench_gpu import _gpu_runner
from gpu_checksum import as_cupy_result, assert_gate, consume_device_result
from noahmp_clm.io import synthetic_forcing
from profile_gpu import _profiler_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 1 or args.steps < 1 or args.output.exists():
        parser.error("Positive counts and a fresh output path are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import cupy as cp
    from noahmp_clm.gpu.cuda import loader
    load_column_kernel = loader.load_column_kernel
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "src").rglob("*.cu"))
    paths += [Path(__file__).resolve(), root / "benchmarks/bench_gpu.py",
              root / "benchmarks/gpu_checksum.py", root / "benchmarks/profile_gpu.py"]
    def source_hashes():
        return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths}
    hashes = source_hashes()
    forcing = synthetic_forcing(n_days=args.steps / 48., dt=1800.)
    if len(forcing) != args.steps:
        raise ValueError("Forcing length mismatch")
    reference = _gpu_runner("jax32", args.n, forcing)()
    reference_arrays = as_cupy_result(reference)
    reference_digest, reference_gate = consume_device_result(
        reference, args.n, args.steps, expected_dtype="float32")
    assert_gate(reference_gate)
    # Disk hashes alone do not identify code imported from an editable checkout.
    loaded_modules = {}
    for name, module in tuple(sys.modules.items()):
        if name == "noahmp_clm" or name.startswith("noahmp_clm."):
            actual = Path(module.__file__).resolve()
            if not actual.is_relative_to(root / "src"):
                raise AssertionError(f"Imported {name} from outside frozen source: {actual}")
            loaded_modules[name] = actual.relative_to(root).as_posix()
    loaded_cuda_hash = hashlib.sha256(loader.KERNEL_SOURCE.encode("utf-8")).hexdigest()
    disk_cuda_hash = hashlib.sha256((root / "src/noahmp_clm/gpu/cuda/column_kernel.cu")
        .read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    if loaded_cuda_hash != disk_cuda_hash:
        raise AssertionError("Loaded CUDA source differs from the recorded disk source")
    runner = _gpu_runner("cuda", args.n, forcing)
    warmup_gates = []
    for _ in range(3):
        result = runner()
        _, gate = consume_device_result(result, args.n, args.steps,
            reference_arrays, reference_digest, expected_dtype="float32")
        assert_gate(gate)
        warmup_gates.append(gate)
        del result
    if source_hashes() != hashes:
        raise AssertionError("Source changed before profiler capture")
    runtime, runtime_path = _profiler_runtime()
    if runtime.cudaProfilerStart() != 0:
        raise RuntimeError("Profiler start failed")
    try:
        result = runner()
    finally:
        stop_status = runtime.cudaProfilerStop()
    if stop_status:
        raise RuntimeError("Profiler stop failed")
    digest, gate = consume_device_result(result, args.n, args.steps,
        reference_arrays, reference_digest, expected_dtype="float32")
    final_hashes = source_hashes()
    if final_hashes != hashes:
        raise AssertionError("Source changed during profiler capture")
    import jax
    import jaxlib
    device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    row = {"status": "diagnostic_only_h4_unvalidated", "accepted_benchmark": False,
           "n": args.n, "steps": args.steps, "sourcehash": hashes,
           "cupy": cp.__version__, "device": str(cp.cuda.Device()),
           "device_name": device_name, "device_pci_bus_id": cp.cuda.Device().pci_bus_id,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "jax": jax.__version__, "jaxlib": jaxlib.__version__,
           "loaded_modules": loaded_modules, "loaded_cuda_sha256_normalized": loaded_cuda_hash,
           "runtime": runtime_path, "kernel_attributes": load_column_kernel().attributes,
           "digest": digest, "finite_and_shape_gate": gate,
           "reference_parity_gate": gate, "warmup_gates": warmup_gates,
           "reference_backend": "jax32", "reference_parity_checked_here": True,
           "source_unchanged": True, "final_sourcehash": final_hashes}
    args.output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: row[key] for key in ("status", "n", "steps", "kernel_attributes")}))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

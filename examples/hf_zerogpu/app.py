"""Bounded Noah-MP compatibility demo; deploy with src/ and the benchmark helper."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import gradio as gr
import numpy as np
import spaces
import torch  # ZeroGPU's supported allocation layer; JAX is checked independently.

from check_closeout import compare_cross, conservation, digest, numpy_trajectory
from probe_gpu_numerics import inputs as h4_inputs
from noahmp_clm.column import ColumnState
from noahmp_clm.gpu.pack import column_state_to_vec


@spaces.GPU(duration=120)
def gpu_smoke(f, i, forcing, precision):
    # JAX must initialize while a real device is allocated, never silently on CPU.
    import jax
    jax.config.update("jax_enable_x64", True)
    from noahmp_clm.gpu.jax_core import run_columns_jax
    devices = jax.devices()
    if not devices or any(d.platform != "gpu" for d in devices):
        raise RuntimeError(f"JAX did not get a GPU: {devices}")
    dtype = np.float32 if precision == "float32" else np.float64
    started = time.perf_counter()
    f, forcing = f.astype(dtype), forcing.astype(dtype)
    result = run_columns_jax(f, i, forcing)
    jax.block_until_ready(result)
    final = tuple(np.asarray(a) for a in result)
    state, integer = jax.device_put((f, i))
    ss, ii, dd = [f.copy()], [i.copy()], []
    for step in range(forcing.shape[1]):
        state, integer, diag = run_columns_jax(state, integer, forcing[:, step:step+1])
        ss.append(np.asarray(state)); ii.append(np.asarray(integer)); dd.append(np.asarray(diag[:, 0]))
    host = np.stack(ss, 1), np.stack(ii, 1), np.stack(dd, 1)
    replay_equal = all(np.array_equal(a, b) for a, b in
                       ((host[0][:, -1], final[0]), (host[1][:, -1], final[1]), (host[2], final[2])))
    elapsed = time.perf_counter()-started
    return host, {"devices": [str(d) for d in devices], "jax": jax.__version__,
                  "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                  "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                  "seconds_jit_compute_full_readback": elapsed,
                  "step_replay_equals_full_batch": replay_equal}


def run_smoke(precision="float32"):
    started = time.perf_counter()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "columns": 1024, "steps": 48, "precision": precision,
              "python": platform.python_version(), "numpy": np.__version__,
              "source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "src").rglob("*.py"))},
              "scope": "Synthetic compatibility smoke; not observational validation or a performance benchmark"}
    try:
        # Both sides start with the same float32-quantized inputs.
        f, one, _ = h4_inputs(1024)
        i = np.broadcast_to(column_state_to_vec(ColumnState())[1], (1024, 33)).copy()
        forcing = np.broadcast_to(one[:, None], (1024, 48, 8)).copy()
        forcing[:, 12:, 7] = 0
        report["input_sha256"] = digest(f, i, forcing)
        tick = time.perf_counter()
        rf, ri, rd = numpy_trajectory(f, i, forcing)
        report["numpy_unique_input_columns"] = len({digest(f[k], i[k], forcing[k]) for k in range(1024)})
        report["numpy_reference_seconds"] = time.perf_counter()-tick
        (af, ai, ad), report["gpu_environment"] = gpu_smoke(f, i, forcing, precision)
        report["checksum_sha256"] = digest(af, ai, ad)
        report["numpy_checksum_sha256"] = digest(rf, ri, rd)
        report["finite"] = bool(np.isfinite(af).all() and np.isfinite(ad).all())
        report["integers_equal"] = bool(np.array_equal(ai, ri))
        labels = [("cold_snow", "melting_initial", "rain", "dry")[c % 4] for c in range(1024)]
        report["m2_comparison"], report["outlier_rows"] = compare_cross(af, ad, rf, rd, forcing, labels=labels)
        report["conservation"] = conservation(af, ad)
        if precision == "float64":
            # The fixed ABI carries padding; match the existing H2/M4 valid-layer mask.
            active = np.ones(rf.shape, dtype=bool)
            active[:, :, 45:] = False
            for offset in (16, 19, 22):
                for layer in range(3):
                    active[:, :, offset+layer] = layer < ri[:, :, 32]
            report["float64_strict_parity"] = bool(np.allclose(af[active], rf[active], rtol=1e-6, atol=1e-6)
                                                  and np.allclose(ad, rd, rtol=1e-6, atol=1e-6))
        report["passed"] = (report["finite"] and report["integers_equal"] and
                            report["gpu_environment"]["step_replay_equals_full_batch"] and
                            report["m2_comparison"]["numeric_thresholds_passed"] and
                            report["conservation"]["surface_accounting_passed"] and
                            report["conservation"]["water_step_passed"] and
                            report.get("float64_strict_parity", True))
    except Exception as exc:
        report.update(passed=False, error=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc())
    report["wall_seconds"] = time.perf_counter()-started
    return json.dumps(report, indent=2, allow_nan=False)


with gr.Blocks(title="Noah-MP short GPU smoke") as demo:
    gr.Markdown("# Noah-MP short GPU smoke\n1024 synthetic columns × 48 half-hour steps. "
                "Compares JAX GPU with a NumPy float64 reference. This is a compatibility check, "
                "not physical validation. Cold compilation and the CPU reference take time.")
    precision = gr.Dropdown(["float32", "float64"], value="float32", label="JAX precision")
    button = gr.Button("Run bounded parity smoke")
    output = gr.Code(label="Measured report", language="json")
    button.click(run_smoke, inputs=precision, outputs=output, api_name="run_smoke")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()

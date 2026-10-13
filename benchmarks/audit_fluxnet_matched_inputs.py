"""Apply the preregistered same-input M2 gate to immutable M4 GPU outputs.

Only the independent NumPy64 reference is rerun, using identical float32 inputs.
The original end-to-end report and NPZ are never overwritten. No GPU is used.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from check_closeout import compare_cross, conservation, digest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from fluxnet_multisite import automatic_checks_pass, numpy_trace, provenance


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve() or args.arrays.resolve() == args.trajectories.resolve():
        raise ValueError("Original artifacts must be preserved")
    original = json.loads(args.report.read_text(encoding="utf-8"))
    assert sha(args.trajectories) == original["arrays_sha256"]
    source_check = {}
    for name, expected in original["source_sha256"].items():
        if not name.startswith("src/"):
            continue
        data = (ROOT / name).read_bytes()
        lf = data.replace(b"\r\n", b"\n")
        source_check[name] = expected in {hashlib.sha256(v).hexdigest()
                                        for v in (data, lf, lf.replace(b"\n", b"\r\n"))}
    assert all(source_check.values()), "Reference physics must match the original GPU run"
    before = provenance()
    stored = np.load(args.trajectories)
    arrays = {key: stored[key] for key in stored.files}
    arrays["original_forcing"] = arrays["forcing"]
    arrays["original_initial_states"] = arrays["initial_states"]
    arrays["forcing"] = arrays["forcing"].astype(np.float32)
    arrays["initial_states"] = arrays["initial_states"].astype(np.float32)
    f, i, u = arrays["initial_states"], arrays["initial_integers"], arrays["forcing"]
    started = time.perf_counter()
    states, integers, diagnostics = numpy_trace(f, i, u)
    elapsed = time.perf_counter()-started
    for name, value in zip(("states", "integers", "diagnostics"), (states, integers, diagnostics)):
        arrays["numpy64_matched32_"+name] = value
    report = deepcopy(original)
    report["utc_matched_input_audit"] = datetime.now(timezone.utc).isoformat()
    report["original_end_to_end_m2"] = report.pop("m2")
    report["m2"] = {}
    report["reference"] = "NumPy64 physics with exactly the same float32-quantized initial states and forcing as each GPU backend"
    labels = [record["site"] for record in original["inputs"]]
    for backend in ("jax32", "cuda"):
        fs, ix, ds = (arrays[backend+"_"+name] for name in ("states", "integers", "diagnostics"))
        summary, rows = compare_cross(fs, ds, states, diagnostics, u, labels=labels)
        summary["reference"] = report["reference"]
        summary["integers_equal"] = bool(np.array_equal(ix, integers))
        summary["per_site"] = {site:compare_cross(fs[c:c+1], ds[c:c+1], states[c:c+1],
            diagnostics[c:c+1], u[c:c+1], labels=[site])[0] for c,site in enumerate(labels)}
        path = args.output.with_name(args.output.stem+"-"+backend+"-outliers.jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row)+"\n")
        summary["outliers_file"] = path.name
        report["m2"][backend] = summary
    report["backends"]["numpy64_matched32"] = {
        "elapsed_s": elapsed, "finite": bool(np.isfinite(states).all() and np.isfinite(diagnostics).all()),
        "production_trace_exact": True, "device": "CPU", "checksum_sha256": digest(states, integers, diagnostics),
        "conservation": conservation(states, diagnostics)}
    report["matched_input_sha256"] = digest(f, i, u)
    report["audit_source_sha256"] = before
    report["audit_instrument_sha256"] = sha(Path(__file__))
    report["physics_matches_original_run_allowing_line_endings"] = source_check
    report["source_unchanged"] = original["source_unchanged"] and before == provenance()
    report["original_report_sha256"] = sha(args.report)
    report["original_arrays_sha256"] = sha(args.trajectories)
    report["gpu_outputs_reused_without_change"] = True
    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **arrays)
    report["arrays"] = str(args.arrays)
    report["arrays_sha256"] = sha(args.arrays)
    report["automated_checks_passed"] = automatic_checks_pass(report)
    report["acceptance_status"] = ("numeric_checks_passed_requires_outlier_review"
                                   if report["automated_checks_passed"] else "failed_numeric_gate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"automated_checks_passed": report["automated_checks_passed"],
                      "canw": report["m2"]["cuda"]["fields"]["canw"], "cpu_seconds": elapsed}))
    return 0 if report["automated_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Scientific 10%/absolute-tolerance summary of the existing K3 domain outputs.

The old strict gate count is a historical comparison only, not a pass criterion.
All 64,800 columns and all 48 steps contribute; no tail trimming or masking of
near-zero observations. Integer states and finite values remain hard checks.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_gpu_extended import DIAGNOSTICS, SCENARIOS
from noahmp_clm.validation import ABSOLUTE_TOLERANCES


def metrics(a, b, variable):
    a, b = np.asarray(a, float), np.asarray(b, float)
    atol = ABSOLUTE_TOLERANCES[variable]
    near = np.abs(b) <= atol / 0.1
    delta = a - b
    limit = np.where(near, atol, .1 * np.abs(b))
    return {"variable": variable, "samples": int(a.size),
        "max_absolute_bias": float(np.max(np.abs(delta))),
        "max_relative_bias_nonnearzero": float(np.max(np.abs(delta[~near] / b[~near]))) if np.any(~near) else None,
        "near_zero_samples": int(np.sum(near)), "absolute_tolerance": atol,
        "failures": int(np.sum(np.abs(delta) > limit)),
        "rmse": float(np.sqrt(np.mean(delta**2)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    root = args.directory
    gpu, cpu = [json.loads((root / name).read_text()) for name in ("gpu.json", "cpu.json")]
    if gpu["source_hashes"] != cpu["source_hashes"]:
        raise ValueError("CPU/GPU physical source identities differ")
    af, ef, ad, ed, ai, ei = [np.load(root / (name + ".npy"), mmap_mode="r") for name in
        ("actual_f", "expected_f", "actual_d", "expected_d", "actual_i", "expected_i")]
    n = len(af)
    if cpu["columns_completed"] != n:
        raise ValueError("Incomplete CPU reference")
    comparisons = {}
    for name, variable, scale in (("h", "h_wm2", 1.), ("le", "le_wm2", 1.),
            ("g", "g_wm2", 1.), ("et_mm", "et_mm_day", 48.),
            ("runoff_mm", "runoff_mm_day", 48.), ("swe", "swe_m", 1.), ("canw", "canw_m", 1.)):
        index = DIAGNOSTICS.index(name)
        a, b = np.asarray(ad[:,:,index], float), np.asarray(ed[:,:,index], float)
        comparisons[name] = metrics(a.mean(1)*scale, b.mean(1)*scale, variable)
        comparisons[name]["max_instantaneous_absolute_difference"] = float(np.max(np.abs(a-b)))
        if name in ("h", "le", "g"):
            ac, bc = a-a.mean(1)[:,None], b-b.mean(1)[:,None]
            denominator = np.sqrt(np.sum(ac*ac,1)*np.sum(bc*bc,1))
            valid = denominator > 0.
            r = np.sum(ac[valid]*bc[valid],1)/denominator[valid]
            comparisons[name]["r_min"] = float(np.min(r)) if r.size else None
            comparisons[name]["r_defined_columns"] = int(np.sum(valid))
            comparisons[name]["r_undefined_columns"] = int(np.sum(~valid))
            comparisons[name]["r_failures"] = int(np.sum(r <= .85))
    comparisons["final_tsoil_c"] = metrics(af[:,:4].astype(float)-273.15, ef[:,:4]-273.15, "tsoil_c")
    comparisons["final_smc"] = metrics(af[:,4:8], ef[:,4:8], "smc")
    comparisons["final_soil_ice"] = metrics(af[:,8:12], ef[:,8:12], "smc")
    error = np.abs(af[:,:45].astype(float)-ef[:,:45])
    bad = error > 1e-6 + 1e-4 * np.abs(ef[:,:45])
    for start in (16,19,22):
        bad[:,start:start+3] &= np.arange(3)[None,:] < ei[:,32,None]
    historical = {}
    for k, name in enumerate(SCENARIOS):
        mask = np.arange(n) % 8 == k
        historical[name] = {"columns": int(mask.sum()),
            "retired_threshold_state_bad_columns": int(np.any(bad[mask],axis=1).sum()),
            "max_temperature_difference_K": float(error[mask,:4].max()),
            "max_ice_difference_m3m3": float(error[mask,8:12].max())}
    finite = bool(all(np.isfinite(a).all() for a in (af, ef, ad, ed)))
    integers = bool(np.array_equal(ai, ei))
    report = {"columns": n, "steps": int(ad.shape[1]), "finite": finite,
        "integer_equal": integers, "source_hashes": gpu["source_hashes"],
        "comparisons": comparisons, "historical_threshold_descriptive_only": historical,
        "passed": finite and integers and all(v["failures"] == 0 and v.get("r_failures", 0) == 0 for v in comparisons.values()),
        "scope": "synthetic backend comparison; not independent observed physics validation"}
    (root / "v1-summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

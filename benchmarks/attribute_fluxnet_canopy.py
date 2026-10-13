"""Attribute every M4 canopy failure without changing its acceptance status.

Reconstruct the frozen ordinary canopy interception/evaporation formula at
saved GPU and NumPy states and temperatures. Includes input quantization.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from check_closeout import compare_cross, source_hashes
from noahmp_clm.gpu.pack import forcing_vec_to_forcing
from noahmp_clm.options import PhysicsOptions
from noahmp_clm.physics import energy as E
from noahmp_clm.physics.precip import partition_precipitation
from noahmp_clm.physics.surface_drag import exchange_velocity


def canopy64(pre, temperature, forcing, options):
    f = forcing_vec_to_forcing(forcing.astype(np.float64))
    names = [field.name for field in fields(PhysicsOptions)]
    rain, snow = partition_precipitation(f.prcp, f.dt, f.tair,
                                         int(options[names.index("OptRainSnowPartition")]))
    assert float(pre[26]) + snow <= 1e-4, "snow surface requires separate reconstruction"
    c, canmax = float(pre[30]), .5e-3 * float(pre[31])
    c += min(rain, max(0., canmax-c))
    ce = exchange_velocity(f, float(pre[33]), float(temperature),
                           int(options[names.index("OptSurfaceDrag")]))
    potential = (E.air_density(f)*E.LV*ce*(E.qsat(float(temperature), f.ps)-f.qair)
                 * min(1., c/max(1e-9, .2*canmax)))
    cap = c*E.LV*1000./f.dt
    return float(max(0., c-min(potential, cap)/E.LV*f.dt/1000.)), bool(potential >= cap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-prefix", default="numpy64",
                        choices=("numpy64", "numpy64_matched32"))
    args = parser.parse_args()
    a = np.load(args.trajectories)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    labels = [entry["site"] for entry in report["inputs"]]
    prefix = args.reference_prefix
    summary, rows = compare_cross(a["cuda_states"], a["cuda_diagnostics"],
                                  a[prefix+"_states"], a[prefix+"_diagnostics"],
                                  a["forcing"], labels=labels)
    evidence = []
    for row in rows:
        assert row["field"] == "canw", "do not misattribute another field to canopy drying"
        c, t = row["column"], row["step_1based"]-1
        gp, np_ = a["cuda_states"][c, t], a[prefix+"_states"][c, t]
        gt, nt = a["cuda_diagnostics"][c, t, 0], a[prefix+"_diagnostics"][c, t, 0]
        u64 = a["forcing"][c, t]
        u32 = u64.astype(np.float32)
        option = a["initial_integers"][c]
        full32, capped32 = canopy64(gp, gt, u32, option)
        temperature64, _ = canopy64(gp, nt, u32, option)
        forcing64, _ = canopy64(gp, nt, u64, option)
        full64, capped64 = canopy64(np_, nt, u64, option)
        terms = {"float32_evaluation_roundoff_m": row["actual"]-full32,
                 "temperature_path_m": full32-temperature64,
                 "forcing_quantization_m": temperature64-forcing64,
                 "previous_state_path_m": forcing64-full64,
                 "reference_reconstruction_m": full64-row["reference"]}
        evidence.append(dict(row, **terms,
            reconstruction_error_m=sum(terms.values())-(row["actual"]-row["reference"]),
            previous_canw_cuda_m=float(gp[30]), previous_canw_numpy_m=float(np_[30]),
            previous_swe_numpy_m=float(np_[26]), precipitation_m_s=float(u64[7]),
            capped_cuda_state=capped32, capped_numpy_state=capped64,
            cause="canopy_exhaustion_roundoff" if capped32 or capped64
                  else "canopy_drying_relative_sensitivity"))
    names = ("float32_evaluation_roundoff_m", "temperature_path_m", "forcing_quantization_m",
             "previous_state_path_m", "reference_reconstruction_m", "reconstruction_error_m")
    result = {"source_sha256": source_hashes(),
              "reference_prefix": prefix,
              "instrument_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "trajectories_sha256": hashlib.sha256(args.trajectories.read_bytes()).hexdigest(),
              "outliers": len(evidence), "by_site": dict(Counter(r["group"] for r in evidence)),
              "by_cause": dict(Counter(r["cause"] for r in evidence)),
              "max_abs_outlier_error_m": max((abs(r["actual"]-r["reference"]) for r in evidence), default=0.),
              "max_canw_in_outliers_m": max((max(r["actual"],r["reference"]) for r in evidence), default=0.),
              "term_max_abs_m": {name:max((abs(r[name]) for r in evidence), default=0.) for name in names},
              "numeric_thresholds_passed": summary["numeric_thresholds_passed"],
              "canw": summary["fields"]["canw"],
              "attribution_changes_acceptance": False,
              "interpretation": "All failed values reconstruct as canopy drying/exhaustion, including one rainy step; pure relative sensitivity near zero. Explaining the errors does not override the preregistered 99.5% gate."}
    result["all_reference_reconstructed"] = result["term_max_abs_m"]["reference_reconstruction_m"] < 1e-15
    result["all_differences_reconstructed"] = result["term_max_abs_m"]["reconstruction_error_m"] < 1e-20
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    with gzip.open(args.output.with_suffix(".jsonl.gz"), "wt", encoding="utf-8") as stream:
        for row in evidence:
            stream.write(json.dumps(row)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "source_sha256"}, indent=2))
    return 0 if result["all_reference_reconstructed"] and result["all_differences_reconstructed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Reconstruct every M2 canopy outlier using the frozen NumPy formula.

This is a diagnostic, not a change to physics or to acceptance tolerances.
Run on the same source as check_closeout.py, with its complete trajectory NPZ.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from check_closeout import compare_cross, source_hashes
from noahmp_clm.gpu.pack import forcing_vec_to_forcing
from noahmp_clm.physics import energy as E
from noahmp_clm.physics.surface_drag import exchange_velocity


def canopy64(pre, temperature, values, option):
    f = forcing_vec_to_forcing(values)
    assert f.prcp == 0 and pre[26] <= 1e-4
    rho = E.air_density(f)
    ce = exchange_velocity(f, float(pre[33]), float(temperature), int(option))
    c = float(pre[30])
    cap = c*E.LV*1000./f.dt
    potential = rho*E.LV*ce*(E.qsat(float(temperature), f.ps)-f.qair)*min(1., c/max(1e-9, .2*.5e-3*float(pre[31])))
    power = min(potential, cap)
    return max(0., c-power/E.LV*f.dt/1000.), bool(potential >= cap)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectories", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    a = np.load(args.trajectories)
    labels = [("cold_snow", "melting_initial", "rain", "dry")[c % 4] for c in range(len(a["initial"]))]
    _, rows = compare_cross(a["cuda_states"], a["cuda_diagnostics"], a["numpy_states"],
                            a["numpy_diagnostics"], a["forcing"], labels=labels)
    evidence = []
    # Surface drag is option index 5 in the fixed ABI; obtain the index from the
    # dataclass order instead of relying on a hand-written magic offset.
    from dataclasses import fields
    from noahmp_clm.options import PhysicsOptions
    drag_index = [f.name for f in fields(PhysicsOptions)].index("OptSurfaceDrag")
    for row in rows:
        assert row["field"] == "canw"
        c, t = row["column"], row["step_1based"]-1
        gp, np_ = a["cuda_states"][c, t], a["numpy_states"][c, t]
        gt, nt = a["cuda_diagnostics"][c, t, 0], a["numpy_diagnostics"][c, t, 0]
        u, option = a["forcing"][c, t], a["options"][c, drag_index]
        both32, clipped32 = canopy64(gp, gt, u, option)
        carry32, _ = canopy64(gp, nt, u, option)
        both64, clipped64 = canopy64(np_, nt, u, option)
        actual, reference = row["actual"], row["reference"]
        terms = {"float32_formula_and_rounding_m": actual-both32,
                 "temperature_path_m": both32-carry32,
                 "previous_state_path_m": carry32-both64,
                 "reference_reconstruction_m": both64-reference}
        evidence.append(dict(row, **terms,
                             reconstruction_error_m=sum(terms.values())-(actual-reference),
                             previous_canw_cuda_m=float(gp[30]), previous_canw_numpy_m=float(np_[30]),
                             capped_cuda_state=clipped32, capped_numpy_state=clipped64,
                             cause="canopy_exhaustion_roundoff" if clipped64 or clipped32 else "canopy_drying_relative_sensitivity"))
    summary = {"source_sha256": source_hashes(), "outliers": len(evidence),
               "unique_config_steps": len({(r["column"] % 156, r["step_1based"]) for r in evidence}),
               "clusters": dict(Counter(r["cause"] for r in evidence)),
               "step_counts": dict(Counter(r["step_1based"] for r in evidence)),
               "max_canw_in_outliers_m": max(max(r["actual"], r["reference"]) for r in evidence),
               "max_abs_outlier_error_m": max(abs(r["actual"]-r["reference"]) for r in evidence),
               "term_max_abs_m": {k: max(abs(r[k]) for r in evidence) for k in (*terms, "reconstruction_error_m")},
               "all_without_precip_or_snow": True,
               "interpretation": "All failures are near-empty canopy storage after rain stops; relative sensitivity, with carry/temperature/evaluation terms retained. No threshold changed."}
    args.output.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    with gzip.open(args.output.with_suffix(".jsonl.gz"), "wt", encoding="utf-8") as stream:
        for row in evidence:
            stream.write(json.dumps(row)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""K1: retain every H4 trajectory and dump every residual difference > .002.

The .002 selector identifies the historical counterexamples; it is not a gate.
The actual gate remains rtol=1e-4, atol=1e-6 for every diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from probe_gpu_numerics import inputs
from noahmp_clm.column import ColumnState
from noahmp_clm.driver import step_column
from noahmp_clm.gpu.pack import column_state_to_vec, vec_to_column_state, forcing_vec_to_forcing

NAMES = ("ts", "rnet", "h", "le", "g", "et_mm", "prcp_mm", "runoff_mm", "melt_mm", "swe", "canw", "residual")
REGIMES = ("cold_snow", "melting", "rain", "dry")
ROOT = Path(__file__).resolve().parents[1]


def context(pre, post, diagnostic):
    """Explicit descriptive flags, not a proof of a phase-transition cause."""
    return {
        "snow_present": bool(pre[26] > 1e-4),
        "snow_onset_or_loss": bool((pre[26] > 1e-4) != (post[26] > 1e-4)),
        "melting": bool(diagnostic[8] > 1e-8),
        "frozen_soil": bool(np.any(pre[8:12] > 1e-8)),
        "temperature_crosses_freezing": bool(np.any((pre[:4] < 273.15) != (post[:4] < 273.15))),
        "soil_ice_changes": bool(np.any(abs(post[8:12] - pre[8:12]) > 1e-6)),
        "near_wilting": bool(np.any(abs(pre[4:8] - pre[40]) <= .01)),
        "wilting_crossing": bool(np.any((pre[4:8] <= pre[40]) != (post[4:8] <= post[40]))),
        "near_zero_h": bool(abs(diagnostic[2]) < .1),
        "near_zero_le": bool(abs(diagnostic[3]) < .1),
        "near_zero_g": bool(abs(diagnostic[4]) < .1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", type=Path, required=True)
    args = p.parse_args()
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.prefix.with_suffix(".npz").exists():
        p.error("Use a fresh prefix")
    import cupy as cp
    from noahmp_clm.gpu.cuda.loader import run_columns_cuda
    f, one, _ = inputs(1024)
    i = np.broadcast_to(column_state_to_vec(ColumnState())[1], (1024, 33)).copy()
    u = np.broadcast_to(one[:, None], (1024, 48, 8)).copy()
    u[:, 12:, 7] = 0
    cf, ci, cd = map(cp.asnumpy, run_columns_cuda(f, i, u))
    # Save every state without changing production driver/ABI. Prove this
    # instrumented replay matches the original multi-step driver bit for bit.
    gs, gi, gd = [f.copy()], [i.copy()], []
    sf, si = cp.asarray(f), cp.asarray(i)
    for t in range(48):
        sf, si, d = run_columns_cuda(sf, si, u[:, t:t+1])
        gs.append(cp.asnumpy(sf)); gi.append(cp.asnumpy(si)); gd.append(cp.asnumpy(d[:, 0]))
    gs, gi = np.stack(gs, 1), np.stack(gi, 1)
    np.testing.assert_array_equal(np.stack(gd, 1), cd)
    np.testing.assert_array_equal(gs[:, -1], cf)
    np.testing.assert_array_equal(gi[:, -1], ci)
    cache, ns, ni, nd = {}, [], [], []
    for c in range(1024):
        key = f[c].tobytes() + i[c].tobytes() + u[c].tobytes()
        if key not in cache:
            state = vec_to_column_state(f[c], i[c], ColumnState().dz)
            ss, ii, dd = [column_state_to_vec(state)[0]], [i[c]], []
            for row in u[c]:
                state, d = step_column(state, forcing_vec_to_forcing(row))
                a, b = column_state_to_vec(state)
                ss.append(a); ii.append(b); dd.append([d[k] for k in NAMES])
            cache[key] = (np.array(ss), np.array(ii), np.array(dd))
        a, b, d = cache[key]; ns.append(a); ni.append(b); nd.append(d)
    ns, ni, nd = np.array(ns), np.array(ni), np.array(nd)
    assert np.all(np.isfinite(cd)) and np.all(np.isfinite(nd))
    np.savez_compressed(args.prefix.with_suffix(".npz"), initial=f, options=i, forcing=u,
                        cuda_states=gs, cuda_options=gi, numpy_states=ns, numpy_options=ni,
                        cuda_diagnostics=cd, numpy_diagnostics=nd, names=np.array(NAMES))
    delta = abs(cd.astype(float) - nd)
    strict = delta > 1e-6 + 1e-4 * abs(nd)
    selected = delta[:, :, 11] > .002
    clusters, flag_counts, step_counts = [], [], []
    for k, name in enumerate(NAMES):
        for r, regime in enumerate(REGIMES):
            mask = (np.arange(1024) % 4 == r)
            clusters.append({"diagnostic": name, "regime": regime,
                             "population": int(mask.sum() * 48),
                             "strict_failed": int(strict[mask, :, k].sum()),
                             "max_abs": float(delta[mask, :, k].max())})
    flags = {key: np.zeros((1024, 48), bool) for key in context(ns[0,0], ns[0,1], nd[0,0])}
    for c in range(1024):
        for t in range(48):
            for key, value in context(ns[c,t], ns[c,t+1], nd[c,t]).items():
                flags[key][c,t] = value
    for flag, mask in flags.items():
        flag_counts.append({"flag": flag, "population": int(mask.sum()),
                            "residual_002_count": int((selected & mask).sum()),
                            "strict_by_diagnostic": {n: int((strict[:,:,k] & mask).sum()) for k,n in enumerate(NAMES)}})
    for t in range(48):
        step_counts.append({"step_1based":t+1,"residual_002_count":int(selected[:,t].sum())})
    with args.prefix.with_name(args.prefix.name + "-rows.jsonl").open("w", encoding="utf-8") as stream:
        for c,t in np.argwhere(selected):
            row = {"column":int(c), "unique_case":int(c % 156), "step_1based":int(t+1),
                   "diagnostic":"residual", "soil":("sand","loam","clay")[c%3],
                   "regime":REGIMES[c%4], "lai":float(f[c,31]),
                   "cuda":float(cd[c,t,11]), "numpy64":float(nd[c,t,11]), "abs_difference":float(delta[c,t,11]),
                   "numpy_context":context(ns[c,t],ns[c,t+1],nd[c,t]),
                   "cuda_context":context(gs[c,t],gs[c,t+1],cd[c,t]),
                   "forcing":u[c,t].tolist(), "options_before":gi[c,t].tolist(),
                   "cuda_state_before":gs[c,t].tolist(), "cuda_state_after":gs[c,t+1].tolist(),
                   "numpy_state_before":ns[c,t].tolist(), "numpy_state_after":ns[c,t+1].tolist(),
                   "cuda_diagnostics":dict(zip(NAMES,cd[c,t].astype(float).tolist())),
                   "numpy_diagnostics":dict(zip(NAMES,nd[c,t].tolist()))}
            stream.write(json.dumps(row,allow_nan=False)+"\n")
    edges = [0, .00025, .0005, .001, .0015, .002, .00225, .0025, .003, float("inf")]
    summary = {"utc":datetime.now(timezone.utc).isoformat(), "device":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
               "N":1024,"steps":48,"unique_inputs":len(cache),"replay_bitwise_equal":True,
               "strict_tolerance":{"rtol":1e-4,"atol":1e-6},
               "residual_002_count":int(selected.sum()),"max_abs_residual_difference":float(delta[:,:,11].max()),
               "residual_002_regime":dict(Counter(REGIMES[c%4] for c,t in np.argwhere(selected))),
               "residual_002_soil":dict(Counter(("sand","loam","clay")[c%3] for c,t in np.argwhere(selected))),
               "residual_002_unique_column_steps":len(set((int(c%156),int(t)) for c,t in np.argwhere(selected))),
               "histogram":{"edges_W_m2":[str(x) for x in edges],"counts":np.histogram(delta[:,:,11],edges)[0].tolist()},
               "clusters":clusters,"flags":flag_counts,"steps_histogram":step_counts,
               "source_sha256":{str(x.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(x.read_bytes()).hexdigest()
                                for x in sorted((ROOT/"src").rglob("*.py"))+sorted((ROOT/"src").rglob("*.cu"))+[Path(__file__)]},
               "input_sha256":hashlib.sha256(f.tobytes()+i.tobytes()+u.tobytes()).hexdigest()}
    args.prefix.with_suffix(".json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("source_sha256","clusters","flags","steps_histogram")}),flush=True)


if __name__ == "__main__":
    main()

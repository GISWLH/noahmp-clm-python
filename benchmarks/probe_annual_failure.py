"""CPU-only replay of the annual smoke's first failed energy solve.

Reads the saved daily input and samples the *unchanged* ordinary residual while
its closure still contains the frozen pre-update state. Returns the original
root, preserving the failing trajectory. No acceptance tolerances are changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from noahmp_clm.column import Forcing
from noahmp_clm.driver import step_column
from noahmp_clm.gpu.pack import column_state_to_vec, vec_to_column_state
from noahmp_clm.physics import energy, snow, surface_drag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/gpu-results/j4-annual-numpy-smoke.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    failed = next(row for row in records if row["event"] == "failure_input")
    day = next(row for row in records if row["event"] == "day" and not row["passed"])
    target_step = day["first_failure"]["coordinate"][1]
    state = vec_to_column_state(failed["initial_fvec"], failed["initial_ivec"], np.array([.1, .3, .6, 1.]))
    forcing = [Forcing(*values) for values in failed["forcing"]]
    original = energy.secant_solve
    capture = {}
    history = []
    for index, f in enumerate(forcing[:target_step + 1]):
        before, before_i = column_state_to_vec(state)

        def wrapper(func, x0, x1, *positional, **keywords):
            root = original(func, x0, x1, *positional, **keywords)
            if index != target_step:
                return root
            cells = dict(zip(func.__code__.co_freevars, (c.cell_contents for c in func.__closure__)))
            s = cells["s"]
            frozen_f, frozen_i = column_state_to_vec(s)
            grid = np.linspace(200., 330., 131)
            values = np.array([func(float(t)) for t in grid], float)
            extended = np.linspace(100., 200., 1001)
            extra_values = np.array([func(float(t)) for t in extended], float)
            crossings = np.flatnonzero(extra_values[:-1] * extra_values[1:] <= 0.)
            refined = []
            for crossing in crossings:
                lo, hi = float(extended[crossing]), float(extended[crossing+1])
                flo = func(lo)
                for _ in range(60):
                    mid = (lo+hi)/2
                    fm = func(mid)
                    if flo*fm <= 0.: hi = mid
                    else: lo, flo = mid, fm
                r = (lo+hi)/2
                refined.append({"root": r, "residual": float(func(r))})
            components = []
            for temperature in [100., 150., root, f.tair, 273.15, 330.]:
                rn = (1.-s.snow_albedo)*f.swdown + .99*(f.lwdown-energy.SIGMA*temperature**4)
                ce = surface_drag.exchange_velocity(f, s.z0m, temperature, s.options.OptSurfaceDrag)
                h = energy.air_density(f)*energy.CP*ce*(temperature-f.tair)
                g = (snow.coupled_snow_flux(temperature, cells["snow_response"])
                     if "snow_response" in cells else snow.surface_snow_flux(s, temperature))
                components.append({"temperature": temperature, "rnet": rn, "h": h,
                                   "g": g, "ce": ce, "F": float(func(temperature))})
            capture.update(root=float(root), initial_guesses=[x0,x1], root_residual=float(func(root)),
                           frozen_fvec=frozen_f.tolist(), frozen_ivec=frozen_i.tolist(),
                           grid_T=grid.tolist(), grid_F=values.tolist(),
                           min_abs_grid_F=float(np.min(np.abs(values))),
                           grid_crossings=int(np.count_nonzero(values[:-1]*values[1:] <= 0.)),
                           extended_T=extended.tolist(), extended_F=extra_values.tolist(),
                           roots_below_200=refined, components=components)
            return root

        energy.secant_solve = wrapper
        try:
            state, diag = step_column(state, f)
        finally:
            energy.secant_solve = original
        after, after_i = column_state_to_vec(state)
        history.append({"step": index, "before_fvec": before.tolist(), "before_ivec": before_i.tolist(),
                        "after_fvec": after.tolist(), "after_ivec": after_i.tolist(),
                        "forcing": vars(f), "diagnostic": {key:value.tolist() if isinstance(value,np.ndarray) else value
                                                           for key,value in diag.items()}})
    rootdir = Path(__file__).resolve().parents[1]
    files = [rootdir/"src/noahmp_clm/driver.py", Path(__file__).resolve()]
    files += sorted((rootdir/"src/noahmp_clm/physics").glob("*.py"))
    output = {"utc": datetime.now(timezone.utc).isoformat(), "backend":"numpy", "gpu_used":False,
              "python":platform.python_version(), "numpy":np.__version__, "platform":platform.platform(),
              "input_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "source_sha256":{str(path.relative_to(rootdir)).replace("\\","/"):hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in files},
              "day_index_zero_based":failed["day_index_zero_based"], "column":failed["column"],
              "step":target_step, "frozen_residual_probe":capture, "state_history":history}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output), "step":target_step,"root":capture["root"],
                      "root_residual":capture["root_residual"], "grid_crossings":capture["grid_crossings"],
                      "roots_below_200":capture["roots_below_200"],
                      "pre_snow_temperature":history[-1]["before_fvec"][19],
                      "prior_snow_temperature":history[-2]["before_fvec"][19],
                      "prior_g":history[-2]["diagnostic"]["g"]}))


if __name__ == "__main__":
    main()

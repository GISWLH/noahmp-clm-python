"""Continuous NumPy annual audit, with every state/forcing retained per day.

Uses the exact original synthetic eight-column forcing and existing 0.5 W/m2
energy gate. Stops on the first failing step without resetting any column.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from check_gpu_extended import (Backend, DIAGNOSTICS, DZ, daily_forcing,
                                digest_arrays, initial_columns, source_hashes)
from noahmp_clm.column import Forcing
from noahmp_clm.driver import step_column
from noahmp_clm.gpu.pack import column_state_to_vec, vec_to_column_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dumps = args.output.with_suffix('.states')
    dumps.mkdir()
    def emit(event, **values):
        record = dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **values)
        line = json.dumps(record, allow_nan=False)
        with args.output.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
        print(line, flush=True)
    backend = Backend.__new__(Backend)
    backend.xp = np
    backend.host = np.asarray
    backend.scalar = lambda x: np.asarray(x).item()
    f, i = initial_columns(8, np.float64)
    states = [vec_to_column_state(f[k], i[k], DZ) for k in range(8)]
    cumulative = np.zeros(8)
    maximum_residual = 0.
    minimum_snow_temperature = 273.15
    maximum_groundwater_depth = 0.
    hashes = source_hashes()
    hashes['benchmarks/check_annual_numpy.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    emit('start', backend='numpy64', requested_days=args.days, source_hashes=hashes,
         initial_state_sha256=digest_arrays(f,i), numpy_version=np.__version__,
         input_kind='original_J4_eight_synthetic_columns', continuous=True)
    started = time.monotonic()
    passed = True
    for day in range(args.days):
        forcing = daily_forcing(np.arange(8), day, np.float64, 20261013)
        initial_f, initial_i = f.copy(), i.copy()
        previous_storage = backend.storage(f)
        fs = np.empty((8,49,70))
        ints = np.empty((8,49,33), dtype=np.int32)
        ds = np.empty((8,48,12))
        fs[:,0], ints[:,0] = f,i
        failure = None
        for step in range(48):
            for column in range(8):
                states[column], diag = step_column(states[column], Forcing(*forcing[column,step]))
                f[column],i[column] = column_state_to_vec(states[column])
                ds[column,step] = [diag[key] for key in DIAGNOSTICS]
            fs[:,step+1], ints[:,step+1] = f,i
            metrics,_ = backend.metrics(f,i,ds[:,step:step+1],backend.storage(fs[:,step]))
            maximum_residual = max(maximum_residual, metrics['energy_abs_residual_max_W_m2'])
            for column in range(8):
                if f[column,26] > 1.e-8:
                    minimum_snow_temperature = min(minimum_snow_temperature, float(np.min(f[column,19:19+i[column,32]])))
            maximum_groundwater_depth = max(maximum_groundwater_depth,float(np.max(f[:,38])))
            if not metrics['passed']:
                failure = dict(day=day,step=step,**metrics['first_failure'])
                passed = False
                break
        count = step+1
        dump = dumps/f'day-{day:03d}.npz'
        np.savez_compressed(dump, f=fs[:,:count+1], i=ints[:,:count+1],
                            diagnostics=ds[:,:count], forcing=forcing[:,:count])
        metrics,water = backend.metrics(f,i,ds[:,:count],previous_storage)
        cumulative += water
        emit('day', day_index_zero_based=day, completed_steps=count,
             forcing_sha256=digest_arrays(forcing), state_dump=str(dump),
             dump_sha256=hashlib.sha256(dump.read_bytes()).hexdigest(),
             water_cumulative_abs_max_m=float(np.max(np.abs(cumulative))),
             first_step_failure=failure, **metrics)
        if failure:
            column = failure['coordinate'][0]
            fail_path = args.output.with_suffix('.failure.json')
            fail_path.write_text(json.dumps(dict(failure=failure,
                f=fs[column,count-1].tolist(),i=ints[column,count-1].tolist(),
                forcing=forcing[column,count-1].tolist(),
                final_f=f[column].tolist(),diagnostics=ds[column,count-1].tolist()),indent=2)+'\n')
            break
    emit('summary', status='passed' if passed else 'failed',
         requested_days=args.days, completed_days=day+1,
         continuous_steps_completed=day*48+count,
         energy_abs_residual_max_W_m2=maximum_residual,
         active_snow_temperature_min_K=minimum_snow_temperature,
         groundwater_depth_max_m=maximum_groundwater_depth,
         water_cumulative_abs_max_m=float(np.max(np.abs(cumulative))),
         all_intermediate_states_checked=True, wall_seconds=time.monotonic()-started)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

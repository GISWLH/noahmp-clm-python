"""K2 annual GPU audit with full small-batch host readback each day.

The production GPU integrator is unchanged.  All daily states and per-step
diagnostics are copied to host, then checked with the original extended gates.
This avoids hundreds of eager scalar GPU synchronizations per tiny daily batch;
it is a correctness audit and never used as a performance benchmark.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from check_gpu_extended import Backend, daily_forcing, digest_arrays, initial_columns, source_hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('jax64','jax32','cuda'),required=True)
    parser.add_argument('--days',type=int,default=365)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh output path')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    dumps=args.output.with_suffix('.states');dumps.mkdir()
    def emit(event,**values):
        line=json.dumps(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**values),allow_nan=False)
        with args.output.open('a',encoding='utf-8') as stream:stream.write(line+'\n')
        print(line,flush=True)
    dtype=np.dtype('float64' if args.backend=='jax64' else 'float32')
    hashes=source_hashes()
    hashes['benchmarks/check_annual_gpu.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    f0,i0=initial_columns(8,dtype)
    emit('start',backend=args.backend,requested_days=args.days,source_hashes=hashes,
         initial_state_sha256=digest_arrays(f0,i0),input_kind='original_J4_eight_synthetic_columns',
         continuous=True,scope='all_daily_states_and_all_half_hour_diagnostics',
         performance_result=False)
    gpu=Backend(args.backend)
    emit('backend',device=gpu.device,library_version=gpu.version,actual_cuda_required=True)
    checker=Backend.__new__(Backend);checker.xp=np;checker.host=np.asarray
    checker.scalar=lambda x:np.asarray(x).item()
    f,i=gpu.xp.asarray(f0),gpu.xp.asarray(i0)
    cumulative=np.zeros(8);absolute=np.zeros(8)
    maximum_residual=0.;minimum_snow_temperature=273.15
    started=time.monotonic();passed=True
    for day in range(args.days):
        forcing=daily_forcing(np.arange(8),day,dtype,20261013)
        result=gpu.run(f,i,gpu.xp.asarray(forcing));gpu.sync(result);gpu.assert_device(result)
        f,i,d=result
        hf,hi,hd=map(gpu.host,result)
        if hf.dtype!=dtype or hd.dtype!=dtype or hi.dtype!=np.int32:
            raise AssertionError('Unexpected GPU dtype')
        if not np.array_equal(hi[:,:32],i0[:,:32]):
            raise AssertionError('Immutable options changed')
        metrics,water=checker.metrics(hf,hi,hd,checker.storage(f0))
        cumulative+=water;absolute+=np.abs(water)
        maximum_residual=max(maximum_residual,metrics['energy_abs_residual_max_W_m2'])
        for column in range(8):
            if hf[column,26]>1.e-8:
                minimum_snow_temperature=min(minimum_snow_temperature,float(np.min(hf[column,19:19+hi[column,32]])))
        path=dumps/f'day-{day:03d}.npz'
        np.savez_compressed(path,initial_f=f0,initial_i=i0,f=hf,i=hi,diagnostics=hd,forcing=forcing)
        emit('day',day_index_zero_based=day,completed_days=day+1,
             forcing_sha256=digest_arrays(forcing),state_dump=str(path),
             dump_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             water_cumulative_abs_max_m=float(np.max(np.abs(cumulative))),
             water_sum_daily_absolute_error_max_m=float(np.max(absolute)),**metrics)
        f0,i0=hf,hi
        if not metrics['passed']:
            passed=False
            column=metrics['first_failure']['coordinate'][0]
            np.savez_compressed(args.output.with_suffix('.failure.npz'),f=hf[column],i=hi[column],
                                diagnostics=hd[column],forcing=forcing[column])
            break
    emit('summary',status='passed' if passed else 'failed',complete=day+1==args.days,
         requested_days=args.days,completed_days=day+1,continuous_steps_completed=(day+1)*48,
         energy_abs_residual_max_W_m2=maximum_residual,
         daily_active_snow_temperature_min_K=minimum_snow_temperature,
         water_cumulative_abs_max_m=float(np.max(np.abs(cumulative))),
         water_sum_daily_absolute_error_max_m=float(np.max(absolute)),
         full_daily_states_and_step_diagnostics_read_back=True,
         wall_seconds=time.monotonic()-started,performance_result=False)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())

"""Device-only plant C/N phases, allowing actual microbial N competition."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..plant_cn import validate_turnover,validate_plan,validate_allocation,_unpack


@lru_cache(maxsize=1)
def _module():
    import cupy as cp
    return cp.RawModule(code=Path(__file__).with_name('plant_cn_kernel.cu').read_text(encoding='utf-8'),
        options=('--std=c++14','--fmad=false','--prec-div=false'),
        name_expressions=('run_plant_turnover','run_plant_plan','run_plant_allocation'))


def _run(stage,a,b,p):
    import cupy as cp
    arrays=tuple(cp.ascontiguousarray(cp.asarray(v,cp.float32)) for v in (a,b,p))
    {'turnover':validate_turnover,'plan':validate_plan,'allocation':validate_allocation}[stage](*arrays,cp)
    n=len(arrays[0]);out=cp.empty((n,{'turnover':34,'plan':24,'allocation':38}[stage]),cp.float32)
    if n:_module().get_function('run_plant_'+stage)((min((n+127)//128,4096),),(128,),(*arrays,out,np.int32(n)))
    if stage=='plan':
        if not bool(cp.all(cp.isfinite(out))) or bool(cp.any(out<0.)):raise RuntimeError('Invalid CUDA plant demand plan')
        return out
    return _unpack(out,cp,stage)


def run_plant_turnover_cuda(state,inputs,parameters):return _run('turnover',state,inputs,parameters)


def plan_plant_allocation_cuda(state,inputs,parameters):return _run('plan',state,inputs,parameters)


def apply_plant_allocation_cuda(plan,supply,parameters):return _run('allocation',plan,supply,parameters)

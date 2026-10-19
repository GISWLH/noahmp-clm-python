"""CUDA float32 closed-column liquid Richards development component."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..richards import _run


@lru_cache(maxsize=1)
def _module():
    import cupy as cp
    source=Path(__file__).with_name('richards_kernel.cu').read_text()
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'),
        name_expressions=('richards_implicit_step','richards_controller_reciprocal'))
    return module


@lru_cache(maxsize=1)
def _kernel():return _module().get_function('richards_implicit_step')


def _controller_reciprocal(value):
    import cupy as cp
    if value.dtype!=cp.float32 or value.ndim!=1 or not 1<=value.size<=10:
        raise ValueError('Richards controller requires 1--10 float32 values')
    value=cp.ascontiguousarray(value);result=cp.empty_like(value)
    _module().get_function('richards_controller_reciprocal')((1,),(32,),(value,result,np.int32(value.size)))
    return result


def _step(old,guess,p,dt):
    import cupy as cp
    old,guess,p=[cp.ascontiguousarray(a) for a in (old,guess,p)];n=len(old)
    result=cp.empty(3*n+2,cp.float32)
    _kernel()((1,),(1,),(old,guess,p,np.float32(dt),result,np.int32(n)))
    return result


def run_richards_cuda(water,hydraulics,dt,**controls):
    import cupy as cp
    water,hydraulics,dt=[cp.asarray(a,cp.float32) for a in (water,hydraulics,dt)]
    return _run(water,hydraulics,dt,cp,_step,**controls)

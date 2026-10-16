"""CUDA tile drainage demand and actual native soil mass/heat transfer."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..tile_drainage import validate_tile_drainage,unpack_tile_drainage,_step_soil


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawModule(code=Path(__file__).with_name('tile_drainage_kernel.cu').read_text(encoding='utf-8'),
        options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_tile_drainage',)).get_function('run_tile_drainage')


def run_tile_drainage_cuda(layers,dt,parameters):
    import cupy as cp
    arrays=tuple(cp.ascontiguousarray(cp.asarray(v,cp.float32)) for v in (layers,dt,parameters))
    validate_tile_drainage(*arrays,cp);n=len(arrays[0]);out=cp.empty((n,10),cp.float32)
    if n:_kernel()((min((n+127)//128,4096),),(128,),(*arrays,out,np.int32(n)))
    return unpack_tile_drainage(out,cp)


def step_tile_soil_cuda(c,p,o,dt,parameters):
    return _step_soil(c,p,o,dt,parameters,'cuda')

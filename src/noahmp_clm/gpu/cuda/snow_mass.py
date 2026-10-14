"""CUDA extensive snow transfers; host handles only validation and packing."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..snow_mass import validate_snow_transfer,unpack_snow_transfer


@lru_cache(maxsize=1)
def _module():
    import cupy as cp
    return cp.RawModule(code=Path(__file__).with_name('snow_mass_kernel.cu').read_text(),
        options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_snow_transfer',))


def run_snow_transfer_cuda(cells,counts,modes,inputs):
    import cupy as cp
    raw=tuple(map(cp.asarray,(counts,modes)))
    if any(v.dtype.kind not in 'iu' for v in raw):raise ValueError('Snow counts and modes must be integers')
    c,u=[cp.ascontiguousarray(cp.asarray(v,dtype=cp.float32)) for v in (cells,inputs)]
    n,m=[cp.ascontiguousarray(v,dtype=cp.int32) for v in raw]
    validate_snow_transfer(c,n,m,u,cp);out=cp.empty((len(c),74),cp.float32)
    if len(c):_module().get_function('run_snow_transfer')((min((len(c)+63)//64,4096),),(64,),
        (c,n,m,u,out,np.int32(len(c))))
    if not bool(cp.all(cp.isfinite(out))):raise RuntimeError('Nonfinite snow mass transfer')
    return unpack_snow_transfer(out)

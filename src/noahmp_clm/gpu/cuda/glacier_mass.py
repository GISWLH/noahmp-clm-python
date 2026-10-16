"""Native CUDA exposed-ice mass transfers, with explicit heat and impurities."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..glacier_mass import validate_glacier_transfer,unpack_glacier_transfer


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    m=cp.RawModule(code=Path(__file__).with_name('glacier_mass_kernel.cu').read_text(),
        options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_glacier_transfer',))
    return m.get_function('run_glacier_transfer')


def run_glacier_transfer_cuda(g,n,mode,u):
    import cupy as cp
    raw=tuple(map(cp.asarray,(n,mode)))
    if any(v.dtype.kind not in 'iu' for v in raw):raise ValueError('Glacier counts/modes must be integers')
    g,u=[cp.ascontiguousarray(cp.asarray(v,dtype=cp.float32)) for v in (g,u)]
    n,mode=[cp.ascontiguousarray(v,dtype=cp.int32) for v in raw];validate_glacier_transfer(g,n,mode,u,cp)
    out=cp.empty((len(g),39),cp.float32)
    if len(g):_kernel()((min((len(g)+63)//64,4096),),(64,),(g,n,mode,u,out,np.int32(len(g))))
    if not bool(cp.all(cp.isfinite(out))):raise RuntimeError('Nonfinite CUDA glacier mass transfer')
    return unpack_glacier_transfer(out)

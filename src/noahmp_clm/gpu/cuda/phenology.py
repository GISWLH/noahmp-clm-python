"""Native CUDA seasonal phenology with integer phase/arming flags."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..phenology import validate_phenology,unpack_phenology


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    module=cp.RawModule(code=Path(__file__).with_name('phenology_kernel.cu').read_text(encoding='utf-8'),
        options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_phenology',))
    return module.get_function('run_phenology')


def run_phenology_cuda(s,o,u,c,p):
    import cupy as cp
    if cp.asarray(o).dtype.kind not in 'iu' or cp.asarray(c).dtype.kind not in 'iu':raise ValueError('Phenology flags must be integers')
    s,u,p=[cp.ascontiguousarray(cp.asarray(v,cp.float32)) for v in (s,u,p)]
    o,c=[cp.ascontiguousarray(cp.asarray(v,cp.int32)) for v in (o,c)]
    validate_phenology(s,o,u,c,p,cp);n=len(s);state=cp.empty_like(s);flags=cp.empty_like(o);diagnostics=cp.empty((n,6),cp.float32)
    if n:_kernel()((min((n+127)//128,4096),),(128,),(s,o,u,c,p,state,flags,diagnostics,np.int32(n)))
    return unpack_phenology(state,flags,diagnostics)

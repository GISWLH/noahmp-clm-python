"""CUDA canopy structure, with explicit C3 crop and snow burial geometry."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..vegetation_structure import validate_structure,unpack_structure


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawKernel(Path(__file__).with_name('vegetation_structure_kernel.cu').read_text(encoding='utf-8'),
        'structure_columns',options=('--std=c++14','--fmad=false','--prec-div=false'))


def run_structure_cuda(c,p,o):
    import cupy as cp
    o=cp.asarray(o)
    if o.dtype.kind not in 'iu':raise ValueError('Structure flags must be integers')
    c,p=[cp.ascontiguousarray(cp.asarray(x,dtype=cp.float32)) for x in (c,p)]
    o=cp.ascontiguousarray(o,dtype=cp.int32);validate_structure(c,p,o,cp)
    n=len(c);out=cp.empty((n,14),cp.float32)
    if n:_kernel()((min((n+127)//128,4096),),(128,),(c,p,o,out,np.int32(n)))
    if not bool(cp.all(cp.isfinite(out))):raise RuntimeError('Nonfinite CUDA canopy structure')
    return unpack_structure(out)

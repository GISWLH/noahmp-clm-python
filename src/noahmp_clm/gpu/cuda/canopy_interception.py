"""Native CUDA canopy interception/phase-specific vapour transfer."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..canopy_interception import validate_canopy_water,unpack_canopy_water


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    source=Path(__file__).with_name('canopy_interception_kernel.cu').read_text(encoding='utf-8')
    return cp.RawKernel(source,'canopy_water_columns',options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'))


def run_canopy_water_cuda(c,area,u,v,mode):
    import cupy as cp
    mode=cp.asarray(mode)
    if mode.dtype.kind not in 'iu':raise ValueError('Canopy modes must be integers')
    c,area,u,v=[cp.ascontiguousarray(cp.asarray(x,dtype=cp.float32)) for x in (c,area,u,v)]
    mode=cp.ascontiguousarray(mode,dtype=cp.int32);validate_canopy_water(c,area,u,v,mode,cp)
    n=len(c);out=cp.empty((n,182),cp.float32)
    if n:_kernel()((min((n+127)//128,4096),),(128,),(c,area,u,v,mode,out,np.int32(n)))
    if not bool(cp.all(cp.isfinite(out))):raise RuntimeError('Nonfinite CUDA canopy water output')
    return unpack_canopy_water(out)

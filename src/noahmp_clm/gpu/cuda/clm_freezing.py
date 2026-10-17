"""CUDA32 explicit-suction freezing, no host physics in the device path."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..clm_freezing import validate_freezing,unpack_freezing

@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawKernel(Path(__file__).with_name('clm_freezing_kernel.cu').read_text(),
        'run_clm_freezing',options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'))

def run_freezing_cuda(inputs):
    import cupy as cp
    a=cp.ascontiguousarray(cp.asarray(inputs,dtype=cp.float32));validate_freezing(a,cp)
    n=len(a);out=cp.empty((n,3),cp.float32)
    if n:_kernel()((min((n+127)//128,4096),),(128,),(a,out,np.int32(n)))
    return unpack_freezing(out)

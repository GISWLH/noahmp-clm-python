"""Native CUDA wet-bulb equation, independent column per thread."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..wetbulb import validate_wetbulb


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    source='\n'.join(Path(__file__).with_name(name).read_text(encoding='utf-8') for name in ('column_kernel.cu','wetbulb_kernel.cu'))
    return cp.RawKernel(source,
        'run_wetbulb',options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'))


def run_wetbulb_cuda(atmosphere):
    import cupy as cp
    a=cp.ascontiguousarray(cp.asarray(atmosphere,dtype=cp.float32));validate_wetbulb(a,cp)
    n=len(a);out=cp.empty((n,2),cp.float32)
    if n:_kernel()((min((n+127)//128,4096),),(128,),(a,out,np.int32(n)))
    return dict(temperature=out[:,0],snow_fraction=out[:,1])

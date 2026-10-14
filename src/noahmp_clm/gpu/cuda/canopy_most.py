"""Native CUDA profile coefficients and actual-flux stability update."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..canopy_most import validate_most,MOST_OUTPUTS


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawKernel(Path(__file__).with_name('canopy_most_kernel.cu').read_text(),
        'run_canopy_most',options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'))


def run_canopy_most_cuda(geometry,atmosphere,iteration):
    import cupy as cp
    arrays=tuple(cp.ascontiguousarray(cp.asarray(v,dtype=cp.float32)) for v in (geometry,atmosphere,iteration))
    validate_most(*arrays,cp);n=len(arrays[0]);out=cp.empty((n,15),cp.float32)
    if n:_kernel()((min((n+127)//128,4096),),(128,),(*arrays,out,np.int32(n)))
    return {name:out[:,k] for k,name in enumerate(MOST_OUTPUTS)}

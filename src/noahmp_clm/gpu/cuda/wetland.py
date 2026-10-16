"""Device methane C partition and conservative pool/flux integration."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..wetland import validate_wetland,unpack_wetland


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawModule(code=Path(__file__).with_name('wetland_kernel.cu').read_text(encoding='utf-8'),
        options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_wetland',)).get_function('run_wetland')


def run_wetland_cuda(state,inputs,parameters):
    import cupy as cp
    arrays=tuple(cp.ascontiguousarray(cp.asarray(v,cp.float32)) for v in (state,inputs,parameters))
    validate_wetland(*arrays,cp);n=len(arrays[0]);out=cp.empty((n,18),cp.float32)
    if n:_kernel()((min((n+127)//128,4096),),(128,),(*arrays,out,np.int32(n)))
    return unpack_wetland(out,cp)

"""CUDA soil/litter C-N cascade, including mineral-N competition."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..decomposition import validate_decomposition,unpack_decomposition


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    module=cp.RawModule(code=Path(__file__).with_name('decomposition_kernel.cu').read_text(encoding='utf-8'),
        options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_decomposition',))
    return module.get_function('run_decomposition')


def run_decomposition_cuda(state,inputs,parameters):
    import cupy as cp
    arrays=tuple(cp.ascontiguousarray(cp.asarray(v,cp.float32)) for v in (state,inputs,parameters))
    validate_decomposition(*arrays,cp);n=len(arrays[0]);out=cp.empty((n,70),cp.float32)
    if n:_kernel()((min((n+127)//128,4096),),(128,),(*arrays,out,np.int32(n)))
    return unpack_decomposition(out,cp)

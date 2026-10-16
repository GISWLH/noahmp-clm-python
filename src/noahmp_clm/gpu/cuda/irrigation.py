"""Native CUDA irrigation requests and externally supplied liquid enthalpy."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..irrigation import validate_irrigation,unpack_irrigation


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawModule(code=Path(__file__).with_name('irrigation_kernel.cu').read_text(encoding='utf-8'),
        options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_irrigation',)).get_function('run_irrigation')


def run_irrigation_cuda(state,layers,climate,flags,parameters):
    import cupy as cp
    if cp.asarray(flags).dtype.kind not in 'iu':raise ValueError('Irrigation checks must be integer flags')
    arrays=tuple(cp.ascontiguousarray(cp.asarray(v,cp.int32 if k==3 else cp.float32)) for k,v in enumerate((state,layers,climate,flags,parameters)))
    validate_irrigation(*arrays,cp);n=len(arrays[0]);out=cp.empty((n,16),cp.float32)
    if n:_kernel()((min((n+127)//128,4096),),(128,),(*arrays,out,np.int32(n)))
    return unpack_irrigation(out,cp)

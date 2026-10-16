"""Native CUDA crop events; host work is shape/domain validation only."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..crop import validate_crop,unpack_crop


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawModule(code=Path(__file__).with_name('crop_kernel.cu').read_text(encoding='utf-8'),
        options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_crop',)).get_function('run_crop')


def run_crop_cuda(state,phase,plant,climate,events,parameters,plant_parameters):
    import cupy as cp
    raw=(state,phase,plant,climate,events,parameters,plant_parameters)
    if any(cp.asarray(raw[k]).dtype.kind not in 'iu' for k in (1,4)):
        raise ValueError('Crop phase and management events must be integer arrays')
    arrays=tuple(cp.ascontiguousarray(cp.asarray(v,cp.int32 if k in (1,4) else cp.float32)) for k,v in enumerate(raw))
    validate_crop(*arrays,cp);n=len(arrays[0]);out=cp.empty((n,43),cp.float32)
    if n:_kernel()((min((n+127)//128,4096),),(128,),(*arrays,out,np.int32(n)))
    return unpack_crop(out,cp)

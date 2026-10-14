"""Native CUDA mixed-phase snow vapor and frost/dew deposition."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..snow_vapor import validate_snow_vapor,unpack_snow_vapor


@lru_cache(maxsize=1)
def _module():
    import cupy as cp
    here=Path(__file__).parent
    source='\n'.join((here/name).read_text() for name in ('snow_mass_kernel.cu','snow_vapor_kernel.cu'))
    return cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_snow_vapor',))


def run_snow_vapor_cuda(cells,counts,inputs):
    import cupy as cp
    raw=cp.asarray(counts)
    if raw.dtype.kind not in 'iu':raise ValueError('Snow counts must be integers')
    c,u=[cp.ascontiguousarray(cp.asarray(v,dtype=cp.float32)) for v in (cells,inputs)]
    n=cp.ascontiguousarray(raw,dtype=cp.int32);validate_snow_vapor(c,n,u,cp)
    out=cp.empty((len(c),56),cp.float32)
    if len(c):_module().get_function('run_snow_vapor')((min((len(c)+63)//64,4096),),(64,),
        (c,n,u,out,np.int32(len(c))))
    if not bool(cp.all(cp.isfinite(out))):raise RuntimeError('Nonfinite snow vapor exchange')
    return unpack_snow_vapor(out)

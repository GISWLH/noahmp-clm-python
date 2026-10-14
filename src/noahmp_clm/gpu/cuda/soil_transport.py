"""CUDA soil liquid transfer with actual extensive heat ledgers."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..soil_transport import validate_soil_transport,unpack_soil_transport


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    source='\n'.join(Path(__file__).with_name(p).read_text() for p in ('enthalpy_kernel.cu','soil_transport_kernel.cu'))
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_soil_transport',))
    return module.get_function('run_soil_transport')


def run_soil_transport_cuda(c,p,o,u):
    import cupy as cp
    raw=cp.asarray(o)
    if raw.dtype.kind not in 'iu':raise ValueError('Soil options must be integers')
    c,p,u=[cp.ascontiguousarray(cp.asarray(v,dtype=cp.float32)) for v in (c,p,u)]
    o=cp.ascontiguousarray(raw,dtype=cp.int32);validate_soil_transport(c,p,o,u,cp)
    n=len(c);out=cp.empty((n,47),cp.float32)
    if n:_kernel()((min((n+63)//64,4096),),(64,),(c,p,o,u,out,np.int32(n)))
    r=unpack_soil_transport(out)
    if not bool(cp.all(r['valid'])) or not bool(cp.all(cp.isfinite(out))):
        raise ValueError('Soil transfer exceeds available liquid or thermodynamic physical domain')
    return r

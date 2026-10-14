"""CUDA VIC transfers with native water/enthalpy state updates."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..vic import validate_vic
from ..soil_transport import validate_soil_transport,unpack_soil_transport


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    source='\n'.join(Path(__file__).with_name(n).read_text(encoding='utf-8') for n in
        ('enthalpy_kernel.cu','soil_transport_kernel.cu'))
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_vic_transport',))
    return module.get_function('run_vic_transport')


def run_vic_transport_cuda(c,p,o,u,v):
    import cupy as cp
    if cp.asarray(o).dtype.kind not in 'iu':raise ValueError('Soil options must be integers')
    c,p,u,v=[cp.ascontiguousarray(cp.asarray(a,cp.float32)) for a in (c,p,u,v)]
    o=cp.ascontiguousarray(cp.asarray(o,cp.int32));validate_soil_transport(c,p,o,u,cp);validate_vic(v,len(c),cp)
    n=len(c);out=cp.empty((n,47),cp.float32)
    if n:_kernel()((min((n+63)//64,4096),),(64,),(c,p,o,u,v,out,np.int32(n)))
    result=unpack_soil_transport(out)
    if not bool(cp.all(result['valid'])) or not bool(cp.all(cp.isfinite(out))):
        raise ValueError('VIC transfer exceeds the thermodynamic physical domain')
    return result

"""CUDA finite glacier mass/enthalpy column component; no host tendencies."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..glacier import validate_glacier,unpack_glacier


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    names=('enthalpy_kernel.cu','column_heat_kernel.cu','glacier_heat_kernel.cu','snow_mass_kernel.cu','glacier_kernel.cu')
    source='\n'.join(Path(__file__).with_name(p).read_text() for p in names)
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_glacier',))
    return module.get_function('run_glacier')


def run_glacier_cuda(c,g,count,b,p,q):
    import cupy as cp
    raw=cp.asarray(count)
    if raw.dtype.kind not in 'iu':raise ValueError('Glacier counts must be integers')
    c,g,b,p,q=[cp.ascontiguousarray(cp.asarray(v,dtype=cp.float32)) for v in (c,g,b,p,q)]
    count=cp.ascontiguousarray(raw,dtype=cp.int32);validate_glacier(c,g,count,b,p,q,cp)
    result=cp.empty((len(c),167),dtype=cp.float32)
    if len(c):_kernel()((min((len(c)+63)//64,4096),),(64,),(c,g,count,b,p,q,result,np.int32(len(c))))
    heat=result[:,90:];thermal=tuple(heat[:,i:i+9] for i in range(0,54,9))+(heat[:,54:62],heat[:,62:71],heat[:,71:])
    out=unpack_glacier((result[:,:45].reshape(-1,5,9),result[:,45:73].reshape(-1,4,7),
        result[:,73:75].astype(cp.int32),result[:,75:90],thermal))
    if not bool(cp.all(cp.isfinite(result))):raise RuntimeError('Nonfinite CUDA glacier output')
    if not bool(cp.all(out['thermal']['converged'])):raise RuntimeError('CUDA glacier heat failed to converge')
    if bool(cp.any((cp.abs(out['water_residual_kgm2'])>out['water_bound_kgm2'])|
        (cp.abs(out['energy_residual_jm2'])>out['energy_bound_jm2']))):raise RuntimeError('CUDA glacier conservation failed')
    return out

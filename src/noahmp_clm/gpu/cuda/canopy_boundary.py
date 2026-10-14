"""Native CUDA fixed surface canopy energy and liquid storage boundary."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..canopy_boundary import validate_boundary,unpack_boundary,radiation_inputs
from ..photosynthesis import pack_leaf_parameters
from .canopy_snow_radiation import _kernel as _radiation_kernel
from .snicar import _tables


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    source='\n'.join(Path(__file__).with_name(p).read_text() for p in
        ('leaf_kernel.cu','radiation_kernel.cu','canopy_energy_kernel.cu'))
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'),
        name_expressions=('run_canopy_boundary',))
    return module.get_function('run_canopy_boundary')


def run_canopy_boundary_cuda(area,u,f,soil,initial,b,c,n,parameters=None,*,max_iterations=100,
                             tolerance=.005,raise_on_failure=True):
    import cupy as cp
    raw=cp.asarray(n)
    if raw.dtype.kind not in 'iu':raise ValueError('Snow counts must be integers')
    arrays=tuple(cp.ascontiguousarray(cp.asarray(v,dtype=cp.float32)) for v in (area,u,f,soil,initial,b,c))
    n=cp.ascontiguousarray(raw,dtype=cp.int32);validate_boundary((*arrays,n),cp)
    if not isinstance(max_iterations,int) or max_iterations<1 or not np.isfinite(tolerance) or tolerance<=0.:
        raise ValueError('Invalid canopy boundary solver controls')
    a,u,f,soil,initial,b,c=arrays
    p=cp.asarray(pack_leaf_parameters(parameters,np.float32))
    size=len(a);out=cp.empty((size,208),cp.float32)
    if size:
        launch=((min((size+63)//64,4096),),(64,))
        ru=cp.ascontiguousarray(radiation_inputs(a,u,cp));radiation=cp.empty((size,584),cp.float32)
        _radiation_kernel(True)(*launch,(a,c,n,ru,*_tables(),radiation,np.int32(size)))
        _kernel()(*launch,(a,u,f,soil,initial,b,radiation,p,out,np.int32(size),np.int32(max_iterations),
            np.float32(tolerance),np.float32(1e-8)))
    result=unpack_boundary(out)
    if not bool(cp.all(cp.isfinite(out))):raise RuntimeError('Nonfinite canopy boundary result')
    if raise_on_failure and not bool(cp.all(result['converged'])):
        raise RuntimeError('CUDA canopy boundary did not converge; inspect with raise_on_failure=False')
    return result

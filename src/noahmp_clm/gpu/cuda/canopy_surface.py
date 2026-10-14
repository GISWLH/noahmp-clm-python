"""Native CUDA nonlinear winter canopy/snow/soil surface time step."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..canopy_surface import validate_surface,unpack_surface
from ..canopy_boundary import radiation_inputs
from ..photosynthesis import pack_leaf_parameters
from .canopy_snow_radiation import _kernel as _radiation_kernel
from .snicar import _tables
from .snow_evolution import _table as _aging


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    names=('leaf_kernel.cu','radiation_kernel.cu','canopy_energy_kernel.cu','enthalpy_kernel.cu',
        'column_heat_kernel.cu','snow_mass_kernel.cu','snow_vapor_kernel.cu','snow_evolution_kernel.cu',
        'soil_transport_kernel.cu','canopy_surface_kernel.cu')
    source='\n'.join(Path(__file__).with_name(name).read_text() for name in names)
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'),
        name_expressions=('run_canopy_surface',))
    return module.get_function('run_canopy_surface')


def run_canopy_surface_cuda(c,g,p,o,a,u,f,v,parameters=None,*,raise_on_failure=True):
    import cupy as cp
    raw=cp.asarray(o)
    if raw.dtype.kind not in 'iu':raise ValueError('Surface options must be integers')
    c,g,p,a,u,f,v=tuple(cp.ascontiguousarray(cp.asarray(x,dtype=cp.float32)) for x in (c,g,p,a,u,f,v))
    o=cp.ascontiguousarray(raw,dtype=cp.int32);arrays=(c,g,p,o,a,u,f,v);validate_surface(arrays,cp)
    size=len(c);out=cp.empty((size,441),cp.float32)
    if size:
        launch=((min((size+31)//32,4096),),(32,))
        optical=cp.ascontiguousarray(c[:,:,cp.array([0,3,4,5,6,7,8])]);counts=cp.ascontiguousarray(o[:,0])
        ru=cp.ascontiguousarray(radiation_inputs(a,u,cp));radiation=cp.empty((size,584),cp.float32)
        _radiation_kernel(True)(*launch,(a,optical,counts,ru,*_tables(),radiation,np.int32(size)))
        _kernel()(*launch,(*arrays,radiation,cp.asarray(pack_leaf_parameters(parameters,np.float32)),_aging(),out,np.int32(size)))
    result=unpack_surface(out)
    if raise_on_failure and (not bool(cp.all(cp.isfinite(out))) or not bool(cp.all(result['converged']))):
        raise RuntimeError('CUDA winter surface failed; inspect with raise_on_failure=False')
    return result

"""CUDA layered canopy phase and joint snow/soil extensive surface."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..canopy_phase_surface import validate_phase_surface,unpack_phase_surface,validate_exchange_extension
from ..canopy_boundary import radiation_inputs
from ..photosynthesis import pack_leaf_parameters
from .canopy_snow_radiation import _kernel as _radiation_kernel
from .snicar import _tables
from .snow_evolution import _table as _aging


@lru_cache(maxsize=8)
def _kernel(extended=False,vic=False,stems=False):
    import cupy as cp
    names=('leaf_kernel.cu','radiation_kernel.cu','canopy_energy_kernel.cu','canopy_phase_energy_kernel.cu',
        'enthalpy_kernel.cu','column_heat_kernel.cu','snow_mass_kernel.cu','snow_vapor_kernel.cu',
        'snow_evolution_kernel.cu','soil_transport_kernel.cu','canopy_surface_kernel.cu','canopy_phase_surface_kernel.cu')
    source='\n'.join(Path(__file__).with_name(name).read_text() for name in names)
    name='run_canopy_phase_surface_stems' if stems else ('run_canopy_phase_surface_vic' if vic else ('run_canopy_phase_surface_extended' if extended else 'run_canopy_phase_surface'))
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'),
        name_expressions=(name,))
    return module.get_function(name)


def run_canopy_phase_surface_cuda(*arrays,parameters=None,raise_on_failure=True,exchange_extension=None,vic=None,stem_area=None,canopy_optics=None):
    import cupy as cp
    if cp.asarray(arrays[3]).dtype.kind not in 'iu':raise ValueError('Surface options must be integers')
    arrays=tuple(cp.ascontiguousarray(cp.asarray(a,dtype=cp.int32 if k==3 else cp.float32)) for k,a in enumerate(arrays))
    stem=None if stem_area is None else cp.ascontiguousarray(cp.asarray(stem_area,cp.float32))
    validate_phase_surface(arrays,cp,stem_area=stem);c,g,p,o,a,u,f,v,condensate=arrays;size=len(c)
    extra=() if exchange_extension is None else (cp.ascontiguousarray(cp.asarray(exchange_extension,cp.float32)),)
    if extra:validate_exchange_extension(extra[0],size,cp)
    if vic is not None:
        from ..vic import validate_vic
        vic=cp.ascontiguousarray(cp.asarray(vic,cp.float32));validate_vic(vic,size,cp)
    arguments=(extra[0] if extra else np.uint64(0),vic) if vic is not None else extra
    if stem is not None:arguments=(stem,extra[0] if extra else np.uint64(0),np.uint64(0) if vic is None else vic)
    out=cp.empty((size,741),cp.float32)
    optical_result=None
    if canopy_optics is not None:
        from ..canopy_optics import validate_canopy_optics
        canopy_optics=cp.ascontiguousarray(cp.asarray(canopy_optics,cp.float32))
        validate_canopy_optics(a if stem is None else a+stem,a,canopy_optics,cp)
    if size:
        launch=((min((size+31)//32,4096),),(32,))
        optical=cp.ascontiguousarray(c[:,:,cp.array([0,3,4,5,6,7,8])]);counts=cp.ascontiguousarray(o[:,0])
        radiation=cp.empty((size,584),cp.float32)
        plant=a if stem is None else cp.ascontiguousarray(a+stem)
        if canopy_optics is None:
            _radiation_kernel(True)(*launch,(plant,optical,counts,cp.ascontiguousarray(radiation_inputs(plant,u,cp)),*_tables(),radiation,np.int32(size)))
        else:
            from .canopy_optics import canopy_snow_optics_raw_cuda
            optical_result=canopy_snow_optics_raw_cuda(plant,optical,counts,radiation_inputs(plant,u,cp),a,canopy_optics)
            radiation=cp.ascontiguousarray(optical_result[:,:584])
        _kernel(bool(extra),vic is not None,stem is not None)(*launch,(*arrays,*arguments,radiation,cp.asarray(pack_leaf_parameters(parameters,np.float32)),_aging(),out,np.int32(size)))
    result=unpack_phase_surface(out)
    if optical_result is not None:
        from ..canopy_optics import attach_optical_absorption
        attach_optical_absorption(result,optical_result)
    if raise_on_failure and (not bool(cp.all(cp.isfinite(out))) or not bool(cp.all(result['converged']))):
        raise RuntimeError('CUDA layered phase surface failed; inspect with raise_on_failure=False')
    return result

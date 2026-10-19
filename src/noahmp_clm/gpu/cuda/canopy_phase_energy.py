"""CUDA layer phase enthalpy and air/leaf energy, without host physics."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..canopy_phase_energy import validate_phase,unpack_phase,PHASE_OUTPUT_SIZE
from ..canopy_boundary import radiation_inputs
from ..photosynthesis import prepare_canopy_leaf_parameters
from .canopy_snow_radiation import _kernel as _radiation_kernel
from .snicar import _tables


@lru_cache(maxsize=2)
def _kernel(profiled=False):
    import cupy as cp
    source='\n'.join(Path(__file__).with_name(p).read_text() for p in
        ('leaf_kernel.cu','radiation_kernel.cu','canopy_energy_kernel.cu','canopy_phase_energy_kernel.cu'))
    if profiled:source='#define CANOPY_LAYER_PARAMETERS\n'+source
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'),name_expressions=('run_canopy_phase',))
    return module.get_function('run_canopy_phase')


def run_canopy_phase_cuda(*arrays,parameters=None,max_iterations=100,tolerance=.005,raise_on_failure=True,canopy_optics=None,return_raw=False):
    import cupy as cp
    raw=cp.asarray(arrays[-1])
    if raw.dtype.kind not in 'iu':raise ValueError('Snow counts must be integers')
    values=tuple(cp.ascontiguousarray(cp.asarray(v,dtype=cp.float32)) for v in arrays[:-1])+(cp.ascontiguousarray(raw,dtype=cp.int32),)
    validate_phase(values,cp);a,u,f,soil,stem,c,b,snow,n=values
    if not isinstance(max_iterations,int) or max_iterations<1 or not np.isfinite(tolerance) or tolerance<=0.:raise ValueError('Invalid phase solver controls')
    size=len(a);out=cp.empty((size,PHASE_OUTPUT_SIZE),cp.float32)
    optical_result=None
    if canopy_optics is not None:
        from ..canopy_optics import validate_canopy_optics
        canopy_optics=cp.ascontiguousarray(cp.asarray(canopy_optics,cp.float32));validate_canopy_optics(a+stem,a,canopy_optics,cp)
    if size:
        launch=((min((size+31)//32,4096),),(32,))
        area=cp.ascontiguousarray(a+stem);ru=cp.ascontiguousarray(radiation_inputs(area,u,cp));rad=cp.empty((size,584),cp.float32)
        if canopy_optics is None:_radiation_kernel(True)(*launch,(area,snow,n,ru,*_tables(),rad,np.int32(size)))
        else:
            from .canopy_optics import canopy_snow_optics_raw_cuda
            optical_result=canopy_snow_optics_raw_cuda(area,snow,n,ru,a,canopy_optics)
            rad=cp.ascontiguousarray(optical_result[:,:584])
        p,profiled=prepare_canopy_leaf_parameters(parameters,cp,size,cp.float32);p=cp.ascontiguousarray(p)
        _kernel(profiled)(*launch,(a,u,f,soil,stem,c,b,rad,p,out,np.int32(size),np.int32(max_iterations),np.float32(tolerance)))
    if not bool(cp.all(cp.isfinite(out))):raise RuntimeError('Nonfinite phase canopy result')
    result=unpack_phase(out)
    if return_raw:result['raw']=out
    if optical_result is not None:
        from ..canopy_optics import attach_optical_absorption
        attach_optical_absorption(result,optical_result)
    if raise_on_failure and not bool(cp.all(result['converged'])):raise RuntimeError('CUDA phase canopy did not converge; inspect with raise_on_failure=False')
    return result

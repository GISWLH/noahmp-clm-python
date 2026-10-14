"""Native CUDA primitives for the developing extensive CLM state."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..enthalpy import validate_soil_thermodynamics,SOIL_THERMODYNAMIC_OUTPUTS


@lru_cache(maxsize=1)
def _soil_kernel():
    import cupy as cp
    module=cp.RawModule(code=Path(__file__).with_name('enthalpy_kernel.cu').read_text(),
        options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_soil_thermodynamics',))
    return module.get_function('run_soil_thermodynamics')


def run_soil_thermodynamics_cuda(inputs,options,*,inverse=False):
    import cupy as cp
    # Validate integer option input before conversion; fractional options must
    # not silently truncate into a supported scheme.
    raw=cp.asarray(options)
    if raw.dtype.kind not in 'iu': raise ValueError('Freezing options must be integers')
    u=cp.ascontiguousarray(cp.asarray(inputs,dtype=cp.float32));o=cp.ascontiguousarray(raw,dtype=cp.int32)
    validate_soil_thermodynamics(u,o,inverse,cp)
    result=cp.empty((len(u),9),dtype=cp.float32)
    if len(u):
        _soil_kernel()((min((len(u)+127)//128,4096),),(128,),
            (u,o,result,np.int32(inverse),np.int32(len(u))))
    if not bool(cp.all(result[:,-1]>0.)):
        raise ValueError('Soil enthalpy is outside the physical domain or inversion did not converge')
    return dict(zip(SOIL_THERMODYNAMIC_OUTPUTS,result.T))

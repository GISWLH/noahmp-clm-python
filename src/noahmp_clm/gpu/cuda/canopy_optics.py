"""CUDA two-tissue shortwave optics without host physical calculations."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..canopy_optics import validate_canopy_optics,unpack_canopy_snow_optics
from ..canopy_snow_radiation import validate_canopy_snow
from .snicar import _tables


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    source='\n'.join(Path(__file__).with_name(n).read_text() for n in
                     ('radiation_kernel.cu','snicar_kernel.cu','canopy_snow_kernel.cu'))
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false'),
                        name_expressions=('run_canopy_snow_optics',))
    return module.get_function('run_canopy_snow_optics')


def canopy_snow_optics_raw_cuda(area,cells,count,inputs,leaf,parameters):
    import cupy as cp
    raw=cp.asarray(count)
    if raw.dtype.kind not in 'iu':raise ValueError('Snow counts must be integers')
    area,cells,inputs,leaf,parameters=(cp.ascontiguousarray(cp.asarray(v,cp.float32)) for v in (area,cells,inputs,leaf,parameters))
    count=cp.ascontiguousarray(raw,cp.int32)
    validate_canopy_snow(area,cells,count,inputs,cp);validate_canopy_optics(area,leaf,parameters,cp)
    out=cp.empty((len(area),624),cp.float32)
    if len(area):_kernel()((min((len(area)+63)//64,4096),),(64,),
        (area,cells,count,inputs,leaf,parameters,*_tables(),out,np.int32(len(area))))
    if not bool(cp.all(cp.isfinite(out))):raise RuntimeError('Nonfinite PFT canopy optics')
    return out


def run_canopy_snow_optics_cuda(area,cells,count,inputs,leaf,parameters):
    return unpack_canopy_snow_optics(canopy_snow_optics_raw_cuda(area,cells,count,inputs,leaf,parameters))

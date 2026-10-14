"""Native CUDA canopy/SNICAR shortwave boundary coupling."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..canopy_snow_radiation import validate_canopy_snow,unpack_canopy_snow
from .snicar import _tables


@lru_cache(maxsize=2)
def _kernel(track_absorption=False):
    import cupy as cp
    source='\n'.join(Path(__file__).with_name(p).read_text() for p in
        ('radiation_kernel.cu','snicar_kernel.cu','canopy_snow_kernel.cu'))
    name='run_canopy_snow_stable' if track_absorption else 'run_canopy_snow'
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=(name,))
    return module.get_function(name)


def run_canopy_snow_cuda(area,cells,counts,inputs):
    import cupy as cp
    raw=cp.asarray(counts)
    if raw.dtype.kind not in 'iu':raise ValueError('Snow counts must be integers')
    a,c,u=[cp.ascontiguousarray(cp.asarray(v,dtype=cp.float32)) for v in (area,cells,inputs)]
    n=cp.ascontiguousarray(raw,dtype=cp.int32);validate_canopy_snow(a,c,n,u,cp)
    out=cp.empty((len(c),584),cp.float32)
    if len(c):_kernel()((min((len(c)+63)//64,4096),),(64,),(a,c,n,u,*_tables(),out,np.int32(len(c))))
    if not bool(cp.all(cp.isfinite(out))):raise RuntimeError('Nonfinite canopy/snow shortwave')
    return unpack_canopy_snow(out)

"""Native CUDA snow microstructure and climatological aerosol deposition."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..snow_evolution import aging_table,validate_snow_evolution,unpack_snow_evolution


@lru_cache(maxsize=1)
def _module():
    import cupy as cp
    here=Path(__file__).parent
    source='\n'.join((here/name).read_text() for name in ('snow_mass_kernel.cu','snow_evolution_kernel.cu'))
    return cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_snow_evolution',))


@lru_cache(maxsize=1)
def _table():
    import cupy as cp
    return cp.asarray(aging_table(np.float32))


def run_snow_evolution_cuda(cells,options,inputs):
    import cupy as cp
    raw=cp.asarray(options)
    if raw.dtype.kind not in 'iu':raise ValueError('Snow options must be integers')
    c,u=[cp.ascontiguousarray(cp.asarray(v,dtype=cp.float32)) for v in (cells,inputs)]
    o=cp.ascontiguousarray(raw,dtype=cp.int32);validate_snow_evolution(c,o,u,cp)
    out=cp.empty((len(c),73),cp.float32)
    if len(c):_module().get_function('run_snow_evolution')((min((len(c)+63)//64,4096),),(64,),
        (c,o,u,_table(),out,np.int32(len(c))))
    if not bool(cp.all(cp.isfinite(out))):raise RuntimeError('Nonfinite snow evolution')
    return unpack_snow_evolution(out)

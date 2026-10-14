"""Native CUDA five-layer, five-band SNICAR approximation."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..snicar import optical_tables,validate_snow_optics,unpack_snow_shortwave


@lru_cache(maxsize=1)
def _module():
    import cupy as cp
    return cp.RawModule(code=Path(__file__).with_name('snicar_kernel.cu').read_text(),
        options=('--std=c++14','--fmad=false','--prec-div=false'),
        name_expressions=('run_snow_shortwave','run_snow_spectrum'))


@lru_cache(maxsize=1)
def _tables():
    import cupy as cp
    return tuple(cp.asarray(x) for x in optical_tables(np.float32))


def run_snow_shortwave_cuda(cells,counts,boundary):
    import cupy as cp
    raw=cp.asarray(counts)
    if raw.dtype.kind not in 'iu':raise ValueError('Snow counts must be integers')
    c,b=[cp.ascontiguousarray(cp.asarray(x,dtype=cp.float32)) for x in (cells,boundary)]
    n=cp.ascontiguousarray(raw,dtype=cp.int32);validate_snow_optics(c,n,b,cp)
    out=cp.empty((len(c),460),cp.float32)
    if len(c):_module().get_function('run_snow_shortwave')((min((len(c)+63)//64,4096),),(64,),
        (c,n,b,*_tables(),out,np.int32(len(c))))
    if not bool(cp.all(cp.isfinite(out))):raise RuntimeError('Nonfinite snow optical solve')
    return unpack_snow_shortwave(out)

"""CUDA implementation of the development soil-water stress component."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..clm_water_stress import validate_water_stress


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    source = Path(__file__).with_name('clm_water_stress_kernel.cu').read_text()
    return cp.RawKernel(source, 'run_clm_water_stress',
                        options=('--std=c++14', '--fmad=false', '--prec-div=true'))


def run_water_stress_cuda(liquid, roots, parameters, options):
    import cupy as cp
    liquid, roots, parameters = (cp.ascontiguousarray(cp.asarray(v, dtype=cp.float32))
                                  for v in (liquid, roots, parameters))
    options = cp.ascontiguousarray(cp.asarray(options, dtype=cp.int32))
    validate_water_stress(liquid, roots, parameters, options, cp)
    count = len(liquid)
    out = cp.empty((count, 5), cp.float32)
    if count:
        _kernel()((min((count+127)//128, 4096),), (128,),
                  (liquid, roots, parameters, options, out, np.int32(count)))
    return dict(root_factor=out[:, 0], layer_factor=out[:, 1:])

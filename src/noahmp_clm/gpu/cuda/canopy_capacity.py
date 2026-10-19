"""CUDA32 capacity profiles and retained temperature history, no host physics."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..canopy_capacity import validate_capacity, unpack_capacity, validate_temperature_memory


@lru_cache(maxsize=2)
def _kernel(name):
    import cupy as cp
    source = Path(__file__).with_name('canopy_capacity_kernel.cu').read_text()
    options=('--std=c++14', '--fmad=false')
    if name=='canopy_capacity':options+=('--prec-div=false',)
    return cp.RawKernel(source, name, options=options)


def run_capacity_cuda(leaf, environment, traits, template):
    import cupy as cp
    arrays = tuple(cp.ascontiguousarray(cp.asarray(v)) for v in (leaf, environment, traits, template))
    validate_capacity(*arrays, cp)
    if arrays[0].dtype != cp.float32:
        raise ValueError('CUDA capacity requires float32 inputs')
    output = cp.empty((len(arrays[0]), 126), cp.float32)
    if len(output): _kernel('canopy_capacity')(((len(output)+127)//128,), (128,), (*arrays, output, np.int32(len(output))))
    return unpack_capacity(output)


def run_temperature_memory_cuda(values, controls, forcing):
    import cupy as cp
    arrays = tuple(cp.ascontiguousarray(cp.asarray(v)) for v in (values, controls, forcing))
    validate_temperature_memory(*arrays, cp)
    if arrays[0].dtype != cp.float32:
        raise ValueError('CUDA temperature memory requires float32 inputs')
    if arrays[1].dtype != cp.int32:
        raise ValueError('CUDA temperature-memory controls require int32')
    size, width = arrays[0].shape
    after = cp.empty_like(arrays[0]); next_controls = cp.empty_like(arrays[1]); means = cp.empty((size, 2), cp.float32)
    if size: _kernel('temperature_memory')(((size+127)//128,), (128,), (*arrays, after, next_controls, means, np.int32(size), np.int32(width)))
    return dict(values=after, controls=next_controls, means=means)

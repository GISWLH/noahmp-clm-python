"""CUDA leaf component; imports and compilation are lazy for CPU installations."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..photosynthesis import pack_leaf_parameters, validate_leaf_inputs


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    module = cp.RawModule(code=Path(__file__).with_name("leaf_kernel.cu").read_text(),
        # XLA uses approximate square root as well as approximate division.
        # A rounded sqrt changes the ci bracket at a few float32 roots; the
        # 6075-case probe isolates this without hiding diffusion residuals.
        options=("--std=c++14", "--fmad=false", "--prec-div=false", "--prec-sqrt=false"),
        name_expressions=("run_leaves",))
    return module.get_function("run_leaves")


def run_leaves_cuda(inputs, parameters=None):
    import cupy as cp
    u = cp.ascontiguousarray(cp.asarray(inputs, dtype=cp.float32))
    validate_leaf_inputs(u, cp)
    p = cp.asarray(pack_leaf_parameters(parameters, np.float32))
    output = cp.empty((u.shape[0], 12), dtype=cp.float32)
    if u.shape[0]:
        _kernel()((min((u.shape[0]+255)//256, 4096),), (256,),
                  (u, p, output, np.int32(u.shape[0])))
    return output

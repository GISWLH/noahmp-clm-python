"""CUDA ten-layer shortwave solver, lazy CuPy import and NVRTC compilation."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..canopy_radiation import (unpack_shortwave, validate_shortwave_inputs,
    unpack_longwave,validate_longwave_inputs)


@lru_cache(maxsize=2)
def _kernel(name="run_shortwave"):
    import cupy as cp
    module = cp.RawModule(code=Path(__file__).with_name("radiation_kernel.cu").read_text(),
        options=("--std=c++14","--fmad=false","--prec-div=false"),
        name_expressions=(name,))
    return module.get_function(name)


def run_shortwave_cuda(leaf_area, inputs):
    import cupy as cp
    lai = cp.ascontiguousarray(cp.asarray(leaf_area,dtype=cp.float32))
    u = cp.ascontiguousarray(cp.asarray(inputs,dtype=cp.float32))
    validate_shortwave_inputs(lai,u,cp)
    result = cp.empty((lai.shape[0],46),dtype=cp.float32)
    if lai.shape[0]:
        _kernel()((min((lai.shape[0]+255)//256,4096),),(256,),
                  (lai,u,result,np.int32(lai.shape[0])))
    return unpack_shortwave(result)


def run_longwave_cuda(leaf_area,temperature,inputs):
    import cupy as cp
    lai,temp,u=[cp.ascontiguousarray(cp.asarray(a,dtype=cp.float32)) for a in (leaf_area,temperature,inputs)]
    validate_longwave_inputs(lai,temp,u,cp)
    result=cp.empty((lai.shape[0],35),dtype=cp.float32)
    if lai.shape[0]:
        _kernel("run_longwave")((min((lai.shape[0]+255)//256,4096),),(256,),
            (lai,temp,u,result,np.int32(lai.shape[0])))
    return unpack_longwave(result)

"""CUDA canopy energy; reuse the tested CUDA leaf and optical components."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..canopy_energy import validate_canopy_batch,unpack_canopy_energy,CANOPY_OUTPUT_SIZE
from ..photosynthesis import pack_leaf_parameters


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    folder=Path(__file__).parent
    source="\n".join((folder/name).read_text() for name in
        ("leaf_kernel.cu","radiation_kernel.cu","canopy_energy_kernel.cu"))
    module=cp.RawModule(code=source,options=("--std=c++14","--fmad=false",
        "--prec-div=false","--prec-sqrt=false"),name_expressions=("run_canopy_energy",))
    return module.get_function("run_canopy_energy")


def run_canopy_energy_cuda(area,inputs,factors,soil,initial,parameters=None,*,max_iterations=100,
                            tolerance=.005,raise_on_failure=True):
    import cupy as cp
    arrays=tuple(cp.ascontiguousarray(cp.asarray(a,dtype=cp.float32)) for a in (area,inputs,factors,soil,initial))
    validate_canopy_batch(*arrays,cp)
    if not isinstance(max_iterations,int) or max_iterations<1 or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError("Invalid canopy solver controls")
    p=cp.asarray(pack_leaf_parameters(parameters,np.float32))
    n=arrays[0].shape[0]
    output=cp.empty((n,CANOPY_OUTPUT_SIZE),dtype=cp.float32)
    if n:
        _kernel()((min((n+63)//64,4096),),(64,),(*arrays,p,output,np.int32(n),
            np.int32(max_iterations),np.float32(tolerance),np.float32(1e-8)))
    result=unpack_canopy_energy(output)
    if raise_on_failure and not bool(cp.all(result["converged"])):
        raise RuntimeError("CUDA canopy energy did not converge; inspect with raise_on_failure=False")
    return result

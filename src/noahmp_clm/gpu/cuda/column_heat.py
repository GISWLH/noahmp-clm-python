"""Native CUDA shared snow/soil heat equation and phase events."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..column_heat import validate_joint_heat,unpack_joint_heat


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    source='\n'.join(Path(__file__).with_name(p).read_text() for p in ('enthalpy_kernel.cu','column_heat_kernel.cu'))
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('run_joint_heat',))
    return module.get_function('run_joint_heat')


def run_joint_heat_cuda(cells,props,options,boundary):
    import cupy as cp
    raw=cp.asarray(options)
    if raw.dtype.kind not in 'iu': raise ValueError('Heat options must be integers')
    c,p,b=[cp.ascontiguousarray(cp.asarray(v,dtype=cp.float32)) for v in (cells,props,boundary)]
    o=cp.ascontiguousarray(raw,dtype=cp.int32)
    validate_joint_heat(c,p,o,b,cp)
    result=cp.empty((len(c),77),dtype=cp.float32)
    if len(c): _kernel()((min((len(c)+63)//64,4096),),(64,),(c,p,o,b,result,np.int32(len(c))))
    fields=tuple(result[:,i:i+9] for i in range(0,54,9))+(result[:,54:62],result[:,62:71],result[:,71:])
    output=unpack_joint_heat(fields)
    if not bool(cp.all(output['converged'])): raise RuntimeError('CUDA joint snow/soil enthalpy solve failed')
    return output

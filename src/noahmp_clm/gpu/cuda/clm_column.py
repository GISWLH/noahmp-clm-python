"""Native CUDA advancement of the developing snow-free CLM column."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..clm_column import validate_clm_batch,site_array,CLM_DIAGNOSTICS
from ..pack import OPTION_FIELDS,INT_SIZE
from ..photosynthesis import pack_leaf_parameters
from ..canopy_energy import unpack_canopy_energy
from .canopy_energy import _kernel as energy_kernel


@lru_cache(maxsize=1)
def _kernels():
    import cupy as cp
    folder=Path(__file__).parent
    header=f"#define CLM_INT_SIZE {INT_SIZE}\n"+"\n".join(
        f"#define CLM_{name} {idx}" for idx,name in enumerate(OPTION_FIELDS))+"\n"
    source=header+"\n".join((folder/name).read_text() for name in
        ("column_kernel.cu","leaf_kernel.cu","radiation_kernel.cu","canopy_energy_kernel.cu","clm_column_kernel.cu"))
    names=("prepare_clm_columns","finish_clm_columns")
    module=cp.RawModule(code=source,options=("--std=c++14","--fmad=false",
        "--prec-div=false","--prec-sqrt=false"),name_expressions=names)
    return tuple(module.get_function(name) for name in names)


def step_clm_columns_cuda(states,options,forcing,site=None,*,leaf_parameters=None,return_canopy=False):
    import cupy as cp
    s=cp.ascontiguousarray(cp.asarray(states,dtype=cp.float32))
    i=cp.ascontiguousarray(cp.asarray(options))
    f=cp.ascontiguousarray(cp.asarray(forcing,dtype=cp.float32))
    n=s.shape[0]
    v=cp.ascontiguousarray(cp.asarray(site_array(n,np.float32) if site is None else site,dtype=cp.float32))
    validate_clm_batch(s,i,f,v,cp)
    arrays=[cp.empty(shape,cp.float32) for shape in ((n,10),(n,23),(n,10,2),(n,10),(n,11),(n,4))]
    c=cp.empty((n,195),cp.float32)
    new=cp.empty_like(s);diag=cp.empty((n,len(CLM_DIAGNOSTICS)),cp.float32)
    if n:
        launch=((min((n+63)//64,4096),),(64,))
        prepare,finish=_kernels()
        prepare(*launch,(s,i,f,v,*arrays,np.int32(n)))
        params=cp.asarray(pack_leaf_parameters(leaf_parameters,np.float32))
        energy_kernel()(*launch,(*arrays[:5],params,c,np.int32(n),np.int32(100),np.float32(.005),np.float32(1e-8)))
        if not bool(cp.all(c[:,68]>0)):
            raise RuntimeError("CLM CUDA canopy did not converge; state was not advanced")
        finish(*launch,(s,i,f,arrays[5],c,new,diag,np.int32(n)))
        if not bool(cp.all(diag[:,21]>0)) or not bool(cp.all(cp.isfinite(new))):
            raise RuntimeError("CLM CUDA water withdrawal or resulting state is invalid")
    return (new,i,diag,unpack_canopy_energy(c)) if return_canopy else (new,i,diag)

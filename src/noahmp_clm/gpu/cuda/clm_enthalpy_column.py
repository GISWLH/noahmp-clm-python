"""Native CUDA extensive CLM column, preserving the new enthalpy state."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..clm_enthalpy_column import validate_column,unpack_column,SURFACE_FIELDS
from .canopy_surface import run_canopy_surface_cuda


@lru_cache(maxsize=2)
def _stages(closed_richards=False):
    import cupy as cp
    names=('column_kernel.cu','enthalpy_kernel.cu','snow_mass_kernel.cu','snow_evolution_kernel.cu',
        'soil_transport_kernel.cu','clm_enthalpy_column_kernel.cu')
    source='\n'.join(Path(__file__).with_name(name).read_text(encoding='utf-8') for name in names)
    if closed_richards:source='#define CLM_RICHARDS_ZERO_FLUX\n'+source
    exports=('prepare_enthalpy_columns','finish_enthalpy_columns')
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'),name_expressions=exports)
    return tuple(module.get_function(name) for name in exports)


def step_clm_enthalpy_columns_cuda(c,g,p,o,site,storage,roots,forcing,*,deposition=None,leaf_parameters=None,return_surface=False,soil_hydrology=None):
    import cupy as cp
    raw=cp.asarray(o)
    if raw.dtype.kind not in 'iu':raise ValueError('CLM options must be integers')
    c,g,p,site,storage,roots,forcing=[cp.ascontiguousarray(cp.asarray(x,dtype=cp.float32)) for x in (c,g,p,site,storage,roots,forcing)]
    o=cp.ascontiguousarray(raw,dtype=cp.int32);n=len(c)
    deposition=cp.zeros((n,5),cp.float32) if deposition is None else cp.ascontiguousarray(cp.asarray(deposition,dtype=cp.float32))
    arrays=c,g,p,o,site,storage,roots;validate_column(arrays,forcing,deposition,cp)
    from ..clm_richards_column import resolve_column_richards,attach_column_richards
    hydrology=resolve_column_richards(soil_hydrology,o,cp)
    pc=cp.empty_like(c);pg=cp.empty_like(g);po=cp.empty((n,2),cp.int32)
    a=cp.empty((n,10),cp.float32);u=cp.empty((n,23),cp.float32);f=cp.empty((n,10,2),cp.float32)
    v=cp.empty((n,7),cp.float32);prep=cp.empty((n,5),cp.float32);valid=cp.empty(n,cp.int32)
    pre_impurity=cp.empty((n,5),cp.float32)
    launch=((min((n+31)//32,4096),),(32,))
    if n:
        prepare,finish=_stages(hydrology is not None)
        prepare(*launch,(*arrays,forcing,pc,pg,po,a,u,f,v,prep,valid,pre_impurity,np.uint64(0),np.int32(n)))
        if not bool(cp.all(valid>0)):raise RuntimeError('CUDA CLM precipitation preparation failed')
    surface=run_canopy_surface_cuda(pc,pg,p,po,a,u,f,v,parameters=leaf_parameters)
    flux=cp.ascontiguousarray(cp.stack([surface[key] for key in SURFACE_FIELDS],axis=1))
    canopy=cp.ascontiguousarray(cp.stack((surface['canopy_water_after_kgm2'],surface['canopy_enthalpy_after_jm2']),axis=1))
    sc=cp.ascontiguousarray(surface['snow_cells']);sg=cp.ascontiguousarray(surface['soil_cells'])
    sn=cp.ascontiguousarray(surface['snow_counts']);imp=cp.ascontiguousarray(surface['impurity_output_kgm2']+pre_impurity)
    nc=cp.empty_like(c);ng=cp.empty_like(g);no=cp.empty_like(o);ns=cp.empty_like(storage);diag=cp.empty((n,26),cp.float32);impurity=cp.empty((n,5),cp.float32)
    if n:
        finish(*launch,(c,g,p,o,site,storage,forcing,deposition,prep,sc,sg,sn,canopy,flux,imp,nc,ng,no,ns,diag,impurity,np.uint64(0),np.int32(n)))
    if not bool(cp.all(diag[:,25]>0.)) or not all(bool(cp.all(cp.isfinite(x))) for x in (nc,ng,ns,diag,impurity)):
        raise RuntimeError('CUDA CLM post-surface water/heat transfer failed')
    result=unpack_column(nc,ng,no,ns,diag,impurity,surface=surface if return_surface else None)
    return attach_column_richards(result,p,o,forcing,c,g,storage,hydrology,'cuda')

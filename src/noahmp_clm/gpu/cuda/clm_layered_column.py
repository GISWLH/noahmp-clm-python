"""CUDA whole-column layered canopy phase state; every transfer runs on device."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..clm_layered_column import validate_layered_column
from ..clm_enthalpy_column import unpack_column,SURFACE_FIELDS
from ..canopy_interception import unpack_canopy_water
from .clm_enthalpy_column import _stages as _column_stages
from .canopy_phase_surface import run_canopy_phase_surface_cuda


@lru_cache(maxsize=1)
def _stages():
    import cupy as cp
    names=('column_kernel.cu','enthalpy_kernel.cu','snow_mass_kernel.cu','snow_evolution_kernel.cu',
        'soil_transport_kernel.cu','canopy_interception_kernel.cu','clm_enthalpy_column_kernel.cu','clm_layered_column_kernel.cu')
    source='\n'.join(Path(__file__).with_name(n).read_text(encoding='utf-8') for n in names)
    exports=('prepare_layered_transfer','finish_layered_transfer')
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'),name_expressions=exports)
    return tuple(module.get_function(n) for n in exports)


def step_clm_layered_columns_cuda(c,g,p,o,site,storage,roots,condensate,forcing,*,deposition=None,leaf_parameters=None,return_surface=False,most_geometry=None,vic_parameters=None):
    import cupy as cp
    if cp.asarray(o).dtype.kind not in 'iu':raise ValueError('CLM options must be integers')
    c,g,p,site,storage,roots,condensate,forcing=[cp.ascontiguousarray(cp.asarray(x,cp.float32)) for x in (c,g,p,site,storage,roots,condensate,forcing)]
    o=cp.ascontiguousarray(cp.asarray(o,cp.int32));n=len(c)
    deposition=cp.zeros((n,5),cp.float32) if deposition is None else cp.ascontiguousarray(cp.asarray(deposition,cp.float32))
    validate_layered_column((c,g,p,o,site,storage,roots,condensate),forcing,deposition,cp)
    from ..vic import column_vic
    vic=column_vic(o,c.dtype,cp,vic_parameters);vic_arg=np.uint64(0) if vic is None else cp.ascontiguousarray(vic)
    new=lambda *shape:cp.empty((n,*shape),cp.float32)
    pc0=new(5,9);pg0=new(4,3);po0=cp.empty_like(o);cc=new(10,2);totals=new(2);prep=new(5);pre=new(182);old=new(3);good=cp.empty(n,cp.int32)
    pc=new(5,9);pg=new(4,3);po=cp.empty((n,2),cp.int32);a=new(10);u=new(23);f=new(10,2);v=new(7);unused=new(5);valid=cp.empty(n,cp.int32)
    launch=((min((n+31)//32,4096),),(32,))
    if n:
        prepare,finish=_stages();base_prepare,base_finish=_column_stages()
        prepare(*launch,(c,g,p,o,site,storage,condensate,forcing,pc0,pg0,po0,cc,totals,prep,pre,old,good,vic_arg,np.int32(n)))
        empty=storage.copy();empty[:,:2]=0.;dry=forcing.copy();dry[:,7]=0.
        base_prepare(*launch,(pc0,pg0,p,po0,site,empty,roots,dry,pc,pg,po,a,u,f,v,unused,valid,np.int32(n)))
        v[:,1:3]=totals
        if not bool(cp.all((good>0)&(valid>0))):raise RuntimeError('CUDA layered precipitation preparation failed')
    surface_arrays=(pc,pg,p,po,a,u,f,v,cc)
    if most_geometry is None:surface=run_canopy_phase_surface_cuda(*surface_arrays,parameters=leaf_parameters,vic=vic)
    else:
        from ..canopy_most_surface import run_most_surface
        if bool(cp.any(o[:,5]!=1)):raise ValueError('MOST geometry requires OptSurfaceDrag=1')
        surface=run_most_surface(surface_arrays,most_geometry,forcing[:,3],site[:,10],site[:,0],backend='cuda',parameters=leaf_parameters,vic=vic)
    flux=cp.ascontiguousarray(cp.stack([surface[key] for key in SURFACE_FIELDS],axis=1))
    sc=cp.ascontiguousarray(surface['snow_cells']);sg=cp.ascontiguousarray(surface['soil_cells']);sn=cp.ascontiguousarray(surface['snow_counts'])
    cond=cp.ascontiguousarray(surface['condensate']);imp=cp.ascontiguousarray(surface['impurity_output_kgm2'])
    pc=new(5,9);pg=new(4,3);pn=cp.empty(n,cp.int32);cc=new(10,2);totals=new(2);pp=new(5);pi=new(5);post=new(182)
    nc=new(5,9);ng=new(4,3);no=cp.empty_like(o);ns=new(3);diag=new(26);impurity=new(5)
    if n:
        finish(*launch,(sc,sg,p,o,site,forcing,prep,sn,cond,imp,pc,pg,pn,cc,totals,pp,pi,post,good,vic_arg,np.int32(n)))
        base_finish(*launch,(c,g,p,o,site,old,forcing,deposition,pp,pc,pg,pn,totals,flux,pi,nc,ng,no,ns,diag,impurity,vic_arg,np.int32(n)))
    if not bool(cp.all(good>0)) or not bool(cp.all(diag[:,25]>0.)) or not all(bool(cp.all(cp.isfinite(x))) for x in (nc,ng,ns,diag,impurity,cc,post)):
        raise RuntimeError('CUDA layered post-surface transfer failed')
    result=unpack_column(nc,ng,no,ns,diag,impurity,surface=surface if return_surface else None)
    result.update(condensate=cc,canopy_interception=unpack_canopy_water(pre),canopy_post_interception=unpack_canopy_water(post))
    if most_geometry is not None:result['most']=surface['most']
    return result

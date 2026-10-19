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


def step_clm_layered_columns_cuda(c,g,p,o,site,storage,roots,condensate,forcing,*,deposition=None,leaf_parameters=None,return_surface=False,most_geometry=None,vic_parameters=None,
                                irrigation_state=None,irrigation_parameters=None,irrigation_input=None,irrigation_check=None,tile_parameters=None,
                                canopy_structure=None,structure_lai_before=None,structure_management=None,prescribed_sai=None,canopy_optics=None,
                                canopy_capacity=None,capacity_environment=None,soil_hydrology=None):
    import cupy as cp
    if cp.asarray(o).dtype.kind not in 'iu':raise ValueError('CLM options must be integers')
    c,g,p,site,storage,roots,condensate,forcing=[cp.ascontiguousarray(cp.asarray(x,cp.float32)) for x in (c,g,p,site,storage,roots,condensate,forcing)]
    o=cp.ascontiguousarray(cp.asarray(o,cp.int32));n=len(c)
    deposition=cp.zeros((n,5),cp.float32) if deposition is None else cp.ascontiguousarray(cp.asarray(deposition,cp.float32))
    validate_layered_column((c,g,p,o,site,storage,roots,condensate),forcing,deposition,cp)
    from ..clm_richards_column import resolve_column_richards,attach_column_richards
    hydrology=resolve_column_richards(soil_hydrology,o,cp)
    from ..clm_structure_column import resolve_column_structure,attach_column_structure
    geometry=resolve_column_structure(canopy_structure,(c,g,p,o,site,storage,roots,condensate),forcing,'cuda',
        previous_lai=structure_lai_before,management=structure_management,prescribed_sai=prescribed_sai)
    area=None if geometry is None else cp.ascontiguousarray(cp.stack((geometry['exposed_lai'],geometry['exposed_sai']),axis=1))
    area_arg=np.uint64(0) if area is None else area
    physical_site=site
    if area is not None:physical_site=site.copy();physical_site[:,0]=area[:,0]+area[:,1]
    from ..irrigation import prepare_column_irrigation,attach_column_irrigation
    ground_input,irrigation=prepare_column_irrigation(g,p,o,site,forcing,irrigation_state,irrigation_parameters,irrigation_input,irrigation_check,'cuda')
    ground_arg=np.uint64(0) if ground_input is None else cp.ascontiguousarray(ground_input)
    from ..vic import column_vic
    vic=column_vic(o,c.dtype,cp,vic_parameters);vic_arg=np.uint64(0) if vic is None else cp.ascontiguousarray(vic)
    new=lambda *shape:cp.empty((n,*shape),cp.float32)
    pc0=new(5,9);pg0=new(4,3);po0=cp.empty_like(o);cc=new(10,2);totals=new(2);prep=new(5);pre=new(182);old=new(3);good=cp.empty(n,cp.int32)
    pc=new(5,9);pg=new(4,3);po=cp.empty((n,2),cp.int32);a=new(10);u=new(23);f=new(10,2);v=new(7);unused=new(5);valid=cp.empty(n,cp.int32)
    pre_impurity=new(5)
    launch=((min((n+31)//32,4096),),(32,))
    if n:
        prepare,finish=_stages();base_prepare,base_finish=_column_stages(hydrology is not None)
        prepare(*launch,(c,g,p,o,site,storage,condensate,forcing,pc0,pg0,po0,cc,totals,prep,pre,old,good,vic_arg,ground_arg,area_arg,np.int32(n)))
        empty=storage.copy();empty[:,:2]=0.;dry=forcing.copy();dry[:,7]=0.
        base_prepare(*launch,(pc0,pg0,p,po0,physical_site,empty,roots,dry,pc,pg,po,a,u,f,v,unused,valid,pre_impurity,vic_arg,np.int32(n)))
        if area is not None:a[:]=area[:,0,None]*cp.float32(.1)
        prep[:,2:4]=prep[:,2:4]+unused[:,2:4]
        v[:,1:3]=totals
        if not bool(cp.all((good>0)&(valid>0))):raise RuntimeError('CUDA layered precipitation preparation failed')
    surface_arrays=(pc,pg,p,po,a,u,f,v,cc)
    from ..clm_capacity_column import resolve_column_capacity,attach_column_capacity
    leaf_parameters,capacity_state=resolve_column_capacity(canopy_capacity,capacity_environment,a,forcing,leaf_parameters,'cuda')
    # Match JAX's explicit reciprocal multiplication. Correctly rounded CuPy
    # division can differ by1ulp (e.g. SAI0.8), changing a nonlinear leaf solve.
    stems={} if area is None else dict(stem_area=cp.ascontiguousarray(cp.broadcast_to(area[:,1,None]*cp.float32(.1),(n,10))))
    if canopy_optics is not None:stems['canopy_optics']=canopy_optics
    if most_geometry is None:surface=run_canopy_phase_surface_cuda(*surface_arrays,parameters=leaf_parameters,vic=vic,**stems)
    else:
        from ..canopy_most_surface import run_most_surface
        if bool(cp.any(o[:,5]!=1)):raise ValueError('MOST geometry requires OptSurfaceDrag=1')
        plant=site[:,0] if area is None else area[:,0]+area[:,1]
        surface=run_most_surface(surface_arrays,most_geometry,forcing[:,3],site[:,10],plant,backend='cuda',parameters=leaf_parameters,vic=vic,**stems)
    flux=cp.ascontiguousarray(cp.stack([surface[key] for key in SURFACE_FIELDS],axis=1))
    sc=cp.ascontiguousarray(surface['snow_cells']);sg=cp.ascontiguousarray(surface['soil_cells']);sn=cp.ascontiguousarray(surface['snow_counts'])
    cond=cp.ascontiguousarray(surface['condensate']);imp=cp.ascontiguousarray(surface['impurity_output_kgm2']+pre_impurity)
    pc=new(5,9);pg=new(4,3);pn=cp.empty(n,cp.int32);cc=new(10,2);totals=new(2);pp=new(5);pi=new(5);post=new(182)
    nc=new(5,9);ng=new(4,3);no=cp.empty_like(o);ns=new(3);diag=new(26);impurity=new(5)
    if n:
        finish(*launch,(sc,sg,p,o,site,forcing,prep,sn,cond,imp,pc,pg,pn,cc,totals,pp,pi,post,good,vic_arg,area_arg,np.int32(n)))
        base_finish(*launch,(c,g,p,o,site,old,forcing,deposition,pp,pc,pg,pn,totals,flux,pi,nc,ng,no,ns,diag,impurity,vic_arg,np.int32(n)))
    if not bool(cp.all(good>0)) or not bool(cp.all(diag[:,25]>0.)) or not all(bool(cp.all(cp.isfinite(x))) for x in (nc,ng,ns,diag,impurity,cc,post)):
        raise RuntimeError('CUDA layered post-surface transfer failed')
    result=unpack_column(nc,ng,no,ns,diag,impurity,surface=surface if return_surface else None)
    result.update(condensate=cc,canopy_interception=unpack_canopy_water(pre),canopy_post_interception=unpack_canopy_water(post))
    if most_geometry is not None:result['most']=surface['most']
    from ..tile_drainage import finish_column_tile
    result=attach_column_structure(result,canopy_structure,geometry)
    result=attach_column_capacity(result,capacity_state)
    result=attach_column_richards(result,p,o,forcing,c,g,old,hydrology,'cuda')
    return finish_column_tile(attach_column_irrigation(result,irrigation),p,o,forcing,tile_parameters,'cuda')

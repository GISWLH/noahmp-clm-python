"""Whole CLM development column retaining ten canopy ice/liquid M/H stores.

Rain/snow interception and unloading surround the native phase surface solve.
Existing stress, drag and drainage stages are reused with precipitation and
aggregate interception disabled after the layered transfer owns those parcels.
"""
from dataclasses import replace
from functools import lru_cache
import numpy as np
from .clm_enthalpy_column import (pack_clm_enthalpy_columns,validate_column,prepare_column_jax,
    finish_column_jax,route_water_jax,unpack_column,SURFACE_FIELDS)
from .canopy_interception import canopy_water_kernel_jax,validate_canopy_water,unpack_canopy_water
from .canopy_energy import _sum
from .snow_evolution import fresh_snow_jax
from .snow_mass import snow_transfer_kernel_jax
from .soil_transport import soil_transport_kernel_jax,compact_soil_parameters
from .canopy_phase_surface import run_canopy_phase_surface_jax


def pack_clm_layered_columns(states,**kwargs):
    states=list(states);dtype=kwargs.get('dtype',np.float64)
    if any(s.canopy_capacity_state is not None for s in states):
        raise NotImplementedError('Canopy capacity needs an extended native ABI retaining temperature history')
    if any(s.canopy_structure_parameters is not None for s in states):
        raise NotImplementedError('Use pack_clm_structure_columns; the base ABI cannot retain stem geometry')
    if any(s.irrigation_state is not None for s in states):
        raise NotImplementedError('Irrigation requires a native entry retaining requests and external water/heat')
    if any(s.bgc_state is not None for s in states):
        raise NotImplementedError('BGC requires an extended native entry retaining all plant/soil C/N stores')
    if any(s.options.OptDynamicVeg!=4 for s in states):
        raise NotImplementedError('Use pack_clm_phenology_columns to retain dynamic vegetation state')
    if any(s.canopy_condensate is None or len(s.canopy_condensate.mass)!=10 for s in states):
        raise ValueError('Every column must carry ten canopy layers')
    base=pack_clm_enthalpy_columns([replace(s,canopy_condensate=None) for s in states],**kwargs)
    condensate=np.asarray([np.column_stack((s.canopy_condensate.mass,s.canopy_condensate.enthalpy)) for s in states],dtype).reshape(-1,10,2)
    return (*base,condensate)


def validate_layered_column(arrays,forcing,deposition,xp):
    *base,condensate=arrays;validate_column(base,forcing,deposition,xp,allow_vic=True)
    c,g,p,o,site,store,roots=base;n=len(c);dtype=c.dtype
    area=xp.stack((xp.broadcast_to(site[:,0,None]/10.,(n,10)),xp.zeros((n,10),dtype)),axis=2)
    u=xp.stack((xp.zeros(n,dtype),xp.zeros(n,dtype),forcing[:,1],forcing[:,0]/2.,site[:,1],forcing[:,3],xp.full(n,270.15,dtype)),axis=1)
    validate_canopy_water(condensate,area,u,xp.zeros_like(condensate),xp.zeros(n,xp.int32),xp)
    eps=xp.finfo(dtype).eps
    if bool(xp.any(abs(condensate.sum(1)-store[:,:2])>16.*eps*xp.maximum(abs(condensate).sum(1),1.))):
        raise ValueError('Column aggregate aliases disagree with canopy layers')


def route_layer_outlets_jax(c,g,n,p,phase,forcing,ledger,vic=None):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;dt,tair,_,wind=forcing[:4];row=lambda *x:j.asarray(x,c.dtype)
    def snowfall():
        fresh=fresh_snow_jax(s(tair-273.15),wind)
        offset=ledger[7]/s(ledger[5]*2100.)
        u=j.concatenate((row(ledger[5],offset,fresh[0],fresh[1],.033,.05,.2,.02,.02,.01,.01),j.zeros(5,c.dtype)))
        out=snow_transfer_kernel_jax(c,n,j.asarray(1),u)
        return out[:45].reshape(5,9),out[45].astype(j.int32)
    c,n=lax.cond(ledger[5]>0.,snowfall,lambda:(c,n))
    c,g,runoff,valid=route_water_jax(c,g,n,p,phase,dt,ledger[4],ledger[6],vic)
    return c,g,n,runoff,valid


def layered_interception_jax(condensate,site,forcing,rain,snow,canopy_area=None):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    area=(j.stack((j.full(10,site[0]/10.,condensate.dtype),j.zeros(10,condensate.dtype)),axis=1)
          if canopy_area is None else j.broadcast_to(s(canopy_area*j.asarray(.1,condensate.dtype)),(10,2)))
    u=j.stack((rain,snow,forcing[1],s(forcing[0]/2.),site[1],forcing[3],j.asarray(270.15,condensate.dtype)))
    return canopy_water_kernel_jax(condensate,area,u,j.zeros_like(condensate),j.asarray(0))


def prepare_layered_jax(c,g,p,o,site,storage,roots,condensate,forcing,vic=None,ground_input=None,canopy_area=None):
    import jax.numpy as j
    from jax import lax
    from . import jax_core as core
    s=lax.optimization_barrier;row=lambda *x:j.asarray(x,c.dtype)
    total=s(forcing[7]*forcing[0]);sm=s(total*core._snow_fraction(forcing[1],forcing[2],forcing[4],o[2]))
    rain=s(s(total-sm)*1000.);snow=s(sm*1000.)
    intercepted=layered_interception_jax(condensate,site,forcing,rain,snow,canopy_area)
    cc=intercepted[:20].reshape(10,2);ledger=intercepted[170:182]
    c,g,n,out,valid=route_layer_outlets_jax(c,g,o[0],p,o[1],forcing,ledger,vic)
    if ground_input is not None:
        c,g,extra,good=route_water_jax(c,g,n,p,o[1],forcing[0],ground_input[0],ground_input[1],vic)
        out=s(out+extra);valid=valid&good
    empty=storage.at[:2].set(0.);dry=forcing.at[7].set(0.)
    physical_site=site if canopy_area is None else site.at[0].set(s(canopy_area[0]+canopy_area[1]))
    pc,pg,po,a,u,f,v,base_prep,good,pre_impurity=prepare_column_jax(c,g,p,o.at[0].set(n),physical_site,empty,roots,dry,vic)
    if canopy_area is not None:a=j.full(10,s(canopy_area[0]*j.asarray(.1,c.dtype)),c.dtype)
    out=s(out+base_prep[2:4])
    v=v.at[1].set(_sum(cc[:,0])).at[2].set(_sum(cc[:,1]))
    prep=j.concatenate((row(s(rain+snow),ledger[8]),out,row(0.)))
    if ground_input is not None:prep=prep.at[:2].set(s(prep[:2]+ground_input))
    old=storage.at[0].set(_sum(condensate[:,0])).at[1].set(_sum(condensate[:,1]))
    return pc,pg,po,a,u,f,v,prep,valid&good,cc,intercepted,old,pre_impurity


def finish_layered_jax(old_c,old_g,p,o,site,old_store,forcing,deposition,prep,c,g,n,condensate,flux,impurity,vic=None,canopy_area=None):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;row=lambda *x:j.asarray(x,c.dtype)
    post=layered_interception_jax(condensate,site,forcing,j.asarray(0.,c.dtype),j.asarray(0.,c.dtype),canopy_area)
    cc=post[:20].reshape(10,2);c,g,n,runoff,valid=route_layer_outlets_jax(c,g,n,p,o[1],forcing,post[170:182],vic)
    drained=snow_transfer_kernel_jax(c,n,j.asarray(3),row(0.,0.,100.,54.5,.033,.05,.2,.02,.02,.01,.01,0.,0.,0.,0.,0.))
    admit=soil_transport_kernel_jax(g,compact_soil_parameters(p,j),j.stack((j.asarray(0),o[1],j.asarray(1))),
        j.concatenate((row(drained[53],drained[54],forcing[0]),j.zeros(4,c.dtype))),vic)
    runoff=s(s(runoff+drained[60:62])+admit[26:28]);valid=valid&(admit[46]>0.)
    impurity=s(impurity+s(drained[62:67]+drained[55:60]))
    canopy=row(_sum(cc[:,0]),_sum(cc[:,1]));prep=prep.at[2:4].set(s(prep[2:4]+runoff)).at[4].set(canopy[0])
    out=finish_column_jax(old_c,old_g,p,o,site,old_store,forcing,deposition,prep,
        drained[:45].reshape(5,9),admit[:12].reshape(4,3),drained[45].astype(j.int32),canopy,flux,impurity,vic)
    return (*out,cc,post,valid)


@lru_cache(maxsize=8)
def _stages(vic=False,external=False,structure=False):
    import jax
    return (jax.jit(jax.vmap(prepare_layered_jax,in_axes=(0,)*9+(0 if vic else None,0 if external else None,0 if structure else None))),
            jax.jit(jax.vmap(finish_layered_jax,in_axes=(0,)*15+(0 if vic else None,0 if structure else None))))


def step_clm_layered_columns_jax(c,g,p,o,site,storage,roots,condensate,forcing,*,deposition=None,leaf_parameters=None,return_surface=False,most_geometry=None,vic_parameters=None,
                               irrigation_state=None,irrigation_parameters=None,irrigation_input=None,irrigation_check=None,tile_parameters=None,
                               canopy_structure=None,structure_lai_before=None,structure_management=None,prescribed_sai=None,canopy_optics=None,
                               canopy_capacity=None,capacity_environment=None):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(c,g,p,o,site,storage,roots,condensate)));forcing=j.asarray(forcing)
    deposition=j.zeros((len(c),5),arrays[0].dtype) if deposition is None else j.asarray(deposition)
    from .vic import column_vic
    validate_layered_column(arrays,forcing,deposition,j)
    from .clm_structure_column import resolve_column_structure,attach_column_structure
    geometry=resolve_column_structure(canopy_structure,arrays,forcing,'jax',previous_lai=structure_lai_before,
        management=structure_management,prescribed_sai=prescribed_sai)
    area=None if geometry is None else j.stack((geometry['exposed_lai'],geometry['exposed_sai']),axis=1)
    from .irrigation import prepare_column_irrigation,attach_column_irrigation
    ground_input,irrigation=prepare_column_irrigation(arrays[1],arrays[2],arrays[3],arrays[4],forcing,
        irrigation_state,irrigation_parameters,irrigation_input,irrigation_check,'jax')
    vic=column_vic(arrays[3],arrays[0].dtype,j,vic_parameters);prepare,finish=_stages(vic is not None,ground_input is not None,area is not None)
    pc,pg,po,a,u,f,v,prep,valid,cc,pre,old,pre_impurity=prepare(*arrays,forcing,vic,ground_input,area)
    if not bool(j.all(valid)):raise RuntimeError('Layered canopy precipitation preparation failed')
    surface_arrays=(pc,pg,arrays[2],po,a,u,f,v,cc)
    from .clm_capacity_column import resolve_column_capacity,attach_column_capacity
    leaf_parameters,capacity_state=resolve_column_capacity(canopy_capacity,capacity_environment,a,forcing,leaf_parameters,'jax')
    stems={} if area is None else dict(stem_area=j.broadcast_to(area[:,1,None]*j.asarray(.1,arrays[0].dtype),(len(c),10)))
    if canopy_optics is not None:stems['canopy_optics']=canopy_optics
    if most_geometry is None:surface=run_canopy_phase_surface_jax(*surface_arrays,parameters=leaf_parameters,vic=vic,**stems)
    else:
        from .canopy_most_surface import run_most_surface
        if bool(j.any(arrays[3][:,5]!=1)):raise ValueError('MOST geometry requires OptSurfaceDrag=1')
        plant=arrays[4][:,0] if area is None else area[:,0]+area[:,1]
        surface=run_most_surface(surface_arrays,most_geometry,forcing[:,3],arrays[4][:,10],plant,backend='jax',parameters=leaf_parameters,vic=vic,**stems)
    flux=j.stack([surface[key] for key in SURFACE_FIELDS],axis=1)
    out=finish(arrays[0],arrays[1],arrays[2],arrays[3],arrays[4],old,forcing,deposition,prep,
        surface['snow_cells'],surface['soil_cells'],surface['snow_counts'],surface['condensate'],flux,surface['impurity_output_kgm2']+pre_impurity,vic,area)
    if not bool(j.all(out[-1])) or not bool(j.all(out[4][:,25]>0.)) or not all(bool(j.all(j.isfinite(x))) for x in out):
        raise RuntimeError('Layered canopy post-surface transfer failed')
    result=unpack_column(*out[:6],surface=surface if return_surface else None)
    result.update(condensate=out[6],canopy_interception=unpack_canopy_water(pre),canopy_post_interception=unpack_canopy_water(out[7]))
    if most_geometry is not None:result['most']=surface['most']
    from .tile_drainage import finish_column_tile
    result=attach_column_structure(result,canopy_structure,geometry)
    result=attach_column_capacity(result,capacity_state)
    return finish_column_tile(attach_column_irrigation(result,irrigation),arrays[2],arrays[3],forcing,tile_parameters,'jax')

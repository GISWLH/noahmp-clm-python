"""Native four-layer field-capacity tile outlet and real soil withdrawal."""
from dataclasses import astuple
from functools import lru_cache
import numpy as np
from ..physics.tile_drainage import TileDrainageParameters


def pack_tile_drainage(liquid,ice,dz,temperature,dt,parameters=TileDrainageParameters(),dtype=np.float64):
    if any(np.shape(v)!=(4,) for v in (liquid,ice,dz,temperature)):
        raise ValueError('Native tile drainage requires four layers')
    return np.column_stack((liquid,ice,dz,temperature)).astype(dtype),np.asarray([dt],dtype),np.asarray(astuple(parameters),dtype)


def validate_tile_drainage(layers,u,p,xp):
    n=len(layers);dtype=layers.dtype
    if (layers.shape!=(n,4,4) or u.shape!=(n,1) or p.shape!=(n,3)
        or dtype not in (xp.float32,xp.float64) or u.dtype!=dtype or p.dtype!=dtype
        or not all(bool(xp.all(xp.isfinite(v))) for v in (layers,u,p))):
        raise ValueError('Invalid tile array shapes, precision or finite inputs')
    depth=layers[:,0,2]
    for k in range(1,4):depth=depth+layers[:,k,2]
    bad=xp.any((layers[:,:,0]<0.)|(layers[:,:,1]<0.)|(layers[:,:,2]<=0.)|
        (layers[:,:,0]+layers[:,:,1]>1000.*layers[:,:,2])|(layers[:,:,3]<150.)|(layers[:,:,3]>400.),axis=1)
    bad|=(u[:,0]<=0.)|(u[:,0]>86400.)|(p[:,0]<=0.)|(p[:,0]>depth)|(p[:,1]<0.)|(p[:,1]>1.)|(p[:,2]<0.)
    if bool(xp.any(bad)):raise ValueError('Invalid tile soil, depth, rate or timestep')


def tile_drainage_kernel_jax(layers,u,p):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;bottom=j.asarray(0.,u.dtype);index=j.asarray(0,j.int32)
    for k in range(4):
        bottom=s(bottom+layers[k,2]);index=index+(p[0]>bottom).astype(j.int32)
    water,ice,z,t=layers[index]
    retained=j.maximum(0.,s(s(s(1000.*p[1])*z)-ice))
    excess=j.minimum(water,j.maximum(0.,s(water-retained)))
    eligible=(t>=273.15)&(ice<=1e-9);capacity=s(p[2]*s(u[0]/86400.))
    removed=j.where(eligible,j.minimum(excess,capacity),0.)
    heat=s(removed*s(334000.+s(4180.*s(t-273.15))))
    demand=j.where(j.arange(4)==index,removed,0.)
    return j.concatenate((demand,j.asarray([index,eligible,excess,capacity,removed,heat],u.dtype)))


def unpack_tile_drainage(out,xp):
    if not bool(xp.all(xp.isfinite(out))):raise RuntimeError('Nonfinite tile outlet')
    return dict(demand_kgm2=out[:,:4],layer_index=out[:,4].astype(xp.int32),eligible=out[:,5]>0.,
        excess_kgm2=out[:,6],capacity_kgm2=out[:,7],drained_kgm2=out[:,8],drained_enthalpy_jm2=out[:,9])


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(tile_drainage_kernel_jax))


def run_tile_drainage_jax(layers,dt,parameters):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(layers,dt,parameters)));validate_tile_drainage(*arrays,j)
    return unpack_tile_drainage(_compiled()(*arrays),j)


def _step_soil(c,p,o,dt,parameters,backend):
    """Derive phases on device, then remove demand with the native M/H operator."""
    if backend=='jax':
        import jax.numpy as xp
        from .enthalpy import run_soil_thermodynamics_jax as inverse
        from .soil_transport import run_soil_transport_jax as transfer
        run=run_tile_drainage_jax
    else:
        import cupy as xp
        from .cuda.enthalpy import run_soil_thermodynamics_cuda as inverse
        from .cuda.soil_transport import run_soil_transport_cuda as transfer
        from .cuda.tile_drainage import run_tile_drainage_cuda as run
    c,p,o=map(xp.asarray,(c,p,o));dtype=c.dtype;n=len(c)
    dt=xp.asarray(dt,dtype);parameters=xp.asarray(parameters,dtype)
    if c.shape!=(n,4,3) or p.shape!=(n,5) or o.shape!=(n,3) or dt.shape!=(n,1) or parameters.shape!=(n,3):
        raise ValueError('Tile soil arrays require native four-layer shapes')
    if bool(xp.any(parameters[:,1]>p[:,0])):raise ValueError('Tile field capacity exceeds soil porosity')
    thermodynamic_parameters=p[:,xp.asarray([0,1,4])]
    therm=inverse(xp.concatenate((c,xp.broadcast_to(thermodynamic_parameters[:,None,:],(n,4,3))),axis=2).reshape(-1,6),xp.repeat(o[:,1],4),inverse=True)
    t=therm['temperature_k'].reshape(n,4);liquid=xp.where(t>=273.15,c[:,:,0],xp.clip(therm['liquid_kgm2'].reshape(n,4),0.,c[:,:,0]))
    out=run(xp.stack((liquid,c[:,:,0]-liquid,c[:,:,2],t),axis=2),dt,parameters)
    options=xp.stack((xp.ones(n,xp.int32),o[:,1],o[:,2]),axis=1)
    inputs=xp.concatenate((xp.zeros((n,2),dtype),dt,out['demand_kgm2']),axis=1)
    actual=transfer(c,p,options,inputs)
    if not bool(xp.all(actual['valid'])):raise RuntimeError('Tile soil withdrawal failed')
    heat=actual['withdrawal_enthalpy_jm2'][:,0]
    for k in range(1,4):heat=heat+actual['withdrawal_enthalpy_jm2'][:,k]
    out.update(drained_enthalpy_jm2=heat,cells=actual['cells'])
    return out


def step_tile_soil_jax(c,p,o,dt,parameters):
    return _step_soil(c,p,o,dt,parameters,'jax')


def finish_column_tile(result,p,o,forcing,parameters,backend):
    """Drain before BGC climate diagnosis and append an explicit deep outlet."""
    if parameters is None:return result
    if backend=='jax':
        import jax.numpy as xp
        from .enthalpy import run_soil_thermodynamics_jax as inverse
    else:
        import cupy as xp
        from .cuda.enthalpy import run_soil_thermodynamics_cuda as inverse
    from .clm_bgc_column import _sum_rows,_set
    from .soil_transport import compact_soil_parameters
    old=result['soil_cells'];n=len(old)
    options=xp.stack((xp.ones(n,xp.int32),o[:,1],o[:,8]),axis=1)
    tile=_step_soil(old,compact_soil_parameters(p,xp),options,forcing[:,:1],parameters,backend)
    new=tile['cells'];mass=tile['drained_kgm2'];heat=tile['drained_enthalpy_jm2'];d=result['diagnostics']
    # Retain the real transfer roundoff in the extensive ledger. The full
    # independent verification separately sums all old/new stores in float64.
    dm=_sum_rows(new[:,:,0]-old[:,:,0],xp);dh=_sum_rows(new[:,:,1]-old[:,:,1],xp)
    d['water_residual_kgm2']=(d['water_residual_kgm2']+dm)+mass
    d['energy_residual_jm2']=(d['energy_residual_jm2']+dh)+heat
    d['runoff_mm']=d['runoff_mm']+mass;d['runoff_deep_mm']=d['runoff_deep_mm']+mass
    d['runoff_enthalpy_jm2']=d['runoff_enthalpy_jm2']+heat
    t=inverse(xp.concatenate((new[:,0,:],p[:,xp.asarray([0,1,6])]),axis=1),o[:,1],inverse=True)['temperature_k']
    ts=xp.where(result['options'][:,0]>0,result['storage'][:,2],t)
    result['storage']=_set(result['storage'],2,ts,backend);d['ts']=ts
    d.update(tile_drainage_mm=mass,tile_drainage_enthalpy_jm2=heat)
    result.update(soil_cells=new,tile_drainage=tile,tile_parameters=xp.asarray(parameters,old.dtype))
    return result

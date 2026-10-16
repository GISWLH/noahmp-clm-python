"""Native bulk CH4 carbon store, linear competition and pressure release."""
from dataclasses import astuple
from functools import lru_cache
import numpy as np
from ..physics.wetland import WetlandParameters

DIAGNOSTICS=('anaerobic_fraction','production_fraction','diffusion_rate_s','oxidation_rate_s',
    'methane_capacity_gc_m2','ebullition_active','methane_production_gc_m2','oxidized_gc_m2',
    'diffused_gc_m2','ebullition_gc_m2','co2_gc_m2','methane_emission_gc_m2','methane_emission_mgch4_m2',
    'carbon_residual_gc_m2','methane_residual_gc_m2','carbon_bound_gc_m2')


def pack_wetland(state,donor,temperature,liquid_theta,ice_theta,porosity,pressure,dt,*,parameters=WetlandParameters(),dtype=np.float64):
    return (np.asarray(astuple(state),dtype),np.asarray([donor,temperature,liquid_theta,ice_theta,porosity,pressure,dt],dtype),
        np.asarray(astuple(parameters),dtype))


def validate_wetland(state,u,p,xp):
    n=len(state);dtype=state.dtype
    if (state.shape!=(n,1) or u.shape!=(n,7) or p.shape!=(n,13) or dtype not in (xp.float32,xp.float64)
        or u.dtype!=dtype or p.dtype!=dtype or not all(bool(xp.all(xp.isfinite(v))) for v in (state,u,p))):
        raise ValueError('Invalid wetland shapes, precision or nonfinite values')
    bad=(state[:,0]<0.)|(u[:,0]<0.)|(u[:,1]<150.)|(u[:,1]>350.)|(u[:,2]<0.)|(u[:,3]<0.)|(u[:,4]<=0.)|(u[:,4]>1.)
    bad|=(u[:,2]+u[:,3]>u[:,4]+8.*xp.finfo(dtype).eps)|(u[:,5]<20000.)|(u[:,5]>120000.)|(u[:,6]<=0.)|(u[:,6]>86400.)
    bad|=(p[:,0]<=0.)|(p[:,1]<0.)|(p[:,1]>=1.)|(p[:,2]<0.)|(p[:,2]>1.)|(p[:,6]<=0.)|xp.any(p[:,7:10]<0.,axis=1)|(p[:,12]<=0.)|(p[:,12]>1.)
    for k in (3,4,10):bad|=(p[:,k]<1.)|(p[:,k]>5.)
    for k in (5,11):bad|=(p[:,k]<250.)|(p[:,k]>330.)
    if bool(xp.any(bad)):raise ValueError('Invalid wetland state, climate or parameters')


def wetland_kernel_jax(old,u,p):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    air=j.maximum(s(s(u[4]-u[2])-u[3]),0.);sat=j.minimum(s(u[2]/u[4]),1.)
    # Exact saturation is a physical endpoint. Approximate device division
    # must not create a spurious oxic fraction when liquid == pore capacity.
    anaerobic=j.where(u[2]>=u[4],1.,j.maximum(s(s(sat-p[1])/s(1.-p[1])),0.))
    ratio=s(p[2]*j.exp(s(j.log(s(p[3]/p[4]))*s(s(u[1]-p[5])/10.))))
    fraction=j.where(u[1]>273.15,s(s(anaerobic*ratio)/s(1.+ratio)),0.)
    capacity=s(s(u[2]+s(air/p[6]))*p[0])
    aqueous=s(s(p[7]*u[2])*u[2])
    gaseous=s(s(p[8]*j.exp(s(j.log(air)*(10./3.))))/s(u[4]*u[4]))
    conductance=s(s(aqueous+s(gaseous/p[6]))/s(.5*p[0]))
    kd=j.where(capacity>0.,s(conductance/j.where(capacity>0.,capacity,1.)),0.)
    ko=j.where((u[1]>273.15)&(u[2]>0.),s(s(p[9]*s(1.-anaerobic))*j.exp(s(j.log(p[10])*s(s(u[1]-p[11])/10.)))),0.)
    limit=s(s(s(s(12.*p[6])*p[12])*u[5])/s(8.31446261815324*u[1]));maximum=s(limit*capacity)
    active=(u[1]>273.15)&(anaerobic>0.)&(u[2]>0.)
    produced=s(u[0]*fraction);k=s(ko+kd);x=s(k*u[6]);survival=j.exp(-x);lost_old=-j.expm1(-x)
    small=s(x*s(.5+s(x*s(-1./6.+s(x*s(1./24.-s(x/120.)))))))
    lost_new=j.where(x<1e-4,small,s(1.-s(lost_old/j.where(x>0.,x,1.))))
    end=s(s(old[0]*survival)+s(produced*s(1.-lost_new)))
    loss=s(s(old[0]*lost_old)+s(produced*lost_new))
    oxidized=j.where(k>0.,s(s(loss*ko)/j.where(k>0.,k,1.)),0.);diffused=s(loss-oxidized)
    bubble=j.where(active,j.maximum(s(end-maximum),0.),0.);end=s(end-bubble);emitted=s(diffused+bubble)
    co2=s(s(u[0]-produced)+oxidized)
    cr=s(s(s(s(end-old[0])+co2)+emitted)-u[0]);mr=s(s(s(s(end-old[0])+oxidized)+emitted)-produced)
    scale=s(s(s(s(end+old[0])+co2)+emitted)+u[0]);bound=s(128.*j.finfo(old.dtype).eps*j.maximum(scale,1.))
    valid=(abs(cr)<=bound)&(abs(mr)<=bound)&(end>=0.)
    return j.asarray([end,anaerobic,fraction,kd,ko,maximum,active,produced,oxidized,diffused,bubble,co2,emitted,
        s(emitted*(16000./12.)),cr,mr,bound,valid],old.dtype)


def unpack_wetland(out,xp):
    result=dict(state=out[:,:1]);result.update(zip(DIAGNOSTICS,out[:,1:17].T))
    result['ebullition_active']=result['ebullition_active']>0.
    if not bool(xp.all(xp.isfinite(out))) or not bool(xp.all(out[:,-1]>0.)):
        raise RuntimeError('Native wetland carbon domain or balance failed')
    return result


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(wetland_kernel_jax))


def run_wetland_jax(state,inputs,parameters):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(state,inputs,parameters)));validate_wetland(*arrays,j)
    return unpack_wetland(_compiled()(*arrays),j)


def column_wetland_environment(g,soil_parameters,temperatures,liquid,pressure,parameters,backend):
    """Depth-average actual device soil; no host physical tendencies."""
    from .clm_bgc_column import _ratio,_sum_rows
    if backend=='jax':import jax.numpy as xp
    else:import cupy as xp
    n=len(g);depth=parameters[:,0];top=xp.zeros(n,g.dtype);thickness=[]
    for k in range(4):
        thickness.append(xp.minimum(g[:,k,2],xp.maximum(depth-top,0.)));top=top+g[:,k,2]
    dz=xp.stack(thickness,axis=1);weights=_ratio(dz,g[:,:,2],backend)
    phase=xp.where(temperatures>=273.15,g[:,:,0],xp.clip(liquid,0.,g[:,:,0]))
    lt=_ratio(_sum_rows(phase*weights,xp),1000.*depth,backend)
    it=_ratio(_sum_rows((g[:,:,0]-phase)*weights,xp),1000.*depth,backend)
    t=_ratio(_sum_rows(temperatures*dz,xp),depth,backend)
    saturation=xp.minimum(_ratio(lt,soil_parameters[:,0],backend),1.)
    fraction=xp.where(lt>=soil_parameters[:,0],1.,xp.maximum(_ratio(saturation-parameters[:,1],1.-parameters[:,1],backend),0.))
    return dict(temperature_k=t,liquid_theta=lt,ice_theta=it,porosity=soil_parameters[:,0],pressure_pa=pressure,
        anaerobic_fraction=fraction,oxygen_factor=1.-.8*fraction)

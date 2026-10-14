"""Native seasonal GDD controller; no host physics in the device path."""
from dataclasses import astuple
from functools import lru_cache
import numpy as np
from ..physics.phenology import PhenologyParameters


def pack_phenology(state,soil_temperature,daylength,increasing,winter,dt,parameters=None,dtype=np.float64):
    p=PhenologyParameters() if parameters is None else parameters
    return (np.asarray([state.gdd,state.leaf_fraction,state.remaining_seconds],dtype),
        np.asarray([state.phase,state.armed],np.int32),np.asarray([soil_temperature,daylength,dt],dtype),
        np.asarray([increasing,winter],np.int32),np.asarray(astuple(p),dtype))


def validate_phenology(s,o,u,c,p,xp):
    n=len(s)
    if (s.shape!=(n,3) or o.shape!=(n,2) or u.shape!=(n,3) or c.shape!=(n,2) or p.shape!=(n,6)
        or s.dtype not in (xp.float32,xp.float64) or u.dtype!=s.dtype or p.dtype!=s.dtype
        or o.dtype.kind not in 'iu' or c.dtype.kind not in 'iu'):
        raise ValueError('Invalid phenology array shapes or precision')
    if not all(bool(xp.all(xp.isfinite(v))) for v in (s,u,p)):raise ValueError('Nonfinite phenology values')
    bad_state=((s[:,0]<0.)|(s[:,1]<0.)|(s[:,1]>1.)|(s[:,2]<0.)|
        (o[:,0]<0)|(o[:,0]>3)|(o[:,1]<0)|(o[:,1]>1))
    bad_climate=((u[:,0]<150.)|(u[:,0]>400.)|(u[:,1]<0.)|(u[:,1]>86400.)|(u[:,2]<=0.)|(u[:,2]>86400.))
    bad_parameters=((p[:,0]<=0.)|(p[:,1]<1.)|(p[:,2]<1.)|(p[:,3]<=0.)|(p[:,3]>=86400.)|(p[:,4]<0.)|(p[:,5]<p[:,4]))
    inconsistent=(((o[:,0]==0)&((s[:,1]!=0.)|(s[:,2]!=0.)))|
        ((o[:,0]==2)&((s[:,1]!=1.)|(s[:,2]!=0.)))|((o[:,0]!=0)&(o[:,1]!=0))|
        ((o[:,0]==1)&(s[:,2]>p[:,1]*86400.))|((o[:,0]==3)&(s[:,2]>p[:,2]*86400.)))
    if bool(xp.any(bad_state|bad_climate|bad_parameters|inconsistent)) or bool(xp.any((c<0)|(c>1))):
        raise ValueError('Invalid phenology state, climate or parameters')


def phenology_kernel_jax(state,options,u,calendar,p):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    gdd,leaf,remaining=state;phase,armed=options;increasing,winter=calendar
    dormant=phase==0;accumulate=dormant&(increasing>0)&((armed>0)|(winter>0))
    gdd=j.where(dormant&((winter>0)|(increasing==0)),0.,gdd)
    armed=j.where(dormant,j.where(increasing==0,0,j.where(winter>0,1,armed)),armed)
    increment=j.where(accumulate,s(j.maximum(0.,s(u[0]-273.15))*s(u[2]/86400.)),0.)
    gdd=s(gdd+increment);onset=accumulate&(gdd>p[0])
    phase=j.where(onset,1,phase);remaining=j.where(onset,s(p[1]*86400.),remaining)
    gdd=j.where(onset,0.,gdd);armed=j.where(onset,0,armed)
    offset=(phase==2)&(increasing==0)&(u[1]<p[3])
    phase=j.where(offset,3,phase);remaining=j.where(offset,s(p[2]*86400.),remaining)
    moving=(phase==1)|(phase==3)
    remaining=j.where(moving,j.maximum(0.,s(remaining-u[2])),remaining)
    leaf=j.where(phase==1,s(1.-remaining/s(p[1]*86400.)),j.where(phase==3,remaining/s(p[2]*86400.),leaf))
    phase=j.where((phase==1)&(remaining==0.),2,j.where((phase==3)&(remaining==0.),0,phase))
    lai=s(p[4]+s(s(p[5]-p[4])*leaf))
    return j.stack((gdd,leaf,remaining)),j.stack((phase,armed)),j.stack((lai,j.maximum(0.,s(leaf-state[1])),
        j.maximum(0.,s(state[1]-leaf)),increment,onset.astype(state.dtype),offset.astype(state.dtype)))


def unpack_phenology(state,flags,diagnostics):
    return dict(state=state,flags=flags,lai=diagnostics[:,0],leaf_gain_fraction=diagnostics[:,1],
        leaf_loss_fraction=diagnostics[:,2],gdd_increment=diagnostics[:,3],
        onset_triggered=diagnostics[:,4]>0.,offset_triggered=diagnostics[:,5]>0.)


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(phenology_kernel_jax))


def run_phenology_jax(s,o,u,c,p):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(s,o,u,c,p)));validate_phenology(*arrays,j)
    return unpack_phenology(*_compiled()(*arrays))

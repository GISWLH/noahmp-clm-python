"""Native crop thermal clock, seed storage, emergence and conservative harvest.

Seven input arrays: crop[5], integer phase[1], plant[8], climate[4],
integer planting/harvest events[2], crop parameters[9], plant parameters[21].
The crop state stores air/soil degree days, age, seed C and seed N.
"""
from dataclasses import astuple
from functools import lru_cache
import numpy as np
from ..physics.crop import CropParameters
from ..physics.plant_cn import PlantParameters
from .plant_cn import pack_plant,validate_plant,_arrays
from .canopy_energy import _sum

DIAGNOSTICS=('seed_input_gc_m2','seed_input_gn_m2','harvest_gc_m2','harvest_gn_m2',
    'planted','emerged','harvested','harvest_reason','growing','target_lai',
    'carbon_residual_gc_m2','nitrogen_residual_gn_m2','carbon_bound_gc_m2','nitrogen_bound_gn_m2','valid')


def pack_crop(state,plant,air_temperature,soil_temperature,dt,*,planting=False,force_harvest=False,
              vegetated_fraction=1.,parameters=CropParameters(),plant_parameters=PlantParameters(),dtype=np.float64):
    b,pp=pack_plant(plant,plant_parameters,dtype=dtype)
    return (np.asarray([state.air_degree_days,state.soil_degree_days,state.age_days,state.seed_carbon,state.seed_nitrogen],dtype),
        np.asarray([state.phase],np.int32),b,np.asarray([air_temperature,soil_temperature,dt,vegetated_fraction],dtype),
        np.asarray([planting,force_harvest],np.int32),np.asarray(astuple(parameters),dtype),pp)


def validate_crop(s,o,b,u,e,p,pp,xp):
    _arrays((5,8,4,9,21),(s,b,u,p,pp),xp);validate_plant(b,pp,xp);n=len(s)
    if o.shape!=(n,1) or e.shape!=(n,2) or o.dtype.kind not in 'iu' or e.dtype.kind not in 'iu':
        raise ValueError('Crop phase and management events must be integer arrays')
    if bool(xp.any((o<0)|(o>3))) or bool(xp.any((e<0)|(e>1))):raise ValueError('Invalid crop phase or event')
    phase=o[:,0]
    bad=(xp.any(s<0.,axis=1)|((s[:,3]==0.)&(s[:,4]!=0.))|
        ((phase!=1)&xp.any(s[:,3:]!=0.,axis=1))|((phase<2)&xp.any(b!=0.,axis=1))|
        ((e[:,0]==1)&((phase!=0)|(e[:,1]==1)))|((e[:,1]==1)&(phase==0)))
    climate=((u[:,0]<150.)|(u[:,0]>350.)|(u[:,1]<150.)|(u[:,1]>350.)|(u[:,2]<=0.)|
        (u[:,2]>86400.)|(u[:,3]<=0.)|(u[:,3]>1.))
    parameters=((p[:,0]<250.)|(p[:,0]>310.)|(p[:,1]<=0.)|(p[:,2]<=0.)|
        (p[:,3]<=0.)|(p[:,3]>=p[:,4])|(p[:,4]>=1.)|(p[:,5]<=0.)|(p[:,6]<=0.)|
        (p[:,7]<0.)|(p[:,7]>1.)|(p[:,8]<=0.))
    if bool(xp.any(bad|climate|parameters)):raise ValueError('Invalid native crop state, weather or parameters')


def crop_kernel_jax(old,options,b,u,events,p,pp):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;row=lambda *v:j.asarray(v,old.dtype)
    tc=lambda a:s(_sum(a[:3])+a[6]);tn=lambda a:s(_sum(a[3:6])+a[7])
    oldc=s(tc(b)+old[3]);oldn=s(tn(b)+old[4]);planted=events[0]>0
    phase=j.where(planted,1,options[0]);active=phase>0
    ic=j.where(planted,p[6],0.);inn=j.where(planted,s(p[6]/pp[3]),0.)
    seedc=j.where(planted,ic,old[3]);seedn=j.where(planted,inn,old[4])
    air=j.where(planted,0.,old[0]);soil=j.where(planted,0.,old[1]);age=j.where(planted,0.,old[2])
    da=s(s(j.clip(s(u[0]-p[0]),0.,p[1])*u[2])/86400.)
    ds=s(s(j.clip(s(u[1]-p[0]),0.,p[1])*u[2])/86400.)
    air=s(air+j.where(active,da,0.));soil=s(soil+j.where(active,ds,0.));age=s(age+j.where(active,s(u[2]/86400.),0.))
    reason=j.where(active,j.where(events[1]>0,3,j.where(air>=p[2],1,j.where(age>=p[5],2,0))),0)
    harvest=reason>0;emerge=(phase==1)&(~harvest)&(soil>=s(p[3]*p[2]))
    ec=j.where(harvest,s(p[7]*s(b[0]+b[1])),0.);en=j.where(harvest,s(p[7]*s(b[3]+b[4])),0.)
    lc=j.where(harvest,s(s(tc(b)+seedc)-ec),0.);ln=j.where(harvest,s(s(tn(b)+seedn)-en),0.)
    ci=j.zeros(7,old.dtype).at[:3].set(s(lc*pp[9:12]));ni=j.zeros(7,old.dtype).at[:3].set(s(ln*pp[9:12]))
    new=b.at[0].set(s(b[0]+j.where(emerge,seedc,0.))).at[3].set(s(b[3]+j.where(emerge,seedn,0.)))
    new=j.where(harvest,j.zeros(8,old.dtype),new)
    seedc=j.where(harvest|emerge,0.,seedc);seedn=j.where(harvest|emerge,0.,seedn)
    phase=j.where(harvest,0,j.where(emerge,2,phase))
    grain=(phase==2)&((air>=s(p[4]*p[2]))|(s(s(new[0]*pp[19])/u[3])>=p[8]))
    phase=j.where(grain,3,phase);growing=(phase==2)|(phase==3)
    newc=s(tc(new)+seedc);newn=s(tn(new)+seedn)
    cr=s(s(s(s(newc-oldc)+ec)+_sum(ci))-ic);nr=s(s(s(s(newn-oldn)+en)+_sum(ni))-inn)
    eps=j.finfo(old.dtype).eps
    cb=s(128.*eps*j.maximum(s(s(s(s(newc+oldc)+ec)+_sum(ci))+ic),1.))
    nb=s(128.*eps*j.maximum(s(s(s(s(newn+oldn)+en)+_sum(ni))+inn),1.))
    valid=(abs(cr)<=cb)&(abs(nr)<=nb)&j.all(new>=0.)&(seedc>=0.)&(seedn>=0.)
    return j.concatenate((row(air,soil,age,seedc,seedn,phase),new,ci,ni,
        row(ic,inn,ec,en,planted,emerge,harvest,reason,growing,j.where(growing,p[8],0.),cr,nr,cb,nb,valid)))


def unpack_crop(output,xp):
    out=dict(state=output[:,:5],phase=output[:,5:6].astype(xp.int32),plant_state=output[:,6:14],
        litter_carbon_gc_m2=output[:,14:21],litter_nitrogen_gn_m2=output[:,21:28])
    out.update(zip(DIAGNOSTICS,output[:,28:].T))
    for key in ('planted','emerged','harvested','growing'):out[key]=out[key]>0.
    out['harvest_reason']=out['harvest_reason'].astype(xp.int32)
    if not bool(xp.all(xp.isfinite(output))) or not bool(xp.all(out['valid']>0.)):
        raise RuntimeError('Native crop C/N budget or state failed')
    return out


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(crop_kernel_jax))


def run_crop_jax(state,phase,plant,climate,events,parameters,plant_parameters):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(state,phase,plant,climate,events,parameters,plant_parameters)))
    validate_crop(*arrays,j)
    return unpack_crop(_compiled()(*arrays),j)

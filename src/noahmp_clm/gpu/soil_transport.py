"""Native extensive soil water transfers and advected enthalpy, four layers.

Modes: 0 top-down water admission, 1 prescribed liquid withdrawal, 2 Campbell
gravity drainage. These transfer primitives do not implement Richards or VIC.
Pore capacities use liquid-equivalent total water, as in the NumPy reference.
"""
from functools import lru_cache
import numpy as np
from .enthalpy import invert_soil_enthalpy_jax


def pack_soil_transport(state,mode,*,mass=0.,enthalpy=0.,dt=1800.,demand=None,
                        permeability_option=1,dtype=np.float64):
    if len(state.water)!=4:raise ValueError('Soil transport requires four layers')
    c=np.column_stack((state.water,state.enthalpy,state.dz)).astype(dtype)
    p=np.asarray([state.params.smcmax,state.params.bexp,state.params.smcref,state.params.satdk],dtype)
    o=np.asarray([mode,state.freezing_option,permeability_option],np.int32)
    u=np.asarray([mass,enthalpy,dt,*([0.]*4 if demand is None else demand)],dtype)
    return c,p,o,u


def validate_soil_transport(c,p,o,u,xp):
    n=len(c)
    if c.shape!=(n,4,3) or p.shape!=(n,4) or o.shape!=(n,3) or u.shape!=(n,7):
        raise ValueError('Soil transfer shapes require (n,4,3), (n,4), (n,3), (n,7)')
    if c.dtype not in (xp.float32,xp.float64) or p.dtype!=c.dtype or u.dtype!=c.dtype or o.dtype.kind not in 'iu':
        raise ValueError('Soil transfers require matching float32/float64 and integer options')
    if not all(bool(xp.all(xp.isfinite(v))) for v in (c,p,u)):
        raise ValueError('Nonfinite soil transfer')
    capacity=1000.*p[:,:1]*c[:,:,2]
    rounding=8.*xp.finfo(c.dtype).eps*xp.maximum(1.,capacity)
    if (bool(xp.any((c[:,:,0]<0.)|(c[:,:,2]<=0.)|(c[:,:,0]>capacity+rounding)))
            or bool(xp.any((p[:,0]<=0.)|(p[:,0]>=1.)|(p[:,1]<=0.)|(p[:,2]<0.)|(p[:,2]>p[:,0])|(p[:,3]<0.)))
            or bool(xp.any((o[:,0]<0)|(o[:,0]>2)|(o[:,1]<1)|(o[:,1]>2)|(o[:,2]<1)|(o[:,2]>2)))
            or bool(xp.any((u[:,0]<0.)|(u[:,2]<=0.)|((u[:,0]==0.)&(u[:,1]!=0.))))
            or bool(xp.any(u[:,3:]<0.))):
        raise ValueError('Invalid soil state, water transfer or hydraulic parameters')


def unpack_soil_transport(out):
    return dict(cells=out[:,:12].reshape(-1,4,3),temperature_offset_k=out[:,12:16],
        temperature_k=out[:,12:16]+273.15,liquid_kgm2=out[:,16:20],ice_kgm2=out[:,20:24],
        incoming=out[:,24:26],runoff=out[:,26:28],withdrawal_mass_kgm2=out[:,28:32],
        withdrawal_enthalpy_jm2=out[:,32:36],interface_mass_kgm2=out[:,36:40],
        interface_enthalpy_jm2=out[:,40:44],bottom_outflow=out[:,44:46],valid=out[:,46]>0.)


def soil_transport_kernel_jax(c,p,o,u):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    por,b,field,satdk=p;mode,phase,permeability=o
    w,h,z=c.T
    initial=invert_soil_enthalpy_jax(w,h,z,por,b,phase)
    ok=j.all(initial[4])
    zeros=j.zeros((4,),c.dtype);pair=j.zeros((2,),c.dtype)
    def admission(_):
        specific=u[1]/j.where(u[0]>0.,u[0],1.)
        def fill(k,carry):
            water,heat,left=carry
            accepted=j.minimum(left,j.maximum(0.,s(s(1000.*por)*z[k])-water[k]))
            return water.at[k].add(accepted),heat.at[k].add(s(accepted*specific)),s(left-accepted)
        nw,nh,left=lax.fori_loop(0,4,fill,(w,h,u[0]))
        return nw,nh,u[:2],j.stack((left,s(left*specific))),zeros,zeros,zeros,zeros,pair,ok
    def withdrawal(_):
        demand=u[3:7];available=initial[3]
        # Inverse H has a finite storage-precision error. This bounded check
        # accepts that representation mismatch, never changes requested flux.
        rounding=8.*j.finfo(c.dtype).eps*j.maximum(w,1.)
        valid=ok&j.all((demand<=available+rounding)&(demand<=w))
        heat=s(demand*s(334000.+s(4180.*initial[0])))
        return s(w-demand),s(h-heat),pair,pair,demand,heat,zeros,zeros,pair,valid
    def drainage(_):
        def transfer(k,carry):
            water,heat,moved,carried,valid=carry
            inv=invert_soil_enthalpy_jax(water[k],heat[k],z[k],por,b,phase)
            liquid=inv[3];ice=s(water[k]-liquid)
            capacity=s(s(1000.*por)*z[k])
            wet=j.clip(water[k]/capacity,0.,1.)
            frozen=j.where(permeability==1,1.,j.exp(s(-7.*ice)/capacity))
            hydraulic=s(s(s(s(1000.*u[2])*satdk)*frozen)*j.power(wet,s(s(2.*b)+3.)))
            excess=j.maximum(0.,water[k]-s(s(1000.*field)*z[k]))
            amount=j.minimum(j.minimum(hydraulic,excess),liquid)
            next_index=j.minimum(k+1,3)
            receiver=j.maximum(0.,s(s(1000.*por)*z[next_index])-water[next_index])
            amount=j.where(k<3,j.minimum(amount,receiver),amount)
            energy=s(amount*s(334000.+s(4180.*inv[0])))
            water=water.at[k].add(-amount);heat=heat.at[k].add(-energy)
            water=water.at[next_index].add(j.where(k<3,amount,0.))
            heat=heat.at[next_index].add(j.where(k<3,energy,0.))
            return water,heat,moved.at[k].set(amount),carried.at[k].set(energy),valid&inv[4]
        nw,nh,mass,heat,valid=lax.fori_loop(0,4,transfer,(w,h,zeros,zeros,ok))
        return nw,nh,pair,pair,zeros,zeros,mass,heat,j.stack((mass[3],heat[3])),valid
    nw,nh,enter,runoff,take,takeh,flow,flowh,bottom,valid=lax.switch(mode,(admission,withdrawal,drainage),None)
    final=invert_soil_enthalpy_jax(nw,nh,z,por,b,phase)
    valid=valid&j.all(final[4])&j.all(nw>=0.)
    cells=j.stack((nw,nh,z),axis=1)
    return j.concatenate((cells.ravel(),final[0],final[3],s(nw-final[3]),enter,runoff,take,takeh,flow,flowh,bottom,j.asarray([valid],c.dtype)))


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(soil_transport_kernel_jax))


def run_soil_transport_jax(c,p,o,u):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(c,p,o,u)));validate_soil_transport(*arrays,j)
    out=_compiled()(*arrays);r=unpack_soil_transport(out)
    if not bool(j.all(r['valid'])) or not bool(j.all(j.isfinite(out))):
        raise ValueError('Soil transfer exceeds available liquid or thermodynamic physical domain')
    return r

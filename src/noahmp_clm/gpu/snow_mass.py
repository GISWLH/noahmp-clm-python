"""Native extensive snow transfers, with separate drainage/runoff ledgers.

Cell columns: mass, enthalpy, thickness, radius, BC and four dust masses.
Modes: 0 regrid, 1 snowfall, 2 rain, 3 liquid drainage, 4 ice sublimation.
This component does not solve energy fluxes or evolve snow microstructure.
"""
from functools import lru_cache
import numpy as np


def pack_snow_transfer(pack,mode,*,amount=0.,temperature=273.15,density=100.,radius=54.5,
                       retention=.033,impermeable_porosity=.05,
                       scavenging=(.2,.02,.02,.01,.01),impurities=(0.,0.,0.,0.,0.),dtype=np.float64):
    cells=np.zeros((5,9),dtype);cells[:,3]=54.5;n=len(pack.mass)
    cells[:n,:4]=np.column_stack((pack.mass,pack.enthalpy,pack.thickness,pack.radius_um))
    cells[:n,4:]=pack.impurities
    args=np.array([amount,temperature-273.15,density,radius,retention,impermeable_porosity,*scavenging,*impurities],dtype)
    return cells,np.asarray(n,np.int32),np.asarray(mode,np.int32),args


def validate_snow_transfer(c,n,mode,u,xp):
    if c.ndim!=3 or c.shape[1:]!=(5,9) or n.shape!=(len(c),) or mode.shape!=n.shape or u.shape!=(len(c),16):
        raise ValueError('Snow transfer shapes must be (n,5,9), (n,), (n,), (n,16)')
    if c.dtype not in (xp.float32,xp.float64) or u.dtype!=c.dtype or n.dtype.kind not in 'iu' or mode.dtype.kind not in 'iu':
        raise ValueError('Snow transfer requires matching floats and integer counts/modes')
    if not bool(xp.all(xp.isfinite(c))) or not bool(xp.all(xp.isfinite(u))):
        raise ValueError('Nonfinite snow transfer input')
    active=xp.arange(5)[None,:]<n[:,None]
    if bool(xp.any((n<0)|(n>5)|(mode<0)|(mode>4))) or bool(xp.any(
        (active&((c[:,:,0]<=0)|(c[:,:,2]<=0)|(c[:,:,3]<=0)))|
        (~active&((c[:,:,0]!=0)|(c[:,:,1]!=0)|(c[:,:,2]!=0))))) or bool(xp.any(c[:,:,4:]<0)) or bool(xp.any((~active[:,:,None])&(c[:,:,4:]!=0))):
        raise ValueError('Invalid snow mass, geometry, tracer or padding')
    if bool(xp.any((u[:,0]<0)|(u[:,4]<0)|(u[:,4]>=1)|(u[:,5]<0)|(u[:,5]>=1))) or bool(xp.any((u[:,6:11]<0)|(u[:,6:11]>1))) or bool(xp.any(u[:,11:]<0)):
        raise ValueError('Invalid snow amount, retention or impurity transfer')
    if bool(xp.any((mode==1)&((u[:,1]<-123.15)|(u[:,1]>0)|(u[:,2]<30)|(u[:,2]>917)|(u[:,3]<=0)))):
        raise ValueError('Invalid snowfall temperature, density or radius')
    if bool(xp.any((mode==2)&((n==0)|(u[:,1]<-73.15)|(u[:,1]>76.85)))):
        raise ValueError('Rain needs existing snow and a 200--350 K temperature')
    supply=xp.sum(c[:,:,0]-xp.clip(c[:,:,1]/334000.,0.,c[:,:,0]),axis=1)
    if bool(xp.any((mode==4)&(u[:,0]>supply+8*xp.finfo(c.dtype).eps*xp.maximum(supply,1.)))):
        raise ValueError('Sublimation exceeds snow ice supply')


def snow_phase_jax(m,h):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    liq=j.clip(h/334000.,0.,m);ice=s(m-liq);safe=j.maximum(m,1e-30)
    offset=j.where(h<0.,h/s(2100.*safe),j.where(h>s(334000.*m),s(h-s(334000.*m))/s(4180.*safe),0.))
    return offset,ice,liq


def snow_regrid_jax(c):
    """Six possible source cells to five depth-based, conservative target cells."""
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    old=j.zeros(7,c.dtype)
    old=lax.fori_loop(0,6,lambda i,x:x.at[i+1].set(s(x[i]+c[i,2])),old)
    depth=old[6];zero=j.zeros_like(depth)
    def row(*v):return j.stack([j.asarray(x,c.dtype) for x in v])
    z=lax.cond(depth<=.03,lambda:row(depth,0,0,0,0),lambda:
      lax.cond(depth<=.04,lambda:row(s(depth/2.),s(depth/2.),0,0,0),lambda:
      lax.cond(depth<=.07,lambda:row(.02,s(depth-.02),0,0,0),lambda:
      lax.cond(depth<=.12,lambda:row(.02,s(s(depth-.02)/2.),s(s(depth-.02)/2.),0,0),lambda:
      lax.cond(depth<=.18,lambda:row(.02,.05,s(depth-.07),0,0),lambda:
      lax.cond(depth<=.29,lambda:row(.02,.05,s(s(depth-.07)/2.),s(s(depth-.07)/2.),0),lambda:
      lax.cond(depth<=.41,lambda:row(.02,.05,.11,s(depth-.18),0),lambda:
      lax.cond(depth<=.64,lambda:row(.02,.05,.11,s(s(depth-.18)/2.),s(s(depth-.18)/2.)),lambda:
      row(.02,.05,.11,.23,s(depth-.41))))))))))
    n=j.sum(z>0.,dtype=j.int32)
    edges=j.zeros(6,c.dtype)
    edges=lax.fori_loop(0,5,lambda i,x:x.at[i+1].set(s(x[i]+z[i])),edges)
    # Both grids have the same physical bottom, independently of summation
    # roundoff in the prescribed target thicknesses.
    edges=edges.at[n].set(depth)
    def source(k,out):
        span=j.maximum(0.,s(j.minimum(edges[1:],old[k+1])-j.maximum(edges[:-1],old[k])))
        span=j.where(j.arange(5)<n,span,0.)
        total=lax.fori_loop(0,5,lambda i,v:s(v+span[i]),zero)
        last=j.max(j.where(span>0.,j.arange(5),-1))
        containing=j.minimum(j.sum((j.arange(5)<n)&(edges[1:]<=old[k]),dtype=j.int32),j.maximum(n-1,0))
        last=j.where(last>=0,last,containing)
        content=c[k].at[2].set(0.).at[3].set(c[k,0]/j.maximum(c[k,3],1e-30))
        def parcel(i,carry):
            values,remaining=carry
            part=j.where(i==last,remaining,j.where((span[i]>0.)&(i<last),s(content*s(span[i]/j.maximum(total,1e-30))),0.))
            part=j.where(c[k,0]>0.,part,0.)
            return values.at[i].set(s(values[i]+part)),s(remaining-part)
        return lax.fori_loop(0,5,parcel,(out,content))[0]
    out=lax.fori_loop(0,6,source,j.zeros((5,9),c.dtype))
    return out.at[:,2].set(z).at[:,3].set(j.where(j.arange(5)<n,out[:,0]/j.maximum(out[:,3],1e-30),54.5)),n


def snow_transfer_kernel_jax(c,n,mode,u):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    ledger=j.zeros((4,7),c.dtype) # incoming, drainage, runoff, sublimation/residue
    def regrid():return (*snow_regrid_jax(j.pad(c,((0,1),(0,0)))),ledger)
    def snowfall():
        fresh=j.concatenate((j.stack((u[0],s(s(u[0]*2100.)*u[1]),u[0]/u[2],u[3])),u[11:]))
        values=j.concatenate((fresh[None,:],c))
        out,nn=snow_regrid_jax(values)
        incoming=j.concatenate((fresh[:2],fresh[4:]))
        return lax.cond(u[0]>0.,lambda:(out,nn,ledger.at[0].set(incoming)),lambda:(c,n,ledger))
    def rain():
        heat=s(u[0]*s(334000.+s(4180.*u[1])))
        out=c.at[0,0].set(s(c[0,0]+u[0])).at[0,1].set(s(c[0,1]+heat))
        return out,n,ledger.at[0,0].set(u[0]).at[0,1].set(heat)
    def drain():
        def layer(idx,carry):
            cells,water,heat,tracer,runoff=carry
            v=cells[idx];v=v.at[0].set(s(v[0]+water)).at[1].set(s(v[1]+heat)).at[4:].set(s(v[4:]+tracer))
            offset,ice,liq=snow_phase_jax(v[0],v[1]);z=j.maximum(v[2],ice/917.);v=v.at[2].set(z)
            pore=j.maximum(0.,s(1.-ice/s(917.*j.maximum(z,1e-30))))
            cap=s(s(s(u[4]*pore)*z)*1000.)
            potential=j.where(ice==0.,liq,j.where(pore>=u[5],j.maximum(0.,s(liq-cap)),0.))
            specific=s(334000.+s(4180.*offset))
            following=cells[j.minimum(idx+1,4)]
            _,ni,_=snow_phase_jax(following[0],following[1])
            blocked=s(1.-ni/s(917.*j.maximum(following[2],1e-30)))<u[5]
            def volume(amount):
                _,ice2,liq2=snow_phase_jax(s(following[0]+amount),s(following[1]+s(amount*specific)))
                return s(ice2/917.+liq2/1000.)
            def bisect(_,bounds):
                low,high=bounds;mid=s(s(low+high)*.5);fits=volume(mid)<=following[2]
                return j.where(fits,mid,low),j.where(fits,high,mid)
            low,_=lax.fori_loop(0,48,bisect,(j.zeros_like(potential),potential))
            accepted=j.where(blocked,0.,j.where(volume(potential)>following[2],low,potential))
            water=j.where((idx+1<n)&(potential>0.),accepted,potential)
            left=s(liq-water);occupied=s(ice/917.+left/1000.)
            overflow=j.where(ice==0.,left,j.where(occupied>z,j.maximum(0.,s(left-s(s(z-ice/917.)*1000.))),0.))
            total=j.where(ice==0.,liq,s(water+overflow));heat=s(water*specific)
            fraction=j.minimum(1.,s(u[6:11]*total)/j.maximum(v[0],1e-30))
            exported=j.where(total==v[0],v[4:],s(v[4:]*fraction))
            split=j.clip(water/j.maximum(total,1e-30),0.,1.)
            tracer=j.where(water==total,exported,j.where(total>0.,s(exported*split),0.))
            run=s(j.concatenate((j.stack((overflow,s(overflow*specific))),s(exported-tracer)))+runoff)
            v=v.at[0].set(s(v[0]-total)).at[1].set(s(v[1]-s(total*specific))).at[4:].set(s(v[4:]-exported))
            v=v.at[1].set(j.where(v[0]==0.,0.,v[1])).at[2].set(j.where(v[0]>0.,v[2],0.))
            return cells.at[idx].set(v),water,heat,tracer,run
        initial=(c,j.zeros((),c.dtype),j.zeros((),c.dtype),j.zeros(5,c.dtype),j.zeros(7,c.dtype))
        values,water,heat,tracer,run=lax.fori_loop(0,n,layer,initial)
        out,nn=snow_regrid_jax(j.pad(values,((0,1),(0,0))))
        return out,nn,ledger.at[1].set(j.concatenate((j.stack((water,heat)),tracer))).at[2].set(run)
    def sublimation():
        def layer(idx,carry):
            values,remaining,actual,energy,residue=carry;v=values[idx]
            offset,ice,_=snow_phase_jax(v[0],v[1]);loss=j.minimum(remaining,ice);heat=s(s(loss*2100.)*offset)
            mass=s(v[0]-loss);z=s(v[2]*s(mass/j.maximum(v[0],1e-30)))
            a=s(v[4:]+residue);residue=j.where(mass==0.,a,0.);a=j.where(mass==0.,0.,a)
            v=j.concatenate((j.stack((mass,j.where(mass==0.,0.,s(v[1]-heat)),z,v[3])),a))
            return values.at[idx].set(v),s(remaining-loss),s(actual+loss),s(energy+heat),residue
        values,_,actual,heat,residue=lax.fori_loop(0,n,layer,(c,u[0],j.zeros((),c.dtype),j.zeros((),c.dtype),j.zeros(5,c.dtype)))
        out,nn=snow_regrid_jax(j.pad(values,((0,1),(0,0))))
        return out,nn,ledger.at[3].set(j.concatenate((j.stack((actual,heat)),residue)))
    cells,count,led=lax.switch(mode,(regrid,snowfall,rain,drain,sublimation))
    return j.concatenate((cells.ravel(),j.array([count],dtype=c.dtype),led.ravel()))


def unpack_snow_transfer(out):
    return dict(cells=out[:,:45].reshape(-1,5,9),counts=out[:,45].astype('int32'),
        incoming=out[:,46:53],drainage=out[:,53:60],runoff=out[:,60:67],sublimation=out[:,67:74])


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(snow_transfer_kernel_jax))


def run_snow_transfer_jax(cells,counts,modes,inputs):
    import jax.numpy as j
    values=tuple(map(j.asarray,(cells,counts,modes,inputs)));validate_snow_transfer(*values,j)
    out=_compiled()(*values)
    if not bool(j.all(j.isfinite(out))):raise RuntimeError('Nonfinite snow mass transfer')
    return unpack_snow_transfer(out)

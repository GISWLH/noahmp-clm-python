"""Native exposed-ice liquid mixing, sublimation and frost/dew deposition.

Mode 0: positive incoming liquid, inputs [mass, enthalpy, five impurities].
Mode 1: signed outward vapor, inputs [mass, surface T-Tf, five zero pads].
The returned impurity outlet includes liquid runoff and exhausted-ice residue.
"""
from functools import lru_cache
import numpy as np


def pack_glacier_transfer(ice,mode,amount,value,*,impurities=(0.,0.,0.,0.,0.),dtype=np.float64):
    g=np.zeros((4,7),dtype);n=len(ice.mass)
    g[:n,:2]=np.column_stack((ice.mass,ice.enthalpy));g[:n,2:]=ice.impurities
    return g,np.asarray(n,np.int32),np.asarray(mode,np.int32),np.asarray([amount,value,*impurities],dtype)


def validate_glacier_transfer(g,n,mode,u,xp):
    if g.shape!=(len(g),4,7) or n.shape!=(len(g),) or mode.shape!=n.shape or u.shape!=(len(g),7):
        raise ValueError('Invalid glacier transfer shapes')
    if g.dtype not in (xp.float32,xp.float64) or u.dtype!=g.dtype or n.dtype.kind not in 'iu' or mode.dtype.kind not in 'iu':
        raise ValueError('Invalid glacier transfer dtypes')
    if not all(bool(xp.all(xp.isfinite(v))) for v in (g,u)):raise ValueError('Nonfinite glacier transfer')
    active=xp.arange(4)[None,:]<n[:,None]
    if bool(xp.any((n<0)|(n>4)|(mode<0)|(mode>1))) or bool(xp.any(
        (active&(g[:,:,0]<=0.))|(g[:,:,1]>0.)|(g[:,:,1]<2100.*g[:,:,0]*(150.-273.15))|
        xp.any(g[:,:,2:]<0.,axis=2)|(~active&xp.any(g!=0.,axis=2)))):raise ValueError('Invalid ice state or padding')
    tol=64.*xp.finfo(g.dtype).eps*xp.maximum(xp.abs(u[:,0]),g[:,0,0])
    if bool(xp.any((mode==0)&((u[:,0]<0.)|(u[:,1]<(334000.+4180.*(200.-273.15))*u[:,0])|
        ((u[:,0]==0.)&(u[:,1]!=0.))|xp.any(u[:,2:]<0.,axis=1)))):
        raise ValueError('Invalid incoming glacier liquid')
    if bool(xp.any((mode==1)&((n==0)|(u[:,0]>g[:,0,0]+tol)|(u[:,1]<-73.15)|(u[:,1]>76.85)|xp.any(u[:,2:]!=0.,axis=1)))):
        raise ValueError('Invalid glacier vapor amount, temperature or padding')


def mix_glacier_liquid_jax(g,n,parcel):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    def mix(carry):
        first,cells,incoming,done=carry;v=s(cells[first]+incoming)
        liquid=j.minimum(v[0],j.maximum(v[1],0.)/334000.);tracer=s(v[2:]*s(liquid/v[0]));mass=s(v[0]-liquid)
        out=j.concatenate((j.stack((liquid,j.maximum(v[1],0.))),tracer))
        cell=j.concatenate((j.stack((mass,j.minimum(v[1],0.))),s(v[2:]-tracer)))
        return first+(mass<=0.).astype(j.int32),cells.at[first].set(cell),out,mass>0.
    first,cells,out,_=lax.while_loop(lambda v:(v[0]<n)&~v[3]&((v[2][0]>0.)|j.any(v[2][2:]>0.)),
        mix,(j.asarray(0,j.int32),g,parcel,j.asarray(False)))
    cells=j.where((j.arange(4)<n-first)[:,None],cells[j.minimum(j.arange(4)+first,3)],0.)
    return cells,n-first,out


def glacier_transfer_kernel_jax(g,n,mode,u):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;zero=j.zeros((),g.dtype);zeros=j.zeros(5,g.dtype)
    def water():
        cells,nn,out=mix_glacier_liquid_jax(g,n,u)
        return cells,nn,out[0],out[1],zero,zero,zero,out[2:]
    def vapor():
        def loss():
            amount=j.where(j.abs(s(u[0]-g[0,0]))<=s(s(64.*j.finfo(g.dtype).eps)*j.maximum(u[0],g[0,0])),g[0,0],u[0])
            fraction=amount/g[0,0];heat=s(fraction*g[0,1]);mass=s(g[0,0]-amount);full=mass==0.
            cells=g.at[0,0].set(mass).at[0,1].set(s(g[0,1]-heat))
            cells=cells.at[1,2:].set(j.where(full&(n>1),s(g[1,2:]+g[0,2:]),g[1,2:]))
            shifted=j.where((j.arange(4)<n-1)[:,None],cells[j.minimum(j.arange(4)+1,3)],0.)
            return j.where(full,shifted,cells),n-full.astype(j.int32),zero,zero,amount,heat,s(2834000.*amount),j.where(full&(n==1),g[0,2:],0.)
        def deposit():
            def frost():
                heat=s(s(u[0]*2100.)*u[1]);cells=g.at[0,0].set(s(g[0,0]-u[0])).at[0,1].set(s(g[0,1]-heat))
                return cells,n,zero,zero,u[0],heat,s(2834000.*u[0]),zeros
            def dew():
                heat=s(u[0]*s(334000.+s(4180.*u[1])))
                cells,nn,out=mix_glacier_liquid_jax(g,n,j.concatenate((j.stack((-u[0],-heat)),zeros)))
                return cells,nn,out[0],out[1],u[0],heat,s(2.5e6*u[0]),out[2:]
            return lax.cond(u[1]<0.,frost,dew)
        return lax.cond(u[0]>0.,loss,lambda:lax.cond(u[0]<0.,deposit,lambda:(g,n,zero,zero,zero,zero,zero,zeros)))
    cells,nn,run,runh,amount,heat,latent,tracer=lax.cond(mode==0,water,vapor)
    return j.concatenate((cells.ravel(),j.array([nn],g.dtype),j.stack((run,runh,amount,heat,latent)),tracer))


def unpack_glacier_transfer(out):
    return dict(cells=out[:,:28].reshape(-1,4,7),counts=out[:,28].astype('int32'),runoff_kgm2=out[:,29],
        runoff_enthalpy_jm2=out[:,30],mass_kgm2=out[:,31],advected_enthalpy_jm2=out[:,32],
        latent_energy_jm2=out[:,33],impurity_out_kgm2=out[:,34:39])


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(glacier_transfer_kernel_jax))


def run_glacier_transfer_jax(g,n,mode,u):
    import jax.numpy as j
    values=tuple(map(j.asarray,(g,n,mode,u)));validate_glacier_transfer(*values,j)
    out=_compiled()(*values)
    if not bool(j.all(j.isfinite(out))):raise RuntimeError('Nonfinite glacier mass transfer')
    return unpack_glacier_transfer(out)

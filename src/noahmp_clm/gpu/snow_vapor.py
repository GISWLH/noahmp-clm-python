"""Native top-layer mixed-phase vapor exchange with separate latent heat.

The mass demand is supplied by the coupled surface solver. This operator
transports condensed-phase enthalpy; latent energy is a distinct boundary
term and is never subtracted from the snow twice.
"""
from functools import lru_cache
import numpy as np
from .snow_mass import pack_snow_transfer,validate_snow_transfer,snow_phase_jax,snow_regrid_jax


def pack_snow_vapor(pack,amount,*,temperature=None,remap=True,dtype=np.float64):
    c,n,_,_=pack_snow_transfer(pack,0,dtype=dtype)
    if temperature is None:
        temperature=float(pack.temperature[0]) if len(pack.mass) else 273.15
    return c,n,np.array([amount,temperature-273.15,int(remap)],dtype)


def validate_snow_vapor(c,n,u,xp):
    if u.shape!=(len(c),3) or u.dtype!=c.dtype or not bool(xp.all(xp.isfinite(u))):
        raise ValueError('Snow vapor needs finite matching (n,3) inputs')
    base=xp.zeros((len(c),16),c.dtype)
    validate_snow_transfer(c,n,xp.zeros_like(n),base,xp)
    amount=u[:,0];supply=c[:,0,0]
    bound=64*xp.finfo(c.dtype).eps*xp.maximum(xp.abs(amount),supply)
    if bool(xp.any((amount>0)&((n==0)|(amount>supply+bound)))):
        raise ValueError('Snow vapor removal exceeds top-layer supply')
    if bool(xp.any((amount<0)&((u[:,1]<-73.15)|(u[:,1]>76.85)|((n==0)&(u[:,1]>=0))))):
        raise ValueError('Snow deposition needs valid temperature; empty dew belongs in soil')
    if bool(xp.any((u[:,2]!=0)&(u[:,2]!=1))):raise ValueError('Invalid snow vapor remap flag')


def snow_vapor_kernel_jax(c,n,u):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    zero=j.zeros((),c.dtype);residue=j.zeros(5,c.dtype)
    empty=j.zeros((5,9),c.dtype).at[:,3].set(54.5)
    def loss():
        supply=c[0,0];bound=s(s(64.*j.finfo(c.dtype).eps)*j.maximum(u[0],supply))
        amount=j.where(j.abs(s(u[0]-supply))<=bound,supply,u[0]);full=amount==supply
        fraction=j.where(full,1.,amount/j.maximum(supply,1e-30))
        _,ice,_=snow_phase_jax(supply,c[0,1])
        ice=j.where(full,ice,s(amount*ice)/j.maximum(supply,1e-30));liquid=s(amount-ice)
        heat=s(fraction*c[0,1])
        cells=c.at[0,0].set(s(supply-amount)).at[0,1].set(s(c[0,1]-heat)).at[0,2].set(s(c[0,2]*s(1.-fraction)))
        cells=cells.at[1,4:].set(j.where(full&(n>1),s(cells[1,4:]+cells[0,4:]),cells[1,4:]))
        residue=j.where(full&(n==1),cells[0,4:],0.)
        shifted=cells[j.minimum(j.arange(5)+1,4)]
        shifted=j.where((j.arange(5)<n-1)[:,None],shifted,empty)
        return j.where(full,shifted,cells),n-j.asarray(full,j.int32),amount,ice,liquid,heat,residue
    def deposit():
        amount=u[0];ice=j.where(u[1]<0.,amount,0.);liquid=s(amount-ice)
        heat=s(s(s(ice*2100.)*u[1])+s(liquid*s(334000.+s(4180.*u[1]))))
        mass=s(c[0,0]-amount);h=s(c[0,1]-heat);_,frozen,_=snow_phase_jax(mass,h)
        z=j.maximum(s(c[0,2]-ice/200.),frozen/917.)
        cells=c.at[0,0].set(mass).at[0,1].set(h).at[0,2].set(z)
        return cells,j.maximum(n,1),amount,ice,liquid,heat,residue
    state=lax.cond(u[0]>0.,loss,lambda:lax.cond(u[0]<0.,deposit,lambda:(c,n,zero,zero,zero,zero,residue)))
    cells,nn,amount,ice,liquid,heat,residue=state
    cells,nn=lax.cond((u[2]>0)&(u[0]!=0)&(nn>0),lambda:snow_regrid_jax(j.pad(cells,((0,1),(0,0)))),lambda:(cells,nn))
    latent=s(s(2.5e6*amount)+s(334000.*ice))
    return j.concatenate((cells.ravel(),j.array([nn],c.dtype),j.stack((amount,heat)),residue,j.stack((ice,liquid,latent))))


def unpack_snow_vapor(out):
    return dict(cells=out[:,:45].reshape(-1,5,9),counts=out[:,45].astype('int32'),
        outward=out[:,46:53],ice_kgm2=out[:,53],liquid_kgm2=out[:,54],latent_energy_jm2=out[:,55])


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(snow_vapor_kernel_jax))


def run_snow_vapor_jax(cells,counts,inputs):
    import jax.numpy as j
    values=tuple(map(j.asarray,(cells,counts,inputs)));validate_snow_vapor(*values,j)
    out=_compiled()(*values)
    if not bool(j.all(j.isfinite(out))):raise RuntimeError('Nonfinite snow vapor exchange')
    return unpack_snow_vapor(out)

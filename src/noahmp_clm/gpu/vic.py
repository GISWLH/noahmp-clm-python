"""Native VIC curves and optional configuration for extensive soil transfers.

Configuration columns: surface enabled, bottom enabled, b, Ds, Dsmax [mm/day],
Ws, exponent, residual pore fraction. Flags allow mixed columns on one device.
"""
from functools import lru_cache
import numpy as np
from ..physics.vic import VICParameters


def pack_vic(parameters=None,*,surface=True,bottom=True,dtype=np.float64):
    p=VICParameters() if parameters is None else parameters
    if not isinstance(p,VICParameters):raise TypeError('Expected VICParameters')
    return np.asarray([surface,bottom,*vars(p).values()],dtype)


def validate_vic(v,n,xp):
    if v.shape!=(n,8) or v.dtype not in (xp.float32,xp.float64) or not bool(xp.all(xp.isfinite(v))):
        raise ValueError('VIC requires finite (n,8) float configuration')
    if bool(xp.any(((v[:,:2]!=0.)&(v[:,:2]!=1.)))) or bool(xp.any(
        (v[:,2]<0.)|(v[:,3]<0.)|(v[:,3]>v[:,5])|(v[:,4]<0.)|(v[:,5]<=0.)|(v[:,5]>=1.)|
        (v[:,6]<1.)|(v[:,7]<0.)|(v[:,7]>=1.))):raise ValueError('Invalid VIC parameters or enable flags')


def column_vic(options,dtype,xp,parameters=None):
    """Encode explicit site parameters; no host hydrology is evaluated."""
    flags=(options[:,6:8]==6)
    if not bool(xp.any(flags)):
        if parameters is not None:raise ValueError('VIC parameters require runoff option 6')
        return None
    if parameters is None or isinstance(parameters,VICParameters):
        values=xp.broadcast_to(xp.asarray(pack_vic(parameters,dtype=np.dtype(dtype))[2:]),(len(options),6))
    else:values=xp.asarray(parameters,dtype)
    if values.shape!=(len(options),6):raise ValueError('VIC site parameters require (n,6)')
    config=xp.concatenate((flags.astype(dtype),values),axis=1);validate_vic(config,len(options),xp)
    return config


def infiltration_jax(incoming,water,capacity,shape):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    deficit=j.maximum(0.,s(capacity-water));exponent=s(1.+shape)
    log_x=j.log(j.where(deficit>0.,deficit/j.maximum(capacity,j.finfo(capacity.dtype).tiny),1.))/exponent
    limit=s(s(exponent*capacity)*j.exp(log_x))
    ratio=j.minimum(incoming/j.where(limit>0.,limit,1.),1.)
    value=s(-deficit*j.expm1(s(exponent*j.log1p(-ratio))))
    value=j.where(incoming>=limit,deficit,value)
    value=j.where(shape==0.,j.minimum(incoming,deficit),value)
    return j.minimum(incoming,j.maximum(0.,value))


def baseflow_jax(liquid,capacity,dt,v):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    residual=s(v[7]*capacity);available=j.maximum(0.,s(liquid-residual))
    wet=available/s(capacity-residual);linear=v[3]/v[5]
    excess=j.maximum(0.,s(wet-v[5]))/s(1.-v[5])
    fraction=s(s(linear*wet)+s(s(1.-linear)*j.power(excess,v[6])))
    return j.minimum(available,s(s(v[4]*s(dt/86400.))*fraction))


@lru_cache(maxsize=1)
def _compiled():
    import jax
    from .soil_transport import soil_transport_kernel_jax
    return jax.jit(jax.vmap(soil_transport_kernel_jax))


def run_vic_transport_jax(c,p,o,u,v):
    import jax.numpy as j
    from .soil_transport import validate_soil_transport,unpack_soil_transport
    c,p,o,u,v=map(j.asarray,(c,p,o,u,v));validate_soil_transport(c,p,o,u,j);validate_vic(v,len(c),j)
    if v.dtype!=c.dtype:raise ValueError('VIC precision mismatch')
    out=_compiled()(c,p,o,u,v);result=unpack_soil_transport(out)
    if not bool(j.all(result['valid'])) or not bool(j.all(j.isfinite(out))):
        raise ValueError('VIC transfer exceeds the thermodynamic physical domain')
    return result

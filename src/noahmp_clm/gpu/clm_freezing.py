"""Native explicit-suction freezing component, independent columns."""
from functools import lru_cache
import numpy as np

def pack_freezing(total,temperature,porosity,bexp,suction_m,dtype=np.float64):
    arrays=np.broadcast_arrays(*[np.asarray(v,dtype) for v in [total,temperature,porosity,bexp,suction_m]])
    result=np.column_stack([v.ravel() for v in arrays]);validate_freezing(result,np)
    return result

def validate_freezing(a,xp):
    if a.ndim!=2 or a.shape[1]!=5 or a.dtype not in (xp.float32,xp.float64):raise ValueError('Expected float N-by5 freezing inputs')
    if not bool(xp.all(xp.isfinite(a))):raise ValueError('Nonfinite freezing input')
    total,t,sat,b,psi=a.T
    bad=(t<150.)|(t>400.)|(sat<=0.)|(sat>=1.)|(total<0.)|(total>sat)|(b<=0.)|(psi<1e-12)|(psi>1e6)
    if bool(xp.any(bad)):raise ValueError('Nonphysical freezing input')

def freezing_kernel_jax(a):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    total,t,sat,b,psi=a
    deficit=j.maximum(s(273.15-t),0.)
    coefficient=s(334000./s(9.80616*psi))
    raw=s(s(coefficient*deficit)/t);ratio=j.maximum(raw,1e-12)
    potential=j.exp(j.minimum(s(j.log(sat)-s(j.log(ratio)/b)),j.log(s(2.*sat))))
    denominator=s(s(s(b*t)*t)*ratio)
    dp=j.where(raw>1e-12,s(s(s(potential*coefficient)*273.15)/denominator),0.)
    four=s(.04*total);width=j.maximum(four,1e-12);dw=j.where(four>1e-12,.04,0.)
    difference=s(total-potential);sign=j.sign(difference)
    rounded=j.maximum(s(width-j.abs(difference)),0.)
    liquid=j.maximum(s(j.minimum(total,potential)-s(s(rounded*rounded)/s(4.*width))),0.)
    dtotal=j.where(difference<0.,1.,j.where(difference>0.,0.,.5))
    dtotal=s(dtotal-s(s(rounded*s(dw-sign))/s(2.*width)))
    dtotal=s(dtotal+s(s(s(rounded*rounded)*dw)/s(s(4.*width)*width)))
    dcap=s(j.where(difference>0.,1.,j.where(difference<0.,0.,.5))-s(s(rounded*sign)/s(2.*width)))
    dt=j.where(liquid>0.,s(dcap*dp),0.);dtotal=j.where(liquid>0.,dtotal,0.)
    unfrozen=t>=273.15
    return j.stack((j.where(unfrozen,total,liquid),j.where(unfrozen,0.,dt),j.where(unfrozen,1.,dtotal)))

def unpack_freezing(a):
    return dict(liquid=a[:,0],d_liquid_d_temperature=a[:,1],d_liquid_d_total=a[:,2])

@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(freezing_kernel_jax))

def run_freezing_jax(inputs):
    import jax.numpy as j
    a=j.asarray(inputs);validate_freezing(a,j)
    return unpack_freezing(_compiled()(a))

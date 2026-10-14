"""Native JAX solution of the Wang wet-bulb depression equation."""
from functools import lru_cache
import math


def validate_wetbulb(atmosphere,xp):
    a=atmosphere
    if a.ndim!=2 or a.shape[1]!=3 or a.dtype not in (xp.float32,xp.float64):
        raise ValueError('Wet-bulb atmosphere requires floating (n,3): T, q, p')
    if (not bool(xp.all(xp.isfinite(a))) or bool(xp.any((a[:,0]<200.)|(a[:,0]>350.)|
            (a[:,1]<0.)|(a[:,1]>=.2)|(a[:,2]<30000.)|(a[:,2]>120000.)))):
        raise ValueError('Invalid wet-bulb atmosphere')


def wetbulb_kernel_jax(atmosphere):
    import jax.numpy as j
    from jax import lax
    b=lax.optimization_barrier;t,q,p=atmosphere
    tc=b(t-273.15);vapor=b(b(q*p)/b(.622+b(.378*q)))
    latent=j.where(t>273.15,2500000.,2834000.)
    gamma=b(b(1004.*p)/b(.622*latent))
    def iteration(_,bounds):
        lower,upper=bounds;middle=b(.5*b(lower+upper))
        es=b(610.8*j.exp(b(b(17.27*middle)/b(237.3+middle))))
        residual=b(b(es-vapor)+b(gamma*b(middle-tc)))
        return j.where(residual<0.,middle,lower),j.where(residual>=0.,middle,upper)
    lower,upper=lax.fori_loop(0,60,iteration,(j.asarray(-150.,t.dtype),j.asarray(100.,t.dtype)))
    offset=b(.5*b(lower+upper));tw=b(273.15+offset)
    argument=b(j.asarray(math.log(6.99e-5),t.dtype)+b(2.*b(offset+3.97)))
    tail=j.exp(-j.abs(argument));den=b(1.+tail)
    fraction=j.where(argument>=0.,tail/den,1./den)
    return j.stack((tw,fraction))


@lru_cache(maxsize=1)
def _batch():
    import jax
    return jax.jit(jax.vmap(wetbulb_kernel_jax))


def run_wetbulb_jax(atmosphere):
    import jax.numpy as j
    a=j.asarray(atmosphere);validate_wetbulb(a,j);out=_batch()(a)
    return dict(temperature=out[:,0],snow_fraction=out[:,1])

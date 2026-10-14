"""Native JAX CLM MOST profile coefficients and actual-flux stability update."""
from functools import lru_cache
import numpy as np

MOST_OUTPUTS=('zeta','raw_zeta','buoyancy','convective_wind','effective_wind','air_density',
    'potential_temperature','friction_velocity','heat_velocity','water_velocity',
    'leaf_velocity','ground_velocity','momentum_integral','heat_integral','water_integral')


def validate_most(geometry,atmosphere,iteration,xp):
    n=len(geometry)
    if geometry.shape!=(n,9) or atmosphere.shape!=(n,7) or iteration.shape!=(n,4):
        raise ValueError('MOST arrays require shapes (n,9), (n,7), (n,4)')
    if (geometry.dtype not in (xp.float32,xp.float64) or atmosphere.dtype!=geometry.dtype
            or iteration.dtype!=geometry.dtype or not all(bool(xp.all(xp.isfinite(v))) for v in (geometry,atmosphere,iteration))):
        raise ValueError('MOST requires finite matching floating precision')
    g,a,s=geometry,atmosphere,iteration
    bad=(g[:,3]<0)|(g[:,7]<=0)|(g[:,8]<=0)
    for h,r in ((0,4),(1,5),(2,6)):bad|=(g[:,r]<=0)|(g[:,r]>=g[:,h]-g[:,3])
    bad|=(a[:,0]<200)|(a[:,0]>350)|(a[:,1]<0)|(a[:,1]>=.2)|(a[:,2]<30000)|(a[:,3]<0)|(a[:,6]<=0)
    bad|=(s[:,0]<-100)|(s[:,0]>2)|(s[:,1]<1)|(s[:,2]<=0)|(s[:,3]<0)
    if bool(xp.any(bad)):raise ValueError('Invalid MOST geometry, atmosphere or iteration')


def most_kernel_jax(g,a,s):
    import jax.numpy as j
    from jax import lax
    b=lax.optimization_barrier
    def psi(z,momentum):
        x=b(b(1.-b(16.*z))**.25)
        if momentum:return b(b(b(2.*j.log(b(b(1.+x)/2.)))+j.log(b(b(1.+b(x*x))/2.)))-b(2.*j.arctan(x)))+j.asarray(np.pi/2.,g.dtype)
        return 2.*j.log(b(b(1.+b(x*x))/2.))
    def primitive(z,inv,momentum):
        x=b(z*inv);edge=-1.574 if momentum else -.465
        def unstable(_):
            def strong(_):
                tail=(1.14*b(b((-x)**(1./3.))-j.asarray((-edge)**(1./3.),g.dtype)) if momentum else
                      .8*b(j.asarray((-edge)**(-1./3.),g.dtype)-b((-x)**(-1./3.))))
                return b(b(j.log(b(edge/x))-psi(j.asarray(edge,g.dtype),momentum))+tail)
            return lax.cond(x>=edge,lambda _:-psi(x,momentum),strong,operand=None)
        def stable(_):
            return lax.cond(x<=1.,lambda _:5.*x,lambda _:b(b(4.+x)+b(4.*j.log(x))),operand=None)
        return b(j.log(z)+lax.cond(x<0.,unstable,stable,operand=None))
    inv=b(s[0]/b(g[0]-g[3]))
    integrals=j.stack([b(primitive(b(g[k]-g[3]),inv,k==0)-primitive(g[k+4],inv,k==0)) for k in range(3)])
    dm,dh,dq=integrals;ustar=b(b(.4*s[1])/dm)
    leaf=b(.01*j.sqrt(b(ustar/s[2])));sparse=j.exp(-s[3])
    soil=b(j.asarray(.4/.13,g.dtype)*b(b(b(g[8]*ustar)/1.5e-5)**(-.45)))
    ground=b(ustar*b(b(soil*sparse)+b(.004*b(1.-sparse))))
    theta=b(a[0]+b(.0098*g[1]));virtual=b(theta*b(1.+b(.61*a[1])))
    pv=b(b(a[1]*a[2])/b(.622+b(.378*a[1])))
    rho=b(b(a[2]-b(.378*pv))/b(287.*a[0]))
    vf=b(b(b(b(1.+b(.61*a[1]))*a[4])/1004.+b(b(.61*theta)*a[5]))/rho)
    buoyancy=b(b(9.81*vf)/virtual);conv=b(j.maximum(0.,b(buoyancy*g[7]))**(1./3.))
    wind=j.maximum(1.,j.sqrt(b(b(a[3]*a[3])+b(conv*conv))))
    raw=b(b(b(-b(g[0]-g[3])*.4)*buoyancy)/b(b(a[6]*a[6])*a[6]))
    return j.stack((j.clip(raw,-100.,2.),raw,buoyancy,conv,wind,rho,theta,ustar,
        b(b(.4*ustar)/dh),b(b(.4*ustar)/dq),leaf,ground,dm,dh,dq))


@lru_cache(maxsize=1)
def _batch():
    import jax
    return jax.jit(jax.vmap(most_kernel_jax))


def run_canopy_most_jax(geometry,atmosphere,iteration):
    import jax.numpy as j
    arrays=tuple(j.asarray(v) for v in (geometry,atmosphere,iteration));validate_most(*arrays,j)
    out=_batch()(*arrays)
    return {name:out[:,k] for k,name in enumerate(MOST_OUTPUTS)}

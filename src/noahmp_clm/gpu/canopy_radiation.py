"""Independent JAX ten-layer single-band shortwave component."""
from functools import lru_cache
RADIATION_LAYERS = 10


def validate_shortwave_inputs(lai, u, xp):
    if lai.ndim != 2 or lai.shape[1] != 10 or u.shape != (lai.shape[0],6):
        raise ValueError("Radiation inputs must have shape (n,10) and (n,6)")
    if not bool(xp.all(xp.isfinite(lai))) or not bool(xp.all(xp.isfinite(u))):
        raise ValueError("Nonfinite radiation input")
    if bool(xp.any(lai<0)) or bool(xp.any((u[:,0]<0)|(u[:,1]<0)|(u[:,2]<0)
            |(u[:,2]>1)|(u[:,3]<=0)|(u[:,3]>1)|(u[:,4]<0)|(u[:,5]<0)
            |(u[:,4]+u[:,5]>1))):
        raise ValueError("Unphysical radiation input")


def shortwave_kernel_jax(lai, u, ground_albedo_direct=None,*,track_absorption=False):
    import jax.numpy as jnp
    from jax import lax
    beam0, diffuse0, ag, mu, refl, tran = u
    direct_albedo = ag if ground_albedo_direct is None else ground_albedo_direct
    k = .5/mu
    omega = refl+tran
    backscatter = .625*refl+.375*tran
    # .5/k == mu analytically. Write the product explicitly: XLA rewrites
    # division by k, while an independently rounded CUDA reciprocal differs.
    # 1/mu >= 1, so log(1+x) needs no small-x correction here.
    beta = (1.+k)*mu*(1.-mu*jnp.log(1.+1./mu))
    a = 1.-omega+backscatter
    width = lai/1024.
    t = 1./(1.+a*width)
    r = backscatter*width*t
    intercepted = -jnp.expm1(-k*width)
    beam_t = 1.-intercepted
    up = omega*beta*intercepted
    down = omega*(1.-beta)*intercepted
    diffuse_abs=(1.-omega)*width*t
    beam_abs=(1.-omega)*intercepted

    def double(_, coeff):
        r, t, up, down, bt, da, ba, interception = coeff
        den = 1.-r*r
        nr, nt = r+t*t*r/den, t*t/den
        nu = up+t*(r*down+up*bt)/den
        nd = down*bt+t*(down+r*up*bt)/den
        nb = bt*bt
        nt = jnp.where(omega == 1.,1.-nr,nt)
        nd = jnp.where(omega == 1.,1.-nb-nu,nd)
        if track_absorption:
            # Integrate absorption under doubling. Keep small deficits out of
            # repeated subtraction from unity, especially for very thin LAI.
            nda=da*(1.+t/(1.-r))
            nba=ba*(1.+bt)+da*(down+up*bt)/(1.-r)
            interception=interception*(2.-interception)
            nt=(1.-nr)-nda;nb=1.-interception;nd=(interception-nu)-nba
            da,ba=nda,nba
        return nr, nt, nu, nd, nb, da, ba, interception
    r,t,up,down,beam_t,diffuse_abs,beam_abs,_=lax.fori_loop(0,10,double,
        (r,t,up,down,beam_t,diffuse_abs,beam_abs,intercepted))
    beam = jnp.zeros((11,), dtype=lai.dtype).at[0].set(beam0)
    def transmit(j, beam):
        return beam.at[j+1].set(beam[j]*beam_t[j])
    beam = lax.fori_loop(0,10,transmit,beam)
    su, sd = up*beam[:-1], down*beam[:-1]
    reflect = jnp.zeros_like(beam).at[10].set(ag)
    source = jnp.zeros_like(beam).at[10].set(direct_albedo*beam[10])

    def eliminate(idx, arrays):
        re, so = arrays
        j = 9-idx
        den = 1.-r[j]*re[j+1]
        return (re.at[j].set(r[j]+t[j]*t[j]*re[j+1]/den),
                so.at[j].set(su[j]+t[j]*(so[j+1]+re[j+1]*sd[j])/den))
    reflect, source = lax.fori_loop(0,10,eliminate,(reflect,source))
    diffuse = jnp.zeros_like(beam).at[0].set(diffuse0)
    def forward(j, d):
        return d.at[j+1].set((t[j]*d[j]+r[j]*source[j+1]+sd[j])/(1.-r[j]*reflect[j+1]))
    diffuse = lax.fori_loop(0,10,forward,diffuse)
    upward = reflect*diffuse+source
    absorbed = beam[:-1]+diffuse[:-1]+upward[1:]-beam[1:]-diffuse[1:]-upward[:-1]
    if track_absorption:
        absorbed=diffuse_abs*(diffuse[:-1]+upward[1:])+beam_abs*beam[:-1]
    ground = ((1.-ag)*(beam[10]+diffuse[10]) if ground_albedo_direct is None else
              (1.-direct_albedo)*beam[10]+(1.-ag)*diffuse[10])
    total = lax.fori_loop(0,10,lambda j,s:s+absorbed[j], jnp.zeros_like(ground))
    residual = beam0+diffuse0-upward[0]-total-ground
    return jnp.concatenate((absorbed,beam,diffuse,upward,jnp.stack((upward[0],ground,residual))))


def unpack_shortwave(result):
    """Views preserve the array backend (NumPy/JAX/CuPy)."""
    return {"absorbed":result[:,:10], "direct":result[:,10:21],
        "diffuse_down":result[:,21:32],"diffuse_up":result[:,32:43],
        "reflected":result[:,43],"ground_absorbed":result[:,44],"residual":result[:,45]}


def run_shortwave_jax(leaf_area, inputs):
    """Inputs: direct, diffuse, ground albedo, zenith cosine, leaf r, leaf t."""
    import jax.numpy as jnp
    lai, u = jnp.asarray(leaf_area), jnp.asarray(inputs)
    validate_shortwave_inputs(lai,u,jnp)
    if lai.dtype not in (jnp.float32,jnp.float64) or u.dtype != lai.dtype:
        raise ValueError("Radiation inputs require matching float32/float64 dtype")
    return unpack_shortwave(_compiled()(lai,u))


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(shortwave_kernel_jax))


def longwave_kernel_jax(lai, temperature, u):
    import jax.numpy as jnp
    from jax import lax
    sky, ground_temperature, emissivity = u
    t = jnp.exp(-lai)
    sigma = 5.670374419e-8
    emission = (1.-t)*sigma*temperature**4
    down = jnp.zeros((11,),dtype=lai.dtype).at[0].set(sky)
    down = lax.fori_loop(0,10,lambda j,d:d.at[j+1].set(t[j]*d[j]+emission[j]),down)
    up = jnp.zeros_like(down).at[10].set((1.-emissivity)*down[10]+emissivity*sigma*ground_temperature**4)
    def propagate(idx,up):
        j=9-idx
        return up.at[j].set(t[j]*up[j+1]+emission[j])
    up = lax.fori_loop(0,10,propagate,up)
    net = down[:-1]+up[1:]-down[1:]-up[:-1]
    ground = down[10]-up[10]
    total = lax.fori_loop(0,10,lambda j,s:s+net[j],jnp.zeros_like(ground))
    residual = sky-up[0]-total-ground
    return jnp.concatenate((net,down,up,jnp.stack((ground,up[0],residual))))


def unpack_longwave(result):
    return {"net_absorbed":result[:,:10],"down":result[:,10:21],"up":result[:,21:32],
        "ground_net":result[:,32],"outgoing":result[:,33],"residual":result[:,34]}


def validate_longwave_inputs(lai, temperature, u, xp):
    if lai.ndim!=2 or lai.shape[1]!=10 or temperature.shape!=lai.shape or u.shape!=(lai.shape[0],3):
        raise ValueError("Longwave inputs must be (n,10), (n,10), (n,3)")
    if not all(bool(xp.all(xp.isfinite(a))) for a in (lai,temperature,u)):
        raise ValueError("Nonfinite longwave input")
    if (bool(xp.any(lai<0)) or bool(xp.any((temperature<200)|(temperature>350)))
            or bool(xp.any((u[:,0]<0)|(u[:,1]<200)|(u[:,1]>350)|(u[:,2]<=0)|(u[:,2]>1)))):
        raise ValueError("Unphysical longwave input")


def run_longwave_jax(leaf_area,temperature,inputs):
    """Inputs: sky LW W/m2, ground K, ground emissivity; extinction = 1."""
    import jax.numpy as jnp
    lai,temp,u=map(jnp.asarray,(leaf_area,temperature,inputs))
    validate_longwave_inputs(lai,temp,u,jnp)
    if lai.dtype not in (jnp.float32,jnp.float64) or u.dtype!=lai.dtype or temp.dtype!=lai.dtype:
        raise ValueError("Longwave inputs require matching float32/float64 dtype")
    return unpack_longwave(_compiled_longwave()(lai,temp,u))


@lru_cache(maxsize=1)
def _compiled_longwave():
    import jax
    return jax.jit(jax.vmap(longwave_kernel_jax))

"""Independent JAX leaf kernel for the NumPy photosynthesis reference.

Input row: absorbed PAR, leaf K, CO2 ppm, RH relative to leaf saturation,
boundary conductance mol/m²/s, nitrogen factor, water stress. Parameters follow
LeafParameters field order. This component is not yet wired to column stepping.
"""
from dataclasses import astuple
from functools import lru_cache
import numpy as np
from ..physics.photosynthesis import LeafParameters

LEAF_OUTPUTS = ("gross", "net", "rubisco", "rubp", "tpu", "respiration",
    "electron_transport", "ci_ppm", "cs_ppm", "gs_mol",
    "surface_relative_humidity", "diffusion_residual")


def pack_leaf_parameters(parameters=None, dtype=np.float64):
    return np.asarray(astuple(parameters or LeafParameters()), dtype=dtype)


def validate_leaf_inputs(u, xp):
    """Eager API validation; compiled component callers enforce this contract."""
    if u.ndim != 2 or u.shape[1] != 7:
        raise ValueError("Leaf inputs must have shape (n, 7)")
    if not bool(xp.all(xp.isfinite(u))):
        raise ValueError("Nonfinite leaf input")
    bad = ((u[:,0]<0)|(u[:,1]<200)|(u[:,1]>350)|(u[:,2]<=0)|(u[:,3]<0)
           |(u[:,4]<=0)|(u[:,5]<0)|(u[:,5]>1)|(u[:,6]<0)|(u[:,6]>1))
    if bool(xp.any(bad)):
        raise ValueError("Unphysical leaf input")


def leaf_kernel_jax(u, p):
    import jax.numpy as jnp
    from jax import lax
    par, temp, ca, rh, gb, nitrogen, stress = u
    growth = jnp.clip(p[8]-273.15, 11., 35.)

    def arrhenius(activation):
        return jnp.exp((activation/(8.314462618*298.15))*(1.-298.15/temp))

    def peaked(activation, deactivation, entropy):
        numerator = 1.+jnp.exp((298.15*entropy-deactivation)/(8.314462618*298.15))
        denominator = 1.+jnp.exp((temp*entropy-deactivation)/(8.314462618*temp))
        return arrhenius(activation)*numerator/denominator

    vc_factor = peaked(72000., 200000., 668.39-1.07*growth)
    vcmax = p[0]*nitrogen*vc_factor
    jmax = p[1]*nitrogen*peaked(50000., 200000., 659.70-.75*growth)
    product = 3.*p[2]*nitrogen*vc_factor
    rd = p[3]*peaked(46390., 150650., 490.)
    kc, ko, gamma = 404.9*arrhenius(79430.), 278400.*arrhenius(36380.), 42.75*arrhenius(37830.)
    light = .5*p[4]*par
    total = light+jmax
    denominator = total+jnp.sqrt(jnp.maximum(total*total-4.*p[5]*light*jmax, 0.))
    transport = jnp.where(denominator > 0., 2.*light*jmax/jnp.maximum(denominator, 1e-30), 0.)

    def evaluate(ci):
        carbon = jnp.maximum(ci-gamma, 0.)
        rubisco = vcmax*carbon/(ci+kc*(1.+p[9]/ko))
        rubp = transport*carbon/(4.*ci+8.*gamma)
        gross = jnp.minimum(jnp.minimum(rubisco, rubp), product)
        net = gross-rd
        cs = jnp.maximum(ca-1.4*net/gb, 1e-9)
        x = p[6]*jnp.maximum(net, 0.)*stress/cs
        b = gb-p[7]-x
        c = gb*(p[7]+x*jnp.minimum(rh, 1.))
        root = jnp.sqrt(b*b+4.*c)
        gs = jnp.where(b >= 0., 2.*c/jnp.maximum(b+root, 1e-30), .5*(root-b))
        residual = net-(ca-ci)/(1.4/gb+1.6/gs)
        return gross, net, rubisco, rubp, cs, gs, residual

    def bisect(_, bracket):
        lo, hi = bracket
        mid = .5*(lo+hi)
        residual = evaluate(mid)[-1]
        return jnp.where(residual < 0., mid, lo), jnp.where(residual >= 0., mid, hi)

    hi = ca+rd*(1.4/gb+1.6/p[7])+1.
    lo, hi = lax.fori_loop(0, 48, bisect, (jnp.zeros_like(hi), hi))
    ci = .5*(lo+hi)
    gross, net, ac, aj, cs, gs, residual = evaluate(ci)
    humidity = (gb*jnp.minimum(rh, 1.)+gs)/(gb+gs)
    return jnp.stack((gross, net, ac, aj, product, rd, transport, ci, cs, gs, humidity, residual))


def run_leaves_jax(inputs, parameters=None):
    import jax.numpy as jnp
    u = jnp.asarray(inputs)
    validate_leaf_inputs(u, jnp)
    if u.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("Leaf inputs must be float32 or float64")
    p = jnp.asarray(pack_leaf_parameters(parameters), dtype=u.dtype)
    return _compiled_leaves()(u, p)


@lru_cache(maxsize=1)
def _compiled_leaves():
    import jax
    return jax.jit(jax.vmap(leaf_kernel_jax, in_axes=(0, None)))

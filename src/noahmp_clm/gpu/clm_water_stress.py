"""Native development-column soil beta with the actual saturated suction.

Parameters retain the extensive column's seven-field order: porosity, B,
field capacity, conductivity, quartz, wilting theta, saturated suction (m).
The released legacy column keeps its separate stress closure.
"""
from functools import lru_cache


def validate_water_stress(liquid, roots, parameters, options, xp):
    n = len(liquid)
    if (liquid.shape != (n, 4) or roots.shape != (n, 4)
            or parameters.shape != (n, 7) or options.shape != (n,)
            or liquid.dtype not in (xp.float32, xp.float64)
            or roots.dtype != liquid.dtype or parameters.dtype != liquid.dtype
            or options.dtype.kind not in 'iu'):
        raise ValueError('Expected matching (n,4) liquid/roots, (n,7) soil and integer (n,) options')
    if (not bool(xp.all(xp.isfinite(liquid)))
            or not bool(xp.all(xp.isfinite(roots)))
            or not bool(xp.all(xp.isfinite(parameters)))
            or bool(xp.any(liquid < 0.)) or bool(xp.any(roots < 0.))
            or bool(xp.any((options < 1) | (options > 3)))):
        raise ValueError('Invalid liquid water, roots, soil or stress option')
    p = parameters
    if bool(xp.any((p[:, 0] <= 0.) | (p[:, 1] <= 0.)
                  | (p[:, 5] < 0.) | (p[:, 2] <= p[:, 5])
                  | (p[:, 2] > p[:, 0]) | (p[:, 6] <= 0.) | (p[:, 6] >= 150.))):
        raise ValueError('Invalid porosity, B, field/wilting theta or saturated suction')
    total = xp.sum(roots, axis=1)
    tolerance = 8.*xp.finfo(liquid.dtype).eps
    if bool(xp.any((total != 0.) & (xp.abs(total-1.) > tolerance))):
        raise ValueError('Root fractions must sum to one or zero')


def water_stress_kernel_jax(liquid, roots, p, option):
    import jax.numpy as j
    from jax import lax
    s = lax.optimization_barrier
    noah = j.clip(s(liquid-p[5])/j.maximum(1e-9, s(p[2]-p[5])), 0., 1.)
    log_wilt = j.log(j.asarray(150., liquid.dtype))
    log_head = s(j.log(p[6])-s(p[1]*j.log(s(j.maximum(.01, liquid)/p[0]))))
    potential = j.where(log_head >= log_wilt, -150.,
                        -j.exp(j.minimum(log_head, log_wilt)))
    clm = s(1.-s(potential/-150.))/s(1.+s(p[6]/-150.))
    ssib = -j.expm1(s(-5.8*j.log(s(-150./potential))))
    beta = j.clip(j.where(option == 1, noah, j.where(option == 2, clm, ssib)), 0., 1.)
    weighted = s(beta*roots)
    root = j.clip(s(s(weighted[0]+weighted[2])+s(weighted[1]+weighted[3])), 0., 1.)
    return j.concatenate((j.reshape(root, (1,)), beta))


@lru_cache(maxsize=1)
def _batch():
    import jax
    return jax.jit(jax.vmap(water_stress_kernel_jax))


def run_water_stress_jax(liquid, roots, parameters, options):
    import jax.numpy as j
    liquid, roots, parameters = map(j.asarray, (liquid, roots, parameters))
    options = j.asarray(options)
    validate_water_stress(liquid, roots, parameters, options, j)
    out = _batch()(liquid, roots, parameters, options)
    return dict(root_factor=out[:, 0], layer_factor=out[:, 1:])

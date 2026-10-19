"""Native JAX capacity profiles and restartable ten-day temperature memory.

Host operations validate and pack arrays only. The NumPy physical reference
is not called by either device entry.
"""
from functools import lru_cache
import numpy as np


def validate_capacity(leaf, environment, traits, template, xp):
    n = len(leaf); dtype = leaf.dtype
    if (leaf.shape != (n, 10) or environment.shape != (n, 3) or traits.shape != (n, 4)
            or template.shape != (n, 10)):
        raise ValueError('Capacity arrays require n by10/3/4/10 shapes')
    if dtype not in (xp.float32, xp.float64) or any(a.dtype != dtype for a in (environment, traits, template)):
        raise ValueError('Capacity arrays require matching floating precision')
    if not all(bool(xp.all(xp.isfinite(a))) for a in (leaf, environment, traits, template)):
        raise ValueError('Nonfinite capacity input')
    day, maximum, temperature = environment.T
    bad = (day < 0.) | (day > maximum) | (maximum <= 0.) | (maximum > 86400.) | (temperature < 200.) | (temperature > 350.)
    if (bool(xp.any(bad)) or bool(xp.any(leaf < 0.)) or bool(xp.any(traits[:, :2] <= 0.))
            or bool(xp.any((traits[:, 2:] < 0.) | (traits[:, 2:] > 1.)))):
        raise ValueError('Unphysical C3 capacity input')
    if (bool(xp.any(template < 0.)) or bool(xp.any((template[:, 4:6] <= 0.) | (template[:, 4:6] > 1.)))
            or bool(xp.any(template[:, 7] <= 0.)) or bool(xp.any((template[:, 8] < 250.) | (template[:, 8] > 320.)))):
        raise ValueError('Invalid gas-exchange template')
    # This also rejects trait combinations that overflow in native precision.
    unscaled = 1. / (traits[:, 0] * traits[:, 1]) * traits[:, 2] * 7.16 * 60. * traits[:, 3]
    if not bool(xp.all(xp.isfinite(unscaled))) or bool(xp.any(.00963 * unscaled - 2.43 > np.log(np.finfo(np.dtype(dtype)).max))):
        raise ValueError('Leaf traits exceed finite C3 capacity range')


def unpack_capacity(raw):
    return dict(leaf_parameters=raw[:, :100].reshape((-1, 10, 10)),
                nitrogen_profile_factor=raw[:, 100:110], leaf_nitrogen_gn_m2_leaf=raw[:, 110:120],
                **{name: raw[:, 120+k] for k, name in enumerate(('unscaled_vcmax25_top', 'vcmax25_top',
                    'nitrogen_extinction', 'daylength_factor', 'mean_temperature_k', 'growth_temperature_k'))})


def capacity_kernel_jax(leaf, environment, traits, template):
    import jax.numpy as j
    from jax import lax
    s = lax.optimization_barrier
    nitrogen = 1. / s(traits[0] * traits[1])
    unscaled = s(s(s(s(nitrogen * traits[2]) * 7.16) * 60.) * traits[3])
    kn = j.exp(s(s(.00963 * unscaled) - 2.43))
    ratio = environment[0] / environment[1]
    dayfactor = j.maximum(.01, s(ratio * ratio))
    growth = j.clip(s(environment[2] - 273.15), 11., 35.)
    top = s(unscaled * dayfactor)
    rows = j.broadcast_to(template, (10, 10))
    factors = j.zeros(10, leaf.dtype)
    def layer(k, carry):
        cumulative, values = carry
        cumulative = s(cumulative + leaf[k])
        midpoint = s(cumulative - s(.5 * leaf[k]))
        scale = j.where(leaf[k] > 0., j.exp(-s(kn * midpoint)), 0.)
        return cumulative, values.at[k].set(scale)
    _, factors = lax.fori_loop(0, 10, layer, (j.asarray(0., leaf.dtype), factors))
    vcmax = s(top * factors)
    rows = rows.at[:, 0].set(vcmax).at[:, 1].set(s(s(2.59 - s(.035 * growth)) * vcmax))
    rows = rows.at[:, 2].set(s(.167 * vcmax)).at[:, 3].set(s(.015 * vcmax)).at[:, 8].set(s(growth + 273.15))
    scalars = j.stack((unscaled, top, kn, dayfactor, environment[2], s(growth + 273.15)))
    return j.concatenate((rows.reshape(-1), factors, s(nitrogen * factors), scalars))


@lru_cache(maxsize=1)
def _capacity_compiled():
    import jax
    return jax.jit(jax.vmap(capacity_kernel_jax))


def run_capacity_jax(leaf, environment, traits, template):
    import jax.numpy as j
    arrays = tuple(map(j.asarray, (leaf, environment, traits, template)))
    validate_capacity(*arrays, j)
    return unpack_capacity(_capacity_compiled()(*arrays))


def validate_temperature_memory(values, controls, forcing, xp):
    n = len(values); dtype = values.dtype
    if (values.ndim != 2 or values.shape[1] < 1 or values.shape[1] > 14400
            or controls.shape != (n, 2) or controls.dtype.kind not in 'iu' or forcing.shape != (n, 3)):
        raise ValueError('Invalid temperature-memory shapes or integer controls')
    if dtype not in (xp.float32, xp.float64) or forcing.dtype != dtype:
        raise ValueError('Memory arrays require matching floating precision')
    width = values.shape[1]; count, cursor = controls.T
    if (not bool(xp.all(xp.isfinite(values))) or not bool(xp.all(xp.isfinite(forcing)))
            or bool(xp.any((values < 200.) | (values > 350.)))
            or bool(xp.any((forcing[:, 1:] < 200.) | (forcing[:, 1:] > 350.)))
            or bool(xp.any(abs(forcing[:, 0] * width - 864000.) > 4.*xp.finfo(dtype).eps*864000.))
            or bool(xp.any((count < 0) | (count > width) | (cursor < 0) | (cursor >= width)
                           | ((count < width) & (cursor != count))))):
        raise ValueError('Invalid retained temperature samples, interval or cursor')


def temperature_memory_kernel_jax(values, controls, forcing):
    import jax.numpy as j
    from jax import lax
    s = lax.optimization_barrier
    count, cursor = controls
    after = values.at[cursor].set(forcing[2]); next_count = j.minimum(count + 1, len(values))
    def total(k, carry):
        old, new = carry
        old = s(old + j.where(k < count, s(values[k] - 273.15), 0.))
        new = s(new + j.where(k < next_count, s(after[k] - 273.15), 0.))
        return old, new
    old, new = lax.fori_loop(0, len(values), total, (j.asarray(0., values.dtype), j.asarray(0., values.dtype)))
    mean = j.where(count > 0, s(273.15 + old / j.maximum(count, 1)), forcing[1])
    next_mean = s(273.15 + new / next_count)
    next_controls = j.stack((next_count, (cursor + 1) % len(values)))
    return after, next_controls, j.stack((mean, next_mean))


@lru_cache(maxsize=1)
def _memory_compiled():
    import jax
    return jax.jit(jax.vmap(temperature_memory_kernel_jax))


def run_temperature_memory_jax(values, controls, forcing):
    import jax.numpy as j
    arrays = tuple(map(j.asarray, (values, controls, forcing)))
    validate_temperature_memory(*arrays, j)
    output = _memory_compiled()(*arrays)
    return dict(values=output[0], controls=output[1], means=output[2])

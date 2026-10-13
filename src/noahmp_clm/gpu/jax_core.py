"""JAX 单柱物理移植。

本模块只接收定长数组，不创建或修改 ``ColumnState``。每个柱的状态由一个
``fvec``、一个 ``ivec`` 和当前强迫向量显式传入；因此 ``vmap`` 后不存在柱间
共享状态。公式与普通版相同，固定次数迭代是 JAX 编译所需的数值等价形式。
"""

from __future__ import annotations

from typing import Callable

try:  # GPU extra 是可选依赖；普通版 doctest 不应因未安装 JAX 而收集失败。
    import jax
    import jax.numpy as jnp
    from jax import lax
    _JAX_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - 依赖缺失环境
    jax = None
    jnp = None
    lax = None
    _JAX_IMPORT_ERROR = exc

from .pack import (
    FLOAT_SIZE,
    INT_SIZE,
    OPTION_FIELDS,
    SNOW_COUNT_INDEX,
)


KARMAN = 0.4
SIGMA = 5.670374419e-8
RD = 287.0
CP = 1004.0
LV = 2.5e6
RHO_W = 1000.0
LF = 3.34e5
RHO_SNOW = 300.0
C_SNOW = RHO_SNOW * 2100.0
GRAVITY = 9.81
FREEZING_POINT = 273.15
if jnp is not None:
    DZ = (0.1, 0.3, 0.6, 1.0)
    LAYER_INDEX = jnp.arange(3)
    SOIL_INDEX = jnp.arange(4)
else:  # 仅保证可选模块可被 doctest 收集；调用时由 _require_jax 报错。
    DZ = None
    LAYER_INDEX = None
    SOIL_INDEX = None
OPTION_INDEX = {name: index for index, name in enumerate(OPTION_FIELDS)}


def _require_jax() -> None:
    """对应普通版: driver.step_column。

    在没有安装 ``.[gpu]`` 时给出清晰错误，而不是导入阶段失败。
    """
    if _JAX_IMPORT_ERROR is not None:
        raise ImportError("JAX GPU backend 未安装，请先 pip install -e '.[gpu]'") from _JAX_IMPORT_ERROR


def _option(ivec: jax.Array, name: str) -> jax.Array:
    """对应普通版: options.PhysicsOptions.__post_init__。

    从整型 ABI 向量读取一个选项；字段索引由 Python 静态元数据确定。
    """
    return ivec[OPTION_INDEX[name]]


def _switch(option: jax.Array, branches: list[Callable], operand) -> jax.Array:
    """对应普通版: 各 physics 模块的方案字典分发。

    Noah-MP 选项是从 1 开始的枚举，JAX ``lax.switch`` 使用从 0 开始的索引。
    对超出当前实现范围的值使用最后一个分支，避免追踪期 Python 异常。
    """
    option = jnp.asarray(option, dtype=jnp.int32)
    index = jnp.minimum(jnp.maximum(option - 1, 0), len(branches) - 1)
    return lax.switch(index, branches, operand)


def _sat_vapor_pressure(t: jax.Array) -> jax.Array:
    """对应普通版: energy.sat_vapor_pressure。

    Tetens 饱和水汽压，方程形式与普通版相同。
    """
    tc = t - 273.15
    return 611.2 * jnp.exp(17.67 * tc / (tc + 243.5))


def _qsat(t: jax.Array, ps: jax.Array) -> jax.Array:
    """对应普通版: energy.qsat。"""
    es = _sat_vapor_pressure(t)
    return 0.622 * es / jnp.maximum(1.0, ps - 0.378 * es)


def _air_density(tair: jax.Array, qair: jax.Array, ps: jax.Array) -> jax.Array:
    """对应普通版: energy.air_density。"""
    return ps / (RD * tair * (1.0 + 0.61 * qair))


def _bulk_ch(wind: jax.Array, z0m: jax.Array) -> jax.Array:
    """对应普通版: energy.bulk_ch。"""
    z0m_safe = jnp.clip(z0m, 1.0e-5, 5.0)
    z0h = z0m_safe / 10.0
    return KARMAN**2 / (jnp.log(10.0 / z0m_safe) * jnp.log(10.0 / z0h))


def _psi_m(zeta: jax.Array) -> jax.Array:
    """对应普通版: surface_drag._psi_m。

    Businger--Dyer 稳定度函数；``jnp.where`` 使稳定/不稳定分支可追踪。
    """
    x = jnp.power(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-12), 0.25)
    unstable = (
        2.0 * jnp.log((1.0 + x) / 2.0)
        + jnp.log((1.0 + x * x) / 2.0)
        - 2.0 * jnp.arctan(x)
        + jnp.pi / 2.0
    )
    return jnp.where(zeta < 0.0, unstable, -5.0 * zeta)


def _psi_h(zeta: jax.Array) -> jax.Array:
    """对应普通版: surface_drag._psi_h。"""
    x = jnp.power(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-12), 0.25)
    unstable = 2.0 * jnp.log((1.0 + x * x) / 2.0)
    return jnp.where(zeta < 0.0, unstable, -5.0 * zeta)


def _most_exchange_velocity(
    wind: jax.Array,
    tair: jax.Array,
    qair: jax.Array,
    ps: jax.Array,
    z0m: jax.Array,
    surface_temperature: jax.Array,
) -> jax.Array:
    """对应普通版: surface_drag.most_exchange_velocity。

    固定 20 次 ``lax.fori_loop``；达到普通版相对变化阈值后冻结状态，
    保持早停的数值语义而不改变编译期循环上限。
    """
    wind = jnp.maximum(wind, 0.3)
    z0m_safe = jnp.clip(z0m, 1.0e-5, 5.0)
    z0h = jnp.maximum(z0m_safe / 10.0, 1.0e-6)
    neutral_m = jnp.log(10.0 / z0m_safe)
    neutral_h = jnp.log(10.0 / z0h)
    rho = _air_density(tair, qair, ps)
    ce0 = _bulk_ch(wind, z0m_safe) * wind

    def body(_, carry):
        ce, obukhov, done = carry
        finite = jnp.abs(obukhov) < 1.0e29
        zeta = jnp.where(finite, 10.0 / obukhov, 0.0)
        den_m = jnp.maximum(1.0, neutral_m - _psi_m(zeta))
        den_h = jnp.maximum(1.0, neutral_h - _psi_h(zeta))
        ustar = KARMAN * wind / den_m
        ce_new = jnp.maximum(1.0e-6, KARMAN**2 * wind / (den_m * den_h))
        sensible = rho * CP * ce_new * (surface_temperature - tair)
        obukhov_new = -rho * CP * tair * ustar**3 / (
            KARMAN * GRAVITY * jnp.where(jnp.abs(sensible) < 1.0e-8, 1.0, sensible)
        )
        obukhov_new = jnp.where(
            jnp.abs(sensible) < 1.0e-8,
            1.0e30,
            jnp.clip(obukhov_new, -1.0e6, 1.0e6),
        )
        finite_new = jnp.abs(obukhov_new) < 1.0e29
        converged = (finite & finite_new & (
            jnp.abs(obukhov_new - obukhov) / jnp.maximum(1.0, jnp.abs(obukhov)) < 1.0e-4
        )) | (~finite & ~finite_new)
        return (jnp.where(done, ce, ce_new),
                jnp.where(done, obukhov, obukhov_new), done | converged)

    ce, _, _ = lax.fori_loop(
        0, 20, body, (ce0, jnp.asarray(1.0e30, dtype=ce0.dtype), jnp.asarray(False))
    )
    return ce


def _exchange_velocity(
    wind: jax.Array,
    tair: jax.Array,
    qair: jax.Array,
    ps: jax.Array,
    z0m: jax.Array,
    surface_temperature: jax.Array,
    option: jax.Array,
) -> jax.Array:
    """对应普通版: surface_drag.exchange_velocity。"""
    most = lambda _: _most_exchange_velocity(
        wind, tair, qair, ps, z0m, surface_temperature
    )
    bulk = lambda _: _bulk_ch(jnp.maximum(wind, 0.3), z0m) * jnp.maximum(wind, 0.3)
    return _switch(option, [most, bulk], None)


def _snow_fraction(
    tair: jax.Array, qair: jax.Array, ps: jax.Array, option: jax.Array
) -> jax.Array:
    """对应普通版: precip.partition_precipitation。

    分支 1--3 对应 Jordan (1991)、BATS 和冰点阈值；4 沿用 Jordan，5 用
    湿球近似（阶段 1 的纯 Python 版预留接口）。
    """
    delta = tair - FREEZING_POINT
    jordan = jnp.where(
        delta > 2.5,
        0.0,
        jnp.where(
            delta <= 0.5,
            1.0,
            jnp.where(delta <= 2.0, jnp.clip(1.0 - (-54.632 + 0.2 * tair), 0.0, 1.0), 0.6),
        ),
    )
    bats = (tair < FREEZING_POINT + 2.2).astype(tair.dtype)
    threshold = (tair < FREEZING_POINT).astype(tair.dtype)
    # 线性湿球替代项：相对湿度越低，湿球越低，越偏向降雪。
    es = _sat_vapor_pressure(tair)
    ea = qair * ps / (0.622 + 0.378 * qair)
    rh = jnp.clip(ea / jnp.maximum(es, 1.0), 0.0, 1.0)
    wetbulb = jnp.clip((FREEZING_POINT + 1.5 - (tair - 3.0 * (1.0 - rh))) / 3.0, 0.0, 1.0)
    return _switch(option, [lambda _: jordan, lambda _: bats, lambda _: threshold,
                            lambda _: jordan, lambda _: wetbulb], None)


def _smooth_water_cap(total, potential):
    width = jnp.maximum(0.04 * total, 1.0e-12)
    rounding = jnp.maximum(width - jnp.abs(total - potential), 0.0)
    return jnp.maximum(0.0, jnp.minimum(total, potential) - rounding * rounding / (4.0 * width))


def _unfrozen_water(
    smc: jax.Array,
    temperature: jax.Array,
    smcmax: jax.Array,
    bexp: jax.Array,
    option: jax.Array,
) -> jax.Array:
    """对应普通版: frozen_soil.unfrozen_water_content。

    Niu--Yang (2006) 直接反推未冻水；Koren (1999) 用固定三次平均迭代。
    """
    total = jnp.maximum(0.0, smc)
    undercooling = jnp.maximum(0.0, FREEZING_POINT - temperature)
    clausius = LF / 461.5 * undercooling / FREEZING_POINT**2
    potential = smcmax * (
        jnp.maximum(clausius, 1.0e-8) / 5.0e-5
    ) ** (-1.0 / jnp.maximum(bexp, 1.0e-6))
    niu = jnp.where(temperature >= FREEZING_POINT, total, _smooth_water_cap(total, potential))

    def koren_body(_, liquid):
        inner = smcmax * (
            jnp.maximum(LF / 461.5 * jnp.maximum(0.0, FREEZING_POINT - temperature)
                        / FREEZING_POINT**2, 1.0e-8) / 5.0e-5
        ) ** (-1.0 / jnp.maximum(bexp, 1.0e-6))
        inner = jnp.where(temperature >= FREEZING_POINT, liquid, _smooth_water_cap(liquid, inner))
        return 0.5 * (liquid + jnp.minimum(smc, inner))

    koren = lax.fori_loop(0, 3, koren_body, niu)
    return _switch(option, [lambda _: niu, lambda _: jnp.minimum(smc, koren)], None)


def _phase_and_capacity(
    smc: jax.Array,
    temperature: jax.Array,
    smcmax: jax.Array,
    bexp: jax.Array,
    option: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """对应普通版: frozen_soil.update_soil_phase/apparent_heat_capacity。

    返回 ``(soil_ice, apparent_heat_capacity)``。
    """
    liquid = _unfrozen_water(smc, temperature, smcmax, bexp, option)
    ice = jnp.maximum(0.0, smc - liquid)
    base = smc * 4.18e6 + (1.0 - smcmax) * 2.0e6
    eps = 0.02
    upper = _unfrozen_water(smc, temperature + eps, smcmax, bexp, option)
    lower = _unfrozen_water(smc, temperature - eps, smcmax, bexp, option)
    phase = RHO_W * LF * jnp.maximum(0.0, (upper - lower) / (2.0 * eps))
    return ice, base + phase


def _soil_stress(
    smc: jax.Array,
    root_frac: jax.Array,
    smcmax: jax.Array,
    smcwlt: jax.Array,
    smcref: jax.Array,
    bexp: jax.Array,
    option: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """对应普通版: stress.soil_water_stress。

    同时返回根区加权胁迫和表层胁迫，分别供冠层和地面蒸发使用。
    """
    noah = jnp.clip((smc - smcwlt) / jnp.maximum(1.0e-9, smcref - smcwlt), 0.0, 1.0)
    effective = jnp.clip(smc / jnp.maximum(smcmax, 1.0e-9), 1.0e-4, 1.0)
    psi = -0.1 * effective ** (-bexp)
    psi_ref = -0.1 * jnp.clip(smcref / jnp.maximum(smcmax, 1.0e-9), 1.0e-4, 1.0) ** (-bexp)
    matric = jnp.clip((-1500.0 - psi) / (-1500.0 - psi_ref), 0.0, 1.0)
    ssib = matric**2
    # Fix the four-layer addition order to avoid batch-shape-dependent rounding.
    weighted = root_frac * _switch(option, [lambda _: noah, lambda _: matric,
                                          lambda _: ssib], None)
    root = (weighted[0] + weighted[2]) + (weighted[1] + weighted[3])
    layer0 = _switch(option, [lambda _: noah[0], lambda _: matric[0], lambda _: ssib[0]], None)
    return jnp.clip(root, 0.0, 1.0), jnp.clip(layer0, 0.0, 1.0)


def _ground_resistance(
    smc0: jax.Array,
    dz0: jax.Array,
    smcmax: jax.Array,
    smcwlt: jax.Array,
    bexp: jax.Array,
    option: jax.Array,
) -> jax.Array:
    """对应普通版: ground_evap.ground_resistance。

    实现 Sakaguchi--Zeng (2009)、Sellers (1992) 及两个选项组合。
    """
    sat = jnp.maximum(smcmax, 1.0e-6)
    liquid = jnp.clip(smc0, 0.0, sat)
    saturation = jnp.minimum(1.0, liquid / sat)
    dry_thickness = dz0 * (jnp.exp((1.0 - saturation) ** 2.0) - 1.0) / (jnp.e - 1.0)
    diffusivity = 2.2e-5 * sat**2 * jnp.maximum(1.0e-6, 1.0 - smcwlt / sat) ** (
        2.0 + 3.0 / jnp.maximum(bexp, 1.0e-6)
    )
    sakaguchi = jnp.maximum(1.0e-3, dry_thickness / diffusivity)
    sellers = jnp.exp(8.25 - 4.225 * jnp.maximum(0.0, smc0 / sat))
    adjusted = jnp.exp(8.25 - 6.0 * jnp.maximum(0.0, smc0 / sat))
    return _switch(option, [lambda _: sakaguchi, lambda _: sellers,
                            lambda _: adjusted, lambda _: sakaguchi], None)


def _snow_conductivity(density: jax.Array, temperature: jax.Array, option: jax.Array) -> jax.Array:
    """对应普通版: snow.snow_thermal_conductivity。"""
    rho = jnp.clip(density, 50.0, 917.0)
    ratio = rho / 1000.0
    stieglitz = 2.2 * ratio**1.88
    anderson = 2.9 * ratio**2.0
    constant = jnp.full_like(ratio, 0.30)
    verseghy = (0.023 + 0.234 * ratio**2.0) * (
        1.0 + 0.05 * jnp.clip(temperature - 268.15, 0.0, 5.0) / 5.0
    )
    douville = stieglitz * (1.0 + 0.1 * ratio)
    return jnp.clip(_switch(option, [lambda _: stieglitz, lambda _: anderson,
                                     lambda _: constant, lambda _: verseghy,
                                     lambda _: douville], None), 0.05, 2.5)


def _add_snowfall(
    swe_layers: jax.Array,
    snow_t_layers: jax.Array,
    snow_density_layers: jax.Array,
    snow_count: jax.Array,
    snowfall: jax.Array,
    snow_temperature: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """对应普通版: snow.add_snowfall。

    新雪按 SnowLayerDivideMod 阈值将三层固定数组重新分配；无降雪时保留
    原有层结构。返回层数组、有效层数、总 SWE 和顶层雪温。
    """
    old_total = jnp.sum(swe_layers)
    new_total = old_total + jnp.maximum(0.0, snowfall)
    old_mask = LAYER_INDEX < snow_count
    weighted_t = jnp.sum(jnp.where(old_mask, swe_layers * snow_t_layers, 0.0))
    weighted_d = jnp.sum(jnp.where(old_mask, swe_layers * snow_density_layers, 0.0))
    mean_t = (weighted_t + snowfall * snow_temperature) / jnp.maximum(new_total, 1.0e-12)
    mean_d = (weighted_d + snowfall * 100.0) / jnp.maximum(new_total, 1.0e-12)
    new_count = jnp.where(new_total <= 0.05, 1, jnp.where(new_total <= 0.15, 2, 3)).astype(jnp.int32)
    new_mask = LAYER_INDEX < new_count
    divided = jnp.where(new_mask, new_total / new_count, 0.0)
    divided_t = jnp.where(new_mask, mean_t, FREEZING_POINT)
    divided_d = jnp.where(new_mask, mean_d, 300.0)
    has_snow = snowfall > 0.0
    return lax.cond(
        has_snow,
        lambda _: (divided, divided_t, divided_d, new_count, new_total),
        lambda _: (swe_layers, snow_t_layers, snow_density_layers, snow_count, old_total),
        None,
    )


def _compact_snow(
    swe_layers: jax.Array,
    snow_t_layers: jax.Array,
    snow_density_layers: jax.Array,
    snow_count: jax.Array,
    dt: jax.Array,
    option: jax.Array,
) -> jax.Array:
    """对应普通版: snow.compact_snow。

    Anderson (1976) 与 Abolafia--Rosenzweig (2024) 选项保留相同的密度状态，
    只改变温度沉降和上覆载荷速率；SWE 严格不变。
    """
    mask = LAYER_INDEX < snow_count
    warm_rate = _switch(option, [lambda _: 1.0e-7, lambda _: 6.0e-8], None)
    load_rate = _switch(option, [lambda _: 2.5e-7, lambda _: 1.5e-7], None)
    overburden = jnp.cumsum(swe_layers[::-1])[::-1] - swe_layers
    rate = warm_rate * jnp.exp(jnp.clip(0.08 * (snow_t_layers - 258.15), -4.0, 4.0))
    updated = jnp.minimum(
        550.0,
        snow_density_layers + jnp.maximum(0.0, dt) * (rate * (550.0 - snow_density_layers)
                                                       + load_rate * overburden * 1.0e4),
    )
    return jnp.where(mask & (jnp.sum(swe_layers) > 1.0e-8), updated, snow_density_layers)


def _snow_surface_flux(
    swe_layers: jax.Array,
    snow_t_layers: jax.Array,
    snow_density_layers: jax.Array,
    option: jax.Array,
    surface_temperature: jax.Array,
) -> jax.Array:
    """对应普通版: snow.surface_snow_flux。"""
    top_swe = jnp.maximum(swe_layers[0], 1.0e-8)
    top_depth = top_swe * RHO_W / jnp.maximum(snow_density_layers[0], 50.0)
    conductivity = _snow_conductivity(snow_density_layers[0], snow_t_layers[0], option)
    return conductivity * (surface_temperature - snow_t_layers[0]) / jnp.maximum(
        0.5 * top_depth, 0.01
    )


def _snow_surface_response(swe, temperature, density, count, dt, option):
    """Schur elimination of the new snow top temperature (ordinary snow.py)."""
    mask = LAYER_INDEX < count
    depth = jnp.where(mask, swe * RHO_W / jnp.maximum(density, 50.0), 1.0e-4)
    cap = jnp.where(mask, C_SNOW * depth / jnp.maximum(dt, 1.0e-6), 1.0e12)
    conductivity = _snow_conductivity(density, temperature, option)
    g01 = jnp.where(count > 1,
        (2.0 / (1.0 / conductivity[0] + 1.0 / conductivity[1]))
        / jnp.maximum(0.5 * (depth[0] + depth[1]), 1.0e-4), 0.0)
    g12 = jnp.where(count > 2,
        (2.0 / (1.0 / conductivity[1] + 1.0 / conductivity[2]))
        / jnp.maximum(0.5 * (depth[1] + depth[2]), 1.0e-4), 0.0)
    factor1 = g12 / jnp.maximum(cap[2] + g12, 1.0e-12)
    diagonal1 = (cap[1] + g01 + g12) - factor1 * g12
    rhs1 = cap[1] * (temperature[1] - temperature[0]) + factor1 * (cap[2] * (temperature[2] - temperature[0]))
    factor0 = g01 / jnp.maximum(diagonal1, 1.0e-12)
    admittance = jnp.maximum((cap[0] + g01) - factor0 * g01, 1.0e-12)
    reference = temperature[0] + (factor0 * rhs1) / admittance
    surface_k = conductivity[0] / jnp.maximum(0.5 * depth[0], 0.01)
    effective_k = surface_k / (1.0 + surface_k / admittance)
    return reference, effective_k, surface_k


def _coupled_snow_flux(ts, response):
    reference, effective_k, surface_k = response
    return jnp.maximum(effective_k * (ts - reference), surface_k * (ts - FREEZING_POINT))


def _thomas3(
    lower: jax.Array, diagonal: jax.Array, upper: jax.Array, rhs: jax.Array
) -> jax.Array:
    """对应普通版: snow._implicit_heat_step 的 Thomas 求解。

    三层雪系统固定展开；不使用 Python 动态循环。
    """
    d0 = jnp.maximum(diagonal[0], 1.0e-12)
    cp0 = upper[0] / d0
    dp0 = rhs[0] / d0
    m1 = jnp.maximum(diagonal[1] - lower[1] * cp0, 1.0e-12)
    cp1 = upper[1] / m1
    dp1 = (rhs[1] - lower[1] * dp0) / m1
    m2 = jnp.maximum(diagonal[2] - lower[2] * cp1, 1.0e-12)
    dp2 = (rhs[2] - lower[2] * dp1) / m2
    x2 = dp2
    x1 = dp1 - cp1 * x2
    x0 = dp0 - cp0 * x1
    return jnp.stack((x0, x1, x2))


def _snow_heat(
    swe_layers: jax.Array,
    snow_t_layers: jax.Array,
    snow_density_layers: jax.Array,
    snow_count: jax.Array,
    dt: jax.Array,
    top_flux: jax.Array,
    option: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """对应普通版: snow.apply_snow_heat。

    三层雪固定隐式耦合；超过 0°C 的 apparent energy 转为融水并前移有效层。
    """
    mask = LAYER_INDEX < snow_count
    thickness = swe_layers * RHO_W / jnp.maximum(snow_density_layers, 50.0)
    # The reference protects inter-layer conductance denominators, but its
    # active-layer heat capacity and melt energy use the actual thickness.
    # Inflating a thin active layer here creates artificial thermal mass.
    safe_thickness = jnp.where(mask, thickness, 1.0e-4)
    conductivity = _snow_conductivity(snow_density_layers, snow_t_layers, option)
    k01 = 2.0 / (1.0 / conductivity[0] + 1.0 / conductivity[1])
    k12 = 2.0 / (1.0 / conductivity[1] + 1.0 / conductivity[2])
    g01 = k01 / jnp.maximum(0.5 * (safe_thickness[0] + safe_thickness[1]), 1.0e-4)
    g12 = k12 / jnp.maximum(0.5 * (safe_thickness[1] + safe_thickness[2]), 1.0e-4)
    g01 = jnp.where(mask[0] & mask[1], g01, 0.0)
    g12 = jnp.where(mask[1] & mask[2], g12, 0.0)
    cap = C_SNOW * safe_thickness / jnp.maximum(dt, 1.0e-6)
    cap = jnp.where(mask, cap, 1.0e12)
    lower = jnp.array([0.0, -g01, -g12], dtype=cap.dtype)
    diagonal = jnp.array([cap[0] + g01, cap[1] + g01 + g12, cap[2] + g12])
    upper = jnp.array([-g01, -g12, 0.0], dtype=cap.dtype)
    rhs = cap * snow_t_layers + jnp.array([top_flux, 0.0, 0.0], dtype=cap.dtype)
    updated = _thomas3(lower, diagonal, upper, rhs)
    melt = jnp.zeros_like(swe_layers)
    cooled = updated
    remaining_heat = jnp.asarray(0.0, dtype=updated.dtype)
    for layer in range(3):
        capacity = C_SNOW * safe_thickness[layer]
        excess = (updated[layer] - FREEZING_POINT) * capacity + remaining_heat
        amount = jnp.minimum(swe_layers[layer], jnp.maximum(excess, 0.0) / (LF * RHO_W))
        melt = melt.at[layer].set(jnp.where(mask[layer], amount, 0.0))
        temperature = jnp.where(remaining_heat > 0.0,
            FREEZING_POINT + jnp.minimum(excess, 0.0) / capacity,
            jnp.minimum(updated[layer], FREEZING_POINT))
        cooled = cooled.at[layer].set(jnp.where(mask[layer], temperature, FREEZING_POINT))
        remaining_heat = jnp.where(mask[layer], jnp.maximum(0.0, excess - amount * (LF * RHO_W)), remaining_heat)
    remaining = jnp.maximum(swe_layers - melt, 0.0)
    active = remaining > 1.0e-8
    # 固定长度 gather 把融空的顶层前移；有效层数不改变数组形状。
    count = jnp.maximum(1, jnp.sum(active.astype(jnp.int32))).astype(jnp.int32)
    indices = jnp.nonzero(active, size=3, fill_value=0)[0]
    valid = LAYER_INDEX < jnp.sum(active.astype(jnp.int32))
    sorted_swe = jnp.where(valid, remaining[indices], 0.0)
    sorted_t = jnp.where(valid, cooled[indices], FREEZING_POINT)
    sorted_d = jnp.where(valid, snow_density_layers[indices], 300.0)
    return sorted_swe, sorted_t, sorted_d, count, jnp.sum(melt), remaining_heat


def _snow_albedo(
    swe: jax.Array,
    snow_temperature: jax.Array,
    snow_age_days: jax.Array,
    dt: jax.Array,
    snowfall: jax.Array,
    option: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """对应普通版: snow.update_snow_albedo。

    BATS 是指数雪龄衰减；CLASS 使用雪温分段的平方根雪龄衰减。
    """
    new_snow = snowfall > 1.0e-6
    age = jnp.where(new_snow, 0.0, jnp.maximum(0.0, snow_age_days) + jnp.maximum(0.0, dt) / 86400.0)
    bats = jnp.clip(0.55 + 0.30 * jnp.exp(-age / 3.0), 0.50, 0.90)
    fresh = jnp.where(snow_temperature < 271.15, 0.84, 0.70)
    rate = jnp.where(snow_temperature < 268.15, 0.02, 0.06)
    floor = jnp.where(snow_temperature < 271.15, 0.65, 0.50)
    clas = jnp.clip(fresh - rate * jnp.sqrt(age), floor, fresh)
    value = _switch(option, [lambda _: bats, lambda _: clas, lambda _: bats], None)
    return lax.cond(
        swe <= 1.0e-6,
        lambda _: (jnp.asarray(0.8, dtype=swe.dtype), jnp.asarray(0.0, dtype=swe.dtype)),
        lambda _: (value, age),
        None,
    )


def _thomas4(
    lower: jax.Array, diagonal: jax.Array, upper: jax.Array, rhs: jax.Array
) -> jax.Array:
    """对应普通版: soil._thomas。

    四层土壤追赶法手工展开，依次对应普通版的前消元和回代步骤。
    """
    b0 = jnp.maximum(diagonal[0], 1.0e-12)
    cp0 = upper[0] / b0
    dp0 = rhs[0] / b0
    b1 = jnp.maximum(diagonal[1] - lower[1] * cp0, 1.0e-12)
    cp1 = upper[1] / b1
    dp1 = (rhs[1] - lower[1] * dp0) / b1
    b2 = jnp.maximum(diagonal[2] - lower[2] * cp1, 1.0e-12)
    cp2 = upper[2] / b2
    dp2 = (rhs[2] - lower[2] * dp1) / b2
    b3 = jnp.maximum(diagonal[3] - lower[3] * cp2, 1.0e-12)
    dp3 = (rhs[3] - lower[3] * dp2) / b3
    x3 = dp3
    x2 = dp2 - cp2 * x3
    x1 = dp1 - cp1 * x2
    x0 = dp0 - cp0 * x1
    return jnp.stack((x0, x1, x2, x3))


def _soil_conductivity(smc: jax.Array, smcmax: jax.Array, quartz: jax.Array) -> jax.Array:
    """对应普通版: soil.thermal_conductivity。"""
    sr = jnp.clip(smc / smcmax, 0.0, 1.0)
    kt_dry = 0.20
    kt_sat = 0.60 + 1.60 * quartz
    ke = jnp.where(sr > 0.1, jnp.clip(jnp.log10(jnp.maximum(sr, 1.0e-6)) + 1.0, 0.0, 1.0), 0.0)
    return kt_dry + (kt_sat - kt_dry) * ke


def _soil_heat(
    tsoil: jax.Array,
    smc: jax.Array,
    soil_ice: jax.Array,
    dt: jax.Array,
    g_flux: jax.Array,
    smcmax: jax.Array,
    bexp: jax.Array,
    quartz: jax.Array,
    supercool_option: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """对应普通版: soil.step_soil_heat。

    四层隐式土壤热传导使用固定 Thomas 展开，底边界保持普通版零通量。
    """
    dz = jnp.asarray(DZ, dtype=tsoil.dtype)
    del soil_ice
    _, cap = _phase_and_capacity(smc, tsoil, smcmax, bexp, supercool_option)
    kt = _soil_conductivity(smc, smcmax, quartz)
    k01 = 2.0 / (1.0 / kt[0] + 1.0 / kt[1])
    k12 = 2.0 / (1.0 / kt[1] + 1.0 / kt[2])
    k23 = 2.0 / (1.0 / kt[2] + 1.0 / kt[3])
    g01 = k01 / (0.5 * (dz[0] + dz[1]))
    g12 = k12 / (0.5 * (dz[1] + dz[2]))
    g23 = k23 / (0.5 * (dz[2] + dz[3]))
    lower = jnp.array([0.0, -g01, -g12, -g23], dtype=tsoil.dtype)
    diagonal = jnp.array([
        cap[0] * dz[0] / dt + g01,
        cap[1] * dz[1] / dt + g01 + g12,
        cap[2] * dz[2] / dt + g12 + g23,
        cap[3] * dz[3] / dt + g23,
    ])
    upper = jnp.array([-g01, -g12, -g23, 0.0], dtype=tsoil.dtype)
    rhs = cap * dz / dt * tsoil
    rhs = rhs.at[0].add(g_flux)
    new_tsoil = _thomas4(lower, diagonal, upper, rhs)
    equilibrium = (g_flux == 0.0) & jnp.all(tsoil == tsoil[0])
    new_tsoil = jnp.where(equilibrium, tsoil, new_tsoil)
    new_ice, _ = _phase_and_capacity(new_tsoil * 0.0 + smc, new_tsoil, smcmax, bexp, supercool_option)
    return new_tsoil, new_ice


def _runoff_scheme(
    incoming: jax.Array,
    smc: jax.Array,
    smcmax: jax.Array,
    smcref: jax.Array,
    terrain_index: jax.Array,
    groundwater_depth: jax.Array,
    option: jax.Array,
    equilibrium: bool,
) -> tuple[jax.Array, jax.Array]:
    """对应普通版: runoff.route_runoff/topmodel_runoff/bats_runoff。"""
    dz = jnp.asarray(DZ, dtype=smc.dtype)
    deficit0 = jnp.sum(jnp.maximum(0.0, smcref - smc) * dz)
    terrain_factor = jnp.exp(-terrain_index)
    deficit = jnp.where(
        equilibrium,
        jnp.maximum(deficit0, jnp.maximum(0.0, groundwater_depth) * smcmax * terrain_factor),
        deficit0,
    )
    deficit = jnp.maximum(0.0, deficit * terrain_factor)
    wetness = jnp.clip(jnp.exp(terrain_index) * jnp.exp(-jnp.maximum(0.0, groundwater_depth) / 2.0), 0.0, 1.0)
    top1 = jnp.maximum(0.0, incoming - deficit) * wetness
    top2 = jnp.clip(top1, 0.0, incoming)
    wet = jnp.mean(jnp.clip(smc / jnp.maximum(smcmax, 1.0e-6), 0.0, 1.0))
    bats = incoming * jnp.clip(wet**3, 0.0, 1.0)
    surface = _switch(option, [lambda _: top1, lambda _: top2,
                               lambda _: jnp.asarray(0.0, dtype=incoming.dtype),
                               lambda _: bats], None)
    surface = jnp.clip(surface, 0.0, incoming)
    return surface, incoming - surface


def _route_runoff(
    incoming: jax.Array,
    smc: jax.Array,
    smcmax: jax.Array,
    smcref: jax.Array,
    terrain_index: jax.Array,
    groundwater_depth: jax.Array,
    surface_option: jax.Array,
    subsurface_option: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """对应普通版: runoff.route_runoff。"""
    surface, subsurface = _runoff_scheme(
        incoming, smc, smcmax, smcref, terrain_index, groundwater_depth,
        surface_option, equilibrium=surface_option == 2,
    )
    second_surface, final_subsurface = _runoff_scheme(
        subsurface, smc, smcmax, smcref, terrain_index, groundwater_depth,
        subsurface_option, equilibrium=subsurface_option == 2,
    )
    same = surface_option == subsurface_option
    return jnp.where(same, surface, surface + second_surface), jnp.where(
        same, subsurface, final_subsurface
    )


def _soil_water(
    smc: jax.Array,
    soil_ice: jax.Array,
    root_frac: jax.Array,
    infil_m: jax.Array,
    transp_m: jax.Array,
    esoil_m: jax.Array,
    dt: jax.Array,
    smcmax: jax.Array,
    smcwlt: jax.Array,
    smcref: jax.Array,
    satdk: jax.Array,
    bexp: jax.Array,
    permeability_option: jax.Array,
    runoff_surface_option: jax.Array,
    runoff_subsurface_option: jax.Array,
    terrain_index: jax.Array,
    groundwater_depth: jax.Array,
    supercool_option: jax.Array,
    tsoil: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """对应普通版: soil.step_soil_water。

    Fill-and-spill、根系吸水、Campbell 排水和径流路由都按四层固定展开。
    """
    dz = jnp.asarray(DZ, dtype=smc.dtype)
    del soil_ice
    work = smc
    remaining = jnp.maximum(0.0, infil_m)
    for index in range(4):
        room = jnp.maximum(0.0, (smcmax - work[index]) * dz[index])
        add = jnp.minimum(remaining, room)
        work = work.at[index].add(add / dz[index])
        remaining = remaining - add
    runoff_surface = jnp.maximum(0.0, remaining)
    phase_ice, _ = _phase_and_capacity(work, tsoil, smcmax, bexp, supercool_option)

    transp_taken = jnp.asarray(0.0, dtype=smc.dtype)
    for index in range(4):
        available = jnp.maximum(0.0, work[index] - smcwlt) * dz[index]
        take = jnp.minimum(transp_m * root_frac[index], available)
        work = work.at[index].add(-take / dz[index])
        transp_taken = transp_taken + take
    available0 = work[0] * dz[0]
    esoil_taken = jnp.minimum(esoil_m, jnp.maximum(0.0, available0))
    work = work.at[0].add(-esoil_taken / dz[0])

    permeability_linear = jnp.ones_like(work)
    permeability_nonlinear = jnp.exp(-7.0 * phase_ice / jnp.maximum(smcmax, 1.0e-6))
    permeability = _switch(permeability_option, [lambda _: permeability_linear,
                                                   lambda _: permeability_nonlinear], None)
    runoff_deep = jnp.asarray(0.0, dtype=smc.dtype)
    for index in range(4):
        q = satdk * permeability[index] * jnp.clip(work[index] / smcmax, 0.0, 1.0) ** (2.0 * bexp + 3.0)
        excess = jnp.maximum(0.0, work[index] - smcref) * dz[index]
        drain = jnp.minimum(excess, q * dt)
        work = work.at[index].add(-drain / dz[index])
        if index < 3:
            work = work.at[index + 1].add(drain / dz[index + 1])
        else:
            runoff_deep = runoff_deep + drain
    for index in range(4):
        excess = jnp.maximum(0.0, work[index] - smcmax) * dz[index]
        work = work.at[index].add(-excess / dz[index])
        if index < 3:
            work = work.at[index + 1].add(excess / dz[index + 1])
        else:
            runoff_deep = runoff_deep + excess
    runoff_surface, runoff_deep2 = _route_runoff(
        runoff_surface + runoff_deep, work, smcmax, smcref, terrain_index,
        groundwater_depth, runoff_surface_option, runoff_subsurface_option,
    )
    final_ice, _ = _phase_and_capacity(work, tsoil, smcmax, bexp, supercool_option)
    return work, final_ice, runoff_surface, runoff_deep2, transp_taken, esoil_taken


def _stomatal_resistance(
    smc: jax.Array,
    root_frac: jax.Array,
    rsmin: jax.Array,
    lai: jax.Array,
    swdown: jax.Array,
    tair: jax.Array,
    qair: jax.Array,
    ps: jax.Array,
    smcmax: jax.Array,
    smcwlt: jax.Array,
    smcref: jax.Array,
    bexp: jax.Array,
    option_water: jax.Array,
    option_stomata: jax.Array,
) -> jax.Array:
    """对应普通版: canopy.stomatal_resistance。

    Ball--Berry 的 A 使用阶段 1 经验光响应曲线，完整 Farquhar 留到阶段 2。
    """
    root_beta, _ = _soil_stress(smc, root_frac, smcmax, smcwlt, smcref, bexp, option_water)
    es = _sat_vapor_pressure(tair)
    ea = qair * ps / (0.622 + 0.378 * qair)
    vpd = jnp.maximum(0.0, es - ea)
    f_light = jnp.clip(swdown / 500.0, 0.02, 1.0)
    f_temp = jnp.clip(1.0 - 0.0016 * (298.0 - tair) ** 2, 0.05, 1.0)
    f_vpd = jnp.clip(1.0 - vpd / 4000.0, 0.10, 1.0)
    jarvis = jnp.clip(rsmin / (f_light * f_temp * f_vpd * jnp.maximum(0.02, root_beta)), rsmin, 5000.0)
    par = jnp.maximum(0.0, swdown) * 2.0
    assimilation = 25.0 * (1.0 - jnp.exp(-par / 450.0)) * f_temp * jnp.maximum(0.02, root_beta)
    rh = jnp.clip(ea / jnp.maximum(es, 1.0), 0.05, 1.0)
    gs_mol = 0.01 + 9.0 * assimilation * rh / 400.0
    gs_ms = gs_mol * 8.314462618 * tair / jnp.maximum(ps, 1.0)
    ball_berry = jnp.clip(1.0 / jnp.maximum(gs_ms, 1.0e-6), rsmin, 5000.0)
    return _switch(option_stomata, [lambda _: ball_berry, lambda _: jarvis], None)


def _penman_monteith(
    rn: jax.Array,
    tair: jax.Array,
    qair: jax.Array,
    ps: jax.Array,
    ra: jax.Array,
    rs: jax.Array,
) -> jax.Array:
    """对应普通版: canopy.penman_monteith。"""
    es = _sat_vapor_pressure(tair)
    ea = qair * ps / (0.622 + 0.378 * qair)
    tc = tair - 273.15
    des_dt = es * 17.67 * 243.5 / (tc + 243.5) ** 2
    gamma = CP * ps / (0.622 * LV)
    rho = _air_density(tair, qair, ps)
    numerator = des_dt * rn + rho * CP * jnp.maximum(0.0, es - ea) / jnp.maximum(ra, 1.0)
    denominator = des_dt + gamma * (1.0 + rs / jnp.maximum(ra, 1.0))
    return jnp.maximum(0.0, numerator / denominator)


def _secant_solve(func: Callable[[jax.Array], jax.Array], x0: jax.Array, x1: jax.Array) -> jax.Array:
    """对应普通版: energy.secant_solve。

    固定 25 次割线更新；收敛后冻结，并保留普通版 131 点扫描/40 次二分回退。
    """
    f0 = func(x0)
    best = x0
    best_abs = jnp.abs(f0)

    def body(_, carry):
        old_x0, old_x1, old_f0, old_best, old_best_abs, stopped, converged = carry
        f1 = func(old_x1)
        better = jnp.abs(f1) < old_best_abs
        best_new = jnp.where(better, old_x1, old_best)
        best_abs_new = jnp.where(better, jnp.abs(f1), old_best_abs)
        denom = f1 - old_f0
        x2_raw = old_x1 - f1 * (old_x1 - old_x0) / jnp.where(jnp.abs(denom) < 1.0e-12, 1.0, denom)
        x2 = jnp.clip(jnp.where(jnp.abs(denom) < 1.0e-12, old_x1, x2_raw), 200.0, 330.0)
        hit = jnp.abs(f1) < 1.0e-3
        freeze = stopped | hit | (jnp.abs(denom) < 1.0e-12)
        return (jnp.where(freeze, old_x0, old_x1),
                jnp.where(freeze, old_x1, x2), jnp.where(freeze, old_f0, f1),
                jnp.where(stopped, old_best, best_new),
                jnp.where(stopped, old_best_abs, best_abs_new), freeze,
                converged | (~stopped & hit))

    _, answer, _, best, _, _, converged = lax.fori_loop(
        0, 25, body, (x0, x1, f0, best, best_abs, jnp.asarray(False), jnp.asarray(False))
    )

    def fallback(_):
        def scan(index, carry):
            previous, bracket = carry
            value = func(jnp.asarray(201.0, dtype=x0.dtype) + index)
            crossing = jnp.isfinite(previous) & jnp.isfinite(value) & (previous * value <= 0.0)
            return value, jnp.where((bracket < 0) & crossing, index, bracket)

        _, bracket = lax.fori_loop(
            0, 130, scan, (func(jnp.asarray(200.0, dtype=x0.dtype)), jnp.asarray(-1, jnp.int32))
        )

        def bisect(_):
            left = jnp.asarray(200.0, dtype=x0.dtype) + bracket
            right = left + 1.0

            def iteration(_, carry):
                lo, hi, flo, result, done = carry
                mid = 0.5 * (lo + hi)
                fm = func(mid)
                hit = jnp.abs(fm) < 1.0e-3
                go_left = flo * fm <= 0.0
                return (jnp.where(done | go_left, lo, mid),
                        jnp.where(done | ~go_left, hi, mid),
                        jnp.where(done | go_left, flo, fm),
                        jnp.where(~done & hit, mid, result), done | hit)

            lo, hi, _, result, done = lax.fori_loop(
                0, 40, iteration, (left, right, func(left), left, jnp.asarray(False))
            )
            return jnp.where(done, result, 0.5 * (lo + hi))

        return lax.cond(bracket >= 0, bisect, lambda _: best, None)

    return lax.cond(converged, lambda _: answer, fallback, None)


def step_column_jax(
    fvec: jax.Array, ivec: jax.Array, forcing_vec: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """对应普通版: driver.step_column。

    单柱 JAX 纯函数；返回更新后的 ``(fvec, ivec, diag_vec)``。diag 顺序为
    ``ts, rnet, h, le, g, et_mm, prcp_mm, runoff_mm, melt_mm, swe, canw,
    residual``。
    """
    _require_jax()
    fvec = jnp.asarray(fvec)
    dz = jnp.asarray(DZ, dtype=fvec.dtype)
    ivec = jnp.asarray(ivec, dtype=jnp.int32)
    dt, tair, qair, wind, ps, swdown, lwdown, prcp = forcing_vec
    tsoil = fvec[0:4]
    smc = fvec[4:8]
    soil_ice = fvec[8:12]
    root_frac = fvec[12:16]
    snow_swe_layers = fvec[16:19]
    snow_t_layers = fvec[19:22]
    snow_density_layers = fvec[22:25]
    ts_old, swe_old, snow_t_old, snow_albedo_old = fvec[25:29]
    snow_age_old, canw_old, lai, fveg = fvec[29:33]
    z0m, albedo, emiss, rsmin = fvec[33:37]
    terrain_index, groundwater_depth = fvec[37:39]
    smcmax, smcwlt, smcref, satdk, bexp, quartz = fvec[39:45]
    snow_count = ivec[SNOW_COUNT_INDEX]

    rho = _air_density(tair, qair, ps)
    ce_initial = _exchange_velocity(wind, tair, qair, ps, z0m, tair, _option(ivec, "OptSurfaceDrag"))
    ra_initial = 1.0 / jnp.maximum(ce_initial, 1.0e-6)
    total_prcp = jnp.maximum(0.0, prcp) * jnp.maximum(0.0, dt)
    snow_fraction = _snow_fraction(
        tair, qair, ps, _option(ivec, "OptRainSnowPartition")
    )
    snow_m = total_prcp * snow_fraction
    rain_m = total_prcp - snow_m
    canmax = 0.5e-3 * lai
    intercepted = jnp.minimum(rain_m, jnp.maximum(0.0, canmax - canw_old))
    canw_intercepted = canw_old + intercepted
    throughfall = rain_m - intercepted
    (
        snow_swe_layers,
        snow_t_layers,
        snow_density_layers,
        snow_count,
        _,
    ) = _add_snowfall(
        snow_swe_layers, snow_t_layers, snow_density_layers, snow_count,
        snow_m, tair,
    )
    snow_density_layers = _compact_snow(
        snow_swe_layers, snow_t_layers, snow_density_layers, snow_count, dt,
        _option(ivec, "OptSnowCompaction"),
    )
    swe = jnp.sum(snow_swe_layers)
    snow_t = jnp.where(swe > 1.0e-8, snow_t_layers[0], FREEZING_POINT)

    rs = _stomatal_resistance(
        smc, root_frac, rsmin, lai, swdown, tair, qair, ps, smcmax, smcwlt,
        smcref, bexp, _option(ivec, "OptSoilWaterTranspiration"),
        _option(ivec, "OptStomataResistance"),
    )
    rn_canopy = fveg * ((1.0 - albedo) * swdown + emiss * (lwdown - SIGMA * tair**4))
    tr_le_initial = _penman_monteith(
        jnp.maximum(0.0, rn_canopy), tair, qair, ps, ra_initial, rs
    )
    snow_cover = jnp.minimum(1.0, swe / 0.02)
    tr_le_initial = tr_le_initial * (1.0 - snow_cover)
    avail_root = jnp.sum(jnp.maximum(0.0, smc - smcwlt) * dz)
    tr_m_potential = tr_le_initial / LV * dt / RHO_W
    tr_le = tr_le_initial * jnp.where(
        tr_m_potential > avail_root,
        avail_root / jnp.maximum(tr_m_potential, 1.0e-12),
        1.0,
    )
    tr_m = tr_le / LV * dt / RHO_W

    option_drag = _option(ivec, "OptSurfaceDrag")
    option_snow_k = _option(ivec, "OptSnowThermConduct")
    option_supercool = _option(ivec, "OptSoilSupercoolWater")

    def snow_energy(_):
        snow_albedo = snow_albedo_old
        snow_emiss = jnp.asarray(0.99, dtype=fvec.dtype)
        response = _snow_surface_response(snow_swe_layers, snow_t_layers,
            snow_density_layers, snow_count, dt, option_snow_k)

        def residual(ts):
            rnet_local = (1.0 - snow_albedo) * swdown + snow_emiss * (lwdown - SIGMA * ts**4)
            ce_local = _exchange_velocity(wind, tair, qair, ps, z0m, ts, option_drag)
            h_local = rho * CP * ce_local * (ts - tair)
            g_local = _coupled_snow_flux(ts, response)
            return rnet_local - h_local - g_local

        ts = _secant_solve(residual, tair, tair + 1.0)
        rnet = (1.0 - snow_albedo) * swdown + snow_emiss * (lwdown - SIGMA * ts**4)
        ce = _exchange_velocity(wind, tair, qair, ps, z0m, ts, option_drag)
        h = rho * CP * ce * (ts - tair)
        g_snow = _coupled_snow_flux(ts, response)
        new_layers = _snow_heat(
            snow_swe_layers, snow_t_layers, snow_density_layers, snow_count,
            dt, g_snow, option_snow_k,
        )
        return (
            ts, rnet, h, jnp.asarray(0.0, dtype=fvec.dtype), g_snow,
            jnp.asarray(0.0, dtype=fvec.dtype), jnp.asarray(0.0, dtype=fvec.dtype),
            new_layers[0], new_layers[1], new_layers[2], new_layers[3], new_layers[4],
            jnp.asarray(0.0, dtype=fvec.dtype), rnet - h - g_snow, new_layers[5] / dt,
        )

    def bare_energy(_):
        beta_root, beta_surface = _soil_stress(
            smc, root_frac, smcmax, smcwlt, smcref, bexp,
            _option(ivec, "OptSoilWaterTranspiration"),
        )
        del beta_root
        ground_res = _ground_resistance(
            smc[0], dz[0], smcmax, smcwlt, bexp,
            _option(ivec, "OptGroundResistanceEvap"),
        )
        kt1 = _soil_conductivity(smc[0:1], smcmax, quartz)[0]
        canw_cap_rate = canw_intercepted * LV * RHO_W / jnp.maximum(dt, 1.0e-6)
        smc0_cap_rate = smc[0] * dz[0] * LV * RHO_W / jnp.maximum(dt, 1.0e-6)

        def residual(ts):
            rnet_local = (1.0 - albedo) * swdown + emiss * (lwdown - SIGMA * ts**4)
            ce_local = _exchange_velocity(wind, tair, qair, ps, z0m, ts, option_drag)
            h_local = rho * CP * ce_local * (ts - tair)
            dqs = _qsat(ts, ps) - qair
            ra_local = 1.0 / jnp.maximum(ce_local, 1.0e-6)
            le_g_local = jnp.minimum(
                beta_surface * rho * LV * jnp.maximum(0.0, dqs) / (ra_local + ground_res),
                smc0_cap_rate,
            )
            le_c_local = jnp.minimum(
                rho * LV * ce_local * dqs * jnp.minimum(1.0, canw_intercepted / jnp.maximum(1.0e-9, 0.2 * canmax)),
                canw_cap_rate,
            )
            g_local = kt1 * (ts - tsoil[0]) / (0.5 * dz[0])
            return rnet_local - h_local - le_g_local - le_c_local - tr_le - g_local

        ts = _secant_solve(residual, tair, tair + 1.0)
        rnet = (1.0 - albedo) * swdown + emiss * (lwdown - SIGMA * ts**4)
        ce = _exchange_velocity(wind, tair, qair, ps, z0m, ts, option_drag)
        h = rho * CP * ce * (ts - tair)
        dqs = _qsat(ts, ps) - qair
        ra = 1.0 / jnp.maximum(ce, 1.0e-6)
        le_g = jnp.minimum(
            beta_surface * rho * LV * jnp.maximum(0.0, dqs) / (ra + ground_res),
            smc0_cap_rate,
        )
        le_c = jnp.minimum(
            rho * LV * ce * dqs * jnp.minimum(1.0, canw_intercepted / jnp.maximum(1.0e-9, 0.2 * canmax)),
            canw_cap_rate,
        )
        g_soil = kt1 * (ts - tsoil[0]) / (0.5 * dz[0])
        ecan = le_c / LV * dt / RHO_W
        return (
            ts, rnet, h, le_g + le_c + tr_le, g_soil, ecan,
            le_g / LV * dt / RHO_W, snow_swe_layers, snow_t_layers,
            snow_density_layers, snow_count, jnp.asarray(0.0, dtype=fvec.dtype),
            jnp.maximum(0.0, canw_intercepted - ecan),
            rnet - h - le_g - le_c - tr_le - g_soil, g_soil,
        )

    snow_surface = swe > 1.0e-4
    energy = lax.cond(snow_surface, snow_energy, bare_energy, None)
    (
        ts, rnet, h, le_total, g_flux, ecan_m, esoil_potential,
        snow_swe_layers, snow_t_layers, snow_density_layers, snow_count,
        melt, canw_after_energy, residual, soil_flux,
    ) = energy
    smc, soil_ice, runoff_surface, runoff_deep, transp_taken, esoil_taken = _soil_water(
        smc, soil_ice, root_frac, throughfall + melt, tr_m, esoil_potential,
        dt, smcmax, smcwlt, smcref, satdk, bexp,
        _option(ivec, "OptSoilPermeabilityFrozen"),
        _option(ivec, "OptRunoffSurface"), _option(ivec, "OptRunoffSubsurface"),
        terrain_index, groundwater_depth, option_supercool, tsoil,
    )
    tsoil, soil_ice = _soil_heat(
        tsoil, smc, soil_ice, dt, soil_flux,
        smcmax, bexp, quartz, option_supercool
    )
    swe = jnp.sum(snow_swe_layers)
    snow_t = jnp.where(swe > 1.0e-8, snow_t_layers[0], FREEZING_POINT)
    snow_albedo, snow_age = _snow_albedo(
        swe, snow_t, snow_age_old, dt, snow_m, _option(ivec, "OptSnowAlbedo")
    )
    canw = jnp.where(snow_surface, canw_intercepted, canw_after_energy)
    et_m = transp_taken + esoil_taken + ecan_m
    runoff_m = runoff_surface + runoff_deep
    diag = jnp.stack((
        ts, rnet, h, le_total, g_flux, et_m * 1000.0,
        prcp * dt * 1000.0, runoff_m * 1000.0, melt * 1000.0,
        swe, canw, residual,
    ))
    updated = fvec
    updated = updated.at[0:4].set(tsoil)
    updated = updated.at[4:8].set(smc)
    updated = updated.at[8:12].set(soil_ice)
    updated = updated.at[16:19].set(snow_swe_layers)
    updated = updated.at[19:22].set(snow_t_layers)
    updated = updated.at[22:25].set(snow_density_layers)
    updated = updated.at[25].set(ts)
    updated = updated.at[26].set(swe)
    updated = updated.at[27].set(snow_t)
    updated = updated.at[28].set(snow_albedo)
    updated = updated.at[29].set(snow_age)
    updated = updated.at[30].set(canw)
    updated_int = ivec.at[SNOW_COUNT_INDEX].set(snow_count.astype(jnp.int32))
    return updated, updated_int, diag


def _run_one_jax(
    fvec: jax.Array, ivec: jax.Array, forcings: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """对应普通版: driver.run_column。"""

    def body(carry, forcing):
        fnew, inew, diag = step_column_jax(carry[0], carry[1], forcing)
        return (fnew, inew), diag

    (fnew, inew), diagnostics = lax.scan(body, (fvec, ivec), forcings)
    return fnew, inew, diagnostics


_BATCH_JIT = (
    jax.jit(jax.vmap(_run_one_jax, in_axes=(0, 0, 0)))
    if jax is not None
    else None
)


def run_columns_jax(
    fvecs: jax.Array, ivecs: jax.Array, forcings: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """对应普通版: driver.run_columns。

    ``fvecs``/``ivecs`` 的首轴是柱轴；`forcings` 可为共享的 ``(T, 8)``，
    也可为每柱独立的 ``(N, T, 8)``。返回最终状态和 ``(N, T, 12)`` 诊断。
    时间轴由 ``lax.scan``，柱轴由 ``vmap`` 实现。
    """
    _require_jax()
    fvecs = jnp.asarray(fvecs)
    ivecs = jnp.asarray(ivecs, dtype=jnp.int32)
    forcing_array = jnp.asarray(forcings)
    if forcing_array.ndim == 2:
        forcing_array = jnp.broadcast_to(
            forcing_array, (fvecs.shape[0], forcing_array.shape[0], forcing_array.shape[1])
        )
    # 广播在 jit 外完成，避免 Windows CPU wheel 的 HLO verifier 将
    # broadcast 与 scan/vmap 合并时产生循环；核心批处理仍由 jax.jit(vmap)
    # 编译，CUDA/JAX Linux 路径无需改变接口。
    return _BATCH_JIT(fvecs, ivecs, forcing_array)


__all__ = ["step_column_jax", "run_columns_jax", "FLOAT_SIZE", "INT_SIZE"]

"""积雪 bulk 状态、反照率与融化。

v0.1 简化：雪盖存在时地表即雪面，能量平衡按雪反照率求解 Ts，
雪内热传导加热雪体，雪温到 0°C 后的盈余能量转为融化。
反照率方案参考 Noah-MP ``SnowAlbedoBatsMod.F90`` 与
``SnowAlbedoClassMod.F90``；SNICAR (option 3) 留给后续阶段。
"""
from __future__ import annotations

import numpy as np

LF = 3.34e5    # 融化潜热 J/kg
RHO_W = 1000.0
RHO_SNOW = 300.0
C_SNOW = RHO_SNOW * 2100.0  # 雪体积热容量 J/(m3 K)
K_SNOW = 0.30               # 雪导热系数 W/(m K)


def snow_depth(swe: float) -> float:
    return swe * RHO_W / RHO_SNOW


def albedo_bats(age_days: float, snow_temperature: float, new_snow: bool = False) -> float:
    """BATS (Dickinson et al., 1993) 指数雪龄衰减，返回雪反照率。"""
    del snow_temperature  # BATS v0.1 方案只按雪龄衰减
    age = 0.0 if new_snow else max(0.0, float(age_days))
    return float(np.clip(0.55 + 0.30 * np.exp(-age / 3.0), 0.50, 0.90))


def albedo_class(age_days: float, snow_temperature: float, new_snow: bool = False) -> float:
    """CLASS (Verseghy, 1991) 的雪龄/温度分段反照率。

    冷雪采用较慢的 grain-growth 衰减，接近融点的湿雪采用较快衰减；
    降雪会将雪龄重置为零。分段系数是单柱基线的 CLASS 简化，保持与
    Noah-MP ``SnowAlbedoClassMod.F90`` 相同的物理方向。
    """
    age = 0.0 if new_snow else max(0.0, float(age_days))
    fresh = 0.84 if float(snow_temperature) < 271.15 else 0.70
    rate = 0.02 if float(snow_temperature) < 268.15 else 0.06
    floor = 0.65 if float(snow_temperature) < 271.15 else 0.50
    return float(np.clip(fresh - rate * np.sqrt(age), floor, fresh))


SNOW_ALBEDO = {1: albedo_bats, 2: albedo_class}


def update_snow_albedo(
    swe: float,
    albedo: float,
    age_days: float,
    snow_temperature: float,
    dt: float,
    snowfall_m: float,
    option: int,
) -> tuple[float, float]:
    """纯函数形式更新 ``(albedo, age_days)``，供状态驱动调用。"""
    if swe <= 1.0e-6:
        return 0.8, 0.0
    if option not in SNOW_ALBEDO:
        raise NotImplementedError(
            f"OptSnowAlbedo={option} 尚未实现；阶段 1 支持 1 (BATS)、2 (CLASS)"
        )
    is_new = snowfall_m > 1.0e-6
    age = 0.0 if is_new else max(0.0, float(age_days)) + max(0.0, float(dt)) / 86400.0
    value = SNOW_ALBEDO[option](age, snow_temperature, new_snow=is_new)
    return float(value), float(age)


def age_albedo(state, dt: float, snowfall_m: float, option: int = 1) -> None:
    """更新状态中的反照率（状态已在 ``step_column`` 入口复制）。"""
    state.snow_albedo, state.snow_age_days = update_snow_albedo(
        state.swe, state.snow_albedo, state.snow_age_days, state.snow_t,
        dt, snowfall_m, option
    )


def apply_snow_heat(state, dt: float, g_snow: float) -> float:
    """雪内热传导通量 g_snow（向下为正 W/m2）加热雪体；返回融水 (m)。原地修改。"""
    if state.swe <= 1e-6:
        return 0.0
    dsnow = max(0.05, snow_depth(state.swe))
    state.snow_t += g_snow * dt / (C_SNOW * dsnow)
    melt = 0.0
    if state.snow_t > 273.15:
        melt = (state.snow_t - 273.15) * C_SNOW * dsnow / (LF * RHO_W)
        melt = min(melt, state.swe)
        state.swe -= melt
        state.snow_t = 273.15
    return melt

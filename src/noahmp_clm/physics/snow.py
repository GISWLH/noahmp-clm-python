"""单层 bulk 积雪。

v0.1 简化：雪盖存在时地表即雪面，能量平衡按雪反照率求解 Ts，
雪内热传导加热雪体，雪温到 0°C 后的盈余能量转为融化。
升华、雪层细分、风吹雪是阶段 1 的工作。
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


def age_albedo(state, dt: float, snowfall_m: float) -> None:
    """反照率衰减 / 新雪重置。原地修改。"""
    if state.swe <= 1e-6:
        state.snow_albedo = 0.8
        state.snow_age_days = 0.0
        return
    if snowfall_m > 1e-6:
        state.snow_age_days = 0.0
    else:
        state.snow_age_days += dt / 86400.0
    state.snow_albedo = 0.55 + 0.30 * np.exp(-state.snow_age_days / 3.0)


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

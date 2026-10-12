"""冻土未冻水、相变热容与冻结渗透率。

未冻水含量参考 Niu & Yang (2006, JHM) 的冻结特征曲线；温度低于冰点时
由 Clausius--Clapeyron/Clapp--Hornberger 关系反推液态水。相变潜热用
apparent heat capacity (AHC) 进入土壤隐式热传导，避免额外的外层迭代。
``OptSoilPermeabilityFrozen=2`` 用冰含量指数衰减导水率。
"""
from __future__ import annotations

import numpy as np

from ..column import ColumnState, SoilParams

FREEZING_POINT = 273.15
LATENT_HEAT = 3.34e5  # J/kg
RHO_WATER = 1000.0
GAS_CONSTANT_WATER = 461.5  # J/(kg K)


def _niu_yang_liquid(smc: np.ndarray, temperature: np.ndarray, p: SoilParams) -> np.ndarray:
    temp = np.asarray(temperature, dtype=float)
    total = np.maximum(0.0, np.asarray(smc, dtype=float))
    undercooling = np.maximum(0.0, FREEZING_POINT - temp)
    clausius = LATENT_HEAT / GAS_CONSTANT_WATER * undercooling / (FREEZING_POINT**2)
    # 0.1 Pa 量级的饱和吸力尺度把 Clausius--Clapeyron 温差映射到
    # Campbell 土壤水势；数值对应 Niu--Yang 冻结曲线的典型量级。
    potential = p.smcmax * (
        np.maximum(clausius, 1.0e-8) / 5.0e-5
    ) ** (-1.0 / max(p.bexp, 1.0e-6))
    return np.where(temp >= FREEZING_POINT, total, np.minimum(total, potential))


def unfrozen_water_content(
    smc: np.ndarray, temperature: np.ndarray, p: SoilParams, option: int = 1
) -> np.ndarray:
    """返回各土层未冻体积含水量 (m3/m3)。"""
    if option == 1:
        return _niu_yang_liquid(smc, temperature, p)
    if option == 2:
        # Koren (1999) 迭代入口：用 Niu--Yang 估计作为初值并固定三次收敛。
        liquid = _niu_yang_liquid(smc, temperature, p)
        for _ in range(3):
            liquid = 0.5 * (liquid + np.minimum(smc, _niu_yang_liquid(liquid, temperature, p)))
        return np.minimum(np.asarray(smc, dtype=float), liquid)
    raise ValueError(f"OptSoilSupercoolWater={option} 无效；允许值为 1 或 2")


def update_soil_phase(state: ColumnState) -> np.ndarray:
    """按状态温度刷新 ``soil_ice``，返回未冻水含量。"""
    liquid = unfrozen_water_content(
        state.smc, state.tsoil, state.soil, state.options.OptSoilSupercoolWater
    )
    state.soil_ice[:] = np.maximum(0.0, state.smc - liquid)
    return liquid


def apparent_heat_capacity(
    smc: np.ndarray, temperature: np.ndarray, p: SoilParams, option: int = 1
) -> np.ndarray:
    """含相变潜热的表观体积热容 (J m-3 K-1)。"""
    base = np.asarray(smc, dtype=float) * 4.18e6 + (1.0 - p.smcmax) * 2.0e6
    temp = np.asarray(temperature, dtype=float)
    eps = 0.02
    upper = unfrozen_water_content(smc, temp + eps, p, option)
    lower = unfrozen_water_content(smc, temp - eps, p, option)
    phase = RHO_WATER * LATENT_HEAT * np.maximum(0.0, (upper - lower) / (2.0 * eps))
    return base + phase


def permeability_factor(
    state: ColumnState, option: int, liquid: np.ndarray | None = None
) -> np.ndarray:
    """返回冻结土壤导水率倍率（linear=1，nonlinear=指数衰减）。"""
    if option == 1:
        return np.ones_like(state.smc, dtype=float)
    if option == 2:
        ice = state.soil_ice if liquid is None else np.maximum(0.0, state.smc - liquid)
        return np.exp(-7.0 * ice / max(state.soil.smcmax, 1.0e-6))
    raise ValueError(f"OptSoilPermeabilityFrozen={option} 无效；允许值为 1 或 2")


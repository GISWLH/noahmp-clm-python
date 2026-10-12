"""地表层动量/热量交换系数。

``most_exchange_velocity`` 是 Monin--Obukhov similarity theory (MOST) 的
标量交换实现，依据 Noah-MP ``ResistanceAboveCanopyMostMod.F90`` 和
``PsychrometricVariableMod.F90`` 的迭代结构重写。稳定度通过 Obukhov 长度
``L = -rho*Cp*T*u*^3/(k*g*H)`` 更新，稳定与不稳定情形分别使用 Businger--
Dyer 相似函数；相对变化小于 ``1e-4`` 或 20 次迭代后停止。

``OptSurfaceDrag=2`` 保留 v0.1 的 Chen et al. (1997) 风格中性 bulk
交换系数。两个方案都返回速度形式的 ``CH = ce`` (m s-1)，同一结果由驱动
用于感热和潜热通量。
"""
from __future__ import annotations

import numpy as np

from ..column import Forcing
from . import energy

GRAVITY = 9.81


def _psi_m(z_over_l: float) -> float:
    """Businger--Dyer 动量稳定度积分函数。"""
    if z_over_l < 0.0:
        x = (1.0 - 16.0 * z_over_l) ** 0.25
        return float(
            2.0 * np.log((1.0 + x) / 2.0)
            + np.log((1.0 + x * x) / 2.0)
            - 2.0 * np.arctan(x)
            + np.pi / 2.0
        )
    return float(-5.0 * z_over_l)


def _psi_h(z_over_l: float) -> float:
    """Businger--Dyer 热量/水汽稳定度积分函数。"""
    if z_over_l < 0.0:
        x = (1.0 - 16.0 * z_over_l) ** 0.25
        return float(2.0 * np.log((1.0 + x * x) / 2.0))
    return float(-5.0 * z_over_l)


def most_exchange_velocity(
    forcing: Forcing,
    z0m: float,
    surface_temperature: float,
    *,
    max_iterations: int = 20,
    tolerance: float = 1.0e-4,
) -> float:
    """计算 MOST 标量交换速度 ``ce = CH * wind`` (m/s)。

    迭代只使用当前柱和当前强迫，因而是无副作用的纯函数。零感热通量
    回退到中性系数；风速下限与原 bulk 方案相同。
    """
    wind = max(float(forcing.wind), 0.3)
    zref = 10.0
    z0m_safe = float(np.clip(z0m, 1.0e-5, 0.5 * zref))
    z0h = max(z0m_safe / 10.0, 1.0e-6)
    neutral_m = np.log(zref / z0m_safe)
    neutral_h = np.log(zref / z0h)
    ce = energy.bulk_ch(wind, z0m_safe) * wind
    rho = energy.air_density(forcing)
    k = energy.KARMAN
    obukhov = np.inf
    for _ in range(max_iterations):
        zeta_m = 0.0 if not np.isfinite(obukhov) else zref / obukhov
        zeta_h = 0.0 if not np.isfinite(obukhov) else zref / obukhov
        den_m = max(1.0, neutral_m - _psi_m(zeta_m))
        den_h = max(1.0, neutral_h - _psi_h(zeta_h))
        ustar = k * wind / den_m
        ch = k * k / (den_m * den_h)
        ce_new = max(1.0e-6, ch * wind)
        sensible = rho * energy.CP * ce_new * (float(surface_temperature) - forcing.tair)
        if abs(sensible) < 1.0e-8:
            obukhov_new = np.inf
        else:
            obukhov_new = -rho * energy.CP * forcing.tair * ustar**3 / (
                k * GRAVITY * sensible
            )
            obukhov_new = float(np.clip(obukhov_new, -1.0e6, 1.0e6))
        if np.isfinite(obukhov) and np.isfinite(obukhov_new):
            rel = abs(obukhov_new - obukhov) / max(1.0, abs(obukhov))
            ce = ce_new
            if rel < tolerance:
                break
        elif not np.isfinite(obukhov) and not np.isfinite(obukhov_new):
            ce = ce_new
            break
        ce = ce_new
        obukhov = obukhov_new
    return float(ce)


def exchange_velocity(
    forcing: Forcing, z0m: float, surface_temperature: float, option: int
) -> float:
    """按 ``OptSurfaceDrag`` 分发交换速度 ``ce`` (m/s)。"""
    if option == 1:
        return most_exchange_velocity(forcing, z0m, surface_temperature)
    if option == 2:
        wind = max(float(forcing.wind), 0.3)
        return float(energy.bulk_ch(wind, z0m) * wind)
    raise ValueError(f"OptSurfaceDrag={option} 无效；允许值为 1 或 2")


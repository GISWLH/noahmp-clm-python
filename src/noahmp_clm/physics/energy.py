"""地表能量平衡：湍流通量 + 对 Ts 的割线迭代求解。

Noah-MP 风格：Rnet(Ts) − H(Ts) − LE(Ts) − G(Ts) = 0，
用割线法迭代地表温度 Ts。F(Ts) 单调递减，迭代稳定。
"""
from __future__ import annotations

import numpy as np

KARMAN = 0.4
SIGMA = 5.670374419e-8
RD = 287.0
CP = 1004.0
LV = 2.5e6


def sat_vapor_pressure(t: float) -> float:
    """Tetens 饱和水汽压 (Pa)。"""
    tc = t - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def qsat(t: float, ps: float) -> float:
    """温度 t、气压 ps 下的饱和比湿 (kg/kg)。"""
    es = sat_vapor_pressure(t)
    return 0.622 * es / max(1.0, ps - 0.378 * es)


def air_density(f) -> float:
    return f.ps / (RD * f.tair * (1.0 + 0.61 * f.qair))


def bulk_ch(wind: float, z0m: float, zref: float = 10.0) -> float:
    """中性层结总体交换系数（无量纲，乘风速得交换速度）。"""
    z0h = z0m / 10.0
    return KARMAN ** 2 / (np.log(zref / z0m) * np.log(zref / z0h))


def secant_solve(func, x0: float, x1: float,
                 tol: float = 1e-3, maxit: int = 25) -> float:
    """割线法求根，Ts 钳制在 [200, 330] K。"""
    f0 = func(x0)
    for _ in range(maxit):
        f1 = func(x1)
        if abs(f1) < tol:
            return x1
        denom = f1 - f0
        if abs(denom) < 1e-12:
            break
        x2 = x1 - f1 * (x1 - x0) / denom
        x2 = float(np.clip(x2, 200.0, 330.0))
        x0, f0, x1 = x1, f1, x2
    return x1

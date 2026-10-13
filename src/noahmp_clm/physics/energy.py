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
    """割线法求根，Ts 钳制在 [200, 330] K。

    割线法在 MOST 稳定度修正与雪导热方案叠加时，两个初值可能落在同一
    侧且函数局部非单调。达到 ``maxit`` 仍未收敛时，固定网格扫描
    ``[200, 330]`` 找到第一个符号括区间，再用二分法收敛；这保持了纯
    函数接口，也避免把未闭合的 Ts 传回能量诊断。方程为 Noah-MP
    技术说明第 3 章的地表能量平衡残差。
    """
    f0 = func(x0)
    best_x, best_f = float(x0), float(f0)
    for _ in range(maxit):
        f1 = func(x1)
        if abs(f1) < abs(best_f):
            best_x, best_f = float(x1), float(f1)
        if abs(f1) < tol:
            return x1
        denom = f1 - f0
        if abs(denom) < 1e-12:
            break
        x2 = x1 - f1 * (x1 - x0) / denom
        x2 = float(np.clip(x2, 200.0, 330.0))
        x0, f0, x1 = x1, f1, x2

    # 固定扫描只在割线没有达到闭合容差时触发；对雪面 F(Ts) 的局部
    # 非单调性，比盲目继续外推更稳健。扫描次数固定，便于后续向量化。
    grid = np.linspace(200.0, 330.0, 131)
    values = np.asarray([float(func(float(x))) for x in grid])
    finite = np.isfinite(values)
    roots = np.flatnonzero(
        finite[:-1] & finite[1:] & (values[:-1] * values[1:] <= 0.0)
    )
    if roots.size:
        i = int(roots[0])
        left, right = float(grid[i]), float(grid[i + 1])
        f_left, f_right = float(values[i]), float(values[i + 1])
        for _ in range(40):
            mid = 0.5 * (left + right)
            f_mid = float(func(mid))
            if abs(f_mid) < tol:
                return mid
            if f_left * f_mid <= 0.0:
                right, f_right = mid, f_mid
            else:
                left, f_left = mid, f_mid
        return 0.5 * (left + right)
    return best_x

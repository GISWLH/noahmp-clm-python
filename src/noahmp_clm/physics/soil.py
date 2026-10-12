"""土壤水热求解器。

热：向后 Euler 隐式求解一维热传导（Thomas 追赶法），步长无条件稳定；
    上边界为地表热通量 G（Neumann），下边界零通量。
水：入渗 fill-and-spill（超渗→地表径流）+ Campbell/Clapp-Hornberger 型
    重力排水 + 根系分层吸水。冻土相变使用表观热容量进入隐式热方程。
"""
from __future__ import annotations

import numpy as np

from ..column import ColumnState
from .frozen_soil import apparent_heat_capacity, permeability_factor, update_soil_phase
from .runoff import route_runoff


def heat_capacity(smc: np.ndarray, p) -> np.ndarray:
    """体积热容量 J/(m3 K)：水 + 干土骨架（Noah 风格简化）。"""
    return smc * 4.18e6 + (1.0 - p.smcmax) * 2.0e6


def thermal_conductivity(smc: np.ndarray, p) -> np.ndarray:
    """Johansen 简化：干/饱和端元 + Kersten 数插值。"""
    sr = np.clip(smc / p.smcmax, 0.0, 1.0)
    kt_dry = 0.20
    kt_sat = 0.60 + 1.60 * p.quartz  # 石英含量加权，砂土 ~2.0，黏土 ~0.9
    ke = np.where(sr > 0.1, np.clip(np.log10(np.maximum(sr, 1e-6)) + 1.0, 0.0, 1.0), 0.0)
    return kt_dry + (kt_sat - kt_dry) * ke


def step_soil_heat(state: ColumnState, dt: float, g_flux: float) -> None:
    """隐式更新各层土壤温度。g_flux: 向下为正的地表热通量 (W/m2)。原地修改。"""
    n = len(state.dz)
    dz = state.dz
    cap = apparent_heat_capacity(
        state.smc, state.tsoil, state.soil, state.options.OptSoilSupercoolWater
    )
    kt = thermal_conductivity(state.smc, state.soil)    # W/(m K)

    # 界面导热系数（调和平均）与界面间距
    kt_if = np.empty(n - 1)
    dz_if = np.empty(n - 1)
    for i in range(n - 1):
        kt_if[i] = 2.0 / (1.0 / kt[i] + 1.0 / kt[i + 1])
        dz_if[i] = 0.5 * (dz[i] + dz[i + 1])

    a = np.zeros(n)  # 下对角
    b = np.zeros(n)  # 主对角
    c = np.zeros(n)  # 上对角
    d = cap * dz / dt * state.tsoil.copy()

    # 第 0 层：上边界通量 G
    k0 = kt_if[0] / dz_if[0]
    b[0] = cap[0] * dz[0] / dt + k0
    c[0] = -k0
    d[0] += g_flux
    # 中间层
    for i in range(1, n - 1):
        ku = kt_if[i - 1] / dz_if[i - 1]
        kd = kt_if[i] / dz_if[i]
        a[i] = -ku
        b[i] = cap[i] * dz[i] / dt + ku + kd
        c[i] = -kd
    # 底层：零通量
    ku = kt_if[n - 2] / dz_if[n - 2]
    a[n - 1] = -ku
    b[n - 1] = cap[n - 1] * dz[n - 1] / dt + ku

    state.tsoil[:] = _thomas(a, b, c, d)
    update_soil_phase(state)


def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray,
            d: np.ndarray) -> np.ndarray:
    """三对角追赶法。"""
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def add_water_topdown(smc: np.ndarray, dz: np.ndarray, smcmax: float,
                      vol: float) -> float:
    """从表层向下 fill-and-spill 加入 vol (m) 水量；返回无法容纳的地表径流 (m)。"""
    smc = smc  # 原地修改
    remaining = vol
    for i in range(len(dz)):
        room = (smcmax - smc[i]) * dz[i]
        add = min(remaining, max(0.0, room))
        smc[i] += add / dz[i]
        remaining -= add
        if remaining <= 1e-12:
            break
    return max(0.0, remaining)


def enforce_cap(smc: np.ndarray, dz: np.ndarray, smcmax: float) -> float:
    """把超孔隙度的水向下挤，底层挤出计为径流 (m)。"""
    runoff = 0.0
    for i in range(len(dz)):
        excess = (smc[i] - smcmax) * dz[i]
        if excess > 0:
            smc[i] = smcmax
            if i < len(dz) - 1:
                smc[i + 1] += excess / dz[i + 1]
            else:
                runoff += excess
    return runoff


def step_soil_water(state: ColumnState, dt: float, infil_m: float,
                    transp_m: float, esoil_m: float) -> dict:
    """土壤水分更新（m 为单位的水量）。返回分量诊断。

    顺序：入渗 → 蒸腾/土壤蒸发提取 → 重力排水。处处保证质量守恒。
    """
    p = state.soil
    n = len(state.dz)
    smc = state.smc
    dz = state.dz

    runoff_surf = add_water_topdown(smc, dz, p.smcmax, infil_m)
    update_soil_phase(state)

    # 蒸腾按根系权重从各层提取（不低于萎蔫点）
    transp_taken = 0.0
    for i in range(n):
        avail = max(0.0, smc[i] - p.smcwlt) * dz[i]
        want = transp_m * state.root_frac[i]
        take = min(want, avail)
        smc[i] -= take / dz[i]
        transp_taken += take
    # 土壤蒸发只从表层提取（可至 0）
    avail0 = smc[0] * dz[0]
    esoil_taken = min(esoil_m, avail0)
    smc[0] -= esoil_taken / dz[0]

    # Campbell/Clapp-Hornberger 型排水：只排田间持水量以上部分
    runoff_deep = 0.0
    for i in range(n):
        q = p.satdk * permeability_factor(
            state, state.options.OptSoilPermeabilityFrozen
        )[i] * (np.clip(smc[i] / p.smcmax, 0.0, 1.0)) ** (2.0 * p.bexp + 3.0)
        excess = max(0.0, smc[i] - p.smcref) * dz[i]
        drain = min(excess, q * dt)
        smc[i] -= drain / dz[i]
        if i < n - 1:
            smc[i + 1] += drain / dz[i + 1]
        else:
            runoff_deep += drain
    runoff_deep += enforce_cap(smc, dz, p.smcmax)
    update_soil_phase(state)

    runoff_surf, runoff_deep = route_runoff(
        runoff_surf + runoff_deep, state, state.options.OptRunoffSurface
    )

    return {"runoff_surf_m": runoff_surf, "runoff_deep_m": runoff_deep,
            "transp_m": transp_taken, "esoil_m": esoil_taken}

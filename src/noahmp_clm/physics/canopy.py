"""冠层：big-leaf，Jarvis/Ball--Berry 气孔阻抗 + Penman--Monteith 蒸腾。

Jarvis 型胁迫函数来自 Noah-MP ``ResistanceCanopyStomataJarvisMod.F90``
的方案结构。Ball--Berry 方案参考 ``ResistanceCanopyStomataBallBerryMod.F90``；
本阶段用经验光合--光响应曲线估计光合速率 ``A``，没有实现阶段 2 才会加入的
完整 Farquhar 生化光合模型。
"""
from __future__ import annotations

import numpy as np

from ..column import ColumnState, Forcing
from .energy import sat_vapor_pressure, air_density
from .stress import soil_water_stress

CP = 1004.0   # 干空气定压比热 J/(kg K)
LV = 2.5e6    # 蒸发潜热 J/kg


def jarvis_rs(state: ColumnState, f: Forcing) -> float:
    """简化 Jarvis 气孔阻抗 (s/m)。各胁迫因子 ∈ (0, 1]。"""
    es = sat_vapor_pressure(f.tair)
    ea = f.qair * f.ps / (0.622 + 0.378 * f.qair)
    vpd = max(0.0, es - ea)  # Pa

    f_light = float(np.clip(f.swdown / 500.0, 0.02, 1.0))
    f_temp = float(np.clip(1.0 - 0.0016 * (298.0 - f.tair) ** 2, 0.05, 1.0))
    f_vpd = float(np.clip(1.0 - vpd / 4000.0, 0.10, 1.0))

    f_smc = max(0.02, soil_water_stress(
        state, state.options.OptSoilWaterTranspiration
    ))

    rs = state.rsmin / (f_light * f_temp * f_vpd * f_smc)
    return float(np.clip(rs, state.rsmin, 5000.0))


def empirical_photosynthesis(state: ColumnState, f: Forcing) -> float:
    """经验光响应曲线给出的 ``A`` (umol CO2 m-2 s-1)。

    ``A = Amax (1-exp(-alpha*PAR)) fT fW`` 是 Ball--Berry 的阶段 1 替代项；
    它只提供合理的量级，不能代表 Farquhar 的羧化/电子传递限制。
    """
    par = max(0.0, float(f.swdown)) * 2.0  # 简化的 PAR (umol photons m-2 s-1)
    light = 1.0 - np.exp(-par / 450.0)
    temp = float(np.clip(1.0 - 0.0016 * (298.0 - f.tair) ** 2, 0.05, 1.0))
    water = max(0.02, soil_water_stress(
        state, state.options.OptSoilWaterTranspiration
    ))
    return float(25.0 * light * temp * water)


def ball_berry_rs(state: ColumnState, f: Forcing) -> float:
    """Ball--Berry 气孔阻抗 (s/m) 的阶段 1 简化实现。

    采用 ``g_s = g0 + g1*A*RH/Ca``，其中 ``A`` 来自
    :func:`empirical_photosynthesis`，参考 Collatz et al. (1991, 1992) 与
    Noah-MP ``ResistanceCanopyStomataBallBerryMod.F90``。400 ppm 的大气 CO2
    是本阶段固定的经验值；完整 Farquhar/CO2 耦合留到阶段 2。
    """
    es = sat_vapor_pressure(f.tair)
    ea = f.qair * f.ps / (0.622 + 0.378 * f.qair)
    rh = float(np.clip(ea / max(es, 1.0), 0.05, 1.0))
    assimilation = empirical_photosynthesis(state, f)
    ca = 400.0  # umol mol-1; stage-1 forcing has no explicit CO2 field
    g0 = 0.01   # mol CO2 m-2 s-1
    g1 = 9.0
    gs_mol = g0 + g1 * assimilation * rh / ca
    # Convert molar conductance to m/s using ideal-gas molar volume.
    gas_constant = 8.314462618
    gs_ms = gs_mol * gas_constant * f.tair / max(f.ps, 1.0)
    return float(np.clip(1.0 / max(gs_ms, 1.0e-6), state.rsmin, 5000.0))


def stomatal_resistance(state: ColumnState, f: Forcing, option: int) -> float:
    """按 ``OptStomataResistance`` 选择 Jarvis 或 Ball--Berry。"""
    schemes = {1: ball_berry_rs, 2: jarvis_rs}
    try:
        return schemes[option](state, f)
    except KeyError as exc:
        raise ValueError(
            f"OptStomataResistance={option} 无效；允许值为 1 或 2"
        ) from exc


def penman_monteith(rn: float, f: Forcing, ra: float, rs: float) -> float:
    """Penman-Monteith 潜热通量 (W/m2)，rn 为冠层可利用能量。"""
    es = sat_vapor_pressure(f.tair)
    ea = f.qair * f.ps / (0.622 + 0.378 * f.qair)
    # 饱和水汽压斜率 d(es)/dT (Pa/K)，Tetens 微分
    tc = f.tair - 273.15
    des_dt = es * 17.67 * 243.5 / (tc + 243.5) ** 2
    gamma = CP * f.ps / (0.622 * LV)  # 干湿表常数
    rho = air_density(f)
    num = des_dt * rn + rho * CP * max(0.0, es - ea) / max(ra, 1.0)
    den = des_dt + gamma * (1.0 + rs / max(ra, 1.0))
    return float(max(0.0, num / den))

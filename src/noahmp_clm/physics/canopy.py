"""冠层：big-leaf，简化 Jarvis 气孔阻抗 + Penman-Monteith 蒸腾。

v0.1 采用简化的 Jarvis 型胁迫函数（光/温/VPD/土壤水），Ball-Berry +
Farquhar 光合耦合是阶段 2（CLM 柱子集）的工作。
"""
from __future__ import annotations

import numpy as np

from ..column import ColumnState, Forcing
from .energy import sat_vapor_pressure, air_density

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

    smc_root = float(np.sum(state.smc * state.root_frac))
    p = state.soil
    f_smc = float(np.clip((smc_root - p.smcwlt) / max(1e-6, p.smcref - p.smcwlt),
                          0.02, 1.0))

    rs = state.rsmin / (f_light * f_temp * f_vpd * f_smc)
    return float(np.clip(rs, state.rsmin, 5000.0))


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

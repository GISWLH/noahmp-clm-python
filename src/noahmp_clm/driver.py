"""批量驱动：N 个完全独立的单柱各自推进。

step_column 是纯函数（状态进、状态出，无原地修改、无全局量），
因此 run_columns 的 Python 循环将来可整体替换为 jax.vmap 或 CUDA kernel，
物理代码不需要改。
"""
from __future__ import annotations

import numpy as np

from .column import ColumnState, Forcing
from .physics import energy as E
from .physics import soil as S
from .physics import snow as SN
from .physics.canopy import penman_monteith, stomatal_resistance
from .physics.precip import partition_precipitation
from .physics.surface_drag import exchange_velocity
from .physics.stress import soil_water_stress
from .physics.ground_evap import ground_resistance

CP = E.CP
LV = E.LV
RHO_W = 1000.0
SIGMA = E.SIGMA


def step_column(state: ColumnState, f: Forcing):
    """推进单个柱一个时间步。返回 (new_state, diag)。"""
    s = state.copy()
    dt = f.dt
    p = s.soil
    rho = E.air_density(f)
    wind_eff = max(f.wind, 0.3)
    ce = exchange_velocity(f, s.z0m, f.tair, s.options.OptSurfaceDrag)
    ra = 1.0 / max(ce, 1e-6)

    # ---- 1. 降水分配：由 state.options 选择纯函数方案 ----
    rain_m, snow_m = partition_precipitation(
        f.prcp, dt, f.tair, s.options.OptRainSnowPartition
    )

    # ---- 2. 冠层截留 ----
    canmax = 0.5e-3 * s.lai
    add = min(rain_m, max(0.0, canmax - s.canw))
    s.canw += add
    throughfall = rain_m - add
    SN.add_snowfall(s, snow_m, f.tair)
    SN.compact_snow(s, dt, s.options.OptSnowCompaction)

    # ---- 3. 冠层蒸腾（Penman-Monteith，可利用能量按 tair 估算） ----
    rs = stomatal_resistance(s, f, s.options.OptStomataResistance)
    rn_canopy = s.fveg * ((1.0 - s.albedo) * f.swdown
                          + s.emiss * (f.lwdown - SIGMA * f.tair ** 4))
    tr_le = penman_monteith(max(0.0, rn_canopy), f, ra, rs)
    snow_cover = min(1.0, s.swe / 0.02)
    tr_le *= (1.0 - snow_cover)
    avail_root = float(np.sum(np.maximum(0.0, s.smc - p.smcwlt) * s.dz))
    tr_m_pot = tr_le / LV * dt / RHO_W
    if tr_m_pot > avail_root:
        tr_le = tr_le * (avail_root / tr_m_pot) if avail_root > 0 else 0.0
    tr_m = tr_le / LV * dt / RHO_W

    # ---- 4. 地表能量平衡：割线迭代求 Ts ----
    snow_surface = s.swe > 1e-4
    if snow_surface:
        # 雪面分支（v0.1：忽略升华，雪下土壤取绝热）
        albedo, emiss = s.snow_albedo, 0.99
        snow_response = SN.snow_surface_response(s, dt)

        def F(ts):
            rnet = (1.0 - albedo) * f.swdown + emiss * (f.lwdown - SIGMA * ts ** 4)
            ce_ts = exchange_velocity(f, s.z0m, ts, s.options.OptSurfaceDrag)
            h = rho * CP * ce_ts * (ts - f.tair)
            g = SN.coupled_snow_flux(ts, snow_response)
            return rnet - h - g

        ts = E.secant_solve(F, f.tair, f.tair + 1.0)
        rnet = (1.0 - albedo) * f.swdown + emiss * (f.lwdown - SIGMA * ts ** 4)
        ce = exchange_velocity(f, s.z0m, ts, s.options.OptSurfaceDrag)
        h = rho * CP * ce * (ts - f.tair)
        g_snow = SN.coupled_snow_flux(ts, snow_response)
        melt = SN.apply_snow_heat(s, dt, g_snow)
        le_g = le_c = 0.0
        ecan_m = esoil_m = 0.0
        g_soil = 0.0
        le_total = 0.0
        residual = rnet - h - g_snow
    else:
        # 裸土/植被分支
        albedo, emiss = s.albedo, s.emiss
        beta = soil_water_stress(
            s, s.options.OptSoilWaterTranspiration, layer=0
        )
        rs_ground = ground_resistance(
            s, s.options.OptGroundResistanceEvap, snow_cover=False
        )
        kt1 = S.thermal_conductivity(s.smc[:1], p)[0]
        dz1 = s.dz[0]
        canw_cap_rate = s.canw * LV * RHO_W / dt          # 截留水允许的最大蒸发功率
        smc0_cap_rate = s.smc[0] * dz1 * LV * RHO_W / dt  # 表层水允许的最大蒸发功率

        def F(ts):
            rnet = (1.0 - albedo) * f.swdown + emiss * (f.lwdown - SIGMA * ts ** 4)
            ce_ts = exchange_velocity(f, s.z0m, ts, s.options.OptSurfaceDrag)
            h = rho * CP * ce_ts * (ts - f.tair)
            dqs = E.qsat(ts, f.ps) - f.qair
            ra_ts = 1.0 / max(ce_ts, 1.0e-6)
            le_g = min(
                beta * rho * LV * max(0.0, dqs) / (ra_ts + rs_ground),
                smc0_cap_rate,
            )
            le_c = min(rho * LV * ce_ts * dqs * min(1.0, s.canw / max(1e-9, 0.2 * canmax)),
                       canw_cap_rate)
            g = kt1 * (ts - s.tsoil[0]) / (0.5 * dz1)
            return rnet - h - le_g - le_c - tr_le - g

        ts = E.secant_solve(F, f.tair, f.tair + 1.0)
        rnet = (1.0 - albedo) * f.swdown + emiss * (f.lwdown - SIGMA * ts ** 4)
        ce = exchange_velocity(f, s.z0m, ts, s.options.OptSurfaceDrag)
        h = rho * CP * ce * (ts - f.tair)
        dqs = E.qsat(ts, f.ps) - f.qair
        ra = 1.0 / max(ce, 1.0e-6)
        le_g = min(
            beta * rho * LV * max(0.0, dqs) / (ra + rs_ground),
            smc0_cap_rate,
        )
        le_c = min(rho * LV * ce * dqs * min(1.0, s.canw / max(1e-9, 0.2 * canmax)),
                   canw_cap_rate)
        g_soil = kt1 * (ts - s.tsoil[0]) / (0.5 * dz1)
        esoil_m = le_g / LV * dt / RHO_W
        ecan_m = le_c / LV * dt / RHO_W
        s.canw = max(0.0, s.canw - ecan_m)
        melt = 0.0
        le_total = le_g + le_c + tr_le
        residual = rnet - h - le_total - g_soil

    s.ts = ts

    # ---- 5. 土壤水分 ----
    wdiag = S.step_soil_water(s, dt, throughfall + melt, tr_m, esoil_m)
    runoff_m = wdiag["runoff_surf_m"] + wdiag["runoff_deep_m"]

    # ---- 6. 土壤热量 ----
    S.step_soil_heat(s, dt, g_soil)

    # ---- 7. 雪反照率老化 ----
    SN.age_albedo(s, dt, snow_m, s.options.OptSnowAlbedo)

    et_m = wdiag["transp_m"] + wdiag["esoil_m"] + ecan_m
    diag = {
        "ts": ts, "rnet": rnet, "h": h, "le": le_total,
        "g": g_snow if snow_surface else g_soil,
        "et_mm": et_m * 1000.0, "prcp_mm": f.prcp * dt * 1000.0,
        "runoff_mm": runoff_m * 1000.0, "melt_mm": melt * 1000.0,
        "swe": s.swe, "canw": s.canw,
        "smc": s.smc.copy(), "tsoil": s.tsoil.copy(),
        "residual": residual,
    }
    return s, diag


def run_column(state: ColumnState, forcings: list[Forcing]):
    """跑完一个柱的全部强迫序列。返回 (diags, final_state)。"""
    s = state
    diags = []
    for f in forcings:
        s, d = step_column(s, f)
        diags.append(d)
    return diags, s


def run_columns(states: list[ColumnState], forcings_list: list[list[Forcing]]):
    """批量跑 N 个柱。循环体就是将来的 vmap/CUDA 轴。

    每个柱的 forcings 可以不同（不同站点/情景），也可以是同一序列。
    """
    return [run_column(s, fs) for s, fs in zip(states, forcings_list)]

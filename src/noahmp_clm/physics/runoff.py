"""地表/地下径流方案。

TOPMODEL 分支参考 Noah-MP ``RunoffSurfaceTopModelEquiMod.F90``：地形指数
控制饱和缺水，地下水位深度控制可达饱和程度。这里把候选径流在地表和
地下出口间重新分配，函数保持水量守恒；VIC/新安江等方案留作后续扩展。
"""
from __future__ import annotations

import numpy as np

from ..column import ColumnState


def topmodel_saturation_deficit(state: ColumnState, equilibrium: bool = False) -> float:
    """返回 TOPMODEL 饱和缺水量 (m)。"""
    p = state.soil
    deficit = float(np.sum(np.maximum(0.0, p.smcref - state.smc) * state.dz))
    terrain_factor = np.exp(-float(state.terrain_index))
    if equilibrium:
        # 平衡水位把地下水位深度换成等效可储水亏缺。
        water_table = max(0.0, float(state.groundwater_depth))
        deficit = max(deficit, water_table * p.smcmax * terrain_factor)
    return float(max(0.0, deficit * terrain_factor))


def topmodel_runoff(incoming_m: float, state: ColumnState, *, equilibrium: bool) -> tuple[float, float]:
    """TOPMODEL 产流，返回 ``(surface, subsurface)`` 并守恒输入水量。"""
    incoming = max(0.0, float(incoming_m))
    deficit = topmodel_saturation_deficit(state, equilibrium=equilibrium)
    # 地下水位越浅、地形指数越大，达到饱和缺水阈值越快。
    wetness = np.clip(
        np.exp(float(state.terrain_index))
        * np.exp(-max(0.0, float(state.groundwater_depth)) / 2.0),
        0.0,
        1.0,
    )
    surface = max(0.0, incoming - deficit) * wetness
    return float(surface), float(incoming - surface)


def bats_runoff(incoming_m: float, state: ColumnState) -> tuple[float, float]:
    """BATS 简化径流：湿润土壤增强地表产流。"""
    incoming = max(0.0, float(incoming_m))
    wet = float(np.mean(np.clip(state.smc / max(state.soil.smcmax, 1.0e-6), 0.0, 1.0)))
    surface = incoming * np.clip(wet**3, 0.0, 1.0)
    return float(surface), float(incoming - surface)


RUNOFF_SCHEMES = {
    1: lambda incoming, state: topmodel_runoff(incoming, state, equilibrium=False),
    2: lambda incoming, state: topmodel_runoff(incoming, state, equilibrium=True),
    3: lambda incoming, state: (0.0, max(0.0, float(incoming))),
    4: bats_runoff,
}


def route_runoff(incoming_m: float, state: ColumnState, option: int) -> tuple[float, float]:
    """按 ``OptRunoffSurface`` 分发并返回守恒的地表/地下径流。"""
    try:
        surface, subsurface = RUNOFF_SCHEMES[option](incoming_m, state)
    except KeyError as exc:
        raise NotImplementedError(
            f"OptRunoffSurface={option} 尚未实现；阶段 1 支持 1、2、3、4"
        ) from exc
    return float(max(0.0, surface)), float(max(0.0, subsurface))


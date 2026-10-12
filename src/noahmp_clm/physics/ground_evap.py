"""地面蒸发/升华的表面阻抗。

``ground_resistance`` 依据 Noah-MP ``ResistanceGroundEvaporationMod.F90``
重新实现 Sakaguchi & Zeng (2009) 的干土层厚度和水汽扩散率关系；Sellers
(1992) 两个方案也保留为可选分支。返回的阻抗是土壤内部扩散阻力 (s/m)，
与 ``surface_drag`` 给出的空气动力阻力串联。
"""
from __future__ import annotations

import numpy as np

from ..column import ColumnState


def resistance_sakaguchi_zeng(state: ColumnState) -> float:
    """Sakaguchi--Zeng (2009) 干表层阻抗 (s/m)。"""
    p = state.soil
    sat = max(float(p.smcmax), 1.0e-6)
    liquid = float(np.clip(state.smc[0], 0.0, sat))
    saturation = min(1.0, liquid / sat)
    # Noah-MP 参数化：dry-layer thickness / reduced vapour diffusivity。
    dry_thickness = state.dz[0] * (
        np.exp((1.0 - saturation) ** 2.0) - 1.0
    ) / (np.e - 1.0)
    diffusivity = 2.2e-5 * sat * sat * max(
        1.0e-6, 1.0 - p.smcwlt / sat
    ) ** (2.0 + 3.0 / max(p.bexp, 1.0e-6))
    return float(max(1.0e-3, dry_thickness / diffusivity))


def resistance_sellers(state: ColumnState, adjusted: bool = False) -> float:
    """Sellers (1992) 指数阻抗及其湿土调低版本 (s/m)。"""
    p = state.soil
    evap_factor = max(0.0, state.smc[0] / max(p.smcmax, 1.0e-6))
    coefficient = 6.0 if adjusted else 4.225
    return float(np.exp(8.25 - coefficient * evap_factor))


def ground_resistance(
    state: ColumnState, option: int, *, snow_cover: bool = False
) -> float:
    """按 ``OptGroundResistanceEvap`` 选择地面阻抗。

    option 4 在无雪时使用 Sakaguchi--Zeng，有雪时采用 Noah-MP 的表值
    ``1 s/m``；雪面潜热的完整升华过程由后续雪模块接入。
    """
    schemes = {
        1: lambda: resistance_sakaguchi_zeng(state),
        2: lambda: resistance_sellers(state, adjusted=False),
        3: lambda: resistance_sellers(state, adjusted=True),
        4: lambda: 1.0 if snow_cover else resistance_sakaguchi_zeng(state),
    }
    try:
        return float(max(1.0e-3, schemes[option]()))
    except KeyError as exc:
        raise ValueError(
            f"OptGroundResistanceEvap={option} 无效；允许值为 1、2、3、4"
        ) from exc


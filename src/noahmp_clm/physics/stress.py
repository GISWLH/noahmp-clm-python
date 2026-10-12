"""土壤水分对蒸腾和蒸发的胁迫因子。

方案入口对应 Noah-MP ``SoilWaterTranspirationMod.F90``：Noah 使用体积含水量
分段线性因子，CLM 与 SSiB 使用 Campbell 型土壤基质势。这里把基质势映射
到 [0, 1] 的可利用水分因子，保持每个方案为无副作用的标量纯函数。
"""
from __future__ import annotations

import numpy as np

from ..column import ColumnState


def _weighted(values: np.ndarray, state: ColumnState, layer: int | None) -> float:
    if layer is not None:
        return float(values[layer])
    return float(np.sum(values * state.root_frac))


def _noah_beta(state: ColumnState, layer: int | None) -> float:
    p = state.soil
    beta = np.clip(
        (state.smc - p.smcwlt) / max(1.0e-9, p.smcref - p.smcwlt), 0.0, 1.0
    )
    return _weighted(beta, state, layer)


def _campbell_potential(smc: np.ndarray, state: ColumnState) -> np.ndarray:
    """以 Campbell/Clapp--Hornberger 关系估计基质势 (kPa)。"""
    p = state.soil
    effective = np.clip(smc / max(p.smcmax, 1.0e-9), 1.0e-4, 1.0)
    # 饱和基质势约 -0.1 kPa；指数 b 控制干燥时的快速下降。
    return -0.1 * effective ** (-p.bexp)


def _matric_beta(state: ColumnState, layer: int | None, ssib: bool) -> float:
    p = state.soil
    psi = _campbell_potential(state.smc, state)
    psi_ref = float(_campbell_potential(np.array([p.smcref]), state)[0])
    psi_wilt = -1500.0  # kPa, Noah-MP/CLM 常用萎蔫势量级
    beta = np.clip((psi_wilt - psi) / (psi_wilt - psi_ref), 0.0, 1.0)
    if ssib:
        # SSiB 对干燥段的阻力增长更陡；保留湿端为 1 的归一化关系。
        beta = beta**2
    return _weighted(beta, state, layer)


def soil_water_stress(
    state: ColumnState, option: int, layer: int | None = None
) -> float:
    """按 ``OptSoilWaterTranspiration`` 返回 [0, 1] 水分胁迫因子。

    ``layer=None`` 返回根区加权值；传入层号则返回该层值，供地面蒸发
    使用。方案 1/2/3 分别为 Noah/CLM/SSiB。
    """
    if option == 1:
        return float(np.clip(_noah_beta(state, layer), 0.0, 1.0))
    if option == 2:
        return float(np.clip(_matric_beta(state, layer, ssib=False), 0.0, 1.0))
    if option == 3:
        return float(np.clip(_matric_beta(state, layer, ssib=True), 0.0, 1.0))
    raise ValueError(
        f"OptSoilWaterTranspiration={option} 无效；允许值为 1、2、3"
    )


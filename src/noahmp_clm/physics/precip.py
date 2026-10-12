"""降水相态分配方案。

实现参考 Noah-MP v5.0 ``AtmosForcingMod.F90`` 中的相态分配入口；代码按
Jordan (1991) 的分段关系和 Noah-MP 的 BATS/冰点阈值定义重新组织，没有
逐行翻译上游 Fortran。冠层截雪和湿球温度方案留给后续阶段。
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

FREEZING_POINT = 273.15  # K


def jordan91_snow_fraction(tair: float) -> float:
    """返回 Jordan (1991) 的冻结降水比例。

    0.5 K 以下全为雪，2.5 K 以上全为雨；中间段采用 Noah-MP
    ``AtmosForcingMod`` 的分段线性/平台关系。输入温度单位为 K。
    """
    delta = float(tair) - FREEZING_POINT
    if delta > 2.5:
        return 0.0
    if delta <= 0.5:
        return 1.0
    if delta <= 2.0:
        # 1 - (-54.632 + 0.2*T[K])，等价于 Noah-MP 的 Jordan 关系。
        return float(np.clip(1.0 - (-54.632 + 0.2 * float(tair)), 0.0, 1.0))
    return 0.6


def bats_snow_fraction(tair: float) -> float:
    """返回 BATS 相态方案的冻结比例（Dickinson et al., 1993）。

    当近地气温低于冰点加 2.2 K 时判为全雪，否则判为全雨；阈值与
    Noah-MP ``AtmosForcingMod.F90`` 的 ``OptRainSnowPartition=2`` 相同。
    """
    return float(float(tair) < FREEZING_POINT + 2.2)


def freezing_threshold_snow_fraction(tair: float) -> float:
    """返回冰点阈值方案的冻结比例（Noah-MP option 3）。"""
    return float(float(tair) < FREEZING_POINT)


SNOW_FRACTIONS: dict[int, Callable[[float], float]] = {
    1: jordan91_snow_fraction,
    2: bats_snow_fraction,
    3: freezing_threshold_snow_fraction,
}


def partition_precipitation(
    prcp_rate: float, dt: float, tair: float, option: int = 1
) -> tuple[float, float]:
    """按选项把降水率拆成雨/雪水量。

    参数
    ------
    prcp_rate:
        总降水率，单位 ``m s-1``。
    dt:
        时间步长，单位 ``s``。
    tair:
        参考高度气温，单位 ``K``。
    option:
        ``OptRainSnowPartition``；阶段 1 实现 1--3。

    返回
    ------
    ``(rain_m, snow_m)``，均为该时间步的水当量（m）。函数不修改输入。
    """
    if option not in SNOW_FRACTIONS:
        raise NotImplementedError(
            f"OptRainSnowPartition={option} 尚未实现；阶段 1 支持 1、2、3"
        )
    total = max(0.0, float(prcp_rate)) * max(0.0, float(dt))
    snow = total * SNOW_FRACTIONS[option](float(tair))
    return total - snow, snow


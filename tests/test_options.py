"""阶段 1 选项框架的契约测试。"""

import pytest

from noahmp_clm import ColumnState, PhysicsOptions


def test_hrldas_defaults_are_explicit() -> None:
    """默认配置与 HRLDAS 示例 namelist 的物理选项一致。"""
    options = PhysicsOptions()
    assert options.OptDynamicVeg == 4
    assert options.OptSurfaceDrag == 1
    assert options.OptStomataResistance == 1
    assert options.OptGroundResistanceEvap == 4
    assert options.OptSnowSoilTempTime == 3
    assert options.OptSnowCompaction == 2
    assert options.OptIrrigation == 0
    assert options.OptCropModel == 0
    assert ColumnState().options == options


def test_options_are_immutable_and_validated() -> None:
    """配置是可安全共享的不可变值对象，错误枚举在入口处失败。"""
    with pytest.raises(ValueError, match="OptSnowAlbedo"):
        PhysicsOptions(OptSnowAlbedo=9)
    with pytest.raises(ValueError, match="OptDynamicVeg"):
        PhysicsOptions(OptDynamicVeg=1)
    options = PhysicsOptions(OptRainSnowPartition=3)
    with pytest.raises(AttributeError):
        options.OptRainSnowPartition = 1  # type: ignore[misc]


def test_column_copy_keeps_options_value() -> None:
    """复制状态不产生跨柱的配置引用或隐式修改。"""
    options = PhysicsOptions(OptSnowAlbedo=2)
    original = ColumnState(options=options)
    copied = original.copy()
    assert copied.options is options
    assert copied.options.OptSnowAlbedo == 2


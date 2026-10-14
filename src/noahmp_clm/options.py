"""Noah-MP v5.0 物理方案开关。

字段名称与 NCAR/noahmp ``src/ConfigVarType.F90`` 的 ``Opt*`` 配置变量
一一对应。默认值取自 NCAR/HRLDAS ``run/examples/NLDAS/namelist.hrldas.NLDAS``
（同仓库 ``namelist.hrldas_example`` 的物理选项段），这样基线在没有显式
传入选项时仍采用 Noah-MP/HRLDAS 的示例配置。

这里仅保存无状态的标量配置；物理模块通过显式参数读取它们，不在柱之间
共享任何状态。方案含义依据 Noah-MP v5.0 ``ConfigVarType.F90`` 的注释，
方程来源会在各物理实现的 docstring 中注明。
"""
from __future__ import annotations

from dataclasses import dataclass


def _check_choice(name: str, value: int, choices: tuple[int, ...]) -> None:
    """检查一个 Noah-MP 枚举值并给出可读错误。"""
    if value not in choices:
        allowed = ", ".join(str(item) for item in choices)
        raise ValueError(f"{name}={value} 无效；允许值为 {{{allowed}}}")


@dataclass(frozen=True)
class PhysicsOptions:
    """Noah-MP v5.0 ``Opt*`` 物理配置的不可变 Python 表示。

    阶段 1 实现的选项会在 ``physics/`` 中按字典分发。阶段外的作物、灌溉、
    湿地、冰川和瓦片排水仍保留配置字段，默认关闭，便于后续扩展而不改变
    ``ColumnState`` 或 ``step_column`` 的调用签名。
    """

    # 与 HRLDAS namelist.hrldas 示例一致；OptDynamicVeg=4 表示使用输入 LAI
    # 和最大植被覆盖度，关闭动态植被。见 ConfigVarType.F90 注释。
    OptDynamicVeg: int = 4
    # 1 Jordan (1991), 2 BATS, 3 freezing-point threshold, 5 wet-bulb
    # (Wang et al., 2019). HRLDAS example: 1.
    OptRainSnowPartition: int = 1
    # 1 Noah soil moisture, 2 CLM matric potential, 3 SSiB matric potential.
    OptSoilWaterTranspiration: int = 1
    # 1 Sakaguchi & Zeng (2009), 2 Sellers (1992), 3 adjusted Sellers,
    # 4 option 1 without snow and table resistance with snow. HRLDAS: 4.
    OptGroundResistanceEvap: int = 4
    # 1 Monin--Obukhov similarity (MOST), 2 original Noah/Chen et al. (1997).
    OptSurfaceDrag: int = 1
    # 1 Ball--Berry, 2 Jarvis. HRLDAS example uses 1.
    OptStomataResistance: int = 1
    # 1 BATS, 2 CLASS, 3 SNICAR (stage 1 leaves SNICAR unimplemented).
    OptSnowAlbedo: int = 1
    # 1 modified two-stream, 2 grid-cell two-stream, 3 vegetated-fraction
    # two-stream. Kept for ConfigVarType parity; stage 1 uses the existing
    # big-leaf radiation approximation. HRLDAS: 3.
    OptCanopyRadiationTransfer: int = 3
    # 1 semi-implicit flux boundary, 2 fully implicit temperature boundary,
    # 3 as 1 with snow cover in skin temperature. HRLDAS: 3.
    OptSnowSoilTempTime: int = 3
    # 1 Stieglitz (Yen, 1965), 2 Anderson (1976), 3 constant,
    # 4 Verseghy (1991), 5 Douville (Yen, 1981). HRLDAS: 1.
    OptSnowThermConduct: int = 1
    # 1 zero heat flux, 2 prescribed deep (8 m) temperature. HRLDAS: 2.
    OptSoilTemperatureBottom: int = 2
    # 1 Niu & Yang (2006) no iteration, 2 Koren et al. (1999) iteration.
    OptSoilSupercoolWater: int = 1
    # 1 TOPMODEL with groundwater, 2 TOPMODEL equilibrium water table,
    # 3 original free drainage, 4 BATS, 5 MMF, 6 VIC, 7 Xinanjiang,
    # 8 Dynamic VIC. HRLDAS: 3.
    OptRunoffSurface: int = 3
    OptRunoffSubsurface: int = 3
    # 1 linear (more permeable), 2 nonlinear (less permeable). HRLDAS: 1.
    OptSoilPermeabilityFrozen: int = 1
    # Dynamic VIC infiltration: 1 Philip, 2 Green--Ampt, 3 Smith--Parlange.
    OptDynVicInfiltration: int = 1
    # 0 off, 1 simple, 2 Hooghoudt. Stage 1: off.
    OptTileDrainage: int = 0
    # 0 off, 1 on, 2 planting/harvesting trigger, 3 LAI trigger.
    OptIrrigation: int = 0
    # 0 geo-em fractions, 1 sprinkler, 2 drip, 3 surface flooding.
    OptIrrigationMethod: int = 0
    # 0 off, 1 Liu et al. (2016) crop model. Stage 1: off.
    OptCropModel: int = 0
    # 1 dominant texture, 2 depth-varying texture, 3 composition/pedotransfer,
    # 4 input properties. HRLDAS: 1.
    OptSoilProperty: int = 1
    # 1 Saxton & Rawls (2006) pedotransfer. HRLDAS: 1.
    OptPedotransfer: int = 1
    # 1 include glacier ice phase change, 2 original Noah treatment. Glacier
    # physics is outside stage 1; HRLDAS keeps the modern value 1.
    OptGlacierTreatment: int = 1
    # 1 Anderson (1976), 2 Abolafia-Rosenzweig et al. (2024). HRLDAS: 2.
    OptSnowCompaction: int = 2
    # 1 Niu & Yang (2007), 2 Abolafia-Rosenzweig et al. (2025). HRLDAS: 1.
    OptSnowCoverGround: int = 1
    # 0 off, 1 uniform wetland, 2 spatial wetland. Stage 1: off.
    OptWetlandModel: int = 0

    # SNICAR selectors are part of ConfigVarType.F90's Opt* interface. They
    # are retained for one-to-one configuration parity; OptSnowAlbedo=3 is
    # deliberately outside the stage 1 implementation. The values below are
    # harmless first-table entries and match the upstream selector conventions.
    OptSnicarSnowShape: int = 1
    OptSnicarRTSolver: int = 1
    OptSnicarBandNum: int = 1
    OptSnicarSolarSpec: int = 1
    OptSnicarSnwOptic: int = 1
    OptSnicarDustOptic: int = 1

    def __post_init__(self) -> None:
        """验证配置枚举，尽早报告拼写或越界的 namelist 选项。"""
        choices = {
            "OptDynamicVeg": (4,5),
            "OptRainSnowPartition": (1, 2, 3, 4, 5),
            "OptSoilWaterTranspiration": (1, 2, 3),
            "OptGroundResistanceEvap": (1, 2, 3, 4),
            "OptSurfaceDrag": (1, 2),
            "OptStomataResistance": (1, 2),
            "OptSnowAlbedo": (1, 2, 3),
            "OptCanopyRadiationTransfer": (1, 2, 3),
            "OptSnowSoilTempTime": (1, 2, 3),
            "OptSnowThermConduct": (1, 2, 3, 4, 5),
            "OptSoilTemperatureBottom": (1, 2),
            "OptSoilSupercoolWater": (1, 2),
            "OptRunoffSurface": tuple(range(1, 9)),
            "OptRunoffSubsurface": tuple(range(1, 9)),
            "OptSoilPermeabilityFrozen": (1, 2),
            "OptDynVicInfiltration": (1, 2, 3),
            "OptTileDrainage": (0, 1, 2),
            "OptIrrigation": (0, 1, 2, 3),
            "OptIrrigationMethod": (0, 1, 2, 3),
            "OptCropModel": (0, 1),
            "OptSoilProperty": (1, 2, 3, 4),
            "OptPedotransfer": (1,),
            "OptGlacierTreatment": (1, 2),
            "OptSnowCompaction": (1, 2),
            "OptSnowCoverGround": (1, 2),
            "OptWetlandModel": (0, 1, 2),
            "OptSnicarSnowShape": (1, 2, 3, 4),
            "OptSnicarRTSolver": (1, 2),
            "OptSnicarBandNum": (1, 2),
            "OptSnicarSolarSpec": (1, 2, 3, 4, 5, 6),
            "OptSnicarSnwOptic": (1, 2, 3),
            "OptSnicarDustOptic": (1, 2, 3),
        }
        for name, allowed in choices.items():
            _check_choice(name, getattr(self, name), allowed)


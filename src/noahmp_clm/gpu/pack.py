"""把单柱状态规约为 JAX/CUDA 可处理的定长 numpy 向量。"""

from __future__ import annotations

from dataclasses import fields
from typing import Iterable

import numpy as np

from ..column import ColumnState, Forcing, SoilParams
from ..options import PhysicsOptions


# 这些顺序是 GPU ABI：后续 JAX 与 CUDA 实现只能依赖这里的偏移表，不能依赖
# dataclass 的隐式内存布局。每一个条目是 (名称, 起始偏移, 元素个数)。
FLOAT_FIELDS: tuple[tuple[str, int, int], ...] = (
    ("tsoil", 0, 4),
    ("smc", 4, 4),
    ("soil_ice", 8, 4),
    ("root_frac", 12, 4),
    ("snow_swe_layers", 16, 3),
    ("snow_t_layers", 19, 3),
    ("snow_density_layers", 22, 3),
    ("ts", 25, 1),
    ("swe", 26, 1),
    ("snow_t", 27, 1),
    ("snow_albedo", 28, 1),
    ("snow_age_days", 29, 1),
    ("canw", 30, 1),
    ("lai", 31, 1),
    ("fveg", 32, 1),
    ("z0m", 33, 1),
    ("albedo", 34, 1),
    ("emiss", 35, 1),
    ("rsmin", 36, 1),
    ("terrain_index", 37, 1),
    ("groundwater_depth", 38, 1),
    ("soil.smcmax", 39, 1),
    ("soil.smcwlt", 40, 1),
    ("soil.smcref", 41, 1),
    ("soil.satdk", 42, 1),
    ("soil.bexp", 43, 1),
    ("soil.quartz", 44, 1),
    ("reserved0", 45, 1),
)

# 70 个槽位与阶段 3 状态规约约定一致。当前 ColumnState 的明确字段占用前
# 46 个槽位，剩余槽位为零填充的 ABI 保留区，给后续 CLM 状态扩展使用。
FLOAT_SIZE = 70
OPTION_FIELDS: tuple[str, ...] = tuple(item.name for item in fields(PhysicsOptions))
OPTION_COUNT = len(OPTION_FIELDS)
SNOW_COUNT_INDEX = OPTION_COUNT
INT_SIZE = OPTION_COUNT + 1
FORCING_SIZE = 8
DZ_DEFAULT = np.array([0.1, 0.3, 0.6, 1.0], dtype=np.float64)


def _as_fixed_layers(values: np.ndarray, name: str, fill: float) -> np.ndarray:
    """对应普通版: column.ColumnState.__post_init__。

    将 1--3 层积雪字段放入固定三层槽位。
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not 1 <= array.size <= 3:
        raise ValueError(f"{name} 必须是一维且包含 1--3 层")
    result = np.full(3, fill, dtype=np.float64)
    result[: array.size] = array
    return result


def _require_soil_shape(state: ColumnState) -> None:
    """对应普通版: column.ColumnState.__post_init__。

    检查固定四层土壤 ABI，避免把可变形状偷偷带入 JIT。
    """
    for name in ("tsoil", "smc", "soil_ice", "root_frac"):
        if np.asarray(getattr(state, name)).shape != (4,):
            raise ValueError(f"{name} 必须正好包含 4 个土壤层")
    if np.asarray(state.dz).shape != (4,):
        raise ValueError("GPU 状态要求 dz 正好包含 4 个土壤层")


def column_state_to_vec(state: ColumnState) -> tuple[np.ndarray, np.ndarray]:
    """对应普通版: ``column.ColumnState`` 的字段序列化。

    返回 ``(fvec, ivec)``。浮点向量使用 float64 保证阶段 A round-trip 的位级
    稳定性；阶段 B 进入 JAX 后再显式选择 float64 或 float32。所有输入数组只
    被读取，不会原地修改。
    """
    _require_soil_shape(state)
    fvec = np.zeros(FLOAT_SIZE, dtype=np.float64)

    fvec[0:4] = np.asarray(state.tsoil, dtype=np.float64)
    fvec[4:8] = np.asarray(state.smc, dtype=np.float64)
    fvec[8:12] = np.asarray(state.soil_ice, dtype=np.float64)
    fvec[12:16] = np.asarray(state.root_frac, dtype=np.float64)

    snow_swe = _as_fixed_layers(state.snow_swe_layers, "snow_swe_layers", 0.0)
    snow_t = _as_fixed_layers(state.snow_t_layers, "snow_t_layers", state.snow_t)
    snow_density = _as_fixed_layers(
        state.snow_density_layers, "snow_density_layers", 300.0
    )
    fvec[16:19] = snow_swe
    fvec[19:22] = snow_t
    fvec[22:25] = snow_density

    scalar_values = (
        state.ts,
        state.swe,
        state.snow_t,
        state.snow_albedo,
        state.snow_age_days,
        state.canw,
        state.lai,
        state.fveg,
        state.z0m,
        state.albedo,
        state.emiss,
        state.rsmin,
        state.terrain_index,
        state.groundwater_depth,
    )
    fvec[25:39] = np.asarray(scalar_values, dtype=np.float64)
    fvec[39:45] = np.asarray(
        (
            state.soil.smcmax,
            state.soil.smcwlt,
            state.soil.smcref,
            state.soil.satdk,
            state.soil.bexp,
            state.soil.quartz,
        ),
        dtype=np.float64,
    )
    # fvec[45:70] remains a deterministic zero-filled extension area.

    ivec = np.empty(INT_SIZE, dtype=np.int32)
    for index, name in enumerate(OPTION_FIELDS):
        ivec[index] = int(getattr(state.options, name))
    ivec[SNOW_COUNT_INDEX] = int(np.asarray(state.snow_swe_layers).size)
    return fvec, ivec


def vec_to_column_state(
    fvec: Iterable[float], ivec: Iterable[int], dz_const: Iterable[float]
) -> ColumnState:
    """对应普通版: ``column.ColumnState`` 的字段反序列化。

    ``dz_const`` 是编译期常量，不占状态向量；``SoilParams.name`` 也不进入
    GPU ABI，解包后使用 ``"packed"`` 作为仅用于显示的标签。
    """
    floats = np.asarray(fvec, dtype=np.float64)
    ints = np.asarray(ivec, dtype=np.int32)
    dz = np.asarray(dz_const, dtype=np.float64)
    if floats.shape != (FLOAT_SIZE,):
        raise ValueError(f"fvec 必须是形状 ({FLOAT_SIZE},)")
    if ints.shape != (INT_SIZE,):
        raise ValueError(f"ivec 必须是形状 ({INT_SIZE},)")
    if dz.shape != (4,):
        raise ValueError("dz_const 必须是形状 (4,)")

    snow_count = int(ints[SNOW_COUNT_INDEX])
    if not 1 <= snow_count <= 3:
        raise ValueError("ivec 中 snow_layer_count 必须为 1、2 或 3")
    option_values = {
        name: int(ints[index]) for index, name in enumerate(OPTION_FIELDS)
    }
    options = PhysicsOptions(**option_values)
    soil = SoilParams(
        name="packed",
        smcmax=float(floats[39]),
        smcwlt=float(floats[40]),
        smcref=float(floats[41]),
        satdk=float(floats[42]),
        bexp=float(floats[43]),
        quartz=float(floats[44]),
    )
    scalar_names = (
        "ts",
        "swe",
        "snow_t",
        "snow_albedo",
        "snow_age_days",
        "canw",
        "lai",
        "fveg",
        "z0m",
        "albedo",
        "emiss",
        "rsmin",
        "terrain_index",
        "groundwater_depth",
    )
    scalar_values = {
        name: float(floats[25 + index]) for index, name in enumerate(scalar_names)
    }
    return ColumnState(
        dz=dz.copy(),
        tsoil=floats[0:4].copy(),
        smc=floats[4:8].copy(),
        soil_ice=floats[8:12].copy(),
        root_frac=floats[12:16].copy(),
        snow_swe_layers=floats[16 : 16 + snow_count].copy(),
        snow_t_layers=floats[19 : 19 + snow_count].copy(),
        snow_density_layers=floats[22 : 22 + snow_count].copy(),
        soil=soil,
        options=options,
        **scalar_values,
    )


def forcing_to_vec(forcing: Forcing) -> np.ndarray:
    """对应普通版: ``column.Forcing`` 的八个强迫字段序列化。"""
    return np.asarray(
        (
            forcing.dt,
            forcing.tair,
            forcing.qair,
            forcing.wind,
            forcing.ps,
            forcing.swdown,
            forcing.lwdown,
            forcing.prcp,
        ),
        dtype=np.float64,
    )


def forcing_vec_to_forcing(vector: Iterable[float]) -> Forcing:
    """对应普通版: ``column.Forcing`` 的强迫向量反序列化。"""
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (FORCING_SIZE,):
        raise ValueError(f"forcing vector 必须是形状 ({FORCING_SIZE},)")
    return Forcing(*[float(value) for value in values])

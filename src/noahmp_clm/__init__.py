"""noahmp_clm: Noah-MP / CLM 单柱物理的纯 Python 基线（普通版本）。

设计约束：单柱状态全封装、柱间零耦合。批量轴（格点）永远独立，
是将来 jax.vmap / CUDA 向量化的轴。
"""

from .column import ColumnState, Forcing, SoilParams, default_soil_params
from .options import PhysicsOptions
from .clm_state import CLMColumnState
from .clm_glacier import GlacierColumnState
from .driver import run_column, run_columns, step_column
from .io import synthetic_forcing

__all__ = [
    "ColumnState", "CLMColumnState", "GlacierColumnState", "Forcing", "SoilParams", "default_soil_params", "PhysicsOptions",
    "run_column", "run_columns", "step_column", "synthetic_forcing",
]

__version__ = "0.2.0"

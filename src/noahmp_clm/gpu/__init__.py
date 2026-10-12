"""GPU 迁移所需的定长状态接口。

GPU 子包不在导入时加载 JAX 或 CuPy，因而基础 numpy 安装仍可使用普通版。
"""

from .pack import (
    FLOAT_FIELDS,
    FLOAT_SIZE,
    FORCING_SIZE,
    INT_SIZE,
    OPTION_FIELDS,
    column_state_to_vec,
    forcing_to_vec,
    forcing_vec_to_forcing,
    vec_to_column_state,
)

__all__ = [
    "FLOAT_FIELDS",
    "FLOAT_SIZE",
    "FORCING_SIZE",
    "INT_SIZE",
    "OPTION_FIELDS",
    "column_state_to_vec",
    "forcing_to_vec",
    "forcing_vec_to_forcing",
    "vec_to_column_state",
]

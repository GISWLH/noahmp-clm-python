"""CUDA complete BGC/crop/irrigation column retaining methane carbon."""
from ..clm_wetland_column import _step


def step_clm_wetland_columns_cuda(arrays,forcing,**kwargs):
    return _step(arrays,forcing,'cuda',**kwargs)

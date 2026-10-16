"""Native CUDA surface irrigation with actual physical/BGC state carry."""
from ..clm_irrigation_column import _step


def step_clm_irrigated_columns_cuda(arrays,forcing,*,inputs=None,checks=None,**kwargs):
    return _step(arrays,forcing,inputs,checks,'cuda',**kwargs)

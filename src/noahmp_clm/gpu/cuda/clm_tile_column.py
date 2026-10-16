"""Native CUDA tile outlet around the retained tagged base column state."""
from ..clm_tile_column import _step


def step_clm_tile_columns_cuda(batch,forcing,**kwargs):
    return _step(batch,forcing,'cuda',**kwargs)

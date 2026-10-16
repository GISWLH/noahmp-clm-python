"""Whole-column BGC orchestration over native CUDA physics and C/N kernels."""
from ..clm_bgc_column import _step
from functools import lru_cache


@lru_cache(maxsize=1)
def _division():
    import cupy as cp
    # Match the division used by JAX and the native plant kernels. CuPy's
    # default precise divide gives 1 where JAX gives 0.99999994, and also
    # changes the seasonal retention ratio. Both feed the physical column.
    return cp.ElementwiseKernel('float32 numerator,float32 denominator','float32 ratio',
        'ratio=numerator/denominator;','clm_bgc_coupling_ratio',
        options=('--fmad=false','--prec-div=false'))


def step_clm_bgc_columns_cuda(c,g,p,o,site,storage,roots,condensate,ps,flags,pp,plant,plant_p,soil,soil_p,bp,
                             forcing,*,environment=None,calendar=None,nitrogen_deposition_gn_m2=0.,oxygen_factor=1.,**kwargs):
    return _step((c,g,p,o,site,storage,roots,condensate,ps,flags,pp,plant,plant_p,soil,soil_p,bp),forcing,
        environment,calendar,nitrogen_deposition_gn_m2,oxygen_factor,'cuda',**kwargs)

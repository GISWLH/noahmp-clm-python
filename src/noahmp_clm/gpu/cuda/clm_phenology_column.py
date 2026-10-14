"""CUDA seasonal column, retaining prognostic GDD/leaf state on device."""
from ..clm_phenology_column import _step


def step_clm_phenology_columns_cuda(c,g,p,o,site,storage,roots,condensate,ps,flags,parameters,
                                     forcing,*,environment,calendar,**kwargs):
    return _step((c,g,p,o,site,storage,roots,condensate,ps,flags,parameters),forcing,
        environment,calendar,'cuda',**kwargs)

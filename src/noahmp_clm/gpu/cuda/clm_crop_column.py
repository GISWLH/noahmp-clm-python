"""Complete annual-crop column over genuine CUDA physical and C/N kernels."""
from ..clm_bgc_column import _step


def step_clm_crop_columns_cuda(c,g,p,o,site,storage,roots,condensate,ps,flags,pp,plant,plant_p,soil,soil_p,bp,
                              crop_state,crop_phase,crop_parameters,forcing,*,management=None,
                              nitrogen_deposition_gn_m2=0.,oxygen_factor=1.,**kwargs):
    return _step((c,g,p,o,site,storage,roots,condensate,ps,flags,pp,plant,plant_p,soil,soil_p,bp),forcing,
        None,None,nitrogen_deposition_gn_m2,oxygen_factor,'cuda',
        crop_arrays=(crop_state,crop_phase,crop_parameters),crop_management=management,**kwargs)

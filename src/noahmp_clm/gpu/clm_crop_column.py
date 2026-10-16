"""Nineteen-array crop column: complete BGC/M/H plus thermal phase and seed.

The native crop operator feeds turnover, canopy radiation/exchange, soil
decomposition and mineral-N competition. Only validation/dispatch runs on host.
"""
from dataclasses import astuple,replace
import numpy as np
from .clm_bgc_column import pack_clm_bgc_columns,next_bgc_arrays,_step
from ..physics.crop import CropManagement


def pack_clm_crop_columns(states,**kwargs):
    states=list(states);dtype=kwargs.get('dtype',np.float64)
    if any(s.options.OptCropModel!=1 or s.crop_state is None for s in states):
        raise ValueError('Every crop column requires OptCropModel=1 and explicit crop/BGC state')
    base=pack_clm_bgc_columns([replace(s,crop_state=None,crop_parameters=None,
        options=replace(s.options,OptCropModel=0)) for s in states],**kwargs)
    cs=np.asarray([[s.crop_state.air_degree_days,s.crop_state.soil_degree_days,s.crop_state.age_days,
        s.crop_state.seed_carbon,s.crop_state.seed_nitrogen] for s in states],dtype).reshape(-1,5)
    phase=np.asarray([[s.crop_state.phase] for s in states],np.int32).reshape(-1,1)
    parameters=np.asarray([astuple(s.crop_parameters) for s in states],dtype).reshape(-1,9)
    return (*base,cs,phase,parameters)


def pack_crop_management(events):
    events=[CropManagement() if e is None else e for e in events]
    if any(not isinstance(e,CropManagement) for e in events):raise ValueError('Expected CropManagement or None')
    return np.asarray([[e.planting,e.force_harvest] for e in events],np.int32).reshape(-1,2)


def step_clm_crop_columns_jax(c,g,p,o,site,storage,roots,condensate,ps,flags,pp,plant,plant_p,soil,soil_p,bp,
                             crop_state,crop_phase,crop_parameters,forcing,*,management=None,
                             nitrogen_deposition_gn_m2=0.,oxygen_factor=1.,**kwargs):
    return _step((c,g,p,o,site,storage,roots,condensate,ps,flags,pp,plant,plant_p,soil,soil_p,bp),forcing,
        None,None,nitrogen_deposition_gn_m2,oxygen_factor,'jax',
        crop_arrays=(crop_state,crop_phase,crop_parameters),crop_management=management,**kwargs)


def next_crop_arrays(previous,result):
    """Carry physical/BGC/seed/thermal stores without importing host tendencies."""
    if len(previous)!=19:raise ValueError('Crop columns require nineteen state arrays')
    return (*next_bgc_arrays(previous[:16],result),result['crop_state'],result['crop_phase'],previous[18])

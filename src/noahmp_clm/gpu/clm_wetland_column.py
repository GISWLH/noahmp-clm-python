"""Conserved CH4-C state appended to native BGC/crop/irrigated columns."""
from dataclasses import astuple,replace
import numpy as np


def pack_clm_wetland_columns(states,**kwargs):
    states=list(states);dtype=kwargs.get('dtype',np.float64)
    if not states or any(s.wetland_state is None or s.options.OptWetlandModel!=1 for s in states):
        raise ValueError('Every wetland column requires option 1 and explicit methane/BGC stores')
    kinds=[(s.crop_state is not None,s.irrigation_state is not None) for s in states]
    if len(set(kinds))!=1:raise ValueError('A wetland batch must use one base state ABI')
    plain=[replace(s,wetland_state=None,wetland_parameters=None,options=replace(s.options,OptWetlandModel=0)) for s in states]
    if kinds[0][1]:from .clm_irrigation_column import pack_clm_irrigated_columns as pack
    elif kinds[0][0]:from .clm_crop_column import pack_clm_crop_columns as pack
    else:from .clm_bgc_column import pack_clm_bgc_columns as pack
    return (*pack(plain,**kwargs),np.asarray([astuple(s.wetland_state) for s in states],dtype),
        np.asarray([astuple(s.wetland_parameters) for s in states],dtype))


def _step(arrays,forcing,backend,**kwargs):
    count=len(arrays)-2
    if count not in (16,18,19,21):raise ValueError('Wetland columns append two arrays to a BGC/crop/irrigated ABI')
    kwargs.update(wetland_state=arrays[-2],wetland_parameters=arrays[-1])
    if count in (18,21):
        if backend=='jax':from .clm_irrigation_column import step_clm_irrigated_columns_jax as run
        else:from .cuda.clm_irrigation_column import step_clm_irrigated_columns_cuda as run
        return run(arrays[:-2],forcing,**kwargs)
    if backend=='jax':
        if count==19:from .clm_crop_column import step_clm_crop_columns_jax as run
        else:from .clm_bgc_column import step_clm_bgc_columns_jax as run
    else:
        if count==19:from .cuda.clm_crop_column import step_clm_crop_columns_cuda as run
        else:from .cuda.clm_bgc_column import step_clm_bgc_columns_cuda as run
    return run(*arrays[:-2],forcing,**kwargs)


def step_clm_wetland_columns_jax(arrays,forcing,**kwargs):
    return _step(arrays,forcing,'jax',**kwargs)


def next_wetland_arrays(previous,result):
    count=len(previous)-2
    if count in (18,21):from .clm_irrigation_column import next_irrigated_arrays as carry
    elif count==19:from .clm_crop_column import next_crop_arrays as carry
    elif count==16:from .clm_bgc_column import next_bgc_arrays as carry
    else:raise ValueError('Unsupported wetland state ABI')
    return (*carry(previous[:-2],result),result['wetland_state'],previous[-1])

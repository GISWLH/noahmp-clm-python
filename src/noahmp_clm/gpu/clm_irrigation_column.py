"""Conservative surface irrigation around native layered/seasonal/BGC/crop columns.

Append request[2] and irrigation parameters[5] to the corresponding base ABI
(8, 11, 16 or 19 arrays). The physical stage diagnoses irrigation after the
actual seasonal/crop/turnover LAI update; soil water/heat never come from host.
"""
from dataclasses import astuple,replace
import numpy as np
from ..physics.irrigation import IrrigationInput


def pack_clm_irrigated_columns(states,**kwargs):
    states=list(states);dtype=kwargs.get('dtype',np.float64)
    if not states or any(s.irrigation_state is None or s.options.OptIrrigation!=1 for s in states):
        raise ValueError('Every irrigated column requires option 1 and explicit request state')
    kinds=[19 if s.crop_state is not None else 16 if s.bgc_state is not None else 11 if s.options.OptDynamicVeg==5 else 8 for s in states]
    if len(set(kinds))!=1:raise ValueError('An irrigated batch must use one base state ABI')
    stripped=[replace(s,irrigation_state=None,irrigation_parameters=None,
        options=replace(s.options,OptIrrigation=0,OptIrrigationMethod=0)) for s in states]
    if kinds[0]==19:from .clm_crop_column import pack_clm_crop_columns as pack
    elif kinds[0]==16:from .clm_bgc_column import pack_clm_bgc_columns as pack
    elif kinds[0]==11:from .clm_phenology_column import pack_clm_phenology_columns as pack
    else:from .clm_layered_column import pack_clm_layered_columns as pack
    base=pack(stripped,**kwargs)
    return (*base,np.asarray([astuple(s.irrigation_state) for s in states],dtype),
        np.asarray([astuple(s.irrigation_parameters) for s in states],dtype))


def pack_irrigation_inputs(inputs,*,dtype=np.float64):
    values=[IrrigationInput() if v is None else v for v in inputs]
    if any(not isinstance(v,IrrigationInput) for v in values):raise ValueError('Expected IrrigationInput or None')
    return (np.asarray([[v.supply_kgm2,v.supply_temperature_k] for v in values],dtype).reshape(-1,2),
        np.asarray([[v.check] for v in values],np.int32).reshape(-1,1))


def _step(arrays,forcing,inputs,checks,backend,**kwargs):
    count=len(arrays)-2
    if count not in (8,11,16,19):raise ValueError('Irrigated columns append two arrays to a supported base ABI')
    if backend=='jax':
        if count==19:from .clm_crop_column import step_clm_crop_columns_jax as run
        elif count==16:from .clm_bgc_column import step_clm_bgc_columns_jax as run
        elif count==11:from .clm_phenology_column import step_clm_phenology_columns_jax as run
        else:from .clm_layered_column import step_clm_layered_columns_jax as run
    else:
        if count==19:from .cuda.clm_crop_column import step_clm_crop_columns_cuda as run
        elif count==16:from .cuda.clm_bgc_column import step_clm_bgc_columns_cuda as run
        elif count==11:from .cuda.clm_phenology_column import step_clm_phenology_columns_cuda as run
        else:from .cuda.clm_layered_column import step_clm_layered_columns_cuda as run
    return run(*arrays[:-2],forcing,irrigation_state=arrays[-2],irrigation_parameters=arrays[-1],
        irrigation_input=inputs,irrigation_check=checks,**kwargs)


def step_clm_irrigated_columns_jax(arrays,forcing,*,inputs=None,checks=None,**kwargs):
    return _step(arrays,forcing,inputs,checks,'jax',**kwargs)


def next_irrigated_arrays(previous,result):
    count=len(previous)-2;base=previous[:-2]
    if count==19:
        from .clm_crop_column import next_crop_arrays
        after=next_crop_arrays(base,result)
    elif count==16:
        from .clm_bgc_column import next_bgc_arrays
        after=next_bgc_arrays(base,result)
    elif count==11:
        from .clm_phenology_column import next_phenology_arrays
        after=next_phenology_arrays(base,result)
    elif count==8:
        after=(result['snow_cells'],result['soil_cells'],base[2],result['options'],base[4],result['storage'],base[6],result['condensate'])
    else:raise ValueError('Unsupported irrigated state ABI')
    return (*after,result['irrigation_state'],previous[-1])

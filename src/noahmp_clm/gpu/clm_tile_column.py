"""Explicit tile configuration around an identified native base column ABI.

Several composed base ABIs have the same array count. Retain a base-kind tag
instead of inferring composition from length and risking loss of C/N stores.
Only that dispatch tag is on the host; all prognostic tendencies are native.
"""
from dataclasses import dataclass,replace,astuple
from importlib import import_module
import numpy as np

_ENTRIES=dict(layered=('clm_layered_column','layered'),seasonal=('clm_phenology_column','phenology'),
    bgc=('clm_bgc_column','bgc'),crop=('clm_crop_column','crop'),irrigated=('clm_irrigation_column','irrigated'),
    wetland=('clm_wetland_column','wetland'))


@dataclass(frozen=True)
class TileColumnBatch:
    base_kind: str
    arrays: tuple
    parameters: object

    def __post_init__(self):
        if self.base_kind not in _ENTRIES:raise ValueError('Unknown tile base column kind')


def pack_clm_tile_columns(states,**kwargs):
    states=list(states);dtype=kwargs.get('dtype',np.float64)
    if not states or any(s.options.OptTileDrainage!=1 or s.tile_drainage_parameters is None for s in states):
        raise ValueError('Every tile column requires option 1 and explicit parameters')
    kinds=['wetland' if s.wetland_state is not None else 'irrigated' if s.irrigation_state is not None
        else 'crop' if s.crop_state is not None else 'bgc' if s.bgc_state is not None
        else 'seasonal' if s.options.OptDynamicVeg==5 else 'layered' for s in states]
    if len(set(kinds))!=1:raise ValueError('A tile batch must use one base column kind')
    module,name=_ENTRIES[kinds[0]];pack=getattr(import_module('.'+module,__package__),'pack_clm_'+name+'_columns')
    plain=[replace(s,tile_drainage_parameters=None,options=replace(s.options,OptTileDrainage=0)) for s in states]
    return TileColumnBatch(kinds[0],pack(plain,**kwargs),np.asarray([astuple(s.tile_drainage_parameters) for s in states],dtype))


def _step(batch,forcing,backend,**kwargs):
    if not isinstance(batch,TileColumnBatch):raise ValueError('Expected a tagged TileColumnBatch')
    module,name=_ENTRIES[batch.base_kind];prefix='.' if backend=='jax' else '.cuda.'
    run=getattr(import_module(prefix+module,__package__),'step_clm_'+name+'_columns_'+backend)
    if batch.base_kind in ('irrigated','wetland'):return run(batch.arrays,forcing,tile_parameters=batch.parameters,**kwargs)
    return run(*batch.arrays,forcing,tile_parameters=batch.parameters,**kwargs)


def step_clm_tile_columns_jax(batch,forcing,**kwargs):
    return _step(batch,forcing,'jax',**kwargs)


def next_tile_batch(previous,result):
    kind=previous.base_kind;a=previous.arrays
    if kind=='layered':after=(result['snow_cells'],result['soil_cells'],a[2],result['options'],a[4],result['storage'],a[6],result['condensate'])
    else:
        module,name=_ENTRIES[kind]
        after=getattr(import_module('.'+module,__package__),'next_'+name+'_arrays')(a,result)
    return replace(previous,arrays=after,parameters=result['tile_parameters'])

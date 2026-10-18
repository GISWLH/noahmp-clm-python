"""Explicit structure extension around the existing native column ABIs.

Three appended arrays retain total SAI, top/bottom and scheme/form flags.
The base arrays continue to carry total LAI. All burial/update tendencies are
computed on device; stripping geometry is permitted only while packing this
extension, where it is retained separately rather than lost.
"""
from dataclasses import replace
from importlib import import_module
import numpy as np

_KINDS={'layered':('clm_layered_column','pack_clm_layered_columns','step_clm_layered_columns',''),
        'seasonal':('clm_phenology_column','pack_clm_phenology_columns','step_clm_phenology_columns','next_phenology_arrays'),
        'bgc':('clm_bgc_column','pack_clm_bgc_columns','step_clm_bgc_columns','next_bgc_arrays'),
        'crop':('clm_crop_column','pack_clm_crop_columns','step_clm_crop_columns','next_crop_arrays'),
        'irrigated':('clm_irrigation_column','pack_clm_irrigated_columns','step_clm_irrigated_columns','next_irrigated_arrays'),
        'wetland':('clm_wetland_column','pack_clm_wetland_columns','step_clm_wetland_columns','next_wetland_arrays')}


def _kind(kind):
    if kind not in _KINDS:raise ValueError('Explicit supported structure column kind required')
    return _KINDS[kind]


def pack_clm_structure_columns(states,*,kind='layered',**kwargs):
    states=list(states);dtype=kwargs.get('dtype',np.float64);module,pack,_,_=_kind(kind)
    if not states or any(s.canopy_structure_parameters is None for s in states):
        raise ValueError('Every structure column requires explicit geometry and total SAI')
    plain=[replace(s,sai=0.,canopy_structure_parameters=None) for s in states]
    base=getattr(import_module('.'+module,__package__),pack)(plain,**kwargs)
    state=np.asarray([[s.sai] for s in states],dtype)
    parameters=np.asarray([[s.canopy_structure_parameters.canopy_top_m,s.canopy_structure_parameters.canopy_bottom_m] for s in states],dtype)
    flags=np.asarray([[int(s.canopy_structure_parameters.scheme),s.canopy_structure_parameters.short_vegetation] for s in states],np.int32)
    return (*base,state,parameters,flags)


def resolve_column_structure(structure,arrays,forcing,backend,*,previous_lai=None,management=None,prescribed_sai=None):
    if structure is None:
        if any(v is not None for v in (previous_lai,management,prescribed_sai)):
            raise ValueError('Structure inputs require retained geometry state')
        return None
    if len(structure)!=3:raise ValueError('Structure extension requires three arrays')
    if backend=='jax':
        import jax.numpy as xp
        from .vegetation_structure import run_structure_jax as run
    elif backend=='cuda':
        import cupy as xp
        from .cuda.vegetation_structure import run_structure_cuda as run
    else:raise ValueError('Invalid structure backend')
    c,_,_,options,site=arrays[:5];n=len(c);dtype=c.dtype
    ss,pp,oo=map(xp.asarray,structure)
    if ss.shape!=(n,1) or pp.shape!=(n,2) or oo.shape!=(n,2) or oo.dtype.kind not in 'iu':
        raise ValueError('Structure extension shapes require n by1/2/2 with integer flags')
    if ss.dtype!=dtype or pp.dtype!=dtype:raise ValueError('Structure state precision mismatch')
    before=site[:,0] if previous_lai is None else xp.asarray(previous_lai,dtype)
    if before.shape!=(n,):raise ValueError('Previous structure LAI must have one value per column')
    zero=xp.zeros(n,dtype);depth=zero
    # Match the reference ordered reduction and use only the old active snow.
    for k in range(5):depth=depth+xp.where(k<options[:,0],c[:,k,2],0.)
    imposed=zero if prescribed_sai is None else xp.broadcast_to(xp.asarray(prescribed_sai,dtype),(n,))
    controls=xp.zeros((n,3),xp.int32) if management is None else xp.asarray(management)
    if controls.shape!=(n,3) or controls.dtype.kind not in 'iu':raise ValueError('Structure management requires integer n by3 flags')
    if bool(xp.any((controls<0)|(controls>1))):raise ValueError('Structure management flags must be boolean')
    # Harvest/planting belong only to dynamic annual geometry; prescribed
    # geometry remains an explicit alternative even in a crop column.
    controls=xp.where((oo[:,0]==3)[:,None],controls,0)
    flags=xp.concatenate((oo,xp.full((n,1),int(prescribed_sai is not None),xp.int32),controls),axis=1)
    inputs=xp.stack((ss[:,0],before,site[:,0],forcing[:,0],depth,imposed),axis=1)
    return run(inputs,pp,flags)


def attach_column_structure(result,structure,geometry):
    if geometry is not None:
        result.update(structure_state=geometry['sai_after'][:,None],structure_parameters=structure[1],
            structure_flags=structure[2],canopy_geometry=geometry,vegetation_structure=geometry)
    return result


def _step(arrays,forcing,backend,kind,**kwargs):
    module,_,name,_=_kind(kind)
    if len(arrays)<11:raise ValueError('Structure column is missing appended state arrays')
    structure=arrays[-3:];base=arrays[:-3]
    package=__package__ if backend=='jax' else __package__+'.cuda'
    run=getattr(import_module('.'+module,package),name+'_'+backend)
    if 'canopy_structure' in kwargs:raise ValueError('Use the retained structure arrays')
    if kind in ('wetland','irrigated'):return run(base,forcing,canopy_structure=structure,**kwargs)
    return run(*base,forcing,canopy_structure=structure,**kwargs)


def step_clm_structure_columns_jax(arrays,forcing,*,kind='layered',**kwargs):
    return _step(arrays,forcing,'jax',kind,**kwargs)


def step_clm_structure_columns_cuda(arrays,forcing,*,kind='layered',**kwargs):
    return _step(arrays,forcing,'cuda',kind,**kwargs)


def next_structure_arrays(previous,result,*,kind='layered'):
    module,_,_,carry=_kind(kind);base=previous[:-3]
    if kind=='layered':
        if len(base)!=8:raise ValueError('Layered structure requires eight base arrays')
        after=(result['snow_cells'],result['soil_cells'],base[2],result['options'],base[4],result['storage'],base[6],result['condensate'])
    else:after=getattr(import_module('.'+module,__package__),carry)(base,result)
    return (*after,result['structure_state'],previous[-2],previous[-1])

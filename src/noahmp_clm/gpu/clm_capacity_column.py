"""Retained temperature-history extension around native CLM column families.

Four appended arrays hold sample values, integer count/cursor, interval/cold
temperature, and PFT traits. Existing physical/C-N/structure prefixes remain
unchanged. The BGC forms therefore contain20 or23 arrays, respectively.
"""
from dataclasses import astuple,replace
from importlib import import_module
import numpy as np
from ..physics.canopy_capacity import CanopyCapacityEnvironment
from .clm_structure_column import _kind


def pack_clm_capacity_columns(states,*,kind='bgc',structured=False,**kwargs):
    states=list(states);dtype=kwargs.get('dtype',np.float64)
    if not states or any(s.canopy_capacity_state is None or s.canopy_capacity_parameters is None for s in states):
        raise ValueError('Every capacity column requires explicit traits and complete temperature history')
    widths={len(s.canopy_capacity_state.values) for s in states}
    if len(widths)!=1:raise ValueError('A capacity batch requires matching temperature-history intervals')
    plain=[replace(s,canopy_capacity_state=None,canopy_capacity_parameters=None) for s in states]
    if structured:
        from .clm_structure_column import pack_clm_structure_columns
        base=pack_clm_structure_columns(plain,kind=kind,**kwargs)
    else:
        module,pack,_,_=_kind(kind)
        base=getattr(import_module('.'+module,__package__),pack)(plain,**kwargs)
    values=np.asarray([s.canopy_capacity_state.values for s in states],dtype)
    controls=np.asarray([[s.canopy_capacity_state.count,s.canopy_capacity_state.cursor] for s in states],np.int32)
    configuration=np.asarray([[s.canopy_capacity_state.time_step_seconds,s.canopy_capacity_state.initial_temperature_k] for s in states],dtype)
    traits=np.asarray([astuple(s.canopy_capacity_parameters) for s in states],dtype)
    return (*base,values,controls,configuration,traits)


def pack_capacity_environment(environments,*,dtype=np.float64):
    values=list(environments)
    if any(not isinstance(v,CanopyCapacityEnvironment) for v in values):
        raise ValueError('Every capacity column requires current and maximum photoperiod')
    return np.asarray([astuple(v) for v in values],dtype).reshape(-1,2)


def resolve_column_capacity(capacity,environment,leaf,forcing,parameters,backend):
    if capacity is None:
        if environment is not None:raise ValueError('Capacity photoperiod requires retained native history')
        return parameters,None
    if len(capacity)!=4:raise ValueError('Capacity extension requires four retained arrays')
    if backend=='jax':
        import jax.numpy as xp
        from .canopy_capacity import run_capacity_jax as profile,run_temperature_memory_jax as memory
    elif backend=='cuda':
        import cupy as xp
        from .cuda.canopy_capacity import run_capacity_cuda as profile,run_temperature_memory_cuda as memory
    else:raise ValueError('Invalid capacity backend')
    from .canopy_capacity import validate_temperature_memory
    from .photosynthesis import prepare_canopy_leaf_parameters
    n=len(leaf);dtype=leaf.dtype
    values=xp.asarray(capacity[0],dtype);controls=xp.asarray(capacity[1])
    configuration=xp.asarray(capacity[2],dtype);traits=xp.asarray(capacity[3],dtype)
    if configuration.shape!=(n,2) or traits.shape!=(n,4):raise ValueError('Invalid retained capacity configuration')
    if environment is None:raise ValueError('Capacity requires explicit current/maximum photoperiod')
    environment=xp.asarray(environment,dtype)
    if environment.shape!=(n,2):raise ValueError('Capacity photoperiod requires n by2 values')
    if bool(xp.any(forcing[:,0]!=configuration[:,0])):
        raise ValueError('Forcing interval disagrees with retained temperature history')
    incoming=xp.stack((configuration[:,0],configuration[:,1],forcing[:,1]),axis=1)
    validate_temperature_memory(values,controls,incoming,xp)
    updated=memory(values,controls.astype(xp.int32),incoming)
    if getattr(parameters,'shape',None)==(n,10):
        # Here each row is a column's gas-exchange template. The retained
        # traits/history still own all layer capacities and growth temperature.
        template=xp.asarray(parameters,dtype)
    else:
        template,profiled=prepare_canopy_leaf_parameters(parameters,xp,n,dtype)
        if profiled:raise ValueError('Retained capacity derives its own layer coefficients; supply gas-exchange templates')
        template=xp.broadcast_to(template,(n,10))
    environment=xp.concatenate((environment,updated['means'][:,:1]),axis=1)
    diagnostic=profile(leaf,environment,traits,template)
    diagnostic.update(temperature_count_before=controls[:,0].astype(xp.int32),temperature_count_after=updated['controls'][:,0],
                      temperature_cursor_after=updated['controls'][:,1],next_mean_temperature_k=updated['means'][:,1])
    state=dict(values=updated['values'],controls=updated['controls'],configuration=configuration,traits=traits,diagnostic=diagnostic)
    return diagnostic['leaf_parameters'],state


def attach_column_capacity(result,state):
    if state is not None:
        result.update(capacity_temperature_values=state['values'],capacity_temperature_controls=state['controls'],
                      capacity_configuration=state['configuration'],capacity_traits=state['traits'],canopy_capacity=state['diagnostic'])
    return result


def _step(arrays,forcing,backend,kind,structured,environment,kwargs):
    if len(arrays)<4:raise ValueError('Missing native capacity state')
    if 'canopy_capacity' in kwargs or 'capacity_environment' in kwargs:
        raise ValueError('Use the retained capacity arrays and explicit environment')
    base,capacity=arrays[:-4],arrays[-4:]
    if structured:
        from .clm_structure_column import step_clm_structure_columns_jax,step_clm_structure_columns_cuda
        run=step_clm_structure_columns_jax if backend=='jax' else step_clm_structure_columns_cuda
        return run(base,forcing,kind=kind,canopy_capacity=capacity,capacity_environment=environment,**kwargs)
    module,_,name,_=_kind(kind);package=__package__ if backend=='jax' else __package__+'.cuda'
    run=getattr(import_module('.'+module,package),name+'_'+backend)
    if kind in ('wetland','irrigated'):
        return run(base,forcing,canopy_capacity=capacity,capacity_environment=environment,**kwargs)
    return run(*base,forcing,canopy_capacity=capacity,capacity_environment=environment,**kwargs)


def step_clm_capacity_columns_jax(arrays,forcing,*,kind='bgc',structured=False,environment=None,**kwargs):
    return _step(arrays,forcing,'jax',kind,structured,environment,kwargs)


def step_clm_capacity_columns_cuda(arrays,forcing,*,kind='bgc',structured=False,environment=None,**kwargs):
    return _step(arrays,forcing,'cuda',kind,structured,environment,kwargs)


def next_capacity_arrays(previous,result,*,kind='bgc',structured=False):
    base=previous[:-4]
    if structured:
        from .clm_structure_column import next_structure_arrays
        after=next_structure_arrays(base,result,kind=kind)
    else:
        module,_,_,carry=_kind(kind)
        if kind=='layered':
            if len(base)!=8:raise ValueError('Layered capacity requires eight base arrays')
            after=(result['snow_cells'],result['soil_cells'],base[2],result['options'],base[4],result['storage'],base[6],result['condensate'])
        else:after=getattr(import_module('.'+module,__package__),carry)(base,result)
    return (*after,result['capacity_temperature_values'],result['capacity_temperature_controls'],
            result['capacity_configuration'],result['capacity_traits'])

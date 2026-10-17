"""Native GDD update followed by the complete layered CLM column.

The extended ABI appends phenology state, integer flags and parameters to the
eight layered-column arrays. Its entry point explicitly selects OptDynamicVeg=5;
the reused transport ABI has no dynamic-vegetation option slot. LAI is an alias
of the prognostic leaf fraction, updated before precipitation interception.
All physical calculations, including inversion of old third-layer soil H,
run on the selected device. No host NumPy driver supplies device tendencies.
"""
from dataclasses import astuple,replace
import numpy as np
from .clm_layered_column import pack_clm_layered_columns,validate_layered_column
from .phenology import validate_phenology
from ..physics.phenology import PhenologyEnvironment


def pack_clm_phenology_columns(states,**kwargs):
    states=list(states);dtype=kwargs.get('dtype',np.float64)
    if any(s.options.OptDynamicVeg!=5 for s in states):
        raise ValueError('Every seasonal column must select OptDynamicVeg=5')
    base=pack_clm_layered_columns([replace(s,options=replace(s.options,OptDynamicVeg=4),
        phenology_state=None,phenology_parameters=None) for s in states],**kwargs)
    ps=np.asarray([[s.phenology_state.gdd,s.phenology_state.leaf_fraction,s.phenology_state.remaining_seconds]
        for s in states],dtype).reshape(-1,3)
    flags=np.asarray([[s.phenology_state.phase,s.phenology_state.armed] for s in states],np.int32).reshape(-1,2)
    parameters=np.asarray([astuple(s.phenology_parameters) for s in states],dtype).reshape(-1,6)
    return (*base,ps,flags,parameters)


def pack_phenology_environment(environments,*,dtype=np.float64):
    values=list(environments)
    if any(not isinstance(e,PhenologyEnvironment) for e in values):raise ValueError('Expected PhenologyEnvironment values')
    return (np.asarray([[e.daylength_seconds,e.latitude_degrees] for e in values],dtype).reshape(-1,2),
        np.asarray([[e.increasing_daylength,e.winter_solstice] for e in values],np.int32).reshape(-1,2))


def validate_seasonal_column(arrays,forcing,environment,calendar,xp):
    if len(arrays)!=11:raise ValueError('Seasonal CLM columns require eleven state arrays')
    n=len(arrays[0]);dtype=arrays[0].dtype
    validate_layered_column(arrays[:8],forcing,xp.zeros((n,5),dtype),xp)
    if environment.shape!=(n,2) or environment.dtype!=dtype:
        raise ValueError('Seasonal environment requires matching (n,2) daylength/latitude values')
    if not bool(xp.all(xp.isfinite(environment))) or bool(xp.any((abs(environment[:,1])<=19.5)|(abs(environment[:,1])>90.))):
        raise ValueError('Seasonal phenology requires explicit extratropical latitude')
    # Temperature is independently validated after inversion of conserved H.
    u=xp.stack((xp.full(n,273.15,dtype),environment[:,0],forcing[:,0]),axis=1)
    ps,flags,parameters=arrays[8:];validate_phenology(ps,flags,u,calendar,parameters,xp)
    lai=parameters[:,4]+(parameters[:,5]-parameters[:,4])*ps[:,1]
    if bool(xp.any(abs(arrays[4][:,0]-lai)>8.*xp.finfo(dtype).eps*xp.maximum(lai,1.))):
        raise ValueError('LAI alias disagrees with the prognostic phenology state')


def _step(arrays,forcing,environment,calendar,backend,**kwargs):
    if backend=='jax':
        import jax.numpy as xp
        from .enthalpy import run_soil_thermodynamics_jax as inverse
        from .phenology import run_phenology_jax as phenology
        from .clm_layered_column import step_clm_layered_columns_jax as column
        arrays=tuple(map(xp.asarray,arrays));forcing=xp.asarray(forcing);environment=xp.asarray(environment)
    else:
        import cupy as xp
        from .cuda.enthalpy import run_soil_thermodynamics_cuda as inverse
        from .cuda.phenology import run_phenology_cuda as phenology
        from .cuda.clm_layered_column import step_clm_layered_columns_cuda as column
        # Reject fractional option flags before applying the float32 ABI.
        if any(xp.asarray(arrays[k]).dtype.kind not in 'iu' for k in (3,9)):
            raise ValueError('CLM and phenology flags must be integers')
        arrays=tuple(xp.asarray(a,xp.int32 if k in (3,9) else xp.float32) for k,a in enumerate(arrays))
        forcing=xp.asarray(forcing,xp.float32);environment=xp.asarray(environment,xp.float32)
    calendar=xp.asarray(calendar)
    validate_seasonal_column(arrays,forcing,environment,calendar,xp)
    c,g,p,o,site,storage,roots,condensate,ps,flags,parameters=arrays
    soil=xp.concatenate((g[:,2,:],p[:,xp.asarray([0,1,6])]),axis=1)
    temperature=inverse(soil,o[:,1],inverse=True)['temperature_k']
    climate=xp.stack((temperature,environment[:,0],forcing[:,0]),axis=1)
    result=phenology(ps,flags,climate,calendar,parameters)
    if backend=='jax':next_site=site.at[:,0].set(result['lai'])
    else:next_site=site.copy();next_site[:,0]=result['lai']
    out=column(c,g,p,o,next_site,storage,roots,condensate,forcing,**kwargs)
    result['lai_before']=site[:,0]
    out.update(site=next_site,phenology=result,phenology_parameters=parameters)
    return out


def step_clm_phenology_columns_jax(c,g,p,o,site,storage,roots,condensate,ps,flags,parameters,
                                    forcing,*,environment,calendar,**kwargs):
    """Advance option 5; retain returned site and phenology state between steps.

    Other keyword arguments match step_clm_layered_columns_jax (VIC, MOST,
    deposition and leaf parameters). Leaf availability is not a carbon pool.
    """
    return _step((c,g,p,o,site,storage,roots,condensate,ps,flags,parameters),forcing,
        environment,calendar,'jax',**kwargs)


def next_phenology_arrays(previous,result):
    """Device-preserving state carry; immutable soil parameters/roots are reused."""
    return (result['snow_cells'],result['soil_cells'],previous[2],result['options'],result['site'],
        result['storage'],previous[6],result['condensate'],result['phenology']['state'],
        result['phenology']['flags'],result['phenology_parameters'])

"""Attach explicit zero-flux Richards transport to a native physical column.

The finish stage must suppress legacy gravity drainage first. This adapter
then advances actual soil W/H and recomputes the full-column diagnostic
ledger; it cannot erase an outlet after the fact or change material Ksat.
"""


def resolve_column_richards(parameters,options,xp):
    if parameters is None:return None
    from ..physics.richards import RichardsHydrology
    if not isinstance(parameters,RichardsHydrology):raise TypeError('Expected RichardsHydrology')
    if bool(xp.any(options[:,7]!=3)):
        raise ValueError('Closed Richards requires subsurface routing flag 3; requested bottom outlets cannot be silently removed')
    return parameters


def _sum_rows(values,xp):
    out=xp.zeros(len(values),values.dtype)
    for k in range(values.shape[1]):out=out+values[:,k]
    return out


def attach_column_richards(result,p,o,forcing,old_c,old_g,old_store,parameters,backend):
    if parameters is None:return result
    if backend=='jax':
        import jax.numpy as xp
        from .richards_enthalpy import run_richards_enthalpy_jax as run
    elif backend=='cuda':
        import cupy as xp
        from .cuda.richards_enthalpy import run_richards_enthalpy_cuda as run
    else:raise ValueError('Unknown Richards column backend')
    before=result['soil_cells']
    properties=xp.stack((p[:,0],p[:,1],p[:,3],p[:,6]),axis=1)
    transport=run(before,properties,o[:,1],forcing[:,0],**parameters.controls())
    ground=transport['cells'];storage=result['storage']
    temperature=xp.where(result['options'][:,0]>0,storage[:,2],transport['temperature_k'][:,0])
    if backend=='jax':storage=storage.at[:,2].set(temperature)
    else:storage=storage.copy();storage[:,2]=temperature
    c=result['snow_cells'];d=dict(result['diagnostics'])
    water0=(_sum_rows(old_c[:,:,0],xp)+_sum_rows(old_g[:,:,0],xp))+old_store[:,0]
    water1=(_sum_rows(c[:,:,0],xp)+_sum_rows(ground[:,:,0],xp))+storage[:,0]
    heat0=(_sum_rows(old_c[:,:,1],xp)+_sum_rows(old_g[:,:,1],xp))+old_store[:,1]
    heat1=(_sum_rows(c[:,:,1],xp)+_sum_rows(ground[:,:,1],xp))+storage[:,1]
    water_residual=(((water1-water0)-d['prcp_mm'])+d['et_mm'])+d['runoff_mm']
    heat_residual=(((((heat1-heat0)-d['precipitation_enthalpy_jm2'])-d['atmospheric_energy_jm2'])+
        d['advected_enthalpy_jm2'])+d['runoff_enthalpy_jm2'])
    d.update(ts=temperature,water_residual_kgm2=water_residual,energy_residual_jm2=heat_residual,
        runoff_surface_mm=d['runoff_mm'],runoff_deep_mm=xp.zeros_like(d['runoff_mm']))
    transport=dict(transport,water_before_kg_m2=before[:,:,0],enthalpy_before_j_m2=before[:,:,1],
        lower_boundary_condition=xp.full(len(before),2,xp.int32))
    return dict(result,soil_cells=ground,storage=storage,diagnostics=d,soil_hydrology=transport)

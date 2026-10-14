"""Device-array MOST feedback around the native layered surface solvers.

The host dispatches bounded iterations and reads their all-converged flag.
Profiles, H/E feedback, convergence checks, state and every physical solve
remain on the selected device. All trials use the identical old M/H arrays.
"""


def run_most_surface(arrays,geometry,wind,leaf_dimension,plant_area,*,backend,parameters=None):
    if backend=='jax':
        import jax.numpy as xp
        from .canopy_most import run_canopy_most_jax as coefficients
        from .canopy_phase_surface import run_canopy_phase_surface_jax as solve
    elif backend=='cuda':
        import cupy as xp
        from .cuda.canopy_most import run_canopy_most_cuda as coefficients
        from .cuda.canopy_phase_surface import run_canopy_phase_surface_cuda as solve
    else:raise ValueError('MOST surface backend must be jax or cuda')
    arrays=tuple(xp.asarray(v) for v in arrays);u=arrays[5];n=len(u);dtype=u.dtype
    geometry=xp.asarray(geometry,dtype);wind=xp.asarray(wind,dtype)
    leaf_dimension=xp.broadcast_to(xp.asarray(leaf_dimension,dtype),(n,))
    plant_area=xp.broadcast_to(xp.asarray(plant_area,dtype),(n,))
    zero=xp.zeros(n,dtype);zeta=zero;effective=xp.maximum(1.,wind)
    done=xp.zeros(n,bool);iterations=xp.zeros(n,xp.int32);previous=None
    lower=zero;upper=zero+2.;stable_bracket=xp.zeros(n,bool)
    atmosphere=xp.stack((u[:,0],u[:,1],u[:,2],wind,zero,zero,zero+.2),axis=1)
    state=xp.stack((zeta,effective,leaf_dimension,plant_area),axis=1)
    current=coefficients(geometry,atmosphere,state)
    names=('friction_velocity','heat_velocity','water_velocity','leaf_velocity','ground_velocity')
    for count in range(1,51):
        trial=xp.concatenate((u[:,:3],xp.stack((current['heat_velocity'],current['ground_velocity'],current['leaf_velocity']),axis=1),u[:,6:]),axis=1)
        extension=xp.stack((current['water_velocity'],current['potential_temperature']),axis=1)
        surface=solve(*arrays[:5],trial,*arrays[6:],parameters=parameters,exchange_extension=extension)
        vapor=surface['et_kgm2']/arrays[7][:,0]
        atmosphere=xp.stack((u[:,0],u[:,1],u[:,2],wind,surface['h'],vapor,current['friction_velocity']),axis=1)
        feedback=coefficients(geometry,atmosphere,state)
        if count==1:stable_bracket=feedback['zeta']>0.
        target_state=xp.stack((feedback['zeta'],feedback['effective_wind'],leaf_dimension,plant_area),axis=1)
        target=coefficients(geometry,atmosphere,target_state)
        mismatch=xp.max(xp.stack([xp.abs(target[k]-current[k])/current[k] for k in names],axis=1),axis=1)
        if previous is not None:
            dh=xp.abs(surface['h']-previous[0]);de=xp.abs(vapor-previous[1])
            dt=xp.max(xp.abs(surface['leaf_temperature']-previous[2]),axis=1)
            accepted=(mismatch<=1e-4)&(dh<=.05)&(de<=2e-8)&(dt<=.01)
            iterations=xp.where((~done)&accepted,xp.asarray(count,xp.int32),iterations);done=done|accepted
            if bool(xp.all(done)):
                surface['most']={**current,**{k:feedback[k] for k in ('zeta','raw_zeta','buoyancy','convective_wind','effective_wind')},
                    'used_zeta':zeta,'used_effective_wind':effective,'iterations':iterations,
                    'coefficient_residual':mismatch,'heat_change_wm2':dh,'vapor_change_kgm2s':de,'leaf_temperature_change_k':dt}
                return surface
        previous=(surface['h'],vapor,surface['leaf_temperature'])
        lower=xp.where(stable_bracket&(~done)&(feedback['zeta']>=zeta),zeta,lower)
        upper=xp.where(stable_bracket&(~done)&(feedback['zeta']<zeta),zeta,upper)
        next_zeta=xp.where(stable_bracket,.5*(lower+upper),.5*(zeta+feedback['zeta']))
        zeta=xp.where(done,zeta,next_zeta)
        effective=xp.where(done|stable_bracket,effective,.5*(effective+feedback['effective_wind']))
        state=xp.stack((zeta,effective,leaf_dimension,plant_area),axis=1)
        current=coefficients(geometry,atmosphere,state)
    raise RuntimeError('Native MOST flux feedback did not converge in 50 iterations')

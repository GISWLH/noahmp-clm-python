"""Native four-layer irrigation requests and externally funded water/heat."""
from dataclasses import astuple
from functools import lru_cache
import numpy as np
from ..physics.irrigation import IrrigationParameters

DIAGNOSTICS=('available_kgm2','threshold_kgm2','deficit_kgm2','frozen_kgm2','eligible','triggered',
    'requested_kgm2','applied_kgm2','applied_enthalpy_jm2','unmet_kgm2','cancelled_kgm2',
    'request_residual_kgm2','request_bound_kgm2')


def pack_irrigation(state,liquid,ice,dz,dt,lai,*,check=False,supply_kgm2=0.,supply_temperature_k=285.,
                    parameters=IrrigationParameters(),dtype=np.float64):
    if not isinstance(check,(bool,np.bool_)):raise ValueError('Irrigation check must be boolean')
    if any(np.shape(v)!=(4,) for v in (liquid,ice,dz)):raise ValueError('Native irrigation requires four soil layers')
    return (np.asarray(astuple(state),dtype),np.column_stack((liquid,ice,dz)).astype(dtype),
        np.asarray([dt,lai,supply_kgm2,supply_temperature_k],dtype),np.asarray([check],np.int32),np.asarray(astuple(parameters),dtype))


def validate_irrigation(state,layers,u,flags,p,xp):
    n=len(state);dtype=state.dtype
    if (state.shape!=(n,2) or layers.shape!=(n,4,3) or u.shape!=(n,4) or p.shape!=(n,5)
        or any(v.dtype!=dtype for v in (layers,u,p)) or dtype not in (xp.float32,xp.float64)
        or not all(bool(xp.all(xp.isfinite(v))) for v in (state,layers,u,p))):
        raise ValueError('Invalid native irrigation shapes, precision or nonfinite values')
    if flags.shape!=(n,1) or flags.dtype.kind not in 'iu' or bool(xp.any((flags<0)|(flags>1))):
        raise ValueError('Irrigation checks must be integer flags')
    bad=(xp.any(state<0.,axis=1)|((state[:,0]==0.)!=(state[:,1]==0.))|((flags[:,0]>0)&(state[:,0]>0.)))
    bad|=xp.any((layers[:,:,0]<0.)|(layers[:,:,1]<0.)|(layers[:,:,2]<=0.)|
        (layers[:,:,0]+layers[:,:,1]>1000.*layers[:,:,2]),axis=1)
    bad|=(u[:,0]<=0.)|(u[:,0]>86400.)|(u[:,1]<0.)|(u[:,2]<0.)|(u[:,3]<273.15)|(u[:,3]>350.)
    bad|=(p[:,0]<=0.)|(p[:,2]<0.)|(p[:,1]<=p[:,2])|(p[:,1]>1.)|(p[:,3]<=0.)|(p[:,3]>1.)|(p[:,4]<=0.)|(p[:,4]>86400.)
    if bool(xp.any(bad)):raise ValueError('Invalid native irrigation request, soil, supply or parameters')


def irrigation_kernel_jax(old,layers,u,flags,p):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;top=j.asarray(0.,old.dtype);depth=j.asarray(0.,old.dtype)
    available=j.asarray(0.,old.dtype);frozen=j.asarray(0.,old.dtype)
    for k in range(4):
        thickness=j.minimum(layers[k,2],j.maximum(s(p[0]-top),0.));fraction=s(thickness/layers[k,2])
        depth=s(depth+thickness);available=s(available+s(layers[k,0]*fraction));frozen=s(frozen+s(layers[k,1]*fraction));top=s(top+layers[k,2])
    target=s(s(1000.*p[1])*depth);wilt=s(s(1000.*p[2])*depth)
    threshold=s(s(p[3]*s(target-wilt))+wilt);deficit=j.maximum(s(threshold-available),0.)
    eligible=(u[1]>0.)&(frozen<=1e-9);cancelled=j.where(eligible,0.,old[0])
    requested=j.where((flags[0]>0)&eligible,deficit,0.)
    pending=j.where(requested>0.,requested,j.where(eligible,old[0],0.))
    seconds=j.where(requested>0.,p[4],j.where(eligible,old[1],0.))
    quota=j.where(pending>0.,j.where(u[0]>=seconds,pending,s(pending*s(u[0]/j.where(seconds>0.,seconds,1.)))),0.)
    finish=(pending==0.)|(u[0]>=seconds)
    pending=j.where(finish,0.,s(pending-quota));seconds=j.where(finish,0.,s(seconds-u[0]))
    applied=j.minimum(quota,u[2]);unmet=s(quota-applied);heat=s(applied*s(334000.+s(4180.*s(u[3]-273.15))))
    residual=s(s(s(s(s(pending-old[0])+applied)+unmet)+cancelled)-requested)
    scale=s(s(s(s(s(pending+old[0])+applied)+unmet)+cancelled)+requested)
    bound=s(128.*j.finfo(old.dtype).eps*j.maximum(scale,1.))
    valid=(abs(residual)<=bound)&(pending>=0.)&(seconds>=0.)
    return j.asarray([pending,seconds,available,threshold,deficit,frozen,eligible,requested>0.,requested,
        applied,heat,unmet,cancelled,residual,bound,valid],old.dtype)


def unpack_irrigation(out,xp):
    result=dict(state=out[:,:2]);result.update(zip(DIAGNOSTICS,out[:,2:15].T))
    for name in ('eligible','triggered'):result[name]=result[name]>0.
    if not bool(xp.all(xp.isfinite(out))) or not bool(xp.all(out[:,15]>0.)):
        raise RuntimeError('Native irrigation request balance or output failed')
    return result


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(irrigation_kernel_jax))


def run_irrigation_jax(state,layers,climate,flags,parameters):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(state,layers,climate,flags,parameters)));validate_irrigation(*arrays,j)
    return unpack_irrigation(_compiled()(*arrays),j)


def prepare_column_irrigation(g,p,o,site,forcing,state,parameters,inputs,checks,backend):
    """Diagnose old soil phases and current managed LAI entirely on device."""
    if state is None:
        if any(v is not None for v in (parameters,inputs,checks)):raise ValueError('Irrigation inputs require request state and parameters')
        return None,None
    if parameters is None:raise ValueError('Irrigation request state requires parameters')
    if backend=='jax':
        import jax.numpy as xp
        from .enthalpy import run_soil_thermodynamics_jax as inverse
        run=run_irrigation_jax
    else:
        import cupy as xp
        from .cuda.enthalpy import run_soil_thermodynamics_cuda as inverse
        from .cuda.irrigation import run_irrigation_cuda as run
    n=len(g);dtype=g.dtype;state=xp.asarray(state,dtype);parameters=xp.asarray(parameters,dtype)
    inputs=xp.broadcast_to(xp.asarray([0.,285.],dtype),(n,2)) if inputs is None else xp.asarray(inputs,dtype)
    checks=xp.zeros((n,1),xp.int32) if checks is None else xp.asarray(checks)
    if inputs.shape!=(n,2):raise ValueError('Irrigation supply/temperature inputs require shape (n,2)')
    thermodynamic_parameters=p[:,xp.asarray([0,1,6])]
    si=xp.concatenate((g,xp.broadcast_to(thermodynamic_parameters[:,None,:],(n,4,3))),axis=2).reshape(-1,6)
    therm=inverse(si,xp.repeat(o[:,1],4),inverse=True)
    temperature=therm['temperature_k'].reshape(n,4);raw=therm['liquid_kgm2'].reshape(n,4)
    liquid=xp.where(temperature>=273.15,g[:,:,0],xp.clip(raw,0.,g[:,:,0]))
    layers=xp.stack((liquid,g[:,:,0]-liquid,g[:,:,2]),axis=2)
    climate=xp.stack((forcing[:,0],site[:,0],inputs[:,0],inputs[:,1]),axis=1)
    out=run(state,layers,climate,checks,parameters)
    if bool(xp.any(parameters[:,1]>p[:,0])):raise ValueError('Irrigation target exceeds soil porosity')
    out['parameters']=parameters
    return xp.stack((out['applied_kgm2'],out['applied_enthalpy_jm2']),axis=1),out


def attach_column_irrigation(result,irrigation):
    if irrigation is None:return result
    # The physical closure stage received total external M/H. Restore the
    # separately measured precipitation from the untouched interception
    # ledger; do not subtract two large nearly equal water/heat amounts.
    pre=result['canopy_interception']['ledger'];d=result['diagnostics']
    d['prcp_mm']=(pre['intercepted_rain_kgm2']+pre['through_rain_kgm2'])+(pre['intercepted_snow_kgm2']+pre['through_snow_kgm2'])
    d['precipitation_enthalpy_jm2']=pre['precipitation_enthalpy_jm2']
    d['irrigation_mm']=irrigation['applied_kgm2'];d['irrigation_enthalpy_jm2']=irrigation['applied_enthalpy_jm2']
    result.update(irrigation=irrigation,irrigation_state=irrigation['state'],irrigation_parameters=irrigation['parameters'])
    return result

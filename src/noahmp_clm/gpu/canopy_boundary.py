"""Fixed snow/soil surface temperature, native optics and canopy water heat.

This is the atmosphere boundary of a prognostic enthalpy column. The caller
must match ground_nonshortwave_heat plus depth-resolved shortwave to column
enthalpy. Intercepted water remains liquid; canopy snow is not implied.
"""
from functools import lru_cache
import numpy as np
from .canopy_energy import (pack_canopy_case,validate_canopy_batch,
    canopy_energy_kernel_jax,unpack_canopy_energy,_sum)
from .canopy_snow_radiation import validate_canopy_snow,canopy_snow_kernel_jax
from .snicar import pack_snow_optics,_jax_tables
from .photosynthesis import pack_leaf_parameters


def pack_canopy_boundary(area,swdown,lwdown,exchange,*,fixed_ground_temperature,
                         ground_is_ice=False,ground_shares_soil=True,snowpack=None,
                         canopy_storage=None,dtype=np.float64,**kwargs):
    from ..physics.snowpack import Snowpack
    base=pack_canopy_case(area,swdown,lwdown,exchange,dtype=dtype,**kwargs)
    b=np.zeros(11,dtype);b[:3]=fixed_ground_temperature,ground_is_ice,ground_shares_soil
    b[6]=1.
    if canopy_storage is not None:
        cs=canopy_storage
        if cs.root_enthalpy.shape!=(4,):raise ValueError('Four root source enthalpies required')
        b[3:]=np.r_[1.,cs.mass,cs.enthalpy,cs.dt,cs.root_enthalpy]
    c,n,_=pack_snow_optics(Snowpack.empty() if snowpack is None else snowpack,[0.,0.],[0.,0.],dtype=dtype)
    return (*base,b,c,n)


def radiation_inputs(area,u,xp):
    fv=xp.where(xp.sum(area,axis=1)>0.,u[:,13],0.)
    return xp.stack((u[:,11],fv,u[:,21],u[:,22],u[:,19],u[:,20]),axis=1)


def validate_boundary(arrays,xp):
    area,u,f,soil,initial,b,c,n=arrays
    validate_canopy_batch(area,u,f,soil,initial,xp)
    if b.shape!=(len(area),11):raise ValueError('Boundary shape requires (n,11)')
    if any(a.dtype!=area.dtype for a in (u,f,soil,initial,b,c)) or area.dtype not in (xp.float32,xp.float64):
        raise ValueError('Boundary floats must have matching float32/float64 dtype')
    if not bool(xp.all(xp.isfinite(b))):raise ValueError('Nonfinite canopy boundary')
    flags=b[:,1:4]
    bad=((b[:,0]<200.)|(b[:,0]>350.)|(b[:,4]<0.)|(b[:,6]<=0.)
        |((b[:,1]>0.)&(b[:,0]>xp.asarray(273.15,dtype=b.dtype)))
        |((b[:,4]==0.)&(b[:,5]!=0.)))
    eps=xp.finfo(b.dtype).eps
    limit=b[:,4]/xp.maximum(b[:,6],xp.finfo(b.dtype).tiny)
    bad|=(b[:,3]>0.)&((soil[:,0]==0.)|(u[:,15]>limit*(1.+8.*eps))
        |(((xp.sum(area,axis=1)==0.)|(u[:,13]==0.))&(b[:,4]>0.)))
    if bool(xp.any(bad)) or bool(xp.any((flags!=0.)&(flags!=1.))):
        raise ValueError('Invalid fixed surface, canopy storage or soil supply')
    validate_canopy_snow(area,c,n,radiation_inputs(area,u,xp),xp)


def unpack_boundary(out):
    result=unpack_canopy_energy(out[:,:195])
    result.update(ground_nonshortwave_heat=out[:,195],snow_absorbed_shortwave=out[:,196:201],
        soil_absorbed_shortwave=out[:,201],ground_latent_heat=out[:,202],
        canopy_storage_change_wm2=out[:,203],canopy_water_after_kgm2=out[:,204],
        canopy_enthalpy_after_jm2=out[:,205],wet_advected_heat_wm2=out[:,206],
        transpiration_advected_heat_wm2=out[:,207])
    return result


def canopy_boundary_kernel_jax(area,u,f,soil,initial,b,c,n,p,ice,impurity,*,max_iterations=100,
                                tolerance=.005,q_tolerance=1e-8):
    import jax.numpy as j
    rinput=j.stack((u[11],j.where(_sum(area)>0.,u[13],0.),u[21],u[22],u[19],u[20]))
    radiation=canopy_snow_kernel_jax(area,c,n,rinput,ice,impurity,track_absorption=True)
    return canopy_energy_kernel_jax(area,u,f,soil,initial,p,max_iterations=max_iterations,
        tolerance=tolerance,q_tolerance=q_tolerance,boundary=b,radiation=radiation)


@lru_cache(maxsize=8)
def _compiled(max_iterations,tolerance,q_tolerance):
    import jax
    from functools import partial
    return jax.jit(jax.vmap(partial(canopy_boundary_kernel_jax,max_iterations=max_iterations,
        tolerance=tolerance,q_tolerance=q_tolerance),in_axes=(0,0,0,0,0,0,0,0,None,None,None)))


def run_canopy_boundary_jax(area,u,f,soil,initial,b,c,n,parameters=None,*,max_iterations=100,
                            tolerance=None,raise_on_failure=True):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(area,u,f,soil,initial,b,c,n)));validate_boundary(arrays,j)
    dtype=arrays[0].dtype
    tol=(.005 if dtype==j.float32 else 5e-5) if tolerance is None else tolerance
    if not isinstance(max_iterations,int) or max_iterations<1 or not np.isfinite(tol) or tol<=0.:
        raise ValueError('Invalid canopy boundary solver controls')
    qtol=1e-8 if dtype==j.float32 else 1e-10
    p=j.asarray(pack_leaf_parameters(parameters,np.dtype(dtype)))
    out=_compiled(max_iterations,float(tol),qtol)(*arrays,p,*_jax_tables(str(dtype)))
    result=unpack_boundary(out)
    if not bool(j.all(j.isfinite(out))):raise RuntimeError('Nonfinite canopy boundary result')
    if raise_on_failure and not bool(j.all(result['converged'])):
        raise RuntimeError('JAX canopy boundary did not converge; inspect with raise_on_failure=False')
    return result

"""Native layered canopy phase coupled to snow and soil extensive stores."""
from functools import lru_cache
import numpy as np
from .canopy_surface import pack_canopy_surface,validate_surface,surface_kernel_jax,unpack_surface,_radiation
from .canopy_phase_energy import validate_phase,unpack_phase
from .canopy_boundary import radiation_inputs
from .photosynthesis import pack_leaf_parameters
from .snicar import _jax_tables
from .snow_evolution import _tables as _aging


def pack_canopy_phase_surface(snow,soil,dt,area,swdown,lwdown,exchange,condensate,*,dtype=np.float64,**kwargs):
    if condensate.mass.shape!=(10,):raise ValueError('Ten canopy condensate layers required')
    if {'canopy_water_kgm2','canopy_enthalpy_jm2'} & kwargs.keys():raise ValueError('Use the layer stores, not both layer and aggregate storage')
    base=pack_canopy_surface(snow,soil,dt,area,swdown,lwdown,exchange,
        canopy_water_kgm2=float(condensate.mass.sum()),canopy_enthalpy_jm2=float(condensate.enthalpy.sum()),dtype=dtype,**kwargs)
    return (*base,np.column_stack((condensate.mass,condensate.enthalpy)).astype(dtype))


def validate_phase_surface(arrays,xp):
    *base,condensate=arrays;validate_surface(base,xp)
    c,g,p,o,a,u,f,v=base;n=len(c);dtype=c.dtype
    supply=xp.zeros((n,10),dtype=dtype)
    supply=xp.concatenate((xp.ones((n,1),dtype=dtype),supply[:,1:]),axis=1)
    boundary=xp.stack((xp.ones(n,dtype),u[:,0],xp.zeros(n,dtype),xp.ones(n,dtype),v[:,0]),axis=1)
    boundary=xp.concatenate((boundary,xp.zeros((n,4),dtype=dtype)),axis=1)
    validate_phase((a,u,f,supply,xp.zeros_like(a),condensate,boundary,c[:,:,[0,3,4,5,6,7,8]],o[:,0]),xp)
    # Aggregate aliases are packing metadata only; device reductions below
    # establish the authoritative sums from individual layer state.
    eps=xp.finfo(dtype).eps
    for k in range(2):
        total=xp.sum(condensate[:,:,k],axis=1)
        if bool(xp.any(abs(total-v[:,1+k])>16.*eps*xp.maximum(xp.sum(abs(condensate[:,:,k]),axis=1),1.))):
            raise ValueError('Aggregate canopy aliases disagree with layer state')


def unpack_phase_surface(out):
    result=unpack_surface(out[:,:441]);phase=unpack_phase(out[:,441:741])
    # Surface LE includes the actual ground phase parcel. Keep it separate
    # from the fixed-boundary canopy solver's predicted ground latent heat.
    result['condensate']=out[:,636:656].reshape(-1,2,10).transpose(0,2,1)
    for key in ('mass','enthalpy','thermal_enthalpy','ice_vapor','liquid_vapor','vapor_enthalpy',
                'vapor_latent_energy','unmet_evaporation','phase_coordinates','root_advected_heat_wm2',
                'transpiration_net_heat_wm2','restarts','max_solver_residual'):
        result[key]=phase[key]
    result['canopy_converged']=phase['converged']
    return result


@lru_cache(maxsize=4)
def _compiled(extended=False,vic=False):
    import jax
    axes=(0,0,0,0,0,0,0,0,0,None,None,0)
    return jax.jit(jax.vmap(surface_kernel_jax,in_axes=axes+(0 if extended else None,0 if vic else None)))


def validate_exchange_extension(extension,n,xp):
    if extension.shape!=(n,2) or not bool(xp.all(xp.isfinite(extension))) or bool(xp.any(
            (extension[:,0]<=0.)|(extension[:,1]<200.)|(extension[:,1]>400.))):
        raise ValueError('Invalid separate water velocity/potential temperature extension')


def run_canopy_phase_surface_jax(*arrays,parameters=None,raise_on_failure=True,exchange_extension=None,vic=None):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,arrays));validate_phase_surface(arrays,j)
    c,g,p,o,a,u,f,v,condensate=arrays;dtype=str(c.dtype)
    radiation=_radiation()(a,c[:,:,j.array([0,3,4,5,6,7,8])],o[:,0],radiation_inputs(a,u,j),*_jax_tables(dtype))
    extra=() if exchange_extension is None else (j.asarray(exchange_extension),)
    if extra:
        validate_exchange_extension(extra[0],len(c),j)
        if extra[0].dtype!=c.dtype:raise ValueError('Exchange extension precision mismatch')
    if vic is not None:
        from .vic import validate_vic
        vic=j.asarray(vic);validate_vic(vic,len(c),j)
        if vic.dtype!=c.dtype:raise ValueError('VIC precision mismatch')
    out=_compiled(bool(extra),vic is not None)(*arrays[:8],radiation,j.asarray(pack_leaf_parameters(parameters,np.dtype(dtype))),_aging(dtype),condensate,
        extra[0] if extra else None,vic)
    result=unpack_phase_surface(out)
    if raise_on_failure and (not bool(j.all(j.isfinite(out))) or not bool(j.all(result['converged']))):
        raise RuntimeError('Layered phase surface failed; inspect with raise_on_failure=False')
    return result

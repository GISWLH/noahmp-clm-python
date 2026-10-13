"""Developing snow-free CLM column on device, using the released packed ABI.

Four soil layers (0.1, 0.3, 0.6, 1 m) and ten canopy layers. Preparation,
canopy energy and water/heat advancement are separate device kernels, allowing
their intermediate budgets to be checked without running host physics.
Winter and the final v1 state layout are still under development.
"""
from functools import lru_cache
import numpy as np
from .pack import FLOAT_SIZE, INT_SIZE, OPTION_FIELDS, DZ_DEFAULT, column_state_to_vec
from .canopy_energy import _sum, _dot, unpack_canopy_energy
from .photosynthesis import pack_leaf_parameters

CLM_DIAGNOSTICS = ("ts","rnet","h","le","g","et_mm","prcp_mm","runoff_mm",
    "melt_mm","swe","canw","residual","gpp_umol_m2s","gpp_gcm2",
    "leaf_respiration_umol_m2s","canopy_temperature","canopy_humidity",
    "water_residual_m","latent_water_residual_wm2","canopy_iterations",
    "canopy_converged","withdrawal_valid")
SITE_FIELDS = ("co2_ppm","cosine_zenith","diffuse_fraction","leaf_dimension","nitrogen_factor")


def pack_clm_columns(states, dtype=np.float64):
    """Check soil geometry before discarding it into the fixed GPU layout."""
    states=list(states)
    if not states:
        return np.empty((0,FLOAT_SIZE),dtype),np.empty((0,INT_SIZE),np.int32)
    for state in states:
        if not np.array_equal(state.dz,DZ_DEFAULT):
            raise ValueError("CLM GPU column currently requires the default four soil thicknesses")
    vectors=[column_state_to_vec(s) for s in states]
    return np.asarray([v[0] for v in vectors],dtype),np.asarray([v[1] for v in vectors],np.int32)


def site_array(n, dtype=np.float64, *, co2_ppm=420.,cosine_zenith=.5,
               diffuse_fraction=.5,leaf_dimension=.04,nitrogen_factor=1.):
    """Scalar or one value per column; explicitly records site assumptions."""
    return np.stack([np.broadcast_to(v,(n,)) for v in (co2_ppm,cosine_zenith,
        diffuse_fraction,leaf_dimension,nitrogen_factor)],axis=1).astype(dtype)


def validate_clm_batch(s,i,f,site,xp):
    n=s.shape[0] if s.ndim==2 else -1
    if s.shape!=(n,FLOAT_SIZE) or i.shape!=(n,INT_SIZE) or f.shape!=(n,8) or site.shape!=(n,5):
        raise ValueError("Invalid CLM column batch shapes")
    if not all(bool(xp.all(xp.isfinite(a))) for a in (s,f,site)):
        raise ValueError("Nonfinite CLM column input")
    if s.dtype not in (xp.float32,xp.float64) or f.dtype!=s.dtype or site.dtype!=s.dtype:
        raise ValueError("CLM state, forcing and site require matching floating dtype")
    if i.dtype!=xp.int32:
        raise ValueError("CLM options require int32")
    def opt(name): return i[:,OPTION_FIELDS.index(name)]
    supported={"OptStomataResistance":(1,),"OptRainSnowPartition":(1,2,3),
        "OptSurfaceDrag":(1,2),"OptSoilWaterTranspiration":(1,2,3),
        "OptGroundResistanceEvap":(1,2,3,4),"OptSoilSupercoolWater":(1,2),
        "OptSoilPermeabilityFrozen":(1,2),"OptRunoffSurface":(1,2,3,4),
        "OptRunoffSubsurface":(1,2,3,4),"OptDynamicVeg":(4,),
        "OptCropModel":(0,),"OptIrrigation":(0,),"OptWetlandModel":(0,),"OptTileDrainage":(0,)}
    for name,choices in supported.items():
        if bool(xp.any(~xp.isin(opt(name),xp.asarray(choices)))):
            raise NotImplementedError(f"CLM GPU {name} supports {choices}")
    bad=(xp.any((s[:,:4]<200)|(s[:,:4]>350),axis=1)
        |xp.any((s[:,4:8]<0)|(s[:,4:8]>s[:,39:40]),axis=1)
        |xp.any(s[:,12:16]<0,axis=1)|(s[:,30]<0)|(s[:,31]<0)|(s[:,32]<0)|(s[:,32]>1)
        |(s[:,34]<0)|(s[:,34]>1)|(s[:,35]<=0)|(s[:,35]>1)
        |(s[:,39]<=0)|(s[:,40]<0)|(s[:,40]>=s[:,41])|(s[:,41]>s[:,39])|(s[:,42]<0)|(s[:,43]<=0)
        |(f[:,0]<=0)|(f[:,1]<200)|(f[:,1]>350)|(f[:,2]<0)|(f[:,2]>=.2)|(f[:,4]<30000)
        |(f[:,5]<0)|(f[:,6]<0)|(site[:,0]<=0)|(site[:,1]<=0)|(site[:,1]>1)
        |(site[:,2]<0)|(site[:,2]>1)|(site[:,3]<=0)|(site[:,4]<0)|(site[:,4]>1))
    if bool(xp.any(bad)):
        raise ValueError("Unphysical CLM column input")
    tair=f[:,1]
    snowy=xp.where(opt("OptRainSnowPartition")==3,tair<273.15,
        xp.where(opt("OptRainSnowPartition")==2,tair<275.35,tair<=275.65))
    if bool(xp.any((s[:,26]>0)|xp.any(s[:,16:19]>0,axis=1)|((f[:,7]>0)&snowy))):
        raise NotImplementedError("CLM winter coupling requires the pending five-layer snow implementation")


def prepare_clm_jax(s,i,f,site):
    import jax.numpy as jnp
    from jax import lax
    from . import jax_core as core
    option=lambda name:core._option(i,name)
    dz=jnp.asarray(core.DZ,dtype=s.dtype)
    liquid=core._unfrozen_water(s[4:8],s[:4],s[39],s[43],option("OptSoilSupercoolWater"))
    ice=jnp.maximum(0.,s[4:8]-liquid)
    # Match the reference's explicit total-minus-phase supply calculation.
    liquid=jnp.maximum(s[4:8]-ice,0.)
    rain=jnp.maximum(0.,f[7])*f[0]
    capacity=.0005*s[32]*s[31]
    intercepted=jnp.where(s[31]>0.,jnp.minimum(s[32]*rain,jnp.maximum(0.,capacity-s[30])),0.)
    canw=s[30]+intercepted
    wet=jnp.where(capacity>0.,jnp.minimum(1.,(canw/jnp.maximum(capacity,1e-30))**(2./3.)),0.)
    stress,_=core._soil_stress(liquid,s[12:16],s[39],s[40],s[41],s[43],option("OptSoilWaterTranspiration"))
    ce=core._exchange_velocity(f[3],f[1],f[2],f[4],s[33],s[25],option("OptSurfaceDrag"))
    z0=jnp.clip(s[33],1e-5,5.)
    # Preserve ustar rounding before the second division in leaf conductance.
    ustar=lax.optimization_barrier(.4*jnp.maximum(f[3],.3)/jnp.log(10./z0))
    leaf_velocity=.01*jnp.sqrt(ustar/site[3])
    bare=(.4/.13)*(z0*ustar/1.5e-5)**(-.45)
    ground_velocity=(bare*jnp.exp(-s[31])+.004*(1.-jnp.exp(-s[31])))*ustar
    conduct=core._soil_conductivity(s[4:5],s[39],s[44])[0]/(.5*dz[0])
    resistance=core._ground_resistance(liquid[0],dz[0],s[39],s[40],s[43],option("OptGroundResistanceEvap"))
    u=jnp.stack((f[1],f[2],f[4],ce,ground_velocity,leaf_velocity,s[0],conduct,site[0],
        resistance,jnp.asarray(0.,s.dtype),f[5],f[6],s[32],wet,canw*1000./f[0],
        jnp.asarray(jnp.inf,s.dtype),jnp.asarray(jnp.inf,s.dtype),s[35],s[34],s[34],site[2],site[1]))
    soil=jnp.concatenate((jnp.ones(1,s.dtype),jnp.maximum(liquid-s[40],0.)*dz*1000./f[0],
        s[12:16],(lax.optimization_barrier(liquid[0]*dz[0])*1000./f[0])[None]))
    factors=jnp.broadcast_to(jnp.stack((site[4],stress)),(10,2))
    return (jnp.full((10,),s[31]/10.),u,factors,soil,jnp.full((11,),f[1]),
        jnp.stack((canw,rain-intercepted,capacity,rain)))


def finish_clm_jax(s,i,f,prep,c):
    import jax.numpy as jnp
    from jax import lax
    from . import jax_core as core
    option=lambda name:core._option(i,name)
    dz=jnp.asarray(core.DZ,dtype=s.dtype)
    ecan=_sum(c[30:40])*f[0]/1000.
    esoil=c[50]*f[0]/1000.
    roots=c[51:55]*f[0]/1000.
    transp=_sum(roots)
    canw=prep[0]-ecan
    tolerance=jnp.asarray(2e-8 if s.dtype==jnp.float32 else 1e-12,s.dtype)
    valid=canw>=-tolerance
    canw=jnp.maximum(0.,canw)
    drip=jnp.maximum(0.,canw-prep[2])
    canw-=drip
    work=s[4:8]
    remaining=prep[1]+drip
    for k in range(4):
        add=jnp.minimum(remaining,jnp.maximum(0.,(s[39]-work[k])*dz[k]))
        work=work.at[k].add(add/dz[k])
        remaining-=add
    runoff=jnp.maximum(0.,remaining)
    liquid=core._unfrozen_water(work,s[:4],s[39],s[43],option("OptSoilSupercoolWater"))
    ice=jnp.maximum(0.,work-liquid)
    valid=valid&jnp.all(roots<=jnp.maximum(work-ice-s[40],0.)*dz+tolerance)
    work=work-roots/dz
    valid=valid&(esoil<=jnp.maximum(0.,work[0]-ice[0])*dz[0]+tolerance)
    work=work.at[0].add(-esoil/dz[0])
    permeability=jnp.where(option("OptSoilPermeabilityFrozen")==2,
        jnp.exp(-7.*ice/jnp.maximum(s[39],1e-6)),1.)
    for k in range(4):
        q=s[42]*permeability[k]*jnp.clip(work[k]/s[39],0.,1.)**(2.*s[43]+3.)
        drain=jnp.minimum(jnp.maximum(0.,work[k]-s[41])*dz[k],q*f[0])
        work=work.at[k].add(-drain/dz[k])
        if k<3: work=work.at[k+1].add(drain/dz[k+1])
        else: runoff+=drain
    for k in range(4):
        excess=jnp.maximum(0.,work[k]-s[39])*dz[k]
        work=work.at[k].set(jnp.where(excess>0.,s[39],work[k]))
        if k<3: work=work.at[k+1].add(excess/dz[k+1])
        else: runoff+=excess
    rs,rd=core._route_runoff(runoff,work,s[39],s[41],s[37],s[38],
        option("OptRunoffSurface"),option("OptRunoffSubsurface"))
    runoff=rs+rd
    temp,ice=core._soil_heat(s[:4],work,ice,f[0],c[28],s[39],s[43],s[44],option("OptSoilSupercoolWater"))
    new=s.at[:4].set(temp).at[4:8].set(work).at[8:12].set(ice).at[25].set(c[10]).at[30].set(canw)
    et=ecan+transp+esoil
    storage0=_dot(s[4:8],dz)+s[30]
    storage1=_dot(work,dz)+canw
    diag=jnp.stack((c[10],c[29],c[26],c[27],c[28],et*1000.,prep[3]*1000.,runoff*1000.,
        jnp.asarray(0.,s.dtype),s[26],canw,c[66],c[13],c[13]*12.0107e-6*f[0],c[14],c[11],c[12],
        storage1-storage0-prep[3]+et+runoff,c[27]-lax.optimization_barrier(2.5e6*et)*1000./f[0],c[67],c[68],valid.astype(s.dtype)))
    return new,diag


@lru_cache(maxsize=1)
def _compiled_stages():
    import jax
    return jax.jit(jax.vmap(prepare_clm_jax)),jax.jit(jax.vmap(finish_clm_jax))


def step_clm_columns_jax(states,options,forcing,site=None,*,leaf_parameters=None,return_canopy=False):
    import jax.numpy as jnp
    from .canopy_energy import _compiled
    s,i,f=map(jnp.asarray,(states,options,forcing))
    v=jnp.asarray(site_array(s.shape[0],s.dtype) if site is None else site)
    validate_clm_batch(s,i,f,v,jnp)
    prepare,finish=_compiled_stages()
    area,u,factors,soil,initial,prep=prepare(s,i,f,v)
    tol,qtol=(.005,1e-8) if s.dtype==jnp.float32 else (5e-5,1e-10)
    c=_compiled(100,tol,qtol)(area,u,factors,soil,initial,jnp.asarray(pack_leaf_parameters(leaf_parameters,np.dtype(s.dtype))))
    if not bool(jnp.all(c[:,68]>0)):
        raise RuntimeError("CLM JAX canopy did not converge; state was not advanced")
    new,d=finish(s,i,f,prep,c)
    if not bool(jnp.all(d[:,21]>0)) or not bool(jnp.all(jnp.isfinite(new))):
        raise RuntimeError("CLM JAX water withdrawal or resulting state is invalid")
    return (new,i,d,unpack_canopy_energy(c)) if return_canopy else (new,i,d)

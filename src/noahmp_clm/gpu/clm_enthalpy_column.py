"""Native extensive CLM development column, precipitation through drainage.

This preserves the current NumPy liquid-canopy profile. Canopy intercepted
ice and flux-feedback MOST are separate pending physical development.
"""
from functools import lru_cache
from dataclasses import fields
import numpy as np
from .canopy_energy import _sum
from .canopy_surface import pack_canopy_surface,run_canopy_surface_jax
from .snow_mass import validate_snow_transfer,snow_transfer_kernel_jax,snow_phase_jax
from .soil_transport import validate_soil_transport,soil_transport_kernel_jax,compact_soil_parameters
from .snow_evolution import fresh_snow_jax
from .enthalpy import invert_soil_enthalpy_jax

OPTION_NAMES=('OptSoilSupercoolWater','OptRainSnowPartition','OptSoilWaterTranspiration',
    'OptGroundResistanceEvap','OptSurfaceDrag','OptRunoffSurface','OptRunoffSubsurface','OptSoilPermeabilityFrozen')
SITE_NAMES=('lai','fveg','z0m','albedo','emiss','terrain_index','groundwater_depth',
    'co2_ppm','cosine_zenith','diffuse_fraction','leaf_dimension','nitrogen_factor')
SURFACE_FIELDS=('et_kgm2','runoff_kgm2','runoff_enthalpy_jm2','atmospheric_energy_jm2',
    'advected_enthalpy_jm2','gpp','h','le','ground_heat','rnet','leaf_respiration',
    'snow_drainage_kgm2','canopy_temperature','canopy_humidity','surface_temperature')
DIAGNOSTICS=('ts','rnet','h','le','g','et_mm','prcp_mm','runoff_mm','runoff_surface_mm',
    'runoff_deep_mm','swe','canw','gpp_umol_m2s','gpp_gcm2','leaf_respiration_umol_m2s',
    'canopy_temperature','canopy_humidity','flux_surface_temperature',
    'precipitation_enthalpy_jm2','runoff_enthalpy_jm2','advected_enthalpy_jm2',
    'atmospheric_energy_jm2','water_residual_kgm2','energy_residual_jm2','snow_drainage_kgm2','valid')


def pack_clm_enthalpy_columns(states,*,dtype=np.float64,co2_ppm=420.,cosine_zenith=.5,
                              diffuse_fraction=.5,leaf_dimension=.04,nitrogen_factor=1.):
    """Encode independent states as (snow, soil, parameters, options, site,
    canopy/surface storage, roots). Five padded snow and four soil layers;
    soil thickness remains an input, not the released fixed-depth ABI.
    """
    from ..clm_state import clm_options
    from ..physics.canopy_energy import CanopyExchange
    states=list(states);count=len(states);baseline=clm_options()
    extras=np.stack([np.broadcast_to(value,(count,)) for value in
        (co2_ppm,cosine_zenith,diffuse_fraction,leaf_dimension,nitrogen_factor)],axis=1)
    rows=[]
    for k,state in enumerate(states):
        if state.bgc_state is not None:
            raise NotImplementedError('BGC requires an extended native entry retaining all plant/soil C/N stores')
        if state.canopy_condensate is not None:
            raise NotImplementedError('Use pack_clm_layered_columns for the canopy phase profile; aggregate packing would lose it')
        for field in fields(state.options):
            if field.name not in OPTION_NAMES and getattr(state.options,field.name)!=getattr(baseline,field.name):
                raise NotImplementedError(f'{field.name} is outside the CLM enthalpy profile')
        if len(state.ground.water)!=4:raise ValueError('Native CLM column requires four soil layers')
        # Encoding only: no precipitation, drag, phase or exchange physics.
        e=CanopyExchange(280.,.005,101325.,.02,.004,.03,280.,0.)
        c,g,p,o,_,_,_,_=pack_canopy_surface(state.snow,state.ground,1.,np.zeros(10),0.,0.,e,dtype=dtype)
        o=np.array([o[0],*[getattr(state.options,name) for name in OPTION_NAMES]],np.int32)
        site=np.asarray([*[getattr(state,name) for name in SITE_NAMES[:7]],*extras[k]],dtype)
        storage=np.asarray([state.canopy_water_kgm2,state.canopy_enthalpy_jm2,state.surface_temperature],dtype)
        rows.append((c,g,p,o,site,storage,np.asarray(state.root_frac,dtype)))
    if not rows:return tuple(np.empty((0,*shape),np.int32 if k==3 else dtype) for k,shape in
        enumerate(((5,9),(4,3),(7,),(9,),(12,),(3,),(4,))))
    return tuple(np.stack([row[k] for row in rows]) for k in range(7))


def validate_column(arrays,forcing,deposition,xp,*,allow_vic=False):
    c,g,p,o,site,storage,roots=arrays;n=len(c)
    if (c.shape!=(n,5,9) or g.shape!=(n,4,3) or p.shape!=(n,7) or o.shape!=(n,9)
        or site.shape!=(n,12) or storage.shape!=(n,3) or roots.shape!=(n,4)
        or forcing.shape!=(n,8) or deposition.shape!=(n,5)):raise ValueError('Invalid extensive CLM batch shapes')
    if o.dtype.kind not in 'iu' or any(x.dtype!=c.dtype for x in (g,p,site,storage,roots,forcing,deposition)):
        raise ValueError('CLM column requires matching float arrays and integer options')
    if not all(bool(xp.all(xp.isfinite(x))) for x in (p,site,storage,roots,forcing,deposition)):
        raise ValueError('Nonfinite extensive CLM forcing or site')
    zero=xp.zeros(n,dtype=xp.int32)
    validate_snow_transfer(c,o[:,0],zero,xp.zeros((n,16),dtype=c.dtype),xp)
    soil_options=xp.stack((zero,o[:,1],o[:,8]),axis=1)
    soil_inputs=xp.concatenate((xp.zeros((n,2),dtype=c.dtype),forcing[:,:1],xp.zeros((n,4),dtype=c.dtype)),axis=1)
    validate_soil_transport(g,compact_soil_parameters(p,xp),soil_options,soil_inputs,xp)
    maxopt=xp.asarray([5,2,5,3,4,2,6 if allow_vic else 4,6 if allow_vic else 4,2])
    if bool(xp.any((o<xp.asarray([0,1,1,1,1,1,1,1,1]))|(o>maxopt))) or bool(xp.any(o[:,2]==4)):
        raise NotImplementedError('Unsupported CLM column option')
    if bool(xp.any(o[:,6:8]==5)) or bool(xp.any(
        xp.any(o[:,6:8]==6,axis=1)&xp.any((o[:,6:8]!=3)&(o[:,6:8]!=6),axis=1))):
        raise NotImplementedError('VIC supports explicit combinations of runoff options 3 and 6')
    bad=((site[:,0]<0.)|(site[:,1]<0.)|(site[:,1]>1.)|(site[:,2]<=0.)|
        (site[:,3]<0.)|(site[:,3]>1.)|(site[:,4]<=0.)|(site[:,4]>1.)|(site[:,7]<=0.)|
        (site[:,8]<=0.)|(site[:,8]>1.)|(site[:,9]<0.)|(site[:,9]>1.)|(site[:,10]<=0.)|
        (site[:,11]<0.)|(site[:,11]>1.)|(storage[:,0]<0.)|((storage[:,0]==0.)&(storage[:,1]!=0.))|
        (storage[:,2]<200.)|(storage[:,2]>350.)|(p[:,4]<0.)|(p[:,4]>1.)|(p[:,5]<0.)|(p[:,5]>p[:,2])|
        (forcing[:,0]<=0.)|(forcing[:,1]<200.)|(forcing[:,1]>350.)|(forcing[:,2]<0.)|(forcing[:,2]>=.2)|
        (forcing[:,3]<0.)|(forcing[:,4]<30000.)|((o[:,2]==5)&(forcing[:,4]>120000.))|xp.any(forcing[:,5:]<0.,axis=1))
    if bool(xp.any(bad)) or bool(xp.any(roots<0.)) or bool(xp.any(deposition<0.)):
        raise ValueError('Invalid extensive CLM state, forcing or site')


def route_water_jax(c,g,n,p,phase,dt,mass,heat,vic=None):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    def into_snow():
        # Carry the authoritative supplied H directly; reconstructing a rain
        # temperature and then H would add a redundant cancellation round trip.
        pack=c.at[0,0].set(s(c[0,0]+mass)).at[0,1].set(s(c[0,1]+heat))
        return pack,g,j.zeros(2,c.dtype),j.asarray(True)
    def into_soil():
        u=j.concatenate((j.stack((mass,heat,dt)),j.zeros(4,c.dtype)))
        out=soil_transport_kernel_jax(g,compact_soil_parameters(p,j),j.stack((j.asarray(0),phase,j.asarray(1))),u,vic)
        return c,out[:12].reshape(4,3),out[26:28],out[46]>0.
    return lax.cond(mass>0.,lambda:lax.cond(n>0,into_snow,into_soil),
        lambda:(c,g,j.zeros(2,c.dtype),j.asarray(True)))


def collapse_molten_snow_jax(c,g,n,p,phase,dt,vic=None):
    """Route a wholly liquid pack to soil, retaining authoritative M/H/tracers."""
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    def collapse():
        mass=_sum(c[:,0]);heat=_sum(c[:,1])
        tracers=j.stack([_sum(c[:,k]) for k in range(4,9)])
        u=j.concatenate((j.stack((mass,heat,dt)),j.zeros(4,c.dtype)))
        out=soil_transport_kernel_jax(g,compact_soil_parameters(p,j),j.stack((j.asarray(0),phase,j.asarray(1))),u,vic)
        empty=j.zeros_like(c).at[:,3].set(54.5)
        return empty,out[:12].reshape(4,3),j.asarray(0,j.int32),out[26:28],tracers,out[46]>0.
    molten=(n>0)&j.all(j.where(j.arange(5)<n,c[:,1]>=s(334000.*c[:,0]),True))
    return lax.cond(molten,collapse,lambda:(c,g,n,j.zeros(2,c.dtype),j.zeros(5,c.dtype),j.asarray(True)))


def prepare_column_jax(c,g,p,o,site,storage,roots,forcing,vic=None):
    import jax.numpy as j
    from jax import lax
    from . import jax_core as core
    s=lax.optimization_barrier;row=lambda *x:j.asarray(x,c.dtype);dt,tair,qair,wind,ps,sw,lw,prcp=forcing
    n,phase=o[:2];capacity=s(s(.5*site[1])*site[0]);mass,heat=storage[:2]
    total=s(prcp*dt);snow_m=s(total*core._snow_fraction(tair,qair,ps,o[2]));rain=s(s(total-snow_m)*1000.);snowfall=s(snow_m*1000.)
    rain_h=s(334000.+s(4180.*s(tair-273.15)));ice_h=s(2100.*s(j.minimum(tair,273.15)-273.15))
    precipitation_h=s(s(rain*rain_h)+s(snowfall*ice_h))
    drip=j.maximum(0.,s(mass-capacity));drip_h=s(s(drip*heat)/j.where(mass>0.,mass,1.))
    mass=s(mass-drip);heat=s(heat-drip_h)
    c,g,runoff,valid=route_water_jax(c,g,n,p,phase,dt,drip,drip_h)
    interception=j.minimum(s(site[1]*rain),j.maximum(0.,s(capacity-mass)))
    mass=s(mass+interception);heat=s(heat+s(interception*rain_h))
    def snow():
        fresh=fresh_snow_jax(s(tair-273.15),wind)
        inputs=j.concatenate((row(snowfall,s(j.minimum(tair,273.15)-273.15),fresh[0],fresh[1],.033,.05,.2,.02,.02,.01,.01),j.zeros(5,c.dtype)))
        out=snow_transfer_kernel_jax(c,n,j.asarray(1),inputs)
        return out[:45].reshape(5,9),out[45].astype(j.int32)
    c,n=lax.cond(snowfall>0.,snow,lambda:(c,n))
    through=s(rain-interception)
    c,g,flow,good=route_water_jax(c,g,n,p,phase,dt,through,s(through*rain_h));runoff=s(runoff+flow);valid=valid&good
    c,g,n,flow,impurity,good=collapse_molten_snow_jax(c,g,n,p,phase,dt,vic)
    runoff=s(runoff+flow);valid=valid&good
    inv=invert_soil_enthalpy_jax(g[:,0],g[:,1],g[:,2],p[0],p[1],p[6],phase)
    liquid=inv[3]/s(1000.*g[:,2])
    stress,_=core._soil_stress(liquid,roots,p[0],p[5],p[2],p[1],o[3])
    velocity=core._exchange_velocity(wind,tair,qair,ps,site[2],storage[2],o[5])
    z0=j.clip(site[2],1e-5,5.);ustar=s(s(.4*j.maximum(wind,.3))/j.log(10./z0))
    leaf_velocity=s(.01*j.sqrt(ustar/site[10]))
    coefficient=s((.4/.13)*j.power(s(s(z0*ustar)/1.5e-5),-.45));decay=j.exp(-site[0])
    ground_velocity=s(s(s(coefficient*decay)+s(.004*s(1.-decay)))*ustar)
    resistance=core._ground_resistance(liquid[0],g[0,2],p[0],p[5],p[1],o[4])
    resistance=j.where((o[4]==4)&(n>0),1.,resistance)
    wet=j.where(capacity>0.,j.minimum(1.,j.power(mass/j.maximum(capacity,1e-30),2./3.)),0.)
    u=row(tair,qair,ps,velocity,ground_velocity,leaf_velocity,s(inv[0][0]+273.15),0.,site[7],resistance,0.,
        sw,lw,site[1],wet,mass/dt,j.inf,j.inf,site[4],site[3],site[3],site[9],site[8])
    area=j.full(10,site[0]/10.,c.dtype);factors=j.broadcast_to(j.stack((site[11],stress)),(10,2))
    v=j.concatenate((row(dt,mass,heat),roots));prep=j.concatenate((row(s(rain+snowfall),precipitation_h),runoff,row(capacity)))
    return c,g,j.stack((n,phase)),area,u,factors,v,prep,valid&j.all(inv[4]),impurity


def runoff_jax(incoming,g,p,site,surface_option,subsurface_option):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;smc=g[:,0]/s(1000.*g[:,2])
    def scheme(amount,option):
        terrain=j.exp(-site[5]);deficit=_sum(s(j.maximum(0.,s(p[2]-smc))*g[:,2]))
        deficit=j.where(option==2,j.maximum(deficit,s(s(j.maximum(0.,site[6])*p[0])*terrain)),deficit)
        deficit=j.maximum(0.,s(deficit*terrain))
        wet=j.clip(s(j.exp(site[5])*j.exp(-j.maximum(0.,site[6])/2.)),0.,1.)
        top=s(j.maximum(0.,s(amount-deficit))*wet)
        saturation=_sum(j.clip(smc/j.maximum(p[0],1e-6),0.,1.))*.25
        bats=s(amount*s(s(saturation*saturation)*saturation))
        out=j.where(option==3,0.,j.where(option==4,bats,top));out=j.clip(out,0.,amount)
        return out,s(amount-out)
    surf,sub=scheme(incoming,surface_option);second,deep=scheme(sub,subsurface_option)
    surf=j.where(surface_option==subsurface_option,surf,s(surf+second));surf=j.clip(surf,0.,incoming)
    return j.stack((surf,s(incoming-surf)))


def finish_column_jax(old_c,old_g,p,o,site,old_store,forcing,deposition,prep,c,g,n,canopy,flux,impurity,vic=None,*,closed_richards=False):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;row=lambda *x:j.asarray(x,c.dtype);dt=forcing[0];phase=o[1]
    mass,heat=canopy;runoff=s(prep[2:4]+flux[1:3]);valid=j.asarray(True)
    drip=j.maximum(0.,s(mass-prep[4]));drip_h=s(s(drip*heat)/j.where(mass>0.,mass,1.))
    mass=s(mass-drip);heat=s(heat-drip_h)
    def overflow():
        pack,soil,out,ok=route_water_jax(c,g,n,p,phase,dt,drip,drip_h,vic)
        drained=snow_transfer_kernel_jax(pack,n,j.asarray(3),row(0.,0.,100.,54.5,.033,.05,.2,.02,.02,.01,.01,0.,0.,0.,0.,0.))
        u=j.concatenate((row(drained[53],drained[54],dt),j.zeros(4,c.dtype)))
        admitted=soil_transport_kernel_jax(soil,compact_soil_parameters(p,j),j.stack((j.asarray(0),phase,j.asarray(1))),u,vic)
        out=s(s(out+drained[60:62])+admitted[26:28]);tracer=s(drained[62:67]+drained[55:60])
        return drained[:45].reshape(5,9),admitted[:12].reshape(4,3),drained[45].astype(j.int32),out,tracer,ok&(admitted[46]>0.)
    c,g,n,out,tracer,valid=lax.cond(drip>0.,overflow,lambda:(c,g,n,j.zeros(2,c.dtype),j.zeros(5,c.dtype),valid))
    runoff=s(runoff+out);impurity=s(impurity+tracer)
    # Zero-input admission diagnoses phase and copies W/H without gravity
    # transfer. Explicit Richards transport follows the finish stage.
    mode=0 if closed_richards else 2
    drained=soil_transport_kernel_jax(g,compact_soil_parameters(p,j),j.stack((j.asarray(mode),phase,o[8])),j.concatenate((row(0.,0.,dt),j.zeros(4,c.dtype))),vic)
    g=drained[:12].reshape(4,3);runoff=s(runoff+drained[44:46]);valid=valid&(drained[46]>0.)
    addition=s(deposition*dt);c=c.at[0,4:].set(j.where(n>0,s(c[0,4:]+addition),c[0,4:]));impurity=s(impurity+j.where(n==0,addition,0.))
    temp=s(j.where(n>0,snow_phase_jax(c[0,0],c[0,1])[0],drained[12])+273.15)
    store=row(mass,heat,temp);options=o.at[0].set(n)
    classified=s(runoff_jax(runoff[0]/1000.,g,p,site,o[6],o[7])*1000.)
    if vic is not None:classified=j.where(j.any(vic[:2]>0.),row(s(runoff[0]-drained[44]),drained[44]),classified)
    water0=s(s(_sum(old_c[:,0])+_sum(old_g[:,0]))+old_store[0]);water1=s(s(_sum(c[:,0])+_sum(g[:,0]))+mass)
    h0=s(s(_sum(old_c[:,1])+_sum(old_g[:,1]))+old_store[1]);h1=s(s(_sum(c[:,1])+_sum(g[:,1]))+heat)
    water_res=s(s(s(s(water1-water0)-prep[0])+flux[0])+runoff[0])
    energy_res=s(s(s(s(s(h1-h0)-prep[1])-flux[3])+flux[4])+runoff[1])
    diag=row(temp,flux[9],flux[6],flux[7],flux[8],flux[0],prep[0],runoff[0],classified[0],classified[1],
        _sum(c[:,0])/1000.,mass/1000.,flux[5],s(s(flux[5]*12.0107e-6)*dt),flux[10],flux[12],flux[13],flux[14],
        prep[1],runoff[1],flux[4],flux[3],water_res,energy_res,flux[11],valid.astype(c.dtype))
    return c,g,options,store,diag,impurity


@lru_cache(maxsize=2)
def _stages(closed_richards=False):
    import jax
    finish=finish_column_jax if not closed_richards else lambda *a:finish_column_jax(*a,closed_richards=True)
    return jax.jit(jax.vmap(prepare_column_jax)),jax.jit(jax.vmap(finish))


def unpack_column(c,g,o,store,diag,impurity,surface=None):
    result=dict(snow_cells=c,soil_cells=g,options=o,storage=store,
        diagnostics={name:diag[:,k] for k,name in enumerate(DIAGNOSTICS)},impurity_output_kgm2=impurity)
    if surface is not None:result['surface']=surface
    return result


def step_clm_enthalpy_columns_jax(c,g,p,o,site,storage,roots,forcing,*,deposition=None,leaf_parameters=None,return_surface=False,soil_hydrology=None):
    """Advance every column on device; forcing is (dt,T,q,wind,p,SW,LW,rain).

    Precipitation has m/s units; optional five-species deposition is kg/m2/s.
    Next step uses returned snow_cells, soil_cells, options and storage with
    the existing parameters, site and roots. return_surface exposes the
    nonlinear exchange/heat diagnostics before final drainage/deposition.
    """
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(c,g,p,o,site,storage,roots)));forcing=j.asarray(forcing)
    deposition=j.zeros((len(c),5),arrays[0].dtype) if deposition is None else j.asarray(deposition)
    validate_column(arrays,forcing,deposition,j)
    from .clm_richards_column import resolve_column_richards,attach_column_richards
    hydrology=resolve_column_richards(soil_hydrology,arrays[3],j);prepare,finish=_stages(hydrology is not None)
    pc,pg,po,a,u,f,v,prep,valid,pre_impurity=prepare(*arrays,forcing)
    if not bool(j.all(valid)):raise RuntimeError('CLM precipitation preparation failed')
    surface=run_canopy_surface_jax(pc,pg,arrays[2],po,a,u,f,v,parameters=leaf_parameters)
    flux=j.stack([surface[key] for key in SURFACE_FIELDS],axis=1)
    canopy=j.stack((surface['canopy_water_after_kgm2'],surface['canopy_enthalpy_after_jm2']),axis=1)
    out=finish(arrays[0],arrays[1],arrays[2],arrays[3],arrays[4],arrays[5],forcing,deposition,prep,
        surface['snow_cells'],surface['soil_cells'],surface['snow_counts'],canopy,flux,surface['impurity_output_kgm2']+pre_impurity)
    if not bool(j.all(out[4][:,25]>0.)) or not all(bool(j.all(j.isfinite(x))) for x in out):
        raise RuntimeError('CLM post-surface water/heat transfer failed')
    result=unpack_column(*out,surface=surface if return_surface else None)
    return attach_column_richards(result,arrays[2],arrays[3],forcing,arrays[0],arrays[1],arrays[5],hydrology,'jax')

"""Native ten-layer condensate interception and phase-specific vapour transfer."""
from functools import lru_cache
import numpy as np
from .canopy_energy import _sum

LEDGER=('intercepted_rain_kgm2','intercepted_snow_kgm2','through_rain_kgm2','through_snow_kgm2',
    'ground_liquid_kgm2','ground_ice_kgm2','ground_liquid_enthalpy_jm2','ground_ice_enthalpy_jm2',
    'precipitation_enthalpy_jm2','water_residual_kgm2','energy_residual_jm2','valid')
LAYER_FIELDS=('temperature','ice','liquid','wet_fraction','dry_fraction','snow_fraction',
    'canopy_drip_kgm2','canopy_snowfall_kgm2','canopy_unloading_kgm2',
    'mass_kgm2','ice_kgm2','liquid_kgm2','advected_enthalpy_jm2','latent_energy_jm2','unmet_evaporation_kgm2')


def pack_canopy_interception(state,rain,snow,air_temperature,dt,leaf_area,*,stem_area=0.,vegetated_fraction=1.,wind=0.,unloading_threshold=270.15,dtype=np.float64):
    if state.mass.shape!=(10,):raise ValueError('Device canopy condensate requires ten layers')
    c=np.column_stack((state.mass,state.enthalpy)).astype(dtype)
    area=np.stack((np.broadcast_to(leaf_area,(10,)),np.broadcast_to(stem_area,(10,))),axis=1).astype(dtype)
    u=np.array([rain,snow,air_temperature,dt,vegetated_fraction,wind,unloading_threshold],dtype)
    return c,area,u,np.zeros((10,2),dtype),np.int32(0)


def pack_canopy_vapor(state,amount,leaf_temperature,*,dtype=np.float64):
    """Explicit iterated leaf temperature is required; packing does no physics."""
    if state.mass.shape!=(10,):raise ValueError('Device canopy condensate requires ten layers')
    c=np.column_stack((state.mass,state.enthalpy)).astype(dtype)
    vapor=np.stack((np.broadcast_to(amount,(10,)),np.broadcast_to(leaf_temperature,(10,))),axis=1).astype(dtype)
    return c,np.zeros((10,2),dtype),np.zeros(7,dtype),vapor,np.int32(1)


def validate_canopy_water(c,area,u,v,mode,xp):
    n=len(c)
    if c.shape!=(n,10,2) or area.shape!=(n,10,2) or u.shape!=(n,7) or v.shape!=(n,10,2) or mode.shape!=(n,):
        raise ValueError('Invalid canopy water batch shapes')
    if c.dtype not in (xp.float32,xp.float64) or mode.dtype.kind not in 'iu' or any(a.dtype!=c.dtype for a in (area,u,v)):
        raise ValueError('Canopy water requires matching floating arrays and integer modes')
    if not all(bool(xp.all(xp.isfinite(a))) for a in (c,area,u,v)) or bool(xp.any((mode<0)|(mode>1))):
        raise ValueError('Invalid canopy water inputs or mode')
    mass,heat=c[:,:,0],c[:,:,1];eps=xp.finfo(c.dtype).eps
    lower=mass*2100.*(150.-273.15);upper=mass*(334000.+4180.*(400.-273.15));rounding=8.*eps*xp.maximum(abs(heat),1.)
    if bool(xp.any((mass<0.)|((mass==0.)&(heat!=0.))|(heat<lower-rounding)|(heat>upper+rounding))) or bool(xp.any(area<0.)):
        raise ValueError('Invalid canopy mass, enthalpy or exposed area')
    bad=((u[:,0]<0.)|(u[:,1]<0.)|(u[:,2]<200.)|(u[:,2]>350.)|(u[:,3]<=0.)|
        (u[:,4]<0.)|(u[:,4]>1.)|(u[:,5]<0.)|(u[:,6]<200.)|(u[:,6]>273.15))
    if bool(xp.any((mode==0)&bad)) or bool(xp.any((mode[:,None]==1)&((v[:,:,1]<200.)|(v[:,:,1]>350.)))):
        raise ValueError('Invalid canopy precipitation or vapour forcing')


def canopy_phase_jax(mass,heat):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;den=j.where(mass>0.,mass,1.)
    liquid=j.clip(heat/334000.,0.,mass);ice=s(mass-liquid)
    offset=j.where(heat<0.,heat/s(2100.*den),j.where(heat>s(mass*334000.),s(heat-s(mass*334000.))/s(4180.*den),0.))
    return s(273.15+j.where(mass>0.,offset,0.)),ice,liquid


def canopy_water_kernel_jax(c,area,u,v,mode):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;mass,heat=c[:,0],c[:,1];zero=j.zeros(10,c.dtype)
    def interception():
        a=s(area[:,0]+area[:,1]);total=_sum(a);weights=a/j.where(total>0.,total,1.)
        # tanh GPU intrinsics differ by an ulp, amplified when warm rain mixes
        # with negative ice enthalpy. This equivalent form is stable at LAI=0.
        em=s(j.expm1(s(-2.*total)));rain_fraction=s((-em)/s(2.+em))
        fi=s(u[4]*rain_fraction);fs=s(u[4]*s(-j.expm1(s(-.5*total))))
        ri=s(fi*u[0]);si=s(fs*u[1]);rh=s(334000.+s(4180.*s(u[2]-273.15)));ih=s(2100.*s(j.minimum(u[2],273.15)-273.15))
        ph=s(s(u[0]*rh)+s(u[1]*ih));m=s(mass+s(weights*s(ri+si)));h=s(heat+s(weights*s(s(ri*rh)+s(si*ih))))
        _,ice,liquid=canopy_phase_jax(m,h);lh=j.where(h>0.,h,0.);hi=s(h-lh)
        cap=s(s(.1*u[4])*a);icap=s(s(6.*u[4])*a)
        kept_liquid=j.minimum(liquid,cap);kept=j.minimum(ice,icap);drip=s(liquid-kept_liquid);fall=s(ice-kept)
        rate=s(u[5]/1.56e5+j.maximum(0.,s(u[2]-u[6]))/1.87e5);unload=s(kept*j.minimum(1.,s(u[3]*rate)));out=s(fall+unload)
        left=s(kept-unload);hl=s((kept_liquid/j.where(liquid>0.,liquid,1.))*lh);hs=s((left/j.where(ice>0.,ice,1.))*hi)
        dh=s(lh-hl);sh=s(hi-hs);newm=s(kept_liquid+left);newh=s(hl+hs)
        tr=s(u[0]-ri);ts=s(u[1]-si);gm=s(tr+_sum(drip));gi=s(ts+_sum(out));gh=s(s(tr*rh)+_sum(dh));gih=s(s(ts*ih)+_sum(sh))
        water=s(s(s(s(s(_sum(newm)-_sum(mass))-u[0])-u[1])+gm)+gi)
        energy=s(s(s(s(_sum(newh)-_sum(heat))-ph)+gh)+gih)
        _,ni,_=canopy_phase_jax(newm,newh)
        wet=j.where(cap>0.,j.minimum(1.,j.power(newm/j.where(cap>0.,cap,1.),2./3.)),0.)
        dry=s(s(1.-wet)*(area[:,0]/j.where(a>0.,a,1.)))
        snow=j.where(icap>0.,j.minimum(1.,j.power(ni/j.where(icap>0.,icap,1.),.15)),0.)
        fields=j.stack((wet,dry,snow,drip,fall,unload,*([zero]*6)))
        ledger=j.asarray([ri,si,tr,ts,gm,gi,gh,gih,ph,water,energy,1.],c.dtype)
        return newm,newh,fields,ledger
    def vapor():
        _,ice,liquid=canopy_phase_jax(mass,heat);cold=v[:,1]<=273.15;available=j.where(cold,ice,liquid)
        loss=j.minimum(j.maximum(v[:,0],0.),available);remaining=s(available-loss)
        deposit=j.maximum(-v[:,0],0.);di=j.where(cold,deposit,0.);dl=s(deposit-di)
        oi=s(j.where(cold,loss,0.)-di);ol=s(j.where(cold,0.,loss)-dl)
        source=j.where(cold,j.minimum(heat,0.),j.maximum(heat,0.))
        retained=s((remaining/j.where(available>0.,available,1.))*source)
        added=s(s(s(di*2100.)*s(v[:,1]-273.15))+s(dl*s(334000.+s(4180.*s(v[:,1]-273.15)))))
        h=s(s(source-retained)-added);actual=s(loss-deposit)
        newm=s(s(remaining+j.where(cold,liquid,ice))+deposit);newh=s(s(s(heat-source)+retained)+added)
        latent=s(s(2.5e6*s(oi+ol))+s(334000.*oi));unmet=j.maximum(s(v[:,0]-loss),0.)
        fields=j.stack((*([zero]*6),actual,oi,ol,h,latent,unmet))
        ledger=j.zeros(12,c.dtype).at[9].set(s(s(_sum(newm)-_sum(mass))+_sum(actual))).at[10].set(s(s(_sum(newh)-_sum(heat))+_sum(h))).at[11].set(1.)
        return newm,newh,fields,ledger
    m,h,fields,ledger=lax.cond(mode==0,interception,vapor);t,ice,liquid=canopy_phase_jax(m,h)
    return j.concatenate((j.stack((m,h),axis=1).reshape(-1),t,ice,liquid,fields.reshape(-1),ledger))


@lru_cache(maxsize=1)
def _kernel():
    import jax
    return jax.jit(jax.vmap(canopy_water_kernel_jax))


def unpack_canopy_water(out):
    result=dict(cells=out[:,:20].reshape((-1,10,2)))
    for k,key in enumerate(LAYER_FIELDS):result[key]=out[:,20+10*k:30+10*k]
    result['ledger']={key:out[:,170+k] for k,key in enumerate(LEDGER)}
    return result


def run_canopy_water_jax(c,area,u,v,mode):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(c,area,u,v,mode)));validate_canopy_water(*arrays,j)
    out=_kernel()(*arrays)
    if not bool(j.all(j.isfinite(out))):raise RuntimeError('Nonfinite canopy water output')
    return unpack_canopy_water(out)

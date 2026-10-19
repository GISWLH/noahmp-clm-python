"""Device canopy/SNICAR multiple reflection with distinct direct albedos."""
from functools import lru_cache
import numpy as np
from .snicar import (pack_snow_optics,validate_snow_optics,snow_shortwave_kernel_jax,
    assemble_snow_output_jax,unpack_snow_shortwave,_jax_tables)
from .canopy_radiation import shortwave_kernel_jax


def pack_canopy_snow_case(area,pack,swdown,*,vegetated_fraction=1.,diffuse_fraction=.5,
                          cosine_zenith=.5,ground_albedo=(.15,.3),dtype=np.float64):
    cells,count,_=pack_snow_optics(pack,[0.,0.],[0.,0.],dtype=dtype)
    return np.asarray(area,dtype),cells,count,np.asarray(
        [swdown,vegetated_fraction,diffuse_fraction,cosine_zenith,*ground_albedo],dtype)


def validate_canopy_snow(area,cells,counts,inputs,xp):
    n=len(cells)
    if area.shape!=(n,10) or inputs.shape!=(n,6):raise ValueError('Canopy/snow shapes require (n,10), (n,6)')
    if area.dtype!=cells.dtype or inputs.dtype!=cells.dtype:raise ValueError('Canopy/snow floats must share dtype')
    if not all(bool(xp.all(xp.isfinite(v))) for v in (area,inputs)) or bool(xp.any(area<0)):
        raise ValueError('Invalid leaf area or canopy/snow boundary')
    if bool(xp.any((inputs[:,0]<0)|(inputs[:,1]<0)|(inputs[:,1]>1)|(inputs[:,2]<0)|(inputs[:,2]>1))):
        raise ValueError('Invalid shortwave or vegetation/diffuse fraction')
    snow_boundary=xp.concatenate((xp.zeros((n,4),cells.dtype),inputs[:,4:6],inputs[:,3:4]),axis=1)
    validate_snow_optics(cells,counts,snow_boundary,xp)


def canopy_snow_kernel_jax(area,cells,count,inputs,ice_table,impurity_table,*,track_absorption=False,
                           photosynthetic_leaf_area=None,canopy_optics=None):
    import jax
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    sw,fv,diff,mu,agv,agn=inputs
    unit_boundary=j.stack((j.ones_like(sw),j.ones_like(sw),j.ones_like(sw),j.ones_like(sw),agv,agn,mu))
    unit=snow_shortwave_kernel_jax(cells,count,unit_boundary,ice_table,impurity_table)
    response=unit[50:310].reshape(2,5,26)
    def sum4(v):return lax.fori_loop(1,5,lambda i,x:s(x+v[i]),j.zeros_like(v[0]))
    ad=j.stack((response[0,0,23],sum4(response[0,:,23])))
    af=j.stack((response[1,0,23],sum4(response[1,:,23])))
    beam=s(s(.5*sw)*s(1.-diff));sky=s(s(.5*sw)*diff)
    if canopy_optics is None:
        canopy=jax.vmap(lambda band:shortwave_kernel_jax(area,j.stack((beam,sky,af[band],mu,
            j.where(band==0,.1,.45),j.where(band==0,.05,.25))),ad[band],track_absorption=track_absorption))(j.arange(2))
    else:
        from .canopy_optics import mixture_kernel_jax
        rho,tau,leaf_fraction=mixture_kernel_jax(area,photosynthetic_leaf_area,canopy_optics)
        canopy=jax.vmap(lambda band:shortwave_kernel_jax(area,j.stack((beam,sky,af[band],mu,rho[band,0],tau[band,0])),
            ad[band],track_absorption=track_absorption,layer_optics=(rho[band],tau[band])))(j.arange(2))
    bottom_beam=s(s(fv*canopy[:,20])+s(s(1.-fv)*beam))
    bottom_sky=s(s(fv*canopy[:,31])+s(s(1.-fv)*sky))
    snow_boundary=j.concatenate((bottom_beam,bottom_sky,j.stack((agv,agn,mu))))
    incoming=j.stack((bottom_beam,bottom_sky))[:,j.array([0,1,1,1,1])]
    conditions=s(response*incoming[:,:,None])
    snow=assemble_snow_output_jax(conditions,unit[310:].reshape(2,5,3,5),unit[45:50]>0,snow_boundary)
    leaf=s(fv*s(canopy[0,:10]+canopy[1,:10]))
    bare=s(s(ad*beam)+s(af*sky))
    reflected=s(s(fv*s(canopy[0,43]+canopy[1,43]))+s(s(1.-fv)*s(bare[0]+bare[1])))
    total=lax.fori_loop(0,10,lambda i,x:s(x+leaf[i]),j.zeros_like(sw))
    total_snow=lax.fori_loop(0,5,lambda i,x:s(x+snow[i]),j.zeros_like(sw))
    residual=s(s(s(sw-reflected)-total)-s(total_snow+snow[40]))
    if canopy_optics is None:
        par=j.where(area>0.,s(canopy[0,:10]*4.6)/j.maximum(area,1e-30),0.)
    else:
        leaf_absorbed=s(canopy[:,:10]*leaf_fraction);stem_absorbed=s(canopy[:,:10]-leaf_absorbed)
        par=j.where(photosynthetic_leaf_area>0.,s(leaf_absorbed[0]*4.6)/j.maximum(photosynthetic_leaf_area,1e-30),0.)
    out=j.concatenate((leaf,par,snow[:5],j.stack((snow[40],reflected,residual)),ad,af,canopy.ravel(),snow))
    return out if canopy_optics is None else j.concatenate((out,leaf_absorbed.ravel(),stem_absorbed.ravel()))


def unpack_canopy_snow(out):
    return dict(canopy_absorbed=out[:,:10],absorbed_par=out[:,10:20],snow_absorbed=out[:,20:25],
        soil_absorbed=out[:,25],reflected=out[:,26],residual=out[:,27],snow_direct_albedo=out[:,28:30],
        snow_diffuse_albedo=out[:,30:32],canopy_bands=out[:,32:124].reshape(-1,2,46),
        snow=unpack_snow_shortwave(out[:,124:]))


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(canopy_snow_kernel_jax,in_axes=(0,0,0,0,None,None)))


def run_canopy_snow_jax(area,cells,counts,inputs):
    import jax.numpy as j
    a,c,n,u=map(j.asarray,(area,cells,counts,inputs));validate_canopy_snow(a,c,n,u,j)
    out=_compiled()(a,c,n,u,*_jax_tables(str(c.dtype)))
    if not bool(j.all(j.isfinite(out))):raise RuntimeError('Nonfinite canopy/snow shortwave')
    return unpack_canopy_snow(out)

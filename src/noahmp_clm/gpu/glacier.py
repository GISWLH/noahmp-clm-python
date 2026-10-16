"""Native glacier heat, melt, snow drainage and exposed-ice refreezing.

Packing is the only host physics representation conversion. Kernels consume
five snow and four finite ice stores; all fluxes and phase changes run on device.
"""
from functools import lru_cache
import numpy as np
from ..physics.glacier import GlacierIce,GlacierParameters
from ..physics.snowpack import Snowpack


def pack_glacier_case(snow,ice,dt,top_flux,*,snow_absorbed=None,ice_absorbed=None,
                      top_temperature=None,top_conductance=0.,parameters=GlacierParameters(),dtype=np.float64):
    if not isinstance(snow,Snowpack) or not isinstance(ice,GlacierIce) or not len(ice.mass):
        raise ValueError('Glacier heat requires snow and non-exhausted ice')
    if top_temperature is None and top_conductance!=0.:raise ValueError('Top exchange requires temperature')
    ns=len(snow.mass);ni=len(ice.mass)
    c=np.zeros((5,9),dtype);c[:,3]=54.5
    c[:ns,:4]=np.column_stack((snow.mass,snow.enthalpy,snow.thickness,snow.radius_um))
    c[:ns,4:]=snow.impurities
    g=np.zeros((4,7),dtype);g[:ni,:2]=np.column_stack((ice.mass,ice.enthalpy));g[:ni,2:]=ice.impurities
    q=np.zeros(9,dtype);q[:ns]=0. if snow_absorbed is None else snow_absorbed
    q[5:5+ni]=0. if ice_absorbed is None else ice_absorbed
    p=parameters
    return (c,g,np.array([ns,ni],np.int32),np.array([dt,top_flux,
        0. if top_temperature is None else top_temperature-273.15,top_conductance],dtype),
        np.array([p.thermal_conductivity_wmk,p.geothermal_wm2,p.ice_albedo,p.emissivity],dtype),q)


def validate_glacier(c,g,count,b,p,q,xp):
    from .snow_mass import validate_snow_transfer
    n=len(c)
    if (c.shape!=(n,5,9) or g.shape!=(n,4,7) or count.shape!=(n,2)
        or b.shape!=(n,4) or p.shape!=(n,4) or q.shape!=(n,9)):
        raise ValueError('Invalid glacier batch shapes')
    if (c.dtype not in (xp.float32,xp.float64) or any(v.dtype!=c.dtype for v in (g,b,p,q))
        or count.dtype.kind not in 'iu'):raise ValueError('Glacier requires matching floats and integer counts')
    if not all(bool(xp.all(xp.isfinite(v))) for v in (c,g,b,p,q)):raise ValueError('Nonfinite glacier input')
    inputs=xp.tile(xp.asarray([0.,0.,100.,54.5,.033,.05,.2,.02,.02,.01,.01,0.,0.,0.,0.,0.],dtype=c.dtype),(n,1))
    validate_snow_transfer(c,count[:,0],xp.full(n,3,dtype=xp.int32),inputs,xp)
    index=xp.arange(4)[None,:];active=index<count[:,1,None]
    if bool(xp.any((count[:,1]<1)|(count[:,1]>4)|(b[:,0]<=0.)|(b[:,3]<0.)|
        (p[:,0]<=0.)|(p[:,1]<0.)|(p[:,2]<0.)|(p[:,2]>1.)|(p[:,3]<=0.)|(p[:,3]>1.))):
        raise ValueError('Invalid glacier counts or boundary parameters')
    if bool(xp.any((active&(g[:,:,0]<=0.))|(g[:,:,1]>0.)|
        (g[:,:,1]<2100.*g[:,:,0]*(150.-273.15))|xp.any(g[:,:,2:]<0.,axis=2)|
        (~active&xp.any(g!=0.,axis=2)))):raise ValueError('Invalid glacier ice state or padding')
    solar_active=xp.concatenate((xp.arange(5)[None,:]<count[:,0,None],active),axis=1)
    if bool(xp.any((~solar_active)&(q!=0.))):raise ValueError('Invalid glacier absorbed-energy padding')


def glacier_kernel_jax(c,g,count,b,p,q):
    import jax.numpy as j
    from jax import lax
    from .glacier_heat import glacier_heat_kernel_jax
    from .snow_mass import snow_transfer_kernel_jax
    s=lax.optimization_barrier;ns,ni=count;n=ns+ni;idx=j.arange(9);active=idx<n;is_snow=idx<ns
    sc=c[j.minimum(idx,4)];ic=g[j.clip(idx-ns,0,3)]
    mass=j.where(active,j.where(is_snow,sc[:,0],ic[:,0]),0.)
    h=j.where(active,j.where(is_snow,sc[:,1],ic[:,1]),0.)
    z=j.where(active,j.where(is_snow,sc[:,2],ic[:,0]/917.),1.)
    rho=j.minimum(mass/z,917.)
    ks=s(.023+s(s(s(7.75e-5*rho)+s(s(1.105e-6*rho)*rho))*j.asarray(2.29-.023,c.dtype)))
    k=j.where(active,j.where(is_snow,ks,p[0]),1.)
    solar=j.where(active,j.where(is_snow,q[j.minimum(idx,4)],q[5+j.clip(idx-ns,0,3)]),0.)
    solar=solar.at[n-1].add(p[1])
    cells=j.stack((mass,h,z,k,solar),axis=1)
    thermal=glacier_heat_kernel_jax(cells,n,b)
    hs=thermal[0];heated=c.at[:,1].set(j.where(j.arange(5)<ns,hs[:5],0.))
    u=j.array([0.,0.,100.,54.5,.033,.05,.2,.02,.02,.01,.01,0.,0.,0.,0.,0.],dtype=c.dtype)
    snow=snow_transfer_kernel_jax(heated,ns,j.asarray(3),u)
    # Melt each ice layer. Pack surviving cold stores, exporting liquid H
    # exactly even when the finite layer melts and its water warms above Tf.
    def melt(i,carry):
        out,nout,run=carry;v=g[i];energy=hs[ns+i]
        liquid=j.minimum(v[0],j.maximum(energy,0.)/334000.)
        tracer=s(v[2:]*s(liquid/v[0]));remain=s(v[0]-liquid)
        parcel=j.concatenate((j.stack((liquid,j.maximum(energy,0.))),tracer))
        cell=j.concatenate((j.stack((remain,j.minimum(energy,0.))),s(v[2:]-tracer)))
        out=lax.cond(remain>0.,lambda x:x.at[nout].set(cell),lambda x:x,out)
        return out,nout+(remain>0.).astype(j.int32),s(run+parcel)
    ice,nice,im=lax.fori_loop(0,ni,melt,(j.zeros_like(g),j.asarray(0,j.int32),j.zeros(7,c.dtype)))
    # Snow bottom water can freeze into exposed ice. Entirely melted cells
    # expose the next one; liquid above a surviving cell runs off immediately.
    def mix(carry):
        first,ice,parcel,done=carry;v=s(ice[first]+parcel)
        liquid=j.minimum(v[0],j.maximum(v[1],0.)/334000.)
        tracer=s(v[2:]*s(liquid/v[0]));remain=s(v[0]-liquid)
        next_parcel=j.concatenate((j.stack((liquid,j.maximum(v[1],0.))),tracer))
        cell=j.concatenate((j.stack((remain,j.minimum(v[1],0.))),s(v[2:]-tracer)))
        return first+(remain<=0.).astype(j.int32),ice.at[first].set(cell),next_parcel,remain>0.
    first,ice,parcel,_=lax.while_loop(lambda x:(x[0]<nice)&~x[3]&((x[2][0]>0.)|j.any(x[2][2:]>0.)),
        mix,(j.asarray(0,j.int32),ice,snow[53:60],j.asarray(False)))
    ice=j.where((j.arange(4)<nice-first)[:,None],ice[j.minimum(j.arange(4)+first,3)],0.)
    new_snow=snow[:45].reshape(5,9);run=s(s(im+snow[60:67])+parcel)
    # Serial extensive sums match CUDA operation order; test ledgers are
    # independently recomputed in float64 from actual input/output stores.
    def stores(c,g):
        start=lax.fori_loop(0,5,lambda i,v:s(v+c[i,:2]),j.zeros(2,c.dtype))
        return lax.fori_loop(0,4,lambda i,v:s(v+g[i,:2]),start)
    old=stores(c,g);new=stores(new_snow,ice)
    residual=s(s(new-old)+run[:2]);residual=residual.at[1].set(s(residual[1]-thermal[8][0]))
    abs_h=lax.fori_loop(0,9,lambda i,v:s(v+j.abs(h[i])),j.zeros((),c.dtype))
    eps=j.finfo(c.dtype).eps
    wm=s(s(64.*eps)*j.maximum(s(old[0]+run[0]),1.))
    wh=s(s(s(64.*eps)*j.maximum(s(s(abs_h+j.abs(run[1]))+j.abs(thermal[8][0])),1.))+s(n*thermal[8][5]))
    newice=lax.fori_loop(0,4,lambda i,v:s(v+ice[i,0]),j.zeros((),c.dtype))
    oldice=lax.fori_loop(0,4,lambda i,v:s(v+g[i,0]),j.zeros((),c.dtype))
    refrozen=s(newice-s(oldice-im[0]))
    diag=j.concatenate((run,j.stack((im[0],snow[53],refrozen,s(p[1]*b[0]),residual[0],residual[1],wm,wh))))
    return new_snow,ice,j.stack((snow[45].astype(j.int32),nice-first)),diag,thermal


def unpack_glacier(result):
    from .column_heat import unpack_joint_heat
    snow,ice,count,d,thermal=result
    return dict(snow_cells=snow,ice_cells=ice,counts=count,runoff_kgm2=d[:,0],runoff_enthalpy_jm2=d[:,1],
        runoff_impurities_kgm2=d[:,2:7],ice_melt_kgm2=d[:,7],snow_drainage_kgm2=d[:,8],
        net_refreezing_kgm2=d[:,9],geothermal_input_jm2=d[:,10],water_residual_kgm2=d[:,11],
        energy_residual_jm2=d[:,12],water_bound_kgm2=d[:,13],energy_bound_jm2=d[:,14],
        thermal=unpack_joint_heat(thermal))


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(glacier_kernel_jax))


def run_glacier_jax(c,g,count,b,p,q):
    import jax.numpy as j
    values=tuple(map(j.asarray,(c,g,count,b,p,q)));validate_glacier(*values,j)
    out=unpack_glacier(_compiled()(*values))
    if not bool(j.all(out['thermal']['converged'])):raise RuntimeError('JAX glacier heat failed to converge')
    if bool(j.any((j.abs(out['water_residual_kgm2'])>out['water_bound_kgm2'])|
        (j.abs(out['energy_residual_jm2'])>out['energy_bound_jm2']))):raise RuntimeError('JAX glacier conservation failed')
    return out

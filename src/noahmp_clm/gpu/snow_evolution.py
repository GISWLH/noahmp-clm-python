"""Native snow compaction, table-driven grain aging and deposition.

Modes: 0 compaction, 1 aging, 2 deposition, 3 compaction then aging.
These operators preserve mass/enthalpy; deposition adds only traced aerosol.
"""
from functools import lru_cache
import numpy as np
from .snow_mass import pack_snow_transfer,validate_snow_transfer,snow_phase_jax,snow_regrid_jax


def pack_snow_evolution(pack,mode,dt,*,gradient=0.,ice_before=None,refrozen_mass=None,
                        deposition_rates=(0.,0.,0.,0.,0.),remap=True,
                        air_temperature=273.15,wind10=0.,dtype=np.float64):
    cells,n,_,_=pack_snow_transfer(pack,0,dtype=dtype);n=int(n)
    u=np.zeros(23,dtype);u[0]=dt;u[1:1+n]=np.broadcast_to(gradient,(n,))
    if ice_before is not None:u[6:6+n]=np.broadcast_to(ice_before,(n,))
    if refrozen_mass is not None:u[11:11+n]=np.broadcast_to(refrozen_mass,(n,))
    u[16:21]=deposition_rates;u[21:]=air_temperature-273.15,wind10
    opts=np.array([n,mode,int(ice_before is not None),int(refrozen_mass is not None),int(remap)],np.int32)
    return cells,opts,u


def validate_snow_evolution(c,o,u,xp):
    if o.shape!=(len(c),5) or o.dtype.kind not in 'iu' or u.shape!=(len(c),23) or u.dtype!=c.dtype:
        raise ValueError('Snow evolution requires (n,5) integer options and matching (n,23) inputs')
    validate_snow_transfer(c,o[:,0],xp.zeros(len(c),dtype=xp.int32),xp.zeros((len(c),16),c.dtype),xp)
    if not bool(xp.all(xp.isfinite(u))) or bool(xp.any(u[:,:21]<0)) or bool(xp.any((u[:,21]<-123.15)|(u[:,21]>56.85)|(u[:,22]<0))):
        raise ValueError('Invalid snow evolution step, gradient, history, deposition or forcing')
    if bool(xp.any((o[:,1]<0)|(o[:,1]>3))) or bool(xp.any((o[:,2:]<0)|(o[:,2:]>1))):
        raise ValueError('Invalid snow evolution operation or flag')
    active=xp.arange(5)[None,:]<o[:,0,None];ice=c[:,:,0]-xp.clip(c[:,:,1]/334000.,0.,c[:,:,0])
    if bool(xp.any(active&(o[:,3,None]>0)&(u[:,11:16]>ice+8*xp.finfo(c.dtype).eps*xp.maximum(c[:,:,0],1.)))):
        raise ValueError('Refrozen mass exceeds available ice')
    for start in (1,6,11):
        if bool(xp.any((~active)&(u[:,start:start+5]!=0))):raise ValueError('Snow history padding must be zero')


def aging_table(dtype=np.float64):
    from ..physics.snicar import snow_tables
    t=snow_tables()
    for name,start,step,size in (('T',223.15,5.,11),('dTdz',0.,10.,31),('sno_dns',50.,50.,8)):
        np.testing.assert_allclose(t['aging_'+name],start+step*np.arange(size),rtol=0,atol=1e-12)
    return np.stack([t['aging_'+key] for key in ('tau','kappa','drdsdt0')]).astype(dtype)


def aging_parameters_jax(offset,gradient,density,table):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    coordinates=j.stack((s(j.clip(offset,-50.,0.)+50.)/5.,j.clip(gradient,0.,300.)/10.,s(j.clip(density,50.,400.)-50.)/50.))
    physical=j.stack((s(273.15+offset),gradient,density));spacing=j.array([5.,10.,50.],offset.dtype)
    tolerance=s(s(8.*j.finfo(offset.dtype).eps)*j.maximum(j.abs(physical),1.))/spacing
    nearest=j.rint(coordinates);coordinates=j.where(j.abs(s(coordinates-nearest))<=tolerance,nearest,coordinates)
    low=j.minimum(j.floor(coordinates).astype(j.int32),j.array([9,29,6]));fraction=s(coordinates-low.astype(offset.dtype))
    def corner(i,result):
        hi=j.array([i&1,(i>>1)&1,(i>>2)&1]);index=low+hi
        w=j.where(hi>0,fraction,s(1.-fraction));weight=s(s(w[0]*w[1])*w[2])
        return s(result+s(weight*table[:,index[0],index[1],index[2]]))
    return lax.fori_loop(0,8,corner,j.zeros(3,offset.dtype))


def fresh_snow_jax(offset,wind):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    thermal=j.where(offset>-15.,s(50.+s(1.7*j.power(j.maximum(s(j.minimum(offset,2.)+15.),0.),1.5))),
        s(s(-3.833*offset)-s(s(.0333*offset)*offset)))
    loading=j.where(wind>.1,s(266.861*j.power(s(s(1.+j.tanh(wind/5.))/2.),8.8)),0.)
    return j.stack((j.clip(s(thermal+loading),30.,917.),s(54.5+s(5.*j.clip(s(offset+30.),0.,30.)))))


def snow_evolution_kernel_jax(c,o,u,table):
    import jax
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;n,mode,pre,ref,remap=o;active=j.arange(5)<n
    mass=c[:,0];safe=j.maximum(mass,1e-30);z=j.maximum(c[:,2],1e-30)
    offset,ice,liquid=snow_phase_jax(mass,c[:,1]);rho=mass/z
    c1=j.exp(s(-.046*j.maximum(s(ice/z-175.),0.)));c2=j.where(liquid/z>.01,2.,1.)
    destructive=s(s(s(2.777e-6*c1)*c2)*j.exp(s(.04*offset)))
    over=lax.fori_loop(0,5,lambda i,v:v.at[i+1].set(s(v[i]+mass[i])),j.zeros(6,c.dtype))[1:]
    over=s(over-s(.5*mass))
    viscosity=s(s(s(4.*7.62237e6)*s(rho/450.))*j.exp(s(s(-.1*offset)+s(.023*rho))))/s(1.+s(60.*liquid)/s(1000.*z))
    rate_c=s(destructive+over/j.maximum(viscosity,1e-30))
    volume=s(ice/917.+liquid/1000.);allowed=(s(1.-volume/z)>.001)&(ice>.1)
    factor=j.exp(s(-u[0]*j.where(allowed,rate_c,0.)))
    melt=j.where(u[6:11]>0.,ice/j.maximum(u[6:11],1e-30),1.)
    factor=s(factor*j.where((pre>0)&allowed,j.minimum(1.,melt),1.))
    depth=j.maximum(s(c[:,2]*factor),volume)
    cells=c.at[:,2].set(j.where(active&((mode==0)|(mode==3)),depth,c[:,2]))
    # Match the column: compact, age at the updated density, then remap.
    aging_density=mass/j.maximum(cells[:,2],1e-30)
    do_age=(mode==1)|(mode==3)
    parameters=jax.vmap(aging_parameters_jax,in_axes=(0,0,0,None))(offset,u[1:6],j.maximum(aging_density,1.),table)
    tau,kappa,rate=parameters.T
    def dry(r):
        y=s(s(r-54.5)+tau);p=s(1.+1./kappa);hours=u[0]/7200.
        inc=s(s(j.log(j.maximum(s(s(p*rate)*hours),1e-30))-j.log(y))+j.log(tau/y)/kappa)
        soft=s(j.maximum(inc,0.)+j.log1p(j.exp(-j.abs(inc))))
        increase=s(y*j.expm1(soft/p))
        return j.where(u[0]>0.,s(r+increase),r)
    radius=j.maximum(c[:,3],54.5);radius=dry(radius)
    fraction=j.minimum(.1,liquid/safe)
    wet=s(s(s(422000.*fraction)*fraction)*fraction)/(4.*np.pi)
    radius=j.cbrt(s(s(s(radius*radius)*radius)+s(s(3.*wet)*u[0])))
    radius=dry(radius)
    frozen=j.minimum(u[11:16],ice)/safe
    radius=j.where(ref>0,1./s(s(1.-frozen)/radius+frozen/1000.),radius)
    radius=j.clip(radius,30.,1500.)
    cells=cells.at[:,3].set(j.where(active&do_age,radius,c[:,3]))
    addition=s(u[16:21]*u[0]);bare=j.where((mode==2)&(n==0),addition,0.)
    cells=cells.at[0,4:].set(j.where((mode==2)&(n>0),s(cells[0,4:]+addition),cells[0,4:]))
    cells,nn=lax.cond((remap>0)&((mode==0)|(mode==3))&(n>0),lambda:snow_regrid_jax(j.pad(cells,((0,1),(0,0)))),lambda:(cells,n))
    clipped=active&do_age&((offset<-50.)|(offset>0.)|(u[1:6]>300.)|(aging_density<50.)|(aging_density>400.))
    parameters=j.where((active&do_age)[:,None],parameters,0.)
    return j.concatenate((cells.ravel(),j.array([nn],c.dtype),bare,parameters.ravel(),fresh_snow_jax(u[21],u[22]),clipped.astype(c.dtype)))


def unpack_snow_evolution(out):
    return dict(cells=out[:,:45].reshape(-1,5,9),counts=out[:,45].astype('int32'),bare_deposition=out[:,46:51],
        aging_parameters=out[:,51:66].reshape(-1,5,3),fresh_density=out[:,66],fresh_radius=out[:,67],table_clipped=out[:,68:73]>0)


@lru_cache(maxsize=2)
def _tables(dtype):
    import jax.numpy as j
    return j.asarray(aging_table(np.dtype(dtype)))


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(snow_evolution_kernel_jax,in_axes=(0,0,0,None)))


def run_snow_evolution_jax(cells,options,inputs):
    import jax.numpy as j
    c,o,u=map(j.asarray,(cells,options,inputs));validate_snow_evolution(c,o,u,j)
    out=_compiled()(c,o,u,_tables(str(c.dtype)))
    if not bool(j.all(j.isfinite(out))):raise RuntimeError('Nonfinite snow evolution')
    return unpack_snow_evolution(out)

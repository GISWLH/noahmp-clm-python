"""Native extensive winter surface: canopy, vapour, soil uptake and joint heat.

Precipitation/interception and gravity drainage remain caller operations.
No host physics is used in either backend's nonlinear surface solve.
"""
from functools import lru_cache
import numpy as np
from .canopy_energy import pack_canopy_case,validate_canopy_batch,_sum,canopy_energy_kernel_jax
from .canopy_boundary import unpack_boundary,radiation_inputs
from .snow_mass import pack_snow_transfer,validate_snow_transfer,snow_phase_jax,snow_transfer_kernel_jax
from .soil_transport import validate_soil_transport,soil_transport_kernel_jax
from .enthalpy import invert_soil_enthalpy_jax
from .column_heat import joint_heat_kernel_jax
from .snow_vapor import snow_vapor_kernel_jax
from .snow_evolution import snow_evolution_kernel_jax,_tables as _aging
from .canopy_snow_radiation import canopy_snow_kernel_jax
from .snicar import _jax_tables
from .photosynthesis import pack_leaf_parameters


def pack_canopy_surface(snow,soil,dt,area,swdown,lwdown,exchange,*,canopy_water_kgm2=0.,
                        canopy_enthalpy_jm2=None,root_weights=None,dtype=np.float64,**kwargs):
    """Pack state and forcing only; soil supply and optics are diagnosed on device."""
    reserved={'soil_supply','fixed_ground_temperature','ground_is_ice','ground_shares_soil',
        'snowpack','canopy_evaporation_limit','ground_evaporation_limit','initial_temperatures',
        'tolerance','temperature_tolerance','canopy_storage'}
    if reserved & kwargs.keys():raise ValueError('Surface owns its supply and thermal boundary controls')
    a,u,f,_,_=pack_canopy_case(area,swdown,lwdown,exchange,dtype=dtype,**kwargs)
    c,n,_,_=pack_snow_transfer(snow,0,dtype=dtype)
    g=np.column_stack((soil.water,soil.enthalpy,soil.dz)).astype(dtype)
    p=np.asarray([soil.params.smcmax,soil.params.bexp,soil.params.smcref,soil.params.satdk,
                  soil.params.quartz,soil.params.smcwlt],dtype)
    o=np.asarray([n,soil.freezing_option],np.int32)
    h=canopy_water_kgm2*(334000.+4180.*(exchange.air_temperature-273.15)) if canopy_enthalpy_jm2 is None else canopy_enthalpy_jm2
    v=np.asarray([dt,canopy_water_kgm2,h,*([0.]*4 if root_weights is None else root_weights)],dtype)
    return c,g,p,o,a,u,f,v


def validate_surface(arrays,xp):
    c,g,p,o,a,u,f,v=arrays;n=len(c)
    if (c.shape!=(n,5,9) or g.shape!=(n,4,3) or p.shape!=(n,6) or o.shape!=(n,2)
            or v.shape!=(n,7)):raise ValueError('Invalid surface state/parameter shapes')
    if any(x.dtype!=c.dtype for x in (g,p,a,u,f,v)) or o.dtype.kind not in 'iu':
        raise ValueError('Surface requires matching floats and integer options')
    zero=xp.zeros(n,dtype=xp.int32)
    su=xp.zeros((n,16),dtype=c.dtype)
    validate_snow_transfer(c,o[:,0],zero,su,xp)
    so=xp.stack((zero,o[:,1],zero+1),axis=1)
    # Immutable JAX arrays are supported by constructing, not assigning.
    gu=xp.concatenate((xp.zeros((n,2),dtype=c.dtype),v[:,:1],xp.zeros((n,4),dtype=c.dtype)),axis=1)
    validate_soil_transport(g,p[:,:4],so,gu,xp)
    supply=xp.zeros((n,10),dtype=c.dtype)
    initial=xp.broadcast_to(u[:,:1],(n,11))
    validate_canopy_batch(a,u,f,supply,initial,xp)
    if (not bool(xp.all(xp.isfinite(v))) or not bool(xp.all(xp.isfinite(p))) or
        bool(xp.any((v[:,0]<=0.)|(v[:,1]<0.)|((v[:,1]==0.)&(v[:,2]!=0.))|
            (((xp.sum(a,axis=1)==0.)|(u[:,13]==0.))&(v[:,1]>0.)))) or bool(xp.any(v[:,3:]<0.)) or
        bool(xp.any((p[:,4]<0.)|(p[:,4]>1.)|(p[:,5]<0.)|(p[:,5]>p[:,0])))):
        raise ValueError('Invalid surface stores, roots or soil properties')


def surface_kernel_jax(c,g,p,o,a,u,f,v,radiation,leaf,aging,condensate=None,exchange_extension=None,vic=None):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;dtype=c.dtype;zero=j.zeros((),dtype);dt=v[0];n=o[0];phase=o[1]
    row=lambda *x:j.asarray(x,dtype)
    if condensate is not None:v=v.at[1].set(_sum(condensate[:,0])).at[2].set(_sum(condensate[:,1]))
    inv=invert_soil_enthalpy_jax(g[:,0],g[:,1],g[:,2],p[0],p[1],phase)
    root_h=s(334000.+s(4180.*inv[0]))
    supply=j.concatenate((row(1.),j.maximum(0.,s(inv[3]-s(s(p[5]*1000.)*g[:,2])))/dt,v[3:7],row(inv[3][0]/dt)))
    u=u.at[15].set(v[1]/dt).at[17].set(j.where(n>0,c[0,0]/dt,j.inf))
    initial=j.full(11,u[0],dtype)
    def evaluate(temp):
        boundary=j.concatenate((row(temp,(temp<273.15).astype(dtype),(n==0).astype(dtype),1.,v[1],v[2],dt),root_h))
        if condensate is None:
            canopy=canopy_energy_kernel_jax(a,u,f,supply,initial,leaf,boundary=boundary,radiation=radiation,
                tolerance=.005 if dtype==j.float32 else 1e-7,q_tolerance=1e-8 if dtype==j.float32 else 1e-12)
            extra=j.zeros(0,dtype)
        else:
            from .canopy_phase_energy import phase_energy_kernel_jax
            b=j.concatenate((row(1.,temp,(temp<273.15).astype(dtype),(n==0).astype(dtype),dt),root_h))
            extra=phase_energy_kernel_jax(a,u,f,supply,j.zeros(10,dtype),condensate,b,None,n,leaf,None,None,
                tolerance=.005 if dtype==j.float32 else 1e-7,radiation=radiation,exchange_extension=exchange_extension)
            canopy=j.concatenate((extra[:195],extra[288:289],extra[293:298],extra[289:291],extra[285:286],
                row(_sum(extra[195:205]),_sum(extra[205:215])),extra[286:288]))
        roots=s(canopy[51:55]*dt);gm=s(canopy[50]*dt);wet=s(_sum(canopy[30:40])*dt)
        demand=roots.at[0].add(j.where((n==0)&(gm>0.),gm,0.))
        tr=soil_transport_kernel_jax(g,p[:4],j.stack((j.asarray(1),phase,j.asarray(1))),j.concatenate((row(0.,0.,dt),demand)))
        ground=tr[:12].reshape(4,3);gh=s(_sum(tr[32:36])-_sum(s(roots*root_h)))
        sw=canopy[196:201];soil_sw=canopy[201]
        def snowy():
            vapor=snow_vapor_kernel_jax(c,n,row(gm,s(temp-273.15),0.));nn=vapor[45].astype(j.int32)
            shifted=j.concatenate((sw[1:],row(0.))).at[0].add(j.where(nn>0,sw[0],0.))
            solar=j.where(nn<n,shifted,sw)
            base=s(soil_sw+j.where((nn<n)&(nn==0),sw[0],0.))
            return vapor[:45].reshape(5,9),nn,ground,vapor[46],vapor[47],vapor[55]/dt,zero,zero,vapor[48:53],solar,base,tr[46]>0.
        def bare():
            offset=s(temp-273.15)
            heat=j.where(gm<0.,s(gm*j.where(temp<273.15,s(2100.*offset),s(334000.+s(4180.*offset)))),gh)
            latent=s(j.where((gm<0.)&(temp<273.15),2834000.,2500000.)*gm)/dt
            added=soil_transport_kernel_jax(ground,p[:4],j.stack((j.asarray(0),phase,j.asarray(1))),
                j.concatenate((row(j.maximum(-gm,0.),j.where(gm<0.,-heat,0.),dt),j.zeros(4,dtype))))
            return c,n,added[:12].reshape(4,3),gm,heat,latent,added[26],added[27],j.zeros(5,dtype),sw,soil_sw,(tr[46]>0.)&(added[46]>0.)
        pack,nn,ground,gm,gh,latent,runoff,runoff_h,residue,sw,soil_sw,valid=lax.cond(n>0,snowy,bare)
        ice_before=snow_phase_jax(pack[:,0],pack[:,1])[1]
        top=s(s(canopy[195]+s(canopy[202]*canopy[50]))-latent)
        cells=j.zeros((9,5),dtype).at[:,2:4].set(1.)
        def cell(k,cells):
            snow=k<nn;soil=(k>=nn)&(k<nn+4);sk=j.minimum(k,4);gk=j.clip(k-nn,0,3)
            value=j.stack((j.where(snow,pack[sk,0],j.where(soil,ground[gk,0],0.)),
                j.where(snow,pack[sk,1],j.where(soil,ground[gk,1],0.)),
                j.where(snow,pack[sk,2],j.where(soil,ground[gk,2],1.)),j.where(snow|soil,0.,1.),
                j.where(snow,sw[sk],j.where(k==nn,soil_sw,0.))))
            return cells.at[k].set(value)
        cells=lax.fori_loop(0,9,cell,cells)
        heat=j.concatenate(joint_heat_kernel_jax(cells,p[j.array([0,1,4])],j.stack((nn,phase)),row(dt,top,0.,0.)))
        pack=pack.at[:,1].set(j.where(j.arange(5)<nn,heat[:5],0.))
        ground=ground.at[:,1].set(heat[nn+j.arange(4)])
        residual=s(heat[18]-s(temp-273.15))
        valid=valid&(canopy[68]>0.)&(heat[75]>0.)&j.all(inv[4])
        return j.concatenate((pack.ravel(),row(nn),ground.ravel(),canopy,heat,ice_before,
            row(gm,gh,latent,wet,_sum(roots),runoff,runoff_h),residue,row(temp,residual,valid.astype(dtype)),extra))
    center=j.clip(s(j.where(n>0,snow_phase_jax(c[0,0],c[0,1])[0],inv[0][0])+273.15),200.,350.)
    candidate=evaluate(center)
    resolution=s(s(2.*j.finfo(dtype).eps)*273.15)
    tol=j.maximum(j.asarray(2e-7,dtype),resolution)
    # Offset secant avoids cancellation between large absolute Kelvin values.
    # The evaluated temperature itself is rounded to storage precision.
    def bracket(_,carry):
        left,right,fl,fr,width,count=carry
        def take_left():
            t=j.maximum(200.,s(center-width));r=evaluate(t);return t,r[361],count+1
        left,fl,count=lax.cond((fl<0.)&(left>200.),take_left,lambda:(left,fl,count))
        def take_right():
            t=j.minimum(350.,s(center+width));r=evaluate(t);return t,r[361],count+1
        right,fr,count=lax.cond((fr>0.)&(right<350.),take_right,lambda:(right,fr,count))
        return left,right,fl,fr,s(2.*width),count
    start=(center,center,candidate[361],candidate[361],j.asarray(2.,dtype),j.asarray(1))
    left,right,fl,fr,_,count=lax.cond(j.abs(candidate[361])>tol,
        lambda:lax.fori_loop(0,8,bracket,start),lambda:start)
    bracketed=(fl>=0.)&(fr<=0.)
    def step(carry):
        k,left,right,fl,fr,details,count=carry
        width=s(right-left);trial=s(left-s(fl*width)/j.where(fr!=fl,s(fr-fl),1.))
        trial=j.clip(trial,s(left+s(.05*width)),s(right-s(.05*width)))
        # The 5% margin can round to an endpoint in a narrow float32 bracket.
        trial=j.where((trial<=left)|(trial>=right),s(left+s(.5*width)),trial)
        result=evaluate(trial);value=result[361]
        return k+1,j.where(value>0.,trial,left),j.where(value>0.,right,trial),j.where(value>0.,value,fl),j.where(value>0.,fr,value),result,count+1
    _,left,right,_,_,candidate,count=lax.while_loop(lambda x:(x[0]<60)&(j.abs(x[5][361])>tol)&bracketed&(s(x[2]-x[1])>resolution),step,
        (j.asarray(0),left,right,fl,fr,candidate,count))
    def plateau():
        exact=evaluate(j.asarray(273.15,dtype))
        return j.where(j.abs(exact[361])<=tol,exact,candidate),count+1
    candidate,count=lax.cond((candidate[45]>0.)&(candidate[284]==0.)&(candidate[360]!=j.asarray(273.15,dtype)),
        plateau,lambda:(candidate,count))
    # A signed bracket narrower than Kelvin representation also resolves the
    # root. Keep its endpoints and the actual residual: never report that the
    # residual test passed when rounding prevented further refinement.
    converged=((j.abs(candidate[361])<=tol)|(bracketed&(s(right-left)<=resolution)))&(candidate[362]>0.)
    details=None if condensate is None else candidate[363:]
    finished=surface_finish_jax(c,g,v,candidate[:363],count,converged,aging,p,phase,details,vic)
    return j.concatenate((finished,row(left,right),candidate[363:]))


def surface_finish_jax(old_snow,old_soil,v,d,count,converged,aging,p,phase,phase_details=None,vic=None):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;dtype=d.dtype;dt=v[0];row=lambda *x:j.asarray(x,dtype)
    pack=d[:45].reshape(5,9);n=d[45].astype(j.int32);ground=d[46:58].reshape(4,3);canopy=d[58:266];heat=d[266:343]
    # Nonuniform three-point derivative matches np.gradient(..., edge_order=1).
    z=j.where(j.arange(6)<n,j.r_[pack[:,2],row(0.)],ground[0,2])
    centers=lax.fori_loop(0,6,lambda k,x:x.at[k].set(s(_sum(j.where(j.arange(6)<=k,z,0.))-s(.5*z[k]))),j.zeros(6,dtype))
    temperatures=j.where(j.arange(6)<n,j.r_[heat[18:23],row(0.)],heat[18+n])
    def grad(k):
        prev=j.maximum(k-1,0);nxt=j.minimum(k+1,5);h1=s(centers[k]-centers[prev]);h2=s(centers[nxt]-centers[k]);h1=j.where(k==0,1.,h1)
        # Difference form is translation invariant and reduces cancellation.
        left=s(temperatures[k]-temperatures[prev])/h1;right=s(temperatures[nxt]-temperatures[k])/h2
        middle=s(s(h2*left)+s(h1*right))/s(h1+h2)
        return j.where(k<n,j.abs(j.where(k==0,right,middle)),0.)
    gradient=j.stack([grad(k) for k in range(5)])
    ice=snow_phase_jax(pack[:,0],pack[:,1])[1];refrozen=j.maximum(0.,s(ice-d[343:348]))
    ev=snow_evolution_kernel_jax(pack,j.stack((n,j.asarray(3),j.asarray(1),j.asarray(1),j.asarray(0))),
        j.concatenate((row(dt),gradient,d[343:348],refrozen,j.zeros(5,dtype),row(0.,0.))),aging)
    drain=snow_transfer_kernel_jax(ev[:45].reshape(5,9),n,j.asarray(3),row(0.,0.,100.,54.5,.033,.05,.2,.02,.02,.01,.01,0.,0.,0.,0.,0.))
    pack=drain[:45].reshape(5,9);nn=drain[45]
    admission=soil_transport_kernel_jax(ground,p[:4],j.stack((j.asarray(0),phase,j.asarray(1))),
        j.concatenate((row(drain[53],drain[54],dt),j.zeros(4,dtype))),vic)
    ground=admission[:12].reshape(4,3)
    runoff=s(s(d[353]+drain[60])+admission[26]);runoff_h=s(s(d[354]+drain[61])+admission[27])
    advected=s(d[349]+s(dt*s(canopy[206]+canopy[207])))
    et=s(s(d[352]+d[348])+d[351]);le=s(s(2500000.*s(_sum(canopy[40:50])+_sum(canopy[30:40])))+d[350])
    if phase_details is not None:le=s(s(_sum(phase_details[255:265])/dt+s(2500000.*_sum(canopy[40:50])))+d[350])
    atmospheric=s(s(s(canopy[29]-canopy[26])-le)*dt)
    old_h=s(s(_sum(old_snow[:,1])+_sum(old_soil[:,1]))+v[2]);new_h=s(s(_sum(pack[:,1])+_sum(ground[:,1]))+canopy[205])
    old_m=s(s(_sum(old_snow[:,0])+_sum(old_soil[:,0]))+v[1]);new_m=s(s(_sum(pack[:,0])+_sum(ground[:,0]))+canopy[204])
    energy=s(s(s(s(new_h-old_h)-atmospheric)+advected)+runoff_h);water=s(s(s(new_m-old_m)+et)+runoff)
    impurity=s(s(d[355:360]+drain[62:67])+drain[55:60])
    summary=j.concatenate((row(et,runoff,runoff_h,advected,atmospheric,energy,water,drain[53],drain[54],le,heat[71]/dt,count,
        (converged&(admission[46]>0.)).astype(dtype)),impurity))
    d=d.at[260].set(j.where(d[348]!=0.,s(d[350]*dt)/j.where(d[348]!=0.,d[348],1.),d[260]))
    return j.concatenate((d,pack.ravel(),row(nn),ground.ravel(),summary))


def unpack_surface(out):
    result=unpack_boundary(out[:,58:266])
    result.update(snow_cells=out[:,363:408].reshape(-1,5,9),snow_counts=out[:,408].astype('int32'),
        soil_cells=out[:,409:421].reshape(-1,4,3),thermal_snow_cells=out[:,:45].reshape(-1,5,9),
        thermal_snow_counts=out[:,45].astype('int32'),thermal_soil_cells=out[:,46:58].reshape(-1,4,3),
        thermal_temperature_offset=out[:,284:293],thermal_layer_residual=out[:,311:320],
        thermal_layer_tolerance=out[:,328:337],surface_temperature=out[:,360],surface_temperature_residual=out[:,361],
        ground_nonshortwave_heat=out[:,339],ground_vapor_mass_kgm2=out[:,348],
        impurity_output_kgm2=out[:,434:439],surface_root_bracket_k=out[:,439:441])
    names=('et_kgm2','runoff_kgm2','runoff_enthalpy_jm2','advected_enthalpy_jm2','atmospheric_energy_jm2',
        'energy_residual_jm2','water_residual_kgm2','snow_drainage_kgm2','snow_drainage_enthalpy_jm2','le','ground_heat')
    result.update({name:out[:,421+k] for k,name in enumerate(names)})
    result.update(surface_evaluations=out[:,432].astype('int32'),converged=out[:,433]>0.)
    return result


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(surface_kernel_jax,in_axes=(0,0,0,0,0,0,0,0,0,None,None)))


@lru_cache(maxsize=1)
def _radiation():
    import jax
    from functools import partial
    return jax.jit(jax.vmap(partial(canopy_snow_kernel_jax,track_absorption=True),in_axes=(0,0,0,0,None,None)))


def run_canopy_surface_jax(c,g,p,o,a,u,f,v,parameters=None,*,raise_on_failure=True):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(c,g,p,o,a,u,f,v)));validate_surface(arrays,j)
    c,g,p,o,a,u,f,v=arrays;dtype=str(c.dtype)
    optical=c[:,:,j.array([0,3,4,5,6,7,8])]
    radiation=_radiation()(a,optical,o[:,0],radiation_inputs(a,u,j),*_jax_tables(dtype))
    out=_compiled()(*arrays,radiation,j.asarray(pack_leaf_parameters(parameters,np.dtype(dtype))),_aging(dtype))
    result=unpack_surface(out)
    if raise_on_failure and (not bool(j.all(j.isfinite(out))) or not bool(j.all(result['converged']))):
        raise RuntimeError('JAX winter surface failed; inspect with raise_on_failure=False')
    return result

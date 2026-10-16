"""Native meteorological glacier column: precipitation, optics, heat and mass.

Five-array ABI: snow (5,9), ice (4,7), four physical parameters, integer
[snow count, ice count, precipitation option, drag option], and site values
[z0m, wind height, temperature height, humidity height, cosine zenith, diffuse].
All extensive transfers and surface roots run on the selected device.
"""
from functools import lru_cache
import numpy as np

DIAGNOSTICS=('ts','rnet','h','le','g','et_mm','prcp_mm','runoff_mm','swe','flux_surface_temperature',
    'precipitation_enthalpy_jm2','runoff_enthalpy_jm2','advected_enthalpy_jm2','atmospheric_energy_jm2',
    'geothermal_input_jm2','water_residual_kgm2','energy_residual_jm2','water_bound_kgm2','energy_bound_jm2',
    'surface_evaluations','heat_velocity','water_velocity')


def pack_clm_glacier_columns(states,*,cosine_zenith=.5,diffuse_fraction=.5,dtype=np.float64):
    from .glacier import pack_glacier_case
    states=list(states)
    if not states:raise ValueError('Glacier packing requires at least one state')
    rows=[]
    mu=np.broadcast_to(cosine_zenith,(len(states),));df=np.broadcast_to(diffuse_fraction,(len(states),))
    for i,state in enumerate(states):
        c,g,count,b,p,q=pack_glacier_case(state.snow,state.ice,1.,0.,parameters=state.parameters,dtype=dtype)
        o=np.r_[count,[state.options.OptRainSnowPartition,state.options.OptSurfaceDrag]].astype(np.int32)
        site=np.asarray([state.z0m,state.wind_height,state.temperature_height,state.humidity_height,mu[i],df[i]],dtype)
        rows.append((c,g,p,o,site))
    return tuple(np.stack([row[i] for row in rows]) for i in range(5))


def validate_glacier_column(arrays,f,dep,xp):
    from .glacier import validate_glacier
    c,g,p,o,site=arrays;n=len(c)
    if o.shape!=(n,4) or site.shape!=(n,6) or f.shape!=(n,8) or dep.shape!=(n,5):raise ValueError('Invalid glacier column shapes')
    if any(v.dtype!=c.dtype for v in (site,f,dep)):raise ValueError('Glacier column float dtypes must match')
    b=xp.concatenate((f[:,:1],xp.zeros((n,3),c.dtype)),axis=1)
    validate_glacier(c,g,o[:,:2],b,p,xp.zeros((n,9),c.dtype),xp)
    if not all(bool(xp.all(xp.isfinite(v))) for v in (site,f,dep)):raise ValueError('Nonfinite glacier column input')
    if bool(xp.any((site[:,0]<1e-5)|(site[:,0]>.5)|(site[:,1]<=site[:,0])|
        (site[:,2]<=site[:,0]*.1)|(site[:,3]<=site[:,0]*.1)|(site[:,4]<=0.)|(site[:,4]>1.)|(site[:,5]<0.)|(site[:,5]>1.))):
        raise ValueError('Invalid glacier optical geometry or measurement heights')
    if bool(xp.any((f[:,1]<200.)|(f[:,1]>330.)|(f[:,2]<0.)|(f[:,2]>=.2)|(f[:,3]<0.)|(f[:,4]<30000.)|xp.any(f[:,5:]<0.,axis=1))) or bool(xp.any(dep<0.)):
        raise ValueError('Invalid glacier meteorology or deposition')
    if bool(xp.any(~((o[:,2]==1)|(o[:,2]==2)|(o[:,2]==3)|(o[:,2]==5))|((o[:,3]!=1)&(o[:,3]!=2)))):
        raise ValueError('Unsupported glacier precipitation or drag option')
    if n and bool(xp.any(o[:,3]!=o[0,3])):raise ValueError('Group glacier batches by drag option')


def prepare_glacier_column_jax(c,g,p,o,site,f):
    import jax.numpy as j
    from jax import lax
    from .jax_core import _snow_fraction
    from .snow_evolution import fresh_snow_jax
    from .snow_mass import snow_transfer_kernel_jax
    from .glacier_mass import mix_glacier_liquid_jax
    s=lax.optimization_barrier;row=lambda *x:j.asarray(x,c.dtype);dt,tair,qair,wind,ps,sw,lw,prcp=f
    ns,ni=o[:2];total=s(prcp*dt);snow=s(total*_snow_fraction(tair,qair,ps,o[2]));rain=s(s(total-snow)*1000.);snow=s(snow*1000.)
    offset=s(tair-273.15);rainh=s(rain*s(334000.+s(4180.*offset)));snowh=s(snow*s(2100.*j.minimum(offset,0.)))
    def snowfall():
        fresh=fresh_snow_jax(offset,wind)
        u=j.concatenate((row(snow,j.minimum(offset,0.),fresh[0],fresh[1],.033,.05,.2,.02,.02,.01,.01),j.zeros(5,c.dtype)))
        out=snow_transfer_kernel_jax(c,ns,j.asarray(1),u)
        return out[:45].reshape(5,9),out[45].astype(j.int32)
    c,ns=lax.cond(snow>0.,snowfall,lambda:(c,ns))
    def rainfall():
        def snowy():return c.at[0,0].set(s(c[0,0]+rain)).at[0,1].set(s(c[0,1]+rainh)),g,ni,j.zeros(7,c.dtype)
        def bare():
            ice,n,out=mix_glacier_liquid_jax(g,ni,j.concatenate((row(rain,rainh),j.zeros(5,c.dtype))))
            return c,ice,n,out
        return lax.cond(ns>0,snowy,bare)
    c,g,ni,out=lax.cond(rain>0.,rainfall,lambda:(c,g,ni,j.zeros(7,c.dtype)))
    momentum=j.log(site[1]/site[0]);rough=s(site[0]*.1);numerator=s(.16*j.maximum(wind,.3))
    hv=numerator/s(momentum*j.log(site[2]/rough));wv=numerator/s(momentum*j.log(site[3]/rough))
    return c,g,o.at[0].set(ns).at[1].set(ni),j.concatenate((row(s(rain+snow),s(rainh+snowh)),out)),row(hv,wv,tair)


def glacier_surface_kernel_jax(c,g,p,o,f,vel,q,reflected):
    import jax.numpy as j
    from jax import lax
    from .glacier import glacier_kernel_jax
    from .glacier_mass import glacier_transfer_kernel_jax
    from .snow_mass import snow_phase_jax
    from .snow_vapor import snow_vapor_kernel_jax
    s=lax.optimization_barrier;dtype=c.dtype;row=lambda *x:j.asarray(x,dtype);ns,ni=o[:2];dt=f[0]
    rho=f[4]/s(s(287.*f[1])*s(1.+s(.61*f[2])))
    supply=j.where(ns>0,c[0,0],g[0,0])
    def evaluate(t):
        offset=s(t-273.15)
        water_es=s(611.2*j.exp(s(17.67*offset)/s(offset+243.5)))
        ice_es=j.exp(s(s(s(9.550426-5723.265/t)+s(3.53068*j.log(t)))-s(.00728332*t)))
        es=j.where(t<273.15,ice_es,water_es);humidity=s(.622*es)/s(f[4]-s(.378*es))
        demand=j.minimum(s(s(s(rho*vel[1])*s(humidity-f[2]))*dt),supply)
        def snow_vapor():
            v=snow_vapor_kernel_jax(c,ns,row(demand,offset,0.));nn=v[45].astype(j.int32)
            shifted=j.r_[q[1:5],j.zeros(1,dtype)].at[0].add(j.where(nn>0,q[0],0.))
            solar=q.at[:5].set(j.where(nn<ns,shifted,q[:5])).at[5].set(s(q[5]+j.where((nn<ns)&(nn==0),q[0],0.)))
            return v[:45].reshape(5,9),g,j.stack((nn,ni)),v[46],v[47],v[55]/dt,row(0.,0.),v[48:53],solar
        def ice_vapor():
            v=glacier_transfer_kernel_jax(g,ni,j.asarray(1),j.r_[row(demand,offset),j.zeros(5,dtype)])
            nn=v[28].astype(j.int32)
            shifted=j.r_[q[6:],j.zeros(1,dtype)].at[0].add(j.where(nn>0,q[5],0.))
            solar=q.at[5:].set(j.where(nn<ni,shifted,q[5:]))
            return c,v[:28].reshape(4,7),j.stack((ns,nn)),v[31],v[32],v[33]/dt,v[29:31],v[34:39],solar
        snow,ice,count,gm,gh,le,flow,tracer,solar=lax.cond(ns>0,snow_vapor,ice_vapor)
        h=s(s(s(rho*1004.)*vel[0])*s(t-vel[2]));t2=s(t*t);lw=s(p[3]*s(f[6]-s(j.asarray(5.670374419e-8,dtype)*s(t2*t2))))
        boundary=row(dt,s(s(lw-h)-le),0.,0.)
        def heat():
            sn,ic,nn,d,th=glacier_kernel_jax(snow,ice,count,boundary,p,solar)
            return j.concatenate((sn.ravel(),ic.ravel(),nn.astype(dtype),d,*th))
        # A last-cell vapor loss has no material left for the heat equation.
        # Do not evaluate a zero-cell matrix; reject this forcing time step.
        flat=lax.cond(count[1]>0,heat,lambda:j.zeros(167,dtype))
        d=flat[75:90];actual=j.where(count[1]>0,flat[99],j.nan);valid=(count[1]>0)&(flat[165]>0.)
        return j.concatenate((flat,row(t,s(actual-t),valid.astype(dtype),gm,gh,le,h,s(s(f[5]-reflected)+lw),
            s(flow[0]+d[0]),s(flow[1]+d[1])),s(tracer+d[2:7]),vel[:2]))
    initial=j.where(ns>0,snow_phase_jax(c[0,0],c[0,1])[0],g[0,1]/s(2100.*j.maximum(g[0,0],1e-30)))
    center=j.clip(s(initial+273.15),200.,350.);candidate=evaluate(center)
    resolution=s(s(2.*j.finfo(dtype).eps)*273.15);tol=j.maximum(j.asarray(2e-7,dtype),resolution)
    def bracket(_,v):
        left,right,fl,fr,width,count=v
        def lo():
            t=j.maximum(200.,s(center-width));out=evaluate(t);return t,out[168],count+1
        left,fl,count=lax.cond((fl<0.)&(left>200.),lo,lambda:(left,fl,count))
        def hi():
            t=j.minimum(350.,s(center+width));out=evaluate(t);return t,out[168],count+1
        right,fr,count=lax.cond((fr>0.)&(right<350.),hi,lambda:(right,fr,count))
        return left,right,fl,fr,s(width*2.),count
    start=(center,center,candidate[168],candidate[168],j.asarray(2.,dtype),j.asarray(1))
    left,right,fl,fr,_,count=lax.cond(j.abs(candidate[168])>tol,lambda:lax.fori_loop(0,8,bracket,start),lambda:start)
    bracketed=(fl>=0.)&(fr<=0.)
    def step(v):
        k,left,right,fl,fr,old,count=v;width=s(right-left)
        trial=s(left-s(fl*width)/j.where(fr!=fl,s(fr-fl),1.));trial=j.clip(trial,s(left+s(.05*width)),s(right-s(.05*width)))
        out=evaluate(trial);value=out[168]
        return k+1,j.where(value>0.,trial,left),j.where(value>0.,right,trial),j.where(value>0.,value,fl),j.where(value>0.,fr,value),out,count+1
    _,left,right,_,_,candidate,count=lax.while_loop(lambda v:(v[0]<60)&(j.abs(v[5][168])>tol)&bracketed&(s(v[2]-v[1])>resolution),
        step,(j.asarray(0),left,right,fl,fr,candidate,count))
    def plateau():
        exact=evaluate(j.asarray(273.15,dtype));return j.where(j.abs(exact[168])<=tol,exact,candidate),count+1
    candidate,count=lax.cond((candidate[108]==0.)&(candidate[167]!=j.asarray(273.15,dtype)),plateau,lambda:(candidate,count))
    good=((j.abs(candidate[168])<=tol)|(bracketed&(s(right-left)<=resolution)))&(candidate[169]>0.)
    return j.concatenate((candidate,row(count,left,right,good.astype(dtype))))


def finish_glacier_column_jax(c,g,p,o,f,dep,prep,d,aging):
    import jax.numpy as j
    from jax import lax
    from .snow_mass import snow_phase_jax
    from .snow_evolution import snow_evolution_kernel_jax
    s=lax.optimization_barrier;dtype=c.dtype;row=lambda *x:j.asarray(x,dtype)
    snow=d[:45].reshape(5,9);ice=d[45:73].reshape(4,7);ns=d[73].astype(j.int32);ni=d[74].astype(j.int32);idx=j.arange(6)
    # Actual post-drainage geometry, followed by conservative compact/age.
    z=j.where(idx<ns,j.r_[snow[:,2],row(1.)],j.maximum(ice[0,0]/917.,1e-30))
    t=j.where(idx<ns,j.r_[snow_phase_jax(snow[:,0],snow[:,1])[0],row(0.)],ice[0,1]/s(2100.*j.maximum(ice[0,0],1e-30)))
    def total(v):return lax.fori_loop(0,len(v),lambda i,x:s(x+v[i]),j.zeros((),dtype))
    centers=j.stack([s(total(j.where(idx<=i,z,0.))-s(.5*z[i])) for i in range(6)])
    def grad(i):
        lo=j.maximum(i-1,0);hi=j.minimum(i+1,5);h1=j.where(i==0,1.,s(centers[i]-centers[lo]));h2=j.maximum(s(centers[hi]-centers[i]),1e-30)
        left=s(t[i]-t[lo])/h1;right=s(t[hi]-t[i])/h2
        middle=s(s(h2*left)+s(h1*right))/s(h1+h2)
        value=j.where(i==0,right,j.where(i==ns+(ni>0).astype(j.int32)-1,left,middle))
        return j.where((i<ns)&(ns+(ni>0).astype(j.int32)>1),j.abs(value),0.)
    gradient=j.stack([grad(i) for i in range(5)])
    u=j.concatenate((row(f[0]),gradient,j.zeros(15,dtype),row(0.,0.)))
    ev=snow_evolution_kernel_jax(snow,j.stack((ns,j.asarray(3),j.asarray(0),j.asarray(0),j.asarray(0))),u,aging)
    snow=ev[:45].reshape(5,9);addition=s(dep*f[0]);tracer=s(d[177:182]+prep[4:9])
    snow=snow.at[0,4:].set(s(snow[0,4:]+j.where(ns>0,addition,0.)))
    ice=ice.at[0,2:].set(s(ice[0,2:]+j.where((ns==0)&(ni>0),addition,0.)))
    tracer=s(tracer+j.where((ns==0)&(ni==0),addition,0.))
    mass0=s(total(c[:,0])+total(g[:,0]));heat0=s(total(c[:,1])+total(g[:,1]))
    mass1=s(total(snow[:,0])+total(ice[:,0]));heat1=s(total(snow[:,1])+total(ice[:,1]))
    run=s(prep[2]+d[175]);runh=s(prep[3]+d[176]);atm=s(s(s(d[174]-d[173])-d[172])*f[0]);geo=s(p[1]*f[0])
    water=s(s(s(s(mass1-mass0)-prep[0])+d[170])+run)
    heat=s(s(s(s(s(s(heat1-heat0)-prep[1])-atm)-geo)+d[171])+runh)
    eps=j.finfo(dtype).eps;wm=s(s(128.*eps)*j.maximum(s(s(mass0+prep[0])+j.abs(d[170])),1.))
    scale=total(j.abs(j.r_[c[:,1],g[:,1],row(prep[1],atm,geo,d[171],runh)]))
    wh=s(s(s(128.*eps)*j.maximum(scale,1.))+d[89])
    ts=j.where(ns>0,s(snow_phase_jax(snow[0,0],snow[0,1])[0]+273.15),j.where(ni>0,s(ice[0,1]/s(2100.*j.maximum(ice[0,0],1e-30))+273.15),0.))
    diag=row(ts,d[174],d[173],d[172],s(d[161]/f[0]-p[1]),d[170],prep[0],run,total(snow[:,0])/1000.,d[167],
        prep[1],runh,d[171],atm,geo,water,heat,wm,wh,d[184],d[182],d[183])
    return snow,ice,o.at[0].set(ns).at[1].set(ni),diag,tracer


@lru_cache(maxsize=1)
def _stages():
    import jax
    return (jax.jit(jax.vmap(prepare_glacier_column_jax)),jax.jit(jax.vmap(glacier_surface_kernel_jax)),
        jax.jit(jax.vmap(finish_glacier_column_jax,in_axes=(0,0,0,0,0,0,0,0,None))))


def unpack_glacier_column(result,surface):
    c,g,o,d,tracer=result
    return dict(snow_cells=c,ice_cells=g,options=o,diagnostics={k:d[:,i] for i,k in enumerate(DIAGNOSTICS)},
        impurity_out_kgm2=tracer,surface=surface,exhausted=o[:,1]==0)


def next_glacier_arrays(previous,result):
    return result['snow_cells'],result['ice_cells'],previous[2],result['options'],previous[4]


def step_clm_glacier_columns_jax(c,g,p,o,site,forcing,*,deposition=None):
    import jax.numpy as j
    from .snicar import run_snow_shortwave_jax
    from .snow_evolution import _tables as aging
    arrays=tuple(map(j.asarray,(c,g,p,o,site)));c,g,p,o,site=arrays;f=j.asarray(forcing)
    dep=j.zeros((len(c),5),c.dtype) if deposition is None else j.asarray(deposition,c.dtype)
    validate_glacier_column(arrays,f,dep,j);prepare,surface,finish=_stages()
    pc,pg,po,prep,vel=prepare(*arrays,f)
    if bool(j.any(po[:,1]==0)):raise RuntimeError('Precipitation exhausted glacier; reduce time step')
    opt=pc[:,:,[0,3,4,5,6,7,8]];beam=.5*f[:,5]*(1.-site[:,5]);sky=.5*f[:,5]*site[:,5]
    boundary=j.stack((beam,beam,sky,sky,p[:,2],p[:,2],site[:,4]),axis=1)
    rad=run_snow_shortwave_jax(opt,po[:,0],boundary)
    q=j.concatenate((rad['absorbed'],rad['ground_absorbed'][:,None],j.zeros((len(c),3),c.dtype)),axis=1)
    def solve(v):return surface(pc,pg,p,po,f,v,q,rad['reflected'])
    if len(c) and int(o[0,3])==1:raw,most=run_glacier_most(solve,site,f,backend='jax')
    else:raw=solve(vel);most=None
    if not bool(j.all(j.isfinite(raw))) or not bool(j.all(raw[:,187]>0.)):raise RuntimeError('JAX glacier surface root failed')
    result=finish(c,g,p,o,f,dep,prep,raw,aging(str(c.dtype)))
    if not all(bool(j.all(j.isfinite(v))) for v in result):raise RuntimeError('Nonfinite JAX glacier column output')
    out=unpack_glacier_column(result,raw)
    d=out['diagnostics']
    if bool(j.any((j.abs(d['water_residual_kgm2'])>d['water_bound_kgm2'])|(j.abs(d['energy_residual_jm2'])>d['energy_bound_jm2']))):
        raise RuntimeError('JAX glacier column conservation failed')
    if most is not None:out['most']=most
    return out


def run_glacier_most(solve,site,f,*,backend):
    """Actual device H/E feedback; every trial retains the same old stores."""
    if backend=='jax':
        import jax.numpy as xp
        from .canopy_most import run_canopy_most_jax as coefficients
    else:
        import cupy as xp
        from .cuda.canopy_most import run_canopy_most_cuda as coefficients
    n=len(f);zero=xp.zeros(n,f.dtype);zeta=zero;effective=xp.maximum(1.,f[:,3]);done=xp.zeros(n,bool)
    geometry=xp.stack((site[:,1],site[:,2],site[:,3],zero,site[:,0],site[:,0]*.1,site[:,0]*.1,zero+1000.,site[:,0]),axis=1)
    lower=zero;upper=zero+2.;iterations=xp.zeros(n,xp.int32);previous=None
    atmosphere=xp.stack((f[:,1],f[:,2],f[:,4],f[:,3],zero,zero,zero+.2),axis=1)
    state=xp.stack((zeta,effective,zero+.04,zero),axis=1);current=coefficients(geometry,atmosphere,state)
    names=('friction_velocity','heat_velocity','water_velocity','leaf_velocity','ground_velocity')
    for count in range(1,51):
        raw=solve(xp.stack((current['heat_velocity'],current['water_velocity'],current['potential_temperature']),axis=1))
        vapor=raw[:,170]/f[:,0];h=raw[:,173]
        atmosphere=xp.stack((f[:,1],f[:,2],f[:,4],f[:,3],h,vapor,current['friction_velocity']),axis=1)
        feedback=coefficients(geometry,atmosphere,state)
        if count==1:stable=feedback['zeta']>0.
        target=coefficients(geometry,atmosphere,xp.stack((feedback['zeta'],feedback['effective_wind'],zero+.04,zero),axis=1))
        mismatch=xp.max(xp.stack([xp.abs(target[k]-current[k])/current[k] for k in names],axis=1),axis=1)
        if previous is not None:
            dh=xp.abs(h-previous[0]);de=xp.abs(vapor-previous[1]);accepted=(mismatch<=1e-4)&(dh<=.05)&(de<=2e-8)
            iterations=xp.where((~done)&accepted,xp.asarray(count,xp.int32),iterations);done=done|accepted
            if bool(xp.all(done)):return raw,dict(**current,used_zeta=zeta,used_effective_wind=effective,
                iterations=iterations,coefficient_residual=mismatch,heat_change_wm2=dh,vapor_change_kgm2s=de)
        previous=h,vapor
        lower=xp.where(stable&(~done)&(feedback['zeta']>=zeta),zeta,lower);upper=xp.where(stable&(~done)&(feedback['zeta']<zeta),zeta,upper)
        zeta=xp.where(done,zeta,xp.where(stable,.5*(lower+upper),.5*(zeta+feedback['zeta'])))
        effective=xp.where(done|stable,effective,.5*(effective+feedback['effective_wind']))
        state=xp.stack((zeta,effective,zero+.04,zero),axis=1);current=coefficients(geometry,atmosphere,state)
    raise RuntimeError('Native glacier MOST did not converge in 50 trials')

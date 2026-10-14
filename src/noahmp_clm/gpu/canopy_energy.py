"""JAX multilayer canopy energy reference and shared batch input contract.

Ten leaf layers, four root layers, one diagnostic canopy air space. Inputs
remain separate from the released column ABI while the new driver is built.
"""
from dataclasses import astuple
from functools import lru_cache
import numpy as np
from .photosynthesis import pack_leaf_parameters, LEAF_OUTPUTS

CANOPY_INPUTS = ("air_temperature","specific_humidity","pressure","atmosphere_velocity",
    "ground_velocity","leaf_velocity","soil_temperature","soil_conductance","co2_ppm",
    "ground_resistance","soil_flux_offset","swdown","lwdown","vegetated_fraction",
    "wet_fraction","canopy_evaporation_limit","transpiration_limit","ground_evaporation_limit",
    "ground_emissivity","visible_ground_albedo","nir_ground_albedo","diffuse_fraction","cosine_zenith")
CANOPY_OUTPUT_SIZE = 195


def pack_canopy_case(area, swdown, lwdown, exchange, *, dtype=np.float64, **kwargs):
    """Pack one NumPy reference call for either GPU backend; no physics here."""
    pending={'fixed_ground_temperature','ground_is_ice','ground_shares_soil','snowpack','canopy_storage'} & kwargs.keys()
    if pending:
        raise NotImplementedError('Winter canopy arguments are not yet in the GPU batch contract: '+', '.join(sorted(pending)))
    area = np.asarray(area,dtype=dtype)
    if area.shape != (10,):
        raise ValueError("GPU canopy requires ten leaf layers")
    alb = kwargs.get("ground_albedo",(.15,.3))
    u = np.asarray((*astuple(exchange),swdown,lwdown,kwargs.get("vegetated_fraction",1.),
        kwargs.get("wet_fraction",0.),kwargs.get("canopy_evaporation_limit",np.inf),
        kwargs.get("transpiration_limit",np.inf),kwargs.get("ground_evaporation_limit",np.inf),
        kwargs.get("ground_emissivity",.97),*alb,kwargs.get("diffuse_fraction",.5),
        kwargs.get("cosine_zenith",.5)),dtype=dtype)
    factors = np.stack(np.broadcast_arrays(np.asarray(kwargs.get("nitrogen_factor",1.),dtype=dtype)+np.zeros(10,dtype),
        np.asarray(kwargs.get("water_stress",1.),dtype=dtype)+np.zeros(10,dtype)),axis=-1)
    soil = np.zeros(10,dtype=dtype)
    if kwargs.get("soil_supply") is not None:
        s = kwargs["soil_supply"]
        if len(s.roots)!=4:
            raise ValueError("GPU canopy requires four root layers")
        soil[:] = np.r_[1.,s.roots,s.weights,s.surface]
    initial = np.asarray(kwargs.get("initial_temperatures",np.full(11,exchange.air_temperature)),dtype=dtype)
    return area,u,factors,soil,initial


def validate_canopy_batch(area,u,factors,soil,initial,xp):
    n = area.shape[0] if area.ndim == 2 else -1
    if (area.shape!=(n,10) or u.shape!=(n,23) or factors.shape!=(n,10,2)
            or soil.shape!=(n,10) or initial.shape!=(n,11)):
        raise ValueError("Canopy shapes must be (n,10), (n,23), (n,10,2), (n,10), (n,11)")
    if not all(bool(xp.all(xp.isfinite(v))) for v in (area,factors,soil,initial,u[:,:15],u[:,18:])):
        raise ValueError("Nonfinite canopy input")
    if bool(xp.any(xp.isnan(u[:,15:18]))) or bool(xp.any(u[:,15:18]<0.)):
        raise ValueError("Invalid evaporation limit")
    bad = ((u[:,0]<200)|(u[:,0]>350)|(u[:,1]<0)|(u[:,1]>=.2)|(u[:,2]<30000)
        |(u[:,3]<=0)|(u[:,4]<=0)|(u[:,5]<=0)|(u[:,6]<200)|(u[:,6]>350)|(u[:,7]<0)
        |(u[:,8]<=0)|(u[:,9]<0)|(u[:,11]<0)|(u[:,12]<0)|(u[:,13]<0)|(u[:,13]>1)
        |(u[:,14]<0)|(u[:,14]>1)|(u[:,18]<=0)|(u[:,18]>1)|(u[:,19]<0)|(u[:,19]>1)
        |(u[:,20]<0)|(u[:,20]>1)|(u[:,21]<0)|(u[:,21]>1)|(u[:,22]<=0)|(u[:,22]>1))
    if (bool(xp.any(bad)) or bool(xp.any(area<0)) or bool(xp.any((factors<0)|(factors>1)))
            or bool(xp.any((initial<200)|(initial>350))) or bool(xp.any(soil<0))
            or bool(xp.any((soil[:,0]!=0)&(soil[:,0]!=1))) or bool(xp.any(soil[:,1]>soil[:,9]))):
        raise ValueError("Unphysical canopy input")


def unpack_canopy_energy(a):
    return {"leaf_temperature":a[:,:10],"ground_temperature":a[:,10],
        "canopy_temperature":a[:,11],"canopy_humidity":a[:,12],"gpp":a[:,13],"leaf_respiration":a[:,14],
        "leaf_sensible":a[:,15:25],"ground_sensible":a[:,25],"h":a[:,26],"le":a[:,27],
        "ground_heat":a[:,28],"rnet":a[:,29],"wet_evaporation":a[:,30:40],"transpiration":a[:,40:50],
        "ground_evaporation":a[:,50],"root_uptake":a[:,51:55],"layer_residual":a[:,55:66],
        "residual":a[:,66],"iterations":a[:,67].astype("int32"),"converged":a[:,68]>0,
        "canopy_water_residual":a[:,69],"canopy_humidity_change":a[:,70],"air_heat_residual":a[:,71],
        "air_water_residual":a[:,72],"outgoing_shortwave":a[:,73],"outgoing_longwave":a[:,74],
        "gas_exchange":{k:a[:,75:195].reshape((-1,10,12))[:,:,j] for j,k in enumerate(LEAF_OUTPUTS)}}


def _sum(a):
    """Fixed left-to-right reduction, also used by the CUDA thread."""
    import jax.numpy as jnp
    from jax import lax
    return lax.fori_loop(0,a.shape[0],lambda j,total:total+a[j],jnp.zeros_like(a[0]))


def _dot(a,b):
    return _sum(a*b)


def _linear_solve(matrix,rhs):
    """Small dense Gaussian elimination with partial pivoting; no BLAS reductions."""
    import jax.numpy as jnp
    from jax import lax
    n = rhs.shape[0]
    indices = jnp.arange(n)
    def eliminate(k,carry):
        a,b = carry
        pivot = jnp.argmax(jnp.where(indices>=k,jnp.abs(a[:,k]),-1.))
        row,bval = a[k],b[k]
        a = a.at[k].set(a[pivot]).at[pivot].set(row)
        b = b.at[k].set(b[pivot]).at[pivot].set(bval)
        factor = jnp.where(indices>k,a[:,k]/a[k,k],0.)
        a = jnp.where((indices[:,None]>k)&(indices[None,:]>k),a-factor[:,None]*a[k][None,:],a)
        b = b-factor*b[k]
        return a,b
    a,b = lax.fori_loop(0,n,eliminate,(matrix,rhs))
    def back(idx,x):
        j = n-1-idx
        value = lax.fori_loop(j+1,n,lambda k,v:v-a[j,k]*x[k],b[j])
        return x.at[j].set(value/a[j,j])
    return lax.fori_loop(0,n,back,jnp.zeros_like(rhs))


def _root_allocation(demand,soil):
    import jax.numpy as jnp
    from jax import lax
    roots,weights = soil[1:5],soil[5:9]
    target = jnp.minimum(demand,_sum(jnp.where(weights>0.,roots,0.)))
    def distribute(_,take):
        remaining = jnp.maximum(0.,target-_sum(take))
        active = jnp.where(take<roots,weights,0.)
        denom = _sum(active)
        return take+jnp.minimum(remaining*active/jnp.maximum(denom,1e-30),roots-take)
    return lax.fori_loop(0,4,distribute,jnp.zeros_like(roots))


def _positive_scale(flux,limit):
    import jax.numpy as jnp
    positive,negative = _sum(jnp.maximum(flux,0.)),_sum(jnp.minimum(flux,0.))
    return jnp.where(positive>0.,jnp.minimum(1.,(limit-negative)/jnp.maximum(positive,1e-30)),1.)


def _water(qcan,qs,gs,area,fv,rho,u,soil,ground_shares_soil=True):
    import jax.numpy as jnp
    boundary = rho*fv*area*u[5]
    drying = qs[:10]>qcan
    kw = boundary*jnp.where(drying,u[14],1.)
    kt = boundary*jnp.where(drying,1.-u[14],0.)*gs/(u[5]+gs)
    wet = kw*(qs[:10]-qcan)
    kw = kw*jnp.where(wet>0.,_positive_scale(wet,u[15]),1.)
    transp = kt*(qs[:10]-qcan)
    root_limit = _sum(jnp.where(soil[5:9]>0.,soil[1:5],0.))
    limit = jnp.where(soil[0]>0.,jnp.minimum(u[16],root_limit),u[16])
    kt = kt*_positive_scale(transp,limit)
    transp = kt*(qs[:10]-qcan)
    uptake = jnp.where(soil[0]>0.,_root_allocation(_sum(transp),soil),jnp.zeros((4,),u.dtype))
    glimit = jnp.where((soil[0]>0.)&ground_shares_soil,jnp.minimum(u[17],jnp.maximum(0.,soil[9]-uptake[0])),u[17])
    under = rho*fv/(1./u[4]+u[9])
    bare = rho*(1.-fv)/(1./u[3]+u[9])
    ground = jnp.stack((under*(qs[10]-qcan),bare*(qs[10]-u[1])))
    scaling = jnp.where(ground>0.,_positive_scale(ground,glimit),1.)
    under,bare = under*scaling[0],bare*scaling[1]
    wet = kw*(qs[:10]-qcan)
    eg,eb = under*(qs[10]-qcan),bare*(qs[10]-u[1])
    residual = _sum(wet)+_sum(transp)+eg-rho*fv*u[3]*(qcan-u[1])
    return wet,transp,eg,eb,kw+kt,under,bare,residual,uptake


def _longwave_operator(area,eps,fv):
    import jax.numpy as jnp
    from jax import lax
    t = jnp.exp(-area)
    down = jnp.zeros((11,12),area.dtype).at[0,11].set(1.)
    def forward(j,d):
        return d.at[j+1].set(t[j]*d[j]).at[j+1,j].add(1.-t[j])
    down = lax.fori_loop(0,10,forward,down)
    up = jnp.zeros_like(down).at[10].set((1.-eps)*down[10]).at[10,10].add(eps)
    def back(idx,a):
        j=9-idx
        return a.at[j].set(t[j]*a[j+1]).at[j,j].add(1.-t[j])
    up = lax.fori_loop(0,10,back,up)
    mat = fv*jnp.concatenate((down[:10]+up[1:]-down[1:]-up[:10],(down[10]-up[10])[None,:]))
    return mat.at[10,10].add(-(1.-fv)*eps).at[10,11].add((1.-fv)*eps)


def canopy_energy_kernel_jax(area,u,factors,soil,initial,p,*,max_iterations=100,tolerance=.005,q_tolerance=1e-8,
                             boundary=None,radiation=None):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from .canopy_radiation import shortwave_kernel_jax
    from .photosynthesis import leaf_kernel_jax
    fv = jnp.where(_sum(area)>0.,u[13],0.)
    active = fv*area>0.
    bands = []
    for i,(reflect,transmit) in enumerate(((.1,.05),(.45,.25))):
        bands.append(shortwave_kernel_jax(area,jnp.stack((u[11]*.5*(1.-u[21]),u[11]*.5*u[21],
            u[19+i],u[22],jnp.asarray(reflect,u.dtype),jnp.asarray(transmit,u.dtype)))))
    bare_sw = u[11]*(1.-.5*(u[19]+u[20]))
    sw = jnp.concatenate((fv*(bands[0][:10]+bands[1][:10]),
        jnp.asarray([fv*(bands[0][44]+bands[1][44])+(1.-fv)*bare_sw])))
    par = jnp.where(area>0.,bands[0][:10]*4.6/jnp.maximum(area,1e-30),0.)
    outgoing_sw = fv*(bands[0][43]+bands[1][43])+(1.-fv)*(u[11]-bare_sw)
    if boundary is not None:
        # Native canopy/SNICAR output; radiation is frozen for a surface solve.
        sw=jnp.concatenate((radiation[:10],jnp.asarray([_sum(radiation[20:25])+radiation[25]])))
        par=radiation[10:20]
        outgoing_sw=radiation[26]
        latent=jnp.full((11,),2.5e6,u.dtype).at[10].add(jnp.where(boundary[1]>0.,334000.,0.))
        weights=area/jnp.maximum(_sum(area),1e-30)
    lw = _longwave_operator(area,u[18],fv)
    rho = u[2]/(287.*u[0]*(1.+.61*u[1]))
    hcan = rho*1004.*jnp.concatenate((fv*area*u[5],jnp.asarray([fv*u[4]])))
    hatm,hbare = rho*1004.*fv*u[3],rho*1004.*(1.-fv)*u[3]
    hden = _sum(hcan)+hatm
    hjac = jnp.diag(hcan)-jnp.where(hden>0.,hcan[:,None]*hcan[None,:]/jnp.maximum(hden,1e-30),0.)
    hjac = hjac.at[10,10].add(hbare)
    mol_density = u[2]/(8.314462618*u[0])
    empty_roots = (soil[0]>0.)&(_sum(jnp.where(soil[5:9]>0.,soil[1:5],0.))==0.)
    capacity_stress = jnp.where((u[16]==0.)|empty_roots,0.,factors[:,1])
    temperature = initial.at[:10].set(jnp.where(active,initial[:10],u[0]))
    if boundary is not None:
        temperature=temperature.at[10].set(boundary[0])

    def iteration(carry):
        count,temp,qcan,_,_ = carry
        tc = temp-273.15
        # Preserve the saturation-pressure rounding shared with NumPy/CUDA.
        # Without this barrier XLA folds 611.2 into the later .622/.378,
        # bypassing es and moving humidity by an ulp before its root solve.
        es = lax.optimization_barrier(611.2*jnp.exp(17.67*tc/(tc+243.5)))
        den = u[2]-.378*es
        qs = .622*es/den
        numerator = lax.optimization_barrier(.622*u[2]*es)
        numerator = lax.optimization_barrier(numerator*17.67)*243.5
        dq = numerator/(den*den*(tc+243.5)*(tc+243.5))
        if boundary is not None:
            t=temp[10]
            ies=lax.optimization_barrier(jnp.exp(9.550426-5723.265/t+3.53068*jnp.log(t)-.00728332*t))
            slope=5723.265/(t*t)+3.53068/t-.00728332
            iden=u[2]-.378*ies
            iqs=.622*ies/iden
            idq=lax.optimization_barrier(.622*u[2]*ies)*slope/(iden*iden)
            qs=qs.at[10].set(jnp.where(boundary[1]>0.,iqs,qs[10]))
            dq=dq.at[10].set(jnp.where(boundary[1]>0.,idq,dq[10]))
            es=es.at[10].set(jnp.where(boundary[1]>0.,ies,es[10]))
        vapor = qcan*u[2]/(.622+.378*qcan)
        leaf_input = jnp.stack((par,temp[:10],jnp.full((10,),u[8]),vapor/es[:10],
            jnp.full((10,),u[5]*mol_density),factors[:,0]*capacity_stress,factors[:,1]),axis=1)
        gas = jax.vmap(leaf_kernel_jax,in_axes=(0,None))(leaf_input,p)
        gs = gas[:,9]/mol_density
        left,right = jnp.minimum(jnp.min(qs),u[1]),jnp.maximum(jnp.max(qs),u[1])
        def bisect(_,bracket):
            lo,hi=bracket
            mid=.5*(lo+hi)
            residual=_water(mid,qs,gs,area,fv,rho,u,soil,True if boundary is None else boundary[2]>0.)[7]
            return jnp.where(residual>0.,mid,lo),jnp.where(residual>0.,hi,mid)
        left,right=lax.fori_loop(0,42,bisect,(left,right))
        qnew=jnp.where(fv>0.,.5*(left+right),u[1])
        wet,tr,eg,eb,kleaf,kunder,kbare,water_res,uptake=_water(qnew,qs,gs,area,fv,rho,u,soil,True if boundary is None else boundary[2]>0.)
        relative_temp=lax.optimization_barrier(temp-u[0])
        air_offset=jnp.where(hden>0.,_dot(hcan,relative_temp)/jnp.maximum(hden,1e-30),0.)
        tair=u[0]+air_offset
        sensible=hcan*(relative_temp-air_offset)
        sensible=sensible.at[10].add(hbare*(temp[10]-u[0]))
        evaporation=jnp.concatenate((wet+tr,jnp.asarray([eg+eb])))
        square=temp*temp
        blackbody=jnp.concatenate((5.670374419e-8*(square*square),u[12:13]))
        net=sw+jax.vmap(_dot,in_axes=(0,None))(lw,blackbody)
        ground=u[7]*(temp[10]-u[6])+u[10]
        residual=(net-sensible-2.5e6*evaporation).at[10].add(-ground)
        if boundary is not None:
            residual=net-sensible-latent*evaporation
            ground=residual[10]
            residual=residual.at[10].set(0.)
            enabled=boundary[3]>0.
            mass=jnp.where(enabled,jnp.maximum(0.,boundary[4]-boundary[6]*_sum(wet)),0.)
            liquid_heat=334000.+4180.*(temp[:10]-273.15)
            enthalpy=mass*_dot(weights,liquid_heat)
            storage=jnp.where(enabled,(enthalpy-boundary[5])/boundary[6],0.)
            root_mean=jnp.where(_sum(uptake)>0.,_dot(uptake,boundary[7:11])/jnp.maximum(_sum(uptake),1e-30),0.)
            wet_adv=jnp.where(enabled,_dot(wet,liquid_heat),0.)
            tr_adv=jnp.where(enabled,_dot(tr,liquid_heat),0.)
            residual=residual.at[:10].add(-jnp.where(enabled,weights*storage+wet*liquid_heat+tr*(liquid_heat-root_mean),0.))
        qerror=jnp.abs(qnew-qcan)
        energy_balanced=jnp.max(jnp.abs(residual))<=tolerance
        converged=energy_balanced&(qerror<=q_tolerance)
        kcan=jnp.concatenate((kleaf,jnp.asarray([kunder])))
        kden=_sum(kcan)+rho*fv*u[3]
        kd=kcan*dq
        ejac=jnp.diag(kd)-jnp.where(kden>0.,kcan[:,None]*kd[None,:]/jnp.maximum(kden,1e-30),0.)
        ejac=ejac.at[10,10].add(kbare*dq[10])
        jac=lw[:,:11]*(4.*5.670374419e-8*(temp*temp*temp))[None,:]-hjac-2.5e6*ejac
        jac=jac.at[10,10].add(-u[7])
        if boundary is not None:
            jac=lw[:,:11]*(4.*5.670374419e-8*(temp*temp*temp))[None,:]-hjac-latent[:,None]*ejac
            jac=jac.at[:10,:10].add(-jnp.where(enabled,4180.*mass/boundary[6]*(weights[:,None]*weights[None,:]),0.))
            jac=jac.at[jnp.arange(10),jnp.arange(10)].add(-jnp.where(enabled,4180.*(wet+tr),0.))
        inactive=jnp.concatenate((~active,jnp.asarray([False])))
        jac=jnp.where(inactive[:,None],-jnp.eye(11,dtype=u.dtype),jac)
        rhs=jnp.where(inactive,u[0]-temp,residual)
        if boundary is not None:
            jac=jac.at[10].set(-jnp.eye(11,dtype=u.dtype)[10])
            rhs=rhs.at[10].set(0.)
        delta=_linear_solve(jac,-rhs)
        next_temp=jnp.where(energy_balanced,temp,jnp.clip(temp+.8*jnp.clip(delta,-5.,5.),200.,350.))
        h,evap=_sum(sensible),_sum(evaporation)
        gpp=fv*_dot(area,jnp.where(active,gas[:,0],0.))
        resp=fv*_dot(area,jnp.where(active,gas[:,5],0.))
        air_h=hatm*air_offset+hbare*(temp[10]-u[0])
        air_e=rho*fv*u[3]*(qnew-u[1])+eb
        out=jnp.concatenate((temp,jnp.stack((tair,qnew,gpp,resp)),sensible[:10],
            jnp.stack((sensible[10],h,2.5e6*evap,ground,_sum(net))),wet,tr,jnp.asarray([eg+eb]),
            uptake,residual,jnp.stack((_sum(residual),count+1.,converged.astype(u.dtype),water_res,
            qerror,h-air_h,evap-air_e,outgoing_sw,u[12]-_sum(net-sw))),gas.reshape(-1)))
        if boundary is not None:
            out=out.at[27].set(_dot(latent,evaporation))
            out=jnp.concatenate((out,jnp.asarray([ground-sw[10]]),radiation[20:26],
                jnp.stack((latent[10],storage,mass,enthalpy,wet_adv,tr_adv))))
        return count+1,next_temp,qnew,out,converged
    carry=(jnp.asarray(0,jnp.int32),temperature,u[1],jnp.zeros((CANOPY_OUTPUT_SIZE+(13 if boundary is not None else 0),),u.dtype),jnp.asarray(False))
    return lax.while_loop(lambda c:(c[0]<max_iterations)&(~c[4]),iteration,carry)[3]


def run_canopy_energy_jax(area,inputs,factors,soil,initial,parameters=None,*,max_iterations=100,
                           tolerance=None,raise_on_failure=True):
    import jax.numpy as jnp
    arrays=tuple(map(jnp.asarray,(area,inputs,factors,soil,initial)))
    validate_canopy_batch(*arrays,jnp)
    dtype=arrays[0].dtype
    if dtype not in (jnp.float32,jnp.float64) or any(a.dtype!=dtype for a in arrays):
        raise ValueError("Canopy inputs require matching float32/float64 dtype")
    if not isinstance(max_iterations,int) or max_iterations<1:
        raise ValueError("max_iterations must be positive")
    tol=(.005 if dtype==jnp.float32 else 5e-5) if tolerance is None else tolerance
    if not np.isfinite(tol) or tol<=0:
        raise ValueError("tolerance must be positive")
    qtol=1e-8 if dtype==jnp.float32 else 1e-10
    p=jnp.asarray(pack_leaf_parameters(parameters,np.dtype(dtype)))
    result=unpack_canopy_energy(_compiled(max_iterations,float(tol),qtol)(*arrays,p))
    if raise_on_failure and not bool(jnp.all(result["converged"])):
        raise RuntimeError("JAX canopy energy did not converge; inspect with raise_on_failure=False")
    return result


@lru_cache(maxsize=8)
def _compiled(max_iterations,tolerance,q_tolerance):
    import jax
    from functools import partial
    return jax.jit(jax.vmap(partial(canopy_energy_kernel_jax,max_iterations=max_iterations,
        tolerance=tolerance,q_tolerance=q_tolerance),in_axes=(0,0,0,0,0,None)))

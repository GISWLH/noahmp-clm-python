"""JAX shared snow/soil enthalpy equation, up to five snow plus four soil cells."""
from functools import lru_cache
import numpy as np
from .enthalpy import invert_soil_enthalpy_jax


def pack_joint_heat_case(snow,soil,dt,top_flux,*,snow_absorbed=None,soil_absorbed=None,
                         top_temperature=None,top_conductance=0.,thermal_conductivity=None,dtype=np.float64):
    """Packing only: k=0 requests native density/Johansen conductivity.

    Cell columns are mass, H, dz, k, absorbed SW. Active snow precedes four
    soil cells; inactive trailing cells have dz/k=1 and zero mass/energy.
    Properties are porosity/exponent/quartz; options are snow count/phase
    option; boundaries are dt/top flux/air temperature offset/top conductance.
    """
    ns=len(snow.mass);n=ns+4
    if len(soil.water)!=4 or ns>5: raise ValueError('GPU heat requires four soil and at most five snow layers')
    cells=np.zeros((9,5),dtype=dtype);cells[:,2:4]=1.
    cells[:n,0]=np.r_[snow.mass,soil.water];cells[:n,1]=np.r_[snow.enthalpy,soil.enthalpy]
    cells[:n,2]=np.r_[snow.thickness,soil.dz]
    cells[:n,3]=0. if thermal_conductivity is None else np.broadcast_to(thermal_conductivity,(n,))
    cells[:ns,4]=0. if snow_absorbed is None else snow_absorbed
    cells[ns:n,4]=0. if soil_absorbed is None else soil_absorbed
    if top_temperature is None and top_conductance!=0.: raise ValueError('Top conductance requires a temperature')
    p=soil.params
    properties=np.array([p.smcmax,p.bexp,p.quartz],dtype)
    # Independently rounding a saturated water mass, depth and porosity may
    # place that mass one ULP outside the rounded pore volume. Round only
    # such boundary masses toward the interior; keep the conserved H input.
    pore=1000.*cells[ns:n,2].astype(float)*float(properties[0])
    boundary_mass=np.asarray(pore,dtype=dtype)
    boundary_mass=np.where(boundary_mass.astype(float)>pore,np.nextafter(boundary_mass,np.asarray(-np.inf,dtype)),boundary_mass)
    cells[ns:n,0]=np.minimum(cells[ns:n,0],boundary_mass)
    return cells,properties,np.array([ns,soil.freezing_option],np.int32),np.array(
        [dt,top_flux,0. if top_temperature is None else top_temperature-273.15,top_conductance],dtype)


def validate_joint_heat(cells,props,options,boundary,xp):
    n=len(cells)
    if cells.shape!=(n,9,5) or props.shape!=(n,3) or options.shape!=(n,2) or boundary.shape!=(n,4):
        raise ValueError('Joint heat shapes must be (n,9,5), (n,3), (n,2), (n,4)')
    if cells.dtype not in (xp.float32,xp.float64) or props.dtype!=cells.dtype or boundary.dtype!=cells.dtype or options.dtype.kind not in 'iu':
        raise ValueError('Heat arrays require matching float32/float64 and integer options')
    if not all(bool(xp.all(xp.isfinite(v))) for v in (cells,props,boundary)):
        raise ValueError('Nonfinite joint heat input')
    ns,phase=options.T;por,b,quartz=props.T
    if bool(xp.any((ns<0)|(ns>5)|((phase!=1)&(phase!=2))|(por<=0)|(por>=1)|(b<=0)|(quartz<0)|(quartz>1)|
        (boundary[:,0]<=0)|(boundary[:,3]<0))): raise ValueError('Invalid joint heat properties or options')
    index=xp.arange(9)[None,:];active=index<ns[:,None]+4;is_snow=index<ns[:,None]
    w,h,z,k,q=[cells[:,:,i] for i in range(5)]
    if bool(xp.any((w<0)|(z<=0)|(k<0)|(is_snow&(w<=0))|
        (active&~is_snow&(w>1000*z*por[:,None]+8*xp.finfo(cells.dtype).eps*xp.maximum(1.,w)))|
        (~active&((w!=0)|(h!=0)|(q!=0))))): raise ValueError('Invalid snow/soil cell mass, geometry or padding')


def joint_heat_kernel_jax(cells,props,options,boundary):
    import jax.numpy as j
    from jax import lax
    stop=lax.optimization_barrier
    mass,h0,z,user_k,solar=cells.T;por,b,quartz=props;ns,phase=options
    dt,qtop,air,exchange=boundary;index=j.arange(9);active=index<ns+4;snow=index<ns
    rho=j.minimum(mass/z,917.)
    ks=stop(.023+stop(stop(stop(7.75e-5*rho)+stop(stop(1.105e-6*rho)*rho))*j.asarray(2.29-.023,cells.dtype)))
    saturation=j.clip(mass/stop(stop(1000.*z)*por),0.,1.)
    kersten=j.where(saturation>.1,j.clip(stop(stop(j.log(j.maximum(saturation,1e-6)))*.4342944819032518)+1.,0.,1.),0.)
    kg=stop(.2+stop(stop(stop(.6+stop(1.6*quartz))-.2)*kersten))
    k=j.where(user_k>0.,user_k,j.where(snow,ks,kg))
    interface=j.where(index[:-1]<ns+3,1./stop(stop(stop(.5*z[:-1])/k[:-1])+stop(stop(.5*z[1:])/k[1:])),0.)
    up=j.r_[j.zeros_like(dt),interface];down=j.r_[interface,j.zeros_like(dt)]
    def diagnose(h):
        safe_mass=j.where(snow,mass,1.)
        upper=stop(334000.*safe_mass)
        ts=j.where(h<0.,h/stop(2100.*safe_mass),j.where(h>upper,stop(h-upper)/stop(4180.*safe_mass),0.))
        ds=j.where(h<0.,1./stop(2100.*safe_mass),j.where(h>upper,1./stop(4180.*safe_mass),0.))
        t,reconstructed,c,liq,ok,_=invert_soil_enthalpy_jax(j.where(snow,0.,mass),j.where(snow,0.,h),z,por,b,phase)
        eps=j.finfo(cells.dtype).eps
        snow_error=stop(stop(eps*stop(j.abs(h)+j.where(h>upper,upper,0.)))*ds)+stop(2.*eps*j.abs(ts))
        soil_error=stop(j.abs(stop(reconstructed-h))/c)+stop(2.*eps*j.abs(t))
        rounding=j.where(active,j.where(snow,snow_error,soil_error),0.)
        return j.where(active,j.where(snow,ts,t),0.),j.where(active,j.where(snow,ds,1./c),0.),j.where(snow,j.clip(h/334000.,0.,mass),liq),j.all(j.where(active&~snow,ok,True)),rounding
    def evaluate(h):
        t,d,liq,valid,rounding=diagnose(h)
        f=stop(interface*stop(t[:-1]-t[1:]))
        top=stop(qtop+stop(exchange*stop(air-t[0])))
        flux=j.r_[top,f,j.zeros_like(top)]
        net=stop(stop(solar+flux[:-1])-flux[1:])
        residual=j.where(active,stop(stop(h-h0)-stop(dt*net)),0.)
        return residual,t,d,liq,valid,f,top,rounding
    # Each reservoir has its own representation error. A large deep-soil H
    # must never excuse failing to melt a tiny snow cell at the surface.
    scale=j.maximum(j.abs(h0),j.maximum(j.abs(dt*(solar+j.where(index==0,qtop,0.))),1.))
    tolerance=j.maximum(j.asarray(2e-7,cells.dtype),8.*j.finfo(cells.dtype).eps*scale)
    def tolerance_at(h,rounding):
        flux_error=stop(stop(up*stop(rounding+j.r_[j.zeros_like(dt),rounding[:-1]]))+
            stop(down*stop(rounding+j.r_[rounding[1:],j.zeros_like(dt)])))
        flux_error=stop(flux_error+j.where(index==0,stop(exchange*rounding),0.))
        return stop(j.maximum(tolerance,8.*j.finfo(cells.dtype).eps*j.abs(h))+stop(stop(2.*dt)*flux_error))
    def norm_of(residual,h,rounding): return j.max(j.abs(residual)/tolerance_at(h,rounding))
    def merit_of(residual,weights):
        # L1 detects progress in other cells when the maximum-residual cell
        # moves by less than one float32 H ULP on a phase-event step.
        return lax.fori_loop(0,9,lambda i,s:stop(s+stop(j.abs(residual[i])/weights[i])),j.zeros_like(dt))
    def linear_solve(derivative,residual):
        lower=stop(-stop(dt*up)*j.r_[j.zeros_like(dt),derivative[:-1]])
        upper=stop(-stop(dt*down)*j.r_[derivative[1:],j.zeros_like(dt)])
        diagonal=stop(1.+stop(stop(dt*stop(stop(up+down)+j.where(index==0,exchange,0.)))*derivative))
        # M-matrix tridiagonal elimination of the enthalpy Newton Jacobian.
        # This is not the released temperature/AHC Thomas scheme.
        rhs=-residual
        def eliminate(i,carry):
            diag,right=carry;ratio=lower[i]/diag[i-1]
            return diag.at[i].set(stop(diag[i]-stop(ratio*upper[i-1]))),right.at[i].set(stop(right[i]-stop(ratio*right[i-1])))
        diagonal,rhs=lax.fori_loop(1,9,eliminate,(diagonal,rhs))
        delta=j.zeros_like(rhs).at[8].set(rhs[8]/diagonal[8])
        def backward(i,d):
            ix=7-i;return d.at[ix].set(stop(rhs[ix]-stop(upper[ix]*d[ix+1]))/diagonal[ix])
        return lax.fori_loop(0,8,backward,delta)
    def iteration(carry):
        count,h,_,_=carry;res,t,deriv,_,valid,_,_,rounding=evaluate(h)
        # Freeze line-search weights for this Newton direction. Precision
        # bounds jump when a snow cell enters/leaves the latent plateau;
        # rescaling the trial by its own bound can reject a decreasing
        # residual exactly at that phase event and stall indefinitely.
        weights=tolerance_at(h,rounding);norm=j.max(j.abs(res)/weights);merit=merit_of(res,weights)
        upper=stop(334000.*mass)
        # Phase-event steps land exactly. A nearby representable H is still
        # inside its current phase; treating several ULPs as an endpoint
        # skips a real event and can oscillate across the latent plateau.
        at_lower=snow&(h==0.);at_upper=snow&(h==upper)
        def active_set(_,d):
            delta=linear_solve(d,res)
            ds=j.where(at_lower,j.where(delta<0.,1./stop(2100.*j.maximum(mass,1e-30)),0.),d)
            return j.where(at_upper,j.where(delta>0.,1./stop(4180.*j.maximum(mass,1e-30)),0.),ds)
        deriv=lax.fori_loop(0,12,active_set,deriv);delta=linear_solve(deriv,res)
        fraction0=-h/j.where(delta!=0.,delta,1.)
        fraction1=stop(upper-h)/j.where(delta!=0.,delta,1.)
        event0=snow&~at_lower&(delta!=0.)&(fraction0>0.)&(fraction0<=1.)
        event1=snow&~at_upper&(delta!=0.)&(fraction1>0.)&(fraction1<=1.)
        scale=j.minimum(j.min(j.where(event0,fraction0,1.)),j.min(j.where(event1,fraction1,1.)))
        def search(s):
            tries,scale,candidate,accepted=s
            trial=stop(h+stop(scale*delta))
            trial=j.where(event0&(scale==fraction0),0.,j.where(event1&(scale==fraction1),upper,trial))
            r,_,_,_,good,_,_,rounding=evaluate(trial)
            take=good&((merit_of(r,weights)<merit)|(norm_of(r,trial,rounding)<=1.))
            return tries+1,scale*.5,j.where(take,trial,candidate),take
        _,_,trial,accepted=lax.while_loop(lambda s:(s[0]<40)&~s[3],search,
            (j.asarray(0),scale,h,j.asarray(False)))
        next_h=j.where(norm<=1.,h,trial)
        next_result=evaluate(next_h)
        done=norm_of(next_result[0],next_h,next_result[7])<=1.
        return count+1,next_h,done,valid&(accepted|done)
    initial_res=evaluate(h0);done=norm_of(initial_res[0],h0,initial_res[7])<=1.
    count,h,done,valid=lax.while_loop(lambda c:(c[0]<80)&~c[2]&c[3],iteration,
        (j.asarray(0),h0,done,initial_res[4]))
    res,t,deriv,liquid,ok,flux,top,rounding=evaluate(h)
    summed=lax.fori_loop(0,9,lambda i,s:stop(s+j.where(active[i],solar[i],0.)),j.zeros_like(dt))
    change=lax.fori_loop(0,9,lambda i,s:stop(s+stop(h[i]-h0[i])),j.zeros_like(dt))
    incoming=stop(dt*stop(top+summed));budget=stop(change-incoming)
    final_tolerance=tolerance_at(h,rounding)
    return (h,j.where(active,t+273.15,0.),t,liquid,mass-liquid,res,flux,final_tolerance,
        j.stack((incoming,budget,top,count.astype(cells.dtype)+1.,(done&valid&ok).astype(cells.dtype),j.max(final_tolerance))))


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(joint_heat_kernel_jax))


def unpack_joint_heat(result):
    h,t,offset,liq,ice,res,flux,tolerance,diag=result
    return dict(enthalpy_jm2=h,temperature_k=t,temperature_offset_k=offset,liquid_kgm2=liq,ice_kgm2=ice,
        layer_residual_jm2=res,interface_flux_wm2=flux,input_energy_jm2=diag[:,0],energy_residual_jm2=diag[:,1],
        top_flux_wm2=diag[:,2],iterations=diag[:,3],converged=diag[:,4]>0,convergence_tolerance_jm2=diag[:,5],
        layer_tolerance_jm2=tolerance)


def run_joint_heat_jax(cells,props,options,boundary):
    import jax.numpy as j
    c,p,o,b=[j.asarray(v) for v in (cells,props,options,boundary)]
    validate_joint_heat(c,p,o,b,j)
    result=unpack_joint_heat(_compiled()(c,p,o,b))
    if not bool(j.all(result['converged'])): raise RuntimeError('JAX joint snow/soil enthalpy solve failed')
    return result

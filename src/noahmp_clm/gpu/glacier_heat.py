"""Device pure-water finite-volume heat equation for up to nine snow/ice cells.

The phase-event/Newton algorithm matches the independently tested snow/soil
solver. This separate evaluator has no mineral heat capacity or unfrozen-water
curve; every active cell is pure water. Glacier storage/exports live in glacier.
"""
def glacier_heat_kernel_jax(cells,n,boundary):
    import jax.numpy as j
    from jax import lax
    stop=lax.optimization_barrier
    mass,h0,z,k,solar=cells.T
    dt,qtop,air,exchange=boundary;index=j.arange(9);active=index<n;snow=active
    interface=j.where(index[:-1]<n-1,1./stop(stop(stop(.5*z[:-1])/k[:-1])+stop(stop(.5*z[1:])/k[1:])),0.)
    up=j.r_[j.zeros_like(dt),interface];down=j.r_[interface,j.zeros_like(dt)]
    def diagnose(h):
        safe_mass=j.where(snow,mass,1.)
        upper=stop(334000.*safe_mass)
        ts=j.where(h<0.,h/stop(2100.*safe_mass),j.where(h>upper,stop(h-upper)/stop(4180.*safe_mass),0.))
        ds=j.where(h<0.,1./stop(2100.*safe_mass),j.where(h>upper,1./stop(4180.*safe_mass),0.))
        eps=j.finfo(cells.dtype).eps
        snow_error=stop(stop(eps*stop(j.abs(h)+j.where(h>upper,upper,0.)))*ds)+stop(2.*eps*j.abs(ts))
        rounding=j.where(active,snow_error,0.)
        return (j.where(active,ts,0.),j.where(active,ds,0.),
            j.where(active,j.clip(h/334000.,0.,mass),0.),j.asarray(True),rounding)
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

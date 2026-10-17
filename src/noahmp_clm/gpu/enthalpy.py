"""JAX thermodynamic primitives for the new extensive snow/soil state.

Internal temperatures are offsets from the physical freezing point. This
avoids losing sub-Kelvin phase information while inverting H in float32;
the released AHC/Thomas scheme is not changed or reformulated here.
"""
from functools import lru_cache

SOIL_THERMODYNAMIC_OUTPUTS=('temperature_offset_k','temperature_k','liquid_kgm2','ice_kgm2',
    'enthalpy_jm2','capacity_jm2k','reconstruction_residual_jm2','iterations','converged')


def validate_soil_thermodynamics(inputs,options,inverse,xp):
    if inputs.ndim!=2 or inputs.shape[1]!=6 or options.shape!=(len(inputs),):
        raise ValueError('Soil thermodynamics requires (n,6) inputs including suction and (n,) phase options')
    if inputs.dtype not in (xp.float32,xp.float64) or options.dtype.kind not in 'iu':
        raise ValueError('Soil inputs require float32/float64 and integer phase options')
    if not bool(xp.all(xp.isfinite(inputs))) or bool(xp.any((options!=1)&(options!=2))):
        raise ValueError('Nonfinite soil thermodynamics or unknown freezing option')
    w,q,z,por,b,psi=inputs.T
    if bool(xp.any((w<0.)|(z<=0.)|(por<=0.)|(por>=1.)|(b<=0.)|
        (psi<1e-12)|(psi>1e6)|(w>1000.*z*por+8.*xp.finfo(inputs.dtype).eps*xp.maximum(1.,w)))):
        raise ValueError('Invalid soil mass, depth, porosity or exponent')
    if not inverse and bool(xp.any((q<-123.15)|(q>126.85))):
        raise ValueError('Soil temperature outside 150--400 K')


def soil_thermodynamics_jax(water,offset,dz,porosity,bexp,saturated_suction_m,option):
    """Pure device expression returning H, dH/dT and liquid mass."""
    import jax.numpy as j
    from jax import lax
    # Barriers make the separately specified device arithmetic order explicit
    # around the compact cap and its derivatives; no host NumPy evaluation.
    stop=lax.optimization_barrier
    def cap(total,potential):
        width=j.maximum(stop(.04*total),1e-12)
        dw=j.where(.04*total>1e-12,1.,0.)*.04
        diff=stop(total-potential);sign=j.sign(diff)
        rounded=j.maximum(stop(width-j.abs(diff)),0.)
        value=stop(j.minimum(total,potential)-stop(rounded*rounded)/stop(4.*width))
        dtotal=stop(j.where(diff<0.,1.,j.where(diff>0.,0.,.5))-
            stop(rounded*stop(dw-sign))/stop(2.*width))
        dtotal=stop(dtotal+stop(stop(rounded*rounded)*dw)/stop(stop(4.*width)*width))
        dpotential=stop(j.where(diff>0.,1.,j.where(diff<0.,0.,.5))-
            stop(rounded*sign)/stop(2.*width))
        return j.maximum(value,0.),j.where(value>0.,dtotal,0.),j.where(value>0.,dpotential,0.)
    theta=water/stop(1000.*dz)
    temperature=stop(273.15+offset)
    coefficient=stop(334000./stop(9.80616*saturated_suction_m))
    raw=stop(stop(coefficient*j.maximum(-offset,0.))/temperature);ratio=j.maximum(raw,1e-12)
    potential=j.exp(j.minimum(stop(j.log(porosity)-stop(j.log(ratio)/bexp)),j.log(stop(2.*porosity))))
    denominator=stop(stop(stop(bexp*temperature)*temperature)*ratio)
    dp=j.where(raw>1e-12,stop(stop(stop(potential*coefficient)*273.15)/denominator),0.)
    liquid,_,dpotential=cap(theta,potential);derivative=stop(dpotential*dp)
    def adjust(_,carry):
        liq,deriv=carry
        limited,ds,dp_=cap(liq,potential)
        slope=stop(stop(ds*deriv)+stop(dp_*dp))
        slope=j.where(limited<theta,slope,0.)
        return stop(.5*stop(liq+j.minimum(theta,limited))),stop(.5*stop(deriv+slope))
    corrected=lax.fori_loop(0,3,adjust,(liquid,derivative))
    liquid=j.where(option==2,corrected[0],liquid)
    derivative=j.where(option==2,corrected[1],derivative)
    liquid=j.where(offset>=0.,water,j.clip(stop(liquid*stop(1000.*dz)),0.,water))
    derivative=stop(j.where(offset>=0.,0.,derivative)*stop(1000.*dz))
    dry=stop(stop(stop(1.-porosity)*2e6)*dz)
    sensible=stop(stop(dry+stop(2100.*water))+stop(2080.*liquid))
    enthalpy=stop(stop(sensible*offset)+stop(334000.*liquid))
    capacity=stop(sensible+stop(stop(stop(2080.*offset)+334000.)*derivative))
    return enthalpy,capacity,liquid


def invert_soil_enthalpy_jax(water,enthalpy,dz,porosity,bexp,saturated_suction_m,option):
    """Safeguarded H inversion; returns offset, H/C/liquid, status and count."""
    import jax.numpy as j
    from jax import lax
    stop=lax.optimization_barrier
    low=j.full_like(water,-123.15);high=j.full_like(water,126.85)
    hlo=soil_thermodynamics_jax(water,low,dz,porosity,bexp,saturated_suction_m,option)[0]
    hhi=soil_thermodynamics_jax(water,high,dz,porosity,bexp,saturated_suction_m,option)[0]
    valid=(enthalpy>=hlo)&(enthalpy<=hhi)
    dry=stop(stop(stop(1.-porosity)*2e6)*dz)
    t=j.clip(stop(enthalpy-stop(334000.*water))/stop(dry+stop(4180.*water)),low,high)
    def tolerance(t,h,c,liquid):
        # H is authoritative. Bound floating evaluation of its two terms,
        # including cancellation near H=0; never replace the conserved H.
        scale=j.maximum(j.maximum(j.abs(enthalpy),j.abs(stop(334000.*liquid))),1.)
        return j.maximum(j.asarray(2e-8,water.dtype),stop(8.*j.finfo(water.dtype).eps*scale))
    def advance(_,carry):
        t,low,high,done,count=carry
        h,c,liq=soil_thermodynamics_jax(water,t,dz,porosity,bexp,saturated_suction_m,option)
        error=stop(h-enthalpy);finished=j.abs(error)<=tolerance(t,h,c,liq)
        count=count+j.where(~done,1.,0.)
        done=done|finished
        low=j.where(error<0.,t,low);high=j.where(error>0.,t,high)
        trial=stop(t-error/c)
        trial=j.where((trial>low)&(trial<high),trial,stop(.5*stop(low+high)))
        return j.where(done,t,trial),low,high,done,count
    initial=(t,low,high,j.zeros_like(water,dtype=bool),j.zeros_like(water))
    _,solved=lax.while_loop(lambda x:(x[0]<64)&j.any(~x[1][3]),
        lambda x:(x[0]+1,advance(x[0],x[1])),(j.asarray(0),initial))
    t,_,_,_,count=solved
    h,c,liq=soil_thermodynamics_jax(water,t,dz,porosity,bexp,saturated_suction_m,option)
    converged=valid&(j.abs(h-enthalpy)<=tolerance(t,h,c,liq))
    return t,h,c,liq,converged,count


def soil_thermodynamics_kernel_jax(inputs,option,*,inverse):
    import jax.numpy as j
    w,q,z,por,b,psi=inputs
    if inverse:
        t,h,c,liq,ok,count=invert_soil_enthalpy_jax(w,q,z,por,b,psi,option)
        residual=h-q
    else:
        t=q;h,c,liq=soil_thermodynamics_jax(w,t,z,por,b,psi,option)
        ok=j.asarray(True);count=j.zeros_like(w);residual=j.zeros_like(w)
    return j.stack((t,t+273.15,liq,w-liq,h,c,residual,count,ok.astype(w.dtype)))


@lru_cache(maxsize=2)
def _compiled_soil(inverse):
    import jax
    return jax.jit(jax.vmap(lambda u,o:soil_thermodynamics_kernel_jax(u,o,inverse=inverse)))


def run_soil_thermodynamics_jax(inputs,options,*,inverse=False):
    """Rows: water kg/m2, T-273.15 K (or H J/m2), dz m, porosity, B, suction m.

    Returns device arrays. The inverse reports a reconstruction error but
    never changes the caller's authoritative enthalpy. Bad domain/inversion
    raises; it cannot silently return a clipped state.
    """
    import jax.numpy as j
    u,o=j.asarray(inputs),j.asarray(options)
    validate_soil_thermodynamics(u,o,inverse,j)
    result=_compiled_soil(inverse)(u,o)
    if not bool(j.all(result[:,-1]>0.)):
        raise ValueError('Soil enthalpy is outside the physical domain or inversion did not converge')
    return dict(zip(SOIL_THERMODYNAMIC_OUTPUTS,result.T))

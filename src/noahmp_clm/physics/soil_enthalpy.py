"""Mass/enthalpy soil state for conservative coupling to the five-layer snow.

Enthalpy is relative to dry soil and ice at 273.15 K. Liquid water therefore
carries fusion enthalpy even above freezing. The retained Niu--Yang/Koren
water-equilibrium options are differentiated analytically; this is a new
physical energy ledger, not a rewrite of the released Thomas solver in delta T.
"""
from dataclasses import dataclass
import numpy as np
from .snowpack import TF,CI,CW,LF
from .clm_freezing import _freezing_potential


def _cap_partials(total,potential):
    width=np.maximum(.04*total,1e-12)
    dw=np.where(.04*total>1e-12,.04,0.)
    diff=total-potential;sign=np.sign(diff)
    rounded=np.maximum(width-np.abs(diff),0.)
    value=np.minimum(total,potential)-rounded**2/(4.*width)
    dtotal=np.where(diff<0,1.,np.where(diff>0,0.,.5))-rounded*(dw-sign)/(2*width)+rounded**2*dw/(4*width**2)
    dpotential=np.where(diff>0,1.,np.where(diff<0,0.,.5))-rounded*sign/(2*width)
    return np.maximum(value,0.),np.where(value>0,dtotal,0.),np.where(value>0,dpotential,0.)


def liquid_and_derivative(theta,temperature,params,option=1):
    """Liquid water equivalent volume and d(liquid)/dT, m3/m3 and K-1.

    CLM5Eq2.6.52 uses the explicit positive saturated suction head. Its compact
    smooth cap is retained. Option2 retains the historical simplified iteration;
    it is not an independently validated implementation of full Koren1999.
    """
    theta,t=np.broadcast_arrays(np.asarray(theta,float),np.asarray(temperature,float))
    if option not in (1,2): raise ValueError('Unknown soil unfrozen-water option')
    b=max(params.bexp,1e-6)
    potential,dp=_freezing_potential(t,params.smcmax,b,params.saturated_suction_m)
    liquid,_,dpot=_cap_partials(theta,potential)
    deriv=dpot*dp
    if option==2:
        for _ in range(3):
            limited,ds,dp_= _cap_partials(liquid,potential)
            derivative_limited=ds*deriv+dp_*dp
            capped=np.minimum(theta,limited)
            derivative_limited=np.where(limited<theta,derivative_limited,0.)
            liquid=.5*(liquid+capped);deriv=.5*(deriv+derivative_limited)
    return np.where(t>=TF,theta,liquid),np.where(t>=TF,0.,deriv)


def enthalpy_and_capacity(water,temperature,dz,params,option=1):
    """Extensive enthalpy J/m2 and its local derivative J/m2/K."""
    water,t,z=np.broadcast_arrays(np.asarray(water,float),temperature,dz)
    liquid,deriv=liquid_and_derivative(water/(1000.*z),t,params,option)
    liquid=liquid*1000.*z;deriv=deriv*1000.*z
    offset=t-TF
    dry=(1.-params.smcmax)*2e6*z
    sensible=dry+CI*water+(CW-CI)*liquid
    h=sensible*offset+LF*liquid
    capacity=sensible+((CW-CI)*offset+LF)*deriv
    return h,capacity


def temperature_from_enthalpy(water,enthalpy,dz,params,option=1):
    """Invert the monotone physical soil energy curve with safeguarded Newton.

    Fixed 150--400 K brackets are input-domain checks, never energy clipping.
    The returned temperatures reconstruct the requested energy to roundoff.
    """
    water,h,z=np.broadcast_arrays(np.asarray(water,float),np.asarray(enthalpy,float),np.asarray(dz,float))
    if (not all(np.all(np.isfinite(v)) for v in (water,h,z)) or np.any(water<0) or np.any(z<=0)
            or not 0<params.smcmax<1 or np.any(water>1000.*params.smcmax*z+1e-9)):
        raise ValueError('Invalid soil water mass, enthalpy, depth or porosity')
    lower=np.full(h.shape,150.);upper=np.full(h.shape,400.)
    hlo=enthalpy_and_capacity(water,lower,z,params,option)[0]
    hhi=enthalpy_and_capacity(water,upper,z,params,option)[0]
    if np.any(h<hlo) or np.any(h>hhi): raise ValueError('Soil enthalpy outside the 150--400 K physical domain')
    dry=(1.-params.smcmax)*2e6*z
    t=np.clip(TF+(h-LF*water)/(dry+CW*water),lower,upper)
    tolerance=np.maximum(2e-8,32.*np.finfo(float).eps*np.maximum(np.abs(h),1.))
    for _ in range(64):
        value,cap=enthalpy_and_capacity(water,t,z,params,option)
        error=value-h
        resolution=4.*np.spacing(np.abs(t))*cap
        done=np.abs(error)<=np.maximum(tolerance,resolution)
        if np.all(done): break
        lower=np.where(error<0.,t,lower);upper=np.where(error>0.,t,upper)
        trial=t-error/cap
        trial=np.where((trial>lower)&(trial<upper),trial,.5*(lower+upper))
        t=np.where(done,t,trial)
    final=enthalpy_and_capacity(water,t,z,params,option)[0]-h
    # Kelvin representation has a finite ULP times heat capacity, especially
    # in a steep freezing curve. Accept that resolution, not physical clipping.
    rounding=4.*np.spacing(np.abs(t))*enthalpy_and_capacity(water,t,z,params,option)[1]
    if np.any(np.abs(final)>np.maximum(tolerance,rounding)):
        raise RuntimeError(f'Soil enthalpy inverse failed: {np.max(np.abs(final)):.9g} J/m2')
    return t


@dataclass(frozen=True)
class SoilEnthalpyState:
    water: np.ndarray
    enthalpy: np.ndarray
    dz: np.ndarray
    params: object
    freezing_option: int=1

    def __post_init__(self):
        w,h,z=[np.array(v,float,copy=True) for v in (self.water,self.enthalpy,self.dz)]
        if w.ndim!=1 or not w.size or h.shape!=w.shape or z.shape!=w.shape:
            raise ValueError('Soil enthalpy state requires matching nonempty layers')
        temperature_from_enthalpy(w,h,z,self.params,self.freezing_option)
        for name,value in (('water',w),('enthalpy',h),('dz',z)):
            value.setflags(write=False);object.__setattr__(self,name,value)

    @classmethod
    def from_column(cls,state):
        water=1000.*np.asarray(state.smc)*state.dz
        h=enthalpy_and_capacity(water,state.tsoil,state.dz,state.soil,state.options.OptSoilSupercoolWater)[0]
        return cls(water,h,state.dz,state.soil,state.options.OptSoilSupercoolWater)

    @property
    def temperature(self):
        return temperature_from_enthalpy(self.water,self.enthalpy,self.dz,self.params,self.freezing_option)

    @property
    def liquid(self):
        return liquid_and_derivative(self.water/(1000*self.dz),self.temperature,self.params,
            self.freezing_option)[0]*1000*self.dz

    @property
    def ice(self): return self.water-self.liquid

    def with_enthalpy(self,enthalpy):
        return SoilEnthalpyState(self.water,enthalpy,self.dz,self.params,self.freezing_option)


def infiltrate(state,mass_kgm2,enthalpy_jm2):
    """Fill pore water capacity top-down, carrying the input water's enthalpy.

    Returns (new_state, runoff_mass, runoff_enthalpy). Zero input mass cannot
    silently dispose of a nonzero heat input: use a thermal boundary instead.
    This transfer primitive is not the final runoff/infiltration scheme.
    """
    if (not np.isfinite(mass_kgm2) or not np.isfinite(enthalpy_jm2) or mass_kgm2<0
            or (mass_kgm2==0 and enthalpy_jm2!=0)):
        raise ValueError('Invalid water mass/enthalpy input')
    if mass_kgm2==0: return state,0.,0.
    specific=enthalpy_jm2/mass_kgm2
    w,h=state.water.copy(),state.enthalpy.copy();left=mass_kgm2
    for j in range(w.size):
        accepted=min(left,max(0.,1000*state.params.smcmax*state.dz[j]-w[j]))
        w[j]+=accepted;h[j]+=accepted*specific;left-=accepted
    return SoilEnthalpyState(w,h,state.dz,state.params,state.freezing_option),float(left),float(left*specific)


def withdraw_liquid(state,demand_kgm2):
    """Remove diagnosed liquid mass and its sensible+fusion enthalpy per layer.

    Atmospheric latent heat is handled separately by the surface solver. An
    overdraw raises rather than silently changing a solved latent heat flux.
    """
    demand=np.asarray(demand_kgm2,float)
    t=state.temperature;available=state.liquid
    if (demand.shape!=state.water.shape or not np.all(np.isfinite(demand)) or np.any(demand<0)
            or np.any(demand>available+1e-10)):
        raise ValueError('Liquid withdrawal exceeds available soil water')
    heat=demand*(LF+CW*(t-TF))
    new=SoilEnthalpyState(state.water-demand,state.enthalpy-heat,state.dz,state.params,state.freezing_option)
    return new,heat

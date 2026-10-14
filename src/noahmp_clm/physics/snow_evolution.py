"""Snow compaction and grain evolution, based on CLM5 technical-note equations.

These routines do not change mass, enthalpy or deposited impurity mass. Call
aging/compaction before remapping if using a per-layer pre-melt ice history.
No drifting-snow redistribution is represented in this single-column module.
"""
import numpy as np
from .snowpack import Snowpack, TF, RHO_ICE, RHO_WATER, regrid
from .snicar import snow_tables


def fresh_snow_properties(air_temperature,wind10=0.):
    """CLM5 2.8.9--10 density and 2.3.72 fresh-grain temperature relationship."""
    t,u=np.broadcast_arrays(np.asarray(air_temperature,float),np.asarray(wind10,float))
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(u)) or np.any(u<0) or np.any((t<150)|(t>330)):
        raise ValueError('Invalid snowfall air temperature or 10 m wind')
    c=t-TF
    thermal=np.where(c>-15.,50.+1.7*np.maximum(np.minimum(c,2.)+15.,0.)**1.5,
        -3.833*c-.0333*c*c)
    wind=np.where(u>.1,266.861*((1.+np.tanh(u/5.))/2.)**8.8,0.)
    density=np.clip(thermal+wind,30.,RHO_ICE)
    radius=54.5+5.*np.clip(c+30.,0.,30.)
    return density,radius


def compact(pack,dt,*,ice_before=None,remap=True):
    """Metamorphism, overburden and melt collapse; preserve extensive budgets.

    Uses CLM5 2.8.33--39 rates. Integrate frozen rates exponentially to prevent
    negative depth for a long step; geometry cannot fall below ice+water volume.
    ``ice_before`` is mass before the current phase-change calculation in the
    same grid. No drift or fractional snow-cover compaction is added.
    """
    if not np.isfinite(dt) or dt<0: raise ValueError('Invalid compaction time step')
    if not pack.mass.size: return pack
    t,ice,liquid=pack.temperature,pack.ice,pack.liquid
    rho=pack.density;rhoice=ice/pack.thickness
    c1=np.exp(-.046*np.maximum(rhoice-175.,0.))
    c2=np.where(liquid/pack.thickness>.01,2.,1.)
    destructive=2.777e-6*c1*c2*np.exp(-.04*(TF-t))
    overburden=np.cumsum(pack.mass)-.5*pack.mass
    viscosity=4.*7.62237e6*(rho/450.)*np.exp(.1*(TF-t)+.023*rho)/(
        1.+60.*liquid/(RHO_WATER*pack.thickness))
    rate=destructive+overburden/viscosity
    solid_volume=ice/RHO_ICE+liquid/RHO_WATER
    allowed=(1.-solid_volume/pack.thickness>.001)&(ice>.1)
    factor=np.exp(-dt*np.where(allowed,rate,0.))
    if ice_before is not None:
        previous=np.asarray(ice_before,float)
        if previous.shape!=ice.shape or np.any(previous<0) or not np.all(np.isfinite(previous)):
            raise ValueError('Pre-melt ice must match the current grid')
        melt_fraction=np.divide(ice,previous,out=np.ones_like(ice),where=previous>0)
        factor*=np.where(allowed,np.minimum(1.,melt_fraction),1.)
    z=np.maximum(pack.thickness*factor,solid_volume)
    new=Snowpack(pack.mass,pack.enthalpy,z,pack.radius_um,pack.impurities)
    return regrid(new) if remap else new


def aging_parameters(temperature,gradient,density):
    """Trilinear interpolation in the supplied UCAR table, clipped to its domain.

    The old file's unit attributes say tau=hours and drdsdt0=m2/kg/hour; the
    description and official SnowSnicarMod usage treat them as the length
    scale and radius growth in micrometres/hour in CLM5 eq. 2.3.73. We retain
    the original data and document this metadata discrepancy, not rescale it
    as specific surface area. Smoother interpolation replaces nearest indices.
    """
    t,g,r=np.broadcast_arrays(temperature,gradient,density)
    if not all(np.all(np.isfinite(x)) for x in (t,g,r)) or np.any(g<0) or np.any(r<=0):
        raise ValueError('Invalid snow aging conditions')
    table=snow_tables();coordinates=[]
    for key,value in zip(('T','dTdz','sno_dns'),(t,g,r)):
        axis=table['aging_'+key];x=np.clip(value,axis[0],axis[-1])
        index=np.clip(np.searchsorted(axis,x,side='right')-1,0,len(axis)-2)
        f=(x-axis[index])/(axis[index+1]-axis[index])
        coordinates.append((index,f))
    result=[]
    for name in ('tau','kappa','drdsdt0'):
        values=np.zeros(t.shape)
        for i in range(8):
            indices=[];weight=np.ones(t.shape)
            for d,(index,f) in enumerate(coordinates):
                high=(i>>d)&1;indices.append(index+high);weight*=f if high else 1.-f
            values+=weight*table['aging_'+name][tuple(indices)]
        result.append(values)
    return tuple(result)


def _dry_growth(radius,hours,tau,kappa,initial_rate):
    # Exact integral of dr/dt = a*(tau/(r-r0+tau))**(1/kappa).
    # Logarithmic form handles the table's tau down to 2.4e-32 and kappa
    # down to .0016 without overflow or subtracting nearly equal large radii.
    if hours==0.: return radius
    y=radius-54.5+tau;p=1.+1./kappa
    log_increment=np.log(p*initial_rate*hours)-np.log(y)+np.log(tau/y)/kappa
    increase=y*np.expm1(np.logaddexp(0.,log_increment)/p)
    return radius+increase


def grain_growth(pack,dt,*,temperature_gradient=0.,refrozen_mass=None):
    """Dry/wet aging with Strang splitting and optical-area refreezing mixture.

    Gradient is K/m at each layer. The driver must compute it from physical
    interface temperatures. The all-zero default is an isothermal assumption.
    Wet growth caps liquid fraction at 0.1, as in the official implementation.
    Fresh snowfall already enters via ``add_snowfall``; do not add it twice.
    """
    if not np.isfinite(dt) or dt<0: raise ValueError('Invalid grain growth time step')
    if not pack.mass.size: return pack
    gradient=np.broadcast_to(temperature_gradient,pack.mass.shape)
    tau,kappa,rate=aging_parameters(pack.temperature,gradient,pack.density)
    r=np.maximum(pack.radius_um,54.5)
    r=_dry_growth(r,dt/7200.,tau,kappa,rate)
    liquid_fraction=np.minimum(.1,pack.liquid/pack.mass)
    wet_coefficient=1e18*4.22e-13*liquid_fraction**3/(4.*np.pi)
    r=np.cbrt(r**3+3.*wet_coefficient*dt)
    r=_dry_growth(r,dt/7200.,tau,kappa,rate)
    if refrozen_mass is not None:
        mass=np.asarray(refrozen_mass,float)
        if (mass.shape!=pack.mass.shape or np.any(mass<0) or np.any(mass>pack.ice+1e-12)
                or not np.all(np.isfinite(mass))):
            raise ValueError('Invalid refrozen snow mass')
        fraction=mass/pack.mass
        r=1./((1.-fraction)/r+fraction/1000.)
    r=np.clip(r,30.,1500.)
    return Snowpack(pack.mass,pack.enthalpy,pack.thickness,r,pack.impurities)


def deposit_impurities(pack,rates_kgm2s,dt):
    """Deposit climatological BC/four dust rates; return state and bare residue.

    Call after drainage so new deposition remains in the top snow layer for
    the next radiation solve. An empty column returns deposition to the caller
    as a bare-surface sink, so the impurity ledger has no untracked loss.
    """
    rates=np.asarray(rates_kgm2s,float)
    if rates.shape!=(5,) or not np.all(np.isfinite(rates)) or np.any(rates<0) or not np.isfinite(dt) or dt<0:
        raise ValueError('Invalid BC/dust deposition or time step')
    addition=rates*dt
    if not pack.mass.size: return pack,addition
    a=pack.impurities.copy();a[0]+=addition
    return Snowpack(pack.mass,pack.enthalpy,pack.thickness,pack.radius_um,a),np.zeros(5)

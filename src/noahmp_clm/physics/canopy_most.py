"""CLM surface-layer transfer driven by actual sensible and water mass flux.

Equations 2.5.7--45 and 2.5.116--122, CLM5 technical note (release-clm5.0).
The profile integrals are written from those published equations, not copied
from CTSM source. Separate measurement heights and lower roughness limits are
retained. These building blocks do not themselves advance a column state.
"""
from dataclasses import dataclass
import math
from .energy import KARMAN, CP, RD

GRAVITY = 9.81


@dataclass(frozen=True)
class MOSTGeometry:
    """Heights above ground and roughness lengths, all metres.

    Site metadata must supply measurement heights and displacement explicitly
    for tower evaluation. Defaults only describe a short reference canopy.
    """
    wind_height: float = 10.
    temperature_height: float = 10.
    humidity_height: float = 10.
    displacement: float = 0.
    momentum_roughness: float = .05
    heat_roughness: float = .005
    water_roughness: float = .005
    boundary_layer_height: float = 1000.
    ground_roughness: float = .01

    def __post_init__(self):
        if not all(math.isfinite(x) for x in vars(self).values()):
            raise ValueError('Nonfinite MOST geometry')
        if self.displacement < 0 or self.boundary_layer_height <= 0 or self.ground_roughness <= 0:
            raise ValueError('Invalid MOST height or roughness')
        for height, roughness in ((self.wind_height,self.momentum_roughness),
                                  (self.temperature_height,self.heat_roughness),
                                  (self.humidity_height,self.water_roughness)):
            if not 0 < roughness < height-self.displacement:
                raise ValueError('Measurement must be above displacement plus roughness')


def _unstable_psi(zeta, momentum):
    x=(1.-16.*zeta)**.25
    if momentum:
        return 2.*math.log((1.+x)/2.)+math.log((1.+x*x)/2.)-2.*math.atan(x)+math.pi/2.
    return 2.*math.log((1.+x*x)/2.)


def profile_integral(height, roughness, inverse_length, *, momentum=False):
    """Positive integral of phi(z/L) d(log z) from z0 to height.

    Analytic primitives cover both endpoints even when roughness itself is
    outside the weakly stable/unstable branch. The CLM integral coefficients
    1.14 and 0.8 are the published rounded constants of equations 32 and 37.
    """
    if (not all(math.isfinite(v) for v in (height,roughness,inverse_length))
            or not 0 < roughness < height):
        raise ValueError('Invalid MOST profile interval')
    def primitive(z):
        zeta=z*inverse_length
        if zeta >= 0.:
            correction=5.*zeta if zeta<=1. else 4.+zeta+4.*math.log(zeta)
        else:
            edge=-1.574 if momentum else -.465
            if zeta>=edge:correction=-_unstable_psi(zeta,momentum)
            else:
                tail=(1.14*((-zeta)**(1./3.)-(-edge)**(1./3.)) if momentum
                      else .8*((-edge)**(-1./3.)-(-zeta)**(-1./3.)))
                correction=math.log(edge/zeta)-_unstable_psi(edge,momentum)+tail
        return math.log(z)+correction
    return primitive(height)-primitive(roughness)


def flux_stability(temperature,humidity,pressure,wind,sensible_heat,water_vapor,
                   friction_velocity,geometry):
    """Return buoyancy, convective wind and bounded target (zm-d)/L.

    H is W/m², E is kg/m²/s, positive out of the column. E must be actual
    vapour mass, including sublimation; LE/Lv is not interchangeable with E.
    Near neutral remains continuous, rather than forcing |zeta| >= 0.01.
    Bounds -100 and +2 follow the CLM stability envelope; raw zeta is returned
    so a saturated bound cannot be mistaken for exact unbounded closure.
    """
    values=(temperature,humidity,pressure,wind,sensible_heat,water_vapor,friction_velocity)
    if (not all(math.isfinite(v) for v in values) or not 200<=temperature<=350
            or not 0<=humidity<.2 or pressure<30000 or wind<0 or friction_velocity<=0):
        raise ValueError('Invalid MOST atmosphere or turbulent flux')
    theta=temperature+.0098*geometry.temperature_height
    virtual=theta*(1.+.61*humidity)
    vapor_pressure=humidity*pressure/(.622+.378*humidity)
    density=(pressure-.378*vapor_pressure)/(RD*temperature)
    virtual_flux=((1.+.61*humidity)*sensible_heat/CP+.61*theta*water_vapor)/density
    buoyancy=GRAVITY*virtual_flux/virtual
    convective=max(0.,buoyancy*geometry.boundary_layer_height)**(1./3.)
    effective=max(1.,math.hypot(wind,convective))
    raw=-(geometry.wind_height-geometry.displacement)*KARMAN*buoyancy/friction_velocity**3
    return dict(zeta=max(-100.,min(2.,raw)),raw_zeta=raw,buoyancy=buoyancy,
                convective_wind=convective,effective_wind=effective,
                air_density=density,potential_temperature=theta)


def canopy_exchange(zeta,effective_wind,geometry,*,leaf_dimension=.04,plant_area=3.):
    """Return MOST heat/water velocities and u*-dependent canopy resistances."""
    if (not all(math.isfinite(v) for v in (zeta,effective_wind,leaf_dimension,plant_area))
            or not -100<=zeta<=2 or effective_wind<1. or leaf_dimension<=0 or plant_area<0):
        raise ValueError('Invalid MOST stability, wind or canopy dimensions')
    zm=geometry.wind_height-geometry.displacement
    inv=zeta/zm
    dm=profile_integral(zm,geometry.momentum_roughness,inv,momentum=True)
    dh=profile_integral(geometry.temperature_height-geometry.displacement,geometry.heat_roughness,inv)
    dq=profile_integral(geometry.humidity_height-geometry.displacement,geometry.water_roughness,inv)
    ustar=KARMAN*effective_wind/dm
    leaf=.01*math.sqrt(ustar/leaf_dimension)
    sparse=math.exp(-plant_area)
    soil_coefficient=KARMAN/.13*(geometry.ground_roughness*ustar/1.5e-5)**(-.45)
    ground=ustar*(soil_coefficient*sparse+.004*(1.-sparse))
    return dict(friction_velocity=ustar,heat_velocity=KARMAN*ustar/dh,
                water_velocity=KARMAN*ustar/dq,leaf_velocity=leaf,ground_velocity=ground,
                momentum_integral=dm,heat_integral=dh,water_integral=dq)


def solve_surface_most(solve_surface,exchange,wind,geometry,*,leaf_dimension=.04,
                       plant_area=3.,time_step,max_iterations=50):
    """Iterate transfer with H/E from a pure, fixed-old-state surface solve.

    ``solve_surface(exchange)`` returns snow, soil, canopy mass, diagnostics.
    Every trial starts from the SAME old extensive stores. Only the final
    converged trial is committed; precipitation/unloading must stay outside
    this iteration. A failed iteration raises without advancing any store.
    """
    from dataclasses import replace
    import numpy as np
    if not isinstance(max_iterations,int) or max_iterations<2:
        raise ValueError('MOST requires at least two surface solves')
    if not math.isfinite(time_step) or time_step<=0:raise ValueError('Positive MOST time step required')
    zeta=0.;effective=max(1.,float(wind));previous=None;history=[]
    relaxation=.5;previous_stability_residual=None;previous_mismatch=None
    stable_bracket=False;lower=0.;upper=2.;bisection_restarts=0
    theta=exchange.air_temperature+.0098*geometry.temperature_height
    for iteration in range(1,max_iterations+1):
        velocities=canopy_exchange(zeta,effective,geometry,leaf_dimension=leaf_dimension,plant_area=plant_area)
        boundary=replace(exchange,atmosphere_velocity=velocities['heat_velocity'],
            atmosphere_water_velocity=velocities['water_velocity'],potential_temperature=theta,
            leaf_velocity=velocities['leaf_velocity'],ground_velocity=velocities['ground_velocity'])
        result=solve_surface(boundary);d=result[-1]
        vapor=float(d['et_kgm2']/time_step)
        target=flux_stability(exchange.air_temperature,exchange.specific_humidity,exchange.pressure,
            wind,float(d['h']),vapor,velocities['friction_velocity'],geometry)
        updated=canopy_exchange(target['zeta'],target['effective_wind'],geometry,
            leaf_dimension=leaf_dimension,plant_area=plant_area)
        mismatch=max(abs(updated[k]-velocities[k])/velocities[k] for k in
            ('friction_velocity','heat_velocity','water_velocity','leaf_velocity','ground_velocity'))
        history.append(dict(zeta=zeta,effective_wind=effective,h=float(d['h']),water_vapor=vapor,
                            target_zeta=target['zeta'],coefficient_residual=mismatch))
        if iteration==1:stable_bracket=target['zeta']>0.
        if previous is not None:
            dh=abs(d['h']-previous['h']);de=abs(vapor-previous['vapor'])
            dt=float(np.max(abs(d['leaf_temperature']-previous['temperature']),initial=0.))
            physical_stable=dh<=.05 and de<=2e-8 and dt<=.01
            accepted=mismatch<=1e-4 and physical_stable
            if not accepted and iteration==max_iterations==50 and physical_stable and mismatch<=.1:
                # One validation solve at the undamped feedback target.
                # Never advance its stores or accept stability of successive
                # iterates alone across a discontinuous inner-solver branch.
                required={'h','le','gpp','et_kgm2','runoff_kgm2','leaf_temperature',
                    'ground_temperature','canopy_temperature','canopy_humidity'}
                if required<=d.keys():
                    checked=solve_surface(replace(boundary,atmosphere_velocity=updated['heat_velocity'],
                        atmosphere_water_velocity=updated['water_velocity'],leaf_velocity=updated['leaf_velocity'],
                        ground_velocity=updated['ground_velocity']))
                    from .most_physical_check import surface_consistent
                    def physical(r):
                        pack,soil,_,diag=r
                        return {**{k:np.asarray(diag[k])[None,...] for k in required},
                            'soil_temperature':soil.temperature[None,:],
                            'soil_water_content':(soil.water/(1000.*soil.dz))[None,:],
                            'swe':np.asarray([pack.mass.sum()/1000.])}
                    accepted=bool(surface_consistent(physical(result),physical(checked),time_step,np)[0])
            if accepted:
                d['most']=dict(**velocities,**target,used_zeta=zeta,used_effective_wind=effective,
                    iterations=iteration,coefficient_residual=mismatch,heat_change_wm2=float(dh),
                    vapor_change_kgm2s=float(de),leaf_temperature_change_k=dt,history=history)
                d['most']['bisection_restarts']=bisection_restarts
                return result
        stability_residual=target['zeta']-zeta
        if previous_stability_residual is not None and not stable_bracket and mismatch>1e-4:
            if stability_residual*previous_stability_residual<0. and mismatch>previous_mismatch:
                # A growing sign-alternating fixed-point map can cycle even
                # with half relaxation. Reduce the trial gain, keeping the
                # same old stores, residual tests and iteration budget.
                relaxation=max(.0625,.5*relaxation)
        previous_stability_residual=stability_residual;previous_mismatch=mismatch
        previous=dict(h=float(d['h']),vapor=vapor,temperature=np.array(d['leaf_temperature'],copy=True))
        if stable_bracket:
            # At the neutral trial F=target-zeta > 0; at zeta=2, F<=0
            # by the physical envelope. Stable buoyancy has no convective
            # wind, so this is a scalar root at the prescribed effective
            # wind. Bisection avoids the near-unit Picard slope in calm,
            # strongly stable columns. Every bracket trial retains old M/H.
            if target['zeta']>=zeta:lower=zeta
            else:upper=zeta
            midpoint=.5*(lower+upper)
            if midpoint==zeta:
                # An inner-solver discontinuity may exhaust the representable
                # bracket. Resume damped feedback; all acceptance tests above
                # and the original iteration budget still apply.
                stable_bracket=False;bisection_restarts+=1
            else:zeta=midpoint
        if not stable_bracket:
            # Preserve the original rounding sequence at the ordinary gain.
            zeta=(.5*(zeta+target['zeta']) if relaxation==.5 else zeta+relaxation*(target['zeta']-zeta))
            effective=(.5*(effective+target['effective_wind']) if relaxation==.5 else effective+relaxation*(target['effective_wind']-effective))
    raise RuntimeError(f'MOST flux feedback did not converge in {max_iterations} iterations; '
                       f'coefficient residual={mismatch:.9g}, zeta={zeta:.9g}')

"""One conservative implicit heat system for snow and underlying soil.

Each interface contributes equal and opposite fluxes to adjacent enthalpy
equations. This establishes the shared snow/soil boundary required by CLM5
technical note 2.6.7--10 without sequentially inventing two interface fluxes.
The released AHC/Thomas path is intentionally not modified by this module.
"""
import numpy as np
from .snowpack import TF,CI,CW,LF
from .snow_heat import conductivity as snow_conductivity
from .soil import thermal_conductivity as soil_conductivity
from .soil_enthalpy import temperature_from_enthalpy,enthalpy_and_capacity


def coupled_heat_step(snow,soil,dt,top_flux,*,snow_absorbed=None,soil_absorbed=None,
                      top_temperature=None,top_conductance=0.,thermal_conductivity=None):
    """Return (snow, soil, ledger) after one jointly implicit thermal step.

    Fluxes/absorbed radiation are W/m2. Optional linear top exchange is solved
    implicitly; the returned top flux includes it. The soil bottom is insulated.
    Water masses do not change in this operation; drainage/infiltration and
    their advected heat are separate conservative transfers. Conductivity is
    frozen at the old state; soil uses the retained Johansen approximation.
    """
    ns=snow.mass.size;ng=soil.water.size;n=ns+ng
    qs=np.zeros(ns) if snow_absorbed is None else np.asarray(snow_absorbed,float)
    qg=np.zeros(ng) if soil_absorbed is None else np.asarray(soil_absorbed,float)
    if (qs.shape!=(ns,) or qg.shape!=(ng,) or not all(np.all(np.isfinite(v)) for v in (qs,qg))
            or not np.all(np.isfinite([dt,top_flux,top_conductance])) or dt<=0. or top_conductance<0.
            or (top_temperature is None and top_conductance>0.)
            or (top_temperature is not None and not np.isfinite(top_temperature))):
        raise ValueError('Invalid coupled thermal boundary or time step')
    z=np.r_[snow.thickness,soil.dz]
    if thermal_conductivity is None:
        k=np.r_[snow_conductivity(snow),soil_conductivity(soil.water/(1000*soil.dz),soil.params)]
    else: k=np.asarray(np.broadcast_to(thermal_conductivity,(n,)),float)
    if not np.all(np.isfinite(k)) or np.any(k<=0.): raise ValueError('Invalid thermal conductivity')
    conduct=1./(.5*z[:-1]/k[:-1]+.5*z[1:]/k[1:])
    operator=np.zeros((n,n))
    for j,g in enumerate(conduct):
        operator[j,j]-=g;operator[j+1,j+1]-=g
        operator[j,j+1]+=g;operator[j+1,j]+=g
    operator[0,0]-=top_conductance
    source=np.r_[qs,qg];source[0]+=top_flux
    if top_temperature is not None: source[0]+=top_conductance*(top_temperature-TF)
    h0=np.r_[snow.enthalpy,soil.enthalpy]
    def diagnose(h):
        hs=h[:ns];m=snow.mass
        ts=np.where(hs<0,hs/(CI*m),np.where(hs>LF*m,(hs-LF*m)/(CW*m),0.))
        ds=np.where(hs<0,1./(CI*m),np.where(hs>LF*m,1./(CW*m),0.))
        tg=temperature_from_enthalpy(soil.water,h[ns:],soil.dz,soil.params,soil.freezing_option)
        cg=enthalpy_and_capacity(soil.water,tg,soil.dz,soil.params,soil.freezing_option)[1]
        return np.r_[ts,tg-TF],np.r_[ds,1./cg]
    def evaluate(h):
        t,d=diagnose(h)
        return h-h0-dt*(source+operator@t),t,d
    h=h0.copy();tol=max(2e-7,64*np.finfo(float).eps*max(1.,np.max(np.abs(h0)),np.max(np.abs(dt*source))))
    for iteration in range(1,81):
        residual,t,derivative=evaluate(h);norm=np.max(np.abs(residual))
        if norm<=tol: break
        # At a latent-heat endpoint T(H) has two one-sided derivatives. A
        # line search using only the plateau derivative can approach the
        # kink forever. Land on each phase event, then select the derivative
        # consistent with the solved direction (semismooth active set).
        hs=h[:ns];upper=LF*snow.mass
        ulp=4.*np.spacing(np.maximum(np.abs(hs),upper))
        at_lower=np.abs(hs)<=ulp;at_upper=np.abs(hs-upper)<=ulp
        for _ in range(2*ns+2):
            jacobian=np.eye(n)-dt*operator*derivative[None,:]
            delta=np.linalg.solve(jacobian,-residual)
            ds=derivative[:ns].copy()
            ds=np.where(at_lower,np.where(delta[:ns]<0.,1./(CI*snow.mass),0.),ds)
            ds=np.where(at_upper,np.where(delta[:ns]>0.,1./(CW*snow.mass),0.),ds)
            if np.array_equal(ds,derivative[:ns]): break
            derivative[:ns]=ds
        scale=1.;events=[]
        for j in range(ns):
            for boundary in (0.,upper[j]):
                if abs(h[j]-boundary)<=ulp[j] or delta[j]==0.: continue
                fraction=(boundary-h[j])/delta[j]
                if 0.<fraction<=1.:
                    events.append((fraction,j,boundary));scale=min(scale,fraction)
        accepted=False
        for _ in range(40):
            candidate=h+scale*delta
            for fraction,j,boundary in events:
                if scale==fraction: candidate[j]=boundary
            try: value=evaluate(candidate)[0]
            except ValueError:
                scale*=.5;continue
            candidate_norm=np.max(np.abs(value))
            # A vanishing snow parcel can cross a complete latent plateau
            # while neighbouring soil H cannot move by one ULP. Landing on
            # a new exact phase endpoint still advances the active set.
            # Retain the original final residual/tolerance and iteration cap.
            phase_event=any(scale==fraction and candidate[j]==boundary and h[j]!=boundary
                            for fraction,j,boundary in events)
            if candidate_norm<norm or (phase_event and candidate_norm==norm):
                h=candidate;accepted=True;break
            scale*=.5
        if not accepted: break
    residual,t,_=evaluate(h)
    if np.max(np.abs(residual))>tol:
        raise RuntimeError(f'Coupled snow/soil enthalpy solve failed: {np.max(np.abs(residual)):.9g} J/m2')
    interface=conduct*(t[:-1]-t[1:])
    top=top_flux+(top_conductance*(top_temperature-TF-t[0]) if top_temperature is not None else 0.)
    incoming=dt*(top+qs.sum()+qg.sum())
    snow_change=float((h[:ns]-h0[:ns]).sum());soil_change=float((h[ns:]-h0[ns:]).sum())
    cross=float(interface[ns-1]) if ns else 0.
    return snow.with_enthalpy(h[:ns]),soil.with_enthalpy(h[ns:]),dict(
        top_flux_wm2=float(top),interface_flux_wm2=interface,snow_soil_flux_wm2=cross,
        snow_enthalpy_change_jm2=snow_change,soil_enthalpy_change_jm2=soil_change,
        input_energy_jm2=float(incoming),residual_jm2=float(snow_change+soil_change-incoming),
        layer_residual_jm2=residual,iterations=iteration,convergence_tolerance_jm2=tol)

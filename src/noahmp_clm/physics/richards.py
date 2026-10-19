"""NumPy development reference for closed-column vertical liquid redistribution.

Mixed pressure-head Richards equations, Campbell retention and harmonic face
conductivity. Downward flux is positive. This water-only component is not yet
wired into the physical column: frozen-pore/enthalpy coupling, runoff boundary
operators and native backends must be validated before that integration.
No upstream implementation is copied. See docs/CLM_RICHARDS.md.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class HydraulicProfile:
    dz_m: np.ndarray
    porosity: np.ndarray
    saturated_conductivity_m_s: np.ndarray
    saturated_suction_m: np.ndarray
    retention_exponent: np.ndarray

    def __post_init__(self):
        dz=np.asarray(self.dz_m,float)
        if dz.ndim!=1 or not len(dz) or not np.all(np.isfinite(dz)) or np.any(dz<=0.):
            raise ValueError('Richards requires positive finite layer thicknesses')
        for name in self.__dataclass_fields__:
            try:value=np.broadcast_to(np.asarray(getattr(self,name),float),dz.shape).copy()
            except ValueError:raise ValueError('Hydraulic profile fields must match the soil layers') from None
            if not np.all(np.isfinite(value)):raise ValueError('Nonfinite hydraulic profile')
            if name=='porosity' and np.any((value<=0.)|(value>=1.)):raise ValueError('Invalid liquid pore volume')
            if name=='saturated_conductivity_m_s' and np.any(value<0.):raise ValueError('Negative hydraulic conductivity')
            if name in ('saturated_suction_m','retention_exponent') and np.any(value<=0.):raise ValueError('Positive retention parameters required')
            value.setflags(write=False);object.__setattr__(self,name,value)

    @property
    def midpoint_m(self):return np.cumsum(self.dz_m)-.5*self.dz_m


def hydraulic_properties(head_m,profile):
    """Liquid volume fraction, dtheta/dhead, conductivity and its derivative.

    Positive pressure above air entry retains saturated pore water while
    pressure gradients still transmit water through saturated cells.
    """
    h=np.asarray(head_m,float);p=profile
    if h.shape!=p.dz_m.shape or not np.all(np.isfinite(h)):raise ValueError('Invalid pressure-head vector')
    suction=np.maximum(-h,p.saturated_suction_m)
    theta=p.porosity*(p.saturated_suction_m/suction)**(1./p.retention_exponent)
    capacity=np.where(h < -p.saturated_suction_m,theta/(p.retention_exponent*suction),0.)
    power=2.*p.retention_exponent+3.
    conductivity=p.saturated_conductivity_m_s*(theta/p.porosity)**power
    derivative=power*conductivity*capacity/theta
    return theta,capacity,conductivity,derivative


def _faces(head,p):
    theta,capacity,k,dk=hydraulic_properties(head,p)
    distance=np.diff(p.midpoint_m);total=k[:-1]+k[1:]
    # A zero-conductivity layer blocks its two faces, without a numerical leak.
    valid=total>0.;denominator=np.where(valid,total,1.)
    face=np.where(valid,2.*k[:-1]*k[1:]/denominator,0.)
    gradient=1.-np.diff(head)/distance
    left=np.where(valid,2.*(k[1:]/denominator)**2*dk[:-1],0.)*gradient+face/distance
    right=np.where(valid,2.*(k[:-1]/denominator)**2*dk[1:],0.)*gradient-face/distance
    return theta,capacity,face*gradient,left,right


def _residual(head,old,p,dt,jacobian=False):
    theta,capacity,q,left,right=_faces(head,p)
    faces=np.r_[0.,q,0.]
    residual=1000.*p.dz_m*theta-old+1000.*dt*np.diff(faces)
    if not jacobian:return residual,q
    matrix=np.diag(1000.*p.dz_m*capacity)
    for j in range(len(q)):
        matrix[j,j]+=1000.*dt*left[j];matrix[j,j+1]+=1000.*dt*right[j]
        matrix[j+1,j]-=1000.*dt*left[j];matrix[j+1,j+1]-=1000.*dt*right[j]
    return residual,q,matrix


def _implicit_step(old,guess,p,dt):
    head=guess.copy();total_capacity=float(np.sum(1000.*p.dz_m*p.porosity))
    for iteration in range(45):
        residual,q,matrix=_residual(head,old,p,dt,True)
        norm=float(np.max(abs(residual)))
        if norm<=1e-10:
            transfer=1000.*dt*q
            # The finite-volume update from the solved flux is authoritative.
            # It telescopes exactly; no residual mass is invented or discarded.
            water=old+np.r_[0.,transfer]-np.r_[transfer,0.]
            maximum=1000.*p.dz_m*p.porosity
            if np.any(water<=0.) or np.any(water>maximum+1e-9):raise RuntimeError('Richards solved flux violates pore bounds')
            return water,head,transfer,norm,iteration+1
        try:direction=np.linalg.solve(matrix,-residual)
        except np.linalg.LinAlgError:raise RuntimeError('Singular mixed Richards pressure solve') from None
        scale=1.;accepted=False
        for _ in range(35):
            candidate=head+scale*direction
            if np.all(np.isfinite(candidate)):
                theta=hydraulic_properties(candidate,p)[0]
                all_saturated=np.all(candidate>=-p.saturated_suction_m)
                if not (all_saturated and float(old.sum())<total_capacity-1e-8):
                    trial,_=_residual(candidate,old,p,dt)
                    if np.max(abs(trial))<norm:
                        head=candidate;accepted=True;break
            scale*=.5
        if not accepted:raise RuntimeError('Richards pressure line search did not decrease the full water residual')
    raise RuntimeError('Richards nonlinear water solve did not converge')


def redistribute_liquid(water_kg_m2,profile,dt_seconds,*,absolute_tolerance_kg_m2=.002,
                        relative_tendency_tolerance=.01,max_substeps=4096):
    """Conservative adaptive redistribution, with zero flux at both boundaries.

    The complete change is carried by signed integrated face transfers. A
    backward-Euler full step and two half steps control temporal error; the
    accepted solution uses the two half steps. Strictly positive liquid is the
    current development domain. No ice, energy, surface input, aquifer or
    lateral groundwater tendency is hidden in this component.
    """
    if not isinstance(profile,HydraulicProfile):raise TypeError('Expected HydraulicProfile')
    p=profile;old=np.asarray(water_kg_m2,float);maximum=1000.*p.dz_m*p.porosity
    if old.shape!=p.dz_m.shape or not np.all(np.isfinite(old)) or np.any(old<=0.) or np.any(old>maximum):
        raise ValueError('Richards reference requires positive liquid within explicit pore capacity')
    if (not math.isfinite(dt_seconds) or dt_seconds<=0. or not math.isfinite(absolute_tolerance_kg_m2)
            or absolute_tolerance_kg_m2<=0. or not math.isfinite(relative_tendency_tolerance)
            or not 0.<relative_tendency_tolerance<=.1 or type(max_substeps) is not int or max_substeps<1):
        raise ValueError('Invalid Richards timestep or temporal controls')
    theta=old/(1000.*p.dz_m)
    head=-p.saturated_suction_m*(p.porosity/theta)**p.retention_exponent
    # Closed, fully saturated columns have an undetermined pressure offset.
    # Select a hydrostatic gauge; water and interface transfers remain exact.
    if np.all(old==maximum):
        head=p.midpoint_m+float(np.max(-p.saturated_suction_m-p.midpoint_m))
        return dict(water_kg_m2=old.copy(),pressure_head_m=head,interface_transfer_kg_m2=np.zeros(len(old)-1),
                    forward_transfer_kg_m2=np.zeros(len(old)-1),reverse_transfer_kg_m2=np.zeros(len(old)-1),
                    mass_residual_kg_m2=0.,mass_bound_kg_m2=32.*np.finfo(float).eps*max(float(2.*old.sum()),1.),
                    accepted_substeps=0,rejected_substeps=0,maximum_nonlinear_residual_kg_m2=0.,maximum_temporal_error_kg_m2=0.)
    current=old.copy();elapsed=0.;step=min(float(dt_seconds),3600.);accepted=rejected=0;max_residual=max_error=0.
    transfers=[[] for _ in range(len(old)-1)]
    while elapsed<dt_seconds:
        if accepted+rejected>=max_substeps:raise RuntimeError('Richards adaptive substep budget exhausted')
        step=min(step,dt_seconds-elapsed)
        try:
            coarse,_,_,_,_=_implicit_step(current,head,p,step)
            half,half_head,first,r1,_=_implicit_step(current,head,p,.5*step)
            fine,next_head,second,r2,_=_implicit_step(half,half_head,p,.5*step)
            error=abs(fine-coarse);tendency=np.maximum(abs(fine-current),abs(coarse-current))
            ratio=float(np.max(error/(absolute_tolerance_kg_m2+relative_tendency_tolerance*tendency)))
        except RuntimeError:
            rejected+=1;step*=.5
            if step<1e-7:raise
            continue
        if ratio>1.:
            rejected+=1;step*=max(.2,.8/math.sqrt(ratio));continue
        for part in (first,second):
            for face,value in enumerate(part):transfers[face].append(float(value))
        current=fine;head=next_head;elapsed+=step;accepted+=1
        max_residual=max(max_residual,r1,r2);max_error=max(max_error,float(error.max()))
        step*=min(2.,max(.5,.8/math.sqrt(max(ratio,1e-12))))
    integrated=np.asarray([math.fsum(v) for v in transfers])
    forward=np.asarray([math.fsum(max(v,0.) for v in values) for values in transfers])
    reverse=np.asarray([math.fsum(max(-v,0.) for v in values) for values in transfers])
    water=old+np.r_[0.,integrated]-np.r_[integrated,0.]
    residual=float(math.fsum(water)-math.fsum(old));bound=32.*np.finfo(float).eps*max(float(abs(water).sum()+abs(old).sum()),1.)
    if abs(residual)>bound:raise RuntimeError('Richards extensive water budget failed')
    return dict(water_kg_m2=water,pressure_head_m=head,interface_transfer_kg_m2=integrated,
        forward_transfer_kg_m2=forward,reverse_transfer_kg_m2=reverse,
        mass_residual_kg_m2=residual,mass_bound_kg_m2=bound,accepted_substeps=accepted,rejected_substeps=rejected,
        maximum_nonlinear_residual_kg_m2=max_residual,maximum_temporal_error_kg_m2=max_error)


def soil_hydraulic_profile(state,ice_impedance_exponent=6.):
    """Preserve the total-pore Campbell law inside the available liquid pores.

    Effective saturated suction and conductivity are transformed together;
    using effective porosity alone would spuriously increase frozen-soil
    conductivity. Ice occupies water-equivalent volume, as in the retained
    extensive soil state. The decimal ice factor is the CLM-style parameter.
    """
    from .soil_enthalpy import SoilEnthalpyState
    if not isinstance(state,SoilEnthalpyState):raise TypeError('Expected SoilEnthalpyState')
    if not math.isfinite(ice_impedance_exponent) or ice_impedance_exponent<0.:
        raise ValueError('Invalid ice impedance')
    p=state.params;liquid=state.liquid;ice=state.water-liquid
    fraction=ice/(1000.*state.dz*p.smcmax);available=p.smcmax-ice/(1000.*state.dz)
    if np.any(liquid<=0.) or np.any(available<=0.):raise NotImplementedError('Dry/fully ice-filled Richards cells require a separate degenerate-limit validation')
    ratio=available/p.smcmax
    suction=p.saturated_suction_m/ratio**p.bexp
    conductivity=p.satdk*ratio**(2.*p.bexp+3.)*10.**(-ice_impedance_exponent*fraction)
    return HydraulicProfile(state.dz,available,conductivity,suction,p.bexp)


def redistribute_soil_enthalpy(state,dt_seconds,*,ice_impedance_exponent=6.,maximum_liquid_fraction=.05,
                               maximum_temperature_change_k=.05,max_substeps=8192):
    """Closed-column water and upwind liquid-enthalpy transport reference.

    Each accepted thermal transport step rebuilds liquid pores/conductivity
    from the actual phase state. Forward/reverse water are retained separately
    so reversal cannot erase transported heat. There is no external energy,
    infiltration, groundwater, lateral exchange or runoff tendency here.
    """
    from .soil_enthalpy import SoilEnthalpyState
    from .snowpack import TF,LF,CW
    if (not isinstance(state,SoilEnthalpyState) or not math.isfinite(dt_seconds) or dt_seconds<=0.
            or not 0.<maximum_liquid_fraction<=.2 or not math.isfinite(maximum_temperature_change_k)
            or maximum_temperature_change_k<=0. or type(max_substeps) is not int or max_substeps<1):
        raise ValueError('Invalid extensive Richards transport controls')
    current=state;elapsed=0.;step=min(float(dt_seconds),3600.);accepted=rejected=0
    water_faces=[[] for _ in range(len(state.water)-1)];heat_faces=[[] for _ in water_faces]
    max_temperature=max_fraction=0.;forward_total=np.zeros(len(water_faces));reverse_total=forward_total.copy()
    while elapsed<dt_seconds:
        if accepted+rejected>=max_substeps:raise RuntimeError('Extensive Richards transport budget exhausted')
        step=min(step,dt_seconds-elapsed);liquid=current.liquid;temperature=current.temperature
        profile=soil_hydraulic_profile(current,ice_impedance_exponent)
        flow=redistribute_liquid(liquid,profile,step)
        forward=flow['forward_transfer_kg_m2'];reverse=flow['reverse_transfer_kg_m2'];net=flow['interface_transfer_kg_m2']
        outward=np.r_[forward,0.]+np.r_[0.,reverse]
        fraction=float(np.max(outward/liquid))
        if fraction>maximum_liquid_fraction:
            rejected+=1;step*=max(.1,.8*maximum_liquid_fraction/fraction)
            if step<1e-7:raise RuntimeError('Richards liquid-availability substep underflow')
            continue
        parcel_enthalpy=LF+CW*(temperature-TF)
        heat=forward*parcel_enthalpy[:-1]-reverse*parcel_enthalpy[1:]
        water=current.water+np.r_[0.,net]-np.r_[net,0.]
        enthalpy=current.enthalpy+np.r_[0.,heat]-np.r_[heat,0.]
        candidate=SoilEnthalpyState(water,enthalpy,current.dz,current.params,current.freezing_option)
        change=float(np.max(abs(candidate.temperature-temperature)))
        if change>maximum_temperature_change_k:
            rejected+=1;step*=max(.1,.8*maximum_temperature_change_k/change)
            if step<1e-7:raise RuntimeError('Richards thermal-transport substep underflow')
            continue
        for j in range(len(net)):water_faces[j].append(float(net[j]));heat_faces[j].append(float(heat[j]))
        forward_total+=forward;reverse_total+=reverse
        current=candidate;elapsed+=step;accepted+=1
        max_temperature=max(max_temperature,change);max_fraction=max(max_fraction,fraction)
        if change<.25*maximum_temperature_change_k and fraction<.25*maximum_liquid_fraction:step*=2.
    mass=np.asarray([math.fsum(v) for v in water_faces]);heat=np.asarray([math.fsum(v) for v in heat_faces])
    final=SoilEnthalpyState(state.water+np.r_[0.,mass]-np.r_[mass,0.],
        state.enthalpy+np.r_[0.,heat]-np.r_[heat,0.],state.dz,state.params,state.freezing_option)
    residual=np.asarray([math.fsum(final.water)-math.fsum(state.water),math.fsum(final.enthalpy)-math.fsum(state.enthalpy)])
    bound=32.*np.finfo(float).eps*np.maximum([abs(state.water).sum()+abs(final.water).sum(),abs(state.enthalpy).sum()+abs(final.enthalpy).sum()],1.)
    if np.any(abs(residual)>bound):raise RuntimeError('Richards extensive water/enthalpy budget failed')
    return final,dict(interface_transfer_kg_m2=mass,interface_enthalpy_j_m2=heat,
        forward_transfer_kg_m2=forward_total,reverse_transfer_kg_m2=reverse_total,
        residual_mass_heat=residual,bound_mass_heat=bound,accepted_substeps=accepted,rejected_substeps=rejected,
        maximum_substep_temperature_change_k=max_temperature,maximum_substep_liquid_fraction=max_fraction)

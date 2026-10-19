"""Closed-column Richards water and liquid enthalpy transport on devices.

The host dispatches adaptive transport steps. Water fluxes, phase inversion,
ice impedance, donor enthalpy and extensive state updates run on the device.
This development component does not change the production column.
"""
import math
from .richards import _sum


def _phase(c,p,option,xp,invert):
    n=len(c);w,h,z=c.T
    broadcast=lambda value:xp.broadcast_to(value,(n,))
    inputs=xp.stack((w,h,z,broadcast(p[0]),broadcast(p[1]),broadcast(p[3])),axis=1)
    return invert(inputs,xp.full((n,),option,dtype=xp.int32),inverse=True)


def _profile(c,p,liquid,xp,ice_exponent):
    z=c[:,2];water=c[:,0];ice=water-liquid;volume=1000.*z
    available=p[0]-ice*xp.reciprocal(volume)
    ratio=available*xp.reciprocal(p[0]);safe=xp.where(available>0.,ratio,1.)
    suction=p[3]*xp.reciprocal(xp.power(safe,p[1]))
    fraction=ice*xp.reciprocal(volume*p[0])
    conductivity=xp.where(available>0.,p[2]*xp.power(10.,-ice_exponent*fraction),0.)
    return xp.stack((z,available,conductivity,suction,xp.broadcast_to(p[1],z.shape),xp.broadcast_to(p[0],z.shape)),axis=1)


def _add(total,correction,value):
    corrected=value-correction;updated=total+corrected
    return updated,(updated-total)-corrected


def _one(c,p,option,duration,xp,invert,redistribute,ice_exponent,fraction_limit,temperature_limit,budget):
    n=len(c);current=c.copy();phase=_phase(current,p,option,xp,invert)
    mass=xp.zeros(n-1,c.dtype);heat=mass.copy();mc=mass.copy();hc=mass.copy();forward=mass.copy();reverse=mass.copy()
    accepted=rejected=hydraulic_steps=hydraulic_rejections=0;elapsed=0.;step=min(duration,3600.)
    max_temperature=xp.asarray(0.,c.dtype);max_fraction=max_temperature;max_hydraulic=max_temperature
    while elapsed<duration:
        if accepted+rejected>=budget:raise RuntimeError('Device Richards heat transport budget exhausted')
        step=min(step,duration-elapsed);liquid=phase['liquid_kgm2'];offset=phase['temperature_offset_k']
        profile=_profile(current,p,liquid,xp,ice_exponent)
        flow=redistribute(liquid[None,:],profile[None,:,:],xp.asarray([step],c.dtype))
        f=flow['forward_transfer_kg_m2'][0];r=flow['reverse_transfer_kg_m2'][0];net=flow['interface_transfer_kg_m2'][0]
        z=xp.zeros(1,c.dtype);outward=xp.concatenate((f,z))+xp.concatenate((z,r))
        incoming=xp.concatenate((z,f))+xp.concatenate((r,z));available=liquid+incoming
        if bool(xp.any((available==0.)&(outward>0.))):raise RuntimeError('Device Richards outflow has no liquid source')
        ratio=outward*xp.reciprocal(xp.where(available>0.,available,1.));fraction=xp.max(ratio)
        if float(fraction)>fraction_limit:
            rejected+=1;step*=max(.1,.8*fraction_limit/float(fraction))
            if step<1e-7:raise RuntimeError('Device Richards liquid-availability substep underflow')
            continue
        # Use the phase solver's offset directly, preserving sub-Kelvin
        # information in float32 without subtracting two absolute temperatures.
        specific=334000.+4180.*offset;carried=f*specific[:-1]-r*specific[1:]
        water=current[:,0]+xp.concatenate((z,net))-xp.concatenate((net,z))
        enthalpy=current[:,1]+xp.concatenate((z,carried))-xp.concatenate((carried,z))
        candidate=xp.stack((water,enthalpy,current[:,2]),axis=1)
        try:next_phase=_phase(candidate,p,option,xp,invert)
        except ValueError:
            rejected+=1;step*=.5
            if step<1e-7:raise
            continue
        change=xp.max(xp.abs(next_phase['temperature_offset_k']-offset))
        if float(change)>temperature_limit:
            rejected+=1;step*=max(.1,.8*temperature_limit/float(change))
            if step<1e-7:raise RuntimeError('Device Richards thermal substep underflow')
            continue
        mass,mc=_add(mass,mc,net);heat,hc=_add(heat,hc,carried);forward=forward+f;reverse=reverse+r
        current=candidate;phase=next_phase;elapsed+=step;accepted+=1
        hydraulic_steps+=int(flow['accepted_substeps'][0]);hydraulic_rejections+=int(flow['rejected_substeps'][0])
        max_hydraulic=xp.maximum(max_hydraulic,flow['maximum_nonlinear_residual_kg_m2'][0])
        max_temperature=xp.maximum(max_temperature,change);max_fraction=xp.maximum(max_fraction,fraction)
        if float(change)<.25*temperature_limit and float(fraction)<.25*fraction_limit:step*=2.
    z=xp.zeros(1,c.dtype)
    water=c[:,0]+xp.concatenate((z,mass))-xp.concatenate((mass,z))
    enthalpy=c[:,1]+xp.concatenate((z,heat))-xp.concatenate((heat,z))
    final=xp.stack((water,enthalpy,c[:,2]),axis=1);last=_phase(final,p,option,xp,invert)
    residual=xp.stack((_sum(water,xp)-_sum(c[:,0],xp),_sum(enthalpy,xp)-_sum(c[:,1],xp)))
    scale=xp.stack((_sum(xp.abs(water),xp)+_sum(xp.abs(c[:,0]),xp),_sum(xp.abs(enthalpy),xp)+_sum(xp.abs(c[:,1]),xp)))
    bound=32.*xp.finfo(c.dtype).eps*xp.maximum(scale,1.)
    if bool(xp.any(xp.abs(residual)>bound)):raise RuntimeError('Device Richards extensive water/heat budget failed')
    return dict(cells=final,temperature_offset_k=last['temperature_offset_k'],temperature_k=last['temperature_k'],
        liquid_kg_m2=last['liquid_kgm2'],ice_kg_m2=last['ice_kgm2'],
        interface_transfer_kg_m2=mass,interface_enthalpy_j_m2=heat,
        forward_transfer_kg_m2=forward,reverse_transfer_kg_m2=reverse,
        residual_mass_heat=residual,bound_mass_heat=bound,
        accepted_substeps=xp.asarray(accepted,xp.int32),rejected_substeps=xp.asarray(rejected,xp.int32),
        maximum_substep_temperature_change_k=max_temperature,maximum_substep_liquid_fraction=max_fraction,
        hydraulic_accepted_substeps=xp.asarray(hydraulic_steps,xp.int32),hydraulic_rejected_substeps=xp.asarray(hydraulic_rejections,xp.int32),
        maximum_hydraulic_residual_kg_m2=max_hydraulic,
        phase_reconstruction_residual_j_m2=last['reconstruction_residual_jm2'],
        phase_iterations=last['iterations'].astype(xp.int32),phase_converged=last['converged']>0.)


def _run(c,p,options,dt,xp,invert,redistribute,*,ice_impedance_exponent=6.,maximum_liquid_fraction=.05,
         maximum_temperature_change_k=.05,max_substeps=8192):
    c,p,options,dt=[xp.asarray(v) for v in (c,p,options,dt)]
    if c.ndim!=3 or c.shape[2]!=3 or not 1<=c.shape[1]<=10 or not len(c) or p.shape!=(len(c),4) or options.shape!=(len(c),) or dt.shape!=(len(c),):
        raise ValueError('Richards heat requires cells(B,L,3), properties(B,4), phase(B), dt(B), 1--10 layers')
    if c.dtype not in (xp.float32,xp.float64) or p.dtype!=c.dtype or dt.dtype!=c.dtype or options.dtype.kind not in 'iu':
        raise ValueError('Richards heat requires matching floating precision and integer phase options')
    if not all(bool(xp.all(xp.isfinite(v))) for v in (c,p,dt)) or bool(xp.any(p[:,2]<0.)) or bool(xp.any(dt<=0.)):
        raise ValueError('Invalid Richards heat inputs')
    if (not math.isfinite(ice_impedance_exponent) or ice_impedance_exponent<0. or not 0.<maximum_liquid_fraction<=.2
            or not math.isfinite(maximum_temperature_change_k) or maximum_temperature_change_k<=0.
            or type(max_substeps) is not int or max_substeps<1):raise ValueError('Invalid Richards heat controls')
    columns=[_one(c[i],p[i],options[i],float(dt[i]),xp,invert,redistribute,ice_impedance_exponent,
        maximum_liquid_fraction,maximum_temperature_change_k,max_substeps) for i in range(len(c))]
    return {key:xp.stack([v[key] for v in columns]) for key in columns[0]}


def run_richards_enthalpy_jax(c,p,options,dt,**controls):
    import jax.numpy as j
    from .enthalpy import run_soil_thermodynamics_jax
    from .richards import run_richards_jax
    return _run(c,p,options,dt,j,run_soil_thermodynamics_jax,run_richards_jax,**controls)

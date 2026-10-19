"""Layer condensate enthalpy coupled to canopy radiation, gas and air exchange.

The time discretisation first heats each existing condensate parcel, then
exchanges vapour at that same leaf temperature. Both operations are solved
together, so final storage plus advected parcel heat equals the pre-vapour
thermal increment. Ice/liquid phase changes retain the full latent plateau.
This NumPy development path is not yet used by the default column driver.
"""
import numpy as np
from .canopy_energy import _saturation, _ice_saturation, _longwave_operator
from .canopy_interception import CanopyCondensate, coverage
from .canopy_radiation import canopy_shortwave
from .photosynthesis import leaf_gas_exchange, R_GAS
from .soil_uptake import _allocate
from .snowpack import TF, CI, CW, LF
from .energy import RD, CP, LV, SIGMA


def _decode(coordinate, mass, phase_active=None):
    """Scaled specific enthalpy for wet layers; temperature offset for dry ones."""
    x=np.asarray(coordinate)
    t=TF+np.minimum(x,0.)*CW/CI+np.maximum(x-LF/CW,0.)
    return np.where(mass>0. if phase_active is None else phase_active,t,TF+x),mass*CW*x


def _equilibrium_vapor(state, amount, temperature, liquid_fraction=None):
    """Conservative mixed-phase extension for the implicit thermal solve.

    Existing liquid/ice proportions approximate the exposed wet areas at
    the melting plateau. Pure phases retain evaporation/sublimation. This
    deliberately differs from the separate CLM temperature-switch component
    at exactly Tf: suppressing all liquid evaporation there makes the joint
    heat equation discontinuous. No phase mass can be overdrawn.
    """
    mass=state.mass;heat=state.enthalpy
    fraction=(np.divide(state.liquid,mass,out=np.asarray(temperature>TF,dtype=float),where=mass>0.)
        if liquid_fraction is None else liquid_fraction)
    loss=np.minimum(np.maximum(amount,0.),mass);deposit=np.maximum(-amount,0.)
    remaining=mass-loss;retained=np.divide(remaining,mass,out=np.zeros_like(mass),where=mass>0.)*heat
    liquid=(loss-deposit)*fraction;ice=(loss-deposit)-liquid
    added=deposit*((1.-fraction)*CI*(temperature-TF)+fraction*(LF+CW*(temperature-TF)))
    advected=heat-retained-added
    final=CanopyCondensate(remaining+deposit,retained+added)
    return final,dict(mass_kgm2=loss-deposit,ice_kgm2=ice,liquid_kgm2=liquid,
        advected_enthalpy_jm2=advected,latent_energy_jm2=LV*(loss-deposit)+LF*ice,
        unmet_evaporation_kgm2=np.maximum(amount-loss,0.))


def solve_canopy_phase_energy(leaf_area, swdown, lwdown, exchange, condensate, dt, *,
        stem_area=0., vegetated_fraction=1., soil_supply=None, root_enthalpy=None,
        transpiration_limit=np.inf, ground_evaporation_limit=np.inf,
        water_stress=1., nitrogen_factor=1., leaf_parameters=None,
        ground_emissivity=.97, ground_albedo=(.15,.3), diffuse_fraction=.5,
        cosine_zenith=.5, fixed_ground_temperature=None, ground_is_ice=False,
        ground_shares_soil=True, snowpack=None, tolerance=5e-5, max_iterations=80):
    """Joint layer phase/temperature and canopy-air humidity solution.

    Condensate and all fluxes refer to total ground area. Leaf/stem areas
    refer to vegetated area. SoilWaterSupply and limits are kg/m2/s, root
    enthalpies J/kg, dt seconds. Return condensate already includes actual
    vapour loss/deposition, but has not yet shed any new capacity excess.
    Fixed-ground operation returns the heat required by the ground column.
    No SciPy solver or host/device surrogate is used by this implementation.
    """
    e=exchange;leaf=np.asarray(leaf_area,float);n=len(condensate.mass)
    stem=np.broadcast_to(stem_area,(n,)).astype(float);area=leaf+stem;fv=float(vegetated_fraction)
    if (leaf.shape!=(n,) or not np.all(np.isfinite(area)) or np.any(leaf<0.) or np.any(stem<0.)
            or not np.isfinite(dt) or dt<=0. or not 0.<=fv<=1. or not np.isfinite(lwdown) or lwdown<0.
            or not 0.<ground_emissivity<=1. or not np.isfinite(tolerance) or tolerance<=0.
            or not isinstance(max_iterations,int) or max_iterations<1
            or any(x<0. or np.isnan(x) for x in (transpiration_limit,ground_evaporation_limit))):
        raise ValueError('Invalid phase-canopy geometry, boundary or solver controls')
    if area.sum()==0.:fv=0.
    active=fv*area>0.;mass=condensate.mass
    if np.any((~active)&(mass>0.)):
        raise ValueError('Water on inactive canopy layers must be shed before the energy solve')
    if np.any((condensate.temperature<200.)|(condensate.temperature>350.)):
        raise ValueError('Initial canopy phase state outside energy temperature domain')
    if fixed_ground_temperature is not None and (not np.isfinite(fixed_ground_temperature) or not 200.<=fixed_ground_temperature<=350.):
        raise ValueError('Invalid fixed ground temperature')
    if ground_is_ice and (fixed_ground_temperature is None or fixed_ground_temperature>TF):
        raise ValueError('Ice ground requires a fixed temperature at or below freezing')
    if snowpack is not None and fixed_ground_temperature is None:
        raise ValueError('Snow radiation requires fixed-ground enthalpy coupling')
    factors=[]
    for value in (water_stress,nitrogen_factor):
        a=np.broadcast_to(value,(n,)).astype(float)
        if not np.all(np.isfinite(a)) or np.any((a<0.)|(a>1.)):raise ValueError('Invalid canopy biochemical factor')
        factors.append(a)
    water_stress,nitrogen_factor=factors
    roots=np.empty(0) if soil_supply is None else soil_supply.roots
    if soil_supply is not None and root_enthalpy is None:
        raise ValueError('Phase energy requires explicit root source enthalpy')
    root_h=np.zeros_like(roots) if root_enthalpy is None else np.asarray(root_enthalpy,float)
    if root_h.shape!=roots.shape or not np.all(np.isfinite(root_h)):
        raise ValueError('Root enthalpy must match soil supply')
    if soil_supply is None and transpiration_limit>0.:
        raise ValueError('Phase energy requires explicit root supply for transpiration heat')
    rooted=0. if soil_supply is None else np.where(soil_supply.weights>0.,roots,0.).sum()
    tr_limit=min(transpiration_limit,rooted)
    biochemical=water_stress if tr_limit>0. else water_stress*0.
    fractions=coverage(condensate,leaf,stem_area=stem,vegetated_fraction=fv)
    if snowpack is None:
        sw=canopy_shortwave(area,swdown,ground_albedo=ground_albedo,diffuse_fraction=diffuse_fraction,cosine_zenith=cosine_zenith)
        bare_sw=swdown*(1.-.5*sum(ground_albedo));shortwave=np.r_[fv*sw['absorbed'],fv*sw['ground_absorbed']+(1.-fv)*bare_sw]
        reflected=fv*sw['reflected']+(1.-fv)*(swdown-bare_sw);snow_sw=np.empty(0);soil_sw=shortwave[-1]
    else:
        from .canopy_snow_radiation import canopy_snow_shortwave
        rad=canopy_snow_shortwave(area,snowpack,swdown,vegetated_fraction=fv,ground_albedo=ground_albedo,
            diffuse_fraction=diffuse_fraction,cosine_zenith=cosine_zenith)
        sw=rad['canopy'];snow_sw=rad['snow_absorbed'];soil_sw=rad['soil_absorbed'];reflected=rad['reflected']
        shortwave=np.r_[rad['canopy_absorbed'],snow_sw.sum()+soil_sw]
    lw=_longwave_operator(area,ground_emissivity,fv)
    rho=e.pressure/(RD*e.air_temperature*(1.+.61*e.specific_humidity));mol=e.pressure/(R_GAS*e.air_temperature)
    heat_weight=rho*CP*np.r_[fv*area*e.leaf_velocity,fv*e.ground_velocity]
    atm_h=rho*CP*fv*e.atmosphere_velocity;bare_h=rho*CP*(1.-fv)*e.atmosphere_velocity
    heat_den=heat_weight.sum()+atm_h;vapor_conductance=rho*fv*area*e.leaf_velocity
    ground_latent=LV+(LF if ground_is_ice else 0.)
    lower=np.r_[np.where(mass>0.,(200.-TF)*CI/CW,200.-TF),200.,0.]
    upper=np.r_[np.where(mass>0.,LF/CW+350.-TF,350.-TF),350.,190.]
    x=np.r_[np.divide(condensate.enthalpy,mass*CW,out=np.full(n,e.air_temperature-TF),where=mass>0.),
        fixed_ground_temperature if fixed_ground_temperature is not None else e.air_temperature,e.specific_humidity*1000.]
    phase_active=mass>0.

    def evaluate(z,details=False):
        leaf_t,thermal_h=_decode(z[:n],mass,phase_active);t=np.r_[leaf_t,z[n]];qc=z[n+1]*.001
        heated=CanopyCondensate(mass,thermal_h)
        qliq,_,es=_saturation(t,e.pressure);qice=_ice_saturation(t,e.pressure)[0]
        liquid_area=np.divide(heated.liquid,mass,
            out=np.where(phase_active,np.clip(z[:n]*CW/LF,0.,1.),np.asarray(leaf_t>TF,dtype=float)),where=mass>0.)
        wet_sat=liquid_area*qliq[:-1]+(1.-liquid_area)*qice[:-1];ground_sat=qice[-1] if ground_is_ice else qliq[-1]
        gas=leaf_gas_exchange(sw['absorbed_par'],leaf_t,co2_ppm=e.co2_ppm,
            relative_humidity=qc*e.pressure/(.622+.378*qc)/es[:-1],boundary_conductance=e.leaf_velocity*mol,
            parameters=leaf_parameters,water_stress=water_stress,nitrogen_factor=nitrogen_factor*biochemical)
        wet_area=np.where(wet_sat>qc,fractions['wet_fraction'],1.)
        wet_demand=vapor_conductance*wet_area*(wet_sat-qc)
        final,parcel=_equilibrium_vapor(heated,dt*wet_demand,leaf_t,liquid_area)
        wet=parcel['mass_kgm2']/dt
        # Leaves transpire liquid water; stem area contributes boundary/heat
        # exchange but has no stomata. Root and soil surface supply are shared.
        tr=vapor_conductance*fractions['dry_fraction']*gas['gs_mol']/(e.leaf_velocity*mol+gas['gs_mol'])*np.maximum(qliq[:-1]-qc,0.)
        tr*=min(1.,tr_limit/tr.sum()) if tr.sum()>0. else 1.
        uptake=np.empty(0) if soil_supply is None else _allocate(tr.sum(),roots,soil_supply.weights)
        surface=ground_evaporation_limit
        if soil_supply is not None and ground_shares_soil:surface=min(surface,max(0.,soil_supply.surface-uptake[0]))
        ge=rho*np.array([fv*(ground_sat-qc),(1.-fv)*(ground_sat-e.specific_humidity)])/(1./np.array([e.ground_velocity,e.water_velocity])+e.ground_resistance)
        positive=np.maximum(ge,0.);negative=np.minimum(ge,0.)
        ge=negative+positive*(min(1.,(surface-negative.sum())/positive.sum()) if positive.sum()>0. else 1.)
        air_offset=float(heat_weight@(t-e.heat_temperature)/heat_den) if heat_den>0. else 0.
        sensible=heat_weight*((t-e.heat_temperature)-air_offset);sensible[-1]+=bare_h*(t[-1]-e.heat_temperature)
        rn=shortwave+lw@np.r_[SIGMA*t**4,lwdown]
        latent=np.r_[parcel['latent_energy_jm2']/dt+LV*tr,ground_latent*ge.sum()]
        root_mean=float(uptake@root_h/uptake.sum()) if uptake.sum()>0. else 0.
        transp_heat=tr*(LF+CW*(leaf_t-TF)-root_mean)
        thermal_increment=(thermal_h-condensate.enthalpy)/dt
        residual=rn-sensible-latent;residual[:-1]-=thermal_increment+transp_heat
        ground_heat=e.soil_conductance*(t[-1]-e.soil_temperature)+e.soil_flux_offset
        if fixed_ground_temperature is not None:ground_heat=residual[-1]
        residual[-1]-=ground_heat
        residual[:-1]=np.where(active,residual[:-1],e.air_temperature-leaf_t)
        if fixed_ground_temperature is not None:residual[-1]=fixed_ground_temperature-t[-1]
        air_water=wet.sum()+tr.sum()+ge[0]-rho*fv*e.water_velocity*(qc-e.specific_humidity)
        qres=air_water*LV if fv>0. else (qc-e.specific_humidity)*1e5
        all_residual=np.r_[residual,qres]
        if not details:return all_residual
        # Rebuild storage and advective terms from the actual final parcel;
        # the solve's pre-vapour identity is not the independent closure test.
        storage=(final.enthalpy-condensate.enthalpy)/dt;advected=parcel['advected_enthalpy_jm2']/dt
        physical=rn-sensible-latent;physical[:-1]-=storage+advected+transp_heat;physical[-1]-=ground_heat
        atm_water=rho*fv*e.water_velocity*(qc-e.specific_humidity)+ge[1]
        return dict(condensate=final,thermal_condensate=heated,leaf_temperature=leaf_t,ground_temperature=float(t[-1]),
            canopy_temperature=e.heat_temperature+air_offset,canopy_humidity=qc,gas_exchange=gas,
            gpp=float(fv*leaf@np.where(active,gas['gross'],0.)),leaf_respiration=float(fv*leaf@np.where(active,gas['respiration'],0.)),
            h=float(sensible.sum()),le=float(latent.sum()),rnet=float(rn.sum()),ground_heat=float(ground_heat),
            ground_nonshortwave_heat=float(ground_heat-shortwave[-1]),snow_absorbed_shortwave=snow_sw,soil_absorbed_shortwave=float(soil_sw),
            outgoing_shortwave=float(reflected),outgoing_longwave=float(lwdown-(rn-shortwave).sum()),
            wet_evaporation=wet,transpiration=tr,ground_evaporation=float(ge.sum()),root_uptake=uptake,
            leaf_sensible=sensible[:-1],ground_sensible=float(sensible[-1]),ground_latent_heat=ground_latent,
            canopy_storage_change_wm2=float(storage.sum()),wet_advected_heat_wm2=float(advected.sum()),
            transpiration_advected_heat_wm2=float(tr@(LF+CW*(leaf_t-TF))),
            root_advected_heat_wm2=float(uptake@root_h),transpiration_net_heat_wm2=float(transp_heat.sum()),vapor_parcels=parcel,
            layer_residual=physical,residual=float(physical.sum()),solver_residual=all_residual,
            air_water_residual=float(wet.sum()+tr.sum()+ge.sum()-atm_water),
            air_heat_residual=float(sensible.sum()-atm_h*air_offset-bare_h*(t[-1]-e.heat_temperature)),
            canopy_water_residual=float((final.mass-mass+dt*wet).sum()),phase_coordinates=z[:n].copy())

    # Test the atmospheric-temperature branch as well as the old enthalpy.
    # Warm old water alone can be a poor starting temperature for a freezing
    # night; at Tf the previous phase proportion supplies the initial guess.
    ambient=x.copy()
    if e.air_temperature!=TF:
        ambient[:n]=np.where(mass>0.,(e.air_temperature-TF)*CI/CW if e.air_temperature<TF
            else LF/CW+e.air_temperature-TF,e.air_temperature-TF)
    old_res=evaluate(x);ambient_res=evaluate(ambient)
    alternative=ambient.copy()
    if ambient_res@ambient_res<old_res@old_res:alternative=x.copy();x=ambient
    restart_available=not np.array_equal(x,alternative);restarts=0
    # Small coupled Newton system, numerical Jacobian and residual-based line
    # search. No approximate-heat-capacity smearing of the latent plateau.
    for iteration in range(1,max_iterations+1):
        residual=evaluate(x)
        if np.max(abs(residual))<=tolerance:break
        detail=evaluate(x,True)
        fresh=(~phase_active)&active&(abs(x[:n])<=.05)&(detail['wet_evaporation']<0.)
        if np.any(fresh):
            # A previously dry leaf can receive both ice and liquid at Tf.
            # Activate a phase coordinate without adding initial water/heat.
            # The .05 K neighborhood changes coordinates only: the latent
            # plateau remains exactly Tf, with no temperature smoothing.
            phase_active|=fresh
            x[:n]=np.where(fresh,np.where(x[:n]<0.,x[:n]*CI/CW,LF/CW+x[:n]),x[:n])
            alternative[:n]=np.where(fresh,np.where(alternative[:n]<0.,alternative[:n]*CI/CW,LF/CW+alternative[:n]),alternative[:n])
            lower[:n]=np.where(fresh,(200.-TF)*CI/CW,lower[:n])
            upper[:n]=np.where(fresh,LF/CW+350.-TF,upper[:n])
            residual=evaluate(x);detail=evaluate(x,True)
        depleted=phase_active&(detail['condensate'].mass==0.)&(x[:n]>0.)&(x[:n]<LF/CW)&(abs(residual[:n])>tolerance)
        if np.any(depleted):
            # With no condensate left, storage plus vapour enthalpy is
            # independent of the arbitrary plateau fraction. Move that
            # redundant coordinate to the cold/warm edge selected by the
            # heat residual, so the dry leaf temperature can respond.
            x[:n]=np.where(depleted,np.where(residual[:n]>0.,LF/CW,0.),x[:n])
            residual=evaluate(x)
        jac=np.empty((n+2,n+2))
        for k in range(n+2):
            # Resolve narrow vapour-supply transitions in float64 rather
            # than averaging their two derivatives across a coarse stencil.
            step=1e-6 if k<n+1 else 1e-7
            lo=x.copy();hi=x.copy();lo[k]=max(lower[k],x[k]-step);hi[k]=min(upper[k],x[k]+step)
            jac[:,k]=(evaluate(hi)-evaluate(lo))/(hi[k]-lo[k])
        delta=np.linalg.lstsq(jac,-residual,rcond=None)[0]
        # Preserve the Newton direction while limiting the largest component;
        # separate clipping of humidity and enthalpy can reverse descent.
        delta/=max(1.,float(np.max(abs(delta)/np.r_[np.full(n,20.),5.,5.])))
        norm=float(residual@residual);accepted=False;chosen_norm=norm
        # Near a latent-plateau edge, the Newton step can be much larger than
        # the distance to the kink. Keep sub-Jacobian steps available rather
        # than falsely declaring a nonsmooth but solvable state stalled.
        for backtrack in range(24):
            trial=np.clip(x+delta*2.**(-backtrack),lower,upper)
            r=evaluate(trial)
            trial_norm=float(r@r)
            if trial_norm<norm:x=trial;accepted=True;chosen_norm=trial_norm;break
        if not accepted or chosen_norm>.999*norm:
            # A Newton humidity direction can point outside the physical
            # interval, or make negligible accepted progress near a bound.
            # At fixed leaf/ground temperatures, bracket the air-water
            # balance inside the same admissible interval, and accept only
            # if the complete coupled residual decreases. This is a new
            # starting point, never a relaxed convergence criterion.
            lo=x.copy();hi=x.copy();lo[-1]=lower[-1];hi[-1]=upper[-1]
            if evaluate(lo)[-1]>=0. and evaluate(hi)[-1]<=0.:
                for _ in range(32):
                    mid=.5*(lo+hi)
                    if evaluate(mid)[-1]>0.:lo=mid
                    else:hi=mid
                trial=.5*(lo+hi);rr=evaluate(trial)
                if rr@rr<chosen_norm:
                    x=trial;restarts+=1
                    continue
            if accepted:continue
            # A centred derivative can cross a latent-plateau kink and point
            # uphill on the current branch. Retry locally only after the
            # original Newton/backtracking and humidity recovery have failed.
            # Exact kinks retain the centred derivative. Accept only a strict
            # reduction of the same complete coupled residual; all final
            # energy/water and convergence requirements are unchanged.
            local_jac=np.empty_like(jac)
            for k in range(n+2):
                step=1e-6 if k<n+1 else 1e-7
                lo=x.copy();hi=x.copy();lo[k]=max(lower[k],x[k]-step);hi[k]=min(upper[k],x[k]+step)
                if k<n and phase_active[k]:
                    if x[k]<0.:hi[k]=min(hi[k],0.)
                    elif 0.<x[k]<LF/CW:lo[k]=max(lo[k],0.);hi[k]=min(hi[k],LF/CW)
                    elif x[k]>LF/CW:lo[k]=max(lo[k],LF/CW)
                local_jac[:,k]=(evaluate(hi)-evaluate(lo))/(hi[k]-lo[k])
            local_delta=np.linalg.lstsq(local_jac,-residual,rcond=None)[0]
            local_delta/=max(1.,float(np.max(abs(local_delta)/np.r_[np.full(n,20.),5.,5.])))
            for backtrack in range(24):
                trial=np.clip(x+local_delta*2.**(-backtrack),lower,upper);rr=evaluate(trial)
                if rr@rr<norm:x=trial;accepted=True;restarts+=1;break
            if accepted:continue
            if restart_available:
                # Lowest initial residual need not select the right side of
                # the latent plateau. Retry the other physical initial guess
                # before declaring a failure; no residual is accepted here.
                x=alternative.copy();restart_available=False;restarts+=1
                continue
            # The fine derivative can lose descent on an ordinary phase
            # edge. Retry the former wider stencil only after every original
            # recovery path has failed; narrow-supply trajectories stay intact.
            # Strict descent of all equations and the original iteration and
            # independent parcel budgets remain mandatory.
            for clipped in (False,True):
                wide_jac=np.empty_like(jac)
                for k in range(n+2):
                    step=1e-4 if k<n+1 else 1e-5
                    lo=x.copy();hi=x.copy()
                    lo[k]=max(lower[k],x[k]-step);hi[k]=min(upper[k],x[k]+step)
                    if clipped and k<n and phase_active[k]:
                        if x[k]<0.:hi[k]=min(hi[k],0.)
                        elif 0.<x[k]<LF/CW:lo[k]=max(lo[k],0.);hi[k]=min(hi[k],LF/CW)
                        elif x[k]>LF/CW:lo[k]=max(lo[k],LF/CW)
                    wide_jac[:,k]=(evaluate(hi)-evaluate(lo))/(hi[k]-lo[k])
                wide_delta=np.linalg.lstsq(wide_jac,-residual,rcond=None)[0]
                wide_delta/=max(1.,float(np.max(abs(wide_delta)/np.r_[np.full(n,20.),5.,5.])))
                for backtrack in range(24):
                    trial=np.clip(x+wide_delta*2.**(-backtrack),lower,upper)
                    rr=evaluate(trial)
                    if rr@rr<norm:x=trial;accepted=True;restarts+=1;break
                if accepted:break
            if accepted:continue
            raise RuntimeError(f'Canopy phase Newton stalled at iteration {iteration}: residual={np.max(abs(residual)):.8g} W/m2')
    else:
        # The last permitted update may have reached the root. Check its
        # accepted state before declaring failure; the loop's residual is
        # from before that update. Keep the same tolerance and update budget.
        residual=evaluate(x)
        if np.max(abs(residual))>tolerance:
            raise RuntimeError(f'Canopy phase energy did not converge: residual={np.max(abs(residual)):.8g} W/m2')
    result=evaluate(x,True);result['iterations']=iteration;result['restarts']=restarts
    if np.max(abs(result['layer_residual']))>tolerance*1.01:
        raise RuntimeError('Final canopy parcel energy does not match the solved heat budget')
    return result

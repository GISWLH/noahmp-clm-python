"""NumPy canopy / snow / soil surface step with an extensive energy ledger.

The thermal boundary and diagnostic canopy are iterated to the same surface
temperature. Vapour/root transfers use the beginning-of-step water supply;
their actual mass and condensed-phase enthalpy are removed before the implicit
heat solve. Optics and supply are frozen during a step; thermal coefficients
are evaluated after mass transfer and frozen within the inner solve.
Precipitation/interception and gravity drainage belong to the
column driver. Shared liquid storage and optional per-layer ice/liquid stores
both retain sensible and fusion enthalpy.
"""
import numpy as np
from .canopy_energy import solve_canopy_energy,CanopyWaterStorage
from .column_heat import coupled_heat_step
from .soil_enthalpy import withdraw_liquid,infiltrate
from .soil_uptake import SoilWaterSupply
from .snowpack import Snowpack,TF,LF,CI,CW,drain_liquid
from .snow_vapor import exchange_vapor
from .snow_evolution import compact,grain_growth

LV=2.5e6


def surface_step(snow,soil,dt,leaf_area,swdown,lwdown,exchange,*,canopy_water_kgm2=0.,
                 canopy_enthalpy_jm2=None,canopy_condensate=None,root_weights=None,wet_fraction=0.,temperature_tolerance=2e-7,**canopy_options):
    """Return snow, soil, canopy water, diagnostics after one surface step.

    Temperature convergence concerns the post-vapour, pre-drain thermal state.
    Meltwater drainage/infiltration is subsequently conservative operator
    splitting. Evaporation is kg/m2/s and all heat budgets are extensive J/m2.
    Positive advected heat leaves the model, independently of turbulent LE.
    Caller-provided root weights may be zero (no accessible roots).
    A supplied CanopyCondensate selects the layered ice/liquid energy solve;
    diagnostics then include the actual final canopy_condensate. The third
    return value remains total canopy mass for compatibility.
    """
    if canopy_condensate is not None:
        if canopy_water_kgm2!=0. or canopy_enthalpy_jm2 is not None:
            raise ValueError('Provide layered condensate or shared liquid storage, not both')
        canopy_water_kgm2=float(canopy_condensate.mass.sum())
        canopy_enthalpy_jm2=float(canopy_condensate.enthalpy.sum())
    if (not np.isfinite(dt) or dt<=0 or not np.isfinite(canopy_water_kgm2) or canopy_water_kgm2<0
            or not np.isfinite(temperature_tolerance) or temperature_tolerance<=0):
        raise ValueError('Invalid surface time step, canopy water or tolerance')
    reserved={'soil_supply','fixed_ground_temperature','ground_is_ice','ground_shares_soil',
        'snowpack','canopy_evaporation_limit','ground_evaporation_limit','initial_temperatures','tolerance','canopy_storage'}
    if reserved & canopy_options.keys(): raise ValueError('Surface step owns its water and thermal boundary controls')
    roots=np.zeros_like(soil.water) if root_weights is None else np.asarray(root_weights,float)
    liquid=soil.liquid
    supply=SoilWaterSupply(np.maximum(liquid-soil.params.smcwlt*1000.*soil.dz,0.)/dt,roots,liquid[0]/dt)
    canopy_heat=(canopy_water_kgm2*(LF+CW*(exchange.air_temperature-TF))
        if canopy_enthalpy_jm2 is None else float(canopy_enthalpy_jm2))
    storage=CanopyWaterStorage(canopy_water_kgm2,canopy_heat,dt,LF+CW*(soil.temperature-TF))
    original_mass=float(snow.mass.sum()+soil.water.sum()+canopy_water_kgm2)
    original_heat=float(snow.enthalpy.sum()+soil.enthalpy.sum()+canopy_heat)
    ns=len(snow.mass);evaluations=0
    def evaluate(temperature):
        nonlocal evaluations
        evaluations+=1
        boundary=dict(fixed_ground_temperature=temperature,ground_is_ice=bool(temperature<TF),
            ground_shares_soil=not ns,snowpack=snow,ground_evaporation_limit=snow.mass[0]/dt if ns else np.inf,
            soil_supply=supply,tolerance=1e-7)
        if canopy_condensate is None:
            result=solve_canopy_energy(leaf_area,swdown,lwdown,exchange,**boundary,
                canopy_evaporation_limit=canopy_water_kgm2/dt,wet_fraction=wet_fraction,canopy_storage=storage,**canopy_options)
        else:
            from .canopy_phase_energy import solve_canopy_phase_energy
            result=solve_canopy_phase_energy(leaf_area,swdown,lwdown,exchange,canopy_condensate,dt,
                root_enthalpy=storage.root_enthalpy,**boundary,**canopy_options)
            result.update(canopy_water_after_kgm2=float(result['condensate'].mass.sum()),
                canopy_enthalpy_after_jm2=float(result['condensate'].enthalpy.sum()))
        root_mass=result['root_uptake']*dt
        ground_mass=float(result['ground_evaporation']*dt)
        wet_mass=float(result['wet_evaporation'].sum()*dt)
        new_canopy=(canopy_water_kgm2-wet_mass if canopy_condensate is None else result['canopy_water_after_kgm2'])
        if new_canopy<-1e-12: raise RuntimeError('Canopy water overdraw')
        new_canopy=max(0.,new_canopy)
        demand=root_mass.copy()
        if not ns and ground_mass>0.: demand[0]+=ground_mass
        ground,water_heat=withdraw_liquid(soil,demand)
        root_heat=float(np.sum(root_mass*(LF+CW*(soil.temperature-TF))))
        ground_heat=float(water_heat.sum()-root_heat)
        runoff=runoff_heat=0.;residue=np.zeros(5)
        sw_snow=result['snow_absorbed_shortwave'].copy()
        sw_soil=np.zeros_like(soil.water);sw_soil[0]=result['soil_absorbed_shortwave']
        if ns:
            pack,vapor=exchange_vapor(snow,ground_mass,deposition_temperature=temperature,remap=False)
            ground_mass=vapor['mass_kgm2']
            ground_heat=vapor['advected_enthalpy_jm2'];latent=vapor['latent_energy_jm2']/dt
            residue+=vapor['residue_kgm2']
            if len(pack.mass)<ns:
                # Explicit old-state optical absorption still belongs in the
                # column after an evaporated top cell vanishes. Transfer its
                # absorbed heat to the new surface, never drop that energy.
                removed=sw_snow[0];sw_snow=sw_snow[1:]
                if len(sw_snow): sw_snow[0]+=removed
                else: sw_soil[0]+=removed
        else:
            pack=snow;latent=LV*ground_mass/dt
            if ground_mass<0.:
                # Frost on exposed soil enters soil ice, not a new snow
                # thermal layer (CLM5 2.5.148, 2.7.5). Creating a layer here
                # changed the surface unknown discontinuously at freezing.
                # The phase-dependent latent and transported enthalpies sum
                # continuously at TF; the soil equilibrium curve retains H.
                if temperature<TF:
                    ground_heat=ground_mass*CI*(temperature-TF)
                    latent=(LV+LF)*ground_mass/dt
                else: ground_heat=ground_mass*(LF+CW*(temperature-TF))
                ground,runoff,runoff_heat=infiltrate(ground,-ground_mass,-ground_heat)
        # The fixed-temperature leaf solve is independent of the choice of
        # ground latent conversion. Use the actual ice/liquid transfer above.
        boundary=float(result['ground_nonshortwave_heat']+
            result['ground_latent_heat']*result['ground_evaporation']-latent)
        ice_before=pack.ice
        pack,ground,thermal=coupled_heat_step(pack,ground,dt,boundary,
            snow_absorbed=sw_snow,soil_absorbed=sw_soil)
        actual=float(pack.temperature[0] if len(pack.mass) else ground.temperature[0])
        return actual-temperature,dict(snow=pack,soil=ground,canopy_water=new_canopy,result=result,
            thermal=thermal,root_mass=root_mass,ground_mass=ground_mass,wet_mass=wet_mass,
            root_heat=root_heat,ground_heat=ground_heat,runoff=runoff,runoff_heat=runoff_heat,
            residue=residue,ground_latent=latent,ice_before=ice_before,temperature=temperature)
    # Safeguarded scalar secant solve, independent of SciPy. The thermal solve
    # carries latent plateaus in enthalpy, so surface temperature is never
    # clipped at freezing and post-melt heat remains available to the soil.
    initial=float(snow.temperature[0] if ns else soil.temperature[0])
    center=float(np.clip(initial,200.,350.));fc,details=evaluate(center)
    if abs(fc)>temperature_tolerance:
        left=right=center;fl=fr=fc
        width=2.
        for _ in range(8):
            if fl<0. and left>200.: left=max(200.,center-width);fl,unused=evaluate(left)
            if fr>0. and right<350.: right=min(350.,center+width);fr,unused=evaluate(right)
            if fl>=0. and fr<=0.: break
            width*=2.
        else: raise RuntimeError('Surface temperature is not bracketed in 200--350 K')
        for iteration in range(60):
            trial=(left*fr-right*fl)/(fr-fl) if fr!=fl else .5*(left+right)
            trial=float(np.clip(trial,left+.05*(right-left),right-.05*(right-left)))
            value,details=evaluate(trial)
            if abs(value)<=temperature_tolerance: break
            if value>0.: left,fl=trial,value
            else: right,fr=trial,value
        else: raise RuntimeError(f'Surface heat/water iteration did not converge: {value:.9g} K')
    # A mixed-phase snow surface is exactly on its latent plateau. A scalar
    # secant can stop one Kelvin ULP below TF and wrongly classify deposition
    # as frost rather than dew. Re-evaluate the exact physical phase event;
    # only accept it if the complete coupled solve still satisfies its root.
    if (len(details['snow'].mass) and details['snow'].temperature[0]==TF
            and details['temperature']!=TF):
        residual,plateau=evaluate(TF)
        if abs(residual)<=temperature_tolerance:details=plateau
    pack,ground=details['snow'],details['soil'];result=details['result']
    thermal_state=dict(thermal_snow_mass=pack.mass.copy(),thermal_snow_enthalpy=pack.enthalpy.copy(),
        thermal_snow_temperature=pack.temperature.copy(),thermal_soil_water=ground.water.copy(),
        thermal_soil_enthalpy=ground.enthalpy.copy(),thermal_soil_temperature=ground.temperature.copy())
    # Grain metamorphism uses the actual snow/soil temperature gradient, not
    # the fixed illustrative gradient used by the earlier component audit.
    if len(pack.mass):
        centers=np.cumsum(np.r_[pack.thickness,ground.dz[0]])-.5*np.r_[pack.thickness,ground.dz[0]]
        gradient=np.abs(np.gradient(np.r_[pack.temperature,ground.temperature[0]],centers))[:-1]
        refrozen=np.maximum(0.,pack.ice-details['ice_before'])
        pack=compact(pack,dt,ice_before=details['ice_before'],remap=False)
        pack=grain_growth(pack,dt,temperature_gradient=gradient,refrozen_mass=refrozen)
    pack,flow=drain_liquid(pack,return_ledger=True)
    ground,overflow,overflow_heat=infiltrate(ground,flow['drainage_kgm2'],flow['drainage_enthalpy_jm2'])
    runoff=details['runoff']+flow['runoff_kgm2']+overflow
    runoff_heat=details['runoff_heat']+flow['runoff_enthalpy_jm2']+overflow_heat
    advected=details['ground_heat']+dt*(result['wet_advected_heat_wm2']+result['transpiration_advected_heat_wm2'])
    et=float(details['root_mass'].sum()+details['ground_mass']+details['wet_mass'])
    le=(LV*float(result['transpiration'].sum()+result['wet_evaporation'].sum())+details['ground_latent']
        if canopy_condensate is None else float(result['vapor_parcels']['latent_energy_jm2'].sum()/dt+
            LV*result['transpiration'].sum()+details['ground_latent']))
    atmospheric=float((result['rnet']-result['h']-le)*dt)
    final_heat=float(pack.enthalpy.sum()+ground.enthalpy.sum()+result['canopy_enthalpy_after_jm2'])
    final_mass=float(pack.mass.sum()+ground.water.sum()+details['canopy_water'])
    diagnostics=dict(result,le=le,ground_heat=details['thermal']['input_energy_jm2']/dt,
        ground_nonshortwave_heat=details['thermal']['top_flux_wm2'],
        ground_latent_heat=(details['ground_latent']*dt/details['ground_mass'] if details['ground_mass'] else result['ground_latent_heat']),
        ground_latent_energy_jm2=details['ground_latent']*dt,
        surface_temperature=details['temperature'],thermal=details['thermal'],surface_evaluations=evaluations,
        et_kgm2=et,runoff_kgm2=runoff,runoff_enthalpy_jm2=runoff_heat,
        advected_enthalpy_jm2=advected,atmospheric_energy_jm2=atmospheric,
        energy_residual_jm2=final_heat-original_heat-atmospheric+advected+runoff_heat,
        water_residual_kgm2=final_mass-original_mass+et+runoff,
        snow_drainage_kgm2=flow['drainage_kgm2'],snow_drainage_enthalpy_jm2=flow['drainage_enthalpy_jm2'],
        impurity_output_kgm2=details['residue']+flow['runoff_impurities_kgm2']+flow['drainage_impurities_kgm2'])
    diagnostics.update(thermal_state)
    if canopy_condensate is not None:diagnostics['canopy_condensate']=result['condensate']
    return pack,ground,details['canopy_water'],diagnostics

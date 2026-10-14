"""Conservative, column-aggregated CENTURY-style litter/soil C-N cascade.

Original implementation from the equations/parameter tables cited in
docs/CLM_DECOMPOSITION.md. Three litter pools and coarse wood feed three SOM
pools (fast, slow, passive). No vertical mixing or plant carbon is hidden here.
All transfers are amounts over the step, in gC or gN per square metre.
"""
from dataclasses import dataclass
import math
import numpy as np

YEAR=365.*86400.
POOLS=('metabolic_litter','cellulose_litter','lignin_litter','coarse_wood','fast','slow','passive')
DONOR=np.array([0,1,2,3,3,4,4,5,5,6])
RECEIVER=np.array([4,4,5,1,2,5,6,4,6,4])


@dataclass(frozen=True)
class DecompositionParameters:
    turnover_years: tuple=(.066,.25,.25,4.1,.17,6.1,270.)
    soil_cn: tuple=(8.,11.,11.)
    sand_fraction: float=.4
    wood_cellulose_fraction: float=.76
    q10: float=1.5
    reference_temperature: float=298.15
    dry_potential_mpa: float=-2.5
    wet_potential_mpa: float=-.002

    def __post_init__(self):
        tau=np.asarray(self.turnover_years,float);cn=np.asarray(self.soil_cn,float)
        if (tau.shape!=(7,) or cn.shape!=(3,) or not np.all(np.isfinite(tau)) or not np.all(np.isfinite(cn))
            or np.any(tau<=0.) or np.any(cn<=0.) or not all(math.isfinite(v) for v in
                (self.sand_fraction,self.wood_cellulose_fraction,self.q10,self.reference_temperature,
                 self.dry_potential_mpa,self.wet_potential_mpa))
            or not 0.<=self.sand_fraction<=1. or not 0.<=self.wood_cellulose_fraction<=1.
            or not 1.<=self.q10<=5. or not 250.<=self.reference_temperature<=330.
            or not self.dry_potential_mpa<self.wet_potential_mpa<0.):
            raise ValueError('Invalid decomposition parameters')
        object.__setattr__(self,'turnover_years',tuple(tau));object.__setattr__(self,'soil_cn',tuple(cn))


@dataclass(frozen=True)
class DecompositionState:
    carbon: np.ndarray
    nitrogen: np.ndarray
    mineral_nitrogen: float=0.

    def __post_init__(self):
        c=np.array(self.carbon,float,copy=True);n=np.array(self.nitrogen,float,copy=True)
        if (c.shape!=(7,) or n.shape!=(7,) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(n))
            or np.any(c<0.) or np.any(n<0.) or np.any((c==0.)&(n>0.))
            or not math.isfinite(self.mineral_nitrogen) or self.mineral_nitrogen<0.):
            raise ValueError('Invalid decomposition C/N stores')
        c.setflags(write=False);n.setflags(write=False)
        object.__setattr__(self,'carbon',c);object.__setattr__(self,'nitrogen',n)


def environment_scalars(temperature,water_potential_mpa,oxygen_factor=1.,parameters=DecompositionParameters()):
    """Q10 and logarithmic matric-potential limitation; negative MPa convention.

    Frozen water potential is bounded by the Clapeyron suction magnitude.
    Oxygen is an explicit external multiplier (one when CH4 is disabled).
    """
    if (not all(math.isfinite(v) for v in (temperature,water_potential_mpa,oxygen_factor))
        or not 150.<=temperature<=350. or water_potential_mpa>0. or not 0.<=oxygen_factor<=1.):
        raise ValueError('Invalid decomposition temperature, matric potential or oxygen factor')
    p=parameters
    temperature_factor=p.q10**((temperature-p.reference_temperature)/10.)
    frozen=-334000.*max(273.15-temperature,0.)/(1000.*temperature)
    psi=min(water_potential_mpa,frozen)
    if psi<=p.dry_potential_mpa:moisture=0.
    elif psi>=p.wet_potential_mpa:moisture=1.
    else:moisture=math.log(p.dry_potential_mpa/psi)/math.log(p.dry_potential_mpa/p.wet_potential_mpa)
    return temperature_factor,moisture,temperature_factor*moisture*oxygen_factor


def cascade(parameters=DecompositionParameters()):
    """Ten pathways: donor fractions and pathway respiration fractions."""
    p=parameters;fast_resp=.17+.68*p.sand_fraction;passive_share=.004/(1.-fast_resp)
    path=np.array([1.,1.,1.,p.wood_cellulose_fraction,1.-p.wood_cellulose_fraction,
        1.-passive_share,passive_share,.42/.45,.03/.45,1.])
    respiration=np.array([.55,.5,.5,0.,0.,fast_resp,fast_resp,.55,.55,.55])
    return path,respiration


def step_decomposition(state,temperature,water_potential_mpa,dt,*,carbon_input=None,nitrogen_input=None,
                       mineral_n_input=0.,plant_n_demand=0.,oxygen_factor=1.,parameters=DecompositionParameters()):
    """Advance independent C/N stores and return explicit transfer ledgers.

    External inputs are step amounts added before decomposition. Decay uses
    an exponential donor survival fraction; incoming cascade material waits
    until the next step. Mineral-N competition shares the *starting* mineral
    supply (including deposition) in proportion to immobilization and plant
    demands. Newly mineralized N becomes available next step. This explicit
    split prevents instantaneous reuse of receiving organic pools.
    """
    if (not math.isfinite(dt) or not 0.<dt<=86400. or not math.isfinite(mineral_n_input)
        or mineral_n_input<0. or not math.isfinite(plant_n_demand) or plant_n_demand<0.):
        raise ValueError('Invalid decomposition timestep or mineral N input/demand')
    inputs=[]
    for value in (carbon_input,nitrogen_input):
        a=np.zeros(7) if value is None else np.asarray(value,float)
        if a.shape!=(7,) or not np.all(np.isfinite(a)) or np.any(a<0.):raise ValueError('Invalid external organic C/N input')
        inputs.append(a)
    ci,ni=inputs;c=state.carbon+ci;n=state.nitrogen+ni
    if np.any((c==0.)&(n>0.)):raise ValueError('Organic nitrogen needs a carbon-bearing pool')
    mineral=state.mineral_nitrogen+mineral_n_input
    tf,wf,modifier=environment_scalars(temperature,water_potential_mpa,oxygen_factor,parameters)
    fraction=-np.expm1(-modifier*dt/(np.asarray(parameters.turnover_years)*YEAR))
    path,respiration=cascade(parameters)
    carbon_out=c[DONOR]*fraction[DONOR]*path
    nitrogen_out=n[DONOR]*fraction[DONOR]*path
    carbon_in=carbon_out*(1.-respiration)
    nitrogen_in=np.array([carbon_in[k]/parameters.soil_cn[j-4] if j>=4 else nitrogen_out[k]
        for k,j in enumerate(RECEIVER)])
    demand=nitrogen_in-nitrogen_out
    potential_immobilization=float(np.maximum(demand,0.).sum())
    total_demand=potential_immobilization+plant_n_demand
    limitation=min(1.,mineral/total_demand) if total_demand>0. else 1.
    scale=np.where(demand>0.,limitation,1.)
    carbon_out*=scale;carbon_in*=scale;nitrogen_out*=scale;nitrogen_in*=scale
    delta_n=nitrogen_in-nitrogen_out
    immobilization=float(np.maximum(delta_n,0.).sum());mineralization=float(np.maximum(-delta_n,0.).sum())
    plant_uptake=plant_n_demand*limitation
    new_c=c.copy();new_n=n.copy()
    for k,(i,j) in enumerate(zip(DONOR,RECEIVER)):
        new_c[i]-=carbon_out[k];new_c[j]+=carbon_in[k]
        new_n[i]-=nitrogen_out[k];new_n[j]+=nitrogen_in[k]
    # Clamp only roundoff at an exhausted pool; its contribution remains in
    # the independently reported C/N budget residual and numerical bound.
    eps=np.finfo(float).eps
    if np.any(new_c < -64.*eps*max(float(c.sum()),1.)) or np.any(new_n < -64.*eps*max(float(n.sum()),1.)):
        raise RuntimeError('Decomposition overdraws an organic pool')
    new_c=np.maximum(new_c,0.);new_n=np.maximum(new_n,0.)
    new_mineral=max(0.,mineral-immobilization-plant_uptake)+mineralization
    result=DecompositionState(new_c,new_n,new_mineral)
    rh=float((carbon_out-carbon_in).sum())
    cr=float(new_c.sum()-state.carbon.sum()-ci.sum()+rh)
    nr=float(new_n.sum()+new_mineral-state.nitrogen.sum()-state.mineral_nitrogen-ni.sum()-mineral_n_input+plant_uptake)
    cb=64.*eps*max(float(abs(new_c).sum()+state.carbon.sum()+ci.sum()+rh),1.)
    nb=64.*eps*max(float(abs(new_n).sum()+new_mineral+state.nitrogen.sum()+state.mineral_nitrogen+ni.sum()+mineral_n_input+plant_uptake),1.)
    if abs(cr)>cb or abs(nr)>nb:raise RuntimeError('Decomposition C/N budget exceeds its numerical bound')
    return result,dict(heterotrophic_respiration_gc_m2=rh,heterotrophic_respiration_umol_m2s=rh/(12.0107e-6*dt),
        temperature_factor=tf,moisture_factor=wf,rate_multiplier=modifier,
        potential_immobilization_gn_m2=potential_immobilization,immobilization_gn_m2=immobilization,
        gross_mineralization_gn_m2=mineralization,plant_uptake_gn_m2=plant_uptake,nitrogen_limitation=limitation,
        carbon_out_gc_m2=carbon_out,carbon_in_gc_m2=carbon_in,nitrogen_out_gn_m2=nitrogen_out,nitrogen_in_gn_m2=nitrogen_in,
        carbon_residual_gc_m2=cr,nitrogen_residual_gn_m2=nr,carbon_bound_gc_m2=cb,nitrogen_bound_gn_m2=nb)

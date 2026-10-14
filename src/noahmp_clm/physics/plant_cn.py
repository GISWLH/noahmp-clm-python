"""Simplified plant C/N allocation, turnover, respiration and leaf-N feedback.

Leaf, aggregate stem and fine-root pools plus nonstructural C/N reserves.
Sources, deliberate simplifications and units are in docs/CLM_PLANT_CN.md.
Photosynthesis supplies gross carbon; leaf dark respiration is counted once.
"""
from dataclasses import dataclass
import math
import numpy as np

DAY=86400.
GC_PER_UMOL=12.0107e-6


@dataclass(frozen=True)
class PlantParameters:
    allocation: tuple=(.3,.4,.3)
    target_cn: tuple=(25.,1./(.1/50.+.9/500.),42.)
    turnover_days: tuple=(365.,30.*365.,365.)
    litter_fractions: tuple=(.3,.45,.25)
    growth_respiration_ratio: float=.11
    maintenance_base: float=2.525e-6
    maintenance_q10: float=1.5
    stem_live_n_fraction: float=(.1/50.)/(.1/50.+.9/500.)
    maximum_cn_multiplier: float=2.
    new_carbon_storage_fraction: float=.1
    reserve_release_days: float=30.
    specific_leaf_area: float=.02
    litter_min_cn_multiplier: float=1.5

    def __post_init__(self):
        for name in ('allocation','target_cn','turnover_days','litter_fractions'):
            x=np.asarray(getattr(self,name),float)
            if x.shape!=(3,) or not np.all(np.isfinite(x)) or np.any(x<0.):raise ValueError('Invalid plant parameter vector')
            if name in ('target_cn','turnover_days') and np.any(x<=0.):raise ValueError('Positive plant C:N and lifetimes required')
            if name in ('allocation','litter_fractions') and abs(float(x.sum())-1.)>1e-12:raise ValueError('Plant fractions must sum to one')
            object.__setattr__(self,name,tuple(x))
        scalars=[v for k,v in vars(self).items() if not isinstance(v,tuple)]
        if (not all(math.isfinite(v) for v in scalars) or not 0.<=self.growth_respiration_ratio<=1.
            or self.maintenance_base<0. or not 1.<=self.maintenance_q10<=5.
            or not 0.<=self.stem_live_n_fraction<=1. or self.maximum_cn_multiplier<1.
            or not 0.<=self.new_carbon_storage_fraction<=1. or self.reserve_release_days<=0.
            or self.specific_leaf_area<=0. or self.litter_min_cn_multiplier<1.):
            raise ValueError('Invalid plant physiological parameters')


@dataclass(frozen=True)
class PlantState:
    carbon: np.ndarray
    nitrogen: np.ndarray
    reserve_carbon: float=0.
    reserve_nitrogen: float=0.

    def __post_init__(self):
        c=np.array(self.carbon,float,copy=True);n=np.array(self.nitrogen,float,copy=True)
        if (c.shape!=(3,) or n.shape!=(3,) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(n))
            or np.any(c<0.) or np.any(n<0.) or np.any((c==0.)&(n>0.))
            or not math.isfinite(self.reserve_carbon) or self.reserve_carbon<0.
            or not math.isfinite(self.reserve_nitrogen) or self.reserve_nitrogen<0.):
            raise ValueError('Invalid plant tissue/reserve C/N state')
        c.setflags(write=False);n.setflags(write=False)
        object.__setattr__(self,'carbon',c);object.__setattr__(self,'nitrogen',n)

    @property
    def total_carbon(self):return float(self.carbon.sum()+self.reserve_carbon)

    @property
    def total_nitrogen(self):return float(self.nitrogen.sum()+self.reserve_nitrogen)


def leaf_nitrogen_factor(state,parameters=PlantParameters()):
    """Fraction of target leaf N per unit carbon, used to scale Vcmax/Jmax/TPU."""
    return min(1.,state.nitrogen[0]*parameters.target_cn[0]/state.carbon[0]) if state.carbon[0]>0. else 0.


def plant_lai(state,vegetated_fraction,parameters=PlantParameters()):
    """LAI per vegetated area from leaf gC per total ground area and SLA."""
    if not math.isfinite(vegetated_fraction) or not 0.<=vegetated_fraction<=1.:raise ValueError('Invalid vegetation fraction')
    if vegetated_fraction==0.:
        if np.any(state.carbon>0.):raise ValueError('Bare ground cannot carry plant tissue')
        return 0.
    return float(state.carbon[0]*parameters.specific_leaf_area/vegetated_fraction)


def _timestep(dt):
    if not math.isfinite(dt) or not 0.<dt<=DAY:raise ValueError('Plant timestep must be positive and at most one day')


def _budget(old,new,carbon_in=0.,carbon_out=0.,nitrogen_in=0.,nitrogen_out=0.):
    cr=new.total_carbon-old.total_carbon-carbon_in+carbon_out
    nr=new.total_nitrogen-old.total_nitrogen-nitrogen_in+nitrogen_out
    eps=np.finfo(float).eps
    cb=64.*eps*max(old.total_carbon+new.total_carbon+carbon_in+carbon_out,1.)
    nb=64.*eps*max(old.total_nitrogen+new.total_nitrogen+nitrogen_in+nitrogen_out,1.)
    if abs(cr)>cb or abs(nr)>nb:raise RuntimeError('Plant C/N budget exceeds numerical bound')
    return dict(carbon_residual_gc_m2=cr,nitrogen_residual_gn_m2=nr,carbon_bound_gc_m2=cb,nitrogen_bound_gn_m2=nb)


def turnover_plant(state,dt,*,leaf_retention=1.,parameters=PlantParameters()):
    """Route seasonal/background mortality to four litter/wood pools.

    Retention is the fraction of currently displayed leaves retained during
    this step. Background turnover uses exponential survival. Free leaf-N
    resorption leaves litter no richer than 1.5 times target leaf C:N by default.
    It moves N to reserves, never destroys it. Stem/root N is not resorbed.
    """
    _timestep(dt)
    if not math.isfinite(leaf_retention) or not 0.<=leaf_retention<=1.:raise ValueError('Invalid seasonal leaf retention')
    p=parameters;loss=-np.expm1(-dt/(np.array(p.turnover_days)*DAY))
    loss[0]=1.-(1.-loss[0])*leaf_retention
    fallen_c=state.carbon*loss;fallen_n=state.nitrogen*loss
    recovered=max(0.,fallen_n[0]-fallen_c[0]/(p.litter_min_cn_multiplier*p.target_cn[0]))
    litter_c=np.zeros(7);litter_n=np.zeros(7)
    litter_c[:3]=(fallen_c[0]+fallen_c[2])*np.array(p.litter_fractions)
    litter_n[:3]=(fallen_n[0]-recovered+fallen_n[2])*np.array(p.litter_fractions)
    litter_c[3]=fallen_c[1];litter_n[3]=fallen_n[1]
    new=PlantState(state.carbon-fallen_c,state.nitrogen-fallen_n,state.reserve_carbon,state.reserve_nitrogen+recovered)
    d=_budget(state,new,carbon_out=float(litter_c.sum()),nitrogen_out=float(litter_n.sum()))
    d.update(litter_carbon_gc_m2=litter_c,litter_nitrogen_gn_m2=litter_n,retranslocated_nitrogen_gn_m2=recovered,
        tissue_carbon_loss_gc_m2=fallen_c,tissue_nitrogen_loss_gn_m2=fallen_n)
    return new,d


@dataclass(frozen=True)
class AllocationPlan:
    state: PlantState
    parameters: PlantParameters
    gpp_gc_m2: float
    dt: float
    available_carbon: float
    potential_growth: np.ndarray
    tissue_n_deficit: np.ndarray
    maintenance_respiration: np.ndarray
    unmet_maintenance: np.ndarray
    soil_n_demand: float


def plan_plant_allocation(state,gpp_gc_m2,leaf_respiration_gc_m2,air_temperature,root_temperature,dt,*,
                          target_leaf_carbon,growing=True,parameters=PlantParameters()):
    """Plan demand before competition with microbial immobilization.

    Root temperature here is a representative temperature. Whole-column
    callers may instead map the weighted Q10 factor to its equivalent T.
    Growth N demand includes repair of existing tissue N deficits. Target
    leaf carbon is a ceiling; seasonal turnover, not this plan, removes leaves.
    """
    _timestep(dt)
    if (not all(math.isfinite(v) for v in (gpp_gc_m2,leaf_respiration_gc_m2,air_temperature,root_temperature,target_leaf_carbon))
        or gpp_gc_m2<0. or leaf_respiration_gc_m2<0. or target_leaf_carbon<0.
        or not 150.<=air_temperature<=350. or not 150.<=root_temperature<=350. or not isinstance(growing,bool)):
        raise ValueError('Invalid plant assimilation, respiration, temperature or growth environment')
    p=parameters
    demand=np.array([leaf_respiration_gc_m2,
        state.nitrogen[1]*p.stem_live_n_fraction*p.maintenance_base*dt*p.maintenance_q10**((air_temperature-293.15)/10.),
        state.nitrogen[2]*p.maintenance_base*dt*p.maintenance_q10**((root_temperature-293.15)/10.)])
    cash=state.reserve_carbon+gpp_gc_m2;maintenance_total=float(demand.sum())
    paid=demand*min(1.,cash/maintenance_total) if maintenance_total>0. else demand
    # A resource-limited award spends the entire supply. Subtracting the sum
    # of rounded proportional awards can invent a tiny positive remainder,
    # which would spuriously create leaves and a finite leaf-N factor.
    available=max(0.,cash-maintenance_total)
    current=max(0.,gpp_gc_m2-maintenance_total)
    stored=max(0.,available-current)
    release=-math.expm1(-dt/(p.reserve_release_days*DAY))
    growth_budget=(1.-p.new_carbon_storage_fraction)*current+stored*release if growing else 0.
    potential=np.array(p.allocation)*(growth_budget/(1.+p.growth_respiration_ratio))
    potential[0]=min(potential[0],max(0.,target_leaf_carbon-state.carbon[0]))
    deficit=np.maximum(state.carbon/np.array(p.target_cn)-state.nitrogen,0.)
    n_demand=max(0.,float(deficit.sum()+(potential/np.array(p.target_cn)).sum())-state.reserve_nitrogen)
    return AllocationPlan(state,p,gpp_gc_m2,dt,available,potential,deficit,paid,demand-paid,n_demand)


def apply_plant_allocation(plan,soil_n_supply=0.):
    """Use the actual mineral-N award, retaining unused C/N in reserves.

    Existing N deficits are repaired first. New growth may dilute tissue N
    up to the configured maximum C:N; beyond that, C growth is limited. This
    makes nitrogen shortage affect both biomass and the subsequent Vcmax.
    """
    old=plan.state;p=plan.parameters
    if not math.isfinite(soil_n_supply) or soil_n_supply<0. or soil_n_supply>plan.soil_n_demand+32.*np.finfo(float).eps*max(plan.soil_n_demand,1.):
        raise ValueError('Soil nitrogen award must lie within the planned demand')
    supply=old.reserve_nitrogen+soil_n_supply;deficit_total=float(plan.tissue_n_deficit.sum())
    repair=plan.tissue_n_deficit*min(1.,supply/deficit_total) if deficit_total>0. else np.zeros(3)
    available=max(0.,supply-deficit_total)
    target_n=plan.potential_growth/np.array(p.target_cn)
    minimum=float(target_n.sum()/p.maximum_cn_multiplier)
    growth=plan.potential_growth*min(1.,available/minimum) if minimum>0. else plan.potential_growth.copy()
    target_n=growth/np.array(p.target_cn);target_total=float(target_n.sum())
    allocated_n=target_n*min(1.,available/target_total) if target_total>0. else np.zeros(3)
    reserve_n=max(0.,available-target_total)
    growth_respiration=p.growth_respiration_ratio*growth
    reserve_c=max(0.,plan.available_carbon-float(growth.sum()+growth_respiration.sum()))
    new=PlantState(old.carbon+growth,old.nitrogen+repair+allocated_n,reserve_c,reserve_n)
    ra=float(plan.maintenance_respiration.sum()+growth_respiration.sum())
    d=_budget(old,new,carbon_in=plan.gpp_gc_m2,carbon_out=ra,nitrogen_in=soil_n_supply)
    d.update(gpp_gc_m2=plan.gpp_gc_m2,npp_gc_m2=plan.gpp_gc_m2-ra,autotrophic_respiration_gc_m2=ra,
        maintenance_respiration_gc_m2=plan.maintenance_respiration,growth_respiration_gc_m2=growth_respiration,
        unmet_maintenance_gc_m2=plan.unmet_maintenance,carbon_growth_gc_m2=growth,
        nitrogen_growth_gn_m2=allocated_n,nitrogen_repair_gn_m2=repair,soil_n_demand_gn_m2=plan.soil_n_demand,
        soil_n_uptake_gn_m2=soil_n_supply,nitrogen_factor=leaf_nitrogen_factor(new,p),
        root_respiration_gc_m2=float(plan.maintenance_respiration[2]+growth_respiration[2]))
    return new,d


def carbon_exchange_diagnostics(gpp_gc_m2,autotrophic_respiration_gc_m2,heterotrophic_respiration_gc_m2,
                                 root_respiration_gc_m2,dt):
    """NEE is positive to the atmosphere; soil respiration includes roots + RH."""
    _timestep(dt)
    values=(gpp_gc_m2,autotrophic_respiration_gc_m2,heterotrophic_respiration_gc_m2,root_respiration_gc_m2)
    if not all(math.isfinite(v) and v>=0. for v in values) or root_respiration_gc_m2>autotrophic_respiration_gc_m2:
        raise ValueError('Invalid carbon exchange ledger')
    ra=autotrophic_respiration_gc_m2;rh=heterotrophic_respiration_gc_m2;nee=ra+rh-gpp_gc_m2
    return dict(gpp_gc_m2=gpp_gc_m2,npp_gc_m2=gpp_gc_m2-ra,nee_gc_m2=nee,
        autotrophic_respiration_gc_m2=ra,heterotrophic_respiration_gc_m2=rh,
        ecosystem_respiration_gc_m2=ra+rh,soil_respiration_gc_m2=root_respiration_gc_m2+rh,
        gpp_umol_m2s=gpp_gc_m2/(GC_PER_UMOL*dt),nee_umol_m2s=nee/(GC_PER_UMOL*dt))

"""Simplified field-capacity tile drainage with explicit liquid enthalpy.

Noah-MP technical note section 3.7, equations 302--304 motivate excess-water
removal from a specified drain layer. This option uses one tube depth, a
column-wide rate cap, and no flow in a frozen drain layer. It is not the
Hooghoudt scheme. See docs/CLM_TILE_DRAINAGE.md for assumptions and units.
"""
from dataclasses import dataclass
import math
import numpy as np
from .snowpack import TF,LF,CW
from .soil_enthalpy import SoilEnthalpyState,withdraw_liquid


@dataclass(frozen=True)
class TileDrainageParameters:
    tube_depth_m: float=1.
    field_capacity: float=.28
    maximum_mm_day: float=10.

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in vars(self).values())
            or self.tube_depth_m<=0. or not 0.<=self.field_capacity<=1. or self.maximum_mm_day<0.):
            raise ValueError('Invalid tile depth, field capacity or drainage rate')


def tile_drainage_demand(liquid,ice,dz,temperature,dt,parameters=TileDrainageParameters()):
    """Liquid removal [kg/m2] and diagnostics, without modifying soil stores.

    At an exact layer interface the shallower layer contains the drain. The
    configured depth must lie inside the model column. Ice is never removed;
    subfreezing drain layers are closed rather than assigned an uncalibrated
    ice permeability. The cap in mm/day is converted to mass per actual step.
    """
    water,frozen,z,t=[np.asarray(v,float) for v in (liquid,ice,dz,temperature)]
    p=parameters
    if (not isinstance(p,TileDrainageParameters) or water.ndim!=1 or water.size==0
        or any(v.shape!=water.shape for v in (frozen,z,t))
        or not all(np.all(np.isfinite(v)) for v in (water,frozen,z,t))
        or np.any(water<0.) or np.any(frozen<0.) or np.any(z<=0.)
        or np.any(water+frozen>1000.*z) or np.any(t<150.) or np.any(t>400.)
        or not math.isfinite(dt) or not 0.<dt<=86400. or p.tube_depth_m>float(z.sum())):
        raise ValueError('Invalid tile drainage soil, depth or timestep')
    layer=int(np.searchsorted(np.cumsum(z),p.tube_depth_m,side='left'))
    retained=max(0.,1000.*p.field_capacity*z[layer]-frozen[layer])
    excess=min(water[layer],max(0.,water[layer]-retained))
    eligible=t[layer]>=TF and frozen[layer]<=1e-9
    capacity=p.maximum_mm_day*(dt/86400.)
    removed=min(excess,capacity) if eligible else 0.
    demand=np.zeros(water.size);demand[layer]=removed
    heat=removed*(LF+CW*(t[layer]-TF))
    return demand,dict(layer_index=layer,eligible=bool(eligible),excess_kgm2=excess,
        capacity_kgm2=capacity,drained_kgm2=removed,drained_enthalpy_jm2=heat)


def step_tile_drainage(soil,dt,parameters=TileDrainageParameters()):
    """Withdraw actual liquid and its heat; return the independent outlet."""
    if not isinstance(soil,SoilEnthalpyState):raise ValueError('Tile drainage requires extensive soil state')
    if parameters.field_capacity>soil.params.smcmax:raise ValueError('Tile field capacity exceeds soil porosity')
    t=soil.temperature;liquid=np.where(t>=TF,soil.water,np.clip(soil.liquid,0.,soil.water))
    demand,d=tile_drainage_demand(liquid,soil.water-liquid,soil.dz,t,dt,parameters)
    after,heat=withdraw_liquid(soil,demand)
    d['drained_enthalpy_jm2']=float(heat.sum())
    water_error=float(after.water.sum()-soil.water.sum())+d['drained_kgm2']
    heat_error=float(after.enthalpy.sum()-soil.enthalpy.sum())+d['drained_enthalpy_jm2']
    eps=np.finfo(float).eps
    water_bound=64.*eps*max(float(after.water.sum()+soil.water.sum())+d['drained_kgm2'],1.)
    heat_bound=64.*eps*max(float(abs(after.enthalpy).sum()+abs(soil.enthalpy).sum())+abs(d['drained_enthalpy_jm2']),1.)
    if abs(water_error)>water_bound or abs(heat_error)>heat_bound:
        raise RuntimeError('Tile drainage water/heat residual exceeds numerical bound')
    d.update(water_residual_kgm2=water_error,heat_residual_jm2=heat_error,
        water_bound_kgm2=water_bound,heat_bound_jm2=heat_bound)
    return after,d

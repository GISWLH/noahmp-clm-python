"""Declared retained operands; NumPy physics is called only by CPU tests."""
from pathlib import Path
import hashlib,json
import numpy as np


def fixture():
    path=Path(__file__).parent/'fixtures/qia3803_coordinate_recovery.npz'
    meta=json.loads(path.with_suffix('.json').read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load(path,allow_pickle=False) as z:return dict(z)


def operands(v):
    from noahmp_clm.column import SoilParams
    from noahmp_clm.physics.snowpack import Snowpack
    from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
    from noahmp_clm.physics.canopy_interception import CanopyCondensate
    from noahmp_clm.physics.canopy_energy import CanopyExchange
    from noahmp_clm.physics.photosynthesis import LeafParameters
    c,g,p,o,leaf,u,f,b,cond=[v['input/surface_a'+str(k)][0].astype(float) for k in range(9)]
    n=int(o[0]);snow=Snowpack(c[:n,0],c[:n,1],c[:n,2],c[:n,3],c[:n,4:])
    props=SoilParams(smcmax=p[0],bexp=p[1],smcref=p[2],satdk=p[3],quartz=p[4],smcwlt=p[5],saturated_suction_m=p[6])
    soil=SoilEnthalpyState(g[:,0],g[:,1],g[:,2],props,int(o[1]));water=CanopyCondensate(cond[:,0],cond[:,1])
    ext=v['input/exchange_extension'][0];exchange=CanopyExchange(*u[:11],atmosphere_water_velocity=float(ext[0]),potential_temperature=float(ext[1]))
    kw=dict(stem_area=v['input/stem_area'][0].astype(float),vegetated_fraction=u[13],transpiration_limit=u[16],ground_emissivity=u[18],ground_albedo=u[19:21],diffuse_fraction=u[21],cosine_zenith=u[22],nitrogen_factor=f[:,0],water_stress=f[:,1],leaf_parameters=LeafParameters(**json.loads(str(v['leaf_parameters_json']))))
    return snow,soil,water,leaf,u,b,exchange,kw


def fixed_operands(v):
    from noahmp_clm.physics.soil_uptake import SoilWaterSupply
    snow,soil,water,leaf,u,b,exchange,kw=operands(v);dt=b[0];temperature=float(v['fixed_ground_temperature'][0])
    kw=dict(kw,soil_supply=SoilWaterSupply(np.maximum(soil.liquid-soil.params.smcwlt*1000.*soil.dz,0.)/dt,b[3:],soil.liquid[0]/dt),root_enthalpy=334000.+4180.*(soil.temperature-273.15),ground_evaporation_limit=snow.mass[0]/dt if len(snow.mass) else np.inf,fixed_ground_temperature=temperature,ground_is_ice=temperature<273.15,ground_shares_soil=not len(snow.mass),snowpack=snow)
    return (leaf,u[11],u[12],exchange,water,dt),kw

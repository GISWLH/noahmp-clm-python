"""Independent equation checks for the extensive surface connection."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
from noahmp_clm.column import ColumnState,default_soil_params
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.canopy_column import surface_step
from noahmp_clm.physics.canopy_energy import CanopyExchange,CanopyWaterStorage,solve_canopy_energy
from noahmp_clm.physics.soil_uptake import SoilWaterSupply

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);args=ap.parse_args()
    start=time.perf_counter();reference=module('surface_reference','tests/test_surface_independent.py')
    leaf=module('leaf_reference','tests/test_canopy_energy.py')
    profiles=[(0,290.,288.,0.,.5,600.),(0,290.,288.,0.,.95,0.),(0,265.,268.,0.,.1,200.),
        (0,269.,266.,0.,.99,0.),(1,265.,268.,-12000.,.4,200.),(3,265.,270.,-6000.,.99,0.),
        (5,280.,271.,100000.,.4,800.),(1,285.,280.,320000.,.4,600.),(3,275.,274.,200000.,.99,0.)]
    maxima=np.zeros(8);rows=[]
    for texture in ('sand','loam','clay'):
        for speed in (.01,.04):
            for profile in profiles:
                ns,air,soil_t,snow_h,rh,solar=profile
                m=np.arange(1,ns+1)*3.;snow=Snowpack(m,m*snow_h,m/250.) if ns else Snowpack.empty()
                soil=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,soil_t),smc=np.full(4,.3),soil=default_soil_params(texture)))
                es=611.2*np.exp(17.67*(air-TF)/(air-TF+243.5));q=rh*.622*es/(101325.-.378*es)
                e=CanopyExchange(air,q,101325.,speed,.004,.03,soil_t,0.)
                _,_,_,d=surface_step(snow,soil,1800.,[0.]*10,solar,300.,e,vegetated_fraction=0.,ground_albedo=(.2,.3))
                h,t,w,m,evap,hflux,le,residual=reference.independent_bare_surface(snow,soil,1800.,solar,300.,e)
                difference=np.array([np.max(np.abs(np.r_[d['thermal_snow_temperature'],d['thermal_soil_temperature']]-t)),
                    np.max(np.abs(np.r_[d['thermal_snow_enthalpy'],d['thermal_soil_enthalpy']]-h)),
                    abs(d['et_kgm2']-evap),abs(d['h']-hflux),abs(d['le']-le),
                    abs(d['energy_residual_jm2']),abs(d['water_residual_kgm2']),np.max(np.abs(residual))])
                maxima=np.maximum(maxima,difference)
                rows.append(dict(texture=texture,atmosphere_velocity=speed,profile=profile,differences=difference.tolist()))
    labels=['temperature_k','enthalpy_jm2','et_kgm2','h_wm2','le_wm2','energy_budget_jm2','water_budget_kgm2','independent_equation_wm2']
    report={'scope':'54 bare surface simultaneous heat/vapor references; 12 fixed-ground canopy storage references; not official output or observations',
        'bare_joint':dict(cases=len(rows),maxima=dict(zip(labels,maxima.tolist())),cases_detail=rows)}
    maxima=np.zeros(5);rows=[]
    for source_t in (265.,280.,300.):
        for mass,temperature,humidity,sw in [(0.,285.,.008,600.),(.3,278.,.008,800.),(.8,305.,.015,50.),(.1,285.,.022,0.)]:
            supply=SoilWaterSupply(np.full(4,1e-4),np.array([.6,.25,.1,.05]),1e-4)
            store=CanopyWaterStorage(mass,mass*(334000.+4180.*(temperature-TF)),1800.,334000.+4180.*(np.arange(4)*2.+source_t-TF))
            e=leaf.boundary(specific_humidity=humidity)
            kw=dict(soil_supply=supply,canopy_evaporation_limit=mass/1800.,wet_fraction=.5,canopy_storage=store,fixed_ground_temperature=295.)
            d=solve_canopy_energy([.3]*10,sw,330.,e,**kw);ref=leaf.independent_energy_reference([.3]*10,sw,330.,e,**kw)
            difference=np.r_[np.max(np.abs(np.r_[d['leaf_temperature'],d['ground_temperature']]-ref['temperature'])),
                [abs(d[k]-ref[k]) for k in ('h','le','ground_heat','gpp')]]
            maxima=np.maximum(maxima,difference)
            rows.append(dict(source_temperature_k=source_t,old_canopy_mass=mass,old_canopy_temperature_k=temperature,differences=difference.tolist()))
    report['canopy_storage']=dict(cases=len(rows),maxima=dict(zip(['temperature_k','h_wm2','le_wm2','g_wm2','gpp_umol_m2s'],maxima.tolist())),cases_detail=rows)
    paths=[Path(__file__),ROOT/'tests/test_surface_independent.py',ROOT/'tests/test_canopy_energy.py']+list((ROOT/'src/noahmp_clm/physics').glob('*.py'))
    report['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report['wall_seconds']=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:{key:value for key,value in report[k].items() if key!='cases_detail'} for k in ('bare_joint','canopy_storage')},indent=2))
    assert report['bare_joint']['maxima']['temperature_k']<2e-6 and report['bare_joint']['maxima']['energy_budget_jm2']<.02
    assert maxima[0]<5e-5 and np.max(maxima[1:])<3e-4


if __name__=='__main__': main()

"""Independent equation audit for the NumPy canopy / snow boundary."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
from noahmp_clm.physics.canopy_energy import CanopyExchange,solve_canopy_energy,_ice_saturation
from noahmp_clm.physics.snowpack import Snowpack
from noahmp_clm.physics.soil_uptake import SoilWaterSupply
from noahmp_clm.physics.canopy_radiation import two_stream_shortwave

ROOT=Path(__file__).resolve().parents[1]
def test_module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f'tests/{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args()
    energy=test_module('test_canopy_energy');radiation=test_module('test_canopy_radiation')
    rng=np.random.default_rng(51628);start=time.perf_counter();rows=[]
    for case in range(60):
        temperature=rng.uniform(245.,273.15);area=rng.uniform(.02,.7,10);fraction=(0.,.6,1.)[case%3]
        mass=rng.uniform(2.,40.,case%5+1);pack=Snowpack.from_ice(mass,temperature,density=rng.uniform(100.,500.))
        e=CanopyExchange(temperature+rng.uniform(-2.,3.),rng.uniform(.0001,.002),90000.,.02,.003,.03,274.,15.)
        options=dict(vegetated_fraction=fraction,fixed_ground_temperature=temperature,
            ground_is_ice=True,snowpack=pack,ground_shares_soil=False,ground_evaporation_limit=1e-6,
            soil_supply=SoilWaterSupply(np.array([0.,2e-6,2e-6,2e-6]),np.array([.6,.3,.1,0.]),0.),
            diffuse_fraction=rng.uniform(0.,1.),cosine_zenith=rng.uniform(.1,1.))
        sw=0. if case%5==0 else rng.uniform(50.,850.);lw=rng.uniform(180.,300.)
        result=solve_canopy_energy(area,sw,lw,e,**options)
        ref=energy.independent_energy_reference(area,sw,lw,e,**options)
        delta={key:abs(float(result[key]-ref[key])) for key in ('h','le','ground_heat','gpp')}
        delta['temperature_k']=float(np.max(np.abs(np.r_[result['leaf_temperature'],result['ground_temperature']]-ref['temperature'])))
        delta['humidity_kgkg']=abs(float(result['canopy_humidity']-ref['canopy_humidity']))
        rows.append(dict(case=case,layers=len(mass),vegetated_fraction=fraction,ground_temperature_k=temperature,
            absolute_error=delta,layer_energy_residual_wm2=float(np.max(np.abs(result['layer_residual']))),
            total_energy_residual_wm2=float(sw+lw-result['outgoing_shortwave']-result['outgoing_longwave']-result['h']-result['le']-result['ground_heat']),
            air_heat_residual_wm2=float(result['air_heat_residual']),air_water_residual_kgm2s=float(result['air_water_residual']),
            iterations=result['iterations']))
    t=np.linspace(208.15,273.15,800);es=_ice_saturation(t,90000.)[2]
    reference=100.*10.**(-9.09718*(273.16/t-1)-3.56654*np.log10(273.16/t)+.876793*(1-t/273.16)+np.log10(6.1071))
    max_flux=max_relative=max_budget=0.
    for af,ad in ((.15,.3),(.5,.85),(.95,.6)):
        for mu in (.1,.5,1.):
            area=np.linspace(.05,.55,10)
            result=two_stream_shortwave(area,400.,200.,leaf_reflectance=.45,leaf_transmittance=.25,
                ground_albedo=af,ground_albedo_direct=ad,cosine_zenith=mu)
            ref=radiation.continuous_reference(area,.45,.25,mu,ag=af,ag_direct=ad)
            for key,value in ref.items():
                error=np.abs(result[key]-value);max_flux=max(max_flux,float(np.max(error)))
                max_relative=max(max_relative,float(np.max(error/np.maximum(np.abs(value),1.))))
            max_budget=max(max_budget,abs(result['residual']))
    report=dict(scope='NumPy fixed snow-surface leaf/air exchange, ice saturation, and canopy/snow radiation boundary; not a time-stepped whole column or observed-site validation',
        cases=60,max_absolute_error={key:max(row['absolute_error'][key] for row in rows) for key in rows[0]['absolute_error']},
        max_layer_energy_residual_wm2=max(row['layer_energy_residual_wm2'] for row in rows),
        max_total_energy_residual_wm2=max(abs(row['total_energy_residual_wm2']) for row in rows),
        max_air_heat_residual_wm2=max(abs(row['air_heat_residual_wm2']) for row in rows),
        max_air_water_residual_kgm2s=max(abs(row['air_water_residual_kgm2s']) for row in rows),
        max_iterations=max(row['iterations'] for row in rows),
        ice_saturation=dict(reference='Smithsonian/Goff-Gratch via NCAR EOL',temperatures=800,
            min_temperature_k=float(t.min()),max_temperature_k=float(t.max()),max_relative_bias_percent=float(100*np.max(np.abs(es/reference-1.)))),
        split_albedo_radiation=dict(cases=9,reference='independent SciPy matrix-exponential boundary-value solve',
            max_flux_difference_wm2=max_flux,max_relative_difference_percent=100*max_relative,max_budget_residual_wm2=max_budget),
        rows=rows,wall_seconds=time.perf_counter()-start)
    sources=[Path(__file__),ROOT/'tests/test_canopy_energy.py',ROOT/'tests/test_canopy_radiation.py',ROOT/'src/noahmp_clm/data/snow_tables.npz']+list((ROOT/'src/noahmp_clm/physics').glob('*.py'))
    report['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_sha256')},indent=2))
    assert report['max_absolute_error']['temperature_k']<.01 and report['max_layer_energy_residual_wm2']<5e-5


if __name__=='__main__': main()

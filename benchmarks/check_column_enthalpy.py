"""Coupled snow/soil thermal and advected-heat audit; not a canopy/site run."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
from noahmp_clm.column import ColumnState,default_soil_params
from noahmp_clm.physics.snowpack import Snowpack,TF,CI,CW,LF,add_snowfall,add_rain,drain_liquid,remove_sublimation
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState,infiltrate,withdraw_liquid
from noahmp_clm.physics.column_heat import coupled_heat_step
from noahmp_clm.physics.snow_evolution import compact,grain_growth,fresh_snow_properties,deposit_impurities
from noahmp_clm.physics.snicar import snow_shortwave

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);args=ap.parse_args()
    spec=importlib.util.spec_from_file_location('independent',ROOT/'tests/test_column_heat.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    start=time.perf_counter();rng=np.random.default_rng(62643)
    report={'scope':'Coupled snow/soil heat and advected mass/enthalpy; prescribed atmospheric exchange, no canopy, no official model or observed-site reference'}
    maxima=np.zeros(3);max_iterations=0
    for case in range(120):
        ns=case%6;m=rng.uniform(.5,50.,ns)
        snow=Snowpack(m,m*rng.uniform(-CI*20,LF*1.1,ns),m/rng.uniform(80,550,ns))
        column=ColumnState(soil=default_soil_params(('sand','loam','clay')[case%3]),
            tsoil=rng.uniform(TF-12,TF+8,4),smc=rng.uniform(.1,.35,4))
        ground=SoilEnthalpyState.from_column(column)
        k=rng.uniform(.05,1.5,ns+4);sw=rng.uniform(0,50,ns);sg=rng.uniform(0,20,4)
        dt=rng.uniform(30,3600);top=rng.uniform(-100,150);air=rng.uniform(TF-20,TF+15);exchange=rng.uniform(0,15)
        s,g,d=coupled_heat_step(snow,ground,dt,top,snow_absorbed=sw,soil_absorbed=sg,
            top_temperature=air,top_conductance=exchange,thermal_conductivity=k)
        href,tref=module.independent_coupled_reference(snow,ground,dt,top,sw,sg,k,air,exchange)
        maxima=np.maximum(maxima,[np.max(np.abs(np.r_[s.enthalpy,g.enthalpy]-href)),
            np.max(np.abs(np.r_[s.temperature,g.temperature]-tref)),abs(d['residual_jm2'])])
        max_iterations=max(max_iterations,d['iterations'])
    report['independent_nonlinear_reference']=dict(cases=120,max_enthalpy_difference_jm2=float(maxima[0]),
        max_temperature_difference_k=float(maxima[1]),max_energy_balance_jm2=float(maxima[2]),max_iterations=max_iterations)
    snow=Snowpack.from_ice([.1],TF,100.)
    ground=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,TF-1.),smc=np.full(4,.25)))
    before=snow.enthalpy.sum()+ground.enthalpy.sum()
    s,g,d=coupled_heat_step(snow,ground,1800.,300.)
    s,dr=drain_liquid(s,return_ledger=True)
    g,over,overh=infiltrate(g,dr['drainage_kgm2'],dr['drainage_enthalpy_jm2'])
    report['complete_melt']=dict(input_jm2=540000.,initial_enthalpy_jm2=float(before),
        final_soil_enthalpy_jm2=float(g.enthalpy.sum()),soil_conductive_gain_jm2=d['soil_enthalpy_change_jm2'],
        meltwater_kgm2=dr['drainage_kgm2'],meltwater_enthalpy_jm2=dr['drainage_enthalpy_jm2'],
        runoff_kgm2=over+dr['runoff_kgm2'],runoff_enthalpy_jm2=overh+dr['runoff_enthalpy_jm2'],
        energy_residual_jm2=float(s.enthalpy.sum()+g.enthalpy.sum()+overh+dr['runoff_enthalpy_jm2']-before-540000.))
    snow=Snowpack.empty()
    ground=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,TF-2.),smc=np.full(4,.25)))
    mass_initial=ground.water.sum();heat_initial=ground.enthalpy.sum()
    mass_in=mass_out=heat_in=heat_out=0.;et=runoff=drainage=0.;max_layers=0;max_swe=0.
    max_mass_error=max_heat_error=max_step_heat=0.;min_t=1000.;max_t=0.;steps_with_snow=0
    trace_in=np.zeros(5);trace_out=np.zeros(5);max_trace_error=0.;daily=[]
    dt=1800.
    for step in range(365*48):
        day=step/48.;hour=(step%48)*.5
        air=TF-2.-12.*np.cos(2*np.pi*(day-15)/365)+3.*np.sin(2*np.pi*hour/24)
        sun=max(0.,np.sin(np.pi*(hour-6)/12))
        solid=.07 if air<TF and (step//48)%5<2 else 0.
        rain=.035 if air>=TF and (step//48)%7==0 else 0.
        if solid:
            rho,radius=fresh_snow_properties(air,2.)
            snow=add_snowfall(snow,solid,min(air,TF),float(rho),float(radius))
            mass_in+=solid;heat_in+=solid*CI*(min(air,TF)-TF)
        if rain:
            incoming=rain*(LF+CW*(air-TF));mass_in+=rain;heat_in+=incoming
            if snow.mass.size: snow=add_rain(snow,rain,air)
            else:
                ground,out,out_h=infiltrate(ground,rain,incoming)
                mass_out+=out;heat_out+=out_h;runoff+=out
        latent=0.
        if snow.mass.size:
            loss=min(.0005,snow.ice.sum())
            snow,advected,residue=remove_sublimation(snow,loss)
            mass_out+=loss;heat_out+=advected;trace_out+=residue;et+=loss
            latent=(2.5e6+LF)*loss/dt
        else:
            available=np.maximum(0.,ground.liquid-1000*.12*ground.dz)
            demand=np.minimum(available,.02*sun*np.array([.6,.25,.1,.05]))
            ground,advected=withdraw_liquid(ground,demand)
            mass_out+=demand.sum();heat_out+=advected.sum();et+=demand.sum()
            latent=2.5e6*demand.sum()/dt
        optics=snow_shortwave(snow,[150*sun,150*sun],[50*sun,50*sun],cosine_zenith=max(.05,sun))
        ice_before=snow.ice.copy()
        snow,ground,d=coupled_heat_step(snow,ground,dt,-20.-latent,
            snow_absorbed=optics['absorbed'],soil_absorbed=[optics['ground_absorbed'],0.,0.,0.],
            top_temperature=air,top_conductance=8.)
        heat_in+=d['input_energy_jm2'];max_step_heat=max(max_step_heat,abs(d['residual_jm2']))
        snow=compact(snow,dt,ice_before=ice_before,remap=False)
        snow=grain_growth(snow,dt,temperature_gradient=10.)
        snow,flow=drain_liquid(snow,return_ledger=True)
        mass_out+=flow['runoff_kgm2'];heat_out+=flow['runoff_enthalpy_jm2'];runoff+=flow['runoff_kgm2']
        trace_out+=flow['runoff_impurities_kgm2']+flow['drainage_impurities_kgm2']
        ground,out,out_h=infiltrate(ground,flow['drainage_kgm2'],flow['drainage_enthalpy_jm2'])
        mass_out+=out;heat_out+=out_h;runoff+=out;drainage+=flow['drainage_kgm2']
        rates=np.array([1.,2.,1.,.5,.25])*1e-12
        snow,residue=deposit_impurities(snow,rates,dt);trace_in+=rates*dt;trace_out+=residue
        mass_error=snow.mass.sum()+ground.water.sum()+mass_out-mass_initial-mass_in
        heat_error=snow.enthalpy.sum()+ground.enthalpy.sum()+heat_out-heat_initial-heat_in
        max_mass_error=max(max_mass_error,abs(mass_error));max_heat_error=max(max_heat_error,abs(heat_error))
        max_trace_error=max(max_trace_error,np.max(np.abs(snow.impurities.sum(0)+trace_out-trace_in)))
        max_swe=max(max_swe,snow.swe);max_layers=max(max_layers,len(snow.mass));steps_with_snow+=bool(len(snow.mass))
        temperatures=np.r_[snow.temperature,ground.temperature]
        min_t=min(min_t,float(temperatures.min()));max_t=max(max_t,float(temperatures.max()))
        if step%48==47:
            daily.append(dict(day=step//48+1,swe_m=snow.swe,soil_temperature_k=ground.temperature.tolist(),
                soil_water_kgm2=ground.water.tolist(),soil_ice_kgm2=ground.ice.tolist(),
                mass_error_kgm2=float(mass_error),heat_error_jm2=float(heat_error)))
            if (step//48+1)%30==0:
                print(f'day {step//48+1}: water={max_mass_error:.9g} kg/m2, enthalpy={max_heat_error:.9g} J/m2',flush=True)
    report['synthetic_year']=dict(steps=17520,steps_with_snow=steps_with_snow,max_layers=max_layers,max_swe_m=max_swe,
        min_temperature_k=min_t,max_temperature_k=max_t,et_kgm2=float(et),runoff_kgm2=float(runoff),
        meltwater_into_soil_kgm2=float(drainage),mass_input_kgm2=mass_in,mass_output_kgm2=float(mass_out),
        max_water_error_kgm2=float(max_mass_error),max_total_enthalpy_error_jm2=float(max_heat_error),
        max_thermal_step_residual_jm2=float(max_step_heat),max_impurity_error_kgm2=float(max_trace_error),daily=daily)
    sources=[Path(__file__),ROOT/'tests/test_column_heat.py',ROOT/'src/noahmp_clm/column.py',ROOT/'src/noahmp_clm/options.py',
        ROOT/'src/noahmp_clm/data/snow_tables.npz']+list((ROOT/'src/noahmp_clm/physics').glob('*.py'))
    report['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report['wall_seconds']=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','synthetic_year')},indent=2))
    print(json.dumps({k:v for k,v in report['synthetic_year'].items() if k!='daily'},indent=2))
    assert maxima[1]<.01 and max_mass_error<1e-7 and max_heat_error<.01


if __name__=='__main__': main()

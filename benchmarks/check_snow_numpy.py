"""Numeric independent-reference and 365-day component budgets for new snow.

This is NOT an official CTSM/HRLDAS or observed-site comparison, and NOT the
coupled CLM column: bottom heat/water are exported to a recorded reservoir.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time
import numpy as np
from scipy.integrate import solve_ivp

from noahmp_clm.physics.snowpack import (Snowpack,TF,CI,LF,add_snowfall,add_rain,
    drain_liquid,remove_sublimation)
from noahmp_clm.physics.snow_heat import heat_step
from noahmp_clm.physics.snicar import snow_shortwave,spectral_fluxes
from noahmp_clm.physics.snow_evolution import (compact,grain_growth,aging_parameters,
    fresh_snow_properties,deposit_impurities)

ROOT=Path(__file__).resolve().parents[1]


def fixture(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path);args=ap.parse_args()
    start=time.perf_counter();rng=np.random.default_rng(42612)
    report={'scope':'NumPy component analytic/SciPy references and synthetic snow-only year; no observational or official model acceptance',
        'heat':{},'radiation':{},'aging':{}}
    independent=fixture('test_snow_heat').independent_scipy_reference
    maxima=np.zeros(4);iterations=0
    for _ in range(200):
        n=rng.integers(1,6);m=rng.uniform(2.,50.,n);z=m/rng.uniform(80.,600.,n)
        h=m*rng.uniform(-CI*25,LF*1.1,n);p=Snowpack(m,h,z)
        dt=rng.uniform(30.,3600.);top=rng.uniform(-300.,600.)
        sw=rng.uniform(0.,100.,n);tb=rng.uniform(TF-15,TF+10);gb=rng.uniform(0,20);k=rng.uniform(.05,1.,n)
        new,d=heat_step(p,dt,top,absorbed=sw,bottom_temperature=tb,bottom_conductance=gb,thermal_conductivity=k)
        ref=independent(p,dt,top,sw,tb,gb,k);rp=p.with_enthalpy(ref)
        maxima=np.maximum(maxima,[np.max(np.abs(new.enthalpy-ref)),np.max(np.abs(new.temperature-rp.temperature)),
            np.max(np.abs(new.liquid-rp.liquid)),abs(d['residual_jm2'])])
        iterations=max(iterations,d['iterations'])
    report['heat']=dict(cases=200,max_enthalpy_error_jm2=float(maxima[0]),max_temperature_error_k=float(maxima[1]),
        max_liquid_error_kgm2=float(maxima[2]),max_balance_error_jm2=float(maxima[3]),max_iterations=iterations)
    independent=fixture('test_snicar').independent_spectrum
    maxima=np.zeros(3)
    for eddington in (False,True):
        for _ in range(120):
            n=rng.integers(1,6);tau=rng.uniform(.001,3,n);w=rng.uniform(.9 if eddington else .1,1.,n)
            g=rng.uniform(.1,.95,n);mu=rng.uniform(.05,1.);alb=rng.uniform(0.,.95)
            beam,diffuse=rng.uniform(0.,800.,2)
            a=spectral_fluxes(tau,w,g,beam,diffuse,alb,mu,eddington=eddington)
            b=independent(tau,w,g,beam,diffuse,alb,mu,eddington)
            differences=[np.max(np.abs(np.asarray(a[key])-b[key])) for key in b]
            relative=[np.max(np.divide(np.abs(np.asarray(a[key])-b[key]),np.abs(b[key]),
                out=np.zeros_like(np.asarray(a[key])),where=np.abs(b[key])>1.)) for key in b]
            maxima=np.maximum(maxima,[max(differences),max(relative),abs(a['residual'])])
    report['radiation']=dict(cases=240,max_flux_error_wm2=float(maxima[0]),
        max_relative_error_above_1wm2_percent=float(maxima[1]*100),max_balance_wm2=float(maxima[2]))
    example={}
    for radius in (54.5,500.,1500.):
        for bc in (0.,1e-6):
            p=Snowpack.from_ice([100.],TF-5,300.,radius)
            p=Snowpack(p.mass,p.enthalpy,p.thickness,p.radius_um,[[bc*100,0.,0.,0.,0.]])
            out=snow_shortwave(p,[300.,300.],[100.,100.])
            example[f'radius_{radius}_bc_kgkg_{bc}']={k:out[k] for k in ('albedo','ground_absorbed','reflected','residual')}
    report['radiation']['examples']=example
    errors=[]
    for temp,grad,rho,rad,fliq in ((TF-15,0.,300.,54.5,0.),(TF-3,100.,100.,100.,0.),
        (TF,50.,200.,54.5,.1),(TF,150.,350.,500.,.033)):
        h=10*LF*fliq if fliq else 10*CI*(temp-TF)
        p=Snowpack([10.],[h],[10/rho],[rad])
        tau,kappa,rate=(float(x[0]) for x in aging_parameters([temp],[grad],[rho]))
        wet=1e18*4.22e-13*fliq**3/(4*np.pi)
        ref=solve_ivp(lambda _,r:rate/3600*(tau/(r-54.5+tau))**(1/kappa)+wet/r**2,
            (0,1800),[rad],rtol=1e-11,atol=1e-11).y[0,-1]
        got=float(grain_growth(p,1800,temperature_gradient=grad).radius_um[0])
        errors.append(dict(temperature_k=temp,gradient_km=grad,density_kgm3=rho,
            radius_um=rad,liquid_fraction=fliq,reference_um=ref,result_um=got,bias_percent=100*(got-ref)/ref))
    report['aging']['cases']=errors
    # Half-hourly accumulation/ablation/refreezing with all advected enthalpy
    # and tracer paths tracked. Radiation sampled daily for this budget test;
    # solar input here is specified heat, not a closed atmospheric energy solve.
    p=Snowpack.empty();dt=1800.;mass_input=mass_output=heat_input=heat_output=0.
    impurity_input=np.zeros(5);impurity_output=np.zeros(5)
    max_water=max_heat=max_tracer=0.;max_layers=0;runoff=drainage=0.;max_swe=0.
    min_temperature=float('inf');max_temperature=-float('inf')
    daily=[]
    for step in range(365*48):
        day=step/48.;hour=(step%48)*.5
        air=TF-2.-12.*np.cos(2*np.pi*(day-15)/365)+3.*np.sin(2*np.pi*hour/24)
        snow=.07 if air<TF and (step//48)%5<2 else 0.
        rain=.035 if air>=TF and (step//48)%7==0 else 0.
        if snow:
            rho,rad=fresh_snow_properties(air,2.)
            p=add_snowfall(p,snow,min(air,TF),float(rho),float(rad))
            mass_input+=snow;heat_input+=snow*CI*(min(air,TF)-TF)
        if rain and p.mass.size:
            p=add_rain(p,rain,air);mass_input+=rain;heat_input+=rain*(LF+4180*(air-TF))
        if p.mass.size:
            ice_before=p.ice
            top=-10.+60.*max(0.,np.sin(np.pi*(hour-6)/12))
            p,d=heat_step(p,dt,top,bottom_temperature=TF-1.,bottom_conductance=1.,
                top_temperature=air,top_conductance=8.)
            heat_input+=d['top_flux_wm2']*dt;heat_output+=d['bottom_flux_wm2']*dt
            p=compact(p,dt,ice_before=ice_before,remap=False)
            p=grain_growth(p,dt,temperature_gradient=10.)
            p,d=drain_liquid(p,return_ledger=True)
            for path in ('drainage','runoff'):
                mass_output+=d[path+'_kgm2'];heat_output+=d[path+'_enthalpy_jm2']
                impurity_output+=d[path+'_impurities_kgm2']
            runoff+=d['runoff_kgm2'];drainage+=d['drainage_kgm2']
            if p.mass.size:
                loss=min(.001,p.ice.sum())
                p,advected,residue=remove_sublimation(p,loss)
                mass_output+=loss;heat_output+=advected;impurity_output+=residue
        rates=np.array([1.,2.,1.,.5,.25])*1e-12
        p,residue=deposit_impurities(p,rates,dt)
        impurity_input+=rates*dt;impurity_output+=residue
        max_water=max(max_water,abs(p.mass.sum()+mass_output-mass_input))
        max_heat=max(max_heat,abs(p.enthalpy.sum()+heat_output-heat_input))
        max_tracer=max(max_tracer,np.max(np.abs(p.impurities.sum(0)+impurity_output-impurity_input)))
        max_layers=max(max_layers,len(p.mass));max_swe=max(max_swe,p.swe)
        if p.mass.size:
            min_temperature=min(min_temperature,float(p.temperature.min()))
            max_temperature=max(max_temperature,float(p.temperature.max()))
        if step%48==47:
            optical=snow_shortwave(p,[300.,300.],[100.,100.])
            daily.append([step//48+1,p.swe,len(p.mass),float(p.enthalpy.sum()),optical['albedo']])
    report['synthetic_year']=dict(steps=365*48,max_layers=max_layers,max_swe_m=max_swe,
        min_snow_temperature_k=min_temperature,max_snow_temperature_k=max_temperature,
        mass_input_kgm2=mass_input,mass_output_kgm2=mass_output,drainage_kgm2=drainage,runoff_kgm2=runoff,
        heat_input_jm2=heat_input,heat_output_jm2=heat_output,
        max_cumulative_water_error_kgm2=max_water,max_cumulative_enthalpy_error_jm2=max_heat,
        max_cumulative_impurity_error_kgm2=max_tracer,daily=daily)
    report['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),ROOT/'tests/test_snow_heat.py',ROOT/'tests/test_snicar.py',
            ROOT/'src/noahmp_clm/physics/canopy_radiation.py',ROOT/'src/noahmp_clm/data/snow_tables.npz']+
            sorted((ROOT/'src/noahmp_clm/physics').glob('snow*.py'))+[ROOT/'src/noahmp_clm/physics/snicar.py']}
    report['wall_seconds']=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','synthetic_year')},indent=2))
    print(json.dumps({k:v for k,v in report['synthetic_year'].items() if k!='daily'},indent=2))
    assert report['heat']['max_temperature_error_k']<.01
    assert report['radiation']['max_relative_error_above_1wm2_percent']<10.
    assert max_water<1e-8 and max_heat<.005 and max_tracer<1e-15


if __name__=='__main__':main()

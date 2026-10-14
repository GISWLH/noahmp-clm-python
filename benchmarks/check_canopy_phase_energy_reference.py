"""Independent phase-canopy equations and a conservative continuous canopy run."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
from test_canopy_phase_energy import cases,independent_reference
from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy
from noahmp_clm.physics.canopy_interception import CanopyCondensate,intercept_precipitation
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.soil_uptake import SoilWaterSupply
from test_gpu_snicar import assert_science_close


def random_cases(count):
    rng=np.random.default_rng(66913)
    for k in range(count):
        area=rng.uniform(.05,.7,10);air=rng.uniform(258.,303.);es=611.2*np.exp(17.67*(air-273.15)/(air-29.65));q=.622*es/(101325.-.378*es)*rng.uniform(.3,1.1)
        e=CanopyExchange(air,q,101325.,rng.uniform(.01,.04),.003,.03,air+rng.uniform(-4.,4.),15.)
        mass=rng.uniform(.001,.4,10)
        old=CanopyCondensate.from_temperature(mass,273.15,liquid_fraction=rng.uniform(0.,1.,10)) if k%4==0 else CanopyCondensate.from_temperature(mass,air+rng.uniform(-5.,5.,10))
        supply=SoilWaterSupply(np.full(4,1e-5),np.full(4,.25),1.5e-5)
        kw=dict(soil_supply=supply,root_enthalpy=np.full(4,334000.+4180.*(air-273.15)),vegetated_fraction=rng.uniform(.2,1.))
        yield f'random_{k}',area,rng.uniform(0.,800.),rng.uniform(.75,1.05)*5.670374419e-8*air**4,e,old,kw


def equation_audit(count):
    maximum={};failures=[];records=[]
    for name,a,sw,lw,e,c,kw in [*cases(),*random_cases(count)]:
        try:
            got=solve_canopy_phase_energy(a,sw,lw,e,c,1800.,**kw)
            ref=independent_reference(a,sw,lw,e,c,1800.,**kw)
            errors=dict(temperature=float(np.max(abs(np.r_[got['leaf_temperature'],got['ground_temperature']]-ref['temperature']))),
                water=float(np.max(abs(got['condensate'].mass-ref['mass']))),enthalpy=float(np.max(abs(got['condensate'].enthalpy-ref['enthalpy']))))
            assert errors['temperature']<=.05,(name,errors)
            for key in ('h','le','gpp'):
                errors[key]=abs(got[key]-ref[key]);floor,absolute=(.1,.01) if key=='gpp' else (10.,1.)
                assert_science_close(got[key],ref[key],relative=.1,zero_floor=floor,absolute=absolute)
            for key,value in errors.items():maximum[key]=max(maximum.get(key,0.),value)
            records.append(dict(name=name,iterations=got['iterations'],energy_residual=got['residual'],air_water_residual=got['air_water_residual'],errors=errors,
                actual={k:float(got[k]) for k in ('h','le','gpp')},reference={k:float(ref[k]) for k in ('h','le','gpp')}))
        except Exception as error:
            failures.append(dict(name=name,error=repr(error)));print(json.dumps(failures[-1]),flush=True)
        if (len(records)+len(failures))%10==0:print(json.dumps(dict(completed=len(records)+len(failures),failures=len(failures),maximum=maximum)),flush=True)
    metrics={}
    for key in ('h','le','gpp'):
        actual=np.array([r['actual'][key] for r in records]);reference=np.array([r['reference'][key] for r in records]);error=actual-reference
        mask=abs(reference)>(.1 if key=='gpp' else 10.)
        metrics[key]=dict(n=len(error),mean_bias=float(error.mean()),rmse=float(np.sqrt(np.mean(error**2))),
            correlation=float(np.corrcoef(actual,reference)[0,1]) if np.std(reference)>0. else None,
            nonzero_n=int(mask.sum()),max_relative_percent=float(np.max(100.*abs(error[mask]/reference[mask]),initial=0.)))
    return dict(cases=len(records)+len(failures),passed=len(records),failures=failures,max_absolute=maximum,metrics=metrics,records=records)


def trajectory(steps):
    area=np.full(10,.3);fv=.8;dt=1800.;states=[CanopyCondensate.from_temperature(np.full(10,.1),t) for t in (285.,263.,273.15)]
    initial=np.array([[c.mass.sum(),c.enthalpy.sum()] for c in states]);rows=[[] for _ in states];maximum=np.zeros((3,2));cumulative=np.zeros((3,2));max_iterations=0
    max_phase=np.zeros((3,2));max_res=0.
    for step in range(steps):
        angle=2.*np.pi*(step%48+.5)/48.
        for column,base in enumerate((285.,263.,273.15)):
            old=states[column];air=base+4.*np.sin(angle);es=611.2*np.exp(17.67*(air-273.15)/(air-29.65));q=.622*es/(101325.-.378*es)*(.7+.2*np.cos(angle))
            precip=.08 if step%24<3 else 0.;rain=precip if air>273.15 else 0.;snow=precip-rain
            wet,p=intercept_precipitation(old,rain,snow,air,dt/2.,area,vegetated_fraction=fv,wind=3.)
            e=CanopyExchange(air,q,101325.,.02,.003,.03,base,15.);sw=500.*max(0.,np.sin(angle));lw=.9*5.670374419e-8*air**4
            soil=SoilWaterSupply(np.full(4,1e-5),np.full(4,.25),1.5e-5);root_h=np.full(4,334000.+4180.*(base-273.15))
            result=solve_canopy_phase_energy(area,sw,lw,e,wet,dt,vegetated_fraction=fv,soil_supply=soil,root_enthalpy=root_h,
                fixed_ground_temperature=base,ground_is_ice=base<=273.15,ground_shares_soil=base>273.15,ground_evaporation_limit=1e-5)
            final,post=intercept_precipitation(result['condensate'],0.,0.,air,dt/2.,area,vegetated_fraction=fv,wind=3.)
            states[column]=final;max_iterations=max(max_iterations,result['iterations']);max_res=max(max_res,float(np.max(abs(result['layer_residual']))))
            outlet_m=sum(d['ground_liquid_kgm2']+d['ground_ice_kgm2'] for d in (p,post))
            outlet_h=sum(d['ground_liquid_enthalpy_jm2']+d['ground_ice_enthalpy_jm2'] for d in (p,post))
            root_m=dt*result['root_uptake'].sum();root_heat=dt*float(result['root_uptake']@root_h)
            vapor_m=dt*(result['wet_evaporation'].sum()+result['transpiration'].sum())
            vapor_heat=result['vapor_parcels']['advected_enthalpy_jm2'].sum()+dt*float(result['transpiration']@(334000.+4180.*(result['leaf_temperature']-273.15)))
            boundary=dt*(sw+lw-result['outgoing_shortwave']-result['outgoing_longwave']-result['h']-result['le']-result['ground_heat'])
            net=np.array([precip+root_m-outlet_m-vapor_m,p['precipitation_enthalpy_jm2']+root_heat+boundary-outlet_h-vapor_heat])
            error=np.array([final.mass.sum()-old.mass.sum(),final.enthalpy.sum()-old.enthalpy.sum()])-net
            assert abs(error[0])<1e-12 and abs(error[1])<10.*5e-5*dt+1e-6,(step,column,error)
            maximum[column]=np.maximum(maximum[column],abs(error));rows[column].append(net)
            totals=np.array([math.fsum(row[k] for row in rows[column]) for k in range(2)])
            cumulative[column]=np.maximum(cumulative[column],abs(np.array([final.mass.sum(),final.enthalpy.sum()])-initial[column]-totals))
            max_phase[column]=np.maximum(max_phase[column],[final.ice.sum(),final.liquid.sum()])
        if step%48==47:print(json.dumps(dict(day=step//48+1,max_step=maximum.tolist(),max_cumulative=cumulative.tolist())),flush=True)
    return dict(steps=steps,columns=3,max_step=maximum.tolist(),max_cumulative=cumulative.tolist(),max_phase=max_phase.tolist(),max_iterations=max_iterations,max_layer_residual=max_res)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--random-count',type=int,default=60);parser.add_argument('--steps',type=int,default=336);args=parser.parse_args();start=time.monotonic()
    report=dict(equations=equation_audit(args.random_count))
    report['sources']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('src/noahmp_clm/physics/canopy_phase_energy.py','tests/test_canopy_phase_energy.py','benchmarks/check_canopy_phase_energy_reference.py')}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['equations']['failures']:raise RuntimeError(f'{len(report["equations"]["failures"])} independent equation cases failed; full report retained')
    if args.steps:report['trajectory']=trajectory(args.steps)
    report['seconds']=time.monotonic()-start;args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()

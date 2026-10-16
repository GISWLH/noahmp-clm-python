"""Synthetic meteorological year for the finite NumPy glacier land unit.

This is a continuity/conservation audit, not an observational validation.
The report retains source hashes, prescribed forcing conventions and budgets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from noahmp_clm.column import Forcing
from noahmp_clm.clm_glacier import GlacierColumnState,step_clm_glacier_column
from noahmp_clm.physics.glacier import GlacierIce
from noahmp_clm.physics.canopy_energy import _saturation,_ice_saturation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days',type=int,default=365);parser.add_argument('--steps-per-day',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.days<1 or args.steps_per_day<1:parser.error('Positive duration and steps required')
    root=Path(__file__).resolve().parents[1]
    manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    state=GlacierColumnState(GlacierIce.from_thickness([.1,.3,.6,99.],263.15))
    initial=np.array([state.mass_kgm2,state.enthalpy_jm2]);incoming=np.zeros(2);outgoing=np.zeros(2);bounds=np.zeros(2)
    stepmax=np.zeros(2);cumulative=np.zeros(2);dep=np.array([1.,.1,.2,.3,.4])*1e-13;initial_tracer=np.zeros(5);tracer_out=np.zeros(5)
    report=dict(status='running',days_requested=args.days,steps_per_day=args.steps_per_day,source_manifest=manifest,
        reference='Synthetic seasonal and diurnal forcing; not CLM/FLUXNET acceptance',daily=[])
    start=time.perf_counter();completed=0
    try:
        for day in range(args.days):
            daily=dict(et_mm=0.,runoff_mm=0.,prcp_mm=0.,h=0.,le=0.,rnet=0.)
            seasonal=np.sin(2.*np.pi*(day-80.)/365.)
            for hour in range(args.steps_per_day):
                phase=(hour+.5)/args.steps_per_day;solar=max(0.,np.sin(2.*np.pi*(phase-.25)))
                air=260.+18.*seasonal+3.*np.sin(2.*np.pi*(phase-.35));pressure=75000.
                q=float((_ice_saturation if air<273.15 else _saturation)(air,pressure)[0])*.75
                f=Forcing(86400./args.steps_per_day,air,q,3.,pressure,(350.+150.*seasonal)*solar,
                    .8*5.670374419e-8*air**4,1e-3/86400. if day%4==0 else 0.)
                old=state;state,d=step_clm_glacier_column(state,f,cosine_zenith=max(.05,solar),deposition_rates=dep)
                inp=np.array([f.prcp*f.dt*1000.,d['precipitation_enthalpy_jm2']+(d['rnet']-d['h']-d['le'])*f.dt+d['geothermal_input_jm2']])
                out=np.array([d['et_mm']+d['runoff_mm'],d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']])
                new=np.array([state.mass_kgm2,state.enthalpy_jm2]);oldv=np.array([old.mass_kgm2,old.enthalpy_jm2])
                residual=new-oldv-inp+out;allowance=np.array([d['water_bound_kgm2'],d['energy_bound_jm2']])
                assert np.all(abs(residual)<=allowance),(day,hour,residual,allowance)
                incoming+=inp;outgoing+=out;bounds+=allowance;stepmax=np.maximum(stepmax,abs(residual))
                residual=new-initial-incoming+outgoing;cumulative=np.maximum(cumulative,abs(residual));assert np.all(abs(residual)<=bounds+1e-6)
                initial_tracer+=dep*f.dt;tracer_out+=d['impurity_out_kgm2']
                tracer=state.snow.impurities.sum(0)+state.ice.impurities.sum(0)+tracer_out-initial_tracer
                assert np.all(abs(tracer)<=1e-14)
                for key in daily:daily[key]+=d[key] if key.endswith('_mm') else d[key]/args.steps_per_day
                completed+=1
            daily.update(day=day+1,swe_mm=float(state.snow.mass.sum()),ice_mass_kgm2=float(state.ice.mass.sum()),
                surface_temperature=state.surface_temperature,snow_layers=len(state.snow.mass),ice_layers=len(state.ice.mass))
            report['daily'].append(daily)
            if day%30==29:print(json.dumps(dict(day=day+1,max_step=stepmax.tolist(),max_cumulative=cumulative.tolist())),flush=True)
        report['status']='complete'
    except Exception as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        report.update(steps_completed=completed,seconds=time.perf_counter()-start,max_step_residual=stepmax.tolist(),
            max_cumulative_residual=cumulative.tolist(),cumulative_numerical_bounds=bounds.tolist(),
            external_input=incoming.tolist(),external_output=outgoing.tolist())
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(output=str(args.output),status=report['status'],steps=completed,seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()

"""Continuous synthetic canopy/snow/soil audit; not a site/reference validation."""
from dataclasses import asdict
from pathlib import Path
import argparse,hashlib,json,time,traceback
import numpy as np
from noahmp_clm.column import ColumnState,Forcing,default_soil_params
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.physics.snowpack import Snowpack,TF

ROOT=Path(__file__).resolve().parents[1]


def encode(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path);ap.add_argument('--days',type=int,default=7)
    ap.add_argument('--layered-canopy',action='store_true',help='Carry ten conservative canopy ice/liquid stores through the whole column')
    ap.add_argument('--most',action='store_true',help='Iterate actual H/E with explicit synthetic tower heights')
    args=ap.parse_args();start=time.perf_counter();dt=1800.
    from noahmp_clm.physics.canopy_most import MOSTGeometry
    geometry=MOSTGeometry(wind_height=15.,temperature_height=12.,humidity_height=10.,displacement=2.) if args.most else None
    report={'scope':__doc__,'days':args.days,'layered_canopy':args.layered_canopy,'most_geometry':None if geometry is None else vars(geometry),'columns':[]}
    sources=[p for p in (ROOT/'src/noahmp_clm').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    sources.append(Path(__file__))
    report['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for case in range(3):
        old=ColumnState(soil=default_soil_params(('sand','loam','clay')[case]),
            tsoil=np.full(4,TF-2. if case<2 else TF+8.),smc=np.full(4,.28 if case<2 else .43),
            ts=TF-3. if case<2 else TF+8.,fveg=(.8,0.,.65)[case],lai=(3.,0.,2.)[case],canw=.0001 if case!=1 else 0.)
        snow=Snowpack.from_ice([3.,5.,8.,15.,30.],TF-3.,250.) if case==1 else Snowpack.empty()
        state=CLMColumnState.from_column(old,snow=snow)
        if args.layered_canopy:state=state.with_layered_canopy()
        water0=state.water_kgm2;heat0=state.enthalpy_jm2
        inflow=et=runoff=net_heat=advected=outheat=0.
        traces=np.zeros(5);traceout=np.zeros(5);rates=np.array([1.,2.,1.,.5,.25])*1e-12
        maximum=np.zeros(5);minimum_t=1e9;maximum_t=-1e9;max_snow=0.;max_layers=0;max_evaluations=0
        daily=[];snow_steps=0;totals=np.zeros(5);max_canopy_ice=0.;max_canopy_liquid=0.
        for step in range(args.days*48):
            day=step/48.;hour=(step%48)*.5
            # Seven-day cold-to-warm transition repeated for longer audits.
            seasonal=-7.+18.*np.sin(np.pi*(day%7)/7.)
            air=TF+seasonal+4.*np.sin(2.*np.pi*(hour-8.)/24.)+(4. if case==2 else 0.)
            sun=max(0.,np.sin(np.pi*(hour-6.)/12.))
            es=611.2*np.exp(17.67*(air-TF)/(air-TF+243.5))
            rh=.96 if step%96<12 else .65
            q=rh*.622*es/(101325.-.378*es)
            precip=2e-6 if step%96<6 else 0.
            f=Forcing(dt,air,q,2.5,101325.,650.*sun,270.+35.*sun,precip)
            try:
                after,d=step_clm_enthalpy_column(state,f,cosine_zenith=max(.05,sun),deposition_rates=rates,most_geometry=geometry)
            except Exception:
                failure=dict(case=case,step=step,state=asdict(state),forcing=asdict(f),traceback=traceback.format_exc())
                args.output.with_suffix('.failure.json').write_text(json.dumps(failure,indent=2,default=encode))
                raise
            inflow+=d['prcp_mm'];et+=d['et_mm'];runoff+=d['runoff_mm']
            net_heat+=d['precipitation_enthalpy_jm2']+d['atmospheric_energy_jm2']
            advected+=d['advected_enthalpy_jm2'];outheat+=d['runoff_enthalpy_jm2']
            traces+=dt*rates;traceout+=d['impurity_output_kgm2']
            mass_error=after.water_kgm2-water0-inflow+et+runoff
            heat_error=after.enthalpy_jm2-heat0-net_heat+advected+outheat
            trace_error=float(np.max(np.abs(after.snow.impurities.sum(0)+traceout-traces)))
            maximum=np.maximum(maximum,[abs(mass_error),abs(heat_error),abs(d['water_residual_kgm2']),abs(d['energy_residual_jm2']),trace_error])
            tt=np.r_[after.tsoil,after.snow.temperature]
            minimum_t=min(minimum_t,float(tt.min()));maximum_t=max(maximum_t,float(tt.max()))
            max_snow=max(max_snow,after.swe);max_layers=max(max_layers,len(after.snow.mass))
            max_evaluations=max(max_evaluations,d['surface_evaluations']);snow_steps+=bool(len(after.snow.mass))
            if args.layered_canopy:
                max_canopy_ice=max(max_canopy_ice,float(after.canopy_condensate.ice.sum()))
                max_canopy_liquid=max(max_canopy_liquid,float(after.canopy_condensate.liquid.sum()))
            totals+=np.array([d['et_mm'],d['h']*dt,d['le']*dt,d['gpp_gcm2'],d['runoff_mm']])
            state=after
            if step%48==47:
                daily.append(dict(day=step//48+1,et_mm=totals[0],h_jm2=totals[1],le_jm2=totals[2],gpp_gcm2=totals[3],runoff_mm=totals[4],
                    swe_m=state.swe,canopy_kgm2=state.canopy_water_kgm2,soil_temperature_k=state.tsoil,soil_water_kgm2=state.ground.water,
                    cumulative_water_error_kgm2=mass_error,cumulative_enthalpy_error_jm2=heat_error))
                if args.layered_canopy:
                    daily[-1].update(canopy_ice_kgm2=state.canopy_condensate.ice.sum(),canopy_liquid_kgm2=state.canopy_condensate.liquid.sum())
                totals[:]=0.
                print(f'column {case}, day {step//48+1}: max water {maximum[0]:.9g} kg/m2, enthalpy {maximum[1]:.9g} J/m2',flush=True)
        result=dict(column=case,steps=args.days*48,snow_steps=snow_steps,max_layers=max_layers,max_swe_m=max_snow,
            min_temperature_k=minimum_t,max_temperature_k=maximum_t,max_surface_evaluations=max_evaluations,
            max_cumulative_water_error_kgm2=maximum[0],max_cumulative_enthalpy_error_jm2=maximum[1],
            max_step_water_error_kgm2=maximum[2],max_step_enthalpy_error_jm2=maximum[3],max_impurity_error_kgm2=maximum[4],
            max_canopy_ice_kgm2=max_canopy_ice,max_canopy_liquid_kgm2=max_canopy_liquid,
            precipitation_kgm2=inflow,et_kgm2=et,runoff_kgm2=runoff,daily=daily)
        report['columns'].append(result);report['wall_seconds']=time.perf_counter()-start
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False,default=encode))
        assert maximum[0]<1e-7 and maximum[1]<.1 and maximum[3]<.02 and maximum[4]<1e-12
    print(json.dumps([{k:v for k,v in r.items() if k!='daily'} for r in report['columns']],indent=2,default=encode))


if __name__=='__main__': main()

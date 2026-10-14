"""Four-backend continuous canopy water/enthalpy ledger; fixed ground boundary."""
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
from check_canopy_phase_gpu import packed,same_precision,ledgers
from test_gpu_canopy_phase_energy import host,reference
from test_gpu_snicar import assert_science_close
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_interception import CanopyCondensate,intercept_precipitation
from noahmp_clm.physics.soil_uptake import SoilWaterSupply
from noahmp_clm.gpu.canopy_phase_energy import pack_canopy_phase,run_canopy_phase_jax
from noahmp_clm.gpu.canopy_interception import pack_canopy_interception,run_canopy_water_jax


def main():
    import cupy as cp
    from noahmp_clm.gpu.cuda.canopy_phase_energy import run_canopy_phase_cuda
    from noahmp_clm.gpu.cuda.canopy_interception import run_canopy_water_cuda
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=336);args=parser.parse_args()
    if args.output.exists():parser.error('Use fresh trajectory result path')
    area=np.full(10,.3);bases=(285.,263.,273.15);fv=.8;dt=1800.;started=time.monotonic()
    names=('numpy64','jax64','jax32','cuda32');states={};initial={};fluxes={};maximum={};cumulative={};cumulative_bounds={};daily={};budgets={};history={}
    for name in names:
        dtype=np.float32 if name.endswith('32') else np.float64
        states[name]=[np.column_stack((np.full(10,.1),CanopyCondensate.from_temperature(np.full(10,.1),t).enthalpy)).astype(dtype) for t in bases]
        initial[name]=np.array([s.astype(float).sum(0) for s in states[name]]);fluxes[name]=[[] for _ in bases]
        maximum[name]=np.zeros(2);cumulative[name]=np.zeros(2);cumulative_bounds[name]=np.zeros((3,2));daily[name]=[];history[name]=[];budgets[name]={}
    report=dict(kind='canopy subsystem; prescribed ground temperature and soil supply, not a full column or observations',steps=0,columns=3,backends=4,same_precision_max=0.,failures=[])
    def save():args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    def convert(v):return cp.asnumpy(v) if isinstance(v,cp.ndarray) else np.asarray(v)
    def water_step(name,cells,air,rain,snow):
        dtype=cells[0].dtype;rows=[pack_canopy_interception(CanopyCondensate(c[:,0],c[:,1]),r,s,t,dt/2.,area,vegetated_fraction=fv,wind=3.,dtype=dtype) for c,t,r,s in zip(cells,air,rain,snow)]
        arrays=tuple(np.stack([r[k] for r in rows]) for k in range(5))
        if name=='numpy64':
            results=[intercept_precipitation(CanopyCondensate(c[:,0],c[:,1]),r,s,t,dt/2.,area,vegetated_fraction=fv,wind=3.) for c,t,r,s in zip(cells,air,rain,snow)]
            return [np.column_stack((state.mass,state.enthalpy)) for state,_ in results],{k:np.array([r[k] for _,r in results]) for k in ('precipitation_enthalpy_jm2','ground_liquid_kgm2','ground_ice_kgm2','ground_liquid_enthalpy_jm2','ground_ice_enthalpy_jm2')},None
        got=host((run_canopy_water_cuda if name=='cuda32' else run_canopy_water_jax)(*arrays),convert)
        return list(got['cells']),got['ledger'],got
    for step in range(args.steps):
        angle=2.*np.pi*(step%48+.5)/48.;air=np.array(bases)+4.*np.sin(angle);es=611.2*np.exp(17.67*(air-273.15)/(air-29.65));q=.622*es/(101325.-.378*es)*(.7+.2*np.cos(angle))
        precip=.08 if step%24<3 else 0.;rain=np.where(air>273.15,precip,0.);snow=precip-rain;sw=500.*max(0.,np.sin(angle));lw=.9*5.670374419e-8*air**4
        samples={}
        for name in names:
            old=states[name];dtype=old[0].dtype;wet,pre,wpre=water_step(name,old,air,rain,snow);rows=[]
            for k in range(3):
                e=CanopyExchange(air[k],q[k],101325.,.02,.003,.03,bases[k],15.)
                supply=SoilWaterSupply(np.full(4,1e-5),np.full(4,.25),1.5e-5)
                rows.append(pack_canopy_phase(area,sw,lw[k],e,CanopyCondensate(wet[k][:,0],wet[k][:,1]),dt,
                    vegetated_fraction=fv,soil_supply=supply,root_enthalpy=np.full(4,334000.+4180.*(bases[k]-273.15)),
                    fixed_ground_temperature=bases[k],ground_is_ice=bases[k]<=273.15,ground_shares_soil=bases[k]>273.15,ground_evaporation_limit=1e-5,dtype=dtype))
            arrays=packed(rows,dtype)
            if name=='numpy64':
                refs=[reference(row) for row in zip(*arrays)]
                keys=('leaf_temperature','h','le','gpp','ground_heat','root_uptake','transpiration','wet_evaporation','outgoing_shortwave','outgoing_longwave')
                got={key:np.array([r[key] for r in refs]) for key in keys};got.update(mass=np.array([r['condensate'].mass for r in refs]),enthalpy=np.array([r['condensate'].enthalpy for r in refs]),
                    vapor_enthalpy=np.array([r['vapor_parcels']['advected_enthalpy_jm2'] for r in refs]))
            else:
                got=host((run_canopy_phase_cuda if name=='cuda32' else run_canopy_phase_jax)(*arrays,raise_on_failure=False),convert)
                if not np.all(got['converged']):
                    report['failures'].append(dict(step=step,backend=name,columns=np.where(~got['converged'])[0].tolist(),residual=got['max_solver_residual'].tolist(),arrays=[v.tolist() for v in arrays]));save();raise RuntimeError('Phase trajectory convergence failed; inputs saved')
                checked=ledgers(got,arrays)
                for key,value in checked.items():budgets[name][key]=max(budgets[name].get(key,0.),value)
            final,post,wpost=water_step(name,[np.column_stack((m,h)) for m,h in zip(got['mass'],got['enthalpy'])],air,np.zeros(3),np.zeros(3));states[name]=final
            samples[name]=(got,wpre,wpost);b=arrays[6].astype(float);u=arrays[1].astype(float)
            out_m=sum(np.asarray(d['ground_liquid_kgm2'],float)+np.asarray(d['ground_ice_kgm2'],float) for d in (pre,post))
            out_h=sum(np.asarray(d['ground_liquid_enthalpy_jm2'],float)+np.asarray(d['ground_ice_enthalpy_jm2'],float) for d in (pre,post))
            root_m=dt*np.asarray(got['root_uptake'],float).sum(1);root_h=dt*(np.asarray(got['root_uptake'],float)*b[:,5:9]).sum(1)
            vapor_m=dt*(np.asarray(got['wet_evaporation'],float).sum(1)+np.asarray(got['transpiration'],float).sum(1))
            vapor_h=np.asarray(got['vapor_enthalpy'],float).sum(1)+dt*(np.asarray(got['transpiration'],float)*(334000.+4180.*(np.asarray(got['leaf_temperature'],float)-273.15))).sum(1)
            boundary=dt*(u[:,11]+u[:,12]-np.asarray(got['outgoing_shortwave'],float)-np.asarray(got['outgoing_longwave'],float)-np.asarray(got['h'],float)-np.asarray(got['le'],float)-np.asarray(got['ground_heat'],float))
            # Rain/snow inputs to the device were quantized to its precision.
            incoming=rain.astype(dtype).astype(float)+snow.astype(dtype).astype(float)
            net=np.stack((incoming+root_m-out_m-vapor_m,np.asarray(pre['precipitation_enthalpy_jm2'],float)+root_h+boundary-out_h-vapor_h),axis=1)
            now=np.array([v.astype(float).sum(0) for v in final]);before=np.array([v.astype(float).sum(0) for v in old]);error=now-before-net
            eps=np.finfo(dtype).eps;bound=np.stack((256.*eps*np.maximum(abs(now[:,0])+abs(before[:,0])+abs(net[:,0]),1.)+1e-12,
                dt*(10.*(.005 if dtype==np.float32 else 5e-5)+128.*eps*(abs(u[:,11])+abs(u[:,12])+abs(np.asarray(got['h'],float))+abs(np.asarray(got['le'],float))+100.))+256.*eps*np.maximum(abs(now[:,1])+abs(before[:,1])+abs(net[:,1]),1.)+1e-6),axis=1)
            assert np.all(abs(error)<=bound),(step,name,error,bound)
            maximum[name]=np.maximum(maximum[name],abs(error).max(0));cumulative_bounds[name]+=bound
            for k in range(3):
                fluxes[name][k].append(net[k]);total=np.array([math.fsum(row[i] for row in fluxes[name][k]) for i in range(2)])
                cumulative[name]=np.maximum(cumulative[name],abs(now[k]-initial[name][k]-total))
            history[name].append(np.stack((vapor_m,np.asarray(got['h'],float),np.asarray(got['le'],float),np.asarray(got['gpp'],float),np.asarray(got['leaf_temperature'],float).mean(1)),axis=1))
            if step%48==47:
                aggregate=np.array(history[name][-48:]).sum(0)
                aggregate[:,1:3]*=dt/1e6;aggregate[:,3]*=dt*12.011e-6;aggregate[:,4]/=48.
                daily[name].append(aggregate)
        for kind in range(3):
            same=same_precision(samples['cuda32'][kind],samples['jax32'][kind]);report['same_precision_max']=max(report['same_precision_max'],max(same['max_absolute'].values()))
            if same['failures']:report['failures'].append(dict(step=step,kind=kind,same_precision=same));save();raise RuntimeError('Trajectory same precision failed')
        report.update(steps=step+1,budgets={name:dict(max_step=maximum[name].tolist(),max_cumulative=cumulative[name].tolist(),cumulative_bound=cumulative_bounds[name].max(0).tolist(),component=budgets[name]) for name in names})
        if step%48==47:save();print(json.dumps(dict(day=step//48+1,budgets=report['budgets'])),flush=True)
    report['daily']={}
    for name in names[1:]:
        metrics={};a=np.array(daily[name]);r=np.array(daily['numpy64'])
        for i,key in enumerate(('canopy_et_kgm2','h_mjm2','le_mjm2','gpp_gcm2','leaf_temperature_k')):
            floor=.01 if i==0 else .1 if i==3 else .864;absolute=1e-4 if i==0 else .01 if i==3 else .0864
            mask=abs(r[:,:,i])>floor;err=abs(a[:,:,i]-r[:,:,i]);metrics[key]=dict(max_absolute=float(err.max()),nonzero_n=int(mask.sum()),max_relative_percent=None if i==4 or not np.any(mask) else float(np.max(100.*err[mask]/abs(r[:,:,i][mask]))))
            if i==4:assert np.all(err<.05)
            else:assert_science_close(a[:,:,i],r[:,:,i],relative=.1,zero_floor=floor,absolute=absolute)
        report['daily'][name]=metrics
    report['seconds']=time.monotonic()-started;save();print(json.dumps(dict(seconds=report['seconds'],same_precision=report['same_precision_max'])),flush=True)


if __name__=='__main__':main()

"""Fixed-budget A-Ci-only global-fit diagnostic; light scoring is a separate command.

No production equations or original observational gates change. This remains
exploratory because the original light-response scores have already been seen.
"""
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import argparse,csv,gzip,hashlib,io,json,time,zipfile
import numpy as np
from scipy.optimize import differential_evolution,least_squares
from noahmp_clm.physics.photosynthesis import LeafParameters,farquhar_rates
import noahmp_clm.physics.photosynthesis as model

ARCHIVE_SHA='936cff54c647e84bb4fc59e380c6d234888f8e47722635b00d24e55068c83cd5'
BOUNDS=((1.,300.),(1.,600.),(.1,50.))
SEED=20261017
DE_OPTIONS=dict(maxiter=1000,popsize=16,tol=1e-9,atol=1e-10,polish=False,workers=1,updating='immediate')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def records(path):
    if sha(path)!=ARCHIVE_SHA:raise ValueError('Original Leafweb source mismatch')
    with zipfile.ZipFile(path) as z:
        return {n:list(csv.DictReader(io.StringIO(z.read(n).decode('cp1252'))))[1:] for n in sorted(z.namelist())}

def points(rows):
    values=np.asarray([[float(r[k]) for k in ['CO2i','PARi','Tleaf','AnetCO2']] for r in rows],float)
    if not np.all(np.isfinite(values)):raise ValueError('Nonfinite experiment')
    return values

def fit_one(task):
    name,dark,calibration,baseline,baseline_rmse,seed=task
    d=points(dark);a=points(calibration);assert len(d)==1 and len(a)>=4
    quantum=1.-float(dark[0]['FoorFs'])/float(dark[0]['FmorFm.'])
    rd=-d[0,3]/float(farquhar_rates(420.,0.,d[0,2]+273.15,LeafParameters(respiration25=1.))['respiration'])
    def parameters(x):
        return LeafParameters(vcmax25=x[0],jmax25=x[1],tpu25=x[2],respiration25=rd,quantum_yield_psii=quantum)
    def rates(x):return farquhar_rates(a[:,0],.84*a[:,1],a[:,2]+273.15,parameters(x))
    def residual(x):return rates(x)['net']-a[:,3]
    def cost(x):
        error=residual(x)
        return float(np.dot(error,error))
    original=np.asarray([baseline[k] for k in ['vcmax25','jmax25','tpu25']])
    old_rmse=float(np.sqrt(cost(original)/len(a)))
    np.testing.assert_allclose(old_rmse,baseline_rmse,rtol=1e-10,atol=1e-10)
    jac=[]
    for k in range(3):
        change=max(abs(original[k]),1.)*1e-5;right=original.copy();left=original.copy()
        right[k]+=change;left[k]-=change;jac.append((residual(right)-residual(left))/(2.*change))
    started=time.monotonic()
    global_fit=differential_evolution(cost,BOUNDS,rng=np.random.default_rng(seed),**DE_OPTIONS)
    polished=least_squares(residual,global_fit.x,bounds=np.asarray(BOUNDS).T,max_nfev=2000,
        ftol=1e-12,xtol=1e-12,gtol=1e-12)
    candidates=[('original27-start',original),('differential-evolution',global_fit.x),('polished-global',polished.x)]
    label,chosen=min(candidates,key=lambda item:cost(item[1]))
    r=rates(chosen);limited=np.argmin(np.stack([r[k] for k in ['rubisco','rubp','tpu']]),axis=0)
    return dict(file=name,calibration_n=len(a),seed=seed,baseline_calibration_rmse=old_rmse,
        baseline_jacobian_column_norm=np.linalg.norm(np.stack(jac,axis=1),axis=0).tolist(),
        chosen_training_only=label,parameters=asdict(parameters(chosen)),calibration_rmse=float(np.sqrt(cost(chosen)/len(a))),
        limiting_counts={k:int(np.sum(limited==i)) for i,k in enumerate(['rubisco','rubp','tpu'])},
        zero_carbon_points=int(np.sum((r['rubisco']==0)&(r['rubp']==0))),
        global_termination=dict(success=bool(global_fit.success),message=str(global_fit.message),nfev=int(global_fit.nfev),nit=int(global_fit.nit),cost=float(global_fit.fun)),
        polished_termination=dict(success=bool(polished.success),nfev=int(polished.nfev),cost=cost(polished.x)),seconds=time.monotonic()-started)

def calibrate(archive,baseline_path,output):
    baseline=json.loads(gzip.decompress(baseline_path.read_bytes()))['multistart']['files']
    by_name={v['file']:v for v in baseline};tasks=[]
    for i,(name,rows) in enumerate(records(archive).items()):
        dark=[r for r in rows if int(r['DataType'])==0]
        training=[r for r in rows if 11<=int(r['DataType'])<=25]
        # Only dark/A-Ci records and old training parameters reach workers.
        prior=by_name[name]
        tasks.append((name,dark,training,prior['parameters'],prior['calibration_rmse'],SEED+i))
    started=time.monotonic();model_sha=sha(model.__file__)
    with ProcessPoolExecutor(max_workers=4) as pool:files=list(pool.map(fit_one,tasks))
    assert sha(model.__file__)==model_sha
    result=dict(scope='Fixed-budget A-Ci-only global optimizer diagnostic, original bounds and biochemical equations. No light measurements passed to optimization workers; no global-optimum proof asserted.',
        archive_sha256=ARCHIVE_SHA,baseline_sha256=sha(baseline_path),instrument_sha256=sha(__file__),model_sha256=model_sha,
        configuration=dict(seed=SEED,bounds=BOUNDS,differential_evolution=DE_OPTIONS,polish_max_nfev=2000,
            selection='Minimum A-Ci sum of squared residuals among old best, DE and polished DE; no light score used'),files=files,seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps([dict(file=v['file'],training_rmse=v['calibration_rmse'],old_training_rmse=v['baseline_calibration_rmse'],
        old_jacobian_columns=v['baseline_jacobian_column_norm'],selected=v['chosen_training_only']) for v in files]),flush=True)

def evaluate(archive,calibration_path,output):
    calibration=json.loads(calibration_path.read_text());assert calibration['instrument_sha256']==sha(__file__)
    assert calibration['model_sha256']==sha(model.__file__) and calibration['archive_sha256']==sha(archive)
    chosen={v['file']:v for v in calibration['files']};files=[]
    for name,rows in records(archive).items():
        data=points([r for r in rows if 31<=int(r['DataType'])<=45]);entry=dict(file=name,status='unmeasured',validation_n=len(data))
        if len(data):
            p=LeafParameters(**chosen[name]['parameters'])
            prediction=farquhar_rates(data[:,0],.84*data[:,1],data[:,2]+273.15,p)['net']
            errors=prediction-data[:,3];rmse=float(np.sqrt(np.mean(errors**2)));nrmse=rmse/float(data[:,3].mean());near=abs(data[:,3])<=1.
            entry.update(status='pass' if nrmse<=.1 else 'fail',relative_mean_bias=float(errors.mean()/data[:,3].mean()),normalized_rmse=nrmse,
                r=float(np.corrcoef(prediction,data[:,3])[0,1]),near_zero_max_abs=float(np.max(abs(errors[near]))) if np.any(near) else None,
                near_zero_failures=int(np.sum(abs(errors[near])>.5)),light_rows=[dict(ci_ppm=float(v[0]),incident_par=float(v[1]),leaf_c=float(v[2]),
                    observed_net=float(v[3]),predicted_net=float(a)) for v,a in zip(data,prediction)])
        files.append(entry)
    result=dict(scope='Separate scoring after A-Ci-only parameter artifact is frozen. Exploratory, not blind; original NRMSE10% and near-zero0.5 gates retained.',
        archive_sha256=sha(archive),calibration_sha256=sha(calibration_path),instrument_sha256=sha(__file__),files=files)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps([{k:v.get(k) for k in ['file','status','normalized_rmse','r']} for v in files]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['calibrate','evaluate']);p.add_argument('archive',type=Path)
    p.add_argument('--baseline',type=Path);p.add_argument('--calibration',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve the previous diagnostic output; use a new path')
    if a.mode=='calibrate':calibrate(a.archive,a.baseline,a.output)
    else:evaluate(a.archive,a.calibration,a.output)

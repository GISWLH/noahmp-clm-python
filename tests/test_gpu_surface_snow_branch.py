"""Nine native saved-state perturbations against EHPC NumPy references."""
from pathlib import Path
import json,hashlib
import numpy as np
import pytest
from noahmp_clm.validation import compare_series

FIXTURES=Path(__file__).with_name('fixtures')

def saved(label):
    path=FIXTURES/'ham_surface_branch_gpu.npz'
    metadata=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['fixture_sha256']
    with np.load(path,allow_pickle=False) as z:
        return tuple(z[f'{label}_a{k}'] for k in range(16)),{k.removeprefix(label+'_'):z[k] for k in z if k.startswith(label+'_') and not k.startswith(label+'_a')}

def solve(backend,label):
    from run_fluxnet_native import Backend,audit_step
    from noahmp_clm.gpu.clm_bgc_column import next_bgc_arrays
    a,v=saved(label);engine=Backend(backend);device=tuple(engine.xp.asarray(x) for x in a)
    out_device=engine.step(*device,v['forcing'],environment=v['environment'],calendar=v['calendar'],most_geometry=v['geometry'],return_surface=True)
    out=engine.host(out_device);next_device=next_bgc_arrays(device,out_device)
    after=tuple(engine.host(x) for x in next_device)
    zeros=np.zeros(len(a[0]),a[0].dtype)
    _,residual,bound,tracer=audit_step(a,after,v['forcing'],np.zeros((len(a[0]),5),a[0].dtype),zeros,out)
    temperature,_=engine.soil(next_device);temperature=engine.host(temperature)
    dt=v['forcing'][:,0].astype(float);rates=86400./dt
    actual={k:np.asarray(out['diagnostics'][k],float) for k in ['h','le','et_mm','runoff_mm','gpp_gc_m2','nee_gc_m2']}
    actual.update(soil_temperature=np.asarray(temperature,float),smc=after[1][:,:,0].astype(float)/(1000.*after[1][:,:,2]),
        swe_m=after[0][:,:,0].astype(float).sum(1)/1000.)
    summary={}
    for key,var,scale,offset in [('h','h_wm2',1.,0.),('le','le_wm2',1.,0.),
        ('et_mm','et_mm_day',rates,0.),('runoff_mm','runoff_mm_day',rates,0.),
        ('gpp_gc_m2','gpp_gc_day',rates,0.),('nee_gc_m2','nee_gc_day',rates,0.),
        ('soil_temperature','tsoil_c',1.,-273.15),('smc','smc',1.,0.),('swe_m','swe_m',1.,0.)]:
        got=actual[key]*scale+offset;ref=v['ref_'+key]*scale+offset
        values=[compare_series([x],[y],var) for x,y in zip(got.ravel(),ref.ravel())]
        assert all(value['bias_pass'] for value in values),(backend,key,values)
        summary[key]=dict(max_absolute_difference=float(abs(got-ref).max()),
            max_nonzero_relative_percent=max([100*abs(x['relative_bias']) for x in values if x['relative_bias'] is not None] or [0.]))
    most=out['surface']['most']
    assert np.all(most['iterations']<=50) and np.all(most['coefficient_residual']<=1e-4)
    print('HaM continuous branch',json.dumps(dict(backend=backend,scope='Component perturbation ensemble, rates are day-equivalent; not temporal/observed scores.',
        metrics=summary,iterations=most['iterations'].tolist(),
        maximum_residual_m_h_c_n=abs(residual).max(0).tolist(),maximum_bounds_m_h_c_n=bound.max(0).tolist(),
        maximum_impurity_residual=float(tracer.max()))),flush=True)
    return out

def test_ham_branch_cuda_jax_same_precision_reference_and_budgets():
    pytest.importorskip('cupy');pytest.importorskip('jax')
    from run_fluxnet_native import compare_same_precision
    cuda=solve('cuda32','f32');jax=solve('jax32','f32')
    print('HaM branch all fields same precision',compare_same_precision(cuda,jax),flush=True)

def test_ham_branch_jax64_reference_and_budgets():
    jax=pytest.importorskip('jax')
    if not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    solve('jax64','f64')

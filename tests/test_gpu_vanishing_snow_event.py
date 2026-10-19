"""Retained microscopic-snow phase event, with offline EHPC oracles."""
from pathlib import Path
import json,os
import numpy as np
import pytest


def fixture(precision):
    with np.load(Path(__file__).parent/'fixtures/vanishing_snow_event_reference.npz',allow_pickle=False) as z:
        prefix=precision+'/'
        return {n[len(prefix):]:z[n] for n in z.files if n.startswith(prefix)}


def host(result):return {k:v.get() if hasattr(v,'get') else np.asarray(v) for k,v in result.items()}


def run(c,backend):
    if backend=='cuda32':from noahmp_clm.gpu.cuda.column_heat import run_joint_heat_cuda as execute
    else:from noahmp_clm.gpu.column_heat import run_joint_heat_jax as execute
    return host(execute(*(c[k] for k in ['cells','props','options','boundary'])))


def budget(c,result):
    cells=c['cells'];h=result['enthalpy_jm2'];bc=c['boundary'];n=5
    change=(h[:,:n].astype(float)-cells[:,:n,1].astype(float)).sum(1)
    incoming=bc[:,0].astype(float)*(result['top_flux_wm2'].astype(float)+cells[:,:n,4].astype(float).sum(1))
    residual=change-incoming
    bound=result['layer_tolerance_jm2'][:,:n].astype(float).sum(1)+16.*np.finfo(cells.dtype).eps*np.maximum(1.,abs(incoming))+1e-5
    assert np.all(abs(residual)<=bound),(residual,bound)
    assert np.all(abs(result['layer_residual_jm2'])<=result['layer_tolerance_jm2'])
    assert np.all(result['converged']) and np.all(result['temperature_offset_k'][:,:n]>0.)
    return dict(maximum_energy_residual_j_m2=float(abs(residual).max()),maximum_original_bound_j_m2=float(bound.max()))


@pytest.mark.parametrize('backend',['jax32','jax64','cuda32'])
def test_extreme_phase_event_against_quantized_ehpc_reference(backend,monkeypatch):
    jax=pytest.importorskip('jax');assert any(d.platform=='gpu' for d in jax.devices())
    if backend=='cuda32':pytest.importorskip('cupy')
    import noahmp_clm.physics.column_heat as physics
    def forbidden(*a,**kw):raise AssertionError('GPU test cannot execute NumPy physics')
    monkeypatch.setattr(physics,'coupled_heat_step',forbidden)
    c=fixture(backend[-2:]);out=run(c,backend);guard=budget(c,out)
    delta=abs(out['temperature_k'][:,:5]-c['temperature_k'])
    assert np.max(delta)<=.01 if backend!='jax64' else np.max(delta)<=1e-7
    np.testing.assert_allclose(out['enthalpy_jm2'][:,:5],c['enthalpy_jm2'],rtol=.1 if backend!='jax64' else 1e-8,atol=.1 if backend!='jax64' else .001)
    print('Vanishing snow native',json.dumps(dict(backend=backend,maximum_temperature_error_k=float(delta.max()),original_budgets=guard,iterations=out['iterations'].tolist())),flush=True)


def test_extreme_phase_event_all_raw_outputs_same_precision():
    pytest.importorskip('jax');pytest.importorskip('cupy');c=fixture('32')
    a,b=run(c,'cuda32'),run(c,'jax32');assert a.keys()==b.keys();maximum={}
    budget(c,a);budget(c,b)
    if os.environ.get('NOAHMP_THIN_EVENT_DIAGNOSTICS'):
        folder=Path(os.environ['NOAHMP_THIN_EVENT_DIAGNOSTICS']);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'allraw32.npz',**{'cuda/'+k:v for k,v in a.items()},**{'jax/'+k:v for k,v in b.items()})
    for key in a:
        assert a[key].dtype==b[key].dtype
        if a[key].dtype.kind in 'biu' or key=='iterations':np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        else:np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-6,err_msg=key)
        maximum[key]=float(abs(a[key].astype(float)-b[key].astype(float)).max(initial=0.))
    print('Vanishing snow allraw',json.dumps(maximum),flush=True)

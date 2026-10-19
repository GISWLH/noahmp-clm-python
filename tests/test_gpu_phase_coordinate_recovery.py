"""Original rejected surface on device, offline EHPC references and raw gates."""
from pathlib import Path
import json,os
import numpy as np
import pytest
from phase_coordinate_recovery_cases import fixture
from test_gpu_canopy_phase_surface import budgets


def host(x):
    if isinstance(x,dict):return {k:host(v) for k,v in x.items()}
    return x.get() if hasattr(x,'get') else np.asarray(x)


def execute(v,backend):
    from noahmp_clm.physics.photosynthesis import LeafParameters
    if backend=='cuda32':from noahmp_clm.gpu.cuda.canopy_phase_surface import run_canopy_phase_surface_cuda as run
    else:from noahmp_clm.gpu.canopy_phase_surface import run_canopy_phase_surface_jax as run
    dtype=np.float64 if backend=='jax64' else np.float32
    a=tuple(v['input/surface_a'+str(k)].astype(np.int32 if k==3 else dtype) for k in range(9))
    if backend=='jax64':
        # The retained packet is float32. These two fields are packing
        # aliases, not independent state: rebuild them in the selected
        # precision from the unchanged authoritative layer stores.
        a[7][:,1:3]=a[8].sum(axis=1)
    kw={k:v['input/'+k].astype(dtype) for k in ['exchange_extension','vic','stem_area'] if 'input/'+k in v}
    out=host(run(*a,parameters=LeafParameters(**json.loads(str(v['leaf_parameters_json']))),**kw))
    return a,out


def flat(d,prefix=''):
    for k,v in d.items():
        if isinstance(v,dict):yield from flat(v,prefix+k+'/')
        else:yield prefix+k,v


@pytest.mark.parametrize('backend',['jax32','jax64','cuda32'])
def test_retained_surface_reference_and_original_extensive_budgets(backend,monkeypatch):
    jax=pytest.importorskip('jax');assert any(d.platform=='gpu' for d in jax.devices())
    if backend=='cuda32':pytest.importorskip('cupy')
    import noahmp_clm.physics.canopy_phase_energy as physics
    import noahmp_clm.physics.canopy_column as column
    def forbidden(*a,**kw):raise AssertionError('GPU result must not use NumPy physics')
    monkeypatch.setattr(physics,'solve_canopy_phase_energy',forbidden);monkeypatch.setattr(column,'surface_step',forbidden)
    v=fixture();a,out=execute(v,backend);guard=budgets(a,out)
    assert np.all(out['converged']) and np.all(out['canopy_converged'])
    assert np.max(out['max_solver_residual'])<=(.005 if backend!='jax64' else 5e-5)
    maximum={}
    for key in ['h','le','gpp']:
        difference=abs(out[key]-v['expected/surface/'+key]);assert np.all(difference<=1.)
        maximum[key]=float(difference.max())
    temperature=float(abs(out['leaf_temperature'][0]-v['expected/surface/leaf_temperature']).max());assert temperature<=.05
    np.testing.assert_allclose(out['condensate'][0,:,0],v['expected/surface/condensate_mass'],rtol=.1,atol=1e-4)
    np.testing.assert_allclose(out['condensate'][0,:,1],v['expected/surface/condensate_enthalpy'],rtol=.1,atol=10.)
    print('3803 native recovery',json.dumps(dict(backend=backend,maximum_leaf_temperature_error_k=temperature,maximum_flux_error=maximum,original_budgets=guard)),flush=True)


def test_retained_surface_all_outputs_same_precision():
    pytest.importorskip('jax');pytest.importorskip('cupy');v=fixture()
    a,left=execute(v,'cuda32');b,right=execute(v,'jax32');budgets(a,left);budgets(b,right)
    l,r=dict(flat(left)),dict(flat(right));assert l.keys()==r.keys();maximum={}
    if os.environ.get('NOAHMP_PHASE_COORDINATE_DIAGNOSTICS'):
        folder=Path(os.environ['NOAHMP_PHASE_COORDINATE_DIAGNOSTICS']);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'allraw32.npz',**{'cuda/'+k:v for k,v in l.items()},**{'jax/'+k:v for k,v in r.items()})
    for key in l:
        assert l[key].dtype==r[key].dtype,key
        if l[key].dtype.kind in 'biu':np.testing.assert_array_equal(l[key],r[key],err_msg=key)
        else:np.testing.assert_allclose(l[key],r[key],rtol=1e-6,atol=1e-6,err_msg=key)
        maximum[key]=float(abs(l[key].astype(float)-r[key].astype(float)).max(initial=0.))
    print('3803 all raw surface',json.dumps(maximum),flush=True)

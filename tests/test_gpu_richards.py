"""Native liquid reference; all NumPy physics fixtures were computed on EHPC."""
from pathlib import Path
import json,os
import numpy as np
import pytest


def case(precision,index):
    with np.load(Path(__file__).parent/'fixtures/richards_reference.npz',allow_pickle=False) as z:
        prefix=precision+'/'+str(index)+'/'
        return {k[len(prefix):]:v for k,v in z.items() if k.startswith(prefix)}


def host(result):return {k:v.get() if hasattr(v,'get') else np.asarray(v) for k,v in result.items()}


@pytest.mark.parametrize('backend',['jax32','jax64','cuda32'])
def test_liquid_native_against_all_ehpc_quantized_references(backend):
    jax=pytest.importorskip('jax')
    assert any(d.platform=='gpu' for d in jax.devices()),'GPU execution is required'
    if backend=='cuda32':
        pytest.importorskip('cupy')
        from noahmp_clm.gpu.cuda.richards import run_richards_cuda as run
    else:
        assert backend!='jax64' or jax.config.x64_enabled
        from noahmp_clm.gpu.richards import run_richards_jax as run
    for index in range(17):
        c=case(backend[-2:],index);got=host(run(*(c[k] for k in ['water','hydraulics','dt'])))
        assert all(v.dtype==c['water'].dtype for v in got.values() if v.dtype.kind=='f')
        theta_error=float(np.max(abs(got['water_kg_m2']-c['expected/water_kg_m2'])/(1000.*c['hydraulics'][:,:,0])))
        expected=c['expected/interface_transfer_kg_m2'];face_error=abs(got['interface_transfer_kg_m2']-expected)
        assert theta_error<=.002
        assert np.all(face_error<=np.where(abs(expected)<=.02,.002,.1*abs(expected)))
        np.testing.assert_array_equal(got['pressure_defined'],c['expected/pressure_defined'])
        assert np.all(abs(got['mass_residual_kg_m2'])<=got['mass_bound_kg_m2'])
        assert np.all(got['water_kg_m2']>=0.)
        print('Richards native reference',json.dumps(dict(backend=backend,index=index,max_soil_water_error_m3m3=theta_error,
            max_face_error_kg_m2=float(face_error.max()),mass_residual_kg_m2=got['mass_residual_kg_m2'].tolist())),flush=True)


def test_liquid_cuda_jax_every_returned_field_same_precision():
    pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.richards import run_richards_jax
    from noahmp_clm.gpu.cuda.richards import run_richards_cuda
    saved={};maximum={}
    for index in range(17):
        c=case('32',index);inputs=[c[k] for k in ['water','hydraulics','dt']]
        a,b=host(run_richards_cuda(*inputs)),host(run_richards_jax(*inputs))
        assert a.keys()==b.keys()
        if os.environ.get('NOAHMP_RICHARDS_DIAGNOSTICS'):
            folder=Path(os.environ['NOAHMP_RICHARDS_DIAGNOSTICS']);folder.mkdir(exist_ok=True,parents=True)
            np.savez_compressed(folder/('case'+str(index)+'.npz'),**{'cuda/'+k:v for k,v in a.items()},**{'jax/'+k:v for k,v in b.items()})
        for key in a:
            assert a[key].dtype==b[key].dtype
            saved[str(index)+'/cuda/'+key]=a[key];saved[str(index)+'/jax/'+key]=b[key]
            if a[key].dtype.kind in 'biu':np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            else:np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-6,err_msg=key)
            maximum[key]=max(maximum.get(key,0.),float(np.max(abs(a[key].astype(float)-b[key].astype(float)))))
    if os.environ.get('NOAHMP_RICHARDS_DIAGNOSTICS'):
        folder=Path(os.environ['NOAHMP_RICHARDS_DIAGNOSTICS']);folder.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(folder/'allraw32.npz',**saved)
    print('Richards all raw same precision',json.dumps(maximum),flush=True)

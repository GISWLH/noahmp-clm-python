"""Native face correction and actual saturated producer cases, offline NumPy only."""
from pathlib import Path
import json
import numpy as np
import pytest
from test_finite_volume import CASES
from test_gpu_richards_enthalpy import audit_closed_storage,host


def backend(name):
    jax=pytest.importorskip('jax');assert any(d.platform=='gpu' for d in jax.devices())
    if name=='cuda32':
        xp=pytest.importorskip('cupy')
        from noahmp_clm.gpu.cuda.richards_enthalpy import run_richards_enthalpy_cuda as run
    else:
        import jax.numpy as xp
        assert name!='jax64' or jax.config.x64_enabled
        from noahmp_clm.gpu.richards_enthalpy import run_richards_enthalpy_jax as run
    return xp,run,xp.float64 if name=='jax64' else xp.float32


def fixture(index,precision):
    with np.load(Path(__file__).parent/'fixtures/richards_saturation_reference.npz',allow_pickle=False) as z:
        p=str(index)+'/'
        values={k[len(p):]:v for k,v in z.items() if k.startswith(p)}
    for k in ['cells','properties','dt','invalid_cells']:values[k]=values[k].astype('float'+precision)
    return values


@pytest.mark.parametrize('name',['jax32','jax64','cuda32'])
def test_native_directed_faces_and_returned_donor_heat(name):
    from noahmp_clm.physics.finite_volume import bounded_interface_transfers,close_integrated_transport
    xp,_,dtype=backend(name)
    for case in CASES:
        old,cap,net=[xp.asarray(v,dtype) for v in case[:3]]
        result=bounded_interface_transfers(old,cap,net,xp=xp,roundoff_only=False)
        got=result.get() if hasattr(result,'get') else np.asarray(result)
        np.testing.assert_array_equal(got,np.asarray(case[3],dtype=got.dtype))
    for direction in [1.,-1.]:
        old=xp.asarray([5,9] if direction>0 else [9,5],dtype);cap=xp.asarray([10,10],dtype)
        net=xp.asarray([2*direction],dtype);f=xp.maximum(net,0.);r=xp.maximum(-net,0.)
        result=close_integrated_transport(old,cap,net,f,r,heat=f*340000.-r*350000.,
            forward_heat=f*340000.,reverse_heat=r*350000.,xp=xp,roundoff_only=False)
        got=host(dict(enumerate(result)))
        expected=[direction,max(direction,0.),max(-direction,0.),340000. if direction>0 else -350000.]
        for k,value in enumerate(expected):np.testing.assert_array_equal(got[k],[value])


@pytest.mark.parametrize('name',['jax32','jax64','cuda32'])
def test_actual_saturated_producers_against_independent_ehpc_numpy(name):
    _,run,_=backend(name)
    for index in range(2):
        c=fixture(index,name[-2:]);got=host(run(*(c[k] for k in ['cells','properties','options','dt'])))
        capacity=np.asarray(1000.,c['cells'].dtype)*c['cells'][:,:,2]*c['properties'][:,0,None]
        assert np.all((got['cells'][:,:,0]>=0.)&(got['cells'][:,:,0]<=capacity))
        theta=float(np.max(abs(got['cells'][:,:,0]-c['expected/water'])/(1000.*c['cells'][:,:,2])))
        temperature=float(np.max(abs(got['temperature_k']-c['expected/temperature'])))
        assert theta<=.002 and temperature<=.05
        actual,bounds=audit_closed_storage(c['cells'],got['cells'])
        assert np.all(abs(got['residual_mass_heat'])<=got['bound_mass_heat'])
        print('Saturated producer native reference',json.dumps(dict(backend=name,index=index,
            maximum_theta_error=theta,maximum_temperature_error_k=temperature,
            actual_mass_heat_residual=actual.tolist(),original_bounds=bounds.tolist())),flush=True)


def test_saturated_all_raw_fields_and_second_step_same_precision():
    _,cuda,_=backend('cuda32');_,jax,_=backend('jax32')
    for index in range(2):
        c=fixture(index,'32');a=b=c['cells']
        for step in range(2):
            left=host(cuda(a,c['properties'],c['options'],c['dt']))
            right=host(jax(b,c['properties'],c['options'],c['dt']));assert left.keys()==right.keys()
            audit_closed_storage(a,left['cells']);audit_closed_storage(b,right['cells'])
            for key in left:
                assert left[key].dtype==right[key].dtype
                if left[key].dtype.kind in 'biu':np.testing.assert_array_equal(left[key],right[key],err_msg=key)
                else:np.testing.assert_allclose(left[key],right[key],rtol=1e-6,atol=1e-6,err_msg=key)
            for result in [left,right]:
                capacity=np.float32(1000.)*result['cells'][:,:,2]*c['properties'][:,0,None]
                assert np.all((result['cells'][:,:,0]>=0.)&(result['cells'][:,:,0]<=capacity))
            a,b=left['cells'],right['cells']
            print('Saturated producer raw carry',json.dumps(dict(index=index,step=step,fields=len(left),
                maximum_absolute={k:float(np.max(abs(left[k].astype(float)-right[k].astype(float)))) for k in left})),flush=True)


def test_original_overfull_input_is_still_rejected():
    for name in ['jax32','cuda32']:
        _,run,_=backend(name);c=fixture(0,'32')
        with pytest.raises(ValueError):run(c['invalid_cells'],c['properties'],c['options'],c['dt'])

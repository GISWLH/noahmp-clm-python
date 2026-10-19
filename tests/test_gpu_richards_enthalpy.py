"""Native conservative water/heat against frozen EHPC NumPy references."""
from pathlib import Path
import json,math,os
import numpy as np
import pytest


def case(precision,index):
    with np.load(Path(__file__).parent/'fixtures/richards_heat_reference.npz',allow_pickle=False) as z:
        prefix=precision+'/'+str(index)+'/'
        return {k[len(prefix):]:v for k,v in z.items() if k.startswith(prefix)}


def host(result):return {k:v.get() if hasattr(v,'get') else np.asarray(v) for k,v in result.items()}


def audit_closed_storage(before,after):
    """Independent promoted sums of actual stores, using the original bound."""
    residual=[];bound=[];epsilon=np.finfo(before.dtype).eps
    for old,new in zip(before,after):
        one=[];limit=[]
        for field in [0,1]:
            one.append(math.fsum(map(float,new[:,field]))-math.fsum(map(float,old[:,field])))
            scale=math.fsum(abs(float(v)) for v in np.r_[old[:,field],new[:,field]])
            limit.append(32.*epsilon*max(scale,1.))
        residual.append(one);bound.append(limit)
    residual=np.asarray(residual);bound=np.asarray(bound)
    assert np.all(abs(residual)<=bound)
    return residual,bound


@pytest.mark.parametrize('backend',['jax32','jax64','cuda32'])
def test_native_heat_against_all_ehpc_quantized_references(backend):
    jax=pytest.importorskip('jax');assert any(d.platform=='gpu' for d in jax.devices())
    if backend=='cuda32':
        pytest.importorskip('cupy')
        from noahmp_clm.gpu.cuda.richards_enthalpy import run_richards_enthalpy_cuda as run
    else:
        assert backend!='jax64' or jax.config.x64_enabled
        from noahmp_clm.gpu.richards_enthalpy import run_richards_enthalpy_jax as run
    for index in range(7):
        c=case(backend[-2:],index);got=host(run(*(c[k] for k in ['cells','properties','options','dt'])))
        assert all(v.dtype==c['cells'].dtype for v in got.values() if v.dtype.kind=='f')
        theta_error=float(np.max(abs(got['cells'][:,:,0]-c['expected/cells'][:,:,0])/(1000.*c['cells'][:,:,2])))
        temperature_error=float(np.max(abs(got['temperature_k']-c['expected/temperature_k'])))
        assert theta_error<=.002 and temperature_error<=.05
        assert np.all(got['phase_converged'])
        assert np.all(abs(got['residual_mass_heat'])<=got['bound_mass_heat'])
        actual,original=audit_closed_storage(c['cells'],got['cells'])
        assert np.all(got['maximum_substep_temperature_change_k']<=.05) and np.all(got['maximum_substep_liquid_fraction']<=.05)
        np.testing.assert_array_equal(got['cells'][:,:,2],c['cells'][:,:,2])
        if index==6:np.testing.assert_array_equal(got['cells'],c['cells'])
        print('Richards native water/heat',json.dumps(dict(backend=backend,index=index,max_soil_water_error_m3m3=theta_error,
            max_temperature_error_k=temperature_error,residual_mass_heat=got['residual_mass_heat'].tolist(),original_bounds=got['bound_mass_heat'].tolist(),
            independent_actual_residual_mass_heat=actual.tolist(),independent_original_bounds=original.tolist())),flush=True)


def test_water_heat_all_returned_fields_cuda_jax32_same_precision_and_state_carry():
    pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.richards_enthalpy import run_richards_enthalpy_jax
    from noahmp_clm.gpu.cuda.richards_enthalpy import run_richards_enthalpy_cuda
    saved={};maximum={}
    for index in range(7):
        c=case('32',index);a=b=c['cells'];dt=c['dt']*.5
        for step in range(2):
            ga=host(run_richards_enthalpy_cuda(a,c['properties'],c['options'],dt))
            gb=host(run_richards_enthalpy_jax(b,c['properties'],c['options'],dt));assert ga.keys()==gb.keys()
            audit_closed_storage(a,ga['cells']);audit_closed_storage(b,gb['cells'])
            for k in ga:
                saved[str(index)+'/'+str(step)+'/cuda/'+k]=ga[k];saved[str(index)+'/'+str(step)+'/jax/'+k]=gb[k]
            if os.environ.get('NOAHMP_RICHARDS_HEAT_DIAGNOSTICS'):
                folder=Path(os.environ['NOAHMP_RICHARDS_HEAT_DIAGNOSTICS']);folder.mkdir(exist_ok=True,parents=True)
                np.savez_compressed(folder/('case'+str(index)+'step'+str(step)+'.npz'),**{'cuda/'+k:v for k,v in ga.items()},**{'jax/'+k:v for k,v in gb.items()})
            for k in ga:
                assert ga[k].dtype==gb[k].dtype
                if ga[k].dtype.kind in 'biu':np.testing.assert_array_equal(ga[k],gb[k],err_msg=k)
                else:np.testing.assert_allclose(ga[k],gb[k],rtol=1e-6,atol=1e-6,err_msg=k)
                maximum[k]=max(maximum.get(k,0.),float(np.max(abs(ga[k].astype(float)-gb[k].astype(float)))))
            a,b=ga['cells'],gb['cells']
    print('Richards heat all raw state carry',json.dumps(maximum),flush=True)

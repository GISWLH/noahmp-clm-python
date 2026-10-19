"""Actual20/23-array C-N/MOST capacity histories and carried native steps."""
import numpy as np
import pytest
from test_gpu_canopy_capacity_layers import saved,capture
from test_gpu_clm_layered_column import host,budgets
from test_gpu_clm_bgc_column import cn_budgets
from test_gpu_snicar import assert_science_close


def inputs(precision,structured):
    data=saved();prefix='full'+precision+'/'+str(int(structured))+'/'
    count=int(data[prefix+'state_count'])
    arrays=tuple(data[prefix+'a'+str(k)] for k in range(count))
    return data,prefix,arrays


def run(backend,precision,structured,most,arrays):
    data,prefix,_=inputs(precision,structured)
    from noahmp_clm.gpu.clm_capacity_column import step_clm_capacity_columns_jax,step_clm_capacity_columns_cuda
    step=step_clm_capacity_columns_jax if backend=='jax' else step_clm_capacity_columns_cuda
    forcing=data[prefix+'forcing']
    kwargs=dict(environment=data[prefix+'environment'],canopy_optics=data[prefix+'optics'],return_surface=True,
                nitrogen_deposition_gn_m2=np.full(len(forcing),1e-5,forcing.dtype),leaf_parameters=data[prefix+'template'])
    if most:kwargs['most_geometry']=data[prefix+'geometry']
    return step(arrays,forcing,kind='bgc',structured=structured,**kwargs)


def original_budgets(arrays,forcing,result):
    return dict(**budgets(arrays[:8],forcing,np.zeros((len(forcing),5),forcing.dtype),result),
                **cn_budgets(arrays[:16],result,np.full(len(forcing),1e-5,forcing.dtype)))


def flattened(value,prefix=''):
    result={}
    for name,one in value.items():
        if isinstance(one,dict):result.update(flattened(one,prefix+name+'/'))
        else:result[prefix+name]=one
    return result


def compare_every_raw(left,right):
    a,b=flattened(left),flattened(right);assert a.keys()==b.keys();worst=0.;ratio=0.
    for name,x in a.items():
        y=b[name];assert x.dtype==y.dtype and x.shape==y.shape,name
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)),name
        if x.dtype.kind in 'biu':np.testing.assert_array_equal(x,y,err_msg=name)
        else:
            np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6,err_msg=name)
            delta=abs(x.astype(float)-y.astype(float));worst=max(worst,float(delta.max(initial=0.)))
            ratio=max(ratio,float((delta/(1e-6+1e-6*abs(y.astype(float)))).max(initial=0.)))
    return dict(raw_fields=len(a),maximum_absolute_difference=worst,maximum_gate_ratio=ratio)


@pytest.mark.parametrize('precision',['32','64'])
def test_native_capacity_bgc_matches_ehpc_and_original_budgets(precision):
    jax=pytest.importorskip('jax')
    if precision=='64' and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_capacity_column import next_capacity_arrays
    for structured in [False,True]:
        data,prefix,original=inputs(precision,structured);forcing=data[prefix+'forcing']
        for most in [False,True]:
            arrays=original
            for index in range(2):
                raw=run('jax',precision,structured,most,arrays);got=host(raw)
                ref=prefix+'ref'+str(int(most))+'/'+str(index)+'/'
                maximum={}
                for name in ['h','le','gpp_gc_m2','nee_gc_m2']:
                    expected=data[ref+name];actual=got['diagnostics'][name];maximum[name]=float(abs(actual-expected).max())
                    cn=name.endswith('gc_m2')
                    assert_science_close(actual,expected,relative=.1,zero_floor=.001 if cn else 10.,absolute=.0001 if cn else 1.)
                np.testing.assert_allclose(got['surface']['leaf_temperature'],data[ref+'leaf_temperature'],rtol=0.,atol=.05)
                np.testing.assert_array_equal(got['capacity_temperature_values'],data[ref+'memory_values'])
                np.testing.assert_array_equal(got['capacity_temperature_controls'],data[ref+'memory_controls'])
                np.testing.assert_allclose(got['canopy_capacity']['next_mean_temperature_k'],data[ref+'next_mean_temperature_k'],rtol=0.,atol=.001)
                snapshot=flattened(got)
                capture('whole-reference'+precision+'-'+str(int(structured))+'-'+str(int(most))+'-'+str(index),**snapshot)
                before=tuple(np.asarray(v) for v in arrays)
                print('whole capacity reference',precision,structured,most,index,dict(maximum_absolute=maximum,
                      original_budgets=original_budgets(before,forcing,got)),flush=True)
                arrays=next_capacity_arrays(arrays,raw,kind='bgc',structured=structured)


def test_native_capacity_cuda_jax_all_fields_and_all_carried_arrays():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_capacity_column import next_capacity_arrays
    for structured in [False,True]:
        data,prefix,original=inputs('32',structured);forcing=data[prefix+'forcing']
        for most in [False,True]:
            cuda=jax=original
            for index in range(2):
                before_cuda=tuple(v.get() if hasattr(v,'get') else np.asarray(v) for v in cuda)
                before_jax=tuple(np.asarray(v) for v in jax)
                a=run('cuda','32',structured,most,cuda);b=run('jax','32',structured,most,jax)
                left,right=host(a,cp.asnumpy),host(b)
                next_cuda=next_capacity_arrays(cuda,a,kind='bgc',structured=structured)
                next_jax=next_capacity_arrays(jax,b,kind='bgc',structured=structured)
                state_left={'next'+str(k):cp.asnumpy(v) for k,v in enumerate(next_cuda)}
                state_right={'next'+str(k):np.asarray(v) for k,v in enumerate(next_jax)}
                capture('whole-allraw32-'+str(int(structured))+'-'+str(int(most))+'-'+str(index),
                        **{'cuda/'+k:v for k,v in flattened(left).items()},**{'jax/'+k:v for k,v in flattened(right).items()},
                        **{'cuda/'+k:v for k,v in state_left.items()},**{'jax/'+k:v for k,v in state_right.items()})
                stats=compare_every_raw(left,right);stats['state']=compare_every_raw(state_left,state_right)
                print('whole capacity all raw',structured,most,index,dict(**stats,
                      cuda_budgets=original_budgets(before_cuda,forcing,left),jax_budgets=original_budgets(before_jax,forcing,right)),flush=True)
                cuda,jax=next_cuda,next_jax

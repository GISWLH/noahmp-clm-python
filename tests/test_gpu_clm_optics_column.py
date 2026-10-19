"""PFT inputs through complete structured C/N, MOST and carried state."""
import numpy as np
import pytest
from test_gpu_canopy_optics import saved,capture
from test_gpu_clm_layered_column import host,budgets
from test_gpu_clm_bgc_column import cn_budgets
from test_gpu_snicar import assert_science_close


def inputs(precision):
    z=saved();prefix='full'+precision+'_'
    return tuple(z[prefix+'a'+str(k)] for k in range(19)),z[prefix+'forcing'],z[prefix+'optics'],z[prefix+'geometry']


def run(backend,precision,most,arrays=None):
    a,f,optics,geometry=inputs(precision);a=a if arrays is None else arrays
    if backend=='jax':
        from noahmp_clm.gpu.clm_structure_column import step_clm_structure_columns_jax as step
        from noahmp_clm.gpu.enthalpy import run_soil_thermodynamics_jax as inverse
        import jax.numpy as xp
    else:
        from noahmp_clm.gpu.clm_structure_column import step_clm_structure_columns_cuda as step
        from noahmp_clm.gpu.cuda.enthalpy import run_soil_thermodynamics_cuda as inverse
        import cupy as xp
    kwargs=dict(canopy_optics=optics,nitrogen_deposition_gn_m2=np.asarray([1e-5],f.dtype),return_surface=True)
    if most:kwargs['most_geometry']=geometry
    result=step(a,f,kind='bgc',**kwargs)
    g=result['soil_cells'];p=xp.asarray(a[2]);o=result['options'];n=len(g)
    parameters=xp.broadcast_to(p[:,None,[0,1,6]],(n,4,3));u=xp.concatenate((g,parameters),axis=2).reshape(-1,6)
    temperature=inverse(u,xp.repeat(o[:,1],4),inverse=True)['temperature_k'].reshape(n,4)
    moisture=g[:,:,0]/(1000.*g[:,:,2])
    return result,temperature,moisture


def audit(a,f,result):
    waterheat=budgets(a[:8],f,np.zeros((len(f),5),f.dtype),result)
    carbon=cn_budgets(a[:16],result,np.asarray([1e-5],f.dtype))
    return dict(**waterheat,**carbon)


@pytest.mark.parametrize('precision',['32','64'])
def test_pft_optics_full_bgc_matches_ehpc_and_original_budgets(precision):
    jax=pytest.importorskip('jax')
    if precision=='64' and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    z=saved();a,f,_,_=inputs(precision)
    for most in [False,True]:
        raw,t,w=run('jax',precision,most);got=host(raw);maximum={}
        for name in ['h','le','gpp_gc_m2','nee_gc_m2']:
            expected=z['fullref'+str(int(most))+'_'+name];actual=got['diagnostics'][name]
            maximum[name]=float(abs(actual-expected).max())
            cn=name.endswith('gc_m2');assert_science_close(actual,expected,relative=.1,zero_floor=.001 if cn else 10.,absolute=.0001 if cn else 1.)
        np.testing.assert_allclose(np.asarray(t),z['fullref'+str(int(most))+'_tsoil_k'],rtol=0.,atol=.05)
        np.testing.assert_allclose(np.asarray(w),z['fullref'+str(int(most))+'_smc_m3m3'],rtol=.1,atol=1e-5)
        print('PFT optics full BGC science',precision,most,dict(maximum_absolute=maximum,original_budgets=audit(a,f,got)),flush=True)


def compare_every_field(left,right,prefix=''):
    assert set(left)==set(right),prefix
    failures=[];maximum={}
    def compare(a,b,path):
        assert set(a)==set(b),path
        for name,value in a.items():
            label=path+'/'+name;other=b[name]
            if isinstance(value,dict):compare(value,other,label);continue
            assert value.shape==other.shape and value.dtype==other.dtype,label
            maximum[label]=float(abs(value.astype(float)-other.astype(float)).max(initial=0.))
            try:
                if value.dtype.kind in 'biu':np.testing.assert_array_equal(value,other,err_msg=label)
                else:np.testing.assert_allclose(value,other,rtol=1e-6,atol=1e-6,err_msg=label)
            except AssertionError:failures.append(label)
    compare(left,right,prefix)
    print('PFT optics complete raw sameprecision',dict(maximum_absolute=maximum,failed_fields=failures),flush=True)
    assert not failures,failures


def test_pft_optics_full_bgc_cuda_all_fields_two_carried_steps():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_structure_column import next_structure_arrays
    for most in [False,True]:
        original,f,_,_=inputs('32');cu_before=ja_before=original
        for step in range(2):
            cu,ct,cw=run('cuda','32',most,cu_before);ja,jt,jw=run('jax','32',most,ja_before)
            left,right=host(cu,cp.asnumpy),host(ja)
            def flat(value,prefix=''):
                out={}
                for name,x in value.items():
                    if isinstance(x,dict):out.update(flat(x,prefix+name+'/'))
                    else:out[prefix+name]=x
                return out
            capture('full32-'+str(int(most))+'-'+str(step),**{'cuda/'+n:v for n,v in flat(left).items()},**{'jax/'+n:v for n,v in flat(right).items()})
            compare_every_field(left,right)
            np.testing.assert_allclose(cp.asnumpy(ct),np.asarray(jt),rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(cp.asnumpy(cw),np.asarray(jw),rtol=1e-6,atol=1e-6)
            print('PFT optics carried BGC budgets',most,step,dict(cuda=audit(cu_before,f,left),jax=audit(ja_before,f,right)),flush=True)
            cu_before=tuple(cp.asnumpy(v) for v in next_structure_arrays(cu_before,cu,kind='bgc'))
            ja_before=tuple(np.asarray(v) for v in next_structure_arrays(ja_before,ja,kind='bgc'))

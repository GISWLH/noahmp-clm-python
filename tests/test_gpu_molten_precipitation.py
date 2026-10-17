"""Real failed BGC/MOST state and fully molten packs on native devices."""
from dataclasses import astuple, replace
import numpy as np
import pytest
from test_molten_precipitation import ham_failure
from test_gpu_clm_layered_column import host, compare_same_precision, budgets, science, reference
from test_gpu_clm_bgc_column import reference as bgc_reference, cn_budgets
from noahmp_clm.gpu.clm_bgc_column import pack_clm_bgc_columns, pack_bgc_environment
from noahmp_clm.physics.canopy_most import MOSTGeometry


def actual_case(dtype):
    state,f,kw=ham_failure()
    arrays=pack_clm_bgc_columns([state],dtype=dtype,co2_ppm=kw['co2_ppm'],cosine_zenith=kw['cosine_zenith'],diffuse_fraction=kw['diffuse_fraction'])
    env,cal=pack_bgc_environment([kw['phenology_environment']],dtype=dtype)
    return arrays,np.array([astuple(f)],dtype),env,cal,np.array([astuple(kw['most_geometry'])],dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_failed_ham_bgc_most_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax
    a,f,e,c,g=actual_case(dtype);dep=np.zeros((1,5),dtype)
    got=host(step_clm_bgc_columns_jax(*a,f,environment=e,calendar=c,most_geometry=g,return_surface=True))
    ref=bgc_reference(tuple(x[0] for x in a),f[0],dep[0],e[0],c[0],0.,1.,most_geometry=MOSTGeometry(*map(float,g[0])))
    for key in ('h','le','gpp_gc_m2','nee_gc_m2'):
        expected=ref['diagnostics'][key];actual=float(got['diagnostics'][key][0])
        assert abs(actual-expected)<=max(.01 if key.endswith('gc_m2') else 1.,.1*abs(expected))
        print('HaM native',str(dtype),key,'model',actual,'reference',expected,'absolute',abs(actual-expected),flush=True)
    assert got['options'][0,0]==0
    print('HaM native M/H',str(dtype),budgets(a[:8],f,dep,got),flush=True)
    print('HaM native C/N',str(dtype),cn_budgets(a,got,np.zeros(1,dtype)),flush=True)


def test_failed_ham_bgc_most_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax
    from noahmp_clm.gpu.cuda.clm_bgc_column import step_clm_bgc_columns_cuda
    a,f,e,c,g=actual_case(np.float32)
    left=host(step_clm_bgc_columns_cuda(*a,f,environment=e,calendar=c,most_geometry=g,return_surface=True),cp.asnumpy)
    right=host(step_clm_bgc_columns_jax(*a,f,environment=e,calendar=c,most_geometry=g,return_surface=True))
    print('HaM whole BGC/MOST same precision',compare_same_precision(left,right),flush=True)


def molten_cases(dtype):
    from noahmp_clm.column import ColumnState,Forcing
    from noahmp_clm.clm_state import CLMColumnState,clm_options
    from noahmp_clm.physics.snowpack import Snowpack,LF,CW
    from noahmp_clm.gpu.clm_layered_column import pack_clm_layered_columns
    states=[]
    for runoff in (3,6):
        opt=replace(clm_options(),OptRunoffSurface=runoff,OptRunoffSubsurface=runoff)
        old=ColumnState(ts=285.,tsoil=np.full(4,285.),smc=np.full(4,.4499),lai=1.,options=opt)
        state=CLMColumnState.from_column(old,options=opt).with_layered_canopy()
        m=np.array([.2,1.]);h=m*(LF+CW*np.array([5.,2.]))
        impurities=np.full((2,5),1e-9)
        states.append(replace(state,snow=Snowpack(m,h,np.array([1e-5,2e-4]),impurities=impurities)))
    arrays=pack_clm_layered_columns(states,dtype=dtype)
    f=np.broadcast_to(np.array([1800.,285.,.006,2.,101325.,0.,350.,0.],dtype),(2,8)).copy()
    return arrays,f,np.zeros((2,5),dtype)


def test_multilayer_liquid_pack_full_column_native_conservation_and_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax
    from noahmp_clm.gpu.cuda.clm_layered_column import step_clm_layered_columns_cuda
    a,f,dep=molten_cases(np.float32)
    left=host(step_clm_layered_columns_cuda(*a,f,return_surface=True),cp.asnumpy)
    right=host(step_clm_layered_columns_jax(*a,f,return_surface=True))
    print('Molten two-layer same precision',compare_same_precision(left,right),flush=True)
    assert np.all(left['options'][:,0]==0)
    np.testing.assert_allclose(left['impurity_output_kgm2'],2e-9,rtol=1e-6,atol=1e-15)
    print('Molten two-layer M/H',budgets(a,f,dep,left),flush=True)
    for k,row in enumerate(zip(*a)):
        ref=reference(row,f[k],dep[k])
        one={key:({x:y[k] for x,y in v.items()} if key=='diagnostics' else v[k]) for key,v in left.items()
             if key not in ('surface','canopy_interception','canopy_post_interception')}
        print('Molten two-layer reference',k,science(one,ref,np.float32),flush=True)

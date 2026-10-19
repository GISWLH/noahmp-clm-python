"""EHPC-generated references; no NumPy forward physics on the GPU host."""
from pathlib import Path
import os
import numpy as np
import pytest
from test_gpu_snicar import assert_science_close


def saved():
    with np.load(Path(__file__).parent/'fixtures/pft_canopy_optics_reference.npz',allow_pickle=False) as z:
        return {n:np.array(v) for n,v in z.items()}


def component(precision):
    z=saved();return [z[precision+'_a'+str(k)] for k in range(4)]+[z[precision+'_leaf'],z[precision+'_optics']]


def capture(name,**arrays):
    folder=os.environ.get('NOAHMP_PFT_DIAGNOSTICS')
    if folder:
        path=Path(folder);path.mkdir(parents=True,exist_ok=True);np.savez_compressed(path/(name+'.npz'),**arrays)


def check_reference(precision,result):
    refs=saved();maximum={}
    for name,value in result.items():
        key=precision+'_ref_'+name
        if key not in refs:continue
        expected=refs[key];maximum[name]=float(abs(np.asarray(value)-expected).max(initial=0.))
        albedo='albedo' in name
        assert_science_close(value,expected,relative=.1,zero_floor=.001 if albedo else 1.,absolute=1e-6 if albedo else .1)
    sw=component(precision)[3][:,0].astype(float)
    budget=sw-result['reflected'].astype(float)-result['canopy_absorbed'].astype(float).sum(1)-result['snow_absorbed'].astype(float).sum(1)-result['soil_absorbed'].astype(float)
    assert np.all(abs(budget)<(.005 if precision=='32' else 1e-8))
    print('PFT optics native reference',precision,dict(maximum_absolute=maximum,maximum_joint_energy_residual_Wm2=float(abs(budget).max())),flush=True)


@pytest.mark.parametrize('precision',['32','64'])
def test_pft_optics_device_matches_ehpc_reference(precision):
    jax=pytest.importorskip('jax')
    if precision=='64' and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.canopy_optics import run_canopy_snow_optics_jax
    result=run_canopy_snow_optics_jax(*component(precision))
    def host(v):return {k:host(x) if isinstance(x,dict) else np.asarray(x) for k,x in v.items()}
    check_reference(precision,host(result))


def test_pft_optics_cuda_all624_raw_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.canopy_optics import canopy_snow_optics_raw_jax,unpack_canopy_snow_optics
    from noahmp_clm.gpu.cuda.canopy_optics import canopy_snow_optics_raw_cuda
    data=component('32');cuda=cp.asnumpy(canopy_snow_optics_raw_cuda(*data));jax=np.asarray(canopy_snow_optics_raw_jax(*data))
    capture('component32-all624',cuda=cuda,jax=jax)
    np.testing.assert_allclose(cuda,jax,rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(cuda[:,168:174],jax[:,168:174])
    result=unpack_canopy_snow_optics(cuda);check_reference('32',result)
    print('PFT optics all624 raw CUDA/JAX32',float(abs(cuda.astype(float)-jax.astype(float)).max()),flush=True)


def test_pft_optics_invalid_geometry_and_passivity():
    from noahmp_clm.gpu.canopy_optics import validate_canopy_optics
    area,_,_,_,leaf,p=component('64');validate_canopy_optics(area,leaf,p,np)
    bad=leaf.copy();bad[0,0]=area[0,0]+1.
    with pytest.raises(ValueError):validate_canopy_optics(area,bad,p,np)
    bad=p.copy();bad[0,0]=.9;bad[0,2]=.2
    with pytest.raises(ValueError):validate_canopy_optics(area,leaf,bad,np)


def test_nonabsorbing_leaf_stem_and_mixed_device_limits_all_raw_fields():
    cp=pytest.importorskip('cupy');jax=pytest.importorskip('jax')
    if not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.canopy_optics import canopy_snow_optics_raw_jax,unpack_canopy_snow_optics
    from noahmp_clm.gpu.cuda.canopy_optics import canopy_snow_optics_raw_cuda
    for precision in ('32','64'):
        data=component(precision);dtype=data[0].dtype;rng=np.random.default_rng(391821)
        leaf=(10.**rng.uniform(-5.,1.,data[0].shape)).astype(dtype)
        stem=(10.**rng.uniform(-5.,1.,data[0].shape)).astype(dtype)
        stem[::3]=0.;leaf[1::3]=0.
        data[0]=leaf+stem;data[4]=leaf
        data[5]=np.tile(np.asarray([.8,.8,.2,.2,.4,.4,.6,.6],dtype),(len(leaf),1))
        raw=np.asarray(canopy_snow_optics_raw_jax(*data));capture('nonabsorbing'+precision,jax=raw)
        result=unpack_canopy_snow_optics(raw)
        tolerance=.005 if precision=='32' else 1e-8
        assert np.max(abs(result['canopy_absorbed']))<tolerance
        # Leaf photon allocation is exactly zero. The transport's raw stem
        # absorption retains finite-precision coefficient/flux roundoff and
        # uses the original hard energy bound, rather than bitwise zero.
        np.testing.assert_array_equal(raw[:,584:604],np.zeros_like(raw[:,584:604]))
        assert np.max(abs(raw[:,604:624]))<tolerance
        energy=data[3][:,0].astype(float)-result['reflected'].astype(float)-result['canopy_absorbed'].astype(float).sum(1)-result['snow_absorbed'].astype(float).sum(1)-result['soil_absorbed'].astype(float)
        assert np.max(abs(energy))<tolerance
        if precision=='32':
            cuda=cp.asnumpy(canopy_snow_optics_raw_cuda(*data));capture('nonabsorbing32-pair',cuda=cuda,jax=raw)
            np.testing.assert_allclose(cuda,raw,rtol=1e-6,atol=1e-6)
            np.testing.assert_array_equal(cuda[:,168:174],raw[:,168:174])
            print('Nonabsorbing all624 CUDA/JAX32 maximum',float(abs(cuda.astype(float)-raw.astype(float)).max()),flush=True)

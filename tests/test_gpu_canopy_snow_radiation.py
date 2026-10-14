import numpy as np
import pytest
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.physics.canopy_snow_radiation import canopy_snow_shortwave
from noahmp_clm.gpu.canopy_snow_radiation import pack_canopy_snow_case,validate_canopy_snow


def cases(dtype,random_count=64):
    rows=[]
    for layers in (0,1,5):
        snow=Snowpack.from_ice(np.full(layers,10.),TF-5.) if layers else Snowpack.empty()
        for fv in (0.,.6,1.):
            for diffuse in (0.,.4,1.):
                rows.append(pack_canopy_snow_case(np.full(10,.3),snow,700.,vegetated_fraction=fv,diffuse_fraction=diffuse,dtype=dtype))
    snow=Snowpack.from_ice([20.,30.],260.)
    rows.append(pack_canopy_snow_case(np.zeros(10),snow,500.,dtype=dtype))
    rows.append(pack_canopy_snow_case(np.full(10,.3),snow,0.,dtype=dtype))
    rng=np.random.default_rng(22189)
    for case in range(random_count):
        ns=case%6;mass=10.**rng.uniform(-4.,2.7,ns)
        pack=Snowpack(mass,-2100.*mass*5,mass/250.,rng.uniform(20.,1800.,ns),mass[:,None]*10.**rng.uniform(-10.,-4.,(ns,5)))
        area=rng.uniform(.01,.8,10);area[::3]=0.
        rows.append(pack_canopy_snow_case(area,pack,rng.uniform(0.,1200.),vegetated_fraction=rng.uniform(0.,1.),
            diffuse_fraction=rng.uniform(0.,1.),cosine_zenith=rng.uniform(.05,1.),ground_albedo=rng.uniform(0.,.5,2),dtype=dtype))
    return tuple(np.stack([row[i] for row in rows]) for i in range(4))


def reference(area,c,n,u):
    n=int(n);area,c,u=np.asarray(area,float),np.asarray(c,float),np.asarray(u,float)
    pack=Snowpack(c[:n,0],np.zeros(n),c[:n,0]/250.,c[:n,1],c[:n,2:])
    r=canopy_snow_shortwave(area,pack,u[0],vegetated_fraction=u[1],diffuse_fraction=u[2],cosine_zenith=u[3],ground_albedo=u[4:6])
    r['snow_absorbed']=np.pad(r['snow_absorbed'],(0,5-n))
    r['absorbed_par']=r['canopy']['absorbed_par']
    return r


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_canopy_snow_device_reference_and_budget(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.canopy_snow_radiation import run_canopy_snow_jax
    data=cases(dtype);result=run_canopy_snow_jax(*data)
    names=('canopy_absorbed','absorbed_par','snow_absorbed','soil_absorbed','reflected','residual','snow_direct_albedo','snow_diffuse_albedo')
    maxima={k:0. for k in names}
    for row in range(len(data[0])):
        ref=reference(*(v[row] for v in data))
        for key in names:
            actual=np.asarray(result[key][row]);expected=ref[key]
            maxima[key]=max(maxima[key],float(np.max(np.abs(actual-expected))))
            # Double precision still uses the established finite-slab canopy
            # approximation shared with its existing NumPy reference.
            from test_gpu_snicar import assert_science_close
            assert_science_close(actual,expected,relative=.1 if dtype is np.float32 else 1e-7,
                absolute=.1 if dtype is np.float32 and 'albedo' not in key else 1e-6,
                zero_floor=.001 if 'albedo' in key else 1.)
        budget=float(data[3][row,0])-float(result['reflected'][row])-np.asarray(result['canopy_absorbed'][row],float).sum()-np.asarray(result['snow_absorbed'][row],float).sum()-float(result['soil_absorbed'][row])
        assert abs(budget)<(.005 if dtype is np.float32 else 1e-8)
    print('canopy/SNICAR',dtype,'max absolute',maxima,flush=True)


def flatten(result,prefix=''):
    for k,v in result.items():
        if isinstance(v,dict):yield from flatten(v,prefix+k+'.')
        else:yield prefix+k,v


def test_canopy_snow_cuda_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.canopy_snow_radiation import run_canopy_snow_jax
    from noahmp_clm.gpu.cuda.canopy_snow_radiation import run_canopy_snow_cuda
    data=cases(np.float32);a=dict(flatten(run_canopy_snow_cuda(*data)));b=dict(flatten(run_canopy_snow_jax(*data)))
    for key in a:
        aa,bb=cp.asnumpy(a[key]),np.asarray(b[key])
        print('canopy/SNICAR CUDA/JAX32',key,np.max(aa!=bb) if aa.dtype==bool else np.max(np.abs(aa-bb)),flush=True)
        np.testing.assert_allclose(aa,bb,rtol=1e-6,atol=1e-6)


def test_canopy_snow_bad_fraction():
    a,c,n,u=cases(np.float64,0);validate_canopy_snow(a,c,n,u,np)
    u[0,1]=1.1
    with pytest.raises(ValueError,match='fraction'):validate_canopy_snow(a,c,n,u,np)

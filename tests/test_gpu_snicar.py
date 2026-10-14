"""SNICAR device columns, independent spectral equations and flux budgets."""
import numpy as np
import pytest
import json
from pathlib import Path
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.physics.snicar import snow_shortwave,bulk_optics,spectral_fluxes
from noahmp_clm.gpu.snicar import pack_snow_optics,validate_snow_optics


def cases(dtype=np.float64,random_count=84):
    packs=[Snowpack.empty(),Snowpack.from_ice([100.],TF-5.,radius_um=54.5),
        Snowpack.from_ice([100.],TF-5.,radius_um=1000.),Snowpack.from_ice([.0001],TF-5.),
        Snowpack.from_ice([2.,5.,10.,20.,40.],TF-5.,radius_um=[20.,30.,54.5,1500.,1800.])]
    fresh=packs[1]
    packs.append(Snowpack(fresh.mass,fresh.enthalpy,fresh.thickness,fresh.radius_um,np.full((1,5),1e-4)))
    rows=[pack_snow_optics(p,[300.,300.],[100.,100.],dtype=dtype) for p in packs]
    rows.append(pack_snow_optics(packs[-1],[0.,0.],[0.,0.],dtype=dtype))
    saved=json.loads((Path(__file__).parent/'fixtures/snow_weak_absorption.json').read_text())
    rows.append((np.asarray(saved['cells'],dtype),np.asarray(saved['count'],np.int32),np.asarray(saved['boundary'],dtype)))
    rng=np.random.default_rng(28809)
    for case in range(random_count):
        n=case%6;mass=10.**rng.uniform(-4.,2.7,n)
        impurity=mass[:,None]*10.**rng.uniform(-10.,-4.,(n,5))
        p=Snowpack(mass,mass*-2100.*5,mass/250.,rng.uniform(20.,1800.,n),impurity)
        rows.append(pack_snow_optics(p,rng.uniform(0.,600.,2),rng.uniform(0.,400.,2),
            ground_albedo=rng.uniform(0.,1.,2),cosine_zenith=rng.uniform(.02,1.),dtype=dtype))
    return tuple(np.stack([row[i] for row in rows]) for i in range(3))


def assert_science_close(actual,expected,*,relative=.1,absolute=.1,zero_floor=1.):
    """Use the relative gate above the floor, absolute gate below it, not both."""
    actual,expected=np.broadcast_arrays(np.asarray(actual),np.asarray(expected))
    normal=np.abs(expected)>zero_floor
    np.testing.assert_allclose(actual[normal],expected[normal],rtol=relative,atol=0.)
    np.testing.assert_allclose(actual[~normal],expected[~normal],rtol=0.,atol=absolute)


def reference(c,n,b):
    c,b=np.asarray(c,float),np.asarray(b,float);n=int(n)
    p=Snowpack(c[:n,0],np.zeros(n),c[:n,0]/250.,c[:n,1],c[:n,2:])
    r=snow_shortwave(p,b[:2],b[2:4],ground_albedo=b[4:6],cosine_zenith=b[6])
    conditions=np.zeros((2,5,26));optics=np.zeros((2,5,3,5))
    for mode in range(2):
        table=bulk_optics(p,diffuse=bool(mode))
        for band in range(5):
            q=r['conditions'][mode][band];v=conditions[mode,band]
            v[:n]=q['absorbed']
            for field,start in (('direct',5),('diffuse_down',11),('diffuse_up',17)):
                v[start:start+6]=np.pad(q[field],(0,5-n),mode='edge')
            v[23:]=q['reflected'],q['ground_absorbed'],q['residual']
            for prop,key in enumerate(('tau','omega','g')):optics[mode,band,prop,:n]=table[key][:,band]
    return dict(absorbed=np.pad(r['absorbed'],(0,5-n)),spectral_absorbed=np.pad(r['spectral_absorbed'],((0,5-n),(0,0))),
        spectral_ground_absorbed=r['spectral_ground_absorbed'],spectral_reflected=r['spectral_reflected'],
        ground_absorbed=r['ground_absorbed'],reflected=r['reflected'],albedo=0. if r['albedo'] is None else r['albedo'],
        residual=r['residual'],albedo_defined=r['albedo'] is not None,
        radius_clipped=np.pad((p.radius_um<30.)|(p.radius_um>1500.),(0,5-n)),conditions=conditions,optics=optics)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_snow_shortwave_jax_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.snicar import run_snow_shortwave_jax
    c,n,b=cases(dtype);got={k:np.asarray(v) for k,v in run_snow_shortwave_jax(c,n,b).items()}
    max_error={k:0. for k in got if got[k].dtype!=bool}
    for row in range(len(c)):
        expected=reference(c[row],n[row],b[row])
        for key,value in expected.items():
            if key in ('albedo_defined','radius_clipped'):np.testing.assert_array_equal(got[key][row],value);continue
            max_error[key]=max(max_error[key],float(np.max(np.abs(got[key][row]-value))))
            if key=='optics':np.testing.assert_allclose(got[key][row],value,rtol=1e-5 if dtype is np.float32 else 1e-12,atol=1e-6)
            else:assert_science_close(got[key][row],value,relative=.1 if dtype is np.float32 else 1e-7,
                absolute=1e-6 if key=='albedo' or dtype is np.float64 else .1,zero_floor=.001 if key=='albedo' else 1.)
        budget=float(b[row,:4].astype(float).sum()-got['reflected'][row]-got['absorbed'][row].astype(float).sum()-got['ground_absorbed'][row])
        assert abs(budget)<(.002 if dtype is np.float32 else 1e-8)
    print('SNICAR',dtype,'maximum absolute',max_error,flush=True)
    assert got['albedo'][1]>got['albedo'][2] and got['albedo'][1]>got['albedo'][5]
    assert np.min(got['absorbed'])>=(-.001 if dtype is np.float32 else -1e-8)


def test_snow_shortwave_cuda_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.snicar import run_snow_shortwave_jax
    from noahmp_clm.gpu.cuda.snicar import run_snow_shortwave_cuda
    inputs=cases(np.float32);a=run_snow_shortwave_cuda(*inputs);b=run_snow_shortwave_jax(*inputs)
    for key in a:
        aa,bb=cp.asnumpy(a[key]),np.asarray(b[key])
        print('SNICAR CUDA/JAX32',key,np.max(aa!=bb) if aa.dtype==bool else np.max(np.abs(aa-bb)),flush=True)
        np.testing.assert_allclose(aa,bb,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_device_spectrum_independent_and_limits(dtype):
    jax=pytest.importorskip('jax');j=pytest.importorskip('jax.numpy')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.snicar import spectrum_jax
    from test_snicar import independent_spectrum
    rng=np.random.default_rng(98082);optics=[];boundary=[];closure=[]
    for case in range(48):
        eddington=case%2==0
        optics.append(np.column_stack((rng.uniform(.001,3.,5),rng.uniform(.9 if eddington else .1,1.,5),rng.uniform(.1,.95,5))))
        boundary.append([*rng.uniform(0.,600.,2),rng.uniform(0.,.95),rng.uniform(.05,1.)]);closure.append(eddington)
    # Pure scattering, including direct-beam resonance for w=.75 and mu=1.
    for w,mu in ((.75,1.),(1.,.1),(1.,1.)):
        optics.append(np.column_stack(([2.,5.,0.,0.,0.],np.full(5,w),np.zeros(5))))
        boundary.append([500.,300.,.4,mu]);closure.append(False)
    v,u=np.array(optics,dtype),np.array(boundary,dtype);e=np.array(closure)
    output=np.asarray(jax.jit(jax.vmap(spectrum_jax))(j.asarray(v),j.asarray(u),j.asarray(e)))
    error=0.
    for row in range(len(v)):
        ref=independent_spectrum(*v[row].astype(float).T,*u[row].astype(float),bool(e[row]))
        expected=np.r_[ref['absorbed'],ref['direct'],ref['diffuse_down'],ref['diffuse_up'],ref['reflected'],ref['ground_absorbed']]
        error=max(error,float(np.max(np.abs(output[row,:25]-expected))))
        np.testing.assert_allclose(output[row,:25],expected,rtol=.1 if dtype is np.float32 else 1e-8,atol=.1 if dtype is np.float32 else 1e-6)
    assert np.max(np.abs(output[-2:,:5]))<(.001 if dtype is np.float32 else 1e-8)
    print('Independent spectrum',dtype,'maximum absolute W/m2',error,flush=True)


def test_snow_optics_contract():
    c,n,b=cases(np.float64,0);validate_snow_optics(c,n,b,np)
    bad=c.copy();bad[0,4,0]=1.
    with pytest.raises(ValueError,match='padding'):validate_snow_optics(bad,n,b,np)
    bad=b.copy();bad[0,6]=0.
    with pytest.raises(ValueError,match='boundary'):validate_snow_optics(c,n,bad,np)
    with pytest.raises(ValueError,match='integer'):validate_snow_optics(c,n.astype(float),b,np)


def test_gpu_thick_snow_analytic_limit():
    jax=pytest.importorskip('jax');j=pytest.importorskip('jax.numpy');cp=pytest.importorskip('cupy')
    from noahmp_clm.gpu.snicar import spectrum_jax
    from noahmp_clm.gpu.cuda.snicar import _module
    inputs=[];boundaries=[];closures=[];expected=[]
    for eddington in (False,True):
        for w,g,mu in ((.9999,.89,.2),(.999,.85,.7),(.98,.92,.3)):
            v=np.zeros((5,3),np.float32);v[0]=[1e6,w,g]
            u=np.array([300.,500.,.2,mu],np.float32)
            inputs.append(v);boundaries.append(u);closures.append(int(eddington))
            w,g=float(v[0,1]),float(v[0,2]);mu=float(u[3]);f=1-w*g*g
            ws=w*(1-g*g)/f;gs=g/(1+g)
            if eddington:a=(7-ws*(4+3*gs))/4;b=(ws*(4-3*gs)-1)/4;beta=(2-3*gs*mu)/4
            else:a=2-ws*(1+gs);b=ws*(1-gs);beta=(1-gs*mu)/2
            lam=np.sqrt(a*a-b*b);rd=b/(a+lam);rb=ws*(beta+rd*(1-beta))/(1+lam*mu)
            expected.append(300*rb+500*rd)
    v,u,e=np.array(inputs),np.array(boundaries),np.array(closures,np.int32)
    jj=np.asarray(jax.jit(jax.vmap(spectrum_jax))(j.asarray(v),j.asarray(u),j.asarray(e>0)))
    out=cp.empty((len(v),26),cp.float32)
    _module().get_function('run_snow_spectrum')((1,),(64,),(cp.asarray(v),cp.asarray(u),cp.asarray(e),out,np.int32(len(v))))
    cc=cp.asnumpy(out);np.testing.assert_allclose(cc,jj,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(cc[:,23],expected,rtol=.1,atol=.1)
    assert np.max(np.abs(cc[:,24]))<1e-6
    error=np.abs(cc[:,23]-expected)
    print('Thick analytic maximum W/m2',error.max(),'relative percent',(error/expected).max()*100,flush=True)

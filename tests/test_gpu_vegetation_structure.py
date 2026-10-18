"""Original NumPy reference and native component checks on real device arrays."""
import json
import numpy as np
import pytest
from noahmp_clm.physics.vegetation_structure import (
    CanopyStructureParameters,step_stem_area,exposed_canopy_geometry)
from noahmp_clm.gpu.vegetation_structure import FIELDS,pack_structure,validate_structure


def cases(dtype):
    rows=[]
    for kind in range(4):
        for short in (False,True):
            p=CanopyStructureParameters(kind,3.,1.,short)
            for old,previous,lai in [(0.,0.,0.),(1.,0.,0.),(2.,4.,0.),(2.,4.,3.),(.01,3.,3.)]:
                for depth in (0.,.05,.2,1.,2.,3.,4.):
                    rows.append(pack_structure(old,previous,lai,1800.,depth,p,dtype=dtype))
            if kind==0:rows.append(pack_structure(.8,5.,0.,1800.,0.,p,prescribed_sai=.6,dtype=dtype))
            if kind==3:
                for flags in [dict(harvested=True),dict(fallow=True),dict(newly_planted=True)]:
                    rows.append(pack_structure(1.,3.,0.,86400.,.1,p,dtype=dtype,**flags))
    rng=np.random.default_rng(182973)
    for _ in range(512):
        kind=int(rng.integers(0,4));top=rng.uniform(.1,30.);bottom=rng.uniform(0.,.9*top)
        p=CanopyStructureParameters(kind,top,bottom,bool(rng.integers(0,2)))
        rows.append(pack_structure(*rng.uniform(0.,8.,3),rng.uniform(1.,86400.),rng.uniform(0.,35.),p,dtype=dtype))
    return tuple(np.stack([v[k] for v in rows]) for k in range(3))


def references(arrays):
    rows=[]
    for c,p,o in zip(*arrays):
        c,p=c.astype(float),p.astype(float)
        parameters=CanopyStructureParameters(int(o[0]),*p,bool(o[1]))
        sai,d=step_stem_area(*c[:4],parameters,prescribed_sai=c[5] if o[2] else None,
            harvested=bool(o[3]),fallow=bool(o[4]),newly_planted=bool(o[5]))
        d.update(exposed_canopy_geometry(c[2],sai,c[4],parameters));rows.append(d)
    return {key:np.asarray([row[key] for row in rows]) for key in FIELDS}


def check_reference(actual,reference,atol):
    maximum=0.;relative=0.
    for key in FIELDS:
        a,b=np.asarray(actual[key],float),reference[key];delta=abs(a-b)
        np.testing.assert_allclose(a,b,rtol=.1,atol=atol,err_msg=key)
        maximum=max(maximum,float(delta.max()))
        mask=abs(b)>1e-4
        if mask.any():relative=max(relative,float((delta[mask]/abs(b[mask])).max()))
        if key in ('exposed_lai','exposed_sai','exposed_plant_area'):
            assert np.all(a[b==0.]==0.),key
    return dict(maximum_absolute=maximum,maximum_relative_percent=100.*relative)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_structure_jax_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype==np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.vegetation_structure import run_structure_jax
    arrays=cases(dtype);out=run_structure_jax(*arrays)
    print('native structure reference',dtype,len(arrays[0]),json.dumps(check_reference(out,references(arrays),
        1e-6 if dtype==np.float32 else 1e-12)),flush=True)


def test_structure_cuda_jax_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.vegetation_structure import run_structure_jax
    from noahmp_clm.gpu.cuda.vegetation_structure import run_structure_cuda
    arrays=cases(np.float32);cuda={k:cp.asnumpy(v) for k,v in run_structure_cuda(*arrays).items()}
    jax={k:np.asarray(v) for k,v in run_structure_jax(*arrays).items()};maximum=0.
    for key in FIELDS:
        np.testing.assert_allclose(cuda[key],jax[key],rtol=1e-6,atol=1e-6,err_msg=key)
        maximum=max(maximum,float(abs(cuda[key]-jax[key]).max()))
    print('native structure same precision',json.dumps(dict(rows=len(arrays[0]),maximum_absolute=maximum,
        cuda_reference=check_reference(cuda,references(arrays),1e-6))),flush=True)


def test_native_structure_rejects_invalid_contract():
    c,p,o=cases(np.float32);validate_structure(c,p,o,np)
    for change in ('negative','time','height','scheme','flag','dynamic_override','crop_flag','harvest_live'):
        cc,pp,oo=c.copy(),p.copy(),o.copy()
        if change=='negative':cc[0,0]=-1.
        if change=='time':cc[0,3]=86401.
        if change=='height':pp[0,0]=pp[0,1]
        if change=='scheme':oo[0,0]=4
        if change=='flag':oo[0,1]=2
        if change=='dynamic_override':oo[0,0]=1;oo[0,2]=1
        if change=='crop_flag':oo[0,3]=1
        if change=='harvest_live':oo[0,0]=3;oo[0,3]=1;cc[0,2]=1.
        with pytest.raises(ValueError):validate_structure(cc,pp,oo,np)

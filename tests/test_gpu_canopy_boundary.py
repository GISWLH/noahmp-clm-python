"""Native winter canopy boundary, independent equations and heat ledgers."""
import numpy as np
import pytest
import json
from pathlib import Path
from noahmp_clm.physics.canopy_energy import CanopyExchange,CanopyWaterStorage,solve_canopy_energy
from noahmp_clm.physics.soil_uptake import SoilWaterSupply
from noahmp_clm.physics.snowpack import Snowpack
from noahmp_clm.physics.photosynthesis import LeafParameters
from noahmp_clm.gpu.photosynthesis import pack_leaf_parameters
from noahmp_clm.gpu.canopy_boundary import pack_canopy_boundary,validate_boundary
from test_gpu_snicar import assert_science_close


def cases(dtype=np.float64,random_count=30):
    rows=[]
    for temp in (250.,265.,273.15,295.):
        for fv in (0.,.7,1.):
            for stored in (False,True):
                roots=np.array([0.,2e-5,2e-5,2e-5]);supply=SoilWaterSupply(roots,np.array([.6,.3,.1,0.]),0.)
                mass=.3 if stored and fv else 0.
                storage=CanopyWaterStorage(mass,mass*(334000.+4180.*(temp-273.15)),1800.,np.full(4,350000.)) if stored else None
                snow=Snowpack.from_ice([4.,12.,30.],min(temp,273.15)) if temp<=273.15 else None
                e=CanopyExchange(temp+2.,.001 if temp<274. else .014,90000.,.02,.004,.03,274.,15.)
                rows.append(pack_canopy_boundary(np.full(10,.3),500.,220.,e,fixed_ground_temperature=temp,
                    ground_is_ice=temp<=273.15,ground_shares_soil=snow is None,snowpack=snow,
                    canopy_storage=storage,soil_supply=supply,canopy_evaporation_limit=mass/1800.,
                    wet_fraction=.5,vegetated_fraction=fv,ground_evaporation_limit=1e-6,dtype=dtype))
    rng=np.random.default_rng(51903)
    for i in range(random_count):
        temp=rng.uniform(250.,302.);n=i%6
        area=rng.uniform(0.,.8,10) if i%7 else np.zeros(10)
        mass=rng.uniform(0.,1.) if area.sum() else 0.;dt=rng.uniform(300.,3600.)
        supply=SoilWaterSupply(rng.uniform(0.,5e-5,4),rng.uniform(0.,1.,4),1e-4)
        storage=CanopyWaterStorage(mass,mass*(334000.+4180.*rng.uniform(-15.,20.)),dt,rng.uniform(3e5,4.5e5,4))
        m=10.**rng.uniform(-3.,2.,n)
        snow=Snowpack(m,np.zeros(n),m/250.,rng.uniform(30.,1500.,n),m[:,None]*10.**rng.uniform(-9.,-5.,(n,5)))
        e=CanopyExchange(temp+rng.uniform(-3.,3.),rng.uniform(.0005,.009),rng.uniform(70000.,102000.),
            rng.uniform(.01,.03),rng.uniform(.002,.008),rng.uniform(.015,.05),280.,20.)
        rows.append(pack_canopy_boundary(area,rng.uniform(0.,900.),rng.uniform(220.,380.),e,
            fixed_ground_temperature=temp,ground_is_ice=temp<273.15,ground_shares_soil=n==0,
            snowpack=snow,canopy_storage=storage,soil_supply=supply,wet_fraction=rng.uniform(0.,1.),
            canopy_evaporation_limit=mass/dt,ground_evaporation_limit=rng.uniform(0.,3e-5),
            vegetated_fraction=rng.uniform(.1,1.),diffuse_fraction=rng.uniform(0.,1.),
            cosine_zenith=rng.uniform(.05,1.),dtype=dtype))
    saved=json.loads((Path(__file__).parent/'fixtures/canopy_thin_leaf_boundary.json').read_text())
    rows.append(tuple(np.asarray(v,np.int32 if k==7 else dtype) for k,v in enumerate(saved['arrays'])))
    return tuple(np.stack([row[k] for row in rows]) for k in range(8))


def reference_arguments(row):
    dtype=row[0].dtype
    area,u,f,soil,initial,b,c,n=[np.asarray(v,float) for v in row];n=int(n)
    pack=Snowpack(c[:n,0],np.zeros(n),c[:n,0]/250.,c[:n,1],c[:n,2:])
    supply=SoilWaterSupply(soil[1:5],soil[5:9],soil[9]) if soil[0] else None
    storage=CanopyWaterStorage(b[4],b[5],b[6],b[7:11]) if b[3] else None
    # A packed limit can round one ulp above the exact quotient. Reference
    # and backend use the same physical store; clamp only that input rounding.
    limit=min(u[15],b[4]/b[6]) if storage is not None else u[15]
    kw=dict(vegetated_fraction=u[13],wet_fraction=u[14],canopy_evaporation_limit=limit,
        transpiration_limit=u[16],ground_evaporation_limit=u[17],ground_emissivity=u[18],
        ground_albedo=u[19:21],diffuse_fraction=u[21],cosine_zenith=u[22],
        nitrogen_factor=f[:,0],water_stress=f[:,1],soil_supply=supply,
        initial_temperatures=initial,leaf_parameters=LeafParameters(*pack_leaf_parameters(dtype=dtype).astype(float)),
        fixed_ground_temperature=b[0],ground_is_ice=bool(b[1]),ground_shares_soil=bool(b[2]),
        snowpack=pack,canopy_storage=storage)
    return area,u[11],u[12],CanopyExchange(*u[:11]),kw


def numpy_reference(arrays):
    results=[]
    for row in zip(*arrays):
        a,s,l,e,kw=reference_arguments(row)
        # float32 freezing is a few microkelvin below the double constant.
        r=solve_canopy_energy(a,s,l,e,**kw)
        r['snow_absorbed_shortwave']=np.pad(r['snow_absorbed_shortwave'],(0,5-len(r['snow_absorbed_shortwave'])))
        results.append(r)
    return results


def check_science(actual,references,dtype):
    errors={}
    for key in references[0]:
        if key in ('gas_exchange','iterations','layer_residual','residual','canopy_humidity_change',
                   'canopy_water_residual','air_heat_residual','air_water_residual'):continue
        a=np.asarray(actual[key]);e=np.asarray([r[key] for r in references]);errors[key]=float(np.max(abs(a-e)))
        if 'temperature' in key:np.testing.assert_allclose(a,e,atol=.05 if dtype==np.float32 else 2e-5,rtol=0.)
        elif key in ('wet_evaporation','transpiration','ground_evaporation','root_uptake','canopy_humidity'):
            assert_science_close(a,e,absolute=1e-7 if dtype==np.float32 else 1e-10,zero_floor=1e-6)
        elif key=='canopy_water_after_kgm2':assert_science_close(a,e,absolute=1e-4,zero_floor=.001)
        else:assert_science_close(a,e,relative=.1 if dtype==np.float32 else 1e-5,
            absolute=.1 if dtype==np.float32 else .0005,zero_floor=1.)
    return errors


def check_ledgers(got,arrays):
    a,u,f,soil,initial,b,c,n=arrays
    enabled=b[:,3]>0.;uptake=np.asarray(got['root_uptake'],float)
    rootheat=np.sum(uptake*b[:,7:11].astype(float),axis=1)*enabled
    # Incoming root liquid heat offsets outgoing transpired liquid heat;
    # vaporization latent heat is a different term, never counted twice.
    energy=(u[:,11].astype(float)+u[:,12]-np.asarray(got['outgoing_shortwave'])-np.asarray(got['outgoing_longwave'])
        -np.asarray(got['h'])-np.asarray(got['le'])-np.asarray(got['ground_heat'])
        -np.asarray(got['canopy_storage_change_wm2'])-np.asarray(got['wet_advected_heat_wm2'])
        -np.asarray(got['transpiration_advected_heat_wm2'])+rootheat)
    water=(np.asarray(got['canopy_water_after_kgm2'],float)-b[:,4]
        +b[:,6].astype(float)*np.asarray(got['wet_evaporation'],float).sum(axis=1))[enabled]
    heat=(np.asarray(got['ground_heat'],float)-np.asarray(got['ground_nonshortwave_heat'])
        -np.asarray(got['snow_absorbed_shortwave'],float).sum(axis=1)-np.asarray(got['soil_absorbed_shortwave']))
    assert np.max(abs(energy))<(.06 if a.dtype==np.float32 else .0006)
    assert np.max(abs(water))<(2e-6 if a.dtype==np.float32 else 1e-12)
    assert np.max(abs(heat))<(.0005 if a.dtype==np.float32 else 1e-9)
    return dict(energy_wm2=float(np.max(abs(energy))),water_kgm2=float(np.max(abs(water))),shortwave_wm2=float(np.max(abs(heat))))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_boundary_jax_reference_and_ledgers(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.canopy_boundary import run_canopy_boundary_jax
    arrays=cases(dtype);got=run_canopy_boundary_jax(*arrays)
    print('boundary science',str(dtype),check_science(got,numpy_reference(arrays),dtype),flush=True)
    print('boundary ledgers',check_ledgers(got,arrays),flush=True)


def test_boundary_cuda_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.canopy_boundary import run_canopy_boundary_jax
    from noahmp_clm.gpu.cuda.canopy_boundary import run_canopy_boundary_cuda
    arrays=cases(np.float32);a=run_canopy_boundary_cuda(*arrays);b=run_canopy_boundary_jax(*arrays)
    for key in b:
        fields=b[key] if isinstance(b[key],dict) else [None]
        for field in fields:
            aa,bb=(a[key][field],b[key][field]) if field else (a[key],b[key])
            aa,bb=cp.asnumpy(aa),np.asarray(bb)
            print('boundary CUDA/JAX32',key,field,np.max(aa!=bb) if aa.dtype.kind in 'ib' else np.max(abs(aa-bb)),flush=True)
            if aa.dtype.kind in 'ib':np.testing.assert_array_equal(aa,bb)
            else:np.testing.assert_allclose(aa,bb,rtol=1e-6,atol=1e-6)


def test_boundary_input_contract():
    arrays=list(cases(np.float64,0));validate_boundary(arrays,np)
    for idx,value in ((0,190.),(1,2.),(4,-1.),(6,0.)):
        bad=[v.copy() for v in arrays];bad[5][0,idx]=value
        with pytest.raises(ValueError):validate_boundary(bad,np)
    bad=[v.copy() for v in arrays];bad[1][1,15]=1.
    with pytest.raises(ValueError):validate_boundary(bad,np)


def test_boundary_against_independent_equations():
    from test_canopy_energy import independent_energy_reference
    arrays=cases(np.float64,0)
    for i in (0,3,5,11,17,21,23):
        a,s,l,e,kw=reference_arguments(tuple(v[i] for v in arrays))
        actual=solve_canopy_energy(a,s,l,e,**kw)
        expected=independent_energy_reference(a,s,l,e,**kw)
        np.testing.assert_allclose(actual['leaf_temperature'],expected['temperature'][:10],atol=5e-5,rtol=0.)
        for key in ('h','le','ground_heat','gpp'):
            np.testing.assert_allclose(actual[key],expected[key],atol=.0003,rtol=1e-5)


def test_boundary_failure_and_empty_batch():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.canopy_boundary import run_canopy_boundary_jax
    from noahmp_clm.gpu.cuda.canopy_boundary import run_canopy_boundary_cuda
    arrays=tuple(v[5:6] for v in cases(np.float32,0))
    for run in (run_canopy_boundary_jax,run_canopy_boundary_cuda):
        got=run(*arrays,max_iterations=1,raise_on_failure=False)
        assert not bool(got['converged'][0])
        with pytest.raises(RuntimeError,match='did not converge'):run(*arrays,max_iterations=1)
        empty=run(*(v[:0] for v in arrays))
        assert empty['snow_absorbed_shortwave'].shape==(0,5)

"""Native crop events against NumPy, independent C/N and full annual cycles."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.physics.crop import CropState,CropPhase,CropParameters,step_crop
from noahmp_clm.physics.plant_cn import PlantState,PlantParameters
from noahmp_clm.gpu.crop import pack_crop,validate_crop
from test_gpu_plant_cn import parameters as plant_parameters
from test_gpu_snicar import assert_science_close
from test_gpu_clm_enthalpy_column import compare_same_precision,host


def cases(dtype=np.float64,count=512):
    rng=np.random.default_rng(51016);rows=[]
    for k in range(count):
        phase=CropPhase(k%4)
        p=CropParameters(base_temperature_k=rng.uniform(273.15,283.15),maturity_degree_days=rng.uniform(800.,2200.),
            emergence_fraction=rng.uniform(.01,.08),grain_fill_fraction=rng.uniform(.4,.7),maximum_age_days=rng.uniform(90.,250.),
            seed_carbon_gc_m2=rng.uniform(.1,5.),harvest_index=rng.uniform(.1,.8),maximum_lai=rng.uniform(3.,7.))
        c=rng.uniform(1.,1000.,3) if phase>=2 else np.zeros(3)
        b=PlantState(c,c/rng.uniform(20.,80.,3),rng.uniform(0.,20.) if phase>=2 else 0.,rng.uniform(0.,1.) if phase>=2 else 0.)
        seed=p.seed_carbon_gc_m2 if phase==1 else 0.
        s=CropState(rng.uniform(0.,1.2*p.maturity_degree_days),rng.uniform(0.,p.maturity_degree_days),
            rng.uniform(0.,1.1*p.maximum_age_days),phase,seed,seed/25.)
        rows.append(pack_crop(s,b,rng.uniform(260.,310.),rng.uniform(265.,305.),(1800.,3600.,86400.)[k%3],
            planting=phase==0 and k%8==0,force_harvest=phase>0 and k%7==0,vegetated_fraction=rng.uniform(.2,1.),parameters=p,dtype=dtype))
    return tuple(np.stack(v) for v in zip(*rows))


def reference(s,o,b,u,e,p,pp):
    s=s.astype(float);b=b.astype(float);u=u.astype(float)
    crop=CropState(*s[:3],int(o[0]),*s[3:]);plant=PlantState(b[:3],b[3:6],b[6],b[7])
    new,after,d=step_crop(crop,plant,*u[:3],planting=bool(e[0]),force_harvest=bool(e[1]),vegetated_fraction=float(u[3]),
        parameters=CropParameters(*p.astype(float)),plant_parameters=plant_parameters(pp))
    return dict(state=np.array([new.air_degree_days,new.soil_degree_days,new.age_days,new.seed_carbon,new.seed_nitrogen]),
        phase=np.array([new.phase],np.int32),plant_state=np.r_[after.carbon,after.nitrogen,after.reserve_carbon,after.reserve_nitrogen],**d)


def independent_budget(arrays,result):
    s=arrays[0].astype(float);b=arrays[2].astype(float);new=result['state'].astype(float);plant=result['plant_state'].astype(float)
    oldc=b[:,:3].sum(1)+b[:,6]+s[:,3];oldn=b[:,3:6].sum(1)+b[:,7]+s[:,4]
    newc=plant[:,:3].sum(1)+plant[:,6]+new[:,3];newn=plant[:,3:6].sum(1)+plant[:,7]+new[:,4]
    ci=result['seed_input_gc_m2'].astype(float);ni=result['seed_input_gn_m2'].astype(float)
    co=result['harvest_gc_m2'].astype(float)+result['litter_carbon_gc_m2'].astype(float).sum(1)
    no=result['harvest_gn_m2'].astype(float)+result['litter_nitrogen_gn_m2'].astype(float).sum(1)
    cr=newc-oldc+co-ci;nr=newn-oldn+no-ni;eps=np.finfo(result['state'].dtype).eps
    cb=128.*eps*np.maximum(newc+oldc+co+ci,1.);nb=128.*eps*np.maximum(newn+oldn+no+ni,1.)
    assert np.all(abs(cr)<=cb),(cr,cb)
    assert np.all(abs(nr)<=nb),(nr,nb)
    return dict(carbon=float(abs(cr).max()),nitrogen=float(abs(nr).max()),carbon_bound=float(cb.max()),nitrogen_bound=float(nb.max()))


def check_reference(arrays,result):
    maximum={}
    for k,row in enumerate(zip(*arrays)):
        ref=reference(*row)
        for key,expected in ref.items():
            if '_residual_' in key or '_bound_' in key:continue
            actual=result[key][k]
            if key in ('phase','planted','emerged','harvested','harvest_reason','growing'):
                np.testing.assert_array_equal(actual,expected,err_msg=key);continue
            error=float(np.max(abs(actual-expected)));maximum[key]=max(maximum.get(key,0.),error)
            if key=='plant_state':
                assert_science_close(actual[[0,1,2,6]],expected[[0,1,2,6]],relative=.1,zero_floor=.001,absolute=.0001)
                assert_science_close(actual[[3,4,5,7]],expected[[3,4,5,7]],relative=.1,zero_floor=.0001,absolute=.00001)
            elif key=='state':
                assert_science_close(actual[:3],expected[:3],relative=.1,zero_floor=.01,absolute=.001)
                assert_science_close(actual[3:],expected[3:],relative=.1,zero_floor=.001,absolute=.0001)
            else:assert_science_close(actual,expected,relative=.1,zero_floor=.001,absolute=.0001)
    return maximum


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_crop_device_reference_and_independent_budget(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.crop import run_crop_jax
    arrays=cases(dtype);result=host(run_crop_jax(*arrays))
    print('crop device reference',str(dtype),check_reference(arrays,result),flush=True)
    print('crop independent budget',str(dtype),independent_budget(arrays,result),flush=True)


def test_crop_same_precision_all_states_events_and_ledgers():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.crop import run_crop_jax
    from noahmp_clm.gpu.cuda.crop import run_crop_cuda
    arrays=cases(np.float32);a=host(run_crop_cuda(*arrays),cp.asnumpy);b=host(run_crop_jax(*arrays))
    print('crop CUDA/JAX all fields',compare_same_precision(a,b),flush=True)
    print('crop CUDA independent budget',independent_budget(arrays,a),flush=True)


def test_crop_continuous_annual_cycles_four_backends():
    cp=pytest.importorskip('cupy');jax=pytest.importorskip('jax')
    if not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.crop import run_crop_jax
    from noahmp_clm.gpu.cuda.crop import run_crop_cuda
    states={};net={};bounds={};worst={};event_days={};same=0.
    for name in ('numpy64','jax64','jax32','cuda32'):
        dtype=np.float64 if name.endswith('64') else np.float32
        rows=[pack_crop(CropState(),PlantState(np.zeros(3),np.zeros(3)),293.15,290.,86400.,planting=True,
            parameters=CropParameters(maturity_degree_days=900.+137.*k,emergence_fraction=.0625,grain_fill_fraction=.625),dtype=dtype) for k in range(8)]
        states[name]=tuple(np.stack(v) for v in zip(*rows));net[name]=np.zeros((8,2));bounds[name]=np.zeros(2);worst[name]=np.zeros(2)
        event_days[name]={'emerged':np.zeros(8,int),'harvested':np.zeros(8,int)}
    for day in range(180):
        results={}
        for name,arrays in states.items():
            s,o,b,u,e,p,pp=arrays;u=u.copy();e=np.zeros_like(e);e[:,0]=day==0
            u[:,0]=292.37+np.arange(8)*.37+3.*np.sin(day/18.)
            u[:,1]=289.13+np.arange(8)*.31+2.*np.sin(day/18.)
            a=(s,o,b,u,e,p,pp)
            if name=='numpy64':
                rows=[reference(*v) for v in zip(*a)];r={key:np.stack([v[key] for v in rows]) for key in rows[0]}
            elif name=='cuda32':r=host(run_crop_cuda(*a),cp.asnumpy)
            else:r=host(run_crop_jax(*a))
            results[name]=r;budget=independent_budget(a,r)
            net[name][:,0]+=r['seed_input_gc_m2'].astype(float)-r['harvest_gc_m2'].astype(float)-r['litter_carbon_gc_m2'].astype(float).sum(1)
            net[name][:,1]+=r['seed_input_gn_m2'].astype(float)-r['harvest_gn_m2'].astype(float)-r['litter_nitrogen_gn_m2'].astype(float).sum(1)
            bounds[name]+=np.array([budget['carbon_bound'],budget['nitrogen_bound']])
            plant=r['plant_state'].astype(float);crop=r['state'].astype(float)
            total=np.column_stack((plant[:,:3].sum(1)+plant[:,6]+crop[:,3],plant[:,3:6].sum(1)+plant[:,7]+crop[:,4]))
            residual=abs(total-net[name]);assert np.all(residual<=bounds[name]);worst[name]=np.maximum(worst[name],residual.max(0))
            for key in event_days[name]:event_days[name][key]=np.where(r[key],day+1,event_days[name][key])
            states[name]=(r['state'],r['phase'],r['plant_state'],u,e,p,pp)
        same=max(same,compare_same_precision(results['cuda32'],results['jax32']))
    for name in states:
        assert np.all(states[name][1]==0) and np.all(states[name][2]==0.) and np.all(states[name][0][:,3:]==0.)
        for key,days in event_days[name].items():
            assert np.all(days>0);np.testing.assert_array_equal(days,event_days['numpy64'][key])
    print('crop annual 8 columns x 180 days x 4 backends',dict(same_precision=same,
        cumulative_cn={k:v.tolist() for k,v in worst.items()},cumulative_bounds={k:v.tolist() for k,v in bounds.items()},
        events={k:{kk:vv.tolist() for kk,vv in v.items()} for k,v in event_days.items()}),flush=True)


def test_crop_native_contract():
    arrays=cases(np.float64,16);validate_crop(*arrays,np)
    bad=list(arrays);bad[1]=bad[1].astype(float)
    with pytest.raises(ValueError,match='integer'):validate_crop(*bad,np)
    bad=[v.copy() for v in arrays];bad[0][0,3]=1.
    with pytest.raises(ValueError):validate_crop(*bad,np)
    bad=[v.copy() for v in arrays];bad[2][0,6]=1.
    with pytest.raises(ValueError):validate_crop(*bad,np)
    bad=[v.copy() for v in arrays];bad[4][0,:]=1
    with pytest.raises(ValueError):validate_crop(*bad,np)

"""Native phase solve against independently validated NumPy physics."""
import numpy as np
import pytest
import json
from functools import lru_cache
from pathlib import Path
from test_canopy_phase_energy import cases
from test_gpu_snicar import assert_science_close
from noahmp_clm.gpu.canopy_phase_energy import pack_canopy_phase,run_canopy_phase_jax,validate_phase
from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy


def reference(row,*,exchange_extension=None,leaf_parameters=None,**solver_controls):
    from dataclasses import replace
    from noahmp_clm.physics.canopy_energy import CanopyExchange
    from noahmp_clm.physics.canopy_interception import CanopyCondensate
    from noahmp_clm.physics.soil_uptake import SoilWaterSupply
    from noahmp_clm.physics.snowpack import Snowpack
    from noahmp_clm.physics.photosynthesis import LeafParameters
    from noahmp_clm.gpu.photosynthesis import pack_leaf_parameters
    dtype=row[0].dtype
    a,u,f,soil,stem,c,b,snow,n=[np.asarray(v,float) for v in row];n=int(n)
    pack=Snowpack(snow[:n,0],np.zeros(n),snow[:n,0]/250.,snow[:n,1],snow[:n,2:]) if n else None
    supply=SoilWaterSupply(soil[1:5],soil[5:9],soil[9]) if soil[0]>0. else None
    kw=dict(stem_area=stem,soil_supply=supply,root_enthalpy=b[5:9] if supply is not None else None,vegetated_fraction=u[13],
        transpiration_limit=u[16],ground_evaporation_limit=u[17],ground_emissivity=u[18],
        ground_albedo=u[19:21],diffuse_fraction=u[21],cosine_zenith=u[22],
        nitrogen_factor=f[:,0],water_stress=f[:,1],fixed_ground_temperature=b[1] if b[0]>0. else None,
        ground_is_ice=bool(b[2]),ground_shares_soil=bool(b[3]),snowpack=pack,
        leaf_parameters=LeafParameters(*pack_leaf_parameters(dtype=dtype).astype(float)) if leaf_parameters is None else leaf_parameters)
    exchange=CanopyExchange(*u[:11])
    if exchange_extension is not None:
        exchange=replace(exchange,atmosphere_water_velocity=float(exchange_extension[0]),
            potential_temperature=float(exchange_extension[1]))
    return solve_canopy_phase_energy(a,u[11],u[12],exchange,CanopyCondensate(c[:,0],c[:,1]),b[4],**kw,**solver_controls)


def batch(dtype):
    rows=[pack_canopy_phase(a,sw,lw,e,c,1800.,dtype=dtype,**kw) for _,a,sw,lw,e,c,kw in cases()]
    saved=json.loads((Path(__file__).parent/'fixtures/canopy_phase_native_regressions.json').read_text())
    rows.extend(tuple(np.asarray(v,np.int32 if k==8 else dtype) for k,v in enumerate(row['arrays'])) for row in saved['cases'])
    for name in ('canopy_phase_humidity_bound.json','canopy_phase_humidity_slow.json'):
        saved=json.loads((Path(__file__).parent/'fixtures'/name).read_text())
        rows.append(tuple(np.asarray(v,np.int32 if k==8 else dtype) for k,v in enumerate(saved['arrays'])))
    return tuple(np.stack([r[k] for r in rows]) for k in range(9))


@pytest.mark.parametrize('fixture',['canopy_phase_humidity_bound.json','canopy_phase_humidity_slow.json'])
def test_humidity_bound_restart_closes_original_coupled_equations(fixture):
    saved=json.loads((Path(__file__).parent/'fixtures'/fixture).read_text())
    arrays=tuple(np.asarray(v,np.int32 if k==8 else np.float64) for k,v in enumerate(saved['arrays']))
    result=reference(arrays,tolerance=1e-7)
    assert result['restarts']>0 and 0.<result['canopy_humidity']<.19
    assert abs(result['layer_residual']).max()<=1.01e-7
    assert abs(result['solver_residual']).max()<=1e-7
    assert abs(result['canopy_water_residual'])<1e-12
    print('humidity restart',fixture,result['iterations'],result['restarts'],result['canopy_humidity'],
        abs(result['layer_residual']).max(),result['canopy_water_residual'],flush=True)


def host(value,convert=np.asarray):
    return {k:host(v,convert) if isinstance(v,dict) else convert(v) for k,v in value.items()}


@pytest.mark.parametrize('dtype',[np.float64,np.float32])
def test_phase_native_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype==np.float64 and not jax.config.x64_enabled:pytest.skip('x64 required')
    a=batch(dtype);got=host(run_canopy_phase_jax(*a,raise_on_failure=False));maximum={}
    print('phase convergence',dtype,got['converged'].tolist(),got['iterations'].tolist(),got['max_solver_residual'].tolist(),flush=True)
    assert np.all(got['converged'])
    for k,row in enumerate(zip(*a)):
        ref=reference(row)
        for key in ('leaf_temperature','ground_temperature','canopy_temperature','h','le','gpp','ground_heat'):
            error=float(np.max(abs(np.asarray(got[key][k],float)-ref[key])));maximum[key]=max(maximum.get(key,0.),error)
            if 'temperature' in key:np.testing.assert_allclose(got[key][k],ref[key],rtol=0.,atol=.05)
            else:assert_science_close(got[key][k],ref[key],relative=.1,zero_floor=.1 if key=='gpp' else 10.,absolute=.01 if key=='gpp' else 1.)
        assert_science_close(got['mass'][k],ref['condensate'].mass,relative=.1,zero_floor=.001,absolute=1e-4)
        assert_science_close(got['enthalpy'][k],ref['condensate'].enthalpy,relative=.1,zero_floor=100.,absolute=10.)
    mass=a[5][:,:,0].astype(float);heat=a[5][:,:,1].astype(float);dt=a[6][:,4].astype(float)
    water=got['mass'].astype(float)-mass+dt[:,None]*got['wet_evaporation'].astype(float)
    energy=(got['enthalpy'].astype(float)-heat+got['vapor_enthalpy'].astype(float))-(got['thermal_enthalpy'].astype(float)-heat)
    magnitude=abs(got['enthalpy'].astype(float))+abs(heat)+abs(got['vapor_enthalpy'].astype(float))+abs(got['thermal_enthalpy'].astype(float))
    assert np.all(abs(water)<=64.*np.finfo(dtype).eps*np.maximum(mass,1.)+1e-12)
    assert np.all(abs(energy)<=64.*np.finfo(dtype).eps*np.maximum(magnitude,1.)+1e-6)
    print('phase science',dtype,maximum,'parcel budgets',np.max(abs(water)),np.max(abs(energy)),flush=True)


def test_phase_input_contract():
    a=list(batch(np.float64));validate_phase(a,np)
    a[6]=a[6].copy();a[6][0,4]=0.
    with pytest.raises(ValueError):validate_phase(a,np)


def test_phase_cuda_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.cuda.canopy_phase_energy import run_canopy_phase_cuda
    arrays=batch(np.float32);ja=host(run_canopy_phase_jax(*arrays,raise_on_failure=False));cu=host(run_canopy_phase_cuda(*arrays,raise_on_failure=False),cp.asnumpy)
    print('CUDA phase convergence',cu['converged'].tolist(),cu['iterations'].tolist(),cu['max_solver_residual'].tolist(),flush=True)
    assert np.all(cu['converged']) and np.all(ja['converged'])
    maximum={};failures=[]
    def compare(a,b,path=''):
        for key in a:
            name=path+key
            if isinstance(a[key],dict):compare(a[key],b[key],name+'.');continue
            av,bv=a[key],b[key]
            maximum[name]=float(np.max(abs(av.astype(float)-bv.astype(float))))
            try:
                if av.dtype.kind in 'biu':np.testing.assert_array_equal(av,bv)
                else:np.testing.assert_allclose(av,bv,rtol=1e-6,atol=1e-6)
            except AssertionError as error:failures.append((name,str(error)))
    compare(cu,ja);print('phase same precision',maximum,flush=True)
    assert not failures,failures


@pytest.mark.parametrize('fixture',['qia_479_phase_jacobian.npz','phase_edge_float64_stencil.npz'],ids=['Qia479-supply','existing-phase-edge'])
def test_saved_float64_jacobian_closes_original_phase_equations(fixture):
    jax=pytest.importorskip('jax')
    if not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    import jax.numpy as j
    from noahmp_clm.gpu.canopy_phase_energy import phase_energy_kernel_jax,unpack_phase
    from noahmp_clm.gpu.snicar import _jax_tables
    with np.load(Path(__file__).parent/'fixtures'/fixture,allow_pickle=False) as saved:
        arrays=tuple(saved['a'+str(k)] for k in range(9));extension=saved['extension'];parameters=saved['leaf_parameters']
    expected=reference(arrays,exchange_extension=extension,tolerance=1e-7,max_iterations=80)
    ice,impurity=_jax_tables('float64');p=j.asarray(parameters);ext=j.asarray(extension)
    fn=jax.jit(lambda *x:phase_energy_kernel_jax(*x,p,ice,impurity,
        exchange_extension=ext,tolerance=1e-7,max_iterations=80))
    result=host(unpack_phase(np.asarray(fn(*map(j.asarray,arrays)))[None,:]))
    assert result['converged'][0] and result['max_solver_residual'][0]<=1e-7
    assert np.max(abs(expected['solver_residual']))<=1e-7
    maximum={}
    for key in ('leaf_temperature','ground_temperature','canopy_temperature','h','le','gpp','ground_heat'):
        actual=result[key][0];target=expected[key];maximum[key]=float(np.max(abs(actual-target)))
        if 'temperature' in key:np.testing.assert_allclose(actual,target,rtol=0.,atol=.05)
        else:assert_science_close(actual,target,relative=.1,zero_floor=.1 if key=='gpp' else 10.,absolute=.01 if key=='gpp' else 1.)
    mass=arrays[5][:,0];heat=arrays[5][:,1];dt=arrays[6][4]
    water=result['mass'][0]-mass+dt*result['wet_evaporation'][0]
    energy=result['enthalpy'][0]-heat+result['vapor_enthalpy'][0]-(result['thermal_enthalpy'][0]-heat)
    magnitude=abs(result['enthalpy'][0])+abs(heat)+abs(result['vapor_enthalpy'][0])+abs(result['thermal_enthalpy'][0])
    assert np.all(abs(water)<=64*np.finfo(float).eps*np.maximum(mass,1.)+1e-12)
    assert np.all(abs(energy)<=64*np.finfo(float).eps*np.maximum(magnitude,1.)+1e-6)
    print('saved float64 phase',fixture,json.dumps(dict(iterations=result['iterations'].tolist(),
        maximum_absolute_difference=maximum,max_solver_residual=result['max_solver_residual'].tolist(),
        water_residual_kgm2=float(abs(water).max()),parcel_heat_residual_jm2=float(abs(energy).max()))),flush=True)


def oldwet_case(precision,fixture='qia_735_phase_oldwet.json'):
    # Quantized references were computed on EHPC from actual rejected calls;
    # no NumPy forward solve is needed in this device regression.
    def decode(value):
        if isinstance(value,list):return [decode(x) for x in value]
        if not isinstance(value,dict):return value
        if '__array__' in value:
            array=np.asarray(decode(value['__array__']),dtype=value['dtype'])
            return array.reshape(value['shape']) if 'shape' in value else array
        if '__float__' in value:return float(value['__float__'])
        return {k:decode(v) for k,v in value.items()}
    saved=json.loads((Path(__file__).parent/'fixtures'/fixture).read_text())
    return decode(saved['quantized'][precision])


@lru_cache(maxsize=2)
def oldwet_jax_kernel(precision):
    import jax
    from noahmp_clm.gpu.canopy_phase_energy import phase_energy_kernel_jax
    from noahmp_clm.gpu.snicar import _jax_tables
    ice,impurity=_jax_tables('float'+precision)
    tolerance=.005 if precision=='32' else 1e-7;budget=100 if precision=='32' else 80
    # Match the native surface entry: traits and MOST extension are dynamic
    # device arguments, rather than constants captured by this test's JIT.
    return jax.jit(lambda *a:phase_energy_kernel_jax(*a[:9],a[9],ice,impurity,exchange_extension=a[10],
                                                  tolerance=tolerance,max_iterations=budget))


@pytest.mark.parametrize('precision',['32','64'])
@pytest.mark.parametrize('fixture',['qia_735_phase_oldwet.json','qia_1110_phase_kink.json'],ids=['existing735','exact1110'])
def test_saved_existing_water_phase_matches_ehpc_reference(precision,fixture):
    jax=pytest.importorskip('jax')
    if precision=='64' and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    import jax.numpy as j
    from noahmp_clm.gpu.canopy_phase_energy import unpack_phase
    case=oldwet_case(precision,fixture);arrays=case['arrays']
    raw=np.asarray(oldwet_jax_kernel(precision)(*map(j.asarray,arrays),j.asarray(case['parameters']),j.asarray(case['extension'])))[None,:]
    got=host(unpack_phase(raw));tolerance=.005 if precision=='32' else 1e-7
    assert got['converged'][0] and got['max_solver_residual'][0]<=tolerance
    differences={}
    for key,target in case['reference'].items():
        actual=got[key][0];differences[key]=float(np.max(abs(actual-target)))
        if 'temperature' in key:np.testing.assert_allclose(actual,target,rtol=0.,atol=.05)
        else:assert_science_close(actual,target,relative=.1,zero_floor=.1 if key=='gpp' else 10.,absolute=.01 if key=='gpp' else 1.)
    assert_science_close(got['mass'][0],case['reference_mass'],relative=.1,zero_floor=.001,absolute=1e-4)
    assert_science_close(got['enthalpy'][0],case['reference_enthalpy'],relative=.1,zero_floor=100.,absolute=10.)
    mass=arrays[5][:,0].astype(float);heat=arrays[5][:,1].astype(float);dt=float(arrays[6][4]);eps=np.finfo(arrays[0].dtype).eps
    water=got['mass'][0].astype(float)-mass+dt*got['wet_evaporation'][0].astype(float)
    thermal=got['thermal_enthalpy'][0].astype(float);advected=got['vapor_enthalpy'][0].astype(float);final=got['enthalpy'][0].astype(float)
    energy=(final-heat+advected)-(thermal-heat)
    assert np.all(abs(water)<=64.*eps*np.maximum(mass,1.)+1e-12)
    assert np.all(abs(energy)<=64.*eps*np.maximum(abs(final)+abs(heat)+abs(advected)+abs(thermal),1.)+1e-6)
    print('saved existing-water native reference',fixture,precision,dict(iterations=got['iterations'].tolist(),
        maximum_absolute_difference=differences,maximum_water_residual_kgm2=float(np.max(abs(water))),
        maximum_parcel_heat_residual_jm2=float(np.max(abs(energy)))) ,flush=True)


@pytest.mark.parametrize('fixture',['qia_735_phase_oldwet.json','qia_1110_phase_kink.json'],ids=['existing735','exact1110'])
def test_saved_existing_water_phase_cuda_all_raw_same_precision(fixture):
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    import jax.numpy as j
    from noahmp_clm.gpu.canopy_phase_energy import unpack_phase,validate_phase
    from noahmp_clm.gpu.canopy_boundary import radiation_inputs
    from noahmp_clm.gpu.cuda.canopy_snow_radiation import _kernel as radiation_kernel
    from noahmp_clm.gpu.cuda.snicar import _tables
    case=oldwet_case('32',fixture);arrays=tuple(cp.ascontiguousarray(cp.asarray(np.asarray(v)[None,...],
        dtype=cp.int32 if k==8 else cp.float32)) for k,v in enumerate(case['arrays']))
    validate_phase(arrays,cp);leaf,u,f,soil,stem,condensate,boundary,snow,count=arrays
    area=cp.ascontiguousarray(leaf+stem);radiation=cp.empty((1,584),cp.float32)
    radiation_kernel(True)((1,),(32,),(area,snow,count,cp.ascontiguousarray(radiation_inputs(area,u,cp)),*_tables(),radiation,np.int32(1)))
    folder=Path(__file__).resolve().parents[1]/'src/noahmp_clm/gpu/cuda'
    code='\n'.join((folder/name).read_text() for name in ('leaf_kernel.cu','radiation_kernel.cu','canopy_energy_kernel.cu','canopy_phase_energy_kernel.cu'))
    entry='''
extern "C" __global__ void saved_oldwet_phase(const float* a,const float* u,const float* f,const float* soil,const float* stem,const float* c,const float* b,const float* rad,const float* p,const float* extension,float* out){
    if(blockIdx.x*blockDim.x+threadIdx.x==0){PhaseContext v;phase_prepare(a,u,f,soil,stem,c,b,p,rad,v);v.water_velocity=extension[0];v.heat_temperature=extension[1];phase_solve(v,out,100,.005f);}
}
'''
    module=cp.RawModule(code=code+entry,options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'),name_expressions=('saved_oldwet_phase',))
    cuda=cp.empty((1,300),cp.float32)
    module.get_function('saved_oldwet_phase')((1,),(32,),(leaf,u,f,soil,stem,condensate,boundary,radiation,
            cp.asarray(case['parameters']),cp.asarray(case['extension']),cuda))
    actual=cp.asnumpy(cuda);expected=np.asarray(oldwet_jax_kernel('32')(*map(j.asarray,case['arrays']),
        j.asarray(case['parameters']),j.asarray(case['extension'])))[None,:]
    import os
    if os.environ.get('NOAHMP_QIA1110_DIAGNOSTICS'):
        folder=Path(os.environ['NOAHMP_QIA1110_DIAGNOSTICS']);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/(Path(fixture).stem+'-raw300.npz'),cuda=actual,jax=expected)
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
    a,b=unpack_phase(actual),unpack_phase(expected)
    assert np.all(a['converged']) and np.all(b['converged'])
    for key,value in a.items():
        if not isinstance(value,dict) and value.dtype.kind in 'biu':np.testing.assert_array_equal(value,b[key])
    print('saved existing-water all300raw sameprecision',fixture,float(np.max(abs(actual.astype(float)-expected.astype(float)))),flush=True)

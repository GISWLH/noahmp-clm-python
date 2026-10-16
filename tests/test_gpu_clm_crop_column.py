"""Crop management must feed the real native physical/BGC column on device."""
from dataclasses import replace
import numpy as np
import pytest
from test_clm_crop_column import case
from test_gpu_clm_bgc_column import reference
from test_gpu_clm_layered_column import budgets,science,host,compare_same_precision
from test_gpu_snicar import assert_science_close
from noahmp_clm.physics.crop import CropManagement
from noahmp_clm.gpu.clm_crop_column import pack_clm_crop_columns,pack_crop_management,next_crop_arrays


def cases(dtype):
    spec=[('fallow',285.),('sown',285.),('emergence',285.),('vegetative',285.),
        ('grain',285.),('mature',285.),('vegetative',270.),('cold_seed',270.)]
    states=[];forcing=[];events=[]
    for k,(kind,temperature) in enumerate(spec):
        state,f=case(kind,temperature)
        state=replace(state,options=replace(state.options,OptRainSnowPartition=5,OptRunoffSurface=6,OptRunoffSubsurface=6))
        states.append(state);forcing.append(list(vars(f).values()))
        events.append(CropManagement(planting=k==0,force_harvest=k==6))
    return pack_clm_crop_columns(states,dtype=dtype),np.asarray(forcing,dtype),pack_crop_management(events)


def cn_budgets(arrays,got,nitrogen):
    op=arrays[11].astype(float);os=arrays[13].astype(float);oc=arrays[16].astype(float)
    p=got['plant_state'].astype(float);s=got['soil_cn_state'].astype(float);c=got['crop_state'].astype(float)
    oldc=op[:,:3].sum(1)+op[:,6]+os[:,:7].sum(1)+oc[:,3]
    oldn=op[:,3:6].sum(1)+op[:,7]+os[:,7:].sum(1)+oc[:,4]
    newc=p[:,:3].sum(1)+p[:,6]+s[:,:7].sum(1)+c[:,3]
    newn=p[:,3:6].sum(1)+p[:,7]+s[:,7:].sum(1)+c[:,4]
    d=got['diagnostics'];crop=got['crop']
    gpp=d['gpp_gc_m2'].astype(float);ra=d['autotrophic_respiration_gc_m2'].astype(float);rh=d['heterotrophic_respiration_gc_m2'].astype(float)
    ec=crop['harvest_gc_m2'].astype(float);en=crop['harvest_gn_m2'].astype(float)
    ic=crop['seed_input_gc_m2'].astype(float);inn=crop['seed_input_gn_m2'].astype(float)
    cr=newc-oldc-gpp+ra+rh+ec-ic;nr=newn-oldn-nitrogen.astype(float)+en-inn
    eps=np.finfo(arrays[0].dtype).eps
    cb=128.*eps*np.maximum(newc+oldc+gpp+ra+rh+ec+ic,1.);nb=128.*eps*np.maximum(newn+oldn+nitrogen+en+inn,1.)
    assert np.all(abs(cr)<=cb),(cr,cb)
    assert np.all(abs(nr)<=nb),(nr,nb)
    return dict(carbon=float(abs(cr).max()),nitrogen=float(abs(nr).max()),carbon_bound_max=float(cb.max()),nitrogen_bound_max=float(nb.max()))


def check_science(arrays,f,dep,events,nitrogen,got,dtype):
    maximum={}
    for k,row in enumerate(zip(*arrays)):
        ref=reference(row[:16],f[k],dep[k],np.zeros(2),np.zeros(2,int),nitrogen[k],1.,crop_data=(*row[16:],events[k]))
        one={key:({a:b[k] for a,b in value.items()} if key=='diagnostics' else value[k]) for key,value in got.items()
            if key in ('snow_cells','soil_cells','options','storage','diagnostics','impurity_output_kgm2','condensate')}
        error=science(one,ref,dtype)
        for key in ('plant_state','soil_cn_state','crop_state'):
            value=got[key][k];expected=ref[key];error[key]=float(np.max(abs(value-expected)))
            if key=='plant_state':ci=[0,1,2,6];ni=[3,4,5,7]
            elif key=='soil_cn_state':ci=list(range(7));ni=list(range(7,15))
            else:
                ci=[3];ni=[4]
                assert_science_close(value[:3],expected[:3],relative=.1,zero_floor=.001,absolute=.0001)
            assert_science_close(value[ci],expected[ci],relative=.1,zero_floor=.001,absolute=.0001)
            assert_science_close(value[ni],expected[ni],relative=.1,zero_floor=.0001,absolute=.00001)
        for key in ('gpp_gc_m2','nee_gc_m2','autotrophic_respiration_gc_m2','heterotrophic_respiration_gc_m2'):
            assert_science_close(got['diagnostics'][key][k],ref['diagnostics'][key],relative=.1,zero_floor=.001,absolute=.0001)
            error[key]=float(abs(got['diagnostics'][key][k]-ref['diagnostics'][key]))
        for key in ('planted','emerged','harvested','harvest_reason','growing'):
            np.testing.assert_array_equal(got['crop'][key][k],ref['crop'][key])
        for key in ('seed_input_gc_m2','harvest_gc_m2','litter_carbon_gc_m2'):
            assert_science_close(got['crop'][key][k],ref['crop'][key],relative=.1,zero_floor=.001,absolute=.0001)
        for key in ('seed_input_gn_m2','harvest_gn_m2','litter_nitrogen_gn_m2'):
            assert_science_close(got['crop'][key][k],ref['crop'][key],relative=.1,zero_floor=.0001,absolute=.00001)
        np.testing.assert_array_equal(got['crop_phase'][k],ref['crop_phase'])
        for key in ('lai_used','lai_after','target_lai','nitrogen_factor_used','nitrogen_factor_after'):
            assert_science_close(got['biogeochemistry'][key][k],ref['biogeochemistry'][key],relative=.1,zero_floor=.001,absolute=.0001)
        for key,value in error.items():
            if np.ndim(value)==0:maximum[key]=max(maximum.get(key,0.),value)
    return maximum


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_crop_native_full_column_reference_and_independent_budgets(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_crop_column import step_clm_crop_columns_jax
    arrays,f,events=cases(dtype);dep=np.zeros((len(f),5),dtype);nitrogen=np.full(len(f),.00001,dtype)
    for step in range(2):
        out=step_clm_crop_columns_jax(*arrays,f,management=events,nitrogen_deposition_gn_m2=nitrogen,deposition=dep,return_surface=True)
        got=host(out)
        print('crop full native',str(dtype),step,check_science(arrays,f,dep,events,nitrogen,got,dtype),flush=True)
        print('crop full budgets',str(dtype),step,dict(**budgets(arrays[:8],f,dep,got),**cn_budgets(arrays,got,nitrogen)),flush=True)
        arrays=next_crop_arrays(arrays,got);events=np.zeros_like(events)


def test_crop_native_full_column_same_precision_and_device_state_carry():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_crop_column import step_clm_crop_columns_jax
    from noahmp_clm.gpu.cuda.clm_crop_column import step_clm_crop_columns_cuda
    arrays,f,events=cases(np.float32);a_arrays=b_arrays=arrays
    dep=np.zeros((len(f),5),np.float32);nitrogen=np.full(len(f),.00001,np.float32)
    for step in range(2):
        a=step_clm_crop_columns_cuda(*a_arrays,f,management=events,nitrogen_deposition_gn_m2=nitrogen,deposition=dep,return_surface=True)
        b=step_clm_crop_columns_jax(*b_arrays,f,management=events,nitrogen_deposition_gn_m2=nitrogen,deposition=dep,return_surface=True)
        ah=host(a,cp.asnumpy);bh=host(b)
        print('crop full same precision',step,compare_same_precision(ah,bh),flush=True)
        before=tuple(cp.asnumpy(v) if isinstance(v,cp.ndarray) else v for v in a_arrays)
        print('crop full CUDA budgets',step,dict(**budgets(before[:8],f,dep,ah),**cn_budgets(before,ah,nitrogen)),flush=True)
        a_arrays=next_crop_arrays(a_arrays,a);b_arrays=next_crop_arrays(b_arrays,b);events=np.zeros_like(events)
        assert all(isinstance(v,cp.ndarray) for k,v in enumerate(a_arrays) if k in (0,1,4,5,7,8,9,11,13,16,17))


def test_crop_extended_contract_preserves_seed_and_thermal_state():
    arrays,f,events=cases(np.float64)
    assert len(arrays)==19 and arrays[16].shape==(8,5) and arrays[17].dtype==np.int32
    assert arrays[16][1,3]==3. and arrays[17][1,0]==1
    np.testing.assert_array_equal(events[:,0],[1,0,0,0,0,0,0,0])
    np.testing.assert_array_equal(events[:,1],[0,0,0,0,0,0,1,0])
    with pytest.raises(ValueError,match='CropManagement'):pack_crop_management([1])
    state,_=case()
    state=replace(state,crop_state=None,crop_parameters=None,options=replace(state.options,OptCropModel=0))
    with pytest.raises(ValueError,match='OptCropModel'):pack_clm_crop_columns([state])

"""Tile outflow in native physical and fully managed C/N/CH4 columns."""
from dataclasses import replace
import numpy as np
import pytest
from test_clm_tile_column import case
from test_gpu_clm_layered_column import reference,budgets,science,host,compare_same_precision
from test_gpu_clm_wetland_column import check_reference as wetland_reference,independent_budgets as wetland_budgets,next_controls
from noahmp_clm.gpu.clm_tile_column import pack_clm_tile_columns,next_tile_batch
from noahmp_clm.gpu.clm_irrigation_column import pack_irrigation_inputs


def cases(kind,dtype):
    states=[];forcing=[];controls=[]
    for name in (['warm','dry','frozen'] if kind=='layered' else ['managed']):
        s,f,kw=case(name)
        if kind=='managed':s=replace(s,options=replace(s.options,OptRunoffSurface=6,OptRunoffSubsurface=6))
        states.append(s);forcing.append(list(vars(f).values()));controls.append(kw)
    kwargs={}
    if kind=='managed':
        kwargs['inputs'],kwargs['checks']=pack_irrigation_inputs([kw['irrigation_input'] for kw in controls],dtype=dtype)
        kwargs['management']=np.zeros((len(states),2),np.int32)
    return pack_clm_tile_columns(states,dtype=dtype),np.asarray(forcing,dtype),kwargs


def check(batch,f,kwargs,got,dtype):
    arrays=batch.arrays
    if batch.base_kind=='wetland':
        return wetland_reference(arrays,f,kwargs,got,dtype,tile_parameters=batch.parameters),wetland_budgets(arrays,f,kwargs,got)
    maximum={}
    for k,row in enumerate(zip(*arrays)):
        ref=reference(row,f[k],np.zeros(5),tile_parameters=batch.parameters[k])
        one={key:({a:b[k] for a,b in value.items()} if key=='diagnostics' else value[k]) for key,value in got.items()
            if key in ('snow_cells','soil_cells','options','storage','diagnostics','impurity_output_kgm2','condensate')}
        for key,value in science(one,ref,dtype).items():
            if np.ndim(value)==0:maximum[key]=max(maximum.get(key,0.),value)
    return maximum,budgets(arrays[:8],f,np.zeros((len(f),5)),got)


@pytest.mark.parametrize('kind',['layered','managed'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_native_tile_whole_reference_and_independent_budgets(kind,dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('x64 required')
    from noahmp_clm.gpu.clm_tile_column import step_clm_tile_columns_jax
    batch,f,kwargs=cases(kind,dtype)
    for step in range(2):
        controls=dict(kwargs,nitrogen_deposition_gn_m2=.0001) if kind=='managed' else kwargs
        got=host(step_clm_tile_columns_jax(batch,f,return_surface=True,**controls))
        errors,budget=check(batch,f,kwargs,got,dtype)
        print('tile whole reference',kind,dtype,step,errors,'budgets',budget,flush=True)
        batch=next_tile_batch(batch,got);kwargs=next_controls(kwargs)


@pytest.mark.parametrize('kind',['layered','managed'])
def test_native_tile_whole_same_precision_and_device_carry(kind):
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_tile_column import step_clm_tile_columns_jax
    from noahmp_clm.gpu.cuda.clm_tile_column import step_clm_tile_columns_cuda
    batch,f,kwargs=cases(kind,np.float32);a=b=batch
    for step in range(2):
        controls=dict(kwargs,nitrogen_deposition_gn_m2=.0001) if kind=='managed' else kwargs
        ja=step_clm_tile_columns_jax(a,f,return_surface=True,**controls);cu=step_clm_tile_columns_cuda(b,f,return_surface=True,**controls)
        ah=host(ja);bh=host(cu,cp.asnumpy)
        print('tile whole same precision',kind,step,compare_same_precision(ah,bh),flush=True)
        _,budget=check(batch,f,kwargs,bh,np.float32)
        print('tile whole CUDA budgets',kind,step,budget,flush=True)
        a=next_tile_batch(a,ja);b=next_tile_batch(b,cu)
        assert isinstance(b.arrays[1],cp.ndarray) and isinstance(b.parameters,cp.ndarray)
        batch=replace(b,arrays=tuple(cp.asnumpy(v) if isinstance(v,cp.ndarray) else np.asarray(v) for v in b.arrays),parameters=cp.asnumpy(b.parameters))
        kwargs=next_controls(kwargs)


def test_tagged_tile_pack_preserves_colliding_base_abis():
    from test_clm_wetland_column import case as wetland_case
    from test_clm_irrigation_column import case as irrigation_case
    wet,_,_=wetland_case();irrig,_,_=irrigation_case('bgc')
    a=pack_clm_tile_columns([wet.with_tile_drainage()]);b=pack_clm_tile_columns([irrig.with_tile_drainage()])
    assert len(a.arrays)==len(b.arrays)==18
    assert a.base_kind=='wetland' and b.base_kind=='irrigated'
    with pytest.raises(ValueError,match='one base'):pack_clm_tile_columns([wet.with_tile_drainage(),irrig.with_tile_drainage()])

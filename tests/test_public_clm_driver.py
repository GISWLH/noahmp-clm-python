"""Public trajectories retain actual extensive and managed physical stores."""
from copy import deepcopy
from dataclasses import asdict,is_dataclass,replace
import numpy as np
import pytest
from noahmp_clm import CLMColumnState,GlacierColumnState,ColumnState,Forcing,run_column,run_columns,step_column
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.clm_glacier import step_clm_glacier_column
from noahmp_clm.physics.glacier import GlacierIce
from noahmp_clm.physics.snowpack import Snowpack
from test_clm_wetland_column import case as wetland_case,budgets
from test_clm_tile_column import case as tile_case
from test_clm_glacier_column import independent as glacier_budget


def same(a,b):
    if is_dataclass(a):a,b=asdict(a),asdict(b)
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    else:assert a==b


@pytest.mark.parametrize('kind',['snow','seasonal','managed'])
def test_public_run_keeps_all_stores_and_stepwise_external_inputs(kind):
    old,f,options=tile_case('managed') if kind=='managed' else wetland_case(kind)
    assert isinstance(old,CLMColumnState)
    original=deepcopy(old);forcing=[f,replace(f,swdown=f.swdown+20.),f]
    inputs=[];manual=old;expected=[];maxima=np.zeros(4)
    for i,one in enumerate(forcing):
        kwargs=dict(options,co2_ppm=390.+10.*i,nitrogen_deposition_gn_m2=(i+1)*1e-5,deposition_rates=np.arange(1,6)*1e-12*(i+1))
        if i and 'irrigation_input' in kwargs:kwargs['irrigation_input']=replace(kwargs['irrigation_input'],check=False)
        inputs.append(kwargs);after,d=step_clm_enthalpy_column(manual,one,**kwargs)
        residual=budgets(manual,after,one,d,kwargs['nitrogen_deposition_gn_m2'])
        maxima=np.maximum(maxima,abs(np.array([residual[k] for k in ('water','heat','carbon','nitrogen')])))
        tracer=after.snow.impurities.sum(0)-manual.snow.impurities.sum(0)+d['impurity_output_kgm2']-kwargs['deposition_rates']*one.dt
        np.testing.assert_allclose(tracer,0.,atol=1e-18,rtol=0.)
        manual=after;expected.append(d)
    actual,final=run_column(old,iter(forcing),step_options=iter(inputs),co2_ppm=420.)
    same(actual,expected);same(final,manual);same(old,original)
    assert len(actual)==3 and final.canopy_condensate.mass.shape==(10,) and final.bgc_state is not None
    if kind=='managed':assert final.crop_state is not None and final.wetland_state is not None and final.irrigation_state is not None and final.tile_drainage_parameters is not None
    print('public CLM original/direct equality',kind,'M/H/C/N maxima',maxima.tolist(),flush=True)


def test_public_five_layer_glacier_trajectory_retains_heat_water_and_impurities():
    old=GlacierColumnState(GlacierIce.from_thickness([.1,.3,.6,9.],263.15),Snowpack.from_ice([2.,5.,11.,23.,30.],263.15,300.))
    original=deepcopy(old);f=Forcing(1800.,260.,.001,3.,80000.,150.,240.,2e-6)
    dep=np.arange(1,6)*1e-12;steps=[dict(deposition_rates=dep),dict(deposition_rates=dep*2.)]
    manual=old;expected=[];maximum=np.zeros(2)
    for inputs in steps:
        after,d=step_clm_glacier_column(manual,f,**inputs)
        maximum=np.maximum(maximum,abs(glacier_budget(manual,after,d,f,inputs['deposition_rates'])))
        manual=after;expected.append(d)
    actual,final=run_column(old,[f,f],step_options=steps)
    same(actual,expected);same(final,manual);same(old,original)
    assert len(old.snow.mass)==5 and 1<=len(final.snow.mass)<=5
    print('public glacier original/direct equality M/H maxima',maximum.tolist(),flush=True)


def test_public_batch_mixes_land_units_without_sharing_state_or_events():
    a,f,kw=wetland_case('snow');b=GlacierColumnState(GlacierIce.from_thickness([.1,1.],263.15))
    icef=Forcing(1800.,260.,.001,3.,80000.,150.,240.,0.)
    states=[a,b,ColumnState()];before=deepcopy(states);forcing=[[f],[icef,icef],[replace(icef,tair=285.,lwdown=320.)]]
    inputs=[[kw],[{},{}],[{}]]
    batch=run_columns(iter(states),iter(forcing),step_options_list=iter(inputs))
    solo=[run_column(s,fs,step_options=one) for s,fs,one in zip(states,forcing,inputs)]
    same(batch,solo);same(states,before)
    assert [len(v[0]) for v in batch]==[1,2,1]
    assert isinstance(batch[0][1],CLMColumnState) and isinstance(batch[1][1],GlacierColumnState) and isinstance(batch[2][1],ColumnState)


@pytest.mark.parametrize('fault',['columns_short','columns_long','options_columns','intervals_short','intervals_long','bad_mapping','second_column_short'])
def test_misaligned_columns_or_step_inputs_are_rejected_before_physics(monkeypatch,fault):
    import noahmp_clm.driver as driver
    called=[];monkeypatch.setattr(driver,'step_column',lambda *a,**k:called.append(a))
    states=[ColumnState(),ColumnState()];fs=[[object()],[object()]];kw=[[{}],[{}]]
    if fault=='columns_short':fs.pop()
    elif fault=='columns_long':fs.append([])
    elif fault=='options_columns':kw.pop()
    elif fault=='intervals_short':kw[0]=[]
    elif fault=='intervals_long':kw[0].append({})
    elif fault=='bad_mapping':kw[0][0]=None
    else:kw[1]=[]
    with pytest.raises((ValueError,TypeError)):driver.run_columns(states,fs,step_options_list=kw)
    assert called==[]


def test_wrong_state_or_keywords_cannot_silently_select_legacy_physics():
    f=Forcing(1800.,285.,.006,3.,101325.,350.,320.,0.)
    with pytest.raises(TypeError,match='Expected'):step_column(object(),f)
    with pytest.raises(TypeError,match='Physics keywords'):step_column(ColumnState(),f,co2_ppm=420.)
    s,_,_=wetland_case('snow')
    with pytest.raises(TypeError):step_column(s,f,unknown_physics_option=True)

"""Structure geometry must retain the actual managed M/H/C/N/CH4 ledgers."""
from dataclasses import replace
import json
import pytest
from noahmp_clm.driver import step_column
from noahmp_clm.physics.vegetation_structure import CanopyStructureParameters


def attach(state):
    crop=state.crop_state is not None
    return state.with_vegetation_structure(CanopyStructureParameters(3 if crop else 1,3.,.5,crop),sai=.8)


def check_structure(old,new,f,d):
    assert new.canopy_structure_parameters==old.canopy_structure_parameters
    used=d['vegetation_structure']['sai_used'] if old.bgc_state is not None else new.sai
    assert d['canopy_geometry']['total_sai']==used
    if old.crop_state is None:
        expected=max(.5,(1.-.5*f.dt/(30.*86400.))*old.sai+max(old.lai-new.lai,0.))
        assert new.sai==pytest.approx(expected,abs=1e-14)
    elif d['crop']['harvested']:
        assert new.sai==.25 and new.lai==0.


@pytest.mark.parametrize('kind',['wet','dry','frozen','snow','crop','harvest','irrigated','bgc_irrigated','seasonal'])
def test_stems_with_actual_wetland_crop_irrigation_and_nitrogen(kind):
    from test_clm_wetland_column import case,budgets
    state,f,kwargs=case(kind);state=attach(state)
    for _ in range(2):
        new,d=step_column(state,f,nitrogen_deposition_gn_m2=.0001,**kwargs)
        residual=budgets(state,new,f,d,.0001);check_structure(state,new,f,d)
        print('NumPy stem wetland',kind,json.dumps(dict(residual=residual,sai_before=state.sai,
            sai_used=d['vegetation_structure']['sai_used'],sai_after=new.sai,methane_stored=new.wetland_state.methane_gc_m2,
            methane_emitted=d['methane_emission_gc_m2'])),flush=True)
        state=new
        if 'crop_management' in kwargs:kwargs['crop_management']=None
        if 'irrigation_input' in kwargs:kwargs['irrigation_input']=replace(kwargs['irrigation_input'],check=False)


@pytest.mark.parametrize('kind',['plain','bgc','crop','snow','limited','frozen','harvest'])
def test_stems_with_actual_irrigation_request_and_external_heat(kind):
    from test_clm_irrigation_column import case,budgets
    from noahmp_clm.physics.crop import CropManagement
    state,f,request=case(kind);state=attach(state)
    event=CropManagement(force_harvest=True) if kind=='harvest' else None
    for _ in range(2):
        new,d=step_column(state,f,irrigation_input=request,crop_management=event)
        residual=budgets(state,new,f,d);check_structure(state,new,f,d)
        print('NumPy stem irrigation',kind,json.dumps(dict(residual=residual,sai_before=state.sai,
            sai_after=new.sai,applied=d['irrigation_mm'],external_heat=d['irrigation_enthalpy_jm2'])),flush=True)
        state=new;request=replace(request,check=False);event=None

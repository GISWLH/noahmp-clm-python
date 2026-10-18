"""Original failed station steps: material-event resolution and hard budgets."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.column import Forcing,SoilParams
from noahmp_clm.options import PhysicsOptions
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.snowpack import Snowpack
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.biogeochemistry import BGCState,BGCParameters
from noahmp_clm.physics.plant_cn import PlantState,PlantParameters
from noahmp_clm.physics.decomposition import DecompositionState,DecompositionParameters
from noahmp_clm.physics.phenology import PhenologyState,PhenologyParameters,PhenologyEnvironment
from noahmp_clm.physics.photosynthesis import LeafParameters
from noahmp_clm.physics.canopy_most import MOSTGeometry
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.validation import compare_series


def saved_case(index):
    path=Path(__file__).parent/'fixtures/numpy_annual_material_events.json'
    r=deepcopy(json.loads(path.read_text())['cases'][index]);s=r['state'];g=s['ground']
    g['params']=SoilParams(**g['params']);s['ground']=SoilEnthalpyState(**g)
    s['snow']=Snowpack(**s['snow']);s['options']=PhysicsOptions(**s['options'])
    roots=np.asarray(s['root_frac'],float);s['root_frac']=roots
    s['canopy_condensate']=CanopyCondensate(**s['canopy_condensate'])
    for key,cls in [('phenology_state',PhenologyState),('phenology_parameters',PhenologyParameters)]:
        if s[key] is not None:s[key]=cls(**s[key])
    b=s['bgc_state'];s['bgc_state']=BGCState(PlantState(**b['plant']),DecompositionState(**b['soil']))
    p=s['bgc_parameters'];s['bgc_parameters']=BGCParameters(PlantParameters(**p['plant']),DecompositionParameters(**p['decomposition']),p['maximum_lai'],p['saturated_potential_mpa'])
    state=CLMColumnState(**s);roots=roots.copy();roots.setflags(write=False);object.__setattr__(state,'root_frac',roots)
    kw=r['kwargs'];kw['leaf_parameters']=LeafParameters(**kw['leaf_parameters']);kw['most_geometry']=MOSTGeometry(**kw['most_geometry'])
    if kw['phenology_environment'] is not None:kw['phenology_environment']=PhenologyEnvironment(**kw['phenology_environment'])
    return state,Forcing(**r['forcing']),kw,r


@pytest.mark.parametrize('index',range(4),ids=['Cng2657','Qia479','Cha4037','HaM6456'])
def test_saved_station_step_resolves_material_or_ordinary_root_and_conserves(index):
    sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
    from run_fluxnet_column import audit_step
    before,forcing,kwargs,record=saved_case(index)
    after,d=step_clm_enthalpy_column(before,forcing,**kwargs)
    _,residual,bound=audit_step(before,after,forcing,d,kwargs['nitrogen_deposition_gn_m2'])
    assert d['surface_root_converged'] or d['surface_vapor_event']
    if d['surface_vapor_event']:
        assert not d['surface_root_converged'] and abs(d['surface_root_residual_k'])>2e-7
        left,right=d['surface_root_bracket_k'];fl,fr=d['surface_root_bracket_residual_k']
        assert right-left<=d['surface_temperature_resolution_k'] and fl>=0. and fr<=0.
        assert len(set(d['surface_root_bracket_snow_layers']))==2
    else:assert abs(d['surface_root_residual_k'])<=2e-7
    assert compare_series([d['h']],[record['native_cuda_h_wm2']],'h_wm2')['bias_pass']
    assert compare_series([d['le']],[record['native_cuda_le_wm2']],'le_wm2')['bias_pass']
    assert d['most']['iterations']<=50
    print('saved station event',json.dumps(dict(case=record['case'],h=d['h'],le=d['le'],
        material_event=d['surface_vapor_event'],ordinary_root=d['surface_root_converged'],
        root_residual_k=d['surface_root_residual_k'],residual_m_h_c_n=residual.tolist(),bound_m_h_c_n=bound.tolist())),flush=True)

"""Fully melted precipitation must not remain a snow thermal boundary."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from noahmp_clm.clm_state import CLMColumnState, clm_options
from noahmp_clm.column import ColumnState, SoilParams, Forcing
from noahmp_clm.options import PhysicsOptions
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState, infiltrate
from noahmp_clm.physics.snowpack import Snowpack, LF, CW
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.biogeochemistry import BGCState, BGCParameters
from noahmp_clm.physics.plant_cn import PlantState, PlantParameters
from noahmp_clm.physics.decomposition import DecompositionState, DecompositionParameters
from noahmp_clm.physics.phenology import PhenologyState, PhenologyParameters, PhenologyEnvironment
from noahmp_clm.physics.canopy_most import MOSTGeometry
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column


def ham_failure(fixture='ham_molten_snow.json'):
    record=json.loads((Path(__file__).parent/'fixtures'/fixture).read_text())
    s=record['state']; g=s['ground']; g['params']=SoilParams(**g['params'])
    s['ground']=SoilEnthalpyState(**g);s['snow']=Snowpack.empty()
    s['options']=PhysicsOptions(**s['options']);s['root_frac']=np.array(s['root_frac'])
    s['canopy_condensate']=CanopyCondensate(**s['canopy_condensate'])
    s['phenology_state']=PhenologyState(**s['phenology_state']);s['phenology_parameters']=PhenologyParameters(**s['phenology_parameters'])
    b=s['bgc_state'];s['bgc_state']=BGCState(PlantState(**b['plant']),DecompositionState(**b['soil']))
    p=s['bgc_parameters'];s['bgc_parameters']=BGCParameters(PlantParameters(**p['plant']),DecompositionParameters(**p['decomposition']),p['maximum_lai'],p['saturated_potential_mpa'])
    return CLMColumnState(**s),Forcing(**record['forcing']),dict(co2_ppm=record['co2'],cosine_zenith=record['mu'],diffuse_fraction=record['diffuse'],
        phenology_environment=PhenologyEnvironment(**record['environment']),most_geometry=MOSTGeometry(**record['most_geometry']))


def test_actual_warm_rain_failure_converges_and_closes_four_budgets():
    sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
    from run_fluxnet_column import audit_step
    state,forcing,kwargs=ham_failure()
    after,d=step_clm_enthalpy_column(state,forcing,**kwargs)
    _,residual,bound=audit_step(state,after,forcing,d,0.)
    assert d['most']['iterations']<=50 and d['most']['coefficient_residual']<=1e-4
    assert not len(after.snow.mass)
    print('HaM warm molten precipitation',json.dumps(dict(iterations=d['most']['iterations'],coefficient_residual=d['most']['coefficient_residual'],
        h=d['h'],le=d['le'],surface_temperature=d['surface_temperature'],step_residual_m_h_c_n=residual.tolist(),bound_m_h_c_n=bound.tolist())),flush=True)


@pytest.mark.parametrize('layered',[False,True])
@pytest.mark.parametrize('runoff',[3,6])
def test_liquid_pack_equals_its_explicit_soil_parcel_with_heat_and_tracers(layered,runoff):
    from noahmp_clm.physics.vic import infiltrate_vic
    opt=replace(clm_options(),OptRunoffSurface=runoff,OptRunoffSubsurface=runoff)
    old=ColumnState(ts=285.,tsoil=np.full(4,285.),smc=np.full(4,.4499),lai=1.,options=opt)
    state=CLMColumnState.from_column(old,options=opt)
    if layered:state=state.with_layered_canopy()
    mass=np.array([.2,1.]);heat=mass*(LF+CW*np.array([5.,2.]))
    tracers=np.array([[1e-9,2e-9,3e-9,4e-9,5e-9],[5e-9,4e-9,3e-9,2e-9,1e-9]])
    state=replace(state,snow=Snowpack(mass,heat,np.array([1e-5,2e-4]),impurities=tracers))
    admit=infiltrate_vic if runoff==6 else infiltrate
    earth,out,out_heat=admit(state.ground,float(mass.sum()),float(heat.sum()))
    expected_initial=replace(state,snow=Snowpack.empty(),ground=earth)
    f=Forcing(1800.,285.,.006,2.,101325.,0.,350.,0.)
    actual,diagnostic=step_clm_enthalpy_column(state,f)
    expected,reference=step_clm_enthalpy_column(expected_initial,f)
    np.testing.assert_allclose(actual.ground.water,expected.ground.water,rtol=0,atol=1e-12)
    np.testing.assert_allclose(actual.ground.enthalpy,expected.ground.enthalpy,rtol=0,atol=1e-6)
    assert diagnostic['runoff_mm']-reference['runoff_mm']==pytest.approx(out,abs=1e-12)
    assert diagnostic['runoff_enthalpy_jm2']-reference['runoff_enthalpy_jm2']==pytest.approx(out_heat,abs=1e-6)
    np.testing.assert_allclose(diagnostic['impurity_output_kgm2'],tracers.sum(0),rtol=1e-12,atol=1e-20)
    assert not len(actual.snow.mass)

"""Real microscopic wet-layer stiffness with independent reference and budgets."""
import numpy as np
import pytest
from phase_coordinate_recovery_cases import fixture,operands,fixed_operands
from test_canopy_phase_energy import independent_reference
from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy
from noahmp_clm.physics.canopy_column import surface_step


@pytest.mark.parametrize('seeded',[False,True])
def test_retained_stiff_phase_reference_and_all_parcels(seeded):
    v=fixture();args,kw=fixed_operands(v)
    got=solve_canopy_phase_energy(*args,**kw,tolerance=1e-7,max_iterations=80,initial_guess=v['stagnation_seed'].astype(float) if seeded else None)
    ref=independent_reference(*args,**kw)
    error=float(abs(np.r_[got['leaf_temperature'],got['ground_temperature']]-ref['temperature']).max());assert error<.05
    for key in ['h','le']:assert abs(got[key]-ref[key])<=1.
    assert abs(got['solver_residual']).max()<=1e-7 and abs(got['layer_residual']).max()<=1.05e-7
    np.testing.assert_allclose(got['condensate'].mass+args[-1]*got['wet_evaporation'],args[-2].mass,rtol=0.,atol=1e-14)
    np.testing.assert_allclose(got['condensate'].mass,ref['mass'],rtol=.1,atol=1e-7)
    np.testing.assert_allclose(got['condensate'].enthalpy,ref['enthalpy'],rtol=.1,atol=.1)
    print('3803 current NumPy independent',dict(seeded=seeded,maximum_temperature_error_k=error,iterations=got['iterations'],maximum_solver_residual_w_m2=float(abs(got['solver_residual']).max())),flush=True)


def test_retained_full_surface_actual_extensive_budgets():
    v=fixture();snow,soil,water,leaf,u,b,exchange,kw=operands(v)
    pack,ground,_,d=surface_step(snow,soil,b[0],leaf,u[11],u[12],exchange,canopy_condensate=water,root_weights=b[3:],**kw)
    assert abs(d['water_residual_kgm2'])<1e-10 and abs(d['energy_residual_jm2'])<.02
    oldm=snow.mass.sum()+soil.water.sum()+water.mass.sum();oldh=snow.enthalpy.sum()+soil.enthalpy.sum()+water.enthalpy.sum()
    newm=pack.mass.sum()+ground.water.sum()+d['condensate'].mass.sum();newh=pack.enthalpy.sum()+ground.enthalpy.sum()+d['condensate'].enthalpy.sum()
    mw=newm-oldm+d['et_kgm2']+d['runoff_kgm2']
    incoming=(u[11]+u[12]-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*b[0]
    mh=newh-oldh-incoming+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    bound=d['thermal']['convergence_tolerance_jm2']+b[0]*abs(d['layer_residual']).sum()+32.*np.finfo(float).eps*max(abs(oldh)+abs(newh)+abs(incoming)+abs(d['advected_enthalpy_jm2'])+abs(d['runoff_enthalpy_jm2']),1.)+1e-5
    assert abs(mw)<=32.*np.finfo(float).eps*max(oldm+newm+abs(d['et_kgm2'])+abs(d['runoff_kgm2']),1.) and abs(mh)<=bound
    np.testing.assert_allclose(ground.temperature,v['expected/surface/soil_temperature'],rtol=0.,atol=.05)
    print('3803 full NumPy surface budgets',dict(water_residual_kg_m2=float(mw),heat_residual_j_m2=float(mh),original_heat_bound_j_m2=float(bound)),flush=True)

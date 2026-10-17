"""Keep a continuous snow-supply branch before solving its temperature root."""
from dataclasses import replace
from pathlib import Path
import json,sys
import numpy as np
import pytest
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.validation import compare_series
from test_molten_precipitation import ham_failure

FIXTURE='ham_surface_branch.json'

def record():
    return json.loads((Path(__file__).with_name('fixtures')/FIXTURE).read_text())

def test_actual_ham_continuous_surface_matches_independent_brent(monkeypatch):
    import noahmp_clm.physics.canopy_most as most
    state,forcing,kwargs=ham_failure(FIXTURE);r=record();seen={}
    class Captured(Exception):pass
    def fixed_boundary(solve_surface,*args,**kw):
        seen.update(solve_surface(CanopyExchange(**r['reference_boundary']))[-1])
        raise Captured
    monkeypatch.setattr(most,'solve_surface_most',fixed_boundary)
    with pytest.raises(Captured):step_clm_enthalpy_column(state,forcing,**kwargs)
    reference=r['reference']
    assert abs(seen['surface_temperature']-reference['scipy_root'])<=.5
    assert compare_series([seen['h']],[reference['scipy_h']],'h_wm2')['bias_pass']
    assert abs(seen['water_residual_kgm2'])<1e-10
    assert abs(seen['energy_residual_jm2'])<1e-5
    print('HaM fixed boundary',json.dumps(dict(temperature=seen['surface_temperature'],
        brent_temperature=reference['scipy_root'],h=seen['h'],brent_h=reference['scipy_h'],
        evaluations=seen['surface_evaluations'])),flush=True)

@pytest.mark.parametrize('duration_factor',[.5,1.,2.])
@pytest.mark.parametrize('precipitation_factor',[.5,1.,2.])
def test_ham_snow_event_whole_step_and_four_budgets(duration_factor,precipitation_factor):
    sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
    from run_fluxnet_column import audit_step
    state,forcing,kwargs=ham_failure(FIXTURE)
    forcing=replace(forcing,dt=forcing.dt*duration_factor,prcp=forcing.prcp*precipitation_factor)
    after,d=step_clm_enthalpy_column(state,forcing,**kwargs)
    _,residual,bound=audit_step(state,after,forcing,d,0.)
    assert d['most']['iterations']<=50 and d['most']['coefficient_residual']<=1e-4
    assert np.all(np.isfinite(after.ground.enthalpy))
    print('HaM snow supply event',json.dumps(dict(duration_factor=duration_factor,
        precipitation_factor=precipitation_factor,iterations=d['most']['iterations'],h=d['h'],le=d['le'],
        residual_m_h_c_n=residual.tolist(),bound_m_h_c_n=bound.tolist())),flush=True)

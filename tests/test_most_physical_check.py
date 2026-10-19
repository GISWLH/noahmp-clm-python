"""A near-neutral discontinuous map must be checked at its feedback target."""
from types import SimpleNamespace
import numpy as np
import pytest
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_most import MOSTGeometry,canopy_exchange,solve_surface_most


@pytest.mark.parametrize('heat,jump,accept,solves',[(.02,0.,True,51),(.02,2.,False,51),(2.,0.,False,50)])
def test_exhausted_most_validates_physics_across_target_branch(heat,jump,accept,solves):
    geometry=MOSTGeometry();wind=1.;neutral=canopy_exchange(0.,wind,geometry)['heat_velocity']
    exchange=CanopyExchange(280.,.003,101325.,.02,.003,.03,278.,10.)
    snow=SimpleNamespace(mass=np.zeros(0))
    soil=SimpleNamespace(temperature=np.full(4,278.),water=np.full(4,28.),dz=np.full(4,.1))
    calls=[]
    def solve(boundary):
        stable=boundary.atmosphere_velocity<neutral
        h=heat if stable else -heat
        temperature=280.+(jump if stable else 0.)
        calls.append((h,temperature))
        return snow,soil,0.,dict(h=h,le=0.,gpp=0.,et_kgm2=0.,runoff_kgm2=0.,
            leaf_temperature=np.full(10,temperature),ground_temperature=278.,
            canopy_temperature=280.,canopy_humidity=.003)
    if accept:
        result=solve_surface_most(solve,exchange,wind,geometry,time_step=1800.)
        assert result[-1]['most']['iterations']==50
        assert result[-1]['most']['coefficient_residual']>1e-4
        assert len(calls)==solves
        # The accepted stores/fluxes belong to the last original trial.
        assert result[-1]['h']==calls[-2][0]
        assert abs(calls[-1][0]-calls[-2][0])==pytest.approx(2.*heat)
    else:
        with pytest.raises(RuntimeError,match='MOST flux feedback did not converge'):
            solve_surface_most(solve,exchange,wind,geometry,time_step=1800.)
        # The large-heat case is rejected already by the conservative
        # coefficient bound; the temperature counterexample reaches the
        # target validation and must still be rejected there.
        assert len(calls)==solves


def test_short_most_budget_does_not_use_physical_fallback():
    geometry=MOSTGeometry();exchange=CanopyExchange(280.,.003,101325.,.02,.003,.03,278.,10.)
    calls=[]
    def solve(boundary):
        calls.append(boundary)
        return None,None,None,dict(h=-.02,et_kgm2=0.,leaf_temperature=np.full(10,280.))
    with pytest.raises(RuntimeError,match='MOST flux feedback did not converge in 2'):
        solve_surface_most(solve,exchange,1.,geometry,time_step=1800.,max_iterations=2)
    assert len(calls)==2

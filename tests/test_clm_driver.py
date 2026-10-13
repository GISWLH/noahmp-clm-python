import numpy as np
import pytest
from noahmp_clm.column import ColumnState, Forcing
from noahmp_clm.clm_driver import step_clm_column


@pytest.mark.parametrize("scenario",["day","night","rain","dew","dry","bare"])
def test_canopy_soil_column_water_latent_and_surface_energy(scenario):
    state = ColumnState(ts=298.,tsoil=np.full(4,295.))
    sw,qa,prcp,lw = 800.,.01,0.,350.
    if scenario in ("night","dew"):
        sw,lw = 0.,280.
    if scenario == "dew":
        qa = .019
    if scenario == "rain":
        prcp = 1e-6
    if scenario == "dry":
        state.smc[:] = state.soil.smcwlt
    if scenario == "bare":
        state.fveg = 0.
    before = state.copy()
    after,d = step_clm_column(state,Forcing(1800,298,qa,3,101325,sw,lw,prcp))
    assert abs(d["water_residual_m"]) < 1e-12
    assert abs(d["latent_water_residual_wm2"]) < 1e-9
    assert abs(d["residual"]) < 6e-4
    assert len(d["leaf_temperature"]) == 10
    np.testing.assert_array_equal(state.smc,before.smc)
    np.testing.assert_array_equal(state.tsoil,before.tsoil)
    assert state.canw == before.canw
    if scenario in ("night","dew","dry","bare"):
        assert d["gpp_umol_m2s"] == 0.
    else:
        assert d["gpp_umol_m2s"] > 0.
    if scenario == "dew":
        assert after.canw > before.canw and d["et_mm"] < 0.


def test_canopy_overflow_and_unsupported_winter_are_explicit():
    state = ColumnState(ts=298.,canw=.002,lai=1.,fveg=.5,tsoil=np.full(4,295.))
    after,d = step_clm_column(state,Forcing(1800.,298.,.019,3.,101325.,0.,280.,0.))
    assert after.canw <= .0005*after.lai*after.fveg
    assert abs(d["water_residual_m"]) < 1e-12
    with pytest.raises(NotImplementedError,match="five-layer snow"):
        step_clm_column(ColumnState(swe=.01),Forcing(1800.,270.,.002,3.,101325.,100.,280.,0.))

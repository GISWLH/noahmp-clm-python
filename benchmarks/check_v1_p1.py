"""Quantify P1 frozen-state sensitivity and melt-out energy destination."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from noahmp_clm import ColumnState, Forcing, default_soil_params
from noahmp_clm.driver import step_column
from noahmp_clm.physics.soil import step_soil_heat, heat_capacity
from noahmp_clm.physics.frozen_soil import apparent_heat_capacity


def main():
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / "tests/fixtures/k3_phase_causal.json").read_text())["heat"]
    probes = []
    for row in rows:
        state = ColumnState(soil=default_soil_params("sand"),
            smc=np.array(row["smc"]), tsoil=np.array(row["old_temperature"]))
        capacity = apparent_heat_capacity(state.smc, state.tsoil, state.soil)
        step_soil_heat(state, 1800., row["g"])
        probes.append(dict(column=row["column"], step=row["step"], input=row["input_backend"],
            capacity=capacity.tolist(), new_temperature=state.tsoil.tolist(),
            historical_new_temperature=row["new_temperature_float64"]))
    final, diag = step_column(ColumnState(swe=.00011, snow_t=273.15),
        Forcing(1800., 295., .004, 2., 1e5, 700., 350., 0.))
    expected = diag["g"] * 1800. - diag["melt_mm"] * 334000.
    soil_gain = float(np.sum(heat_capacity(final.smc, final.soil)*final.dz*(final.tsoil-285.)))
    print(json.dumps(dict(phase_probes=probes, melt_out=dict(
        input_heat_jm2=diag["g"]*1800., melt_latent_jm2=diag["melt_mm"]*334000.,
        soil_received_jm2=diag["snow_excess_heat_jm2"],
        soil_sensible_gain_jm2=soil_gain, soil_budget_error_jm2=soil_gain-expected,
        unrouted_jm2=expected-diag["snow_excess_heat_jm2"])), indent=2))


if __name__ == "__main__":
    main()

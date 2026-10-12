"""阶段 A：固定状态向量的位级 round-trip 测试。"""

from __future__ import annotations

import numpy as np

from noahmp_clm.column import ColumnState, Forcing, SoilParams
from noahmp_clm.gpu.pack import (
    FLOAT_SIZE,
    FORCING_SIZE,
    INT_SIZE,
    column_state_to_vec,
    forcing_to_vec,
    forcing_vec_to_forcing,
    vec_to_column_state,
)
from noahmp_clm.options import PhysicsOptions


def _random_state(rng: np.random.Generator, snow_layers: int) -> ColumnState:
    soil = SoilParams(
        name="random",
        smcmax=float(rng.uniform(0.38, 0.55)),
        smcwlt=float(rng.uniform(0.04, 0.25)),
        smcref=float(rng.uniform(0.16, 0.42)),
        satdk=float(10.0 ** rng.uniform(-6.2, -4.1)),
        bexp=float(rng.uniform(3.5, 10.0)),
        quartz=float(rng.uniform(0.1, 0.95)),
    )
    options = PhysicsOptions(
        OptRainSnowPartition=int(rng.choice([1, 2, 3, 5])),
        OptSoilWaterTranspiration=int(rng.choice([1, 2, 3])),
        OptGroundResistanceEvap=int(rng.choice([1, 2, 3, 4])),
        OptSurfaceDrag=int(rng.choice([1, 2])),
        OptStomataResistance=int(rng.choice([1, 2])),
        OptSnowAlbedo=int(rng.choice([1, 2])),
        OptSnowThermConduct=int(rng.choice([1, 2, 3, 4, 5])),
        OptSoilSupercoolWater=int(rng.choice([1, 2])),
        OptSoilPermeabilityFrozen=int(rng.choice([1, 2])),
        OptSnowCompaction=int(rng.choice([1, 2])),
        OptSnowCoverGround=int(rng.choice([1, 2])),
    )
    dz = np.array([0.1, 0.3, 0.6, 1.0], dtype=np.float64)
    swe_layers = rng.uniform(0.0, 0.08, snow_layers)
    return ColumnState(
        dz=dz,
        tsoil=rng.uniform(245.0, 305.0, 4),
        smc=rng.uniform(soil.smcwlt, soil.smcmax, 4),
        ts=float(rng.uniform(240.0, 310.0)),
        swe=float(swe_layers.sum()),
        snow_t=float(rng.uniform(245.0, 273.15)),
        snow_albedo=float(rng.uniform(0.45, 0.95)),
        snow_age_days=float(rng.uniform(0.0, 30.0)),
        canw=float(rng.uniform(0.0, 0.003)),
        lai=float(rng.uniform(0.0, 7.0)),
        fveg=float(rng.uniform(0.0, 1.0)),
        z0m=float(rng.uniform(0.005, 0.2)),
        albedo=float(rng.uniform(0.1, 0.4)),
        emiss=float(rng.uniform(0.9, 0.99)),
        rsmin=float(rng.uniform(60.0, 250.0)),
        soil=soil,
        root_frac=rng.dirichlet(np.ones(4)),
        options=options,
        terrain_index=float(rng.uniform(-2.0, 4.0)),
        groundwater_depth=float(rng.uniform(0.1, 8.0)),
        soil_ice=rng.uniform(0.0, 0.2, 4),
        snow_swe_layers=swe_layers,
        snow_t_layers=rng.uniform(245.0, 273.15, snow_layers),
        snow_density_layers=rng.uniform(100.0, 450.0, snow_layers),
    )


def test_state_pack_round_trip_is_bitwise_for_50_columns() -> None:
    rng = np.random.default_rng(20261012)
    for index in range(50):
        state = _random_state(rng, snow_layers=index % 3 + 1)
        fvec, ivec = column_state_to_vec(state)
        restored = vec_to_column_state(fvec, ivec, state.dz)

        assert fvec.shape == (FLOAT_SIZE,)
        assert ivec.shape == (INT_SIZE,)
        assert np.array_equal(restored.dz, state.dz)
        for name in ("tsoil", "smc", "soil_ice", "root_frac"):
            assert np.array_equal(getattr(restored, name), getattr(state, name))
        for name in (
            "snow_swe_layers",
            "snow_t_layers",
            "snow_density_layers",
        ):
            assert np.array_equal(getattr(restored, name), getattr(state, name))
        for name in (
            "ts",
            "swe",
            "snow_t",
            "snow_albedo",
            "snow_age_days",
            "canw",
            "lai",
            "fveg",
            "z0m",
            "albedo",
            "emiss",
            "rsmin",
            "terrain_index",
            "groundwater_depth",
        ):
            assert getattr(restored, name) == getattr(state, name)
        assert restored.soil.__dict__ | {"name": state.soil.name} == state.soil.__dict__
        assert restored.options == state.options


def test_forcing_pack_round_trip() -> None:
    forcing = Forcing(1800.0, 278.5, 0.004, 3.2, 101325.0, 420.0, 315.0, 2e-6)
    vector = forcing_to_vec(forcing)
    assert vector.shape == (FORCING_SIZE,)
    assert forcing_vec_to_forcing(vector) == forcing

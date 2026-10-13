"""J3: device option dispatch, strict float32 matrix and two runoff outlets.

The runoff component probe is essential: total runoff alone cannot distinguish
the routing selectors. It compiles the production device functions unchanged.
These checks do not modify or replace the separate NumPy64 H4 tolerance gate.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from noahmp_clm.column import ColumnState, default_soil_params
from noahmp_clm.gpu.cuda import loader
from noahmp_clm.gpu.pack import OPTION_FIELDS, column_state_to_vec, forcing_to_vec, vec_to_column_state
from noahmp_clm.options import PhysicsOptions
from noahmp_clm.physics.runoff import route_runoff

_path = Path(__file__).resolve().parents[1] / "benchmarks/check_gpu_options.py"
_spec = importlib.util.spec_from_file_location("cuda_option_scenarios", _path)
_scenarios = importlib.util.module_from_spec(_spec)
# dataclasses resolves annotations through the module registry.
import sys
sys.modules[_spec.name] = _scenarios
_spec.loader.exec_module(_scenarios)

VALUES = [(name, value) for name, values in loader.PHYSICS_OPTION_VALUES.items() for value in values]


@pytest.fixture(scope="module")
def gpu():
    required = os.environ.get("NOAHMP_REQUIRE_GPU") == "1"
    try:
        import cupy as cp
        import jax
        jax.config.update("jax_enable_x64", True)
        assert not cp.cuda.runtime.is_hip and cp.cuda.runtime.getDeviceCount() > 0
        assert jax.default_backend() == "gpu"
        assert all("cuda" in str(d.client.platform_version).lower() for d in jax.devices())
    except (ImportError, RuntimeError, AssertionError) as exc:
        if required:
            raise
        pytest.skip(f"Actual NVIDIA CUDA/JAX required: {exc}")
    return cp, jax


def test_cuda_option_inventory_has_35_values_and_preserves_reserved_abi():
    expected = {name: tuple(v for v in spec.values if v not in spec.unsupported)
                for name, spec in _scenarios.SPECS.items() if spec.category == "physics"}
    assert loader.PHYSICS_OPTION_VALUES == expected
    assert len(expected) == 12 and len(VALUES) == 35
    defaults = PhysicsOptions()
    for index, name in enumerate(OPTION_FIELDS):
        if name not in expected:
            assert loader._OPTION_RANGES[index] == (getattr(defaults, name),)


@pytest.mark.parametrize("name,value", [
    ("OptRainSnowPartition", 4), ("OptRainSnowPartition", 5), ("OptSnowAlbedo", 3),
    *[(name, value) for name in ("OptRunoffSurface", "OptRunoffSubsurface") for value in range(5, 9)],
    ("OptCanopyRadiationTransfer", 1), ("OptDynamicVeg", 1),
])
def test_cuda_rejects_unimplemented_selector_in_any_column(gpu, name, value):
    cp, _ = gpu
    _, integers = column_state_to_vec(ColumnState())
    options = cp.asarray(np.tile(integers[:, None], (1, 5)))
    options[OPTION_FIELDS.index(name), 3] = value
    with pytest.raises(ValueError, match=name):
        loader._check_options(options)


def _pack(cases):
    packed = [column_state_to_vec(state) for _, state, _ in cases]
    f = np.array([x[0] for x in packed], np.float32)
    i = np.array([x[1] for x in packed], np.int32)
    u = np.array([[forcing_to_vec(row) for row in forcing] for _, _, forcing in cases], np.float32)
    return f, i, u


def _strict_report(actual, expected, labels):
    failures = []
    for group, a, b in zip(("state", "integer", "diagnostics"), actual, expected):
        assert a.shape == b.shape
        if group == "integer":
            np.testing.assert_array_equal(a, b)
            continue
        assert a.dtype == b.dtype == np.float32
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        ratio = np.abs(a64 - b64) / (1e-6 + 1e-6 * np.abs(b64))
        bad = ~np.isfinite(ratio) | (ratio > 1)
        if np.any(bad):
            first = tuple(np.argwhere(bad)[0])
            index = np.unravel_index(np.argmax(np.where(np.isfinite(ratio), ratio, np.inf)), a.shape)
            failures.append({"group": group, "count": int(np.sum(bad)),
                             "first": [int(v) for v in first], "first_case": labels[first[0]],
                             "worst": [int(v) for v in index], "max_ratio": float(ratio[index]),
                             "cuda": float(a[index]), "jax32": float(b[index])})
    print(json.dumps({"cases": len(labels), "failures": failures}), flush=True)
    assert not failures, json.dumps(failures)


def _run_compare(gpu, cases):
    cp, jax = gpu
    from noahmp_clm.gpu.jax_core import run_columns_jax
    f, i, u = _pack(cases)
    expected = run_columns_jax(f, i, u)
    jax.block_until_ready(expected)
    actual = loader.run_columns_cuda(f, i, u)
    actual = tuple(cp.asnumpy(v) for v in actual)
    expected = tuple(np.asarray(v) for v in expected)
    _strict_report(actual, expected, [name for name, _, _ in cases])


@pytest.mark.parametrize("name,value", VALUES)
def test_cuda_implemented_option_eight_scenarios_24_steps(gpu, name, value):
    options = replace(PhysicsOptions(), **{name: value})
    _run_compare(gpu, _scenarios.scenarios(24, options))


def test_cuda_mixed_all_options_1024_columns_48_steps(gpu):
    base = _scenarios.scenarios(48, PhysicsOptions())
    cases = []
    for index in range(1024):
        name, state, forcing = base[index % len(base)]
        choices = {name: values[(index // (field % 4 + 1) + field) % len(values)]
                   for field, (name, values) in enumerate(loader.PHYSICS_OPTION_VALUES.items())}
        cases.append((f"{index}:{name}", replace(state, options=replace(state.options, **choices)), forcing))
    _run_compare(gpu, cases)


RUNOFF_PROBE = r'''
extern "C" __global__ void probe_option_runoff(
    const float* __restrict__ state, const float* __restrict__ incoming,
    const int* __restrict__ options, float* __restrict__ output, int n) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=n) return;
    float s[70]; for(int j=0;j<70;++j) s[j]=state[c*70+j];
    Runoff r=route_runoff(incoming[c],s,options[c*2],options[c*2+1]);
    output[c*2]=r.surface; output[c*2+1]=r.subsurface;
}
'''


def test_cuda_runoff_both_outlets_16_pairs(gpu):
    cp, jax = gpu
    import jax.numpy as jnp
    from noahmp_clm.gpu.jax_core import _route_runoff
    states, incoming, options, ordinary = [], [], [], []
    for texture in ("sand", "loam", "clay"):
        soil = default_soil_params(texture)
        for saturation in (0.15, 0.8, 1.0):
            for terrain, groundwater in ((-1.0, 3.0), (1.5, 0.1), (0.2, 0.8)):
                state = ColumnState(soil=soil, smc=soil.smcmax*np.array([saturation, 0.3, 0.7, 0.9]),
                                    terrain_index=terrain, groundwater_depth=groundwater)
                f, i = column_state_to_vec(state)
                f = f.astype(np.float32)
                quantized = vec_to_column_state(f, i, state.dz)
                for a in range(1, 5):
                    for b in range(1, 5):
                        for amount in (0., 0.003, 0.05, 0.3):
                            value = np.float32(amount)
                            states.append(f); incoming.append(value); options.append((a, b))
                            ordinary.append(route_runoff(float(value), quantized, a, b))
    f, u, o = np.array(states, np.float32), np.array(incoming, np.float32), np.array(options, np.int32)
    module = cp.RawModule(code=loader.KERNEL_SOURCE+RUNOFF_PROBE,
                          options=("--std=c++14", "--fmad=false", "--prec-div=false"),
                          name_expressions=("probe_option_runoff",))
    result = cp.empty((len(f), 2), cp.float32)
    module.get_function("probe_option_runoff")(((len(f)+127)//128,), (128,),
        (cp.asarray(f), cp.asarray(u), cp.asarray(o), result, np.int32(len(f))))
    actual = cp.asnumpy(result)

    def reference(s, value, opt):
        a, b = _route_runoff(value, s[4:8], s[39], s[41], s[37], s[38], opt[0], opt[1])
        return jnp.stack((a, b))

    expected = np.asarray(jax.jit(jax.vmap(reference))(f, u, o))
    for label, target in (("jax32", expected), ("numpy64", np.array(ordinary))):
        print(label, "two-outlet max_abs", float(np.max(np.abs(actual-target))), flush=True)
        np.testing.assert_allclose(actual, target, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.sum(axis=1), u, rtol=1e-6, atol=1e-6)
    # Distinct outlets are observed; a stub returning (0, incoming) cannot pass.
    assert np.any(actual[:, 0] > 0.) and np.any(actual[:, 1] > 0.)

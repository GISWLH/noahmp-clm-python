"""Preserve the exact 12-column Cng surface-root contraction regression.

The fixture contains the original failed2657 input and independent EHPC
MI1537 NumPy references, exported by MZ1545 without physical recomputation.
Both GPU backends must satisfy the original physical, root-convergence,
and same-precision gates; the old GPU false-position-only solver fails.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest


def fixture(folder=None):
    folder = Path(__file__).with_name('fixtures') if folder is None else Path(folder)
    meta = json.loads((folder / 'cng_surface_root_contraction.json').read_text())
    path = folder / 'cng_surface_root_contraction.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['fixture_sha256']
    assert meta['original_diagnostic_input_sha256'] == 'd40644c23c9b31feabd3032528157ef38b9bd54b694a9191862704ab1ac59ffb'
    assert meta['numpy_reference_pbs_job'] == '1537.manager'
    assert meta['numpy_reference_actual_exit'] == 0
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    assert set(data) == set(meta['original_input_arrays']) | set(meta['independent_reference_arrays'])
    assert len(meta['original_input_arrays']) == 23 and len(meta['independent_references']) == 12
    for key, contract in {**meta['original_input_arrays'], **meta['independent_reference_arrays']}.items():
        value = data[key]
        assert list(value.shape) == contract['shape'] and str(value.dtype) == contract['dtype']
        assert hashlib.sha256(value.tobytes()).hexdigest() == contract['raw_sha256']
    return data, meta


def flatten(value, prefix=''):
    for name, item in value.items():
        if isinstance(item, dict):
            yield from flatten(item, prefix + name + '/')
        else:
            yield prefix + name, np.asarray(item)


def test_actual_cng_bounded_surface_root_original_gates(monkeypatch):
    """The old GPU false-position-only loop must fail this physical input."""
    jax = pytest.importorskip('jax')
    pytest.importorskip('cupy')
    assert any(device.platform == 'gpu' for device in jax.devices())
    import noahmp_clm.clm_enthalpy_driver as cpu_driver
    import run_fluxnet_native as native
    import run_fluxnet_column as runner
    from noahmp_clm.validation import compare_series

    def forbidden(*args, **kwargs):
        raise AssertionError('GPU result must not call NumPy forward physics')

    monkeypatch.setattr(cpu_driver, 'step_clm_enthalpy_column', forbidden)
    data, meta = fixture()
    binding = meta['binding']
    arrays = tuple(data['a' + str(k)] for k in range(16))
    kwargs = dict(environment=data['calendar_values'], calendar=data['calendar_flags'],
                  nitrogen_deposition_gn_m2=data['nitrogen'], deposition=data['deposition'],
                  most_geometry=data['geometry'], leaf_parameters=runner.LeafParameters(**binding['leaf_parameters']),
                  return_surface=True)
    if 'canopy_optics' in data:
        kwargs['canopy_optics'] = data['canopy_optics']
    hydrology = runner.hydrology_configuration(binding['cases'][0]['configuration'])
    if hydrology is not None:
        kwargs['soil_hydrology'] = hydrology
    outputs = {}
    for backend in ['cuda32', 'jax32']:
        engine = native.Backend(backend)
        devices = tuple(engine.xp.asarray(v) for v in arrays)
        devices = engine.update_forcing(devices, data['environment'])
        before = tuple(engine.host(v) for v in devices)
        for actual, expected in zip(before, arrays):
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()
        result_device = engine.step(*devices, data['forcing'], **kwargs)
        result = engine.host(result_device)
        after_device = native.next_bgc_arrays(devices, result_device)
        after = tuple(engine.host(v) for v in after_device)
        _, residual, bound, tracer = native.audit_step(before, after, data['forcing'], data['deposition'], data['nitrogen'], result)
        surface = result['surface']
        assert np.all(surface['converged']) and np.all(surface['canopy_converged'])
        tolerance = 2 * np.finfo(np.float32).eps * 273.15
        assert np.isfinite(surface['surface_root_bracket_k']).all()
        width = np.diff(surface['surface_root_bracket_k'], axis=1)[:, 0]
        assert np.all(width >= 0.)
        assert np.isfinite(surface['surface_temperature_residual']).all()
        assert np.all((abs(surface['surface_temperature_residual']) <= tolerance) | (width <= tolerance))
        metrics = {}
        for key, variable in [('h', 'h_wm2'), ('le', 'le_wm2')]:
            actual, expected = result['diagnostics'][key], data['numpy/' + key]
            scores = [compare_series([float(a)], [float(e)], variable) for a, e in zip(actual, expected)]
            assert all(score['bias_pass'] for score in scores)
            metrics[key + '_maximum_absolute_difference'] = float(abs(actual - expected).max())
        # Backend.soil already returns host NumPy arrays for either backend.
        temperature, _ = engine.soil(after_device)
        np.testing.assert_allclose(temperature, data['numpy/soil_temperature'], rtol=0., atol=.05)
        flat = dict(flatten(result))
        flat.update({'before' + str(k): v for k, v in enumerate(before)})
        flat.update({'next' + str(k): v for k, v in enumerate(after)})
        assert len(flat) == 345
        outputs[backend] = flat
        print('Cng bounded root', backend, json.dumps(dict(metrics=metrics,
              maximum_original_residual_m_h_c_n=abs(residual).max(axis=0).tolist(),
              maximum_original_bound_m_h_c_n=bound.max(axis=0).tolist(),
              maximum_impurity_residual=float(abs(tracer).max()),
              bracket_width_k=width.tolist(), temperature_residual_k=surface['surface_temperature_residual'].tolist())), flush=True)
    assert outputs['cuda32'].keys() == outputs['jax32'].keys()
    maximum = {}
    for key, left in outputs['cuda32'].items():
        right = outputs['jax32'][key]
        assert left.dtype == right.dtype and left.shape == right.shape
        if left.dtype.kind in 'biu' or key.startswith('before'):
            assert left.tobytes() == right.tobytes(), key
        else:
            assert left.dtype == np.dtype('float32')
            assert np.isfinite(left).all() and np.isfinite(right).all(), key
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6, err_msg=key)
        maximum[key] = float(abs(left.astype(float) - right.astype(float)).max(initial=0.))
    print('Cng all345 same precision', json.dumps(maximum), flush=True)

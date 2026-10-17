"""Independent accounting and input identity for the native station runner."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
import run_fluxnet_native as runner


def closed_column():
    n = 2
    a = [np.zeros((n, 1)) for _ in range(16)]
    a[0] = np.zeros((n, 5, 9)); a[1] = np.zeros((n, 4, 3)); a[3] = np.ones((n, 9), int)
    a[5] = np.full((n, 3), 999.)  # Rounded aliases must not enter the ledger.
    a[7] = np.zeros((n, 10, 2)); a[11] = np.zeros((n, 8)); a[13] = np.zeros((n, 15))
    a[1][:, :, 0] = 10.; a[1][:, :, 1] = 100.
    a[7][:, 0, 0] = 2.; a[7][:, 0, 1] = -300.
    a[11][:, :3] = 1.; a[11][:, 3:6] = .1
    a[13][:, :7] = 2.; a[13][:, 7:] = .2
    f = np.zeros((n, 8)); f[:, 0] = 1800.; f[:, 1] = 280.; f[:, 4] = 101325.
    d = {k: np.zeros(n) for k in ('et_mm', 'runoff_mm', 'advected_enthalpy_jm2', 'runoff_enthalpy_jm2',
        'h', 'le', 'gpp_gc_m2', 'autotrophic_respiration_gc_m2', 'heterotrophic_respiration_gc_m2')}
    sf = {'outgoing_shortwave': np.zeros(n), 'outgoing_longwave': np.zeros(n),
          'thermal_layer_tolerance': np.zeros((n, 9)), 'layer_residual': np.zeros((n, 10))}
    out = dict(diagnostics=d, surface=sf, impurity_output_kgm2=np.zeros((n, 5)))
    return tuple(a), f, out


def test_authoritative_canopy_and_cn_stores():
    a, f, out = closed_column()
    np.testing.assert_allclose(runner.totals(a), [[42., 100., 17., 1.9]]*2)
    incoming, residual, bound, tracer = runner.audit_step(a, a, f, np.zeros((2, 5)), np.zeros(2), out)
    assert not np.any(incoming) and not np.any(residual) and not np.any(tracer)
    assert np.all(bound > 0)


@pytest.mark.parametrize('array,cell', [(1, (0, 0, 0)), (7, (0, 0, 1)),
                                       (11, (0, 6)), (13, (0, 14))])
def test_missing_water_heat_carbon_or_nitrogen_rejected(array, cell):
    a, f, out = closed_column(); after = [v.copy() for v in a]; after[array][cell] += 1.
    with pytest.raises(RuntimeError, match='M/H/C/N'):
        runner.audit_step(a, after, f, np.zeros((2, 5)), np.zeros(2), out)


def test_each_column_uses_own_bound_and_no_cross_column_cancellation():
    a, f, out = closed_column(); after = [v.copy() for v in a]
    # Opposing mass errors must not cancel across the batch. A huge reservoir
    # in the second column must not enlarge the first column's bound.
    a[1][1, 0, 0] = 1e20; after[1][1, 0, 0] = 1e20
    after[1][0, 0, 0] += .001
    with pytest.raises(RuntimeError, match='M/H/C/N'):
        runner.audit_step(a, after, f, np.zeros((2, 5)), np.zeros(2), out)


def test_impurity_loss_rejected():
    a, f, out = closed_column()
    with pytest.raises(RuntimeError, match='impurity'):
        runner.audit_step(a, a, f, np.full((2, 5), 1e-12), np.zeros(2), out)


def test_same_precision_checks_all_fields_and_flags():
    original = {'nested': {'value': np.array([2.], np.float32)}, 'phase': np.array([2])}
    assert runner.compare_same_precision(original, original) == 0.
    with pytest.raises(AssertionError):
        runner.compare_same_precision(original, {'nested': original['nested'], 'phase': np.array([3])})
    with pytest.raises(AssertionError):
        runner.compare_same_precision(original, {'nested': {'value': np.array([2.01])}, 'phase': original['phase']})
    with pytest.raises(ValueError, match='fields'):
        runner.compare_same_precision(original, {'nested': original['nested']})


def batch_fixture(tmp_path):
    data = tmp_path/'input.npz'
    np.savez(data, time_start=['200301010000', '200301010030'], time_end=['200301010030', '200301010100'],
        forcing=np.array([[1800., 280., .005, 2., 101325., 0., 250., 0.]]*2), environment=np.array([[380., .1, .5]]*2))
    meta = dict(prepared_npz_sha256=runner.sha(data), archive_name='FLX_CN-Qia_FLUXNET2015_FULLSET.zip',
                latitude=26.7414, longitude=115.0581, utc_offset_hours=8)
    config = dict(site='CN-Qia', latitude=26.7414, longitude=115.0581, utc_offset_hours=8,
                  aerosol_deposition_kg_m2_s=[0.]*5, nitrogen_deposition_gn_m2_s=0., leaf_parameters={})
    for name, value in [('meta.json', meta), ('config.json', config)]:
        (tmp_path/name).write_text(json.dumps(value), encoding='utf-8')
    row = dict(case='CN-Qia-a', input='input.npz', input_manifest='meta.json', configuration='config.json')
    path = tmp_path/'batch.json'; path.write_text(json.dumps({'cases': [row]}), encoding='utf-8')
    return path, row, config


def test_batch_retains_original_time_and_site_identity(tmp_path):
    path, _, _ = batch_fixture(tmp_path); loaded = runner.load_batch(path)
    assert len(loaded) == 1 and len(loaded[0]['stamps']) == 2
    assert loaded[0]['stamps'][0].isoformat() == '2003-01-01T00:00:00'


@pytest.mark.parametrize('change', ['site', 'geometry', 'duplicate', 'path', 'forcing'])
def test_batch_rejects_mismatched_or_ambiguous_inputs(tmp_path, change):
    path, row, config = batch_fixture(tmp_path)
    if change == 'site': config['site'] = 'CN-Cng'
    if change == 'geometry': config['latitude'] += 1.
    if change in ('site', 'geometry'):
        (tmp_path/'config.json').write_text(json.dumps(config), encoding='utf-8')
    if change == 'duplicate': path.write_text(json.dumps({'cases': [row, row]}), encoding='utf-8')
    if change == 'path':
        row['case'] = '../escape'; path.write_text(json.dumps({'cases': [row]}), encoding='utf-8')
    if change == 'forcing':
        with (tmp_path/'input.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError): runner.load_batch(path)

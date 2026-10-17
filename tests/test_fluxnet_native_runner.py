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


def committed_fixture(tmp_path, *, end='200301010100', start='200301010000'):
    """Small on-disk transaction; physical correctness is tested separately."""
    path, _, _ = batch_fixture(tmp_path)
    cases = runner.load_batch(path)
    c = cases[0]
    identity = dict(backend='cuda32', compare_jax32=False, source_sha256={'source': 'unchanged'},
        cases=[dict(case=c['case'], configuration_sha256=runner.sha(c['paths']['configuration']),
            forcing_npz_sha256=runner.sha(c['paths']['input']),
            forcing_manifest_sha256=runner.sha(c['paths']['input_manifest']))])
    folder = tmp_path/'previous'; cp = folder/c['case']; cp.mkdir(parents=True)
    arrays, _, _ = closed_column()
    state = {f'a{k}': v[:1].copy() for k, v in enumerate(arrays)}
    state.update(index=np.array(2), initial=runner.totals(tuple(state[f'a{k}'] for k in range(16))),
        net=np.ones((1, 4)), bounds=np.full((1, 4), 2.), maximum=np.full((1, 4), .1),
        stepmax=np.full((1, 4), .01), tracermax=np.zeros((1, 5)))
    runner.write_npz(folder/'state.npz', **state)
    runner.write_npz(cp/'trace.npz', time_start=np.array([start, '200301010030']),
                     time_end=np.array(['200301010030', end]), h_wm2=np.array([1., 2.]))
    r = dict(steps_completed=2, model_npz='trace.npz', model_npz_sha256=runner.sha(cp/'trace.npz'),
             batch_checkpoint_sha256=runner.sha(folder/'state.npz'))
    runner.atomic_json(cp/'report-generation.json', r)
    runner.atomic_json(cp/'report.json', r)
    report = dict(status='complete', steps_completed=2, steps_requested=2, identity=identity,
        checkpoint_file='state.npz', checkpoint_sha256=runner.sha(folder/'state.npz'),
        case_reports={c['case']:dict(file='report-generation.json', sha256=runner.sha(cp/'report-generation.json'))})
    runner.atomic_json(folder/'batch.json', report)
    return folder, cases, identity, state


def test_committed_generation_survives_stale_or_partial_alias(tmp_path):
    folder, cases, _, expected = committed_fixture(tmp_path)
    (folder/cases[0]['case']/'report.json').write_text('{interrupted alias', encoding='utf-8')
    _, saved, rows = runner.read_generation(folder)
    for key in expected: np.testing.assert_array_equal(saved[key], expected[key])
    np.testing.assert_array_equal(rows[0][1]['h_wm2'], [1., 2.])
    # Incomplete next-generation files do not affect the committed pointer.
    (folder/'state-orphan.npz').write_bytes(b'incomplete')
    assert runner.read_generation(folder)[0]['steps_completed'] == 2


@pytest.mark.parametrize('target', ['state.npz', 'CN-Qia-a/report-generation.json', 'CN-Qia-a/trace.npz'])
def test_committed_generation_rejects_corruption(tmp_path, target):
    folder, _, _, _ = committed_fixture(tmp_path)
    with (folder/target).open('ab') as stream: stream.write(b'corrupt')
    with pytest.raises(ValueError, match='hash'): runner.read_generation(folder)


def test_continuation_retains_every_array_and_original_ledger(tmp_path):
    folder, cases, identity, expected = committed_fixture(tmp_path)
    cases[0]['data']['time_start'][0] = '200301010100'
    link, _, saved = runner.carry_generation(folder, 'continue', cases, identity)
    for key in expected: np.testing.assert_array_equal(saved[key], expected[key])
    assert link['previous_ledger_steps'] == 2 and link['previous_segments'] == 1
    assert link['batch_report_sha256'] == runner.sha(folder/'batch.json')


@pytest.mark.parametrize('change', ['gap', 'configuration', 'order', 'backend', 'shadow', 'source', 'incomplete'])
def test_continuation_rejects_identity_time_or_completion_changes(tmp_path, change):
    folder, cases, identity, _ = committed_fixture(tmp_path)
    cases[0]['data']['time_start'][0] = '200301010100'
    if change == 'gap': cases[0]['data']['time_start'][0] = '200301010130'
    elif change == 'configuration': identity['cases'][0]['configuration_sha256'] = 'different'
    elif change == 'order': identity['cases'][0]['case'] = 'different'
    elif change == 'backend': identity['backend'] = 'jax64'
    elif change == 'shadow': identity['compare_jax32'] = True
    elif change == 'source': identity['source_sha256'] = {'source':'changed'}
    else:
        r = json.loads((folder/'batch.json').read_text()); r['status'] = 'running'
        runner.atomic_json(folder/'batch.json', r)
    with pytest.raises(ValueError): runner.carry_generation(folder, 'continue', cases, identity)


def test_cycle_requires_same_forcing_and_whole_calendar_block(tmp_path):
    folder, cases, identity, _ = committed_fixture(tmp_path, end='200401010000')
    assert runner.carry_generation(folder, 'cycle', cases, identity)[0]['mode'] == 'cycle'
    identity['cases'][0]['forcing_npz_sha256'] = 'changed'
    with pytest.raises(ValueError, match='identical forcing'):
        runner.carry_generation(folder, 'cycle', cases, identity)


def test_week_repetition_cannot_masquerade_as_climate_cycle(tmp_path):
    folder, cases, identity, _ = committed_fixture(tmp_path)
    with pytest.raises(ValueError, match='calendar years'):
        runner.carry_generation(folder, 'cycle', cases, identity)


@pytest.mark.parametrize('monthly', [False, True])
def test_segmented_run_matches_continuous_state_trace_and_ledger(tmp_path, monkeypatch, monthly):
    """Exercise runner transactions with an exactly conserved test tendency.

    This is an orchestration test, not an alternate physical backend or a
    scientific comparison. The real CUDA split trajectory is run separately.
    """
    from test_clm_bgc_column import case
    state, _, _ = case(None)
    monkeypatch.setattr(runner, 'build_state', lambda config: state)

    class LedgerBackend:
        xp = np
        def __init__(self, name): pass
        def host(self, value): return value
        def update_forcing(self, arrays, env):
            a = list(arrays); a[4] = a[4].copy(); a[4][:, 7:10] = env
            return tuple(a)
        def step(self, *args, **kwargs):
            a, forcing = args[:16], args[16]; n = len(forcing)
            fields = {0:'snow_cells', 1:'soil_cells', 3:'options', 4:'site', 5:'storage',
                      7:'condensate', 8:'phenology_state', 9:'phenology_flags',
                      11:'plant_state', 13:'soil_cn_state'}
            out = {name:a[k].copy() for k, name in fields.items()}
            out['plant_state'][:, 6] += .125
            out['plant_state'][:, 7] += kwargs['nitrogen_deposition_gn_m2']
            d = {key:np.zeros(n) for key in (*runner.TRACE.values(),
                 'advected_enthalpy_jm2', 'runoff_enthalpy_jm2')}
            d['gpp_gc_m2'][:] = .125; d['nee_gc_m2'][:] = -.125
            out.update(diagnostics=d, impurity_output_kgm2=kwargs['deposition']*forcing[:, :1], surface=dict(
                outgoing_shortwave=forcing[:, 5], outgoing_longwave=forcing[:, 6],
                thermal_layer_tolerance=np.zeros((n, 9)), layer_residual=np.zeros((n, 10))))
            return out
        def soil(self, arrays):
            return np.full((len(arrays[0]), 4), 285.), arrays[1][:, :, 0]

    monkeypatch.setattr(runner, 'Backend', LedgerBackend)
    path, row, config = batch_fixture(tmp_path)
    config.update(scope='test instrument only', most_geometry={}, leaf_dimension_m=.04, phenology=None)
    if monthly:
        del config['aerosol_deposition_kg_m2_s']
        config['aerosol_deposition_monthly_kg_m2_s'] = (np.arange(1, 13)[:, None]*np.arange(1, 6)*1e-12).tolist()
        config['nitrogen_deposition_gn_m2_s'] = 1e-7
    runner.atomic_json(tmp_path/'config.json', config)
    with np.load(tmp_path/'input.npz') as z: original = dict(z)
    for name, starts, ends in [
        ('all', ['200301010000','200301010030','200301010100','200301010130'],
                ['200301010030','200301010100','200301010130','200301010200']),
        ('first', ['200301010000','200301010030'], ['200301010030','200301010100']),
        ('second', ['200301010100','200301010130'], ['200301010130','200301010200'])]:
        forcing = np.repeat(original['forcing'][:1], len(starts), axis=0)
        env = np.repeat(original['environment'][:1], len(starts), axis=0)
        data = tmp_path/(name+'.npz')
        runner.write_npz(data, forcing=forcing, environment=env, time_start=np.array(starts), time_end=np.array(ends))
        meta = json.loads((tmp_path/'meta.json').read_text()); meta['prepared_npz_sha256'] = runner.sha(data)
        runner.atomic_json(tmp_path/(name+'-meta.json'), meta)
        runner.atomic_json(tmp_path/(name+'-batch.json'), {'cases':[dict(row,
            input=data.name, input_manifest=name+'-meta.json')]})
    runner.run(tmp_path/'all-batch.json', tmp_path/'continuous')
    runner.run(tmp_path/'first-batch.json', tmp_path/'first')
    runner.run(tmp_path/'second-batch.json', tmp_path/'second', initial_run=tmp_path/'first',
               initial_mode='continue', stop_after=1)
    # A stale/partial compatibility alias cannot prevent exact resumption.
    (tmp_path/'second/CN-Qia-a/report.json').write_text('{broken alias')
    runner.run(tmp_path/'second-batch.json', tmp_path/'second', initial_run=tmp_path/'first',
               initial_mode='continue', resume=True)
    full, expected, full_rows = runner.read_generation(tmp_path/'continuous')
    end, actual, end_rows = runner.read_generation(tmp_path/'second')
    _, _, first_rows = runner.read_generation(tmp_path/'first')
    assert full['ledger_steps'] == end['ledger_steps'] == 4 and end['ledger_segments'] == 2
    assert end['steps_completed'] == 2 and end['status'] == 'complete'
    for key in expected:
        if key != 'index': np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    for key, values in full_rows[0][1].items():
        np.testing.assert_array_equal(np.concatenate((first_rows[0][1][key], end_rows[0][1][key])), values)
    assert actual['net'][0, 2] == .5
    if monthly:
        assert actual['net'][0, 3] == pytest.approx(4*1800*1e-7)

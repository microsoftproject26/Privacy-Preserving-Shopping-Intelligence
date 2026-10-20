"""Admission, numerical, accounting and CALIB personal-evaluator controls."""
from __future__ import annotations

import copy
import json
import math
import sys
from types import SimpleNamespace

import ppsi_env  # noqa: F401 — activate the inherited isolation before kernels
import pytest
import torch

from experiment_extensions.federated import localpasses as L
from fl.client import LocalSolver, ItemLRSolver, PFConfig, client_update
from fl.numerics import enforce_strict_fp32, state_digest
import ppsi_paths

sys.path.insert(0, str(ppsi_paths.code_dir('tests_fl')))   # the FL test kit (03_federated_learning/tests/research)
from fl_testkit import tiny, one_client  # noqa: E402


@pytest.fixture(autouse=True)
def strict():
    torch.set_num_threads(1)
    enforce_strict_fp32()


@pytest.fixture
def cold():
    return L.select_cell('cold_FA_eta0.001_P2_E6_s2026')


@pytest.fixture
def warm():
    return L.select_cell('pb100_FA_eta3e-05_P4_E6_s2026')


def identity():
    return {'checkpoint_sha256': L.INIT_SHA, 'state_digest': L.INIT_DIGEST, 'seed': 2026}


@pytest.mark.parametrize('role', ['MAIN', 'MAIN_TEST', 'TEST2', 'RETRO_TEST', 'UNSEEN_VAL'])
def test_other_roles_refused_before_source_read(monkeypatch, role):
    monkeypatch.setattr(L, 'load_protocol', lambda *a: (_ for _ in ()).throw(AssertionError('source read')))
    assert L.main(['--protocol', 'never', '--role', role, '--cell', 'cold', '--gpu', '1', '--out', 'never']) == 2


@pytest.mark.parametrize('device', [None, '0', '0,1'])
def test_early_device_export_required(monkeypatch, device):
    if device is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', device)
    monkeypatch.setattr(L, 'load_protocol', lambda *a: (_ for _ in ()).throw(AssertionError('source read')))
    assert L.main(['--protocol', 'never', '--cell', 'cold', '--gpu', '1', '--out', 'never']) == 2


def test_no_duplicate_PB100_P2_or_arbitrary_cell():
    for name in ['pb100_FA_eta3e-05_P2_E6_s2026', 'cold_FA_eta0.01_P4_E6_s2026', 'cold_FA_eta0.001_P1_E6_s2026']:
        with pytest.raises(L.Refused, match='six-cell'):
            L.select_cell(name)


def test_matched_consumed_exposure_rounds_and_different_cohort_schedule():
    assert [L.rounds(p) for p in (1, 2, 4)] == [3414, 1707, 854]
    # These are planned expected exposures. Actual dropouts and heterogeneous
    # client sizes are recorded by the inherited per-round counters.
    for p in (1, 2, 4):
        planned = L.rounds(p) * 64 * .9 * p / 32768
        assert 6 <= planned < 6.005
    assert L.rounds(4) * 64 < L.rounds(2) * 64
    with pytest.raises(L.Refused):
        L.rounds(4, n=131072)


def test_specs_derive_same_registered_prox_and_private_rules(cold):
    from v4int import launch_fl as F
    for method in ('FP', 'PF'):
        spec = L.fixed_spec(dict(cold, method=method))
        if method == 'FP':
            assert spec['cfg']['mu'] == F.KNOBS['mu'] == .01
        else:
            assert spec['cfg']['pf'] == {'lr': L.PEAK, 'lam': F.KNOBS['pf_lam'], 'clip': F.KNOBS['pf_clip']}
            assert spec['cfg']['pf']['lr'] != spec['cfg']['solver']['lr']
        assert spec['cfg']['dp'] is None


def test_warm_init_and_item_ratio_exact(warm):
    s = L.fixed_spec(warm, init_identity=identity())
    assert s['cfg']['solver']['item_lr_mult'] == .1
    for field, value in [('seed', 2027), ('state_digest', 'wrong'), ('checkpoint_sha256', 'wrong')]:
        ident = dict(identity(), **{field: value})
        with pytest.raises(L.Refused, match='own PB100'):
            L.fixed_spec(warm, init_identity=ident)
    with pytest.raises(L.Refused, match='needs'):
        L.fixed_spec(warm)


def test_cold_cannot_accept_warm_checkpoint(cold):
    with pytest.raises(L.Refused, match='cold'):
        L.fixed_spec(cold, init_identity=identity())


@pytest.mark.parametrize('passes', [2, 4])
def test_real_local_solver_consumption_and_repeatability(passes):
    worker = tiny(10, tied=False)
    theta = tiny(3, tied=False).broadcast_state(clone=True)
    c = one_client('owned-calib', 19, seed=11, invalid_frac=.1)
    sol = LocalSolver(lr=.1, optimizer='sgd_m', weight_decay=0., passes=passes, batch_size=16)
    a = client_update(worker, theta, c, sol, round_idx=2, seed=2026, clone_upload=True)
    b = client_update(worker, theta, c, sol, round_idx=2, seed=2026, clone_upload=True)
    assert a.n_consumed == a.n_valid * passes
    assert a.steps == a.steps_per_pass * passes
    assert state_digest(a.upload) == state_digest(b.upload)


@pytest.mark.parametrize('passes', [2, 4])
def test_registered_prox_changes_after_first_step_and_mu0_equals_FA(passes):
    theta = tiny(3, tied=False).broadcast_state(clone=True)
    c = one_client('owned-prox', 4, seed=11)
    sol = LocalSolver(lr=.1, optimizer='sgd_m', weight_decay=0., passes=passes, batch_size=16)
    a = client_update(tiny(10, tied=False), theta, c, sol, round_idx=2, seed=2026, clone_upload=True)
    zero = client_update(tiny(10, tied=False), theta, c, sol, round_idx=2, seed=2026, mu=0., clone_upload=True)
    prox = client_update(tiny(10, tied=False), theta, c, sol, round_idx=2, seed=2026, mu=.01, clone_upload=True)
    assert state_digest(a.upload) == state_digest(zero.upload)
    assert state_digest(a.upload) != state_digest(prox.upload)
    once = copy.copy(sol)
    from dataclasses import replace
    once = replace(once, passes=1)
    f = client_update(tiny(10, tied=False), theta, c, once, round_idx=2, seed=2026, clone_upload=True)
    p = client_update(tiny(10, tied=False), theta, c, once, round_idx=2, seed=2026, mu=.01, clone_upload=True)
    assert state_digest(f.upload) == state_digest(p.upload)


@pytest.mark.parametrize('passes', [2, 4])
def test_pf_private_zero_control_upload_isolation_and_persistence(passes):
    theta = tiny(3, tied=False).broadcast_state(clone=True)
    c = one_client('owned-personal', 19, seed=11)
    sol = ItemLRSolver(lr=.1, optimizer='sgd_m', item_lr_mult=.1, weight_decay=0., passes=passes)
    a = client_update(tiny(10, tied=False), theta, c, sol, round_idx=2, seed=2026, clone_upload=True)
    zero = client_update(tiny(10, tied=False), theta, c, sol, round_idx=2, seed=2026,
                        pf=PFConfig(lr=0., lam=0.), clone_upload=True)
    assert state_digest(a.upload) == state_digest(zero.upload)
    actual = client_update(tiny(10, tied=False), theta, c, sol, round_idx=2, seed=2026,
                          pf=PFConfig(lr=L.PEAK, lam=1e-4, clip=1.), clone_upload=True)
    assert set(actual.upload) == set(a.upload) == set(tiny(10, tied=False).manifest.shared_keys)
    assert not set(tiny(10, tied=False).manifest.buffer_keys) & set(actual.upload)
    assert actual.personal is not None and torch.count_nonzero(actual.personal.p) > 0
    again = client_update(tiny(10, tied=False), theta, c, sol, round_idx=3, seed=2026,
        pf=PFConfig(lr=L.PEAK, lam=1e-4, clip=1.), personal=actual.personal, clone_upload=True)
    assert again.personal.visits == actual.personal.visits + 1
    assert again.personal.n_consumed == actual.personal.n_consumed + again.n_consumed


def test_pf_CALIB_harness_replaces_MAIN_population_without_mutating_original():
    main = SimpleNamespace(evalp='MAIN_PROVIDER', key_base={'eval_manifest_id': 'MAIN'},
                           _row_labels=lambda: {'manifest_role': 'MAIN'})
    calib = SimpleNamespace(manifest_id=L.EVAL)
    keys = {'eval_manifest_id': L.EVAL, 'train_manifest_id': L.TRAIN}
    proxy = L.paired_calib_harness(main, calib, keys)
    assert proxy.evalp is calib and proxy.key_base == keys
    assert proxy._row_labels()['manifest_role'] == 'CALIBRATION'
    assert main.evalp == 'MAIN_PROVIDER' and main.key_base['eval_manifest_id'] == 'MAIN'
    with pytest.raises(L.Refused, match='CALIB provider'):
        L.paired_calib_harness(main, SimpleNamespace(manifest_id='MAIN'), keys)


def test_context_restores_spec_metadata_and_personal_dispatch(cold):
    from v4int import launch_fl as F
    from v4int import launch_device as D
    keys = ['ALL_ARMS', 'METHOD_CODE', 'base_arm', 'resolve_spec', 'write_json', 'spec_tuning', 'is_warm_arm']
    before, pf = {k: getattr(F, k) for k in keys}, D.pf_evaluator
    with pytest.raises(RuntimeError):
        with L.launcher_context(cold, {'numerical_code_sha256': 'toy'}):
            assert F.base_arm('EXT_LP_COLD_FA') == 'FA'
            spec = L.fixed_spec(cold)
            assert F.spec_tuning(spec)['local_passes'] == 2
            bad = copy.deepcopy(spec)
            bad['cfg']['endpoint_rounds'] += 1
            with pytest.raises(L.Refused, match='exact own'):
                F.check_warm_spec(bad)
            with pytest.raises(L.Refused, match='nested'):
                with L.launcher_context(cold, {}):
                    pass
            raise RuntimeError('synthetic exception')
    assert all(getattr(F, k) is v for k, v in before.items()) and D.pf_evaluator is pf


def test_personal_dispatch_uses_captured_CALIB_keys_even_when_MAIN_keys_are_passed(cold, monkeypatch):
    from v4int import launch_fl as F
    from v4int import launch_device as D
    import v4base.identity as I
    tr = SimpleNamespace(evalp='MAIN_PROVIDER', key_base={'eval_manifest_id': 'MAIN'},
                         _row_labels=lambda: {'manifest_role': 'MAIN'})
    provider = SimpleNamespace(manifest_id=L.EVAL)
    kb = {'train_manifest_id': L.TRAIN, 'eval_manifest_id': L.EVAL}
    received = {}
    monkeypatch.setattr(F, 'role_evaluator', lambda *args: 'shared_CALIB_evaluator')

    def personal(proxy, keys):
        received.update(proxy=proxy, keys=keys)
        return 'personal_CALIB_evaluator'

    def external(ep, **kw):
        assert ep is provider
        return ['own-CALIB-external-key']

    monkeypatch.setattr(D, 'pf_evaluator', personal)
    monkeypatch.setattr(I, 'eval_client_key_hex_by_eval_user_key', external)
    with L.launcher_context(dict(cold, method='PF'), {'numerical_code_sha256': 'toy'}):
        assert F.role_evaluator(tr, provider, kb, {}) == 'shared_CALIB_evaluator'
        assert D.pf_evaluator(tr, ['wrong-MAIN-key']) == 'personal_CALIB_evaluator'
        assert received['proxy'].evalp is provider
        assert received['keys'] == ['own-CALIB-external-key']
    assert tr.evalp == 'MAIN_PROVIDER'


def test_context_does_not_replace_numerical_execution_kernels(cold):
    from v4int import launch_fl as F
    from fl import client, runtime, server
    before = (F.execute, F.run, client.client_update, runtime.FLRun.step, server.FedAdam.step)
    with L.launcher_context(cold, {'numerical_code_sha256': 'toy'}):
        assert before == (F.execute, F.run, client.client_update, runtime.FLRun.step, server.FedAdam.step)


def test_rows_name_actual_SGD_lifecycle_exploratory_scope_and_private_AdamW(cold):
    from v4int import launch_fl as F
    cell = dict(cold, method='PF')
    with L.launcher_context(cell, {'numerical_code_sha256': 'toy'}):
        r = F.fl_rows([{'run_label': 'PRODUCTION'}], arm='EXT_LP_COLD_PF', run_id='own',
                      efe_raw=1., mark=1., digest='toy', role='CALIBRATION')[0]
        assert r['optimizer_lifecycle'] == 'FRESH_SGDM_PER_VISIT'
        assert r['pf_private_optimizer_lifecycle'] == 'PERSISTENT_ADAMW_PER_CLIENT'
        assert r['run_label'] == r['comparison_status'] == L.LABEL
        assert r['manifest_role'] == 'CALIBRATION'


def test_mechanical_inclusive_tie_keeps_negative_outcomes():
    pool = [{'cell': 'p4', 'passes': 4, 'server_lr': .001, 'value': .101},
            {'cell': 'p2', 'passes': 2, 'server_lr': .001, 'value': .100}]
    assert L.choose(pool)['cell'] == 'p2'  # exactly0.001 inclusive
    pool[0]['value'] = .10100001
    assert L.choose(pool)['cell'] == 'p4'
    for r in pool:
        r['value'] -= .1  # no forced-positive/no-improvement filter
    assert L.choose(pool)['cell'] == 'p4'
    with pytest.raises(L.Refused):
        L.choose([dict(pool[0], value=float('nan'))])


def engine_pb_fixture(values):
    """Use the actual inherited evaluation method, without a training kernel."""
    from fl.runtime import FLRun, RunState
    run = FLRun.__new__(FLRun)
    run.state, run.cfg, run.server, run.ckpt = RunState(), SimpleNamespace(best_delta=.0002), None, None
    iterator = iter(values)
    run.eval_fn = lambda _: next(iterator)
    for unit in range(1, 13):
        FLRun._evaluate(run, unit, unit - 1)
    summary = {'evals': run.state.evals, 'best_metric': run.state.best_metric,
               'best_half_efe': run.state.best_half_efe}
    scores = {f'{unit/2:.1f}': value for unit, value in enumerate(values, 1)}
    return summary, scores


def test_retained_PB_exact_float_boundary_matches_inherited_engine():
    threshold = .1 + .0002
    values = [.1, math.nextafter(threshold, -math.inf), threshold,
              math.nextafter(threshold, math.inf)] + [.1] * 8
    summary, scores = engine_pb_fixture(values)
    pb = L.retained_pb(summary, scores, n_rounds=12)
    assert pb['mark'] == 2. and pb['value'] == values[3]
    assert pb['improvements'][:4] == [True, False, False, True]
    assert L.fixed_declaration()['best_delta'] == .0002


def test_retained_PB_changes_selection_at_inclusive_winner_boundary():
    p4_summary, p4_scores = engine_pb_fixture([.101] + [.10100001] * 11)
    pb = L.retained_pb(p4_summary, p4_scores, n_rounds=12)
    pool = [{'cell': 'p2', 'passes': 2, 'server_lr': .001, 'value': .100},
            {'cell': 'p4', 'passes': 4, 'server_lr': .001, 'value': pb['value']}]
    assert pb['mark'] == .5 and L.choose(pool)['cell'] == 'p2'
    # Raw maximum would silently change the declared inclusive tie decision.
    assert L.choose([pool[0], dict(pool[1], value=max(p4_scores.values()))])['cell'] == 'p4'


@pytest.mark.parametrize('mutation', ['mark0', 'wrong_flag', 'wrong_best_mark', 'wrong_best_metric',
                                    'out_of_order', 'score_drift', 'wrong_phase'])
def test_chronological_PB_mutations_refused(mutation):
    summary, scores = engine_pb_fixture([.1] * 12)
    if mutation == 'mark0':
        summary['evals'].insert(0, [0, 0, 1., True, 'MAIN'])
        scores['0.0'] = 1.
    elif mutation == 'wrong_flag':
        summary['evals'][1][3] = True
    elif mutation == 'wrong_best_mark':
        summary['best_half_efe'] = 12
    elif mutation == 'wrong_best_metric':
        summary['best_metric'] += .00001
    elif mutation == 'out_of_order':
        summary['evals'][1][1] = 0
    elif mutation == 'score_drift':
        summary['evals'][1][2] = math.nextafter(.1, math.inf)
    else:
        summary['evals'][1][4] = 'CONTINUATION'
    with pytest.raises(L.Refused):
        L.retained_pb(summary, scores, n_rounds=12)


@pytest.fixture(scope='module')
def actual_cold_P1():
    from v4int.verify import DataPaths, verify_manifest_by_registry
    from v4eval.manifest import build_eval_manifest
    from v4int.providers import EVAL_MANIFEST_COLUMNS
    directory = L.DONOR / 'runs_x_rees46/COLD/G/FA_eta0.001_tau1e-6_E6'
    vm = verify_manifest_by_registry(L.EVAL, L.WORK / 'contracts/manifest_registry/REGISTRY_V2.json',
        DataPaths.default(L.WORK), root=L.WORK, expect_kind='EVAL')
    manifest = build_eval_manifest(vm.header, {c: vm.columns[c] for c in EVAL_MANIFEST_COLUMNS if c in vm.columns})
    cell = {'cell': 'cold_FA_eta0.001_P1_E6_s2026_HISTORICAL', 'line': 'cold', 'method': 'FA',
            'passes': 1, 'server_lr': .001, 'seed': 2026}
    summary, side = L.read(directory / 'RUN_SUMMARY.json'), L.read(directory / 'ARM_SPEC.json')
    scores = {f'{u/2:.1f}': L.read(directory / 'evals' / f'eval_{u/2:.1f}.result.json')['E2E']['macro']['mrr@20']
              for u in range(1, 13)}
    return directory, cell, manifest, summary, side, scores


def test_actual_historical_P1_retains_4point5_despite_raw_endpoint_max(actual_cold_P1):
    directory, cell, manifest, summary, side, scores = actual_cold_P1
    result = L.read_curve(directory, cell, manifest=manifest)
    assert result['best_mark'] == 4.5 and result['value'] == .1139379621389816
    assert max(scores, key=scores.get) == '6.0' and scores['6.0'] == .1140739310791782
    assert result['retained_checkpoint']['round'] == 2564
    assert result['retained_checkpoint']['state_digest'] == L.read(directory / 'evals/eval_4.5.result.json')['model_sha256']


@pytest.mark.parametrize('mutation', ['raw_max_mark', 'false_improvement', 'changed_score'])
def test_actual_historical_PB_summary_mutation_refused(actual_cold_P1, mutation):
    _, _, _, summary, _, scores = actual_cold_P1
    summary = copy.deepcopy(summary)
    if mutation == 'raw_max_mark':
        summary['best_half_efe'], summary['best_metric'] = 12, scores['6.0']
    elif mutation == 'false_improvement':
        summary['evals'][-1][3] = True
    else:
        summary['evals'][-1][2] = math.nextafter(summary['evals'][-1][2], math.inf)
    with pytest.raises(L.Refused):
        L.retained_pb(summary, scores, n_rounds=3414)


@pytest.mark.parametrize('field,value', [('round', 3414), ('run_id', 'another'),
                                       ('config_digest', 'changed'), ('phase', 'CONTINUATION')])
def test_actual_retained_checkpoint_identity_mutation_refused(actual_cold_P1, monkeypatch, field, value):
    import fl.checkpoint as C
    directory, _, _, summary, side, scores = actual_cold_P1
    real = C.load_checkpoint
    monkeypatch.setattr(C, 'load_checkpoint', lambda p: dict(real(p), **{field: value}))
    pb = L.retained_pb(summary, scores, n_rounds=3414)
    with pytest.raises(L.Refused, match='checkpoint identity'):
        L.verify_retained_checkpoint(directory, side, summary, pb, L.read(directory / 'evals/eval_4.5.result.json'))


def test_real_historical_CALIB_evaluator_identity_and_tampered_population_refusal():
    p = L.DONOR / 'runs_x_rees46/PB100/S2/eta3e-05_c0.1_ilr0.1_tau1e-6_E6_P2/evals/eval_5.5.result.json'
    r = L.read(p)
    m = SimpleNamespace(n=158135, hashes=r['hashes'])
    L.check_result_identity(r, m)
    for field, value in [('manifest_id', 'MAIN_VALIDATION_5_100_131072'), ('n_rows', 1), ('evaluator_code_sha256', 'wrong')]:
        with pytest.raises(L.Refused, match='population or exact frozen'):
            L.check_result_identity(dict(r, **{field: value}), m)
    bad = copy.deepcopy(r)
    bad['eval_config']['eval_device_class'] = 'CPU-FP32'
    with pytest.raises(L.Refused, match='population or exact frozen'):
        L.check_result_identity(bad, m)


def test_FP_PF_require_winner_and_can_reuse_historical_PB100_P2():
    name = 'pb100_FA_eta3e-05_P2_E6_s2026_HISTORICAL'
    winner = {'cell': name, 'line': 'pb100', 'passes': 2, 'server_lr': 3e-5, 'seed': 2026}
    sel = {'winners': {'pb100': winner}}
    cell = L.select_cell(name, method='PF', selection=sel)
    assert cell['cell'] == 'pb100_PF_eta3e-05_P2_E6_s2026'
    assert L.fixed_spec(cell, init_identity=identity())['cfg']['method'] == 'PF'
    with pytest.raises(L.Refused, match='mechanical'):
        L.select_cell(name, method='FP')
    with pytest.raises(L.Refused, match='mechanical'):
        L.select_cell('not_winner', method='FP', selection=sel)


def test_partial_resume_and_immutable_complete(tmp_path):
    out = tmp_path / 'new'
    b = {'numerical': 'same', 'base': 'same'}
    L.output_guard(out, b, root=tmp_path)
    assert L.output_guard(out, b, resume=True, root=tmp_path) == out
    with pytest.raises(L.Refused):
        L.output_guard(out, dict(b, numerical='other'), resume=True, root=tmp_path)
    (out / 'LOCALPASSES_COMPLETE.json').write_text('{}')
    with pytest.raises(L.Refused, match='never reruns'):
        L.output_guard(out, b, resume=True, root=tmp_path)
    with pytest.raises(L.Refused):
        L.output_guard(tmp_path / 'absent', b, resume=True, root=tmp_path)


def test_symlink_and_old_outputs_refused(tmp_path):
    phase = tmp_path / 'phase'
    phase.mkdir()
    old = tmp_path / 'old'
    old.mkdir()
    (phase / 'alias').symlink_to(old, target_is_directory=True)
    with pytest.raises(L.Refused):
        L.output_guard(phase / 'alias' / 'run', {}, root=phase)
    assert not list(old.iterdir())


def test_tampered_protocol_self_sha_and_finite_json(tmp_path):
    p = tmp_path / 'bad'
    p.write_text('{"x": NaN}')
    with pytest.raises(L.Refused, match='nonfinite'):
        L.read(p)
    p.write_text(json.dumps({'format': L.FORMAT, 'declaration': L.fixed_declaration(), 'self_sha256': 'wrong'}))
    with pytest.raises(L.Refused, match='fixed CALIB'):
        L.load_protocol(p)

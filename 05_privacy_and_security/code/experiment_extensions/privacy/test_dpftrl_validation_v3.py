"""CPU-only scorer contracts, actual tiny rich-model interfaces, and negative controls.

All synthetic run/review files stay under pytest's extension scratch directory.
No real endpoint, TEST data, private entropy or CUDA is used by these tests.
"""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import ppsi_env  # noqa: F401
import numpy as np
import pytest
import torch

from fl.numerics import enforce_strict_fp32
from fl.synthetic import make_tiny_adapter
from v4eval import frozen_config as FC
from v4eval.manifest import build_eval_manifest
from v4eval.predict import predict_adapter
from v4eval.bootstrap import point
from . import dpftrl_train_v4 as D
from . import dpftrl_validation_v3 as V


@pytest.fixture(autouse=True)
def strict():
    torch.set_num_threads(2)
    enforce_strict_fp32(warn_only=False)
    FC.apply_runtime_flags(FC.load())
    assert not torch.cuda.is_initialized()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    return path


def tensor_file(payload):
    stream = io.BytesIO(); torch.save(payload, stream); stream.seek(0)
    return stream


def payload(adapter=None, mechanism='tree', eta=.1):
    adapter = adapter or make_tiny_adapter(K=7, d=4, tied=True, dropout=0., seed=19)
    c = D.select_cell(mechanism, eta)
    return adapter, D.model_only_payload(adapter.broadcast_state(), D.SharedVector(adapter), c, D.budget(mechanism)), c


def manifest():
    # User10 has two rankable and one OOV row, user20 is OOV-only, user30
    # has a rankable row plus censored row; unequal users expose decision-macro bugs.
    return build_eval_manifest({'manifest_id': 'SYNTH_CALIB_VALIDATION', 'v4_split': 'VALIDATION', 'catalogue_K': 7}, {
        'decision_id': np.arange(7, dtype=np.int64) + 10,
        'eval_user_key': np.array([10, 10, 10, 20, 30, 30, 40], dtype=np.int64),
        'eval_mask': np.array([1, 1, 1, 1, 1, 0, 0], dtype=bool),
        'target_class': np.array([0, 1, -1, -1, 2, -1, -1], dtype=np.int64),
        'target_oov': np.array([0, 0, 1, 1, 0, 0, 0], dtype=bool),
        'censor_class': np.array(['NONE'] * 5 + ['CEN_NO_NEXT'] * 2)})


def test_full_e2e_macro_oov_only_user_and_censored_hand_oracle():
    m = manifest(); st = V.validation_stats(m, np.array([1, 2, -1, -1, 4, -1, -1]))
    assert st.user_keys.tolist() == [10, 20, 30]
    assert st.N.tolist() == [3., 1., 1.] and st.NR.tolist() == [2., 0., 1.]
    assert st.S.tolist() == [1.5, 0., .25]
    assert point(st, 'macro_e2e') == pytest.approx((.5 + 0 + .25) / 3)
    assert point(st, 'micro_e2e') == .35
    assert point(st, 'macro_e2e') != point(st, 'micro_e2e')
    assert point(st, 'macro_e2e') != pytest.approx((.75 + .25) / 2)


@pytest.mark.parametrize('ranks', [[1, 2, -1, -1], [0, 2, -1, -1, 4, -1, -1],
    [8, 2, -1, -1, 4, -1, -1], [1, 2, 1, -1, 4, -1, -1],
    [1, 2, -1, -1, 4, 1, -1], [1, 2, -2, -1, 4, -1, -1],
    [1., 2., -1., -1., 4., -1., -1.]])
def test_rank_alignment_fullK_oov_censor_and_integer_negative_controls(ranks):
    with pytest.raises(V.Refused): V.validation_stats(manifest(), np.array(ranks))


@pytest.mark.parametrize('mechanism', ['tree', 'local'])
@pytest.mark.parametrize('eta', [.1, 1.])
@pytest.mark.parametrize('tied', [False, True])
def test_model_only_shared_roundtrip_inserts_fixed_buffers_and_aliases(mechanism, eta, tied):
    a = make_tiny_adapter(K=7, d=4, tied=tied, dropout=0., seed=19)
    b = make_tiny_adapter(K=7, d=4, tied=tied, dropout=0., seed=20)
    _, p, c = payload(a, mechanism, eta)
    before = {k: v.clone() for k, v in b.extract_buffers().items()}
    V.load_model_only(tensor_file(p), b, c)
    for k in a.manifest.shared_keys:
        assert torch.equal(a.broadcast_state()[k], b.broadcast_state()[k])
    for k in before: assert torch.equal(before[k], b.broadcast_state()[k])
    assert set(p['shared_theta']).isdisjoint(a.manifest.alias_keys)
    assert set(p['shared_theta']).isdisjoint(a.manifest.buffer_keys)


@pytest.mark.parametrize('field,value', [('format', 'other'), ('public_cell', D.select_cell('local', .1)),
    ('privacy_certified', True), ('pretraining_preprocessing_tuning_accounted', True),
    ('transport_isolation_implemented', True), ('execution_scope', 'DEPLOYED'),
    ('conditional_fixed_recipe_accountant', D.budget('local')), ('PRIVATE_ENTROPY', 123), ('INTERNAL_COUNTERS', {})])
def test_export_missing_private_scope_cell_accountant_refused(field, value):
    a, p, c = payload(); p[field] = value
    with pytest.raises(V.Refused): V.load_model_only(tensor_file(p), a, c)


@pytest.mark.parametrize('mutation', ['missing', 'extra_buffer', 'float64', 'float16', 'nan', 'shape', 'nontensor'])
def test_shared_geometry_nonfinite_dtype_and_unknown_keys_refused(mutation):
    a, p, c = payload(); shared = p['shared_theta']; k = next(iter(shared))
    if mutation == 'missing': shared.pop(k)
    elif mutation == 'extra_buffer': shared['fake_buffer'] = torch.tensor(1.)
    elif mutation in ('float64', 'float16'): shared[k] = shared[k].to(getattr(torch, mutation))
    elif mutation == 'nan': shared[k].reshape(-1)[0] = float('nan')
    elif mutation == 'shape': shared[k] = torch.zeros(1)
    else: shared[k] = 1.
    with pytest.raises((V.Refused, D.Refused)): V.load_model_only(tensor_file(p), a, c)


def test_query_strips_truth_without_losing_features_or_decision_alignment():
    a = {'decision_id': torch.tensor([5]), 'item_tokens': torch.tensor([[3]]), 'target_class': torch.tensor([7]),
         'target_oov': True, 'labels': 'truth', 'event_numeric_features': torch.ones(1, 1, 2)}
    q = V.query_inputs(a)
    assert set(q) == {'decision_id', 'item_tokens', 'event_numeric_features'}
    assert q['item_tokens'] is a['item_tokens'] and a['target_class'].item() == 7


def test_actual_small_rich_D64_full_frozen_predict_and_truth_poison_interface():
    from v4models import build
    from v4models.schema import load_schema
    from v4models.synthetic import synthetic_examples_v2
    built = build('SASREC', load_schema(), 41, 'cpu', K=7, variant='SASREC_D64_B2', dropout=(0.,))
    a, m = built.adapter, manifest()
    batch = synthetic_examples_v2(5, 7, built.widths, torch.Generator().manual_seed(37), max_len=5, window=50, with_names=True)
    batch['decision_id'] = torch.tensor(m.decision_id[:5]); batch['target_class'] = torch.tensor(m.target_class[:5])
    a.module.eval()
    with torch.no_grad():
        q = a.query(V.query_inputs(batch))
        poisoned = dict(batch, target_class=torch.full((5,), -987654))
        assert torch.equal(q, a.query(V.query_inputs(poisoned)))
    cfg = FC.load()
    art = predict_adapter(a, [V.query_inputs(batch)], m, config=cfg, model_sha256='synthetic', device='cpu')
    res = V.evaluate_utility(m, art, cfg)
    from v4eval import evaluator_code_sha256
    assert res['evaluator_code_sha256'] == evaluator_code_sha256()
    value = V.metric_recount(m, art, res, actual_population=False)
    assert value == pytest.approx(point(V.validation_stats(m, art.ranks), 'macro_e2e'), abs=1e-16)
    assert np.all(art.ranks[m.rankable] >= 1) and np.all(art.ranks[~m.rankable] == -1)
    # Independent full score-matrix target-rank oracle, quoted tie rule.
    with torch.no_grad(): scores = a.logits(q)
    for row in np.flatnonzero(m.rankable):
        t = m.target_class[row]; s = scores[row]; oracle = 1 + int((s > s[t]).sum()) + int(((s == s[t]) & (torch.arange(7) < t)).sum())
        assert art.ranks[row] == oracle
    rows = V.utility_rows(res, D.select_cell('local', .1), {'synthetic': True})
    assert len(rows) == 48 and {r['method'] for r in rows} == {'LDP_MESSAGE_ROLE_SIMULATOR'}
    assert all(not r['privacy_certified'] and not r['utility_release_inside_privacy_accountant'] for r in rows)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('changed', ['score', 'n_users', 'n_decisions'])
def test_independent_metric_recount_detects_false_result_denominators(changed):
    m = manifest(); ranks = np.array([1, 2, -1, -1, 4, -1, -1]); value = point(V.validation_stats(m, ranks), 'macro_e2e')
    result = {'E2E': {'n_users': 3, 'n_decisions': 5, 'macro': {'mrr@20': value}}}
    if changed == 'score': result['E2E']['macro']['mrr@20'] = .999
    else: result['E2E'][changed] += 1
    with pytest.raises(V.Refused): V.metric_recount(m, SimpleNamespace(ranks=ranks), result, actual_population=False)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), None, 'not a number'])
def test_independent_metric_recount_refuses_nonfinite_or_nonnumeric_result(bad):
    m = manifest(); ranks = np.array([1, 2, -1, -1, 4, -1, -1])
    result = {'E2E': {'n_users': 3, 'n_decisions': 5, 'macro': {'mrr@20': bad}}}
    with pytest.raises(V.Refused): V.metric_recount(m, SimpleNamespace(ranks=ranks), result, actual_population=False)


@pytest.mark.parametrize('role,split', [('MAIN', 'VALIDATION'), ('TEST', 'VALIDATION'), ('CALIBRATION', 'TEST'), ('CALIBRATION', 'TRAIN')])
def test_role_split_refusal_precedes_any_file_or_data_access(monkeypatch, role, split):
    monkeypatch.setattr(D, 'sha', lambda p: pytest.fail('opened before role gate'))
    with pytest.raises(V.Refused): V.load_protocol('/absent', 'x', role=role, split=split)
    with pytest.raises(V.Refused): V.build_protocol('/absent', role=role, split=split)
    with pytest.raises(V.Refused): V.prepare(None, None, None, None, role=role, split=split)
    with pytest.raises(V.Refused): V.score(None, None, None, None, None, None, 'tree', .1, None, None, None, None, role=role, split=split)


@pytest.fixture
def own_run(tmp_path, monkeypatch):
    root = tmp_path / 'DPFTRL_CALIB_04'; monkeypatch.setattr(V, 'RUN_ROOT', root)
    cells = [D.select_cell(m, e) for m in ('tree', 'local') for e in (.1, 1.)]
    cell = cells[0]; directory = root / cell['cell_id']; directory.mkdir(parents=True)
    review = {'status': 'PASS', 'reviewer': '/synthetic/independent', 'protocol_sha256': V.TRAIN_PROTOCOL_SHA,
        'preparation_sha256': V.TRAIN_PREPARATION_SHA, 'module_sha256': V.TRAIN_MODULE_SHA,
        'tests_sha256': V.TRAIN_TESTS_SHA, 'run_accountant_concurrence': True, 'modes': ['tree', 'local'],
        'fixed_public_S': 1., 'TEST_approved': False,
        'approved_output_directories': {c['cell_id']: str(root / c['cell_id']) for c in cells}}
    review_path = write(tmp_path / 'SYNTHETIC_NOT_AN_APPROVAL.json', review); review_sha = D.sha(review_path)
    started = {'cell': cell, 'protocol_sha256': V.TRAIN_PROTOCOL_SHA, 'preparation_sha256': V.TRAIN_PREPARATION_SHA,
        'review_sha256': review_sha, 'module_sha256': V.TRAIN_MODULE_SHA, 'role': 'CALIBRATION', 'split': 'VALIDATION',
        'gpu_ownership_check': {'synthetic': True}}
    a, p, _ = payload(); model = directory / 'MODEL_ONLY/model.pt'; model.parent.mkdir(); torch.save(p, model)
    complete = {'status': 'TRAIN_ONLY_NOT_UTILITY_COMPLETE', 'binding': started, 'model_file_sha256': D.sha(model),
        'TEST_read': False, 'privacy_certified': False, 'validation_scored': False, 'utility_complete': False,
        'execution_scope': V.declaration()['execution_scope']}
    write(directory / 'STARTED.json', started); write(directory / 'COMPLETE.json', complete)
    return {'protocol': {'run_root': str(root), 'cells': cells}, 'directory': directory, 'started': started,
        'complete': complete, 'review': review, 'review_path': review_path, 'review_sha': review_sha, 'model': model}


def completed(r):
    return V.own_completed_export(r['protocol'], 'tree', .1, r['review_path'], r['review_sha'])


def test_completed_export_only_opens_three_whitelisted_run_files(own_run, monkeypatch):
    r = own_run; read_before = D.read; opened = []
    def read(path):
        opened.append(str(path)); assert 'INTERNAL' not in str(path) and 'PRIVATE' not in str(path)
        return read_before(path)
    monkeypatch.setattr(D, 'read', read)
    identity = completed(r)
    assert set(identity['files']) == {'STARTED.json', 'COMPLETE.json', 'MODEL_ONLY/model.pt'}
    # Schema/preparation are fixed public component inputs, outside the run.
    assert set(opened) == {str(r['directory'] / 'STARTED.json'), str(r['directory'] / 'COMPLETE.json'),
        str(r['review_path']), str(V.TRAIN_PREPARATION), str(V.PRODUCTION_SCHEMA)}
    run_reads = [p for p in opened if V.RUN_ROOT in Path(p).parents]
    assert sorted(run_reads) == sorted([str(r['directory'] / 'STARTED.json'), str(r['directory'] / 'COMPLETE.json')])


@pytest.mark.parametrize('field,value', [('status', 'COMPLETE'), ('model_file_sha256', '0' * 64),
    ('TEST_read', True), ('privacy_certified', True), ('validation_scored', True), ('utility_complete', True),
    ('execution_scope', 'DEPLOYED')])
def test_wrong_training_completion_or_checkpoint_hash_refused(own_run, field, value):
    r = own_run; obj = copy.deepcopy(r['complete']); obj[field] = value; write(r['directory'] / 'COMPLETE.json', obj)
    with pytest.raises(V.Refused): completed(r)


@pytest.mark.parametrize('field,value', [('role', 'MAIN'), ('split', 'TEST'), ('protocol_sha256', '0' * 64),
    ('preparation_sha256', '0' * 64), ('module_sha256', '0' * 64), ('review_sha256', '0' * 64),
    ('cell', D.select_cell('tree', 1.)), ('gpu_ownership_check', None), ('EXTRA', 'unknown')])
def test_training_binding_role_source_recipe_and_review_refused(own_run, field, value):
    r = own_run; obj = copy.deepcopy(r['started']); obj[field] = value; write(r['directory'] / 'STARTED.json', obj)
    with pytest.raises(V.Refused): completed(r)


def test_changed_checkpoint_bytes_without_rewriting_receipt_refused(own_run):
    own_run['model'].write_bytes(b'other checkpoint')
    with pytest.raises(V.Refused): completed(own_run)


def test_binding_mismatch_between_completion_and_start_refused(own_run):
    r = own_run; obj = copy.deepcopy(r['complete']); obj['binding']['role'] = 'MAIN'; write(r['directory'] / 'COMPLETE.json', obj)
    with pytest.raises(V.Refused): completed(r)


def test_cross_cell_copy_cannot_impersonate_other_endpoint(own_run):
    r = own_run
    with pytest.raises(V.Refused): V.own_completed_export(r['protocol'], 'local', .1, r['review_path'], r['review_sha'])


def test_training_review_must_pin_canonical_output_map(own_run):
    r = own_run; changed = dict(r['review'], approved_output_directories={})
    write(r['review_path'], changed); changed_sha = D.sha(r['review_path'])
    r['started']['review_sha256'] = changed_sha; write(r['directory'] / 'STARTED.json', r['started'])
    r['complete']['binding'] = r['started']; write(r['directory'] / 'COMPLETE.json', r['complete'])
    with pytest.raises(V.Refused, match='output directories'):
        V.own_completed_export(r['protocol'], 'tree', .1, r['review_path'], changed_sha)


def test_cpu_component_review_never_authorizes_trained_export(own_run):
    r = own_run; obj = dict(r['review'], status='CPU_REVIEW_PASS_PENDING_ROOT_GPU')
    write(r['review_path'], obj); h = D.sha(r['review_path'])
    r['started']['review_sha256'] = h; write(r['directory'] / 'STARTED.json', r['started'])
    r['complete']['binding'] = r['started']; write(r['directory'] / 'COMPLETE.json', r['complete'])
    with pytest.raises(D.Refused): V.own_completed_export(r['protocol'], 'tree', .1, r['review_path'], h)


def test_symlink_completed_model_refused_before_deserialization(own_run, tmp_path):
    p = own_run['model']; moved = tmp_path / 'foreign.pt'; p.rename(moved); p.symlink_to(moved)
    with pytest.raises(V.Refused, match='physical|symlink'): completed(own_run)


def test_wrong_catalogue_and_rich_model_refused_before_scoring():
    with pytest.raises(V.Refused): V.require_actual_model(make_tiny_adapter(K=7, d=4, dropout=0.))


def test_protocol_source_pin_drift_and_selfhash_refused(tmp_path, monkeypatch):
    src = tmp_path / 'source.py'; src.write_text('original')
    def body(): return {'declaration': V.declaration(), 'sources': [D.pin(src)]}
    monkeypatch.setattr(V, 'protocol_body', body)
    p = tmp_path / 'score_protocol.json'; V.build_protocol(p); h = D.sha(p)
    V.load_protocol(p, h)
    src.write_text('changed')
    with pytest.raises(V.Refused, match='source/data'): V.load_protocol(p, h)
    src.write_text('original'); obj = D.read(p); obj['declaration']['eval_batch'] = 16; write(p, obj)
    with pytest.raises(V.Refused): V.load_protocol(p, D.sha(p))


def test_private_or_TEST_paths_are_not_part_of_declared_input_inventory():
    dec = V.declaration()
    assert dec['input_whitelist'] == ['STARTED.json', 'COMPLETE.json', 'MODEL_ONLY/model.pt']
    assert not dec['TEST_allowed'] and not dec['automatic_retry'] and not dec['resume']
    assert not dec['four_cell_joint_epsilon8_claim'] and not dec['private_training_state_read']
    assert dec['OOV_only_users'] == 10 and dec['full_manifest_rows'] == 158135



@pytest.mark.parametrize('case', ['idle', 'busy', 'foreign_memory', 'smi_failure', 'cuda_mismatch'])
def test_actual_score_entrypoint_inherited_preflight_true_runner(monkeypatch, tmp_path, case):
    """Reach scorer's true inherited preflight with deterministic subprocess observations."""
    import subprocess
    from v4int import launch_central as LC
    from v4sup.gpu import default_runner
    assert V.default_runner is default_runner
    assert not hasattr(LC, 'default_runner')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0' if case != 'cuda_mismatch' else '1')
    monkeypatch.setattr(D, 'require_early_cuda', lambda gpu: None)
    monkeypatch.setattr(V, 'load_protocol', lambda *a: {})
    monkeypatch.setattr(V, 'preparation_gate', lambda *a: {'actual_provider_proof': {}, 'provider_fingerprint': 'f', 'provider_manifest_binding': {}, 'population': {}})
    monkeypatch.setattr(V, 'review_gate', lambda *a: None)
    monkeypatch.setattr(V, 'own_completed_export', lambda *a: {})
    monkeypatch.setattr(D, 'fresh', lambda path: tmp_path / 'NEVER_CREATED')
    ep = SimpleNamespace(fingerprint='f', manifest_binding={}, manifest=None)
    monkeypatch.setattr(V, 'actual_cpu_state', lambda *a: {'proof': {}, 'ep': ep})
    monkeypatch.setattr(V, 'population_proof', lambda *a: {})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    calls = []
    def observed(argv, **kw):
        calls.append(argv)
        assert kw == {'capture_output': True, 'text': True, 'timeout': 60}
        if case == 'smi_failure':
            return subprocess.CompletedProcess(argv, 1, '', 'diagnostic')
        if '--query-gpu=index,pci.bus_id' in argv:
            out = '0, 00000000:01:00.0\n1, 00000000:02:00.0\n'
        elif '--query-compute-apps=gpu_bus_id,pid' in argv:
            out = '00000000:01:00.0, 123\n' if case == 'busy' else ''
        else:
            assert '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu' in argv
            out = '0, GPU-test-0, A40, %s, 46068, 0\n1, GPU-test-1, A40, 0, 46068, 0\n' % (4096 if case == 'foreign_memory' else 0)
        return subprocess.CompletedProcess(argv, 0, out, '')
    monkeypatch.setattr(subprocess, 'run', observed)
    error = V.Refused if case == 'idle' else LC.LaunchRefused
    with pytest.raises(error, match='CUDA scoring requires actual root-owned physical GPU' if case == 'idle' else None):
        V.score('p', 'ps', 'prep', 'preps', 'r', 'rs', 'tree', .1, 'tr', 'trs', tmp_path/'unused', 0)
    assert len(calls) == (0 if case == 'cuda_mismatch' else 3 if case in ('idle', 'foreign_memory') else 2)
    assert not (tmp_path/'NEVER_CREATED').exists()
    assert not torch.cuda.is_initialized()

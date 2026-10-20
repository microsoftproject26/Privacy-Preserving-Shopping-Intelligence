"""Causal-pair, donor-formula and exact full-catalogue sparse rank controls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
import torch

from experiment_extensions.baselines import rees46_training_free as T
from v4eval.ranking import rank_from_scores


def batch(items, lengths=None, targets=None):
    a = np.asarray(items, dtype=np.int64)
    out = {'item_tokens': a, 'lengths': np.full(len(a), a.shape[1], dtype=np.int64) if lengths is None else np.asarray(lengths)}
    if targets is not None:
        out['target_class'] = np.asarray(targets, dtype=np.int64)
        out['loss_mask'] = np.ones(len(a), dtype=bool)
    return out


def test_train_offsets_end_at_target_without_window_pair_overcount():
    b = batch([[3,4,5], [3,4,0]], [3,2], [3,2])
    keys = T.causal_pair_keys(b, 4, 3)
    assert [k.tolist() for k in keys] == [[2*4+3, 1*4+2], [1*4+3, 0*4+2], [0*4+3]]
    assert sum(map(len, keys)) == 5
    # The earlier 0->1 pair is not repeatedly manufactured from each later window.
    assert 1 not in np.concatenate(keys)


def test_unknown_train_source_preserves_original_offset():
    keys = T.causal_pair_keys(batch([[3,2,5]], targets=[1]), 4, 3)
    assert [k.tolist() for k in keys] == [[9], [], [1]]


@pytest.mark.parametrize('target,loss', [(-1, True), (4, True), (0, False)])
def test_ineligible_training_refused(target, loss):
    b = batch([[3]], targets=[target]); b['loss_mask'] = np.array([loss])
    with pytest.raises(ValueError):
        T.causal_pair_keys(b, 4)


@pytest.mark.parametrize('length', [-1, 4])
def test_malformed_history_length_refused(length):
    with pytest.raises(ValueError):
        T.history_matrix(batch([[3,4,5]], [length]), 4, 'MC')


def test_target_and_padding_never_enter_query():
    b = batch([[3,4,100,101]], [2], [0])
    changed = dict(b, target_class=np.array([3]), item_tokens=np.array([[3,4,3,5]]))
    for method in T.METHODS:
        np.testing.assert_array_equal(T.history_matrix(b, 4, method).toarray(), T.history_matrix(changed, 4, method).toarray())
    assert T.history_matrix(b, 4, 'MC').toarray().tolist() == [[0,1,0,0]]


def test_unknown_query_not_replaced_by_previous_known_item():
    H = T.history_matrix(batch([[3,2]]), 4, 'MC')
    assert H.nnz == 0
    H = T.history_matrix(batch([[3,2]]), 4, 'SeqRules')
    assert H.toarray()[0,0] == np.float32(0.25)
    H = T.history_matrix(batch([[3,2]]), 4, 'PCTM')
    assert H.toarray()[0,0] == np.float32(0.6) / np.float32(1.6)


def test_duplicate_query_items_sum_their_original_weights():
    H = T.history_matrix(batch([[3,3,3]]), 4, 'SeqRules')
    assert H.nnz == 1
    assert H.data[0] == np.float32(1 + 1/4 + 1/9)


def test_donor_mc_seqrules_pctm_formulas():
    K = 4
    offsets = []
    for d in range(10):
        m = np.zeros((K,K), dtype=np.float32)
        m[0,1], m[0,2], m[2,3] = d+1, 2, 3
        offsets.append(sparse.csr_matrix(m))
    frequency = np.array([4, 2, 1, 3], dtype=np.int64)
    models = T.models_from_offsets(offsets, frequency)
    # Independently apply the pinned donor equations, operation order and dtype.
    counts = offsets[0].toarray()
    inverse = np.zeros(K, dtype=np.float32)
    total = counts.sum(axis=1)
    inverse[total > 0] = 1 / total[total > 0]
    np.testing.assert_array_equal(models['MC'][0].toarray(), counts * inverse[:,None])
    seq = None; pctm = None
    for d, matrix in enumerate(offsets, 1):
        sw = np.float32(1) / np.log10(np.float32(d)+np.float32(1.7))
        pw = np.float32(1) / np.sqrt(np.float32(d))
        seq = matrix.multiply(sw) if seq is None else seq+matrix.multiply(sw)
        pctm = matrix.multiply(pw) if pctm is None else pctm+matrix.multiply(pw)
    np.testing.assert_array_equal(models['SeqRules'][0].toarray(), seq.toarray())
    expected = pctm.copy(); expected.data = np.log1p(expected.data * K / 11250).astype(np.float32)
    np.testing.assert_array_equal(models['PCTM'][0].toarray(), expected.toarray())
    log_pop = np.log((frequency+1)/(frequency.sum()+K)).astype(np.float32)
    np.testing.assert_array_equal(models['PCTM'][1], np.float32(-0.04)*log_pop)


def test_pruning_ties_keep_smaller_class_id_and_no_wrong_reorder():
    m = sparse.csr_matrix(np.array([[1,2,2,1],[0,0,0,0]], dtype=np.float32))
    p = T.prune_rows(m, 2)
    np.testing.assert_array_equal(p.toarray(), [[0,2,2,0],[0,0,0,0]])
    p = T.prune_rows(m, 1)
    np.testing.assert_array_equal(p.toarray(), [[0,2,0,0],[0,0,0,0]])


@pytest.mark.parametrize('boost', [-0.04, 0.0, 0.5])
@pytest.mark.parametrize('method', T.METHODS)
def test_sparse_ranks_exact_dense_oracle_all_ties_and_unknown(boost, method):
    rng = np.random.default_rng(901)
    K = 37
    matrix = rng.integers(0, 4, (K,K)).astype(np.float32)
    matrix[rng.random((K,K)) < 0.88] = 0
    frequency = rng.integers(0, 4, K)
    bg = (np.float32(boost)*np.log((frequency+1)/(frequency.sum()+K)).astype(np.float32))
    scorer = T.TrainingFreeScorer(method, sparse.csr_matrix(matrix), bg)
    tokens = rng.integers(0, K+4, (120, 7))
    lengths = rng.integers(0, 8, 120)
    b = batch(tokens, lengths)
    targets = rng.integers(0, K, 120)
    ranks, ties, before = scorer.ranks(b, targets)
    reference = rank_from_scores(torch.from_numpy(scorer.dense_oracle(b)), torch.from_numpy(targets), K_expected=K, class_chunk=11)
    np.testing.assert_array_equal(ranks, reference.ranks.numpy())
    np.testing.assert_array_equal(ties, reference.n_tied.numpy())
    np.testing.assert_array_equal(before, reference.eq_before.numpy())


def test_zero_background_rank_and_tie_counts_cover_all_K():
    b = T.BackgroundRanks(np.zeros(158486, dtype=np.float32))
    assert b.rank([], [], 12345) == (12346, 158485, 12345)


def test_float32_addition_rounding_cancellation_and_signed_zero():
    for bg in [np.array([-0., +0., -0., 0.], dtype=np.float32),
               np.array([1, 1+1e-7, 2, -2], dtype=np.float32)]:
        index = T.BackgroundRanks(bg)
        cols = np.array([0,2,3]); scores = np.array([1e-9, -2, 2], dtype=np.float32)
        dense = bg.copy(); dense[cols] += scores
        ref = rank_from_scores(torch.from_numpy(np.repeat(dense[None],4,axis=0)), torch.arange(4))
        for target in range(4):
            assert index.rank(cols, scores, target) == (int(ref.ranks[target]), int(ref.n_tied[target]), int(ref.eq_before[target]))


def test_unrankable_keeps_minus_one_and_does_not_change_other_rows():
    scorer = T.TrainingFreeScorer('MC', sparse.csr_matrix((4,4), dtype=np.float32), np.zeros(4, dtype=np.float32))
    ranks, ties, before = scorer.ranks(batch([[3],[3],[3]]), np.array([1,-1,3]))
    np.testing.assert_array_equal(ranks, [2,-1,4]); np.testing.assert_array_equal(ties, [3,-1,3])
    np.testing.assert_array_equal(before, [1,-1,3])


@pytest.mark.parametrize('phase', ['MAIN_TEST', 'RETRO_TEST', 'TEST', 'UNSEEN_TEST'])
def test_test_refusal_happens_without_creating_output(tmp_path, phase):
    out = tmp_path/'new'
    with pytest.raises(ValueError, match='TEST'):
        T.reserve_output(out, 'SCORING', phase)
    assert not out.exists()


def test_write_once_output_identity(tmp_path):
    out = tmp_path/'new'
    T.reserve_output(out, 'TRAIN_FIT', 'CALIBRATION')
    old = (out/'STARTED.json').read_bytes()
    with pytest.raises(FileExistsError):
        T.reserve_output(out, 'TRAIN_FIT', 'CALIBRATION')
    assert (out/'STARTED.json').read_bytes() == old


def test_fit_changed_file_identity_refused(tmp_path, monkeypatch):
    fit = tmp_path/'fit'; fit.mkdir()
    f = fit/'evidence'; f.write_bytes(b'original')
    receipt = {'status':'COMPLETE', 'phase':'CALIBRATION', 'protocol':T.PROTOCOL,
               'source_binding':{'x':'y'}, 'files_sha256':{'evidence': T.sha_file(f)}}
    (fit/'FIT_RECEIPT.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(T, 'source_binding', lambda: {'x':'y'})
    f.write_bytes(b'changed')
    with pytest.raises(ValueError, match='file identity'):
        T.load_fit(fit, 'CALIBRATION')


def test_nonfinite_sparse_evidence_refused():
    with pytest.raises(ValueError, match='nonfinite'):
        T.BackgroundRanks(np.zeros(2, dtype=np.float32)).rank([0], [np.nan], 1)
    with pytest.raises(ValueError, match='background'):
        T.BackgroundRanks(np.array([0, np.inf]))


def test_negative_result_is_not_relabelled_success():
    assert T.PROTOCOL['selection'] == 'NONE'
    assert T.PROTOCOL['test_supported'] is False
    assert T.PROTOCOL['label'] == 'POST_HOC_EXPLORATORY_DESCRIPTIVE'


def synthetic_provider():
    from types import SimpleNamespace
    from v4eval.manifest import build_eval_manifest
    did = np.array([10,20,30,40,50], dtype=np.int64)
    cols = {'decision_id':did, 'eval_user_key':np.array([0,0,1,1,2]),
            'eval_mask':np.array([True,True,True,False,True]),
            'target_class':np.array([0,1,-1,-1,2]), 'target_oov':np.array([False,False,True,False,False]),
            'censor_class':np.array(['NONE','NONE','NONE','CEN_NO_NEXT','NONE']),
            'v4_split':np.full(5,'VALIDATION')}
    manifest = build_eval_manifest({'manifest_id':'SYNTHETIC_VALIDATION', 'catalogue_K':4,
                                    'candidate_order_hash':'SYNTHETIC'}, cols)
    inputs = batch([[3],[3],[2],[3]])
    inputs['decision_id'] = did[[0,1,2,4]]
    inputs['target_class'] = cols['target_class'][[0,1,2,4]]
    return SimpleNamespace(manifest=manifest, batches=lambda size: [inputs]), inputs


def test_full_artifact_unchanged_metric_evaluator_preserves_oov_and_client_macro():
    from v4eval import frozen_config as fc
    from v4eval.evaluate import evaluate
    from v4base.scoring import build_rows
    provider, _ = synthetic_provider()
    models = {method:T.TrainingFreeScorer(method, sparse.csr_matrix((4,4), dtype=np.float32), np.zeros(4,dtype=np.float32)) for method in T.METHODS}
    cfg = fc.load(); fc.apply_runtime_flags(cfg)
    arts = T.predict_sparse_models(models, provider, cfg, 'SYNTHETIC')
    art = arts['MC']
    np.testing.assert_array_equal(art.ranks, [1,2,-1,-1,3])
    np.testing.assert_array_equal(art.n_tied, [3,3,-1,-1,3])
    result = evaluate(provider.manifest, art, config=cfg, evaluator_code_sha256='SYNTHETIC')
    assert result['E2E']['n_decisions'] == 4
    assert result['E2E']['n_users'] == 3
    assert result['RANKABLE']['n_decisions'] == 3
    assert result['E2E']['micro']['mrr@20'] == pytest.approx((1+1/2+1/3)/4)
    assert result['E2E']['macro']['mrr@20'] == pytest.approx((.75+0+1/3)/3)
    key = {field:'SYNTHETIC' for field in ('data_protocol_hash','feature_contract_hash','label_policy_hash','population_hash','candidate_order_hash')}
    rows = build_rows(result, dict(key, split='VALIDATION'), method='MC')
    assert len(rows) == 48
    assert all(row['run_label']=='BASELINE' for row in rows)


@pytest.mark.parametrize('change', ['reorder','duplicate','wrong_target','non_tail_short'])
def test_full_stream_identity_and_batch_controls_fail(change):
    from v4eval import frozen_config as fc
    provider, inputs = synthetic_provider()
    models = {method:T.TrainingFreeScorer(method, sparse.csr_matrix((4,4), dtype=np.float32), np.zeros(4,dtype=np.float32)) for method in T.METHODS}
    if change == 'reorder': inputs['decision_id'] = inputs['decision_id'][::-1].copy()
    elif change == 'duplicate': inputs['decision_id'][1] = inputs['decision_id'][0]
    elif change == 'wrong_target': inputs['target_class'][0] = 3
    else:
        provider.batches = lambda size: [{key:val[:2] for key,val in inputs.items()}, {key:val[2:] for key,val in inputs.items()}]
    cfg = fc.load(); fc.apply_runtime_flags(cfg)
    from v4eval.errors import AlignmentError
    from v4eval.frozen_config import FrozenConfigError
    with pytest.raises((AlignmentError, FrozenConfigError)):
        T.predict_sparse_models(models, provider, cfg, 'SYNTHETIC')


@pytest.mark.parametrize('operation', ['fit','evaluate'])
@pytest.mark.parametrize('issue', ['stale_registry','alias_test','wrong_train'])
def test_registry_pin_and_role_allowlist_refuse_before_provider_or_output(tmp_path, monkeypatch, operation, issue):
    from v4int import providers, verify
    root = tmp_path/'data_root'
    registry = root/'contracts/manifest_registry/REGISTRY_V2.json'
    registry.parent.mkdir(parents=True)
    role = {'train':'MAIN_TRAIN_5_100_131072', 'eval':'MAIN_VALIDATION_5_100_131072'}
    if issue == 'alias_test': role['eval'] = 'MAIN_TEST_5_100_131072'
    if issue == 'wrong_train': role['train'] = 'PRETRAIN_TRAIN_5_100_NONMAIN'
    registry.write_text(json.dumps({'roles':{'MAIN':role}}))
    if issue != 'stale_registry':
        # Remove only the SHA defense in this fixture, retaining the exact-ID guard.
        monkeypatch.setattr(T, 'REGISTRY_SHA256', T.sha_file(registry))
    def forbidden(*args, **kwargs):
        raise AssertionError('provider/data-path construction before registry guard')
    monkeypatch.setattr(providers, 'StoreTrainProvider', forbidden)
    monkeypatch.setattr(providers, 'StoreEvalProvider', forbidden)
    monkeypatch.setattr(verify.DataPaths, 'default', forbidden)
    out = tmp_path/'new_output'
    expected = 'registry file identity' if issue == 'stale_registry' else 'role IDs differ'
    with pytest.raises(ValueError, match=expected):
        if operation == 'fit': T.fit('MAIN', root, out)
        else: T.evaluate_validation('MAIN', root, tmp_path/'missing_fit', out)
    assert not out.exists()


def test_actual_registry_roles_and_sha_are_bound_without_reading_dataset():
    paths, registry, train, evaluation = T.paths_and_role('CALIBRATION', None)
    assert T.sha_file(registry) == T.REGISTRY_SHA256
    assert (train,evaluation) == T.ROLE_IDS['CALIBRATION']
    assert T.PROTOCOL['registry_file_sha256'] == T.REGISTRY_SHA256

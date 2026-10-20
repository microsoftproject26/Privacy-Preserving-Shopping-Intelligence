"""Canary identity, actual local updates, score and privacy-transcript controls."""
import json
from types import SimpleNamespace
import numpy as np
import ppsi_env  # noqa: F401
import pytest
import torch
from experiment_extensions.privacy import canary_data as C
from fl.client import ClientData, LocalSolver, client_update, valid_rows
from fl.families import SASRecAdapter
from fl.numerics import enforce_strict_fp32, state_digest
from snap_v1.sasrec_data import check_right_padded
from v4models.sasrec_v2 import SASRecConfig, build_sasrec


@pytest.fixture(autouse=True)
def strict():
    enforce_strict_fp32(warn_only=False)
    torch.set_num_threads(1)


def definitions():
    return C.CanaryDefinitions(np.array([[0, 1, 2], [3, 4, 1]]), np.array([4, 2]), 5, 8)


def model(seed=1):
    cfg = SASRecConfig('canary-toy', 'item_only', 8, 1, 2, dropout=.2, max_len=8)
    return build_sasrec(cfg, seed=seed, device='cpu', product_idx_of_class=np.arange(5)+3,
                        vocab_sizes={'item_tokens': 8})


def test_positive_count_long_tail_and_deterministic_class_ties():
    counts = np.array([0, 4, 1, 1, 9, 0, 2, 10])
    assert C.long_tail_pool(counts, .5).tolist() == [2, 3, 6]
    assert 0 not in C.long_tail_pool(counts, 1)
    with pytest.raises(ValueError):
        C.long_tail_pool(np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError):
        C.long_tail_pool(np.array([-1, 2]))


def test_public_definitions_reference_distribution_and_rng_isolation():
    np.random.seed(17)
    state = np.random.get_state()
    tstate = torch.get_rng_state().clone()
    c, r = C.public_definitions(np.array([1, 3, 4]), 5, n_canaries=20, n_references=100,
                               history_length=3, max_len=8)
    d, s = C.public_definitions(np.array([1, 3, 4]), 5, n_canaries=20, n_references=100,
                               history_length=3, max_len=8)
    assert c.commitment() == d.commitment() and r.commitment() == s.commitment()
    assert np.isin(c.histories, [1, 3, 4]).all() and np.isin(r.targets, [1, 3, 4]).all()
    assert c.count == 20 and r.count == 100
    current = np.random.get_state()
    assert state[0] == current[0] and np.array_equal(state[1], current[1]) and state[2:] == current[2:]
    assert torch.equal(tstate, torch.get_rng_state())


def test_definitions_copy_private_mutable_inputs_and_pin_public_content():
    h, t = np.array([[0, 1]]), np.array([2])
    a = C.CanaryDefinitions(h, t, 5, 8)
    pin = a.commitment()
    h[:] = 4
    t[:] = 3
    assert a.commitment() == pin and a.targets[0] == 2
    with pytest.raises(ValueError):
        a.histories[0, 0] = 3


@pytest.mark.parametrize('histories,targets', [([[0, 1]], [-1]), ([[0, 5]], [1]),
                                               ([[0, 1]], [5]), ([[0, 1]], [1, 2]),
                                               ([[0.2, 1]], [1])])
def test_invalid_definition_classes_alignment_and_dtype(histories, targets):
    with pytest.raises(ValueError):
        C.CanaryDefinitions(np.array(histories), np.array(targets), 5, 8)


def test_real_right_padding_class_offset_and_ten_copies():
    b = definitions().batch(np.array([1]), repeats=10)
    check_right_padded(b)
    assert b['item_tokens'].shape == (10, 8)
    assert torch.equal(b['item_tokens'][0], torch.tensor([6, 7, 4, 0, 0, 0, 0, 0]))
    assert b['target_class'].tolist() == [2]*10 and valid_rows(b).numel() == 10
    assert b['position_ids'][0].tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
    empty = definitions().batch(np.array([], dtype=np.int64), repeats=10)
    assert all(v.shape[0] == 0 for v in empty.values()) and valid_rows(empty).numel() == 0


def test_private_fair_coins_use_independent_system_entropy(monkeypatch):
    calls = []
    monkeypatch.setattr(C.secrets, 'token_bytes', lambda n: calls.append(n) or bytes(range(n)))
    state = torch.get_rng_state().clone()
    a = C.private_fair_signs(8)
    assert calls == [8] and a.tolist() == [-1, 1, -1, 1, -1, 1, -1, 1]
    assert torch.equal(state, torch.get_rng_state())


def roster(signs=(1, -1)):
    real = ClientData('real', definitions().batch(np.array([0])))
    return C.FixedRosterCanaries(['real'], lambda k: real, definitions(), np.array(signs),
                                train_manifest_id=C.CALIB_TRAIN), real


def test_fixed_roster_and_receipt_identical_across_all_membership_changes():
    a, real = roster((1, -1))
    for signs in [(1, 1), (-1, -1), (-1, 1)]:
        b, _ = roster(signs)
        assert a.keys == b.keys and a.public_receipt() == b.public_receipt()
    assert a.get_client('real') is real
    assert valid_rows(a.get_client(a.keys[1]).examples).numel() == 10
    assert valid_rows(a.get_client(a.keys[2]).examples).numel() == 0
    receipt = json.dumps(a.public_receipt())
    assert 'private_signs' not in receipt and 'inclusion_count' not in receipt and 'seed' not in receipt


@pytest.mark.parametrize('role', ['MAIN_TRAIN_5_100_131072', 'RETRO_TEST', 'CALIB_TEST', 'CALIBRATION'])
def test_no_other_training_or_eval_manifest(role):
    with pytest.raises(ValueError, match='CALIB TRAIN'):
        C.FixedRosterCanaries([], lambda k: None, definitions(), np.array([1, -1]), train_manifest_id=role)


def test_collisions_unknown_clients_and_invalid_membership_refused():
    with pytest.raises(ValueError, match='collide'):
        C.FixedRosterCanaries([C.PREFIX+'000000'], lambda k: None, definitions(), np.array([1, -1]),
                              train_manifest_id=C.CALIB_TRAIN)
    with pytest.raises(ValueError):
        C.FixedRosterCanaries([], lambda k: None, definitions(), np.array([1, 0]), train_manifest_id=C.CALIB_TRAIN)
    a, _ = roster()
    with pytest.raises(KeyError):
        a.get_client('unregistered-client')


def test_actual_SASRec_SGD_included_consumption_and_absent_zero_delta():
    a, _ = roster()
    adapter = SASRecAdapter(model())
    theta = adapter.broadcast_state(clone=True)
    solver = LocalSolver(lr=.1, optimizer='sgd_m', passes=2, batch_size=16)
    absent = client_update(adapter, theta, a.get_client(a.keys[2]), solver, round_idx=0, seed=2026, clone_upload=True)
    before = {k: theta[k] for k in adapter.manifest.shared_keys}
    assert absent.n_consumed == absent.steps == 0 and state_digest(absent.upload) == state_digest(before)
    included = client_update(adapter, theta, a.get_client(a.keys[1]), solver, round_idx=0, seed=2026, clone_upload=True)
    assert included.n_valid == 10 and included.n_consumed == 20 and included.steps == 2
    assert state_digest(included.upload) != state_digest(before)


def test_full_catalogue_scores_equal_dense_CE_and_restore_flags_RNG_state():
    m = model()
    m.train()
    m.blocks[0].ln1.eval()
    flags = [v.training for v in m.modules()]
    state = {k: v.clone() for k, v in m.state_dict().items()}
    rng = torch.get_rng_state().clone()
    batch = definitions().batch(np.array([0, 1]))
    score = C.full_catalogue_logprob(m, batch, 5)
    m.eval()
    logits = m(item_tokens=batch['item_tokens'], lengths=batch['lengths'], attention_mask=batch['attention_mask'])
    oracle = -torch.nn.functional.cross_entropy(logits, batch['target_class'], reduction='none')
    torch.testing.assert_close(score, oracle, atol=1e-6, rtol=0)
    for module, flag in zip(m.modules(), flags):
        module.training = flag
    C.full_catalogue_logprob(m, batch, 5)
    assert [v.training for v in m.modules()] == flags and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(state[k], v) for k, v in m.state_dict().items())


def test_scorer_never_receives_labels_and_refuses_partial_or_invalid_logits():
    class Fixed(torch.nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
        def forward(self, **kw):
            assert set(kw) == {'item_tokens', 'lengths', 'attention_mask'}
            z = torch.ones(len(kw['lengths']), 5)
            if self.kind == 'partial':
                return z[:, :4]
            if self.kind == 'nonfinite':
                z[0, 0] = float('nan')
            return z.double() if self.kind == 'double' else z
    b = definitions().batch(np.array([0, 1]))
    torch.testing.assert_close(C.full_catalogue_logprob(Fixed('ok'), b, 5), torch.full((2,), -np.log(5), dtype=torch.float32))
    for kind in ['partial', 'nonfinite', 'double']:
        m = Fixed(kind)
        m.train()
        with pytest.raises(ValueError):
            C.full_catalogue_logprob(m, b, 5)
        assert m.training


def test_identical_models_zero_drift_and_full_score_batching_equivalence():
    m, n = model(1), model(2)
    d = definitions()
    assert np.array_equal(C.drift_scores(m, m, d), np.zeros(2))
    np.testing.assert_allclose(C.drift_scores(m, n, d, batch_size=1), C.drift_scores(m, n, d, batch_size=2), atol=1e-6)


def test_finite_reference_ties_bounds_and_no_full_secret_space_extrapolation():
    refs = np.array([1., 2., 2., 3.])
    c = np.array([0., 2., 3., 4.])
    e = C.finite_reference_exposure(c, refs)
    np.testing.assert_allclose(e, -np.log2(np.array([5, 4, 2, 1])/5))
    assert e.min() == 0 and e.max() == np.log2(len(refs)+1)
    for values in [[], [np.nan], [np.inf]]:
        with pytest.raises(ValueError):
            C.finite_reference_exposure(np.array(values), refs)
        with pytest.raises(ValueError):
            C.finite_reference_exposure(c, np.array(values))


def test_TF32_and_autocast_refused_before_model_forward(monkeypatch):
    from fl.numerics import NumericsError
    m = model()
    monkeypatch.setattr(m, 'forward', lambda **kw: (_ for _ in ()).throw(AssertionError('forward started')))
    b = definitions().batch(np.array([0]))
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', True)
    with pytest.raises(NumericsError):
        C.full_catalogue_logprob(m, b, 5)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(ValueError, match='autocast'):
            C.full_catalogue_logprob(m, b, 5)


def test_warn_only_nondeterministic_runtime_refused_before_forward(monkeypatch):
    m = model()
    monkeypatch.setattr(m, 'forward', lambda **kw: (_ for _ in ()).throw(AssertionError('forward started')))
    b = definitions().batch(np.array([0]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.raises(ValueError, match='STRICT'):
        C.full_catalogue_logprob(m, b, 5)
    torch.use_deterministic_algorithms(False)
    with pytest.raises(ValueError, match='STRICT'):
        C.full_catalogue_logprob(m, b, 5)
    enforce_strict_fp32(warn_only=False)


@pytest.mark.parametrize('signs', [[True, True], [1., -1.]])
def test_noninteger_private_signs_refused(signs):
    with pytest.raises(ValueError):
        roster(signs)

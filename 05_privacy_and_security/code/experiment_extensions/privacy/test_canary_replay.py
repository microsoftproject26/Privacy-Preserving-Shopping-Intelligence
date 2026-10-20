"""Actual CPU FLRun bit-flip, entropy and release controls; no real providers."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import ppsi_env  # noqa: F401,E402
from collections import OrderedDict
from dataclasses import replace
import numpy as np
import pytest
import torch

from . import canary_replay as R
from .canary_data import CanaryDefinitions, PREFIX
from fl.client import ClientData
from fl.numerics import enforce_strict_fp32, state_digest
from fl.synthetic import client_examples, make_tiny_adapter


@pytest.fixture(autouse=True)
def strict():
    torch.set_num_threads(1)
    enforce_strict_fp32(warn_only=False)
    yield
    assert not torch.cuda.is_initialized()


def fixture(signs, *, mode='UNFROZEN', rounds=4, passes=2, empty_real=False, group=4):
    defs = CanaryDefinitions(np.array([[1, 2, 3], [5, 6, 7]]), np.array([3, 7]), 12, 8)
    clients = [ClientData('toy_real::' + str(i), client_examples(0 if empty_real else 3 + 2*i,
        12, 8, np.random.default_rng(i+17), invalid_frac=0)) for i in range(2)]
    server = make_tiny_adapter(K=12, d=4, seed=11, tied=False)
    worker = make_tiny_adapter(K=12, d=4, seed=31, tied=False)
    schedule = R.ToySchedule(mode=mode, rounds=rounds, passes=passes, group_size=group)
    entropy = R.PrivateEntropy(771, 881) if mode == 'DP' else None
    return (clients, defs, np.asarray(signs, dtype=np.int8)), dict(adapter=server, workers=[worker],
        schedule=schedule, entropy=entropy)


@pytest.mark.parametrize('mode', ['FROZEN', 'UNFROZEN', 'DP'])
def test_actual_FLRun_bitflip_keeps_plan_full_LR_sequence_and_fixed_endpoint(mode):
    args, kwargs = fixture([-1, -1], mode=mode)
    before = torch.get_rng_state().clone()
    absent = R.replay_synthetic(*args, **kwargs)
    assert torch.equal(before, torch.get_rng_state())
    args2, kwargs2 = fixture([1, -1], mode=mode)
    included = R.replay_synthetic(*args2, **kwargs2)
    a, b = absent._run, included._run
    assert a.plan.keys == b.plan.keys and a.plan.digest() == b.plan.digest()
    assert all(a.plan.round_keys(r) == b.plan.round_keys(r) for r in range(4))
    assert a.cfg.digest() == b.cfg.digest() and a.state.cursor == b.state.cursor == 4
    assert a.state.done and b.state.done and a.state.evals == b.state.evals == []
    assert a.lr_log == b.lr_log and [row[4] for row in a.lr_table['rows']] == [row[4] for row in b.lr_table['rows']]
    assert a.n_decisions_total == b.n_decisions_total == 8 + 2*10
    assert a.n_valid_by_key[PREFIX+'000000'] == 0 and b.n_valid_by_key[PREFIX+'000000'] == 10
    assert a.counters.exposures == 4*2*8 and b.counters.exposures == a.counters.exposures + 4*2*10
    assert state_digest(a.server.broadcast()) != state_digest(b.server.broadcast())
    if mode == 'DP':
        assert a.cfg.dp.denominator == b.cfg.dp.denominator == 4
        assert a.plan.sampling == 'poisson' and a.cfg.dropout_p == 0
    else:
        assert a.plan.sampling == 'sweep' and a.cfg.dp is None


def test_shadow_ten_absent_counts_fail_real_runtime_planned_consumed_guard():
    args, kwargs = fixture([-1, -1])
    private = R.replay_synthetic(*args, **kwargs)
    # Build a fresh actual run with the same client path but the WRONG shadow
    # count in its private runtime map. No wrapper can call this a valid replay.
    from fl.runtime import FLRun
    from fl.server import Server
    adapter = make_tiny_adapter(K=12, d=4, seed=11, tied=False)
    worker = make_tiny_adapter(K=12, d=4, seed=31, tied=False)
    run = private._run
    shadow = dict(run.n_valid_by_key, **{PREFIX+'000000': 10, PREFIX+'000001': 10})
    bad = FLRun(run.cfg, run.plan, Server(adapter, init_sha256=state_digest(adapter.broadcast_state())),
        run.get_client, run.n_decisions_total, workers=[worker], n_valid_by_key=shadow)
    from fl.runtime import LRTableError
    with pytest.raises(LRTableError, match='planned LR table'): bad.step()


def test_original_nonDP_all_zero_cohort_is_explicit_refusal_not_shadow_training():
    from fl.aggregate import EmptyRoundError
    # Full sweep contains positive real rows, so the LR table is valid. A
    # scheduled single absent canary still has zero aggregate weight.
    args, kwargs = fixture([-1, -1], group=1)
    with pytest.raises(EmptyRoundError, match='zero total weight'):
        R.replay_synthetic(*args, **kwargs)


def test_DP_all_zero_and_empty_rounds_release_noise_original_path(monkeypatch):
    from fl.runtime import FLRun
    original_step = FLRun.step
    private_steps = []
    def observe(run):
        before, exposures = run.server.digest(), run.counters.exposures
        out = original_step(run)
        private_steps.append((out, before, run.server.digest(), run.counters.exposures - exposures))
        return out
    monkeypatch.setattr(FLRun, 'step', observe)
    args, kwargs = fixture([-1, -1], mode='DP', rounds=16, group=1)
    result = R.replay_synthetic(*args, **kwargs)
    zero = [s for s in private_steps if s[3] == 0]
    assert zero and result._run.counters.noop_visits > 0
    assert any(s[0]['dp'].get('noise_only') is True for s in zero)
    assert any(s[0]['dp'].get('n_noop', 0) > 0 for s in zero)
    assert all(s[1] != s[2] and s[0]['dp']['denominator'] == 1 for s in zero)
    assert result._run.cfg.dp.denominator == 1 and result._run.state.done


@pytest.mark.parametrize('mode', ['UNFROZEN', 'DP'])
def test_whole_plan_without_any_exposures_has_explicit_inherited_LR_refusal(mode):
    from fl.runtime import LRTableError
    args, kwargs = fixture([-1, -1], mode=mode, empty_real=True)
    with pytest.raises(LRTableError, match='no round of the plan has planned exposures'):
        R.replay_synthetic(*args, **kwargs)


def test_frozen_item_tables_remain_exact_while_unfrozen_tables_move():
    args, kwargs = fixture([1, -1], mode='FROZEN')
    initial = kwargs['adapter'].broadcast_state(clone=True)
    result = R.replay_synthetic(*args, **kwargs)
    from fl.runtime import FROZEN_ITEM_KEYS
    final = result._run.server.broadcast()
    assert all(torch.equal(initial[k], final[k]) for k in FROZEN_ITEM_KEYS)
    assert any(not torch.equal(initial[k], final[k]) for k in final if k not in FROZEN_ITEM_KEYS and final[k].is_floating_point())
    args, kwargs = fixture([1, -1])
    initial = kwargs['adapter'].broadcast_state(clone=True)
    result = R.replay_synthetic(*args, **kwargs)
    assert any(not torch.equal(initial[k], result._run.server.broadcast()[k]) for k in FROZEN_ITEM_KEYS)


def test_final_model_release_whitelist_excludes_private_fields_and_copies_state():
    args, kwargs = fixture([1, -1], mode='DP')
    result = R.replay_synthetic(*args, **kwargs)
    public = result.model_only_payload()
    assert set(public) == {'format', 'theta', 'mechanism', 'fixed_cell_accountant_applicable',
                           'private_tuning_accounted', 'privacy_certified'}
    assert set(public['mechanism']) == {'scope', 'mode', 'rounds', 'local_passes', 'lr_shape',
                                      'optimizer', 'fixed_denominator'}
    assert not public['privacy_certified'] and not public['fixed_cell_accountant_applicable']
    saved = OrderedDict((k, v.clone()) for k, v in public['theta'].items())
    with torch.no_grad():
        next(result._run.server.adapter.module.parameters()).add_(1.)
    assert all(torch.equal(saved[k], v) for k, v in public['theta'].items())
    with torch.no_grad(): next(result._run.server.adapter.module.parameters()).fill_(float('nan'))
    with pytest.raises(R.Refused): result.model_only_payload()


def test_private_Gaussian_matches_inherited_law_and_ignores_public_seed():
    from fl import dp
    template = OrderedDict(a=torch.zeros(13), b=torch.zeros(2, 7))
    original = dp.noise_like
    expected = original(list(template), template, seed=881, round_idx=2, std=.3)
    public = original(list(template), template, seed=2026, round_idx=2, std=.3)
    ambient = torch.get_rng_state().clone()
    with R.private_gaussian(R.PrivateEntropy(771, 881)):
        first = dp.noise_like(list(template), template, seed=2026, round_idx=2, std=.3)
        second = dp.noise_like(list(template), template, seed=2027, round_idx=2, std=.3)
        assert all(torch.equal(expected[k], first[k]) and torch.equal(first[k], second[k]) for k in template)
        assert any(not torch.equal(public[k], first[k]) for k in template)
    assert dp.noise_like is original and torch.equal(ambient, torch.get_rng_state())


def test_private_Gaussian_restores_on_exception_and_rejects_nesting():
    from fl import dp
    original = dp.noise_like
    with pytest.raises(RuntimeError):
        with R.private_gaussian(R.PrivateEntropy(771, 881)):
            with pytest.raises(R.Refused, match='nested'):
                with R.private_gaussian(R.PrivateEntropy(11, 12)): pass
            raise RuntimeError('intentional failure')
    assert dp.noise_like is original


def test_private_Poisson_exact_original_sampling_new_population_and_resume_identity():
    from fl.participation import ParticipationPlan
    keys = [f'toy_public::{i:05}' for i in range(33768)]
    plan = R.private_poisson_plan(keys, group_size=1024, entropy=R.PrivateEntropy(771, 881),
                                  public_roster_id='EID_E2_PUBLIC_ROSTER:CONTROL')
    expected = ParticipationPlan.build(keys, seed=771, manifest_hash='EID_E2_PUBLIC_ROSTER:CONTROL',
                                       group_size=1024, sampling='poisson')
    repeat = R.private_poisson_plan(keys, group_size=1024, entropy=R.PrivateEntropy(771, 999),
                                    public_roster_id='EID_E2_PUBLIC_ROSTER:CONTROL')
    assert plan.inclusion_probability == 1024/33768 and plan.n_clients == 33768
    for r in (0, 1, 383): assert plan.round_keys(r) == expected.round_keys(r) == repeat.round_keys(r)


def test_new_fixed_cell_accountant_is_STANDARD_exact_population_not_old_noise():
    from fl.accountant import epsilon
    record = R.dp_accountant()
    assert (record['N'], record['m'], record['T'], record['q']) == (33768, 1024, 384, 1024/33768)
    assert record['achieved_epsilon'] == epsilon(1024, 33768, 384, record['z'], 5e-6) <= 8
    assert record['conversion'].startswith('standard')
    assert epsilon(1024, 33768, 384, .5642, 5e-6) > 8
    record['N'] = 1
    assert R.dp_accountant()['N'] == 33768
    declaration = R.public_study_declaration()
    assert declaration['potential_decisions_normalizer'] == 656067
    assert declaration['dp_expected_client_passes'] == 384*1024/33768*2
    assert not declaration['MAIN_allowed'] and not declaration['TEST_allowed']


@pytest.mark.parametrize('role', ['MAIN', 'MAIN_TRAIN_5_100_131072', 'TEST', 'CALIB_VALIDATION_5_100_32768'])
def test_wrong_role_refuses_before_numerical_or_private_inputs(role):
    with pytest.raises(R.Refused):
        R.replay_synthetic(None, None, None, adapter=None, workers=None, role=role)


@pytest.mark.parametrize('sampling,noise', [(1, 1), (True, 2), (-1, 2), (1, 2**63), (1., 2)])
def test_private_entropy_invalid_or_reused_stream_refused(sampling, noise):
    with pytest.raises(R.Refused): R.PrivateEntropy(sampling, noise)


def test_no_old_manifest_impersonation_or_production_models_admitted():
    with pytest.raises(R.Refused):
        R.private_poisson_plan(['a','b'], group_size=1, entropy=R.PrivateEntropy(1,2),
                               public_roster_id='OLD_MANIFEST_SHA')
    args, kwargs = fixture([1,-1])
    with pytest.raises(R.Refused): R.replay_synthetic(*args, **dict(kwargs, adapter=object()))
    args[0][0].key = 'real_customer_key'
    with pytest.raises(R.Refused): R.replay_synthetic(*args, **kwargs)

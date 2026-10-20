"""T30/T31 -- LO: no messages, theta0 hash identity, and replay-checksum repeatability.

Derived from 07_LOCAL_ONLY_AND_DEVICE.md SS1-SS2 ("Each client starts from the SAME permitted theta0 ...
learns solely on its own TRAIN records, with no aggregation or shared refresh" / "A procedural lock freezes
initialization, seed, training inputs, algorithm and budget BEFORE held-out queries are evaluated") and the
fl.local_only public contract. Not read from `code/tests_fl`.
"""
from __future__ import annotations

import torch

from fl.client import LocalSolver
from fl.comm import LEDGER
from fl.local_only import LORecipe, ProvenanceError, run_local_only
from fl.numerics import state_digest

from _toolkit import fresh_adapter, one_client


# ---------------------------------------------------------------------------------------- D1: no messages
def test_lo_run_leaves_the_comm_ledger_untouched():
    before = LEDGER.snapshot()
    adapter = fresh_adapter(K=8, d=4, tied=True, seed=1)
    theta0 = adapter.broadcast_state(clone=True)
    recipe = LORecipe(theta0_digest=state_digest(theta0), seed=1, passes=2, eval_at=(2,),
                      solver=LocalSolver(lr=1e-3, passes=2, batch_size=16))
    client = one_client("lo-client", n=6, K=8, L=5, seed=2)
    run_local_only(adapter, theta0, client, recipe)
    after = LEDGER.snapshot()
    assert before == after, "07 SS1-2 / local_only.py: 'there is NO channel' -- the ledger must be untouched (T30)"


# ---------------------------------------------------------------------------------------- D2: theta0 hash identity
def test_lo_refuses_a_theta0_that_does_not_match_the_recipe_digest():
    adapter = fresh_adapter(K=8, d=4, tied=True, seed=1)
    theta0 = adapter.broadcast_state(clone=True)
    wrong_digest = state_digest(fresh_adapter(K=8, d=4, tied=True, seed=2).broadcast_state())
    recipe = LORecipe(theta0_digest=wrong_digest, seed=1, passes=2, eval_at=(2,),
                      solver=LocalSolver(lr=1e-3, passes=2, batch_size=16))
    client = one_client("lo-client", n=6, K=8, L=5, seed=2)
    try:
        run_local_only(adapter, theta0, client, recipe)
        assert False, "must refuse a theta0/recipe digest mismatch (07 SS2 procedural lock, T30/T31)"
    except ProvenanceError:
        pass


def test_lo_accepts_the_matching_theta0_digest():
    adapter = fresh_adapter(K=8, d=4, tied=True, seed=1)
    theta0 = adapter.broadcast_state(clone=True)
    recipe = LORecipe(theta0_digest=state_digest(theta0), seed=1, passes=2, eval_at=(2,),
                      solver=LocalSolver(lr=1e-3, passes=2, batch_size=16))
    client = one_client("lo-client", n=6, K=8, L=5, seed=2)
    res = run_local_only(adapter, theta0, client, recipe)   # must not raise
    assert res.recipe_digest == recipe.digest()


# ---------------------------------------------------------------------------------------- D3: replay checksum
def test_lo_replay_is_checksum_repeatable_on_a_fresh_adapter():
    theta0 = fresh_adapter(K=8, d=4, tied=True, seed=1).broadcast_state(clone=True)
    digest0 = state_digest(theta0)
    recipe = LORecipe(theta0_digest=digest0, seed=5, passes=4, eval_at=(2, 4),
                      solver=LocalSolver(lr=1e-3, passes=4, batch_size=16))
    client = one_client("replay-client", n=10, K=8, L=5, seed=3)

    def probe_eval(adapter, budget):
        # a deterministic scalar computed from the current shared state (eval mode; no RNG dependence)
        return round(float(sum(p.abs().sum() for p in adapter.shared_parameters())), 6)

    run_a = run_local_only(fresh_adapter(K=8, d=4, tied=True, seed=1), theta0, client, recipe, eval_hook=probe_eval)
    run_b = run_local_only(fresh_adapter(K=8, d=4, tied=True, seed=1), theta0, client, recipe, eval_hook=probe_eval)

    assert run_a.final_digest == run_b.final_digest, "07 SS2: replay must be checksum-repeatable (T31)"
    assert run_a.evals == run_b.evals, "07 SS2: eval-hook outputs must repeat identically on replay (T31)"
    assert run_a.n_consumed == run_b.n_consumed == recipe.passes * client.examples["target_class"].shape[0]


# ---------------------------------------------------------------------------------------- D4: no-label client
def test_lo_no_label_client_is_recorded_and_evaluated_at_theta0():
    theta0 = fresh_adapter(K=8, d=4, tied=True, seed=1).broadcast_state(clone=True)
    recipe = LORecipe(theta0_digest=state_digest(theta0), seed=5, passes=3, eval_at=(1, 3),
                      solver=LocalSolver(lr=1e-3, passes=3, batch_size=16))
    client = one_client("no-label", n=8, K=8, L=5, seed=1, invalid_frac=1.0)   # every target_class == -1

    calls = []

    def probe_eval(adapter, budget):
        calls.append(budget)
        return budget

    adapter = fresh_adapter(K=8, d=4, tied=True, seed=1)
    res = run_local_only(adapter, theta0, client, recipe, eval_hook=probe_eval)
    assert res.no_label is True, "07 SS1: a client with no supervised label must be recorded, not removed (T30)"
    assert res.n_valid == 0 and res.n_consumed == 0 and res.steps == 0
    assert calls == [1, 3], "07 SS1: evaluation at every registered budget still runs, using theta0 (T30)"
    # theta0 is untouched: no optimizer update happened. `final_digest` covers SHARED keys only (extract_shared),
    # while `theta0` here is the full broadcast state (shared + buffers) -- compare the matching shared subset.
    theta0_shared_only = state_digest({k: theta0[k] for k in adapter.manifest.shared_keys})
    assert res.final_digest == theta0_shared_only


# ---------------------------------------------------------------------------------------- tail-batch, LO's own n_consumed
def test_lo_tail_batch_step_and_weight_accounting():
    theta0 = fresh_adapter(K=8, d=4, tied=True, seed=1).broadcast_state(clone=True)
    recipe = LORecipe(theta0_digest=state_digest(theta0), seed=5, passes=3, eval_at=(3,),
                      solver=LocalSolver(lr=1e-3, passes=3, batch_size=16))
    client = one_client("lo-tail", n=10, K=8, L=5, seed=1)     # ceil(10/16)=1 batch/pass (a lone tail batch)
    adapter = fresh_adapter(K=8, d=4, tied=True, seed=1)
    res = run_local_only(adapter, theta0, client, recipe)
    assert res.steps == 3, "07 SS1: passes(3) * ceil(10/16)(1) = 3 steps"
    assert res.n_consumed == 30, "07 SS1: n_consumed = passes * n_valid = 3*10 = 30"

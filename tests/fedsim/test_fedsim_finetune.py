"""Full-model on-device fine-tuning FT_C / FT_FA / FT_KD (finetune.py).

Proves (tiny synthetic family, CPU): the state after k passes of the one evaluated trajectory equals a separate k-pass
run bit for bit (so budgets 1 / 2 / 4 come from one trajectory); runs are deterministic per (seed, client) and paired
across arms; every client starts from the untouched base (no carry-over; the base tensors are never mutated); ALL
shared tensors are fine-tuned; the base provenance and a non-causal support are refused; the records carry n support
events, updates = passes x ceil(n / 16), bytes_up = 0, and no channel / ledger is touched; the budget grid is
enforced; support_cutoff is required (multi-query clients: the earliest decision_ts); server / live-worker models
are refused. Negative controls: the latest decision_ts as the cutoff, a runner that skips the base reload.
"""
from __future__ import annotations

import pytest
import torch
from fedsim_testkit import assert_raises, nc, one_client, tiny

from ppsi.fedsim.comm import LEDGER
from ppsi.fedsim.finetune import (
    CausalityError,
    FTProvenanceError,
    FTRecipe,
    FTRunner,
    run_ft_grid,
    support_cutoff_for,
)
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.server import Server

SEED = 2026
PEAK = 0.05


def base():
    a = tiny(3, tied=False)
    a.server_post_aggregate()
    return a.broadcast_state(clone=True), state_digest(a.broadcast_state())


BASE, BASE_D = base()
CUT = 10 ** 6                                          # a cutoff after every synthetic support timestamp


def _with_ts(client, ts):
    ex = dict(client.examples)
    ex["target_ts"] = torch.as_tensor(ts, dtype=torch.long)
    return type(client)(client.key, ex)


def _ts_client(key, n, seed, **kw):
    return _with_ts(one_client(key, n, seed=seed, **kw), list(range(n)))


CA = _ts_client("user-a", 37, 11, invalid_frac=0.1)
CB = _ts_client("user-b", 21, 12)


def recipe(**kw):
    d = {"arm": "FT_C", "base_digest": BASE_D, "seed": SEED, "peak_lr": PEAK, "lr_frac": 0.3}
    d.update(kw)
    return FTRecipe(**d)


def digest_hook(adapter, budget):
    return state_digest(adapter.extract_shared())


def ft(w, rec, client, hook=None, cutoff=CUT):
    return FTRunner(w, BASE, rec).run(client, hook, support_cutoff=cutoff)


def test_nested_budgets_equal_separate_runs():
    w = tiny(9, tied=False)
    full = ft(w, recipe(), CA, digest_hook)
    for b in (1, 2, 4):
        sep = ft(tiny(9, tied=False), recipe(passes=(b,)), CA, digest_hook)
        assert sep.evals[b] == full.evals[b] == sep.final_digest
        assert sep.updates[b] == full.updates[b]
    assert full.final_digest == full.evals[4]


def test_deterministic_and_paired_across_arms():
    r1 = ft(tiny(9, tied=False), recipe(), CA, digest_hook)
    r2 = ft(tiny(10, tied=False), recipe(), CA, digest_hook)
    assert r1.record() == r2.record()
    r3 = ft(tiny(9, tied=False), recipe(seed=SEED + 1), CA, digest_hook)
    assert r3.final_digest != r1.final_digest
    r4 = ft(tiny(9, tied=False), recipe(arm="FT_FA"), CA, digest_hook)
    assert r4.final_digest == r1.final_digest, "same base + seed + client: the arm label does not change the stream"


def test_each_client_starts_from_base_and_base_untouched():
    before = {k: v.clone() for k, v in BASE.items()}
    w = tiny(9, tied=False)
    run = FTRunner(w, BASE, recipe())
    run.run(CA, support_cutoff=CUT)
    b_after_a = run.run(CB, digest_hook, support_cutoff=CUT)
    b_alone = ft(tiny(9, tied=False), recipe(), CB, digest_hook)
    assert b_after_a.record() == b_alone.record()
    assert all(torch.equal(before[k], BASE[k]) for k in BASE)


def test_all_shared_tensors_fine_tuned():
    w = tiny(9, tied=False)
    ft(w, recipe(), CA)
    after = w.extract_shared()
    changed = [k for k in w.manifest.shared_keys if not torch.equal(after[k], BASE[k])]
    assert changed == list(w.manifest.shared_keys)


def test_records_bytes_updates_and_ledger():
    snap = LEDGER.snapshot()
    res = ft(tiny(9, tied=False), recipe(), CA, digest_hook)
    n = int((CA.examples["target_class"] >= 0).sum())
    assert res.n_support_events == n and res.bytes_up == 0
    assert res.updates == {b: b * (-(-n // 16)) for b in (1, 2, 4)}
    assert res.n_consumed == {b: b * n for b in (1, 2, 4)}
    assert res.bytes_down_once == tiny(9, tied=False).manifest.shared_bytes
    assert LEDGER.snapshot() == snap


def test_eval_base_and_no_support_client():
    empty = _ts_client("user-empty", 5, 1, invalid_frac=1.0)
    res = ft(tiny(9, tied=False), recipe(eval_base=True), empty, digest_hook)
    assert res.no_support and res.n_support_events == 0 and set(res.evals) == {0, 1, 2, 4}
    assert len(set(res.evals.values())) == 1 and res.evals[0] == state_digest(
        {k: BASE[k] for k in tiny(9, tied=False).manifest.shared_keys})


def test_provenance_refused():
    bad = {k: v.clone() for k, v in BASE.items()}
    k0 = next(iter(bad))
    bad[k0] = bad[k0] + 1.0
    assert_raises(FTProvenanceError, FTRunner, tiny(9, tied=False), bad, recipe())


def test_causal_support_enforced():
    n = CB.examples["target_class"].shape[0]
    ft(tiny(9, tied=False), recipe(), CB, cutoff=n)
    leaky = _with_ts(CB, list(range(n - 1)) + [n])
    assert_raises(CausalityError, ft, tiny(9, tied=False), recipe(), leaky, cutoff=n)
    no_ts = one_client("user-nots", 9, seed=5)
    assert_raises(CausalityError, ft, tiny(9, tied=False), recipe(), no_ts, cutoff=n)


def test_support_cutoff_required():
    assert_raises(CausalityError, ft, tiny(9, tied=False), recipe(), CB, cutoff=None)
    assert_raises(TypeError, FTRunner(tiny(9, tied=False), BASE, recipe()).run, CB)


def test_multi_query_cutoff_is_earliest_decision_ts():
    n = CB.examples["target_class"].shape[0]
    decisions = [n + 5, n - 3, n + 40]                   # the client's evaluated queries
    assert support_cutoff_for(decisions) == n - 3
    assert_raises(CausalityError, ft, tiny(9, tied=False), recipe(), CB, cutoff=support_cutoff_for(decisions))
    early = _with_ts(CB, [0] * n)
    ft(tiny(9, tied=False), recipe(), early, cutoff=support_cutoff_for(decisions))
    assert_raises(CausalityError, support_cutoff_for, [])


def test_server_and_live_worker_adapters_refused():
    srv_model = tiny(9, tied=False)
    Server(srv_model)
    assert_raises(ValueError, FTRunner, srv_model, BASE, recipe())
    w = tiny(9, tied=False)
    w.fl_role = "worker"
    assert_raises(ValueError, FTRunner, w, BASE, recipe())


def test_grid_and_cells():
    for bad in ({"lr_frac": 0.2}, {"passes": (3,)}, {"passes": (4, 2)}, {"arm": "FT_X"}, {"batch_size": 32}):
        with pytest.raises(ValueError):
            recipe(**bad)
    out = run_ft_grid(tiny(9, tied=False), BASE, CA, arm="FT_KD", base_digest=BASE_D, seed=SEED, peak_lr=PEAK,
                      eval_hook=digest_hook, support_cutoff=CUT)
    assert set(out["cells"]) == {(f, b) for f in (0.1, 0.3) for b in (1, 2, 4)}
    assert out["cells"][(0.1, 1)] != out["cells"][(0.3, 1)]
    assert [r["lr"] for r in out["records"]] == [0.1 * PEAK, 0.3 * PEAK]


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a multi-query client fine-tuned with the LATEST decision_ts accepts support after its earlier queries")
def test_nc_latest_decision_cutoff():
    n = CB.examples["target_class"].shape[0]
    decisions = [n + 5, n - 3, n + 40]
    assert_raises(CausalityError, ft, tiny(9, tied=False), recipe(), CB, cutoff=max(decisions))


@nc("a runner that does not reload the base carries client A's fine-tuning into client B")
def test_nc_no_base_reload():
    w = tiny(9, tied=False)
    run = FTRunner(w, BASE, recipe())
    run.run(CA, support_cutoff=CUT)
    w.load_state_ = lambda state: None                                  # the deliberate defect
    b_after_a = run.run(CB, digest_hook, support_cutoff=CUT)
    b_alone = ft(tiny(9, tied=False), recipe(), CB, digest_hook)
    assert b_after_a.final_digest == b_alone.final_digest

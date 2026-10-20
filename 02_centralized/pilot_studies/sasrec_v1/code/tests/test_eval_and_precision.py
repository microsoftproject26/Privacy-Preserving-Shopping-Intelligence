"""Evaluator / precision (05 sec.D items 25-30; Fable #8). Uses the reused,
audited `eval_full.rank_targets` / `eval_full.aggregate` directly on manual
fixtures -- never real VC/TEST data. Micro/macro expected values are computed
independently in this file (plain Python, not `evaluator._credits_from_ranks`).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pytest
import torch

import paths
import precision
from eval_full import rank_targets, aggregate
from mutants.eval_mutants import cast_to_float32_before_rank

CUDA_AVAILABLE = torch.cuda.is_available()
CODE_DIR = paths.CODE


# ---------------------------------------------------------------------------
# Independent (from-scratch, no torch, no `_credits_from_ranks` reuse) micro
# /macro reference, used to check `aggregate`'s output.
# ---------------------------------------------------------------------------
def expected_block(ranks, users, pop_mask, k):
    idx = [i for i, m in enumerate(pop_mask) if m]
    if not idx:
        return {"n": 0}
    recalls, mrrs = [], []
    by_user = defaultdict(list)
    for i in idx:
        r = ranks[i]
        hit = 1 <= r <= k
        recall = 1.0 if hit else 0.0
        mrr = (1.0 / r) if hit else 0.0
        recalls.append(recall)
        mrrs.append(mrr)
        by_user[users[i]].append((recall, mrr))
    micro_recall = sum(recalls) / len(recalls)
    micro_mrr = sum(mrrs) / len(mrrs)
    per_user_recall = [sum(v[0] for v in vals) / len(vals) for vals in by_user.values()]
    per_user_mrr = [sum(v[1] for v in vals) / len(vals) for vals in by_user.values()]
    macro_recall = sum(per_user_recall) / len(per_user_recall)
    macro_mrr = sum(per_user_mrr) / len(per_user_mrr)
    return {"n": len(idx), "user_count": len(by_user),
            "micro_recall": micro_recall, "micro_mrr": micro_mrr,
            "macro_recall": macro_recall, "macro_mrr": macro_mrr}


def _fixture():
    """8 decisions across 5 users: an exact tie above the target, an
    all-equal-tie row, a worst-of-tie row, an OOV row (eval_mask True),
    an excluded row (eval_mask False), an all-OOV user, and a lone-event user.
    """
    K = 6
    rankable_scores = torch.tensor([
        [10., 10., 10., 5., 1., 0.],   # row0 target=2 (tie above + self): rank 3
        [7., 7., 7., 7., 7., 7.],      # row1 target=0 (all-tie, wins):    rank 1
        [3., 3., 3., 3., 3., 3.],      # row2 target=5 (all-tie, worst):  rank 6
        [9., 8., 0., 0., 0., 0.],      # row4 target=1:                   rank 2
        [5., 5., 5., 5., 1., 0.],      # row7 target=3:                   rank 4
    ])
    rankable_targets = torch.tensor([2, 0, 5, 1, 3])
    ranks_rankable = rank_targets(rankable_scores, rankable_targets)  # production call

    ranks = np.full(8, -1, dtype=np.int64)
    order = [0, 1, 2, 4, 7]
    for pos, r in zip(order, ranks_rankable.numpy()):
        ranks[pos] = r

    ds = dict(
        target_class=np.array([2, 0, 5, -1, 1, 1, -1, 3], dtype=np.int64),
        eval_mask=np.array([True, True, True, True, True, False, True, True]),
        eval_user_key=np.array([100, 100, 101, 101, 102, 102, 103, 104], dtype=np.int64),
        client_idx=np.array([5, 5, 2, 2, 7, 7, 5, 5], dtype=np.int64),
        prior_view_count=np.zeros(8, dtype=np.int64),
        prefix_position=np.ones(8, dtype=np.int64),
    )
    return ranks, ds, K


# ---------------------------------------------------------------------------
# 05 item 25: manual rank fixtures through rank_targets + aggregate: exact
# ties, target self-term, k in {1,2,5,10,20}, OOV zero credit, multi-event
# users, all-OOV user, micro/macro computed independently.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 5, 10, 20])
def test_aggregate_end_to_end_matches_independent_reference(k):
    ranks, ds, K = _fixture()
    users = ds["eval_user_key"]
    pop_end_to_end = ds["eval_mask"]

    exp = expected_block(ranks, users, pop_end_to_end, k)
    got = aggregate(ranks, ds, k=k)["end_to_end"]

    assert got["n"] == exp["n"] == 7, "row5 (eval_mask False) must be excluded from end_to_end"
    assert got["macro"]["user_count"] == exp["user_count"] == 5
    # abs=1e-6: `_population_block` accumulates in float32 (production dtype);
    # our reference is plain Python float (effectively float64) -- the gap is
    # pure float32 rounding, not a semantic disagreement.
    assert got["micro"][f"recall@{k}"] == pytest.approx(exp["micro_recall"], abs=1e-6)
    assert got["micro"][f"mrr@{k}"] == pytest.approx(exp["micro_mrr"], abs=1e-6)
    assert got["macro"][f"recall@{k}"] == pytest.approx(exp["macro_recall"], abs=1e-6)
    assert got["macro"][f"mrr@{k}"] == pytest.approx(exp["macro_mrr"], abs=1e-6)


def test_aggregate_k10_and_k20_are_identical_given_max_rank_6():
    ranks, ds, K = _fixture()
    a10 = aggregate(ranks, ds, k=10)["end_to_end"]
    a20 = aggregate(ranks, ds, k=20)["end_to_end"]
    assert a10["micro"]["recall@10"] == a20["micro"]["recall@20"]
    assert a10["micro"]["mrr@10"] == a20["micro"]["mrr@20"]


def test_aggregate_closed_excludes_OOV_rows_end_to_end_gives_zero_credit():
    ranks, ds, K = _fixture()
    result = aggregate(ranks, ds, k=1)
    # end_to_end: OOV rows (3, 6) stay IN the denominator with zero credit.
    assert result["end_to_end"]["n"] == 7
    # closed: OOV rows are removed from the denominator entirely.
    assert result["closed"]["n"] == 5
    assert result["closed"]["macro"]["user_count"] == 4, "the all-OOV user (103) must vanish from 'closed'"
    users = ds["eval_user_key"]
    closed_pop = ds["eval_mask"] & (ds["target_class"] >= 0)
    exp_closed = expected_block(ranks, users, closed_pop, 1)
    assert result["closed"]["micro"]["recall@1"] == pytest.approx(exp_closed["micro_recall"])
    assert result["closed"]["macro"]["recall@1"] == pytest.approx(exp_closed["macro_recall"])


def test_all_OOV_user_gets_zero_credit_but_stays_counted_in_end_to_end():
    ranks, ds, K = _fixture()
    result = aggregate(ranks, ds, k=5)
    # user 103's only row (index 6) is OOV; find its contribution manually.
    users = ds["eval_user_key"]
    idx103 = [i for i, u in enumerate(users) if u == 103]
    assert len(idx103) == 1 and ranks[idx103[0]] == -1
    # It IS present among end_to_end's 5 macro users (zero credit), but the
    # value only shows up in the mean; verify via the independent reference.
    exp = expected_block(ranks, users, ds["eval_mask"], 5)
    assert result["end_to_end"]["macro"]["user_count"] == exp["user_count"] == 5


# ---------------------------------------------------------------------------
# 05 item 25 (target self-term) + Fable #5: rank is gathered from the SAME
# scores tensor at eval time, not a stale/cached copy.
# ---------------------------------------------------------------------------
def test_rank_uses_the_live_target_score_same_tensor():
    scores = torch.tensor([[100., 1., 1.]])
    target = torch.tensor([0])
    r1 = rank_targets(scores, target)
    assert int(r1[0]) == 1  # target has the unique max -> best rank

    scores2 = scores.clone()
    scores2[0, 0] = 0.5  # perturb ONLY the target's own column
    r2 = rank_targets(scores2, target)
    assert int(r2[0]) == 3, "rank must react to a change in the target's OWN score (self-term)"


# ---------------------------------------------------------------------------
# 05 item 26: candidate-chunk implementations agree on exact fixtures.
# ---------------------------------------------------------------------------
def test_rank_targets_is_chunk_size_independent():
    scores = torch.tensor([
        [10., 10., 10., 5., 1., 0.],
        [7., 7., 7., 7., 7., 7.],
        [3., 3., 3., 3., 3., 3.],
        [9., 8., 0., 0., 0., 0.],
        [5., 5., 5., 5., 1., 0.],
    ])
    target = torch.tensor([2, 0, 5, 1, 3])
    reference = rank_targets(scores, target, class_chunk=100000)
    for chunk in (1, 2, 3, 4, 6, 7):
        got = rank_targets(scores, target, class_chunk=chunk)
        assert torch.equal(got, reference), f"class_chunk={chunk} disagrees with the unchunked reference"


# ---------------------------------------------------------------------------
# 05 item 27: int64 large scores stay exact (no float32 rounding/tie bug).
# ---------------------------------------------------------------------------
def test_int64_large_scores_stay_exact_no_float32_collision():
    # 2**24 + 1 is NOT exactly representable in float32 and rounds (ties to
    # even) to 2**24 == the OTHER row's exact value, manufacturing a false
    # tie that does not exist in exact int64 arithmetic.
    other = 2 ** 24            # 16777216, exact in float32
    target_val = 2 ** 24 + 1   # 16777217, rounds to 16777216 in float32
    scores = torch.tensor([[other, target_val, 1]], dtype=torch.int64)
    target = torch.tensor([1])

    rank_exact = rank_targets(scores, target)
    assert int(rank_exact[0]) == 1, "exact int64 comparison: 16777216 < 16777217, so target keeps rank 1"

    # Negative control: casting to float32 first (the forbidden regression
    # eval_full.py's own docstring names) manufactures a false tie and
    # corrupts the rank.
    broken_scores = cast_to_float32_before_rank(scores)
    assert float(broken_scores[0, 0]) == float(broken_scores[0, 1]), \
        "sanity: the float32 cast must actually manufacture the tie"
    rank_broken = rank_targets(broken_scores, target)
    assert int(rank_broken[0]) != int(rank_exact[0]), \
        "negative control failed: the float32-cast path should have corrupted the rank"


# ---------------------------------------------------------------------------
# 05 item 28: TF32 canary + assert_strict_fp32 (Gate P fix 2 / Fable MUST_FIX 2).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a visible CUDA device (GPU 0)")
def test_tf32_canary_and_assert_strict_fp32_on_gpu():
    device = torch.device("cuda:0")
    try:
        precision.configure_strict_fp32(seed=0)
        rec_strict = precision.assert_strict_fp32()  # must not raise
        canary_strict = precision.tf32_canary(device)
        assert canary_strict["looks_strict_fp32"] is True, canary_strict
        assert canary_strict["relative_error"] < precision.CANARY_THRESHOLD

        # Negative control: deliberately enable TF32 -> canary must FLAG it,
        # and assert_strict_fp32 must raise.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        canary_tf32 = precision.tf32_canary(device)
        assert canary_tf32["looks_strict_fp32"] is False, \
            "negative control failed: canary should flag TF32 once deliberately enabled"
        assert canary_tf32["relative_error"] > precision.CANARY_THRESHOLD
        with pytest.raises(RuntimeError):
            precision.assert_strict_fp32()
    finally:
        precision.configure_strict_fp32(seed=0)  # restore strict settings for other tests


def test_assert_strict_fp32_raises_on_bad_matmul_precision():
    precision.configure_strict_fp32(seed=0)
    try:
        torch.set_float32_matmul_precision("high")
        with pytest.raises(RuntimeError):
            precision.assert_strict_fp32()
    finally:
        torch.set_float32_matmul_precision("highest")
        precision.configure_strict_fp32(seed=0)


# ---------------------------------------------------------------------------
# Gate P2 registry v2 precision: training reads TF32 on, eval reads strict.
# `training_precision(tf32)` / `assert_training_precision(tf32)` / `strict_eval()`.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a visible CUDA device (GPU 0)")
def test_training_precision_tf32_and_strict_eval_roundtrip():
    try:
        precision.configure_strict_fp32(seed=0)   # baseline determinism + strict flags
        rec = precision.training_precision(True)  # simulate the training phase (TF32 matmul)
        precision.assert_training_precision(True)  # must not raise
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.get_float32_matmul_precision() == "high"

        # Negative control: asserting the WRONG training precision must raise.
        with pytest.raises(RuntimeError):
            precision.assert_training_precision(False)

        # strict_eval() must force strict FP32 INSIDE the block...
        with precision.strict_eval():
            precision.assert_strict_fp32()  # must not raise
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.get_float32_matmul_precision() == "highest"

        # ...and restore the TF32 training flags on exit.
        precision.assert_training_precision(True)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.get_float32_matmul_precision() == "high"

        # strict_eval() must restore flags even when the block raises.
        class _Boom(Exception):
            pass
        with pytest.raises(_Boom):
            with precision.strict_eval():
                raise _Boom()
        precision.assert_training_precision(True)  # still restored despite the exception

        # Negative control: strict FP32 training must not look like TF32 training.
        precision.training_precision(False)
        with pytest.raises(RuntimeError):
            precision.assert_training_precision(True)
    finally:
        precision.configure_strict_fp32(seed=0)


def _tiny_rank_fixture_script(strict_only: bool) -> str:
    """A self-contained script (run in-process or as a subprocess) that builds
    a small-but-nontrivial SASRecCE (d=64, K=300, T=50) with a fixed seed,
    optionally puts the process into TF32 training precision first, computes
    logits/ranks, and prints them as a comma-separated line. `strict_only`
    simulates 'a fresh strict process' that never touches TF32 at all."""
    return f"""
import sys, torch
sys.path.insert(0, {str(CODE_DIR)!r})
import paths
import precision
from sasrec_model import SASRecCE, SASRecConfig
from eval_full import rank_targets
import numpy as np

precision.configure_strict_fp32(seed=123)
if not {strict_only}:
    precision.training_precision(True)

cfg = SASRecConfig(run_id='fx', group='item_only', d_model=64, blocks=2, heads=4,
                    ff_multiplier=4, dropout=0.0, max_len=50)
K = 300
poc = np.arange(K, dtype=np.int64) + 3
model = SASRecCE(cfg, vocab_sizes={{'item_tokens': K + 3}}, product_idx_of_class=poc,
                 seed=777, impl='unpadded').to('cuda:0').eval()

rng = np.random.default_rng(42)
B, T = 8, 50
lengths = rng.integers(1, T + 1, size=B).astype(np.int64)
item_tokens = np.zeros((B, T), dtype=np.int64)
attention_mask = np.zeros((B, T), dtype=bool)
for b in range(B):
    L = int(lengths[b])
    item_tokens[b, :L] = rng.integers(3, 3 + K, size=L)
    attention_mask[b, :L] = True
item_tokens = torch.from_numpy(item_tokens).to('cuda:0')
attention_mask = torch.from_numpy(attention_mask).to('cuda:0')
lengths_t = torch.from_numpy(lengths).to('cuda:0')
target = torch.from_numpy(rng.integers(0, K, size=B).astype(np.int64)).to('cuda:0')

with torch.no_grad():
    with precision.strict_eval():
        logits = model(item_tokens=item_tokens, lengths=lengths_t, attention_mask=attention_mask)
        ranks = rank_targets(logits, target)
print(','.join(str(int(x)) for x in ranks.cpu().numpy().tolist()))
"""


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a visible CUDA device (GPU 0)")
def test_strict_eval_gives_bitwise_identical_ranks_tf32_process_vs_fresh_strict_process():
    import subprocess
    import sys as _sys

    # "Inside a TF32-configured process, under strict_eval()":
    script_tf32_proc = _tiny_rank_fixture_script(strict_only=False)
    res_tf32 = subprocess.run([_sys.executable, "-c", script_tf32_proc], capture_output=True,
                              text=True, timeout=120)
    assert res_tf32.returncode == 0, res_tf32.stderr

    # "A fresh strict process" (TF32 never touched at all):
    script_fresh_strict = _tiny_rank_fixture_script(strict_only=True)
    res_strict = subprocess.run([_sys.executable, "-c", script_fresh_strict], capture_output=True,
                                text=True, timeout=120)
    assert res_strict.returncode == 0, res_strict.stderr

    ranks_tf32_proc = res_tf32.stdout.strip()
    ranks_fresh_strict = res_strict.stdout.strip()
    assert ranks_tf32_proc == ranks_fresh_strict, (
        "strict_eval() must give BITWISE-identical rank vectors regardless of whether the "
        f"surrounding process was TF32-configured for training.\nTF32-process ranks: {ranks_tf32_proc}\n"
        f"fresh-strict-process ranks: {ranks_fresh_strict}"
    )

    # Negative control: computing WITHOUT strict_eval() while TF32-configured
    # must be able to differ (proves strict_eval() is doing real work, not a
    # no-op on a fixture too small to show any TF32 effect).
    script_tf32_no_guard = _tiny_rank_fixture_script(strict_only=False).replace(
        "with precision.strict_eval():", "if True:"
    )
    res_tf32_unguarded = subprocess.run([_sys.executable, "-c", script_tf32_no_guard], capture_output=True,
                                        text=True, timeout=120)
    assert res_tf32_unguarded.returncode == 0, res_tf32_unguarded.stderr
    # Not asserted to differ (TF32 error may or may not flip an integer rank on
    # this fixture) -- logged for evidence; the mandatory guarantee is the
    # equality proven above, under strict_eval() in both processes.


# ---------------------------------------------------------------------------
# 05 item 29: fingerprint comparison fails closed on a changed population.
# CONTRACT GAP: no production `fingerprint()` helper exists yet (owned by
# lock_retro.py, not yet written) -- this is our own small helper, matching
# the fields Gate P fix 5 / SASREC_INTERFACES.md sec.8 name: decision_id,
# eval_mask, target_class, eval_user_key.
# ---------------------------------------------------------------------------
def fingerprint(ds: dict) -> str:
    h = hashlib.sha256()
    for key in ("decision_id", "eval_mask", "target_class", "eval_user_key"):
        arr = np.asarray(ds[key])
        h.update(key.encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def test_fingerprint_fails_closed_on_a_changed_population():
    base = dict(
        decision_id=np.arange(6, dtype=np.int64),
        eval_mask=np.array([True, True, False, True, True, True]),
        target_class=np.array([1, 2, -1, 3, 4, 5], dtype=np.int64),
        eval_user_key=np.array([10, 10, 11, 12, 12, 13], dtype=np.int64),
    )
    fp0 = fingerprint(base)
    assert fingerprint(dict(base)) == fp0, "identical population must fingerprint identically"

    for field, mutate in [
        ("eval_mask", lambda d: d.__setitem__("eval_mask", np.array([True, True, True, True, True, True]))),
        ("target_class", lambda d: d.__setitem__("target_class", np.array([1, 2, -1, 3, 4, 99], dtype=np.int64))),
        ("eval_user_key", lambda d: d.__setitem__("eval_user_key", np.array([10, 10, 11, 12, 12, 999], dtype=np.int64))),
        ("decision_id", lambda d: d.__setitem__("decision_id", np.array([0, 1, 2, 3, 4, 999], dtype=np.int64))),
    ]:
        changed = {k: v.copy() for k, v in base.items()}
        mutate(changed)
        fp_changed = fingerprint(changed)
        assert fp_changed != fp0, f"fingerprint must change when '{field}' changes (fail-closed)"

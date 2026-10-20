"""The per-seed data order is reproducible (package 05 §B "identical data order policy by seed").

  * pass_permutation(seed, pass, N) is a pure function: equal across calls, a permutation of 0..N-1, different across
    passes and seeds, with a recorded sha256 per pass;
  * inside the trainer, the decisions served are exactly the pass permutations of the canonical order in consecutive
    effective batches (tail at its own size), identical for both families at the same seed and for two runs of the same
    seed, different for another seed;
Negative control: the GRU1 rule default_rng(seed + pass) gives seed 2026 pass 1 the permutation of seed 2027 pass 0.
"""
from __future__ import annotations

import numpy as np

from trainkit import art  # noqa: E402  (TRAIN-03: sealed artifact root)
from trainkit import B, N, cfg, nc, providers, read, scripted, trainer
from v4train.order import naive_gru1_permutation, pass_permutation, permutation_sha256
from v4train.schedule import eval_marks

VALS = {m: 0.1 for m in eval_marks("MAIN")}


def _check_seeds_independent(perm_fn):
    a = perm_fn(2026, 1, 1000)
    b = perm_fn(2027, 0, 1000)
    assert not np.array_equal(a, b), "two seeds must never share a pass permutation"
    for s in (2026, 2027):
        ps = [perm_fn(s, p, 1000) for p in range(8)]
        assert len({permutation_sha256(p) for p in ps}) == 8


def test_pass_permutation_is_pure_and_distinct():
    p1, p2 = pass_permutation(2026, 3, 5000), pass_permutation(2026, 3, 5000)
    assert np.array_equal(p1, p2) and np.array_equal(np.sort(p1), np.arange(5000))
    assert permutation_sha256(p1) == permutation_sha256(p2)
    _check_seeds_independent(pass_permutation)


def _served(work, name, family, seed):
    tp, ep = providers()
    tr = trainer(work / name, cfg(name, family, seed=seed), tp, ep, eval_override=scripted(VALS))
    tr.run()
    return tp, tr


def test_trainer_serves_the_seeded_order_for_both_families(work):
    tp_g, tr_g = _served(work, "g", "GRU", 2026)
    tp_s, _ = _served(work, "s", "SASREC", 2026)
    tp_g2, _ = _served(work, "g2", "GRU", 2026)
    tp_o, _ = _served(work, "o", "GRU", 2027)
    did = tp_g.decision_ids()
    expect = np.concatenate([did[pass_permutation(2026, p, N)] for p in range(6)])
    assert tp_g.access_log == expect.tolist(), "served order = pass permutations of the canonical order"
    assert tp_s.access_log == tp_g.access_log, "both families see identical data at the same seed"
    assert tp_g2.access_log == tp_g.access_log, "reproducible across runs"
    assert tp_o.access_log != tp_g.access_log and sorted(tp_o.access_log) == sorted(tp_g.access_log)
    rec = read(art(work / "g") / "RUN_SUMMARY.json")
    assert rec["updates"] == 6 * -(-N // B)
    st = read(work / "g" / "RUN_START.json")
    assert "SeedSequence" in st["order_policy"]
    shas = tr_g.state.perm_sha256
    assert [shas[str(p)] for p in range(6)] == [permutation_sha256(pass_permutation(2026, p, N)) for p in range(6)]


@nc("the GRU1 order rule default_rng(seed + pass) collides across seeds 2026 / 2027")
def test_nc_naive_order_collides_across_seeds():
    _check_seeds_independent(naive_gru1_permutation)

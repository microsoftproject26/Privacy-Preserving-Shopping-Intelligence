"""T20 — resume is bitwise-deterministic, with an interruption in the middle of an EFE (both families, real v4eval).

A reference run (A) trains 6.0 EFE uninterrupted; run B is killed after update 9 (pass 2, position 64 of 200: mid-EFE,
between the 1.0 and 1.5 marks) with extra `latest` saves every 4 updates, then resumed from `latest` (update 8) in a
fresh trainer object. Required equal, bitwise: the final state digest, the best / endpoint_6.0 weights (state digest
and file sha256), the evaluation history (marks, EFE, values), the result rows, the final global RNG state and the
logical metrics log (timings excluded; a replayed window counted once). Also: a crash inside a checkpoint write (after
the temp file, before the replace) leaves the previous `latest` valid and the resumed run is still bitwise equal; a
resume with another configuration, or with a drifted order policy, is refused.
Negative controls: a resume that does not restore the RNG state, and one that drops the AdamW state, are not bitwise.
"""
from __future__ import annotations

import pytest
import torch

import v4train.trainer as T
from trainkit import art  # noqa: E402  (TRAIN-03: sealed artifact root)
from trainkit import assert_raises, cfg, interrupt_after, logical_metrics, nc, read, trainer
from v4train.checkpoint import CheckpointError, sha256_file
from v4train.trainer import InjectedInterruption, TrainerError

FAMS = ["GRU", "SASREC"]


def _final(run_dir):
    s = read(art(run_dir) / "RUN_SUMMARY.json")
    ck = read(art(run_dir) / "CHECKPOINTS.json")
    return {"digest": s["final_state_digest"], "rng": s["final_rng_sha256"],
            "evals": [(e["mark"], e["efe"], e["value"], e["improved"]) for e in s["evals"]],
            "rows": read(art(run_dir) / "RESULT_ROWS.json")["rows"],
            "best": (ck["best"]["meta"]["state_digest"], ck["best"]["sha256"]),
            "endpoint": (ck["endpoint_6.0"]["meta"]["state_digest"], ck["endpoint_6.0"]["sha256"]),
            "metrics": logical_metrics(run_dir), "updates": s["updates"], "exposures": s["exposures"]}


def _reference(work, family):
    c = cfg(f"t20_{family}", family, checkpoint_every=4)
    trainer(work / "A", c).run()
    return c, _final(work / "A")


def _interrupted(work, family, c, *, patch_resume=None, **kw):
    b = trainer(work / "B", c, fault=interrupt_after(9), **kw)
    with pytest.raises(InjectedInterruption):
        b.run()
    assert read(art(work / "B") / "CHECKPOINTS.json")["latest"]["meta"]["updates"] == 8
    torch.manual_seed(12345)                                  # a foreign global RNG state before the resume
    b2 = trainer(work / "B", c)
    if patch_resume:
        patch_resume(b2)
    b2.resume()
    return _final(work / "B")


def _assert_bitwise(a, b):
    for k in ("digest", "rng", "evals", "rows", "best", "endpoint", "updates", "exposures", "metrics"):
        assert a[k] == b[k], f"resume is not bitwise: {k} differs"


@pytest.mark.parametrize("family", FAMS)
def test_t20_resume_mid_efe_is_bitwise(work, family):
    c, a = _reference(work, family)
    _assert_bitwise(a, _interrupted(work, family, c))


def test_t20_crash_inside_a_checkpoint_write(work):
    c, a = _reference(work, "GRU")
    seen = {"n": 0}

    def ck_fault(kind, stage):
        if kind == "latest" and stage == "after_tmp":
            seen["n"] += 1
            if seen["n"] == 4:           # latest writes at updates 4 (0.5), 7 (1.0), 8, 11 (1.5): crash in the 4th
                raise InjectedInterruption("crash between the temp file and the replace")

    b = trainer(work / "B", c, checkpoint_fault=ck_fault)
    with pytest.raises(InjectedInterruption):
        b.run()
    idx = read(art(work / "B") / "CHECKPOINTS.json")["latest"]
    assert idx["meta"]["updates"] == 8 and sha256_file(art(work / "B") / idx["file"]) == idx["sha256"]
    assert not list((art(work / "B") / "checkpoints").glob("*.tmp.*")), "no temp file left behind"
    trainer(work / "B", c).resume()
    _assert_bitwise(a, _final(work / "B"))


def test_resume_refusals(work):
    c = cfg("refuse", "GRU", checkpoint_every=4)
    b = trainer(work / "B", c, fault=interrupt_after(9))
    with pytest.raises(InjectedInterruption):
        b.run()
    other = cfg("refuse", "GRU", checkpoint_every=4, log_every=7)          # another configuration
    assert_raises(TrainerError, trainer(work / "B", other).resume)
    assert_raises(TrainerError, trainer(work / "B", c).run)                 # never restart over existing evidence
    import v4train.order as O
    orig = T.pass_permutation
    try:
        T.pass_permutation = O.naive_gru1_permutation                        # a drifted order policy
        assert_raises(TrainerError, trainer(work / "B", c).resume)
    finally:
        T.pass_permutation = orig
    p = art(work / "B") / "checkpoints" / "latest.pt"
    p.write_bytes(p.read_bytes()[:-10])                                      # a truncated resumable
    assert_raises(CheckpointError, trainer(work / "B", c).resume)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("resume without restoring the RNG state (dropout stream) is not bitwise")
def test_nc_t20_no_rng_restore(work):
    c, a = _reference(work, "GRU")

    def patch(tr):
        tr._restore_rng = lambda rng: None
    _assert_bitwise(a, _interrupted(work, "GRU", c, patch_resume=patch))


@nc("resume that drops the AdamW moments is not bitwise")
def test_nc_t20_no_optimizer_restore(work):
    c, a = _reference(work, "GRU")

    def patch(tr):
        tr.opt.load_state_dict = lambda sd: None
    _assert_bitwise(a, _interrupted(work, "GRU", c, patch_resume=patch))

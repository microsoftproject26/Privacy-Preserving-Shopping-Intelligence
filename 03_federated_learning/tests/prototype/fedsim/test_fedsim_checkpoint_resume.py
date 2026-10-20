"""Checkpoint / resume for FA / FP / PF under interruption.

Fixture: the tiny UNTIED model with per-step recentering (SASRec-like) from a registered theta0, 30 synthetic
clients in groups of 8 (4 rounds per sweep, a partial last group of 6), 2 passes, the s(EFE) schedule, evaluation at
every 0.5 EFE with strictly-best saving, run to the 6.0-EFE endpoint (12 rounds).

Proves: a run interrupted at any round (or inside the checkpoint write at any of its 4 crash points) and resumed in
fresh objects ends bitwise-equal to the uninterrupted run — server state, PF store with moments, counters, ledger,
evaluation history, running best and LR log; stale temp files are removed with their sha256 logged; corrupted or
truncated checkpoints and mismatched configs / plans / init hashes are refused; retention = latest + best +
endpoint while live, latest removed at completion (calibration: everything), each removal sha256-logged; the disk
guard refuses a write that would cross the 8 GiB floor. Does not prove: GPU resume.
"""
from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path

import pytest
import torch
from fedsim_testkit import assert_raises, clients, nc, one_client, solver, tiny

import ppsi.fedsim.checkpoint as flck
import ppsi.fedsim.participation as flp
import ppsi.fedsim.personal as flpers
import ppsi.fedsim.runtime as flr
from ppsi.fedsim.checkpoint import (
    CheckpointCorrupt,
    CheckpointError,
    CheckpointManager,
    DiskRefused,
    SimulatedCrash,
    load_checkpoint,
    sha256_file,
)
from ppsi.fedsim.client import PFConfig
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.participation import ParticipationPlan
from ppsi.fedsim.personal import PersonalStore
from ppsi.fedsim.runtime import FLRun, RunConfig
from ppsi.fedsim.server import Server

WORK = Path(__file__).resolve().parent / "_tmp_ckpt"
SEED = 2026
PEAK = 0.05
CS = clients(30, seed=141, invalid_frac=0.15)
BY_KEY = {c.key: c for c in CS}
N_DEC = sum(int((c.examples["target_class"] >= 0).sum()) for c in CS)
PLAN = ParticipationPlan.build(list(BY_KEY), seed=SEED, manifest_hash="ckpt-tests", group_size=8)
EVAL_BATCH = one_client("user-eval", 24, seed=9).examples


@pytest.fixture
def work(request):
    d = WORK / request.node.name.replace("[", "_").replace("]", "")
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    with contextlib.suppress(OSError):
        WORK.rmdir()


def registered_theta0():
    a = tiny(1, tied=False)
    a.server_post_aggregate()                       # the one registration-time recentering (C7 analogue)
    return a.broadcast_state(clone=True), state_digest(a.broadcast_state())


THETA0, INIT = registered_theta0()


def eval_fn(server):
    a = server.adapter
    a.module.eval()
    with torch.no_grad():
        ce = torch.nn.functional.cross_entropy(a.scores(EVAL_BATCH), EVAL_BATCH["target_class"])
    return -float(ce)


def cfg_for(method, **over):
    kw = {"FA": {}, "FP": {"mu": 0.05}, "PF": {"pf": PFConfig(lr=PEAK, lam=1e-4)}}[method]
    kw = {"ckpt_every": 1, "exposure_point": "end", **kw, **over}           # the study value, explicit
    return RunConfig(f"run-{method}", method, SEED, PEAK, solver(), **kw)


def new_run(method, ckpt=None, resume=False, **over):
    a = tiny(1, tied=False)
    a.load_state_(THETA0)
    srv = Server(a, init_sha256=INIT)
    store = PersonalStore(a.query_dim) if method == "PF" else None
    args = (cfg_for(method, **over), PLAN, srv, BY_KEY.get, N_DEC)
    kw = {"workers": [tiny(200, tied=False)], "store": store, "ckpt": ckpt, "eval_fn": eval_fn}
    return FLRun.resume(*args, **kw) if resume else FLRun(*args, **kw)


def fingerprint_run(run) -> dict:
    s = run.summary()
    s["lr_log"] = [list(x) for x in run.lr_log]
    s["ledger_full"] = run.ledger.state_dict()
    return s


_REF = {}


def reference(method):
    if method not in _REF:
        _REF[method] = fingerprint_run(new_run(method).run())
    return _REF[method]


def check_resume(method, work, stop_after):
    ref = reference(method)
    assert ref["done"] and ref["rounds"] == 12 and ref["counters"]["efe"] == 6.0
    first = new_run(method, CheckpointManager(work, f"r-{method}"))
    first.run(max_rounds=stop_after)                               # ... the process dies here
    second = new_run(method, CheckpointManager(work, f"r-{method}"), resume=True)
    assert second.state.cursor == stop_after
    try:
        second.run()
    except flr.LRTableError as e:                                  # the planned-table guard caught a bad resume
        raise AssertionError(f"{method}: resumed run disagrees with its planned LR table: {e}") from None
    got = fingerprint_run(second)
    for k in ref:
        assert got[k] == ref[k], f"{method}: resumed run differs in {k}"


# ------------------------------------------------------------------------------------------------ resume
@pytest.mark.parametrize("method", ["FA", "FP", "PF"])
def test_resume_bitwise(method, work):
    check_resume(method, work, stop_after=5)


def test_resume_at_sweep_boundary_and_first_round(work):
    check_resume("PF", work, stop_after=4)
    shutil.rmtree(work / "r-PF")
    check_resume("PF", work, stop_after=1)


def test_rng_states_restored(work):
    run = new_run("FA", CheckpointManager(work, "rng"))
    run.run(max_rounds=3)
    torch.manual_seed(12345)
    before = torch.get_rng_state().clone()
    run.ckpt.save_latest(run.state_dict())
    torch.manual_seed(999)
    new_run("FA", CheckpointManager(work, "rng"), resume=True)
    assert torch.equal(torch.get_rng_state(), before)


# ------------------------------------------------------------------------------------------------ interrupted writes
@pytest.mark.parametrize("point", flck.CRASH_POINTS)
def test_crash_inside_checkpoint_write(point, work, monkeypatch):
    ref = reference("PF")
    mgr = CheckpointManager(work, "crash")
    run = new_run("PF", mgr)
    run.run(max_rounds=4)
    prev = load_checkpoint(mgr.dir / mgr.LATEST)
    orig = CheckpointManager.save_latest
    monkeypatch.setattr(CheckpointManager, "save_latest", lambda self, p, crash_at=None: orig(self, p, crash_at=point))
    with pytest.raises(SimulatedCrash):
        run.step()                                                   # round 4 finishes, its checkpoint write crashes
    monkeypatch.setattr(CheckpointManager, "save_latest", orig)
    latest = load_checkpoint(mgr.dir / mgr.LATEST)                   # always a VALID state
    expect_cursor = 5 if point == "after_replace" else 4
    assert latest["cursor"] == expect_cursor
    if point != "after_replace":
        assert latest["__content_digest__"] == prev["__content_digest__"], "the previous checkpoint must be intact"
        assert list(mgr.dir.glob("*.tmp.*")), "the interrupted write leaves its temp file"
    mgr2 = CheckpointManager(work, "crash")                          # restart: stale temp removed, sha256 logged
    assert not list(mgr2.dir.glob("*.tmp.*"))
    if point != "after_replace":
        log = [json.loads(x) for x in mgr2.removals.read_text().splitlines()]
        assert log and all(len(e["sha256"]) == 64 and "stale temp" in e["reason"] for e in log)
    resumed = new_run("PF", mgr2, resume=True).run()
    got = fingerprint_run(resumed)
    assert all(got[k] == ref[k] for k in ref), "resume after a crashed write must end bitwise-equal"


def test_corrupted_or_truncated_checkpoint_refused(work):
    mgr = CheckpointManager(work, "corrupt")
    new_run("FA", mgr).run(max_rounds=2)
    p = mgr.dir / mgr.LATEST
    good = p.read_bytes()
    b = bytearray(good)
    b[len(b) // 2] ^= 0x01
    p.write_bytes(bytes(b))
    assert_raises(CheckpointCorrupt, load_checkpoint, p)
    assert_raises(CheckpointCorrupt, new_run, "FA", mgr, True)
    p.write_bytes(good[: len(good) // 3])
    assert_raises(CheckpointCorrupt, load_checkpoint, p)


def test_resume_refuses_foreign_config_plan_or_init(work):
    mgr = CheckpointManager(work, "run-FP")
    new_run("FP", mgr).run(max_rounds=2)
    assert_raises(CheckpointError, new_run, "FP", mgr, True, mu=0.1)               # different mu
    other_plan = ParticipationPlan.build(list(BY_KEY), seed=SEED + 1, manifest_hash="ckpt-tests", group_size=8)
    a = tiny(1, tied=False); a.load_state_(THETA0)
    assert_raises(CheckpointError, FLRun.resume, cfg_for("FP"), other_plan, Server(a, init_sha256=INIT), BY_KEY.get,
                  N_DEC, workers=[tiny(200, tied=False)], ckpt=mgr)
    ok = new_run("FP", mgr, resume=True, ckpt_every=7)                              # cadence is operational only
    assert ok.state.cursor == 2


# ------------------------------------------------------------------------------------------------ retention / disk
def test_retention_latest_best_endpoint(work):
    mgr = CheckpointManager(work, "keep")
    run = new_run("FA", mgr).run()
    assert run.state.done and mgr.files() == ["best.pt", "endpoint_6.0.pt", "latest.pt"]
    ep = load_checkpoint(mgr.dir / mgr.ENDPOINT)
    assert ep["kind"] == "weights" and ep["efe"] == 6.0 and ep["round"] == 12
    best = load_checkpoint(mgr.dir / mgr.BEST)
    assert best["kind"] == "weights" and run.state.best_metric is not None
    h = sha256_file(mgr.dir / mgr.LATEST)
    mgr.complete()
    assert mgr.files() == ["best.pt", "endpoint_6.0.pt"]
    log = [json.loads(x) for x in mgr.removals.read_text().splitlines()]
    assert log[-1]["sha256"] == h and log[-1]["path"].endswith("latest.pt")


def test_retention_calibration_keeps_no_weights(work):
    mgr = CheckpointManager(work, "calib", retain_weights=False)
    new_run("FA", mgr).run()
    hashes = {n: sha256_file(mgr.dir / n) for n in mgr.files()}
    mgr.complete()
    assert mgr.files() == []
    log = {Path(e["path"]).name: e["sha256"] for e in map(json.loads, mgr.removals.read_text().splitlines())}
    assert log == hashes


def test_disk_guard(work):
    GiB = 2 ** 30

    class Usage:
        def __init__(self, free):
            self.free, self.total, self.used = free, 100 * GiB, 0

    mgr = CheckpointManager(work, "disk", disk_usage_fn=lambda _p: Usage(int(8.0 * GiB)))
    run = new_run("FA", mgr)
    assert_raises(DiskRefused, run.step)                               # free - temp < 8 GiB floor
    mgr2 = CheckpointManager(work, "disk2", disk_usage_fn=lambda _p: Usage(int(9.0 * GiB)))
    new_run("FA", mgr2).run(max_rounds=1)                              # below the 10 GiB target: PAUSE, recorded
    ev = [json.loads(x) for x in mgr2.events.read_text().splitlines()]
    assert ev[-1]["disk"] == "PAUSE"


# ------------------------------------------------------------------------------------------------ negative controls
@nc("resume drops the PF optimizer moments")
def test_nc_resume_without_pf_moments(work, monkeypatch):
    orig = PersonalStore.from_state_dict.__func__

    def no_moments(cls, d):
        d = dict(d)
        d["has_opt"] = torch.zeros_like(d["has_opt"])
        return orig(cls, d)
    monkeypatch.setattr(flpers.PersonalStore, "from_state_dict", classmethod(no_moments))
    check_resume("PF", work, stop_after=5)


@nc("resume drops the exposure counters (the LR schedule restarts)")
def test_nc_resume_without_counters(work, monkeypatch):
    monkeypatch.setattr(flp.ExposureCounters, "from_state_dict",
                        classmethod(lambda cls, d: cls(int(d["n_decisions_total"]), int(d["n_clients"]),
                                                       int(d["rounds_per_sweep"]))))
    check_resume("FA", work, stop_after=5)


@nc("resume replays the last round (cursor off by one)")
def test_nc_resume_replays_a_round(work, monkeypatch):
    orig = FLRun.load_state_dict

    def off_by_one(self, d):
        orig(self, d)
        self.state.cursor -= 1
    monkeypatch.setattr(FLRun, "load_state_dict", off_by_one)
    check_resume("FA", work, stop_after=5)


def check_crash_leaves_valid_state(work, save):
    mgr = CheckpointManager(work, "nonatomic")
    run = new_run("FA", mgr)
    run.run(max_rounds=2)
    try:
        save(run.state_dict(), mgr.dir / mgr.LATEST)
    except SimulatedCrash:
        pass
    try:
        load_checkpoint(mgr.dir / mgr.LATEST)                          # must still be a valid checkpoint
    except CheckpointCorrupt as e:
        raise AssertionError(f"a crash during the write left no valid checkpoint: {e}") from None


def _non_atomic_save(payload, path):
    """WRONG: writes the target in place and dies half-way."""
    import io
    buf = io.BytesIO()
    torch.save(payload, buf)
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data[: len(data) // 2])
    raise SimulatedCrash("mid-write")


def test_atomic_save_crash_leaves_valid_state(work):
    check_crash_leaves_valid_state(work, lambda p, path: flck.atomic_save(p, path, crash_at="after_fsync"))


@nc("non-atomic in-place checkpoint write")
def test_nc_non_atomic_write(work):
    check_crash_leaves_valid_state(work, _non_atomic_save)

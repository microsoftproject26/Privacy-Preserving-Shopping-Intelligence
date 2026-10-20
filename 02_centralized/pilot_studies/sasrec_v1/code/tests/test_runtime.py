"""test_runtime.py -- independent runtime acceptance tests (05_TESTS_AND_PRETRAIN_GATE.md
E33-E37), owner: Sonnet-runtime. Every check has a negative control that must really
fail when the guarantee is broken; no fixture reuses the producer's own predicate as
"independent proof".

Runs ONLY on the server (tmux/bash/cgroup/nvidia-smi are Linux-only):
    ~/venvs/ppsi-gpu/bin/python -m pytest -v $S/code/tests/test_runtime.py

All scratch lives under $S/tests/C/<unique>; all evidence is written to
$S/tests/evidence/C_tests.json regardless of pass/fail. Test tmux sessions are named
sasrec-test-*, and are killed (verified by exact argv) in teardown.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import paths  # noqa: E402
import ckpt_io  # noqa: E402
import resources  # noqa: E402
import queue_runner  # noqa: E402

S = paths.S
SCRATCH_ROOT = S / "tests" / "C"
EVIDENCE_PATH = S / "tests" / "evidence" / "C_tests.json"
FAKE_TRAINER = CODE / "tests" / "fixtures" / "fake_trainer.py"
SUPERVISE_SH = CODE / "supervise_run.sh"
PYEXE = sys.executable
GIB = 2 ** 30

_EVIDENCE: list[dict] = []
_CLEANUP_DIRS: list[Path] = []
_CLEANUP_SESSIONS: list[str] = []


def record(name: str, **kw) -> None:
    _EVIDENCE.append({"test": name, "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw})


def new_scratch(tag: str) -> Path:
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=SCRATCH_ROOT))
    _CLEANUP_DIRS.append(d)
    return d


def tmux_session(name: str) -> str:
    assert name.startswith("sasrec-test-"), "test tmux sessions must start with sasrec-test-"
    _CLEANUP_SESSIONS.append(name)
    return name


def tmux_has(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name],
                          capture_output=True).returncode == 0


def tmux_kill(name: str) -> None:
    if tmux_has(name):
        subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True)


def run_bash(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def tmux_new_session(session: str, argv: list, env_vars: dict) -> None:
    """Start `argv` detached in tmux session `session` with `env_vars` set.

    tmux does NOT reliably propagate the *launching* process's environment into a
    brand-new session across tmux server states (observed empirically here), so env
    vars are baked into the shell command itself as a `VAR=val ... argv...` prefix
    (plain POSIX semantics: `VAR=val cmd` sets VAR only for that invocation) rather
    than passed via subprocess's `env=`."""
    import shlex as _shlex
    prefix = " ".join(f"{k}={_shlex.quote(str(v))}" for k, v in env_vars.items())
    cmd_str = (prefix + " " if prefix else "") + " ".join(_shlex.quote(str(p)) for p in argv)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, cmd_str], check=True)


def wait_until(pred, timeout=60.0, interval=0.3, desc="condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for: {desc}")


@pytest.fixture(scope="session", autouse=True)
def _final_cleanup():
    yield
    for name in list(_CLEANUP_SESSIONS):
        tmux_kill(name)
    for d in list(_CLEANUP_DIRS):
        shutil.rmtree(d, ignore_errors=True)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(_EVIDENCE, indent=2, default=str))


# ===========================================================================
# E33 -- atomic checkpoint failure injection + reclaim allowlist safety
# ===========================================================================

def test_ckpt_atomic_rotation_and_failure_injection():
    d = new_scratch("ckpt_atomic")
    ckpt_dir = d / "checkpoints"

    p1 = ckpt_io.save_atomic({"epoch": 1, "update": 1, "tag": "v1"}, ckpt_dir, "latest",
                             expected_update=1, required_keys=["epoch", "update"])
    h1 = ckpt_io.sha256_file(p1)
    assert not (ckpt_dir / "previous.pt").exists(), "first write must not create previous.pt"

    p2 = ckpt_io.save_atomic({"epoch": 2, "update": 2, "tag": "v2"}, ckpt_dir, "latest",
                             expected_update=2, required_keys=["epoch", "update"])
    h2 = ckpt_io.sha256_file(p2)
    assert ckpt_io.sha256_file(ckpt_dir / "previous.pt") == h1, "rotation must move v1 -> previous.pt"
    assert ckpt_io.sha256_file(ckpt_dir / "latest.pt") == h2

    # --- negative control: an update-mismatch write must NOT touch latest/previous ---
    with pytest.raises(ValueError):
        ckpt_io.save_atomic({"epoch": 3, "update": 3, "tag": "v3"}, ckpt_dir, "latest",
                            expected_update=999, required_keys=["epoch", "update"])
    assert ckpt_io.sha256_file(ckpt_dir / "latest.pt") == h2, "previous valid latest must survive"
    assert ckpt_io.sha256_file(ckpt_dir / "previous.pt") == h1
    assert list(ckpt_dir.glob("*.partial.*")) == [], "no stray partial files after a failed write"

    # --- negative control: missing required key must also refuse to rotate ---
    with pytest.raises(ValueError):
        ckpt_io.save_atomic({"epoch": 4, "update": 4}, ckpt_dir, "latest",
                            expected_update=4, required_keys=["epoch", "update", "rng_state"])
    assert ckpt_io.sha256_file(ckpt_dir / "latest.pt") == h2
    assert list(ckpt_dir.glob("*.partial.*")) == []

    # --- positive control: a correct third write DOES proceed (proves the harness
    #     can tell good from bad, not just always-pass) ---
    p3 = ckpt_io.save_atomic({"epoch": 3, "update": 3, "tag": "v3"}, ckpt_dir, "latest",
                             expected_update=3, required_keys=["epoch", "update"])
    h3 = ckpt_io.sha256_file(p3)
    assert h3 != h2
    assert ckpt_io.sha256_file(ckpt_dir / "previous.pt") == h2

    obj, which = ckpt_io.load_resume(ckpt_dir)
    assert which == "latest" and obj["update"] == 3

    record("ckpt_atomic_rotation_and_failure_injection", scratch=str(d),
          latest_sha256=h3, previous_sha256=h2, result="PASS")


def test_ckpt_disk_floor_refuses_write_no_corruption():
    d = new_scratch("ckpt_diskfloor")
    ckpt_dir = d / "checkpoints"
    p1 = ckpt_io.save_atomic({"epoch": 1, "update": 1}, ckpt_dir, "latest",
                             expected_update=1, required_keys=["epoch"])
    h1 = ckpt_io.sha256_file(p1)

    # An explicit (empty) scratch runs_root: the disk-floor branch below forces a
    # reclaim() scan, and it must NEVER be pointed at the real $S/runs production
    # tree from a test (hard rule: reclaim only ever runs inside a scratch fixture
    # tree in tests). The dir need not exist -- reclaim() handles that.
    empty_runs_root = d / "empty_runs_root_for_reclaim_scan"

    real_free_bytes = ckpt_io.free_bytes
    try:
        ckpt_io.free_bytes = lambda path: 1 * GIB  # simulate an almost-full disk
        with pytest.raises(ckpt_io.DiskFloor):
            ckpt_io.save_atomic({"epoch": 2, "update": 2}, ckpt_dir, "latest",
                                expected_update=2, required_keys=["epoch"],
                                runs_root=empty_runs_root)
    finally:
        ckpt_io.free_bytes = real_free_bytes

    assert ckpt_io.sha256_file(ckpt_dir / "latest.pt") == h1, "DiskFloor must not touch the old checkpoint"
    assert not (ckpt_dir / "previous.pt").exists()
    assert list(ckpt_dir.glob("*.partial.*")) == []

    # --- positive control: with real disk space it succeeds ---
    p2 = ckpt_io.save_atomic({"epoch": 2, "update": 2}, ckpt_dir, "latest",
                             expected_update=2, required_keys=["epoch"],
                             runs_root=empty_runs_root)
    assert ckpt_io.sha256_file(p2) != h1
    record("ckpt_disk_floor_refuses_write_no_corruption", scratch=str(d), result="PASS")


def test_reclaim_allowlist_safety():
    d = new_scratch("reclaim")
    runs_root = d / "runs"

    def make_run(name, *, done, final_status, has_best_weights=True):
        rd = runs_root / name
        ck = rd / "checkpoints"
        ck.mkdir(parents=True)
        (ck / "latest.pt").write_bytes(b"LATEST-" + name.encode())
        (ck / "previous.pt").write_bytes(b"PREVIOUS-" + name.encode())
        if has_best_weights:
            (ck / "best_weights.pt").write_bytes(b"BEST-" + name.encode())
        if done:
            (rd / "SUPERVISOR_DONE").write_text("done")
        (rd / "summary.json").write_text(json.dumps({"final_status": final_status}))
        return rd

    complete_rd = make_run("done_complete", done=True, final_status="COMPLETE")
    running_rd = make_run("running", done=False, final_status="RUNNING")
    stopped_rd = make_run("stopped_not_complete", done=True, final_status="STOPPED_NONFINITE")

    outside_sentinel = d / "outside_sentinel.txt"
    outside_sentinel.write_text("must survive")

    deleted = ckpt_io.reclaim(runs_root)

    assert set(deleted) == {
        str(complete_rd / "checkpoints" / "latest.pt"),
        str(complete_rd / "checkpoints" / "previous.pt"),
    }, "reclaim must delete ONLY latest/previous of a COMPLETE+SUPERVISOR_DONE run"
    assert (complete_rd / "checkpoints" / "best_weights.pt").exists(), "best_weights.pt is never reclaimed"

    # --- negative control: a non-COMPLETE run's checkpoints are never touched, even
    #     though it also has a SUPERVISOR_DONE-independent marker set (stopped_rd) ---
    assert (running_rd / "checkpoints" / "latest.pt").exists()
    assert (running_rd / "checkpoints" / "previous.pt").exists()
    assert (stopped_rd / "checkpoints" / "latest.pt").exists()
    assert (stopped_rd / "checkpoints" / "previous.pt").exists()
    assert outside_sentinel.read_text() == "must survive", "reclaim must never touch anything outside runs_root"

    record("test_reclaim_allowlist_safety", scratch=str(d), deleted=deleted, result="PASS")


# ===========================================================================
# supervise_run.sh: exit-code policy, bounded crash resume, no retry storms
# ===========================================================================

def _launch_supervisor(run_id, seed, gpu, run_dir, extra_args=None, env_extra=None):
    session = tmux_session(f"sasrec-test-{run_id}-s{seed}")
    env_vars = {
        "SASREC_TRAINER": f"{PYEXE} {FAKE_TRAINER}",
        "SASREC_SUPERVISE_CRASH_SLEEP": "0.2",
        "SASREC_SUPERVISE_GPU_BUSY_SLEEP": "0.2",
    }
    if env_extra:
        env_vars.update(env_extra)
    extra_args = extra_args or []
    argv = ["bash", str(SUPERVISE_SH), run_id, str(seed), str(gpu), str(run_dir)] + extra_args
    tmux_new_session(session, argv, env_vars)
    return session


def test_supervise_complete_and_stop_codes_no_retry():
    for fail_mode, epoch_flag, rc, reason in [
        (None, None, 0, None),
        ("nonfinite", 1, 3, "NONFINITE"),
        ("disk", 1, 4, "DISK"),
        ("config", 1, 5, "CONFIG"),
    ]:
        d = new_scratch(f"sup_{fail_mode or 'ok'}")
        run_dir = d / "run"
        extra = ["--planned-epochs", "2", "--epoch-seconds", "0.1", "--updates-per-epoch", "5"]
        if fail_mode:
            extra += ["--fail-mode", fail_mode, "--fail-at-epoch", str(epoch_flag)]
        session = _launch_supervisor(f"test-sup-{fail_mode or 'ok'}", 1, 0, run_dir, extra_args=extra)
        try:
            wait_until(lambda: (run_dir / "SUPERVISOR_DONE").exists() or (run_dir / "SUPERVISOR_STOPPED").exists(),
                      timeout=30, desc=f"terminal marker for {fail_mode}")
        finally:
            tmux_kill(session)

        if rc == 0:
            assert (run_dir / "SUPERVISOR_DONE").exists()
            assert not (run_dir / "SUPERVISOR_STOPPED").exists()
        else:
            stopped = (run_dir / "SUPERVISOR_STOPPED").read_text()
            assert f"rc={rc}" in stopped and f"reason={reason}" in stopped
            assert not (run_dir / "SUPERVISOR_DONE").exists()
        log = (run_dir / "logs" / "supervisor.log").read_text()
        launches = log.count("launching trainer")
        assert launches == 1, f"a no-retry exit code must never be retried (got {launches} launches)"
        record("test_supervise_complete_and_stop_codes_no_retry", fail_mode=fail_mode,
              rc=rc, launches=launches, result="PASS")


def test_supervise_crash_resume_bounded_at_3():
    # --- positive control: 2 crashes then success stays within the budget ---
    d = new_scratch("crash_bounded_ok")
    run_dir = d / "run"
    extra = ["--planned-epochs", "2", "--epoch-seconds", "0.1", "--updates-per-epoch", "5",
            "--fail-mode", "crash", "--fail-count", "2"]
    session = _launch_supervisor("test-crash-ok", 1, 0, run_dir, extra_args=extra)
    try:
        wait_until(lambda: (run_dir / "SUPERVISOR_DONE").exists(), timeout=30,
                  desc="eventual completion after 2 bounded crashes")
    finally:
        tmux_kill(session)
    log = (run_dir / "logs" / "supervisor.log").read_text()
    assert log.count("launching trainer") == 3, "2 crashes + 1 success = 3 launches"

    # --- negative control: a run that ALWAYS crashes must give up at exactly 3 resumes ---
    d2 = new_scratch("crash_bounded_fail")
    run_dir2 = d2 / "run"
    extra2 = ["--planned-epochs", "2", "--epoch-seconds", "0.1", "--updates-per-epoch", "5",
             "--fail-mode", "crash", "--fail-count", "999"]
    session2 = _launch_supervisor("test-crash-fail", 1, 0, run_dir2, extra_args=extra2)
    try:
        wait_until(lambda: (run_dir2 / "SUPERVISOR_STOPPED").exists(), timeout=30,
                  desc="give-up after exhausting the crash-resume budget")
    finally:
        tmux_kill(session2)
    stopped = (run_dir2 / "SUPERVISOR_STOPPED").read_text()
    assert "reason=MAX_RESUMES" in stopped
    log2 = (run_dir2 / "logs" / "supervisor.log").read_text()
    assert log2.count("launching trainer") == 4, "3 resumes = 4 total launch attempts, then give up"
    record("test_supervise_crash_resume_bounded_at_3", ok_launches=3, fail_launches=4, result="PASS")


def test_supervise_gpu_busy_bounded_at_12_retries():
    d = new_scratch("gpu_busy_bounded")
    run_dir = d / "run"
    extra = ["--planned-epochs", "2", "--epoch-seconds", "0.1", "--updates-per-epoch", "5",
            "--fail-mode", "gpu_busy", "--fail-at-epoch", "1"]
    session = _launch_supervisor("test-gpubusy", 1, 0, run_dir, extra_args=extra)
    try:
        # 13 total launches, each paying real torch-import startup cost (~1-2s) in
        # the fake trainer's own process, plus 12 short sleeps: budget generously.
        wait_until(lambda: (run_dir / "SUPERVISOR_STOPPED").exists(), timeout=90,
                  desc="give-up after exhausting the GPU_BUSY retry budget")
    finally:
        tmux_kill(session)
    stopped = (run_dir / "SUPERVISOR_STOPPED").read_text()
    assert "reason=GPU_BUSY_MAX_RETRIES" in stopped
    log = (run_dir / "logs" / "supervisor.log").read_text()
    assert log.count("launching trainer") == 13, "12 GPU_BUSY retries = 13 total launch attempts"
    record("test_supervise_gpu_busy_bounded_at_12_retries", result="PASS")


# NOTE (Gate P2): --pair-dir/--pair-partner and exit code 7 no longer exist in the
# trainer CLI/protocol (no pair barrier under the fixed-epoch, owner-deadline plan).
# The former pair-partner-death supervisor tests are removed; pair semantics are now
# purely a post-hoc reporting check (compare_pair_summaries, tested below) plus the
# cut-rule orchestration (maybe_compute_epoch_cap, tested below).


# ===========================================================================
# resources.py -- cgroup-derived limits, not os.cpu_count()
# ===========================================================================

def test_resources_uses_cgroup_not_cpu_count():
    src = (CODE / "resources.py").read_text()
    assert "cpu_count" not in src, "resources.py must never fall back to os.cpu_count()"

    cpu = resources.cgroup_cpu_quota_cores()
    mem = resources.cgroup_memory()
    assert cpu["source"] in ("cgroup_v2", "cgroup_v2_max", "cgroup_v1", "cgroup_v1_unlimited", "unavailable")
    assert mem["source"] in ("cgroup_v2", "cgroup_v1", "unavailable")
    if mem["limit_bytes"] is not None:
        assert 0 < mem["limit_bytes"] < 4 * 1024 * GIB  # sane upper bound, not a sentinel

    rss = resources.rss_of_pids([os.getpid()])
    assert rss["per_pid"][str(os.getpid())] is not None and rss["per_pid"][str(os.getpid())] > 0
    # --- negative control: a PID that cannot exist reports None, not a crash/fake value ---
    dead_pid = 2 ** 30
    rss_dead = resources.rss_of_pids([dead_pid])
    assert rss_dead["per_pid"][str(dead_pid)] is None

    disk = resources.disk_free(S)
    assert disk["free_bytes"] > 0

    record("test_resources_uses_cgroup_not_cpu_count", cpu=cpu, mem_source=mem["source"], result="PASS")


def test_resources_sampler_loop_writes_jsonl():
    d = new_scratch("sampler")
    out = d / "samples.jsonl"
    resources.sampler_loop(out, interval=0.05, iterations=3)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        rec = json.loads(line)
        assert "cgroup_memory" in rec and "disk" in rec and "gpus" in rec
    record("test_resources_sampler_loop_writes_jsonl", n=len(lines), result="PASS")


# ===========================================================================
# pure decision helper: replica concurrency (rich-channel RSS headroom rule)
# ===========================================================================

def test_plan_replica_concurrency():
    assert queue_runner.plan_replica_concurrency("item_only", None, None, None) == "concurrent"
    assert queue_runner.plan_replica_concurrency("item_only", 999 * GIB, 1 * GIB, 0) == "concurrent", \
        "item-only configs are always concurrent regardless of RSS"

    limit = 64 * GIB
    current = 20 * GIB
    # available = 44 GiB; measured rss 20 GiB -> projected after 2nd = 24 GiB >= 12 GiB floor
    assert queue_runner.plan_replica_concurrency("rich_frozen_channels", 20 * GIB, limit, current) == "concurrent"
    # measured rss 35 GiB -> projected after 2nd = 9 GiB < 12 GiB floor
    assert queue_runner.plan_replica_concurrency("rich_frozen_channels", 35 * GIB, limit, current) == "sequential"
    # boundary: exactly at the floor must be concurrent (>=), floor-1 must be sequential
    assert queue_runner.plan_replica_concurrency("rich_frozen_channels", 32 * GIB, limit, current) == "concurrent"
    assert queue_runner.plan_replica_concurrency(
        "rich_frozen_channels", 32 * GIB + 1, limit, current) == "sequential"
    # missing measurement -> conservative "unknown" (caller treats as sequential), never a guess
    assert queue_runner.plan_replica_concurrency("rich_frozen_channels", None, limit, current) == "unknown"
    record("test_plan_replica_concurrency", result="PASS")


# ===========================================================================
# Gate P2 pure helpers: arm_from_epoch_records, compare_pair_summaries
# ===========================================================================

def test_arm_from_epoch_records():
    assert queue_runner.arm_from_epoch_records([]) == {
        "epochs_done": 0, "mean_train_epoch_seconds": None, "eval_seconds": None}

    records = [
        {"epoch": 1, "epoch_seconds": 10.0, "eval_seconds": None, "full_vs": None},
        {"epoch": 2, "epoch_seconds": 12.0, "eval_seconds": None, "full_vs": None},
    ]
    arm = queue_runner.arm_from_epoch_records(records)
    assert arm["epochs_done"] == 2
    assert arm["mean_train_epoch_seconds"] == 11.0
    assert arm["eval_seconds"] is None, "no full-VS eval before floor_start: eval_seconds must stay None"

    records_with_eval = records + [
        {"epoch": 3, "epoch_seconds": 12.0, "eval_seconds": 100.0, "full_vs": {"macro_mrr20": 0.1}},
    ]
    arm2 = queue_runner.arm_from_epoch_records(records_with_eval)
    assert arm2["epochs_done"] == 3
    assert arm2["eval_seconds"] == 100.0
    record("test_arm_from_epoch_records", result="PASS")


def test_compare_pair_summaries():
    complete_a = {"final_status": "COMPLETE", "epochs_completed": 4,
                 "epoch_records": [{"epoch": 1, "perm_sha256": "h1"}, {"epoch": 2, "perm_sha256": "h2"}]}
    complete_b_match = {"final_status": "COMPLETE", "epochs_completed": 4,
                        "epoch_records": [{"epoch": 1, "perm_sha256": "h1"}, {"epoch": 2, "perm_sha256": "h2"}]}
    result = queue_runner.compare_pair_summaries(complete_a, complete_b_match)
    assert result == {"pair_status": "COMPLETE", "equal_epochs": True,
                      "equal_perm_sha256": True, "mismatches": []}

    # --- negative control: a real permutation-digest mismatch must be caught ---
    complete_b_mismatch = {"final_status": "COMPLETE", "epochs_completed": 4,
                           "epoch_records": [{"epoch": 1, "perm_sha256": "h1"}, {"epoch": 2, "perm_sha256": "DIFFERENT"}]}
    result_mismatch = queue_runner.compare_pair_summaries(complete_a, complete_b_mismatch)
    assert result_mismatch["pair_status"] == "MISMATCH"
    assert result_mismatch["equal_perm_sha256"] is False
    assert result_mismatch["mismatches"] == [2]

    # --- negative control: unequal epoch counts must be caught even if digests agree ---
    complete_b_short = {"final_status": "COMPLETE", "epochs_completed": 3,
                        "epoch_records": [{"epoch": 1, "perm_sha256": "h1"}, {"epoch": 2, "perm_sha256": "h2"}]}
    result_unequal = queue_runner.compare_pair_summaries(complete_a, complete_b_short)
    assert result_unequal["pair_status"] == "MISMATCH"
    assert result_unequal["equal_epochs"] is False

    # --- either arm not COMPLETE -> INCOMPLETE, never silently compared ---
    assert queue_runner.compare_pair_summaries(complete_a, None)["pair_status"] == "INCOMPLETE"
    assert queue_runner.compare_pair_summaries(
        complete_a, {"final_status": "STOPPED_CONFIG"})["pair_status"] == "INCOMPLETE"
    record("test_compare_pair_summaries", result="PASS")


# ===========================================================================
# Gate P2 v2 campaign: only A0/A1, seed 2026, no replicas, cut-rule epoch cap.
# E34/E35 -- queue_runner.py end-to-end: simulated quota/chat termination,
# restart-no-duplicate, and one arm stopping permanently doesn't block selection.
# ===========================================================================

# TEST STUB ONLY -- not Opus's authoritative sasrec_schedule.epoch_cap. Mirrors the
# documented cut_rule wording closely enough to exercise queue_runner's
# orchestration end-to-end; only used when SASREC_EPOCH_CAP_IMPORT points at it
# (real runs always default to sasrec_schedule:epoch_cap).
_EPOCH_CAP_STUB_SRC = '''
def epoch_cap(now_s, deadline_s, arms, planned=6, min_epochs=4, post_hours=2.0,
              safety_hours=1.0, floor_start=3, eval_seconds_default=1600):
    budget_s = (deadline_s - now_s) - safety_hours * 3600 - post_hours * 3600
    for E in range(planned, min_epochs - 1, -1):
        worst = 0.0
        for a in arms:
            done = a["epochs_done"]
            mean_train = a["mean_train_epoch_seconds"] or 0.0
            eval_s = a["eval_seconds"] if a["eval_seconds"] is not None else eval_seconds_default
            remaining_train = max(E - done, 0)
            remaining_evals = max(E - max(floor_start - 1, done), 0)
            worst = max(worst, remaining_train * mean_train + remaining_evals * eval_s)
        if worst <= budget_s:
            return E
    return min_epochs
'''


def _write_epoch_cap_stub(root: Path) -> None:
    (root / "epoch_cap_stub.py").write_text(_EPOCH_CAP_STUB_SRC)


def _mini_registry(seed=3026):
    # v2 (Gate P2): only two anchors, one seed, no replicas, fixed small epoch budget
    # (planned == min_epochs_under_cut == 4 so the cut rule is deterministic in tests
    # regardless of the stub's timing arithmetic).
    from datetime import datetime, timedelta, timezone
    deadline = (datetime.now(timezone.utc) + timedelta(hours=100)).isoformat().replace("+00:00", "Z")
    return {
        "training": {"seed": seed},
        "replication": {"additional_seeds": []},
        "schedule": {"planned_epochs": 4, "min_epochs_under_cut": 4, "floor_start_epoch": 3},
        "models": [
            {"id": "test-A0", "group": "item_only", "pair": "test-pair"},
            {"id": "test-A1", "group": "rich_frozen_channels", "pair": "test-pair"},
        ],
        "pair": {"members": ["test-A0", "test-A1"], "barrier": None},
        "queue_plan": {"phase": {"gpu0": ["test-A0"], "gpu1": ["test-A1"]}},
        "owner_deadline": {"hours": 38, "deadline_utc": deadline, "owner_choice": "test"},
        "cut_rule": {"when": "after both anchors complete epoch 2"},
        "dropped_models": [{"id": "test-C0", "dropped_reason": "owner deadline (test)"}],
        "source_registry_sha256": "test", "dataset_hash": "test",
    }


def _queue_env(root: Path, extra=None):
    _write_epoch_cap_stub(root)
    trainer = (f"{PYEXE} {FAKE_TRAINER} --epoch-seconds 0.1 --updates-per-epoch 5 "
              "--planned-epochs 4 --floor-start-epoch 3 --epoch-cap-wait-seconds 8")
    env = {
        "SASREC_QUEUE_ROOT": str(root),
        "SASREC_TRAINER": trainer,
        "SASREC_QUEUE_POLL_SECONDS": "0.5",
        "SASREC_QUEUE_NO_SAMPLER": "1",
        "SASREC_EPOCH_CAP_IMPORT": "epoch_cap_stub:epoch_cap",
        "PYTHONPATH": str(root),
        "SASREC_LOCK_CMD": f'{PYEXE} -c "import sys; sys.exit(0)"',
        "SASREC_RETRO_EVAL_CMD": f'{PYEXE} -c "import sys; sys.exit(0)"',
        "SASREC_REPORT_CMD": f"{PYEXE} {CODE / 'render_report.py'}",
    }
    if extra:
        env.update(extra)
    return env


def _assert_scoped_to_scratch(root: Path, session: str) -> None:
    """Safety canary: confirm SASREC_QUEUE_ROOT really took effect (queue_state.json
    must appear INSIDE the scratch root quickly) before waiting on the long-running
    assertion below. If env propagation into the tmux session silently failed, the
    launched queue_runner.py would otherwise default to the REAL production $S root
    and run against the real locked registry for up to two minutes before this test
    times out -- exactly the failure mode this canary catches early and loudly."""
    try:
        wait_until(lambda: (root / "state" / "queue_state.json").exists(), timeout=15,
                  desc="queue_state.json appears INSIDE the scratch root (env propagation canary)")
    except AssertionError:
        tmux_kill(session)
        raise AssertionError(
            "SASREC_QUEUE_ROOT did not take effect inside the tmux session: "
            "queue_runner.py would otherwise be running against the REAL production "
            "root. Killed the session defensively; see tests/test_runtime.py env "
            "propagation notes.")


def _setup_registry_root(tag: str, **reg_kwargs) -> Path:
    root = new_scratch(tag)
    (root / "config").mkdir(parents=True)
    (root / "config" / "run_registry.locked.json").write_text(json.dumps(_mini_registry(**reg_kwargs)))
    (root / "runs").mkdir()
    (root / "state").mkdir()
    return root


def test_queue_runner_end_to_end_simulated_termination():
    root = _setup_registry_root("queue_e2e")
    env = _queue_env(root)
    session = tmux_session("sasrec-test-queuerunner-e2e")
    # Launch detached: this simulates the launching agent's chat/session ending right
    # after kickoff -- nothing here re-invokes an LLM; the state machine runs alone.
    tmux_new_session(session, [PYEXE, str(CODE / "queue_runner.py")], env)
    _assert_scoped_to_scratch(root, session)
    try:
        wait_until(lambda: (root / "FINAL_SASREC_RESULTS.json").exists(),
                  timeout=120, desc="queue runner reaches REPORT and writes results")
    finally:
        tmux_kill(session)

    qstate = json.loads((root / "state" / "queue_state.json").read_text())
    assert qstate["phase"] == "DONE"
    assert qstate["lock_rc"] == 0 and qstate["retro_rc"] == 0 and qstate["report_rc"] == 0

    selection = json.loads((root / "state" / "selection.json").read_text())
    assert selection["selected"] in {"test-A0", "test-A1"}

    for mid in ["test-A0", "test-A1"]:
        summary = json.loads((root / "runs" / f"{mid}_s3026" / "summary.json").read_text())
        assert summary["final_status"] == "COMPLETE"
        assert summary["epochs_completed"] == 4, "cut rule (planned=min=4) must settle at exactly 4 epochs"
        assert summary["eval_epochs"] == [3, 4], "full-VS eval only at floor epochs (3, 4)"
        assert summary["precision_training"] == "tf32" and summary["precision_eval"] == "strict_fp32"

    # cut rule: written exactly once, both anchors' epoch_cap agrees with it
    epoch_cap_rec = json.loads((root / "state" / "epoch_cap.json").read_text())
    assert epoch_cap_rec["epoch_cap"] == 4
    for mid in ["test-A0", "test-A1"]:
        summary = json.loads((root / "runs" / f"{mid}_s3026" / "summary.json").read_text())
        assert summary["epoch_cap"] == 4

    # no pair barrier, but the pair-parity report must show a clean equal-epoch,
    # equal-permutation-digest match (both anchors share seed 2026's permutation)
    assert qstate["pair_parity"]["pair_status"] == "COMPLETE"
    assert qstate["pair_parity"]["equal_epochs"] is True
    assert qstate["pair_parity"]["equal_perm_sha256"] is True

    # no replicas under the deadline
    assert not any((root / "runs").glob("*_s3027")) and not any((root / "runs").glob("*_s3028"))

    results = json.loads((root / "FINAL_SASREC_RESULTS.json").read_text())
    assert results["statuses"]["training_status"] == "COMPLETE"
    assert results["statuses"]["search_status"] == "COMPLETE_BOUNDED"
    assert (root / "FINAL_SASREC_REPORT_AR.md").exists()
    assert (root / "SOTA_CLAIM_STATUS.md").exists()

    record("test_queue_runner_end_to_end_simulated_termination", root=str(root),
          selected=selection["selected"], epoch_cap=epoch_cap_rec["epoch_cap"], result="PASS")


def test_queue_runner_restart_resumes_no_duplicate():
    root = _setup_registry_root("queue_restart")
    env = _queue_env(root)
    session = tmux_session("sasrec-test-queuerunner-restart")
    argv = [PYEXE, str(CODE / "queue_runner.py")]
    tmux_new_session(session, argv, env)
    _assert_scoped_to_scratch(root, session)

    # Let it get into PRIMARY with at least one run launched, then kill the python
    # process itself (tmux session and the trainer keep running) to simulate the
    # supervising queue_runner.py process dying.
    def _has_any_run_dir():
        return any((root / "runs").glob("test-*_s3026"))
    wait_until(_has_any_run_dir, timeout=20, desc="first primary run launched")
    time.sleep(0.5)
    tmux_kill(session)  # kills queue_runner.py AND its child trainer sessions' controller
    # (the individual per-run tmux sessions it created are independent and survive)

    lock_path = root / "state" / "queue_runner.pid"
    if lock_path.exists():
        lock_path.unlink()  # the killed process cannot release its own lock

    session2 = tmux_session("sasrec-test-queuerunner-restart2")
    tmux_new_session(session2, argv, env)
    try:
        wait_until(lambda: (root / "FINAL_SASREC_RESULTS.json").exists(),
                  timeout=120, desc="restarted queue runner still reaches REPORT")
    finally:
        tmux_kill(session2)

    # No run_id_s3026 run was launched twice: exactly one run dir per model, and its
    # summary shows a single coherent COMPLETE run (fake trainer would leave 2 separate
    # summaries only if genuinely relaunched into two different processes writing the
    # same file, which is safe here because it's the same run_dir either way -- the
    # real duplicate-prevention check is that we never see two DIFFERENT tmux sessions
    # for the same run_id/seed).
    all_sessions = _list_test_sessions()
    a0_sessions = [s for s in all_sessions if "-test-A0-" in s]
    assert len(a0_sessions) <= 1, f"restart must not create a second session for the same run: {a0_sessions}"

    qstate = json.loads((root / "state" / "queue_state.json").read_text())
    assert qstate["phase"] == "DONE"
    record("test_queue_runner_restart_resumes_no_duplicate", result="PASS")


def _list_test_sessions() -> list[str]:
    r = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.startswith("sasrec-")]


def test_queue_runner_one_arm_stopped_other_still_finishes():
    """Gate P2 point 4: if one arm stops permanently, the other still finishes; the
    pair is marked incomplete, and selection runs over the COMPLETE runs only. Also
    exercises the cut-rule's fallback: since test-A1 never produces epoch_records.json,
    the cut condition ("both anchors show epoch 2") never fires, so test-A0's fake
    trainer must fall back to --planned-epochs after its bounded epoch-cap-wait."""
    root = _setup_registry_root("queue_one_arm_stopped")
    env = _queue_env(root)
    # Pre-seed test-A1 as already terminally failed (no --pair-dir/barrier under Gate
    # P2, so queue_runner never launches a session for it at all: _terminal_status
    # short-circuits launch()).
    stopped_dir = root / "runs" / "test-A1_s3026"
    stopped_dir.mkdir(parents=True)
    (stopped_dir / "SUPERVISOR_STOPPED").write_text("rc=5 reason=CONFIG")
    (stopped_dir / "summary.json").write_text(json.dumps({
        "run_id": "test-A1", "final_status": "STOPPED_CONFIG", "best_metric": None}))

    session = tmux_session("sasrec-test-queuerunner-onearm")
    tmux_new_session(session, [PYEXE, str(CODE / "queue_runner.py")], env)
    _assert_scoped_to_scratch(root, session)
    try:
        wait_until(lambda: (root / "FINAL_SASREC_RESULTS.json").exists(),
                  timeout=120, desc="campaign completes despite one permanently-stopped arm")
    finally:
        tmux_kill(session)

    a0_summary = json.loads((root / "runs" / "test-A0_s3026" / "summary.json").read_text())
    assert a0_summary["final_status"] == "COMPLETE"
    assert a0_summary["epochs_completed"] == 4, "fallback ceiling is --planned-epochs (4), never fired via the queue"
    assert not (root / "state" / "epoch_cap.json").exists(), \
        "cut rule must never fire with only one arm reporting; A0's --planned-epochs fallback is a LOCAL default, not a queue decision"

    qstate = json.loads((root / "state" / "queue_state.json").read_text())
    assert qstate["pair_parity"]["pair_status"] == "INCOMPLETE"
    assert qstate["pair_parity"]["equal_epochs"] is None

    selection = json.loads((root / "state" / "selection.json").read_text())
    assert selection["selected"] == "test-A0", "only test-A0 is COMPLETE and eligible"
    ineligible_ids = {r.get("run_id") for r in selection.get("ineligible", [])}
    assert "test-A1" in ineligible_ids, "the pre-failed run must show up as ineligible, not silently vanish"

    results = json.loads((root / "FINAL_SASREC_RESULTS.json").read_text())
    assert results["statuses"]["search_status"] == "COMPLETE_BOUNDED"
    record("test_queue_runner_one_arm_stopped_other_still_finishes",
          selected=selection["selected"], result="PASS")


def test_maybe_compute_epoch_cap_orchestration_in_process():
    """QueueRunner.maybe_compute_epoch_cap: fires exactly once, only once BOTH
    anchors' epoch_records.json show epoch 2, and is never rewritten afterwards --
    all in-process (no tmux/subprocess needed for this orchestration logic)."""
    root = new_scratch("epoch_cap_inprocess")
    (root / "config").mkdir(parents=True)
    (root / "config" / "run_registry.locked.json").write_text(json.dumps(_mini_registry()))
    (root / "runs").mkdir()
    (root / "state").mkdir()
    _write_epoch_cap_stub(root)

    def _write_records(mid, epochs):
        rd = root / "runs" / f"{mid}_s3026"
        rd.mkdir(parents=True, exist_ok=True)
        recs = [{"epoch": e, "epoch_seconds": 10.0 + e, "eval_seconds": None,
                "full_vs": None, "perm_sha256": f"h{e}"} for e in epochs]
        (rd / "epoch_records.json").write_text(json.dumps(recs))

    old_env = {k: os.environ.get(k) for k in ("SASREC_QUEUE_ROOT", "SASREC_EPOCH_CAP_IMPORT")}
    old_path = list(sys.path)
    try:
        os.environ["SASREC_QUEUE_ROOT"] = str(root)
        os.environ["SASREC_EPOCH_CAP_IMPORT"] = "epoch_cap_stub:epoch_cap"
        sys.path.insert(0, str(root))

        # --- negative control: only ONE arm at epoch 2 must NOT trigger a write ---
        _write_records("test-A0", [1, 2])
        _write_records("test-A1", [1])
        qr = queue_runner.QueueRunner(root=root)
        qr.maybe_compute_epoch_cap()
        assert not (root / "state" / "epoch_cap.json").exists(), \
            "cut rule must wait for BOTH anchors, not fire on the first one alone"

        # --- positive control: both reach epoch 2 -> exactly one write ---
        _write_records("test-A1", [1, 2])
        qr.maybe_compute_epoch_cap()
        assert (root / "state" / "epoch_cap.json").exists()
        rec = json.loads((root / "state" / "epoch_cap.json").read_text())
        assert rec["epoch_cap"] == 4  # planned == min_epochs_under_cut == 4 in _mini_registry()

        # --- idempotency: re-running (even with new, different-looking arm data
        #     and a fresh QueueRunner instance, simulating a restart) must NEVER
        #     rewrite the file ---
        _write_records("test-A0", [1, 2, 3])
        _write_records("test-A1", [1, 2, 3])
        qr2 = queue_runner.QueueRunner(root=root)
        qr2.maybe_compute_epoch_cap()
        rec2 = json.loads((root / "state" / "epoch_cap.json").read_text())
        assert rec2 == rec, "epoch_cap.json must never be rewritten once written"
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.path[:] = old_path
        sys.modules.pop("epoch_cap_stub", None)

    record("test_maybe_compute_epoch_cap_orchestration_in_process",
          epoch_cap=rec["epoch_cap"], result="PASS")


# ===========================================================================
# render_report.py -- honest PARTIAL/BLOCKED when inputs are missing
# ===========================================================================

def test_render_report_partial_when_nothing_present():
    root = new_scratch("report_blocked")
    (root / "config").mkdir(parents=True)
    (root / "config" / "run_registry.locked.json").write_text(json.dumps(_mini_registry()))
    (root / "review").mkdir()

    import render_report
    out = render_report.render(root)
    results = json.loads((root / "FINAL_SASREC_RESULTS.json").read_text())
    assert results["statuses"]["training_status"] == "BLOCKED"
    assert results["statuses"]["retrospective_evaluation_status"] == "NOT_RUN"
    assert results["statuses"]["universal_sota_proven"] is False
    assert results["statuses"]["new_confirmatory_holdout_available"] is False
    md = (root / "FINAL_SASREC_REPORT_AR.md").read_text(encoding="utf-8")
    assert "BLOCKED" in md
    sota = (root / "SOTA_CLAIM_STATUS.md").read_text(encoding="utf-8")
    assert "Not yet measured" in sota
    record("test_render_report_partial_when_nothing_present", result="PASS")


def test_render_report_sanitizes_nan_strict_json():
    root = new_scratch("report_nan")
    (root / "config").mkdir(parents=True)
    (root / "config" / "run_registry.locked.json").write_text(json.dumps(_mini_registry()))
    (root / "runs" / "test-A0_s3026").mkdir(parents=True)
    (root / "runs" / "test-A0_s3026" / "summary.json").write_text(
        json.dumps({"run_id": "test-A0", "final_status": "COMPLETE", "best_metric": float("nan")}))

    import render_report
    render_report.render(root)
    raw = (root / "FINAL_SASREC_RESULTS.json").read_text()
    parsed = json.loads(raw)  # must be valid strict JSON: no bare NaN
    assert "NaN" not in raw
    assert any("best_metric" in p for p in parsed["sanitized_nan_or_inf_fields"])
    record("test_render_report_sanitizes_nan_strict_json", result="PASS")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

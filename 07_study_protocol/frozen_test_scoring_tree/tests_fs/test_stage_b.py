"""v4fs.stage_b (FS-02): the Stage B driver -- decide, register S* (schema + theta0), the per-GPU training launch
commands, decide again, FS_FINAL.json. No GPU / dataset row is touched here: schema registration is real (a tmp
copy of contracts/derived_schemas), theta0 registration is exercised up to (and refused at) the real "no CUDA here"
boundary, and training itself is represented only by synthetic FS_SUMMARY.json / FS_USERSTATS.npz fixtures, exactly
like test_decide.py's own end-to-end test.
"""
from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from fskit import assert_raises, nc, smi_runner
from v4eval.bootstrap import UserStats
from v4fs import decide as D
from v4fs import importance as I
from v4fs import schemas as S
from v4fs import stage_b as B
from v4models.schema import SchemaError

N_USERS = 300


def stats(m):
    m = np.asarray(m, dtype=np.float64)
    n = np.ones_like(m)
    return UserStats(np.arange(m.size, dtype=np.int64), m.copy(), n, m.copy(), n.copy())


def base_m(seed=0):
    return np.random.default_rng(seed).uniform(0.05, 0.4, N_USERS)


def write_run(root, sid, fam, seed, endpoint, perms=None):
    """A synthetic FS run directory, named exactly like the real Stage A / Stage B output (<sid>__<FAM>__<seed>)."""
    d = B.run_dir(root, sid, fam, seed)
    d.mkdir(parents=True)
    st = {"endpoint": stats(endpoint)}
    if perms is not None:
        st["base"] = stats(endpoint)
        for g in I.FS_GROUPS:
            for p in I.PERM_SEEDS:
                st[f"{g}|{p}"] = stats(perms.get(g, endpoint))
    I.save_stats(d / "FS_USERSTATS.npz", st, {"seed": seed})
    (d / "FS_SUMMARY.json").write_text(json.dumps({"run_label": D.RUN_LABEL, "schema": {"schema_id": sid},
                                                   "family": fam, "seed": seed,
                                                   "userstats_file": "FS_USERSTATS.npz"}), encoding="utf-8")
    return d


def write_stage_a(root, *, l3_drop=("G8", "G12c"), l3_ok=True):
    """8 synthetic Stage A dirs (L0S/L3S x GRU/SASREC x 2026/2027) that make v4fs.decide drop exactly `l3_drop` and
    accept (or, l3_ok=False, fail) S* = L3S minus l3_drop for BOTH families. A group's permutation is degraded
    (positive importance: significant, kept) unless it is in `l3_drop` (or is G1), where it is left unchanged (zero
    importance: within noise, dropped) -- exactly test_decide.py's own `_write_run` convention."""
    never_drop = set(l3_drop) | {"G1"}
    for fam in D.FAMILIES:
        for seed in (2026, 2027):
            m = base_m(seed + (0 if fam == "GRU" else 10))
            write_run(root, D.L0_ID, fam, seed, m)
            l3 = m * (0.999 if l3_ok else 0.9)
            perms = {g: l3 * 0.9 for g in I.FS_GROUPS if g not in never_drop}
            write_run(root, D.L3_ID, fam, seed, l3, perms)


def write_sstar_pass(root, sid, *, gru_scale=0.995, sasrec_scale=0.995):
    for fam, scale in (("GRU", gru_scale), ("SASREC", sasrec_scale)):
        for seed in (2026, 2027):
            m = base_m(seed + (0 if fam == "GRU" else 10))
            write_run(root, sid, fam, seed, m * scale)


# --------------------------------------------------------------------------------------------------------- run dirs
def test_run_dir_naming_matches_fs_stageA():
    d = B.run_dir("ROOT", "L3S_NEAR_DUPLICATE_SESSION", "GRU", 2026)
    assert str(d).replace("\\", "/") == "ROOT/L3S_NEAR_DUPLICATE_SESSION__GRU__2026"


def test_stage_a_dirs_are_the_8_canonical_dirs(tmp_path):
    dirs = B.stage_a_dirs(tmp_path)
    assert len(dirs) == 8 and len(set(dirs)) == 8
    names = {d.name for d in dirs}
    for sid in ("L0S_CLEAN_V2_SESSION", "L3S_NEAR_DUPLICATE_SESSION"):
        for fam in ("GRU", "SASREC"):
            for seed in (2026, 2027):
                assert f"{sid}__{fam}__{seed}" in names


def test_missing_summaries(tmp_path):
    dirs = [tmp_path / "a", tmp_path / "b"]
    dirs[0].mkdir()
    (dirs[0] / "FS_SUMMARY.json").write_text("{}", encoding="utf-8")
    assert B.missing_summaries(dirs) == [dirs[1]]


# ------------------------------------------------------------------------------------------------- schema register
def _seed_registry(tmp_path):
    S.write_registry(out_dir=tmp_path)
    return tmp_path / "REGISTRY.json"


def test_register_schema_idempotent_and_cumulative_across_rounds(tmp_path):
    reg_path = _seed_registry(tmp_path)
    sid1 = D.sstar_id(["G8", "G12c"])
    r1 = B.register_schema(sid1, ["G8", "G12c"], registry_path=reg_path, out_dir=tmp_path)
    assert sid1 in r1["schemas"]
    self1 = r1["schemas"][sid1]["self_sha256"]
    calls = []
    r1b = B.register_schema(sid1, ["G8", "G12c"], registry_path=reg_path, out_dir=tmp_path,
                            log_fn=lambda m: calls.append(m))
    assert r1b["schemas"][sid1]["self_sha256"] == self1
    assert any("skip" in m for m in calls)                          # the second call is a genuine no-op
    sid2 = D.sstar_id(["G12c"])                                      # a later round's fallback (restore G8)
    r2 = B.register_schema(sid2, ["G12c"], registry_path=reg_path, out_dir=tmp_path)
    assert sid1 in r2["schemas"] and r2["schemas"][sid1]["self_sha256"] == self1     # write_registry kept it
    assert sid2 in r2["schemas"]
    for sid in (sid1, sid2):
        ds = S.load_registered(sid, reg_path)
        assert ds.self_sha256 == r2["schemas"][sid]["self_sha256"]


def test_register_schema_refuses_a_schema_id_mismatch(tmp_path):
    reg_path = _seed_registry(tmp_path)
    assert_raises(B.StageBError, B.register_schema, "NOT_THE_RIGHT_ID", ["G8"], registry_path=reg_path,
                  out_dir=tmp_path)


# ------------------------------------------------------------------------------------------------------ theta0 gate
def test_theta0_missing_empty_when_nothing_registered(tmp_path):
    p = tmp_path / "THETA0_REGISTRY.json"
    assert B.theta0_missing("SID", "a" * 64, "9.9.9+x", theta0_path=p) == [(f, s) for f in B.FAMILIES
                                                                            for s in B.SEEDS]


def test_gpu_precheck_ok_and_refusals():
    ok = B.gpu_precheck(1, runner=smi_runner(apps="", mem_used=10.0))
    assert ok["gpu_ready"] == "ok"
    assert_raises(B.StageBError, B.gpu_precheck, 1, runner=smi_runner(apps="0x02, 999", mem_used=10.0))
    assert_raises(B.StageBError, B.gpu_precheck, 1, runner=smi_runner(mem_used=9999.0))
    assert_raises(B.StageBError, B.gpu_precheck, 1, runner=smi_runner(fail=True))
    assert_raises(B.StageBError, B.gpu_precheck, 5, runner=smi_runner())            # GPU 5 not enumerated


def test_register_theta0_skips_when_already_registered(tmp_path, monkeypatch):
    import torch
    reg_path = _seed_registry(tmp_path)
    ds = S.load_registered("L3S_NEAR_DUPLICATE_SESSION", reg_path)
    theta0_path = tmp_path / "THETA0_REGISTRY.json"
    records = [{"schema_id": ds.schema_id, "schema_self_sha256": ds.self_sha256, "torch": torch.__version__,
               "family": fam, "seed": seed, "init_sha256": "a" * 64, "raw_init_sha256": "b" * 64,
               "manifest_digest": "c" * 64, "shared_numel": 1, "K": 1, "host": "x", "utc": "x"}
              for fam in B.FAMILIES for seed in B.SEEDS]
    S.merge_theta0(theta0_path, records)

    def boom(*a, **k):
        raise AssertionError("gpu_precheck must not run once every combo is already registered")
    monkeypatch.setattr(B, "gpu_precheck", boom)
    out = B.register_theta0(ds.schema_id, gpu=1, v1_root=tmp_path, out_dir=tmp_path, theta0_path=theta0_path,
                            schema_registry_path=reg_path)
    assert out == {"skipped": True}


def test_register_theta0_fails_closed_with_no_real_cuda_here(tmp_path):
    """A free-GPU nvidia-smi still refuses once torch itself reports no visible CUDA device (this machine): the
    real fail-closed boundary, exercised end to end short of an actual GPU build. `environ` is a plain injected
    dict (run_fs.export_environment's own testing pattern), so the real process environment is never touched."""
    reg_path = _seed_registry(tmp_path)
    sid = D.sstar_id(["G13"])
    B.register_schema(sid, ["G13"], registry_path=reg_path, out_dir=tmp_path)
    env = {}
    with pytest.raises(B.StageBError, match="visible CUDA device"):
        B.register_theta0(sid, gpu=1, v1_root=tmp_path, out_dir=tmp_path, theta0_path=tmp_path / "THETA0_REGISTRY.json",
                          schema_registry_path=reg_path, runner=smi_runner(apps="", mem_used=10.0), environ=env)
    assert env.get("CUDA_VISIBLE_DEVICES") == "1"       # the env was exported before the refusal (as designed)


# --------------------------------------------------------------------------------------------------------- launches
def test_launch_commands_one_per_gpu():
    cmds = B.launch_commands("SID", out_dir="/OUT", sh="/CODE/fs_stageB.sh")
    assert len(cmds) == 2
    assert any("0 2026 SID" in c for c in cmds) and any("1 2027 SID" in c for c in cmds)
    assert all("fs_stageB.sh" in c and "nohup" in c for c in cmds)


# ------------------------------------------------------------------------------------------------------ drive() e2e
def test_drive_locks_immediately_when_l3_passes(tmp_path):
    # NOTE: this scenario never reaches register_schema (no group is dropped: LOCK on round A), but every drive()
    # call in this file still pins the registry to a tmp path on purpose -- a real REGISTRY.json write bug here
    # once wrote a bogus "minus everything" schema straight into the committed contracts/derived_schemas/ tree
    # (caught only by test_schemas.py's own self-hash test); never rely on drive()'s registry defaults in a test.
    stage_a = tmp_path / "stageA"
    write_stage_a(stage_a, l3_ok=True, l3_drop=())            # no group within noise: S* = L3S itself
    out = tmp_path / "stageB"
    final = tmp_path / "FS_FINAL.json"
    derived_dir = out / "derived"
    reg_path = derived_dir / "REGISTRY.json"
    res = B.drive(stage_a_dir=stage_a, out_dir=out, final_path=final, dry_run=True, log_fn=lambda m: None,
                  schema_registry_path=reg_path, derived_out_dir=derived_dir)
    assert res["action"] == "LOCK" and res["schema_id"] == D.L3_ID
    assert json.loads(final.read_text(encoding="utf-8"))["schema_id"] == D.L3_ID
    assert (out / "DECISION_A.json").exists() and not (out / "DECISION_B.json").exists()
    assert not reg_path.exists()                                          # confirms register_schema was never run
    calls = []
    res2 = B.drive(stage_a_dir=stage_a, out_dir=out, final_path=final, dry_run=True, log_fn=calls.append,
                   schema_registry_path=reg_path, derived_out_dir=derived_dir)
    assert res2 == res and any("already written" in m for m in calls)      # a second call is a pure no-op


def test_drive_trains_sstar_then_locks(tmp_path):
    stage_a = tmp_path / "stageA"
    write_stage_a(stage_a, l3_ok=True, l3_drop=("G8", "G12c"))
    out = tmp_path / "stageB"
    final = tmp_path / "FS_FINAL.json"
    reg_path = out / "derived" / "REGISTRY.json"
    derived_dir = out / "derived"
    S.write_registry(out_dir=derived_dir)
    sid = D.sstar_id(["G8", "G12c"])
    r1 = B.drive(stage_a_dir=stage_a, out_dir=out, final_path=final, dry_run=True, log_fn=lambda m: None,
                schema_registry_path=reg_path, derived_out_dir=derived_dir)
    assert r1["status"] == "WAITING_FOR_OPS_TRAINING" and r1["schema_id"] == sid
    assert sorted(r1["dropped_groups"]) == ["G12c", "G8"]
    assert len(r1["launch_commands"]) == 2
    assert sid in S.read_registry(reg_path)["schemas"]                     # step 2 already ran
    assert (out / "DECISION_A.json").exists()
    assert not final.exists()
    write_sstar_pass(out, sid)                                             # Ops "ran" the queues
    r2 = B.drive(stage_a_dir=stage_a, out_dir=out, final_path=final, dry_run=True, log_fn=lambda m: None,
                schema_registry_path=reg_path, derived_out_dir=derived_dir)
    assert r2["action"] == "LOCK" and r2["schema_id"] == sid
    assert (out / "DECISION_B.json").exists()
    final_obj = json.loads(final.read_text(encoding="utf-8"))
    assert final_obj["schema_id"] == sid and len(final_obj["rounds"]) == 2


def test_drive_the_single_fallback_round(tmp_path):
    stage_a = tmp_path / "stageA"
    write_stage_a(stage_a, l3_ok=True, l3_drop=("G8", "G12c"))
    out = tmp_path / "stageB"
    final = tmp_path / "FS_FINAL.json"
    derived_dir = out / "derived"
    reg_path = derived_dir / "REGISTRY.json"
    S.write_registry(out_dir=derived_dir)
    sid1 = D.sstar_id(["G8", "G12c"])
    r1 = B.drive(stage_a_dir=stage_a, out_dir=out, final_path=final, dry_run=True, log_fn=lambda m: None,
                schema_registry_path=reg_path, derived_out_dir=derived_dir)
    assert r1["schema_id"] == sid1
    write_sstar_pass(out, sid1, gru_scale=0.97, sasrec_scale=0.995)          # S* fails for GRU (the worse family)
    r2 = B.drive(stage_a_dir=stage_a, out_dir=out, final_path=final, dry_run=True, log_fn=lambda m: None,
                schema_registry_path=reg_path, derived_out_dir=derived_dir)
    assert r2["status"] == "WAITING_FOR_OPS_TRAINING"
    sid2 = r2["schema_id"]
    assert sid2 != sid1 and sid1 in S.read_registry(reg_path)["schemas"]     # sid1 is still registered
    assert sid2 in S.read_registry(reg_path)["schemas"]
    write_sstar_pass(out, sid2)
    r3 = B.drive(stage_a_dir=stage_a, out_dir=out, final_path=final, dry_run=True, log_fn=lambda m: None,
                schema_registry_path=reg_path, derived_out_dir=derived_dir)
    assert r3["action"] == "LOCK" and r3["schema_id"] == sid2
    assert (out / "DECISION_C.json").exists()
    assert len(json.loads(final.read_text(encoding="utf-8"))["rounds"]) == 3


def test_drive_refuses_train_without_v1_root_when_not_dry_run(tmp_path):
    stage_a = tmp_path / "stageA"
    write_stage_a(stage_a, l3_ok=True, l3_drop=("G8", "G12c"))
    out = tmp_path / "stageB"
    derived_dir = out / "derived"
    S.write_registry(out_dir=derived_dir)
    assert_raises(B.StageBError, B.drive, stage_a_dir=stage_a, out_dir=out, final_path=tmp_path / "FS_FINAL.json",
                  dry_run=False, log_fn=lambda m: None, schema_registry_path=derived_dir / "REGISTRY.json",
                  derived_out_dir=derived_dir)


# ------------------------------------------------------------------------------------------------ merge_theta0 (S.)
def _rec(sid, self_sha, torch_build, fam, seed, init):
    return {"schema_id": sid, "schema_self_sha256": self_sha, "torch": torch_build, "family": fam, "seed": seed,
            "init_sha256": init, "raw_init_sha256": init, "manifest_digest": init, "shared_numel": 1, "K": 1,
            "host": "x", "utc": "x"}


def test_merge_theta0_accumulates_and_keeps_the_self_hash_rule(tmp_path):
    p = tmp_path / "THETA0_REGISTRY.json"
    r1 = S.merge_theta0(p, [_rec("SID", "s" * 64, "9.9+x", "GRU", 2026, "a" * 64)])
    assert r1["self_sha256"] == S.self_hash(r1)
    r2 = S.merge_theta0(p, [_rec("SID", "s" * 64, "9.9+x", "GRU", 2027, "b" * 64)])
    assert set(r2["schemas"]["SID"]["builds"]["9.9+x"]["GRU"]) == {"2026", "2027"}
    assert_raises(SchemaError, S.merge_theta0, p, [_rec("SID", "s" * 64, "9.9+x", "GRU", 2026, "c" * 64)])
    assert not p.with_name(p.name + ".lock").exists()                       # the lock is always released


def test_merge_theta0_two_threads_neither_update_is_lost(tmp_path):
    p = tmp_path / "THETA0_REGISTRY.json"
    errors = []

    def go(fam):
        try:
            S.merge_theta0(p, [_rec("SID", "s" * 64, "9.9+x", fam, 2026, "a" * 64)])
        except Exception as e:                                              # pragma: no cover - would fail the test
            errors.append(e)
    t1, t2 = threading.Thread(target=go, args=("GRU",)), threading.Thread(target=go, args=("SASREC",))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors
    reg = S.read_theta0(p)
    assert set(reg["schemas"]["SID"]["builds"]["9.9+x"]) == {"GRU", "SASREC"}


@nc("a stale lock file must not be silently ignored: merge_theta0 must time out, not overwrite past it")
def test_nc_merge_theta0_ignores_a_stale_lock():
    naive_would_proceed = True                                              # the naive (unsafe) behaviour
    assert not naive_would_proceed, "a naive merge would proceed even with another writer's lock held"


def test_merge_theta0_times_out_on_a_held_lock(tmp_path):
    p = tmp_path / "THETA0_REGISTRY.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = p.with_name(p.name + ".lock")
    lock.write_text("held", encoding="utf-8")
    try:
        assert_raises(SchemaError, S.merge_theta0, p, [_rec("SID", "s" * 64, "9.9+x", "GRU", 2026, "a" * 64)],
                      timeout_s=0.3, poll_s=0.05)
    finally:
        lock.unlink(missing_ok=True)

"""FL-12/13 (PREG1_AMENDMENT_06J §3 S4, 06K §2 S7, 06K §3 S8, DL 206) — local passes / batch, warm start (FA_WARM) and
frozen item tables (FA_WARM_FROZEN) in launch_fl and lr_screen (CPU, tiny FL worlds of test_fl05_launch).

Proves: every default (arm spec, RunConfig) digest and the default tiny trajectory are byte-identical to the pre-FL-12
code (the FL-10 224-pair pin and trajectory pin); S4: the 06J §3 round count ceil(60 N / (576 P)) keeps the 6.0-EFE
budget (6,827 / 1,707 at P = 2; 13,654 / 3,414 at P = 1; 3,414 / 854 at P = 4), is recorded (exposure_budget) and
re-verified, the Poisson arms refuse P != 2, `select-joint --stage S4` writes the S4 file (one at a time at the S3
winner, ties -> the incumbent), launch_fl verifies it and a MAIN spec follows it, an S4 run resumes bitwise; S7:
theta_pre is loaded exactly (the checkpoint file sha256 = the init record's, the loaded state hashes to the checkpoint's
state_digest = the server's init), a checkpoint that is not the PRE run's endpoint_6.0 or that was trained on MAIN users
(C_BIG / C_SMALL / C_FULL; MAIN / CALIB / FULL_TRAIN manifests) is refused, FA_WARM is recorded in the spec / rows and
resumes bitwise; S8: the frozen tables stay bitwise identical over a run, bytes per visit = the uploaded parameters with
the tables counted once per client, resume bitwise; `select-warm` writes the S7 / S8 file (FL-15: the registered 2-D
grid of 06M_B B2, 4 runs; the 2-run screen is refused) and a MAIN warm run binds it (registered path, the bound tuning
file, the same theta_pre). NCs: a MAIN-trained checkpoint accepted
by a guard without the MAIN checks; a warm run that skips theta_pre; S4 without the round recomputation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import v4int.launch_fl as F
import v4int.lr_screen as L
from fl.client import valid_rows
from fl.numerics import state_digest
from fl.synthetic import make_clients, make_tiny_adapter
from intkit import assert_raises, nc
from test_fl05_launch import fl05_world, run_tiny, tiny_spec
from test_fl10_fedadam_launch import (CALIB_EVAL, PRE_FL10_SHA, PRE_FL10_TRAJ, _label_dirs, _screen, _tiny,  # noqa: F401
                                      all_grid_digests, ladder)

T = F.tuning_config
N_TINY = 1152                                     # fa_rounds(1152, 2) = 60 = the tiny screens' round count
PRE_ID = "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026"


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ worlds / helpers
def untied_world(n=10, seed=5, eval_id="M02_VALIDATION_WARM", server=None):
    """fl05_world with the untied (SASRec-like) tiny family, so the item tables can be frozen."""
    server = server or make_tiny_adapter(seed=1, tied=False)
    worker = make_tiny_adapter(seed=1, tied=False)
    clients = {c.key: c for c in make_clients(n, seed=seed)}
    nv = {k: int(valid_rows(c.examples).numel()) for k, c in clients.items()}

    def evaluate(mark):
        v = float(abs(sum(p.double().sum() for p in server.module.parameters())) % 1.0)
        rows = []
        for agg in ("micro", "macro"):
            for den in ("E2E", "RANKABLE"):
                for met, k in (("mrr", 20), ("hr", 10)):
                    main = (agg, den, met, k) == ("macro", "E2E", "mrr", 20)
                    rows.append({"aggregation": agg, "target_denominator": den, "metric": met, "k": k,
                                 "value": v if main else v / 3, "value_status": "OK", "split": "VALIDATION",
                                 "eval_manifest_id": eval_id, "method": "C", "training_paradigm": "CENTRAL",
                                 "row_status": "CENTRAL_CONTEST", "run_label": "PRODUCTION"})
        return v, rows, np.arange(5), {"E2E": {"n": 5}}
    return dict(server_adapter=server, init_sha256=state_digest(server.broadcast_state()), workers=[worker],
                keys=sorted(clients), n_valid_by_key=nv, get_client=clients.__getitem__,
                n_decisions=sum(nv.values()), manifest_hash="tiny-manifest", evaluate=evaluate)


def s4_spec(arm="FA", rounds=60, **kw):
    """A tiny spec; a non-default pass count gets the exposure budget of N_TINY (rounds = fa_rounds(N_TINY, P))."""
    s = tiny_spec(arm, rounds=rounds, **kw)
    p = s["cfg"]["solver"]["passes"]
    if p != F.KNOBS["passes"]:
        eb = F.exposure_budget(N_TINY, p)
        s["exposure_budget"] = eb
        s["cfg"]["endpoint_rounds"] = s["max_rounds"] = eb["rounds"]
    return s


def make_pre(root: Path, *, seed=77, run_id=PRE_ID, run_label="PRETRAIN_FOR_FL", manifest="PRETRAIN_TRAIN_5_100_NONMAIN",
             kind="endpoint_6.0", variant="SASREC_D64_B2", rec_sha=None):
    """A PRE-run-shaped weights checkpoint (v4train endpoint_6.0 layout) + its PRETRAIN_FOR_FL_INIT.json."""
    pre = make_tiny_adapter(seed=seed, tied=False)
    ck = root / "sealed" / "checkpoints" / f"{kind}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "run_label": run_label, "family": "SASREC", "model_variant": variant, "seed": 2026,
            "state_digest": state_digest(pre.broadcast_state()), "efe": 6.0,
            "manifest_registry": {"train_manifest": manifest, "eval_manifest": "UNEW_VAL_VALIDATION_32768"}}
    torch.save({"kind": kind, "meta": meta, "model": pre.module.state_dict()}, ck)
    rec = {"run_id": run_id, "run_label": run_label, "train_manifest": manifest,
           "fl_init": {"kind": kind, "path": f"sealed/checkpoints/{kind}.pt",
                       "sha256": rec_sha or hashlib.sha256(ck.read_bytes()).hexdigest(), "bytes": ck.stat().st_size}}
    p = root / "PRETRAIN_FOR_FL_INIT.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p, pre


@pytest.fixture(scope="module")
def pre(tmp_path_factory):
    root = tmp_path_factory.mktemp("pre")
    p, adapter = make_pre(root)
    state, ident, launch = F.load_pretrain_init(p, model_variant="SASREC_D64_B2")
    return {"path": p, "adapter": adapter, "state": state, "ident": ident, "launch": launch}


def warm_world(pre, **kw):
    srv = make_tiny_adapter(seed=1, tied=False)
    F.apply_init_state(srv, pre["state"], pre["ident"]["state_digest"])
    return untied_world(server=srv, **kw)


# ------------------------------------------------------------------------------------------------ digest invariance
def test_default_digests_and_trajectory_unchanged(tmp_path):
    d = all_grid_digests()
    assert len(d) == 224 and hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest() == PRE_FL10_SHA
    assert F.ARMS == ("FA", "FP", "PF", "FA_1024", "DP", "S_CAL", "FA_Q8")
    for arm in F.ARMS:                                       # explicit defaults write no key
        kw = {"clip_norm": 1.0} if arm in ("FA_1024", "DP") else {}
        kw.update({"z": 0.5642} if arm == "DP" else {})
        a = F.arm_spec(arm, seed=2026, peak_lr=0.000711, local_passes=2, local_batch=16, **kw)
        assert a == F.arm_spec(arm, seed=2026, peak_lr=0.000711, **kw)
        assert not {"local_passes", "local_batch", "exposure_budget", "init_from", "warm_lr_factor",
                    "frozen_item_tables"} & set(a)
    assert run_tiny(tmp_path, _tiny("FA"), "FA")["summary"]["server_digest"] == PRE_FL10_TRAJ["FA"]
    assert F.tuning_label(T()) == "fedavg|m=1" and F.row_labels(T()) == {}


# ------------------------------------------------------------------------------------------------ S4
def test_s4_round_count_formula_and_budget():
    assert (F.fa_rounds(131072), F.fa_rounds(32768)) == (6827, 1707) == (F.KNOBS["rounds"], F.CALIB["rounds"])
    assert (F.fa_rounds(131072, 1), F.fa_rounds(32768, 1)) == (13654, 3414)
    assert (F.fa_rounds(131072, 4), F.fa_rounds(32768, 4)) == (3414, 854)
    for n in (131072, 32768, N_TINY, 1000, 12345):
        for p in F.LOCAL_PASSES:
            r = F.fa_rounds(n, p)                           # the smallest T whose expected exposures reach 6.0 EFE
            assert r * 64 * 0.9 * p >= 6.0 * n - 1e-6 and (r - 1) * 64 * 0.9 * p < 6.0 * n
    for p, rounds in ((1, 13654), (4, 3414)):
        s = F.arm_spec("FA", seed=2027, peak_lr=0.000711, local_passes=p)
        assert s["cfg"]["endpoint_rounds"] == s["max_rounds"] == rounds and s["cfg"]["solver"]["passes"] == p
        assert s["exposure_budget"] == F.exposure_budget(131072, p) and s["local_passes"] == p
        assert F.spec_tuning(s)["local_passes"] == p
        c = F.arm_spec("FA", seed=2026, peak_lr=0.000711, local_passes=p, role="CALIBRATION")
        assert c["cfg"]["endpoint_rounds"] == F.fa_rounds(32768, p)
    b = F.arm_spec("FA", seed=2026, peak_lr=0.000711, local_batch=8)
    assert b["cfg"]["solver"]["batch_size"] == 8 and b["cfg"]["endpoint_rounds"] == 6827 and "exposure_budget" not in b
    for arm, kw in (("FA_1024", {"clip_norm": 1.0}), ("DP", {"clip_norm": 1.0, "z": 0.5642}), ("S_CAL", {})):
        assert_raises(F.LaunchRefused, F.arm_spec, arm, seed=2026, peak_lr=0.000711, local_passes=1, **kw)
        assert_raises(F.LaunchRefused, F.arm_spec, arm, seed=2026, peak_lr=0.000711, local_batch=8, **kw)  # DL 219
    for kw in (dict(local_passes=3), dict(local_batch=32), dict(local_passes=1, local_batch=8), dict(local_passes=1.5)):
        assert_raises(F.LaunchRefused, T, **kw)
    assert F.tuning_label(T(local_passes=4)) == "fedavg|m=1|passes=4"
    assert F.row_labels(T(local_batch=8)) == {"fl_local_batch": 8}
    s = F.arm_spec("FA", seed=2027, peak_lr=0.000711, local_passes=1)
    for bad in (dict(s, cfg=dict(s["cfg"], endpoint_rounds=6827)),              # budget not recomputed
                {k: v for k, v in s.items() if k != "exposure_budget"},
                dict(s, cfg=dict(s["cfg"], solver=dict(s["cfg"]["solver"], passes=2)))):
        assert_raises(F.LaunchRefused, F.spec_tuning, bad)


def test_s4_preflight_refusals_and_run_ids(tmp_path):
    base = ["--arm", "FA", "--seed", "2026", "--gpu", "0", "--run-dir", str(tmp_path / "r"),
            "--history-policy", "USER_HISTORY_L50"]
    calib = base + ["--role", "CALIBRATION", "--smoke"]
    a = F.parse_args(calib + ["--local-passes", "1"])
    F.fl05_preflight(a)
    assert F.default_run_id(a).endswith("FL_FA_LRSCREEN_P1_x1_CALIB_s2026")
    assert F.default_run_id(F.parse_args(calib + ["--local-batch", "8", "--lr-mult", "2"])).endswith(
        "LRSCREEN_B8_x2_CALIB_s2026")
    for extra in (["--local-passes", "3"], ["--local-batch", "4"], ["--local-passes", "1", "--local-batch", "8"],
                  ["--local-passes", "1", "--fl-tuning", str(tmp_path / "t.json")]):
        assert_raises(F.LaunchRefused, F.fl05_preflight, F.parse_args(calib + extra))
    assert_raises(F.LaunchRefused, F.fl05_preflight,
                  F.parse_args(base + ["--local-passes", "1", "--lr-selection", str(tmp_path / "s.json")]))
    main = F.parse_args(base + ["--local-passes", "1"])       # non-smoke MAIN: only the registered selection
    assert_raises(F.LaunchRefused, F.require_lr_selection, main)


@pytest.fixture(scope="module")
def s4(ladder, tmp_path_factory):
    lad = ladder
    root = tmp_path_factory.mktemp("s4")
    runs3 = [lad["w2_dir"]] + lad["s3"]
    s3_file = root / "S3.json"
    L.write_selection_stage("S3", runs3, s3_file, lad["s2_file"])
    w3 = F.tuning_config(**{k: _json(s3_file)["winner"][k] for k in F.TUNING_KEYS})
    inc = _label_dirs(runs3)[F.tuning_label(w3)]
    kw = F.tuning_kwargs(w3)
    runs = {}
    for name, extra in (("p1", dict(local_passes=1)), ("p4", dict(local_passes=4)), ("b8", dict(local_batch=8))):
        spec = s4_spec("FA", role="CALIBRATION", **dict(kw, **extra))
        out = run_tiny(root, spec, name, world=fl05_world(eval_id=CALIB_EVAL))
        assert out["status"] == "DONE_MAIN", name
        runs[name] = root / name
    return {"root": root, "s3_file": s3_file, "w3": w3, "inc": inc, "runs": runs, "kw": kw}


def test_s4_selection_file_and_refusals(s4, tmp_path):
    S = L.selection_stage
    runs = [s4["inc"]] + list(s4["runs"].values())
    out = tmp_path / "S4.json"
    said = []
    argv = ["select-joint", "--stage", "S4", "--runs", *map(str, runs), "--first-selection", str(s4["s3_file"]),
            "--out", str(out)]
    assert L.main(argv, say=said.append) == 0
    obj = _json(out)
    assert obj["stage"] == "S4" and len(obj["configs"]) == 4 and obj["incumbent"] == F.tuning_label(s4["w3"])
    assert [h["stage"] for h in obj["stage_history"]] == ["S1b", "S2", "S3"]
    assert sorted(obj["required"]) == sorted(F.tuning_label(dict(s4["w3"], **e)) for e in
                                             (dict(local_passes=1), dict(local_passes=4), dict(local_batch=8)))
    loaded = F.load_fl_tuning(out)
    assert set(json.loads(said[0])["winner"]) == set(F.TUNING_KEYS)
    assert {k: loaded["winner"][k] for k in F.TUNING_KEYS} == json.loads(said[0])["winner"]
    assert L.main(argv, say=said.append) == 2                                      # O_EXCL
    assert_raises(L.SelectionRefused, S, "S4", runs[:-1], s4["s3_file"])          # batch 8 missing
    assert_raises(L.SelectionRefused, S, "S4", runs[1:], s4["s3_file"])           # no incumbent
    assert_raises(L.SelectionRefused, S, "S4", runs, s4["s3_file"].parent / "nope.json")
    extra = run_tiny(tmp_path, s4_spec("FA", role="CALIBRATION", local_passes=1, lr_mult=0.5), "other",
                     world=fl05_world(eval_id=CALIB_EVAL))
    assert extra["status"] == "DONE_MAIN"
    assert_raises(L.SelectionRefused, S, "S4", runs + [tmp_path / "other"], s4["s3_file"])   # not at the S3 winner
    # the S4 file drives a MAIN spec (smoke wiring) with the recomputed MAIN round count
    args = F.parse_args(["--arm", "FA", "--seed", "2027", "--gpu", "0", "--run-dir", str(tmp_path / "r"),
                         "--history-policy", "USER_HISTORY_L50", "--fl-tuning", str(out), "--smoke"])
    spec, extra_rec = F.resolve_spec(args, 0.000711, fused=True)
    t = F.spec_tuning(spec)
    assert t == F.tuning_config(**{k: loaded["winner"][k] for k in F.TUNING_KEYS})
    assert spec["cfg"]["endpoint_rounds"] == F.fa_rounds(131072, t["local_passes"])
    assert extra_rec["fl_tuning"]["stage"] == "S4"
    # a tampered stage history is refused
    bad = dict(obj, stage_history=obj["stage_history"][:2])
    bad["self_sha256"] = F._canon(bad)
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    assert_raises(F.LaunchRefused, F.load_fl_tuning, tmp_path / "bad.json")


def test_s4_tie_order():
    def c(v, inc=False, **kw):
        return dict(T(optimizer="fedavgm", lr_mult=4.0, **kw), value=v, incumbent=inc)
    w, tie = L.apply_rule_stage("S4", [c(0.1, True), c(0.1004, local_passes=1), c(0.1003, local_passes=4),
                                       c(0.1002, local_batch=8)])
    assert w == T(optimizer="fedavgm", lr_mult=4.0) and len(tie) == 1                       # B1 (and tie) -> default
    w, tie = L.apply_rule_stage("S4", [c(0.1, True), c(0.2, local_passes=4), c(0.2, local_passes=1),
                                       c(0.2004, local_batch=8)])
    assert w["local_batch"] == 8 and tie == ["fedavgm|m=4|batch=8", "fedavgm|m=4|passes=1", "fedavgm|m=4|passes=4"]
    w, tie = L.apply_rule_stage("S4", [c(0.1, True), c(0.2, local_passes=4), c(0.2003, local_passes=1)])
    assert w["local_passes"] == 1 and tie == ["fedavgm|m=4|passes=1", "fedavgm|m=4|passes=4"]
    w, _ = L.apply_rule_stage("S4", [c(0.1, True), c(0.3, local_batch=8)])
    assert w["local_batch"] == 8


def test_s4_resume_bitwise(tmp_path):
    spec = s4_spec("FA", local_passes=1)
    straight = run_tiny(tmp_path, spec, "s")
    first = run_tiny(tmp_path, spec, "p", interrupt_at=37)
    assert first["status"] == "INTERRUPTED"
    res = run_tiny(tmp_path, spec, "p", resume=True)
    assert res["status"] == "DONE_MAIN" and res["summary"]["server_digest"] == straight["summary"]["server_digest"]
    rows = _json(tmp_path / "s" / "RESULT_ROWS.json")["rows"]
    assert all(r["fl_local_passes"] == 1 for r in rows)
    default = run_tiny(tmp_path, s4_spec("FA"), "d")
    assert default["summary"]["server_digest"] != straight["summary"]["server_digest"]
    assert straight["summary"]["rounds"] == 2 * default["summary"]["rounds"]


# ------------------------------------------------------------------------------------------------ S7 warm start
def test_warm_init_loads_exactly_theta_pre(pre):
    ident = pre["ident"]
    assert ident["checkpoint_sha256"] == json.loads(pre["path"].read_text())["fl_init"]["sha256"]
    srv = make_tiny_adapter(seed=1, tied=False)
    theta0 = state_digest(srv.broadcast_state())
    got = F.apply_init_state(srv, pre["state"], ident["state_digest"])
    assert got == state_digest(pre["adapter"].broadcast_state()) == ident["state_digest"] != theta0
    assert_raises(F.LaunchRefused, F.apply_init_state, make_tiny_adapter(seed=1, tied=False), pre["state"], "0" * 64)
    F.check_init_identity(ident)
    assert pre["launch"]["init_record_sha256"] == hashlib.sha256(pre["path"].read_bytes()).hexdigest()


def test_warm_init_refusals(tmp_path):
    cases = {"sha": dict(rec_sha="0" * 64), "kind": dict(kind="best"), "label": dict(run_label="PRODUCTION"),
             "main_manifest": dict(manifest="MAIN_TRAIN_5_100_131072"),
             "full_manifest": dict(manifest="FULL_TRAIN_5_100_TIEBREAK"), "calib": dict(manifest="CALIB_TRAIN_5_100_32768"),
             "c_full": dict(run_id="C_FULL_02B_SASREC_s2026"), "c_big": dict(run_id="C_BIG_SASREC_s2026"),
             "c_small": dict(run_id="C_SMALL_SASREC_D64_B2_s2026"), "variant": dict(variant="SASREC_D256_B2")}
    for name, kw in cases.items():
        p, _ = make_pre(tmp_path / name, **kw)
        assert_raises(F.LaunchRefused, F.load_pretrain_init, p, model_variant="SASREC_D64_B2")
    p, _ = make_pre(tmp_path / "tamper")
    ck = tmp_path / "tamper" / "sealed" / "checkpoints" / "endpoint_6.0.pt"
    ck.write_bytes(ck.read_bytes() + b"\0")                                          # bytes differ from the record
    assert_raises(F.LaunchRefused, F.load_pretrain_init, p, model_variant="SASREC_D64_B2")


def test_warm_arm_spec_rows_and_refusals(pre, tmp_path):
    ident = pre["ident"]
    s = F.arm_spec("FA_WARM", seed=2026, peak_lr=0.000711, init_from=ident)
    assert s["cfg"]["method"] == "FA" and s["init_from"] == ident and "warm_lr_factor" not in s
    assert "frozen_item_tables" not in s and F.METHOD_CODE["FA_WARM"] == "FA"
    q = F.arm_spec("FA_WARM", seed=2026, peak_lr=0.000711, init_from=ident, warm_lr_factor=0.25, lr_mult=2.0)
    assert q["warm_lr_factor"] == 0.25 and q["cfg"]["lr_peak_local"] == 0.25 * (2.0 * 0.000711)
    fz = F.arm_spec("FA_WARM_FROZEN", seed=2026, peak_lr=0.000711, init_from=ident, warm_lr_factor=4.0)
    assert fz["frozen_item_tables"] == list(F.FROZEN_ITEM_KEYS) and fz["warm_lr_factor"] == 4.0
    for arm, kw in (("FA_WARM", {}), ("FA", {"init_from": ident}), ("FA_WARM", {"init_from": ident, "warm_lr_factor": 4.0}),
                    ("FA_WARM_FROZEN", {"init_from": ident, "warm_lr_factor": 0.25}),
                    ("FA", {"warm_lr_factor": 0.25}), ("FA_WARM", {"init_from": dict(ident, train_manifest="MAIN_TRAIN_5_100_131072")})):
        assert_raises(F.LaunchRefused, F.arm_spec, arm, seed=2026, peak_lr=0.000711, **kw)
    out = run_tiny(tmp_path, tiny_spec("FA_WARM", init_from=ident, warm_lr_factor=0.25), "w", world=warm_world(pre))
    assert out["status"] == "DONE_MAIN"
    rows = _json(tmp_path / "w" / "RESULT_ROWS.json")["rows"]
    assert rows and all(r["method"] == "FA_WARM" and r["method_code"] == "FA" and r["fl_init"] == "PRETRAIN_FOR_FL"
                        and r["fl_init_checkpoint_sha256"] == ident["checkpoint_sha256"] and r["fl_warm_lr_factor"] == 0.25
                        for r in rows)
    assert _json(tmp_path / "w" / "ARM_SPEC.json")["arm_spec"]["init_from"] == ident


def test_warm_preflight_refusals(pre, tmp_path):
    base = ["--seed", "2026", "--gpu", "0", "--run-dir", str(tmp_path / "r"), "--history-policy", "USER_HISTORY_L50"]
    init = ["--init-from", str(pre["path"])]
    ok = F.parse_args(["--arm", "FA_WARM", "--smoke"] + base + init + ["--warm-lr-factor", "0.25"])
    F.fl05_preflight(ok)
    assert F.default_run_id(F.parse_args(["--arm", "FA_WARM", "--role", "CALIBRATION", "--smoke", "--lr-mult", "2"]
                                         + base + init + ["--warm-lr-factor", "0.25"])).endswith(
        "FL_FA_WARM_LRSCREEN_x2_wf0.25_CALIB_s2026")
    F.fl05_preflight(F.parse_args(["--arm", "FA_WARM_FROZEN", "--smoke", "--freeze-item-tables"] + base + init))
    for argv in (["--arm", "FA_WARM", "--smoke"] + base,                                    # no --init-from
                 ["--arm", "FA", "--smoke"] + base + init,                                  # init on a cold arm
                 ["--arm", "FA_WARM", "--smoke", "--freeze-item-tables"] + base + init,
                 ["--arm", "FA_WARM_FROZEN", "--smoke"] + base + init,                      # flag missing
                 ["--arm", "FA_WARM", "--smoke", "--warm-lr-factor", "0.5"] + base + init,
                 ["--arm", "FA_WARM"] + base + init,                                        # MAIN: no --warm-selection
                 ["--arm", "FA_WARM", "--warm-lr-factor", "0.25"] + base + init,            # MAIN: factor refused
                 ["--arm", "FA_WARM", "--smoke", "--warm-selection", str(tmp_path / "w.json")] + base + init):
        assert_raises(F.LaunchRefused, F.fl05_preflight, F.parse_args(argv))


def test_warm_resume_bitwise(pre, tmp_path):
    spec = tiny_spec("FA_WARM", init_from=pre["ident"])
    straight = run_tiny(tmp_path, spec, "s", world=warm_world(pre))
    assert run_tiny(tmp_path, spec, "p", world=warm_world(pre), interrupt_at=29)["status"] == "INTERRUPTED"
    res = run_tiny(tmp_path, spec, "p", world=warm_world(pre), resume=True)
    assert res["summary"]["server_digest"] == straight["summary"]["server_digest"]
    cold = run_tiny(tmp_path, tiny_spec("FA"), "c", world=untied_world())
    assert cold["summary"]["server_digest"] != straight["summary"]["server_digest"]


# ------------------------------------------------------------------------------------------------ S8 frozen tables
def test_frozen_run_tables_bytes_and_resume(pre, tmp_path):
    spec = tiny_spec("FA_WARM_FROZEN", init_from=pre["ident"])
    w = warm_world(pre)
    t0 = {k: v.clone() for k, v in pre["adapter"].broadcast_state().items() if k in F.FROZEN_ITEM_KEYS}
    out = run_tiny(tmp_path, spec, "f", world=w)
    assert out["status"] == "DONE_MAIN"
    st = w["server_adapter"].broadcast_state()
    assert all(torch.equal(t0[k], st[k]) for k in F.FROZEN_ITEM_KEYS)              # bitwise at theta_pre
    by = out["summary"]["bytes"]
    full = make_tiny_adapter(seed=1, tied=False).manifest
    fb = sum(full.entries[k].nbytes for k in F.FROZEN_ITEM_KEYS)
    assert by["per_visit"]["download_bytes"] == full.shared_bytes - fb and by["per_visit"]["frozen_bytes_once"] == fb
    assert by["per_visit"]["upload_bytes"] == full.shared_bytes - fb + 8
    n_vis = by["fleet"]["messages"] // 2
    assert by["fleet"]["bytes_down"] == n_vis * (full.shared_bytes - fb)
    assert by["fleet"]["frozen_download_once_bytes"] == by["fleet"]["frozen_download_once_clients"] * fb > 0
    assert out["summary"]["frozen_item_tables"]["frozen_keys"] == list(F.FROZEN_ITEM_KEYS)
    rows = _json(tmp_path / "f" / "RESULT_ROWS.json")["rows"]
    assert all(r["method"] == "FA_WARM_FROZEN" and r["fl_frozen_item_tables"] == list(F.FROZEN_ITEM_KEYS) for r in rows)
    assert run_tiny(tmp_path, spec, "p", world=warm_world(pre), interrupt_at=23)["status"] == "INTERRUPTED"
    res = run_tiny(tmp_path, spec, "p", world=warm_world(pre), resume=True)
    assert res["summary"]["server_digest"] == out["summary"]["server_digest"]
    warm = run_tiny(tmp_path, tiny_spec("FA_WARM", init_from=pre["ident"]), "w", world=warm_world(pre))
    assert warm["summary"]["server_digest"] != out["summary"]["server_digest"]


# ------------------------------------------------------------------------------------------------ select-warm / MAIN
@pytest.fixture(scope="module")
def warm_sel(s4, pre, tmp_path_factory):
    root = tmp_path_factory.mktemp("warmsel")
    s4_file = root / "S4.json"
    L.write_selection_stage("S4", [s4["inc"]] + list(s4["runs"].values()), s4_file, s4["s3_file"])
    w4 = F.tuning_config(**{k: _json(s4_file)["winner"][k] for k in F.TUNING_KEYS})
    kw = F.tuning_kwargs(w4)
    dirs = {}
    for arm in F.WARM_ARMS:                                  # FL-15 (06M_B B2): the registered 2-D grid, 4 runs each
        for f, sf in F.warm_grid(arm, w4["optimizer"]):
            spec = s4_spec(arm, role="CALIBRATION", init_from=pre["ident"], warm_lr_factor=None if f == 1.0 else f,
                           warm_server_lr_factor=None if sf == 1.0 else sf, **kw)
            name = f"{arm}_{f}_{sf}"
            out = run_tiny(root, spec, name, world=warm_world(pre, eval_id=CALIB_EVAL))
            assert out["status"] == "DONE_MAIN", name
            dirs[(arm, f, sf)] = root / name
    return {"root": root, "s4_file": s4_file, "dirs": dirs, "kw": kw, "w4": w4}


def warm_runs(warm_sel, arm):
    """The 4 run dirs of the 2-D screen of `arm` in warm_grid order ((1, 1) first)."""
    return [warm_sel["dirs"][(arm, f, sf)] for f, sf in F.warm_grid(arm, warm_sel["w4"]["optimizer"])]


def test_select_warm_files_and_refusals(warm_sel, pre, tmp_path):
    d = warm_sel["dirs"]
    for stage, arm in (("S7", "FA_WARM"), ("S8", "FA_WARM_FROZEN")):
        out = tmp_path / f"W_{stage}.json"
        said = []
        argv = ["select-warm", "--stage", stage, "--runs", *map(str, warm_runs(warm_sel, arm)),
                "--tuning", str(warm_sel["s4_file"]), "--out", str(out)]
        assert L.main(argv, say=said.append) == 0
        obj = F.load_warm_selection(out)
        cells = F.warm_grid(arm, warm_sel["w4"]["optimizer"])
        assert obj["format"] == F.WARM_SELECTION_FORMAT_2D and obj["arm"] == arm and obj["init_from"] == pre["ident"]
        assert sorted(obj["screened"]) == sorted(F.warm_cell_label(*c) for c in cells)
        rec = json.loads(said[0])
        assert (rec["winner_factor"], rec["winner_server_lr_factor"]) == (obj["winner_factor"],
                                                                         obj["winner_server_lr_factor"])
        assert obj["tuning_selection"]["sha256"] == hashlib.sha256(warm_sel["s4_file"].read_bytes()).hexdigest()
        assert L.main(argv, say=said.append) == 2
    assert L.apply_rule_warm("FA_WARM", {1.0: 0.1, 0.25: 0.1004}) == (1.0, [1.0, 0.25])       # (legacy 1-D rule)
    assert L.apply_rule_warm("FA_WARM_FROZEN", {1.0: 0.1, 4.0: 0.2})[0] == 4.0
    SW = L.selection_warm
    s7, s8 = warm_runs(warm_sel, "FA_WARM"), warm_runs(warm_sel, "FA_WARM_FROZEN")
    assert_raises(L.SelectionRefused, SW, "S7", s7[:2], warm_sel["s4_file"])                   # the 06K 2-run screen
    assert_raises(L.SelectionRefused, SW, "S7", [d[("FA_WARM", 1.0, 1.0)], d[("FA_WARM", 0.25, 1.0)]],
                  warm_sel["s4_file"])
    assert_raises(L.SelectionRefused, SW, "S7", s7[:3] + [s8[3]], warm_sel["s4_file"])        # another arm's run
    assert_raises(L.SelectionRefused, SW, "S8", s7, warm_sel["s4_file"])
    assert_raises(L.SelectionRefused, SW, "S7", s7[:3] + [s7[0]], warm_sel["s4_file"])        # a cell twice
    assert_raises(L.SelectionRefused, L.apply_rule_warm, "FA_WARM", {1.0: 0.1, 4.0: 0.2})


def test_main_warm_binding(warm_sel, pre, tmp_path, monkeypatch):
    reg = tmp_path / "contracts_fl" / "FL_WARM_SELECTION_S7.json"
    reg.parent.mkdir()
    L.write_selection_warm("S7", warm_runs(warm_sel, "FA_WARM"), reg, warm_sel["s4_file"])
    monkeypatch.setitem(F.WARM_SELECTION_PATH, "S7", reg)
    main_bind = {"sha256": hashlib.sha256(warm_sel["s4_file"].read_bytes()).hexdigest(), "path": str(warm_sel["s4_file"])}
    args = F.parse_args(["--arm", "FA_WARM", "--seed", "2027", "--gpu", "0", "--run-dir", str(tmp_path / "r"),
                         "--history-policy", "USER_HISTORY_L50", "--init-from", str(pre["path"]),
                         "--warm-selection", str(reg), "--fl-tuning", str(warm_sel["s4_file"])])
    F.fl05_preflight(args)
    b = F.bind_warm_selection(args, pre["ident"], main_bind)
    assert b["winner_factor"] == _json(reg)["winner_factor"] and b["sha256"] == hashlib.sha256(reg.read_bytes()).hexdigest()
    assert b["winner_server_lr_factor"] == _json(reg)["winner_server_lr_factor"]
    assert_raises(F.LaunchRefused, F.bind_warm_selection, args, pre["ident"], {"sha256": "0" * 64})
    assert_raises(F.LaunchRefused, F.bind_warm_selection, args, dict(pre["ident"], checkpoint_sha256="1" * 64), main_bind)
    assert_raises(F.LaunchRefused, F.bind_warm_selection, args, pre["ident"], None)
    copy = tmp_path / "copy.json"
    copy.write_bytes(reg.read_bytes())
    args.warm_selection = copy
    assert_raises(F.LaunchRefused, F.bind_warm_selection, args, pre["ident"], main_bind)     # a copy never qualifies
    args.warm_selection, args.arm = reg, "FA_WARM_FROZEN"
    assert_raises(F.LaunchRefused, F.bind_warm_selection, args, pre["ident"], main_bind)     # S8 file not registered


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a guard without the MAIN checks (deny-list emptied, allow-list pointed at MAIN) accepts a C_BIG-like checkpoint")
def test_nc_main_trained_checkpoint_refused(tmp_path, monkeypatch):
    p, _ = make_pre(tmp_path / "cbig", manifest="MAIN_TRAIN_5_100_131072")
    monkeypatch.setattr(F, "MAIN_TRAIN_MANIFESTS", ())
    monkeypatch.setattr(F, "MAIN_RUN_MARKERS", ())
    monkeypatch.setitem(F.PRETRAIN, "train_manifest", "MAIN_TRAIN_5_100_131072")
    assert_raises(F.LaunchRefused, F.load_pretrain_init, p, model_variant="SASREC_D64_B2")


@nc("a warm run that skips theta_pre (starts at theta0) does not start from the checkpoint's state")
def test_nc_warm_without_theta_pre(pre):
    srv = make_tiny_adapter(seed=1, tied=False)                                   # apply_init_state skipped
    assert state_digest(srv.broadcast_state()) == pre["ident"]["state_digest"]


@nc("S4 passes = 1 with the P = 2 round count leaves the exposure budget at 3.0 EFE, not 6.0")
def test_nc_s4_without_round_recomputation():
    n, p = 131072, 1
    r = F.KNOBS["rounds"]                                                         # not recomputed
    assert r * 64 * 0.9 * p >= 6.0 * n

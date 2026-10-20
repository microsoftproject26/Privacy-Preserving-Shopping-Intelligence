"""P2-07 + P2-13 - v4int.reeval_p2, the Phase-2 shim around the UNEDITED v4int.reeval (CPU; the GPU / pin / data-root
evaluation stays server-only, like every reeval run). Real fixtures where they exist (the PRE / C_FULL RUN_START server
records of FIX-REEVAL, the real registry extensions, the real launch_fl P2 spec builder, the real fl.peft payload, the
real p2_select anchor reader, the real TEST_EXPOSURE_REGISTER tool); synthetic run dirs otherwise.

  * reeval.py is byte-identical (pinned sha) before / after, and every patched reeval attribute is restored after each
    P2 call, refusals included;
  * P2-07: P2_ANCHOR_PRE_CALIB on CALIBRATION passes every PRE source check (real extension binding, init record) and
    plans the CALIBRATION role key; only theta_pre (the sha pin) is admitted on CALIBRATION, only the anchor label;
    the rows written through the shim (real T3 rows on the INT-01 CPU env) are accepted by p2_select.read_anchor;
  * P2-13: dry runs per source kind - P2_WSTAR FL (frozen recipe; WC5 PF + store; merged LoRA when P2-04 is wired),
    P2_C1 FULL / FROZEN central, P2_C2 central source (real FULL extension) - with their refusals; a merged-LoRA payload
    loads into an unwrapped harness adapter through reeval.load_fl;
  * TEST: refused while the DATA-09 gate is closed or any part of the P2 lock binding is missing (P2_METHODS_LOCK,
    P2_SCRIPT_FREEZE, the two LOCKED_EVALUATION entries, the arm / checkpoint binding, declared not run); admitted only
    with all of them; TEST2 and Phase-1 labels (S5) refused through the P2 path;
  * the admitted / pending labels are exactly the P2 labels of P2_CONTRASTS_v3.
Negative controls (test_nc_*, strict xfail on AssertionError).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest
import torch

import v4int.central_ft as CF
import v4int.launch_fl as LF
from intkit import nc
from test_ana05_reeval import rehash
from test_gate_b_reeval import argv as src_argv, real_start, source_run
from v4int import reeval as R
from v4int import reeval_p2 as RP
import ppsi_paths

V4 = ppsi_paths.PROJECT_ROOT.resolve()       # the ppsi_v4 workspace (contracts, company records)
VAR, REC = "SASREC_D64_B2", "SMALL_SASREC_D64_B2"
PB, CB = "PRACTICAL_BEST", "CONTROLLED_BUDGET"
MAIN_TRAIN, MAIN_VAL = "MAIN_TRAIN_5_100_131072", "MAIN_VALIDATION_5_100_131072"
REEVAL_SHA_AT_TASK_START = "833f0c981b0622e717d3309fe9a782d46905f67da45c2c74fa8c063fc7ec09f1"
PATCHED = ("ROLES", "SOURCE_ROLES", "CENTRAL_SOURCES", "source_fl", "source_central", "load_central",
           "check_warm_line", "check_fl_payload", "label_rows", "write_outputs")
ORIG = {k: getattr(R, k) for k in PATCHED}
ORIG_TABLE = copy.deepcopy(R.CENTRAL_SOURCES)
INIT = {"kind": "PRETRAIN_FOR_FL", "run_id": "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026", "run_label": "PRETRAIN_FOR_FL",
        "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN", "fl_init_kind": "endpoint_6.0",
        "checkpoint_sha256": RP.PRE_CHECKPOINT_SHA256, "checkpoint_bytes": 1234, "state_digest": RP.PRE_STATE_DIGEST,
        "model_variant": VAR, "family": "SASREC", "seed": 2026}
INCUMBENT = {"server_opt": "fedadam", "server_lr": 0.001, "client_lr": 0.1}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def restored() -> bool:
    return (all(getattr(R, k) is v for k, v in ORIG.items()) and R.CENTRAL_SOURCES == ORIG_TABLE
            and not getattr(R, "__p2_context__", False))


def refused(fn, msg) -> str:
    """The P2 call must refuse with `msg` (an AssertionError otherwise, so a negative control fails at an assert)."""
    got = None
    try:
        fn()
    except R.ReevalRefused as e:
        got = str(e)
    assert got is not None and msg in got, (msg, got)
    assert restored(), "a reeval attribute was not restored after the refusal"
    return got


def args_for(cp, tmp, *, kind="central", label="P2_WSTAR", role="MAIN", seed=2026, rule=PB, sha=None, dry=True,
             **kw):
    a = {"--ckpt": str(cp), "--ckpt-kind": kind, "--expect-sha256": sha or (_sha(cp) if Path(cp).is_file() else "0" * 64),
         "--variant": VAR, "--recipe-id": REC, "--seed": str(seed), "--role": role, "--arm-label": label,
         "--selection-rule": rule, "--checkpoint-id": f"{label}:ckpt", "--gpu": "0", "--out-dir": str(tmp / "out"),
         "--history-policy": "USER_HISTORY_L50"}
    a.update({f"--{k.replace('_', '-')}": str(v) for k, v in kw.items()})
    return R.parse_args([x for kv in a.items() for x in kv] + (["--dry-run"] if dry else []))


# ------------------------------------------------------------------------------------------------ pin / restoration
def test_reeval_is_pinned_and_unchanged():
    assert RP.REEVAL_SHA256 == REEVAL_SHA_AT_TASK_START == _sha(ppsi_paths.locate("code/v4int/reeval.py"))
    assert RP.check_reeval_unchanged() == REEVAL_SHA_AT_TASK_START and restored()


def test_a_changed_reeval_is_refused(monkeypatch):
    monkeypatch.setattr(RP, "REEVAL_SHA256", "0" * 64)
    refused(lambda: RP.run(args_for(V4 / "missing.pt", Path("."))), "refuses to wrap it")


# ------------------------------------------------------------------------------------------------ P2-07 the anchor
def anchor_args(cp, t, **kw):
    kw = dict(dict(role="CALIBRATION", arm_label=RP.ANCHOR_LABEL, selection_rule=CB), **kw)
    return src_argv(cp, t, "PRE", **kw)


def check_anchor_dry_run(tmp_path, monkeypatch):
    t = tmp_path / "anchor"
    cp = source_run(t, "PRE")
    monkeypatch.setattr(RP, "PRE_CHECKPOINT_SHA256", _sha(cp))         # the synthetic stand-in for theta_pre's bytes
    try:
        out = RP.run(anchor_args(cp, t))
    except R.ReevalRefused as e:
        raise AssertionError(f"the PRE anchor on CALIBRATION refused: {e}")
    assert restored()
    cs = out["central_source"]
    assert out["dry_run"] and out["role"] == "CALIBRATION" and out["arm_label"] == RP.ANCHOR_LABEL, out
    assert cs["source"] == "PRE" and cs["arm_label"] == RP.ANCHOR_LABEL and cs["checks"]["all_ok"], cs
    assert cs["binding"]["view_checks"]["all_ok"] and out["planned_key"].startswith("ROLE_KEY"), out
    assert out["pre_init_record"]["fl_init_sha256"] == _sha(cp) and out["checkpoint_file"]["file"] == "endpoint_6.0.pt"
    lab = out["row_labels"]
    assert lab["central_source"] == "PRE" and lab["p2_anchor"] == "PRE_CALIB" and lab["study"] == "PREG2", lab
    p2 = out["p2_reeval"]
    assert p2["arm_label"] == RP.ANCHOR_LABEL and p2["checks"]["anchor"]["central_source"] == "PRE"
    assert p2["preg2_sha256"] == RP.PREG2_SHA256 and p2["reeval_sha256"] == REEVAL_SHA_AT_TASK_START
    assert p2["role_reading"].startswith("SELECTION_ANCHOR") and not (t / "out").exists()
    return t, cp


def test_anchor_dry_run_pre_on_calibration(tmp_path, monkeypatch):
    check_anchor_dry_run(tmp_path, monkeypatch)


def test_anchor_admits_theta_pre_only(tmp_path, monkeypatch):
    t = tmp_path / "a"
    cp = source_run(t, "PRE")
    refused(lambda: RP.run(anchor_args(cp, t)), "theta_pre only")         # the real pin: a stand-in is refused
    refused(lambda: RP.run(anchor_args(cp, t, expect_sha256=RP.PRE_CHECKPOINT_SHA256)), "--expect-sha256")
    monkeypatch.setattr(RP, "PRE_CHECKPOINT_SHA256", _sha(cp))
    for kw, msg in ((dict(selection_rule=PB), "--selection-rule"), (dict(seed=2027), "--seed"),
                    (dict(arm_label="PRE"), "S5"), (dict(arm_label="C_SMALL"), "admits only"),
                    (dict(arm_label="P2_WSTAR"), "admits only"), (dict(role="MAIN"), "that label only on"),
                    (dict(registry_ext=str(t / "x.json")), "--registry-ext")):
        refused(lambda: RP.run(anchor_args(cp, t, **kw)), msg)
    t2 = tmp_path / "init"                                            # the sha must be the run's own fl_init record
    cp2 = source_run(t2, "PRE", init_over={"sha256": "a" * 64})
    monkeypatch.setattr(RP, "PRE_CHECKPOINT_SHA256", _sha(cp2))
    refused(lambda: RP.run(anchor_args(cp2, t2)), "PRETRAIN_FOR_FL_INIT")
    t3 = tmp_path / "cfull"                                           # another source's checkpoint on CALIBRATION
    cp3 = source_run(t3, "C_FULL", ckpt="endpoint_6.0.pt")
    monkeypatch.setattr(RP, "PRE_CHECKPOINT_SHA256", _sha(cp3))
    refused(lambda: RP.run(src_argv(cp3, t3, "PRE", role="CALIBRATION", arm_label=RP.ANCHOR_LABEL,
                                    selection_rule=CB)), "differs from requested")
    t4 = tmp_path / "main_pair"                                       # a MAIN-pair central run claimed by the label
    cp4 = t4 / "run" / "sealed" / "checkpoints" / "endpoint_6.0.pt"
    cp4.parent.mkdir(parents=True)
    cp4.write_bytes(b"main-pair")
    (t4 / "run" / "RUN_START.json").write_text(json.dumps(
        {"run_id": "C_SMALL_s2026", "config": {"family": "SASREC", "model_variant": VAR, "recipe_id": REC, "seed": 2026},
         "key_base": {"split": "VALIDATION", "run_label": "PRODUCTION", "train_manifest_id": MAIN_TRAIN},
         "train_provider": {"manifest_id": MAIN_TRAIN}, "eval_provider": {"manifest_id": MAIN_VAL}}), encoding="utf-8")
    monkeypatch.setattr(RP, "PRE_CHECKPOINT_SHA256", _sha(cp4))
    refused(lambda: RP.run(anchor_args(cp4, t4)), "not the registered PRE central source")
    with pytest.raises(R.ReevalRefused, match="re-evaluation role"):       # the unedited reeval keeps refusing CALIB
        R.run(src_argv(cp, t, "PRE", role="CALIBRATION", selection_rule=CB))


def check_anchor_pin_refuses(tmp_path):
    t = tmp_path / "pin"
    cp = source_run(t, "PRE")
    refused(lambda: RP.run(anchor_args(cp, t)), "theta_pre only")


def test_anchor_pin_refuses_another_checkpoint(tmp_path):
    check_anchor_pin_refuses(tmp_path)


@nc("CALIBRATION admits any PRE-shaped checkpoint (no theta_pre sha pin)")
def test_nc_anchor_without_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(RP, "anchor_plan", lambda args: {"label": RP.ANCHOR_LABEL, "kind": "ANCHOR",
                                                         "role": "CALIBRATION", "test": False, "seed": 2026, "rule": CB})
    check_anchor_pin_refuses(tmp_path)


# the anchor dir written through the shim on REAL T3 rows (INT-01 CPU env) -> p2_select.read_anchor
CALIB_OVER = {"manifest_role": "CALIBRATION", "train_manifest_id": "CALIB_TRAIN_5_100_32768",
              "eval_manifest_id": RP.ANCHOR_EVAL_MANIFEST}


@pytest.fixture(scope="module")
def calib_eval(providers):
    from test_int06_reeval import _harness, _kb, _mk
    _, ep = providers
    bm = _mk()
    bm.module.eval()
    kb = dict(_kb(), train_manifest_id=CALIB_OVER["train_manifest_id"], eval_manifest_id=RP.ANCHOR_EVAL_MANIFEST)
    return R.evaluate_role(_harness(bm.adapter, ep), ep, kb, dict(CALIB_OVER))


def write_anchor(out: Path, calib_eval, *, through_shim=True, label=RP.ANCHOR_LABEL) -> Path:
    _v, rows, ranks, res = calib_eval
    a = R.parse_args(["--ckpt", "endpoint_6.0.pt", "--ckpt-kind", "central", "--expect-sha256", RP.PRE_CHECKPOINT_SHA256,
                      "--variant", VAR, "--recipe-id", REC, "--seed", "2026", "--role", "CALIBRATION",
                      "--arm-label", label, "--selection-rule", CB, "--checkpoint-id", "PRE:endpoint_6.0",
                      "--gpu", "0", "--out-dir", str(out), "--history-policy", "USER_HISTORY_L50"])
    kw = dict(checkpoint_sha256=RP.PRE_CHECKPOINT_SHA256, source_run_id=INIT["run_id"], efe=6.0, extra={})
    rec = {"format": R.RECORD_FORMAT, "arm_label": label, "role": "CALIBRATION"}
    wkw = dict(tag="CALIBRATION", run_id=f"REEVAL_{label}_CALIBRATION_s2026_{CB}", role="CALIBRATION", result=res,
               ranks=ranks)
    if through_shim:
        plan = {"label": label, "kind": "ANCHOR", "role": "CALIBRATION", "test": False, "seed": 2026, "rule": CB,
                "test_lock": None}
        with RP.p2_context({"plan": plan, "checks": {}, "ckpt_name": "endpoint_6.0.pt", "spec": None}):
            R.write_outputs(out, rows=R.label_rows(rows, a, **kw), record=rec, **wkw)
        assert restored()
    else:
        R.write_outputs(out, rows=R.label_rows(rows, a, **kw), record=rec, **wkw)
    return out


def check_anchor_dir(out: Path) -> dict:
    from v4int import p2_select as PS
    try:
        got = PS.read_anchor(out)
    except PS.P2SelectionRefused as e:
        raise AssertionError(f"p2_select.read_anchor refused the anchor dir: {e}")
    assert got["eval_manifest_id"] == PS.CALIB_EVAL_MANIFEST == RP.ANCHOR_EVAL_MANIFEST
    assert got["result_rows_sha256"] == _sha(out / "RESULT_ROWS.json")
    return got


def test_anchor_rows_are_read_by_p2_select(tmp_path, calib_eval):
    out = write_anchor(tmp_path / "anchor_out", calib_eval)
    got = check_anchor_dir(out)
    assert got["value"] == got["value"]                                # finite (read_anchor refuses NaN); never printed
    rows = json.loads((out / "RESULT_ROWS.json").read_text(encoding="utf-8"))
    rec = json.loads((out / "REEVAL_RECORD.json").read_text(encoding="utf-8"))
    assert rows["role"] == "CALIBRATION" and all(r["method"] == RP.ANCHOR_LABEL for r in rows["rows"])
    assert all(r["p2_anchor"] == "PRE_CALIB" and r["study"] == "PREG2" for r in rows["rows"])
    assert rec["p2_reeval"]["arm_label"] == RP.ANCHOR_LABEL and rec["result_rows_sha256"] == got["result_rows_sha256"]


@nc("the anchor written without the P2 relabel (a Phase-1 PRE row) is read as PRE_CALIB")
def test_nc_anchor_without_p2_label(tmp_path, calib_eval):
    check_anchor_dir(write_anchor(tmp_path / "pre_out", calib_eval, through_shim=False, label="PRE"))


# ------------------------------------------------------------------------------------------------ P2-13 FL (P2_W)
def p2_spec(recipe=None, *, seed=2026, init=None):
    return LF.p2_arm_spec("P2_W", recipe=dict(recipe or INCUMBENT), seed=seed, peak_lr=1e-3, role="MAIN",
                          init_from=dict(init or INIT))


def selection_file(tmp, monkeypatch) -> Path:
    p = tmp / "contracts" / "fl" / "P2_WARM_SELECTION_FINAL.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"stub": "the registered P2_WARM_SELECTION_FINAL"}', encoding="utf-8")
    monkeypatch.setattr(RP, "SELECTION_FINAL", p)
    return p


def _theta():
    from fl.synthetic import make_tiny_adapter
    return {k: v.clone() for k, v in make_tiny_adapter(seed=1).broadcast_state().items()}


def fl_run(tmp, sel, *, spec=None, seed=2026, ckpt="best.pt", efe=None, rid=None, arm="P2_W", label="PRODUCTION",
           role="MAIN", sel_over=None, launch_over=None, payload_over=None, legs=1, theta=None):
    from fl.checkpoint import atomic_save
    spec = spec or p2_spec(seed=seed)
    rid = rid or f"P2_WSTAR_SMALL_s{seed}"
    run = tmp / "fl"
    run.mkdir(parents=True, exist_ok=True)
    srec = {"source": "p2_selection", "path": str(sel), "sha256": _sha(sel), "self_sha256": "f" * 64,
            "format": LF.P2_SELECTION_FORMAT, "stage": "FINAL", "label": spec["p2_label"], "recipe": spec["p2_recipe"],
            "init_from_checked": True, "flags": [], "authority": LF.P2_AUTHORITY}
    srec.update(sel_over or {})
    launch = {"arm": arm, "run_id": rid, "arm_spec": spec, "arm_spec_sha256": LF.spec_digest(spec),
              "fl_config_digest": "C", "label": label, "fl05": {"role": role, "lr_mult": 1.0},
              "small_recipe": {"model_variant": VAR, "recipe_id": REC, "recipe_sha256": "R"}, "p2_selection": srec}
    launch.update(launch_over or {})
    for k in range(legs):
        (run / f"LAUNCH_{k}.json").write_text(json.dumps(launch), encoding="utf-8")
    theta = theta if theta is not None else _theta()
    payload = {"kind": "weights", "run_id": rid, "config_digest": "C", "round": 9,
               "efe": efe if efe is not None else (6.0 if ckpt == "endpoint_6.0.pt" else 3.0), "phase": "MAIN",
               "theta": theta}
    payload.update(payload_over or {})
    cp = run / "ckpt" / rid / ckpt
    atomic_save(payload, cp)
    return cp, rid, theta


def pf_store(tmp, rid, theta) -> Path:
    from fl.checkpoint import atomic_save
    from fl.numerics import state_digest
    from fl.personal import PersonalStore
    p = tmp / "fl" / "PF_STORE_best.pt"
    atomic_save({"kind": "pf_store", "run_id": rid, "config_digest": "C", "store": PersonalStore(8).state_dict(),
                 "theta_digest": state_digest(theta)}, p)
    return p


def check_fl_dry_runs(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    for seed, ckpt, rule in ((2026, "best.pt", PB), (2026, "endpoint_6.0.pt", CB), (2027, "best.pt", PB)):
        t = tmp_path / f"fl_{seed}_{rule}"
        cp, rid, _ = fl_run(t, sel, seed=seed, ckpt=ckpt)
        try:
            out = RP.run(args_for(cp, t, kind="fl", seed=seed, rule=rule))
        except R.ReevalRefused as e:
            raise AssertionError(f"P2_WSTAR s{seed} {rule} refused: {e}")
        assert restored()
        p2 = out["p2_reeval"]
        fl = p2["checks"]["fl"]
        assert out["dry_run"] and out["source_run_id"] == rid and "warm_line" not in out, out
        assert fl["launch_arm"] == "P2_W" and fl["check_p2_spec"] and fl["p2_selection"]["sha256"] == _sha(sel)
        assert fl["theta_pre"]["checkpoint_sha256"] == RP.PRE_CHECKPOINT_SHA256 and fl["method"] == "FA"
        assert p2["source_kind"] == "FL" and p2["checks"]["payload"]["merged_lora"] is None
        assert out["fl_round"] == out["fl_efe"] == RP.SEALED                # S8: no PRACTICAL_BEST mark printed
        assert p2["role_reading"].startswith("POST_VALIDATION_EXPLORATORY")
    t = tmp_path / "wc5"                                                  # WC5 personal head: a PF run + its store
    spec = p2_spec(dict(INCUMBENT, personal_head=True))
    cp, rid, theta = fl_run(t, sel, spec=spec)
    refused(lambda: RP.run(args_for(cp, t, kind="fl")), "needs --ckpt-kind pf")
    out = RP.run(args_for(cp, t, kind="pf", pf_store=pf_store(t, rid, theta)))
    assert out["p2_reeval"]["source_kind"] == "FL_PF" and out["p2_reeval"]["checks"]["fl"]["method"] == "PF"
    assert "pf_store_sha256" in out and restored()


def test_fl_p2_wstar_dry_runs(tmp_path, monkeypatch):
    check_fl_dry_runs(tmp_path, monkeypatch)


def fl_refusal_cases(tmp_path, monkeypatch) -> list:
    sel = selection_file(tmp_path, monkeypatch)
    other_init = dict(INIT, checkpoint_sha256="b" * 64)
    edited = p2_spec()
    edited["cfg"]["solver"]["lr"] = 0.01                                   # a hand edit, re-digested by the launcher
    tampered = p2_spec()
    tampered["max_rounds"] = 10
    cases = [
        ("phase1_arm", dict(arm="FA_WARM_FROZEN"), {}, "not the Phase-2 arm P2_W"),
        ("calib_screen", dict(role="CALIBRATION"), {}, "only MAIN P2_W runs"),
        ("smoke", dict(label="SMOKE_NOT_EVIDENCE"), {}, "smoke"),
        ("recipe_file_id", dict(rid="P2_W_FEDADAM_eta0.001_c0.1_SMALL_s2026"), {}, "not the bound MAIN run"),
        ("hand_edit", dict(spec=edited), {}, "check_p2_spec"),
        ("digest", dict(launch_over={"arm_spec": tampered}), {}, "does not hash"),
        ("theta_pre", dict(spec=p2_spec(init=other_init)), {}, "not the registered theta_pre"),
        ("sel_missing", dict(launch_over={"p2_selection": None}), {}, "no bound p2_selection"),
        ("sel_recipe", dict(sel_over={"recipe": dict(p2_spec()["p2_recipe"], client_lr=0.03)}), {}, "recipe / label"),
        ("sel_sha", dict(sel_over={"sha256": "c" * 64}), {}, "not the file the run was bound to"),
        ("sel_path", dict(sel_over={"path": str(tmp_path / "copy" / "P2_WARM_SELECTION_FINAL.json")}), {},
         "not the registered"),
        ("mark0", dict(efe=0.0), {}, "mark 0 is never a candidate"),
        ("endpoint_early", dict(ckpt="endpoint_6.0.pt", efe=5.5), dict(rule=CB), "scored mark >= 6.0"),
        ("lora_on_dense", dict(payload_over={"p2_lora": {"format": "PPSI_V4_P2_LORA_1"}}), {}, "no LoRA rank"),
        ("seed", dict(seed=2027), dict(seed=2026), "seed differs"),
        ("rule_file", {}, dict(rule=CB), "--selection-rule CONTROLLED_BUDGET scores endpoint_6.0.pt"),
        ("kind_central", {}, dict(kind="central"), "--ckpt-kind fl/pf")]
    out = []
    for tag, run_kw, arg_kw, msg in cases:
        t = tmp_path / tag
        cp, _, _ = fl_run(t, sel, **run_kw)
        out.append((tag, cp, t, dict(dict(kind="fl"), **arg_kw), msg))
    return out


def test_fl_refusals(tmp_path, monkeypatch):
    for tag, cp, t, kw, msg in fl_refusal_cases(tmp_path, monkeypatch):
        refused(lambda: RP.run(args_for(cp, t, **kw)), msg)
    t = tmp_path / "legs"                                                 # every LAUNCH leg agrees (p2_selection too)
    cp, _, _ = fl_run(t, selection_file(tmp_path, monkeypatch), legs=2)
    other = json.loads((t / "fl" / "LAUNCH_1.json").read_text(encoding="utf-8"))
    other["p2_selection"] = dict(other["p2_selection"], self_sha256="0" * 64)
    (t / "fl" / "LAUNCH_1.json").write_text(json.dumps(other), encoding="utf-8")
    refused(lambda: RP.run(args_for(cp, t, kind="fl")), "p2_selection differs")
    t = tmp_path / "absent"                                               # the registered selection must be present
    cp, _, _ = fl_run(t, selection_file(tmp_path, monkeypatch))
    monkeypatch.setattr(RP, "SELECTION_FINAL", tmp_path / "nowhere" / "contracts" / "fl" / "P2_WARM_SELECTION_FINAL.json")
    refused(lambda: RP.run(args_for(cp, t, kind="fl")), "not on this host")


def test_unedited_reeval_refuses_a_p2_w_checkpoint(tmp_path, monkeypatch):
    """The Phase-1 path cannot score a P2_W run under a Phase-1 label (check_warm_line: no block arm for P2_W)."""
    t = tmp_path / "p1"
    cp, _, _ = fl_run(t, selection_file(tmp_path, monkeypatch))
    for label in ("FA_WARM_FROZEN", "FA", "P2_WSTAR"):
        with pytest.raises(R.ReevalRefused, match="block arm"):
            R.run(args_for(cp, t, kind="fl", label=label))


def check_phase1_fl_run_refused(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    t = tmp_path / "phase1"
    spec = {"cfg": {"seed": 2026}, "arm": "FA_WARM_FROZEN", "init_from": INIT, "frozen_item_tables": list(LF.FROZEN_ITEM_KEYS)}
    cp, _, _ = fl_run(t, sel, rid="P2_WSTAR_SMALL_s2026", arm="FA_WARM_FROZEN",
                      launch_over={"arm_spec": spec, "arm_spec_sha256": "S"})
    refused(lambda: RP.run(args_for(cp, t, kind="fl")), "not the Phase-2 arm")


def test_phase1_fl_run_refused_as_p2_wstar(tmp_path, monkeypatch):
    check_phase1_fl_run_refused(tmp_path, monkeypatch)


@nc("the P2 FL source checks replaced by reeval's Phase-1 source_fl (a Phase-1 warm run scored as P2_WSTAR)")
def test_nc_phase1_source_fl(tmp_path, monkeypatch):
    def phase1_source(ckpt, args, ctx):
        src = ORIG["source_fl"](ckpt, args)
        ctx["spec"] = src["launch"].get("arm_spec") or {}
        return src
    monkeypatch.setattr(RP, "p2_source_fl", phase1_source)
    check_phase1_fl_run_refused(tmp_path, monkeypatch)


# ------------------------------------------------------------------------------------------------ merged LoRA (WC1-WC4)
KC = 64


def _built(seed=2026):
    from v4models import build, load_schema
    return build("SASREC", load_schema(), seed, K=KC, variant=VAR)


def lora_payload(r=4):
    """A real fl.peft merged-LoRA weights payload (install_peft's lora_weights_payload) of a K=64 SASREC_D64_B2 whose
    'theta_pre' is the built model itself; returns (payload, theta_pre digest)."""
    import fl.peft as P
    from fl.numerics import state_digest
    bm = _built()
    pre = state_digest(bm.adapter.broadcast_state())
    P.apply_lora(bm.adapter.module, r=r, alpha=float(r), seed=2026)
    P.freeze_all_but(bm.adapter, P.lora_keys(bm.adapter.module))
    P.lora_init_sha256(bm.adapter, pre)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for k, p in bm.adapter.module.named_parameters():
            if k.endswith(P.LORA_A) or k.endswith(P.LORA_B):
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    base = {"kind": "weights", "run_id": "P2_WSTAR_SMALL_s2026", "config_digest": "C", "round": 9, "efe": 3.0,
            "phase": "MAIN"}
    return P.lora_weights_payload(base, bm.adapter), pre


def lora_spec(r=4, seed=2026, init=None):
    spec = p2_spec(seed=seed, init=init)
    spec["p2_recipe"] = dict(spec["p2_recipe"], lora_rank=r, lora_lr_factor=10)
    return spec


def test_merged_lora_payload_checks_and_reeval_load(monkeypatch):
    from fl.numerics import state_digest
    payload, pre = lora_payload(4)
    fresh = _built(seed=5)                                     # an unwrapped harness adapter (reeval.build_fl_harness)
    R.load_fl(fresh.adapter, payload)                          # keys / shapes / dtypes of the base graph; digest kept
    assert state_digest(fresh.adapter.broadcast_state()) == payload["p2_lora"]["merged_theta_digest"]
    refused(lambda: RP.check_lora_payload(payload, lora_spec(4)), "lineage theta_pre digest")   # the real pin
    monkeypatch.setattr(RP, "PRE_STATE_DIGEST", pre)
    rec = RP.check_lora_payload(payload, lora_spec(4))
    assert rec["r"] == 4 and rec["alpha"] == 4.0 and rec["merged_theta_digest"] == state_digest(payload["theta"])
    refused(lambda: RP.check_lora_payload(payload, lora_spec(16)), "LoRA meta")
    refused(lambda: RP.check_lora_payload(dict(payload, p2_lora=None), lora_spec(4)), "merged-LoRA payload")
    bad = dict(payload, theta=dict(payload["theta"]))
    k0 = sorted(bad["theta"])[0]
    bad["theta"][k0] = bad["theta"][k0] + 1.0
    refused(lambda: RP.check_lora_payload(bad, lora_spec(4)), "merged_theta_digest")
    wrapped = dict(payload, theta=dict(payload["theta"], **{"blocks.0.qkv.lora_A": torch.zeros(4, 64)}))
    refused(lambda: RP.check_lora_payload(wrapped, lora_spec(4)), "not the merged base-architecture state")
    spec_seed = lora_spec(4)
    spec_seed["lora"] = {"r": 4, "alpha": 4.0, "targets": ["qkv", "out", "ff1", "ff2"], "seed": 7}
    refused(lambda: RP.check_lora_payload(payload, spec_seed), "LoRA meta")       # the P2-04 spec seed is compared


def test_lora_wstar_fails_closed_until_p2_04_is_wired_then_dry_runs(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    payload, pre = lora_payload(4)
    spec = lora_spec(4, init=dict(INIT, state_digest=pre))       # theta_pre = the test model (digest pinned below)
    t = tmp_path / "lora"
    cp, _, _ = fl_run(t, sel, spec=spec, theta=payload["theta"], payload_over={"p2_lora": payload["p2_lora"]})
    refused(lambda: RP.run(args_for(cp, t, kind="fl")), "LoRA cells (WC1-WC4) are not installed")   # launch_fl f4e56b80
    monkeypatch.setattr(LF, "check_p2_spec", lambda s: None)              # stand-in for the P2-04 wiring
    monkeypatch.setattr(RP, "PRE_STATE_DIGEST", pre)
    out = RP.run(args_for(cp, t, kind="fl"))
    assert out["p2_reeval"]["source_kind"] == "FL_MERGED_LORA" and restored()
    assert out["p2_reeval"]["checks"]["payload"]["merged_lora"]["r"] == 4
    t2 = tmp_path / "lora_missing"                                        # a LoRA recipe with a plain payload
    cp2, _, _ = fl_run(t2, sel, spec=spec)
    refused(lambda: RP.run(args_for(cp2, t2, kind="fl")), "merged-LoRA payload")


# ------------------------------------------------------------------------------------------------ P2-13 central C1
def c1_run(tmp, monkeypatch, *, variant="FULL", seed=2026, peak=1e-4, ckpt="best.pt", kind="AUTO", ident_over=None,
           kb_over=None, start_over=None, sel_over=None, meta_over=None, rid=None) -> Path:
    label = CF.C1_ARMS[variant]
    selp = tmp / "contracts" / "fl" / "P2_C1_SELECTION.json"
    selp.parent.mkdir(parents=True, exist_ok=True)
    selp.write_text('{"stub": "the registered P2_C1_SELECTION"}', encoding="utf-8")
    monkeypatch.setattr(RP, "SELECTION_C1", selp)
    ident = {"arm": label, "variant": variant, "role": "MAIN", "peak_lr": peak,
             "frozen_item_tables": list(CF.FROZEN_ITEM_KEYS) if variant == "FROZEN" else None,
             "init_checkpoint_sha256": RP.PRE_CHECKPOINT_SHA256, "init_checkpoint_bytes": 1234,
             "init_state_digest": RP.PRE_STATE_DIGEST, "init_record_sha256": "a" * 64, "init_run_id": INIT["run_id"],
             "continuation": CF.CONTINUATION_RULE}
    ident.update(ident_over or {})
    sel = {"path": str(selp), "sha256": _sha(selp), "self_sha256": "e" * 64, "format": CF.C1_SELECTION_FORMAT,
           "stage": CF.C1_SELECTION_STAGE, "variant": variant, "peak_lr": peak, "flags": [],
           "reverified_by": "v4int.p2_select.load_p2_selection"}
    sel.update(sel_over or {})
    start = {"run_id": rid or CF.c1_run_id(variant, "MAIN", peak, seed), "seed": seed,
             "config": {"family": "SASREC", "model_variant": VAR, "recipe_id": REC, "seed": seed},
             "config_sha256": "c" * 64, "init_sha256": "d" * 64,
             "key_base": dict({"split": "VALIDATION", "run_label": label, "train_manifest_id": MAIN_TRAIN,
                               "eval_manifest_id": MAIN_VAL,
                               "initialization_block": f"P2_THETA_PRE|SASREC|checkpoint_sha256={RP.PRE_CHECKPOINT_SHA256}"
                                                       f"|state_digest={RP.PRE_STATE_DIGEST}"}, **(kb_over or {})),
             "train_provider": {"manifest_id": MAIN_TRAIN}, "eval_provider": {"manifest_id": MAIN_VAL},
             "p2_c1": {"identity": ident, "selection": sel, "trained_numel": 1000}}
    start.update(start_over or {})
    run = tmp / "run"
    cp = run / "sealed" / "checkpoints" / ckpt
    cp.parent.mkdir(parents=True, exist_ok=True)
    (run / "RUN_START.json").write_text(json.dumps(start), encoding="utf-8")
    mark = 6.0 if ckpt == "endpoint_6.0.pt" else 3.0
    meta = dict({"model_variant": VAR, "seed": seed, "config_sha256": "c" * 64, "init_sha256": "d" * 64, "efe": mark,
                 "mark": mark, "p2_c1": dict(ident)}, **(meta_over or {}))
    k = {"best.pt": "best", "endpoint_6.0.pt": "endpoint_6.0"}.get(ckpt) if kind == "AUTO" else kind
    torch.save({"kind": k, "meta": meta, "model": {"w": torch.zeros(2)}}, cp)
    return cp


def test_c1_dry_runs(tmp_path, monkeypatch):
    for i, (variant, seed, ckpt, rule) in enumerate((("FULL", 2026, "best.pt", PB), ("FROZEN", 2026, "best.pt", PB),
                                                     ("FULL", 2027, "endpoint_6.0.pt", CB),
                                                     ("FROZEN", 2027, "endpoint_6.0.pt", CB))):
        t = tmp_path / f"c1_{i}"
        cp = c1_run(t, monkeypatch, variant=variant, seed=seed, ckpt=ckpt)
        label = CF.C1_ARMS[variant]
        try:
            out = RP.run(args_for(cp, t, label=label, seed=seed, rule=rule))
        except R.ReevalRefused as e:
            raise AssertionError(f"{label} s{seed} {rule} refused: {e}")
        assert restored()
        c1 = out["p2_reeval"]["checks"]["c1"]
        assert out["dry_run"] and out["source_run_id"] == CF.c1_run_id(variant, "MAIN", 1e-4, seed), out
        assert c1["identity"]["arm"] == label and c1["key_base_run_label"] == label and c1["checkpoint_kind"] in (
            "best", "endpoint_6.0")
        assert out["p2_reeval"]["source_kind"] == "CENTRAL_C1" and "central_source" not in out


def test_c1_refusals(tmp_path, monkeypatch):
    cases = [
        ("calib_run", dict(ident_over={"role": "CALIBRATION"}, rid=CF.c1_run_id("FULL", "CALIBRATION", 1e-4, 2026)),
         {}, "not a MAIN C1 run"),
        ("frozen_as_full", dict(variant="FROZEN"), dict(label="P2_C1_FT_CENTRAL_FULL"), "not a MAIN C1 run"),
        ("theta_pre", dict(ident_over={"init_checkpoint_sha256": "b" * 64}), {}, "init_checkpoint_sha256"),
        ("continuation", dict(ident_over={"continuation": None}), {}, "continuation"),
        ("off_grid", dict(ident_over={"peak_lr": 2e-4}), {}, "peak_lr"),
        ("run_label", dict(kb_over={"run_label": "PRODUCTION"}), {}, "run_label"),
        ("init_block", dict(kb_over={"initialization_block": "I_META_RANDOM|SASREC|seed=2026"}), {}, "P2_THETA_PRE"),
        ("providers", dict(start_over={"eval_provider": {"manifest_id": "CALIB_VALIDATION_5_100_32768"}}), {},
         "MAIN pair"),
        ("sel_peak", dict(sel_over={"peak_lr": 3e-4}), {}, "C1 selection"),
        ("sel_sha", dict(sel_over={"sha256": "0" * 64}), {}, "not the file the C1 run was bound to"),
        ("sel_absent", dict(start_over={"p2_c1": {"identity": None}}), {}, "not a MAIN C1 run"),
        ("ckpt_kind", dict(kind="endpoint_6.0"), {}, "carries kind"),
        ("meta_p2c1", dict(meta_over={"p2_c1": {"arm": "X"}}), {}, "meta p2_c1"),
        ("mark0", dict(meta_over={"mark": 0.0}), {}, "mark >= 0.5"),
        ("ep_mark", dict(ckpt="endpoint_6.0.pt", meta_over={"mark": 5.5}), dict(rule=CB), "meta mark"),
        ("seed", {}, dict(seed=2028), "seeds (2026, 2027)"),
        ("main_pair", dict(start_over={"p2_c1": None}, rid="C_SMALL_s2026"), {}, "not a MAIN C1 run")]
    for tag, run_kw, arg_kw, msg in cases:
        t = tmp_path / tag
        cp = c1_run(t, monkeypatch, **run_kw)
        kw = dict(dict(label=CF.C1_ARMS[run_kw.get("variant", "FULL")]), **arg_kw)
        refused(lambda: RP.run(args_for(cp, t, **kw)), msg)
    t = tmp_path / "pre_as_c1"                                            # a PRE checkpoint presented as a C1 arm
    cp = source_run(t, "PRE", ckpt="best.pt", kind="best")
    refused(lambda: RP.run(args_for(cp, t, label="P2_C1_FT_CENTRAL_FULL")), "not the registered PRE central source")
    t = tmp_path / "absent"
    cp = c1_run(t, monkeypatch)
    monkeypatch.setattr(RP, "SELECTION_C1", tmp_path / "none" / "contracts" / "fl" / "P2_C1_SELECTION.json")
    refused(lambda: RP.run(args_for(cp, t, label="P2_C1_FT_CENTRAL_FULL")), "not on this host")


def test_c1_checkpoint_checked_on_load_central_in_a_real_run(tmp_path, monkeypatch):
    """A real (non-dry) run checks the C1 checkpoint object on reeval.load_central (after the CUDA export), before the
    original loader touches a weight; the RUN_START checks of source_central come first."""
    cp = c1_run(tmp_path / "real", monkeypatch)
    args = args_for(cp, tmp_path / "real", label="P2_C1_FT_CENTRAL_FULL", dry=False)
    plan = RP.label_plan(args)
    ctx = {"plan": dict(plan, test_lock=None), "checks": {}, "ckpt_name": "best.pt", "spec": None}
    reached = []
    monkeypatch.setattr(R, "load_central", lambda bm, obj, start, a: reached.append(obj) or {"ok": True})
    with RP.p2_context(ctx):
        src = R.source_central(cp, args)                          # RUN_START checks only (no torch load: not a dry run)
        assert ctx["checks"]["c1"]["checkpoint_kind"] is None
        good = torch.load(cp, map_location="cpu", weights_only=False)
        bad = dict(good, meta=dict(good["meta"], p2_c1={"arm": "X"}))
        with pytest.raises(R.ReevalRefused, match="meta p2_c1"):
            R.load_central(None, bad, src["start"], args)
        assert not reached and getattr(R, "__p2_context__", False)
        assert R.load_central(None, good, src["start"], args) == {"ok": True} and reached
        assert ctx["checks"]["c1"]["checkpoint_kind"] == "best"


# ------------------------------------------------------------------------------------------------ P2-13 central C2
def c2_start(seed=2026) -> dict:
    """The real C_FULL s2026 RUN_START as the C2 shim writes it (run id / run label / SMALL variant + recipe; the same
    REGISTRY_V2_FULL_EXT binding and trainer view), config_sha256 recomputed."""
    st = copy.deepcopy(real_start("C_FULL"))
    rid = f"P2_C2_CFULL_SMALL_{VAR}_s{seed}"
    st["run_id"] = st["config"]["run_id"] = rid
    st["run_label"] = st["key_base"]["run_label"] = "P2_C2_CFULL_SMALL"
    st["model_variant"] = st["config"]["model_variant"] = st["key_base"]["model_variant"] = VAR
    st["config"]["recipe_id"] = st["key_base"]["recipe_id"] = REC
    return rehash(st)


def test_c2_dry_run_as_a_registered_source(tmp_path):
    for ckpt, rule in (("best.pt", PB), ("endpoint_6.0.pt", CB)):
        t = tmp_path / f"c2_{rule}"
        cp = source_run(t, "C_FULL", ckpt=ckpt, start=c2_start(), meta_over={"model_variant": VAR})
        try:
            out = RP.run(args_for(cp, t, label="P2_C2_CFULL_SMALL", rule=rule))
        except R.ReevalRefused as e:
            raise AssertionError(f"C2 {rule} refused: {e}")
        assert restored() and "C_FULL" in R.CENTRAL_SOURCES
        cs = out["central_source"]
        assert cs["source"] == "P2_C2_CFULL_SMALL" and cs["checks"]["all_ok"] and cs["binding"]["view_checks"]["all_ok"]
        assert cs["pair"] == {"train": "FULL_TRAIN_5_100_TIEBREAK", "eval": MAIN_VAL}
        assert out["planned_key"].startswith("SOURCE_TRAIN_SIDE"), out["planned_key"]
        assert out["checkpoint_meta"]["efe"] == out["checkpoint_meta"]["mark"] == RP.SEALED
        assert out["row_labels"]["central_source"] == "P2_C2_CFULL_SMALL" and out["row_labels"]["study"] == "PREG2"
        assert out["p2_reeval"]["source_kind"] == "CENTRAL_C2"


def test_c2_refusals(tmp_path):
    t = tmp_path / "cfull"                                                # the Phase-1 C_FULL run (D256) is not C2
    cp = source_run(t, "C_FULL")
    refused(lambda: RP.run(args_for(cp, t, label="P2_C2_CFULL_SMALL")), "differs from requested")
    t = tmp_path / "s2027"
    st = c2_start(2027)
    st["seed"] = st["config"]["seed"] = 2027
    cp = source_run(t, "C_FULL", start=rehash(st), meta_over={"model_variant": VAR, "seed": 2027})
    refused(lambda: RP.run(args_for(cp, t, label="P2_C2_CFULL_SMALL", seed=2027)), "seeds (2026,)")
    t = tmp_path / "c2"
    cp = source_run(t, "C_FULL", start=c2_start(), meta_over={"model_variant": VAR})
    refused(lambda: RP.run(args_for(cp, t, label="C_FULL")), "PREG2 §6.3 S5")
    st = c2_start()
    st["config"]["peak_lr"] = 1.0                                         # not re-hashed: FIX-REEVAL binding refuses
    t = tmp_path / "unbound"
    cp = source_run(t, "C_FULL", start=st, meta_over={"model_variant": VAR, "config_sha256": st["config_sha256"]})
    refused(lambda: RP.run(args_for(cp, t, label="P2_C2_CFULL_SMALL")), "config_sha256 = RunConfig.sha256")
    t = tmp_path / "p1path"                                               # the Phase-1 path refuses the C2 run
    cp = source_run(t, "C_FULL", start=c2_start(), meta_over={"model_variant": VAR})
    with pytest.raises(R.ReevalRefused, match="C_FULL central source"):
        R.run(src_argv(cp, t, "C_FULL", variant=VAR, recipe_id=REC))


# ------------------------------------------------------------------------------------------------ TEST roles (T0a)
def write_register(path: Path, entries) -> Path:
    from v4data09 import d9_gate as G
    tool = G._register_tool()
    out, prev = [], None
    for i, (purpose, lock_sha, desc) in enumerate(entries, 1):
        e = {"entry_id": i, "utc": "2026-10-03T00:00:00Z", "actor_role": "OPS", "source": "test_p2_reeval",
             "split": "RETRO_TEST", "access_purpose": purpose, "observations": "NONE",
             "observations_description": desc, "decision": "test", "all_methods_lock_sha256": lock_sha,
             "manifest_header_sha256": None, "model_checkpoint_sha256": None, "code_sha256": None,
             "receipt_path": None, "prev_entry_sha256": prev}
        e["entry_sha256"] = hashlib.sha256(tool.canon(e)).hexdigest()
        out.append(e)
        prev = e["entry_sha256"]
    path.write_text(json.dumps({"register_id": "TEST_EXPOSURE_REGISTER", "entries": out}), encoding="utf-8")
    assert tool.verify(json.loads(path.read_text(encoding="utf-8"))) == []
    return path


def gate_env(tmp, monkeypatch, *, ckpt_sha, p1=True, p2=True, freeze=True, lock=True, arms=None, dnr=(),
             data09_open=True, p2_desc="LOCKED_EVALUATION for P2_METHODS_LOCK (Phase-2 sitting)", lock_over=None):
    g = tmp / "gates"
    g.mkdir(parents=True, exist_ok=True)
    aml = g / "ALL_METHODS_LOCK.json"
    aml.write_text('{"lock": "phase-1 stand-in"}', encoding="utf-8")
    p2l, frz = g / "P2_METHODS_LOCK.json", g / "P2_SCRIPT_FREEZE.json"
    if lock:
        arms = arms if arms is not None else [{"arm": "P2_WSTAR", "seed": 2026, "rule": PB,
                                               "checkpoint_sha256": ckpt_sha}]
        obj = {"format": RP.P2_LOCK_FORMAT, "draft": False, "preg2_sha256": RP.PREG2_SHA256,
               "p2_contrasts_self_sha256": RP.CONTRASTS_SELF_SHA256, "bootstrap_seed": 20261004, "sealed_arms": arms,
               "phase1_reused": [], "declared_not_run": list(dnr), "selection_files_sha256": {}}
        obj.update(lock_over or {})
        p2l.write_text(json.dumps(obj), encoding="utf-8")
    if freeze:
        frz.write_text('{"freeze": "stand-in"}', encoding="utf-8")
    ents = [("MANIFEST_BUILD", _sha(aml), "manifest build")]
    if p1:
        ents.append(("LOCKED_EVALUATION", _sha(aml), "LOCKED_EVALUATION for ALL_METHODS_LOCK"))
    if p2 and lock:
        ents.append(("LOCKED_EVALUATION", _sha(p2l), p2_desc))
    reg = write_register(g / "TEST_EXPOSURE_REGISTER.json", ents)
    for k, v in (("AML_PATH", aml), ("REGISTER_PATH", reg), ("P2_METHODS_LOCK", p2l), ("P2_SCRIPT_FREEZE", frz)):
        monkeypatch.setattr(RP, k, v)
    monkeypatch.setattr(R, "test_gate_ok", lambda role: bool(data09_open))
    return g


def main_test_args(cp, t):
    return args_for(cp, t, kind="fl", role="MAIN_TEST", registry_ext=str(t / "REGISTRY_V2_TEST_EXT.json"))


def check_test_refusals(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    cases = [(dict(data09_open=False), "DATA-09 TEST gate is closed"), (dict(lock=False), "P2_METHODS_LOCK"),
             (dict(freeze=False), "P2_SCRIPT_FREEZE"), (dict(p1=False), "Phase-1 ALL_METHODS_LOCK"),
             (dict(p2=False), "binds P2_METHODS_LOCK"), (dict(p2_desc="phase-2 lock"), "observations_description"),
             (dict(arms=[]), "binds 0 entries"),
             (dict(arms=[{"arm": "P2_WSTAR", "seed": 2026, "rule": CB, "checkpoint_sha256": "0" * 64}]),
              "binds 0 entries"),
             (dict(arms=[{"arm": "P2_WSTAR", "seed": 2026, "rule": PB, "checkpoint_sha256": "0" * 64}]),
              "to checkpoint"),
             (dict(dnr=[{"arm": "P2_WSTAR", "seed": None, "reason": "test"}]), "declared not run"),
             (dict(dnr=[{"arm": "P2_WSTAR", "seed": 2026, "reason": "test"}]), "declared not run"),
             (dict(lock_over={"draft": True}), "draft"), (dict(lock_over={"format": "OTHER"}), "format"),
             (dict(lock_over={"preg2_sha256": "0" * 64}), "preg2_sha256")]
    for i, (kw, msg) in enumerate(cases):
        t = tmp_path / f"g{i}"
        cp, _, _ = fl_run(t, sel)
        gate_env(t, monkeypatch, ckpt_sha=_sha(cp), **kw)
        refused(lambda: RP.run(main_test_args(cp, t)), msg)


def test_test_role_refused_without_gate_or_p2_lock(tmp_path, monkeypatch):
    check_test_refusals(tmp_path, monkeypatch)


def test_test_role_admitted_with_gate_and_p2_lock(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    t = tmp_path / "ok"
    cp, rid, _ = fl_run(t, sel)
    gate_env(t, monkeypatch, ckpt_sha=_sha(cp))
    out = RP.run(main_test_args(cp, t))
    tl = out["p2_reeval"]["test_lock"]
    assert out["dry_run"] and out["source_run_id"] == rid and restored()
    assert tl["arm_binding"]["checkpoint_sha256"] == _sha(cp) and tl["locked_evaluation_entry_ids"] == {"phase1": 2,
                                                                                                       "p2": 3}
    assert tl["p2_methods_lock"]["sha256"] == _sha(t / "gates" / "P2_METHODS_LOCK.json")
    assert out["p2_reeval"]["role_reading"].startswith("P2_SITTING_RETRO_TEST")


def test_test_role_refused_when_p2_lock_missing(tmp_path, monkeypatch):
    """A P2 TEST reeval refuses without its lock, even after a real lock has been created."""
    monkeypatch.setattr(RP, "P2_METHODS_LOCK", tmp_path / "absent" / "P2_METHODS_LOCK.json")
    assert not RP.P2_METHODS_LOCK.exists()
    got = refused(lambda: RP.run(main_test_args(tmp_path / "missing" / "fl" / "ckpt" / "P2_WSTAR_SMALL_s2026" / "best.pt",
                                           tmp_path)), "refused")
    assert "DATA-09" in got or "P2_METHODS_LOCK" in got


@nc("a MAIN_TEST reeval without the P2 lock binding (only the DATA-09 gate)")
def test_nc_test_without_p2_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(RP, "p2_test_gate", lambda args, plan: None)
    check_test_refusals(tmp_path, monkeypatch)


def test_test2_and_other_roles_refused(tmp_path):
    missing = tmp_path / "missing.pt"
    for role, msg in (("MAIN_TEST2", "owner O2 = NO"), ("TEST2_DEC2019", "owner O2 = NO"),
                      ("UNSEEN_HOLDOUT_TEST", "not a Phase-2 scoring role"), ("SPARSE_PANEL", "not a Phase-2 scoring"),
                      ("UNSEEN_VAL", "not a Phase-2 scoring"), ("TRUE_COLD", "not a Phase-2 scoring")):
        refused(lambda: RP.run(args_for(missing, tmp_path, kind="fl", role=role)), msg)


# ------------------------------------------------------------------------------------------------ labels
def check_phase1_labels_refused(tmp_path):
    t = tmp_path / "pre_main"
    cp = source_run(t, "PRE")                     # a real, admissible Phase-1 source checkpoint (reeval scores it)
    assert R.run(src_argv(cp, t, "PRE", selection_rule=CB))["dry_run"]
    for role in ("MAIN", "MAIN_TEST", "CALIBRATION"):
        for label in RP.S5_LABELS:
            refused(lambda: RP.run(src_argv(cp, t, "PRE", role=role, arm_label=label, selection_rule=CB)), "S5")
    for label in ("C_SMALL", "FA", "FA_WARM", "DP8_FROZEN", "C_BIG", "P2_W", "P2_FOO"):
        refused(lambda: RP.run(src_argv(cp, t, "PRE", arm_label=label, selection_rule=CB)), "not a Phase-2 arm")
    for label, why in RP.PENDING_LABELS.items():
        refused(lambda: RP.run(src_argv(cp, t, "PRE", arm_label=label, selection_rule=CB)), why)


def test_phase1_labels_refused_through_the_p2_path(tmp_path):
    check_phase1_labels_refused(tmp_path)


@nc("a P2 path without the label admission (Phase-1 PRE on MAIN scored through reeval_p2)")
def test_nc_no_label_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(RP, "label_plan", lambda args: {"label": args.arm_label, "kind": "C1", "role": args.role,
                                                        "test": False, "seed": 2026, "rule": args.selection_rule})
    monkeypatch.setattr(RP, "check_central_source", lambda *a, **k: None)
    check_phase1_labels_refused(tmp_path)


def test_labels_are_the_p2_contrast_labels():
    obj = json.loads(RP.CONTRASTS_PATH.read_text(encoding="utf-8"))
    assert obj["self_sha256"] == RP.CONTRASTS_SELF_SHA256 and obj["draft"] is False
    text = " ".join(c["contrast"] for c in obj["confirmatory_contrasts"])
    found = set(re.findall(r"\b(?:FT_)?P2_[A-Z0-9_]+\b", text))
    assert found == set(RP.ARMS) | set(RP.PENDING_LABELS), sorted(found ^ (set(RP.ARMS) | set(RP.PENDING_LABELS)))
    assert set(RP.S5_LABELS) <= set(re.findall(r"\b[A-Z][A-Z0-9_]+\b", text)) | {"FT_PRE"}
    assert not set(RP.ARMS) & set(R.CENTRAL_SOURCES) and RP.ANCHOR_LABEL not in found


def test_reeval_unchanged_after_the_suite():
    assert _sha(ppsi_paths.locate("code/v4int/reeval.py")) == REEVAL_SHA_AT_TASK_START and restored()

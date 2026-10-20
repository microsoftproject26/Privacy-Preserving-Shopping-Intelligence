"""P2-DEVLOCK piece A - v4int.launch_device_p2, the FT_P2_WSTAR shim around the UNEDITED v4int.launch_device (CPU; dry
runs, the torch-free base binding, the base load into a tiny adapter, the FT_C cell on the base in the tiny device world
of test_int04_device; no real data, no GPU, no run). Real pieces: launch_fl.p2_arm_spec / check_p2_spec / spec_digest /
fl_run_config, the registered SELECTION_FT_C.json, the register tool (via v4data09.d9_gate), reeval_p2.p2_lock_arm.

  * launch_device.py is byte-identical (pinned sha); every patched launch_device attribute is restored after each call,
    refusals and exceptions included; no nesting;
  * a MAIN dry run binds the W* base (best.pt of P2_WSTAR_SMALL_s<seed>, STATUS DONE_MAIN, ARM_SPEC / LAUNCH agree,
    check_p2_spec, theta_pre, the registered P2_WARM_SELECTION_FINAL by sha) and records its sha256; no efe / round;
  * refusals: other arm / role / seed, smoke, run-dir name, base kind, --init-from, endpoint_6.0.pt, another seed, an
    unfinished run, a tampered spec, a CALIBRATION screen, a Phase-1 FL run, init_from != theta_pre, a selection sha
    mismatch / absent file, a PF recipe, a sha pin mismatch, an unregistered FT selection file;
  * the start record is what tools/p2_block_index --p2-dev reads (kind P2_WSTAR, seed, best.pt, file_sha256, run_id);
    rows carry study PREG2 + the base identity; the FT_P2_WSTAR cell is bitwise the FT_C cell on the same start;
  * MAIN_TEST: admitted only with the DATA-09 gate + P2 lock + freeze + both LOCKED_EVALUATION entries + the sealed
    W* / FT_P2_WSTAR base; refused on this host now (no P2_METHODS_LOCK).
Negative control (test_nc_*, strict xfail on AssertionError): the TEST gate removed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import v4int.launch_device as D
import v4int.launch_fl as LF
from fl.numerics import state_digest
from fl.synthetic import make_tiny_adapter
from intkit import nc
from test_int04_device import K, PEAK, _frozen, _plans, _world
from v4int import launch_device_p2 as LDP
from v4int import test_roles as TR

VAR = "SASREC_D64_B2"
PB = "PRACTICAL_BEST"
LAUNCH_DEVICE_SHA_AT_TASK_START = "1968ca9550ed9ac70d9310753aa5aa33243e7bb79cabc6e4fa7cd95ad701642c"
ORIG = {k: getattr(D, k) for k in LDP.PATCHED}
INIT = {"kind": "PRETRAIN_FOR_FL", "run_id": "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026", "run_label": "PRETRAIN_FOR_FL",
        "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN", "fl_init_kind": "endpoint_6.0",
        "checkpoint_sha256": LDP.PRE_CHECKPOINT_SHA256, "checkpoint_bytes": 1234, "state_digest": LDP.PRE_STATE_DIGEST,
        "model_variant": VAR, "family": "SASREC", "seed": 2026}
INCUMBENT = {"server_opt": "fedadam", "server_lr": 0.001, "client_lr": 0.1}
SEL_C = {"arm": "FT_C", "cell": "FT_C_f0.3_P2", "selection_role": "UNSEEN_VAL", "self_sha256": "s" * 64}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _w(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def restored() -> bool:
    return all(getattr(D, k) is v for k, v in ORIG.items()) and not getattr(D, "__p2_device_context__", False)


def refused(fn, msg) -> str:
    got = None
    try:
        fn()
    except D.LaunchRefused as e:
        got = str(e)
    assert got is not None and msg in got, (msg, got)
    assert restored(), "a launch_device attribute was not restored after the refusal"
    return got


# ------------------------------------------------------------------------------------------------ fixtures
def p2_spec(seed=2026, recipe=None, init=None):
    return LF.p2_arm_spec("P2_W", recipe=dict(recipe or INCUMBENT), seed=seed, peak_lr=1e-3, role="MAIN",
                          init_from=dict(init or INIT))


def selection_file(tmp, monkeypatch) -> Path:
    p = tmp / "contracts" / "fl" / "P2_WARM_SELECTION_FINAL.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"stub": "the registered P2_WARM_SELECTION_FINAL"}', encoding="utf-8")
    monkeypatch.setattr(LDP, "SELECTION_FINAL", p)
    return p


def wstar_run(root: Path, sel: Path, *, seed=2026, spec=None, rid=None, arm="P2_W", status="DONE_MAIN",
              role="MAIN", label="PRODUCTION", sel_over=None, launch_over=None, side_over=None, payload_over=None,
              efe=3.0) -> Path:
    """A finished MAIN P2_W run (value-free records + fl weights payloads of the tiny untied model)."""
    from fl.checkpoint import atomic_save
    spec = spec or p2_spec(seed)
    rid = rid or f"P2_WSTAR_SMALL_s{seed}"
    run = root / f"P2_WSTAR_s{seed}"
    cd = LF.fl_run_config(spec, rid, 1).digest()
    srec = {"source": "p2_selection", "path": str(sel), "sha256": _sha(sel), "self_sha256": "f" * 64,
            "format": LF.P2_SELECTION_FORMAT, "stage": "FINAL", "label": spec["p2_label"], "recipe": spec["p2_recipe"],
            "init_from_checked": True, "flags": [], "authority": LF.P2_AUTHORITY}
    srec.update(sel_over or {})
    launch = {"arm": arm, "run_id": rid, "arm_spec": spec, "arm_spec_sha256": LF.spec_digest(spec),
              "fl_config_digest": cd, "label": label, "fl05": {"role": role, "lr_mult": 1.0}, "p2_selection": srec}
    launch.update(launch_over or {})
    _w(run / "LAUNCH_0.json", launch)
    side = {"arm": arm, "run_id": rid, "arm_spec_sha256": LF.spec_digest(spec), "arm_spec": spec}
    side.update(side_over or {})
    _w(run / "ARM_SPEC.json", side)
    _w(run / "STATUS.json", {"run_id": rid, "arm": arm, "status": status, "utc": "2026-10-03T00:00:00Z"})
    theta = {k: v.clone() for k, v in make_tiny_adapter(K=K, seed=5, tied=False).broadcast_state().items()}
    for f, e in (("best.pt", efe), ("endpoint_6.0.pt", 6.0)):
        payload = {"kind": "weights", "run_id": rid, "config_digest": cd, "round": 9, "efe": e, "phase": "MAIN",
                   "theta": theta}
        payload.update(payload_over or {})
        atomic_save(payload, run / "ckpt" / rid / f)
    return run / "ckpt" / rid / "best.pt"


def argv(base, tmp, *, seed=2026, role="MAIN", dry=True, extra=(), run_dir=None):
    rd = run_dir or (tmp / LDP.RUN_DIR_NAME.get(role, "FT_P2_WSTAR_s{seed}").format(seed=seed))
    a = ["--arm", "FT_P2_WSTAR", "--seed", str(seed), "--gpu", "0", "--run-dir", str(rd), "--history-policy",
         "USER_HISTORY_L50", "--role", role, "--base", str(base)]
    return a + list(extra) + (["--dry-run"] if dry else [])


def pre_of(base, tmp, **kw) -> dict:
    return LDP.preflight(LDP.parse_args(argv(base, tmp, **kw)), environ={})


# ------------------------------------------------------------------------------------------------ pin / context
def test_launch_device_pinned_and_context_restored(monkeypatch):
    assert _sha(D.__file__) == LDP.LAUNCH_DEVICE_SHA256 == LAUNCH_DEVICE_SHA_AT_TASK_START
    with LDP.p2_device_context():
        assert "FT_P2_WSTAR" in D.ARMS and D.FT_WARM_BASE["FT_P2_WSTAR"] == "P2_WSTAR"
        assert D.SELECTION_ARM["FT_P2_WSTAR"] == D.FT_PROCEDURE["FT_P2_WSTAR"] == "FT_C"
        try:
            with LDP.p2_device_context():
                raise AssertionError("nested context admitted")
        except D.LaunchRefused as e:
            assert "no nesting" in str(e)
    assert restored() and "FT_P2_WSTAR" not in D.ARMS and "FT_P2_WSTAR" not in D.FT_WARM_BASE
    try:
        with LDP.p2_device_context():
            raise KeyError("boom")
    except KeyError:
        pass
    assert restored()
    monkeypatch.setattr(LDP, "LAUNCH_DEVICE_SHA256", "0" * 64)
    refused(LDP.check_launch_device_unchanged, "refuses to wrap it")


# ------------------------------------------------------------------------------------------------ MAIN dry run
def test_main_dry_run_binds_the_wstar_base(tmp_path, monkeypatch, capsys):
    sel = selection_file(tmp_path, monkeypatch)
    base = wstar_run(tmp_path / "runs", sel)
    assert LDP.main(argv(base, tmp_path)) == 0
    out = json.loads(capsys.readouterr().out)
    assert restored()
    p2 = out["preflight"]["p2_device"]
    b = p2["base"]
    assert b["kind"] == "P2_WSTAR" and b["seed"] == 2026 and b["checkpoint"] == "best.pt" and b["status"] == "DONE_MAIN"
    assert b["file_sha256"] == _sha(base) and b["run_id"] == "P2_WSTAR_SMALL_s2026" and b["launch_arm"] == "P2_W"
    assert b["p2_selection"]["sha256"] == _sha(sel) and b["init_from"]["checkpoint_sha256"] == LDP.PRE_CHECKPOINT_SHA256
    assert "efe" not in json.dumps(b) and "round" not in b                       # S8: the PB mark is never recorded
    fw = out["preflight"]["ft_fa_warm"]
    assert fw["selection"]["file_sha256"] == D.FT_PRE_SELECTION_SHA256 and fw["base_arm"] == "P2_WSTAR"
    assert p2["launch_device_sha256"] == LDP.LAUNCH_DEVICE_SHA256 and p2["test_lock"] is None
    assert out["preflight"]["arm"] == "FT_P2_WSTAR" and out["cells"][0]["cell"].startswith("FT_P2_WSTAR_f")
    pre = pre_of(base, tmp_path, seed=2026, extra=("--base-sha256", _sha(base)))     # the optional sha pin
    assert pre["p2_device"]["base"]["file_sha256"] == _sha(base) and restored()


def test_admission_refusals(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    base = wstar_run(tmp_path / "runs", sel)
    ok = argv(base, tmp_path)
    cases = [(["--arm", "FT_FA_WARM"] + ok[2:], "FT_P2_WSTAR only"),
             (argv(base, tmp_path, role="CALIBRATION"), "runs on --role"),
             (argv(base, tmp_path, role="UNSEEN_HOLDOUT"), "runs on --role"),
             (argv(base, tmp_path, role="UNSEEN_VAL"), "runs on --role"),
             (argv(base, tmp_path, role="SPARSE_PANEL_TEST"), "runs on --role"),
             (argv(base, tmp_path, seed=2028, run_dir=tmp_path / "FT_P2_WSTAR_s2028"), "registered at seeds"),
             (ok + ["--smoke"], "--smoke is refused"),
             (argv(base, tmp_path, run_dir=tmp_path / "FT_P2_WSTAR_MAIN_s2026"), "--run-dir must be named"),
             (ok + ["--base-kind", "fl"], "--base-kind"),
             (ok + ["--init-from", str(base)], "--init-from refused"),
             (argv(base, tmp_path, role="MAIN_TEST"), "--base-sha256 is required"),
             (ok + ["--base-sha256", "0" * 64], "!= --base-sha256"),
             (ok + ["--selection", str(sel)], "--selection")]
    for a, msg in cases:
        refused(lambda: LDP.preflight(LDP.parse_args(a), environ={}), msg)


def test_base_refusals(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    i = iter(range(100))

    def case(msg, *, seed=2026, path_fn=None, **kw):
        t = tmp_path / f"c{next(i)}"
        run_seed = kw.pop("run_seed", seed)
        b = wstar_run(t, sel, seed=run_seed, **kw)
        b = path_fn(b) if path_fn else b
        refused(lambda: pre_of(b, t, seed=seed), msg)

    case("Q5 base_best_only", path_fn=lambda b: b.parent / "endpoint_6.0.pt")
    case("is not the MAIN W* run", seed=2027, run_seed=2026)                       # another seed's W*
    case("is not the MAIN W* run", rid="P2_W_FEDADAM_eta0.001_c0.1_SMALL_s2026")      # a recipe-file (smoke) run
    case("not a DONE_MAIN", status="RUNNING")
    case("not a DONE_MAIN", arm="FA_WARM_FROZEN")                                  # a Phase-1 warm FL run
    case("does not hash", side_over={"arm_spec_sha256": "0" * 64})
    case("CALIBRATION screen", role="CALIBRATION")
    case("LAUNCH label", label="SMOKE_NOT_EVIDENCE")
    case("differs from ARM_SPEC", launch_over={"arm_spec_sha256": "1" * 64})
    tam = p2_spec()
    tam["cfg"] = dict(tam["cfg"], dropout_p=0.2)                                   # a hand-edited spec, re-hashed
    case("check_p2_spec", spec=tam)
    case("theta_pre", spec=p2_spec(init=dict(INIT, checkpoint_sha256="a" * 64)))
    case("is not the file the base run was bound to", sel_over={"sha256": "2" * 64})
    case("p2_selection is", sel_over={"stage": "A"})
    case("recipe / label differ", sel_over={"label": "OTHER"})
    case("not the registered contracts/fl", sel_over={"path": str(tmp_path / "copy" / "P2_WARM_SELECTION_FINAL.json")})
    t = tmp_path / "nosel"
    b = wstar_run(t, sel)
    monkeypatch.setattr(LDP, "SELECTION_FINAL", tmp_path / "gone" / "contracts" / "fl" / "P2_WARM_SELECTION_FINAL.json")
    refused(lambda: pre_of(b, t), "is not on this host")


def test_pf_recipe_base_fails_closed(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    try:
        pf = p2_spec(recipe=dict(INCUMBENT, personal_head=True))
    except LF.LaunchRefused:                                    # the P2 recipe itself refuses a PF incumbent: nothing to do
        return
    b = wstar_run(tmp_path, sel, spec=pf)
    refused(lambda: pre_of(b, tmp_path), "personal-head")


# ------------------------------------------------------------------------------------------------ load / rows / procedure
def _resolve(base, tmp):
    args = LDP.parse_args(argv(base, tmp, dry=False))
    LDP.check_args(args)
    with LDP.p2_device_context():
        st, b, rec = D.resolve_start(args, None, make_tiny_adapter(K=K, seed=1, tied=False), theta0="t0",
                                     model_variant=VAR)
        lab = D.ft_warm_row_labels(rec, {"self_sha256": "q" * 64})
    assert restored()
    return st, b, rec, lab


def test_resolve_start_loads_the_base_and_labels_rows(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    base = wstar_run(tmp_path / "runs", sel)
    from fl.checkpoint import load_checkpoint
    st, b, rec, lab = _resolve(base, tmp_path)
    assert b["digest"] == state_digest(st) == state_digest(load_checkpoint(base)["theta"])
    for k, v in (("kind", "P2_WSTAR"), ("seed", 2026), ("checkpoint", "best.pt"), ("file_sha256", _sha(base)),
                 ("run_id", "P2_WSTAR_SMALL_s2026"), ("base_kind", D.WARM_BASE_KIND)):      # = p2_block_index p2_dev
        assert rec[k] == v, k
    assert "state" not in rec and rec["merged_lora"] is None and not rec["run_id"].startswith("PRETRAIN_FOR_FL")
    assert lab["study"] == "PREG2" and lab["p2_arm"] == "FT_P2_WSTAR" and lab["base_source"] == "P2_WSTAR"
    assert lab["base_checkpoint_sha256"] == _sha(base) and lab["base_p2_selection_sha256"] == _sha(sel)
    assert lab["base_init_checkpoint_sha256"] == LDP.PRE_CHECKPOINT_SHA256 and lab["ft_selection_self_sha256"] == "q" * 64
    for i, (kw, msg) in enumerate(((dict(payload_over={"config_digest": "X"}), "config_digest"),
                                   (dict(efe=0.0), "mark >= 0.5"),
                                   (dict(payload_over={"kind": "train_state"}), "not the fl weights"))):
        t = tmp_path / f"p{i}"
        b2 = wstar_run(t, sel, **kw)
        refused(lambda: _resolve(b2, t), msg)


def _row_fn(spec, over):
    from v4eval.rows import COMPARISON_KEY
    kb = {f: "x" for f in COMPARISON_KEY}
    kb.update(aggregation="macro", target_denominator="E2E")
    return lambda res, cell, selected: D.device_rows(res, kb, {"run_label": "PRODUCTION"}, spec=spec, cell=cell,
                                                     selected=selected, down_bytes=7, overrides=over)


def test_ft_p2_wstar_cell_is_the_ft_c_procedure_on_the_base(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    base = wstar_run(tmp_path / "runs", sel)
    st, _b, rec, lab = _resolve(base, tmp_path)
    w = _world()
    out = {}
    with LDP.p2_device_context():
        reg = D.load_selection(D.FT_PRE_SELECTION)
        s_reg = D.device_spec("FT_P2_WSTAR", seed=2026, peak_lr=PEAK, role="MAIN", start_digest="x", selection=reg)
        assert s_reg["selected_cell"] == "FT_P2_WSTAR_f0.3_P4" and s_reg["procedure_selection_cell"] == "FT_C_f0.3_P4"
        assert s_reg["ft_procedure"] == "FT_C" and s_reg["base_arm"] == "P2_WSTAR"
        for arm in ("FT_P2_WSTAR", "FT_C"):
            spec = D.device_spec(arm, seed=2026, peak_lr=PEAK, role="MAIN", start_digest=state_digest(st),
                                 selection=SEL_C)
            out[arm] = D.execute(spec=spec, run_dir=tmp_path / arm, worker=make_tiny_adapter(K=K, seed=3, tied=False),
                                 start_state=st, plans=_plans(w), get_support=lambda q: w.sup[q.key],
                                 get_eval_batch=w.get_eval_batch, manifest=w.m, frozen=_frozen(),
                                 row_fn=_row_fn(spec, lab if arm == "FT_P2_WSTAR" else None), say=lambda *_: None,
                                 n_shards=2)
    assert restored()
    a = np.load(tmp_path / "FT_P2_WSTAR" / "cells" / "FT_P2_WSTAR_f0.3_P2.ranks_int32.npy")
    b = np.load(tmp_path / "FT_C" / "cells" / "FT_C_f0.3_P2.ranks_int32.npy")
    assert np.array_equal(a, b)
    assert out["FT_P2_WSTAR"]["values"]["FT_P2_WSTAR_f0.3_P2"] == out["FT_C"]["values"]["FT_C_f0.3_P2"]
    rows = json.loads((tmp_path / "FT_P2_WSTAR" / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert rows and {r["method"] for r in rows} == {"FT_P2_WSTAR"} and {r["study"] for r in rows} == {"PREG2"}
    assert {r["ft_procedure"] for r in rows} == {"FT_C"} and {r["base_checkpoint_sha256"] for r in rows} == {_sha(base)}
    summ = json.loads((tmp_path / "FT_P2_WSTAR" / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert summ["arm"] == "FT_P2_WSTAR" and summ["status"] == "DONE" and summ["selected_cell"] == "FT_P2_WSTAR_f0.3_P2"
    assert summ["spec"]["seed"] == 2026 and summ["spec"]["base_arm"] == "P2_WSTAR"


# ------------------------------------------------------------------------------------------------ TEST (T0a)
def write_register(path: Path, entries) -> Path:
    from v4data09 import d9_gate as G
    tool = G._register_tool()
    out, prev = [], None
    for i, (purpose, lock_sha, desc) in enumerate(entries, 1):
        e = {"entry_id": i, "utc": "2026-10-03T00:00:00Z", "actor_role": "OPS", "source": "test_p2_device_ft",
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


def gate_env(tmp, monkeypatch, *, base_sha, p1=True, p2=True, freeze=True, lock=True, arms=None, dnr=(),
             data09_open=True, p2_desc="LOCKED_EVALUATION for P2_METHODS_LOCK (Phase-2 sitting)"):
    g = tmp / "gates"
    g.mkdir(parents=True, exist_ok=True)
    aml = g / "ALL_METHODS_LOCK.json"
    aml.write_text('{"lock": "phase-1 stand-in"}', encoding="utf-8")
    p2l, frz = g / "P2_METHODS_LOCK.json", g / "P2_SCRIPT_FREEZE.json"
    if lock:
        arms = arms if arms is not None else [{"arm": "P2_WSTAR", "seed": 2026, "rule": PB, "checkpoint_sha256": base_sha},
                                              {"arm": "FT_P2_WSTAR", "seed": 2026, "rule": PB,
                                               "base_checkpoint_sha256": base_sha}]
        _w(p2l, {"format": "PPSI_V4_P2_METHODS_LOCK_1", "draft": False, "preg2_sha256": LDP.PREG2_SHA256,
                 "p2_contrasts_self_sha256": "e98580564b618f411db7e7e328fbcedb6e1134f4be5b65806b34e12229373799",
                 "bootstrap_seed": 20261004, "sealed_arms": arms, "phase1_reused": [], "declared_not_run": list(dnr),
                 "selection_files_sha256": {}})
    if freeze:
        frz.write_text('{"freeze": "stand-in"}', encoding="utf-8")
    ents = [("MANIFEST_BUILD", _sha(aml), "manifest build")]
    if p1:
        ents.append(("LOCKED_EVALUATION", _sha(aml), "LOCKED_EVALUATION for ALL_METHODS_LOCK"))
    if p2 and lock:
        ents.append(("LOCKED_EVALUATION", _sha(p2l), p2_desc))
    reg = write_register(g / "TEST_EXPOSURE_REGISTER.json", ents)
    for k, v in (("AML_PATH", aml), ("REGISTER_PATH", reg), ("P2_METHODS_LOCK", p2l), ("P2_SCRIPT_FREEZE", frz)):
        monkeypatch.setattr(LDP, k, v)
    monkeypatch.setattr(TR, "test_gate_ok", lambda role, **kw: bool(data09_open))


def mt_argv(base, t):
    return argv(base, t, role="MAIN_TEST", extra=("--base-sha256", _sha(base), "--registry-ext",
                                                  str(t / "REGISTRY_V2_TEST_EXT.json")))



def check_test_refusals(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    z = "0" * 64
    cases = [(dict(data09_open=False), "DATA-09 TEST gate is closed"), (dict(lock=False), "P2_METHODS_LOCK"),
             (dict(freeze=False), "P2_SCRIPT_FREEZE"), (dict(p1=False), "Phase-1 ALL_METHODS_LOCK"),
             (dict(p2=False), "binds P2_METHODS_LOCK"), (dict(p2_desc="phase-2 lock"), "observations_description"),
             (dict(arms=[]), "binds 0 entries"),
             (dict(arms=[{"arm": "P2_WSTAR", "seed": 2026, "rule": PB, "checkpoint_sha256": "BASE"}]),
              "binds 0 entries for (FT_P2_WSTAR"),
             (dict(arms=[{"arm": "P2_WSTAR", "seed": 2026, "rule": PB, "checkpoint_sha256": z},
                         {"arm": "FT_P2_WSTAR", "seed": 2026, "rule": PB, "base_checkpoint_sha256": "BASE"}]),
              "to checkpoint"),
             (dict(arms=[{"arm": "P2_WSTAR", "seed": 2026, "rule": PB, "checkpoint_sha256": "BASE"},
                         {"arm": "FT_P2_WSTAR", "seed": 2026, "rule": PB, "base_checkpoint_sha256": z}]), "to base"),
             (dict(dnr=[{"arm": "FT_P2_WSTAR", "seed": None, "reason": "test"}]), "declared not run"),
             (dict(dnr=[{"arm": "P2_WSTAR", "seed": 2026, "reason": "test"}]), "declared not run")]
    for i, (kw, msg) in enumerate(cases):
        t = tmp_path / f"g{i}"
        b = wstar_run(t / "runs", sel)
        if kw.get("arms"):
            kw["arms"] = [{k: (_sha(b) if v == "BASE" else v) for k, v in e.items()} for e in kw["arms"]]
        gate_env(t, monkeypatch, base_sha=_sha(b), **kw)
        refused(lambda: LDP.preflight(LDP.parse_args(mt_argv(b, t)), environ={}), msg)


def test_test_role_refused_without_gate_or_p2_lock(tmp_path, monkeypatch):
    check_test_refusals(tmp_path, monkeypatch)


def test_test_role_admitted_with_gate_and_p2_lock(tmp_path, monkeypatch):
    sel = selection_file(tmp_path, monkeypatch)
    b = wstar_run(tmp_path / "runs", sel)
    gate_env(tmp_path, monkeypatch, base_sha=_sha(b))
    pre = LDP.preflight(LDP.parse_args(mt_argv(b, tmp_path)), environ={})
    tl = pre["p2_device"]["test_lock"]
    assert restored() and pre["test_gate"]["role"] == "MAIN_TEST" and pre["p2_device"]["base"]["file_sha256"] == _sha(b)
    assert tl["locked_evaluation_entry_ids"] == {"phase1": 2, "p2": 3}
    assert tl["arm_binding"]["device"]["base_checkpoint_sha256"] == _sha(b)
    assert tl["arm_binding"]["wstar"]["checkpoint_sha256"] == _sha(b)
    assert tl["p2_methods_lock"]["sha256"] == _sha(tmp_path / "gates" / "P2_METHODS_LOCK.json")


def test_test_role_refused_on_this_host_now(tmp_path, monkeypatch):
    """The real state (no P2_METHODS_LOCK exists yet): a MAIN_TEST FT_P2_WSTAR launch refuses, whatever DATA-09 says."""
    assert not LDP.P2_METHODS_LOCK.exists()
    sel = selection_file(tmp_path, monkeypatch)
    b = wstar_run(tmp_path / "runs", sel)
    got = refused(lambda: LDP.preflight(LDP.parse_args(mt_argv(b, tmp_path)), environ={}), "refused")
    assert "DATA-09" in got or "P2_METHODS_LOCK" in got


@nc("a MAIN_TEST FT_P2_WSTAR launch without the P2 lock binding (only launch_device's DATA-09 hook)")
def test_nc_test_without_p2_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(LDP, "p2_device_test_gate", lambda args: None)
    check_test_refusals(tmp_path, monkeypatch)

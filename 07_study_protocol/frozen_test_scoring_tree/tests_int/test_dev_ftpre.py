"""DEV-FTPRE (PREG1_AMENDMENT_06K §2, 06L-C §C7 / §C16) — the FT_PRE device arm of v4int.launch_device and the no-FT
UNSEEN_VAL references of v4int.reeval (CPU, tiny synthetic models / manifests of test_int04_device; no real data).

FT_PRE proves: the base is theta_pre ONLY, loaded from the PRE run's PRETRAIN_FOR_FL_INIT.json (sha-checked, the loaded
state = the checkpoint's state_digest); every MAIN-trained / non-PRE / tampered checkpoint refuses; --base / --base-kind /
--select / UNSEEN_VAL refuse; the selection is the registered FT_C one (contracts/device/SELECTION_FT_C.json, the
default; a real run refuses any other file); the FT_PRE cell equals the FT_C procedure on the same start bitwise, its
rows are labelled FT_PRE with the base identity; an FT_C run from theta_pre refuses; every pre-existing LO / FT_C / FT_FA
spec digest is unchanged. No-FT: C_SMALL (central) and FA (cold-start FL) are the only UNSEEN_VAL references, their rows
are DESCRIPTIVE, the record carries no_ft_reference, and tools/ana_block_index.py admits the written directory.
NCs (test_nc_*): the FT_PRE argument guard removed; the MAIN-users guard emptied; an FT_C run from theta_pre accepted;
the UNSEEN_VAL arm restriction removed; UNSEEN_VAL rows without the DESCRIPTIVE labels.
FIX-DEV-FTPRE (REV-DEV-FTPRE M1 / M3 / M5): the theta_pre refusal and the DESCRIPTIVE labelling are tested through the
helpers run() takes its start / rows / record from (launch_device.resolve_start, reeval.label_rows / add_no_ft_record),
so the two NCs remove the real check inside that wiring; the registered SELECTION_FT_C.json is pinned by its DL 231 file
sha256 (NC: the pin trusted by path); PRETRAIN_FOR_FL_INIT.json is written once and never rewritten.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import v4int.launch_device as D
import v4int.launch_fl as F
from fl.numerics import state_digest
from fl.synthetic import make_tiny_adapter
from intkit import assert_raises, nc
from test_int04_device import K, PEAK, _frozen, _plans, _world
from v4int import pretrain_small as PS
from v4int import reeval as RV

V4 = Path(__file__).resolve().parents[2]
VAR, REC = "SASREC_D64_B2", "SMALL_SASREC_D64_B2"
PRE_ID = "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026"
SEL_C = {"arm": "FT_C", "cell": "FT_C_f0.3_P2", "selection_role": "UNSEEN_VAL", "self_sha256": "s" * 64}
# the LO / FT_C / FT_FA spec digests of the pre-DEV-FTPRE launcher (MEASURED on launch_device 48eee710 before the edit)
PRE_DEVFTPRE_DIGESTS = {
    ("LO", "MAIN"): "89d676d214b78f5e254d94744819d6f13d0e3e0b22c3ef8ce29f9a4cc6295ba5",
    ("LO", "UNSEEN_VAL"): "43d839a486072387dbab29445f586071580ffe797e618f99b6f6773cfc74d7c7",
    ("FT_C", "MAIN"): "2b0b377d22b304f82b93837b6088f00a2d8da2fcbfb57a2ced5f0c90cc4cb7a0",
    ("FT_C", "UNSEEN_VAL"): "a3c37589020206751a703c09f46bd2dcec098caf1cb2e3560bc37ce0a7c030d1",
    ("FT_FA", "UNSEEN_HOLDOUT"): "f03ae74c6039727788ab5a0d5e637d220594c5e531a75d55dfd8c52a4a019720"}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_pre(root: Path, *, seed=77, run_id=PRE_ID, run_label="PRETRAIN_FOR_FL", manifest="PRETRAIN_TRAIN_5_100_NONMAIN",
             kind="endpoint_6.0", variant=VAR, rec_sha=None):
    """A PRE-run-shaped v4train endpoint_6.0 checkpoint of the tiny K-class model + its PRETRAIN_FOR_FL_INIT.json."""
    pre = make_tiny_adapter(K=K, seed=seed, tied=False)
    ck = root / "sealed" / "checkpoints" / f"{kind}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "run_label": run_label, "family": "SASREC", "model_variant": variant, "seed": 2026,
            "state_digest": state_digest(pre.broadcast_state()), "efe": 6.0, "mark": 6.0,
            "manifest_registry": {"train_manifest": manifest, "eval_manifest": "UNEW_VAL_VALIDATION_32768"}}
    torch.save({"kind": kind, "meta": meta, "model": pre.module.state_dict()}, ck)
    rec = {"run_id": run_id, "run_label": run_label, "train_manifest": manifest,
           "fl_init": {"kind": kind, "path": f"sealed/checkpoints/{kind}.pt",
                       "sha256": rec_sha or _sha(ck), "bytes": ck.stat().st_size}}
    p = root / "PRETRAIN_FOR_FL_INIT.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p, pre


def _args(**kw):
    a = dict(arm="FT_PRE", family="SASREC", seed=2026, gpu=0, run_dir=Path("r"), history_policy="USER_HISTORY_L50",
             role="MAIN", select=False, selection=None, base=None, base_kind=None, base_sha256=None, init_from=None,
             n_shards=4, shards=None, smoke=False, smoke_clients=8, dry_run=True, allow_shared_gpu=False,
             registry_ext=None)
    a.update(kw)
    return SimpleNamespace(**a)


def _selection_file(tmp_path, arm="FT_C", cell="FT_C_f0.3_P4", name="SEL.json"):
    obj = {"format": D.SELECTION_FORMAT, "arm": arm, "seed": 2026, "selection_role": "UNSEEN_VAL", "cell": cell,
           "eval_manifest_content_sha256": "c" * 64}
    obj["self_sha256"] = D._canon(obj)
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------ digests / spec
def test_existing_spec_digests_unchanged():
    sel = {"LO": {"arm": "LO", "cell": "LO_P4"}, "FT_C": {"arm": "FT_C", "cell": "FT_C_f0.3_P4"},
           "FT_FA": {"arm": "FT_FA", "cell": "FT_FA_f0.1_P2"}}
    for (arm, role), want in PRE_DEVFTPRE_DIGESTS.items():
        kw = dict(select=True) if role == "UNSEEN_VAL" else dict(selection=sel[arm])
        got = D.spec_digest(D.device_spec(arm, seed=2026, peak_lr=0.000711, role=role, start_digest="s" * 64,
                                          fused=True, **kw))
        assert got == want, (arm, role)
    assert D.FT_PROCEDURE["FT_C"] == "FT_C" and D.FT_PROCEDURE["FT_FA"] == "FT_FA"


def test_ft_pre_spec_applies_the_ft_c_selection():
    s = D.device_spec("FT_PRE", seed=2026, peak_lr=PEAK, role="MAIN", start_digest="x", selection=SEL_C)
    assert s["arm"] == "FT_PRE" and s["selected_cell"] == "FT_PRE_f0.3_P2" and len(s["cells"]) == 1
    assert s["cells"][0] == {"cell": "FT_PRE_f0.3_P2", "passes": 2, "lr_frac": 0.3}
    assert s["ft_procedure"] == "FT_C" and s["procedure_selection_cell"] == "FT_C_f0.3_P2"
    assert s["procedure_selection_arm"] == "FT_C" and s["procedure_selection_self_sha256"] == "s" * 64
    for kw, msg in ((dict(selection=dict(SEL_C, arm="FT_PRE")), "not FT_C"),
                    (dict(selection=dict(SEL_C, cell="LO_P2")), "not a cell of the FT_C grid"),
                    (dict(select=True, role="UNSEEN_VAL"), "--select refused"),
                    (dict(selection=SEL_C, role="UNSEEN_VAL"), "never run on the UNSEEN_VAL"),
                    (dict(), "needs --selection")):
        args = dict(seed=2026, peak_lr=PEAK, role="MAIN", start_digest="x")
        args.update(kw)
        with pytest.raises(D.LaunchRefused, match=msg):
            D.device_spec("FT_PRE", **args)


def test_registered_ft_c_selection_is_the_default_and_verifies():
    reg = V4 / "contracts" / "device" / "SELECTION_FT_C.json"
    assert D.FT_PRE_SELECTION == reg
    assert D.FT_PRE_SELECTION_SHA256 == _sha(reg)                   # FIX-DEV-FTPRE M5: the DL 231 file sha256 (e47965d2)
    obj = D.load_selection(reg)                                     # the registered file (DL 231) verifies
    assert obj["arm"] == "FT_C" and obj["cell"] == "FT_C_f0.3_P4"
    rec = D.check_ft_pre_selection_file(reg, smoke=False)
    assert rec["is_registered_file"] and rec["file_sha256"] == _sha(reg) == rec["registered_sha256"]
    s = D.device_spec("FT_PRE", seed=2026, peak_lr=PEAK, role="MAIN", start_digest="x", selection=obj)
    assert s["selected_cell"] == "FT_PRE_f0.3_P4" and s["procedure_selection_cell"] == "FT_C_f0.3_P4"


# ------------------------------------------------------------------------------------------------ preflight refusals
def test_ft_pre_preflight_refusals(tmp_path):
    p, _ = make_pre(tmp_path / "pre")
    gate = tmp_path / "G1.json"
    gate.write_text("{}", encoding="utf-8")
    pre = D.preflight(_args(init_from=p), environ={}, gate_path=gate)          # default = the registered selection
    assert pre["ft_pre"]["selection"]["is_registered_file"] and pre["ft_pre"]["init_record_sha256"] == _sha(p)
    other = _selection_file(tmp_path)
    cases = [(dict(), "needs --init-from"), (dict(init_from=tmp_path / "nope.json"), "does not exist"),
             (dict(init_from=p, base=tmp_path / "b.pt"), "--base / --base-kind are refused"),
             (dict(init_from=p, base_kind="central"), "--base / --base-kind are refused"),
             (dict(init_from=p, select=True, role="UNSEEN_VAL"), "--select"),
             (dict(init_from=p, role="UNSEEN_VAL"), "never run on the UNSEEN_VAL"),
             (dict(init_from=p, selection=other), "registered"),                    # a real run: the registered file
             (dict(init_from=p, selection=_selection_file(tmp_path, arm="FT_FA", cell="FT_FA_f0.3_P4",
                                                          name="FA.json")), "registered FT_C selection"),
             (dict(arm="FT_C", init_from=p, base=tmp_path / "b.pt", base_kind="central",
                   selection=other), "--init-from is for --arm FT_PRE only")]
    for kw, msg in cases:
        with pytest.raises(D.LaunchRefused, match=msg):
            D.preflight(_args(**kw), environ={}, gate_path=gate)
    smoke = D.preflight(_args(init_from=p, selection=other, smoke=True), environ={}, gate_path=gate)
    assert smoke["ft_pre"]["selection"]["is_registered_file"] is False                # a smoke may use another file
    assert D.parse_args(["--arm", "FT_PRE", "--seed", "2026", "--gpu", "0", "--run-dir", "r", "--history-policy",
                         "USER_HISTORY_L50", "--init-from", str(p)]).init_from == p


# ------------------------------------------------------------------------------------------------ theta_pre base
def test_load_pre_base_is_exactly_theta_pre(tmp_path):
    p, pre = make_pre(tmp_path / "pre")
    worker = make_tiny_adapter(K=K, seed=1, tied=False)
    theta0 = state_digest(worker.broadcast_state())
    b = D.load_pre_base(worker, p, model_variant=VAR)
    assert b["digest"] == state_digest(pre.broadcast_state()) == state_digest(b["state"]) != theta0
    assert b["kind"] == "PRE" and b["run_id"] == PRE_ID and b["file_sha256"] == json.loads(p.read_text())["fl_init"][
        "sha256"] and b["init_record_sha256"] == _sha(p)
    F.check_init_identity(b["init_from"])
    D.refuse_theta_pre_base("FT_PRE", {k: v for k, v in b.items() if k != "state"})
    lab = D.ft_pre_row_labels(b, {"self_sha256": "q" * 64})
    assert lab["base_source"] == "PRE" and lab["base_train_manifest_id"] == "PRETRAIN_TRAIN_5_100_NONMAIN"
    assert lab["base_checkpoint_sha256"] == b["file_sha256"] and lab["base_init_record_sha256"] == _sha(p)
    D.load_pre_base(make_tiny_adapter(K=K, seed=1, tied=False), p, model_variant=VAR, want_sha256=b["file_sha256"])
    assert_raises(D.LaunchRefused, D.load_pre_base, make_tiny_adapter(K=K, seed=1, tied=False), p,
                  model_variant=VAR, want_sha256="0" * 64)


def test_load_pre_base_refusals(tmp_path):
    cases = {"sha": dict(rec_sha="0" * 64), "kind": dict(kind="best"), "label": dict(run_label="PRODUCTION"),
             "main": dict(manifest="MAIN_TRAIN_5_100_131072"), "full": dict(manifest="FULL_TRAIN_5_100_TIEBREAK"),
             "calib": dict(manifest="CALIB_TRAIN_5_100_32768"), "c_full": dict(run_id="C_FULL_02B_SASREC_s2026"),
             "c_big": dict(run_id="C_BIG_SASREC_s2026"), "c_small": dict(run_id="C_SMALL_SASREC_D64_B2_s2026"),
             "variant": dict(variant="SASREC_D256_B3")}
    for name, kw in cases.items():
        p, _ = make_pre(tmp_path / name, **kw)
        assert_raises(D.LaunchRefused, D.load_pre_base, make_tiny_adapter(K=K, seed=1, tied=False), p,
                      model_variant=VAR)
    p, _ = make_pre(tmp_path / "tamper")
    ck = tmp_path / "tamper" / "sealed" / "checkpoints" / "endpoint_6.0.pt"
    ck.write_bytes(ck.read_bytes() + b"\0")
    assert_raises(D.LaunchRefused, D.load_pre_base, make_tiny_adapter(K=K, seed=1, tied=False), p, model_variant=VAR)


def test_theta_pre_base_is_ft_pre_both_ways():
    for arm in ("FT_C", "FT_FA"):
        with pytest.raises(D.LaunchRefused, match="is FT_PRE"):
            D.refuse_theta_pre_base(arm, {"kind": "central", "run_id": PRE_ID})
        D.refuse_theta_pre_base(arm, {"kind": "central", "run_id": "02B_SASREC_MAIN_s2026"})
    D.refuse_theta_pre_base("LO", {"kind": "theta0"})
    for bad in ({"kind": "central", "run_id": PRE_ID}, {"kind": "PRE", "run_id": "C_SMALL_s2026",
                                                        "init_from": {"run_id": "C_SMALL_s2026"}}):
        assert_raises(D.LaunchRefused, D.refuse_theta_pre_base, "FT_PRE", bad)


def _replaced_registered_selection(tmp_path, monkeypatch):
    """FIX-DEV-FTPRE M5: the contracts copy replaced by another verified FT_C selection (e.g. a seed-2027 one)."""
    other = _selection_file(tmp_path, name="SELECTION_FT_C.json")
    monkeypatch.setattr(D, "FT_PRE_SELECTION", other)
    return other


def test_ft_pre_selection_pinned_by_sha256(tmp_path, monkeypatch):
    p, _ = make_pre(tmp_path / "pre")
    gate = tmp_path / "G1.json"
    gate.write_text("{}", encoding="utf-8")
    other = _replaced_registered_selection(tmp_path, monkeypatch)
    with pytest.raises(D.LaunchRefused, match="pinned sha256"):     # the path's bytes are no longer trusted
        D.check_ft_pre_selection_file(other, smoke=False)
    with pytest.raises(D.LaunchRefused, match="pinned sha256"):     # the default --selection of a real preflight
        D.preflight(_args(init_from=p), environ={}, gate_path=gate)
    rec = D.check_ft_pre_selection_file(other, smoke=True)          # smoke: unchanged (any verified FT_C file)
    assert rec["is_registered_file"] and rec["file_sha256"] == _sha(other) != D.FT_PRE_SELECTION_SHA256
    assert D.preflight(_args(init_from=p, smoke=True), environ={}, gate_path=gate)["ft_pre"]


def _start_args(**kw):
    return _args(**dict(dict(dry_run=False), **kw))


def test_resolve_start_is_the_run_start_and_refuses_theta_pre_for_ft_c(tmp_path):
    """launch_device.run takes its start ONLY from resolve_start: LO = theta0 of the harness, FT_PRE = theta_pre from
    the init record, FT_C / FT_FA = --base; an FT_C / FT_FA --base that is the theta_pre checkpoint refuses there."""
    p, pre = make_pre(tmp_path / "pre")
    ck = tmp_path / "pre" / "sealed" / "checkpoints" / "endpoint_6.0.pt"
    st, base, rec = D.resolve_start(_start_args(init_from=p), None, make_tiny_adapter(K=K, seed=1, tied=False),
                                    theta0="t0", model_variant=VAR)
    assert state_digest(st) == state_digest(pre.broadcast_state()) and base["kind"] == "PRE"
    assert rec["kind"] == "PRE" and rec["init_record_sha256"] == _sha(p) and rec["init_from"]["run_id"] == PRE_ID
    assert "state" not in rec and D.ft_pre_row_labels(rec, None)["base_init_record_sha256"] == _sha(p)
    h = SimpleNamespace(adapter=make_tiny_adapter(K=K, seed=5, tied=False), built=SimpleNamespace(init_sha256="i"))
    st, base, rec = D.resolve_start(_start_args(arm="LO"), h, None, theta0="t0", model_variant=VAR)
    assert base is None and rec == {"kind": "theta0", "init_sha256": "i", "theta0_registered": "t0"}
    assert state_digest(st) == state_digest(h.adapter.broadcast_state())
    make_pre(tmp_path / "c", run_id="02B_SASREC_MAIN_s2026")                         # a non-theta_pre central base
    cq = tmp_path / "c" / "sealed" / "checkpoints" / "endpoint_6.0.pt"
    st, base, rec = D.resolve_start(_start_args(arm="FT_C", base=cq, base_kind="central"), None,
                                    make_tiny_adapter(K=K, seed=1, tied=False), theta0="t0", model_variant=VAR)
    assert rec["kind"] == "central" and rec["run_id"] == "02B_SASREC_MAIN_s2026" and rec["file_sha256"] == _sha(cq)
    for arm in ("FT_C", "FT_FA"):                                   # theta_pre as a --base: refused in the wiring
        with pytest.raises(D.LaunchRefused, match="is FT_PRE"):
            D.resolve_start(_start_args(arm=arm, base=ck, base_kind="central"), None,
                            make_tiny_adapter(K=K, seed=1, tied=False), theta0="t0", model_variant=VAR)


# ------------------------------------------------------------------------------------------------ PRETRAIN_FOR_FL_INIT (M3)
def _pre_run(tmp_path, data=b"endpoint-weights"):
    ck = tmp_path / "sealed" / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "endpoint_6.0.pt").write_bytes(data)
    return ck / "endpoint_6.0.pt"


def test_fl_init_record_is_written_once(tmp_path, monkeypatch):
    """FIX-DEV-FTPRE M3: a second call (the --fl-init-only path / a resume leg) returns the existing record unchanged,
    with no utc churn (the file sha256 = the init_record_sha256 FT_PRE and the PRE re-evaluation compare)."""
    _pre_run(tmp_path)
    rec = PS.fl_init_record(tmp_path, PRE_ID)
    out = tmp_path / PS.FL_INIT_FILE
    before = out.read_bytes()
    monkeypatch.setattr(PS.time, "strftime", lambda *a: "2099-01-01T00:00:00Z")     # a later call: another utc
    again = PS.fl_init_record(tmp_path, PRE_ID)
    assert again == rec == json.loads(before) and out.read_bytes() == before
    assert again["utc"] != "2099-01-01T00:00:00Z"


def test_fl_init_record_refuses_another_theta_pre(tmp_path):
    ck = _pre_run(tmp_path)
    PS.fl_init_record(tmp_path, PRE_ID)
    out = tmp_path / PS.FL_INIT_FILE
    before = out.read_bytes()
    with pytest.raises(PS.PretrainRefused, match="never rewritten"):                 # another run id
        PS.fl_init_record(tmp_path, PRE_ID.replace("s2026", "s2027"))
    ck.write_bytes(b"other-endpoint-weights")                                          # another endpoint_6.0
    with pytest.raises(PS.PretrainRefused, match="never rewritten"):
        PS.fl_init_record(tmp_path, PRE_ID)
    assert out.read_bytes() == before
    out.write_text("{not json", encoding="utf-8")
    with pytest.raises(PS.PretrainRefused, match="unreadable"):
        PS.fl_init_record(tmp_path, PRE_ID)


# ------------------------------------------------------------------------------------------------ FT_PRE = FT_C procedure
def _row_fn(spec):
    from v4eval.rows import COMPARISON_KEY
    kb = {f: "x" for f in COMPARISON_KEY}
    kb.update(aggregation="macro", target_denominator="E2E")
    return lambda res, cell, selected: D.device_rows(res, kb, {"run_label": "PRODUCTION"}, spec=spec, cell=cell,
                                                     selected=selected, down_bytes=7,
                                                     overrides={"base_source": "PRE"} if spec["arm"] == "FT_PRE"
                                                     else None)


def test_ft_pre_cell_is_the_ft_c_procedure_on_theta_pre(tmp_path):
    w = _world()
    p, _ = make_pre(tmp_path / "pre")
    base = D.load_pre_base(make_tiny_adapter(K=K, seed=1, tied=False), p, model_variant=VAR)
    start = base["state"]
    out = {}
    for arm in ("FT_PRE", "FT_C"):
        spec = D.device_spec(arm, seed=2026, peak_lr=PEAK, role="MAIN", start_digest=state_digest(start),
                             selection=SEL_C)
        out[arm] = D.execute(spec=spec, run_dir=tmp_path / arm, worker=make_tiny_adapter(K=K, seed=3, tied=False),
                             start_state=start, plans=_plans(w), get_support=lambda q: w.sup[q.key],
                             get_eval_batch=w.get_eval_batch, manifest=w.m, frozen=_frozen(), row_fn=_row_fn(spec),
                             say=lambda *_: None, n_shards=2)
    a = np.load(tmp_path / "FT_PRE" / "cells" / "FT_PRE_f0.3_P2.ranks_int32.npy")
    b = np.load(tmp_path / "FT_C" / "cells" / "FT_C_f0.3_P2.ranks_int32.npy")
    assert np.array_equal(a, b) and out["FT_PRE"]["values"]["FT_PRE_f0.3_P2"] == out["FT_C"]["values"]["FT_C_f0.3_P2"]
    summ = json.loads((tmp_path / "FT_PRE" / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert summ["arm"] == "FT_PRE" and summ["selected_cell"] == "FT_PRE_f0.3_P2" and summ["status"] == "DONE"
    rows = json.loads((tmp_path / "FT_PRE" / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert rows and {r["method"] for r in rows} == {"FT_PRE"} and {r["method_code"] for r in rows} == {"FT_PRE"}
    assert {r["ft_procedure"] for r in rows} == {"FT_C"} and {r["procedure_selection_cell"] for r in rows} == \
        {"FT_C_f0.3_P2"} and {r["device_selected"] for r in rows} == {True}
    assert {r["base_source"] for r in rows} == {"PRE"} and {r["comparison_status"] for r in rows} == \
        {"DOWNSTREAM_RETENTION"}
    rec = json.loads(next((tmp_path / "FT_PRE" / "shards").glob("shard_*.json")).read_text(encoding="utf-8"))
    assert all(t["kind"] == "FT" for r in rec["records"] for t in r["trajectories"])


# ------------------------------------------------------------------------------------------------ no-FT UNSEEN_VAL (reeval)
def _central(tmp_path, run_id="C_SMALL_SASREC_D64_B2_s2026", cfg=None, kb=None):
    run = tmp_path / "run"
    cp = run / "sealed" / "checkpoints" / "best.pt"
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b"synthetic-checkpoint-for-preflight")
    start = {"run_id": run_id, "config": dict({"family": "SASREC", "model_variant": VAR, "recipe_id": REC,
                                               "seed": 2026}, **(cfg or {})),
             "key_base": dict({"split": "VALIDATION", "run_label": "PRODUCTION"}, **(kb or {}))}
    (run / "RUN_START.json").write_text(json.dumps(start), encoding="utf-8")
    return cp


def _fl_run(tmp_path, arm="FA", init_from=None):
    from fl.checkpoint import atomic_save
    run, rid = tmp_path / "fl", f"FL_{arm}_SMALL_s2026"
    run.mkdir(parents=True, exist_ok=True)
    spec = {"cfg": {"seed": 2026}, **({"init_from": init_from} if init_from else {})}
    launch = {"arm": arm, "run_id": rid, "arm_spec_sha256": "S", "fl_config_digest": "C", "label": "PRODUCTION",
              "fl05": {"role": "MAIN", "lr_mult": 1.0}, "arm_spec": spec,
              "small_recipe": {"model_variant": VAR, "recipe_id": REC, "recipe_sha256": "R"}}
    (run / "LAUNCH_0.json").write_text(json.dumps(launch), encoding="utf-8")
    cp = run / "ckpt" / rid / "best.pt"
    atomic_save({"kind": "weights", "run_id": rid, "config_digest": "C", "round": 5, "efe": 3.0, "phase": "MAIN",
                 "theta": make_tiny_adapter(seed=1).broadcast_state()}, cp)
    return cp


def _rv(cp, tmp_path, *, kind="central", arm="C_SMALL", role="UNSEEN_VAL", variant=VAR, recipe=REC):
    return RV.parse_args(["--ckpt", str(cp), "--ckpt-kind", kind, "--expect-sha256", _sha(cp), "--variant", variant,
                          "--recipe-id", recipe, "--seed", "2026", "--role", role, "--arm-label", arm,
                          "--selection-rule", "PRACTICAL_BEST", "--checkpoint-id", "x:best", "--gpu", "0",
                          "--out-dir", str(tmp_path / "out"), "--history-policy", "USER_HISTORY_L50", "--dry-run"])


def test_no_ft_references_dry_run(tmp_path):
    got = RV.run(_rv(_central(tmp_path / "c"), tmp_path / "c"))
    assert got["dry_run"] and got["no_ft_reference"]["comparison_status"] == "DESCRIPTIVE"
    assert got["no_ft_reference"]["no_ft_reference_arm"] == "C_SMALL" and got["no_ft_reference"]["planned"]
    got = RV.run(_rv(_fl_run(tmp_path / "f"), tmp_path / "f", kind="fl", arm="FA"))
    assert got["dry_run"] and got["no_ft_reference"]["no_ft_reference_arm"] == "FA"
    main = RV.run(_rv(_central(tmp_path / "m"), tmp_path / "m", role="MAIN"))
    assert "no_ft_reference" not in main                                   # every other role: unchanged


def test_no_ft_refusals(tmp_path):
    cp = _central(tmp_path / "c")
    for kw, msg in ((dict(arm="C_BIG"), "only the no-FT references"), (dict(arm="FT_C"), "only the no-FT"),
                    (dict(arm="PRE"), "not the registered PRE"),   # 06O: PRE admitted, this C_SMALL run is not PRE
                    (dict(arm="FA"), "only the no-FT"),
                    (dict(arm="C_SMALL", variant="SASREC_D256_B3", recipe="02B_SASREC"), "only the no-FT")):
        with pytest.raises(RV.ReevalRefused, match=msg):
            RV.run(_rv(cp, tmp_path / "c", **kw))
    with pytest.raises(RV.ReevalRefused, match="only the no-FT"):                       # C_SMALL as an FL checkpoint
        RV.run(_rv(_fl_run(tmp_path / "f0"), tmp_path / "f0", kind="fl", arm="C_SMALL"))
    with pytest.raises(RV.ReevalRefused, match="cold-start FA"):                         # FA_Q8 relabelled as FA
        RV.run(_rv(_fl_run(tmp_path / "f1", arm="FA_Q8"), tmp_path / "f1", kind="fl", arm="FA"))
    with pytest.raises(RV.ReevalRefused, match="cold-start FA"):                         # a warm FA (FA_WARM) as FA
        RV.run(_rv(_fl_run(tmp_path / "f2", init_from={"kind": "PRETRAIN_FOR_FL"}), tmp_path / "f2", kind="fl",
                   arm="FA"))
    with pytest.raises(RV.ReevalRefused, match="MAIN-pair central run"):
        RV.check_no_ft_source(SimpleNamespace(role="UNSEEN_VAL", arm_label="C_SMALL"), {"central_source": "PRE"})


def _label_args(role="UNSEEN_VAL", arm="C_SMALL"):
    return SimpleNamespace(role=role, arm_label=arm, selection_rule="PRACTICAL_BEST", checkpoint_id="C_SMALL_s2026:best",
                           ckpt_kind="central")


def _labelled(role="UNSEEN_VAL"):
    rows = RV.label_rows([{"method": "C"}], _label_args(role), checkpoint_sha256="k", source_run_id="x", efe=1.0,
                         extra={"exposures": 3})
    return rows, RV.add_no_ft_record({"role": role}, _label_args(role))


def test_reeval_labels_rows_and_record_through_run_helpers():
    """reeval.run labels its rows with label_rows and its record with add_no_ft_record: DESCRIPTIVE + no_ft_reference on
    UNSEEN_VAL; every other role keeps comparison_status REEVAL and an unchanged record."""
    rows, rec = _labelled()
    assert rows[0]["comparison_status"] == "DESCRIPTIVE" and rows[0]["no_ft_reference"] is True
    assert rows[0]["row_status"] == "REEVAL" and rows[0]["exposures"] == 3 and rows[0]["method"] == "C_SMALL"
    assert rec == {"role": "UNSEEN_VAL", "no_ft_reference": RV.no_ft_labels("UNSEEN_VAL", "C_SMALL")}
    rows, rec = _labelled("MAIN")
    assert rows[0]["comparison_status"] == "REEVAL" and "no_ft_reference" not in rows[0] and rec == {"role": "MAIN"}


def test_no_ft_rows_end_to_end_and_index_admission(tmp_path):
    """C_SMALL on a synthetic UNSEEN_VAL manifest through reeval's own evaluate_role / relabel / write_outputs: the rows
    are the frozen T3 evaluation, labelled DESCRIPTIVE, and tools/ana_block_index.py admits the directory as the
    registered no-FT reference on role UNSEEN_VAL."""
    from v4eval.evaluate import evaluate
    from v4eval.predict import predict_adapter
    from v4eval.rows import COMPARISON_KEY
    w = _world()
    fz = _frozen()
    ad = make_tiny_adapter(K=K, seed=4, tied=False)
    ad.module.eval()
    kb = {f: "x" for f in COMPARISON_KEY}
    rep = SimpleNamespace(manifest=w.m, batches=w.full_batches, manifest_id=w.m.manifest_id)
    ev = "UNEW_VAL_VALIDATION_32768"
    over = {"manifest_role": "UNSEEN_VAL", "train_manifest_id": "UNEW_VAL_SUPPORT_32768", "eval_manifest_id": ev}
    h = RV.central_harness(ad, frozen=fz, device=torch.device("cpu"), key_base=kb, labels={"run_label": "PRODUCTION"},
                           seed=2026, evaluator_sha="e", evalp=rep)
    v, rows, ranks, res = RV.evaluate_role(h, rep, kb, over)
    rows = RV.label_rows(rows, _label_args(), checkpoint_sha256="k" * 64, source_run_id="C_SMALL_s2026", efe=5.5,
                         extra={})                                              # run()'s own labelling step
    art = predict_adapter(ad, w.full_batches(fz.eval_batch), w.m, config=fz, model_sha256="m")
    ref = evaluate(w.m, art, config=fz, evaluator_code_sha256="e")
    assert v == ref["E2E"]["macro"]["mrr@20"] and np.array_equal(ranks, art.ranks)
    assert {r["comparison_status"] for r in rows} == {"DESCRIPTIVE"} and {r["no_ft_reference"] for r in rows} == {True}
    assert {r["row_status"] for r in rows} == {"REEVAL"} and {r["manifest_role"] for r in rows} == {"UNSEEN_VAL"}
    assert {r["method"] for r in rows} == {"C_SMALL"} and {r["eval_manifest_id"] for r in rows} == {ev}
    assert RV.no_ft_labels("MAIN", "C_SMALL") == {}
    out = tmp_path / "REEVAL_A40" / "C_SMALL_UNSEEN_VAL_s2026_PB"
    rec = RV.write_outputs(out, tag="UNSEEN_VAL", run_id="REEVAL_C_SMALL_UNSEEN_VAL_s2026_PRACTICAL_BEST",
                           role="UNSEEN_VAL", rows=rows, result=res, ranks=ranks, record={
                               "format": RV.RECORD_FORMAT, "arm_label": "C_SMALL", "seed": 2026, "role": "UNSEEN_VAL",
                               "selection_rule": "PRACTICAL_BEST", "checkpoint_id": "C_SMALL_s2026:best",
                               "checkpoint_sha256": "k" * 64, "device_class": "CPU-SYN", "eval_manifest_id": ev,
                               "no_ft_reference": RV.add_no_ft_record({}, _label_args())["no_ft_reference"]})
    assert rec["status"] == "COMPLETE"
    tools = V4 / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("_devftpre_ana_block_index", tools / "ana_block_index.py")
    BI = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = BI
    spec.loader.exec_module(BI)
    lines = []
    rc = BI.main(["--split", "VALIDATION", "--device-class", "CPU-SYN", "--reeval", str(tmp_path / "REEVAL_A40"),
                  "--out", str(tmp_path / "idx")], say=lines.append)
    assert rc == 0, lines
    idx = json.loads((tmp_path / "idx" / "BLOCK_INDEX.json").read_text(encoding="utf-8"))
    assert [(e["arm"], e["role"], e["rule"]) for e in idx["entries"]] == [("C_SMALL", "UNSEEN_VAL", "PRACTICAL_BEST")]


# ------------------------------------------------------------------------------------------------ negative controls
@nc("an FT_PRE argument guard that is not called accepts --base for FT_PRE")
def test_nc_ft_pre_accepts_base(tmp_path, monkeypatch):
    p, _ = make_pre(tmp_path / "pre")
    gate = tmp_path / "G1.json"
    gate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(D, "check_ft_pre_args", lambda args: {})                     # MUTANT
    assert_raises(D.LaunchRefused, D.preflight, _args(init_from=p, base=tmp_path / "b.pt", base_kind="central",
                                                      selection=D.FT_PRE_SELECTION), environ={}, gate_path=gate)


@nc("a theta_pre guard without the MAIN checks accepts a C_BIG-like (MAIN-trained) base for FT_PRE")
def test_nc_ft_pre_main_trained_base(tmp_path, monkeypatch):
    p, _ = make_pre(tmp_path / "cbig", manifest="MAIN_TRAIN_5_100_131072")
    monkeypatch.setattr(F, "MAIN_TRAIN_MANIFESTS", ())                               # MUTANT (deny-list emptied,
    monkeypatch.setattr(F, "MAIN_RUN_MARKERS", ())                                   # allow-list pointed at MAIN)
    monkeypatch.setitem(F.PRETRAIN, "train_manifest", "MAIN_TRAIN_5_100_131072")
    assert_raises(D.LaunchRefused, D.load_pre_base, make_tiny_adapter(K=K, seed=1, tied=False), p, model_variant=VAR)


@nc("an FT_C launch whose base is theta_pre is accepted (a device run from theta_pre must be FT_PRE)")
def test_nc_ft_c_from_theta_pre(tmp_path, monkeypatch):
    make_pre(tmp_path / "pre")
    ck = tmp_path / "pre" / "sealed" / "checkpoints" / "endpoint_6.0.pt"
    monkeypatch.setattr(D, "refuse_theta_pre_base", lambda arm, rec: None)           # MUTANT (the check run() calls
    assert_raises(D.LaunchRefused, D.resolve_start,                                  # through resolve_start removed)
                  _start_args(arm="FT_C", base=ck, base_kind="central"), None,
                  make_tiny_adapter(K=K, seed=1, tied=False), theta0="t0", model_variant=VAR)


@nc("reeval without the 06L-C §C16 arm restriction evaluates C_BIG on the UNSEEN_VAL selection panel")
def test_nc_no_ft_restriction_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(RV, "check_no_ft_args", lambda args: None)                   # MUTANT (both the argument
    monkeypatch.setattr(RV, "check_no_ft_source", lambda args, src: None)            # and the source checks)
    assert_raises(RV.ReevalRefused, RV.run, _rv(_central(tmp_path / "c"), tmp_path / "c", arm="C_BIG"))


@nc("UNSEEN_VAL no-FT rows without the DESCRIPTIVE labels")
def test_nc_no_ft_rows_not_descriptive(monkeypatch):
    monkeypatch.setattr(RV, "no_ft_labels", lambda role, arm: {})                    # MUTANT (the labels run() adds
    rows, rec = _labelled()                                                          # through label_rows removed)
    assert rows[0]["comparison_status"] == "DESCRIPTIVE" and rows[0].get("no_ft_reference") is True
    assert rec.get("no_ft_reference")


@nc("the registered FT_C selection trusted by path: a replaced contracts copy is accepted by a real FT_PRE run")
def test_nc_ft_pre_selection_trusted_by_path(tmp_path, monkeypatch):
    other = _replaced_registered_selection(tmp_path, monkeypatch)
    monkeypatch.setattr(D, "FT_PRE_SELECTION_SHA256", _sha(other))                   # MUTANT (pin = the path's bytes)
    assert_raises(D.LaunchRefused, D.check_ft_pre_selection_file, other, smoke=False)

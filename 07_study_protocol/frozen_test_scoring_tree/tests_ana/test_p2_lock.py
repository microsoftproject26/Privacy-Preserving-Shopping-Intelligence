"""P2-DEVLOCK piece B - tools/p2_lock.py, the P2-00 lock tool (synthetic run dirs of value-free records + checkpoint
bytes; the REAL contracts/PREG2.md, P2_CONTRASTS_v3, ALL_METHODS_LOCK and register tool, all read-only; outputs under
pytest tmp only - company/gates/P2_METHODS_LOCK.json is never written here).

  * the lock is written once (O_EXCL; a second write refuses, bytes unchanged; a dry run writes nothing);
  * its self_sha256 verifies with retention_report._self_ok (utf8-compact rule) and is accepted by both readers:
    tools/p2_report.check_p2_lock and v4int/reeval_p2.p2_lock_arm (every sealed non-device entry; FT_P2_WSTAR's base =
    the sealed W* best.pt; a declared-not-run arm refuses); S5 entries bound to ALL_METHODS_LOCK (FT_PRE null);
  * refusals (nothing written): a sealed run not DONE_MAIN (FL, C1), C2 still improving, an expected sha mismatch, a
    selection binding mismatch (W* FINAL, C1), a missing checkpoint, an uncovered / doubly listed (arm, seed), a device
    arm without its W*, init_from != theta_pre, a PREG2 sha mismatch, a Phase-1 pin mismatch, a device base pin mismatch;
  * the register-entry helper gives exactly the register tool's FIELDS, names P2_METHODS_LOCK, is appended by the
    unchanged tool and accepted by p2_report.check_locked_evaluations;
  * the freeze is accepted by p2_report.check_p2_freeze and refuses a retention_report sha != 9a3b2c2a;
  * the code-tree digest is deterministic, skips __pycache__, changes with any file, needs v4int/p2_select.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import retention_report as R
import p2_lock as PL
import p2_report as P2
from v4int import reeval_p2 as RP

PB, CB = "PRACTICAL_BEST", "CONTROLLED_BUDGET"
LABEL = "FEDADAM_eta0.001_c0.1"
DNR = [{"arm": "P2_WSTAR_FA1024", "seed": None, "reason": "declared not run at lock time (06D §8): test"},
       {"arm": "P2_WSTAR_DP8", "seed": None, "reason": "declared not run at lock time (06D §8): test"},
       {"arm": "P2_WSTAR_DP1", "seed": 2026, "reason": "declared not run at lock time (06D §8): test"},
       {"arm": "P2_WSTAR_DPFTRL8", "seed": None, "reason": "Part B unfinished at lock time (06D §8): test"},
       {"arm": "P2_WSTAR_SECAGG", "seed": None, "reason": "Part B unfinished at lock time (06D §8): test"}]


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _w(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _ck(p: Path, tag: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(hashlib.sha256(tag.encode()).digest() * 8)


def make_world(root: Path) -> dict:
    """Selection files, a code tree, finished MAIN runs (W* x2, C1 x4, C2, one FT_P2_WSTAR device run) and the plan."""
    fin = {"format": PL.FINAL_FORMAT, "stage": "FINAL", "winner": LABEL, "winner_recipe": {"server_opt": "fedadam"}}
    fin["self_sha256"] = hashlib.sha256(json.dumps(fin, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    sel = {"P2_WARM_SELECTION_FINAL": _w(root / "contracts" / "fl" / "P2_WARM_SELECTION_FINAL.json", fin),
           "P2_WARM_SELECTION_A": _w(root / "contracts" / "fl" / "P2_WARM_SELECTION_A.json", {"stub": "A"}),
           "P2_C1_SELECTION": _w(root / "contracts" / "fl" / "P2_C1_SELECTION.json", {"stub": "C1"})}
    tree = root / "code_p2r1"
    (tree / "v4int").mkdir(parents=True)
    (tree / "v4int" / "p2_select.py").write_text("# the selection tool (stub)\n", encoding="utf-8")
    (tree / "v4int" / "reeval_p2.py").write_text("# stub\n", encoding="utf-8")
    runs, sealed = root / "runs", []
    for s in (2026, 2027):
        rid = f"P2_WSTAR_SMALL_s{s}"
        rd = runs / f"P2_WSTAR_s{s}"
        spec = {"arm": "P2_W", "study": "PREG2", "role": "MAIN", "cfg": {"seed": s, "method": "FA"},
                "init_from": {"checkpoint_sha256": PL.PRE_CHECKPOINT_SHA256}, "p2_label": LABEL}
        _w(rd / "STATUS.json", {"run_id": rid, "arm": "P2_W", "status": "DONE_MAIN", "utc": "x"})
        _w(rd / "ARM_SPEC.json", {"arm": "P2_W", "run_id": rid, "arm_spec_sha256": "a" * 64, "arm_spec": spec})
        _w(rd / "LAUNCH_0.json", {"arm": "P2_W", "run_id": rid, "arm_spec_sha256": "a" * 64, "label": "PRODUCTION",
                                  "fl05": {"role": "MAIN"}, "p2_selection": {"sha256": _sha(sel["P2_WARM_SELECTION_FINAL"]),
                                                                             "label": LABEL}})
        for f in ("best.pt", "endpoint_6.0.pt"):
            _ck(rd / "ckpt" / rid / f, f"{rid}/{f}")
        sealed.append({"arm": "P2_WSTAR", "seed": s, "run_dir": str(rd)})
        for arm, var in PL.C1_ARMS.items():
            rd = runs / f"{arm}_s{s}"
            rid = f"{arm}_MAIN_lr0.0001_s{s}"
            _w(rd / "RUN_START.json", {"run_id": rid, "config": {"seed": s},
                                       "p2_c1": {"identity": {"arm": arm, "variant": var, "role": "MAIN",
                                                              "init_checkpoint_sha256": PL.PRE_CHECKPOINT_SHA256},
                                                 "selection": {"sha256": _sha(sel["P2_C1_SELECTION"])}}})
            _w(rd / "STATUS.json", {"run_id": rid, "status": "DONE_MAIN"})
            for f in ("best.pt", "endpoint_6.0.pt"):
                _ck(rd / "sealed" / "checkpoints" / f, f"{rid}/{f}")
            sealed.append({"arm": arm, "seed": s, "run_dir": str(rd)})
    rd = runs / "P2_C2_CFULL_SMALL_s2026"
    rid = "P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026"
    _w(rd / "RUN_START.json", {"run_id": rid, "config": {"seed": 2026}, "key_base": {"run_label": "P2_C2_CFULL_SMALL"}})
    _w(rd / "STATUS.json", {"run_id": rid, "status": "DONE_CONTINUATION"})
    for f in ("best.pt", "endpoint_6.0.pt"):
        _ck(rd / "sealed" / "checkpoints" / f, f"{rid}/{f}")
    sealed.append({"arm": "P2_C2_CFULL_SMALL", "seed": 2026, "run_dir": str(rd)})
    wbest = runs / "P2_WSTAR_s2026" / "ckpt" / "P2_WSTAR_SMALL_s2026" / "best.pt"
    dd = runs / "FT_P2_WSTAR_s2026"
    _w(dd / "STATUS.json", {"spec_sha256": "s" * 64, "status": "DONE"})
    _w(dd / "LAUNCH_0.json", {"start": {"kind": "P2_WSTAR", "seed": 2026, "checkpoint": "best.pt",
                                        "file_sha256": _sha(wbest), "run_id": "P2_WSTAR_SMALL_s2026"}})
    sealed += [{"arm": "FT_P2_WSTAR", "seed": 2026, "run_dir": str(dd)}, {"arm": "FT_P2_WSTAR", "seed": 2027}]
    plan = {"format": PL.PLAN_FORMAT, "sealed": sealed, "declared_not_run": [dict(d) for d in DNR],
            "selection_files": {k: str(v) for k, v in sel.items()}, "code_tree": {"name": "code_p2r1", "path": str(tree)}}
    return {"root": root, "plan": plan, "sel": sel, "tree": tree, "runs": runs, "wbest": wbest}


def plan_file(w: dict) -> Path:
    return _w(w["root"] / "P2_LOCK_PLAN.json", w["plan"])


def item(w, arm, seed) -> dict:
    return next(i for i in w["plan"]["sealed"] if (i["arm"], i["seed"]) == (arm, seed))


# ------------------------------------------------------------------------------------------------ the lock
def test_lock_written_once_and_accepted_by_both_readers(tmp_path):
    w = make_world(tmp_path / "w")
    out = tmp_path / "gates" / "P2_METHODS_LOCK.json"
    said = []
    assert PL.main(["lock", "--plan", str(plan_file(w)), "--out", str(out), "--dry-run"], say=said.append) == 0
    assert not out.exists() and "nothing written" in said[-1]
    rec = PL.write_lock(plan_file(w), out, say=said.append)
    raw = out.read_bytes()
    assert rec["file_sha256"] == _sha(out) and rec["n_sealed"] == 16 and rec["n_declared_not_run"] == 5
    with pytest.raises(PL.LockRefused, match="already exists"):                      # O_EXCL: written once
        PL.write_lock(plan_file(w), out, say=said.append)
    with pytest.raises(PL.LockRefused, match="already exists"):
        PL._write_new(out, b"{}")
    assert out.read_bytes() == raw
    obj = json.loads(raw.decode("utf-8"))
    assert R._self_ok(obj) == "VERIFIED" and P2._utf8_compact_self(obj) == obj["self_sha256"]
    assert obj["format"] == "PPSI_V4_P2_METHODS_LOCK_1" and obj["draft"] is False and obj["bootstrap_seed"] == 20261004
    assert obj["preg2_sha256"] == RP.PREG2_SHA256 and obj["p2_contrasts_self_sha256"] == RP.CONTRASTS_SELF_SHA256
    got = P2.check_p2_lock(out)                                                     # the P2-12 reader
    assert got["n_sealed"] == 16 and got["n_phase1_reused"] == 10 and len(got["declared_not_run"]) == 5
    assert got["sealed"][("FT_P2_WSTAR", 2026, PB)]["base_checkpoint_sha256"] == _sha(w["wbest"])
    assert got["sealed"][("FT_P2_WSTAR", 2027, PB)]["base_checkpoint_sha256"] == \
        got["sealed"][("P2_WSTAR", 2027, PB)]["checkpoint_sha256"]
    assert got["phase1_reused"][("PRE", 2026, PB)] == RP.PRE_CHECKPOINT_SHA256 and \
        got["phase1_reused"][("FT_PRE", 2027, PB)] is None
    assert set(got["selection_files_sha256"]) == {"P2_WARM_SELECTION_FINAL", "P2_WARM_SELECTION_A", "P2_C1_SELECTION"}
    n = 0
    for e in obj["sealed_arms"]:                                                    # the P2-13 reader (one adapter)
        if e["arm"] == "FT_P2_WSTAR":
            assert "checkpoint_sha256" not in e and (e["main_run"] is None) == (e["seed"] == 2027)
            continue
        b = RP.p2_lock_arm(obj, label=e["arm"], seed=e["seed"], rule=e["rule"], checkpoint_sha256=e["checkpoint_sha256"])
        assert b["checkpoint_sha256"] == e["checkpoint_sha256"]
        n += 1
    assert n == 14
    with pytest.raises(RP.P2ReevalRefused, match="declared not run"):
        RP.p2_lock_arm(obj, label="P2_WSTAR_DP8", seed=2026, rule=CB, checkpoint_sha256="0" * 64)
    c2 = [e for e in obj["sealed_arms"] if e["arm"] == "P2_C2_CFULL_SMALL"]
    assert [e["rule"] for e in c2] == [CB, PB] and obj["code_tree"]["selection_tool"]["path"] == "v4int/p2_select.py"
    assert obj["phase1_lock"]["file_sha256"] == PL.AML_SHA256 == _sha(PL.AML_PATH)
    assert PL.main(["verify", "--lock", str(out)], say=said.append) == 0


def refusal_cases():
    def status(arm, seed, **kw):
        def f(w):
            p = Path(item(w, arm, seed)["run_dir"]) / "STATUS.json"
            _w(p, dict(json.loads(p.read_text(encoding="utf-8")), **kw))
        return f

    def launch_sel(w):
        p = Path(item(w, "P2_WSTAR", 2027)["run_dir"]) / "LAUNCH_0.json"
        o = json.loads(p.read_text(encoding="utf-8"))
        o["p2_selection"]["sha256"] = "1" * 64
        _w(p, o)

    def c1_sel(w):
        p = Path(item(w, "P2_C1_FT_CENTRAL_FROZEN", 2026)["run_dir"]) / "RUN_START.json"
        o = json.loads(p.read_text(encoding="utf-8"))
        o["p2_c1"]["selection"]["sha256"] = "2" * 64
        _w(p, o)

    def not_pre(w):
        p = Path(item(w, "P2_WSTAR", 2026)["run_dir"]) / "ARM_SPEC.json"
        o = json.loads(p.read_text(encoding="utf-8"))
        o["arm_spec"]["init_from"]["checkpoint_sha256"] = "3" * 64
        _w(p, o)

    def no_ckpt(w):
        (Path(item(w, "P2_C1_FT_CENTRAL_FULL", 2027)["run_dir"]) / "sealed" / "checkpoints" / "endpoint_6.0.pt").unlink()

    def expect(w):
        item(w, "P2_WSTAR", 2026)["expect_sha256"] = {PB: "4" * 64}

    def uncovered(w):
        w["plan"]["declared_not_run"] = [d for d in w["plan"]["declared_not_run"] if d["arm"] != "P2_WSTAR_DP1"]

    def both(w):
        w["plan"]["declared_not_run"].append({"arm": "P2_WSTAR", "seed": 2027, "reason": "test"})

    def no_wstar(w):
        w["plan"]["sealed"] = [i for i in w["plan"]["sealed"] if (i["arm"], i["seed"]) != ("P2_WSTAR", 2027)]
        w["plan"]["declared_not_run"].append({"arm": "P2_WSTAR", "seed": 2027, "reason": "test"})

    def p1_pin(w):
        w["plan"]["phase1_reused"] = [{"arm": "PRE", "seed": 2026, "rule": PB, "checkpoint_sha256": "5" * 64}]

    def base_pin(w):
        item(w, "FT_P2_WSTAR", 2027)["expect_base_sha256"] = "6" * 64

    def dev_start(w):
        p = Path(item(w, "FT_P2_WSTAR", 2026)["run_dir"]) / "LAUNCH_0.json"
        o = json.loads(p.read_text(encoding="utf-8"))
        o["start"]["checkpoint"] = "endpoint_6.0.pt"
        _w(p, o)

    def bad_seed(w):
        w["plan"]["sealed"].append({"arm": "P2_C2_CFULL_SMALL", "seed": 2027, "run_dir": "x"})

    return [(status("P2_WSTAR", 2026, status="RUNNING"), "is not DONE_MAIN"),
            (status("P2_C1_FT_CENTRAL_FULL", 2027, status="INTERRUPTED"), "is not DONE_MAIN"),
            (status("P2_C2_CFULL_SMALL", 2026, status="DONE_MAIN", still_improving=True), "STILL_IMPROVING"),
            (status("FT_P2_WSTAR", 2026, status="PARTIAL"), "is not DONE"),
            (launch_sel, "is not the recorded P2_WARM_SELECTION_FINAL"), (c1_sel, "P2_C1_SELECTION"),
            (not_pre, "theta_pre"), (no_ckpt, "does not exist"), (expect, "differs from the expected"),
            (uncovered, "neither sealed nor declared not run"), (both, "both sealed and declared not run"),
            (no_wstar, "is not sealed"), (p1_pin, "is not ALL_METHODS_LOCK's"), (base_pin, "expected base"),
            (dev_start, "is not the sealed P2_WSTAR"), (bad_seed, "not a registered P2 (arm, seed)")]


def test_refusals_write_nothing(tmp_path):
    for i, (mutate, msg) in enumerate(refusal_cases()):
        w = make_world(tmp_path / f"c{i}")
        mutate(w)
        out = tmp_path / f"c{i}" / "gates" / "P2_METHODS_LOCK.json"
        with pytest.raises(PL.LockRefused, match=msg.replace("(", r"\(").replace(")", r"\)")):
            PL.write_lock(plan_file(w), out, say=lambda *_: None)
        assert not out.exists(), msg
    assert PL.main(["lock", "--plan", str(tmp_path / "missing.json"), "--out", str(tmp_path / "x.json")]) == 2


def test_registered_input_refusals(tmp_path, monkeypatch):
    w = make_world(tmp_path / "w")
    other = tmp_path / "PREG2.md"
    other.write_text("not the registered PREG2\n", encoding="utf-8")
    monkeypatch.setattr(PL, "PREG2_PATH", other)
    with pytest.raises(PL.LockRefused, match="PREG2"):
        PL.write_lock(plan_file(w), tmp_path / "o1.json", say=lambda *_: None)
    monkeypatch.setattr(PL, "PREG2_PATH", PL.V4 / "contracts" / "PREG2.md")
    monkeypatch.setattr(PL, "CONTRASTS_PATH", PL.V4 / "company" / "gates" / "P2_CONTRASTS_v2.json")
    with pytest.raises(PL.LockRefused, match="P2_CONTRASTS_v3"):
        PL.write_lock(plan_file(w), tmp_path / "o2.json", say=lambda *_: None)
    assert not (tmp_path / "o1.json").exists() and not (tmp_path / "o2.json").exists()


# ------------------------------------------------------------------------------------------------ register / freeze
def test_register_entry_is_appended_and_read_by_p2_report(tmp_path, monkeypatch):
    w = make_world(tmp_path / "w")
    out = tmp_path / "gates" / "P2_METHODS_LOCK.json"
    PL.write_lock(plan_file(w), out, say=lambda *_: None)
    said = []
    assert PL.main(["register-entry", "--lock", str(out)], say=said.append) == 0
    entry = json.loads(said[-1])
    tool = PL.register_tool()
    assert sorted(entry) == sorted(tool.FIELDS) and entry == PL.register_entry(out)
    assert entry["access_purpose"] == "LOCKED_EVALUATION" and entry["split"] == "RETRO_TEST"
    assert entry["all_methods_lock_sha256"] == _sha(out) and "P2_METHODS_LOCK" in entry["observations_description"]
    assert entry["code_sha256"] == json.loads(out.read_text(encoding="utf-8"))["code_tree"]["aggregate_sha256"]
    monkeypatch.setattr(tool, "REG", tmp_path / "TEST_EXPOSURE_REGISTER.json")
    p1 = dict(entry, all_methods_lock_sha256=PL.AML_SHA256, source="company/gates/ALL_METHODS_LOCK.json",
              observations_description="LOCKED_EVALUATION for ALL_METHODS_LOCK (Phase-1 sitting)")
    tool.append(p1)
    tool.append(entry)
    reg = json.loads(tool.REG.read_text(encoding="utf-8"))
    assert tool.verify(reg) == []
    got = P2.check_locked_evaluations(tool.REG, PL.AML_SHA256, _sha(out))
    assert got["phase1_entry_id"] == 1 and got["p2_entry_id"] == 2
    tampered = tmp_path / "tampered.json"
    o = json.loads(out.read_text(encoding="utf-8"))
    o["bootstrap_seed"] = 1
    tampered.write_text(json.dumps(o), encoding="utf-8")
    with pytest.raises(PL.LockRefused, match="self_sha256"):
        PL.register_entry(tampered)


def test_freeze_is_read_by_p2_report_and_pins_retention_report(tmp_path, monkeypatch):
    out = tmp_path / "P2_SCRIPT_FREEZE.json"
    rec = PL.write_freeze(out, say=lambda *_: None)
    got = P2.check_p2_freeze(out)
    assert got["files_sha256"] == rec["files_sha256"] and rec["files_sha256"]["tools/retention_report.py"] == PL.RR_SHA256
    with pytest.raises(PL.LockRefused, match="already exists"):
        PL.write_freeze(out, say=lambda *_: None)
    monkeypatch.setattr(PL, "FREEZE_PINS", dict(PL.FREEZE_PINS, **{"tools/retention_report.py": "0" * 64}))
    with pytest.raises(PL.LockRefused, match="freeze refuses"):
        PL.write_freeze(tmp_path / "f2.json", say=lambda *_: None)
    assert not (tmp_path / "f2.json").exists()


def test_code_tree_digest(tmp_path):
    w = make_world(tmp_path / "w")
    a = PL.code_tree_digest(w["tree"])
    (w["tree"] / "__pycache__").mkdir()
    (w["tree"] / "__pycache__" / "x.cpython-311.pyc").write_bytes(b"\0")
    assert PL.code_tree_digest(w["tree"]) == a and a["n_files"] == 2
    (w["tree"] / "v4int" / "reeval_p2.py").write_text("# changed\n", encoding="utf-8")
    assert PL.code_tree_digest(w["tree"])["aggregate_sha256"] != a["aggregate_sha256"]
    (w["tree"] / "v4int" / "p2_select.py").unlink()
    with pytest.raises(PL.LockRefused, match="p2_select.py"):
        PL.code_tree_digest(w["tree"])


# ------------------------------------------------------------------------------------------------ W* DP arms (P2-WIRE-DP)
def add_dp_run(w: dict, arm: str, seed: int, rid=None, spec_arm=None) -> dict:
    """A finished MAIN run of a W* DP-chain arm as launch_fl writes it: STATUS / ARM_SPEC / LAUNCH arm = the label, run id
    <arm>_SMALL_s<seed>, role MAIN, init_from = theta_pre, PRODUCTION LAUNCH leg."""
    rid = rid or f"{arm}_SMALL_s{seed}"
    label = spec_arm or arm
    rd = w["runs"] / f"{arm}_s{seed}"
    spec = {"arm": label, "study": "PREG2", "role": "MAIN", "cfg": {"seed": seed, "method": "FA"},
            "plan": {"sampling": "poisson"}, "init_from": {"checkpoint_sha256": PL.PRE_CHECKPOINT_SHA256}}
    _w(rd / "STATUS.json", {"run_id": rid, "arm": label, "status": "DONE_MAIN", "utc": "x"})
    _w(rd / "ARM_SPEC.json", {"arm": label, "run_id": rid, "arm_spec_sha256": "b" * 64, "arm_spec": spec})
    _w(rd / "LAUNCH_0.json", {"arm": label, "run_id": rid, "arm_spec_sha256": "b" * 64, "label": "PRODUCTION",
                              "fl05": {"role": "MAIN"}})
    for f in ("best.pt", "endpoint_6.0.pt"):
        _ck(rd / "ckpt" / rid / f, f"{rid}/{f}")
    w["plan"]["declared_not_run"] = [d for d in w["plan"]["declared_not_run"] if d["arm"] != arm]
    w["plan"]["sealed"] = [i for i in w["plan"]["sealed"] if (i["arm"], i["seed"]) != (arm, seed)]
    w["plan"]["sealed"].append({"arm": arm, "seed": seed, "run_dir": str(rd)})
    return w


def test_dp_arm_runs_are_sealed_and_wrong_ones_refused(tmp_path):
    w = make_world(tmp_path / "ok")
    for s in (2026, 2027):
        add_dp_run(w, "P2_WSTAR_DP8", s)
        add_dp_run(w, "P2_WSTAR_FA1024", s)
    out = tmp_path / "ok" / "gates" / "P2_METHODS_LOCK.json"
    PL.write_lock(plan_file(w), out, say=lambda *_: None)
    obj = json.loads(out.read_text(encoding="utf-8"))
    got = {(e["arm"], e["seed"], e["rule"]) for e in obj["sealed_arms"]}
    assert {("P2_WSTAR_DP8", s, r) for s in (2026, 2027) for r in (PB, CB)} <= got
    assert ("P2_WSTAR_DP1", 2026, PB) not in got                       # still declared not run
    e = next(e for e in obj["sealed_arms"] if (e["arm"], e["seed"], e["rule"]) == ("P2_WSTAR_DP8", 2026, CB))
    b = RP.p2_lock_arm(obj, label="P2_WSTAR_DP8", seed=2026, rule=CB, checkpoint_sha256=e["checkpoint_sha256"])
    assert b["checkpoint_sha256"] == e["checkpoint_sha256"]
    cases = [("DP1 s2027 is not registered", lambda w: add_dp_run(w, "P2_WSTAR_DP1", 2027), "not a registered P2"),
             ("wrong run id", lambda w: add_dp_run(w, "P2_WSTAR_DP8", 2026, rid="P2_WSTAR_DP8_X_SMALL_s2026"),
              "run id"),
             ("a P2_W run under the DP8 dir", lambda w: add_dp_run(w, "P2_WSTAR_DP8", 2026, spec_arm="P2_W"),
              "is not a run of"),
             ("a DP1 run labelled DP8", lambda w: add_dp_run(w, "P2_WSTAR_DP8", 2026, spec_arm="P2_WSTAR_DP1"),
              "is not a run of")]
    for i, (_why, mutate, msg) in enumerate(cases):
        w = make_world(tmp_path / f"bad{i}")
        mutate(w)
        o2 = tmp_path / f"bad{i}" / "gates" / "L.json"
        with pytest.raises(PL.LockRefused, match=msg):
            PL.write_lock(plan_file(w), o2, say=lambda *_: None)
        assert not o2.exists()
    w = make_world(tmp_path / "notpre")                                 # init_from must still be theta_pre
    add_dp_run(w, "P2_WSTAR_DP8", 2026)
    p = Path(item(w, "P2_WSTAR_DP8", 2026)["run_dir"]) / "ARM_SPEC.json"
    o = json.loads(p.read_text(encoding="utf-8"))
    o["arm_spec"]["init_from"]["checkpoint_sha256"] = "3" * 64
    _w(p, o)
    with pytest.raises(PL.LockRefused, match="theta_pre"):
        PL.write_lock(plan_file(w), tmp_path / "notpre" / "L.json", say=lambda *_: None)

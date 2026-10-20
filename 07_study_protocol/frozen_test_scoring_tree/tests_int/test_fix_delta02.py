"""FIX-DELTA-02 (REV-DELTA-02 B1, M1, M2, M3, m5, m6; amendment 06J §1-§4, DECISION_LOG row 219) — CPU, synthetic
configuration records and the tiny FL world of tests_int/test_fix_delta01_fl_binding.py (S2 -> S3 -> S4 from real tiny
screen runs; the registered contracts/fl paths are monkeypatched into the test's tmp directory).

Proves:
  B1  06J §4(ii) in lr_screen.stage_decision (= apply_rule_stage) at S2 / S3 / S4: the best candidate replaces the
      incumbent only with a gain >= 0.002 (else no measurable gain: winner = incumbent, tie set = [incumbent]); the
      stage file records gain / best_candidate / no_measurable_gain / tie_band_set / min_gain; the S2 FedAdam tau edge
      condition uses the same decision; launch_fl.load_fl_tuning recomputes all of it (tampered fields refused).
  M3  a condition-true 06J edge group with no run given refuses select-joint (EdgeRunsMissing naming the runs and the
      launch_fl flags), a partial group is refused, the tau condition waits for a required eta edge, an edge whose
      condition does not hold is refused; load_fl_tuning re-checks required and edge runs from the file's configs.
  M1  FA_1024 / DP / S_CAL under an S4 file whose winner has passes 4 or batch 8 run passes 2 / batch 16 with the
      winner's optimizer / LR / shape fields, recorded in the arm spec (poisson_local_rule), LAUNCH and rows; FA takes
      the winner as is; a default-local winner adds no key (plain spec digest); tampered records are refused.
  M2  a non-smoke MAIN run refuses LR_SELECTION_EXT.json and S2 / S3 stage files (the positive S4 binding is in
      test_fix_delta01_fl_binding.py).
  m5  select-joint refuses runs whose ARM_SPEC device records differ.
  m6  the frozen-table bytes record carries the bootstrap / first-visit-dropout note.
NCs (strict xfail on AssertionError): each MAJOR item's pre-fix behaviour asserted against the fixed code.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import v4int.launch_fl as F
import v4int.lr_screen as L
from intkit import assert_raises, nc
from test_fix_delta01_fl_binding import _json, _sha, main_args, register, world  # noqa: F401  (module fixture)
from test_fl05_launch import PEAK, _args, run_tiny, tiny_spec

T = F.tuning_config
LAB = F.tuning_label


def cfg(value, incumbent=False, excluded=False, **kw):
    t = T(**kw)
    return dict(t, label=LAB(t), value=value, incumbent=incumbent, excluded=excluded)


def A1_LOW(m):
    """FL-14 (06L A1): the FedAvgM m {1, 2} cells (deduped against m*), below every other value (no A1 edge)."""
    return [cfg(0.01, optimizer="fedavgm", lr_mult=x) for x in (1.0, 2.0) if x != m]


def _accepted(fn, *a, exc=F.LaunchRefused, **kw):
    try:
        fn(*a, **kw)
        return True
    except exc:
        return False


# ================================================================================================ B1 (06J §4(ii))
def test_b1_min_gain_all_stages():
    assert L.MIN_GAIN == 0.002
    inc2 = cfg(0.1000, True, lr_mult=8.0)
    for v, wins in ((0.1012, False), (0.1019, False), (0.1021, True), (0.1300, True)):
        d = L.stage_decision("S2", [inc2, cfg(v, optimizer="fedadam", server_lr=0.01, lr_mult=8.0),
                                    cfg(0.0990, optimizer="fedavgm", lr_mult=8.0)])
        assert d["best_candidate"] == "fedadam|eta=0.01|m=8" and d["gain"] == pytest.approx(v - 0.1)
        assert d["no_measurable_gain"] is (not wins) and d["min_gain"] == 0.002 and d["incumbent"] == "fedavg|m=8"
        assert (d["winner"]["optimizer"] == "fedadam") is wins
        assert d["tie_set"] == (["fedadam|eta=0.01|m=8"] if wins else ["fedavg|m=8"])
        assert d["tie_band_set"][0] == "fedadam|eta=0.01|m=8"                  # the raw 0.0005 band is recorded
    inc3 = cfg(0.2, True, optimizer="fedavgm", lr_mult=8.0)
    w, tie = L.apply_rule_stage("S3", [inc3, cfg(0.2015, optimizer="fedavgm", lr_mult=8.0, lr_shape="fl_const")])
    assert w == T(optimizer="fedavgm", lr_mult=8.0) and tie == ["fedavgm|m=8"]
    w, _ = L.apply_rule_stage("S3", [inc3, cfg(0.2025, optimizer="fedavgm", lr_mult=8.0, lr_shape="fl_const")])
    assert w["lr_shape"] == "fl_const"
    inc4 = cfg(0.3, True, lr_mult=8.0)
    w, tie = L.apply_rule_stage("S4", [inc4, cfg(0.3013, lr_mult=8.0, local_passes=1),
                                       cfg(0.2, lr_mult=8.0, local_passes=4), cfg(0.29, lr_mult=8.0, local_batch=8)])
    assert w["local_passes"] == 2 and tie == ["fedavg|m=8"]
    w, tie = L.apply_rule_stage("S4", [inc4, cfg(0.3023, lr_mult=8.0, local_passes=1),
                                       cfg(0.3020, lr_mult=8.0, local_batch=8)])
    assert w["local_batch"] == 8 and tie == ["fedavg|m=8|batch=8", "fedavg|m=8|passes=1"]      # band after the gain
    d = L.stage_decision("S4", [inc4])                                          # every candidate excluded
    assert d["gain"] is None and d["best_candidate"] is None and d["no_measurable_gain"] and d["tie_set"] == ["fedavg|m=8"]
    for bad in ([cfg(0.1, lr_mult=8.0), cfg(0.2, optimizer="fedavgm", lr_mult=8.0)],          # no incumbent
                [inc2, cfg(0.1, True, lr_mult=4.0)]):                                          # two incumbents
        assert_raises(L.SelectionRefused, L.stage_decision, "S2", bad)


def test_b1_tau_edge_uses_the_decision():
    m = 8.0
    inc = T(lr_mult=m)
    req = [LAB(t) for t in L.stage_required("S2", inc, m)]

    def s2(adam01_m8):                                   # the best base FedAdam run: eta 0.1 at m* (no eta edge)
        return [cfg(0.1, True, lr_mult=m), cfg(0.09, optimizer="fedadam", server_lr=0.01, lr_mult=1.0),
                cfg(0.09, optimizer="fedadam", server_lr=0.01, lr_mult=m),
                cfg(0.09, optimizer="fedadam", server_lr=0.1, lr_mult=1.0),
                cfg(adam01_m8, optimizer="fedadam", server_lr=0.1, lr_mult=m),
                cfg(0.08, optimizer="fedavgm", lr_mult=m)] + A1_LOW(m)
    groups, pending = L.edge_groups("S2", s2(0.1012), LAB(inc), req)
    assert groups == [] and pending is None                                    # +0.0012: the incumbent carries on
    assert L.check_stage_runs("S2", s2(0.1012), LAB(inc), req) == []
    groups, _ = L.edge_groups("S2", s2(0.1030), LAB(inc), req)
    assert [[LAB(t) for t in g[1]] for g in groups] == [["fedadam|eta=0.1|tau=1e-05|m=8"]]
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S2", s2(0.1030), LAB(inc), req)
    assert [t["label"] for t in ei.value.needed] == ["fedadam|eta=0.1|tau=1e-05|m=8"]
    assert "--server-opt fedadam --server-lr 0.1 --server-tau 1e-05 --lr-mult 8" in str(ei.value)
    ok = s2(0.1030) + [cfg(0.1, optimizer="fedadam", server_lr=0.1, server_tau=1e-5, lr_mult=m)]
    assert [e["labels"] for e in L.check_stage_runs("S2", ok, LAB(inc), req)] == [["fedadam|eta=0.1|tau=1e-05|m=8"]]


# ================================================================================================ M3 (required edges)
def _s2_eta_case(m=8.0):
    return [cfg(0.10, True, lr_mult=m), cfg(0.11, optimizer="fedadam", server_lr=0.01, lr_mult=1.0),
            cfg(0.15, optimizer="fedadam", server_lr=0.01, lr_mult=m),
            cfg(0.11, optimizer="fedadam", server_lr=0.1, lr_mult=1.0),
            cfg(0.11, optimizer="fedadam", server_lr=0.1, lr_mult=m),
            cfg(0.11, optimizer="fedavgm", lr_mult=m)] + A1_LOW(m)


def test_m3_s2_edges_required_in_order():
    m = 8.0
    inc, req = LAB(T(lr_mult=m)), [LAB(t) for t in L.stage_required("S2", T(lr_mult=m), m)]
    base = _s2_eta_case(m)
    with pytest.raises(L.EdgeRunsMissing) as ei:                                 # the eta edge first
        L.check_stage_runs("S2", base, inc, req)
    assert [t["label"] for t in ei.value.needed] == ["fedadam|eta=0.003|m=8", "fedadam|eta=0.001|m=8"]
    assert "tau edge condition" in str(ei.value) and str(ei.value).count("launch_fl --role CALIBRATION") == 2
    eta = [cfg(0.12, optimizer="fedadam", server_lr=0.003, lr_mult=m),
           cfg(0.12, optimizer="fedadam", server_lr=0.001, lr_mult=m)]
    with pytest.raises(L.SelectionRefused, match="must all be present") as ei:                 # partial: refused
        L.check_stage_runs("S2", base + eta[:1], inc, req)
    assert not isinstance(ei.value, L.EdgeRunsMissing)
    with pytest.raises(L.EdgeRunsMissing) as ei:                                 # then the tau edge (FedAdam wins)
        L.check_stage_runs("S2", base + eta, inc, req)
    assert [t["label"] for t in ei.value.needed] == ["fedadam|eta=0.01|tau=1e-05|m=8"]
    full = base + eta + [cfg(0.13, optimizer="fedadam", server_lr=0.01, server_tau=1e-5, lr_mult=m)]
    edges = L.check_stage_runs("S2", full, inc, req)
    assert [x["labels"] for x in edges] == [["fedadam|eta=0.003|m=8", "fedadam|eta=0.001|m=8"],
                                            ["fedadam|eta=0.01|tau=1e-05|m=8"]]
    assert_raises(L.SelectionRefused, L.check_stage_runs, "S2", full + [cfg(0.1, lr_mult=4.0)], inc, req)   # extra
    assert_raises(L.SelectionRefused, L.check_stage_runs, "S2", base[:-1] + eta, inc, req)   # required missing


def test_m3_s3_and_s4_edges():
    inc_t = T(optimizer="fedavgm", lr_mult=8.0)
    inc, req_t = LAB(inc_t), L.stage_required("S3", inc_t, 8.0)
    req = [LAB(t) for t in req_t]

    def s3(v001, v01, v1):
        return [cfg(0.2, True, optimizer="fedavgm", lr_mult=8.0),
                cfg(0.19, optimizer="fedavgm", lr_mult=8.0, lr_shape="fl_const")] + [
            cfg(v, optimizer="fedavgm", lr_shape="fl_const", client_opt="sgd_m", client_lr=lr)
            for lr, v in ((0.01, v001), (0.1, v01), (1.0, v1))] + [
            dict(t, label=LAB(t), value=0.05, incumbent=False, excluded=False) for t in L.a2_cells()]   # FL-14: 06L A2
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S3", s3(0.1, 0.1, 0.3), inc, req)
    assert [t["client_lr"] for t in ei.value.needed] == [3.0, 10.0]
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S3", s3(0.3, 0.1, 0.1), inc, req)
    assert [t["client_lr"] for t in ei.value.needed] == [0.003]
    hi = [cfg(0.1, optimizer="fedavgm", lr_shape="fl_const", client_opt="sgd_m", client_lr=lr) for lr in (3.0, 10.0)]
    assert len(L.check_stage_runs("S3", s3(0.1, 0.1, 0.3) + hi, inc, req)) == 1
    assert L.check_stage_runs("S3", s3(0.1, 0.3, 0.1), inc, req) == []                        # lr 0.1 best: none
    assert_raises(L.SelectionRefused, L.check_stage_runs, "S3", s3(0.1, 0.3, 0.1) + hi, inc, req)   # not allowed
    inc4 = T(lr_mult=8.0)
    req4 = [LAB(t) for t in L.stage_required("S4", inc4, 8.0)]
    c4 = [cfg(0.3, True, lr_mult=8.0), cfg(0.1, lr_mult=8.0, local_passes=1), cfg(0.1, lr_mult=8.0, local_passes=4),
          cfg(0.1, lr_mult=8.0, local_batch=8)]
    assert L.check_stage_runs("S4", c4, LAB(inc4), req4) == []


# ------------------------------------------------------------------------------------------------ files (world)
def forge(obj: dict, values=None, drop=(), keep_edges=False) -> dict:
    """A re-signed stage file whose derived fields follow the fixed rule for new values (configs in `drop` removed)."""
    o = json.loads(json.dumps(obj))
    o["configs"] = [c for c in o["configs"] if c["label"] not in drop]
    for c in o["configs"]:
        if values and c["label"] in values:
            c["value"], c["excluded"] = values[c["label"]], False
    fin = [c for c in o["configs"] if not c["excluded"]]
    d = L.stage_decision(o["stage"], fin)
    o["winner"] = dict(d["winner"], label=LAB(d["winner"]))
    for k in ("tie_set", "tie_band_set", "gain", "best_candidate", "no_measurable_gain", "min_gain"):
        o[k] = d[k]
    o["values"] = {c["label"]: c["value"] for c in fin}
    o["excluded"] = [c for c in o["configs"] if c["excluded"]]
    if not keep_edges:
        try:
            o["edges"] = L.check_stage_runs(o["stage"], o["configs"], o["incumbent"], o["required"])
        except L.SelectionRefused:
            o["edges"] = []
    o["self_sha256"] = F._canon(o)
    return o


def _write(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return p


def test_stage_files_record_the_decision(world):
    for f in (world["s2_file"], world["s3_file"], world["tun"]):
        obj = F.load_fl_tuning(f)
        d = L.stage_decision(obj["stage"], [c for c in obj["configs"] if not c["excluded"]])
        assert {k: obj[k] for k in ("gain", "best_candidate", "no_measurable_gain", "tie_band_set", "min_gain")} == \
            {k: d[k] for k in ("gain", "best_candidate", "no_measurable_gain", "tie_band_set", "min_gain")}
        assert obj["tie_set"] == d["tie_set"] and "06J §4(ii)" in obj["rule"]
        assert obj["device"] == L.read_run(obj["configs"][0]["run_dir"])["device"]
    s4 = _json(world["tun"])
    assert s4["applies_to_exception"]["arms"] == ["FA_1024", "DP", "S_CAL"]
    assert (s4["applies_to_exception"]["local_passes"], s4["applies_to_exception"]["local_batch"]) == (2, 16)


def test_load_fl_tuning_rechecks_b1_and_m3(world, tmp_path):
    obj = _json(world["s2_file"])
    F.load_fl_tuning(_write(tmp_path / "ok.json", forge(obj)))                  # forge() itself is faithful
    for k, v in (("gain", 0.5), ("no_measurable_gain", not obj["no_measurable_gain"]), ("min_gain", 0.0005),
                 ("best_candidate", "fedavg|m=32"), ("tie_band_set", []), ("required", obj["required"][:-1]),
                 ("edges", obj["edges"] + [{"rule": "x", "labels": ["fedavg|m=32"]}]), ("incumbent", "fedavg|m=32")):
        bad = dict(obj, **{k: v})
        bad["self_sha256"] = F._canon(bad)
        assert_raises(F.LaunchRefused, F.load_fl_tuning, _write(tmp_path / f"b_{k}.json", bad))
    # the pre-FIX rule's winner: a FedAdam candidate + 0.0012 over the incumbent recorded as the winner
    inc = obj["incumbent"]
    iv = next(c["value"] for c in obj["configs"] if c["label"] == inc)
    adam = next(c["label"] for c in obj["configs"] if c["optimizer"] == "fedadam" and c["server_lr"] == 0.1
                and c["server_tau"] == 1e-3)
    vals = {c["label"]: iv - (0.02 if (c["optimizer"], c["lr_mult"]) == ("fedavgm", 1.0) else 0.01)   # FL-14: no A1
            for c in obj["configs"] if c["label"] != inc}
    vals[adam] = iv + 0.0012
    edge_labels = [lab for e in obj["edges"] for lab in e["labels"]]
    f = forge(obj, vals, drop=edge_labels)
    assert f["winner"]["label"] == inc and f["no_measurable_gain"]
    F.load_fl_tuning(_write(tmp_path / "b1_ok.json", f))
    old = dict(f, winner=dict(next(c for c in f["configs"] if c["label"] == adam)), tie_set=[adam])
    old["winner"] = {k: old["winner"][k] for k in F.TUNING_KEYS}
    old["self_sha256"] = F._canon(old)
    assert_raises(F.LaunchRefused, F.load_fl_tuning, _write(tmp_path / "b1_old.json", old))
    # M3: the eta edge condition holds (best base FedAdam = eta 0.01) but no edge run is recorded
    adam01 = next(c["label"] for c in obj["configs"] if c["optimizer"] == "fedadam" and c["server_lr"] == 0.01
                  and c["server_tau"] == 1e-3)
    vals = {c["label"]: iv - (0.02 if (c["optimizer"], c["lr_mult"]) == ("fedavgm", 1.0) else 0.01)   # FL-14: no A1
            for c in obj["configs"] if c["label"] != inc}
    vals[adam01] = iv + 0.05
    g = forge(obj, vals, drop=edge_labels)
    assert g["edges"] == []
    with pytest.raises(F.LaunchRefused, match="edge"):
        F.load_fl_tuning(_write(tmp_path / "m3.json", g))


def test_select_joint_refuses_mixed_devices(world, tmp_path):
    runs = []
    for i, d in enumerate(world["runs4"]):
        dst = tmp_path / f"r{i}"
        shutil.copytree(d, dst)
        runs.append(dst)
    L.selection_stage("S4", runs, world["s3_file"])
    side = _json(runs[1] / "ARM_SPEC.json")
    side["device"] = {"eval_device_class": "A40-FP32-DET", "device_pin_sha256": None}
    (runs[1] / "ARM_SPEC.json").write_text(json.dumps(side), encoding="utf-8")
    with pytest.raises(L.SelectionRefused, match="device"):
        L.selection_stage("S4", runs, world["s3_file"])


# ================================================================================================ M1 (DL 219)
def _s_record(tmp_path) -> Path:
    rec = F.s_record_obj({"S": 0.5, "n_norms": 20, "rounds_0based": list(range(20)), "rule": "median"},
                         run_id="FL_S_CAL_SMALL_s2026", seed=2026, config_digest="c" * 64, plan_digest="p" * 64,
                         init_sha256="i" * 64, smoke=False)
    return _write(tmp_path / "S_RECORD.json", rec)


def _reg_forged(world, tmp_path, monkeypatch, target: str) -> tuple:
    """The world's S4 file re-signed so that `target` (passes=4 / batch=8 / the incumbent) wins, at the registered path."""
    obj = _json(world["tun"])
    inc = obj["incumbent"]
    iv = next(c["value"] for c in obj["configs"] if c["label"] == inc)
    lab = inc if target == "inc" else next(c["label"] for c in obj["configs"] if c["label"].endswith(target))
    vals = {c["label"]: iv - 0.01 for c in obj["configs"] if c["label"] != inc}
    if lab != inc:
        vals[lab] = iv + 0.01
    f = forge(obj, vals)
    reg = tmp_path / f"reg_{target.replace('=', '')}"
    reg.mkdir()
    shutil.copy(world["first"], reg / F.FIRST_SCREEN_NAME)
    shutil.copy(world["ext"], reg / F.LR_SELECTION_EXT_NAME)
    _write(reg / F.FL_TUNING_NAME, f)
    register(monkeypatch, reg, _sha(world["first"]))
    return reg / F.FL_TUNING_NAME, T(**{k: f["winner"][k] for k in F.TUNING_KEYS})


def _poisson_args(tmp_path, arm, tun):
    extra = ["--fl-tuning", str(tun)]
    if arm in ("FA_1024", "DP"):
        extra += ["--clip-record", str(_s_record(tmp_path))]
    if arm == "DP":
        extra += ["--epsilon", "8"]
    return _args(tmp_path, *extra, smoke=False, arm=arm, seed="2026")


@pytest.mark.parametrize("target", ["passes=4", "batch=8"])
def test_m1_poisson_arms_keep_p2_b16(world, tmp_path, monkeypatch, target):
    tun, w = _reg_forged(world, tmp_path, monkeypatch, target)
    assert (w["local_passes"], w["local_batch"]) == ((4, 16) if target == "passes=4" else (2, 8))
    fa, fa_extra = F.resolve_spec(main_args(tmp_path, "--fl-tuning", str(tun), arm="FA"), PEAK, fused=True)
    assert F.spec_tuning(fa) == w and "poisson_local_rule" not in fa and "poisson_local_rule" not in fa_extra
    for arm in ("S_CAL", "FA_1024", "DP"):
        spec, extra = F.resolve_spec(_poisson_args(tmp_path, arm, tun), PEAK, fused=True)
        t = F.spec_tuning(spec)
        assert (t["local_passes"], t["local_batch"]) == (2, 16)
        assert {k: t[k] for k in F.POISSON_TAKEN_FIELDS} == {k: w[k] for k in F.POISSON_TAKEN_FIELDS}
        assert spec["cfg"]["solver"]["passes"] == 2 and spec["cfg"]["solver"]["batch_size"] == 16
        assert spec["cfg"]["endpoint_rounds"] == 384 and "exposure_budget" not in spec
        assert spec["poisson_local_rule"] == F.poisson_rule_record(w["local_passes"], w["local_batch"])
        assert extra["poisson_local_rule"]["applied"] and extra["main_selection"]["stage"] == "S4"
        assert extra["poisson_local_rule"]["fields_taken"] == list(F.POISSON_TAKEN_FIELDS)
        assert "row 219" in extra["poisson_local_rule"]["authority"]
        plain = dict(spec)
        plain.pop("poisson_local_rule")
        assert F.spec_digest(plain) != F.spec_digest(spec)                      # the record is in the digest


def test_m1_default_local_winner_adds_nothing(world, tmp_path, monkeypatch):
    tun, w = _reg_forged(world, tmp_path, monkeypatch, "inc")
    assert (w["local_passes"], w["local_batch"]) == (2, 16)
    spec, extra = F.resolve_spec(_poisson_args(tmp_path, "S_CAL", tun), PEAK, fused=True)
    assert "poisson_local_rule" not in spec and extra["poisson_local_rule"]["applied"] is False
    assert F.spec_digest(spec) == F.spec_digest(F.arm_spec("S_CAL", seed=2026, peak_lr=PEAK, fused=True,
                                                           **F.tuning_kwargs(w)))


def test_m1_spec_record_checks_and_rows(tmp_path):
    rec = F.poisson_rule_record(4, 16)
    s = F.arm_spec("DP", seed=2027, peak_lr=PEAK, clip_norm=0.5, z=0.5642, lr_mult=8.0, poisson_local_rule=rec)
    assert F.spec_tuning(s)["local_passes"] == 2 and s["poisson_local_rule"] == rec
    for arm, bad, kw in (("FA", rec, {}), ("DP", F.poisson_rule_record(2, 16), {"clip_norm": 0.5, "z": 0.5642}),
                         ("DP", dict(rec, extra=1), {"clip_norm": 0.5, "z": 0.5642}),
                         ("DP", F.poisson_rule_record(3, 16), {"clip_norm": 0.5, "z": 0.5642}),
                         ("S_CAL", rec, {"local_batch": 8})):
        assert_raises(F.LaunchRefused, F.arm_spec, arm, seed=2026, peak_lr=PEAK, poisson_local_rule=bad, **kw)
    tampered = dict(s, poisson_local_rule=dict(rec, selected_local_passes=2))
    assert_raises(F.LaunchRefused, F.spec_tuning, tampered)
    out = run_tiny(tmp_path, tiny_spec("FA_1024", rounds=8, clip_norm=0.5, poisson_local_rule=F.poisson_rule_record(2, 8)),
                   "p")
    assert out["status"] == "DONE_MAIN"
    rows = _json(tmp_path / "p" / "RESULT_ROWS.json")["rows"]
    assert rows and all(r["fl_poisson_local_rule"] == "DL219_P2_B16" and r["fl_selected_local_batch"] == 8
                        and r["fl_selected_local_passes"] == 2 for r in rows)
    plain = run_tiny(tmp_path, tiny_spec("FA_1024", rounds=8, clip_norm=0.5), "q")
    assert plain["summary"]["server_digest"] == out["summary"]["server_digest"]    # same numerics (P2 / B16)
    assert all("fl_poisson_local_rule" not in r for r in _json(tmp_path / "q" / "RESULT_ROWS.json")["rows"])


# ================================================================================================ M2
def test_m2_ext_and_early_stage_files_refused(world, tmp_path, monkeypatch):
    register(monkeypatch, world["reg_a"], _sha(world["first"]))                  # EXT registered, no tuning file
    assert_raises(F.LaunchRefused, F.require_lr_selection, main_args(tmp_path, "--lr-selection", str(world["ext"])))
    for src in (world["s2_file"], world["s3_file"]):
        reg = tmp_path / src.stem
        reg.mkdir()
        shutil.copy(world["first"], reg / F.FIRST_SCREEN_NAME)
        shutil.copy(world["ext"], reg / F.LR_SELECTION_EXT_NAME)
        shutil.copy(src, reg / F.FL_TUNING_NAME)
        register(monkeypatch, reg, _sha(world["first"]))
        with pytest.raises(F.LaunchRefused, match="S4"):
            F.require_lr_selection(main_args(tmp_path, "--fl-tuning", str(reg / F.FL_TUNING_NAME)))


# ================================================================================================ m6
def test_m6_frozen_bytes_note():
    from fl.runtime import freeze_item_tables
    from fl.synthetic import make_tiny_adapter
    a = make_tiny_adapter(seed=1, tied=False)
    freeze_item_tables(a, F.FROZEN_ITEM_KEYS)
    rec = F.bytes_record(a.manifest, SimpleNamespace(visit_counts=[1, 0, 2]),
                         SimpleNamespace(snapshot=lambda: (6, 10, 10, 0)), frozen_keys=F.FROZEN_ITEM_KEYS)
    assert "bootstrap_bytes_once" in rec["frozen_bootstrap_note"] and "dropped at its first visit" in \
        rec["frozen_bootstrap_note"]
    assert "frozen_bootstrap_note" not in F.bytes_record(make_tiny_adapter(seed=1, tied=False).manifest,
                                                         SimpleNamespace(visit_counts=[1]),
                                                         SimpleNamespace(snapshot=lambda: (2, 1, 1, 0)))


# ================================================================================================ negative controls
@nc("06J §4(ii) dropped: a FedAdam run gaining +0.0012 over the S2 incumbent is the winner")
def test_nc_b1_s2_small_gain_wins():
    w, _ = L.apply_rule_stage("S2", [cfg(0.1, True, lr_mult=8.0), cfg(0.1012, optimizer="fedadam", server_lr=0.01,
                                                                      lr_mult=8.0)])
    assert w["optimizer"] == "fedadam"


@nc("06J §4(ii) dropped: local passes 1 gaining +0.0013 over the S4 incumbent is the winner")
def test_nc_b1_s4_small_gain_wins():
    w, _ = L.apply_rule_stage("S4", [cfg(0.1, True, lr_mult=8.0), cfg(0.1013, lr_mult=8.0, local_passes=1)])
    assert w["local_passes"] == 1


@nc("the tau edge condition read without 06J §4(ii) requires a tau run for a +0.0012 FedAdam gain")
def test_nc_b1_tau_edge_raw_best():
    m = 8.0
    c = [cfg(0.1, True, lr_mult=m), cfg(0.09, optimizer="fedadam", server_lr=0.01, lr_mult=1.0),
         cfg(0.09, optimizer="fedadam", server_lr=0.01, lr_mult=m), cfg(0.09, optimizer="fedadam", server_lr=0.1, lr_mult=1.0),
         cfg(0.1012, optimizer="fedadam", server_lr=0.1, lr_mult=m),
         cfg(0.08, optimizer="fedavgm", lr_mult=m)] + A1_LOW(m)
    groups, _ = L.edge_groups("S2", c, LAB(T(lr_mult=m)), [LAB(t) for t in L.stage_required("S2", T(lr_mult=m), m)])
    assert groups


@nc("select-joint writes the S2 file while the registered eta edge condition holds and its runs are absent")
def test_nc_m3_missing_edge_runs_accepted():
    m = 8.0
    assert _accepted(L.check_stage_runs, "S2", _s2_eta_case(m), LAB(T(lr_mult=m)),
                     [LAB(t) for t in L.stage_required("S2", T(lr_mult=m), m)], exc=L.SelectionRefused), \
        "refused (as it must be)"


@nc("a DP arm under an S4 file whose winner has 4 local passes runs 4 passes")
def test_nc_m1_poisson_takes_s4_passes():
    assert F.poisson_tuning("DP", T(lr_mult=8.0, local_passes=4))[0]["local_passes"] == 4


@nc("an FA_1024 arm under an S4 file whose winner has batch 8 runs batch 8 (DL 219 says 16)")
def test_nc_m1_poisson_takes_s4_batch():
    assert F.poisson_tuning("FA_1024", T(lr_mult=8.0, local_batch=8))[0]["local_batch"] == 8


@nc("a non-smoke MAIN run binds the registered LR_SELECTION_EXT.json (no S4 file)")
def test_nc_m2_ext_binds_main(world, tmp_path, monkeypatch):
    register(monkeypatch, world["reg_a"], _sha(world["first"]))
    assert _accepted(F.require_lr_selection, main_args(tmp_path, "--lr-selection", str(world["ext"]))), \
        "refused (as it must be)"


@nc("a non-smoke MAIN run binds an S3-stage FL_TUNING_SELECTION.json at the registered path")
def test_nc_m2_s3_binds_main(world, tmp_path, monkeypatch):
    reg = tmp_path / "reg"
    reg.mkdir()
    shutil.copy(world["first"], reg / F.FIRST_SCREEN_NAME)
    shutil.copy(world["ext"], reg / F.LR_SELECTION_EXT_NAME)
    shutil.copy(world["s3_file"], reg / F.FL_TUNING_NAME)
    register(monkeypatch, reg, _sha(world["first"]))
    assert _accepted(F.require_lr_selection, main_args(tmp_path, "--fl-tuning", str(reg / F.FL_TUNING_NAME))), \
        "refused (as it must be)"


@nc("select-joint accepts runs evaluated on two device classes")
def test_nc_m5_mixed_devices_accepted(world, tmp_path):
    runs = []
    for i, d in enumerate(world["runs4"]):
        dst = tmp_path / f"r{i}"
        shutil.copytree(d, dst)
        runs.append(dst)
    side = _json(runs[0] / "ARM_SPEC.json")
    side["device"] = {"eval_device_class": "A40-FP32-DET", "device_pin_sha256": None}
    (runs[0] / "ARM_SPEC.json").write_text(json.dumps(side), encoding="utf-8")
    assert _accepted(L.selection_stage, "S4", runs, world["s3_file"], exc=L.SelectionRefused), "refused (as it must be)"

"""ANA-06 — the FL-16 warm method set and PREG1_AMENDMENT_06N in tools/retention_report.py / tools/ana_block_index.py
(synthetic blocks only; no RESULT_ROWS value of a real run is read):

  * the 12 launch_fl warm labels map to their block arms (= the FL-16 run-dir names; DP_WARM_FROZEN eps 8 = DP8_FROZEN,
    eps 1 = DP1_WARM_FROZEN; DP_WARM eps 8 / 1 = DP8_WARM / DP1_WARM; S_CAL_WARM* excluded), each bound to its FL run and
    its variant's S7 / S8 file; a run whose cold-arm semantics do not match its label refuses (report and index);
  * FA_WARM* (06N N2) chosen mechanically from the two selection files (higher CALIB winner value, |Δ| <= 0.0005 -> S8);
  * the registered 06P set (PREG1_AMENDMENT_06P) — DESCRIPTIVE '<id>*' rows when v3 is loaded (v3 confirmatory set
    unchanged), CONFIRMATORY with company/gates/ANA_CONTRASTS_v4.json (primary pair K20b + K21, Holm over the enumerated
    secondary family, the tertiary band) — known answers;
  * REV-06O-ANA05 MINOR 1 (rows of an entry beside a run folder must name that run), DL 287 m3 (no *_WARM* FT_FA / FT_C
    base), the warm S record content check (REV-FL-16 m5, ANA side), the 06N N3.3 N33_ cold re-runs and the 06N N4
    STOPPED_REALLOCATED dirs in the index.
Negative controls (test_nc_*, strict xfail on AssertionError)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import ana_block_index as BI
import anakit
import layoutkit
import retention_report as R
import test_gate_b as GB
from test_ana02 import edit_index
from test_ana04 import add_fl, pbv
from test_retention_report import _set_rows_field, check_refuses, nc, result, run

M, PB, CB = "MAIN", "PRACTICAL_BEST", "CONTROLLED_BUDGET"
S8_LINE = ("FP_WARM_FROZEN", "PF_WARM_FROZEN", "FA_Q8_WARM_FROZEN", "FA_1024_WARM_FROZEN", "DP8_FROZEN")


# ------------------------------------------------------------------------------------------------ fixtures
def set_star_values(blk, v7, v8) -> None:
    """Rewrite the registered S7 / S8 files so the selected cell carries a CALIB value, and re-point every warm entry's
    LAUNCH binding (sha256 / self_sha256) at the new bytes."""
    idx = blk.index()
    for st, v, arm in (("S7", v7, "FA_WARM"), ("S8", v8, "FA_WARM_FROZEN")):
        p = blk.root / "records" / f"FL_WARM_SELECTION_{st}.json"
        obj = anakit.warm_selection_obj(arm)
        obj.pop("self_sha256")
        obj["values"] = {obj["winner"]["label"]: v}
        obj = anakit.canon(obj)
        text = json.dumps(obj, indent=1, sort_keys=True) + "\n"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())
        idx["records"][R.WARM_SELECTION_RECORD[st]] = f"records/{p.name}"
        for e in idx["entries"]:
            if R.warm_stage_of(e["arm"]) == st and e.get("warm_selection"):
                e["warm_selection"].update(sha256=R._sha_bytes(text.encode()), self_sha256=obj["self_sha256"])
    blk.write_index(idx)


def n06_block(tmp: Path, *, v7=0.170582, v8=0.170709, dp1=True):
    """v3 registry; FA (cold, 2 seeds), FA_WARM (S7) and FA_WARM_FROZEN (S8) run dirs, PRE, C_FULL s2026 and the S8 warm
    line (FP / PF / FA_Q8 / FA_1024 _WARM_FROZEN and DP8_FROZEN 2 seeds, DP1_WARM_FROZEN s2026)."""
    blk = anakit.build(tmp)
    anakit.jdump(blk.contrasts, anakit.contrasts_obj(3))
    anakit.add_warm_start(blk, fa_warm_seeds=())                     # PRE only (ONE theta_pre)
    add_fl(blk, "FA_WARM")
    GB.add_s8(blk)
    for arm in S8_LINE:
        add_fl(blk, arm)
    if dp1:
        add_fl(blk, "DP1_WARM_FROZEN", (2026,))
    set_star_values(blk, v7, v8)
    return blk


@pytest.fixture(scope="module")
def n06(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("n06")
    blk = n06_block(tmp)
    rc, text = run(blk, tmp / "out")
    assert rc == 0, text
    return blk, result(tmp / "out"), (tmp / "out" / R.REPORT_NAME).read_text(encoding="utf-8")


def _row(res, kid):
    return next(c for c in res["ana06"]["contrasts"] if c["id"] == kid)


def warm_s_record(S=1.0, n=20625, **over) -> dict:
    obj = {"format": R.S_FORMAT_WARM, "S": S, "n_norms": n, "rounds_0based": list(range(20)), "rule": "median",
           "run_id": "FL_S_CAL_WARM_FROZEN_SMALL_s2026", "seed": 2026, "config_digest": "c" * 64,
           "plan_digest": "d" * 64, "init_sha256": "e" * 64, "label": "S_CALIBRATION", "arm": "S_CAL_WARM_FROZEN",
           "warm_variant": "FA_WARM_FROZEN", "stage": "S8", "init_from": dict(anakit.INIT_FROM),
           "frozen_item_tables": list(anakit.FROZEN_KEYS)}
    obj.update(over)
    return anakit.canon(obj)


def add_s_record(blk, obj) -> None:
    p = blk.root / "records" / "S_RECORD_WARM_FROZEN.json"
    anakit.jdump(p, obj)
    edit_index(blk, lambda idx: idx["records"].__setitem__("S_RECORD_WARM_FROZEN", f"records/{p.name}"))


# ------------------------------------------------------------------------------------------------ tables
def test_warm_line_tables_equal_launch_fl():
    import v4int.launch_fl as F
    from v4int import reeval as RV
    assert R.LAUNCH_WARM_LINE == F.WARM_LINE and R.S_FORMAT_WARM == F.S_FORMAT_WARM
    assert R.S_CAL_WARM_ARM == F.S_CAL_WARM_ARM and tuple(R.WARM_LINE_EXCLUDED) == tuple(
        a for a in F.WARM_LINE if F.base_arm(a) == "S_CAL")
    cold = {"DP8": "DP", "DP1": "DP"}
    for arm, (lab, base, variant, eps) in R.WARM_LINE_ARMS.items():
        assert F.WARM_LINE[lab] == (cold.get(base, base), variant), arm
        assert R.METHOD_CODE_OF[base] == F.METHOD_CODE[lab] and (eps is not None) == (F.base_arm(lab) == "DP")
        assert RV.WARM_BLOCK_ARM[(lab, eps)] == arm and arm in R.ARMS and arm in R.FL_ARMS
        assert (arm in R.FROZEN_ONCE_ARMS) == F.is_frozen_arm(lab) and arm in R.WARM_START_ARMS
    assert R.WARM_LINE_ARMS["DP8_FROZEN"][:4] == ("DP_WARM_FROZEN", "DP8", "FA_WARM_FROZEN", 8.0)
    assert len(set(R.WARM_LINE_ARMS)) == 12 and len(R.LAUNCH_WARM_LINE) == 12


def test_fa_warm_star_rule():
    def rec(v7, v8):
        return {"_WARM_WINNER_CALIB": {"S7": {"value": v7}, "S8": {"value": v8}}}
    assert R.fa_warm_star(rec(0.170582, 0.170709), "VALIDATION", None)["arm"] == "FA_WARM_FROZEN"   # DL 292-306 case
    assert R.fa_warm_star(rec(0.1720, 0.1700), "VALIDATION", None)["arm"] == "FA_WARM"
    tie = R.fa_warm_star(rec(0.1704, 0.1700), "VALIDATION", None)
    assert tie["arm"] == "FA_WARM_FROZEN" and tie["tie"] is True                     # |Δ| <= 0.0005 -> S8
    one = R.fa_warm_star({"_WARM_WINNER_CALIB": {"S7": {"value": 0.17}}}, "VALIDATION", None)
    assert one["status"] == "UNDETERMINED" and one["arm"] is None
    val = {"ana06": {"fa_warm_star": {"stage": "S8"}}}
    assert R.fa_warm_star({}, "TEST", val)["status"] == "FROM_VALIDATION_RESULT"
    with pytest.raises(R.AnaRefused, match="VALIDATION_RESULT chose"):
        R.fa_warm_star(rec(0.1720, 0.1700), "TEST", val)


# ------------------------------------------------------------------------------------------------ 06N known answers
def test_06n_rows_known_answers(n06):
    blk, res, _rep = n06
    a = res["ana06"]
    assert a["status"] == "COMPUTED" and a["fa_warm_star"]["arm"] == "FA_WARM_FROZEN"
    assert a["fa_warm_star"]["calib_winner"]["S8"]["value"] == 0.170709
    k = _row(res, "K20b*")
    assert k["contrast"] == "R(FA_WARM_FROZEN / C_FULL)" and k["tier_06N"] == "primary" and k["label"] == "DESCRIPTIVE"
    want = np.mean([100.0 * pbv(blk, "FA_WARM_FROZEN", s) / pbv(blk, "C_FULL", 2026) for s in (2026, 2027)])
    assert k["by_rule"][PB]["point"] == pytest.approx(want, abs=1e-9) and k["by_rule"][PB]["seeds"] == [2026, 2027]
    k21 = _row(res, "K21*")
    want = np.mean([pbv(blk, "FA_WARM_FROZEN", s) - pbv(blk, "PRE", 2026) for s in (2026, 2027)])
    assert k21["contrast"] == "FA_WARM_FROZEN − PRE" and k21["by_rule"][PB]["point"] == pytest.approx(want, abs=1e-12)
    ki = _row(res, "K_INIT*")
    want = np.mean([pbv(blk, "FA_WARM_FROZEN", s) - pbv(blk, "FA", s) for s in (2026, 2027)])
    assert ki["by_rule"][PB]["point"] == pytest.approx(want, abs=1e-12)
    k12 = _row(res, "K12w*")
    assert k12["contrast"] == "R(DP8_FROZEN / FA_1024_WARM_FROZEN)" and k12["confirmatory_rule"] == CB
    want = np.mean([100.0 * pbv(blk, "DP8_FROZEN", s, CB) / pbv(blk, "FA_1024_WARM_FROZEN", s, CB) for s in (2026, 2027)])
    assert k12["by_rule"][CB]["point"] == pytest.approx(want, abs=1e-9)
    assert _row(res, "K13w*")["by_rule"][CB]["seeds"] == [2026] and _row(res, "K14w*")["contrast"] == \
        "R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN)"
    assert "equivalence" in _row(res, "K14w*")["by_rule"][PB]                       # K14w: TOST ±1 % (06P P4)
    assert {c["id"] for c in a["contrasts"]} == {f"{k}*" for k in R.N06_IDS}
    k4 = _row(res, "K4w*")                                                          # 06P P4: vs C_FULL s2026 (shared)
    assert k4["contrast"] == "R(FP_WARM_FROZEN / C_FULL)" and k4["by_rule"][PB]["seeds"] == [2026, 2027]
    assert k4["by_rule"][PB]["point"] == pytest.approx(
        np.mean([100.0 * pbv(blk, "FP_WARM_FROZEN", s) / pbv(blk, "C_FULL", 2026) for s in (2026, 2027)]), abs=1e-9)
    assert _row(res, "P5*")["by_rule"][PB]["status"] == "NOT_AVAILABLE" and k4["tier_06N"] == "secondary"
    assert all(b["label"] == "DESCRIPTIVE" for c in a["contrasts"] for b in c["by_rule"].values())
    assert len(res["labels"]["CONFIRMATORY"]) == 26 and res["labels"]["hierarchy"]["K2"] == "primary"   # v3 unchanged
    assert a["needs_registration"] and {r["id"] for r in a["references"]} == {"K2", "S8-S7"}


def test_warm_line_bound_and_reported(n06):
    _blk, res, rep = n06
    arms = res["ana06"]["warm_line"]["arms"]
    assert set(arms) == set(S8_LINE) | {"DP1_WARM_FROZEN"}
    for arm, w in arms.items():
        assert w["status"] == "BOUND_AND_FILE_VERIFIED" and w["stage"] == "S8" and w["frozen"] is True, arm
        assert {r["seed"] for r in w["bound_runs"]} == ({2026} if arm == "DP1_WARM_FROZEN" else {2026, 2027})
    assert arms["DP8_FROZEN"]["launch_arm"] == "DP_WARM_FROZEN" and arms["DP8_FROZEN"]["base_arm"] == "DP8"
    tab = {r["arm"] for r in res["retention"]["MAIN"][PB]}
    assert set(S8_LINE) <= tab and "## 12c. ANA-06" in rep and "Needs registration" in rep
    assert "_WARM_WINNER_CALIB" not in res["records"]


def test_star_undetermined_without_values(tmp_path):
    blk = anakit.build(tmp_path)
    add_fl(blk, "FA_WARM")
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    a = result(tmp_path / "o")["ana06"]
    assert a["status"] == "NOT_AVAILABLE" and a["contrasts"] == [] and a["fa_warm_star"]["status"] == "UNDETERMINED"


def test_s7_star_flagged_against_06p(tmp_path):
    blk = n06_block(tmp_path, v7=0.1720, v8=0.1700, dp1=False)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    res = result(tmp_path / "o")
    assert res["ana06"]["status"] == "COMPUTED_STAR_CONFLICT"                       # 06P P1 registers S8
    assert _row(res, "K20b*")["contrast"] == "R(FA_WARM_FROZEN / C_FULL)"


# ------------------------------------------------------------------------------------------------ refusals (report)
def test_warm_line_semantics_refused(tmp_path):
    blk = n06_block(tmp_path / "a", dp1=False)
    p = blk.root / "MAIN" / "FL_FP_WARM_FROZEN_s2026" / "RUN_SUMMARY.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s["arm_spec"]["cfg"]["mu"] = 0.0                                                # FedProx without mu: not FP
    anakit.jdump(p, s)
    check_refuses(blk, tmp_path / "a", "cfg.mu")
    blk = n06_block(tmp_path / "b", dp1=False)
    p = blk.root / "MAIN" / "FL_DP8_FROZEN_s2026" / "RUN_SUMMARY.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s["arm"] = "DP"                                                                 # the legacy (unlaunchable) shape
    anakit.jdump(p, s)
    check_refuses(blk, tmp_path / "b", "not DP_WARM_FROZEN")


@nc("a report without the warm-line semantics check (warm_line_base_problem) must not pass the refusals")
def test_nc_semantics_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "warm_line_base_problem", lambda arm, summ: None)
    blk = n06_block(tmp_path / "a", dp1=False)
    p = blk.root / "MAIN" / "FL_FP_WARM_FROZEN_s2026" / "RUN_SUMMARY.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s["arm_spec"]["cfg"]["mu"] = 0.0
    anakit.jdump(p, s)
    check_refuses(blk, tmp_path / "a", "cfg.mu")


def test_warm_line_unbound_refused(tmp_path):
    blk = n06_block(tmp_path, dp1=False)
    edit_index(blk, lambda idx: [e.pop("warm_selection") for e in idx["entries"] if e["arm"] == "PF_WARM_FROZEN"])
    check_refuses(blk, tmp_path, "not bound to the registered 2-D")


def check_minor1(tmp):
    blk = n06_block(tmp, dp1=False)
    _set_rows_field(blk, "MAIN/FL_FA_Q8_WARM_FROZEN_s2027/RESULT_ROWS.json", "fl_run_id", "FL_OTHER_SMALL_s2027",
                    where={"selection_rule": CB})                                   # the CB entry beside the run folder
    check_refuses(blk, tmp, "REV-06O-ANA05 MINOR 1")


def test_minor1_rows_must_name_the_run_folder(tmp_path):
    check_minor1(tmp_path)


@nc("the pre-ANA-06 binding: an entry beside a run folder bound by an equal warm_selection only")
def test_nc_minor1_reverted(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "run_rows_problem", lambda e, src, rid, asha: None)
    check_minor1(tmp_path)


def check_ft_warm_base(tmp):
    blk = anakit.build(tmp)
    edit_index(blk, lambda idx: [e.update(start_run_id="FL_FP_WARM_FROZEN_SMALL_s2026") for e in idx["entries"]
                                 if e["arm"] == "FT_FA" and e["seed"] == 2026])
    check_refuses(blk, tmp, "DL 287 m3")


def test_ft_fa_on_a_warm_line_base_refused(tmp_path):
    check_ft_warm_base(tmp_path)
    blk = anakit.build(tmp_path / "rows")
    _set_rows_field(blk, "MAIN/FT_C_s2026/RESULT_ROWS.json", "base_run_id", "FL_DP_WARM_FROZEN_eps8_SMALL_s2026")
    check_refuses(blk, tmp_path / "rows", "DL 287 m3")


@nc("a report without the DL 287 m3 rule admits an FT_FA from a warm-line base")
def test_nc_ft_warm_base_admitted(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "warm_base_run_problem", lambda arm, *n: None)
    check_ft_warm_base(tmp_path)


def test_warm_s_record_bound_and_content_checked(tmp_path):
    blk = n06_block(tmp_path / "a", dp1=False)
    add_s_record(blk, warm_s_record(S=1.0))
    rc, text = run(blk, tmp_path / "a" / "o")
    assert rc == 0, text
    sr = result(tmp_path / "a" / "o")["ana06"]["s_record_warm"]["S_RECORD_WARM_FROZEN"]
    assert sr["status"] == "BOUND" and len(sr["runs_bound"]) == 4                  # FA_1024 + DP8 x 2 seeds
    blk = n06_block(tmp_path / "b", dp1=False)
    add_s_record(blk, warm_s_record(S=2.0))
    check_refuses(blk, tmp_path / "b", "clip norm")
    blk = n06_block(tmp_path / "c", dp1=False)
    add_s_record(blk, warm_s_record(S=0.0))
    check_refuses(blk, tmp_path / "c", "REV-FL-16 m5")


@nc("a warm S record loader without the S content check (the self-sha check only)")
def test_nc_s_record_content_unchecked(tmp_path, monkeypatch):
    real = R.s_record_warm
    monkeypatch.setattr(R, "s_record_warm", lambda obj, v, f, rel: dict(real(dict(obj, S=1.0, **{
        "self_sha256": R._canon_compact(dict(obj, S=1.0))}), v, f, rel), S=obj["S"]))
    blk = n06_block(tmp_path, dp1=False)
    add_s_record(blk, warm_s_record(S=0.0))
    check_refuses(blk, tmp_path, "REV-FL-16 m5")


def test_s_cal_rounds_pinned_to_launch_fl():
    import v4int.launch_fl as F
    assert R.S_CAL_ROUNDS == F.KNOBS["s_cal_rounds"]


_S_BAD = [{"rounds_0based": list(range(19))}, {"rounds_0based": list(range(1, 21))},
          {"rounds_0based": [True] + list(range(1, 20))}, {"n": 0}, {"n": -5}, {"n": True}, {"rule": "mean"}]


def test_warm_s_content_mirror_refused(tmp_path):
    for i, over in enumerate(_S_BAD):
        blk = n06_block(tmp_path / str(i), dp1=False)
        add_s_record(blk, warm_s_record(**over))
        check_refuses(blk, tmp_path / str(i), "REV-FL-16 m5")


@pytest.mark.parametrize("over", [{"rounds_0based": list(range(19))}, {"rounds_0based": list(range(1, 21))},
                                  {"rounds_0based": [True] + list(range(1, 20))}, {"n": 0}, {"n": -5},
                                  {"n": True}, {"rule": "mean"}], ids=["short", "shifted", "bool_round", "n0", "nneg",
                                                                       "nbool", "rule"])
@nc("a warm S record whose S content no S calibration can write (rounds != 0..19, n_norms <= 0 / bool, non-median rule)")
def test_nc_warm_s_content_mirror(tmp_path, monkeypatch, over):
    real = R.s_record_warm
    good = {"n_norms": 20625, "rounds_0based": list(range(20)), "rule": "median"}

    def lax(obj, v, f, rel):                                   # a loader that lets the mutated field through
        fixed = dict(obj, **{k: good[k] for k in good if k in obj})
        fixed["self_sha256"] = R._canon_compact(fixed)
        return dict(real(fixed, v, f, rel))
    monkeypatch.setattr(R, "s_record_warm", lax)
    blk = n06_block(tmp_path, dp1=False)
    add_s_record(blk, warm_s_record(**over))
    check_refuses(blk, tmp_path, "REV-FL-16 m5")


@nc("a report without the 06N layer (no K20b* on FA_WARM*)")
def test_nc_06n_rows_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "n06_items", lambda star: [])
    blk = n06_block(tmp_path, dp1=False)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    assert any(c["id"] == "K20b*" for c in result(tmp_path / "o")["ana06"]["contrasts"])


@nc("a FA_WARM* rule without the 0.0005 tie band (a near-tie goes to S7)")
def test_nc_star_tie_band_removed(monkeypatch):
    monkeypatch.setattr(R, "STAR_TIE", 0.0)
    rec = {"_WARM_WINNER_CALIB": {"S7": {"value": 0.1704}, "S8": {"value": 0.1700}}}
    assert R.fa_warm_star(rec, "VALIDATION", None)["arm"] == "FA_WARM_FROZEN"


# ------------------------------------------------------------------------------------------------ the index builder
def warm_fl_dir(R_: Path, name: str, src_blk, arm: str, seed: int, *, launch_arm=None, eps=None):
    """An FL-16 run dir in the server layout: the anakit run (RUN_SUMMARY in launch_fl shape) + LAUNCH_0 with the S8
    warm binding (and dp_z for DP)."""
    d = R_ / "FL" / name
    shutil.copytree(src_blk.root / "MAIN" / f"FL_{arm}_s{seed}", d)
    s = json.loads((d / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    s["arm_spec"]["cfg"]["seed"] = seed
    anakit.jdump(d / "RUN_SUMMARY.json", s)
    la = {"label": "PRODUCTION", "arm": launch_arm or s["arm"], "fl05": {"role": "MAIN"}, "device_pin_sha256": anakit.PIN,
          "warm_selection": anakit.warm_binding(arm)}
    if eps is not None:
        la["dp_z"] = {"epsilon": eps}
    anakit.jdump(d / "LAUNCH_0.json", la)
    return d


def index_layout(tmp: Path):
    blk = n06_block(tmp / "blk", dp1=True)                         # the source of the warm FL run dirs
    lay = layoutkit.build_layout(tmp / "lay", anakit.build(tmp / "plain"))
    R_ = lay["R"]
    for seed in (2026, 2027):
        for arm in ("FP_WARM_FROZEN", "FA_1024_WARM_FROZEN"):
            warm_fl_dir(R_, f"{arm}_s{seed}", blk, arm, seed)
        warm_fl_dir(R_, f"DP8_FROZEN_s{seed}", blk, "DP8_FROZEN", seed, eps=8.0)
    warm_fl_dir(R_, "DP1_WARM_FROZEN_s2026", blk, "DP1_WARM_FROZEN", 2026, eps=1.0)
    for seed in (2026, 2027):                                       # 06N N3.3: the cold FP re-run (from scratch)
        for tag in ("PB", "CB"):                                    # (the plain layout's FP comes from REEVAL dirs)
            shutil.rmtree(R_ / "REEVAL_A40" / f"FP_s{seed}_{tag}")
        d = R_ / "FL" / f"N33_FP_s{seed}"
        shutil.copytree(R_ / "FL" / f"FA_s{seed}", d)
        s = json.loads((d / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
        anakit.jdump(d / "RUN_SUMMARY.json", dict(s, arm="FP"))
        anakit.jdump(d / "LAUNCH_0.json", {"label": "PRODUCTION", "arm": "FP", "fl05": {"role": "MAIN"},
                                           "device_pin_sha256": anakit.PIN})
    for name in ("FP_s2026", "FA_1024_s2026"):                     # 06N N4 / DL 317 D2: stopped, by name (no marker)
        (R_ / "FL" / name).mkdir()
        anakit.jdump(R_ / "FL" / name / "STATUS.json", {"status": "RUNNING"})
    d = R_ / "FL" / "PF_s2027"                                      # any dir with Ops' marker (Fable audit M5)
    d.mkdir()
    anakit.jdump(d / "STATUS.json", {"status": "RUNNING"})
    anakit.jdump(d / "STOPPED_REALLOCATED.json", {"status": "STOPPED_REALLOCATED", "replacement": "N33_PF_s2027"})
    rec = tmp / "srec"
    anakit.jdump(rec / "S_RECORD_WARM_FROZEN.json", warm_s_record())
    args = lay["args"] + [str(rec / "S_RECORD_WARM_FROZEN.json")]
    return blk, lay, R_, args


def test_index_maps_the_warm_line(tmp_path):
    _blk, _lay, R_, args = index_layout(tmp_path)
    lines = []
    assert BI.main(args + ["--out", str(tmp_path / "idx")], say=lines.append) == 0, lines
    idx = json.loads((tmp_path / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))
    keys = {(e["arm"], e["seed"], e["rule"]) for e in idx["entries"] if e["role"] == M}
    assert {("DP8_FROZEN", s, r) for s in (2026, 2027) for r in R.RULES} <= keys
    assert ("DP1_WARM_FROZEN", 2026, CB) in keys and ("FP_WARM_FROZEN", 2027, PB) in keys
    assert all(e.get("warm_selection", {}).get("stage") == "S8" for e in idx["entries"] if e["arm"] in S8_LINE)
    fp = [e for e in idx["entries"] if e["arm"] == "FP" and e.get("run_dir")]
    assert {e["seed"] for e in fp} == {2026, 2027} and all(e.get("cold_rerun") == "06N N3.3" for e in fp)
    exc = {Path(x["dir"]).name: x["reason"] for x in idx["excluded"]}
    assert "STOPPED_REALLOCATED" in exc["FP_s2026"] and "STOPPED_REALLOCATED" in exc["FA_1024_s2026"]
    assert "STOPPED_REALLOCATED" in exc["PF_s2027"]
    assert idx["records"]["S_RECORD_WARM_FROZEN"] == "records/S_RECORD_WARM_FROZEN.json"


@nc("an index builder without the FL-16 label map (the run's arm used as is)")
def test_nc_index_label_map_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(BI, "warm_line_block_arm", lambda d, label, launch: label)
    test_index_maps_the_warm_line(tmp_path)


@nc("an index builder without the DL 317 D2 name rule (the stopped dirs without a marker are read)")
def test_nc_index_stopped_dirs_read(tmp_path, monkeypatch):
    monkeypatch.setattr(BI, "STOPPED_REALLOCATED", ())
    test_index_maps_the_warm_line(tmp_path)


@nc("an index builder that ignores Ops' STOPPED_REALLOCATED.json marker (Fable audit M5)")
def test_nc_index_stopped_marker_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(BI, "STOPPED_MARKER", "NO_SUCH_MARKER.json")
    test_index_maps_the_warm_line(tmp_path)


def check_index_refuses(args, out, needle):
    lines = []
    assert BI.main(args + ["--out", str(out)], say=lines.append) == BI.REFUSED, lines
    assert needle in lines[0], lines


def test_index_warm_line_refusals(tmp_path):
    _blk, _lay, R_, args = index_layout(tmp_path)
    d = R_ / "FL" / "DP8_FROZEN_s2026"                                # eps 1 run in the eps-8 directory
    la = json.loads((d / "LAUNCH_0.json").read_text(encoding="utf-8"))
    anakit.jdump(d / "LAUNCH_0.json", dict(la, dp_z={"epsilon": 1.0}))
    check_index_refuses(args, tmp_path / "i1", "directory arm DP8_FROZEN")
    anakit.jdump(d / "LAUNCH_0.json", dict(la, dp_z={"epsilon": 4.0}))
    check_index_refuses(args, tmp_path / "i2", "not a registered warm-line arm")
    anakit.jdump(d / "LAUNCH_0.json", la)
    s = json.loads((d / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    anakit.jdump(d / "RUN_SUMMARY.json", dict(s, arm="DP"))           # the legacy DP + frozen tables shape
    check_index_refuses(args, tmp_path / "i3", "froze its item tables")
    anakit.jdump(d / "RUN_SUMMARY.json", s)
    p = R_ / "FL" / "FP_WARM_FROZEN_s2026" / "RUN_SUMMARY.json"
    s2 = json.loads(p.read_text(encoding="utf-8"))
    anakit.jdump(p, dict(s2, arm_spec=dict(s2["arm_spec"], frozen_item_tables=None)))
    check_index_refuses(args, tmp_path / "i4", "frozen_item_tables")
    anakit.jdump(p, s2)
    shutil.copytree(R_ / "FL" / "FA_s2026", R_ / "FL" / "N33_FA_s2026")   # FA is not a 06N N3.3 re-run arm
    check_index_refuses(args, tmp_path / "i5", "06N N3.3")
    shutil.rmtree(R_ / "FL" / "N33_FA_s2026")
    d = R_ / "DEV" / "FT_FA_s2026"                                     # DL 287 m3 on the index side
    anakit.jdump(d / "LAUNCH_0.json", {"label": "PRODUCTION", "arm": "FT_FA",
                                       "start": {"run_id": "FL_FP_WARM_FROZEN_SMALL_s2026", "kind": "fl"}})
    check_index_refuses(args, tmp_path / "i6", "DL 287 m3")


# ------------------------------------------------------------------------------------------------ the v4 (06P DRAFT) registry
V4_FILE = R.V4 / "company" / "gates" / "ANA_CONTRASTS_v4.json"


def draft_copy(tmp: Path) -> Path:
    obj = json.loads(V4_FILE.read_text(encoding="utf-8"))
    obj.pop("self_sha256")
    obj["draft"] = True
    p = tmp / "ANA_CONTRASTS_v4_draftcopy.json"
    anakit.jdump(p, anakit.canon(obj))
    return p


def test_v4_file_is_the_06p_registry(tmp_path):
    obj = json.loads(V4_FILE.read_text(encoding="utf-8"))
    assert obj == R.v4_registry_object() and obj["draft"] is False and R._self_ok(obj) == "VERIFIED"
    md = (R.V4 / "contracts" / "PREG1_AMENDMENT_06P.md").read_text(encoding="utf-8")
    assert len(obj["amendment_06P_text"]) == 7 and all(t in md for t in obj["amendment_06P_text"])   # verbatim
    assert R._sha_file(R.V4 / "contracts" / "PREG1_AMENDMENT_06P.md") == obj["amendment_06P_file_sha256"]
    assert obj["hierarchy"]["primary"] == ["K20b", "K21"] and "K2" in obj["hierarchy"]["secondary"]
    assert {"K4w", "K5w", "K12w", "K13w", "K14w", "K_INIT", "P5"} <= set(obj["hierarchy"]["secondary"])
    assert len(obj["confirmatory_contrasts"]) == 33 and obj["holm_family"] == list(R.HOLM_FAMILY_V4)
    erratum = ["K1", "K2", "K3", "K8", "K9b", "K9c", "K11b", "K12", "K17", "K20", "K20c", "K_INIT", "P5", "K4w", "K5w",
               "K12w", "K13w", "K14w"]                                     # 06P ERRATUM 1 (DL 324): the binding list
    assert len(obj["holm_family"]) == 18 and obj["holm_family"] == erratum and obj["amendment_06P_erratum_1"] in md
    assert sorted(obj["hierarchy"]["secondary"]) == sorted(erratum)
    assert len(obj["hierarchy"]["tertiary"]) == 13 and obj["registered_seeds"]["K13w"] == [2026]
    assert R.load_contrasts(V4_FILE, split="TEST")["n_contrasts"] == 33                 # registered: TEST too
    dr = draft_copy(tmp_path)
    assert {"K20b": "R(FA_WARM_FROZEN / C_FULL)", "K21": "FA_WARM_FROZEN − PRE"}.items() <= \
        {c["id"]: c["contrast"] for c in obj["confirmatory_contrasts"]}.items()
    c = R.load_contrasts(dr, split="VALIDATION", allow_draft=True)
    assert c["format"] == R.CONTRAST_FORMAT_V4 and c["n_contrasts"] == 33 and c["draft"] is True
    for kw in (dict(split="VALIDATION"), dict(split="TEST", allow_draft=True)):
        with pytest.raises(R.AnaRefused, match="DRAFT"):
            R.load_contrasts(dr, **kw)
    bad = dict(obj, hierarchy=dict(obj["hierarchy"], primary=["K2"]))
    anakit.jdump(tmp_path / "bad.json", bad)
    with pytest.raises(R.AnaRefused, match="hierarchy"):
        R.load_contrasts(tmp_path / "bad.json", split="VALIDATION", allow_draft=True)


def check_draft_refused(tmp):
    refused, dr = 0, draft_copy(tmp)
    for kw in (dict(split="VALIDATION"), dict(split="TEST", allow_draft=True)):
        try:
            R.load_contrasts(dr, **kw)
        except R.AnaRefused:
            refused += 1
    assert refused == 2


@nc("a v4 loader that ignores the draft flag")
def test_nc_draft_loads_by_default(tmp_path, monkeypatch):
    real = R.load_contrasts_v4
    monkeypatch.setattr(R, "load_contrasts_v4", lambda obj, raw, *, split, allow_draft: real(obj, raw, split="VALIDATION",
                                                                                            allow_draft=True))
    check_draft_refused(tmp_path)


@pytest.fixture(scope="module")
def n06_v4(n06, tmp_path_factory):
    blk = n06[0]
    tmp = tmp_path_factory.mktemp("n06v4")
    shutil.copy2(V4_FILE, tmp / "v4.json")
    old = blk.contrasts
    blk.contrasts = tmp / "v4.json"
    try:
        rc, text = run(blk, tmp / "out")
    finally:
        blk.contrasts = old
    assert rc == 0, text
    return blk, result(tmp / "out"), n06[1]


def test_v4_makes_the_06n_rows_confirmatory(n06_v4):
    blk, res, res3 = n06_v4
    conf = {c["id"]: c for c in res["confirmatory"]}
    assert len(conf) == 33 and res["labels"]["hierarchy"]["K20b"] == "primary" and \
        res["labels"]["hierarchy"]["K2"] == "secondary"
    k = conf["K20b"]
    assert k["contrast"] == "R(FA_WARM_FROZEN / C_FULL)" and k["by_rule"][PB]["label"] == "CONFIRMATORY"
    assert k["by_rule"][PB]["point"] == pytest.approx(_row(res3, "K20b*")["by_rule"][PB]["point"], abs=1e-12)
    assert conf["K12w"]["confirmatory_rule"] == CB and conf["K13w"]["registered_seeds"] == [2026]
    assert conf["K4w"]["by_rule"][PB]["point"] == pytest.approx(
        np.mean([100.0 * pbv(blk, "FP_WARM_FROZEN", s) / pbv(blk, "C_FULL", 2026) for s in (2026, 2027)]), abs=1e-9)
    assert conf["P5"]["by_rule"][PB]["status"] == "NOT_AVAILABLE"                   # no FT_FA_WARM_FROZEN entry here
    assert res["holm_band"]["family_size"] == 18 and k["by_rule"][PB]["holm"]["status"] == "NOT_IN_HOLM_FAMILY"
    assert res["holm_band"]["family"] == list(R.HOLM_FAMILY_V4) and res["holm_band"]["tertiary_band"]["family_size"] == 13
    assert conf["K4w"]["by_rule"][PB]["holm"]["status"] == "COMPUTED"
    assert conf["K14"]["by_rule"][PB]["holm"].get("family") == "TERTIARY_BAND"
    assert res["ana06"]["status"] == "IN_LOADED_FILE" and res["inputs"]["contrasts_format"] == R.CONTRAST_FORMAT_V4
    assert {d["id"] for d in res["registered_descriptive"]} == set(R.V4_DESCRIPTIVE_IDS)
    assert R.HIERARCHY == R._V3_PROFILE["HIERARCHY"] and R.HIERARCHY.get("K2") == "primary"   # v3 restored


def test_v4_refuses_an_s7_star(tmp_path):
    blk = n06_block(tmp_path, v7=0.1720, v8=0.1700, dp1=False)
    shutil.copy2(V4_FILE, tmp_path / "v4.json")
    blk.contrasts = tmp_path / "v4.json"
    rc, text = run(blk, tmp_path / "o")
    assert rc == R.REFUSED and "06P P1" in text, text


def check_v3_after_v4(tmp):
    out = tmp / "claims"
    for f, extra in ((V4_FILE, []), (anakit.V3_FILE, [])):
        o = out / f.stem
        assert R.main(["--claims-only", "--contrasts", str(f), "--out", str(o), *extra], say=lambda *_: None) == 0
    v3 = json.loads((out / anakit.V3_FILE.stem / R.CLAIM_NAME).read_text(encoding="utf-8"))
    assert {c["id"]: c["tier"] for c in v3["contrasts"]}["K2"] == "primary"
    assert R._sha_file(out / anakit.V3_FILE.stem / R.CLAIM_NAME).startswith("9ce7103a")   # the r6 v3 claims output


def test_claims_only_v3_unchanged_after_v4(tmp_path):
    check_v3_after_v4(tmp_path)


@nc("a run that leaves the v4 registry tables in place (no v3 restore)")
def test_nc_profile_not_restored(tmp_path, monkeypatch):
    for k in R.V4_PROFILE:                                     # monkeypatch restores every table after the test
        monkeypatch.setattr(R, k, getattr(R, k))

    def no_restore(fmt):                                       # MUTANT: switch to v4, never back
        if fmt == R.CONTRAST_FORMAT_V4:
            for k, v in R.V4_PROFILE.items():
                setattr(R, k, v)
    monkeypatch.setattr(R, "set_registry_profile", no_restore)
    check_v3_after_v4(tmp_path)


# ------------------------------------------------------------------------------------------------ REV-ANA-06 minors
def test_index_dp8_frozen_needs_two_seeds_and_stopped_reeval_refused(tmp_path):
    _blk, _lay, R_, args = index_layout(tmp_path)
    shutil.rmtree(R_ / "FL" / "DP8_FROZEN_s2027")                     # m1: DP8_FROZEN is a 2-seed arm (06N / 06P K12w)
    check_index_refuses(args, tmp_path / "i1", "DP8_FROZEN/")
    _blk, _lay, R_, args = index_layout(tmp_path / "b")              # m2: a REEVAL of a STOPPED dir is never evidence
    for name, marker in (("FP_s2026", False), ("OLD_FP_RUN", True)):
        stopped = R_ / "FL_T_SRC" / name
        stopped.mkdir(parents=True)
        if marker:
            anakit.jdump(stopped / "STOPPED_REALLOCATED.json", {"status": "STOPPED_REALLOCATED"})
        p = R_ / "REEVAL_A40" / "C_SMALL_s2026_PB" / "REEVAL_RECORD.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        anakit.jdump(p, dict(rec, inputs={"source": {"run_dir": str(stopped), "launch_files": {}}}))
        check_index_refuses(args, tmp_path / f"i2{name}", "STOPPED_REALLOCATED")


def test_v4_claim_register_lists_k14w_equivalent(tmp_path):
    for f in (V4_FILE, anakit.V3_FILE):
        assert R.main(["--claims-only", "--contrasts", str(f), "--out", str(tmp_path / f.stem)], say=lambda *_: None) == 0
    v4 = json.loads((tmp_path / V4_FILE.stem / R.CLAIM_NAME).read_text(encoding="utf-8"))
    v3 = json.loads((tmp_path / anakit.V3_FILE.stem / R.CLAIM_NAME).read_text(encoding="utf-8"))
    assert "K14w" in v4["flags"]["EQUIVALENT"] and "K14 " in v4["flags"]["EQUIVALENT"]
    assert v3["flags"]["EQUIVALENT"] == "K14 only (06L-C C15)"


def check_p5_best_only(tmp):
    import test_ana05 as T5
    blk = n06_block(tmp, dp1=False)
    T5.add_ft_warm(blk, "FT_FA_WARM_FROZEN", (2026,))
    T5.add_ft_warm(blk, "FT_FA_WARM_FROZEN", (2027,), sb_over={"checkpoint": "endpoint_6.0.pt"})   # not best.pt
    shutil.copy2(V4_FILE, tmp / "v4.json")
    blk.contrasts = tmp / "v4.json"
    rc, text = run(blk, tmp / "o")
    assert rc == 0, text
    p5 = next(c for c in result(tmp / "o")["confirmatory"] if c["id"] == "P5")["by_rule"][PB]
    want = pbv(blk, "FT_FA_WARM_FROZEN", 2026) - pbv(blk, "FA_WARM_FROZEN", 2026)
    assert p5["seeds"] == [2026] and p5["point"] == pytest.approx(want, abs=1e-12)          # the best.pt base only
    assert p5["seeds_status"] == "INCOMPLETE_SEEDS" and p5["missing_registered_seeds"] == [2027]


def test_p5_pairs_only_a_best_pt_base(tmp_path):
    check_p5_best_only(tmp_path)


@nc("a P5 without the 06P P5 best.pt base filter (an endpoint-based FT run pairs)")
def test_nc_p5_any_base(tmp_path, monkeypatch):
    monkeypatch.setitem(R.V4_PROFILE, "BASE_BEST_ONLY", ())
    check_p5_best_only(tmp_path)

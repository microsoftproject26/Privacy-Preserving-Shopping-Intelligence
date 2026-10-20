"""FL-05 item 3 — v4int.lr_screen (amendment 06B §1): the local-LR selection from three CALIBRATION FA runs, and its
use by launch_fl --lr-selection (CPU, the tiny FL world of test_fl05_launch).

Proves: the rule (best PRACTICAL_BEST value; |delta| <= 0.0005 ties -> closest to 1, order 1, 0.5, 2); the O_EXCL,
self-hashed LR_SELECTION.json with the three values and the run dirs' RESULT_ROWS / RUN_SUMMARY sha256; every run is
verified (status, FA / CALIBRATION / 2026, spec digest, 12 marks, the one PRACTICAL_BEST E2E macro mrr@20 row = the
run's own evaluation at its best mark, one evaluation manifest, specs identical apart from the multiplier);
launch_fl verifies the file (format, self-sha256, rule recomputed). Negative control: plain argmax ignores the tie rule.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import v4int.launch_fl as F
import v4int.lr_screen as L
from intkit import assert_raises, nc
from test_fl05_launch import fl05_world, run_tiny, tiny_spec

CALIB_EVAL = "M06_CALIB_VALIDATION_WARM"


@pytest.fixture(scope="module")
def screen(tmp_path_factory):
    root = tmp_path_factory.mktemp("fl05_screen")
    dirs = {}
    for m in (0.5, 1.0, 2.0):
        name = f"x{F.LR_MULT_TAG[m]}"
        out = run_tiny(root, tiny_spec("FA", role="CALIBRATION", lr_mult=m), name,
                       world=fl05_world(eval_id=CALIB_EVAL))
        assert out["status"] == "DONE_MAIN"
        dirs[m] = root / name
    main = run_tiny(root, tiny_spec("FA"), "main", world=fl05_world(eval_id=CALIB_EVAL))
    assert main["status"] == "DONE_MAIN"
    return {"root": root, "dirs": dirs, "main": root / "main"}


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def test_apply_rule_and_ties():
    assert L.apply_rule({0.5: 0.100, 1.0: 0.101, 2.0: 0.105}) == (2.0, [2.0])
    assert L.apply_rule({0.5: 0.100, 1.0: 0.1046, 2.0: 0.105}) == (1.0, [1.0, 2.0])       # 0.0004 -> tie
    assert L.apply_rule({0.5: 0.100, 1.0: 0.1044, 2.0: 0.105}) == (2.0, [2.0])            # 0.0006 -> no tie
    assert L.apply_rule({0.5: 0.1047, 1.0: 0.1040, 2.0: 0.105}) == (0.5, [0.5, 2.0])      # 1 out; 0.5 before 2
    assert L.apply_rule({0.5: 0.2, 1.0: 0.2, 2.0: 0.2}) == (1.0, [1.0, 0.5, 2.0])
    assert L.apply_rule({0.5: 0.3, 1.0: 0.1, 2.0: 0.2}) == (0.5, [0.5])
    assert_raises(L.SelectionRefused, L.apply_rule, {0.5: 0.1, 1.0: 0.1})


def test_select_writes_a_verified_selection_once(screen, tmp_path):
    runs = [screen["dirs"][m] for m in (2.0, 0.5, 1.0)]                         # any order
    out = tmp_path / "LR_SELECTION.json"
    said = []
    assert L.main(["select", "--runs", *map(str, runs), "--out", str(out)], say=said.append) == 0
    obj = _json(out)
    rec = json.loads(said[0])
    assert set(rec) == {"path", "sha256", "selected_lr_mult", "self_sha256"}                # no value printed
    # FL-09: the CLI writes the 06F §3 EXT format; over {0.5, 1, 2} its rule is the 06B rule (same order 1, 0.5, 2)
    assert obj["format"] == F.LR_SELECTION_EXT_FORMAT and obj["self_sha256"] == F._canon(obj)
    vals = {m: obj["values"][F.LR_MULT_TAG[m]] for m in (0.5, 1.0, 2.0)}
    assert (obj["selected_lr_mult"], obj["tie_set"]) == L.apply_rule(vals) == L.apply_rule_ext(vals)
    for r in obj["runs"]:
        d = Path(r["run_dir"])
        assert r["result_rows_sha256"] == L._sha(d / "RESULT_ROWS.json")
        assert r["run_summary_sha256"] == L._sha(d / "RUN_SUMMARY.json")
        best_row = [x for x in _json(d / "RESULT_ROWS.json")["rows"] if x["selection_rule"] == "PRACTICAL_BEST"
                    and x["aggregation"] == "macro" and x["target_denominator"] == "E2E" and x["metric"] == "mrr"][0]
        assert r["value"] == best_row["value"] == vals[r["lr_mult"]]
        assert r["max_of_12_marks"] >= r["value"] and r["max_of_12_marks"] - r["value"] <= 0.0002 + 1e-12
    assert [r["lr_mult"] for r in obj["runs"]] == [1.0, 0.5, 2.0]
    assert L.main(["select", "--runs", *map(str, runs), "--out", str(out)], say=said.append) == 2      # O_EXCL
    assert F.load_lr_selection(out)["selected_lr_mult"] == obj["selected_lr_mult"]
    argv = ["--arm", "FP", "--seed", "2027", "--gpu", "0", "--run-dir", str(tmp_path / "r"),
            "--history-policy", "USER_HISTORY_L50", "--lr-selection", str(out)]
    args = F.parse_args(argv)
    m, prov = F.resolve_lr_mult(args)
    assert m == obj["selected_lr_mult"] and prov["self_sha256"] == obj["self_sha256"]
    assert prov["sha256"] == L._sha(out)
    # FIX-DELTA-01 F2: a non-smoke MAIN spec refuses an unregistered file; the smoke spec shows the wiring
    assert_raises(F.LaunchRefused, F.resolve_spec, args, 0.000711, fused=True)
    spec, extra = F.resolve_spec(F.parse_args(argv + ["--smoke"]), 0.000711, fused=True)
    assert spec["lr_mult"] == m and extra["lr_mult_source"]["source"] == "lr_selection"
    assert "main_selection" not in extra


def test_selection_file_tamper_refused(screen, tmp_path):
    out = tmp_path / "S.json"
    L.write_selection([screen["dirs"][m] for m in (0.5, 1.0, 2.0)], out)
    obj = _json(out)
    bad = dict(obj, values=dict(obj["values"], **{"2": obj["values"]["2"] + 1.0}))   # value edited, stale self hash
    p1 = tmp_path / "t1.json"
    p1.write_text(json.dumps(bad), encoding="utf-8")
    assert_raises(F.LaunchRefused, F.load_lr_selection, p1)
    wrong = [m for m in (0.5, 1.0, 2.0) if m != obj["selected_lr_mult"]][0]
    bad2 = dict(obj, selected_lr_mult=wrong)                                     # re-hashed, but not the rule's choice
    bad2["self_sha256"] = F._canon(bad2)
    p2 = tmp_path / "t2.json"
    p2.write_text(json.dumps(bad2), encoding="utf-8")
    assert_raises(F.LaunchRefused, F.load_lr_selection, p2)


def test_selection_refuses_bad_run_sets(screen, tmp_path):
    d = screen["dirs"]
    sel = lambda runs: L.selection(runs)                                          # noqa: E731
    assert_raises(L.SelectionRefused, sel, [d[0.5], d[1.0]])
    assert_raises(L.SelectionRefused, sel, [d[0.5], d[0.5], d[2.0]])
    assert_raises(L.SelectionRefused, sel, [d[0.5], screen["main"], d[2.0]])      # a MAIN-role run
    t = tmp_path / "tampered"
    shutil.copytree(d[2.0], t)
    rr = _json(t / "RESULT_ROWS.json")
    for r in rr["rows"]:
        if r["selection_rule"] == "PRACTICAL_BEST":
            r["value"] = r["value"] + 0.01                                        # no longer the run's own evaluation
    (t / "RESULT_ROWS.json").write_text(json.dumps(rr), encoding="utf-8")
    assert_raises(L.SelectionRefused, sel, [d[0.5], d[1.0], t])
    other = run_tiny(tmp_path, tiny_spec("FA", rounds=48, role="CALIBRATION", lr_mult=2.0), "o",
                     world=fl05_world(eval_id=CALIB_EVAL))
    assert other["status"] == "DONE_MAIN"
    assert_raises(L.SelectionRefused, sel, [d[0.5], d[1.0], tmp_path / "o"])      # differs beyond the multiplier


# ------------------------------------------------------------------------------------------------ negative control
@nc("plain argmax (no tie rule) picks a different multiplier inside the 0.0005 tie band")
def test_nc_argmax_without_tie_rule():
    vals = {0.5: 0.0990, 1.0: 0.1000, 2.0: 0.1003}
    argmax = max(vals, key=vals.get)                                              # MUTANT: no tie rule
    assert argmax == L.apply_rule(vals)[0]

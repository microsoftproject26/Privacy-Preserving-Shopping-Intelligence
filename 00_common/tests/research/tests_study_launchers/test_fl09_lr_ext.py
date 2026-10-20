"""FL-09 (PREG1_AMENDMENT_06F §1, §3) — extended local-LR multipliers and the extended selection file (CPU, the tiny FL
world of test_fl05_launch).

Proves: the registered multiplier set is exactly {0.5, 1, 2, 4, 8, 16, 32} (anything else refused; CALIB run ids use the
tag); every arm-spec digest and fl RunConfig digest at m in {0.5, 1, 2} (all arms, seeds, fused modes, DP z, the CALIB
screen: 96 pairs) is byte-identical to the PRE-FL-09 code (aggregate sha256 below, computed 2026-09-29 from launch_fl.py
sha256 2ab8c9bf74aad694392dae97917f842fa9c6bf19e46787a7d5767fda80112c79); the 06F §3 rule (best value; ties
|delta| <= 0.0005 -> smallest |log2 m|, then the smaller m); `lr_screen select` over >= 3 verified runs writes the O_EXCL
self-hashed PPSI_V4_FL_LR_SELECTION_EXT_1 file (values, screened, excluded, tie set, first-screen sha256) and prints only
m + sha; a non-finite run is excluded and recorded; launch_fl verifies the EXT file (rule re-applied); once
LR_SELECTION_EXT.json exists a non-smoke MAIN run refuses the 06B file. Negative controls: an unregistered multiplier
accepted; a missing first-screen run accepted; the 06B file accepted for MAIN while the EXT file exists; plain argmax.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest

import v4int.launch_fl as F
import v4int.lr_screen as L
from intkit import assert_raises, nc
from test_fl05_launch import _args, fl05_world, run_tiny, tiny_spec

PEAK_REG = 0.000711
CALIB_EVAL = "M06_CALIB_VALIDATION_WARM"
PRE_FL09_SHA = "d5d96769283a969c3ebf2453a88aa60a0839b8d5892073e8f6d54997b1b2caf1"   # 96 (spec, RunConfig) pairs


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def old_grid_digests() -> dict:
    out = {}
    for arm in ("FA", "FP", "PF", "FA_Q8", "FA_1024", "DP", "S_CAL"):
        for seed in (2026, 2027):
            if arm == "S_CAL" and seed != 2026:
                continue
            for fused in (True, None):
                for z in ((0.5642, 1.3094) if arm == "DP" else (None,)):
                    for m in (0.5, 1.0, 2.0):
                        kw = {}
                        if arm in ("FA_1024", "DP"):
                            kw["clip_norm"] = 1.2345
                        if arm == "DP":
                            kw["z"] = z
                        s = F.arm_spec(arm, seed=seed, peak_lr=PEAK_REG, fused=fused, lr_mult=m, **kw)
                        out[f"{arm}|{seed}|{fused}|{z}|{m}|MAIN"] = [
                            F.spec_digest(s), F.fl_run_config(s, f"FL_{arm}_SMALL_s{seed}", 64).digest()]
    for m in (0.5, 1.0, 2.0):
        for fused in (True, None):
            s = F.arm_spec("FA", seed=2026, peak_lr=PEAK_REG, fused=fused, lr_mult=m, role="CALIBRATION")
            out[f"FA|2026|{fused}|None|{m}|CALIBRATION"] = [F.spec_digest(s),
                                                            F.fl_run_config(s, "FL_FA_LRSCREEN", 64).digest()]
    return out


# ------------------------------------------------------------------------------------------------ item 1: multipliers
def test_registered_multiplier_set_and_tags():
    assert F.LR_MULTS == (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    assert [F.LR_MULT_TAG[m] for m in F.LR_MULTS] == ["0.5", "1", "2", "4", "8", "16", "32"]
    for m in F.LR_MULTS:
        assert F.lr_mult_value(m) == m and F.lr_mult_value(F.LR_MULT_TAG[m]) == m
    for bad in (3, 6, 64, 0.25, 1.5, 0.0, -4, float("nan"), float("inf"), "four", None):
        assert_raises(F.LaunchRefused, F.lr_mult_value, bad)
    for m in (4.0, 8.0, 16.0, 32.0):                             # the new values scale the local LR only
        base = F.arm_spec("PF", seed=2026, peak_lr=PEAK_REG, fused=True)
        s = F.arm_spec("PF", seed=2026, peak_lr=PEAK_REG, fused=True, lr_mult=m)
        assert s["cfg"]["lr_peak_local"] == m * PEAK_REG == s["cfg"]["solver"]["lr"]
        assert s["cfg"]["pf"] == base["cfg"]["pf"] and s["knobs"] == base["knobs"] and s["plan"] == base["plan"]
        assert s["lr_mult"] == m and F.spec_digest(s) != F.spec_digest(base)
    for m in ("4", "8", "16", "32"):
        a = _args(Path("."), "--role", "CALIBRATION", "--lr-mult", m, smoke=False)
        assert F.default_run_id(a) == f"FL_FA_LRSCREEN_x{m}_CALIB_s2026"
        F.fl05_preflight(a)
    assert_raises(F.LaunchRefused, F.fl05_preflight, _args(Path("."), "--role", "CALIBRATION", "--lr-mult", "64"))


def test_old_grid_digests_byte_identical_to_pre_fl09_code():
    d = old_grid_digests()
    assert len(d) == 96
    assert hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest() == PRE_FL09_SHA


# ------------------------------------------------------------------------------------------------ item 3: the rule
def test_apply_rule_ext_ties_by_log2_distance_then_smaller_m():
    R = L.apply_rule_ext
    assert R({0.5: 0.10, 1.0: 0.11, 2.0: 0.12, 4.0: 0.13, 8.0: 0.14}) == (8.0, [8.0])
    assert R({1.0: 0.1000, 2.0: 0.1004, 4.0: 0.1003}) == (1.0, [1.0, 2.0, 4.0])          # all within 0.0005
    assert R({0.5: 0.2000, 1.0: 0.1000, 2.0: 0.2000}) == (0.5, [0.5, 2.0])               # |log2| tie -> smaller m
    assert R({4.0: 0.2000, 8.0: 0.2001, 16.0: 0.2004}) == (4.0, [4.0, 8.0, 16.0])
    assert R({4.0: 0.2000, 8.0: 0.2001, 16.0: 0.2006}) == (16.0, [16.0])                 # 0.0006 -> no tie
    assert R({0.5: 0.3, 1.0: 0.3, 2.0: 0.3, 4.0: 0.3, 8.0: 0.3, 16.0: 0.3, 32.0: 0.3})[1] == \
        [1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 32.0]
    for vals in ({0.5: 0.1, 1.0: 0.1047, 2.0: 0.105}, {0.5: 0.2, 1.0: 0.2, 2.0: 0.2}, {0.5: 0.3, 1.0: 0.1, 2.0: 0.2}):
        assert R(vals) == L.apply_rule(vals)                                            # extends the 06B order
    assert_raises(L.SelectionRefused, R, {})
    assert_raises(L.SelectionRefused, R, {3.0: 0.1, 1.0: 0.1})
    assert_raises(L.SelectionRefused, R, {1.0: float("nan"), 2.0: 0.1})


# ------------------------------------------------------------------------------------------------ item 3: selection
@pytest.fixture(scope="module")
def screen(tmp_path_factory):
    root = tmp_path_factory.mktemp("fl09_screen")
    dirs = {}
    for m in (0.5, 1.0, 2.0, 4.0, 8.0):
        name = f"x{F.LR_MULT_TAG[m]}"
        out = run_tiny(root, tiny_spec("FA", role="CALIBRATION", lr_mult=m), name, world=fl05_world(eval_id=CALIB_EVAL))
        assert out["status"] == "DONE_MAIN"
        dirs[m] = root / name
    first = root / "contracts_fl" / "LR_SELECTION.json"                           # the 06B file (three runs)
    first.parent.mkdir()
    L.write_selection([dirs[m] for m in (0.5, 1.0, 2.0)], first)
    return {"root": root, "dirs": dirs, "first": first}


def _vals(dirs):
    return {m: L.read_run(d)["value"] for m, d in dirs.items()}


def test_select_ext_writes_verified_file_once(screen, tmp_path):
    d = screen["dirs"]
    out = tmp_path / F.LR_SELECTION_EXT_NAME
    said = []
    argv = ["select", "--runs", *(str(d[m]) for m in (8.0, 0.5, 4.0, 1.0, 2.0)), "--first-screen", str(screen["first"]),
            "--out", str(out)]
    assert L.main(argv, say=said.append) == 0
    rec = json.loads(said[0])
    assert set(rec) == {"path", "sha256", "selected_lr_mult", "self_sha256"}                    # no value printed
    obj = _json(out)
    assert obj["format"] == F.LR_SELECTION_EXT_FORMAT and obj["self_sha256"] == F._canon(obj)
    assert rec["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    vals = _vals(d)
    assert obj["values"] == {F.LR_MULT_TAG[m]: vals[m] for m in (1.0, 0.5, 2.0, 4.0, 8.0)}
    assert obj["screened"] == [1.0, 0.5, 2.0, 4.0, 8.0] and obj["excluded"] == []
    assert (obj["selected_lr_mult"], obj["tie_set"]) == L.apply_rule_ext(vals)
    assert obj["first_screen_file_sha256"] == hashlib.sha256(screen["first"].read_bytes()).hexdigest()
    assert obj["edge"]["largest_screened"] == 8.0 and obj["edge"]["next_registered_multiplier"] == 16.0
    assert [r["lr_mult"] for r in obj["runs"]] == obj["screened"]
    for r in obj["runs"]:
        assert r["result_rows_sha256"] == L._sha(Path(r["run_dir"]) / "RESULT_ROWS.json")
    assert L.main(argv, say=said.append) == 2                                                   # O_EXCL
    # launch_fl: the EXT file verifies, its m is used, the provenance records the format
    assert F.load_lr_selection(out)["selected_lr_mult"] == obj["selected_lr_mult"]
    argv = ["--arm", "FP", "--seed", "2027", "--gpu", "0", "--run-dir", str(tmp_path / "r"),
            "--history-policy", "USER_HISTORY_L50", "--lr-selection", str(out)]
    args = F.parse_args(argv)
    # FIX-DELTA-01 F2: this file is not the registered contracts/fl file (and lacks x16): a non-smoke MAIN run refuses
    assert_raises(F.LaunchRefused, F.require_lr_selection, args)
    m, prov = F.resolve_lr_mult(args)
    assert m == obj["selected_lr_mult"] and prov["format"] == F.LR_SELECTION_EXT_FORMAT
    spec, extra = F.resolve_spec(F.parse_args(argv + ["--smoke"]), PEAK_REG, fused=True)     # the wiring (smoke)
    assert spec["lr_mult"] == m and extra["lr_mult_source"]["self_sha256"] == obj["self_sha256"]


def test_ext_file_tamper_refused(screen, tmp_path):
    d = screen["dirs"]
    obj = L.selection_ext([d[m] for m in (0.5, 1.0, 2.0, 4.0)])
    assert obj["first_screen_file_sha256"] is None

    def write(o, name):
        p = tmp_path / name
        p.write_text(json.dumps(o), encoding="utf-8")
        return p
    F.load_lr_selection(write(obj, "ok.json"))
    stale = dict(obj, values=dict(obj["values"], **{"4": obj["values"]["4"] + 1.0}))          # stale self hash
    assert_raises(F.LaunchRefused, F.load_lr_selection, write(stale, "t1.json"))
    for field, val in (("selected_lr_mult", [m for m in (0.5, 1.0, 2.0, 4.0) if m != obj["selected_lr_mult"]][0]),
                       ("tie_set", obj["tie_set"] + [16.0]), ("screened", obj["screened"] + [16.0]),
                       ("values", dict(obj["values"], **{"3": 0.5}))):
        bad = dict(obj, **{field: val})
        bad["self_sha256"] = F._canon(bad)                                                      # re-hashed
        assert_raises(F.LaunchRefused, F.load_lr_selection, write(bad, f"t_{field}.json"))


def test_selection_ext_refusals_and_exclusion(screen, tmp_path):
    d = screen["dirs"]
    S = L.selection_ext
    assert_raises(L.SelectionRefused, S, [d[1.0], d[2.0]])                                      # < 3 runs
    assert_raises(L.SelectionRefused, S, [d[1.0], d[2.0], d[2.0]])                              # m twice
    assert_raises(L.SelectionRefused, S, [d[1.0], d[2.0], d[4.0]], screen["first"])            # 0.5 missing
    t = tmp_path / "x8_nonfinite"
    shutil.copytree(d[8.0], t)
    rr = _json(t / "RESULT_ROWS.json")
    for r in rr["rows"]:
        if r["selection_rule"] == "PRACTICAL_BEST":
            r["value"] = None                                                                   # a non-finite metric
    (t / "RESULT_ROWS.json").write_text(json.dumps(rr), encoding="utf-8")
    obj = S([d[0.5], d[1.0], d[2.0], d[4.0], t], screen["first"])
    assert obj["screened"] == [1.0, 0.5, 2.0, 4.0, 8.0] and "8" not in obj["values"]
    assert [e["lr_mult"] for e in obj["excluded"]] == [8.0] and "non-finite" in obj["excluded"][0]["reason"]
    assert obj["selected_lr_mult"] != 8.0 and obj["selected_lr_mult"] == L.apply_rule_ext(
        {m: v for m, v in _vals({k: d[k] for k in (0.5, 1.0, 2.0, 4.0)}).items()})[0]
    p = tmp_path / "ext.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    assert F.load_lr_selection(p)["selected_lr_mult"] == obj["selected_lr_mult"]              # excluded m verified
    assert_raises(L.SelectionRefused, L.selection, [d[0.5], d[1.0], t])                        # 06B path: refused


# ------------------------------------------------------------------------------------------------ item 2: 06B superseded
def _main_args(tmp_path, sel, smoke=False, *extra):
    return _args(tmp_path, "--lr-selection", str(sel), *extra, smoke=smoke, arm="FP")


def test_main_run_refuses_06b_file_once_ext_exists(screen, tmp_path, monkeypatch):
    monkeypatch.setattr(F, "LR_SELECTION_EXT_PATH", tmp_path / "registered" / F.LR_SELECTION_EXT_NAME)
    monkeypatch.setattr(F, "FL_TUNING_PATH", tmp_path / "registered" / F.FL_TUNING_NAME)
    d = screen["dirs"]
    folder = tmp_path / "contracts_fl"
    folder.mkdir()
    first = folder / "LR_SELECTION.json"
    shutil.copy(screen["first"], first)
    # FIX-DELTA-01 F2 (06F §1): the 06B file is refused for MAIN outright, even before any EXT file exists
    assert_raises(F.LaunchRefused, F.require_lr_selection, _main_args(tmp_path, first))
    ext = folder / F.LR_SELECTION_EXT_NAME
    L.write_selection_ext([d[m] for m in (0.5, 1.0, 2.0, 4.0)], ext, first)
    assert_raises(F.LaunchRefused, F.require_lr_selection, _main_args(tmp_path, first))       # sibling EXT exists
    assert_raises(F.LaunchRefused, F.require_lr_selection, _main_args(tmp_path, ext))         # F2: not registered
    F.require_lr_selection(_main_args(tmp_path, first, True))                                  # smoke: not refused
    other = tmp_path / "elsewhere"
    other.mkdir()
    shutil.copy(screen["first"], other / "LR_SELECTION.json")
    assert_raises(F.LaunchRefused, F.require_lr_selection, _main_args(tmp_path, other / "LR_SELECTION.json"))
    F.LR_SELECTION_EXT_PATH.parent.mkdir()
    shutil.copy(ext, F.LR_SELECTION_EXT_PATH)                                                  # the registered path
    assert_raises(F.LaunchRefused, F.require_lr_selection, _main_args(tmp_path, other / "LR_SELECTION.json"))
    assert_raises(F.LaunchRefused, F.require_lr_selection, _main_args(tmp_path, tmp_path / "missing.json"))
    assert_raises(F.LaunchRefused, F.run, _main_args(tmp_path, first), {}, {})                 # before any import


# ------------------------------------------------------------------------------------------------ negative controls
@nc("an unregistered multiplier (64) is accepted")
def test_nc_unregistered_multiplier_accepted():
    try:
        F.lr_mult_value(64)
        accepted = True
    except F.LaunchRefused:
        accepted = False
    assert accepted, "64 was refused (as it must be)"


@nc("the extended selection accepts a run set that omits a first-screen run")
def test_nc_missing_screened_run_accepted(screen):
    d = screen["dirs"]
    try:
        L.selection_ext([d[1.0], d[2.0], d[4.0], d[8.0]], screen["first"])
        accepted = True
    except L.SelectionRefused:
        accepted = False
    assert accepted, "the missing x0.5 run was refused (as it must be)"


@nc("the pre-FL-09 requirement (any --lr-selection) lets a MAIN run use the 06B file while the EXT file exists")
def test_nc_06b_file_accepted_while_ext_exists(screen, tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    first = folder / "LR_SELECTION.json"
    shutil.copy(screen["first"], first)
    L.write_selection_ext([screen["dirs"][m] for m in (0.5, 1.0, 2.0)], folder / F.LR_SELECTION_EXT_NAME, first)

    def mutant(args):                                                  # MUTANT: the FL-07 check only
        if not args.smoke and args.role == "MAIN" and args.lr_selection is None:
            raise F.LaunchRefused("no --lr-selection")
    assert_raises(F.LaunchRefused, mutant, _main_args(tmp_path, first))


@nc("plain argmax (no 06F tie rule) picks a different multiplier inside the 0.0005 tie band")
def test_nc_argmax_ignores_log2_tie_order():
    vals = {1.0: 0.1000, 4.0: 0.1003, 8.0: 0.1004}
    assert max(vals, key=vals.get) == L.apply_rule_ext(vals)[0]
    assert math.isfinite(vals[8.0])

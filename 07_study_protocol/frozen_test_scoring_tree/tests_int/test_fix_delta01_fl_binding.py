"""FIX-DELTA-01 F2 (REV-DELTA-01; amendments 06F §1, 06G §3 / 06J) — launch_fl binds every non-smoke MAIN FL run to the
latest REGISTERED selection file (CPU, the tiny FL world of test_fl05_launch; the registered contracts/fl paths are
monkeypatched into the test's tmp directory).

Proves: the registered LR_SELECTION_EXT.json made by the real `lr_screen select` (x0.5 ... x16 [+ x32], first screen)
binds a MAIN run (file sha256 + first-screen sha256 recorded in LAUNCH main_selection; arm spec digest unchanged); a
preview / copy / sibling file, the 06B file (even at the EXT path), an absent first-screen sha, a first-screen sha that
is not the registered file's, a screen without x16, a broken x32 edge rule (either direction) and an lr_peak_base other
than the SMALL recipe peak are refused; once FL_TUNING_SELECTION.json exists it wins (--lr-selection refused, the
tuning file's S1b source must be the registered EXT by sha, m* = the EXT selection); smoke / CALIBRATION / dry runs are
not bound. Negative controls: the pre-F2 check (any self-consistent EXT file) accepts the preview; the check without the
x32 edge rule accepts a screen whose x16 won; the check without the peak binding accepts a foreign lr_peak_base.
FIX-DELTA-02 M2 (REV-DELTA-02): a non-smoke MAIN run binds ONLY the FINAL (stage S4) FL_TUNING_SELECTION.json; the EXT
file is never a MAIN input (its 06F content rules are checked as the S4 file's S1b source, check_ext_for_main), and S2 /
S3 stage files at the registered path are refused. The world now builds S2 -> S3 -> S4 (edge runs screened as
select-joint names them, FIX-DELTA-02 M3).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import v4int.launch_fl as F
import v4int.lr_screen as L
from intkit import assert_raises, nc
from test_fl05_launch import PEAK, _args, fl05_world, run_tiny, tiny_spec
from test_fl10_fedadam_launch import _label_dirs, complete_edges, screen_point

CALIB_EVAL = "M06_CALIB_VALIDATION_WARM"


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _screen(root, name, **kw):
    out = run_tiny(root, tiny_spec("FA", role="CALIBRATION", **kw), name, world=fl05_world(eval_id=CALIB_EVAL))
    assert out["status"] == "DONE_MAIN", name
    return root / name


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Real screen runs x0.5..x16 (+ x32 when x16 wins without it, 06F §1), the 06B first screen, the EXT selection in
    reg_a (first + EXT), the S2 / S3 stage files and the FINAL S4 tuning file in reg_b (first + EXT +
    FL_TUNING_SELECTION.json = S4)."""
    import torch
    from fl.numerics import enforce_strict_fp32
    torch.set_num_threads(1)
    enforce_strict_fp32()                                  # module fixture: runs before the autouse strict fixture
    root = tmp_path_factory.mktemp("fixdelta01_fl")
    dirs = {m: _screen(root, f"x{F.LR_MULT_TAG[m]}", lr_mult=m) for m in F.FL06F_SCREEN_SET}
    if L.apply_rule_ext({m: L.read_run(d)["value"] for m, d in dirs.items()})[0] == F.FL06F_EDGE:
        dirs[32.0] = _screen(root, "x32", lr_mult=32.0)
    reg_a, reg_b = root / "reg_a", root / "reg_b"
    reg_a.mkdir()
    first = reg_a / F.FIRST_SCREEN_NAME
    L.write_selection([dirs[m] for m in (0.5, 1.0, 2.0)], first)
    ext = reg_a / F.LR_SELECTION_EXT_NAME
    L.write_selection_ext(list(dirs.values()), ext, first)
    m_star = _json(ext)["selected_lr_mult"]
    s2 = [_screen(root, f"adam_{lr}_{m}", lr_mult=m, server_opt="fedadam", server_lr=lr) for lr, m in F.fedadam_grid(m_star)]
    s2.append(_screen(root, "avgm", lr_mult=m_star, server_opt="fedavgm"))
    s2 += [screen_point(root, f"avgm_a1_{F.num_tag(t['lr_mult'])}", t)                  # FL-14: 06L A1 FedAvgM m {1, 2}
           for t in L.stage_required("S2", F.tuning_config(lr_mult=m_star), m_star)[5:]]
    runs2, _e2 = complete_edges(root, "S2", [dirs[m_star]] + s2, ext, "s2")          # FIX-DELTA-02 M3
    s2_file = root / "S2.json"
    L.write_selection_stage("S2", runs2, s2_file, ext)
    w2 = F.tuning_config(**{k: _json(s2_file)["winner"][k] for k in F.TUNING_KEYS})
    s3 = [screen_point(root, f"s3_{i}", t) for i, t in enumerate(L.stage_required("S3", w2, m_star))]
    runs3, _e3 = complete_edges(root, "S3", [_label_dirs(runs2)[F.tuning_label(w2)]] + s3, s2_file, "s3")
    s3_file = root / "S3.json"
    L.write_selection_stage("S3", runs3, s3_file, s2_file)
    w3 = F.tuning_config(**{k: _json(s3_file)["winner"][k] for k in F.TUNING_KEYS})
    s4 = [screen_point(root, f"s4_{i}", t) for i, t in enumerate(L.stage_required("S4", w3, m_star))]
    runs4 = [_label_dirs(runs3)[F.tuning_label(w3)]] + s4
    reg_b.mkdir()
    shutil.copy(first, reg_b / F.FIRST_SCREEN_NAME)
    shutil.copy(ext, reg_b / F.LR_SELECTION_EXT_NAME)
    L.write_selection_stage("S4", runs4, reg_b / F.FL_TUNING_NAME, s3_file)       # FIX-DELTA-02 M2: the FINAL file
    return {"root": root, "dirs": dirs, "reg_a": reg_a, "reg_b": reg_b, "first": first, "ext": ext, "m_star": m_star,
            "tun": reg_b / F.FL_TUNING_NAME, "s2_file": s2_file, "s3_file": s3_file, "runs4": runs4}


def register(monkeypatch, reg: Path, first_sha: str):
    monkeypatch.setattr(F, "FIRST_SCREEN_PATH", reg / F.FIRST_SCREEN_NAME)
    monkeypatch.setattr(F, "LR_SELECTION_EXT_PATH", reg / F.LR_SELECTION_EXT_NAME)
    monkeypatch.setattr(F, "FL_TUNING_PATH", reg / F.FL_TUNING_NAME)
    monkeypatch.setattr(F, "FIRST_SCREEN_SHA256_PREFIX", first_sha[:8])


def main_args(tmp_path, *extra, smoke=False, arm="FP"):
    return _args(tmp_path, *extra, smoke=smoke, arm=arm, seed="2027")


# ------------------------------------------------------------------------------------------------ EXT (never MAIN)
def test_ext_is_never_a_main_input(world, tmp_path, monkeypatch):
    """FIX-DELTA-02 M2: the registered EXT file does not bind a MAIN run (06G §4 / 06I §3 / 06J §3: the S4 file); its
    06F content rules are still enforced (check_ext_for_main, applied to the S4 file's S1b source)."""
    register(monkeypatch, world["reg_a"], _sha(world["first"]))
    a = main_args(tmp_path, "--lr-selection", str(world["ext"]))
    assert_raises(F.LaunchRefused, F.require_lr_selection, a)
    assert_raises(F.LaunchRefused, F.resolve_spec, a, PEAK, fused=True)
    assert_raises(F.LaunchRefused, F.require_lr_selection, main_args(tmp_path))          # no FINAL file registered
    assert F.registered_selection() is None
    F.check_ext_for_main(F.load_lr_selection(world["ext"]), world["ext"], peak_lr=PEAK)
    obj = _json(world["ext"])
    assert set(F.FL06F_SCREEN_SET) <= set(obj["screened"])
    rest = {F.lr_mult_value(t): v for t, v in obj["values"].items() if t != "32"}
    assert (32.0 in obj["screened"]) == (L.apply_rule_ext(rest)[0] == 16.0)
    # the SMALL recipe peak differs from the file's lr_peak_base -> refused
    assert_raises(F.LaunchRefused, F.check_ext_for_main, F.load_lr_selection(world["ext"]), world["ext"],
                  peak_lr=0.000711)


def test_preview_copy_sibling_and_06b_refused(world, tmp_path, monkeypatch):
    register(monkeypatch, world["reg_a"], _sha(world["first"]))
    prev = tmp_path / "LR_SELECTION_EXT_PREVIEW.json"
    shutil.copy(world["ext"], prev)
    copy = tmp_path / "c" / F.LR_SELECTION_EXT_NAME
    copy.parent.mkdir()
    shutil.copy(world["ext"], copy)
    for p in (prev, copy, world["first"], tmp_path / "missing.json"):
        assert_raises(F.LaunchRefused, F.require_lr_selection, main_args(tmp_path, "--lr-selection", str(p)))
        assert_raises(F.LaunchRefused, F.run, main_args(tmp_path, "--lr-selection", str(p)), {}, {})
    assert_raises(F.LaunchRefused, F.require_lr_selection, main_args(tmp_path))                   # no option
    assert_raises(F.LaunchRefused, F.require_lr_selection, main_args(tmp_path, "--lr-mult", "2"))
    assert_raises(F.LaunchRefused, F.require_lr_selection,                                       # tuning not registered
                  main_args(tmp_path, "--fl-tuning", str(world["tun"])))
    reg = tmp_path / "reg06b"                                             # the 06B file AT the registered EXT path
    reg.mkdir()
    shutil.copy(world["first"], reg / F.FIRST_SCREEN_NAME)
    shutil.copy(world["first"], reg / F.LR_SELECTION_EXT_NAME)
    register(monkeypatch, reg, _sha(world["first"]))
    assert_raises(F.LaunchRefused, F.require_lr_selection,
                  main_args(tmp_path, "--lr-selection", str(reg / F.LR_SELECTION_EXT_NAME)))
    empty = tmp_path / "none"                                             # nothing registered: the 06B file refused
    empty.mkdir()
    register(monkeypatch, empty, _sha(world["first"]))
    assert_raises(F.LaunchRefused, F.require_lr_selection, main_args(tmp_path, "--lr-selection", str(world["first"])))


def test_not_bound_for_smoke_calibration_and_dry_run(world, tmp_path, monkeypatch):
    register(monkeypatch, world["reg_a"], _sha(world["first"]))
    prev = tmp_path / "LR_SELECTION_EXT_PREVIEW.json"
    shutil.copy(world["ext"], prev)
    smoke = main_args(tmp_path, "--lr-selection", str(prev), smoke=True)
    F.require_lr_selection(smoke)
    assert "main_selection" not in F.resolve_spec(smoke, PEAK, fused=True)[1]
    F.require_lr_selection(_args(tmp_path, "--role", "CALIBRATION", "--lr-mult", "4", smoke=False))
    dry = main_args(tmp_path, "--lr-selection", str(prev), "--dry-run")
    assert "main_selection" not in F.resolve_spec(dry, PEAK, fused=True)[1]            # FL-07: dry run not refused


# ------------------------------------------------------------------------------------------------ EXT content rules
def _ext_obj(values: dict, *, first_sha, excluded=(), peak=PEAK) -> dict:
    sel, tie = L.apply_rule_ext(values)
    scr = sorted(list(values) + list(excluded), key=L.ext_order_key)
    obj = {"format": F.LR_SELECTION_EXT_FORMAT, "values": {F.LR_MULT_TAG[m]: v for m, v in values.items()},
           "screened": scr, "excluded": [{"lr_mult": m, "reason": "non-finite"} for m in excluded],
           "selected_lr_mult": sel, "tie_set": tie, "first_screen_file_sha256": first_sha,
           "first_screen": {"sha256": first_sha}, "lr_peak_base": peak}
    obj["self_sha256"] = F._canon(obj)
    return obj


def _put(reg: Path, obj: dict) -> Path:
    p = reg / F.LR_SELECTION_EXT_NAME
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return p


RISING = {0.5: 0.10, 1.0: 0.11, 2.0: 0.12, 4.0: 0.13, 8.0: 0.14, 16.0: 0.15}        # x16 wins -> x32 required
PEAKED = {0.5: 0.10, 1.0: 0.11, 2.0: 0.12, 4.0: 0.16, 8.0: 0.14, 16.0: 0.13}        # x4 wins -> no x32


def _ext_world(world, tmp_path, monkeypatch):
    reg = tmp_path / "reg"
    reg.mkdir()
    shutil.copy(world["first"], reg / F.FIRST_SCREEN_NAME)
    fs = _sha(world["first"])
    register(monkeypatch, reg, fs)
    return reg, fs


def _binds(tmp_path, p, peak=PEAK):
    """FIX-DELTA-02 M2: the 06F content rules of an EXT file as the S4 file's S1b source (check_ext_for_main; the EXT
    file itself never binds a MAIN run)."""
    F.check_ext_for_main(F.load_lr_selection(p), p, peak_lr=peak)
    return {"sha256": _sha(p)}


def test_ext_content_rules(world, tmp_path, monkeypatch):
    reg, fs = _ext_world(world, tmp_path, monkeypatch)
    ok = [{**RISING, 32.0: 0.16}, PEAKED, {**RISING, 32.0: 0.10}, {**PEAKED, 16.0: 0.1601}]
    for vals in ok:                                                                    # x32 exactly per the rule
        assert _binds(tmp_path, _put(reg, _ext_obj(vals, first_sha=fs)))["sha256"] == _sha(reg / F.LR_SELECTION_EXT_NAME)
    p = _put(reg, _ext_obj({k: v for k, v in RISING.items() if k != 16.0}, first_sha=fs, excluded=(16.0,)))
    _binds(tmp_path, p)                                          # x16 excluded (non-finite): screened, cannot win
    bad = [_ext_obj(RISING, first_sha=fs),                                             # x16 won, x32 missing
           _ext_obj({**PEAKED, 32.0: 0.10}, first_sha=fs),                       # x32 screened, x16 did not win
           _ext_obj({k: v for k, v in PEAKED.items() if k != 16.0}, first_sha=fs),     # x16 not screened
           _ext_obj({k: v for k, v in PEAKED.items() if k != 0.5}, first_sha=fs),      # x0.5 not screened
           _ext_obj(PEAKED, first_sha=None),                                           # no first-screen sha
           _ext_obj(PEAKED, first_sha="0" * 64),                                       # not the registered 06B file
           _ext_obj(PEAKED, first_sha=fs, peak=0.000711)]                              # lr_peak_base != recipe peak
    for obj in bad:
        F.load_lr_selection(_put(reg, obj))                                            # self-consistent files ...
        assert_raises(F.LaunchRefused, _binds, tmp_path, reg / F.LR_SELECTION_EXT_NAME)   # ... refused for MAIN
    fr = dict(_ext_obj(PEAKED, first_sha=fs), first_screen={"sha256": "1" * 64})
    fr["self_sha256"] = F._canon(fr)
    assert_raises(F.LaunchRefused, _binds, tmp_path, _put(reg, fr))                    # record != field
    p = _put(reg, _ext_obj(PEAKED, first_sha=fs))
    monkeypatch.setattr(F, "FIRST_SCREEN_SHA256_PREFIX", "ffffffff" if not fs.startswith("ffffffff") else "00000000")
    assert_raises(F.LaunchRefused, _binds, tmp_path, p)                                # not the DL 181 file
    monkeypatch.setattr(F, "FIRST_SCREEN_SHA256_PREFIX", fs[:8])
    (reg / F.FIRST_SCREEN_NAME).unlink()
    assert_raises(F.LaunchRefused, _binds, tmp_path, p)                                # first screen absent


# ------------------------------------------------------------------------------------------------ FL_TUNING (S4) binds
def test_final_s4_tuning_file_binds_main(world, tmp_path, monkeypatch):
    register(monkeypatch, world["reg_b"], _sha(world["first"]))
    ext_b = world["reg_b"] / F.LR_SELECTION_EXT_NAME
    assert _json(world["tun"])["stage"] == "S4"
    assert_raises(F.LaunchRefused, F.require_lr_selection, main_args(tmp_path, "--lr-selection", str(ext_b)))
    assert_raises(F.LaunchRefused, F.require_lr_selection, main_args(tmp_path))
    a = main_args(tmp_path, "--fl-tuning", str(world["tun"]))
    F.require_lr_selection(a)
    spec, extra = F.resolve_spec(a, PEAK, fused=True)
    b = extra["main_selection"]
    assert b["kind"] == "fl_tuning" and b["sha256"] == _sha(world["tun"]) and b["format"] == F.FL_TUNING_FORMAT
    assert b["stage"] == "S4" and b["ext"]["sha256"] == _sha(ext_b) and b["first_screen_file_sha256"] == _sha(world["first"])
    win = _json(world["tun"])["winner"]
    assert F.spec_tuning(spec) == F.tuning_config(**{k: win[k] for k in F.TUNING_KEYS})
    assert_raises(F.LaunchRefused, F.resolve_spec, a, 0.000711, fused=True)            # peak binding
    copy = tmp_path / F.FL_TUNING_NAME
    shutil.copy(world["tun"], copy)
    assert_raises(F.LaunchRefused, F.require_lr_selection, main_args(tmp_path, "--fl-tuning", str(copy)))
    # the S1b source must be the registered EXT file (by sha); an absent EXT refuses
    reg = tmp_path / "reg"
    reg.mkdir()
    shutil.copy(world["first"], reg / F.FIRST_SCREEN_NAME)
    shutil.copy(world["tun"], reg / F.FL_TUNING_NAME)
    register(monkeypatch, reg, _sha(world["first"]))
    t = main_args(tmp_path, "--fl-tuning", str(reg / F.FL_TUNING_NAME))
    assert_raises(F.LaunchRefused, F.require_lr_selection, t)                          # EXT absent
    obj = _json(ext_b)
    (reg / F.LR_SELECTION_EXT_NAME).write_text(json.dumps(obj) + "\n", encoding="utf-8")   # same content, other bytes
    F.load_lr_selection(reg / F.LR_SELECTION_EXT_NAME)
    assert_raises(F.LaunchRefused, F.require_lr_selection, t)                          # sha != the S1b record
    shutil.copy(ext_b, reg / F.LR_SELECTION_EXT_NAME)
    F.require_lr_selection(t)                                                          # the exact bytes: bound


def test_s2_s3_stage_files_refused_for_main(world, tmp_path, monkeypatch):
    """FIX-DELTA-02 M2: an S2 / S3 stage file at the registered path never binds a MAIN run (smoke is not bound)."""
    for stage, src in (("S2", world["s2_file"]), ("S3", world["s3_file"])):
        reg = tmp_path / f"reg_{stage}"
        reg.mkdir()
        shutil.copy(world["first"], reg / F.FIRST_SCREEN_NAME)
        shutil.copy(world["ext"], reg / F.LR_SELECTION_EXT_NAME)
        shutil.copy(src, reg / F.FL_TUNING_NAME)
        register(monkeypatch, reg, _sha(world["first"]))
        assert F.load_fl_tuning(reg / F.FL_TUNING_NAME)["stage"] == stage             # a valid stage file ...
        a = main_args(tmp_path, "--fl-tuning", str(reg / F.FL_TUNING_NAME))
        assert_raises(F.LaunchRefused, F.require_lr_selection, a)                      # ... but not FINAL
        assert_raises(F.LaunchRefused, F.check_tuning_for_main, _json(src), src)
        F.require_lr_selection(main_args(tmp_path, "--fl-tuning", str(reg / F.FL_TUNING_NAME), smoke=True))


# ------------------------------------------------------------------------------------------------ negative controls
@nc("the pre-F2 check (format + self-sha + rule of any file) accepts the EXT preview for a MAIN run")
def test_nc_preview_accepted_by_pre_f2_check(world, tmp_path):
    prev = tmp_path / "LR_SELECTION_EXT_PREVIEW.json"
    shutil.copy(world["ext"], prev)

    def mutant(args):                                                  # MUTANT: FL-09's check only
        if F.load_lr_selection(args.lr_selection)["format"] != F.LR_SELECTION_EXT_FORMAT:
            raise F.LaunchRefused("not EXT")
    assert_raises(F.LaunchRefused, mutant, main_args(tmp_path, "--lr-selection", str(prev)))


@nc("a MAIN check without the 06F x32 edge rule accepts a screen whose x16 won but x32 was never run")
def test_nc_edge_rule_dropped(world, tmp_path, monkeypatch):
    reg, fs = _ext_world(world, tmp_path, monkeypatch)
    p = _put(reg, _ext_obj(RISING, first_sha=fs))
    monkeypatch.setattr(F, "FL06F_EDGE", 1e9)                          # MUTANT: the edge can never win
    monkeypatch.setattr(F, "FL06F_EDGE_NEXT", 2e9)
    assert_raises(F.LaunchRefused, _binds, tmp_path, p)


@nc("a MAIN check without the SMALL-peak binding accepts an EXT file screened at another lr_peak_base")
def test_nc_peak_binding_dropped(world, tmp_path, monkeypatch):
    reg, fs = _ext_world(world, tmp_path, monkeypatch)
    p = _put(reg, _ext_obj(PEAKED, first_sha=fs, peak=0.000711))
    assert_raises(F.LaunchRefused, _binds, tmp_path, p, None)          # MUTANT: peak_lr not passed

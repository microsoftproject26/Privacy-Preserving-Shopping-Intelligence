"""P2-05 C1 (v4int.central_ft; PREG2 §4.2) - CPU tests with synthetic providers (tests_train/trainkit) and
read-only reads of the real REGISTRY_V2.

Proves: the launch contract (derived run id, registered LR grid, seeds per role, no continuation, no manifest / peak /
lock overrides); the MAIN binding of contracts/fl/P2_C1_SELECTION.json by file sha256 (registered path, self-hash,
stage, the variant's peak); the CALIBRATION trainer view (the trainer's MAIN role = REGISTRY_V2's CALIB pair, entries
byte-identical); the peak wrapper (the recipe rule on a copy with the recipe's peak, the actual peak = the shim's grid
value); the labels (P2_C1_FT_CENTRAL_FULL / _FROZEN, row_status P2_CONTROL, c_matched_eligible False, refused by
v4train.selection everywhere, other runs untouched); on the REAL CentralTrainer: theta_pre loaded after construction and
before step 0 and recorded in RUN_START, a non-PRE or MAIN-trained init refused, the frozen item tables bit-unchanged
after 50 updates and at 6.0 EFE while the rest trains, the FULL variant trains them, no continuation (the resumable of a
STILL_IMPROVING run deleted, continue_run refused), mark 0 never a PRACTICAL_BEST candidate, read_c1_run, a resume with
another theta_pre refused. NCs: an unregistered LR accepted; the FULL variant keeping the item tables fixed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

import v4int.central_ft as CF
from intkit import assert_raises, nc
import ppsi_paths

V4 = ppsi_paths.PROJECT_ROOT.resolve()        # the ppsi_v4 workspace (contracts, company records)
if str(ppsi_paths.code_dir("tests_train")) not in sys.path:
    sys.path.append(str(ppsi_paths.code_dir("tests_train")))          # trainkit only (appended: tests_int modules win)
import trainkit as T  # noqa: E402

from fl.numerics import state_digest  # noqa: E402

REG_V2 = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2.json"
ARGV = ["--family", "SASREC", "--seed", "2026", "--phase", "MAIN", "--recipe-id", "SMALL_SASREC_D64_B2", "--gpu", "0",
        "--run-dir", "RUNDIR", "--history-policy", "USER_HISTORY_L50"]
FK = CF.FROZEN_ITEM_KEYS


def _shim(role="CALIBRATION", lr=1e-4, frozen=False, extra=()):
    argv = ["--role", role, "--peak-lr", repr(lr), "--init-from", "init.json"] + \
        (["--p2-freeze-item-tables"] if frozen else []) + list(extra) + ["--"] + ARGV
    return CF.split_argv(argv)


def _args(largv, extra=()):
    from v4int.launch_central import parse_args
    return parse_args(list(largv) + list(extra))


def _swap(old, new, argv=ARGV):
    return [new if x == old else x for x in argv]


# ------------------------------------------------------------------------------------------------ contract
def test_constants_match_the_registered_values():
    from fl.runtime import FROZEN_ITEM_KEYS
    from v4int.smoke import SMOKE_RUN_PREFIX
    import v4int.launch_fl as F
    assert CF.FROZEN_ITEM_KEYS == tuple(FROZEN_ITEM_KEYS) == tuple(F.FROZEN_ITEM_KEYS)
    assert CF.SMOKE_PREFIX == SMOKE_RUN_PREFIX
    assert CF.LR_GRID == (3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    assert CF.PRE_CHECKPOINT_SHA256.startswith("cf08588c") and CF.PRE_STATE_DIGEST.startswith("ee33345b")
    reeval_pre = json.loads((V4 / "company" / "evidence" / "reeval_records" / "PRE_SHIM_0.json").read_text("utf-8"))
    assert reeval_pre["fl_init"]["sha256"] == CF.PRE_CHECKPOINT_SHA256           # the PRE run's own FL init record
    reg = json.loads(REG_V2.read_text(encoding="utf-8"))
    assert reg["roles"]["CALIBRATION"] == {"train": "CALIB_TRAIN_5_100_32768", "eval": "CALIB_VALIDATION_5_100_32768"}


def test_launch_contract_derived_run_id_and_refusals():
    shim, largv = _shim()
    assert CF.check_launch_args(_args(largv), shim) == "P2_C1_FT_CENTRAL_FULL_CALIB_lr1e-4_s2026"
    shim, largv = _shim(frozen=True, lr=3e-5)
    assert CF.check_launch_args(_args(largv), shim) == "P2_C1_FT_CENTRAL_FROZEN_CALIB_lr3e-5_s2026"
    assert CF.check_launch_args(_args(largv, ["--smoke"]), shim) == \
        "SMOKE_NOT_EVIDENCE__P2_C1_FT_CENTRAL_FROZEN_CALIB_lr3e-5_s2026"
    shim, largv = _shim(role="MAIN", lr=3e-4, frozen=True)
    assert CF.check_launch_args(_args(_swap("2026", "2027", largv)), shim) == "P2_C1_FT_CENTRAL_FROZEN_MAIN_lr3e-4_s2027"
    for lr in CF.LR_GRID:                                             # every registered value (screen + edges)
        s, lv = _shim(lr=lr)
        CF.check_launch_args(_args(lv), s)
    shim, largv = _shim()
    for extra in (["--peak-lr", "1e-4"], ["--train-manifest", "MAIN_TRAIN_5_100_131072"], ["--eval-manifest", "X"],
                  ["--role", "TUNING"], ["--calibration-record", "c.json"], ["--continue-run"],
                  ["--continuation-budget", "b.json"], ["--run-id", "SMALL_SASREC_D64_B2_MAIN_s2026"],
                  ["--run-id", "P2_C1_FT_CENTRAL_FROZEN_CALIB_lr1e-4_s2026"], ["--allow-shared-gpu"],
                  ["--coschedule-06g"], ["--transform-state", "t.json"], ["--lock-path", "l.json"],
                  ["--theta0-variants", "t0.json"], ["--hp-registry", "hp.json"], ["--tuned-recipes", "x.json"]):
        assert_raises(CF.CentralFTRefused, CF.check_launch_args, _args(largv, extra), shim)
    CF.check_launch_args(_args(largv, ["--allow-shared-gpu", "--coschedule-06g"]), shim)    # 06G §5 co-scheduling
    assert_raises(CF.CentralFTRefused, CF.check_launch_args, _args(_swap("2026", "2027", largv)), shim)   # CALIB 2026
    assert_raises(CF.CentralFTRefused, CF.check_launch_args, _args(_swap("SMALL_SASREC_D64_B2", "02B_SASREC", largv)),
                  shim)
    for bad in (5e-4, 2e-4, 1e-2):
        s, lv = _shim(lr=bad)
        assert_raises(CF.CentralFTRefused, CF.check_launch_args, _args(lv), s)
    s, lv = _shim(extra=["--c1-selection", "x.json"])                 # the screen produces the selection
    assert_raises(CF.CentralFTRefused, CF.check_launch_args, _args(lv), s)


def test_main_smoke_refused_before_any_data_load(tmp_path):
    shim, largv = _shim(role="MAIN", lr=3e-4)
    err = None
    try:
        CF.check_launch_args(_args(largv, ["--smoke"]), shim)
    except CF.CentralFTRefused as e:
        err = str(e)
    assert err and "S8" in err
    rd = str(tmp_path / "run")
    init = tmp_path / "PRETRAIN_FOR_FL_INIT.json"
    init.write_text("{}", encoding="utf-8")
    argv = ["--role", "MAIN", "--peak-lr", "3e-4", "--init-from", str(init), "--"] +         [rd if x == "RUNDIR" else x for x in ARGV] + ["--smoke"]
    assert CF.main(argv) == 2 and not Path(rd).exists()               # refused before launch_central runs
    s2, l2 = _shim(role="CALIBRATION", lr=3e-4)                        # the CALIB smoke stays allowed
    assert CF.check_launch_args(_args(l2, ["--smoke"]), s2).startswith("SMOKE_NOT_EVIDENCE__")


@nc("a MAIN-role smoke (MAIN_VALIDATION into SMOKE_EVAL.json) must be refused (PREG2 §6.3 S8)")
def test_nc_main_smoke_accepted():
    s, lv = _shim(role="MAIN", lr=3e-4)
    try:
        CF.check_launch_args(_args(lv, ["--smoke"]), s)
        accepted = True
    except CF.CentralFTRefused:
        accepted = False
    assert accepted


@nc("a peak LR outside the registered C1 grid must be refused")
def test_nc_unregistered_lr_accepted():
    s, lv = _shim(lr=5e-4)
    try:
        CF.check_launch_args(_args(lv), s)
        accepted = True
    except CF.CentralFTRefused:
        accepted = False
    assert accepted


def _selection(tmp_path, *, stage="C1", full=1e-4, frozen=3e-4, tamper=False, fmt=CF.C1_SELECTION_FORMAT):
    """The P2-06 C1 output layout (v4int.p2_select.selection_c1): format, stage, variants.<v>.winner_peak_lr."""
    obj = {"format": fmt, "stage": stage, "variants": {
        "FULL": {"winner_peak_lr": full, "run_label": "P2_C1_FT_CENTRAL_FULL"},
        "FROZEN": {"winner_peak_lr": frozen, "run_label": "P2_C1_FT_CENTRAL_FROZEN"}}}
    obj["self_sha256"] = CF.canonical_self_sha256(obj)
    if tamper:
        obj["variants"]["FULL"]["winner_peak_lr"] = 3e-5
    p = tmp_path / "P2_C1_SELECTION.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _ok(path, stage):
    assert stage == "C1"
    return {}


def test_main_binds_the_c1_selection_by_sha(tmp_path):
    import v4int.launch_fl as F
    p, sha = _selection(tmp_path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["self_sha256"] == F._canon(obj)                         # the same rule as every selection file
    b = CF.bind_c1_selection(p, variant="FROZEN", peak_lr=3e-4, expect_sha256=sha, registered=p, reverify=_ok)
    assert b["sha256"] == sha and b["peak_lr"] == 3e-4 and b["stage"] == "C1"
    assert CF.bind_c1_selection(None, variant="FULL", peak_lr=1e-4, expect_sha256=sha, registered=p,
                                reverify=_ok)["variant"] == "FULL"
    bad = [dict(variant="FULL", peak_lr=3e-4, expect_sha256=sha, registered=p),          # the other variant's peak
           dict(variant="FULL", peak_lr=1e-4, expect_sha256=None, registered=p),         # no sha pin
           dict(variant="FULL", peak_lr=1e-4, expect_sha256="0" * 64, registered=p),     # wrong pin
           dict(variant="FULL", peak_lr=1e-4, expect_sha256=sha, registered=tmp_path / "other.json")]  # a copy
    for kw in bad:
        assert_raises(CF.CentralFTRefused, CF.bind_c1_selection, p, reverify=_ok, **kw)

    def refuse(path, stage):
        raise ValueError("decision does not re-derive")
    assert_raises(CF.CentralFTRefused, CF.bind_c1_selection, p, variant="FULL", peak_lr=1e-4, expect_sha256=sha,
                  registered=p, reverify=refuse)                       # the P2-06 verifier refuses -> no launch
    for i, kw in enumerate((dict(stage="WA"), dict(tamper=True), dict(fmt="PPSI_V4_P2_WARM_SELECTION_1"))):
        d = tmp_path / f"case{i}"
        d.mkdir()
        q, s2 = _selection(d, **kw)
        assert_raises(CF.CentralFTRefused, CF.bind_c1_selection, q, variant="FULL", peak_lr=1e-4, expect_sha256=s2,
                      registered=q, reverify=_ok)


def test_main_refusals(tmp_path, capsys):
    assert CF.main(["--role", "CALIBRATION"] + ARGV) == 2                       # no '--'
    rd = str(tmp_path / "run")
    base = [rd if x == "RUNDIR" else x for x in ARGV]
    init = tmp_path / "PRETRAIN_FOR_FL_INIT.json"
    assert CF.main(["--role", "CALIBRATION", "--peak-lr", "1e-4", "--init-from", str(init), "--"] + base) == 2  # absent
    init.write_text("{}", encoding="utf-8")
    assert CF.main(["--role", "MAIN", "--peak-lr", "1e-4", "--init-from", str(init), "--"] + base) == 2   # no selection
    assert CF.main(["--role", "CALIBRATION", "--peak-lr", "5e-4", "--init-from", str(init), "--"] + base) == 2
    assert not Path(rd).exists()


# ------------------------------------------------------------------------------------------------ labels / peak / view
def _fake_cfg(run_id, test_overrides=False):
    return types.SimpleNamespace(run_id=run_id, test_overrides=test_overrides, role=None, manifest_role="MAIN",
                                 population="COHORT", data_version="CLEAN_V2_POP5100_V2")


def test_run_id_label_map():
    assert CF.p2_label_of_run_id("P2_C1_FT_CENTRAL_FULL_MAIN_lr1e-4_s2026") == "P2_C1_FT_CENTRAL_FULL"
    assert CF.p2_label_of_run_id("SMOKE_NOT_EVIDENCE__P2_C1_FT_CENTRAL_FROZEN_CALIB_lr1e-4_s2026") == \
        "P2_C1_FT_CENTRAL_FROZEN"
    assert CF.p2_label_of_run_id("P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026") == "P2_C2_CFULL_SMALL"
    for rid in ("C_FULL_02B_SASREC_s2026", "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026", "SMALL_SASREC_D64_B2_MAIN_s2026",
                "02B_SASREC_MAIN_s2026", "", None):
        assert CF.p2_label_of_run_id(rid) is None and CF.c1_variant_of_run_id(rid) is None
    assert CF.c1_variant_of_run_id("P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026") is None


@pytest.fixture
def real_classes(monkeypatch):
    """Every P2-05 class wrapper target on the REAL v4train classes, restored after the test."""
    import v4train.recipe as RC
    import v4train.trainer as TR
    monkeypatch.setattr(RC.RunConfig, "run_label", RC.RunConfig.__dict__["run_label"])
    monkeypatch.setattr(RC.RunConfig, "_validate_train02", RC.RunConfig.__dict__["_validate_train02"])
    for name in ("__init__", "_run_start_record", "_meta", "_summary", "_check_start_identity", "continue_run",
                 "_finish_main", "_row_labels"):
        monkeypatch.setattr(TR.CentralTrainer, name, TR.CentralTrainer.__dict__[name])
    for cls in (RC.RunConfig, TR.CentralTrainer):
        for flag in ("__p2_label__", "__p2_c1_peak__", "__p2_c1__"):
            monkeypatch.setattr(cls, flag, False, raising=False)
    CF.install_p2_label_wrappers(RC, TR)
    CF.install_p2_label_wrappers(RC, TR)                               # idempotent
    CF.install_validate_wrapper(RC)
    CF.install_trainer_wrappers(TR)
    CF.install_trainer_wrappers(TR)                                    # idempotent
    monkeypatch.setattr(CF, "_CTX", None)
    return RC, TR


def test_label_wrapper_on_the_real_runconfig_and_rows(real_classes):
    RC, _ = real_classes
    from v4train.selection import SelectionError, assert_c_matched_eligible, refuse_run_labels
    prop = RC.RunConfig.__dict__["run_label"]
    for rid, lab in (("P2_C1_FT_CENTRAL_FULL_MAIN_lr1e-4_s2026", "P2_C1_FT_CENTRAL_FULL"),
                     ("P2_C1_FT_CENTRAL_FROZEN_CALIB_lr3e-4_s2026", "P2_C1_FT_CENTRAL_FROZEN"),
                     ("P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026", "P2_C2_CFULL_SMALL"),
                     ("SMALL_SASREC_D64_B2_MAIN_s2026", "PRODUCTION"), ("C_FULL_02B_SASREC_s2026", "PRODUCTION")):
        assert prop.fget(_fake_cfg(rid)) == lab
    assert prop.fget(_fake_cfg("SMOKE_NOT_EVIDENCE__P2_C1_FT_CENTRAL_FULL_x", True)) == "SMOKE_NOT_EVIDENCE"
    assert "run_label" not in RC.RunConfig.__dataclass_fields__        # derived: never in the config hash
    for lab in ("P2_C1_FT_CENTRAL_FULL", "P2_C1_FT_CENTRAL_FROZEN", "P2_C2_CFULL_SMALL"):
        row = dict({"family": "SASREC", "selection_rule": "PRACTICAL_BEST"}, **CF.p2_row_fields(lab), run_label=lab)
        assert row["row_status"] == "P2_CONTROL" and row["c_matched_eligible"] is False and row["method"] == lab
        for stage in ("COHORT", "TIEBREAK", "CALIBRATION", "C_MATCHED"):
            assert_raises(SelectionError, refuse_run_labels, [row], stage)
        assert_raises(SelectionError, assert_c_matched_eligible, row)


def test_validate_wrapper_checks_the_recipe_with_its_own_peak():
    seen = []

    def orig(self, fam):
        seen.append((self.run_id, self.peak_lr))

    class Err(ValueError):
        pass

    ctx = {"peak_lr": 1e-4, "recipe_peak_lr": 7.5e-4}
    w = CF.make_validate_wrapper(orig, Err, ctx_getter=lambda: ctx)
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class C:
        run_id: str
        peak_lr: float
    w(C("P2_C1_FT_CENTRAL_FULL_CALIB_lr1e-4_s2026", 1e-4), "SASREC")
    w(C("SMALL_SASREC_D64_B2_MAIN_s2026", 7.5e-4), "SASREC")            # any other run: unchanged
    assert seen == [("P2_C1_FT_CENTRAL_FULL_CALIB_lr1e-4_s2026", 7.5e-4), ("SMALL_SASREC_D64_B2_MAIN_s2026", 7.5e-4)]
    assert_raises(Err, w, C("P2_C1_FT_CENTRAL_FULL_CALIB_lr3e-4_s2026", 3e-4), "SASREC")      # not the shim's peak
    ctx["peak_lr"] = 5e-4
    assert_raises(Err, w, C("P2_C1_FT_CENTRAL_FULL_CALIB_lr5e-4_s2026", 5e-4), "SASREC")      # not a grid value
    w2 = CF.make_validate_wrapper(orig, Err, ctx_getter=lambda: None)
    assert_raises(Err, w2, C("P2_C1_FT_CENTRAL_FULL_CALIB_lr1e-4_s2026", 1e-4), "SASREC")     # no shim context


def test_calibration_view_resolves_main_to_the_calib_pair(tmp_path):
    from v4train.provider import expected_manifests
    from v4train.registry import load_manifest_registry
    reg = json.loads(REG_V2.read_text(encoding="utf-8"))
    sha = hashlib.sha256(REG_V2.read_bytes()).hexdigest()
    obj = CF.calibration_view_obj(reg, registry_rel_path="contracts/manifest_registry/REGISTRY_V2.json",
                                  registry_file_sha256=sha)
    p = tmp_path / "REGISTRY_V2_TRAINER_VIEW.json"
    p.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    vreg = load_manifest_registry(str(p))
    pair = ("CALIB_TRAIN_5_100_32768", "CALIB_VALIDATION_5_100_32768")
    assert vreg.role("MAIN") == vreg.role("CONTINUATION") == vreg.role("CALIBRATION") == pair
    assert tuple(expected_manifests("COHORT", "MAIN", registry=vreg)) == pair
    assert all(obj["manifests"][m] == reg["manifests"][m] for m in pair) and set(obj["manifests"]) == set(pair)
    assert obj["derived_from"]["file_sha256"] == sha and obj["data_version"] == reg["data_version"]


def test_install_rewires_launch_central(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(CF, "install_p2_label_wrappers", lambda: calls.append("labels") or {"stub": 1})
    monkeypatch.setattr(CF, "install_validate_wrapper", lambda: calls.append("peak"))
    monkeypatch.setattr(CF, "install_trainer_wrappers", lambda: calls.append("trainer"))
    monkeypatch.setattr(CF, "_CTX", None)

    class Refused(RuntimeError):
        pass

    recipe = {"recipe_id": "SMALL_SASREC_D64_B2", "model_variant": "SASREC_D64_B2", "peak_lr": 7.5e-4,
              "recipe_sha256": "r"}
    LC = types.SimpleNamespace(G1_GATE=None, LaunchRefused=Refused, _recipe_values=lambda a, f: dict(recipe),
                               bind_data_protocol_lock_v2=lambda paths, *, smoke, gate_path=None: {"lock": 1},
                               build_trainer_registry_view=lambda *a, **k: pytest.fail("original view for CALIB"))
    ctx, record = {"role": "CALIBRATION", "peak_lr": 3e-5, "freeze": True, "init_record": "x"}, {}
    CF.install(LC, ctx, record)
    assert CF._CTX is ctx and ctx["error"] is Refused
    t = LC._recipe_values(None, "SASREC")
    assert t["peak_lr"] == 3e-5 and t["recipe_peak_lr"] == 7.5e-4 == ctx["recipe_peak_lr"]
    assert LC.bind_data_protocol_lock_v2(None, smoke=True) == {"lock": 1} and calls == ["labels", "peak", "trainer"]
    v = LC.build_trainer_registry_view(REG_V2, tmp_path / "run", registry_rel_path="contracts/manifest_registry/"
                                       "REGISTRY_V2.json", registry_file_sha256="f")
    assert json.loads(Path(v["path"]).read_text())["roles"]["MAIN"]["train"] == "CALIB_TRAIN_5_100_32768"
    assert list((tmp_path / "run").glob("P2_C1_SHIM_*.json")) and record["trainer_view"]["role"] == "CALIBRATION"
    LC2 = types.SimpleNamespace(G1_GATE=None, LaunchRefused=Refused, bind_data_protocol_lock_v2=None,
                                build_trainer_registry_view=None,
                                _recipe_values=lambda a, f: dict(recipe, recipe_id="02B_SASREC",
                                                                 model_variant="SASREC_D256_B3"))
    CF.install(LC2, dict(ctx), {})
    assert_raises(Refused, LC2._recipe_values, None, "SASREC")


# ------------------------------------------------------------------------------------------------ the real trainer
N400 = 400                                     # 6.0 EFE = 75 updates at B 32 (>= 50)
from v4train.provider import SyntheticEvalProvider, SyntheticTrainProvider  # noqa: E402
# the production CALIBRATION view: the trainer's MAIN role = the CALIB pair (synthetic rows, the real manifest ids)
_PROV = (SyntheticTrainProvider(N400, T.K, T.WIDTHS, seed=3, manifest_id="CALIB_TRAIN_5_100_32768"),
         SyntheticEvalProvider(T.NEVAL, T.K, T.WIDTHS, seed=4, manifest_id=CF.CALIB_EVAL_ID))
REG400 = T.write_registry({"MAIN": _PROV, "CONTINUATION": _PROV, "CALIBRATION": _PROV},
                          name="REGISTRY_P2_05_N400.json")
MARKS = tuple(round(0.5 * i, 6) for i in range(1, 13))
UP = {m: 0.10 + 0.01 * i for i, m in enumerate(MARKS)}            # strictly improving to 6.0 (STILL_IMPROVING)
DOWN = {m: 0.30 - 0.01 * i for i, m in enumerate(MARKS)}          # best at the first mark 0.5


def _cfg(variant="FROZEN", role="CALIBRATION", seed=2026, **kw):
    rid = CF.c1_run_id(variant, role, 1e-3, seed)
    return T.cfg(rid, "SASREC", seed=seed, peak=1e-3, manifest_registry=REG400, **kw)


def make_pre(root: Path, *, seed=91, manifest="PRETRAIN_TRAIN_5_100_NONMAIN", run_id="PRETRAIN_FOR_FL_SYNTH_s2026"):
    """A PRE-run-shaped endpoint_6.0 checkpoint of the synthetic SASRec model (perturbed theta0 of another seed) and its
    PRETRAIN_FOR_FL_INIT.json (the v4int.pretrain_small record layout)."""
    prev = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                 # the launcher's export (STRICT trainer)
    try:
        tr = T.trainer(root / "pre_build", T.cfg("PRE_SYNTH_BUILD", "SASREC", seed=seed))
    finally:
        if prev is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in tr.module.parameters():
            p.add_(0.01 * torch.randn(p.shape, generator=g))
    digest = state_digest(tr.adapter.broadcast_state())
    ck = root / "sealed" / "checkpoints" / "endpoint_6.0.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "run_label": "PRETRAIN_FOR_FL", "family": "SASREC", "model_variant": tr.built.variant,
            "seed": 2026, "state_digest": digest, "efe": 6.0,
            "manifest_registry": {"train_manifest": manifest, "eval_manifest": "UNEW_VAL_VALIDATION_32768"}}
    torch.save({"kind": "endpoint_6.0", "meta": meta, "model": tr.module.state_dict()}, ck)
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    rec = {"run_id": run_id, "run_label": "PRETRAIN_FOR_FL", "train_manifest": manifest,
           "fl_init": {"kind": "endpoint_6.0", "path": "sealed/checkpoints/endpoint_6.0.pt", "sha256": sha,
                       "bytes": ck.stat().st_size}}
    p = root / "PRETRAIN_FOR_FL_INIT.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return {"path": p, "sha": sha, "digest": digest, "state": {k: v.detach().clone() for k, v in
                                                               tr.module.state_dict().items()}}


@pytest.fixture(scope="module")
def pre(tmp_path_factory):
    return make_pre(tmp_path_factory.mktemp("pre"))


def _ctx(pre, *, freeze=True, role="CALIBRATION", pins=True):
    c = {"role": role, "peak_lr": 1e-3, "recipe_peak_lr": 1e-3, "freeze": freeze, "init_record": str(pre["path"]),
         "selection": None, "error": CF.CentralFTRefused}
    if pins:
        c.update(pin_checkpoint_sha256=pre["sha"], pin_state_digest=pre["digest"])
    return c


def _build(run_dir, cfg, *, values=UP, fault=None):
    return T.trainer(run_dir, cfg, *_PROV, eval_override=T.scripted(values), fault=fault)


def _art_json(rd, name):
    return json.loads((Path(rd) / "sealed" / name).read_text(encoding="utf-8"))


def test_mark0_is_never_evaluated():
    from v4train.schedule import eval_marks
    marks = tuple(round(m, 6) for m in eval_marks("MAIN"))
    assert marks == MARKS == tuple(round(m, 6) for m in CF.EVAL_MARKS) and 0.0 not in marks and marks[0] == 0.5


def test_frozen_c1_loads_theta_pre_before_step0_and_keeps_tables(real_classes, pre, tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(CF, "_CTX", _ctx(pre))
    seen = {}
    holder = {}

    def fault(u):
        if u == 50:
            sd = holder["tr"].module.state_dict()
            seen[50] = {k: sd[k].detach().clone() for k in sd}

    rd = tmp_path / "run"
    tr = _build(rd, _cfg(), fault=fault)
    holder["tr"] = tr
    assert state_digest(tr.adapter.broadcast_state()) == pre["digest"]               # theta_pre, before step 0
    assert tr.state.updates == 0 and len(tr.opt.state) == 0 and tr.auto_continue is False
    shared = set(tr.adapter.manifest.shared_keys)
    assert not (set(FK) & shared) and all(not dict(tr.module.named_parameters())[k].requires_grad for k in FK)
    n_opt = sum(p.numel() for g in tr.opt.param_groups for p in g["params"])
    assert n_opt == tr._p2_c1["trained_numel"] == tr.adapter.manifest.shared_numel < tr._p2_c1["total_shared_numel"]
    tr.run()
    assert 50 in seen and tr.state.updates >= 75
    end = tr.module.state_dict()
    for k in FK:                                                                      # bit-unchanged at 50 and at 6.0
        assert torch.equal(seen[50][k], pre["state"][k]) and torch.equal(end[k], pre["state"][k]), k
    moved = [k for k in end if k not in FK and end[k].is_floating_point() and not torch.equal(end[k], pre["state"][k])]
    assert "blocks.0.qkv.weight" in moved
    start = json.loads((rd / "RUN_START.json").read_text(encoding="utf-8"))
    ident = start["p2_c1"]["identity"]
    assert ident["init_checkpoint_sha256"] == pre["sha"] and ident["init_state_digest"] == pre["digest"]
    assert ident["frozen_item_tables"] == list(FK) and ident["continuation"] == "DISABLED_FIXED_6.0_EFE"
    assert start["p2_c1"]["loaded_state_digest"] == pre["digest"] and start["key_base"]["initialization_block"] \
        .startswith("P2_THETA_PRE|SASREC|checkpoint_sha256=" + pre["sha"])
    summ = _art_json(rd, "RUN_SUMMARY.json")                                          # no continuation
    assert summ["status"] == "DONE_MAIN" and summ["phase"] == "MAIN" and summ["latest_kept"] is False
    assert summ["continuation"] is None and "latest" not in summ["checkpoints"] and not tr.ckpt.exists("latest")
    assert not (rd / "sealed" / "RESULT_ROWS_EXTENDED.json").exists() and summ["p2_c1"]["arm"] == ident["arm"]
    rows = _art_json(rd, "RESULT_ROWS.json")["rows"]
    assert rows and all(r["method"] == "P2_C1_FT_CENTRAL_FROZEN" and r["row_status"] == "P2_CONTROL"
                        and r["c_matched_eligible"] is False and r["downstream_seed_allowed"] is False
                        and r["p2_init_checkpoint_sha256"] == pre["sha"] and r["p2_role"] == "CALIBRATION"
                        and r["manifest_role"] == "CALIBRATION" and r["p2_frozen_item_tables"] == list(FK)
                        for r in rows)
    got = CF.read_c1_run(rd)
    assert got["arm"] == "P2_C1_FT_CENTRAL_FROZEN" and got["mark"] == 6.0 and got["value"] == UP[6.0]
    assert got["trained_numel"] == tr._p2_c1["trained_numel"]
    import v4train.trainer as TR
    assert_raises(TR.TrainerError, tr.continue_run)


def test_full_c1_trains_tables_and_mark0_is_never_practical_best(real_classes, pre, tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(CF, "_CTX", _ctx(pre, freeze=False))
    rd = tmp_path / "run"
    tr = _build(rd, _cfg("FULL"), values=DOWN)
    assert set(FK) <= set(tr.adapter.manifest.shared_keys) and tr._p2_c1["freeze"] is None
    tr.run()
    end = tr.module.state_dict()
    assert all(not torch.equal(end[k], pre["state"][k]) for k in FK)
    got = CF.read_c1_run(rd)
    assert got["mark"] == 0.5 and got["value"] == DOWN[0.5]          # the first mark, never theta_pre (mark 0)
    assert _art_json(rd, "RUN_SUMMARY.json")["practical_best"]["mark"] == 0.5


@nc("the FULL variant must train the item tables (keeping them fixed is the FROZEN arm)")
def test_nc_full_variant_keeps_tables_fixed(real_classes, pre, tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(CF, "_CTX", _ctx(pre, freeze=False))
    tr = _build(tmp_path / "run", _cfg("FULL"), fault=T.interrupt_after(5))
    try:
        tr.run()
    except T.InjectedInterruption:
        pass
    end = tr.module.state_dict()
    assert all(torch.equal(end[k], pre["state"][k]) for k in FK)


def test_init_refusals(real_classes, pre, tmp_path, monkeypatch):
    import v4train.trainer as TR
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(CF, "_CTX", _ctx(pre, pins=False))            # the real PRE pins: a synthetic init is refused
    assert_raises(CF.CentralFTRefused, _build, tmp_path / "a", _cfg())
    main_pre = make_pre(tmp_path / "main_pre", manifest="MAIN_TRAIN_5_100_131072")     # trained on MAIN users
    monkeypatch.setattr(CF, "_CTX", _ctx(main_pre))
    assert_raises(CF.CentralFTRefused, _build, tmp_path / "b", _cfg())
    cbig = make_pre(tmp_path / "cbig", run_id="C_SMALL_SASREC_s2026")
    monkeypatch.setattr(CF, "_CTX", _ctx(cbig))
    assert_raises(CF.CentralFTRefused, _build, tmp_path / "c", _cfg())
    monkeypatch.setattr(CF, "_CTX", _ctx(pre, freeze=False))           # run id FROZEN, flag off
    assert_raises(CF.CentralFTRefused, _build, tmp_path / "d", _cfg("FROZEN"))
    monkeypatch.setattr(CF, "_CTX", None)                              # a P2_C1 run id without the shim: refused by
    import v4train.recipe as RC                                        # the peak wrapper (validate) or the trainer one
    assert_raises((RC.RecipeError, TR.TrainerError), _build, tmp_path / "e", _cfg())
    monkeypatch.setattr(CF, "_CTX", _ctx(pre))
    plain = T.trainer(tmp_path / "f", T.cfg("SMALL_PLAIN_s2026", "SASREC", manifest_registry=REG400), *_PROV)
    assert not hasattr(plain, "_p2_c1") and plain.auto_continue == plain.cfg.effective_auto_continuation
    assert state_digest(plain.adapter.broadcast_state()) != pre["digest"]       # other runs: untouched theta0


def test_resume_reapplies_and_refuses_another_theta_pre(real_classes, pre, tmp_path, monkeypatch):
    import v4train.trainer as TR
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(CF, "_CTX", _ctx(pre))
    rd = tmp_path / "run"
    cfg = _cfg(checkpoint_every=5)
    tr = _build(rd, cfg, fault=T.interrupt_after(10))
    try:
        tr.run()
    except T.InjectedInterruption:
        pass
    other = make_pre(tmp_path / "pre2", seed=93)
    monkeypatch.setattr(CF, "_CTX", _ctx(other))
    assert_raises(TR.TrainerError, _build(rd, cfg).resume)            # RUN_START's p2_c1 identity differs
    monkeypatch.setattr(CF, "_CTX", _ctx(pre))
    out = _build(rd, cfg).resume()
    assert out["status"] == "DONE_MAIN" and CF.read_c1_run(rd)["mark"] == 6.0


def test_selection_reader_reads_calib_rows_only_main_stays_sealed(tmp_path, monkeypatch):
    """PREG2 §6.3 S8: read_c1_run refuses a MAIN-role C1 run (or a CALIB identity on another eval manifest) from
    RUN_START alone, before any value-bearing file is opened."""
    opened = []
    real = CF._strict
    monkeypatch.setattr(CF, "_strict", lambda p: opened.append(Path(p).name) or real(p))
    for i, (role, ev) in enumerate((("MAIN", "MAIN_VALIDATION_5_100_131072"), ("CALIBRATION", "MAIN_VALIDATION_5_100_131072"))):
        rd = tmp_path / f"r{i}"
        (rd / "sealed").mkdir(parents=True)
        rid = CF.c1_run_id("FULL", role, 1e-4, 2026)
        ident = {"arm": "P2_C1_FT_CENTRAL_FULL", "variant": "FULL", "role": role}
        (rd / "RUN_START.json").write_text(json.dumps({"run_id": rid, "p2_c1": {"identity": ident},
                                                       "eval_provider": {"manifest_id": ev}}), encoding="utf-8")
        for name in ("RUN_SUMMARY.json", "RESULT_ROWS.json"):
            (rd / "sealed" / name).write_text('{"practical_best": {"value": 0.5}}', encoding="utf-8")
        opened.clear()
        err = None
        try:
            CF.read_c1_run(rd)
        except CF.CentralFTRefused as e:
            err = str(e)
        assert err and "S8" in err and opened == ["RUN_START.json"], (err, opened)

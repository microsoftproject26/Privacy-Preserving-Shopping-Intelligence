"""TRAIN-02R1 — the gate fixes F1, F3 and F6 (F2 and the launcher side of F6: tests_int/test_train02_launcher.py; F4 and
the 02R2 registry re-issue: test_train02_variants_hp.py).

  * F1: a non-finite or UNDEFINED CKPT_AVG metric records outcome CKPT_AVG_NON_FINITE (value None, no rows), still
    deletes the snapshots and the average (sha256 logged) and finishes DONE_TUNING; the P2 driver keeps the valid
    endpoint and never carries a missing CKPT_AVG;
  * F3: in production mode a missing THETA0_REGISTRY entry fails the theta0 builder (no write); the real run needs EQUAL
    for GRU_D128 / SASREC_D256_B2 x seeds 2026 / 2027;
  * F6: `p2_driver register-02b` writes TUNED_RECIPES_02B.json (O_EXCL, self-hash) from verified stage-2 files and
    refuses a trigger / a mismatch / another HP registry; a MAIN RunConfig with `recipe_id` accepts off-default values
    only as that recipe; a recipe not in 02b, an HP-trial id, a TUNING run with a recipe, a recipe with a calibration
    record are refused; a MAIN run with a recipe records recipe_id in RUN_START, the key base and every row, and a
    resume refuses a changed tuned-recipes file.
Negative controls: the old raise on a non-finite CKPT_AVG; a guard that passes a missing entry; a recipe lookup that
falls back to another recipe.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest

import v4train.ckpt_avg as CA
import v4train.p2_driver as D
import v4train.trainer as T
import v4train.tuned_recipes as TR
from trainkit import KEY, SCHEMA, assert_raises, cfg, interrupt_after, nc, raises_with, read, registry_keys, scripted, \
    trainer
from test_train02_tuning import VALS, tcfg
from v4train import hp_registry as H
from v4train import theta0_variants as T0
from v4train.recipe import RecipeError, RunConfig
from v4train.trainer import EvalOutcome, TrainerError

HP = H.load_hp_registry()
TRIALS = {t["hp_id"]: t for fam in HP["families"].values() for t in fam}


# ================================================================================================ F1
class AvgEvaluator:
    def __init__(self, mode):
        self.mode, self.ev = mode, scripted(VALS)

    def __call__(self, tr, mark):
        if "ckpt_avg" in tr.ckpt.index:
            if self.mode == "nan":
                return EvalOutcome(value=float("nan"), rows=[], secondary={})
            raise TrainerError("primary metric UNDEFINED on SYN: no rankable row")
        return self.ev(tr, mark)


def _check_non_finite_ckpt_avg(work, mode):
    tr = trainer(work / mode, tcfg(mode, "P2_GRU_D256_S1_s4"), eval_override=AvgEvaluator(mode))
    try:
        s = tr.run()
    except Exception as e:                                                        # the pre-F1 behaviour
        raise AssertionError(f"a non-finite CKPT_AVG failed the run: {type(e).__name__}: {e}")
    rd = work / mode
    assert s["status"] == "DONE_TUNING" and s["endpoint"]["value"] == 0.25
    assert s["ckpt_avg"]["outcome"] == CA.OUTCOME_NON_FINITE and s["ckpt_avg"]["value"] is None
    ca = read(rd / "CKPT_AVG.json")
    assert ca["status"] == "COMPLETE" and ca["outcome"] == "CKPT_AVG_NON_FINITE" and ca["non_finite_reason"]
    assert ca["rows_file"] is None and not (rd / "CKPT_AVG_ROWS.json").exists()
    deleted = {Path(json.loads(x)["deleted"]).stem for x in (rd / "RETENTION_LOG.jsonl").read_text(
        encoding="utf-8").splitlines()}
    assert deleted == {"latest", "snapshot_5.0", "snapshot_5.5", "snapshot_6.0", "ckpt_avg"}
    assert not list((rd / "checkpoints").glob("*.pt"))
    v = D.run_values(s)
    assert v["finite"] and v["endpoint"] == 0.25 and v["ckpt_avg"] is None
    assert v["ckpt_avg_outcome"] == "CKPT_AVG_NON_FINITE"
    assert D.exclusions({"r": v}, 0.25)["r"] is None, "the valid endpoint is kept"
    assert D.ckpt_avg_carry([0.25, 0.26], [v["ckpt_avg"], 0.30])["carry"] is False


@pytest.mark.parametrize("mode", ["nan", "undefined"])
def test_non_finite_ckpt_avg_keeps_the_endpoint(work, mode):
    _check_non_finite_ckpt_avg(work, mode)


@nc("the pre-F1 CKPT_AVG step that raises on a non-finite metric (the run stays PENDING_CKPT_AVG)")
def test_nc_non_finite_ckpt_avg_raises(work, monkeypatch):
    def old(v):
        if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
            raise CA.CkptAvgError("non-finite CKPT_AVG metric")                              # MUTANT
        return True
    monkeypatch.setattr(CA, "_finite", old)
    _check_non_finite_ckpt_avg(work, "nan")


# ================================================================================================ F3
def _check_guard_verdict():
    eq = {f"{v}/{s}": {"status": "EQUAL"} for v in ("GRU_D128", "SASREC_D256_B2") for s in (2026, 2027)}
    assert T0.guard_verdict(eq, production=True, require_coverage=True)["ok"]
    missing = dict(eq, **{"GRU_D128/2027": {"status": "NO_THETA0_REGISTRY_ENTRY_FOR_THIS_TORCH_BUILD"}})
    assert not T0.guard_verdict(missing, production=True, require_coverage=True)["ok"]
    assert T0.guard_verdict(missing, production=False, require_coverage=False)["ok"], "tests (small K) only"
    part = {k: v for k, v in eq.items() if not k.endswith("2027")}
    assert not T0.guard_verdict(part, production=True, require_coverage=True)["ok"], "both seeds x both builds"
    assert not T0.guard_verdict(dict(eq, **{"GRU_D128/2026": {"status": "DIFFERENT"}}), production=False,
                                require_coverage=False)["ok"]


def test_theta0_guard_verdict():
    _check_guard_verdict()


@nc("a production guard that passes a missing THETA0_REGISTRY entry (the pre-F3 rule)")
def test_nc_guard_passes_missing_entry(monkeypatch):
    real = T0.guard_verdict
    monkeypatch.setattr(T0, "guard_verdict", lambda g, **kw: real(g, production=False, require_coverage=False))
    _check_guard_verdict()


def test_theta0_missing_entry_blocks_the_write_at_full_k(tmp_path, monkeypatch):
    from v4fs.schemas import load_registered
    monkeypatch.setattr(T0, "registered_current", lambda *a, **k: None)
    res = T0.measure("L3_NEAR_DUPLICATE", [2026], ["GRU_D128"], K=T0.FULL_K, schema=load_registered("L3_NEAR_DUPLICATE"),
                     log=lambda *_: None)
    assert res["guard"]["GRU_D128/2026"]["status"].startswith("NO_THETA0") and not res["ok"]
    raises_with(T0.Theta0VariantsError, "failed a guard", T0.write, res, tmp_path / "t.json")
    assert not (tmp_path / "t.json").exists()


# ================================================================================================ F6 (+ TRAIN-02R2)
import p2kit  # noqa: E402
from p2kit import make_p2, register as _register, rehash_stage, under_v4  # noqa: E402


@pytest.fixture(autouse=True)
def _p2_cleanup():
    yield
    p2kit.cleanup()                                   # the V4-internal P2 fixture dirs (TRAIN-02R3)


def _registered(tmp_path):
    return _register(tmp_path)


def test_register_02b(tmp_path, capsys):
    out, w = _registered(tmp_path)
    reg = TR.load_tuned_recipes(out)
    assert set(reg["recipes"]) == {"02B_GRU", "02B_SASREC"} and reg["self_sha256"] == w["self_sha256"]
    g = reg["recipes"]["02B_GRU"]
    t = TRIALS["P2_GRU_D256_S1_s4"]
    assert (g["model_variant"], g["peak_lr"], g["dropout"], g["schedule"], g["source_hp_sha256"]) == \
        (t["model_variant"], t["peak_lr"], t["dropout"], t["schedule"], t["sha256"])
    assert reg["recipes"]["02B_SASREC"]["ckpt_avg_carried_to_p4"] is True and g["ckpt_avg_carried_to_p4"] is False
    for k in ("stage1_file", "stage2_file"):
        assert not Path(g[k]["path"]).is_absolute(), "V4-relative (R3)"
        assert g[k]["file_sha256"] == hashlib.sha256(TR.v4_abs(g[k]["path"]).read_bytes()).hexdigest()
    assert b"\r" not in out.read_bytes()
    raises_with(TR.TunedRecipeError, "never replaced", D.register_02b, under_v4(tmp_path / "p2"), out)
    assert D.main(["register-02b", "--p2-dir", str(under_v4(tmp_path / "p2")), "--out", str(out)]) == 2
    capsys.readouterr()
    assert not TR.REGISTRY_PATH.exists(), "TUNED_RECIPES_02B.json is written after P2, never here"


def test_register_one_family_at_a_time(tmp_path, capsys):
    """(a): GRU first (version 1, O_EXCL), then SASRec (version 2; v1 archived with O_EXCL; 02B_GRU byte-identical)."""
    p2 = make_p2(tmp_path / "p2", ("GRU", "SASREC"))
    out = tmp_path / "TUNED_RECIPES_02B.json"
    assert D.main(["register-02b", "--p2-dir", str(p2), "--out", str(out), "--family", "GRU"]) == 0
    capsys.readouterr()
    r1 = TR.load_tuned_recipes(out)
    assert set(r1["recipes"]) == {"02B_GRU"} and r1["version"] == 1
    w2 = D.register_02b(p2, out, families=("SASREC",))
    r2 = TR.load_tuned_recipes(out)
    assert set(r2["recipes"]) == {"02B_GRU", "02B_SASREC"} and r2["version"] == 2
    assert r2["recipes"]["02B_GRU"] == r1["recipes"]["02B_GRU"]
    arch = Path(w2["archived_previous"])
    assert arch.name == "TUNED_RECIPES_02B.v1.json" and TR.load_tuned_recipes(arch)["self_sha256"] == r1["self_sha256"]
    assert json.loads(out.read_text(encoding="utf-8"))["previous_self_sha256"] == r1["self_sha256"]
    raises_with(TR.TunedRecipeError, "never replaced", D.register_02b, p2, out, families=("GRU",))
    raises_with(TR.TunedRecipeError, "families must be distinct", D.register_02b, p2, tmp_path / "x.json",
                families=("GRU", "GRU"))


def _check_forged_stage2_refused(tmp_path):
    p2 = make_p2(tmp_path / "p2", ("GRU", "SASREC"))
    s2p = p2 / "P2_GRU_STAGE2.json"
    orig = json.loads(s2p.read_text(encoding="utf-8"))
    # (1) an edited, re-hashed stage-2 file: the D128 config recorded as tuned (it lost on the 2-seed mean)
    forged = json.loads(json.dumps(orig))
    d128 = forged["selection"]["configs"]["GRU_D128"]
    t = TRIALS[d128["hp_id"]]
    forged["selection"]["tuned"] = {"hp_id": t["hp_id"], "model_variant": t["model_variant"], "peak_lr": t["peak_lr"],
                                    "schedule": t["schedule"], "dropout": t["dropout"], "mean_6.0": d128["mean"]}
    rehash_stage(s2p, forged)
    raises_with(TR.TunedRecipeError, "differs from the recomputed", D.register_02b, p2, tmp_path / "o1.json",
                families=("GRU",))
    # (2) a tuned hp_id that is not among the stage-2 configs (a stage-1 trial never re-run at 2027)
    other = [x for x in HP["families"]["GRU"] if x["hp_id"] not in json.loads((p2 / "P2_GRU_STAGE1.json").read_text(
        encoding="utf-8"))["selection"]["stage2_plan"].values()][0]
    forged = json.loads(json.dumps(orig))
    forged["selection"]["tuned"] = {"hp_id": other["hp_id"], "model_variant": other["model_variant"],
                                    "peak_lr": other["peak_lr"], "schedule": other["schedule"],
                                    "dropout": other["dropout"], "mean_6.0": 0.99}
    rehash_stage(s2p, forged)
    raises_with(TR.TunedRecipeError, "differs from the recomputed", D.register_02b, p2, tmp_path / "o2.json",
                families=("GRU",))
    # (3) a stage-2 file that names another stage-1 file
    forged = json.loads(json.dumps(orig))
    forged["stage1_file"] = {"self_sha256": "0" * 64}
    rehash_stage(s2p, forged)
    raises_with(TR.TunedRecipeError, "does not name this stage-1 file", D.register_02b, p2, tmp_path / "o3.json",
                families=("GRU",))
    # (4) a forged stage-1 plan (re-hashed): the recorded plan differs from the recomputed one
    s2p.write_text(json.dumps(orig, indent=1, sort_keys=True), encoding="utf-8")
    s1p = p2 / "P2_GRU_STAGE1.json"
    s1 = json.loads(s1p.read_text(encoding="utf-8"))
    s1["selection"]["stage2_plan"]["GRU_D128"] = HP["families"]["GRU"][0]["hp_id"]
    rehash_stage(s1p, s1)
    raises_with(TR.TunedRecipeError, "", D.register_02b, p2, tmp_path / "o4.json", families=("GRU",))


def test_forged_stage_files_refused(tmp_path):
    _check_forged_stage2_refused(tmp_path)


@nc("a registration that trusts the stage-2 file's recorded selection (the pre-R2 behaviour)")
def test_nc_trusts_recorded_selection(tmp_path, monkeypatch):
    def lax(fam, s1, s2, hp):
        tuned = s2["selection"]["tuned"]
        return {"tuned": tuned, "trial": H.trial(hp, tuned["hp_id"]),
                "carry": bool(s2["selection"].get("ckpt_avg_carried_to_p4"))}
    monkeypatch.setattr(TR, "verify_stages", lax)                                               # MUTANT
    _check_forged_stage2_refused(tmp_path)


def test_register_02b_refusals_and_reverification(tmp_path):
    p2 = make_p2(tmp_path / "p2", ("GRU",))
    raises_with(TR.TunedRecipeError, "does not exist", D.register_02b, p2, tmp_path / "o.json", families=("SASREC",))
    out, _ = _registered(tmp_path / "t")
    reg = TR.load_tuned_recipes(out)
    s2 = TR.v4_abs(reg["recipes"]["02B_SASREC"]["stage2_file"]["path"])
    s2.write_text(s2.read_text(encoding="utf-8") + " ", encoding="utf-8")                  # the stage file changed
    raises_with(TR.TunedRecipeError, "missing or changed", TR.load_tuned_recipes, out)
    obj = json.loads(out.read_text(encoding="utf-8"))
    obj["recipes"]["02B_GRU"]["peak_lr"] = 1e-3
    out.write_text(json.dumps(obj), encoding="utf-8")
    raises_with(TR.TunedRecipeError, "self-hash", TR.load_tuned_recipes, out)


BASE = dict(run_id="m", exposure_point="end", cuda_visible_devices="0", determinism_mode="STRICT", key=dict(KEY),
            **registry_keys())


def _main_cfg(reg, rid="02B_GRU", fam="GRU", **over):
    r = TR.recipe(TR.load_tuned_recipes(reg), rid if rid in ("02B_GRU", "02B_SASREC") else "02B_GRU")
    kw = dict(BASE, family=fam, seed=2026, phase="MAIN", recipe_id=rid, recipe_registry=str(reg),
              peak_lr=r["peak_lr"], model_variant=r["model_variant"], dropout=tuple(r["dropout"]),
              schedule=r["schedule"], theta0_init_sha256="a" * 64, test_overrides=True)
    kw.update(over)
    return RunConfig(**kw)


def _check_recipe_rule(tmp_path):
    reg, _ = _registered(tmp_path)
    c = _main_cfg(reg)
    c.validate()
    assert (c.run_label, c.effective_variant, c.effective_schedule) == ("SMOKE_NOT_EVIDENCE", "GRU_D256", "S1")
    assert {"recipe_id", "recipe_registry"} <= set(c.as_json())
    for over, field in ((dict(peak_lr=1e-3), "peak_lr"), (dict(schedule="S0"), "schedule"),
                        (dict(dropout=(0.2, 0.1)), "dropout"), (dict(model_variant="GRU_D128"), "model_variant")):
        raises_with(RecipeError, f"'{field}'", dataclasses.replace(c, **over).validate)
    raises_with(RecipeError, "not in the tuned-recipes registry", _main_cfg(reg, rid="P2_GRU_D256_S1_s4").validate)
    raises_with(RecipeError, "not in the tuned-recipes registry", _main_cfg(reg, rid="02B_GRU_X").validate)
    raises_with(RecipeError, "is a GRU recipe, not SASREC", _main_cfg(reg, fam="SASREC", model_variant=None,
                                                                      dropout=None).validate)


def test_main_recipe_rule(tmp_path):
    _check_recipe_rule(tmp_path)
    reg = tmp_path / "TUNED_RECIPES_02B.json"
    c = _main_cfg(reg)
    raises_with(RecipeError, "hp_id is valid for role TUNING only",
                dataclasses.replace(c, recipe_id=None, recipe_registry=None, hp_id="P2_GRU_D256_S1_s4").validate)
    raises_with(RecipeError, "never a recipe_id", dataclasses.replace(c, role="TUNING", phase="CALIBRATION", hp_id="P2_GRU_D256_S1_s4").validate)
    raises_with(RecipeError, "MAIN-phase COHORT run only", dataclasses.replace(c, phase="CALIBRATION").validate)
    raises_with(RecipeError, "not a calibration record", dataclasses.replace(c, calibration_record="c.json").validate)
    raises_with(RecipeError, "needs theta0_init_sha256", dataclasses.replace(c, theta0_init_sha256=None,
                                                                        test_overrides=False).validate)
    dataclasses.replace(c, theta0_init_sha256=None, test_overrides=True).validate()
    raises_with(RecipeError, "does not exist", dataclasses.replace(c, recipe_registry=str(tmp_path / "no.json"))
                .validate)
    raises_with(RecipeError, "not in the registered set",                         # without a recipe: unchanged rules
                dataclasses.replace(c, recipe_id=None, recipe_registry=None).validate)


@nc("a recipe lookup that falls back to another registered recipe")
def test_nc_recipe_lookup_fallback(tmp_path, monkeypatch):
    real = TR.recipe
    monkeypatch.setattr(TR, "recipe", lambda reg, rid: dict(real(reg, "02B_GRU"), recipe_id=rid))     # MUTANT
    _check_recipe_rule(tmp_path)


def test_main_run_with_a_recipe_records_it(work, tmp_path):
    reg, _ = _registered(tmp_path)
    r = TR.recipe(TR.load_tuned_recipes(reg), "02B_GRU")
    c = cfg("p3", "GRU", recipe_id="02B_GRU", recipe_registry=str(reg), peak=r["peak_lr"],
            model_variant=r["model_variant"], dropout=tuple(r["dropout"]), schedule=r["schedule"])
    vals = {round(0.5 * i, 6): 0.2 + 0.001 * i for i in range(1, 13)}
    tr = trainer(work / "p3", c, eval_override=scripted(vals))
    assert tr.key_base["recipe_id"] == "02B_GRU" and tr.key_base["recipe_sha256"] == r["sha256"]
    s = tr.run()
    start = read(work / "p3" / "RUN_START.json")
    assert start["tuned_recipe"]["recipe_id"] == "02B_GRU" and start["lr_schedule_id"] == "S1"
    assert start["model_variant"] == "GRU_D256" and s["status"] == "DONE_MAIN"
    rows = read(work / "p3" / "sealed" / "RESULT_ROWS.json")["rows"]                  # TRAIN-03 seal
    assert rows and all(r_["recipe_id"] == "02B_GRU" and r_["model_variant"] == "GRU_D256" for r_ in rows)
    a = trainer(work / "cut", c, eval_override=scripted(vals), fault=interrupt_after(10))
    assert_raises(T.InjectedInterruption, a.run)
    obj = json.loads(reg.read_text(encoding="utf-8"))
    reg.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")        # same content, other bytes
    # TRAIN-03: the resume binds the RECIPE entry (recipe_sha256), so a re-written registry file with the same GRU
    # entry (e.g. the part-2 registration) resumes; a tampered registry is refused by its own self-hash
    assert trainer(work / "cut", c, eval_override=scripted(vals)).resume()["status"] == "DONE_MAIN"
    obj["recipes"]["02B_GRU"]["peak_lr"] = 1e-3
    reg.write_text(json.dumps(obj), encoding="utf-8")
    raises_with(RecipeError, "self-hash mismatch", trainer, work / "cut2", c, eval_override=scripted(vals))


# ================================================================================================ TRAIN-02R2 F2
def _check_production_pin(tmp_path, monkeypatch):
    reg, _ = _registered(tmp_path)
    c = dataclasses.replace(_main_cfg(reg), test_overrides=False)
    raises_with(RecipeError, "must use the production tuned-recipes registry", c.validate)       # any other path
    monkeypatch.setattr(TR, "REGISTRY_PATH", reg)                                              # the production path
    gate = tmp_path / "G1B_APPROVED.json"
    monkeypatch.setattr(TR, "G1B_GATE", gate)
    raises_with(RecipeError, "not approved (G1b)", c.validate)                                   # no G1b file
    part = lambda h: [{"part": 1, "family": "GRU", "self_sha256": h}]                   # noqa: E731 (TRAIN-03 format)
    gate.write_text(json.dumps(TR.g1b_obj(part("0" * 64))), encoding="utf-8")
    raises_with(RecipeError, "not the one pinned at G1b", c.validate)
    gate.write_text(json.dumps({"gate": "G1b"}), encoding="utf-8")
    raises_with(RecipeError, "is not a PPSI_V4_G1B_APPROVED_1 object", c.validate)                # not the G1B format
    gate.write_text(json.dumps(TR.g1b_obj(part(TR.load_tuned_recipes(reg)["self_sha256"]))), encoding="utf-8")
    c.validate()                                                                                  # pinned: accepted


def test_non_test_recipe_needs_the_pinned_production_registry(tmp_path, monkeypatch):
    _check_production_pin(tmp_path, monkeypatch)


@nc("a non-test recipe run that accepts any self-consistent tuned-recipes file (no G1b pin)")
def test_nc_no_production_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(TR, "check_production_registry", lambda path, reg, gate_path=None: None)   # MUTANT
    _check_production_pin(tmp_path, monkeypatch)


# ================================================================================================ TRAIN-02R3
R5_STAGE_KEYS = {"format", "task", "family", "stage", "seed", "hp_registry", "stage1_file", "runs", "values",
                 "selection", "run_label", "rules", "utc", "self_sha256"}
R5_RUN_KEYS = {"hp_id", "seed", "run_dir", "cmd", "already_done", "exit_code", "wall_s"}
R5_VALUE_KEYS = {"status", "endpoint", "ckpt_avg", "ckpt_avg_outcome", "finite", "shared_numel", "exit_code",
                 "values_at_marks", "peak_lr", "model_variant", "seed"}


def test_r5_driver_format_is_what_02b_consumes(tmp_path):
    """The stage files the R5-pack driver (TRAIN-02R1 p2_driver, sha ff557696...) writes: exactly these keys; the
    run_stage / run_values code that writes them is unchanged since TRAIN-02R1 (only register-02b changed). 02b
    registers them without any field the R5 driver does not write."""
    out, _ = _registered(tmp_path)
    p2 = under_v4(tmp_path / "p2")
    for fam in ("GRU", "SASREC"):
        for k in (1, 2):
            obj = json.loads((p2 / f"P2_{fam}_STAGE{k}.json").read_text(encoding="utf-8"))
            assert set(obj) == R5_STAGE_KEYS, set(obj) ^ R5_STAGE_KEYS
            assert all(set(r) == R5_RUN_KEYS for r in obj["runs"])
            assert all(set(v) == R5_VALUE_KEYS for v in obj["values"].values())
    reg = TR.load_tuned_recipes(out)
    ev = reg["recipes"]["02B_GRU"]["run_evidence"]
    assert len(ev["stage1"]) == 12 and len(ev["stage2"]) == 2
    for e in list(ev["stage1"].values()) + list(ev["stage2"].values()):
        assert not Path(e["run_dir"]).is_absolute() and len(e["summary_sha256"]) == 64 and len(e["result_rows_sha256"]) == 64


def _forge_evidence(p2, delta):
    """Rewrite every run's RUN_SUMMARY / RESULT_ROWS with shifted values; the stage pair stays self-consistent."""
    for fam in ("GRU",):
        for k in (1, 2):
            for run in json.loads((p2 / f"P2_{fam}_STAGE{k}.json").read_text(encoding="utf-8"))["runs"]:
                p2kit._run(run["cmd"], delta)


def _check_evidence_bound(tmp_path):
    p2 = make_p2(tmp_path / "p2", ("GRU",))
    _forge_evidence(p2, 0.002)
    raises_with(TR.TunedRecipeError, "differ from the ones recomputed", D.register_02b, p2, tmp_path / "o.json",
                families=("GRU",))
    assert not (tmp_path / "o.json").exists()


def test_values_bound_to_run_summaries(tmp_path):
    _check_evidence_bound(tmp_path)


@nc("a registration that does not re-read the run dirs (values trusted from the stage files)")
def test_nc_values_not_bound_to_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(TR, "run_evidence", lambda stage, stored=None: {})                        # MUTANT
    _check_evidence_bound(tmp_path)


def test_evidence_reverified_on_load_and_endpoint_rows(tmp_path):
    out, _ = _registered(tmp_path)
    reg = TR.load_tuned_recipes(out)
    e = reg["recipes"]["02B_GRU"]["run_evidence"]["stage2"]
    rd = TR.v4_abs(next(iter(e.values()))["run_dir"])
    sp = rd / "RUN_SUMMARY.json"
    sp.write_text(sp.read_text(encoding="utf-8") + " ", encoding="utf-8")         # same values, other bytes
    raises_with(TR.TunedRecipeError, "changed since registration", TR.load_tuned_recipes, out)
    # an endpoint row that disagrees with the RUN_SUMMARY is refused at registration
    p2 = make_p2(tmp_path / "q", ("GRU",))
    s1 = json.loads((p2 / "P2_GRU_STAGE1.json").read_text(encoding="utf-8"))
    rp = TR.v4_abs(TR.v4_rel(s1["runs"][0]["run_dir"])) / "RESULT_ROWS.json"
    rows = json.loads(rp.read_text(encoding="utf-8"))
    rows["rows"][0]["value"] = 0.999
    rp.write_text(json.dumps(rows), encoding="utf-8")
    raises_with(TR.TunedRecipeError, "does not match the RUN_SUMMARY endpoint", D.register_02b, p2,
                tmp_path / "o2.json", families=("GRU",))


def test_02b_loads_from_another_cwd(tmp_path, monkeypatch):
    out, _ = _registered(tmp_path)
    obj = json.loads(out.read_text(encoding="utf-8"))
    for r in obj["recipes"].values():
        assert not Path(r["stage1_file"]["path"]).is_absolute() and not Path(r["stage2_file"]["path"]).is_absolute()
    elsewhere = tmp_path / "elsewhere" / "deep"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    assert set(TR.load_tuned_recipes(out.resolve())["recipes"]) == {"02B_GRU", "02B_SASREC"}
    raises_with(TR.TunedRecipeError, "not under the V4 root", TR.v4_rel, tmp_path / "outside")
    raises_with(TR.TunedRecipeError, "not a V4-relative path", TR.v4_abs, "../escape")

"""The FULL_TRAIN population (the PREG1 T1 tie-break) with the same recipe — no other change.

  * a FULL_TRAIN run trains on M_FULL_TRAIN and evaluates on the full warm VALIDATION manifest with the identical
    recipe, schedule, exposure point, stopping rule and evaluator; only the population differs (config hash aside);
  * a non-test FULL_TRAIN run needs N = 40,180,608 exactly and the cohort-calibrated peak (calibration_record);
    CALIBRATION or a continuation budget are refused (no new calibration, no continuation);
  * its rows are ARCHITECTURE_TIEBREAK / population ALL / HISTORICAL_REFERENCE / never C_MATCHED; its summary says so;
  * read rule support: while a tie-break run is live, STATUS.json and EVENTS.jsonl carry no metric value (sealed);
  * retention: strictly-best + endpoint_6.0 only — the 6.0 resumable is deleted even when the run is STILL_IMPROVING,
    and continue_run is refused;
  * roles: a COHORT config refuses M_FULL_TRAIN, a FULL_TRAIN config refuses M01.
Negative control: a tie-break run that keeps its resumable like a still-improving contest run.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from trainkit import art  # noqa: E402  (TRAIN-03: sealed artifact root)
from trainkit import (FULL_EVAL_MANIFEST, FULL_TRAIN_MANIFEST, K, WIDTHS, assert_raises, cfg, nc, providers, ramp,
                      read, scripted, trainer, write_registry)
from v4train.provider import SyntheticTrainProvider
from v4train.recipe import FULL_TRAIN_N, RecipeError
from v4train.schedule import eval_marks
from v4train.selection import SelectionError, assert_c_matched_eligible
from v4train.trainer import TrainerError

MAIN = eval_marks("MAIN")


def _full(work, name="F", vals=None):
    c = cfg(name, "GRU", population="FULL_TRAIN")
    tr = trainer(work / name, c, eval_override=scripted(vals or ramp(MAIN, 6.0)))
    return tr, tr.run()


def _check_tiebreak_retention(run_dir):
    files = sorted(p.name for p in (art(run_dir) / "checkpoints").glob("*.pt"))
    assert files == ["best.pt", "endpoint_6.0.pt"], f"tie-break keeps strictly-best + endpoint only, found {files}"


def test_full_train_run_same_recipe_and_labels(work):
    tr, s = _full(work)
    assert tr.train.manifest_id == FULL_TRAIN_MANIFEST and tr.evalp.manifest.manifest_id == FULL_EVAL_MANIFEST
    assert s["status"] == "DONE_MAIN" and s["population"] == "FULL_TRAIN" and s["row_status"] == "ARCHITECTURE_TIEBREAK"
    assert s["still_improving"] and not s["latest_kept"]
    _check_tiebreak_retention(work / "F")
    rows = read(art(work / "F") / "RESULT_ROWS.json")["rows"]
    assert rows and all(r["row_status"] == "ARCHITECTURE_TIEBREAK" and r["population_scope"] == "ALL"
                        and r["comparison_status"] == "HISTORICAL_REFERENCE" and r["c_matched_eligible"] is False
                        and r["downstream_seed_allowed"] is False for r in rows)
    for r in rows:
        assert_raises(SelectionError, assert_c_matched_eligible, r)
    st = read(work / "F" / "RUN_START.json")
    cohort = trainer(work / "C", cfg("F", "GRU"), eval_override=scripted(ramp(MAIN, 6.0)))
    norm = lambda d: json.loads(json.dumps(d, sort_keys=True))     # noqa: E731  (tuples -> lists as on disk)
    same = {k: v for k, v in norm(st["config"]).items() if k not in ("population",)}
    other = {k: v for k, v in norm(cohort._run_start_record({"state": "OK"})["config"]).items()
             if k not in ("population",)}
    assert same == other, "the tie-break differs from the contest only in the population"
    assert st["planned_lr_table"]["sha256"] == cohort.planned["sha256"]


def test_tiebreak_values_are_sealed_while_live(work):
    _full(work)
    events = [json.loads(x) for x in (work / "F" / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines()]
    evs = [e for e in events if e.get("event") == "EVAL"]
    assert len(evs) == 12 and all(e["value"] == "SEALED_UNTIL_STAGE_COMPLETE" and e["improved"] is None for e in evs)
    assert "last_value" not in read(work / "F" / "STATUS.json") or \
        read(work / "F" / "STATUS.json").get("last_value") in (None, "SEALED_UNTIL_STAGE_COMPLETE")


def test_full_train_guards(work):
    assert_raises(RecipeError, cfg("x", "GRU", population="FULL_TRAIN", phase="CALIBRATION").validate)
    assert_raises(RecipeError, cfg("x", "GRU", population="FULL_TRAIN", continuation_budget="b.json").validate)
    non_test = dataclasses.replace(cfg("x", "GRU", population="FULL_TRAIN"), test_overrides=False,
                                   effective_batch=256)
    assert_raises(RecipeError, non_test.validate)                                   # no calibration record
    with_cal = dataclasses.replace(non_test, calibration_record="CALIBRATION_GRU.json")
    with_cal.validate()
    tp = SyntheticTrainProvider(1024, K, WIDTHS, seed=3, manifest_id=FULL_TRAIN_MANIFEST)
    _, ep = providers(population="FULL_TRAIN")
    with_cal = dataclasses.replace(with_cal, manifest_registry=write_registry({"FULL_TRAIN": (tp, ep)}))  # N registered
    with pytest.raises(TrainerError, match="the full-TRAIN tie-break has N = 40180608"):
        trainer(work / "n", with_cal, tp, ep)                                          # N != 40,180,608 (HISTORICAL)
    assert FULL_TRAIN_N == 40_180_608
    tpc, epc = providers()
    assert_raises(TrainerError, trainer, work / "a", cfg("a", "GRU"), tp, ep)          # cohort config, full manifests
    assert_raises(TrainerError, trainer, work / "b", cfg("b", "GRU", population="FULL_TRAIN"), tpc, epc)
    tr, _ = _full(work, "F2")
    assert_raises(TrainerError, trainer(work / "F2", cfg("F2", "GRU", population="FULL_TRAIN"),
                                        eval_override=scripted(ramp(MAIN, 6.0))).continue_run)


@nc("a tie-break run that keeps its 6.0 resumable like a STILL_IMPROVING contest run")
def test_nc_tiebreak_keeps_resumable(work, monkeypatch):
    import v4train.trainer as T
    orig = T.CentralTrainer._finish_main

    def as_contest(self):
        self.tiebreak = False                                     # MUTANT: the retention ignores the tie-break rule
        orig(self)
    monkeypatch.setattr(T.CentralTrainer, "_finish_main", as_contest)
    _full(work)
    _check_tiebreak_retention(work / "F")

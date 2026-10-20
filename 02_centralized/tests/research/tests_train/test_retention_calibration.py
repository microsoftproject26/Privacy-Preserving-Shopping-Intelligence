"""Checkpoint retention (DISK_RETENTION_PLAN rules 1, 2, 4; E12.1.8) and calibration mode (G0 D6 L37).

  * a main run that is not STILL_IMPROVING ends with exactly best.pt + endpoint_6.0.pt: `latest` is deleted at 6.0 and
    the deletion is logged with the removed file's sha256; CHECKPOINTS.json matches the files; best is the
    PRACTICAL_BEST mark, the endpoint is 6.0; the int32 rank file of every evaluation is retained;
  * a STILL_IMPROVING run keeps its 6.0 resumable (the only state a continuation may start from);
  * calibration: two arms (1.0 and 0.5 x 1e-3), seed 2026, the same fresh theta0, 2.0 EFE, evaluated on M06; no weight
    file remains anywhere, every arm's resumable deletion is logged, the chosen peak is the arm with the higher
    MRR@20 at 2.0 (tie -> the executed peak), recorded in CALIBRATION_<FAMILY>.json; a main run restarts fresh from
    build(seed) and refuses a peak that differs from the calibration record;
Negative controls: a retention step that skips the deletion; a calibration that keeps its resumable.
"""
from __future__ import annotations

import pytest

from trainkit import art  # noqa: E402  (TRAIN-03: sealed artifact root)
from trainkit import BIG_DISK, K, N, SCHEMA, assert_raises, cfg, nc, providers, ramp, read, scripted, trainer
from v4train.calibration import choose_peak, run_calibration
from v4train.checkpoint import CheckpointStore, sha256_file
from v4train.schedule import eval_marks
from v4train.trainer import TrainerError

MAIN_MARKS = eval_marks("MAIN")


def _check_retention(run_dir, expect_latest: bool):
    files = sorted(p.name for p in (art(run_dir) / "checkpoints").glob("*.pt"))
    want = ["best.pt", "endpoint_6.0.pt"] + (["latest.pt"] if expect_latest else [])
    assert files == sorted(want), f"retained {files}, expected {sorted(want)}"
    idx = read(art(run_dir) / "CHECKPOINTS.json")
    assert sorted(idx) == sorted(w[:-3] for w in want)
    for k, rec in idx.items():
        assert sha256_file(art(run_dir) / rec["file"]) == rec["sha256"]
    s = read(art(run_dir) / "RUN_SUMMARY.json")
    assert idx["best"]["meta"]["mark"] == s["practical_best"]["mark"] and idx["endpoint_6.0"]["meta"]["mark"] == 6.0
    if not expect_latest:
        log = (art(run_dir) / "RETENTION_LOG.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(log) == 1 and s["latest_deleted_sha256"] in log[0] and "latest.pt" in log[0]


def test_retention_main_not_still_improving_real_eval(work):
    s = trainer(work / "r", cfg("ret", "GRU")).run()                       # real v4eval; GRU saturates early
    assert not s["still_improving"]
    _check_retention(work / "r", expect_latest=False)
    ranks = sorted(p.name for p in (art(work / "r") / "evals").glob("*.ranks_int32.npy"))
    assert len(ranks) == 12, "one retained int32 rank file per evaluation (DISK rule 4)"


def test_retention_still_improving_keeps_the_resumable(work):
    s = trainer(work / "r", cfg("ret2", "GRU"), eval_override=scripted(ramp(MAIN_MARKS, 6.0))).run()
    assert s["still_improving"]
    _check_retention(work / "r", expect_latest=True)
    assert read(art(work / "r") / "CHECKPOINTS.json")["latest"]["meta"]["exposures"] == 6 * N


def _calibrate(work, family="GRU", override=None):
    tp, ep = providers("CALIBRATION")
    kw = {"eval_override": override} if override else {}
    return run_calibration(family, SCHEMA, tp, ep, work / "cal", key=dict(cfg("x", family).key), exposure_point="end",
                           cuda_visible_devices=cfg("x", family).cuda_visible_devices,
                           determinism_mode=cfg("x", family).determinism_mode,
                           manifest_registry=cfg("x", family).manifest_registry,       # TRAIN-01R2 keys, explicit
                           data_version=cfg("x", family).data_version,
                           effective_batch=32,
                           test_overrides=True, model_kw={"K": K}, disk_usage_fn=BIG_DISK, trainer_kw=kw, log_every=5)


def test_calibration_keeps_no_weights_and_chooses_by_mrr_at_2(work):
    rec = _calibrate(work)
    assert not list((work / "cal").rglob("*.pt")), "calibration must keep no weights"
    assert [a["multiplier"] for a in rec["arms"]] == [1.0, 0.5] and rec["seed"] == 2026
    assert [a["peak_lr"] for a in rec["arms"]] == [1e-3, 5e-4]
    assert len({a["init_sha256"] for a in rec["arms"]}) == 1, "both arms start from the same fresh theta0"
    for a in rec["arms"]:
        assert [c["mark"] for c in a["curve"]] == [0.5, 1.0, 1.5, 2.0] and a["latest_deleted_sha256"]
        log = (work / "cal" / f"peak_x{a['multiplier']:g}" / "RETENTION_LOG.jsonl").read_text(encoding="utf-8")
        assert a["latest_deleted_sha256"] in log
    v = {a["multiplier"]: a["value_at_2.0"] for a in rec["arms"]}
    assert rec["chosen_multiplier"] == (1.0 if v[1.0] >= v[0.5] - 1e-8 else 0.5)
    assert rec["chosen_peak_lr"] == rec["chosen_multiplier"] * 1e-3
    on_disk = read(work / "cal" / "CALIBRATION_GRU.json")
    assert on_disk["eval_manifest"] == "M06_CALIB_VALIDATION_WARM" and on_disk["train_manifest"] == "M05_CALIB_TRAIN"
    assert on_disk["weights_retained"] == [] and on_disk["main_runs_restart_fresh"] is True
    # TRAIN-01R2: a test calibration is labelled SMOKE_NOT_EVIDENCE and records its registry / data version / schema
    assert on_disk["run_label"] == "SMOKE_NOT_EVIDENCE" and on_disk["evidence"] is False
    assert on_disk["data_version"] == cfg("x", "GRU").data_version and on_disk["manifest_registry"]["role"] == \
        "CALIBRATION" and on_disk["schema"]["schema_id"] == SCHEMA.schema_id
    # the main run restarts fresh and must use the calibrated peak
    wrong = 5e-4 if rec["chosen_peak_lr"] == 1e-3 else 1e-3
    bad = cfg("main_bad", "GRU", peak=wrong, calibration_record=str(work / "cal" / "CALIBRATION_GRU.json"))
    assert_raises(TrainerError, trainer(work / "m_bad", bad).run)
    good = cfg("main_ok", "GRU", peak=rec["chosen_peak_lr"],
               calibration_record=str(work / "cal" / "CALIBRATION_GRU.json"))
    tr = trainer(work / "m_ok", good, eval_override=scripted({m: 0.1 for m in MAIN_MARKS}))
    assert tr.built.init_sha256 == rec["arms"][0]["init_sha256"]              # fresh build(2026), no weights reused
    start_rec = tr._run_start_record({"state": "OK"})
    assert start_rec["calibration_record"]["chosen_peak_lr"] == rec["chosen_peak_lr"]


def test_choose_peak_rule():
    assert choose_peak({1.0: 0.30, 0.5: 0.31}) == 0.5
    assert choose_peak({1.0: 0.31, 0.5: 0.30}) == 1.0
    assert choose_peak({1.0: 0.3, 0.5: 0.3 + 5e-9}) == 1.0                  # exact tie keeps the executed peak


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a retention step that never deletes the 6.0 resumable of a finished run")
def test_nc_retention_skips_deletion(work, monkeypatch):
    monkeypatch.setattr(CheckpointStore, "delete", lambda self, kind, reason: None)
    trainer(work / "r", cfg("ret_nc", "GRU"), eval_override=scripted(ramp(MAIN_MARKS, 3.0))).run()
    _check_retention(work / "r", expect_latest=False)


@nc("a calibration that keeps its resumable weights")
def test_nc_calibration_keeps_weights(work, monkeypatch):
    monkeypatch.setattr(CheckpointStore, "delete", lambda self, kind, reason: None)
    monkeypatch.setattr(CheckpointStore, "weight_files", lambda self: [])
    tp, ep = providers("CALIBRATION")
    trainer(work / "cal", cfg("cal_nc", "GRU", phase="CALIBRATION"), tp, ep,
            eval_override=scripted({m: 0.1 for m in eval_marks("CALIBRATION")})).run()
    assert not list((work / "cal").rglob("*.pt")), "calibration must keep no weights"

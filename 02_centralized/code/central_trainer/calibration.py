"""v4train.calibration — the bounded, symmetric LR calibration of one family (Fable G0 D6 L37; PKG 08 §4).

Two arms, peak LR in {1.0, 0.5} x the executed 1e-3, each a FRESH init at seed 2026 (v4models.build; the same theta0
for both arms), trained on M05_CALIB_TRAIN with the common schedule up to 2.0 EFE and evaluated every 0.5 EFE on
M06_CALIB_VALIDATION_WARM. The chosen peak is the arm with the higher client-macro MRR@20 at 2.0 EFE; an exact tie
(|delta| <= 1e-8) keeps the executed peak (1.0 x), the verified successful configuration (PLAN; PKG 08 §4 "otherwise
use verified successful configs"). Metrics, curves and the choice go to CALIBRATION_<FAMILY>.json. No weights are
retained (DISK_RETENTION_PLAN rule 1): each arm's resumable is deleted when it completes, with its sha256 logged, and
the function refuses to report success if any weight file remains. The main runs restart fresh from build(seed).
TRAIN-01R2: the train / eval manifests are the manifest registry's CALIBRATION role (RunConfig manifest_registry and
data_version are required and forwarded through **config_kw). A non-test calibration whose run label would not be
PRODUCTION (e.g. the HISTORICAL data version) is refused before any arm runs; a PRODUCTION calibration is chosen by
v4train.selection.calibration_choice from the arms' 2.0-EFE rows (PRODUCTION rows only) and the record is evidence. A
test_overrides calibration records the same arithmetic choice labelled SMOKE_NOT_EVIDENCE (evidence false), which no
non-test run accepts (selection.accept_calibration_record).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Mapping, Optional

from .checkpoint import read_json, write_json_atomic
from .recipe import RECIPE, RunConfig
from .selection import CALIBRATION_TIE_TOL as TIE_TOL
from .selection import calibration_choice, choose_peak
from .trainer import CentralTrainer


class CalibrationError(RuntimeError):
    pass


def run_calibration(family: str, schema, train, evalp, out_dir, *, key: Mapping, exposure_point: str,
                    cuda_visible_devices: str,
                    device: str = "cpu", effective_batch: int = RECIPE.effective_batch, test_overrides: bool = False,
                    model_kw: Optional[Mapping] = None, disk_usage_fn: Callable = shutil.disk_usage,
                    trainer_kw: Optional[Mapping] = None, **config_kw) -> dict:
    """`exposure_point` (PREG1 T2 P1) and `cuda_visible_devices` are required: the caller states them; never defaulted."""
    out = Path(out_dir)
    fam = str(family).upper()
    cfgs = [RunConfig(run_id=f"CAL_{fam}_x{mult:g}", family=fam, seed=RECIPE.calibration_seed, phase="CALIBRATION",
                      exposure_point=exposure_point, cuda_visible_devices=cuda_visible_devices,
                      peak_lr=mult * RECIPE.executed_peak_lr,
                      effective_batch=effective_batch, key=dict(key), test_overrides=test_overrides, **config_kw)
            for mult in RECIPE.calibration_multipliers]
    for cfg in cfgs:
        cfg.validate()
    label = cfgs[0].run_label
    if not test_overrides and label != "PRODUCTION":
        raise CalibrationError(f"a non-test calibration must be PRODUCTION; data_version {cfgs[0].data_version!r} "
                               f"gives {label} (refused before any arm runs)")
    arms, arm_rows, starts = [], {}, []
    for mult, cfg in zip(RECIPE.calibration_multipliers, cfgs):
        run_dir = out / f"peak_x{mult:g}"
        tr = CentralTrainer(cfg, schema, train, evalp, run_dir, device=device,
                            disk_usage_fn=disk_usage_fn, **dict(model_kw or {}), **dict(trainer_kw or {}))
        s = tr.run()
        end = [e for e in s["evals"] if abs(e["mark"] - 2.0) < 1e-9][0]
        arm_rows[mult] = read_json(run_dir / end["rows_file"])["rows"]
        starts.append(read_json(run_dir / "RUN_START.json"))
        arms.append({"multiplier": mult, "peak_lr": cfg.peak_lr, "run_id": cfg.run_id, "init_sha256": s["init_sha256"],
                     "config_sha256": s["config_sha256"], "curve": [{"mark": e["mark"], "efe": e["efe"],
                                                                     "value": e["value"]} for e in s["evals"]],
                     "value_at_2.0": s["value_at_end"], "latest_deleted_sha256": s["latest_deleted_sha256"],
                     "run_label": s["run_label"]})
    left = sorted(str(p.relative_to(out)) for p in out.rglob("*.pt"))
    if left:
        raise CalibrationError(f"calibration must keep no weights; found {left}")
    if len({a["init_sha256"] for a in arms}) != 1:
        raise CalibrationError("the two arms did not start from the same fresh theta0")
    values = {a["multiplier"]: a["value_at_2.0"] for a in arms}
    if label == "PRODUCTION":
        choice = calibration_choice(arm_rows, RECIPE.executed_peak_lr)      # PRODUCTION rows only (TRAIN-01R2)
        if choice["values_at_2.0"] != {float(m): float(v) for m, v in values.items()}:
            raise CalibrationError("the arms' 2.0 rows disagree with their run summaries")
        chosen = choice["chosen_multiplier"]
    else:
        chosen = choose_peak(values)                                          # test mechanics only; not evidence
    rec = {"task": "TRAIN-01 calibration", "family": fam, "seed": RECIPE.calibration_seed,
           "train_manifest": train.manifest_id, "eval_manifest": evalp.manifest.manifest_id,
           "metric": "client_macro_mrr@20 (E2E) at 2.0 EFE", "arms": arms, "chosen_multiplier": chosen,
           "chosen_peak_lr": chosen * RECIPE.executed_peak_lr,
           "rule": "higher value at 2.0 EFE; |delta| <= 1e-8 keeps the executed peak (1.0 x)",
           "exposure_point": exposure_point, "run_label": label, "evidence": label == "PRODUCTION",
           "data_version": cfgs[0].data_version, "manifest_registry": starts[0]["manifest_registry"],
           "schema": {k: starts[0]["schema"][k] for k in ("schema_id", "schema_self_sha256")},
           "determinism_mode": cfgs[0].determinism_mode,
           "weights_retained": [], "main_runs_restart_fresh": True, "test_overrides": test_overrides}
    rec["file_sha256"] = write_json_atomic(out / f"CALIBRATION_{fam}.json", rec)
    return rec


__all__ = ["CalibrationError", "choose_peak", "run_calibration", "TIE_TOL"]

"""PREG1 T2 (the LR exposure point) and T3 (the frozen evaluation config), both binding.

T2 — "end": the LR of a central update is peak * s((a + n) / N).
  * P1: `exposure_point` is a REQUIRED run-config key; a config without it is refused; a non-test run with any value
    other than the lock's "end" is refused ("start" survives only as a tested alternative under test_overrides);
  * P2: the planned per-update LR table is computed before training and hashed in RUN_START; min LR > 0; the final main
    update ends at 6N exposures with LR 0.01 x peak; every trained update's LR equals its table entry;
  * P3: v4train.schedule.s_main / s_continuation equal fl.schedule.s_of_efe bit for bit on the knots and 10,000 grid
    points (6.0 endpoint within 1e-12 in both) — when FL-02's fl.schedule is present;
  * the knots in v4train equal the lock's s_knots.
T3 — every evaluation number comes from ONE EvalConfig mirroring v4eval.frozen_config.load() (EVAL-03's real loader;
the object it returns is what v4eval requires): score_block 158,486, class_chunk 32,768, eval_batch 1,024, 1 GiB cache, FP64, the
deterministic flags; the rows carry eval_score_block, eval_batch and the RUNTIME eval_device_class; a non-test run on a
device other than the frozen class, or with a flag off, is refused; a non-test run must use exactly the frozen config.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trainkit import art  # noqa: E402  (TRAIN-03: sealed artifact root)
from trainkit import B, N, assert_raises, cfg, read, registry_keys, scripted, trainer, write_registry
from v4train import schedule as S
from v4train.evalconfig import EvalConfigError, frozen, load_frozen_eval_config
from v4train.recipe import RecipeError, RunConfig
from v4train.schedule import eval_marks, lr_for_update, planned_lr_table, s_continuation, s_main
from v4train.trainer import TrainerError

VALS = {m: 0.1 for m in eval_marks("MAIN")}
LOCK = json.loads((Path(__file__).resolve().parents[2] / "contracts" / "PREG1_LOCK_VALUES.json")
                  .read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ T2
def test_t2_lock_values_and_required_key():
    t2 = LOCK["T2_lr_exposure_point"]
    assert S.LR_EXPOSURE_POINT == t2["exposure_point"] == "end"
    assert [list(k) for k in S.KNOTS] == t2["s_knots"]
    dm = dict(determinism_mode="STRICT", **registry_keys())                          # TRAIN-01R1 / TRAIN-01R2 keys
    assert_raises(RecipeError, RunConfig(run_id="x", family="GRU", seed=2026, cuda_visible_devices="0", **dm).validate)  # P1: key missing
    assert_raises(RecipeError, RunConfig(run_id="x", family="GRU", seed=2026, exposure_point="start", cuda_visible_devices="0", **dm).validate)
    assert_raises(RecipeError, RunConfig(run_id="x", family="GRU", seed=2026, exposure_point="mid", cuda_visible_devices="0", **dm).validate)
    RunConfig(run_id="x", family="GRU", seed=2026, exposure_point="end", cuda_visible_devices="0", **dm).validate()
    assert_raises(RecipeError, RunConfig(run_id="x", family="GRU", seed=2026, exposure_point="end", **dm).validate)  # no GPU key
    assert_raises(RecipeError, trainer, Path("unused"), cfg("x", "GRU", exposure_point=None))
    Nn = 3_906_929
    assert lr_for_update(1e-3, 0, 256, Nn, "MAIN", point="end") > 0.0
    assert lr_for_update(1e-3, 0, 256, Nn, "MAIN", point="start") == 0.0
    assert lr_for_update(1e-3, 6 * 1000 - 8, 8, 1000, "MAIN", point="end") == pytest.approx(1e-5)
    assert_raises(S.ScheduleError, lr_for_update, 1e-3, 0, 8, 1000, "MAIN", point="END")


def test_t2_p2_planned_table():
    for Nn, Bb in ((200, 32), (1000, 256), (40_180_608 // 64, 256)):
        t = planned_lr_table(1e-3, Nn, Bb, "MAIN", "end")
        assert t["min_lr"] > 0.0 and t["final_exposures"] == 6 * Nn and t["last_lr"] == pytest.approx(1e-5, abs=1e-15)
        assert t["n_updates"] == 6 * -(-Nn // Bb) and len(t["sha256"]) == 64
    st = planned_lr_table(1e-3, 200, 32, "MAIN", "start")
    assert st["min_lr"] == 0.0 and st["last_lr"] > 1e-5
    assert planned_lr_table(1e-3, 200, 32, "MAIN", "end")["sha256"] != st["sha256"]


def _lrs(run_dir):
    recs = [json.loads(x) for x in (art(run_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    return [(r["update"], r["exposures"], r["lr_last"]) for r in recs if r["kind"] == "monitor"]


@pytest.mark.parametrize("point", ["end", "start"])
def test_t2_every_update_uses_the_planned_lr(work, point):
    c = cfg(f"lr_{point}", "GRU", log_every=1, exposure_point=point)
    s = trainer(work / "r", c, eval_override=scripted(VALS)).run()
    rows = _lrs(work / "r")
    plan = planned_lr_table(1e-3, N, B, "MAIN", point)
    assert len(rows) == s["updates"] == plan["n_updates"] == 6 * -(-N // B)
    prev = 0
    for (u, exp_after, lr), planned in zip(rows, plan["lr"]):
        at = exp_after if point == "end" else prev
        assert lr == planned == 1e-3 * s_main(at / N), f"update {u}"
        prev = exp_after
    st = read(work / "r" / "RUN_START.json")
    assert st["exposure_point"] == point and st["planned_lr_table"]["sha256"] == plan["sha256"]
    assert st["planned_lr_table"]["min_lr"] == plan["min_lr"]


def test_t2_start_point_refused_outside_tests(work):
    import dataclasses as dc
    c = dc.replace(cfg("st", "GRU", exposure_point="start"), test_overrides=False)
    assert_raises(RecipeError, c.validate)


def test_t2_p3_matches_fl_schedule_bit_for_bit():
    fls = pytest.importorskip("fl.schedule")
    if not hasattr(fls, "s_of_efe"):
        pytest.skip("fl.schedule has no s_of_efe yet (FL-02)")
    grid = [k[0] for k in S.KNOTS] + list(np.linspace(0.0, 6.0, 10_000))
    for e in grid:
        assert s_main(e) == fls.s_of_efe(e), f"s differs at {e!r}"
    for e in list(np.linspace(6.0, 8.0, 10_001))[1:]:
        assert s_continuation(e) == fls.s_of_efe(e, continuation=True), f"continuation differs at {e!r}"
    assert abs(s_main(6.0) - 0.01) <= 1e-12 and abs(fls.s_of_efe(6.0) - 0.01) <= 1e-12


def test_t2_p3_endpoint_tolerance_1e12():
    """PREG1 T2 P3: "6.0 endpoint tolerance 1e-12 in both" — restored after FL-02's reconciliation note."""
    assert S.END_TOL == 1e-12
    assert s_main(6.0 + 1e-12) == s_main(6.0) == s_main(6.0 + 5e-13)
    assert_raises(S.ScheduleError, s_main, 6.0 + 2e-12)
    assert abs(s_main(6.0) - 0.01) <= 1e-12 and s_main(6.0) == 0.009999999999999995   # never exact 0.01


def test_t2_p3_tolerance_zone_matches_fl_when_fl_has_it():
    fls = pytest.importorskip("fl.schedule")
    tol = getattr(fls, "EDGE_TOL", None)
    if tol != S.END_TOL:
        pytest.skip(f"fl.schedule EDGE_TOL is {tol!r}, the lock asks 1e-12 in both (FL-02 to align)")
    for e in (6.0 + 1e-13, 6.0 + 1e-12):
        assert float.hex(s_main(e)) == float.hex(fls.s_of_efe(e))


# ------------------------------------------------------------------------------------------------ T3
def test_t3_frozen_eval_config_is_the_lock():
    t3 = LOCK["T3_eval_numerics"]
    ec = load_frozen_eval_config()
    assert (ec.score_block, ec.class_chunk, ec.eval_batch_size, ec.cache_bytes) == \
        (t3["score_block"], t3["class_chunk"], t3["eval_batch"], t3["cache_bytes"]) == (158486, 32768, 1024, 1 << 30)
    assert ec.eval_device_class == t3["comparison_key_additions"]["eval_device_class"] == "A16-FP32-DET"
    assert ec.flags == t3["flags"] and ec.metric_accumulation == "FP64"
    assert RunConfig(run_id="x", family="GRU", seed=2026, exposure_point="end",
                     determinism_mode="STRICT", **registry_keys()).eval_config == ec                 # the default
    assert {f.name for f in dataclasses.fields(RunConfig)} >= {"eval_config"}
    assert "score_block" not in {f.name for f in dataclasses.fields(RunConfig)}, "no evaluation value outside it"
    other = dataclasses.replace(ec, eval_batch_size=512)
    assert_raises(RecipeError, RunConfig(run_id="x", family="GRU", seed=2026, exposure_point="end", cuda_visible_devices="0",
                                         determinism_mode="STRICT", eval_config=other, **registry_keys()).validate)   # non-test: frozen only
    for bad in (dataclasses.replace(ec, score_block=0), dataclasses.replace(ec, cache_bytes=-1),
                dataclasses.replace(ec, metric_accumulation="FP32")):
        assert_raises(EvalConfigError, bad.validate)


def test_t3_rows_carry_the_frozen_key_and_runtime_device(work):
    trainer(work / "d", cfg("ec", "GRU")).run()
    rows = read(art(work / "d") / "RESULT_ROWS.json")["rows"]
    ec = load_frozen_eval_config()
    assert {int(r["eval_score_block"]) for r in rows} == {ec.score_block}
    assert {int(r["eval_batch"]) for r in rows} == {ec.eval_batch_size}
    assert {r["eval_device_class"] for r in rows} == {"CPU-FP32-DET"}, "a CPU test row never claims the A16 class"
    assert {r["eval_config"]["frozen"] for r in rows} == {True}
    st = read(work / "d" / "RUN_START.json")
    assert st["eval_config"]["score_block"] == 158486 and st["eval_config_sha256"] == ec.sha256()
    assert st["eval_device_class"] == "CPU-FP32-DET"


def test_t3_runtime_flags_and_device_refusals():
    from v4eval.frozen_config import FrozenConfigError, apply_runtime_flags, check_runtime_flags
    from fl.numerics import enforce_strict_fp32
    fz = frozen()
    enforce_strict_fp32(deterministic=True)                    # WARN_ONLY training flags fail the T3 flag check
    assert_raises(FrozenConfigError, check_runtime_flags, fz, torch.device("cpu"))
    apply_runtime_flags(fz)
    check_runtime_flags(fz, torch.device("cpu"))               # the evaluation flags the trainer switches to
    enforce_strict_fp32(deterministic=True)
    tp, ep = _providers_1024()
    c = dataclasses.replace(cfg("dev", "GRU"), test_overrides=False, effective_batch=256,
                            manifest_registry=write_registry({"MAIN": (tp, ep)}))       # the 1024-row providers bound
    with pytest.raises(TrainerError, match="evaluation device class"):
        trainer(Path("unused_dev"), c, tp, ep)                                       # CPU != A16-FP32-DET


def _providers_1024():
    from trainkit import providers
    return providers(n=1024)

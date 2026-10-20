"""TRAIN-01R1 — the explicit, recorded determinism mode of the central trainer (PREG1_AMENDMENT_01 item 4).

Evidence: SYS-04 (company/handoffs/SYS-04.md) ran STRICT deterministic algorithms (warn_only off, with
CUBLAS_WORKSPACE_CONFIG=:4096:8) on the A16 for both families' training ops: no raise, bitwise run-to-run, and equal
to warn-only bit for bit. The proposed lock value is STRICT, pending Fable's confirmation at G1.

  * `determinism_mode` is a REQUIRED run-config key: None is refused, only "STRICT" and "WARN_ONLY" exist, and a value
    other than DETERMINISM_MODE_LOCK is refused unless test_overrides. The mode enters the config hash and the
    precision id (derived from the config, never a constant);
  * STRICT refuses to start unless the launcher exported CUBLAS_WORKSPACE_CONFIG ":4096:8" or ":16:8" (env
    monkeypatched); WARN_ONLY starts without it on the CPU;
  * the trainer applies the mode (STRICT -> use_deterministic_algorithms(True) with warn_only off; WARN_ONLY -> warn_only
    on) from either prior torch state, reads the state back, restores it after every real v4eval evaluation and fails
    closed when it drifts mid-run;
  * the mode and the captured CUBLAS setting are recorded in RUN_START (next to the effective numerics), in every
    checkpoint (the file's meta and CHECKPOINTS.json), in RUN_SUMMARY and in every result row, with the mode-derived
    precision_id;
  * a resume or a continuation whose mode or CUBLAS setting differs from the recorded one is refused; the refusal
    writes nothing and the original launch still resumes.
Negative controls: a trainer that ignores the mode; rows that do not record it; a resume that ignores a mode change.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import pytest
import torch

import v4train.trainer as T
from trainkit import art  # noqa: E402  (TRAIN-03: sealed artifact root)
from trainkit import KEY, assert_raises, cfg, interrupt_after, nc, ramp, read, registry_keys, scripted, trainer
from v4train.checkpoint import CheckpointError, CheckpointStore
from v4train.recipe import (CUBLAS_WORKSPACE_CONFIGS_STRICT, DETERMINISM_MODE_LOCK, DETERMINISM_MODES, PRECISION_ID,
                            RecipeError, RunConfig, precision_id)
from v4train.schedule import eval_marks
from v4train.trainer import BUDGET_SCHEMA, InjectedInterruption, TrainerError

MODES = ("STRICT", "WARN_ONLY")
VALS = {m: 0.1 for m in eval_marks("MAIN")}
ENV = "CUBLAS_WORKSPACE_CONFIG"
RESUME_REFUSAL = "determinism mode / CUBLAS_WORKSPACE_CONFIG"


def _raises_with(exc, text, fn, *a, **kw):
    """fn must raise `exc` and its message must name the intended cause (not merely any refusal)."""
    try:
        fn(*a, **kw)
    except exc as e:
        err = e
    else:
        raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__} ({text!r})")
    assert text in str(err), f"refused for another reason: {err}"
    return err


def _torch_mode() -> str:
    assert torch.are_deterministic_algorithms_enabled(), "deterministic algorithms are off"
    return "WARN_ONLY" if torch.is_deterministic_algorithms_warn_only_enabled() else "STRICT"


def _other_mode_state(mode: str) -> None:
    torch.use_deterministic_algorithms(True, warn_only=(mode == "STRICT"))


def _other_cublas() -> str:
    return next(v for v in CUBLAS_WORKSPACE_CONFIGS_STRICT if v != os.environ[ENV])


def _snapshot(run_dir) -> dict:
    return {str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run_dir.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------------------------------------ the key
def test_required_key_lock_value_and_precision_id(work):
    assert DETERMINISM_MODES == MODES and DETERMINISM_MODE_LOCK == "STRICT"
    assert CUBLAS_WORKSPACE_CONFIGS_STRICT == (":4096:8", ":16:8")
    assert RunConfig.__dataclass_fields__["determinism_mode"].default is None, "no code default"
    base = dict(run_id="d", family="GRU", seed=2026, key=dict(KEY), exposure_point="end", cuda_visible_devices="0",
                **registry_keys())                                                             # TRAIN-01R2 keys
    _raises_with(RecipeError, "determinism_mode is a required", RunConfig(**base).validate)                 # missing
    _raises_with(RecipeError, "determinism_mode is a required",
                 RunConfig(**base, determinism_mode=None, test_overrides=True).validate)                    # even in tests
    for bad in ("strict", "warn_only", "DETERMINISTIC", "OFF", "", "STRICT "):
        _raises_with(RecipeError, "determinism_mode must be one of",
                     RunConfig(**base, determinism_mode=bad, test_overrides=True).validate)
    _raises_with(RecipeError, "determinism_mode is locked to 'STRICT'",
                 RunConfig(**base, determinism_mode="WARN_ONLY").validate)                                   # no override
    RunConfig(**base, determinism_mode="STRICT").validate()
    RunConfig(**base, determinism_mode="WARN_ONLY", test_overrides=True).validate()
    _raises_with(RecipeError, "determinism_mode is a required", trainer, work / "none",
                 cfg("none", "GRU", determinism_mode=None))
    _raises_with(RecipeError, "determinism_mode is locked", trainer, work / "w",
                 dataclasses.replace(cfg("w", "GRU", determinism_mode="WARN_ONLY"), test_overrides=False))
    assert not (work / "none").exists() and not (work / "w").exists()
    s, w = cfg("t", "GRU"), cfg("t", "GRU", determinism_mode="WARN_ONLY")
    assert s.determinism_mode == "STRICT", "trainkit states the key explicitly (the RunConfig default is None)"
    assert s.sha256() != w.sha256() and s.as_json()["determinism_mode"] == "STRICT"
    assert PRECISION_ID == precision_id("STRICT") == s.precision_id
    assert w.precision_id == precision_id("WARN_ONLY") != PRECISION_ID
    assert s.precision_id.endswith(",deterministic_algorithms=STRICT")
    assert w.precision_id.endswith(",deterministic_algorithms=WARN_ONLY")
    assert_raises(RecipeError, precision_id, "OFF")


def test_strict_needs_the_launcher_cublas_setting(work, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    _raises_with(TrainerError, "determinism_mode STRICT needs CUBLAS_WORKSPACE_CONFIG", trainer, work / "unset",
                 cfg("unset", "GRU"))
    for bad in ("", ":4096:2", ":8:8", "4096:8", " :4096:8", ":4096:8 "):
        monkeypatch.setenv(ENV, bad)
        _raises_with(TrainerError, "determinism_mode STRICT needs CUBLAS_WORKSPACE_CONFIG", trainer, work / "bad",
                     cfg("bad", "GRU"))
    assert not (work / "unset").exists() and not (work / "bad").exists(), "refused before anything is created"
    for i, ok in enumerate(CUBLAS_WORKSPACE_CONFIGS_STRICT):
        monkeypatch.setenv(ENV, ok)
        tr = trainer(work / f"ok{i}", cfg(f"ok{i}", "GRU"))
        assert tr.cublas_workspace_config == ok and tr.key_base["cublas_workspace_config"] == ok
        assert _torch_mode() == "STRICT"
    monkeypatch.delenv(ENV)
    tr = trainer(work / "warn", cfg("warn", "GRU", determinism_mode="WARN_ONLY"))      # CPU: WARN_ONLY needs no env
    assert tr.cublas_workspace_config == "UNSET" and tr.key_base["cublas_workspace_config"] == "UNSET"
    assert _torch_mode() == "WARN_ONLY"


# ------------------------------------------------------------------------------------------------ applied
def _check_applied(work, mode: str):
    """Build a trainer in `mode` from the OTHER torch state: the trainer must switch the process to its mode."""
    _other_mode_state(mode)
    tr = trainer(work / f"applied_{mode}", cfg(f"applied_{mode}", "GRU", determinism_mode=mode))
    assert _torch_mode() == mode, f"the trainer did not apply determinism_mode {mode}"
    return tr


@pytest.mark.parametrize("mode", MODES)
def test_mode_applied_from_either_torch_state(work, mode):
    _check_applied(work, mode)


def test_effective_state_is_verified_and_drift_fails_closed(work, monkeypatch):
    orig = T.enforce_strict_fp32
    # an enforce that does not honour warn_only (the SYS-04 probe-wrapper pattern) is caught by the read-back
    monkeypatch.setattr(T, "enforce_strict_fp32",
                        lambda deterministic=True, warn_only=True: orig(deterministic, warn_only=not warn_only))
    _raises_with(TrainerError, "is not the configured determinism_mode", trainer, work / "v", cfg("v", "GRU"))
    monkeypatch.setattr(T, "enforce_strict_fp32", orig)

    def flip(u: int) -> None:                             # something re-arms warn_only in the middle of a STRICT run
        if u == 2:
            torch.use_deterministic_algorithms(True, warn_only=True)
    tr = trainer(work / "d", cfg("d", "GRU", log_every=3), fault=flip, eval_override=scripted(VALS))
    _raises_with(TrainerError, "is not the configured determinism_mode", tr.run)       # caught at the next sync
    assert read(work / "d" / "STATUS.json")["status"] == "FAILED"


# ------------------------------------------------------------------------------------------------ recorded
def _check_rows(run_dir, mode: str, cublas: str) -> None:
    files = [art(run_dir) / "RESULT_ROWS.json"] + sorted((art(run_dir) / "evals").glob("*.rows.json"))
    n = 0
    for f in files:
        for r in read(f)["rows"]:
            n += 1
            assert r.get("determinism_mode") == mode, f"{f.name}: a row does not record determinism_mode {mode}"
            assert r.get("cublas_workspace_config") == cublas, f"{f.name}: a row does not record the CUBLAS setting"
            assert r.get("precision_id") == precision_id(mode), f"{f.name}: precision_id is not the mode's"
    assert n > 0, "no rows"


def _check_recorded(run_dir, mode: str, cublas: str) -> None:
    eff = {"deterministic_algorithms": True, "warn_only": mode == "WARN_ONLY"}
    st = read(run_dir / "RUN_START.json")
    assert st["config"]["determinism_mode"] == mode
    assert st["determinism"] == {"mode": mode, "cublas_workspace_config": cublas, "torch_effective": eff}
    assert st["numerics"]["deterministic_algorithms"] is True
    assert st["numerics"]["deterministic_warn_only"] is (mode == "WARN_ONLY"), "the precision record states the mode"
    assert st["precision_id"] == st["key_base"]["precision_id"] == precision_id(mode)
    assert st["key_base"]["determinism_mode"] == mode and st["key_base"]["cublas_workspace_config"] == cublas
    idx = read(art(run_dir) / "CHECKPOINTS.json")
    assert sorted(idx) == ["best", "endpoint_6.0"]
    for kind, rec in idx.items():
        assert rec["meta"]["determinism"] == st["determinism"], f"{kind}: checkpoint index"
        obj = torch.load(art(run_dir) / rec["file"], map_location="cpu", weights_only=True)
        assert obj["meta"]["determinism"] == st["determinism"], f"{kind}: checkpoint file"
    assert read(art(run_dir) / "RUN_SUMMARY.json")["determinism"] == st["determinism"]
    _check_rows(run_dir, mode, cublas)


@pytest.mark.parametrize("mode", MODES)
def test_mode_applied_and_recorded_real_eval(work, mode):
    c = cfg(f"rec_{mode}", "GRU", determinism_mode=mode, checkpoint_every=4)
    _other_mode_state(mode)
    tr = trainer(work / "r", c, fault=interrupt_after(9))
    assert _torch_mode() == mode
    with pytest.raises(InjectedInterruption):
        tr.run()                                                # real v4eval at 0.5 and 1.0 on the way
    assert _torch_mode() == mode, "the mode is restored after every evaluation (T3 flags switch back)"
    live = read(art(work / "r") / "CHECKPOINTS.json")["latest"]
    assert live["meta"]["determinism"]["mode"] == mode and live["meta"]["updates"] == 8
    obj = torch.load(art(work / "r") / live["file"], map_location="cpu", weights_only=True)
    assert obj["meta"]["determinism"] == live["meta"]["determinism"]
    _other_mode_state(mode)                                     # a foreign torch state before the resume
    s = trainer(work / "r", c).resume()
    assert s["status"] == "DONE_MAIN" and _torch_mode() == mode
    _check_recorded(work / "r", mode, os.environ[ENV])


# ------------------------------------------------------------------------------------------------ resume
def _check_resume_refuses_mode_changes(work, monkeypatch) -> None:
    c = cfg("rs", "GRU", checkpoint_every=4)
    with pytest.raises(InjectedInterruption):
        trainer(work / "B", c, fault=interrupt_after(9), eval_override=scripted(VALS)).run()
    snap = _snapshot(work / "B")
    _raises_with(TrainerError, RESUME_REFUSAL, trainer(work / "B", dataclasses.replace(c, determinism_mode="WARN_ONLY"),
                                                       eval_override=scripted(VALS)).resume)      # another mode
    cur = os.environ[ENV]
    monkeypatch.setenv(ENV, _other_cublas())
    _raises_with(TrainerError, RESUME_REFUSAL, trainer(work / "B", c, eval_override=scripted(VALS)).resume)  # CUBLAS
    monkeypatch.setenv(ENV, cur)
    assert _snapshot(work / "B") == snap, "a refused resume writes nothing"
    s = trainer(work / "B", c, eval_override=scripted(VALS)).resume()        # the original launch still resumes
    assert s["status"] == "DONE_MAIN"


def test_resume_refuses_a_changed_mode_or_cublas(work, monkeypatch):
    _check_resume_refuses_mode_changes(work, monkeypatch)


def test_resume_checks_the_checkpoint_record(work):
    c = cfg("ck", "GRU", checkpoint_every=4)
    with pytest.raises(InjectedInterruption):
        trainer(work / "B", c, fault=interrupt_after(9), eval_override=scripted(VALS)).run()
    p = art(work / "B") / "CHECKPOINTS.json"
    orig = p.read_bytes()
    idx = json.loads(orig)
    idx["latest"]["meta"]["determinism"]["mode"] = "WARN_ONLY"              # a record that disagrees with the start
    p.write_text(json.dumps(idx), encoding="utf-8")
    _raises_with(TrainerError, "latest checkpoint index", trainer(work / "B", c, eval_override=scripted(VALS)).resume)
    p.write_bytes(orig)
    assert trainer(work / "B", c, eval_override=scripted(VALS)).resume()["status"] == "DONE_MAIN"
    st = CheckpointStore(work / "S", "cfg-sha")                              # the store refuses a record-less file
    _raises_with(CheckpointError, "determinism mode record", st.save, "best",
                 {"meta": {"config_sha256": "cfg-sha"}, "model": {}})
    ids = {"manifest_registry": {"file_sha256": "f" * 64, "self_sha256": "s" * 64, "data_version": "DV",
                                 "role": "MAIN"},
           "schema": {"schema_id": "FEATURE_SCHEMA_CLEAN_V2", "schema_self_sha256": "5" * 64}}      # TRAIN-01R2 records
    st.save("best", {"meta": {"config_sha256": "cfg-sha", "determinism": {"mode": "STRICT",
                                                                         "cublas_workspace_config": ":4096:8"},
                              **ids},
                     "model": {}})


def test_continuation_refuses_a_changed_mode_or_cublas(work, monkeypatch):
    run_id = "cm"
    budget = work / "budget.json"
    budget.write_text(json.dumps({"schema": BUDGET_SCHEMA, "run_id": run_id, "max_efe": 8.0,
                                  "approved_by": "synthetic-test", "recorded_utc": "2026-09-26T00:00:00Z"}),
                      encoding="utf-8")
    vals = ramp(eval_marks("MAIN"), 6.0)
    for i, m in enumerate(eval_marks("CONTINUATION")):
        vals[round(m, 6)] = vals[6.0] + 0.001 * (i + 1)
    c = cfg(run_id, "GRU", continuation_budget=str(budget))
    assert trainer(work / "r", c, eval_override=scripted(vals)).run()["still_improving"]
    _raises_with(TrainerError, RESUME_REFUSAL, trainer(work / "r", dataclasses.replace(c, determinism_mode="WARN_ONLY"),
                                                       eval_override=scripted(vals)).continue_run)
    cur = os.environ[ENV]
    monkeypatch.setenv(ENV, _other_cublas())
    _raises_with(TrainerError, RESUME_REFUSAL, trainer(work / "r", c, eval_override=scripted(vals)).continue_run)
    monkeypatch.setenv(ENV, cur)
    assert trainer(work / "r", c, eval_override=scripted(vals)).continue_run()["status"] == "DONE_CONTINUATION"


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a trainer that ignores determinism_mode (the pre-R1 hard-coded warn-only call)")
def test_nc_trainer_ignores_the_mode(work, monkeypatch):
    monkeypatch.setattr(T.CentralTrainer, "_apply_determinism", lambda self: T.enforce_strict_fp32(deterministic=True))
    _check_applied(work, "STRICT")


@nc("result rows that do not record the determinism mode")
def test_nc_rows_without_the_mode(work, monkeypatch):
    orig = T.CentralTrainer._row_labels

    def labels(self):
        lab = orig(self)
        lab.pop("determinism_mode")                          # MUTANT: the key is not recorded in the rows
        return lab
    monkeypatch.setattr(T.CentralTrainer, "_row_labels", labels)
    trainer(work / "r", cfg("nc_rows", "GRU"), eval_override=scripted(VALS)).run()
    _check_rows(work / "r", "STRICT", os.environ[ENV])


@nc("a resume that ignores a determinism-mode change")
def test_nc_resume_ignores_a_mode_change(work, monkeypatch):
    monkeypatch.setattr(T.CentralTrainer, "_check_resume_determinism", lambda self, recorded, where: None)
    _check_resume_refuses_mode_changes(work, monkeypatch)

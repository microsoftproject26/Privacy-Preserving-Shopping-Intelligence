"""Tests of the run orchestrator (run_pipeline.py / <stage>/run.py, the ppsi_pipeline package).

Everything happens in temporary directories: ``PPSI_RESULTS_ROOT`` is pointed at a ``tmp_path``
copy (the package re-reads the variable per call, unlike ``ppsi_paths`` which reads it at import
time), so no test touches the repository's own folders.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from ppsi_pipeline import _select_stages, build_plan, execute, main, render_command
from ppsi_pipeline.decide import COMPLETE_STATUSES, Action, decide, move_aside, read_run_state
from ppsi_pipeline.units import REGISTRY, STAGE_ORDER, Unit

FL_UNIT = Unit(id="03_federated_learning/x", run="main_study/r/X",
               server_run="r/X", command="python launch.py --run-dir {run_dir}",
               resume="--resume", resumable=("ckpt/*/latest.pt",))
CENTRAL_UNIT = Unit(id="02_centralized/x", run="main_study/r/X", server_run="r/X",
                    command="python launch.py --run-dir {run_dir}", resume="--resume",
                    resumable=("checkpoints/latest.pt", "sealed/checkpoints/latest.pt"))
DEVICE_UNIT = Unit(id="04_on_device/x", run="main_study/r/X", server_run="r/X",
                   command="python launch.py --run-dir {run_dir}", resume="--resume",
                   resumable=("shards/shard_*.json",))
NO_RESUME_UNIT = Unit(id="01_data/x", run="main_study/r/X", server_run="r/X",
                      command="python launch.py --out {run_dir}", resume="none")


@pytest.fixture()
def results_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary results tree; the orchestrator decides on <root>/<stage>/results/runs/..."""
    root = tmp_path / "copy"
    monkeypatch.setenv("PPSI_RESULTS_ROOT", str(root))
    return root


def _run_dir(root: Path, unit: Unit) -> Path:
    return root / unit.stage / "results" / "runs" / unit.run


def _write_summary(run_dir: Path, payload: object, name: str = "RUN_SUMMARY.json") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    target = run_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------- the decision itself
def test_start_when_the_run_directory_is_absent(tmp_path: Path) -> None:
    state = read_run_state(tmp_path / "missing", FL_UNIT.summary_names, FL_UNIT.resumable)
    action, why = decide(state, FL_UNIT.resume)
    assert action is Action.START and "no run directory" in why


@pytest.mark.parametrize("status", sorted(COMPLETE_STATUSES))
def test_skip_on_every_complete_status(tmp_path: Path, status: str) -> None:
    run_dir = tmp_path / "run"
    _write_summary(run_dir, {"status": status, "arm": "FA"})
    state = read_run_state(run_dir, FL_UNIT.summary_names, FL_UNIT.resumable)
    assert state.complete and state.status == status
    assert decide(state, FL_UNIT.resume)[0] is Action.SKIP


@pytest.mark.parametrize("status", ["RUNNING", "INTERRUPTED", "PARTIAL", "PENDING_CKPT_AVG"])
def test_incomplete_statuses_resume_from_the_checkpoint(tmp_path: Path, status: str) -> None:
    run_dir = tmp_path / "run"
    _write_summary(run_dir, {"status": status})
    (run_dir / "ckpt" / "FL_X").mkdir(parents=True)
    (run_dir / "ckpt" / "FL_X" / "latest.pt").write_bytes(b"pt")
    state = read_run_state(run_dir, FL_UNIT.summary_names, FL_UNIT.resumable)
    action, why = decide(state, FL_UNIT.resume)
    assert action is Action.RESUME and "latest.pt" in why and "--resume" in why


def test_corrupt_summary_is_incomplete_and_resumes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(run_dir, "{not json")
    (run_dir / "ckpt" / "FL_X" / "latest.pt").parent.mkdir(parents=True)
    (run_dir / "ckpt" / "FL_X" / "latest.pt").write_bytes(b"pt")
    state = read_run_state(run_dir, FL_UNIT.summary_names, FL_UNIT.resumable)
    assert not state.complete and state.summary is None and "does not parse" in (state.problem or "")
    assert decide(state, FL_UNIT.resume)[0] is Action.RESUME


def test_summary_that_is_not_an_object_is_incomplete(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(run_dir, json.dumps(["DONE_MAIN"]))  # a JSON list parses but has no status
    state = read_run_state(run_dir, FL_UNIT.summary_names, FL_UNIT.resumable)
    assert not state.complete and "not a JSON object" in (state.problem or "")


def test_summary_without_a_status_field_is_incomplete(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(run_dir, {"arm": "FA"})
    state = read_run_state(run_dir, FL_UNIT.summary_names, FL_UNIT.resumable)
    assert not state.complete and "no status field" in (state.problem or "")


def test_checkpoint_without_a_summary_cannot_resume_without_force(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "ckpt" / "FL_X" / "latest.pt").parent.mkdir(parents=True)
    (run_dir / "ckpt" / "FL_X" / "latest.pt").write_bytes(b"pt")
    state = read_run_state(run_dir, FL_UNIT.summary_names, FL_UNIT.resumable)
    action, why = decide(state, NO_RESUME_UNIT.resume)
    assert action is Action.START and "cannot resume" in why


def test_sealed_summary_location_counts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"  # sealed central MAIN runs keep RUN_SUMMARY under sealed/
    _write_summary(run_dir, {"status": "DONE_MAIN"}, name="sealed/RUN_SUMMARY.json")
    state = read_run_state(run_dir, CENTRAL_UNIT.summary_names, CENTRAL_UNIT.resumable)
    assert state.complete and state.summary_path == run_dir / "sealed" / "RUN_SUMMARY.json"


def test_device_shard_markers_are_the_resumable_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(run_dir, {"status": "PARTIAL"})  # PARTIAL lives in STATUS.json only
    (run_dir / "shards").mkdir()
    (run_dir / "shards" / "shard_0007.json").write_text("{}", encoding="utf-8")
    state = read_run_state(run_dir, DEVICE_UNIT.summary_names, DEVICE_UNIT.resumable)
    assert state.resumable is not None and state.resumable.name == "shard_0007.json"
    assert decide(state, DEVICE_UNIT.resume)[0] is Action.RESUME


# ------------------------------------------------------------------------- the --force move-aside
def test_move_aside_keeps_everything(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(run_dir, {"status": "DONE_MAIN"})
    backup = move_aside(run_dir, "20260101-000000")
    assert backup is not None and backup.name == "run.bak-20260101-000000"
    assert not run_dir.exists() and (backup / "RUN_SUMMARY.json").is_file()


def test_move_aside_suffixed_on_collision_and_none_when_absent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "run.bak-20260101-000000").mkdir()
    backup = move_aside(run_dir, "20260101-000000")
    assert backup is not None and backup.name == "run.bak-20260101-000000-2"
    assert move_aside(tmp_path / "missing", "20260101-000000") is None


@pytest.fixture()
def one_unit(monkeypatch: pytest.MonkeyPatch) -> Unit:
    """Pin the stage-03 registry to exactly FL_UNIT, so plan/execute tests are hermetic."""
    monkeypatch.setattr("ppsi_pipeline.REGISTRY", {"03_federated_learning": (FL_UNIT,)})
    return FL_UNIT


def test_force_moves_aside_and_plans_a_fresh_start(results_root: Path, one_unit: Unit) -> None:
    run_dir = _run_dir(results_root, FL_UNIT)
    _write_summary(run_dir, {"status": "DONE_MAIN"})
    plan = build_plan(["03_federated_learning"], force=True)
    mine = [item for item in plan if item.unit.id == FL_UNIT.id]
    assert len(mine) == 1 and mine[0].action is Action.START
    assert mine[0].backup is not None and mine[0].backup.is_dir() and not run_dir.exists()
    assert "forced fresh start" in mine[0].reason


# ------------------------------------------------------------------------------ the registry
def test_registry_covers_every_stage_with_unique_units() -> None:
    assert tuple(REGISTRY) == STAGE_ORDER
    ids = [unit.id for units in REGISTRY.values() for unit in units]
    assert len(ids) == len(set(ids))
    for units in REGISTRY.values():
        for unit in units:
            assert unit.resume == "none" or unit.resumable, unit.id
            assert ".." not in unit.run and not Path(unit.run).is_absolute()


def test_render_command_fills_the_run_dir_and_appends_resume(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PPSI_RUNS_ROOT", str(tmp_path / "server"))
    command = render_command(FL_UNIT, resume=True)
    assert "server/r/X" in command  # run directories render posix-style on every platform
    assert command.endswith(" --resume")
    assert "--resume" not in render_command(FL_UNIT, resume=False)


# ------------------------------------------------------------------------------ the CLI
def test_dry_run_prints_the_plan_and_touches_nothing(results_root: Path,
                                                     capsys: pytest.CaptureFixture) -> None:
    assert main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "plan for 6 stage(s)" in out and "START" in out
    assert not (results_root / "03_federated_learning").exists()


def test_from_results_with_no_results_exits_zero(results_root: Path,
                                                 capsys: pytest.CaptureFixture) -> None:
    assert main(["--from-results"]) == 0
    out = capsys.readouterr().out
    assert out.count("no results found") == 6 and "nothing to do" in out


def test_from_results_rebuilds_tables_and_figures(results_root: Path) -> None:
    base = results_root / "03_federated_learning" / "results"
    _write_summary(base / "runs" / "main_study" / "runs_ds3" / "FL_T" / "FA_s2026",
                   {"status": "DONE_MAIN", "arm": "FA"})
    (base / "exploratory_warm_tau_grid_calibration.csv").write_text(
        "eta,value\n3e-06,0.171311\n1e-05,0.171237\n", encoding="utf-8")
    assert main(["--from-results", "--stages", "03"]) == 0
    summary = base / "summary"
    figures = results_root / "03_federated_learning" / "figures"
    assert (summary / "runs_inventory.csv").is_file() and (summary / "runs_inventory.md").is_file()
    assert (summary / "warm_tau_grid_calibration.csv").is_file()
    assert (figures / "warm_tau_grid_calibration.png").is_file()
    assert (figures / "warm_tau_grid_calibration.pdf").is_file()
    rows = (summary / "runs_inventory.csv").read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "phase,run,arm,status" and "DONE_MAIN" in rows[1]


def test_from_results_skips_a_missing_source_with_a_warning(results_root: Path,
                                                             capsys: pytest.CaptureFixture) -> None:
    (results_root / "04_on_device" / "results").mkdir(parents=True)
    assert main(["--from-results", "--stages", "04"]) == 0
    out = capsys.readouterr().out
    assert "warning" in out and "device_latency.csv" in out
    assert not (results_root / "04_on_device" / "figures").exists()


def test_stage_arguments_accept_numbers_and_names() -> None:
    assert _select_stages(["3", "04"]) == ["03_federated_learning", "04_on_device"]
    assert _select_stages(["06_evaluation_and_benchmarks"]) == ["06_evaluation_and_benchmarks"]
    with pytest.raises(SystemExit, match="unknown stage"):
        _select_stages(["99"])


def test_stage_run_py_pins_its_stage(results_root: Path) -> None:
    with pytest.raises(SystemExit, match="manages 01_data only"):
        main(["--stages", "03"], only_stage="01_data")
    assert main(["--dry-run"], only_stage="01_data") == 0


def test_execute_reports_pinned_trees_instead_of_launching(results_root: Path, one_unit: Unit,
                                                            capsys: pytest.CaptureFixture,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    launched = []
    monkeypatch.setattr("ppsi_pipeline.subprocess.run",
                        lambda *a, **k: launched.append(a) or type("R", (), {"returncode": 0})())
    monkeypatch.setenv("PPSI_PROJECT_ROOT", str(results_root / "nowhere"))
    run_dir = _run_dir(results_root, FL_UNIT)
    _write_summary(run_dir, {"status": "RUNNING"})
    (run_dir / "ckpt" / "FL_X").mkdir(parents=True)
    (run_dir / "ckpt" / "FL_X" / "latest.pt").write_bytes(b"pt")
    plan = build_plan(["03_federated_learning"])
    assert [item.action for item in plan] == [Action.RESUME]
    assert execute(plan) == 0
    out = capsys.readouterr().out
    assert not launched
    assert "not launched" in out and "07_study_protocol" in out

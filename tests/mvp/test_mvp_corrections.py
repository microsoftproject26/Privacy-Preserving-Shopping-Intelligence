"""Regressions for the pre-run correction pass.

Each test here exists because of a specific defect or a specific runbook requirement:
the Python 3.11 canary construction, the frozen-source guard, the refusal to invent a
TRAIN history count, and completed-round federated recovery.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ppsi.federated.mvp_runner import build_trainer, model_identity
from ppsi.training.state import pack_shared_state

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli():
    return _load("mvp_cli", "scripts/mvp/run_mvp.py")


@pytest.fixture(scope="module")
def prepare():
    return _load("mvp_prepare", "scripts/mvp/prepare_mvp.py")


# --- the Python 3.11 canary construction path ---------------------------------------


def test_canary_builds_its_dropout_free_fixture_on_python_311(cli, policy):
    """copy.replace does not exist on 3.11; this executes the real construction."""
    identity = model_identity(policy)
    fixture = cli.dropout_free_identity(identity)
    assert fixture.config.dropout == 0.0
    assert identity.config.dropout == float(policy["model"]["dropout"]) > 0.0
    # Everything except dropout must survive, or the fixture would compare two models.
    assert fixture.config.channels == identity.config.channels
    assert fixture.config.hidden == identity.config.hidden
    assert fixture.config.core == identity.config.core
    assert fixture.spec is identity.spec
    assert fixture.category_count == identity.category_count


def test_the_dropout_free_fixture_actually_trains_and_agrees(cli, prepared_dir, policy):
    """The canary's central-versus-Flower agreement check, on the synthetic bundle."""
    from ppsi.federated.mvp_data import load_prepared
    from ppsi.federated.mvp_runner import client_workload, workload_batches
    from ppsi.models.session_gru import common_initialization
    from ppsi.training.flower import FlowerLocalAdapter
    from ppsi.training.t1_mvp_objective import T1ContributingWeightPolicy

    prepared = load_prepared(prepared_dir)
    identity = model_identity(policy)
    fixture = cli.dropout_free_identity(identity)
    state, _ = common_initialization(13, config=fixture.config, batch_spec=fixture.spec)
    workload = client_workload(
        prepared, prepared.population[0], server_round=1, seed=13, batch_size=2
    )
    batch = workload_batches(prepared, workload, fixture.spec)[0]

    central_model, central_core = build_trainer(fixture, policy, state)
    central_core.train_step(batch)
    central_after = pack_shared_state(central_model, central_model.shared_state_spec())

    flower_model, flower_core = build_trainer(fixture, policy, state)
    adapter = FlowerLocalAdapter(
        core=flower_core,
        shared_state_spec=flower_model.shared_state_spec(),
        aggregation_weight_policy=T1ContributingWeightPolicy(),
    )
    result = adapter.fit(
        pack_shared_state(flower_model, flower_model.shared_state_spec()), [batch], outer_round=1
    )
    worst = max(
        float(torch.abs(central_after[key] - result.shared_state[key]).max().item())
        for key in central_after
    )
    assert worst <= 1e-6


# --- the frozen source guard --------------------------------------------------------


class _FreezeContext:
    """The smallest object the freeze helpers need, pointed at a temporary tree."""

    def __init__(self, public: Path, config_rel: str = "policy.json") -> None:
        self.run = "mvp-t1-001"
        self.public = public
        self.config_rel = config_rel

    def public_path(self, name: str) -> str:
        return (self.public / name).as_posix()


def test_source_freeze_detects_drift_and_refuses_to_continue(cli, tmp_path, monkeypatch):
    public = tmp_path / "evidence"
    public.mkdir()
    source = tmp_path / "behaviour.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "policy.json"
    config.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "SOURCE_FILES", ("behaviour.py",))
    monkeypatch.setattr(
        cli,
        "source_snapshot",
        lambda config_rel: {
            "schema": "mvp_source_snapshot_v1",
            "version": "1",
            "git_head": "a" * 40,
            "files_sha256": {
                "behaviour.py": cli.file_sha256(source),
                config_rel: cli.file_sha256(config),
            },
        },
    )
    ctx = _FreezeContext(Path("evidence"))

    frozen = cli.freeze_source(ctx)
    assert frozen["freeze_status"] == "FROZEN_BEFORE_SCIENTIFIC_EXECUTION"
    assert cli.verify_frozen_source(ctx, stage="canary")["drift"] == []
    # The active execution policy is frozen alongside the code, and checked with it.
    assert "policy.json" in cli.read_json(ctx.public_path(cli.SOURCE_FREEZE_NAME))["files_sha256"]

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(cli.StageError, match="SOURCE_DRIFT before centralized"):
        cli.verify_frozen_source(ctx, stage="centralized")
    with pytest.raises(cli.StageError, match="new run id"):
        cli.freeze_source(ctx)


def test_a_stage_cannot_run_before_the_freeze_exists(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    ctx = _FreezeContext(Path("evidence"))
    with pytest.raises(cli.StageError, match="cannot run before the source freeze exists"):
        cli.verify_frozen_source(ctx, stage="federated")


def test_the_named_stages_all_verify_the_freeze(cli):
    """The runbook names four stages; none of them may skip the check."""
    text = (ROOT / "scripts/mvp/run_mvp.py").read_text(encoding="utf-8")
    for stage in ("canary", "centralized", "federated", "compare"):
        assert f'verify_frozen_source(ctx, stage="{stage}")' in text


# --- TRAIN history must be measured, never defaulted --------------------------------


def test_measured_train_history_is_returned_including_a_genuine_zero(prepare, tmp_path):
    from ppsi.federated.clients import client_id_from_user

    counts = pd.Series({11: 7, 12: 0}, name="events")
    clients = [client_id_from_user("11"), client_id_from_user("12")]
    measured = prepare.require_measured_history(counts, clients, evidence_dir=tmp_path)
    assert measured[clients[0]] == 7
    # A measured zero is a real observation and is kept.
    assert measured[clients[1]] == 0
    assert not (tmp_path / "FAILED_missing_train_history.json").exists()


def test_a_validation_client_without_measured_history_fails_preparation(prepare, tmp_path):
    from ppsi.federated.clients import client_id_from_user

    counts = pd.Series({11: 7}, name="events")
    clients = [client_id_from_user("11"), client_id_from_user("99")]
    with pytest.raises(prepare.MissingTrainHistoryError, match="never recorded as a zero"):
        prepare.require_measured_history(counts, clients, evidence_dir=tmp_path)
    evidence = json.loads(
        (tmp_path / "FAILED_missing_train_history.json").read_text(encoding="utf-8")
    )
    assert evidence["clients_without_measured_train_history"] == 1
    assert evidence["client_ids"] == [client_id_from_user("99")]


def test_preparation_never_substitutes_a_zero_for_a_missing_measurement():
    text = (ROOT / "scripts/mvp/prepare_mvp.py").read_text(encoding="utf-8")
    assert "per_client.get(" not in text
    assert "FAIL_CLOSED_NEVER_DEFAULT_TO_ZERO" in text


# --- completed-round federated recovery ---------------------------------------------


def _identity(**overrides) -> dict:
    base = {
        "run": "mvp-t1-001",
        "policy_sha256": "a" * 64,
        "source_freeze_sha256": "b" * 64,
        "data_manifest_sha256": "c" * 64,
        "common_initialization_sha256": "d" * 64,
        "schedule_digest": "e" * 64,
        "seed": 13,
        "rounds": 20,
        "clients_per_round": 50,
        "batch_size": 64,
    }
    base.update(overrides)
    return base


def _trace(server_round: int) -> dict:
    return {
        "selection_digest": f"{server_round:064d}",
        "selected_client_count": 50,
        "aggregated_digest": f"{server_round:064x}",
    }


def _persist(cli, directory: Path, server_round: int, identity: dict) -> dict:
    return cli.persist_completed_round(
        directory,
        run="mvp-t1-001",
        server_round=server_round,
        state={"weight": torch.full((2, 2), float(server_round))},
        identity=identity,
        trace=_trace(server_round),
        round_record={"server_round": server_round, "rows": 100 * server_round},
        ledger_records=[
            {
                "server_round": index,
                "client_id": "client-v1-" + "a" * 64,
                "direction": direction,
                "payload_bytes": 1000 + index,
            }
            for index in range(1, server_round + 1)
            for direction in ("download", "upload")
        ],
        evaluations=[{"server_round": 0, "headline": {"value": 0.1}}],
    )


def test_a_completed_round_round_trips_and_the_pointer_advances(cli, tmp_path):
    identity = _identity()
    assert cli.load_last_completed_round(tmp_path, identity) is None

    _persist(cli, tmp_path, 1, identity)
    first = cli.load_last_completed_round(tmp_path, identity)
    assert first["completed_server_round"] == 1
    assert first["communication_totals_through_round"]["upload_transmissions"] == 1

    _persist(cli, tmp_path, 2, identity)
    second = cli.load_last_completed_round(tmp_path, identity)
    assert second["completed_server_round"] == 2
    assert second["communication_totals_through_round"]["upload_bytes"] == 1001 + 1002
    # Every earlier round stays on disk; a resume never erases prior evidence.
    assert (tmp_path / cli.COMPLETED_ROUNDS_DIR / "round_01.checkpoint.json").is_file()
    state = torch.load(
        tmp_path / second["aggregate"]["state_uri"], map_location="cpu", weights_only=True
    )
    assert float(state["weight"][0, 0]) == 2.0


def test_an_incompatible_identity_refuses_to_resume(cli, tmp_path):
    _persist(cli, tmp_path, 1, _identity())
    with pytest.raises(cli.StageError, match="INCOMPATIBLE_RESUME"):
        cli.load_last_completed_round(tmp_path, _identity(data_manifest_sha256="f" * 64))
    with pytest.raises(cli.StageError, match="INCOMPATIBLE_RESUME"):
        cli.load_last_completed_round(tmp_path, _identity(seed=42))


def test_an_altered_aggregate_state_is_refused(cli, tmp_path):
    identity = _identity()
    payload = _persist(cli, tmp_path, 1, identity)
    torch.save({"weight": torch.zeros(2, 2)}, tmp_path / payload["aggregate"]["state_uri"])
    with pytest.raises(cli.StageError, match="missing or altered"):
        cli.load_last_completed_round(tmp_path, identity)


def test_an_altered_checkpoint_is_refused(cli, tmp_path):
    identity = _identity()
    _persist(cli, tmp_path, 1, identity)
    checkpoint = tmp_path / cli.COMPLETED_ROUNDS_DIR / "round_01.checkpoint.json"
    body = json.loads(checkpoint.read_text(encoding="utf-8"))
    body["round_record"]["rows"] = 999999
    checkpoint.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(cli.StageError, match="does not match the hash"):
        cli.load_last_completed_round(tmp_path, identity)


def test_a_half_written_round_never_becomes_the_resume_point(cli, tmp_path):
    """The pointer is written last, so an interrupted round keeps the previous one."""
    identity = _identity()
    _persist(cli, tmp_path, 1, identity)
    # Simulate a crash after the state file but before the pointer was replaced.
    torch.save(
        {"weight": torch.full((2, 2), 2.0)},
        tmp_path / cli.COMPLETED_ROUNDS_DIR / "round_02.state.pt",
    )
    resumed = cli.load_last_completed_round(tmp_path, identity)
    assert resumed["completed_server_round"] == 1


def test_resume_rekeys_the_schedule_and_keeps_the_logical_round(cli):
    """A resumed run must ask clients for the logical round, not Flower's round one."""
    strategy_class = cli.build_mvp_strategy_class()
    assert "round_offset" in strategy_class.__init__.__code__.co_varnames
    text = (ROOT / "scripts/mvp/run_mvp.py").read_text(encoding="utf-8")
    assert 'conf["server_round"] = logical' in text
    assert "index + completed" in text


def test_a_finished_federated_run_is_not_silently_re_executed(cli):
    text = (ROOT / "scripts/mvp/run_mvp.py").read_text(encoding="utf-8")
    assert "already completed round" in text
    assert "is not re-executed in place" in text


def test_ledger_rows_restore_exactly(cli):
    rows = [
        {
            "server_round": 1,
            "client_id": "client-v1-" + "a" * 64,
            "direction": "download",
            "payload_bytes": 512,
        },
        {
            "server_round": 1,
            "client_id": "client-v1-" + "a" * 64,
            "direction": "upload",
            "payload_bytes": 640,
        },
    ]
    restored = cli._restore_ledger(rows)
    assert cli._ledger_rows(restored) == rows
    assert restored.run_totals()["upload_bytes"] == 640
    assert restored.run_totals()["download_bytes"] == 512


def test_totals_of_counts_each_transmission_once(cli):
    rows = [
        {"server_round": 1, "client_id": "x", "direction": "download", "payload_bytes": 10},
        {"server_round": 1, "client_id": "y", "direction": "download", "payload_bytes": 20},
        {"server_round": 1, "client_id": "x", "direction": "upload", "payload_bytes": 30},
    ]
    totals = cli._totals_of(rows)
    assert totals == {
        "download_bytes": 30,
        "upload_bytes": 30,
        "download_transmissions": 2,
        "upload_transmissions": 1,
    }


def test_numpy_is_available_for_the_state_round_trip():
    """Guards the torch/numpy pairing the recovery state relies on."""
    assert np.asarray(torch.zeros(2)).shape == (2,)

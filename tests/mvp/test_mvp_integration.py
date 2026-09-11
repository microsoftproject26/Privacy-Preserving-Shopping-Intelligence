"""Integration tests: the two regimes really do share one implementation.

The expensive parts of the pilot are covered here on the synthetic bundle, including a
real two-round Flower simulation through the same ClientApp and strategy subclass the
20-round run uses. A test that only exercised helper functions would not have caught the
kind of defect these stages actually produce.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ppsi.federated.communication import CommunicationLedger
from ppsi.federated.mvp_data import load_prepared
from ppsi.federated.mvp_runner import (
    build_trainer,
    client_workload,
    evaluate_full_validation,
    model_identity,
    reset_client_stream,
    workload_batches,
)
from ppsi.federated.mvp_support import exposure_digest, require_complete_replies
from ppsi.training.flower import FlowerLocalAdapter
from ppsi.training.state import pack_shared_state
from ppsi.training.t1_mvp_objective import T1ContributingWeightPolicy

ROOT = Path(__file__).resolve().parents[2]


def _load_cli():
    spec = importlib.util.spec_from_file_location("mvp_cli", ROOT / "scripts/mvp/run_mvp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _initial_state(policy):
    identity = model_identity(policy)
    model, _ = build_trainer(identity, policy, None)
    return identity, {k: v.detach().clone() for k, v in model.state_dict().items()}


def _replay_centralized(prepared, policy, state):
    """One persistent core over the whole schedule; no averaging anywhere."""
    identity = model_identity(policy)
    model, core = build_trainer(identity, policy, state)
    exposure = []
    for entry in prepared.schedule:
        for client in entry["selected_client_ids"]:
            workload = client_workload(
                prepared,
                client,
                server_round=entry["server_round"],
                seed=13,
                batch_size=int(policy["pilot"]["batch_size"]),
            )
            exposure.append((entry["server_round"], client, workload.decision_keys))
            reset_client_stream(seed=13, server_round=entry["server_round"], client_id=client)
            for batch in workload_batches(prepared, workload, identity.spec):
                core.train_step(batch)
    return model, exposure


def _replay_fedavg(prepared, policy, state):
    """The same schedule, averaged each round, with the optimizer reset per round."""
    from scripts.federated.fl_synthetic_smoke import weighted_average_state_dicts

    identity = model_identity(policy)
    server_model, _ = build_trainer(identity, policy, state)
    server_state = pack_shared_state(server_model, server_model.shared_state_spec())
    exposure = []
    for entry in prepared.schedule:
        updates = {}
        # Processing follows the recorded selection order, exactly as the centralized
        # replay does; replies are only sorted by opaque id before they are averaged.
        for client in entry["selected_client_ids"]:
            workload = client_workload(
                prepared,
                client,
                server_round=entry["server_round"],
                seed=13,
                batch_size=int(policy["pilot"]["batch_size"]),
            )
            exposure.append((entry["server_round"], client, workload.decision_keys))
            model, core = build_trainer(identity, policy, state)
            adapter = FlowerLocalAdapter(
                core=core,
                shared_state_spec=model.shared_state_spec(),
                aggregation_weight_policy=T1ContributingWeightPolicy(),
            )
            reset_client_stream(seed=13, server_round=entry["server_round"], client_id=client)
            result = adapter.fit(
                server_state,
                workload_batches(prepared, workload, identity.spec),
                outer_round=entry["server_round"],
            )
            assert result.aggregation_weight == workload.rows
            updates[client] = (dict(result.shared_state), result.aggregation_weight)
        server_state = weighted_average_state_dicts([updates[c] for c in sorted(updates)])
    return server_state, exposure


def test_both_regimes_see_identical_exposure(prepared_dir, policy):
    prepared = load_prepared(prepared_dir)
    _, state = _initial_state(policy)
    _, central_exposure = _replay_centralized(prepared, policy, state)
    _, federated_exposure = _replay_fedavg(prepared, policy, state)
    assert exposure_digest(central_exposure) == exposure_digest(federated_exposure)


def test_round_zero_predictions_are_identical_before_any_training(prepared_dir, policy):
    prepared = load_prepared(prepared_dir)
    identity, state = _initial_state(policy)
    left, _ = build_trainer(identity, policy, state)
    right, _ = build_trainer(identity, policy, state)
    _, left_ranks = evaluate_full_validation(left, prepared, identity.spec, batch_size=4)
    _, right_ranks = evaluate_full_validation(right, prepared, identity.spec, batch_size=4)
    assert np.array_equal(left_ranks, right_ranks)


def test_training_actually_moves_the_model(prepared_dir, policy):
    prepared = load_prepared(prepared_dir)
    _, state = _initial_state(policy)
    trained, _ = _replay_centralized(prepared, policy, state)
    moved = [
        not torch.equal(state[key], value.detach()) for key, value in trained.state_dict().items()
    ]
    assert any(moved)


def test_an_incomplete_reply_set_is_refused(prepared_dir):
    prepared = load_prepared(prepared_dir)
    selected = list(prepared.schedule[0]["selected_client_ids"])
    require_complete_replies(selected, list(selected))
    with pytest.raises(ValueError, match="partial aggregation forbidden"):
        require_complete_replies(selected, selected[:1])


def test_the_ledger_measures_real_serialized_payloads(policy):
    flwr_app = pytest.importorskip("flwr.app")
    _, state = _initial_state(policy)
    record = flwr_app.ArrayRecord.from_torch_state_dict(state)
    ledger = CommunicationLedger()
    ledger.measure(record, server_round=1, client_id="client-v1-" + "a" * 64, direction="download")
    totals = ledger.run_totals()
    assert totals["download_bytes"] > 0
    assert totals["upload_bytes"] == 0
    assert totals["download_transmissions"] == 1
    expected = sum(p.numel() * p.element_size() for p in state.values())
    # Serialized payloads carry their own framing, so they are at least the tensor bytes.
    assert totals["download_bytes"] >= expected


def test_strategy_subclass_extends_the_proven_strategy():
    from scripts.federated.fl_real_smoke import RealDataTracingFedAvg

    cli = _load_cli()
    strategy_class = cli.build_mvp_strategy_class()
    assert issubclass(strategy_class, RealDataTracingFedAvg)
    assert "ledger" in strategy_class.__init__.__code__.co_varnames


def test_two_round_flower_simulation_through_the_real_client_app(prepared_dir, policy, tmp_path):
    """A real Ray-backed Flower run over the synthetic bundle, end to end."""
    pytest.importorskip("flwr.simulation")
    from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
    from flwr.serverapp import ServerApp
    from flwr.simulation import run_simulation

    cli = _load_cli()
    policy_path = tmp_path / "execution.v1.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    prepared = load_prepared(prepared_dir)
    _, state = _initial_state(policy)
    selected_ids = {
        int(e["server_round"]) - 1: list(e["selected_client_ids"]) for e in prepared.schedule
    }
    digests = {int(e["server_round"]) - 1: e["selected_digest"] for e in prepared.schedule}
    ledger = CommunicationLedger()
    strategy_class = cli.build_mvp_strategy_class()
    holder: dict = {}

    app = ServerApp()

    @app.main()
    def server_main(grid, context) -> None:
        strategy = strategy_class(
            ledger=ledger,
            expected_clients=2,
            selected_ids_by_round=selected_ids,
            selection_digests_by_round=digests,
            fraction_train=1.0,
            fraction_evaluate=0.0,
            min_train_nodes=2,
            min_evaluate_nodes=0,
            min_available_nodes=2,
            weighted_by_key="num-examples",
        )
        strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord.from_torch_state_dict(state),
            num_rounds=2,
            train_config=ConfigRecord(
                {
                    "prepared_dir": str(prepared_dir),
                    "data_manifest_sha256": prepared.data_manifest_sha256,
                    "policy_path": str(policy_path),
                    "experiment_seed": 13,
                }
            ),
            evaluate_fn=lambda server_round, arrays: MetricRecord(
                {"full_validation_executed": 0.0}
            ),
        )
        holder["tracing_log"] = strategy.tracing_log
        holder["replies"] = strategy.reply_ids_by_round
        holder["oracle"] = strategy.oracle_by_round

    run_simulation(
        server_app=app,
        client_app=cli.build_mvp_client_app(),
        num_supernodes=2,
        backend_name="ray",
        backend_config={
            "init_args": {"num_cpus": 1, "num_gpus": 0, "include_dashboard": False},
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
        },
    )

    tracing = holder["tracing_log"]
    assert len(tracing) == 2
    assert all(entry["aggregation_oracle_pass"] for entry in tracing)
    # The gate is now the derived per-tensor bound, recorded round by round.
    for logical_round in (1, 2):
        oracle = holder["oracle"][logical_round]
        assert oracle["oracle_pass"] is True
        assert oracle["contributing_client_count"] == 2
        assert oracle["worst_ratio"] <= 1.0
        assert oracle["oracle_policy_id"] == "SCALE_AWARE_FLOAT32_FORWARD_ERROR_V1"
    for server_round, entry in enumerate(tracing, start=1):
        assert holder["replies"][server_round] == sorted(selected_ids[server_round - 1])
        assert entry["contributing_examples"] == 6  # two clients, three rows each
    totals = ledger.run_totals()
    assert totals["download_transmissions"] == totals["upload_transmissions"] == 4
    assert totals["download_bytes"] > 0 and totals["upload_bytes"] > 0

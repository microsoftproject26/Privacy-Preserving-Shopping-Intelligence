"""Contract tests for the scoped T1 MVP integration modules."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ppsi.federated.mvp_data import load_prepared
from ppsi.federated.mvp_runner import (
    build_trainer,
    client_workload,
    headline_metric,
    model_identity,
    reset_client_stream,
    workload_batches,
)
from ppsi.federated.mvp_support import exposure_digest


def test_prepared_bundle_loads_read_only_and_partitions_train_rows(prepared_dir):
    prepared = load_prepared(prepared_dir)
    assert prepared.train.rows == 12
    assert prepared.validation.rows == 10
    assert len(prepared.population) == 4
    covered = sorted(int(r) for c in prepared.population for r in prepared.rows_for(c))
    assert covered == list(range(12))
    for array in prepared.train.arrays.values():
        assert not array.flags.writeable


def test_row_index_that_misses_a_row_is_refused(prepared_dir):
    index = json.loads((prepared_dir / "pilot_client_rows.json").read_text(encoding="utf-8"))
    index["client_rows"][index["population"][0]] = [0, 1]
    (prepared_dir / "pilot_client_rows.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="partition"):
        load_prepared(prepared_dir)


def test_row_index_that_double_counts_a_row_is_refused(prepared_dir):
    index = json.loads((prepared_dir / "pilot_client_rows.json").read_text(encoding="utf-8"))
    index["client_rows"][index["population"][0]] = [0, 1, 1]
    (prepared_dir / "pilot_client_rows.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="partition"):
        load_prepared(prepared_dir)


def test_scheduled_client_outside_the_population_is_refused(prepared_dir):
    schedule = json.loads((prepared_dir / "round_schedule.json").read_text(encoding="utf-8"))
    schedule["rounds"][0]["selected_client_ids"] = ["client-v1-" + "0" * 64]
    (prepared_dir / "round_schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the prepared population"):
        load_prepared(prepared_dir)


@pytest.mark.parametrize("rows", [[-1, 0], [0, 99], [0, 0]])
def test_row_selection_outside_the_split_is_refused(prepared_dir, rows):
    prepared = load_prepared(prepared_dir)
    with pytest.raises(ValueError):
        prepared.train.subset_rows(np.asarray(rows))


def test_local_epoch_visits_every_client_row_exactly_once(prepared_dir, policy):
    prepared = load_prepared(prepared_dir)
    client = prepared.population[0]
    workload = client_workload(prepared, client, server_round=1, seed=13, batch_size=2)
    owned = sorted(int(r) for r in prepared.rows_for(client))
    assert sorted(int(r) for r in workload.row_order) == owned
    batched = sorted(int(r) for batch in workload.batch_rows for r in batch)
    assert batched == owned
    assert len(workload.decision_keys) == len(set(workload.decision_keys))


def test_local_order_depends_on_the_round_but_membership_does_not(prepared_dir):
    prepared = load_prepared(prepared_dir)
    client = prepared.population[0]
    first = client_workload(prepared, client, server_round=1, seed=13, batch_size=2)
    second = client_workload(prepared, client, server_round=2, seed=13, batch_size=2)
    assert set(first.row_order.tolist()) == set(second.row_order.tolist())
    assert first.decision_keys != second.decision_keys or len(first.row_order) == 1


def test_local_schedule_is_reproducible(prepared_dir):
    prepared = load_prepared(prepared_dir)
    client = prepared.population[1]
    a = client_workload(prepared, client, server_round=3, seed=13, batch_size=2)
    b = client_workload(prepared, client, server_round=3, seed=13, batch_size=2)
    assert a.decision_keys == b.decision_keys


def test_client_rng_stream_is_deterministic_and_round_specific():
    first = reset_client_stream(seed=13, server_round=1, client_id="client-v1-" + "a" * 64)
    repeat = reset_client_stream(seed=13, server_round=1, client_id="client-v1-" + "a" * 64)
    other = reset_client_stream(seed=13, server_round=2, client_id="client-v1-" + "a" * 64)
    assert first == repeat != other


def test_model_identity_serializes_the_whole_resolved_configuration(policy):
    identity = model_identity(policy)
    payload = identity.to_dict()
    assert payload["model_config"]["core"] == "gru"
    assert payload["model_config"]["channels"] == ("category_id", "event_type_id")
    # Defaults the policy never mentions must still be recorded.
    assert "widths" in payload["model_config"] and "t3_hidden" in payload["model_config"]
    assert payload["batch_spec"]["candidate_continuous_dim"] == 1
    assert len(payload["batch_spec"]["history_categorical"]) == 5


def test_a_non_adam_optimizer_is_a_policy_change_not_a_default(policy):
    identity = model_identity(policy)
    policy = dict(policy)
    policy["pilot"] = dict(
        policy["pilot"], optimizer=dict(policy["pilot"]["optimizer"], name="SGD")
    )
    with pytest.raises(ValueError, match="policy change"):
        build_trainer(identity, policy, None)


def test_real_batches_pass_the_canonical_validator(prepared_dir, policy):
    from ppsi.training.batch import validate_phase1_batch

    prepared = load_prepared(prepared_dir)
    identity = model_identity(policy)
    workload = client_workload(
        prepared, prepared.population[0], server_round=1, seed=13, batch_size=2
    )
    for batch in workload_batches(prepared, workload, identity.spec):
        validate_phase1_batch(batch, identity.spec)
        assert bool(batch.t1_present.all())
        assert not bool(batch.t2_present.any()) and not bool(batch.t3_present.any())


def test_the_objective_refuses_a_batch_that_smuggles_in_another_task(prepared_dir, policy):
    from ppsi.training.t1_mvp_objective import T1MVPObjective

    prepared = load_prepared(prepared_dir)
    identity = model_identity(policy)
    workload = client_workload(
        prepared, prepared.population[0], server_round=1, seed=13, batch_size=2
    )
    batch = workload_batches(prepared, workload, identity.spec)[0]
    model, _ = build_trainer(identity, policy, None)
    smuggled = batch.to("cpu")
    smuggled.t2_present[:] = True
    with pytest.raises(ValueError, match="cannot consume present T2/T3"):
        T1MVPObjective()(smuggled, model(batch))


def test_exposure_digest_is_order_sensitive_and_reproducible(prepared_dir):
    prepared = load_prepared(prepared_dir)
    records = []
    for entry in prepared.schedule:
        for client in entry["selected_client_ids"]:
            workload = client_workload(
                prepared, client, server_round=entry["server_round"], seed=13, batch_size=2
            )
            records.append((entry["server_round"], client, workload.decision_keys))
    assert exposure_digest(records) == exposure_digest(list(records))
    shuffled = [(r, c, list(reversed(keys))) for r, c, keys in records]
    assert exposure_digest(shuffled) != exposure_digest(records)


def test_headline_refuses_an_unavailable_slice():
    with pytest.raises(ValueError, match="not a zero"):
        headline_metric({"slices": {"next_distinct": {"status": "ZERO_SUPPORT"}}})


def test_full_validation_produces_raw_one_based_ranks(prepared_dir, policy):
    from ppsi.federated.mvp_runner import evaluate_full_validation

    prepared = load_prepared(prepared_dir)
    identity = model_identity(policy)
    model, _ = build_trainer(identity, policy, None)
    summary, ranks = evaluate_full_validation(model, prepared, identity.spec, batch_size=4)
    assert ranks.shape == (prepared.validation.rows,)
    assert int(ranks.min()) >= 1 and int(ranks.max()) <= 588
    assert summary["decision_count"] == prepared.validation.rows
    assert summary["slices"]["next_distinct"]["status"] == "AVAILABLE"
    assert headline_metric(summary)["support_decisions"] == prepared.validation.rows


def test_the_same_state_evaluates_identically_on_two_freshly_built_models(prepared_dir, policy):
    from ppsi.federated.mvp_runner import evaluate_full_validation

    prepared = load_prepared(prepared_dir)
    identity = model_identity(policy)
    left, _ = build_trainer(identity, policy, None)
    state = {k: v.detach().clone() for k, v in left.state_dict().items()}
    right, _ = build_trainer(identity, policy, state)
    _, left_ranks = evaluate_full_validation(left, prepared, identity.spec, batch_size=4)
    _, right_ranks = evaluate_full_validation(right, prepared, identity.spec, batch_size=4)
    assert np.array_equal(left_ranks, right_ranks)


def test_validation_metadata_is_aligned_with_the_prepared_rows(prepared_dir):
    prepared = load_prepared(prepared_dir)
    meta = prepared.validation_metadata()
    assert len(meta["client_ids"]) == prepared.validation.rows
    assert meta["category_changed"].shape == (prepared.validation.rows,)
    assert meta["history_count_basis"] == "TRUE_RAW_TRAIN_EVENT_ROWS_PER_CLIENT"


def test_a_validation_client_without_a_measured_history_count_is_refused(prepared_dir):
    index = json.loads((prepared_dir / "validation_clients.json").read_text(encoding="utf-8"))
    index["train_history_counts"].pop(next(iter(index["train_history_counts"])))
    (prepared_dir / "validation_clients.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="measured TRAIN history count"):
        load_prepared(prepared_dir).validation_metadata()


def test_deterministic_execution_pins_threads_and_streams(policy):
    from ppsi.federated.mvp_runner import set_deterministic_execution

    set_deterministic_execution(policy, seed=13)
    assert torch.get_num_threads() == int(policy["resources"]["torch_threads"])

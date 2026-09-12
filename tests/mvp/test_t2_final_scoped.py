"""Focused tests for T2 final scoped matched protocol, loss, Flower smoke, and identity."""

from __future__ import annotations

from pathlib import Path

import torch

from ppsi.federated.sampling import sample_clients
from ppsi.models.batch_spec import phase1_batch_spec_v1
from ppsi.models.session_gru import SessionGRUConfig, build_model
from ppsi.training.batch import Phase1Batch
from ppsi.training.core import LocalTrainerCore, TrainerPolicy
from ppsi.training.flower import FlowerLocalAdapter
from ppsi.training.state import pack_shared_state
from ppsi.training.t2_mvp_objective import T2ContributingWeightPolicy, T2MVPObjective, masked_t2_sum
from scripts.federated.fl_synthetic_smoke import weighted_average_state_dicts

ROOT = Path(__file__).resolve().parents[2]


def make_dummy_t2_batch(size: int = 4, t2_present: bool = True, seed: int = 13) -> Phase1Batch:
    torch.manual_seed(seed)
    spec = phase1_batch_spec_v1()
    length = 5
    lengths = torch.tensor([3, 4, 2, 5][:size], dtype=torch.int64)
    history_mask = torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)

    return Phase1Batch(
        history_categorical_ids={
            "category_id": torch.randint(0, 10, (size, length)),
            "product_bucket": torch.randint(0, 100, (size, length)),
            "event_type_id": torch.randint(0, 4, (size, length)),
            "brand_bucket": torch.randint(0, 20, (size, length)),
            "price_band": torch.randint(0, 5, (size, length)),
        },
        history_continuous_features=torch.rand((size, length, 1)),
        lengths=lengths,
        history_mask=history_mask,
        query_categorical_ids={
            "query_category_id": torch.randint(0, 10, (size,)),
            "query_product_bucket": torch.randint(0, 100, (size,)),
            "query_brand_bucket": torch.randint(0, 20, (size,)),
            "query_price_band": torch.randint(0, 5, (size,)),
        },
        query_continuous_features=torch.zeros((size, spec.query_continuous_dim), dtype=torch.float32),
        candidate_ids=torch.full((size, 1), spec.candidate_id_pad_id, dtype=torch.int64),
        candidate_categorical_ids={
            "candidate_category_id": torch.full((size, 1), 589, dtype=torch.int64),
            "candidate_price_band": torch.full((size, 1), 5, dtype=torch.int64),
        },
        candidate_continuous_features=torch.zeros((size, 1, spec.candidate_continuous_dim), dtype=torch.float32),
        candidate_mask=torch.zeros((size, 1), dtype=torch.bool),
        t1_target=torch.zeros(size, dtype=torch.int64),
        t2_target=torch.tensor([[1.0], [0.0], [1.0], [0.0]][:size], dtype=torch.float32),
        t3_gains=torch.zeros(size, 1, dtype=torch.float32),
        t1_present=torch.zeros(size, dtype=torch.bool),
        t2_present=torch.ones(size, dtype=torch.bool) if t2_present else torch.zeros(size, dtype=torch.bool),
        t3_present=torch.zeros(size, dtype=torch.bool),
    )


def test_masked_t2_loss_positive_negative_and_gradients():
    logits = torch.tensor([1.5, -0.5, 2.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 0.0])
    present = torch.tensor([True, True, False])

    numerator, support = masked_t2_sum(logits, targets, present)
    assert support == 2
    assert numerator is not None
    assert torch.isfinite(numerator)

    numerator.backward()
    assert logits.grad is not None
    # Masked row (index 2) must receive 0 gradient
    assert logits.grad[2].item() == 0.0
    # Present rows (indices 0 and 1) must have non-zero gradients
    assert logits.grad[0].item() != 0.0
    assert logits.grad[1].item() != 0.0


def test_t2_objective_no_present_examples():
    objective = T2MVPObjective()
    batch = make_dummy_t2_batch(size=3, t2_present=False)

    class DummyOutput:
        t2_logit = torch.tensor([[0.5], [-1.0], [2.0]])

    res = objective(batch, DummyOutput())
    assert res.status.name == "NO_CONTRIBUTING_TASK"
    assert res.total_loss is None
    assert res.contributing_examples == 0


def test_tiny_t2_flower_two_clients_one_round_smoke():
    spec = phase1_batch_spec_v1()
    model_config = SessionGRUConfig(
        channels=("category_id", "event_type_id"),
        use_gap=True,
        hidden=32,
        layers=1,
        dropout=0.0,
        core="gru",
    )
    server_model = build_model(seed=13, batch_spec=spec, config=model_config)
    server_state = pack_shared_state(server_model, server_model.shared_state_spec())

    client_updates = []
    for c_idx in range(2):
        client_model = build_model(seed=13, batch_spec=spec, config=model_config)
        client_opt = torch.optim.Adam(client_model.parameters(), lr=0.01)
        core = LocalTrainerCore(
            model=client_model,
            batch_spec=spec,
            objective=T2MVPObjective(),
            optimizer=client_opt,
            policy=TrainerPolicy(gradient_clip_norm=1.0),
            device="cpu",
        )
        adapter = FlowerLocalAdapter(
            core=core,
            shared_state_spec=client_model.shared_state_spec(),
            aggregation_weight_policy=T2ContributingWeightPolicy(),
        )
        batches = [make_dummy_t2_batch(size=4, t2_present=True, seed=c_idx + 10)]
        fit_res = adapter.fit(server_state, batches, outer_round=1)
        assert isinstance(fit_res.aggregation_weight, int) and fit_res.aggregation_weight > 0
        client_updates.append((dict(fit_res.shared_state), fit_res.aggregation_weight))

    aggregated_state = weighted_average_state_dicts(client_updates)
    assert set(aggregated_state.keys()) == set(server_state.keys())
    for k in aggregated_state:
        assert torch.isfinite(aggregated_state[k]).all()


def test_t2_matched_identity_invariants():
    seed = 13
    eligible_cids = [f"client_{i:04d}" for i in range(250)]

    pop_r1 = sample_clients(eligible_cids, experiment_seed=seed, round_index=0, clients_per_round=200, sampler_version="mvp_population_v1")
    pop_r2a = sample_clients(eligible_cids, experiment_seed=seed, round_index=0, clients_per_round=200, sampler_version="mvp_population_v1")
    assert pop_r1.selected_client_ids == pop_r2a.selected_client_ids
    assert pop_r1.selected_digest == pop_r2a.selected_digest

    sched_r1 = []
    sched_r2a = []
    for r in range(10):
        s_r1 = sample_clients(pop_r1.selected_client_ids, experiment_seed=seed, round_index=r, clients_per_round=20, sampler_version="mvp_population_v1")
        s_r2a = sample_clients(pop_r2a.selected_client_ids, experiment_seed=seed, round_index=r, clients_per_round=20, sampler_version="mvp_population_v1")
        sched_r1.append(s_r1.selected_client_ids)
        sched_r2a.append(s_r2a.selected_client_ids)

    assert sched_r1 == sched_r2a

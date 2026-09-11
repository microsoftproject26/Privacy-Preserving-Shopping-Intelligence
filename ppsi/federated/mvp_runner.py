"""One training and evaluation implementation shared by both MVP regimes.

The centralized run and the FedAvg run of ``mvp-t1-001`` differ in exactly two declared
ways: who averages the updates, and whether the optimizer survives a round boundary.
Everything else — the model, the objective, the batches, the within-client order, the RNG
stream, the evaluator — comes from this module, so a measured difference between the two
cannot be an accident of two separate implementations.

Nothing here decides policy. Sizes, seeds and lifecycles arrive from
``config/mvp/execution.v1.json``; the frozen contracts arrive from ``ppsi`` and are used
unchanged.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from ppsi.data.batching import windows_to_batch
from ppsi.evaluation.t1 import evaluate_t1_ranks, rank_targets_from_scores
from ppsi.federated.mvp_data import PreparedMVP
from ppsi.federated.mvp_support import client_epoch_order, client_rng_seed
from ppsi.models.batch_spec import Phase1BatchSpec, phase1_batch_spec_v1
from ppsi.models.session_gru import SessionGRU, SessionGRUConfig, build_model
from ppsi.training.core import LocalTrainerCore, TrainerPolicy
from ppsi.training.t1_mvp_objective import T1MVPObjective


@dataclass(frozen=True, slots=True)
class MVPModelIdentity:
    """The resolved architecture, serialized in full rather than by its changed fields."""

    config: SessionGRUConfig
    spec: Phase1BatchSpec
    category_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_config": asdict(self.config),
            "batch_spec": _spec_to_dict(self.spec),
            "category_count": self.category_count,
            "factory": "ppsi.models.session_gru.build_model",
            "spec_factory": "ppsi.models.batch_spec.phase1_batch_spec_v1",
        }


def _spec_to_dict(spec: Phase1BatchSpec) -> dict[str, Any]:
    """The complete resolved batch spec, including every channel it carries."""
    return {
        "schema": spec.schema,
        "version": spec.version,
        "history_categorical": [_channel_to_dict(c) for c in spec.history_categorical],
        "query_categorical": [_channel_to_dict(c) for c in spec.query_categorical],
        "candidate_categorical": [_channel_to_dict(c) for c in spec.candidate_categorical],
        "history_continuous_dim": spec.history_continuous_dim,
        "query_continuous_dim": spec.query_continuous_dim,
        "candidate_continuous_dim": spec.candidate_continuous_dim,
        "candidate_id_pad_id": spec.candidate_id_pad_id,
        "candidate_id_vocab_size": spec.candidate_id_vocab_size,
        "t1_absent_fill": spec.t1_absent_fill,
        "t2_absent_fill": spec.t2_absent_fill,
        "t3_absent_fill": spec.t3_absent_fill,
    }


def _channel_to_dict(channel) -> dict[str, Any]:
    return {"name": channel.name, "pad_id": channel.pad_id, "vocab_size": channel.vocab_size}


def model_identity(policy: dict) -> MVPModelIdentity:
    """The exact accepted architecture named by the execution policy."""
    model_cfg = policy["model"]
    config = SessionGRUConfig(
        channels=tuple(model_cfg["channels"]),
        use_gap=bool(model_cfg["use_gap"]),
        hidden=int(model_cfg["hidden"]),
        layers=int(model_cfg["layers"]),
        dropout=float(model_cfg["dropout"]),
        core=str(model_cfg["core"]),
    )
    spec = phase1_batch_spec_v1()
    return MVPModelIdentity(
        config=config, spec=spec, category_count=int(model_cfg["category_count"])
    )


def set_deterministic_execution(policy: dict, *, seed: int) -> None:
    """Pin threads and RNG streams before anything reads them."""
    threads = int(policy["resources"]["torch_threads"])
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def reset_client_stream(*, seed: int, server_round: int, client_id: str) -> int:
    """Reset the local RNG streams so dropout is independent of worker assignment.

    The derived value is an execution stream, not an identity: the client is still the
    same client, and its data selection does not change.
    """
    derived = client_rng_seed(seed=seed, server_round=server_round, client_id=client_id)
    random.seed(derived)
    np.random.seed(derived % 2**32)
    torch.manual_seed(derived)
    return derived


def build_trainer(
    identity: MVPModelIdentity,
    policy: dict,
    state: dict[str, torch.Tensor] | None = None,
) -> tuple[SessionGRU, LocalTrainerCore]:
    """The frozen model plus the one shared trainer core, on CPU."""
    seed = int(policy["pilot"]["seed"])
    optimizer_cfg = policy["pilot"]["optimizer"]
    model = build_model(seed, batch_spec=identity.spec, config=identity.config)
    if state is not None:
        model.load_state_dict(state, strict=True)
    if optimizer_cfg["name"] != "Adam":
        raise ValueError("the MVP policy declares Adam; another optimizer is a policy change")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    core = LocalTrainerCore(
        model=model,
        batch_spec=identity.spec,
        objective=T1MVPObjective(),
        optimizer=optimizer,
        scheduler=None,
        policy=TrainerPolicy(gradient_clip_norm=float(optimizer_cfg["gradient_clip_norm"])),
        device=str(policy["resources"]["device"]),
    )
    return model, core


@dataclass
class ClientWorkload:
    """One client's local epoch: every one of its frozen rows, exactly once."""

    client_id: str
    server_round: int
    row_order: np.ndarray
    batch_rows: list[np.ndarray] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return len(self.row_order)

    @property
    def decision_keys(self) -> list[str]:
        return [str(int(row)) for row in self.row_order]


def client_workload(
    prepared: PreparedMVP,
    client_id: str,
    *,
    server_round: int,
    seed: int,
    batch_size: int,
) -> ClientWorkload:
    """Deterministic local schedule: one permutation of all the client's rows."""
    rows = prepared.train.subset_rows(prepared.rows_for(client_id))
    order = client_epoch_order(rows, seed=seed, server_round=server_round, client_id=client_id)
    if len(order) != len(rows) or set(order.tolist()) != set(rows.tolist()):
        raise ValueError("the local epoch order must be a permutation of the client's own rows")
    workload = ClientWorkload(client_id=client_id, server_round=server_round, row_order=order)
    for start in range(0, len(order), batch_size):
        # Sorting inside a fetched batch is an I/O choice; membership is already fixed.
        workload.batch_rows.append(np.sort(order[start : start + batch_size]))
    return workload


def workload_batches(prepared: PreparedMVP, workload: ClientWorkload, spec: Phase1BatchSpec):
    """Materialize one client's batches from the read-only memory maps."""
    windows = prepared.train.windows()
    return [windows_to_batch(windows, rows, spec, validate=False) for rows in workload.batch_rows]


@torch.no_grad()
def evaluate_full_validation(
    model: SessionGRU,
    prepared: PreparedMVP,
    spec: Phase1BatchSpec,
    *,
    batch_size: int,
    category_count: int = 588,
) -> tuple[dict[str, Any], np.ndarray]:
    """Score every frozen VALIDATION decision and rank the target with the official rule.

    Ranks are collected in the frozen decision order and evaluated once at the end:
    per-batch metrics averaged together are a different quantity from the metric the
    protocol defines.
    """
    windows = prepared.validation.windows()
    total = prepared.validation.rows
    model.eval()
    ranks = np.empty(total, dtype=np.int64)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        rows = np.arange(start, stop, dtype=np.int64)
        batch = windows_to_batch(windows, rows, spec, validate=False)
        logits = model(batch).t1_logits
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite validation logits")
        chunk = rank_targets_from_scores(
            logits.to(torch.float64), batch.t1_target, category_count=category_count
        )
        ranks[start:stop] = chunk.to(torch.int64).numpy()
    if ranks.min() < 1 or ranks.max() > category_count:
        raise ValueError("a raw T1 rank left the 1..C range; zero-for-miss is not this convention")

    meta = prepared.validation_metadata()
    summary = evaluate_t1_ranks(
        ranks,
        meta["category_changed"],
        meta["client_ids"],
        train_history_counts=meta["train_history_counts"],
        mrr_cutoff=20,
    )
    return summary.to_dict(), ranks


def headline_metric(summary: dict[str, Any]) -> dict[str, Any]:
    """The frozen headline slice, with its own support, never a zero stand-in."""
    slices = summary.get("slices", {})
    info = slices.get("next_distinct")
    if not info or info.get("status") != "AVAILABLE":
        raise ValueError("the headline next_distinct slice is unavailable; it is not a zero")
    return {
        "metric_id": "t1.next_distinct.mrr_at_20.macro",
        "value": float(info["mrr_at_20_macro"]),
        "micro": float(info["mrr_at_20_micro"]),
        "accuracy_at_1_macro": float(info["accuracy_at_1_macro"]),
        "accuracy_at_1_micro": float(info["accuracy_at_1_micro"]),
        "support_clients": int(info["client_count"]),
        "support_decisions": int(info["decision_count"]),
    }

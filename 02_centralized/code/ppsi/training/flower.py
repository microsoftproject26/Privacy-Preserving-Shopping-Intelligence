"""Thin Flower-local adapter that reuses the shared LocalTrainerCore."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from ppsi.training.batch import Phase1Batch
from ppsi.training.centralized import AdapterRunSummary, merge_local_summaries
from ppsi.training.core import LocalTrainerCore
from ppsi.training.sampler import TrainingCursor
from ppsi.training.state import (
    SharedStateSpec,
    load_shared_state,
    pack_shared_state,
    shared_state_digest,
    validate_packed_shared_state,
)


class AggregationWeightPolicy(Protocol):
    """S1-PR-06 will freeze the scientific client-update weighting policy."""

    policy_id: str

    def __call__(self, summary: AdapterRunSummary) -> int: ...


@dataclass(frozen=True, slots=True)
class ContributingRowsSmokeWeightPolicy:
    """Non-scientific policy used only by the #18 synthetic contract smoke."""

    policy_id: str = "contributing_rows_smoke_weight_v1_NON_SCIENTIFIC"

    def __call__(self, summary: AdapterRunSummary) -> int:
        return summary.contributing_examples


@dataclass(frozen=True, slots=True)
class FlowerFitResult:
    shared_state: Mapping[str, Tensor]
    aggregation_weight: int
    summary: AdapterRunSummary
    received_state_digest: str
    updated_state_digest: str

    def scalar_metrics(self) -> dict[str, int | float]:
        metrics: dict[str, int | float] = {
            "num-examples": self.aggregation_weight,
            "examples_processed": self.summary.examples_processed,
            "contributing_examples": self.summary.contributing_examples,
            "optimizer_steps": self.summary.optimizer_steps,
            "skipped_steps": self.summary.skipped_steps,
        }
        task_means: list[float] = []
        for task, stat in sorted(self.summary.task_stats.items()):
            key = task.lower()
            metrics[f"{key}_loss_numerator"] = stat.numerator
            metrics[f"{key}_loss_denominator"] = stat.denominator
            if stat.mean is not None:
                task_means.append(stat.mean)
        # Non-scientific smoke diagnostic matching the equal-weight smoke objective.
        metrics["train_loss"] = sum(task_means)
        return metrics


class FlowerLocalAdapter:
    """Parameter translation plus the same core used by centralized execution."""

    optimizer_lifecycle = "RESET_EACH_SERVER_ROUND"

    def __init__(
        self,
        core: LocalTrainerCore,
        shared_state_spec: SharedStateSpec,
        aggregation_weight_policy: AggregationWeightPolicy,
    ) -> None:
        self.core = core
        self.shared_state_spec = shared_state_spec
        self.aggregation_weight_policy = aggregation_weight_policy

    def fit(
        self,
        incoming_state: Mapping[str, Tensor],
        batches: Iterable[Phase1Batch],
        *,
        outer_round: int,
        local_epoch: int = 0,
        cursor: TrainingCursor | None = None,
    ) -> FlowerFitResult:
        validate_packed_shared_state(incoming_state, self.shared_state_spec)
        received_digest = shared_state_digest(incoming_state)
        start_cursor = cursor or TrainingCursor(
            outer_round=outer_round,
            local_epoch=local_epoch,
            next_batch_index=0,
            optimizer_step=0,
        )
        if start_cursor.outer_round not in {None, outer_round}:
            raise ValueError("Flower cursor outer_round does not match fit round")
        if start_cursor.local_epoch != local_epoch:
            raise ValueError("Flower cursor local_epoch does not match fit epoch")

        if cursor is None:
            self.core.reset_optimizer_for_new_server_round()
            load_shared_state(self.core.model, incoming_state, self.shared_state_spec)
        elif self.core.optimizer_step_count != cursor.optimizer_step:
            raise ValueError("Flower cursor optimizer_step does not match restored trainer state")
        summaries = [self.core.train_step(batch) for batch in batches]
        final_cursor = TrainingCursor(
            outer_round=outer_round,
            local_epoch=local_epoch,
            next_batch_index=start_cursor.next_batch_index + len(summaries),
            optimizer_step=self.core.optimizer_step_count,
        )
        merged = merge_local_summaries(summaries, final_cursor=final_cursor)
        updated = pack_shared_state(self.core.model, self.shared_state_spec)
        aggregation_weight = self.aggregation_weight_policy(merged)
        if isinstance(aggregation_weight, bool) or not isinstance(aggregation_weight, int):
            raise TypeError("AggregationWeightPolicy must return an int")
        if aggregation_weight < 0:
            raise ValueError("AggregationWeightPolicy must not return a negative weight")
        return FlowerFitResult(
            shared_state=updated,
            aggregation_weight=aggregation_weight,
            summary=merged,
            received_state_digest=received_digest,
            updated_state_digest=shared_state_digest(updated),
        )

    def evaluate(
        self,
        incoming_state: Mapping[str, Tensor],
        batches: Iterable[Phase1Batch],
        *,
        outer_round: int,
    ) -> AdapterRunSummary:
        load_shared_state(self.core.model, incoming_state, self.shared_state_spec)
        summaries = [self.core.eval_step(batch) for batch in batches]
        cursor = TrainingCursor(
            outer_round=outer_round,
            local_epoch=0,
            next_batch_index=0,
            optimizer_step=self.core.optimizer_step_count,
        )
        return merge_local_summaries(summaries, final_cursor=cursor)

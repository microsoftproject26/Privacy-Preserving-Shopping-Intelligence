"""Thin centralized orchestration over the shared LocalTrainerCore."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ppsi.training.batch import Phase1Batch
from ppsi.training.core import LocalTrainerCore
from ppsi.training.outputs import LocalStepSummary, SupportUnit
from ppsi.training.sampler import TrainingCursor


@dataclass(frozen=True, slots=True)
class MergedTaskStat:
    numerator: float
    denominator: int
    support_unit: SupportUnit

    @property
    def mean(self) -> float | None:
        return None if self.denominator == 0 else self.numerator / self.denominator


@dataclass(frozen=True, slots=True)
class AdapterRunSummary:
    task_stats: Mapping[str, MergedTaskStat]
    examples_processed: int
    contributing_examples: int
    optimizer_steps: int
    skipped_steps: int
    final_cursor: TrainingCursor


class CentralizedAdapter:
    """Own outer-loop iteration only; all local semantics stay in the core."""

    def __init__(self, core: LocalTrainerCore) -> None:
        self.core = core

    def train_batches(
        self,
        batches: Iterable[Phase1Batch],
        *,
        cursor: TrainingCursor,
    ) -> AdapterRunSummary:
        summaries: list[LocalStepSummary] = []
        next_batch_index = cursor.next_batch_index
        for batch in batches:
            summary = self.core.train_step(batch)
            summaries.append(summary)
            next_batch_index += 1
        final_cursor = TrainingCursor(
            outer_round=cursor.outer_round,
            local_epoch=cursor.local_epoch,
            next_batch_index=next_batch_index,
            optimizer_step=self.core.optimizer_step_count,
        )
        return merge_local_summaries(summaries, final_cursor=final_cursor)

    def evaluate_batches(
        self,
        batches: Iterable[Phase1Batch],
        *,
        cursor: TrainingCursor,
    ) -> AdapterRunSummary:
        summaries = [self.core.eval_step(batch) for batch in batches]
        return merge_local_summaries(summaries, final_cursor=cursor)


def merge_local_summaries(
    summaries: Iterable[LocalStepSummary],
    *,
    final_cursor: TrainingCursor,
) -> AdapterRunSummary:
    numerator: dict[str, float] = {}
    denominator: dict[str, int] = {}
    units: dict[str, SupportUnit] = {}
    examples_processed = 0
    contributing_examples = 0
    optimizer_steps = 0
    skipped_steps = 0

    for summary in summaries:
        examples_processed += summary.examples_processed
        contributing_examples += summary.contributing_examples
        optimizer_steps += int(summary.optimizer_step_performed)
        skipped_steps += int(not summary.optimizer_step_performed)
        for task, stat in summary.task_stats.items():
            if task in units and units[task] != stat.support_unit:
                raise ValueError(f"Cannot merge different support units for {task}")
            units[task] = stat.support_unit
            numerator[task] = numerator.get(task, 0.0) + stat.numerator
            denominator[task] = denominator.get(task, 0) + stat.denominator

    merged = {
        task: MergedTaskStat(numerator[task], denominator[task], units[task])
        for task in sorted(numerator)
    }
    return AdapterRunSummary(
        task_stats=merged,
        examples_processed=examples_processed,
        contributing_examples=contributing_examples,
        optimizer_steps=optimizer_steps,
        skipped_steps=skipped_steps,
        final_cursor=final_cursor,
    )

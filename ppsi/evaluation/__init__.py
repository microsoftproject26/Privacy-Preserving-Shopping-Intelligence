"""Evaluation package for Privacy-Preserving Shopping Intelligence."""

from ppsi.evaluation.t1 import (
    T1EvaluationSummary,
    evaluate_t1_ranks,
    history_bucket,
    metric_records_from_t1_summary,
    preflight_t1_task_examples,
    rank_targets_from_rankings,
    rank_targets_from_scores,
    validate_t1_dense_labels,
    validate_t1_evaluator_config,
)

__all__ = [
    "T1EvaluationSummary",
    "evaluate_t1_ranks",
    "history_bucket",
    "metric_records_from_t1_summary",
    "preflight_t1_task_examples",
    "rank_targets_from_rankings",
    "rank_targets_from_scores",
    "validate_t1_dense_labels",
    "validate_t1_evaluator_config",
]

"""Deterministic T1 Evaluation Harness for Shopping Intelligence.

This module implements the frozen T1 evaluation metrics according to
ADR-001, Gate G1, and the S2-PR-07 evaluator specification.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import Tensor

from ppsi.training.identity import file_sha256
from scripts.experiments.schemas import validate_metric_record

# Canonical allowed top-level keys in config/evaluation/t1_evaluator.v1.json
_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "schema",
        "version",
        "task",
        "cohort",
        "category_count",
        "label_semantics",
        "score_universe",
        "tie_break",
        "invalid_input_policy",
        "empty_slice_policy",
        "headline_slice",
        "headline_metric_id",
        "mrr_cutoff",
        "slices",
        "averaging",
        "history_buckets",
        "emit_metrics",
        "deferred_secondary_metrics",
        "real_validation_task_examples",
        "merged_input_evidence",
        "vocabulary",
        "accepted_protocol",
        "g1_gate",
        "edge_decision",
    }
)

_VALID_HISTORY_BUCKET_IDS = (
    "BELOW_10_RETAINED_C1",
    "10_19",
    "20_49",
    "50_99",
    "100_plus",
)


def validate_t1_evaluator_config(config: dict[str, Any]) -> None:
    """Validate T1 evaluator configuration dictionary against canonical specification."""
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dict, got {type(config).__name__}")

    unknown_keys = set(config.keys()) - _ALLOWED_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown top-level config fields: {sorted(unknown_keys)}")

    if config.get("schema") != "t1_evaluator_config_v1":
        raise ValueError(
            f"Invalid schema: {config.get('schema')!r}, expected 't1_evaluator_config_v1'"
        )

    if str(config.get("version")) != "1":
        raise ValueError(f"Invalid version: {config.get('version')!r}, expected '1'")

    if config.get("task") != "T1":
        raise ValueError(f"Invalid task: {config.get('task')!r}, expected 'T1'")

    if config.get("cohort") != "C1":
        raise ValueError(f"Invalid cohort: {config.get('cohort')!r}, expected 'C1'")

    category_count = config.get("category_count")
    if category_count != 588:
        raise ValueError(f"Invalid category_count: {category_count!r}, expected 588")

    if config.get("tie_break") != "ASCENDING_DENSE_CATEGORY_CODE":
        raise ValueError(f"Invalid tie_break: {config.get('tie_break')!r}")

    if config.get("invalid_input_policy") != "ERROR_NO_FALLBACK":
        raise ValueError(f"Invalid invalid_input_policy: {config.get('invalid_input_policy')!r}")

    if config.get("empty_slice_policy") != "ZERO_SUPPORT_NO_METRIC":
        raise ValueError(f"Invalid empty_slice_policy: {config.get('empty_slice_policy')!r}")

    if config.get("headline_slice") != "next_distinct":
        raise ValueError(f"Invalid headline_slice: {config.get('headline_slice')!r}")

    if config.get("headline_metric_id") != "t1.next_distinct.mrr_at_20.macro":
        raise ValueError(f"Invalid headline_metric_id: {config.get('headline_metric_id')!r}")

    if config.get("mrr_cutoff") != 20:
        raise ValueError(f"Invalid mrr_cutoff: {config.get('mrr_cutoff')!r}, expected 20")

    emit_metrics = config.get("emit_metrics")
    if emit_metrics != ["mrr_at_20", "accuracy_at_1"]:
        raise ValueError(f"Invalid emit_metrics: {emit_metrics!r}")


def validate_t1_dense_labels(labels: Any, category_count: int = 588) -> list[int]:
    """Validate numeric storage values as dense category codes in [0, category_count).

    Rejects nulls, booleans, non-finite values, non-integral values, and out-of-range codes.
    """
    if labels is None:
        raise ValueError("Labels cannot be None")

    if isinstance(labels, (Tensor, np.ndarray)):
        raw_list = labels.tolist()
    elif isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        raw_list = list(labels)
    else:
        try:
            raw_list = list(labels)
        except TypeError as exc:
            raise TypeError(
                f"Labels must be an iterable/sequence, got {type(labels).__name__}"
            ) from exc

    validated: list[int] = []
    for idx, val in enumerate(raw_list):
        if val is None:
            raise ValueError(f"Label at index {idx} is null/None")

        if isinstance(val, bool):
            raise TypeError(
                f"Label at index {idx} is boolean ({val!r}); booleans are not valid category codes"
            )

        try:
            float_val = float(val)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Label at index {idx} cannot be converted to float: {val!r}") from exc

        if not math.isfinite(float_val):
            raise ValueError(f"Label at index {idx} is non-finite: {float_val!r}")

        if not float_val.is_integer():
            raise ValueError(f"Label at index {idx} is not an integer code: {float_val!r}")

        int_val = int(float_val)
        if not (0 <= int_val < category_count):
            raise ValueError(
                f"Label at index {idx} is out of bounds [0, {category_count}): {int_val}"
            )

        validated.append(int_val)

    return validated


def rank_targets_from_scores(
    scores: Tensor | np.ndarray | Sequence[Sequence[float]],
    targets: Tensor | np.ndarray | Sequence[int],
    category_count: int = 588,
) -> Tensor:
    """Compute 1-based target ranks from a full score matrix [N, category_count].

    Ties are broken deterministically by ASCENDING_DENSE_CATEGORY_CODE.
    """
    if not isinstance(scores, Tensor):
        scores_t = torch.as_tensor(scores, dtype=torch.float64)
    else:
        scores_t = scores

    if scores_t.ndim != 2:
        raise ValueError(f"Scores must be 2-dimensional [N, C], got shape {list(scores_t.shape)}")

    num_rows, num_cols = scores_t.shape
    if num_rows < 1:
        raise ValueError(f"Scores must contain at least one row, got {num_rows}")

    if num_cols != category_count:
        raise ValueError(f"Scores width {num_cols} does not match category_count {category_count}")

    if not torch.all(torch.isfinite(scores_t)):
        raise ValueError("Scores contain non-finite (NaN or Inf) values")

    validated_targets = validate_t1_dense_labels(targets, category_count=category_count)
    if len(validated_targets) != num_rows:
        raise ValueError(
            f"Number of targets ({len(validated_targets)}) does not match scores rows ({num_rows})"
        )

    targets_t = torch.tensor(validated_targets, dtype=torch.int64, device=scores_t.device)

    row_indices = torch.arange(num_rows, device=scores_t.device)
    target_scores = scores_t[row_indices, targets_t]  # shape [N]

    # Scores strictly greater than target score
    greater = (scores_t > target_scores.unsqueeze(1)).sum(dim=1)

    # Scores exactly equal to target score with strictly lower category code
    codes = torch.arange(category_count, device=scores_t.device).unsqueeze(0)
    equal_lower = ((scores_t == target_scores.unsqueeze(1)) & (codes < targets_t.unsqueeze(1))).sum(
        dim=1
    )

    ranks = 1 + greater + equal_lower
    return ranks.to(torch.int64)


def rank_targets_from_rankings(
    rankings: Tensor | np.ndarray | Sequence[Sequence[int]],
    targets: Tensor | np.ndarray | Sequence[int],
    category_count: int = 588,
) -> Tensor:
    """Compute 1-based target ranks from ordered category permutations [N, category_count].

    Each row must be a strict permutation of integers 0..category_count-1.
    """
    if not isinstance(rankings, Tensor):
        rankings_t = torch.as_tensor(rankings, dtype=torch.int64)
    else:
        rankings_t = rankings.to(torch.int64)

    if rankings_t.ndim != 2:
        raise ValueError(
            f"Rankings must be 2-dimensional [N, C], got shape {list(rankings_t.shape)}"
        )

    num_rows, num_cols = rankings_t.shape
    if num_rows < 1:
        raise ValueError(f"Rankings must contain at least one row, got {num_rows}")

    if num_cols != category_count:
        raise ValueError(
            f"Rankings width {num_cols} does not match category_count {category_count}"
        )

    # Validate full permutation for each row
    sorted_rankings, _ = torch.sort(rankings_t, dim=1)
    expected_row = (
        torch.arange(category_count, device=rankings_t.device).unsqueeze(0).expand(num_rows, -1)
    )
    if not torch.equal(sorted_rankings, expected_row):
        raise ValueError(
            f"Each ranking row must be a strict permutation of 0..{category_count - 1} "
            "with no duplicates or missing categories"
        )

    validated_targets = validate_t1_dense_labels(targets, category_count=category_count)
    if len(validated_targets) != num_rows:
        raise ValueError(
            f"Number of targets ({len(validated_targets)}) does not match rankings rows ({num_rows})"
        )

    targets_t = torch.tensor(validated_targets, dtype=torch.int64, device=rankings_t.device)

    # Position is index where rankings[i, pos] == targets[i]
    matches = rankings_t == targets_t.unsqueeze(1)
    matched_positions = matches.nonzero()

    if matched_positions.shape[0] != num_rows:
        raise ValueError("Each target must appear exactly once in its corresponding ranking row")

    ranks = matched_positions[:, 1] + 1
    return ranks.to(torch.int64)


def history_bucket(count: int | None) -> str | None:
    """Map a non-negative TRAIN event count to its canonical ADR history bucket."""
    if count is None:
        return None

    if isinstance(count, bool):
        raise TypeError(f"History count cannot be boolean: {count!r}")

    try:
        float_count = float(count)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"History count must be a numeric integer, got {count!r}") from exc

    if not math.isfinite(float_count) or not float_count.is_integer():
        raise ValueError(f"History count must be an integer: {count!r}")

    c = int(float_count)
    if c < 0:
        raise ValueError(f"History count must be non-negative, got {c}")

    if 0 <= c <= 9:
        return "BELOW_10_RETAINED_C1"
    if 10 <= c <= 19:
        return "10_19"
    if 20 <= c <= 49:
        return "20_49"
    if 50 <= c <= 99:
        return "50_99"
    return "100_plus"


@dataclasses.dataclass(frozen=True)
class T1EvaluationSummary:
    """Structured summary of T1 ranking evaluation metrics across canonical slices."""

    status: str
    decision_count: int
    client_count: int
    mrr_cutoff: int
    slices: dict[str, dict[str, Any]]
    history_stratification_status: str
    history_buckets: dict[str, dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "decision_count": self.decision_count,
            "client_count": self.client_count,
            "mrr_cutoff": self.mrr_cutoff,
            "slices": self.slices,
            "history_stratification_status": self.history_stratification_status,
        }
        if self.history_buckets is not None:
            result["history_buckets"] = self.history_buckets
        return result


def _compute_slice_metrics(
    indices: list[int],
    ranks: list[int],
    client_ids: list[Any],
    mrr_cutoff: int,
) -> dict[str, Any]:
    """Compute micro and macro Accuracy@1 and MRR@20 for a set of decision indices."""
    decision_count = len(indices)
    if decision_count == 0:
        return {
            "status": "ZERO_SUPPORT",
            "decision_count": 0,
            "client_count": 0,
            "mrr_at_20_micro": None,
            "mrr_at_20_macro": None,
            "accuracy_at_1_micro": None,
            "accuracy_at_1_macro": None,
        }

    slice_ranks = [ranks[i] for i in indices]
    slice_clients = [client_ids[i] for i in indices]

    # Micro contributions
    acc_contribs: list[float] = [1.0 if r == 1 else 0.0 for r in slice_ranks]
    mrr_contribs: list[float] = [1.0 / r if r <= mrr_cutoff else 0.0 for r in slice_ranks]

    mrr_micro = sum(mrr_contribs) / decision_count
    acc_micro = sum(acc_contribs) / decision_count

    # Macro contributions (one vote per client with >=1 decision in slice)
    client_acc_map: dict[Any, list[float]] = defaultdict(list)
    client_mrr_map: dict[Any, list[float]] = defaultdict(list)
    for c_id, a_val, m_val in zip(slice_clients, acc_contribs, mrr_contribs):
        client_acc_map[c_id].append(a_val)
        client_mrr_map[c_id].append(m_val)

    client_count = len(client_acc_map)
    client_acc_means = [sum(vals) / len(vals) for vals in client_acc_map.values()]
    client_mrr_means = [sum(vals) / len(vals) for vals in client_mrr_map.values()]

    mrr_macro = sum(client_mrr_means) / client_count
    acc_macro = sum(client_acc_means) / client_count

    return {
        "status": "AVAILABLE",
        "decision_count": decision_count,
        "client_count": client_count,
        "mrr_at_20_micro": float(mrr_micro),
        "mrr_at_20_macro": float(mrr_macro),
        "accuracy_at_1_micro": float(acc_micro),
        "accuracy_at_1_macro": float(acc_macro),
    }


def evaluate_t1_ranks(
    ranks: Sequence[int] | Tensor | np.ndarray,
    category_changed: Sequence[bool] | Tensor | np.ndarray,
    client_ids: Sequence[Any],
    train_history_counts: Sequence[int | None] | Mapping[Any, int | None] | None = None,
    mrr_cutoff: int = 20,
) -> T1EvaluationSummary:
    """Evaluate 1-based target ranks across overall and next_distinct slices."""
    if isinstance(ranks, (Tensor, np.ndarray)):
        ranks_list = [int(r) for r in ranks.tolist()]
    else:
        ranks_list = [int(r) for r in ranks]

    if isinstance(category_changed, (Tensor, np.ndarray)):
        cat_changed_list = [bool(c) for c in category_changed.tolist()]
    else:
        cat_changed_list = [bool(c) for c in category_changed]

    clients_list = list(client_ids)
    num_decisions = len(ranks_list)

    if len(cat_changed_list) != num_decisions:
        raise ValueError(
            f"Length mismatch: ranks ({num_decisions}) vs category_changed ({len(cat_changed_list)})"
        )
    if len(clients_list) != num_decisions:
        raise ValueError(
            f"Length mismatch: ranks ({num_decisions}) vs client_ids ({len(clients_list)})"
        )

    for idx, r in enumerate(ranks_list):
        if r < 1:
            raise ValueError(f"Rank at index {idx} must be >= 1, got {r}")

    # Process train_history_counts if provided
    history_buckets_by_decision: list[str | None] = []
    history_available = False

    if train_history_counts is not None:
        client_history_verified: dict[Any, int | None] = {}
        if isinstance(train_history_counts, Mapping):
            for c_id, raw_cnt in train_history_counts.items():
                if raw_cnt is not None:
                    _ = history_bucket(raw_cnt)  # validates count
                client_history_verified[c_id] = raw_cnt
            for c_id in clients_list:
                c_cnt = client_history_verified.get(c_id)
                history_buckets_by_decision.append(history_bucket(c_cnt))
                if c_cnt is not None:
                    history_available = True
        else:
            hist_seq = list(train_history_counts)
            if len(hist_seq) != num_decisions:
                raise ValueError(
                    f"Length mismatch: ranks ({num_decisions}) vs train_history_counts ({len(hist_seq)})"
                )
            for c_id, raw_cnt in zip(clients_list, hist_seq):
                if c_id in client_history_verified:
                    if client_history_verified[c_id] != raw_cnt:
                        raise ValueError(
                            f"Inconsistent train_history_count for client {c_id!r}: "
                            f"{client_history_verified[c_id]!r} vs {raw_cnt!r}"
                        )
                else:
                    if raw_cnt is not None:
                        _ = history_bucket(raw_cnt)  # validates count
                    client_history_verified[c_id] = raw_cnt

                history_buckets_by_decision.append(history_bucket(raw_cnt))
                if raw_cnt is not None:
                    history_available = True

    overall_indices = list(range(num_decisions))
    next_distinct_indices = [i for i in range(num_decisions) if cat_changed_list[i]]

    slices: dict[str, dict[str, Any]] = {
        "overall": _compute_slice_metrics(overall_indices, ranks_list, clients_list, mrr_cutoff),
        "next_distinct": _compute_slice_metrics(
            next_distinct_indices, ranks_list, clients_list, mrr_cutoff
        ),
    }

    bucket_results: dict[str, dict[str, Any]] | None = None
    if history_available:
        bucket_results = {}
        for b_id in _VALID_HISTORY_BUCKET_IDS:
            b_indices = [i for i, b in enumerate(history_buckets_by_decision) if b == b_id]
            b_distinct_indices = [i for i in b_indices if cat_changed_list[i]]
            bucket_results[b_id] = {
                "overall": _compute_slice_metrics(b_indices, ranks_list, clients_list, mrr_cutoff),
                "next_distinct": _compute_slice_metrics(
                    b_distinct_indices, ranks_list, clients_list, mrr_cutoff
                ),
            }

    unique_total_clients = len(set(clients_list))
    return T1EvaluationSummary(
        status="PASS",
        decision_count=num_decisions,
        client_count=unique_total_clients,
        mrr_cutoff=mrr_cutoff,
        slices=slices,
        history_stratification_status="AVAILABLE" if history_available else "NOT_AVAILABLE",
        history_buckets=bucket_results,
    )


def metric_records_from_t1_summary(
    summary: T1EvaluationSummary | dict[str, Any],
    cohort: str = "C1",
    task: str = "T1",
) -> list[dict[str, Any]]:
    """Convert T1 evaluation summary to canonical metric_record_v1 records.

    Emits only metrics for slices that have non-zero support, and strictly validates
    each record with validate_metric_record.
    """
    summary_dict = summary.to_dict() if isinstance(summary, T1EvaluationSummary) else summary
    slices = summary_dict.get("slices", {})
    records: list[dict[str, Any]] = []

    # Canonical order: next_distinct (headline) first, then overall (diagnostic)
    for slice_name in ("next_distinct", "overall"):
        slice_info = slices.get(slice_name)
        if not slice_info or slice_info.get("status") != "AVAILABLE":
            continue

        decision_count = slice_info["decision_count"]
        client_count = slice_info["client_count"]

        # MRR@20 Macro
        rec_mrr_macro = {
            "schema": "metric_record_v1",
            "version": "1",
            "metric_id": f"t1.{slice_name}.mrr_at_20.macro",
            "task": task,
            "cohort": cohort,
            "value": float(slice_info["mrr_at_20_macro"]),
            "direction": "MAXIMIZE",
            "unit": "FRACTION",
            "support": int(client_count),
        }
        validate_metric_record(rec_mrr_macro)
        records.append(rec_mrr_macro)

        # MRR@20 Micro
        rec_mrr_micro = {
            "schema": "metric_record_v1",
            "version": "1",
            "metric_id": f"t1.{slice_name}.mrr_at_20.micro",
            "task": task,
            "cohort": cohort,
            "value": float(slice_info["mrr_at_20_micro"]),
            "direction": "MAXIMIZE",
            "unit": "FRACTION",
            "support": int(decision_count),
        }
        validate_metric_record(rec_mrr_micro)
        records.append(rec_mrr_micro)

        # Accuracy@1 Macro
        rec_acc_macro = {
            "schema": "metric_record_v1",
            "version": "1",
            "metric_id": f"t1.{slice_name}.accuracy_at_1.macro",
            "task": task,
            "cohort": cohort,
            "value": float(slice_info["accuracy_at_1_macro"]),
            "direction": "MAXIMIZE",
            "unit": "FRACTION",
            "support": int(client_count),
        }
        validate_metric_record(rec_acc_macro)
        records.append(rec_acc_macro)

        # Accuracy@1 Micro
        rec_acc_micro = {
            "schema": "metric_record_v1",
            "version": "1",
            "metric_id": f"t1.{slice_name}.accuracy_at_1.micro",
            "task": task,
            "cohort": cohort,
            "value": float(slice_info["accuracy_at_1_micro"]),
            "direction": "MAXIMIZE",
            "unit": "FRACTION",
            "support": int(decision_count),
        }
        validate_metric_record(rec_acc_micro)
        records.append(rec_acc_micro)

    return records


_REQUIRED_TASK_EXAMPLE_COLUMNS = (
    "client",
    "session",
    "decision_order",
    "current_category",
    "label_value",
    "category_changed",
    "task_mask",
    "status",
    "cohort",
    "split",
)


def preflight_t1_task_examples(
    parquet_path: Path | str,
    category_count: int = 588,
) -> dict[str, Any]:
    """Validate frozen T1 TaskExample parquet schema and return public-safe aggregates.

    Fails if required columns are missing, contain nulls, or violate C1/VALIDATION contracts.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"TaskExample file not found: {path}")

    df = pl.read_parquet(path)

    missing_cols = [col for col in _REQUIRED_TASK_EXAMPLE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"TaskExample file {path} missing required columns: {missing_cols}")

    # Fail on null before checking equality
    for col in _REQUIRED_TASK_EXAMPLE_COLUMNS:
        null_cnt = df[col].null_count()
        if null_cnt > 0:
            raise ValueError(f"Column {col!r} contains {null_cnt} null values in {path}")

    if (df["split"] != "VALIDATION").any():
        raise ValueError(f"Found non-VALIDATION split values in {path}")

    if (df["cohort"] != "C1").any():
        raise ValueError(f"Found non-C1 cohort values in {path}")

    if (~df["task_mask"]).any():
        raise ValueError(f"Found non-True task_mask values in {path}")

    if (df["status"] != "OBSERVED").any():
        raise ValueError(f"Found non-OBSERVED status values in {path}")

    if df["category_changed"].dtype != pl.Boolean:
        raise ValueError(f"category_changed must be boolean, got {df['category_changed'].dtype}")

    # Validate label values
    labels = df["label_value"]
    if (labels.is_nan() | labels.is_infinite()).any():
        raise ValueError("label_value contains NaN or Infinite values")

    # Integral check
    cast_float = labels.cast(pl.Float64)
    if ((cast_float - cast_float.floor()) != 0.0).any():
        raise ValueError("label_value contains non-integral float values")

    cast_int = cast_float.cast(pl.Int64)
    min_label = int(cast_int.min())  # type: ignore[arg-type]
    max_label = int(cast_int.max())  # type: ignore[arg-type]

    if min_label < 0 or max_label >= category_count:
        raise ValueError(
            f"label_value codes [{min_label}, {max_label}] out of bounds [0, {category_count})"
        )

    row_count = len(df)
    unique_clients = df["client"].n_unique()
    cat_changed_count = int(df["category_changed"].sum())

    return {
        "file_path": str(path),
        "file_sha256": file_sha256(path),
        "row_count": row_count,
        "unique_client_count": unique_clients,
        "category_changed_row_count": cat_changed_count,
        "label_min": min_label,
        "label_max": max_label,
        "all_labels_integral_in_range": True,
        "all_rows_valid_contract": True,
        "category_count": category_count,
    }

"""Result rows and the comparison key.

A row carries the result fields (value, populations, cost and deployment measurements) plus the identity of
everything that can change a number: topology, objective, precision, evaluator code, bootstrap matrix and the
evaluation numerics. The comparison key must match within a block; a mismatched row is a HISTORICAL_REFERENCE and is
never labelled a same-conditions effect. Intended differences (method, training paradigm, local personalisation,
optimiser lifecycle, seed) are recorded, never compared.
Method codes: C central, FA FedAvg, FP FedProx, PF personalised FedAvg, CPF central model with on-device
personalisation, LO local-only.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from .errors import RowKeyError
from .values import Undefined, to_jsonable

RESULT_FIELDS = ("status", "family", "method", "seed", "initialization_block", "data_protocol_hash",
                 "population_hash", "candidate_order_hash", "feature_contract_hash", "query_order_hash",
                 "label_policy_hash", "split", "aggregation", "metric", "k", "target_denominator", "value",
                 "n_decisions", "n_users", "oov_count", "coverage", "checkpoint_id", "selection_rule", "exposures",
                 "efe", "rounds", "local_steps", "up_bytes", "down_bytes", "private_state_bytes", "wall_seconds",
                 "gpu_hours", "peak_ram_bytes", "peak_vram_bytes", "device_name", "inference_p50_ms",
                 "inference_p95_ms", "matched_central_row_id", "retained_percent", "relative_loss_percent", "ci",
                 "evidence_paths", "previously_seen_evaluation_period")
ADDED_FIELDS = ("topology_hash", "objective_id", "precision_id", "evaluator_code_sha256", "bootstrap_W_sha256",
                "eval_config", "value_status", "eval_score_block", "eval_batch", "eval_device_class",
                "n_target_tied", "value_pessimistic")
COMPARISON_KEY = ("data_protocol_hash", "feature_contract_hash", "split", "label_policy_hash", "population_hash",
                  "query_order_hash", "candidate_order_hash", "initialization_block", "topology_hash",
                  "objective_id", "precision_id", "evaluator_code_sha256", "eval_score_block", "eval_batch",
                  "eval_device_class", "eval_stream", "metric", "k", "aggregation", "target_denominator")
# make_row requires every key field except the evaluation-numerics additions (rows_from_result always sets those from
# the result's eval_config; scripted rows may add them afterwards). comparison_status compares ALL of them: float32
# block scores are not bit-identical across block widths or batch shapes (ranking.py), so rows that differ in any of
# them are not same-conditions rows.
EVAL_KEY_ADDITIONS = ("eval_score_block", "eval_batch", "eval_device_class", "eval_stream")
INTENDED_DIFFERENCES = ("method", "training_paradigm", "local_personalization", "optimizer_lifecycle", "seed")
SELECTION_RULES = ("PRACTICAL_BEST", "CONTROLLED_BUDGET", "PRACTICAL_BEST_EXTENDED")
METHODS = ("C", "FA", "FP", "PF", "CPF", "LO")
AGGREGATIONS = ("micro", "macro")
DENOMINATORS = ("E2E", "RANKABLE")


def make_row(key: Mapping, *, method: str, seed: int | None, value, n_decisions: int, n_users: int,
             oov_count: int, coverage, selection_rule: str | None = None, efe: float | None = None,
             extra: Mapping | None = None) -> dict:
    missing = [f for f in COMPARISON_KEY if f not in EVAL_KEY_ADDITIONS and key.get(f) in (None, "")]
    if missing:
        raise RowKeyError(f"result row lacks comparison-key fields {missing}")
    if key["aggregation"] not in AGGREGATIONS or key["target_denominator"] not in DENOMINATORS:
        raise RowKeyError("aggregation must be micro|macro and target_denominator E2E|RANKABLE")
    if selection_rule is not None and selection_rule not in SELECTION_RULES:
        raise RowKeyError(f"selection_rule must be one of {SELECTION_RULES}")
    row = {f: None for f in RESULT_FIELDS + ADDED_FIELDS}
    row.update({f: key[f] for f in COMPARISON_KEY})
    for f in ("family", "eval_config", "bootstrap_W_sha256", "training_paradigm", "local_personalization",
              "optimizer_lifecycle", "eval_score_block", "eval_batch", "eval_device_class", "eval_stream"):
        if f in key:
            row[f] = key[f]
    row.update(status="MEASURED", method=method, seed=seed, n_decisions=int(n_decisions), n_users=int(n_users),
               oov_count=int(oov_count), selection_rule=selection_rule, efe=efe, evidence_paths=[],
               previously_seen_evaluation_period=True)
    if isinstance(value, Undefined):
        row["value"], row["value_status"] = None, f"UNDEFINED:{value.reason}"
    else:
        row["value"], row["value_status"] = float(value), "OK"
    row["coverage"] = to_jsonable(coverage)
    if extra:
        for k, v in extra.items():
            if k in COMPARISON_KEY:
                raise RowKeyError(f"extra may not override comparison-key field {k}")
            row[k] = v
    return row


def rows_from_result(result: Mapping, key_base: Mapping, *, method: str, seed: int | None,
                     selection_rule: str | None = None, efe: float | None = None) -> list[dict]:
    """The full grid of one evaluate() result as rows (query_order_hash = eval_ordered hash).

    The evaluation-numerics key additions come from the result's eval_config; every row carries the tie diagnostic
    (n_target_tied and the value under pessimistic ranks). Rows of trained models (METHODS) must come from the
    adapter path. A row on a device other than the config's reference device class is labelled
    MEASURED_OTHER_DEVICE."""
    cfg = result.get("eval_config") or {}
    if not cfg.get("frozen"):
        raise RowKeyError("result rows need a result produced under a fixed EvalConfig")
    if method in METHODS and cfg.get("path") != "adapter_stream":
        raise RowKeyError(f"{method} rows must come from the adapter path, not {cfg.get('path')!r}")
    base = dict(key_base)
    base["query_order_hash"] = result["hashes"]["eval_ordered_decision_id_sha256"]
    base["evaluator_code_sha256"] = result.get("evaluator_code_sha256") or base.get("evaluator_code_sha256")
    base["eval_config"] = cfg
    base["eval_score_block"] = int(cfg["score_block"])
    base["eval_batch"] = int(cfg["eval_batch"])
    base["eval_device_class"] = str(cfg["eval_device_class"])
    base["eval_stream"] = str(cfg.get("eval_stream") or "")
    if base["eval_stream"] not in ("ALL_ROWS", "E2E_ROWS"):
        raise RowKeyError("the result carries no eval_stream (ALL_ROWS / E2E_ROWS)")
    ref = cfg.get("reference_device_class")
    status = "MEASURED" if ref is None or cfg["eval_device_class"] == ref else "MEASURED_OTHER_DEVICE"
    ties = result["tie_diagnostic"]
    rows = []
    for den in DENOMINATORS:
        blk = result[den]
        pess = ties["pessimistic"][den]
        for agg in AGGREGATIONS:
            for name, v in blk[agg].items():
                metric, k = name.split("@")
                key = dict(base, aggregation=agg, target_denominator=den, metric=metric, k=int(k))
                pv = pess[agg][name]
                rows.append(make_row(key, method=method, seed=seed, value=v, n_decisions=blk["n_decisions"],
                                     n_users=blk["n_users"], oov_count=result["oov"]["oov_count"],
                                     coverage=result["coverage"], selection_rule=selection_rule, efe=efe,
                                     extra={"status": status, "n_target_tied": int(ties["n_target_tied"]),
                                            "value_pessimistic": to_jsonable(pv)}))
    return rows


def comparison_status(a: Mapping, b: Mapping) -> tuple[str, list[str]]:
    diff = [f for f in COMPARISON_KEY if a.get(f) != b.get(f)]
    return ("SAME_CONDITIONS", []) if not diff else ("HISTORICAL_REFERENCE", diff)


def check_block(rows: Iterable[Mapping]) -> None:
    """Refuse a block whose rows do not share one comparison key (apart from metric/k/aggregation/denominator)."""
    rows = list(rows)
    fixed = [f for f in COMPARISON_KEY if f not in ("metric", "k", "aggregation", "target_denominator")]
    for r in rows[1:]:
        diff = [f for f in fixed if r.get(f) != rows[0].get(f)]
        if diff:
            raise RowKeyError(f"rows of one comparison block differ in {diff}: HISTORICAL_REFERENCE only")

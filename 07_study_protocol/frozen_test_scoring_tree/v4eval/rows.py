"""v4eval.rows — result rows and the comparison key (EVALUATION_CONTRACT_V4 §9, E9.1-E9.3; E12.2).

A row carries every RESULT_ROW.template.json field plus the V4 additions (topology_hash, objective_id, precision_id,
evaluator_code_sha256, bootstrap_W_sha256, eval_config). The comparison key must match within a block; a mismatched
row is a HISTORICAL_REFERENCE and is never labelled a same-conditions paradigm effect. Intended differences
(method, training paradigm, local personalization, optimizer lifecycle, seed) are recorded, never compared.
Every V4 row has previously_seen_evaluation_period = true (E9.2).
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Tuple

from .errors import RowKeyError
from .values import Undefined, to_jsonable

TEMPLATE_FIELDS = ("status", "family", "method", "seed", "initialization_block", "data_protocol_hash", "population_hash",
                   "candidate_order_hash", "feature_contract_hash", "query_order_hash", "label_policy_hash", "split",
                   "aggregation", "metric", "k", "target_denominator", "value", "n_decisions", "n_users", "oov_count",
                   "coverage", "checkpoint_id", "selection_rule", "exposures", "efe", "rounds", "local_steps",
                   "up_bytes", "down_bytes", "private_state_bytes", "wall_seconds", "gpu_hours", "peak_ram_bytes",
                   "peak_vram_bytes", "device_name", "inference_p50_ms", "inference_p95_ms", "matched_central_row_id",
                   "retained_percent", "relative_loss_percent", "ci", "evidence_paths", "previously_seen_evaluation_period")
ADDED_FIELDS = ("topology_hash", "objective_id", "precision_id", "evaluator_code_sha256", "bootstrap_W_sha256",
                "eval_config", "value_status", "eval_score_block", "eval_batch", "eval_device_class",
                "eval_batch_amendment_id", "n_target_tied", "value_pessimistic")
COMPARISON_KEY = ("data_protocol_hash", "feature_contract_hash", "split", "label_policy_hash", "population_hash",
                  "query_order_hash", "candidate_order_hash", "initialization_block", "topology_hash", "objective_id",
                  "precision_id", "evaluator_code_sha256", "eval_score_block", "eval_batch", "eval_device_class",
                  "eval_stream", "metric", "k", "aggregation", "target_denominator")
T3_KEY_ADDITIONS = ("eval_score_block", "eval_batch", "eval_device_class", "eval_stream")
# make_row requires every key field except the T3 additions (rows_from_result always sets those from the frozen
# eval_config; scripted rows may add them afterwards). comparison_status compares ALL of them.
# T3 comparison_key_additions (contracts/PREG1_LOCK_VALUES.json): eval_score_block, eval_batch, eval_device_class.
# float32 block scores are not bit-identical across block widths or batch shapes (ranking.py), so rows that differ in
# any of the three are not same-conditions rows.
INTENDED_DIFFERENCES = ("method", "training_paradigm", "local_personalization", "optimizer_lifecycle", "seed")
SELECTION_RULES = ("PRACTICAL_BEST", "CONTROLLED_BUDGET", "PRACTICAL_BEST_EXTENDED")
METHODS = ("C", "FA", "FP", "PF", "CPF", "LO")
AGGREGATIONS = ("micro", "macro")
DENOMINATORS = ("E2E", "RANKABLE")


def make_row(key: Mapping, *, method: str, seed: Optional[int], value, n_decisions: int, n_users: int,
             oov_count: int, coverage, selection_rule: Optional[str] = None, efe: Optional[float] = None,
             extra: Optional[Mapping] = None) -> dict:
    missing = [f for f in COMPARISON_KEY if f not in T3_KEY_ADDITIONS and key.get(f) in (None, "")]
    if missing:
        raise RowKeyError(f"result row lacks comparison-key fields {missing}")
    if key["aggregation"] not in AGGREGATIONS or key["target_denominator"] not in DENOMINATORS:
        raise RowKeyError("aggregation must be micro|macro and target_denominator E2E|RANKABLE")
    if selection_rule is not None and selection_rule not in SELECTION_RULES:
        raise RowKeyError(f"selection_rule must be one of {SELECTION_RULES}")
    row = {f: None for f in TEMPLATE_FIELDS + ADDED_FIELDS}
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


def rows_from_result(result: Mapping, key_base: Mapping, *, method: str, seed: Optional[int],
                     selection_rule: Optional[str] = None, efe: Optional[float] = None) -> List[dict]:
    """The full registered grid of one evaluate() result as rows (query_order_hash = eval_ordered hash).

    The T3 comparison-key additions come from the result's frozen eval_config; every row carries the tie diagnostic
    (n_target_tied and the value under pessimistic ranks). Model rows of C / FA / FP / PF / CPF / LO must come from the
    frozen adapter path. A row on a device other than the frozen class is labelled REHEARSAL_NON_FROZEN_DEVICE."""
    cfg = result.get("eval_config") or {}
    if not cfg.get("frozen"):
        raise RowKeyError("result rows need a result produced under the frozen T3 configuration")
    if method in METHODS and cfg.get("path") != "adapter_stream":
        raise RowKeyError(f"{method} rows must come from the frozen adapter path, not {cfg.get('path')!r}")
    base = dict(key_base)
    base["query_order_hash"] = result["hashes"]["eval_ordered_decision_id_sha256"]
    base["evaluator_code_sha256"] = result.get("evaluator_code_sha256") or base.get("evaluator_code_sha256")
    base["eval_config"] = cfg
    base["eval_score_block"] = int(cfg["score_block"])
    base["eval_batch"] = int(cfg["eval_batch"])
    base["eval_device_class"] = str(cfg["eval_device_class"])
    base["eval_stream"] = str(cfg.get("eval_stream") or "")
    if base["eval_stream"] not in ("ALL_ROWS", "E2E_ROWS"):
        raise RowKeyError("the result carries no registered eval_stream (ALL_ROWS / E2E_ROWS)")
    status = ("MEASURED" if cfg["eval_device_class"] == cfg.get("device_class_frozen") else
              "MEASURED_PINNED_DEVICE" if cfg.get("device_pin_sha256") else "REHEARSAL_NON_FROZEN_DEVICE")
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
                                     extra={"status": status, "eval_batch_amendment_id": cfg.get("eval_batch_amendment_id"),
                                            **({"device_pin_sha256": cfg["device_pin_sha256"]} if cfg.get("device_pin_sha256") else {}),
                                            "n_target_tied": int(ties["n_target_tied"]),
                                            "value_pessimistic": to_jsonable(pv)}))
    return rows


def comparison_status(a: Mapping, b: Mapping) -> Tuple[str, List[str]]:
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

"""Schema-checked lookups and the claim vocabulary.

An automatic report looks up actual schema keys; a missing bootstrap becomes MISSING_EVIDENCE, never a fabricated
success or failure. `lookup` validates every path against RESULT_SCHEMA: an unknown key raises SchemaKeyError (a typo
can never read a default); a known but absent optional block returns MISSING_EVIDENCE.
Claims: "A exceeds B" iff the CI lower bound > 0; "B exceeds A" iff the upper bound < 0; else no detectable difference.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .errors import SchemaKeyError
from .metrics import metric_names
from .values import MISSING_EVIDENCE, Undefined

_METRIC = {n: None for n in metric_names()}
_POP = {"n_decisions": None, "n_users": None, "micro": _METRIC, "macro": _METRIC}
_BLOCKS = {"E2E": _POP, "RANKABLE": _POP, "coverage": None}
_CONTRAST = {"name": None, "kind": None, "statistic": None, "a": None, "b": None, "point": None, "value_a": None,
             "value_b": None, "ci95": None, "n_users": None, "n_decisions": None, "n_resamples": None, "seed": None,
             "W_sha256": None, "user_set_sha256": None, "n_resamples_zero_denominator": None, "seeds": None,
             "per_seed": None}
RESULT_SCHEMA = {
    "manifest_id": None, "catalogue_K": None, "n_rows": None, "hashes": {"ordered_decision_id_sha256": None,
    "eval_ordered_decision_id_sha256": None, "rankable_ordered_decision_id_sha256": None}, "model_sha256": None,
    "eval_config": None, "evaluator_code_sha256": None, "k_grid": None, **_BLOCKS,
    "oov": {"oov_count": None, "oov_rate": None, "by_cause": {"BELOW_SUPPORT": None, "ABSENT_FROM_TRAIN": None}},
    "censoring": {"counts": None, "n_censored": None, "censored_share": None},
    "tie_diagnostic": {"n_ranked": None, "n_target_tied": None, "max_n_tied": None, "pessimistic": _BLOCKS},
    "strata": {"*": {"*": _BLOCKS}},
    "identity_checks": {"*": {"micro": None, "macro": None}},
    "bootstrap": {"*": _CONTRAST},
}
OPTIONAL_BLOCKS = ("bootstrap", "strata", "oov.by_cause")


def lookup(result: Mapping, path: Sequence[str]) -> Any:
    schema: Any = RESULT_SCHEMA
    node: Any = result
    walked = []
    for key in path:
        if not isinstance(schema, dict):
            raise SchemaKeyError(f"{'.'.join(walked)} is a leaf; cannot look up {key!r}")
        if key in schema:
            schema = schema[key]
        elif "*" in schema:
            schema = schema["*"]
        else:
            raise SchemaKeyError(f"{'.'.join(walked + [key])} is not a key of the result schema")
        walked.append(key)
        if not isinstance(node, Mapping) or key not in node:
            return MISSING_EVIDENCE
        node = node[key]
    return node


def claim(contrast: Any) -> str:
    if contrast is None or contrast == MISSING_EVIDENCE or not isinstance(contrast, Mapping):
        return MISSING_EVIDENCE
    ci = contrast.get("ci95", MISSING_EVIDENCE)
    if ci == MISSING_EVIDENCE or ci is None:
        return MISSING_EVIDENCE
    if isinstance(ci, Undefined):
        return "UNDEFINED"
    lo, hi = float(ci[0]), float(ci[1])
    if lo > 0:
        return "A_EXCEEDS_B"
    if hi < 0:
        return "B_EXCEEDS_A"
    return "NO_DETECTABLE_DIFFERENCE"


def claim_for(result: Mapping, contrast_name: str) -> str:
    return claim(lookup(result, ("bootstrap", contrast_name)))

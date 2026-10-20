"""v4base.scoring — the ONE frozen T3 evaluator path every baseline goes through, and BASELINE-labelled result rows.

`v4eval.predict.predict_scores` (its own docstring: "Precomputed full-K score batches ... Model rows of C / FA / FP /
PF / CPF / LO must come from predict_adapter; rows.py refuses this path for them") is exactly the externally-produced
-score path BASE-01 needs: a baseline is not one of those six adapter methods, so `rows_from_result` never invokes
the adapter-path check for it, and the SAME frozen `evaluate()` (identity checks, tie diagnostic, strata, the full
registered metric grid) runs bit-for-bit as for a trained model's row. No v4eval file is read for anything other
than its documented public functions; nothing here is monkeypatched. This resolves the task packet's "if the frozen
evaluator cannot take externally produced scores without editing v4eval, STOP" clause: it can, so BASE-01 is not
BLOCKED on that account.

Sentinel policy across the three baselines (see pop.py/last.py/sknn.py for the per-method rationale): POP uses raw
int64 counts (a zero-copy broadcast row), LAST uses int64 recency with sentinel -1, SKNN uses float32 similarity
sums with the standard 0.0 floor. None of the three ever puts a non-finite float into a score tensor, so
`v4eval.ranking._refuse_nonfinite` never fires for a correct baseline run (a real bug that DID produce inf/NaN would
still be caught there, which is the point of not working around it).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Tuple

import torch

from v4eval import evaluator_code_sha256 as _v4eval_code_sha256
from v4eval import frozen_config as fc
from v4eval.evaluate import evaluate
from v4eval.hashing import code_sha256
from v4eval.predict import predict_scores
from v4eval.rows import rows_from_result

HERE = Path(__file__).resolve().parent

BASELINE_KEY_PLACEHOLDERS = {
    # v4eval.rows.COMPARISON_KEY fields that only mean something for a trained adapter (C/FA/FP/PF/CPF/LO); a
    # baseline has none of these, so it carries a fixed, clearly-named placeholder instead of a fabricated hash.
    "topology_hash": "BASELINE_NO_TOPOLOGY", "objective_id": "BASELINE_NO_OBJECTIVE",
    "precision_id": "BASELINE_NO_PRECISION", "initialization_block": "BASELINE_NO_INITIALIZATION",
}


def v4base_code_sha256() -> str:
    """sha256 of every `v4base/*.py` file (name + bytes, sorted) — this baseline's own "model" identity
    (`model_sha256` on the row/receipt), via the SAME `v4eval.hashing.code_sha256` rule `evaluator_code_sha256()`
    uses for v4eval, not a reinvented one."""
    return code_sha256(sorted(HERE.glob("*.py")))


def evaluator_code_sha256() -> str:
    return _v4eval_code_sha256()


def score_batches(scorer, batches: Iterable[Mapping]) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Adapts an EvalProvider.batches(bs) stream (raw Clean V2 batch dicts) into predict_scores' own
    (decision_id, scores[B, K]) tuple stream."""
    for b in batches:
        did = b["decision_id"]
        did = did if torch.is_tensor(did) else torch.as_tensor(did)
        yield did, scorer.score_batch(b)


def run_baseline_evaluation(scorer, manifest, batches: Iterable[Mapping], *, model_sha256: str,
                            config: Optional[fc.FrozenEvalConfig] = None) -> dict:
    """batches must be the manifest's OWN evaluation stream (`EvalProvider.batches(cfg.eval_batch)`), in manifest
    order, exactly `cfg.eval_batch` rows per batch except an unpadded tail — `predict_scores`' cursor enforces this
    (T3), so a caller cannot silently feed a resampled or reordered stream."""
    cfg = fc.require_frozen(config) if config is not None else fc.load()
    fc.apply_runtime_flags(cfg)      # the frozen T3 determinism flags (CPU: float32 matmul precision, det. algos)
    art = predict_scores(score_batches(scorer, batches), manifest, model_sha256=model_sha256, config=cfg)
    return evaluate(manifest, art, config=cfg, evaluator_code_sha256=evaluator_code_sha256())


def build_rows(result: dict, key_base: Mapping, *, method: str) -> list:
    """BASELINE-labelled rows (Amendment 02 §4: "label BASELINE; never used for selection"). `v4train.recipe.RUN_LABELS`
    has no BASELINE value and is never edited to add one (BASE-01 does not touch v4train); `run_label` here is a
    plain descriptive field these rows carry for Ops/Fable, never fed to `v4train.selection.refuse_run_labels` (that
    gate is COHORT/TIEBREAK/CALIBRATION/C_MATCHED trainer evidence only, and BASELINE rows are never passed to it)."""
    base = dict(BASELINE_KEY_PLACEHOLDERS)
    base.update(key_base)
    base.setdefault("family", f"BASELINE_{method}")
    rows = rows_from_result(result, base, method=method, seed=None, selection_rule=None)
    for r in rows:
        r["run_label"] = "BASELINE"
    return rows


__all__ = ["BASELINE_KEY_PLACEHOLDERS", "v4base_code_sha256", "evaluator_code_sha256", "score_batches",
           "run_baseline_evaluation", "build_rows"]

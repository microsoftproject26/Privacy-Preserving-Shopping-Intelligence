"""v4eval.evaluate — one aligned manifest + prediction artifact -> the full registered result (contract §3-§6).

Populations (§3.1): E2E = eval_mask rows (OOV rows score 0); RANKABLE = eval_mask AND target_class >= 0, identical for
every method on the manifest (T35, proven by rankable_ordered_decision_id_sha256). Censored rows are counted per class
and never enter a denominator (E2.5). Strata come only from allowed manifest fields (E3.5). The grid is
{micro, macro} x {mrr, hr (= recall), ndcg} x k in {1, 5, 10, 20} x {E2E, RANKABLE} (§5.3). Both coverage identities
are recomputed on every result and must hold within 1e-9 (E4.6), otherwise IdentityError.
evaluate() requires the frozen T3 config (frozen_config.load()) and an artifact produced under exactly that config;
the per-row tie diagnostic (T3) is always reported: n_target_tied and the pessimistic-rank grid.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .errors import IdentityError
from .frozen_config import FrozenEvalConfig, check_artifact_config, resolve
from .manifest import CENSOR_CLASSES, EvalManifest, stratum_masks
from .metrics import (IDENTITY_TOL, K_GRID, credits_from_ranks, coverage, identity_errors, identity_terms,
                      population_block)
from .predict import PredictionArtifact, align
from .values import Undefined

DEFAULT_STRATA = ("cold_stratum", "subperiod", "panel_label")


def _blocks(credits: dict, mask: np.ndarray, rank_mask: np.ndarray, users: np.ndarray) -> dict:
    return {"E2E": population_block(credits, mask, users),
            "RANKABLE": population_block(credits, mask & rank_mask, users),
            "coverage": coverage(mask, mask & rank_mask)}


def evaluate(manifest: EvalManifest, art: PredictionArtifact, *, config: Optional[FrozenEvalConfig] = None,
             ks: Iterable[int] = K_GRID, strata: Optional[Iterable[str]] = None,
             evaluator_code_sha256: Optional[str] = None) -> dict:
    cfg = resolve(config)                          # defaults to frozen_config.load()
    check_artifact_config(cfg, art.header.get("eval_config"))
    align(manifest, art)
    ks = tuple(ks)
    credits = credits_from_ranks(art.ranks, ks)
    E = manifest.eval_mask
    R = manifest.rankable
    users = manifest.eval_user_key
    res = {"manifest_id": manifest.manifest_id, "catalogue_K": manifest.catalogue_K, "n_rows": manifest.n,
           "hashes": dict(manifest.hashes), "model_sha256": art.header.get("model_sha256"),
           "eval_config": art.header.get("eval_config"), "evaluator_code_sha256": evaluator_code_sha256,
           "k_grid": list(ks)}
    res.update(_blocks(credits, E, R, users))
    n_e = int(E.sum())
    n_oov = int(manifest.target_oov.sum())
    res["oov"] = {"oov_count": n_oov, "oov_rate": (Undefined("EMPTY_POPULATION") if n_e == 0 else n_oov / n_e)}
    if "oov_cause" in manifest.fields:
        oc = manifest.fields["oov_cause"]
        res["oov"]["by_cause"] = {c: int(np.count_nonzero(E & (oc == c))) for c in ("BELOW_SUPPORT", "ABSENT_FROM_TRAIN")}
    counts = {c: int(np.count_nonzero(manifest.censor_class == c)) for c in CENSOR_CLASSES if c != "NONE"}
    n_cen = int(sum(counts.values()))
    res["censoring"] = {"counts": counts, "n_censored": n_cen,
                        "censored_share": (Undefined("EMPTY_POPULATION") if manifest.n == 0 else n_cen / manifest.n)}
    tied = art.n_tied[R]
    res["tie_diagnostic"] = {"n_ranked": int(R.sum()), "n_target_tied": int(np.count_nonzero(tied > 0)),
                              "max_n_tied": int(tied.max()) if tied.size else 0,
                              "pessimistic": _blocks(credits_from_ranks(art.pessimistic_ranks, ks), E, R, users)}
    fields = DEFAULT_STRATA if strata is None else tuple(strata)
    res["strata"] = {}
    for f in fields:
        if strata is None and f not in manifest.fields:
            continue
        res["strata"][f] = {v: _blocks(credits, E & m, R, users) for v, m in stratum_masks(manifest, f).items()}
    checks = {}
    for name in credits:
        err = identity_errors(identity_terms(credits[name], E, R, users))
        if not isinstance(err["micro"], Undefined) and (err["micro"] > IDENTITY_TOL or err["macro"] > IDENTITY_TOL):
            raise IdentityError(f"{name}: micro err {err['micro']}, macro err {err['macro']}")
        checks[name] = err
    res["identity_checks"] = checks
    return res

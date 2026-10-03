"""Shared helpers for the full-catalogue evaluator tests (synthetic fixtures only; no data rows are read)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from ppsi.evaluation.fullrank.config import EvalConfig
from ppsi.evaluation.fullrank.evaluate import evaluate
from ppsi.evaluation.fullrank.manifest import build_eval_manifest
from ppsi.evaluation.fullrank.predict import predict_scores

K = 24


def nc(reason: str):
    """Negative control: the wrapped key-property check must raise AssertionError (strict xfail)."""
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason="NEGATIVE CONTROL: " + reason)


def assert_raises(exc, fn, *a, **kw):
    """Like pytest.raises, but a missing exception is an AssertionError (so negative controls can use it)."""
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc}")


def hexkey(u: int) -> str:
    return hashlib.sha256(f"synthetic-user-{u}".encode()).hexdigest()


def columns(rows) -> dict:
    """rows: list of (eval_user_key, kind, target_class) with kind in {'R' rankable, 'O' OOV, 'C' censored}."""
    n = len(rows)
    cols = {"decision_id": np.arange(n, dtype=np.int64) * 7 + 1000,
            "eval_user_key": np.array([r[0] for r in rows], dtype=np.int64),
            "eval_mask": np.array([r[1] != "C" for r in rows]),
            "target_class": np.array([r[2] if r[1] == "R" else -1 for r in rows], dtype=np.int64),
            "target_oov": np.array([r[1] == "O" for r in rows]),
            "censor_class": np.array(["NONE" if r[1] != "C" else "CEN_NO_NEXT" for r in rows])}
    return cols


def manifest_from_rows(rows, *, K_: int = K, header=None, extra=None):
    cols = columns(rows)
    if extra:
        cols.update(extra)
    h = {"manifest_id": "SYNTH_VAL", "split": "VALIDATION", "catalogue_K": K_}
    if header:
        h.update(header)
    return build_eval_manifest(h, cols)


def random_rows(n_users: int = 40, seed: int = 0, K_: int = K, oov: float = 0.1, cens: float = 0.1,
                max_rows: int = 9):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for _ in range(int(rng.integers(1, max_rows + 1))):
            x = rng.random()
            kind = "C" if x < cens else ("O" if x < cens + oov else "R")
            rows.append((u, kind, int(rng.integers(0, K_))))
    return rows


def random_manifest(n_users: int = 40, seed: int = 0, K_: int = K, strata: bool = True, **kw):
    rows = random_rows(n_users, seed, K_, **kw)
    extra = None
    if strata:
        rng = np.random.default_rng(seed + 1)
        per_user = rng.choice(["WARM", "COLD_NO_TRAIN_EVENTS", "COLD_TRAIN_EVENTS_NO_LABEL"], size=n_users)
        users = np.array([r[0] for r in rows])
        sub = np.where(np.arange(len(rows)) % 3 == 0, "LATE", "EARLY")
        extra = {"cold_stratum": per_user[users], "subperiod": sub}
    return manifest_from_rows(rows, K_=K_, extra=extra)


def scores_for(m, seed: int = 0, dtype=torch.float32, quantize: float = 0.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(m.n, m.catalogue_K, generator=g, dtype=torch.float64)
    if quantize:
        s = torch.round(s / quantize) * quantize          # forces exact ties
    return s.to(dtype)


def config() -> EvalConfig:
    return EvalConfig()


def artifact_from_scores(m, scores: torch.Tensor, **kw):
    """Precomputed-score artifact (E2E-row batches of exactly eval_batch rows, unpadded tail)."""
    cfg = config()
    b = cfg.eval_batch
    rows = np.flatnonzero(m.eval_mask)                    # the evaluation stream: E2E rows in manifest order
    batches = [(torch.as_tensor(m.decision_id[rows[a:a + b]]), scores[rows[a:a + b]])
               for a in range(0, rows.size, b)]
    return predict_scores(batches, m, config=cfg, model_sha256="synthetic-model", **kw)


def evaluate_default(m, art, **kw):
    return evaluate(m, art, config=config(), **kw)

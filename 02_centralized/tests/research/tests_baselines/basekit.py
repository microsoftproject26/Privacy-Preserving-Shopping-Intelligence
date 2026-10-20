"""tests_base.basekit — small, seeded, in-memory BatchProvider / EvalProvider stand-ins WITH session_offset_tokens
turned on (`v4train.provider.Synthetic{Train,Eval}Provider` do not turn them on, and SKNN needs them; USER_HISTORY_L50
policy). Deterministic, tiny scale (K, L, N all small) — test fixtures only, never real data. Never edits v4train,
v4eval or v4models; only calls their documented public functions (`v4models.synthetic.synthetic_examples_v2`,
`v4eval.manifest.build_eval_manifest`).
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import pytest
import torch

from v4eval.manifest import build_eval_manifest
from v4eval.predict import stream_rows
from v4models.schema import InputWidths
from v4models.synthetic import synthetic_examples_v2, take

K = 24
L = 8
WIDTHS = InputWidths(numeric_dim=2, flags_dim=1, user_context_dim=1, user_mask_dim=1, view_id="TESTS_BASE_VIEW")


def _repeat_last_target(ex: dict, K: int) -> torch.Tensor:
    """The same "learnable" synthetic target v4train.provider._repeat_last_targets uses: the class of the window's
    own last real item (kept local/duplicated here on purpose — a two-line formula, not worth an import from a
    private helper of a package tests_base never edits)."""
    last = ex["item_tokens"].gather(1, (ex["lengths"] - 1).clamp(min=0).view(-1, 1)).squeeze(1)
    return (last - 3).clamp(0, K - 1)


class FixtureTrainProvider:
    """A minimal v4train.provider.BatchProvider: manifest_id / split / n_decisions / decision_ids() / batch(pos)."""

    split = "TRAIN"

    def __init__(self, n: int, K: int = K, *, seed: int = 0, max_len: int = L, manifest_id: str = "FIXTURE_TRAIN",
                did_start: int = 1000):
        g = torch.Generator().manual_seed(int(seed))
        ex = synthetic_examples_v2(int(n), int(K), WIDTHS, g, max_len=max_len, window=max_len, with_session_offset=True)
        ex["target_class"] = _repeat_last_target(ex, K)
        ex["decision_id"] = torch.arange(did_start, did_start + 2 * n, 2, dtype=torch.int64)
        self._ex = ex
        self.K = int(K)
        self.manifest_id = manifest_id

    @property
    def n_decisions(self) -> int:
        return int(self._ex["decision_id"].shape[0])

    def decision_ids(self) -> np.ndarray:
        return self._ex["decision_id"].numpy().copy()

    def batch(self, positions) -> dict:
        idx = torch.as_tensor(np.asarray(positions, dtype=np.int64))
        return take(self._ex, idx)


class FixtureEvalProvider:
    """A minimal v4train.provider.EvalProvider: `.manifest` (a real v4eval.manifest.EvalManifest, so the TEST gate
    and ordered-hash checks run) + `.batches(bs)` over its E2E (eval_mask) rows, in manifest order."""

    def __init__(self, n: int, K: int = K, *, seed: int = 1, max_len: int = L, manifest_id: str = "FIXTURE_VALIDATION",
                n_users: Optional[int] = None, oov_every: int = 17, censor_every: int = 13,
                header_extra: Optional[dict] = None, did_start: int = 50_000):
        g = torch.Generator().manual_seed(int(seed))
        ex = synthetic_examples_v2(int(n), int(K), WIDTHS, g, max_len=max_len, window=max_len, with_session_offset=True)
        tgt = _repeat_last_target(ex, K).numpy().astype(np.int64)
        i = np.arange(n)
        censored = (i % censor_every) == censor_every - 1
        oov = ((i % oov_every) == oov_every - 1) & ~censored
        eval_mask = ~censored
        tgt = np.where(eval_mask & ~oov, tgt, -1)
        n_users = n_users or max(1, n // 3)
        users = (i * 7919) % n_users
        did = np.arange(did_start, did_start + 2 * n, 2, dtype=np.int64)
        cols = {"decision_id": did, "eval_user_key": users.astype(np.int64), "eval_mask": eval_mask,
               "target_class": tgt, "target_oov": oov, "censor_class": np.where(censored, "CEN_NO_NEXT", "NONE")}
        header = {"manifest_id": manifest_id, "catalogue_K": int(K), "candidate_order_hash": "FIXTURE-CANDIDATES"}
        header.update(header_extra or {})
        self.manifest = build_eval_manifest(header, cols)
        ex["decision_id"] = torch.as_tensor(did)
        ex["target_class"] = torch.as_tensor(tgt)
        self._ex = ex
        self.K = int(K)

    def batches(self, batch_size: int) -> Iterator[dict]:
        rows = stream_rows(self.manifest)
        n = rows.size
        bs = int(batch_size)
        for s in range(0, n, bs):
            idx = torch.as_tensor(rows[s:s + bs], dtype=torch.long)
            yield take(self._ex, idx)


def nc(reason: str):
    """Negative control (same convention as tests_data/data_testkit.py, duplicated locally so tests_base stays
    self-contained per the BASE-01 hard rule): the wrapped check must raise AssertionError (strict xfail)."""
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason="NEGATIVE CONTROL: " + reason)


def assert_raises(exc, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc}")


MINIMAL_KEY_BASE = {
    "data_protocol_hash": "FIXTURE_DPH", "feature_contract_hash": "FIXTURE_FCH", "split": "VALIDATION",
    "label_policy_hash": "FIXTURE_LPH", "population_hash": "FIXTURE_POPH", "candidate_order_hash": "FIXTURE-CANDIDATES",
}


__all__ = ["K", "L", "WIDTHS", "FixtureTrainProvider", "FixtureEvalProvider", "MINIMAL_KEY_BASE", "nc", "assert_raises"]

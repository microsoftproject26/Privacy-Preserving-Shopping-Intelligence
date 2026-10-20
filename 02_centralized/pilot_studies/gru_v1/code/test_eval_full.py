"""Tests for `eval_full.py` -- Wave-1 Sonnet C.

Run on the server, pinned to GPU 1:
    CUDA_VISIBLE_DEVICES=1 ~/venvs/ppsi-gpu/bin/python -m pytest code/test_eval_full.py -v

`full_store.py` (Sonnet A's `CompactStore`/`DecisionSet`) does not exist yet,
so this file defines a small, clearly-labelled stand-in `DecisionSet`
(`_StandInDecisionSet`) plus two stand-in stores:

  * `_CacheReaderStore` -- wraps the pilot's (frozen, never edited)
    `window_cache.CacheReader`. Unlike a real `CompactStore` (which stores
    each event ONCE and reconstructs a decision's window from a compact,
    non-redundant per-event array via `end_row`/`length`), the pilot cache
    already stores one full L=50 window PER decision, so `end_row` here is
    simply an identity row index into the cache. This is enough to exercise
    `evaluate_model`'s iteration/forward/rank contract and the parity-gate
    (iii) regression numbers; it is not a test of `CompactStore`'s own
    window-reconstruction logic (that is Sonnet A's / Opus's integration
    concern).
  * `_TinyStore` -- a fully synthetic, tiny F0-only store for the
    RNG/training-mode-restoration unit test (no GPU checkpoint dependency).

Sections:
  1. `rank_targets` vs `evaluator._ranks_chunked` (heavy ties, float + int64,
     multiple chunk widths) + an explicit float32-precision-loss regression.
  2. `aggregate` reproduces the frozen T8/T9 fixtures from `test_evaluator.py`
     (imported, never edited) and matches `evaluator.evaluate()` exactly on
     random small problems.
  3. `evaluate_model`: RNG/training-mode save-restore + bit-identical repeat
     on a synthetic model/store.
  4. Parity gate (iii), tolerance frozen at |delta| <= 1e-6: the pilot best
     checkpoints (F0, F4) evaluated through `evaluate_model` + `_CacheReaderStore`
     on the pilot's 10k VS cohort, at eval_batch 512, must reproduce the
     recorded pilot `eval_full` record for the checkpoint's own update.
  5. `per_user_metric` / `paired_bootstrap`.
"""
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CODE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = CODE_DIR / "vendor"
for _p in (str(CODE_DIR), str(VENDOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
# FULL_RUN_PROTOCOL.md sec. 4: strict FP32 (TF32 off for matmul and cuDNN).
# The pilot's checkpoints were trained AND evaluated under these same global
# backend flags (train.py:set_determinism(), called once, process-wide,
# before the pilot's own run_vs_evaluation) -- without disabling TF32 here
# too, the parity gate (iii) comparison picks up a small (~1e-5 relative)
# numerical difference from TF32's reduced mantissa in the GRU's matmuls,
# which is enough to flip a handful of near-tied ranks. This must be set
# before any CUDA matmul/GRU call, i.e. before any model is built/loaded.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import model as model_mod  # noqa: E402
import train as train_mod  # noqa: E402
from evaluator import evaluate, _ranks_chunked  # noqa: E402
from window_cache import CacheReader  # noqa: E402
from eval_full import (  # noqa: E402
    rank_targets, aggregate, evaluate_model, per_user_metric, paired_bootstrap,
)
from test_evaluator import (  # noqa: E402  -- frozen fixtures, imported, never edited
    _make_t9_rows, _make_t9_oov_extension_rows, _t8_rows_and_scores, EXPECTED, T8_KCUT,
)

WORKSPACE = Path(
    "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1"
)
CACHE_VS = WORKSPACE / "cache" / "vs"
COHORT_PARQUET = WORKSPACE / "artifacts" / "validation_select_cohort.parquet"
PILOT_RUNS = {
    "F0_item_only": WORKSPACE / "experiments" / "gru" / "T-M_F0_item_only",
    "F4_rich_side_information": WORKSPACE / "experiments" / "gru" / "T-M_F4_rich_side_information",
}


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available -- this suite requires the GPU-1 slot")
    return torch.device("cuda:0")


# ===========================================================================
# Stand-in DecisionSet / stores (full_store.py does not exist yet -- see the
# module docstring).
# ===========================================================================

@dataclass
class _StandInDecisionSet:
    split: str
    kind: str
    decision_id: np.ndarray
    end_row: np.ndarray
    length: np.ndarray
    target_class: np.ndarray
    loss_mask: np.ndarray
    eval_mask: np.ndarray
    client_idx: np.ndarray
    user_bucket: np.ndarray
    eval_user_key: Optional[np.ndarray]
    prior_view_count: np.ndarray
    prefix_position: np.ndarray

    def __len__(self) -> int:
        return int(self.decision_id.shape[0])

    def take(self, positions) -> "_StandInDecisionSet":
        positions = np.asarray(positions)

        def sub(a):
            return a[positions] if a is not None else None

        return _StandInDecisionSet(
            split=self.split, kind=self.kind, decision_id=sub(self.decision_id),
            end_row=sub(self.end_row), length=sub(self.length), target_class=sub(self.target_class),
            loss_mask=sub(self.loss_mask), eval_mask=sub(self.eval_mask), client_idx=sub(self.client_idx),
            user_bucket=sub(self.user_bucket), eval_user_key=sub(self.eval_user_key),
            prior_view_count=sub(self.prior_view_count), prefix_position=sub(self.prefix_position),
        )

    def sha256(self) -> str:
        import hashlib
        return hashlib.sha256(np.ascontiguousarray(self.decision_id).tobytes()).hexdigest()


class _CacheReaderStore:
    """Stand-in for `CompactStore`, backed by the pilot's `CacheReader`. See
    the module docstring for exactly what this does and does not test."""

    def __init__(self, cache_dir: Path):
        self.reader = CacheReader(cache_dir)
        self.channels = "F4"

    def batch(self, ds, positions):
        rows = ds.end_row[np.asarray(positions)]
        return self.reader.read(rows)

    def decision_item_tokens(self, ds):
        raw = self.reader.read(ds.end_row)
        item_tokens = raw["item_tokens"].numpy()
        lengths = raw["lengths"].numpy()
        return item_tokens[np.arange(len(ds)), lengths - 1]


def _build_vs_cohort_ds(cache_vs_dir: Path, cohort_parquet: Path) -> _StandInDecisionSet:
    reader = CacheReader(cache_vs_dir)
    n = reader.header["row_count"]
    raw = reader.read(np.arange(n))
    cohort = pq.read_table(cohort_parquet, columns=["eval_user_key"])
    eval_user_key = np.asarray(cohort["eval_user_key"].to_numpy(zero_copy_only=False), dtype=np.int64)
    assert eval_user_key.shape[0] == n, "cohort parquet row count != cache/vs row count"
    return _StandInDecisionSet(
        split="VALIDATION_SELECT", kind="eval",
        decision_id=raw["decision_id"].numpy(), end_row=np.arange(n, dtype=np.int64),
        length=raw["lengths"].numpy(), target_class=raw["target_class"].numpy(),
        loss_mask=raw["loss_mask"].numpy(), eval_mask=raw["eval_mask"].numpy(),
        client_idx=raw["client_id"].numpy(), user_bucket=np.zeros(n, dtype=np.int16),
        eval_user_key=eval_user_key, prior_view_count=np.zeros(n, dtype=np.int64),
        prefix_position=np.ones(n, dtype=np.int64),
    )


class _TinyStore:
    """Fully synthetic, tiny F0-only store (no GPU checkpoint dependency) --
    for the RNG/training-mode unit test."""

    def __init__(self, item_tokens, lengths, target_class, eval_mask, client_idx):
        self.item_tokens = item_tokens
        self.lengths = lengths
        self.target_class = target_class
        self.eval_mask = eval_mask
        self.client_idx = client_idx

    def batch(self, ds, positions):
        idx = ds.end_row[np.asarray(positions)]
        L = self.item_tokens.shape[1]
        it = self.item_tokens[idx]
        lengths = self.lengths[idx]
        attn = np.arange(L)[None, :] < lengths[:, None]
        return {
            "item_tokens": torch.as_tensor(it, dtype=torch.long),
            "attention_mask": torch.as_tensor(attn, dtype=torch.bool),
            "lengths": torch.as_tensor(lengths, dtype=torch.long),
            "target_class": torch.as_tensor(self.target_class[idx], dtype=torch.long),
            "eval_mask": torch.as_tensor(self.eval_mask[idx], dtype=torch.bool),
            "loss_mask": torch.zeros(len(idx), dtype=torch.bool),
            "decision_id": torch.as_tensor(idx, dtype=torch.long),
            "client_id": torch.as_tensor(self.client_idx[idx], dtype=torch.long),
            "last_valid_index": torch.as_tensor(lengths - 1, dtype=torch.long),
        }


# ===========================================================================
# 1. rank_targets vs evaluator._ranks_chunked
# ===========================================================================

@pytest.mark.parametrize("score_kind", ["integer_valued_float32", "int64"])
@pytest.mark.parametrize("chunk_width", [1, 7, 4096, 32768, 500])  # 500 == this test's own K
def test_rank_targets_matches_evaluator_ranks_chunked(score_kind, chunk_width, device):
    rng = np.random.default_rng(hash((score_kind, chunk_width)) % (2**31))
    n, K = 64, 500  # small integer range -> heavy ties
    raw = rng.integers(0, 5, size=(n, K))
    target = rng.integers(0, K, size=n).astype(np.int64)

    if score_kind == "integer_valued_float32":
        scores_np = raw.astype(np.float32)
        scores_cpu = torch.as_tensor(scores_np, dtype=torch.float32)
        reference_scores = scores_cpu
    else:
        scores_np = raw.astype(np.int64)
        scores_cpu = torch.as_tensor(scores_np, dtype=torch.int64)
        reference_scores = scores_cpu.to(torch.float32)  # exact: values are tiny (< 2**24)

    target_t = torch.as_tensor(target, dtype=torch.long)
    expected = _ranks_chunked(reference_scores, target_t, chunk_size=chunk_width)

    got_cpu = rank_targets(scores_cpu, target_t, class_chunk=chunk_width)
    assert torch.equal(got_cpu, expected)

    got_gpu = rank_targets(scores_cpu.to(device), target_t.to(device), class_chunk=chunk_width)
    assert torch.equal(got_gpu.cpu(), expected)


def test_rank_targets_int64_avoids_float32_collision():
    """v0=2**24 and v1=v0+1 are distinct int64 values that COLLIDE once cast
    to float32 (float32 cannot exactly represent 2**24+1). ranking on int64
    must NOT treat them as tied; a float32 cast silently does -- exactly the
    exactness bug `baselines_full.py`'s composite score exists to avoid at
    full scale (max_composite_score exceeds 2**24 there)."""
    v0 = 1 << 24
    v1 = v0 + 1
    scores = torch.tensor([[v0, v1, 0, 0]], dtype=torch.int64)
    target = torch.tensor([1], dtype=torch.long)  # target column holds v1

    rank_int64 = rank_targets(scores, target, class_chunk=4096)
    assert int(rank_int64.item()) == 1  # v1 > v0 in int64: no tie, target is the strict max

    rank_via_float32 = rank_targets(scores.to(torch.float32), target, class_chunk=4096)
    assert int(rank_via_float32.item()) == 2  # v0/v1 collide once cast to float32: a false tie


# ===========================================================================
# 2. aggregate reproduces T8/T9 and matches evaluator.evaluate() exactly.
# ===========================================================================

def _ranks_for(scores: np.ndarray, target: np.ndarray, eval_mask: np.ndarray,
               chunk_size: int) -> np.ndarray:
    n = scores.shape[0]
    ranks = np.full(n, -1, dtype=np.int64)
    need = eval_mask & (target >= 0)
    need_idx = np.nonzero(need)[0]
    if need_idx.size:
        r = rank_targets(torch.as_tensor(scores[need_idx], dtype=torch.float32),
                          torch.as_tensor(target[need_idx], dtype=torch.long), class_chunk=chunk_size)
        ranks[need_idx] = r.numpy()
    return ranks


def _with_slice_fields(rows: dict) -> dict:
    """Adapts a test_evaluator.py fixture dict (evaluator.evaluate()'s "rows"
    naming: `client_id`) into aggregate()'s DecisionSet-style naming
    (`client_idx`), plus the two reporting-only slice fields aggregate()
    always requires."""
    n = rows["target_class"].shape[0]
    out = {k: v for k, v in rows.items() if k != "client_id"}
    out["client_idx"] = rows["client_id"]
    out["prior_view_count"] = np.zeros(n, dtype=np.int64)
    out["prefix_position"] = np.ones(n, dtype=np.int64)
    return out


def test_aggregate_matches_t9_fixture():
    scores, rows = _make_t9_rows()
    ranks = _ranks_for(scores, rows["target_class"], rows["eval_mask"], chunk_size=7)
    result = aggregate(ranks, _with_slice_fields(rows), k=20)

    assert result["end_to_end"]["macro"]["mrr@20"] == pytest.approx(0.5, abs=1e-9)
    assert result["end_to_end"]["micro"]["mrr@20"] == pytest.approx(0.625, abs=1e-9)
    assert result["end_to_end"]["macro"]["user_count"] == 3
    assert result["user_count"] == 3


def test_aggregate_matches_t9_oov_extension_fixture():
    scores, rows = _make_t9_oov_extension_rows()
    ranks = _ranks_for(scores, rows["target_class"], rows["eval_mask"], chunk_size=11)
    result = aggregate(ranks, _with_slice_fields(rows), k=20)

    assert result["end_to_end"]["macro"]["mrr@20"] == pytest.approx(0.4166666666666667, abs=1e-6)
    assert result["closed"]["macro"]["mrr@20"] == pytest.approx(0.5, abs=1e-9)
    assert result["end_to_end"]["n"] == 5
    assert result["closed"]["n"] == 4


@pytest.mark.parametrize("chunk_size", [1, 4, 11])
def test_aggregate_matches_t8_fixture(chunk_size):
    scores, rows = _t8_rows_and_scores()
    ranks = _ranks_for(scores, rows["target_class"], rows["eval_mask"], chunk_size=chunk_size)
    result = aggregate(ranks, _with_slice_fields(rows), k=T8_KCUT)

    for pop in ("end_to_end", "closed"):
        assert result[pop]["n"] == EXPECTED[pop]["n"], pop
        for scope in ("micro", "macro"):
            for name, expected_val in EXPECTED[pop][scope].items():
                got = result[pop][scope][name]
                assert got == pytest.approx(expected_val, abs=1e-6), f"{pop}.{scope}.{name}"


def test_aggregate_matches_evaluator_evaluate_random():
    rng = np.random.default_rng(42)
    for _trial in range(8):
        n = int(rng.integers(5, 40))
        K = int(rng.integers(10, 60))
        scores = rng.integers(0, 8, size=(n, K)).astype(np.float32)  # heavy ties
        target_class = rng.integers(-1, K, size=n).astype(np.int64)  # some OOV (-1)
        eval_mask = rng.integers(0, 2, size=n).astype(bool)
        n_users = max(1, n // 3)
        eval_user_key = rng.integers(0, n_users, size=n).astype(np.int64)
        client_idx = rng.choice([2, 3, 5, 7], size=n).astype(np.int64)

        eval_rows = {"target_class": target_class, "eval_mask": eval_mask,
                     "eval_user_key": eval_user_key, "client_id": client_idx}
        expected = evaluate(lambda: [torch.as_tensor(scores, dtype=torch.float32)], eval_rows,
                             k=5, chunk_size=7)

        ranks = _ranks_for(scores, target_class, eval_mask, chunk_size=7)
        ds_rows = {"target_class": target_class, "eval_mask": eval_mask,
                   "eval_user_key": eval_user_key, "client_idx": client_idx,
                   "prior_view_count": np.zeros(n, dtype=np.int64),
                   "prefix_position": np.ones(n, dtype=np.int64)}
        got = aggregate(ranks, ds_rows, k=5)

        def _cmp(a, b, path):
            if isinstance(a, float) and math.isnan(a):
                assert isinstance(b, float) and math.isnan(b), path
            else:
                assert a == b or abs(a - b) < 1e-12, f"{path}: {a} != {b}"

        for pop in ("end_to_end", "closed"):
            _cmp(got[pop]["n"], expected[pop]["n"], f"{pop}.n")
            for scope in ("micro", "macro"):
                for name, val in expected[pop][scope].items():
                    _cmp(got[pop][scope][name], val, f"{pop}.{scope}.{name}")
        _cmp(got["user_count"], expected["user_count"], "user_count")
        for cohort in ("warm", "unseen"):
            for pop in ("end_to_end", "closed"):
                for scope in ("micro", "macro"):
                    for name, val in expected["cohorts"][cohort][pop][scope].items():
                        _cmp(got["cohorts"][cohort][pop][scope][name], val,
                             f"cohorts.{cohort}.{pop}.{scope}.{name}")


# ===========================================================================
# 3. evaluate_model: RNG / training-mode restoration + bit-identical repeat.
# ===========================================================================

def test_evaluate_model_rng_and_training_mode_restored(device):
    K = 12
    vocab = {"item_tokens": K + 3}
    poc = np.arange(3, 3 + K, dtype=np.int64)
    torch.manual_seed(123)
    model = model_mod.build_model("F0_item_only", vocab, product_idx_of_class=poc).to(device)
    model.train()

    n, L = 17, 5
    rng = np.random.default_rng(3)
    lengths = rng.integers(1, L + 1, size=n).astype(np.int64)
    item_tokens = rng.integers(3, 3 + K, size=(n, L)).astype(np.int64)
    for i in range(n):
        item_tokens[i, lengths[i]:] = 0
    target_class = rng.integers(0, K, size=n).astype(np.int64)
    eval_mask = np.ones(n, dtype=bool)
    client_idx = np.full(n, 7, dtype=np.int64)

    store = _TinyStore(item_tokens, lengths, target_class, eval_mask, client_idx)
    ds = _StandInDecisionSet(
        split="VALIDATION_SELECT", kind="eval", decision_id=np.arange(n, dtype=np.int64),
        end_row=np.arange(n, dtype=np.int64), length=lengths, target_class=target_class,
        loss_mask=np.zeros(n, dtype=bool), eval_mask=eval_mask, client_idx=client_idx,
        user_bucket=np.zeros(n, dtype=np.int16), eval_user_key=np.arange(n, dtype=np.int64),
        prior_view_count=np.zeros(n, dtype=np.int64), prefix_position=np.ones(n, dtype=np.int64),
    )

    torch_state_before = torch.get_rng_state().clone()

    result_a = evaluate_model(model, store, ds, device, eval_batch=4, k=5, return_ranks=True,
                               prefetch=False)
    assert model.training is True  # restored to the pre-call state

    torch_state_after = torch.get_rng_state()
    assert torch.equal(torch_state_before, torch_state_after)

    draw_after = torch.rand(3)
    torch.set_rng_state(torch_state_before)
    draw_expected = torch.rand(3)
    assert torch.equal(draw_after, draw_expected)

    result_b = evaluate_model(model, store, ds, device, eval_batch=4, k=5, return_ranks=True,
                               prefetch=False)
    assert np.array_equal(result_a["ranks"], result_b["ranks"])
    assert result_a["end_to_end"]["micro"]["mrr@5"] == result_b["end_to_end"]["micro"]["mrr@5"]


# ===========================================================================
# 4. Parity gate (iii) -- pilot best checkpoints on the 10k VS cohort.
# ===========================================================================

def _pilot_best_eval_record(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    best_metric = summary["best_metric"]
    lines = [json.loads(l) for l in (run_dir / "metrics.jsonl").open(encoding="utf-8")]
    evals = [l for l in lines if "eval_full" in l]
    for e in evals:
        if abs(e["eval_client_macro_mrr@20"] - best_metric) < 1e-9:
            return e["eval_full"]
    raise AssertionError(f"no eval_full record in {run_dir}/metrics.jsonl matches best_metric")


def _load_pilot_model(run_dir: Path, group: str, device):
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    poc = train_mod.load_product_idx_of_class()
    model = model_mod.build_model(group, train_mod.VOCAB_SIZES, product_idx_of_class=poc)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@pytest.mark.parametrize("group,run_key", [
    ("F0_item_only", "F0_item_only"),
    ("F4_rich_side_information", "F4_rich_side_information"),
])
def test_parity_gate_iii_pilot_cohort(group, run_key, device):
    run_dir = PILOT_RUNS[run_key]
    if not (run_dir / "checkpoints" / "best.pt").exists():
        pytest.skip(f"pilot checkpoint not found at {run_dir}")

    model = _load_pilot_model(run_dir, group, device)
    store = _CacheReaderStore(CACHE_VS)
    ds = _build_vs_cohort_ds(CACHE_VS, COHORT_PARQUET)

    result_a = evaluate_model(model, store, ds, device, eval_batch=512, k=20, return_ranks=True)
    result_b = evaluate_model(model, store, ds, device, eval_batch=512, k=20, return_ranks=True)

    assert np.array_equal(result_a["ranks"], result_b["ranks"]), "two evaluations must be bit-identical"

    expected = _pilot_best_eval_record(run_dir)
    tol = 1e-6
    assert result_a["end_to_end"]["n"] == expected["end_to_end"]["n"]
    assert result_a["end_to_end"]["macro"]["user_count"] == expected["end_to_end"]["macro"]["user_count"]
    for scope in ("micro", "macro"):
        for metric in ("mrr@20", "recall@20"):
            got = result_a["end_to_end"][scope][metric]
            exp = expected["end_to_end"][scope][metric]
            assert abs(got - exp) <= tol, f"end_to_end.{scope}.{metric}: got {got} expected {exp}"


# ===========================================================================
# 5. per_user_metric / paired_bootstrap
# ===========================================================================

def test_per_user_metric_macro_mean_matches_aggregate():
    scores, rows = _make_t9_rows()
    ranks = _ranks_for(scores, rows["target_class"], rows["eval_mask"], chunk_size=7)
    ds_rows = _with_slice_fields(rows)

    agg = aggregate(ranks, ds_rows, k=20)
    keys, vals = per_user_metric(ranks, ds_rows, metric="mrr", k=20, denominator="end_to_end")

    assert vals.mean() == pytest.approx(agg["end_to_end"]["macro"]["mrr@20"], abs=1e-12)
    assert keys.shape[0] == agg["end_to_end"]["macro"]["user_count"]


def test_paired_bootstrap_identical_models_diff_zero():
    rng = np.random.default_rng(0)
    keys = np.arange(200)
    vals = rng.random(200)
    result = paired_bootstrap(keys, vals, keys, vals, n=200, seed=2026)
    assert result["diff"] == 0.0
    assert result["ci95"][0] == pytest.approx(0.0, abs=1e-12)
    assert result["ci95"][1] == pytest.approx(0.0, abs=1e-12)
    assert result["n_users"] == 200


def test_paired_bootstrap_known_positive_difference_excludes_zero():
    rng = np.random.default_rng(1)
    n_users = 500
    keys = np.arange(n_users)
    vals_a = rng.normal(loc=0.20, scale=0.02, size=n_users)
    vals_b = rng.normal(loc=0.15, scale=0.02, size=n_users)
    result = paired_bootstrap(keys, vals_a, keys, vals_b, n=1000, seed=2026)
    assert result["diff"] > 0
    assert result["ci95"][0] > 0


def test_paired_bootstrap_deterministic_with_seed():
    rng = np.random.default_rng(2)
    n_users = 300
    keys = np.arange(n_users)
    vals_a = rng.random(n_users)
    vals_b = rng.random(n_users)
    r1 = paired_bootstrap(keys, vals_a, keys, vals_b, n=500, seed=2026)
    r2 = paired_bootstrap(keys, vals_a, keys, vals_b, n=500, seed=2026)
    assert r1["ci95"] == r2["ci95"]
    assert r1["diff"] == r2["diff"]


def test_paired_bootstrap_requires_same_user_set():
    keys_a = np.array([1, 2, 3])
    keys_b = np.array([1, 2, 4])
    vals = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        paired_bootstrap(keys_a, vals, keys_b, vals)

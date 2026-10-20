"""common.py (owner: Sonnet C) -- G0 ONNX export feasibility smoke, shared helpers.

CPU only. Random-initialised weights only. No dataset rows beyond the frozen
catalogue mapping (product_idx_of_class) that every model-building script in
this repo already reads to learn K / the item<->class map -- no event/user
rows are touched.

Import order (isolation contract, see ppsi_env.py): ppsi_env FIRST, then the
snapshot copies of train/model/sasrec_model. Never import gru_v1/sasrec_v1/
feature_opt_v1 directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

CODE = Path(__file__).resolve().parent.parent  # .../ppsi_v4/code
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import ppsi_env  # noqa: E402  (must precede any snapshot module import)

import torch  # noqa: E402
from torch import nn  # noqa: E402

import train as reuse_train  # noqa: E402  (snapshot copy)
import model as gru_model  # noqa: E402
from sasrec_model import SASRecConfig, SASRecCE  # noqa: E402

torch.set_num_threads(4)

L = 50  # frozen context-window width (rees46_next_item_shared_v4_localtime)
NUM, FLAGS, UC, UM = 19, 15, 12, 6
SASREC_ENTRY = {"id": "A1_RICH256", "group": "rich_frozen_channels", "d_model": 256, "blocks": 2,
                "heads": 4, "ff_multiplier": 4, "dropout": 0.2}
SEED = 2026

SIDE_TOKEN_CHANNELS = ("category_tokens", "brand_tokens", "main_category_tokens",
                       "daypart_tokens", "weekday_tokens")

_ISOLATION_SEEN = None
_POC = None


def isolate_and_verify() -> dict:
    global _ISOLATION_SEEN
    if _ISOLATION_SEEN is None:
        _ISOLATION_SEEN = ppsi_env.assert_isolated()
        ppsi_env.verify_snapshot()
    return _ISOLATION_SEEN


def load_poc():
    global _POC
    if _POC is None:
        _POC = reuse_train.load_product_idx_of_class()
    return _POC


def vocab_sizes() -> dict:
    return dict(reuse_train.VOCAB_SIZES)


def build_gru() -> nn.Module:
    torch.manual_seed(SEED)  # ContextGRU has no explicit seed arg -- pin the global RNG
    poc = load_poc()
    m = gru_model.build_model("F4_rich_side_information", reuse_train.VOCAB_SIZES, product_idx_of_class=poc)
    return m.eval()


def build_sasrec() -> nn.Module:
    poc = load_poc()
    m = SASRecCE(SASRecConfig.from_registry(SASREC_ENTRY), vocab_sizes=dict(reuse_train.VOCAB_SIZES),
                product_idx_of_class=poc, seed=SEED)
    return m.eval()


def get_K() -> int:
    return int(load_poc().shape[0])


# ---------------------------------------------------------------------------
# Synthetic input scenarios -- shapes/vocab match g0_bench_fl_cost.py:batch(),
# token conventions (PAD=0, MISSING=1, OOV=2, KNOWN>=3) match
# snap_v1/reuse/vendor/rees46_eda/config.py and snap_v1/synthetic_store.py.
# ---------------------------------------------------------------------------

def _lengths_for(batch_size: int, mode, gen: torch.Generator) -> torch.Tensor:
    """mode: an int (every row gets that fixed length) or 'mixed' (random 1..L,
    at least one row forced to L so any trace using this batch bakes T=L)."""
    if mode == "mixed":
        lengths = torch.randint(1, L + 1, (batch_size,), generator=gen)
        lengths[0] = L
        return lengths
    return torch.full((batch_size,), int(mode), dtype=torch.long)


def make_batch(batch_size: int, mode, gen: torch.Generator, K: int, *, missing: bool = False) -> dict:
    """Build one synthetic input dict (all tensors CPU), right-padded, following
    the frozen store's layout contract (real events at columns 0..len-1,
    position_ids = 1..len on real columns, 0 on PAD).

    missing=True: every real-position side-channel token is the vocab's
    MISSING(1) sentinel, event_quality_flags are all 1 ("on"), and the
    corresponding numeric features are zeroed (matches the "zero-filled
    channel + its missing indicator" convention in rees46_eda/config.py).
    """
    V = vocab_sizes()
    b = batch_size
    lengths = _lengths_for(b, mode, gen)
    ar = torch.arange(L)
    am = (ar[None, :] < lengths[:, None])  # [B, L] bool, real-event positions

    def real_tok(vocab_n: int) -> torch.Tensor:
        if missing:
            t = torch.ones((b, L), dtype=torch.long)  # MISSING everywhere real
        else:
            t = torch.randint(3, vocab_n, (b, L), generator=gen)
        return t * am  # PAD(0) past length

    item_tokens = torch.randint(3, V["item_tokens"], (b, L), generator=gen) * am  # item itself always real
    out = {
        "item_tokens": item_tokens,
        "category_tokens": real_tok(V["category_tokens"]),
        "brand_tokens": real_tok(V["brand_tokens"]),
        "main_category_tokens": real_tok(V["main_category_tokens"]),
        "daypart_tokens": real_tok(V["daypart_tokens"]),
        "weekday_tokens": real_tok(V["weekday_tokens"]),
        "lengths": lengths,
        "attention_mask": am,
        "position_ids": (ar[None, :] + 1) * am,
    }
    if missing:
        out["event_numeric_features"] = torch.zeros(b, L, NUM)
        out["event_quality_flags"] = torch.ones(b, L, FLAGS, dtype=torch.int8) * am[..., None].to(torch.int8)
    else:
        out["event_numeric_features"] = torch.randn(b, L, NUM, generator=gen) * am[..., None]
        out["event_quality_flags"] = (torch.rand(b, L, FLAGS, generator=gen) < 0.05).to(torch.int8) * am[..., None].to(torch.int8)
    out["user_context"] = torch.randn(b, UC, generator=gen)
    out["user_context_masks"] = torch.ones(b, UM)
    out["target_class"] = torch.randint(0, K, (b,), generator=gen)
    return out


GRU_FULL_ARGS = ("item_tokens", "lengths", "attention_mask", "category_tokens", "brand_tokens",
                 "main_category_tokens", "daypart_tokens", "weekday_tokens", "event_numeric_features",
                 "event_quality_flags", "user_context", "user_context_masks")
SASREC_FULL_ARGS = ("item_tokens", "lengths", "attention_mask", "position_ids", "category_tokens",
                    "brand_tokens", "main_category_tokens", "daypart_tokens", "weekday_tokens",
                    "event_numeric_features", "event_quality_flags", "user_context", "user_context_masks")


class GRUFullWrapper(nn.Module):
    """Positional wrapper: full path item_tokens+side -> logits[B,K]."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, item_tokens, lengths, attention_mask, category_tokens, brand_tokens,
                main_category_tokens, daypart_tokens, weekday_tokens, event_numeric_features,
                event_quality_flags, user_context, user_context_masks):
        return self.model(item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask,
                          category_tokens=category_tokens, brand_tokens=brand_tokens,
                          main_category_tokens=main_category_tokens, daypart_tokens=daypart_tokens,
                          weekday_tokens=weekday_tokens, event_numeric_features=event_numeric_features,
                          event_quality_flags=event_quality_flags, user_context=user_context,
                          user_context_masks=user_context_masks)


class GRUQueryWrapper(nn.Module):
    """Positional wrapper: query-encoder only -> q[B,128] (pre-head, split option)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, item_tokens, lengths, attention_mask, category_tokens, brand_tokens,
                main_category_tokens, daypart_tokens, weekday_tokens, event_numeric_features,
                event_quality_flags, user_context, user_context_masks):
        return self.model.query(item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask,
                                category_tokens=category_tokens, brand_tokens=brand_tokens,
                                main_category_tokens=main_category_tokens, daypart_tokens=daypart_tokens,
                                weekday_tokens=weekday_tokens, event_numeric_features=event_numeric_features,
                                event_quality_flags=event_quality_flags, user_context=user_context,
                                user_context_masks=user_context_masks)


class SASRecFullWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, item_tokens, lengths, attention_mask, position_ids, category_tokens, brand_tokens,
                main_category_tokens, daypart_tokens, weekday_tokens, event_numeric_features,
                event_quality_flags, user_context, user_context_masks):
        return self.model(item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask,
                          position_ids=position_ids, category_tokens=category_tokens, brand_tokens=brand_tokens,
                          main_category_tokens=main_category_tokens, daypart_tokens=daypart_tokens,
                          weekday_tokens=weekday_tokens, event_numeric_features=event_numeric_features,
                          event_quality_flags=event_quality_flags, user_context=user_context,
                          user_context_masks=user_context_masks)


class SASRecQueryWrapper(nn.Module):
    """query-encoder only -> q[B,256] (post final_norm, pre untied head)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, item_tokens, lengths, attention_mask, position_ids, category_tokens, brand_tokens,
                main_category_tokens, daypart_tokens, weekday_tokens, event_numeric_features,
                event_quality_flags, user_context, user_context_masks):
        return self.model.query(item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask,
                                position_ids=position_ids, category_tokens=category_tokens, brand_tokens=brand_tokens,
                                main_category_tokens=main_category_tokens, daypart_tokens=daypart_tokens,
                                weekday_tokens=weekday_tokens, event_numeric_features=event_numeric_features,
                                event_quality_flags=event_quality_flags, user_context=user_context,
                                user_context_masks=user_context_masks)


def gru_args(batch: dict) -> tuple:
    return tuple(batch[k] for k in GRU_FULL_ARGS)


def sasrec_args(batch: dict) -> tuple:
    return tuple(batch[k] for k in SASREC_FULL_ARGS)


def rss_mb() -> float:
    """Process peak RSS in MB (Linux: ru_maxrss is KB)."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Export attempt (try default exporter, fall back to the other mode) + parity
# metrics + onnxruntime latency helper. Shared by run_gru.py / run_sasrec.py.
# ---------------------------------------------------------------------------

def try_export(wrapper: nn.Module, example_args: tuple, path: str, input_names: list[str],
                output_names: list[str], dynamic_axes: dict, opset_version: int = 18) -> dict:
    """Try torch.onnx.export's default exporter (dynamo=True) then, on failure,
    the other mode (dynamo=False). Returns a dict recording exactly which
    exporter worked (or that neither did), opset, warnings, and error text."""
    import traceback
    import warnings as warnings_mod

    result = {"path": path, "opset_version": opset_version, "input_names": list(input_names),
              "output_names": list(output_names), "dynamic_axes": dynamic_axes, "attempts": []}
    for dynamo in (True, False):
        attempt = {"exporter": "dynamo (torch.export-based, default)" if dynamo else "legacy (TorchScript-trace-based, dynamo=False)"}
        try:
            with warnings_mod.catch_warnings(record=True) as wlist:
                warnings_mod.simplefilter("always")
                wrapper.eval()
                torch.onnx.export(wrapper, example_args, path, dynamo=dynamo,
                                  input_names=list(input_names), output_names=list(output_names),
                                  dynamic_axes=dynamic_axes, opset_version=opset_version,
                                  do_constant_folding=True)
            attempt["status"] = "ok"
            attempt["warnings"] = [str(w.message)[:500] for w in wlist]
            size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else None
            attempt["onnx_file_size_mb"] = size_mb
            result["attempts"].append(attempt)
            result["status"] = "ok"
            result["exporter_used"] = attempt["exporter"]
            result["onnx_file_size_mb"] = size_mb
            return result
        except Exception as e:
            attempt["status"] = "failed"
            attempt["error_type"] = type(e).__name__
            attempt["error_text"] = "".join(traceback.format_exception_only(type(e), e))[:2000]
            attempt["error_traceback_tail"] = "".join(traceback.format_exc().splitlines(True)[-25:])
            result["attempts"].append(attempt)
    result["status"] = "failed"
    result["exporter_used"] = None
    return result


def compare_logits(logits_pt: torch.Tensor, logits_onnx, target_class: torch.Tensor, topk: int = 20) -> dict:
    """Parity metrics between a PyTorch logits tensor and an onnxruntime output
    (numpy array) of the same [B,K] shape: max abs diff, fraction of rows whose
    top-k index SET matches exactly, and whether a random target class's exact
    rank (0 = highest logit) is identical."""
    import numpy as np
    lo = torch.as_tensor(logits_onnx)
    lp = logits_pt.detach()
    max_abs_diff = float((lp - lo).abs().max())
    B = lp.shape[0]
    topk_pt = lp.topk(topk, dim=-1).indices
    topk_on = lo.topk(topk, dim=-1).indices
    overlap_fracs = []
    for i in range(B):
        s_pt, s_on = set(topk_pt[i].tolist()), set(topk_on[i].tolist())
        overlap_fracs.append(len(s_pt & s_on) / topk)
    rank_matches = []
    for i in range(B):
        t = int(target_class[i])
        rank_pt = int((lp[i] > lp[i, t]).sum())
        rank_on = int((lo[i] > lo[i, t]).sum())
        rank_matches.append(rank_pt == rank_on)
    return {
        "max_abs_logit_diff": max_abs_diff,
        "mean_top20_overlap_frac": float(np.mean(overlap_fracs)),
        "min_top20_overlap_frac": float(np.min(overlap_fracs)),
        "all_rows_top20_identical": bool(all(f == 1.0 for f in overlap_fracs)),
        "target_rank_identical_frac": float(np.mean(rank_matches)),
        "all_target_ranks_identical": bool(all(rank_matches)),
    }


def compare_vectors(v_pt: torch.Tensor, v_onnx) -> dict:
    """Parity for a query-encoder-only export: just the vector, no head."""
    lo = torch.as_tensor(v_onnx)
    lp = v_pt.detach()
    return {"max_abs_diff": float((lp - lo).abs().max()), "mean_abs_diff": float((lp - lo).abs().mean())}


def filtered_feed(sess, names, args) -> dict:
    """Build the onnxruntime feed dict, restricted to the input names the
    exported graph actually declares. Some inputs (e.g. SASRec's
    `position_ids`) are used ONLY inside a validation assert whose result
    never reaches the traced output (`_layout`'s `bad.item()` check derives
    real positions from `lengths`/`attention_mask`, not from `position_ids`);
    the legacy exporter's dead-code elimination then drops that input from the
    ONNX graph entirely, so re-feeding it by name raises
    'Invalid input name'. This is a genuine export finding (recorded in the
    report), not a bug in this harness -- we just must not feed dropped
    inputs back in."""
    declared = {i.name for i in sess.get_inputs()}
    dropped = [n for n in names if n not in declared]
    feed = {n: a.numpy() for n, a in zip(names, args) if n in declared}
    return feed, dropped


def onnxruntime_session(path: str, intra_op_threads: int = 4):
    import time as time_mod

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_op_threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time_mod.perf_counter()
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    load_s = time_mod.perf_counter() - t0
    return sess, load_s


def time_session(sess, feed: dict, warm: int = 5, reps: int = 50) -> dict:
    import time as time_mod
    for _ in range(warm):
        sess.run(None, feed)
    xs = []
    for _ in range(reps):
        t0 = time_mod.perf_counter()
        sess.run(None, feed)
        xs.append((time_mod.perf_counter() - t0) * 1e3)
    xs.sort()
    n = len(xs)
    p50 = xs[n // 2]
    p95 = xs[min(n - 1, int(0.95 * (n - 1)))]
    return {"p50_ms": p50, "p95_ms": p95, "n": n, "min_ms": xs[0], "max_ms": xs[-1]}


PARITY_SCENARIOS = [
    ("len1_b1", 1, 1, False),
    ("len1_b16", 1, 16, False),
    ("len8_b1", 8, 1, False),
    ("len8_b16", 8, 16, False),
    ("len50_b1", 50, 1, False),
    ("len50_b16", 50, 16, False),
    ("missing_b4_len8", 8, 4, True),
    ("mixed_pad_b16", "mixed", 16, False),
]

def onnx_total_size_mb(path: str) -> float:
    """Sum the main .onnx file plus any external-data companion files PyTorch
    may have written alongside it (e.g. <name>.onnx.data / <name>_data)."""
    p = Path(path)
    if not p.exists():
        return 0.0
    total = p.stat().st_size
    for sib in p.parent.glob(p.name + "*"):
        if sib != p:
            total += sib.stat().st_size
    return total / (1024 * 1024)


LATENCY_SCENARIOS = [
    ("len1_b1", 1, 1),
    ("len1_b16", 1, 16),
    ("len8_b1", 8, 1),
    ("len8_b16", 8, 16),
    ("len50_b1", 50, 1),
    ("len50_b16", 50, 16),
]

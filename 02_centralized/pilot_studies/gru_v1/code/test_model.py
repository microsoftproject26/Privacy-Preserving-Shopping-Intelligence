"""Tests for ContextGRU-v1 (Task T-A), per the brief's "TESTS TO RUN" section.

Run with the LOCAL python from `.local_ai\\gru_v1`:
    <local python> -m pytest code\\test_model.py -v
"""
from __future__ import annotations

import re

import numpy as np
import pytest
import torch
from torch import nn

from model import ContextGRU, build_model, parameter_inventory

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ALL_CATEGORICAL = (
    "category_tokens", "brand_tokens", "main_category_tokens",
    "daypart_tokens", "weekday_tokens",
)
TIMESTEP_KEYS = (
    "item_tokens", "category_tokens", "brand_tokens", "main_category_tokens",
    "daypart_tokens", "weekday_tokens", "event_numeric_features", "event_quality_flags",
)
ENCODE_KEYS = ("item_tokens", "lengths", "attention_mask") + TIMESTEP_KEYS[1:]


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


def make_batch(rng, B, L, vocab_sizes, lengths, *, categorical=(), numeric_dim=0,
               flags_dim=0, user_context=False, left_pad=False):
    """Build a synthetic right-padded (or left-padded) batch as a dict of torch
    tensors, using named keys matching ContextGRU.forward's signature.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    assert lengths.shape == (B,)

    def fill_tokens(vocab_size):
        arr = np.zeros((B, L), dtype=np.int64)
        for i, ln in enumerate(lengths):
            sl = slice(L - int(ln), L) if left_pad else slice(0, int(ln))
            arr[i, sl] = rng.integers(3, vocab_size, size=int(ln))
        return arr

    batch: dict[str, np.ndarray] = {"item_tokens": fill_tokens(vocab_sizes["item_tokens"])}
    for ch in categorical:
        batch[ch] = fill_tokens(vocab_sizes[ch])

    attn = np.zeros((B, L), dtype=bool)
    for i, ln in enumerate(lengths):
        if left_pad:
            attn[i, L - int(ln):] = True
        else:
            attn[i, :int(ln)] = True
    batch["attention_mask"] = attn
    batch["lengths"] = lengths

    if numeric_dim:
        num = np.zeros((B, L, numeric_dim), dtype=np.float32)
        for i, ln in enumerate(lengths):
            sl = slice(L - int(ln), L) if left_pad else slice(0, int(ln))
            num[i, sl] = rng.standard_normal((int(ln), numeric_dim)).astype(np.float32)
        batch["event_numeric_features"] = num
    if flags_dim:
        flg = np.zeros((B, L, flags_dim), dtype=np.int8)
        for i, ln in enumerate(lengths):
            sl = slice(L - int(ln), L) if left_pad else slice(0, int(ln))
            flg[i, sl] = rng.integers(0, 2, size=(int(ln), flags_dim)).astype(np.int8)
        batch["event_quality_flags"] = flg
    if user_context:
        batch["user_context"] = rng.standard_normal((B, 12)).astype(np.float32)
        batch["user_context_masks"] = rng.integers(0, 2, size=(B, 6)).astype(np.float32)

    return {k: torch.from_numpy(v) for k, v in batch.items()}


def _build_f4_fixture(rng, K=20, B=4, L=50):
    vocab = dict(item_tokens=K + 3, category_tokens=8, brand_tokens=6,
                 main_category_tokens=4, daypart_tokens=4, weekday_tokens=5)
    poc = np.arange(K) + 3
    model = build_model("F4_rich_side_information", vocab, product_idx_of_class=poc)
    lengths = np.array([1, L, 17, 33], dtype=np.int64)[:B]
    batch = make_batch(rng, B, L, vocab, lengths, categorical=ALL_CATEGORICAL,
                        numeric_dim=19, flags_dim=15, user_context=True)
    return model, batch, lengths, L


def _slice_row(batch, i, ln, timestep_keys):
    per_timestep = set(timestep_keys) | {"attention_mask"}
    row = {}
    for k, v in batch.items():
        if k == "lengths":
            continue
        if k in per_timestep:
            row[k] = v[i:i + 1, :ln]
        else:
            row[k] = v[i:i + 1]
    row["lengths"] = batch["lengths"][i:i + 1]
    return row


def _per_row_reference(model, encode_batch, lengths):
    rows = []
    for i in range(lengths.shape[0]):
        row = _slice_row(encode_batch, i, int(lengths[i]), TIMESTEP_KEYS)
        with torch.no_grad():
            rows.append(model.encode(**row)[0])
    return torch.stack(rows)


def _trailing_pad_encode(model, encode_batch):
    """Deliberately broken readout: fixed last time index L-1 instead of
    each row's own `lengths-1` (via h_n[-1]). Broken per FABLE_PLAN T1."""
    kwargs = {k: v for k, v in encode_batch.items() if k not in ("lengths", "attention_mask")}
    x = model._embed_and_project(**kwargs)
    lengths = encode_batch["lengths"]
    packed = nn.utils.rnn.pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed_out, _ = model.gru(packed)
    output, _ = nn.utils.rnn.pad_packed_sequence(
        packed_out, batch_first=True, total_length=x.shape[1])
    return output[:, -1]


def _left_pad(encode_batch, lengths, L):
    """Re-lay the same per-timestep content out as LEFT-padded, keeping the
    same `lengths` tensor -- the second deliberately-broken T1 variant."""
    out = dict(encode_batch)
    for k in TIMESTEP_KEYS:
        if k not in encode_batch:
            continue
        arr = encode_batch[k]
        shifted = torch.zeros_like(arr)
        for i, ln in enumerate(lengths):
            ln = int(ln)
            shifted[i, L - ln:L] = arr[i, :ln]
        out[k] = shifted
    attn = torch.zeros_like(encode_batch["attention_mask"])
    for i, ln in enumerate(lengths):
        attn[i, L - int(ln):L] = True
    out["attention_mask"] = attn
    out["lengths"] = encode_batch["lengths"]
    return out


# ---------------------------------------------------------------------------
# 1. Shapes
# ---------------------------------------------------------------------------

def test_shapes_f4_returns_2d(rng):
    K = 1000
    vocab = dict(item_tokens=K + 3, category_tokens=20, brand_tokens=15,
                 main_category_tokens=8, daypart_tokens=8, weekday_tokens=10)
    poc = np.arange(K) + 3
    model = build_model("F4_rich_side_information", vocab, product_idx_of_class=poc)
    model.eval()

    B, L = 8, 50
    lengths = np.array([1, 50, 25, 2, 49, 10, 33, 50], dtype=np.int64)
    batch = make_batch(rng, B, L, vocab, lengths, categorical=ALL_CATEGORICAL,
                        numeric_dim=19, flags_dim=15, user_context=True)
    with torch.no_grad():
        logits = model(**batch)

    assert logits.shape == (B, K)
    assert logits.dim() == 2, "output must be [B,K], never [B,L,K]"


# ---------------------------------------------------------------------------
# 2. T4 -- class-map / tied-head candidates
# ---------------------------------------------------------------------------

def test_t4_class_map_contiguous_vs_permuted(rng):
    K = 40
    vocab = dict(item_tokens=K + 3)
    spec = {"categorical": ["item_tokens"], "numeric_dim": 0, "flags_dim": 0,
            "user_context": False}

    poc_contig = np.arange(K) + 3
    model_c = ContextGRU(spec, vocab, product_idx_of_class=poc_contig)
    assert model_c.contiguous_head is True

    perm = rng.permutation(K)
    poc_perm = (perm + 3).astype(np.int64)
    model_p = ContextGRU(spec, vocab, product_idx_of_class=poc_perm)
    assert model_p.contiguous_head is False
    model_p.eval()

    B, L = 4, 6
    lengths = np.array([1, 6, 3, 6], dtype=np.int64)
    batch = make_batch(rng, B, L, vocab, lengths)

    with torch.no_grad():
        logits = model_p(**batch)
        q = model_p.query(item_tokens=batch["item_tokens"], lengths=batch["lengths"],
                           attention_mask=batch["attention_mask"])
        w = model_p.item_embed.weight[torch.from_numpy(poc_perm).long()]
        brute = q @ w.T + model_p.output_bias

    assert w.shape == (K, 128)
    assert not torch.equal(w[0], model_p.item_embed.weight[0])  # sanity: not row PAD
    max_delta = torch.max(torch.abs(logits - brute)).item()
    assert max_delta <= 1e-5


# ---------------------------------------------------------------------------
# 3. T5 -- tied gradient + parameter inventory
# ---------------------------------------------------------------------------

def test_t5_tied_gradient_and_inventory(rng):
    K = 50
    vocab = dict(item_tokens=K + 3, category_tokens=12, brand_tokens=9,
                 main_category_tokens=5, daypart_tokens=5, weekday_tokens=6)
    poc = np.arange(K) + 3
    model = build_model("F4_rich_side_information", vocab, product_idx_of_class=poc)
    model.train()

    B, L = 6, 10
    lengths = np.array([10, 1, 5, 10, 3, 7], dtype=np.int64)
    batch = make_batch(rng, B, L, vocab, lengths, categorical=ALL_CATEGORICAL,
                        numeric_dim=19, flags_dim=15, user_context=True)
    target = torch.from_numpy(rng.integers(0, K, size=B).astype(np.int64))

    logits = model(**batch)
    loss = torch.nn.functional.cross_entropy(logits, target)
    model.zero_grad()
    loss.backward()

    grad = model.item_embed.weight.grad
    assert grad is not None
    t0 = int(target[0].item())
    assert torch.any(grad[3 + t0] != 0).item(), "gradient must reach the tied class row"
    assert torch.all(grad[0] == 0), "PAD row (index 0) must receive exactly zero gradient"

    item_shape = (vocab["item_tokens"], 128)
    item_params = [p for p in model.parameters() if tuple(p.shape) == item_shape]
    assert len(item_params) == 1, "exactly one item Parameter of shape [item_rows,128]"

    inv = parameter_inventory(model)
    print("parameter_inventory (T5 fixture):", inv)
    assert inv["total"] == sum(p.numel() for p in model.parameters())
    assert inv["total"] > 0


# ---------------------------------------------------------------------------
# 4. T6 -- feature-group exclusion (F0)
# ---------------------------------------------------------------------------

def test_t6_feature_group_exclusion_f0(rng):
    K = 30
    vocab_full = dict(item_tokens=K + 3, category_tokens=9, brand_tokens=7,
                       main_category_tokens=4, daypart_tokens=5, weekday_tokens=6)
    poc = np.arange(K) + 3
    model = build_model("F0_item_only", vocab_full, product_idx_of_class=poc)
    model.eval()

    sd = model.state_dict()
    forbidden = re.compile(r"category|brand|main|daypart|weekday|user|numeric|flags", re.IGNORECASE)
    bad = [k for k in sd if forbidden.search(k)]
    assert bad == [], f"F0 state_dict leaks non-item params: {bad}"

    B, L = 5, 12
    lengths = np.array([12, 1, 6, 12, 4], dtype=np.int64)
    batch = make_batch(rng, B, L, vocab_full, lengths, categorical=ALL_CATEGORICAL,
                        numeric_dim=19, flags_dim=15, user_context=True)

    base_inputs = {"item_tokens": batch["item_tokens"], "lengths": batch["lengths"],
                   "attention_mask": batch["attention_mask"]}
    with torch.no_grad():
        out1 = model(**base_inputs)

    perm = torch.from_numpy(rng.permutation(B))
    extra_keys = ALL_CATEGORICAL + ("event_numeric_features", "event_quality_flags",
                                     "user_context", "user_context_masks")
    permuted_extras = {k: batch[k][perm] for k in extra_keys}

    full_inputs = dict(base_inputs)
    full_inputs.update(permuted_extras)  # F0 must never read these
    with torch.no_grad():
        out2 = model(**full_inputs)

    assert torch.equal(out1, out2), "F0 output must be bit-identical to permuting non-item channels"


# ---------------------------------------------------------------------------
# 5. T1 -- padding invariance (+ deliberately-broken xfail variants)
# ---------------------------------------------------------------------------

def test_t1_padding_invariance(rng):
    torch.backends.cuda.matmul.allow_tf32 = False
    model, batch, lengths, L = _build_f4_fixture(rng)
    model.eval()

    encode_batch = {k: batch[k] for k in ENCODE_KEYS if k in batch}
    with torch.no_grad():
        h_batched = model.encode(**encode_batch)
    ref = _per_row_reference(model, encode_batch, lengths)

    max_delta = torch.max(torch.abs(h_batched - ref)).item()
    assert max_delta <= 1e-5


@pytest.mark.xfail(strict=True, reason="trailing-PAD readout (output[:, L-1]) must diverge "
                                        "from the correct per-row reference for short rows")
def test_t1_broken_trailing_pad_readout(rng):
    torch.backends.cuda.matmul.allow_tf32 = False
    model, batch, lengths, L = _build_f4_fixture(rng)
    model.eval()

    encode_batch = {k: batch[k] for k in ENCODE_KEYS if k in batch}
    ref = _per_row_reference(model, encode_batch, lengths)
    with torch.no_grad():
        h_broken = _trailing_pad_encode(model, encode_batch)

    max_delta = torch.max(torch.abs(h_broken - ref)).item()
    assert max_delta <= 1e-5


@pytest.mark.xfail(strict=True, reason="left-padded input packed with right-padding lengths "
                                        "must diverge from the correct per-row reference")
def test_t1_broken_left_padded_input(rng):
    torch.backends.cuda.matmul.allow_tf32 = False
    model, batch, lengths, L = _build_f4_fixture(rng)
    model.eval()

    encode_batch = {k: batch[k] for k in ENCODE_KEYS if k in batch}
    ref = _per_row_reference(model, encode_batch, lengths)
    left_batch = _left_pad(encode_batch, lengths, L)
    with torch.no_grad():
        h_broken = model.encode(**left_batch)

    max_delta = torch.max(torch.abs(h_broken - ref)).item()
    assert max_delta <= 1e-5

"""v4dev.scoring — the fixed-signature, single-decision scoring wrapper (DEV-01 item 1).

`ScoringModule` wraps a `v4models.build.BuiltModel` (module + FL adapter) into a plain `nn.Module` whose `forward`
takes one positional tensor per input key (a fixed order, `input_keys`) and returns `[B, K]` scores:
    q = adapter.query(dict(zip(input_keys, tensors)))
    scores = adapter.logits(q)                      # BaseAdapter.logits: q @ W.t() (+ bias), the whole catalogue
This is bit-for-bit `ranking.block_scores(q, W, b, 0, K)` with `score_block == K` — the frozen T3 single-block rule
(v4eval/ranking.py), so an ONNX export of this graph is the same scoring function T3 evaluates, not a proxy of it.

`input_keys(family, module)` mirrors the exact per-family kwarg selection the reviewed FL adapters use (never
duplicated ad hoc): GRU via `train.build_named_kwargs` order (item_tokens, lengths, attention_mask, then
`CATEGORICAL_CHANNELS` in order, then numeric / flags / user-context iff the module was built with them), SASRec via
`fl.families.SASREC_KEYS` (its adapter takes every key present, unfiltered). Both are introspected from the built
module. DEV-04: a derived-schema / SCHEMA-01R1-extension model (e.g. C_SMALL) takes the FULL Clean V2 batch its
adapter takes (the projection to the kept channels runs inside the graph, `adapter.project`), minus the keys the
projection removes, plus its extension keys (`extension_input_keys`: extension tokens, then the decision-level
extension floats); a base-schema model's signature is unchanged.

Batch shape: right-padded windows of the model's own `max_len` (v4models.synthetic.L_MAX = 50), exactly the Clean V2
tensor layout every other V2 path uses — "the model's own input features for one decision" (DEV-01 item 1) is this
window, not a single event. A fixed batch of 1 and a fixed window width give ONNX a fully static shape (no
`dynamic_axes`): this is the "fixed batch 1 inference signature" the task calls for.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

import sys as _sys

import v4models  # noqa: F401  (side effect: imports + registers the v4models.build SUBMODULE in sys.modules)
from v4models.synthetic import L_MAX, synthetic_examples_v2

# v4models/__init__.py does `from .build import ..., build, ...`, which REBINDS the package attribute `v4models.build`
# to the `build(...)` function it re-exports; the submodule object is still registered under its own key and is only
# reachable there (`import v4models.build as x` resolves via getattr on the package too, and would give the function).
reuse_train = _sys.modules["v4models.build"].reuse_train   # the snapshot train.py module (CATEGORICAL_CHANNELS)


class ScoringError(ValueError):
    pass


def gru_input_keys(module, projection=None) -> Tuple[str, ...]:
    """`train.build_named_kwargs` order for a ContextGRU-family module (v4models.gru_v2.ContextGRU / ContextGRUD), then
    the SCHEMA-01R1 extension inputs of a derived-schema model (`extension_input_keys`)."""
    keys = ["item_tokens", "lengths", "attention_mask"]
    for ch in reuse_train.CATEGORICAL_CHANNELS:
        if ch in module._categorical_channels:
            keys.append(ch)
    if int(module.numeric_dim) > 0:
        keys.append("event_numeric_features")
    if int(module.flags_dim) > 0:
        keys.append("event_quality_flags")
    if bool(module.use_user_context):
        keys += ["user_context", "user_context_masks"]
    return tuple(keys) + extension_input_keys(module, projection)


def extension_input_keys(module, projection=None) -> Tuple[str, ...]:
    """DEV-04: the SCHEMA-01R1 extension batch keys a derived-schema model consumes, AFTER the base keys: its
    extension token channels (`v4models.build.extension_tokens_of`, the adapter's own `_with_extensions` set) and
    the decision-level extension floats its projection appends to user_context / user_context_masks
    (`ChannelProjection.ext_columns`). Empty for every base-schema model (the DEV-01 signature is unchanged)."""
    keys = list(_sys.modules["v4models.build"].extension_tokens_of(module))
    if projection is not None:
        for blk in ("user_context", "user_context_masks"):
            keys += [c for c in (projection.ext_columns.get(blk) or ()) if c not in keys]
    return tuple(keys)


def _projected_out(projection) -> set:
    """Base batch keys a derived projection removes (dropped token channels, a whole float block pair the model has
    no branch for): the model never reads them, so they are not part of its scoring signature."""
    if projection is None:
        return set()
    out = set(projection.tokens_dropped)
    if not projection.event_blocks_on:
        out |= {"event_numeric_features", "event_quality_flags"}
    if not projection.user_blocks_on:
        out |= {"user_context", "user_context_masks"}
    return out


def sasrec_input_keys(module, projection=None) -> Tuple[str, ...]:
    """`fl.families.SASREC_KEYS`, in order (the SASRec adapter takes every one of these present in the batch), minus
    the keys a derived projection removes, then the extension inputs (`extension_input_keys`)."""
    from fl.families import SASREC_KEYS
    drop = _projected_out(projection)
    return tuple(k for k in SASREC_KEYS if k not in drop) + extension_input_keys(module, projection)


def input_keys(family: str, module, projection=None) -> Tuple[str, ...]:
    fam = str(family).upper()
    if fam == "GRU":
        return gru_input_keys(module, projection)
    if fam == "SASREC":
        return sasrec_input_keys(module, projection)
    raise ScoringError(f"unknown family {family!r}; expected GRU or SASREC")


_GRU_EMBED_KEYS = ("item_tokens", "category_tokens", "brand_tokens", "main_category_tokens", "daypart_tokens",
                  "weekday_tokens", "session_offset_tokens", "session_mode_main_category_tokens",
                  "event_numeric_features", "event_quality_flags")


def gru_dense_query(module, batch: Mapping[str, Tensor]) -> Tensor:
    """A pack-free reimplementation of `ContextGRU.query` (v4models/gru_v2.py), numerically equivalent and ONNX-
    traceable — v4models is out of DEV-01's scope, so this lives in v4dev instead of a code change there.

    Why: `ContextGRU.encode` feeds `nn.utils.rnn.pack_padded_sequence(x, lengths, ...)` into `self.gru` and reads
    only the final hidden state (`h_n[-1]`; the packed OUTPUT sequence is never unpacked). Both ONNX exporters refuse
    this: the classic TorchScript exporter's RNN peephole fusion assumes a pack/unpack PAIR (undocumented; it fails
    deep in `_C._jit_pass_onnx_peephole` with "invalid vector subscript", or, with `input_names` given, a clean
    "Cannot export individual pack_padded_sequence ... must occur in pairs"), and `torch.export` (the `dynamo=True`
    path) fails tracing `nn.GRU`'s internal packed-input branch with a `FakeTensor` shape error. Neither is a v4dev
    bug: PyTorch's ONNX/export tooling does not support "pack, never unpack" RNN usage.

    Equivalence: `nn.GRU`'s recurrence is strictly causal and per-row independent of the batch dimension, so the
    hidden state at any real timestep `t <= lengths-1` is bit-for-bit the same whether the row is truncated there
    (packed) or continues over trailing zero-padded rows (dense) — packing only skips computing timesteps whose
    result is discarded anyway. This calls `module`'s OWN `_embed_and_project` / `gru` / `user_ctx_proj` /
    `readout_proj` / `readout_dropout` (the checkpoint's real weights, no reimplementation of their math), runs the
    GRU densely (no pack/unpack) over the full window, and gathers each row's own `lengths - 1` timestep instead of
    the encode() docstring's forbidden `output[:, L-1]`. Checked against `ContextGRU.query` directly in
    tests_dev/test_scoring.py (parametrized over both families; GRU is checked to float32 tolerance, not
    `torch.equal`, since the dense and packed cuDNN/ATen kernels are not guaranteed bit-identical)."""
    # `batch` already carries exactly this model's required keys (ScoringModule.forward / batch_to_inputs build it
    # from `input_keys`, computed by `gru_input_keys` from these same module attributes), so a plain presence filter
    # against `_GRU_EMBED_KEYS` is exactly `_embed_and_project`'s own required set -- no re-derivation needed here.
    kw = {k: batch[k] for k in _GRU_EMBED_KEYS if k in batch}
    x = module._embed_and_project(**kw)                     # [B, L, d]
    output, _ = module.gru(x)                                # dense (unpacked): zero initial state, no state carry
    lengths = batch["lengths"]
    # `.clamp(min=0)` is defensive ONLY (never lets a length-0 row IndexError the gather below): it must never be the
    # thing that decides a length-0 or non-right-padded row is acceptable. Real `--rows` decisions are validated
    # (refused, not clamped) by `v4dev.bench.validate_rows_layout` before they ever reach this function; synthetic
    # rows (v4models.synthetic.synthetic_examples_v2) already guarantee `lengths >= 1` by construction.
    idx = (lengths.long() - 1).clamp(min=0)
    h_last = output[torch.arange(output.shape[0], device=output.device), idx]     # [B, d] -- row's own last step

    if module.user_ctx_proj is not None:
        uc_in = torch.cat([batch["user_context"], batch["user_context_masks"]], dim=-1)
        uc = module.user_ctx_proj(uc_in)
        readout_in = torch.cat([h_last, uc], dim=-1)
    else:
        readout_in = h_last
    q = module.readout_proj(readout_in)
    q = module.readout_dropout(q)
    return q


class ScoringModule(nn.Module):
    """One decision's own input features -> `[B, K]` scores (DEV-01 item 1). `built` is a v4models.build.BuiltModel;
    the wrapped module and its parameters are registered as-is (no copies), so this graph scores with exactly the
    checkpoint's weights. GRU uses `gru_dense_query` (ONNX-traceable; see its docstring); SASRec uses the real
    `adapter.query` (SASRecCE has no pack/unpack step) — `v4dev.checkpointio.build_scoring_module` sets its `impl`
    to `"padded_reference"` for export (the mask-based path; `"unpadded"`'s Python-level `flat_idx` gather is not
    ONNX-safe either, though it never crashes -- both `impl`s are the same weights, verified numerically equivalent
    by v4models' own tests)."""

    def __init__(self, built) -> None:
        super().__init__()
        self.built_family = str(built.family).upper()
        self.module = built.module               # registers parameters/buffers for tracing + state_dict
        self.adapter = built.adapter              # plain object (query / head_weight / head_bias / logits)
        self.K = int(built.K)
        self.projection = getattr(built, "projection", None)
        self.input_keys: Tuple[str, ...] = input_keys(self.built_family, built.module, self.projection)
        if self.projection is not None and not self.projection.is_identity:
            # DEV-04: ChannelProjection creates its column-index tensors lazily and caches them; a first trace would
            # record them as freshly built constants and a second trace as captured tensors, so two exports of the
            # same model would differ byte-wise (v4dev.export's determinism check). Build them once, here, up front.
            for key in self.projection.index:
                self.projection._idx(key, torch.device("cpu"))

    def forward(self, *tensors: Tensor) -> Tensor:
        if len(tensors) != len(self.input_keys):
            raise ScoringError(f"expected {len(self.input_keys)} inputs {self.input_keys}, got {len(tensors)}")
        batch = dict(zip(self.input_keys, tensors))
        if self.built_family == "GRU":                       # the adapter's own check + projection, then dense GRU
            q = gru_dense_query(self.module, self.adapter.project(batch))
        else:
            q = self.adapter.query(batch)                    # SASRecV2Adapter.query projects itself
        return self.adapter.logits(q)


def example_batch(built, n: int = 1, *, seed: int = 20260926, min_len: int = 1, max_len: int = L_MAX) -> dict:
    """A synthetic Clean-V2-layout batch of `n` decisions sized for `built` (its widths and vocab_sizes) — used for
    ONNX export tracing (n = 1) and for the bench's synthetic latency/parity queries (n = 1000).
    DEV-04: the batch is the FULL Clean V2 batch the adapter consumes (`adapter.input_widths`: the base widths for a
    derived-schema model, whose adapter projects it), plus the SCHEMA-01R1 extension keys when the model keeps any
    (drawn after every other tensor, so a base-schema batch is unchanged)."""
    gen = torch.Generator().manual_seed(int(seed))
    widths = getattr(built.adapter, "input_widths", None) or built.widths
    ext = set(extension_input_keys(built.module, getattr(built, "projection", None)))
    intent = bool(ext - {"session_offset_tokens"})
    return synthetic_examples_v2(int(n), int(built.K), widths, gen, vocab=dict(built.vocab_sizes),
                                 max_len=max_len, min_len=min_len, window=max_len,
                                 with_session_offset=bool(ext), with_session_intent=intent)


def example_inputs(built, scoring: ScoringModule, n: int = 1, *, seed: int = 20260926,
                   min_len: int = 1) -> Tuple[Tensor, ...]:
    """The ordered tensor tuple `ScoringModule.forward` takes, from a fresh synthetic batch of `n` decisions.
    `min_len` (default 1, realistic varying lengths — bench's synthetic latency/parity queries): `v4dev.export`
    always traces with `min_len = L_MAX` (every row full-length) instead, so `SASRecCE._layout`'s data-dependent
    `T = lengths.max()` bakes to the SAFE, maximal T = L_MAX at trace time — the exported graph then never truncates
    below the full static window, so it stays correct for any real (shorter) length fed at actual inference (the
    validity/causal masks, real tensor inputs, still restrict attention/output to each row's own real length)."""
    batch = example_batch(built, n, seed=seed, min_len=min_len)
    return tuple(batch[k] for k in scoring.input_keys)


def batch_to_inputs(scoring: ScoringModule, batch: Mapping[str, Tensor]) -> Tuple[Tensor, ...]:
    """A real (or synthetic) Clean-V2 batch dict -> the ordered tensor tuple `scoring.input_keys` needs."""
    missing = [k for k in scoring.input_keys if k not in batch]
    if missing:
        raise ScoringError(f"batch is missing this model's input keys: {missing}")
    return tuple(batch[k] for k in scoring.input_keys)


__all__ = ["ScoringError", "ScoringModule", "input_keys", "gru_input_keys", "sasrec_input_keys", "extension_input_keys",
           "example_batch",
           "example_inputs", "batch_to_inputs"]

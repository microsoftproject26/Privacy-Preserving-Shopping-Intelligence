"""The fixed-signature, single-decision scoring wrapper that is exported to ONNX.

`ScoringModule` wraps a `ppsi.seqrec.build.BuiltModel` (module + adapter) into a plain `nn.Module` whose `forward`
takes one positional tensor per input key (a fixed order, `input_keys`) and returns `[B, K]` scores:
    q = adapter.query(dict(zip(input_keys, tensors)))
    scores = adapter.logits(q)                      # q @ W.t() (+ bias) over the whole catalogue
This is bit for bit the evaluator's single-block scoring (ppsi.evaluation.fullrank.ranking.block_scores with
score_block >= K), so an ONNX export of this graph is the same scoring function the evaluator measures, not a proxy.

`input_keys(family, module)` mirrors the adapters' own per-family kwarg selection: GRU in `features.gru_kwargs`
order (item_tokens, lengths, attention_mask, then the side token channels in order, then numeric / flags /
user-context iff the module was built with them), SASRec in `features.SASREC_KEYS` order.

Batch shape: right-padded windows of the model's own max_len (50), the same layout every other path uses. A fixed
batch of 1 and a fixed window width give ONNX a fully static shape (no dynamic axes).
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ppsi.seqrec.features import SASREC_KEYS, SIDE_TOKEN_CHANNELS, check_batch_layout
from ppsi.seqrec.synthetic import L_MAX, synthetic_batch


class ScoringError(ValueError):
    pass


def gru_input_keys(module) -> tuple[str, ...]:
    """`features.gru_kwargs` order for a ContextGRU."""
    keys = ["item_tokens", "lengths", "attention_mask"]
    for ch in SIDE_TOKEN_CHANNELS:
        if ch in module._categorical_channels:
            keys.append(ch)
    if int(module.numeric_dim) > 0:
        keys.append("event_numeric_features")
    if int(module.flags_dim) > 0:
        keys.append("event_quality_flags")
    if bool(module.use_user_context):
        keys += ["user_context", "user_context_masks"]
    return tuple(keys)


def sasrec_input_keys(module) -> tuple[str, ...]:
    """`features.SASREC_KEYS`, in order (the SASRec adapter takes every one of these present in the batch)."""
    return tuple(SASREC_KEYS)


def input_keys(family: str, module) -> tuple[str, ...]:
    fam = str(family).upper()
    if fam == "GRU":
        return gru_input_keys(module)
    if fam == "SASREC":
        return sasrec_input_keys(module)
    raise ScoringError(f"unknown family {family!r}; expected GRU or SASREC")


_GRU_EMBED_KEYS = ("item_tokens", "category_tokens", "brand_tokens", "main_category_tokens", "daypart_tokens",
                   "weekday_tokens", "event_numeric_features", "event_quality_flags")


def gru_dense_query(module, batch: Mapping[str, Tensor]) -> Tensor:
    """A pack-free, ONNX-traceable equivalent of `ContextGRU.query`.

    Why: `ContextGRU.encode` feeds `pack_padded_sequence(x, lengths, ...)` into the GRU and reads only the final
    hidden state (the packed output sequence is never unpacked). Both ONNX exporters refuse this: the TorchScript
    exporter's RNN fusion expects a pack / unpack PAIR, and `torch.export` fails tracing the GRU's packed-input branch.

    Equivalence: the GRU recurrence is strictly causal and independent per row, so the hidden state at a real
    timestep t <= lengths-1 is the same whether the row is truncated there (packed) or continues over trailing
    zero-padded steps (dense); packing only skips steps whose result is discarded anyway. This calls the module's
    OWN `_embed_and_project` / `gru` / `user_ctx_proj` / `readout_proj` / `readout_dropout` (the real weights), runs
    the GRU densely over the full window, and gathers each row's own `lengths - 1` step. Checked against
    `ContextGRU.query` in the tests (to float32 tolerance: dense and packed kernels need not be bit-identical)."""
    kw = {k: batch[k] for k in _GRU_EMBED_KEYS if k in batch}
    x = module._embed_and_project(**kw)                     # [B, L, d]
    output, _ = module.gru(x)                                # dense (unpacked): zero initial state, no state carry
    lengths = batch["lengths"]
    # `.clamp(min=0)` is defensive only (a length-0 row must not IndexError the gather): real rows are validated
    # (refused, not clamped) by bench.validate_rows_layout before they reach this function, and synthetic rows
    # always have lengths >= 1.
    idx = (lengths.long() - 1).clamp(min=0)
    h_last = output[torch.arange(output.shape[0], device=output.device), idx]     # [B, d]: the row's own last step
    if module.user_ctx_proj is not None:
        uc = module.user_ctx_proj(torch.cat([batch["user_context"], batch["user_context_masks"]], dim=-1))
        readout_in = torch.cat([h_last, uc], dim=-1)
    else:
        readout_in = h_last
    q = module.readout_proj(readout_in)
    return module.readout_dropout(q)


class ScoringModule(nn.Module):
    """One decision's input features -> `[B, K]` scores. The wrapped module and its parameters are registered as-is
    (no copies), so this graph scores with exactly the loaded weights. GRU uses `gru_dense_query` (ONNX-traceable);
    SASRec uses the real `adapter.query` with `impl = "padded_reference"` (set by io.build_scoring_module)."""

    def __init__(self, built) -> None:
        super().__init__()
        self.built_family = str(built.family).upper()
        self.module = built.module               # registers parameters / buffers for tracing + state_dict
        self.adapter = built.adapter              # plain object (query / head_weight / head_bias / logits)
        self.K = int(built.K)
        self.input_keys: tuple[str, ...] = input_keys(self.built_family, built.module)

    def forward(self, *tensors: Tensor) -> Tensor:
        if len(tensors) != len(self.input_keys):
            raise ScoringError(f"expected {len(self.input_keys)} inputs {self.input_keys}, got {len(tensors)}")
        batch = dict(zip(self.input_keys, tensors, strict=True))
        if self.built_family == "GRU":                       # the adapter's own layout check, then the dense GRU
            check_batch_layout(batch, self.adapter.input_widths)
            q = gru_dense_query(self.module, batch)
        else:                                                # SASRecAdapter.query checks the layout itself
            q = self.adapter.query(batch)
        return self.adapter.logits(q)


def example_batch(built, n: int = 1, *, seed: int = 20260926, min_len: int = 1, max_len: int = L_MAX) -> dict:
    """A synthetic batch of `n` decisions sized for `built` (its widths and vocab sizes): the export trace (n = 1)
    and the bench's synthetic latency / parity queries."""
    gen = torch.Generator().manual_seed(int(seed))
    return synthetic_batch(int(n), int(built.K), built.widths, gen, vocab=dict(built.vocab_sizes),
                           max_len=max_len, min_len=min_len, window=max_len)


def example_inputs(built, scoring: ScoringModule, n: int = 1, *, seed: int = 20260926,
                   min_len: int = 1) -> tuple[Tensor, ...]:
    """The ordered tensor tuple `ScoringModule.forward` takes, from a fresh synthetic batch of `n` decisions.
    The export always traces with `min_len = L_MAX` (every row full-length): SASRec's layout check bakes the
    data-dependent `T = lengths.max()` at trace time, and tracing at the maximal T keeps the graph valid for any
    shorter real length at inference (the validity and causal masks still restrict each row to its own events)."""
    batch = example_batch(built, n, seed=seed, min_len=min_len)
    return tuple(batch[k] for k in scoring.input_keys)


def batch_to_inputs(scoring: ScoringModule, batch: Mapping[str, Tensor]) -> tuple[Tensor, ...]:
    """A batch dict -> the ordered tensor tuple `scoring.input_keys` needs."""
    missing = [k for k in scoring.input_keys if k not in batch]
    if missing:
        raise ScoringError(f"batch is missing this model's input keys: {missing}")
    return tuple(batch[k] for k in scoring.input_keys)


__all__ = ["ScoringError", "ScoringModule", "batch_to_inputs", "example_batch", "example_inputs", "gru_dense_query",
           "gru_input_keys", "input_keys", "sasrec_input_keys"]

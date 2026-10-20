"""gru_export_adapter.py (owner: Sonnet, EDGE-02) -- pack-free export adapter for
ContextGRU (F4_rich_side_information).

EDGE-01 found that ContextGRU.encode()'s
`nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)`
cannot be exported to ONNX under either exporter mode in this torch build
(g0/export_smoke/GRU_REPORT.json): the legacy exporter's own ONNX peephole
pass crashes on the sort/scatter/restore ops that PackedSequence's
`enforce_sorted=False` path needs. This module works around that at
INFERENCE/EXPORT time only -- it never touches `snap_v1` and never retrains
or re-initialises anything.

Design (task EDGE-02.md): a thin wrapper `nn.Module` holding a REFERENCE to an
unmodified `ContextGRU` (no weights copied -- `run_gru_adapter.py` verifies
this with `data_ptr()` equality between every adapter parameter and its
`ContextGRU` counterpart, and with an `is`-identity check on the wrapped
module itself). It re-implements only `encode()`:

  1. the SAME `_embed_and_project` call (identical embedding/projection/
     dropout path, eval mode) as `ContextGRU.encode()`;
  2. the SAME `model.gru` `nn.GRU` module (not a new one), run over the FULL
     right-padded `[B, 50, *]` tensor at once, zero initial state (h0
     omitted, matching `ContextGRU`'s own construction comment);
  3. `h_last = out.gather(1, (lengths - 1))`.

Why this equals the packed path: `nn.utils.rnn.pack_padded_sequence(x,
lengths, enforce_sorted=False)` feeds the GRU exactly `x[i, 0:lengths[i], :]`
for each row `i` (anything at or past column `lengths[i]` is never read), and
`h_n[-1]` is the hidden state after those `lengths[i]` steps from a zero
initial state. A forward (non-bidirectional) GRU is causal -- its hidden
state at time `t` is a function of `x[i, 0:t+1, :]` and `h0` only, never of
later timesteps -- so a full, unpacked unroll's output at time index
`lengths[i] - 1` is the SAME function of the SAME inputs, PROVIDED the row's
real events occupy absolute positions `[0, lengths[i])`, i.e. the
right-padding guarantee this task cites from `DATA_CONTRACT.md:116`
("valid positions are [0, length)").

This identity is a property of the GRU recurrence itself -- it does not
depend on what is sitting in the *un-read* region for the packed path, nor,
symmetrically, on what is sitting before `lengths[i]-1` for the full-unroll
path, as long as real positions are `[0, lengths[i])`. It says nothing about
what happens if that placement assumption is violated (e.g. left-padded
input) -- see `run_gru_adapter.py`'s negative control, which moves the same
per-row content to the OTHER side of the L=50 window and reports, explicitly,
what does and does not still match once that assumption no longer holds.

`query()`/`forward()` are re-implemented the same way as
`ContextGRU.query`/`ContextGRU.forward`, calling the model's own
`user_ctx_proj`/`readout_proj`/`readout_dropout`/`item_embed`/`output_bias`/
`contiguous_head`/`_contig_start`/`product_idx_of_class` attributes directly
-- never copied, never re-created, never re-initialised.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

__all__ = ["GRUExportAdapter"]


class GRUExportAdapter(nn.Module):
    """Pack-free view onto an unmodified ContextGRU instance (see module docstring)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model  # reference, not a copy -- registers `model` as a child module

    # -- pack-free encode: full padded GRU + gather at (lengths - 1) --------
    def encode(
        self,
        *,
        item_tokens: Tensor,
        lengths: Tensor,
        attention_mask: Tensor,
        category_tokens: Optional[Tensor] = None,
        brand_tokens: Optional[Tensor] = None,
        main_category_tokens: Optional[Tensor] = None,
        daypart_tokens: Optional[Tensor] = None,
        weekday_tokens: Optional[Tensor] = None,
        event_numeric_features: Optional[Tensor] = None,
        event_quality_flags: Optional[Tensor] = None,
    ) -> Tensor:  # [B, 256] -- backbone readout, BEFORE user-context fusion
        m = self.model
        if attention_mask.shape != item_tokens.shape:
            raise ValueError("attention_mask and item_tokens must share shape [B,L]")

        x = m._embed_and_project(
            item_tokens=item_tokens, category_tokens=category_tokens,
            brand_tokens=brand_tokens, main_category_tokens=main_category_tokens,
            daypart_tokens=daypart_tokens, weekday_tokens=weekday_tokens,
            event_numeric_features=event_numeric_features,
            event_quality_flags=event_quality_flags,
        )                                        # [B, L, 128] -- model's own path

        out, _ = m.gru(x)                         # [B, L, 256], zero initial state (h0 omitted)
        B, _L, H = out.shape
        idx = (lengths - 1).view(B, 1, 1).expand(B, 1, H)
        h_last = out.gather(1, idx).squeeze(1)     # [B, 256] -- causally == h_n[-1] (right-padded x)
        return h_last

    def query(
        self,
        *,
        item_tokens: Tensor,
        lengths: Tensor,
        attention_mask: Tensor,
        category_tokens: Optional[Tensor] = None,
        brand_tokens: Optional[Tensor] = None,
        main_category_tokens: Optional[Tensor] = None,
        daypart_tokens: Optional[Tensor] = None,
        weekday_tokens: Optional[Tensor] = None,
        event_numeric_features: Optional[Tensor] = None,
        event_quality_flags: Optional[Tensor] = None,
        user_context: Optional[Tensor] = None,
        user_context_masks: Optional[Tensor] = None,
    ) -> Tensor:  # [B, 128] -- post-readout query, pre-head
        m = self.model
        h_last = self.encode(
            item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask,
            category_tokens=category_tokens, brand_tokens=brand_tokens,
            main_category_tokens=main_category_tokens, daypart_tokens=daypart_tokens,
            weekday_tokens=weekday_tokens, event_numeric_features=event_numeric_features,
            event_quality_flags=event_quality_flags,
        )

        if m.user_ctx_proj is not None:
            if user_context is None or user_context_masks is None:
                raise ValueError("this model requires user_context and user_context_masks")
            uc_in = torch.cat([user_context, user_context_masks], dim=-1)  # [B, 18]
            uc = m.user_ctx_proj(uc_in)                                    # [B, 64]
            readout_in = torch.cat([h_last, uc], dim=-1)                   # [B, 320]
        else:
            readout_in = h_last                                           # [B, 256]

        q = m.readout_proj(readout_in)   # [B, 128] -- model's own submodule
        q = m.readout_dropout(q)         # model's own dropout (inert in eval mode)
        return q

    def forward(
        self,
        *,
        item_tokens: Tensor,
        lengths: Tensor,
        attention_mask: Tensor,
        category_tokens: Optional[Tensor] = None,
        brand_tokens: Optional[Tensor] = None,
        main_category_tokens: Optional[Tensor] = None,
        daypart_tokens: Optional[Tensor] = None,
        weekday_tokens: Optional[Tensor] = None,
        event_numeric_features: Optional[Tensor] = None,
        event_quality_flags: Optional[Tensor] = None,
        user_context: Optional[Tensor] = None,
        user_context_masks: Optional[Tensor] = None,
    ) -> Tensor:  # [B, K]
        m = self.model
        q = self.query(
            item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask,
            category_tokens=category_tokens, brand_tokens=brand_tokens,
            main_category_tokens=main_category_tokens, daypart_tokens=daypart_tokens,
            weekday_tokens=weekday_tokens, event_numeric_features=event_numeric_features,
            event_quality_flags=event_quality_flags, user_context=user_context,
            user_context_masks=user_context_masks,
        )

        if m.contiguous_head:
            w_out = m.item_embed.weight[m._contig_start:m._contig_start + m.K]
        else:
            w_out = m.item_embed.weight.index_select(0, m.product_idx_of_class)

        logits = q @ w_out.T + m.output_bias  # [B, K] -- model's own tied head, never copied
        return logits

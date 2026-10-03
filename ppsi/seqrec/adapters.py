"""Federated-simulator adapters for the two families (the `ppsi.fedsim.adapter` protocol).

  GRU     query dim = the item dimension. Tied head: the item_embed rows of the catalogue classes, taken with the
          model's own indexing (contiguous slice [3, 3+K) or index_select on product_idx_of_class): there is ONE item
          Parameter; logits = q @ W.T + b exactly as ContextGRU.forward. product_idx_of_class is a persistent int64
          buffer: FIXED (never averaged, verified identical).
  SASRec  untied output table W_out [K, d] + bias, logits = addmm(b, q, W_out.T) exactly as SASRecCE.forward;
          post_step and server_post_aggregate = recenter_output_().

Every batch is checked against the declared input widths (float block widths, name arrays when present,
feature_view) before the query is computed.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ppsi.fedsim.adapter import BaseAdapter, build_manifest

from .features import InputWidths, check_batch_layout, gru_kwargs, sasrec_kwargs
from .gru import ContextGRU
from .sasrec import SASRecCE


class GRUAdapter(BaseAdapter):
    family = "GRU"

    def __init__(self, module: ContextGRU, widths: InputWidths):
        if not isinstance(module, ContextGRU):
            raise TypeError("GRUAdapter needs a ContextGRU")
        super().__init__(module, build_manifest(module), query_dim=module.readout_proj.out_features, K=module.K)
        self.widths = widths
        self.input_widths = widths

    def query(self, batch):
        check_batch_layout(batch, self.input_widths)
        return self.module.query(**gru_kwargs(batch, self.module))

    def head_weight(self) -> Tensor:
        m = self.module
        if m.contiguous_head:
            return m.item_embed.weight[m._contig_start:m._contig_start + m.K]
        return m.item_embed.weight.index_select(0, m.product_idx_of_class)

    def head_bias(self) -> Tensor:
        return self.module.output_bias

    def logits(self, q: Tensor) -> Tensor:
        return q @ self.head_weight().T + self.module.output_bias        # ContextGRU.forward, verbatim


class SASRecAdapter(BaseAdapter):
    family = "SASREC"
    has_post_step = True

    def __init__(self, module: SASRecCE, widths: InputWidths):
        if not isinstance(module, SASRecCE):
            raise TypeError("SASRecAdapter needs a SASRecCE")
        super().__init__(module, build_manifest(module), query_dim=module.d, K=module.K)
        self.widths = widths
        self.input_widths = widths

    def query(self, batch):
        check_batch_layout(batch, self.input_widths)
        return self.module.query(**sasrec_kwargs(batch))

    def head_weight(self) -> Tensor:
        return self.module.output_weight()

    def head_bias(self) -> Tensor:
        return self.module.output_bias

    def logits(self, q: Tensor) -> Tensor:
        return torch.addmm(self.module.output_bias, q, self.module.output_weight().t())   # SASRecCE.forward

    def post_step(self) -> None:
        self.module.recenter_output_()

    def server_post_aggregate(self) -> None:
        self.module.recenter_output_()


ADAPTERS = {"GRU": GRUAdapter, "SASREC": SASRecAdapter}


def make_adapter(module, widths: InputWidths) -> BaseAdapter:
    """Adapter for an existing module (e.g. after loading a checkpoint)."""
    if isinstance(module, ContextGRU):
        return GRUAdapter(module, widths)
    if isinstance(module, SASRecCE):
        return SASRecAdapter(module, widths)
    raise TypeError(f"not a sequence-model family module: {type(module).__name__}")


def module_forward(adapter: BaseAdapter, batch: dict) -> Tensor:
    """The family module's own forward() on a batch (reference for adapter-equivalence checks)."""
    m = adapter.module
    if isinstance(m, ContextGRU):
        return m(**gru_kwargs(batch, m))
    return m(**sasrec_kwargs(batch))


__all__ = ["ADAPTERS", "GRUAdapter", "SASRecAdapter", "make_adapter", "module_forward"]

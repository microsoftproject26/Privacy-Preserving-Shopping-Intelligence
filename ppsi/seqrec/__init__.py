"""Sequential next-item models: ContextGRU and SASRecCE, their federated adapters and a deterministic builder.

  features   token channels, InputWidths (the four float widths) and batch layout checks
  gru        ContextGRU: feature-group GRU with a tied item-embedding head
  sasrec     SASRecCE: causal Transformer with an untied, recentered output table
  variants   model size variants (cloud sizes and the small on-device size)
  adapters   ppsi.fedsim adapters (query / head / logits / post-step recentering)
  build      build(family, widths, seed, device) -> BuiltModel (module, adapter, theta0 hash)
  monitors   gradient and common-mode drift monitors
  synthetic  synthetic batches for tests and benchmarks
"""
from __future__ import annotations

from .adapters import ADAPTERS, GRUAdapter, SASRecAdapter, make_adapter, module_forward
from .build import FAMILIES, TOPOLOGY, BuiltModel, build
from .features import InputWidths, SchemaError, WidthError, check_batch_layout

__all__ = ["ADAPTERS", "FAMILIES", "TOPOLOGY", "BuiltModel", "GRUAdapter", "InputWidths", "SASRecAdapter",
           "SchemaError", "WidthError", "build", "check_batch_layout", "make_adapter", "module_forward"]

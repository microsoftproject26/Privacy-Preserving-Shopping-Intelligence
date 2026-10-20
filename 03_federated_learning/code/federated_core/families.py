"""fl.families — adapters for the two executed families, built from the byte snapshot `code/snap_v1` via ppsi_env.

`ppsi_env` is imported FIRST (mandatory isolation, T43): it puts snap_v1 at the front of sys.path, pre-imports the
reused modules from it, and scrubs foreign code trees. Nothing in the snapshot is edited.

  GRU    ContextGRU F4_rich_side_information (item 128 / hidden 256, query dim 128). Tied head: the item_embed rows
         of the catalogue classes, taken with the model's own indexing (contiguous slice [3, 3+K) or index_select on
         product_idx_of_class) — there is ONE item Parameter; logits = q @ W.T + b exactly as ContextGRU.forward.
         product_idx_of_class is a persistent int64 buffer: FIXED (never averaged, verified identical).
  SASRec v3 A1 (d 256, 2 blocks, 4 heads, rich), untied output table W_out [K, 256] + bias, logits =
         addmm(b, q, W_out.T) exactly as SASRecCE.forward; post_step and server_post_aggregate = recenter_output_().

A small-K override is supported by both constructors (product_idx_of_class = arange(3, 3+K), item vocab K+3); the
full catalogue map can be read from an explicit local catalogue path (never by editing the snapshot's paths).

CLEAN V2 (FL-02; Ops decision on MODEL-01): the families used by FL runs, receipts and benchmarks are the Clean V2
models of `code/v4models` (widths 15 / 15 / 12 / 4 from the frozen schema), reached through `build_v2` (=
v4models.build with load_schema(); returns the BuiltModel with its adapter and the REGISTERED theta0 hash
`init_sha256`; SASRec theta0 is already recentered once) and `synthetic_examples_v2`. `build_family` and
`synthetic_examples` below build the v1-width SNAPSHOT models and exist only for v1-parity checks. The adapter classes
here are the base classes of v4models' registered ADAPTERS (query / head / logits / post_step unchanged).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
import ppsi_env  # noqa: E402,F401  (must precede every snapshot import)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

import model as gru_model  # noqa: E402  (snapshot copy: snap_v1/reuse/model.py)
import train as reuse_train  # noqa: E402  (snapshot copy: snap_v1/reuse/train.py)
from sasrec_model import SASRecCE, SASRecConfig  # noqa: E402  (snapshot copy)

from .adapter import BaseAdapter, build_manifest  # noqa: E402
from .numerics import seeded_global_rng  # noqa: E402

GRU_GROUP = "F4_rich_side_information"
SASREC_ENTRY = {"id": "A1_RICH256", "group": "rich_frozen_channels", "d_model": 256, "blocks": 2, "heads": 4,
                "ff_multiplier": 4, "dropout": 0.2}                      # as in code/g0_bench_fl_cost.py
L, NUM, FLAGS, UC, UM = 50, 19, 15, 12, 6
SASREC_KEYS = ("item_tokens", "lengths", "attention_mask", "position_ids", "category_tokens", "brand_tokens",
               "main_category_tokens", "daypart_tokens", "weekday_tokens", "event_numeric_features",
               "event_quality_flags", "user_context", "user_context_masks")


def load_catalogue_poc(catalogue_parquet: str | Path) -> np.ndarray:
    """product_idx_of_class from an EXPLICIT local catalogue file (read-only)."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(catalogue_parquet)).read(columns=["product_idx"])["product_idx"].to_numpy()


def _vocab(K: int) -> dict:
    v = dict(reuse_train.VOCAB_SIZES)
    v["item_tokens"] = int(K) + 3
    return v


class GRUAdapter(BaseAdapter):
    def __init__(self, module):
        super().__init__(module, build_manifest(module), query_dim=module.readout_proj.out_features, K=module.K)

    def query(self, batch):
        return self.module.query(**reuse_train.build_named_kwargs(batch, self.module))

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
    has_post_step = True

    def __init__(self, module):
        super().__init__(module, build_manifest(module), query_dim=module.d, K=module.K)

    def query(self, batch):
        return self.module.query(**{k: batch[k] for k in SASREC_KEYS if k in batch})

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


def build_v2(family: str, *, seed: int, K: Optional[int] = None, poc: Optional[np.ndarray] = None,
             device: str = "cpu"):
    """The Clean V2 family (v4models.build with the frozen schema) -> v4models.build.BuiltModel. Use
    `Server(bm.adapter, init_sha256=bm.init_sha256)` / `bm.init_sha256` as the registered theta0 hash."""
    from v4models import build, load_schema             # lazy: v4models itself imports this module
    return build(family, load_schema(), seed, device, K=K, poc=poc)


def synthetic_examples_v2(n: int, K: int, gen: torch.Generator) -> dict:
    """n synthetic decisions in the Clean V2 layout (v4models.synthetic; no data rows)."""
    from v4models import load_schema
    from v4models.synthetic import synthetic_examples_v2 as _syn
    return _syn(n, K, load_schema().widths, gen)


def build_family(family: str, *, seed: int, K: Optional[int] = None, poc: Optional[np.ndarray] = None,
                 device: str = "cpu") -> BaseAdapter:
    """V1-PARITY ONLY (snapshot widths 19/15/12/6). family in {"GRU", "SASREC"}; a small K or the catalogue map."""
    if (K is None) == (poc is None):
        raise ValueError("give exactly one of K (small override) or poc (catalogue map)")
    if poc is None:
        poc = np.arange(3, 3 + int(K), dtype=np.int64)
    K = int(poc.shape[0])
    vocab = _vocab(K)            # item vocab K + 3 (PAD, MISSING, OOV); equals the frozen 158,489 at the full K
    dev = torch.device(device)
    if family == "GRU":
        with seeded_global_rng(seed, torch.device("cpu")):   # ContextGRU uses the default (global-RNG) init
            m = gru_model.build_model(GRU_GROUP, vocab, product_idx_of_class=poc)
        return GRUAdapter(m.to(dev))
    if family == "SASREC":
        m = SASRecCE(SASRecConfig.from_registry(SASREC_ENTRY), vocab_sizes=vocab, product_idx_of_class=poc, seed=seed)
        return SASRecAdapter(m.to(dev))
    raise ValueError(f"unknown family {family!r}")


def synthetic_examples(n: int, K: int, gen: torch.Generator, vocab: Optional[dict] = None) -> dict:
    """V1-PARITY ONLY: n synthetic rich decisions in the snapshot layout (widths 19/15/12/6) — no data rows."""
    V = vocab or _vocab(K)
    lengths = torch.randint(1, L + 1, (n,), generator=gen)
    am = torch.arange(L)[None, :] < lengths[:, None]
    tok = lambda v: torch.randint(3, v, (n, L), generator=gen) * am  # noqa: E731
    return {"item_tokens": tok(V["item_tokens"]), "category_tokens": tok(V["category_tokens"]),
            "brand_tokens": tok(V["brand_tokens"]), "main_category_tokens": tok(V["main_category_tokens"]),
            "daypart_tokens": tok(V["daypart_tokens"]), "weekday_tokens": tok(V["weekday_tokens"]),
            "lengths": lengths, "attention_mask": am,
            "event_numeric_features": torch.randn(n, L, NUM, generator=gen) * am[..., None],
            "event_quality_flags": (torch.rand(n, L, FLAGS, generator=gen) < 0.05).to(torch.int8),
            "user_context": torch.randn(n, UC, generator=gen), "user_context_masks": torch.ones(n, UM),
            "position_ids": torch.arange(1, L + 1)[None, :] * am,
            "target_class": torch.randint(0, K, (n,), generator=gen)}


def module_forward(adapter: BaseAdapter, batch: dict) -> Tensor:
    """The family's own forward() on a batch (reference for adapter-equivalence checks; snapshot or V2 modules)."""
    if isinstance(adapter, GRUAdapter):
        return adapter.module(**reuse_train.build_named_kwargs(batch, adapter.module))
    return adapter.module(**{k: batch[k] for k in SASREC_KEYS if k in batch})

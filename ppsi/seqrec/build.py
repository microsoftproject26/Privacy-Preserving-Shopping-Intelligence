"""`build(family, widths, seed, device)`: a deterministic model, its federated adapter and its initial-state hash.

Families (only the input widths and the size variant change):
  GRU     ContextGRU, rich channels, item 128, hidden 256, one layer, tied item-embedding head (+ output_bias), input
          dropout 0.20, readout dropout 0.10. Init = PyTorch default init drawn from the global CPU RNG seeded with
          `seed` inside a fork, so the caller's RNG is untouched.
  SASREC  SASRecCE d 256, 2 blocks, 4 heads, ff x4, dropout 0.2, input LayerNorm, no sqrt(d), untied output table
          W_out [K, d] + bias with per-step recentering. Init = the model's own seeded generator. The initial state of
          every method is the RECENTERED init, so `build` recenters once (on CPU, before any device move) and reports
          both the raw and the recentered state hash.

Initial-state hash: ppsi.fedsim.numerics.state_digest over the adapter's broadcast state (shared parameters under
canonical keys plus persistent buffers).
`variant` (ppsi.seqrec.variants): None builds the family's default size; `dropout` (GRU (input, readout), SASRec (p,)
or p; None = the variant default) goes into the module build kwargs only, so theta0 does not depend on it.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ppsi.fedsim.adapter import BaseAdapter, ParamManifest
from ppsi.fedsim.numerics import seeded_global_rng, state_digest

from . import gru, sasrec, variants
from .adapters import ADAPTERS
from .features import REES46_VOCAB_SIZES, InputWidths, as_widths

FAMILIES = ("GRU", "SASREC")
TOPOLOGY = {
    "GRU": {"class": "ContextGRU", "channels": "rich (6 token channels + numeric + flags + late-fused user context)",
            "item_dim": 128, "hidden": 256, "layers": 1, "head": "tied: item_embed rows [3, 3+K) + output_bias",
            "proj_dropout": 0.20, "readout_dropout": 0.10, "query_dim": 128},
    "SASREC": {"class": "SASRecCE", "entry": dict(variants.VARIANTS["SASREC_D256_B2"]["sasrec_entry"]),
               "input_layernorm": True, "sqrt_d_input_scale": False,
               "head": "untied output_embed [K, d] + output_bias; recenter_output_() after every optimizer step",
               "query_dim": 256},
}


@dataclass
class BuiltModel:
    family: str
    seed: int
    device: str
    widths: InputWidths
    K: int
    vocab_sizes: dict
    module: nn.Module
    adapter: BaseAdapter
    init_sha256: str            # state_digest(broadcast_state) of the returned theta0 (recentered for SASRec)
    raw_init_sha256: str        # before the recentering (== init_sha256 for GRU)
    recentered: bool
    manifest_digest: str
    inventory: dict = field(default_factory=dict)
    variant: str | None = None
    dropout: tuple | None = None

    @property
    def manifest(self) -> ParamManifest:
        return self.adapter.manifest

    def theta0(self, clone: bool = True) -> OrderedDict[str, torch.Tensor]:
        return self.adapter.broadcast_state(clone)

    def record(self) -> dict:
        return {"family": self.family, "seed": self.seed, "device": self.device, "K": self.K,
                "widths": self.widths.as_json(), "vocab_sizes": dict(self.vocab_sizes),
                "init_sha256": self.init_sha256, "raw_init_sha256": self.raw_init_sha256,
                "recentered_theta0": self.recentered, "manifest_digest": self.manifest_digest,
                "shared_numel": self.manifest.shared_numel, "inventory": dict(self.inventory),
                "variant": self.variant, "dropout": list(self.dropout) if self.dropout is not None else None}


def load_catalogue_poc(catalogue_parquet: str | Path) -> np.ndarray:
    """product_idx_of_class from an explicit local catalogue file with a `product_idx` column (read-only)."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(catalogue_parquet)).read(columns=["product_idx"])["product_idx"].to_numpy()


def default_vocab(K: int) -> dict:
    """The REES46 side-channel vocabularies with an item vocabulary of K + 3 (PAD, MISSING, OOV + K classes)."""
    v = dict(REES46_VOCAB_SIZES)
    v["item_tokens"] = int(K) + 3
    return v


def _class_map(K, poc, catalogue):
    given = [x is not None for x in (K, poc, catalogue)]
    if sum(given) != 1:
        raise ValueError("give exactly one of K (arange(3, 3+K)), poc or catalogue (parquet path)")
    if catalogue is not None:
        poc = load_catalogue_poc(catalogue)
    elif poc is None:
        poc = np.arange(3, 3 + int(K), dtype=np.int64)
    return np.asarray(poc, dtype=np.int64)


def construct_module(family: str, widths: InputWidths, seed: int, *, vocab_sizes: dict, poc: np.ndarray,
                     impl: str = "unpadded", variant: str | None = None, dropout=None) -> nn.Module:
    """The raw, un-recentered module on CPU (deterministic per seed; the caller's global RNG is left untouched)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    vname, ventry = variants.resolve(family, variant)
    drop = variants.dropout_of(family, vname, dropout)
    if family == "GRU":
        with seeded_global_rng(int(seed), torch.device("cpu")):     # ContextGRU uses the default (global-RNG) init
            return gru.build_model(widths.gru_spec(), vocab_sizes, product_idx_of_class=poc,
                                   proj_dropout=drop[0], readout_dropout=drop[1], item_dim=int(ventry["d"]))
    cfg = sasrec.SASRecConfig.from_entry(ventry["sasrec_entry"])
    if drop[0] != cfg.dropout:
        cfg = replace(cfg, dropout=drop[0])
    side = tuple(c for c in widths.categorical if c != "item_tokens")
    with seeded_global_rng(int(seed), torch.device("cpu")):         # every tensor is re-initialised from `seed` anyway
        return sasrec.SASRecCE(cfg, vocab_sizes=vocab_sizes, product_idx_of_class=poc, seed=int(seed),
                               impl=impl, widths=widths, side_channels=side)


def inventory(module: nn.Module) -> dict:
    inv = (gru.parameter_inventory(module) if isinstance(module, gru.ContextGRU)
           else sasrec.parameter_inventory(module))
    unique = {id(p) for p in module.parameters()}
    inv["unique_parameter_tensors"] = len(unique)
    inv["registered_parameter_names"] = len(list(module.named_parameters(remove_duplicate=False)))
    return inv


def build(family: str, widths: InputWidths, seed: int, device: str | torch.device = "cpu", *,
          K: int | None = None, poc=None, catalogue: str | Path | None = None, vocab_sizes: dict | None = None,
          impl: str = "unpadded", variant: str | None = None, dropout=None) -> BuiltModel:
    """Build one family model and its federated adapter.

    family   "GRU" or "SASREC" (case-insensitive).
    widths   the InputWidths of the feature layout.
    seed     the per-seed init; identical seeds give bitwise-identical theta0 (and init_sha256).
    device   the module is built (and, for SASRec, recentered) on CPU, then moved.
    Class map: exactly one of K (arange(3, 3+K)), poc or catalogue (a local catalogue.parquet).
    vocab_sizes  default: `default_vocab(K)`.
    variant  a ppsi.seqrec.variants name of this family; None = the default size.
    dropout  the per-site dropout (GRU (input, readout); SASRec (p,) or p); None = the variant default."""
    fam = str(family).upper()
    if fam not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    widths = as_widths(widths)
    poc = _class_map(K, poc, catalogue)
    Kc = int(poc.shape[0])
    vocab = dict(vocab_sizes) if vocab_sizes is not None else default_vocab(Kc)
    vname, _ = variants.resolve(fam, variant)
    drop = variants.dropout_of(fam, vname, dropout)
    m = construct_module(fam, widths, seed, vocab_sizes=vocab, poc=poc, impl=impl, variant=vname, dropout=drop)
    tmp = ADAPTERS[fam](m, widths)
    raw = state_digest(tmp.broadcast_state())
    recentered = False
    if fam == "SASREC":
        m.recenter_output_()                                        # theta0 is the recentered init
        recentered = True
        init = state_digest(tmp.broadcast_state())
    else:
        init = raw
    dev = torch.device(device)
    m = m.to(dev)
    adapter = ADAPTERS[fam](m, widths)
    return BuiltModel(family=fam, seed=int(seed), device=str(dev), widths=widths, K=Kc, vocab_sizes=vocab, module=m,
                      adapter=adapter, init_sha256=init, raw_init_sha256=raw, recentered=recentered,
                      manifest_digest=adapter.manifest.digest(), inventory=inventory(m), variant=vname,
                      dropout=drop)


__all__ = ["FAMILIES", "TOPOLOGY", "BuiltModel", "build", "construct_module", "default_vocab", "inventory",
           "load_catalogue_poc"]

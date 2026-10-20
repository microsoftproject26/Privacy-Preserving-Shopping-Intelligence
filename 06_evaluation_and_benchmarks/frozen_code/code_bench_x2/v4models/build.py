"""v4models.build — the Clean V2 constructor `build(family, schema, seed, device)` and the FL adapter registration.

Families (package 05 section A; the executed topologies, only the input widths change):
  GRU     ContextGRU (v4models/gru_v2.py): F4-rich channels, item 128, hidden 256, one layer, tied item-embedding head
          (+ output_bias), proj dropout 0.20, readout dropout 0.10. Init = PyTorch default init drawn from the global
          CPU RNG seeded with `seed` inside a fork (exactly as fl.families.build_family), so the caller's RNG is untouched.
  SASREC  SASRec v3 A1 (v4models/sasrec_v2.py): d 256, 2 blocks, 4 heads, ff x4, dropout 0.2, input LayerNorm, no
          sqrt(d), UNTIED output table W_out [K, d] + bias with per-step recentering. Init = the model's own seeded
          generator. Fable G0(b) C7: the per-seed theta0 of EVERY method is the RECENTERED init, so `build` recenters
          once (on CPU, before any device move) and reports both the raw and the recentered state sha256.

Initial-state hash: fl.numerics.state_digest over the adapter's broadcast state (shared parameters under canonical keys
plus persistent buffers) — the same digest fl.local_only.LORecipe records as theta0 provenance.

Adapters: `GRUV2Adapter` / `SASRecV2Adapter` subclass the reviewed fl.families adapters (query / head_weight /
head_bias / logits / post_step / server_post_aggregate unchanged) and add one thing: every batch is checked against
the schema (float block widths, the Clean V2 name arrays when present, feature_view) before the query is computed.
`ADAPTERS` is the registration table.

SCHEMA-01 (compact derived schemas, v4models.derived): `build` also accepts a DerivedSchema. The module is sized from
its kept channels (GRU: the categorical list and widths of the spec; SASRec: `side_channels` and the widths; a branch
without inputs is not built), and the adapter carries the schema's ChannelProjection: `query` checks the FULL Clean V2
batch against the BASE widths, then projects it by name index, so providers stay unchanged. At L0 (the full Clean V2
as a derived schema) the projection is the identity and returns the batch object itself: module, theta0 and forward
are bitwise the MODEL-01 build (tests_models/test_schema01_derived.py).
SCHEMA-01R1: a derived schema may keep the extension channel `session_offset_tokens` (v4models.derived.
EXTENSION_CHANNELS; vocab 9 added to the vocab only for such a model). The adapters pass that batch key to the module
only when the module uses it; for every other model the query path is the MODEL-01 call, unchanged.
TRAIN-02 item 0a (model variants, v4models.variants; PREG1_AMENDMENT_02R1 §1): `build(..., variant=None, dropout=None)`.
variant None (or the family's current-build name GRU_D128 / SASREC_D256_B2) constructs exactly the current module
(ContextGRU via gru_v2.build_model; SASRecConfig.from_registry(SASREC_ENTRY)), bitwise as before; GRU_D256 uses
v4models.variants.ContextGRUD (the `d` argument), the SASRec variants a SASRecConfig from the variant entry. `dropout`
(GRU (input, readout), SASRec (p,) or p; None = the variant default 0.20 / 0.10, 0.2) goes into the module build kwargs
only; it has no parameters, so theta0 does not depend on it. The FL adapters are variant-agnostic: query_dim / K / the
manifest are derived from the module (fl.families: readout_proj.out_features, module.d, build_manifest).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
import ppsi_env  # noqa: E402,F401  (must precede every snapshot import)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import train as reuse_train  # noqa: E402  (snapshot copy: VOCAB_SIZES, build_named_kwargs)
from fl.adapter import BaseAdapter, ParamManifest  # noqa: E402
from fl.families import SASREC_ENTRY, GRUAdapter, SASRecAdapter, load_catalogue_poc  # noqa: E402
from fl.numerics import seeded_global_rng, state_digest  # noqa: E402

from . import gru_v2, sasrec_v2, variants  # noqa: E402
from .derived import EXTENSION_CHANNELS, ChannelProjection, as_projection  # noqa: E402
from .schema import InputWidths, as_widths, check_batch_layout  # noqa: E402

FAMILIES = ("GRU", "SASREC")
TOPOLOGY = {
    "GRU": {"module": "v4models/gru_v2.py", "class": "ContextGRU", "channels": "F4-rich (6 token channels + numeric "
            "+ flags + late-fused user context)", "item_dim": 128, "hidden": 256, "layers": 1,
            "head": "tied: item_embed rows [3, 3+K) + output_bias", "proj_dropout": 0.20, "readout_dropout": 0.10,
            "query_dim": 128},
    "SASREC": {"module": "v4models/sasrec_v2.py", "class": "SASRecCE", "registry_entry": dict(SASREC_ENTRY),
               "input_layernorm": True, "sqrt_d_input_scale": False,
               "head": "untied output_embed [K, d] + output_bias; recenter_output_() after every optimizer step",
               "query_dim": int(SASREC_ENTRY["d_model"])},
}
V1_MAPPING_FILES = {"catalogue": "mappings/catalogue.parquet",
                    "category_tokens": "mappings/vocab_category_id.parquet",
                    "brand_tokens": "mappings/vocab_brand.parquet",
                    "main_category_tokens": "mappings/vocab_main_category.parquet"}
CLOSED_VOCABS = {"daypart_tokens": 3 + 5, "weekday_tokens": 3 + 7}   # declared closed forms (PAD/MISSING/OOV + bands)


def extension_tokens_of(module: nn.Module) -> tuple:
    """The SCHEMA-01R1 extension token channels a family module consumes (empty for every MODEL-01 model)."""
    return tuple(c for c in getattr(module, "_categorical_channels", ()) if c in EXTENSION_CHANNELS)


def _with_extensions(kw: dict, batch, ext: tuple) -> dict:
    for c in ext:
        if batch.get(c) is None:
            raise ValueError(f"this model's schema keeps {c}, but the batch has no {c!r} key (DATA-05)")
        kw[c] = batch[c]
    return kw


def _projection_for(widths: InputWidths, projection: Optional[ChannelProjection]) -> Optional[ChannelProjection]:
    if projection is not None and projection.model != widths:
        raise ValueError("the projection's model widths differ from the adapter's widths")
    return projection


class GRUV2Adapter(GRUAdapter):
    family = "GRU"

    def __init__(self, module: gru_v2.ContextGRU, widths: InputWidths,
                 projection: Optional[ChannelProjection] = None):
        if not isinstance(module, gru_v2.ContextGRU):
            raise TypeError("GRUV2Adapter needs a v4models.gru_v2.ContextGRU")
        super().__init__(module)
        self.widths = widths
        self.projection = _projection_for(widths, projection)
        self.input_widths = projection.base if projection is not None else widths

    def project(self, batch):
        """The full Clean V2 batch -> this model's inputs (checked against the BASE layout first)."""
        check_batch_layout(batch, self.input_widths)
        return batch if self.projection is None else self.projection.apply(batch)

    def query(self, batch):
        b = self.project(batch)
        ext = extension_tokens_of(self.module)
        if not ext:
            return super().query(b)                               # the MODEL-01 call
        return self.module.query(**_with_extensions(reuse_train.build_named_kwargs(b, self.module), b, ext))


class SASRecV2Adapter(SASRecAdapter):
    family = "SASREC"

    def __init__(self, module: sasrec_v2.SASRecCE, widths: InputWidths,
                 projection: Optional[ChannelProjection] = None):
        if not isinstance(module, sasrec_v2.SASRecCE):
            raise TypeError("SASRecV2Adapter needs a v4models.sasrec_v2.SASRecCE")
        super().__init__(module)
        self.widths = widths
        self.projection = _projection_for(widths, projection)
        self.input_widths = projection.base if projection is not None else widths

    def project(self, batch):
        """The full Clean V2 batch -> this model's inputs (checked against the BASE layout first)."""
        check_batch_layout(batch, self.input_widths)
        return batch if self.projection is None else self.projection.apply(batch)

    def query(self, batch):
        b = self.project(batch)
        ext = extension_tokens_of(self.module)
        if not ext:
            return super().query(b)                               # the MODEL-01 call
        from fl.families import SASREC_KEYS
        return self.module.query(**_with_extensions({k: b[k] for k in SASREC_KEYS if k in b}, b, ext))


ADAPTERS = {"GRU": GRUV2Adapter, "SASREC": SASRecV2Adapter}


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
    raw_init_sha256: str        # before the C7 recentering (== init_sha256 for GRU)
    recentered: bool
    manifest_digest: str
    inventory: dict = field(default_factory=dict)
    schema_id: Optional[str] = None            # SCHEMA-01: set when built from a derived schema
    schema_self_sha256: Optional[str] = None
    projection: Optional[ChannelProjection] = None
    variant: Optional[str] = None               # TRAIN-02: the model variant name (v4models.variants)
    dropout: Optional[tuple] = None             # TRAIN-02: the per-site dropout the module was built with

    @property
    def manifest(self) -> ParamManifest:
        return self.adapter.manifest

    def theta0(self, clone: bool = True) -> "OrderedDict[str, torch.Tensor]":
        return self.adapter.broadcast_state(clone)

    def record(self) -> dict:
        return {"family": self.family, "seed": self.seed, "device": self.device, "K": self.K,
                "widths": self.widths.as_json(), "vocab_sizes": dict(self.vocab_sizes),
                "init_sha256": self.init_sha256, "raw_init_sha256": self.raw_init_sha256,
                "recentered_theta0": self.recentered, "manifest_digest": self.manifest_digest,
                "shared_numel": self.manifest.shared_numel, "inventory": dict(self.inventory),
                **({"schema_id": self.schema_id, "schema_self_sha256": self.schema_self_sha256,
                    "projection": self.projection.as_json() if self.projection is not None else None}
                   if self.schema_id is not None else {})}


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vocab_sizes_from_mappings(v1_root: str | Path, *, verify_sha256: bool = True) -> tuple:
    """(vocab_sizes, product_idx_of_class) read from the frozen v1 mapping files (read-only), asserted against the
    executed VOCAB_SIZES (DATA_CONTRACT_CLEAN_V2 section 7: vocab sizes unchanged, the builder asserts them)."""
    import pyarrow.parquet as pq
    root = Path(v1_root)
    if verify_sha256:
        listed = {e["path"]: e["sha256"] for e in json.loads((root / "FILES.json").read_text())["files"]}
        for rel in V1_MAPPING_FILES.values():
            got = _sha256_file(root / rel)
            if listed.get(rel) != got:
                raise RuntimeError(f"{rel}: sha256 {got} != FILES.json {listed.get(rel)}")
    poc = pq.read_table(root / V1_MAPPING_FILES["catalogue"], columns=["product_idx"])["product_idx"].to_numpy()
    K = int(poc.shape[0])
    if not np.array_equal(poc, np.arange(3, 3 + K)):
        raise RuntimeError("catalogue product_idx is not arange(3, 3+K)")
    vocab = {"item_tokens": K + 3}
    for ch in ("category_tokens", "brand_tokens", "main_category_tokens"):
        idx = np.sort(pq.read_table(root / V1_MAPPING_FILES[ch], columns=["idx"])["idx"].to_numpy())
        if not np.array_equal(idx, np.arange(3, 3 + idx.shape[0])):
            raise RuntimeError(f"{ch}: vocabulary idx is not contiguous from 3")
        vocab[ch] = int(idx[-1]) + 1
    vocab.update(CLOSED_VOCABS)
    if vocab != dict(reuse_train.VOCAB_SIZES):
        raise RuntimeError(f"mapping-file vocab sizes {vocab} != executed VOCAB_SIZES {reuse_train.VOCAB_SIZES}")
    return vocab, poc


def _default_vocab(K: int) -> dict:
    v = dict(reuse_train.VOCAB_SIZES)
    v["item_tokens"] = int(K) + 3        # PAD, MISSING, OOV + K classes; equals the executed 158,489 at the full K
    return v


def _class_map(K, poc, catalogue, v1_root):
    given = [x is not None for x in (K, poc, catalogue, v1_root)]
    if sum(given) != 1:
        raise ValueError("give exactly one of K (small override), poc, catalogue (parquet path) or v1_root")
    vocab = None
    if catalogue is not None:
        poc = load_catalogue_poc(catalogue)
    elif v1_root is not None:
        vocab, poc = vocab_sizes_from_mappings(v1_root)
    elif poc is None:
        poc = np.arange(3, 3 + int(K), dtype=np.int64)
    poc = np.asarray(poc, dtype=np.int64)
    return poc, vocab


def construct_module(family: str, widths: InputWidths, seed: int, *, vocab_sizes: dict, poc: np.ndarray,
                     impl: str = "unpadded", variant: Optional[str] = None, dropout=None) -> nn.Module:
    """The raw, un-recentered module on CPU (deterministic per seed; the caller's global RNG is left untouched).
    `variant` / `dropout` (TRAIN-02): None builds the current module exactly as before."""
    ext = [c for c in widths.categorical if c in EXTENSION_CHANNELS and c not in vocab_sizes]
    if ext:                                                       # SCHEMA-01R1: only a model that keeps an extension
        vocab_sizes = dict(vocab_sizes, **{c: int(EXTENSION_CHANNELS[c]["vocab"]) for c in ext})
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    vname, ventry = variants.resolve(family, variant)
    drop = variants.dropout_of(family, vname, dropout)
    if family == "GRU":
        with seeded_global_rng(int(seed), torch.device("cpu")):     # ContextGRU uses the default (global-RNG) init
            if variants.is_current_build(vname):                    # the current build: the ContextGRU class itself
                return gru_v2.build_model(widths.gru_spec(), vocab_sizes, product_idx_of_class=poc,
                                          proj_dropout=drop[0], readout_dropout=drop[1])
            return variants.ContextGRUD(widths.gru_spec(), vocab_sizes, product_idx_of_class=poc,
                                        proj_dropout=drop[0], readout_dropout=drop[1], d=int(ventry["d"]))
    if family == "SASREC":
        if variants.is_current_build(vname):
            if dict(ventry["sasrec_entry"]) != dict(SASREC_ENTRY):
                raise RuntimeError("the SASREC_D256_B2 variant entry differs from fl.families.SASREC_ENTRY")
            cfg = sasrec_v2.SASRecConfig.from_registry(SASREC_ENTRY)
        else:
            cfg = sasrec_v2.SASRecConfig.from_registry(ventry["sasrec_entry"])
        if drop[0] != cfg.dropout:
            from dataclasses import replace as _replace
            cfg = _replace(cfg, dropout=drop[0])
        side = tuple(c for c in widths.categorical if c != "item_tokens")    # SCHEMA-01; L0: == SIDE_CHANNELS
        with seeded_global_rng(int(seed), torch.device("cpu")):     # every tensor is re-initialised from `seed` anyway
            return sasrec_v2.SASRecCE(cfg, vocab_sizes=vocab_sizes, product_idx_of_class=poc, seed=int(seed),
                                      impl=impl, widths=widths, side_channels=side)
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def inventory(module: nn.Module) -> dict:
    inv = gru_v2.parameter_inventory(module) if isinstance(module, gru_v2.ContextGRU) \
        else sasrec_v2.parameter_inventory(module)
    unique = {id(p) for p in module.parameters()}
    inv["unique_parameter_tensors"] = len(unique)
    inv["registered_parameter_names"] = len(list(module.named_parameters(remove_duplicate=False)))
    return inv


def build(family: str, schema, seed: int, device: str | torch.device = "cpu", *, K: Optional[int] = None,
          poc=None, catalogue: Optional[str | Path] = None, v1_root: Optional[str | Path] = None,
          vocab_sizes: Optional[dict] = None, impl: str = "unpadded", variant: Optional[str] = None,
          dropout=None) -> BuiltModel:
    """Build one Clean V2 family model and its FL adapter.

    family   "GRU" or "SASREC" (case-insensitive).
    schema   a CleanV2Schema (load_schema()), a DerivedSchema (v4models.derived; the adapter projects the full
             Clean V2 batch onto its kept channels) or an InputWidths (the v1 reference widths: parity tests only).
    seed     the per-seed init; identical seeds give bitwise-identical theta0 (and init_sha256).
    device   the module is built (and, for SASRec, recentered) on CPU, then moved.
    Class map: exactly one of K (small synthetic override: arange(3, 3+K)), poc, catalogue (explicit local
    catalogue.parquet) or v1_root (reads and hash-checks the v1 mapping files and asserts the vocab sizes).
    variant  TRAIN-02: a v4models.variants name of this family; None = the current build (GRU_D128 / SASREC_D256_B2).
    dropout  TRAIN-02: the per-site dropout (GRU (input, readout); SASRec (p,) or p); None = the variant default."""
    fam = str(family).upper()
    if fam not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    widths = as_widths(schema)
    projection = as_projection(schema)
    poc, file_vocab = _class_map(K, poc, catalogue, v1_root)
    Kc = int(poc.shape[0])
    vocab = dict(vocab_sizes) if vocab_sizes is not None else (file_vocab or _default_vocab(Kc))
    for c in widths.categorical:                                  # SCHEMA-01R1 extension vocab (L0 / L3: no change)
        if c in EXTENSION_CHANNELS:
            vocab.setdefault(c, int(EXTENSION_CHANNELS[c]["vocab"]))
    vname, _ = variants.resolve(fam, variant)
    drop = variants.dropout_of(fam, vname, dropout)
    m = construct_module(fam, widths, seed, vocab_sizes=vocab, poc=poc, impl=impl, variant=vname, dropout=drop)
    tmp = ADAPTERS[fam](m, widths, projection)
    raw = state_digest(tmp.broadcast_state())
    recentered = False
    if fam == "SASREC":
        m.recenter_output_()                                        # C7: the registered theta0 is the recentered init
        recentered = True
        init = state_digest(tmp.broadcast_state())
    else:
        init = raw
    dev = torch.device(device)
    m = m.to(dev)
    adapter = ADAPTERS[fam](m, widths, projection)
    derived = projection is not None
    return BuiltModel(family=fam, seed=int(seed), device=str(dev), widths=widths, K=Kc, vocab_sizes=vocab, module=m,
                      adapter=adapter, init_sha256=init, raw_init_sha256=raw, recentered=recentered,
                      manifest_digest=adapter.manifest.digest(), inventory=inventory(m),
                      schema_id=schema.schema_id if derived else None,
                      schema_self_sha256=schema.self_sha256 if derived else None, projection=projection,
                      variant=vname, dropout=drop)


def make_adapter(module: nn.Module, widths, projection: Optional[ChannelProjection] = None) -> BaseAdapter:
    """Adapter for an existing V2 module (e.g. after loading a checkpoint). `widths` may also be a DerivedSchema
    (its model widths and projection are used)."""
    if getattr(widths, "is_derived_schema", False):
        if projection is not None:
            raise ValueError("give a DerivedSchema or (widths, projection), not both")
        widths, projection = widths.widths, widths.projection
    if isinstance(module, gru_v2.ContextGRU):
        return GRUV2Adapter(module, widths, projection)
    if isinstance(module, sasrec_v2.SASRecCE):
        return SASRecV2Adapter(module, widths, projection)
    raise TypeError(f"not a v4models family module: {type(module).__name__}")


def module_forward(adapter: BaseAdapter, batch: dict) -> torch.Tensor:
    """The family module's own forward() on a batch (reference for adapter-equivalence checks). A derived-schema
    adapter's projection is applied first (the module consumes the projected inputs)."""
    proj = getattr(adapter, "projection", None)
    if proj is not None:
        batch = proj.apply(batch)
    m = adapter.module
    ext = extension_tokens_of(m)
    if isinstance(m, gru_v2.ContextGRU):
        return m(**_with_extensions(reuse_train.build_named_kwargs(batch, m), batch, ext))
    from fl.families import SASREC_KEYS
    return m(**_with_extensions({k: batch[k] for k in SASREC_KEYS if k in batch}, batch, ext))


__all__ = ["FAMILIES", "TOPOLOGY", "ADAPTERS", "GRUV2Adapter", "SASRecV2Adapter", "BuiltModel", "build",
           "construct_module", "inventory", "make_adapter", "module_forward", "vocab_sizes_from_mappings",
           "extension_tokens_of"]

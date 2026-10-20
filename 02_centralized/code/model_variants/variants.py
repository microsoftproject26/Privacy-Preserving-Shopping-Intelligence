"""v4models.variants — the registered model variants of the P2 size search (TRAIN-02 item 0a; PREG1_AMENDMENT_02R1 §1).

Five variants, exactly as 02R1 §1 / Fable's follow-up ruling (b) define them:
  GRU_D128        the current GRU build (v4models.gru_v2.ContextGRU, unchanged): item embedding 128, input_proj -> 128,
                  LayerNorm(128), nn.GRU(128, 256), readout_proj -> 128, tied head (+ output_bias).
  GRU_D256        item embedding 256, input_proj -> 256, LayerNorm(256), nn.GRU(256, 256) (hidden unchanged),
                  readout_proj -> 256, tied head. Built by `ContextGRUD` below (the `d` argument).
  SASREC_D256_B2  the current SASRec build (SASRecConfig.from_registry(fl.families.SASREC_ENTRY), unchanged).
  SASREC_D256_B3  one more causal block (3 blocks).
  SASREC_D384_B2  d_model 384, 4 heads (head dim 96), ff x4, 2 blocks, untied, C7 recentering.
Every other width (side embeddings, numeric / flag projection 32, user context 64 / USER_HIDDEN, max_len 50) is
unchanged; the SASRec variants differ from the current build only in the SASRecConfig fields named in their entry.

A sixth variant, registered by TRAIN-04 (PREG1_AMENDMENT_03 §1/§3), is a DOWNSTREAM SIZE, not a P2 contest size:
  SASREC_D64_B2   d_model 64, 4 heads (head dim 16), ff x4, 2 blocks, untied, C7 recentering; everything else identical
                  to SASREC_D256_B2 (heads count, ff multiplier, tying, dropout sites, recentering). Parameter-matched
                  to GRU_D128 (the downstream SMALL GRU size): at the full catalogue, shared_numel and the per-visit
                  dense payload (down + up, FP32) come out close to GRU_D128's (both well under the 170 MB SMALL cap
                  of PREG1_AMENDMENT_03 §1). It is NOT added to `SIZE_ORDER` (that table is P2's contest-size ladder);
                  `DOWNSTREAM_SMALL` below is the separate SMALL-size mapping for the downstream matrix.

`VARIANTS` is the code table; `registry_obj()` wraps it as the self-hashed registry file
contracts/model_variants/MODEL_VARIANT_REGISTRY.json, which `python -B -m v4models.variants --write <path>` creates
with O_EXCL (an existing file is never overwritten). `load_variant_registry(path)` refuses a file whose self-hash does
not verify or whose entries differ from this table (the file and the code cannot drift apart).
`v4models.build.build(..., variant=None | name, dropout=None | value)`: None is the family's current build.

ContextGRUD mirrors ContextGRU.__init__ statement for statement with the literal 128 of the item dimension replaced
by `d` (item embedding, input_proj out, input_norm, the GRU input size, readout_proj out); every forward method is
inherited unchanged. At d = 128 it builds a module bitwise identical to ContextGRU (same keys, shapes, init draws; test
tests_train/test_train02_variants.py), but GRU_D128 never uses it: the current build keeps the ContextGRU class itself.
gru_v2.py (and its DIFF_VS_SNAPSHOT provenance) is therefore untouched. GRU_MIRRORED_INIT_SHA256 pins the parent
__init__ source this mirror was written against; the test refuses a drift.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Optional

import torch  # noqa: F401  (torch.as_tensor below)
from torch import nn

from . import gru_v2

FORMAT = "PPSI_V4_MODEL_VARIANT_REGISTRY_1"
SELF_KEY = "self_sha256"
V4 = Path(__file__).resolve().parents[2]
REGISTRY_PATH = V4 / "contracts" / "model_variants" / "MODEL_VARIANT_REGISTRY.json"
SOURCE = ("contracts/PREG1_AMENDMENT_02R1.md §1; company/reviews/BEST_BACKBONE_FABLE_FOLLOWUP_SIZE.md (b), (d); "
          "TRAIN-02 item 0a")
GRU_MIRRORED_INIT_SHA256 = "360c9f8745003b7bf795351e5c7998323c913a34b15fe2ccecee40c05efd9239"   # sha256 of the parent __init__ source

_SASREC_BASE = {"group": "rich_frozen_channels", "heads": 4, "ff_multiplier": 4, "dropout": 0.2}
VARIANTS = {
    "GRU_D128": {"family": "GRU", "current_build": True, "d": 128, "hidden": 256, "layers": 1, "head": "tied",
                 "dropout_sites": ["input", "readout"], "dropout_default": [0.2, 0.1],
                 "definition": "current build: item embedding 128, input_proj -> 128, LayerNorm(128), nn.GRU(128, 256), "
                               "readout_proj -> 128, tied head + output_bias"},
    "GRU_D256": {"family": "GRU", "current_build": False, "d": 256, "hidden": 256, "layers": 1, "head": "tied",
                 "dropout_sites": ["input", "readout"], "dropout_default": [0.2, 0.1],
                 "definition": "item embedding 256, input_proj -> 256, LayerNorm(256), nn.GRU(256, 256), "
                               "readout_proj -> 256, tied head + output_bias (hidden unchanged)"},
    "SASREC_D256_B2": {"family": "SASREC", "current_build": True, "head": "untied", "recenter": "C7",
                       "dropout_sites": ["all"], "dropout_default": [0.2],
                       "sasrec_entry": dict(_SASREC_BASE, id="A1_RICH256", d_model=256, blocks=2),
                       "definition": "current build: SASRecConfig.from_registry(fl.families.SASREC_ENTRY)"},
    "SASREC_D256_B3": {"family": "SASREC", "current_build": False, "head": "untied", "recenter": "C7",
                       "dropout_sites": ["all"], "dropout_default": [0.2],
                       "sasrec_entry": dict(_SASREC_BASE, id="SASREC_D256_B3", d_model=256, blocks=3),
                       "definition": "the current build with one more causal block (3 blocks)"},
    "SASREC_D384_B2": {"family": "SASREC", "current_build": False, "head": "untied", "recenter": "C7",
                       "dropout_sites": ["all"], "dropout_default": [0.2],
                       "sasrec_entry": dict(_SASREC_BASE, id="SASREC_D384_B2", d_model=384, blocks=2),
                       "definition": "d_model 384, 4 heads (head dim 96), ff x4, 2 blocks, untied, C7 recentering"},
    "SASREC_D64_B2": {"family": "SASREC", "current_build": False, "head": "untied", "recenter": "C7",
                      "dropout_sites": ["all"], "dropout_default": [0.2],
                      "sasrec_entry": dict(_SASREC_BASE, id="SASREC_D64_B2", d_model=64, blocks=2),
                      "definition": "TRAIN-04 / PREG1_AMENDMENT_03 §1: downstream SMALL SASRec, d_model 64, 4 heads "
                                    "(head dim 16), ff x4, 2 blocks, untied, C7 recentering; parameter-matched to "
                                    "GRU_D128; everything else identical to SASREC_D256_B2"},
}
DEFAULT_VARIANT = {"GRU": "GRU_D128", "SASREC": "SASREC_D256_B2"}
SIZE_ORDER = {"GRU": ("GRU_D128", "GRU_D256"), "SASREC": ("SASREC_D256_B2", "SASREC_D256_B3", "SASREC_D384_B2")}
# PREG1_AMENDMENT_03 §1: the downstream SMALL size per family (NOT a P2 SIZE_ORDER entry — a separate, smaller table
# for the BIG/SMALL downstream matrix; C_SMALL / FA / FP / PF / CPF / LO all build at this variant per family).
DOWNSTREAM_SMALL = {"GRU": "GRU_D128", "SASREC": "SASREC_D64_B2"}


class VariantError(ValueError):
    pass


def self_sha256(obj: Mapping) -> str:
    """sha256 of the canonical JSON of `obj` without "self_sha256" (the v4models.derived.self_hash rule)."""
    body = {k: v for k, v in obj.items() if k != SELF_KEY}
    enc = json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(enc.encode("utf-8")).hexdigest()


def resolve(family: str, variant: Optional[str] = None) -> tuple:
    """(name, entry) of a variant of `family`; None is the family's current build."""
    fam = str(family).upper()
    if fam not in DEFAULT_VARIANT:
        raise VariantError(f"unknown family {family!r}")
    name = DEFAULT_VARIANT[fam] if variant is None else str(variant)
    e = VARIANTS.get(name)
    if e is None:
        raise VariantError(f"unknown model variant {name!r} (registered: {sorted(VARIANTS)})")
    if e["family"] != fam:
        raise VariantError(f"model variant {name} is a {e['family']} variant, not {fam}")
    return name, copy.deepcopy(e)


def is_current_build(name: str) -> bool:
    return bool(VARIANTS[name]["current_build"])


def dropout_of(family: str, variant: Optional[str], dropout=None) -> tuple:
    """The per-site dropout tuple of a build: GRU (input, readout), SASRec (p,); None -> the variant default."""
    name, e = resolve(family, variant)
    want = len(e["dropout_default"])
    if dropout is None:
        return tuple(float(x) for x in e["dropout_default"])
    vals = tuple(float(x) for x in (dropout if isinstance(dropout, (list, tuple)) else (dropout,)))
    if len(vals) != want or not all(0.0 <= x < 1.0 for x in vals):
        raise VariantError(f"{name}: dropout must be {want} value(s) in [0, 1) ({e['dropout_sites']}), got {dropout!r}")
    return vals


def topology(name: str) -> dict:
    return {"model_variant": name, **{k: v for k, v in VARIANTS[name].items() if k != "definition"}}


def registry_obj() -> dict:
    obj = {"format": FORMAT, "task": "TRAIN-02 item 0a", "source": SOURCE,
           "variants": copy.deepcopy(VARIANTS), "default_variant": dict(DEFAULT_VARIANT),
           "size_order": {k: list(v) for k, v in SIZE_ORDER.items()},
           "theta0": "per variant x seed in contracts/model_variants/THETA0_VARIANTS.json (v4train.theta0_variants; "
                     "CPU build, SASRec C7-recentered); derived_schemas/REGISTRY.json and THETA0_REGISTRY.json unchanged"}
    obj[SELF_KEY] = self_sha256(obj)
    return obj


def write_exclusive(path: Path, text: str) -> str:
    """Create `path` with O_EXCL (refuse an existing file), LF line endings, fsync; returns the file sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(data).hexdigest()


def write_registry(path: Path = REGISTRY_PATH) -> dict:
    obj = registry_obj()
    text = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    return {"path": str(path), "file_sha256": write_exclusive(Path(path), text), "self_sha256": obj[SELF_KEY]}


def load_variant_registry(path=REGISTRY_PATH) -> dict:
    """Read and verify the variant registry: strict JSON, self-hash, and every entry equal to the code table."""
    p = Path(path)
    if not p.is_file():
        raise VariantError(f"model variant registry {p} does not exist")
    data = p.read_bytes()
    obj = json.loads(data.decode("utf-8"),
                     parse_constant=lambda c: (_ for _ in ()).throw(VariantError(f"non-strict JSON constant {c}")))
    if not isinstance(obj, dict) or obj.get("format") != FORMAT:
        raise VariantError(f"{p}: not a {FORMAT} object")
    if obj.get(SELF_KEY) != self_sha256(obj):
        raise VariantError(f"{p}: self-hash mismatch (stated {obj.get(SELF_KEY)!r}, recomputed {self_sha256(obj)})")
    if obj.get("variants") != VARIANTS or obj.get("default_variant") != DEFAULT_VARIANT:
        raise VariantError(f"{p}: the registered variants differ from the v4models.variants code table")
    return {"path": str(p), "file_sha256": hashlib.sha256(data).hexdigest(), "self_sha256": obj[SELF_KEY],
            "variants": obj["variants"]}


# ------------------------------------------------------------------------------------------------ GRU d argument
class ContextGRUD(gru_v2.ContextGRU):
    """ContextGRU with the item dimension `d` (item embedding, input_proj, input_norm, GRU input, readout_proj).

    The body mirrors gru_v2.ContextGRU.__init__ (parameter creation order, hence the seeded default-init draws, is the
    same); only the five 128s of the item dimension read `d`. The hidden size (256) and every other width are the
    parent's. All forward methods are inherited."""

    def __init__(self, spec: dict, vocab_sizes: dict, *, product_idx_of_class, proj_dropout: float = 0.20,
                 readout_dropout: float = 0.10, d: int = 128) -> None:
        nn.Module.__init__(self)
        d = int(d)
        if d < 1:
            raise ValueError("d must be a positive int")
        self.d_item = d

        categorical = list(spec["categorical"])
        if "item_tokens" not in categorical:
            raise ValueError("item_tokens is mandatory in every feature group")

        numeric_dim = int(spec.get("numeric_dim", 0))
        flags_dim = int(spec.get("flags_dim", 0))
        use_user_context = bool(spec.get("user_context", False))
        if use_user_context:
            try:
                user_context_dim = int(spec["user_context_dim"])
                user_mask_dim = int(spec["user_mask_dim"])
            except KeyError as e:
                raise KeyError(
                    "a user-context spec must carry user_context_dim and user_mask_dim "
                    f"(taken from the feature schema); missing {e}"
                ) from None
        else:
            user_context_dim = user_mask_dim = 0

        self.spec = dict(spec)
        self._categorical_channels = tuple(categorical)
        self.numeric_dim = numeric_dim
        self.flags_dim = flags_dim
        self.use_user_context = use_user_context
        self.user_context_dim = user_context_dim
        self.user_mask_dim = user_mask_dim

        width = 0
        for ch in categorical:
            if ch not in self._EMBED_DIM:
                raise ValueError(f"unknown categorical channel {ch!r}")
            if ch not in vocab_sizes:
                raise KeyError(f"vocab_sizes missing entry for required channel {ch!r}")
            dim = d if ch == "item_tokens" else self._EMBED_DIM[ch]          # the item dimension reads d
            padding_idx = 0 if ch == "item_tokens" else None
            if ch in ("session_offset_tokens", "session_mode_main_category_tokens"):
                padding_idx = 0
            emb = nn.Embedding(int(vocab_sizes[ch]), dim, padding_idx=padding_idx)
            setattr(self, self._ATTR[ch], emb)
            width += dim
        self.item_vocab_size = int(vocab_sizes["item_tokens"])

        if numeric_dim > 0 or flags_dim > 0:
            self.numeric_flags_proj = nn.Sequential(
                nn.Linear(numeric_dim + flags_dim, 32), nn.GELU()
            )
            width += 32
        else:
            self.numeric_flags_proj = None

        self.concat_width = width
        self.input_proj = nn.Linear(width, d)                                  # d (parent: 128)
        self.input_norm = nn.LayerNorm(d)                                      # d (parent: 128)
        self.input_dropout = nn.Dropout(proj_dropout)

        self.gru = nn.GRU(d, 256, num_layers=1, batch_first=True, dropout=0)   # input d (parent: 128); hidden 256

        if use_user_context:
            self.user_ctx_proj = nn.Sequential(
                nn.Linear(user_context_dim + user_mask_dim, 64),
                nn.GELU(),
            )
            readout_in_dim = 256 + 64
        else:
            self.user_ctx_proj = None
            readout_in_dim = 256

        self.readout_proj = nn.Linear(readout_in_dim, d)                       # d (parent: 128)
        self.readout_dropout = nn.Dropout(readout_dropout)

        poc = torch.as_tensor(product_idx_of_class, dtype=torch.long)
        if poc.ndim != 1:
            raise ValueError("product_idx_of_class must be 1-D")
        K = int(poc.shape[0])
        if K == 0:
            raise ValueError("product_idx_of_class must be non-empty")
        if int(poc.min()) < 0 or int(poc.max()) >= self.item_vocab_size:
            raise IndexError(
                "product_idx_of_class references rows outside the item table "
                f"(item vocab size {self.item_vocab_size})"
            )
        contiguous = bool(torch.equal(poc, torch.arange(3, 3 + K, dtype=torch.long)))
        self.contiguous_head = contiguous
        self._contig_start = 3 if contiguous else None
        self.register_buffer("product_idx_of_class", poc)
        self.K = K
        self.output_bias = nn.Parameter(torch.zeros(K))


def parent_init_source_sha256() -> str:
    import inspect
    return hashlib.sha256(inspect.getsource(gru_v2.ContextGRU.__init__).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="write / verify the model variant registry (TRAIN-02 item 0a)")
    ap.add_argument("--write", default=None, help="create this registry file (O_EXCL: an existing file is refused)")
    ap.add_argument("--verify", default=None, help="verify this registry file against the code table")
    args = ap.parse_args(argv)
    if args.write:
        print(json.dumps(write_registry(Path(args.write)), indent=1))
    if args.verify:
        r = load_variant_registry(args.verify)
        print(json.dumps({k: r[k] for k in ("path", "file_sha256", "self_sha256")}, indent=1))
    return 0


__all__ = ["FORMAT", "VARIANTS", "DEFAULT_VARIANT", "SIZE_ORDER", "DOWNSTREAM_SMALL", "REGISTRY_PATH", "VariantError",
           "resolve", "is_current_build", "dropout_of", "topology", "registry_obj", "write_registry",
           "load_variant_registry", "write_exclusive", "self_sha256", "ContextGRUD", "parent_init_source_sha256",
           "GRU_MIRRORED_INIT_SHA256"]

if __name__ == "__main__":
    raise SystemExit(main())

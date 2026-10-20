"""Model size variants of the two families.

  GRU_D128        the default GRU: item embedding 128, input_proj -> 128, LayerNorm(128), nn.GRU(128, 256),
                  readout_proj -> 128, tied head (+ output_bias).
  GRU_D256        item dimension 256 (item embedding, input_proj, LayerNorm, GRU input, readout_proj); hidden 256.
  SASREC_D256_B2  the default SASRec: d_model 256, 2 causal blocks, 4 heads, ff x4, untied head, recentering.
  SASREC_D256_B3  one more causal block (3 blocks).
  SASREC_D384_B2  d_model 384, 4 heads (head dim 96), 2 blocks.
  SASREC_D64_B2   d_model 64, 4 heads (head dim 16), 2 blocks: the small on-device SASRec, parameter-matched to
                  GRU_D128.
Every other width (side embeddings, numeric / flag projection 32, user context 64, max_len 50) is unchanged; the
SASRec variants differ from the default only in the SASRecConfig fields named in their entry. `SIZE_ORDER` lists the
larger cloud sizes per family; `SMALL_VARIANT` is the small (on-device) size per family.
"""
from __future__ import annotations

import copy

_SASREC_BASE = {"group": "rich_frozen_channels", "heads": 4, "ff_multiplier": 4, "dropout": 0.2}
VARIANTS = {
    "GRU_D128": {"family": "GRU", "current_build": True, "d": 128, "hidden": 256, "layers": 1, "head": "tied",
                 "dropout_sites": ["input", "readout"], "dropout_default": [0.2, 0.1]},
    "GRU_D256": {"family": "GRU", "current_build": False, "d": 256, "hidden": 256, "layers": 1, "head": "tied",
                 "dropout_sites": ["input", "readout"], "dropout_default": [0.2, 0.1]},
    "SASREC_D256_B2": {"family": "SASREC", "current_build": True, "head": "untied", "recenter": "per_step",
                       "dropout_sites": ["all"], "dropout_default": [0.2],
                       "sasrec_entry": dict(_SASREC_BASE, id="SASREC_D256_B2", d_model=256, blocks=2)},
    "SASREC_D256_B3": {"family": "SASREC", "current_build": False, "head": "untied", "recenter": "per_step",
                       "dropout_sites": ["all"], "dropout_default": [0.2],
                       "sasrec_entry": dict(_SASREC_BASE, id="SASREC_D256_B3", d_model=256, blocks=3)},
    "SASREC_D384_B2": {"family": "SASREC", "current_build": False, "head": "untied", "recenter": "per_step",
                       "dropout_sites": ["all"], "dropout_default": [0.2],
                       "sasrec_entry": dict(_SASREC_BASE, id="SASREC_D384_B2", d_model=384, blocks=2)},
    "SASREC_D64_B2": {"family": "SASREC", "current_build": False, "head": "untied", "recenter": "per_step",
                      "dropout_sites": ["all"], "dropout_default": [0.2],
                      "sasrec_entry": dict(_SASREC_BASE, id="SASREC_D64_B2", d_model=64, blocks=2)},
}
DEFAULT_VARIANT = {"GRU": "GRU_D128", "SASREC": "SASREC_D256_B2"}
SIZE_ORDER = {"GRU": ("GRU_D128", "GRU_D256"), "SASREC": ("SASREC_D256_B2", "SASREC_D256_B3", "SASREC_D384_B2")}
SMALL_VARIANT = {"GRU": "GRU_D128", "SASREC": "SASREC_D64_B2"}


class VariantError(ValueError):
    pass


def resolve(family: str, variant: str | None = None) -> tuple:
    """(name, entry) of a variant of `family`; None is the family's default build."""
    fam = str(family).upper()
    if fam not in DEFAULT_VARIANT:
        raise VariantError(f"unknown family {family!r}")
    name = DEFAULT_VARIANT[fam] if variant is None else str(variant)
    e = VARIANTS.get(name)
    if e is None:
        raise VariantError(f"unknown model variant {name!r} (known: {sorted(VARIANTS)})")
    if e["family"] != fam:
        raise VariantError(f"model variant {name} is a {e['family']} variant, not {fam}")
    return name, copy.deepcopy(e)


def is_current_build(name: str) -> bool:
    return bool(VARIANTS[name]["current_build"])


def dropout_of(family: str, variant: str | None, dropout=None) -> tuple:
    """The per-site dropout tuple of a build: GRU (input, readout), SASRec (p,); None -> the variant default."""
    name, e = resolve(family, variant)
    want = len(e["dropout_default"])
    if dropout is None:
        return tuple(float(x) for x in e["dropout_default"])
    vals = tuple(float(x) for x in (dropout if isinstance(dropout, (list, tuple)) else (dropout,)))
    if len(vals) != want or not all(0.0 <= x < 1.0 for x in vals):
        raise VariantError(f"{name}: dropout must be {want} value(s) in [0, 1) ({e['dropout_sites']}), "
                           f"got {dropout!r}")
    return vals


def topology(name: str) -> dict:
    return {"model_variant": name, **copy.deepcopy(VARIANTS[name])}


__all__ = ["DEFAULT_VARIANT", "SIZE_ORDER", "SMALL_VARIANT", "VARIANTS", "VariantError", "dropout_of",
           "is_current_build", "resolve", "topology"]

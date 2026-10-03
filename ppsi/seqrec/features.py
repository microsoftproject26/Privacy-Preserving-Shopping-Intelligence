"""Input layout shared by both model families: token channels, float input widths and batch checks.

A batch is one row per next-item decision with a right-padded window of past events:

  item_tokens, category_tokens, brand_tokens, main_category_tokens, daypart_tokens, weekday_tokens   int64 [B, L]
  lengths [B], attention_mask bool [B, L] (a contiguous prefix of `lengths` real events), position_ids [B, L]
  event_numeric_features float [B, L, numeric_dim], event_quality_flags int8 [B, L, flags_dim]
  user_context float [B, user_context_dim], user_context_masks float [B, user_mask_dim]
  target_class int64 [B] (class j <-> item token j + 3; tokens 0 / 1 / 2 are PAD / MISSING / OOV)

`InputWidths` is what the models are sized from: the four float widths and, optionally, the column names in tensor
order. `check_batch_layout` refuses a batch whose float widths, name arrays or feature view disagree with it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

CATEGORICAL = ("item_tokens", "category_tokens", "brand_tokens", "main_category_tokens", "daypart_tokens",
               "weekday_tokens")
SIDE_TOKEN_CHANNELS = CATEGORICAL[1:]
# Vocabulary sizes of the REES46 catalogue (PAD, MISSING and OOV included); item_tokens is K + 3.
REES46_VOCAB_SIZES = {"item_tokens": 158489, "category_tokens": 663, "brand_tokens": 3956,
                      "main_category_tokens": 16, "daypart_tokens": 8, "weekday_tokens": 10}
# batch key of each float block -> the name-array key a batch may carry for it
NAME_KEYS = {"event_numeric_features": "event_numeric_feature_names",
             "event_quality_flags": "event_quality_flag_names",
             "user_context": "user_context_names",
             "user_context_masks": "user_context_mask_names"}
SASREC_KEYS = ("item_tokens", "lengths", "attention_mask", "position_ids") + SIDE_TOKEN_CHANNELS + (
    "event_numeric_features", "event_quality_flags", "user_context", "user_context_masks")


class SchemaError(ValueError):
    pass


class WidthError(ValueError):
    pass


@dataclass(frozen=True)
class InputWidths:
    """The four float input widths plus (optionally) the column names in tensor order."""
    numeric_dim: int
    flags_dim: int
    user_context_dim: int
    user_mask_dim: int
    view_id: str = "rich"
    numeric_names: tuple = ()
    flag_names: tuple = ()
    user_context_names: tuple = ()
    user_mask_names: tuple = ()
    categorical: tuple = CATEGORICAL

    def __post_init__(self):
        for f in ("numeric_dim", "flags_dim", "user_context_dim", "user_mask_dim"):
            if int(getattr(self, f)) < 0:
                raise SchemaError(f"{f} must be >= 0")
        for dim, names in ((self.numeric_dim, self.numeric_names), (self.flags_dim, self.flag_names),
                           (self.user_context_dim, self.user_context_names),
                           (self.user_mask_dim, self.user_mask_names)):
            if names and len(names) != dim:
                raise SchemaError(f"name array of length {len(names)} does not match width {dim}")

    def width_of(self, key: str) -> int:
        return {"event_numeric_features": self.numeric_dim, "event_quality_flags": self.flags_dim,
                "user_context": self.user_context_dim, "user_context_masks": self.user_mask_dim}[key]

    def names_of(self, key: str) -> tuple:
        return {"event_numeric_features": self.numeric_names, "event_quality_flags": self.flag_names,
                "user_context": self.user_context_names, "user_context_masks": self.user_mask_names}[key]

    def gru_spec(self) -> dict:
        """The ContextGRU group spec: the late-fused user branch exists iff user ctx + masks > 0."""
        return {"categorical": list(self.categorical), "numeric_dim": int(self.numeric_dim),
                "flags_dim": int(self.flags_dim),
                "user_context": int(self.user_context_dim) + int(self.user_mask_dim) > 0,
                "user_context_dim": int(self.user_context_dim), "user_mask_dim": int(self.user_mask_dim)}

    def as_json(self) -> dict:
        return {"view_id": self.view_id, "numeric_dim": self.numeric_dim, "flags_dim": self.flags_dim,
                "user_context_dim": self.user_context_dim, "user_mask_dim": self.user_mask_dim,
                "categorical": list(self.categorical), "numeric_names": list(self.numeric_names),
                "flag_names": list(self.flag_names), "user_context_names": list(self.user_context_names),
                "user_mask_names": list(self.user_mask_names)}


def as_widths(widths) -> InputWidths:
    if isinstance(widths, InputWidths):
        return widths
    raise TypeError(f"expected InputWidths, got {type(widths).__name__}")


def _as_names(v) -> tuple:
    if isinstance(v, (str, bytes)):
        raise WidthError("a name array must be a sequence of names, not a single string")
    return tuple(x.decode() if isinstance(x, bytes) else str(x) for x in v)


def check_batch_layout(batch: Mapping, widths: InputWidths) -> None:
    """Refuse a batch whose float block widths, name arrays or feature_view disagree with `widths`.

    Width: every float block present must have last dimension == the declared width. Names: when the batch carries
    the name arrays (`event_numeric_feature_names`, ...), they must equal the declared names in order. View: a
    `feature_view` key, when present, must equal the declared view id."""
    for key, name_key in NAME_KEYS.items():
        t = batch.get(key)
        if t is not None and int(t.shape[-1]) != widths.width_of(key):
            raise WidthError(f"{key} has width {int(t.shape[-1])}; the feature layout ({widths.view_id}) expects "
                             f"{widths.width_of(key)}")
        if name_key in batch and widths.names_of(key):
            got = _as_names(batch[name_key])
            if got != widths.names_of(key):
                raise WidthError(f"{name_key} {got} != declared names {widths.names_of(key)}")
    fv = batch.get("feature_view")
    if fv is not None and str(fv) != widths.view_id:
        raise WidthError(f"feature_view {fv!r} != declared view {widths.view_id!r}")


def names_sequence(widths: InputWidths) -> dict:
    """The name-array keys of a batch for `widths` (used by synthetic batches and tests)."""
    return {NAME_KEYS[k]: list(widths.names_of(k)) for k in NAME_KEYS if widths.names_of(k)}


def gru_kwargs(batch: Mapping, model) -> dict:
    """Named forward() kwargs for a ContextGRU, derived from the model's own channels and widths."""
    kw = {"item_tokens": batch["item_tokens"], "lengths": batch["lengths"],
          "attention_mask": batch["attention_mask"]}
    for ch in SIDE_TOKEN_CHANNELS:
        if ch in model._categorical_channels:
            kw[ch] = batch[ch]
    if model.numeric_dim > 0:
        kw["event_numeric_features"] = batch["event_numeric_features"]
    if model.flags_dim > 0:
        kw["event_quality_flags"] = batch["event_quality_flags"]
    if model.use_user_context:
        kw["user_context"] = batch["user_context"]
        kw["user_context_masks"] = batch["user_context_masks"]
    return kw


def sasrec_kwargs(batch: Mapping) -> dict:
    """Named forward() kwargs for a SASRecCE: every SASRec key the batch carries."""
    return {k: batch[k] for k in SASREC_KEYS if k in batch}


__all__ = [
    "CATEGORICAL",
    "NAME_KEYS",
    "REES46_VOCAB_SIZES",
    "SASREC_KEYS",
    "SIDE_TOKEN_CHANNELS",
    "InputWidths",
    "SchemaError",
    "WidthError",
    "as_widths",
    "check_batch_layout",
    "gru_kwargs",
    "names_sequence",
    "sasrec_kwargs",
]

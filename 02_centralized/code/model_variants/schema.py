"""v4models.schema — the input-width object both Clean V2 model families are sized from (OI-05).

`load_schema()` reads `contracts/FEATURE_SCHEMA_CLEAN_V2.json`, verifies its self-hash (the canonicalization stated in
the file), and derives the four float input widths from `tensor_layout_v2` (the name arrays). The derived widths are
cross-checked against every other place the schema states them (`summary_counts.final_widths`,
`family_sizing.GRU.group_spec`, `family_sizing.SASRec.constants_v2`, the 53-channel arithmetic) and any disagreement is
a hard `SchemaError`. Nothing here reads dataset rows.

`InputWidths` is what the models consume. `v1_reference_widths(schema)` returns the executed v1 widths 19 / 15 / 12 / 6
(`family_sizing.SASRec.constants_v1`); it exists ONLY for the snapshot-parity tests (same weights, same logits), never
for a Clean V2 run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

CODE = Path(__file__).resolve().parents[1]
PPSI = CODE.parent
DEFAULT_SCHEMA_PATH = PPSI / "contracts" / "FEATURE_SCHEMA_CLEAN_V2.json"
CATEGORICAL = ("item_tokens", "category_tokens", "brand_tokens", "main_category_tokens", "daypart_tokens",
               "weekday_tokens")
VIEW_ID_V2 = "clean_v2_rich"
VIEW_ID_V1_REFERENCE = "v1_rich_reference"
# batch key of each float block -> the name-array key the Clean V2 view adds (DATA_CONTRACT_CLEAN_V2 section 7)
NAME_KEYS = {"event_numeric_features": "event_numeric_feature_names",
             "event_quality_flags": "event_quality_flag_names",
             "user_context": "user_context_names",
             "user_context_masks": "user_context_mask_names"}


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
    view_id: str
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
                           (self.user_context_dim, self.user_context_names), (self.user_mask_dim, self.user_mask_names)):
            if names and len(names) != dim:
                raise SchemaError(f"name array of length {len(names)} does not match width {dim}")

    def width_of(self, key: str) -> int:
        return {"event_numeric_features": self.numeric_dim, "event_quality_flags": self.flags_dim,
                "user_context": self.user_context_dim, "user_context_masks": self.user_mask_dim}[key]

    def names_of(self, key: str) -> tuple:
        return {"event_numeric_features": self.numeric_names, "event_quality_flags": self.flag_names,
                "user_context": self.user_context_names, "user_context_masks": self.user_mask_names}[key]

    def gru_spec(self) -> dict:
        """The ContextGRU group spec (DATA_CONTRACT_CLEAN_V2 section 7, GRU sizing). SCHEMA-01: the late-fused user
        branch exists iff user ctx + masks > 0 (a compact schema without the user block builds no user_ctx_proj);
        the categorical list is this object's (a derived schema's kept tokens, in the canonical order)."""
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


@dataclass(frozen=True)
class CleanV2Schema:
    path: str
    schema_id: str
    schema_version: str
    self_sha256: str                # the stated and verified self-hash
    file_sha256: str                # sha256 of the file bytes (provenance)
    widths: InputWidths
    channels_total: int
    raw: Mapping = field(repr=False, compare=False, default_factory=dict)

    @property
    def dead_in_train_flags(self) -> tuple:
        """The event flags the schema discloses as DEAD_IN_TRAIN (their first-layer columns get zero gradient)."""
        out = []
        for ch in self.raw.get("channels", []):
            if ch.get("block") == "event_flag" and "DEAD_IN_TRAIN" in str(ch.get("disclosure") or ""):
                out.append(ch["name"])
        return tuple(out)


def canonical_self_hash(obj: Mapping) -> str:
    """sha256 over json.dumps(obj minus 'schema_self_sha256', sort_keys, ensure_ascii, compact, allow_nan=False)."""
    body = {k: v for k, v in obj.items() if k != "schema_self_sha256"}
    enc = json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(enc.encode("utf-8")).hexdigest()


def _names(layout: Mapping, key: str) -> tuple:
    try:
        names = tuple(str(n) for n in layout[key]["names"])
    except KeyError as e:
        raise SchemaError(f"tensor_layout_v2.{key}.names missing ({e})") from None
    if len(set(names)) != len(names):
        raise SchemaError(f"duplicate names in tensor_layout_v2.{key}")
    return names


def widths_from_schema_obj(obj: Mapping) -> tuple:
    """(InputWidths, channels_total) from a parsed schema; every stated width must agree."""
    layout = obj["tensor_layout_v2"]
    num, flg = _names(layout, "event_numeric_features"), _names(layout, "event_quality_flags")
    uc, um = _names(layout, "user_context"), _names(layout, "user_context_masks")
    cat = tuple(layout["categorical"]["keys"])
    if cat != CATEGORICAL:
        raise SchemaError(f"categorical keys {cat} != {CATEGORICAL}")
    view = layout.get("view_id")
    if view != VIEW_ID_V2:
        raise SchemaError(f"view_id {view!r} != {VIEW_ID_V2!r}")
    w = InputWidths(len(num), len(flg), len(uc), len(um), view_id=view, numeric_names=num, flag_names=flg,
                    user_context_names=uc, user_mask_names=um, categorical=cat)
    fw = obj["summary_counts"]["final_widths"]
    stated = {"final_widths": (fw["event_numeric_features"], fw["event_quality_flags"], fw["user_context"],
                               fw["user_context_masks"])}
    gs = obj["family_sizing"]["GRU"]["group_spec"]
    stated["family_sizing.GRU.group_spec"] = (gs["numeric_dim"], gs["flags_dim"], gs["user_context_dim"],
                                              gs["user_mask_dim"])
    c2 = obj["family_sizing"]["SASRec"]["constants_v2"]
    stated["family_sizing.SASRec.constants_v2"] = (c2["NUMERIC_DIM"], c2["FLAGS_DIM"], c2["USER_CTX_DIM"],
                                                   c2["USER_MASK_DIM"])
    derived = (w.numeric_dim, w.flags_dim, w.user_context_dim, w.user_mask_dim)
    for where, vals in stated.items():
        if tuple(int(v) for v in vals) != derived:
            raise SchemaError(f"{where} states widths {vals}, the tensor layout gives {derived}")
    if tuple(gs["categorical"]) != CATEGORICAL or gs.get("user_context") is not True:
        raise SchemaError("family_sizing.GRU.group_spec categorical/user_context disagree with the layout")
    total = sum(derived) + len(cat) + int(fw["output_catalogue"])
    if total != int(fw["channels_total"]) or total != int(obj["summary_counts"]["kept_channels_total"]):
        raise SchemaError(f"channel arithmetic {total} != channels_total {fw['channels_total']}")
    return w, total


def load_schema(path: Optional[str | Path] = None, *, verify_self_hash: bool = True) -> CleanV2Schema:
    p = Path(path) if path is not None else DEFAULT_SCHEMA_PATH
    data = p.read_bytes()
    obj = json.loads(data.decode("utf-8"))
    stated = obj.get("schema_self_sha256")
    if verify_self_hash:
        got = canonical_self_hash(obj)
        if stated != got:
            raise SchemaError(f"schema self-hash mismatch: stated {stated}, recomputed {got}")
    widths, total = widths_from_schema_obj(obj)
    return CleanV2Schema(path=str(p), schema_id=str(obj.get("schema_id")), schema_version=str(obj.get("schema_version")),
                         self_sha256=str(stated), file_sha256=hashlib.sha256(data).hexdigest(), widths=widths,
                         channels_total=total, raw=obj)


def v1_reference_widths(schema: CleanV2Schema) -> InputWidths:
    """The executed v1 rich widths (19/15/12/6) from `family_sizing.SASRec.constants_v1` — parity tests only."""
    c1 = schema.raw["family_sizing"]["SASRec"]["constants_v1"]
    return InputWidths(int(c1["NUMERIC_DIM"]), int(c1["FLAGS_DIM"]), int(c1["USER_CTX_DIM"]), int(c1["USER_MASK_DIM"]),
                       view_id=VIEW_ID_V1_REFERENCE)


def as_widths(schema) -> InputWidths:
    """The MODEL widths: a CleanV2Schema's, an InputWidths itself, or a derived schema's (v4models.derived; its
    kept channels — the adapter projects the full Clean V2 batch onto them)."""
    if isinstance(schema, CleanV2Schema):
        return schema.widths
    if isinstance(schema, InputWidths):
        return schema
    if getattr(schema, "is_derived_schema", False):
        return schema.widths
    raise TypeError(f"expected CleanV2Schema, DerivedSchema or InputWidths, got {type(schema).__name__}")


def _as_names(v) -> tuple:
    if isinstance(v, (str, bytes)):
        raise WidthError("a name array must be a sequence of names, not a single string")
    return tuple(x.decode() if isinstance(x, bytes) else str(x) for x in v)


def check_batch_layout(batch: Mapping, widths: InputWidths) -> None:
    """Refuse a batch whose float block widths, name arrays or feature_view disagree with `widths`.

    Width: every float block present must have last dimension == the schema width. Names: when the batch carries the
    Clean V2 name arrays (`event_numeric_feature_names`, ...), they must equal the schema names in order. View: a
    `feature_view` key, when present, must equal the schema view id."""
    for key, name_key in NAME_KEYS.items():
        t = batch.get(key)
        if t is not None and int(t.shape[-1]) != widths.width_of(key):
            raise WidthError(f"{key} has width {int(t.shape[-1])}; the feature schema ({widths.view_id}) expects "
                             f"{widths.width_of(key)}")
        if name_key in batch and widths.names_of(key):
            got = _as_names(batch[name_key])
            if got != widths.names_of(key):
                raise WidthError(f"{name_key} {got} != schema names {widths.names_of(key)}")
    fv = batch.get("feature_view")
    if fv is not None and str(fv) != widths.view_id:
        raise WidthError(f"feature_view {fv!r} != schema view {widths.view_id!r}")


def names_sequence(widths: InputWidths) -> dict:
    """The name-array keys of a Clean V2 batch for `widths` (used by synthetic batches and tests)."""
    return {NAME_KEYS[k]: list(widths.names_of(k)) for k in NAME_KEYS if widths.names_of(k)}


__all__ = ["SchemaError", "WidthError", "InputWidths", "CleanV2Schema", "DEFAULT_SCHEMA_PATH", "CATEGORICAL",
           "VIEW_ID_V2", "NAME_KEYS", "canonical_self_hash", "widths_from_schema_obj", "load_schema",
           "v1_reference_widths", "as_widths", "check_batch_layout", "names_sequence"]

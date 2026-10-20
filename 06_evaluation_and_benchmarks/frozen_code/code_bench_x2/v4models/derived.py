"""v4models.derived — compact DERIVED schemas of FEATURE_SCHEMA_CLEAN_V2 and the channel projection (SCHEMA-01).

A derived schema keeps a subset of the Clean V2 channels. It is a JSON object (contracts/derived_schemas/<id>.json):

  {"format": "PPSI_V4_DERIVED_SCHEMA_1", "schema_id": ...,
   "base": "FEATURE_SCHEMA_CLEAN_V2 2.1.0 self <64-hex self-hash of the base schema>",
   "keep": {"categorical": [...], "event_numeric_features": [...], "event_quality_flags": [...],
            "user_context": [...], "user_context_masks": [...]},
   "widths": {"tokens", "numeric_dim", "flags_dim", "user_context_dim", "user_mask_dim"},
   ..., "self_sha256": sha256 over the object minus "self_sha256" (sort_keys, ensure_ascii, compact, allow_nan=False,
                       the canonicalization of the base schema's own self-hash)}

Rules (every violation is a SchemaError): the base line must name exactly the loaded base schema (id, version,
self-hash); `item_tokens` is always kept; every kept list is a duplicate-free SUBSEQUENCE of the base name array in the
base order (so the L0 projection is the identity and the init order is canonical); the stated widths equal the list
lengths; the self-hash verifies.

Projection (the owner speed directive: providers stay unchanged). Providers keep serving the FULL Clean V2 batch. The
family adapter (v4models.build) first checks that batch against the BASE widths / names / view, then
`ChannelProjection.apply` selects the kept float columns by name index (index_select on the last dim), drops the
unkept token channels, and drops a whole block pair when the model has no branch for it (event numeric + flags when
both widths are 0; user ctx + masks when both are 0). At L0 (everything kept) `apply` returns the batch object itself:
no op is added, so the L0 forward is bitwise today's.

SCHEMA-01R1 (owner directives DECISION_LOG 47 and 48): EXTENSION channels are DATA-05 batch keys that are not in
FEATURE_SCHEMA_CLEAN_V2 but may be kept by a derived schema, each AFTER the base names of its keep block:
  categorical (per-event embeddings, int64 [B, 50], token 0 = PAD: a zero row with no gradient)
    session_offset_tokens               vocab 9: min(offset, 7) + 1 on valid positions (1 = the current session); dim 4
    session_mode_main_category_tokens   the session's (prefix) mode main category (DL 48, G13); dim 8; vocab 17 =
                                        the main-category vocabulary 16 + UNKNOWN (PROVISIONAL until DATA-05 reports the
                                        exact size: a different size changes the schema self-hash and the theta0)
  user_context (decision-level float32 [B])    current_matches_session_mode, session_mode_share        (DL 48, G13)
  user_context_masks (float32 [B])             current_matches_session_mode_available                  (DL 48, G13)
A schema that keeps one records its contract under "extension_channels" (hashed with the schema); the loader requires
it to equal EXTENSION_CHANNELS. The projection passes a kept extension token through, APPENDS the kept decision-level
extension floats as columns after the kept base columns of user_context / user_context_masks (so the models' existing
user branch consumes them: no model change), and removes unkept extension keys (non-identity projections only). A
schema with an extension is never `is_full` (L0S is not L0: the L0 bitwise guard applies to L0 only).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import torch

from .schema import CATEGORICAL, NAME_KEYS, CleanV2Schema, InputWidths, SchemaError, load_schema

FORMAT = "PPSI_V4_DERIVED_SCHEMA_1"
SELF_KEY = "self_sha256"
CANONICALIZATION = ("sha256 over json.dumps(object minus 'self_sha256', sort_keys=True, ensure_ascii=True, "
                    "separators=(',', ':'), allow_nan=False), UTF-8")
BLOCKS = ("event_numeric_features", "event_quality_flags", "user_context", "user_context_masks")
KEEP_KEYS = ("categorical",) + BLOCKS
EVENT_BLOCKS = ("event_numeric_features", "event_quality_flags")
USER_BLOCKS = ("user_context", "user_context_masks")
EXTENSION_CHANNELS = {
    "session_offset_tokens": {"block": "categorical", "kind": "categorical_per_event", "dtype": "int64",
                              "shape": "[B,50]", "vocab": 9, "pad_token": 0, "embed_dim": {"GRU": 4, "SASREC": 4},
                              "values": "0 = PAD; min(offset, 7) + 1 on valid positions (1 = the current session)",
                              "source": "DATA-05 batch key (owner directive DECISION_LOG 47)"},
    "session_mode_main_category_tokens": {
        "block": "categorical", "kind": "categorical_per_event", "dtype": "int64", "shape": "[B,50]", "vocab": 17,
        "vocab_status": "PROVISIONAL (main-category vocabulary 16 + UNKNOWN); DATA-05 reports the exact size",
        "pad_token": 0, "embed_dim": {"GRU": 8, "SASREC": 8},
        "values": "0 = PAD; past session: that session's mode known main category; current session: the causal prefix "
                  "mode; UNKNOWN when no known category yet",
        "group": "G13 SESSION_INTENT", "source": "DATA-05 batch key (owner design DECISION_LOG 48)"},
    "current_matches_session_mode": {
        "block": "user_context", "kind": "decision_float", "dtype": "float32", "shape": "[B]",
        "values": "1 iff the current event's known main category equals the current-session prefix mode",
        "mask": "current_matches_session_mode_available", "group": "G13 SESSION_INTENT",
        "source": "DATA-05 batch key (owner design DECISION_LOG 48)"},
    "session_mode_share": {
        "block": "user_context", "kind": "decision_float", "dtype": "float32", "shape": "[B]",
        "values": "count(mode) / known-category events of the current session so far", "group": "G13 SESSION_INTENT",
        "source": "DATA-05 batch key (owner design DECISION_LOG 48)"},
    "current_matches_session_mode_available": {
        "block": "user_context_masks", "kind": "decision_mask", "dtype": "float32", "shape": "[B]",
        "values": "1 iff current_matches_session_mode is defined", "group": "G13 SESSION_INTENT",
        "source": "DATA-05 batch key (owner design DECISION_LOG 48)"},
}
EXTENSION_BY_BLOCK = {k: tuple(c for c, m in EXTENSION_CHANNELS.items() if m["block"] == k)
                      for k in ("categorical", "event_numeric_features", "event_quality_flags", "user_context",
                                "user_context_masks")}
EXTENSION_TOKENS = EXTENSION_BY_BLOCK["categorical"]
WIDTH_KEYS = {"categorical": "tokens", "event_numeric_features": "numeric_dim", "event_quality_flags": "flags_dim",
              "user_context": "user_context_dim", "user_context_masks": "user_mask_dim"}


def self_hash(obj: Mapping) -> str:
    body = {k: v for k, v in obj.items() if k != SELF_KEY}
    enc = json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(enc.encode("utf-8")).hexdigest()


def base_line(base: CleanV2Schema) -> str:
    return f"{base.schema_id} {base.schema_version} self {base.self_sha256}"


def base_names(base: CleanV2Schema) -> dict:
    w = base.widths
    return {"categorical": tuple(w.categorical), "event_numeric_features": tuple(w.numeric_names),
            "event_quality_flags": tuple(w.flag_names), "user_context": tuple(w.user_context_names),
            "user_context_masks": tuple(w.user_mask_names)}


def allowed_names(base: CleanV2Schema) -> dict:
    """The names a derived schema may keep: the base arrays, each followed by its block's extension names."""
    return {k: v + EXTENSION_BY_BLOCK[k] for k, v in base_names(base).items()}


def full_keep(base: CleanV2Schema) -> dict:
    """The L0 keep lists: every Clean V2 channel, in the base order."""
    return {k: list(v) for k, v in base_names(base).items()}


def keep_from_names(base: CleanV2Schema, names) -> dict:
    """Keep lists (base order) for a set of channel names; an unknown name is refused; item_tokens is always kept."""
    names = set(str(n) for n in names)
    known = allowed_names(base)
    unknown = sorted(names - set().union(*map(set, known.values())))
    if unknown:
        raise SchemaError(f"not Clean V2 channels: {unknown}")
    keep = {k: [n for n in v if n in names] for k, v in known.items()}
    if "item_tokens" not in keep["categorical"]:
        keep["categorical"].insert(0, "item_tokens")
    return keep


def widths_of_keep(keep: Mapping) -> dict:
    return {WIDTH_KEYS[k]: len(keep[k]) for k in KEEP_KEYS}


def make_derived(base: CleanV2Schema, schema_id: str, keep: Mapping, **extra) -> dict:
    """A verified derived-schema JSON object (with its self-hash). `extra` fields (provenance) are hashed too."""
    keep = {k: [str(n) for n in keep[k]] for k in KEEP_KEYS}
    obj = {"format": FORMAT, "schema_id": str(schema_id), "base": base_line(base), "keep": keep,
           "widths": widths_of_keep(keep), "n_information_channels": sum(len(v) for v in keep.values()) + 1,
           "n_information_channels_note": "tokens + numeric + flags + user context + masks + the output catalogue",
           "self_sha256_canonicalization": CANONICALIZATION}
    ext = [c for k in KEEP_KEYS for c in keep[k] if c in EXTENSION_CHANNELS]
    if ext:                                          # SCHEMA-01R1: only a schema that keeps an extension records it
        obj["extension_channels"] = {c: EXTENSION_CHANNELS[c] for c in ext}
    for k, v in extra.items():
        if k in obj or k == SELF_KEY:
            raise SchemaError(f"extra field {k!r} collides with a schema field")
        obj[k] = v
    obj[SELF_KEY] = self_hash(obj)
    derived_from_obj(obj, base)                       # the object must pass its own loader
    return obj


def _subsequence(kept, allowed, what: str) -> tuple:
    kept = tuple(kept)
    if len(set(kept)) != len(kept):
        raise SchemaError(f"{what}: duplicate names")
    missing = [n for n in kept if n not in allowed]
    if missing:
        raise SchemaError(f"{what}: {missing} are not base channels")
    if kept != tuple(n for n in allowed if n in kept):
        raise SchemaError(f"{what}: names must follow the base order (a subsequence of {list(allowed)})")
    return kept


@dataclass(frozen=True)
class ChannelProjection:
    """Full Clean V2 batch (base widths) -> the derived schema's inputs (model widths), by name index."""
    base: InputWidths
    model: InputWidths
    index: Mapping                  # float block key -> tuple of kept base column indices
    tokens_dropped: tuple
    event_blocks_on: bool
    user_blocks_on: bool
    extensions_dropped: tuple = ()     # SCHEMA-01R1: extension keys the schema does not keep (removed if present)
    ext_columns: Mapping = field(default_factory=dict)   # block -> kept decision-level extension keys (appended)
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def is_identity(self) -> bool:
        return (not self.tokens_dropped and self.event_blocks_on and self.user_blocks_on
                and not any(self.ext_columns.values())
                and all(tuple(self.index[k]) == tuple(range(self.base.width_of(k))) for k in BLOCKS))

    def _idx(self, key: str, device) -> torch.Tensor:
        ck = (key, str(device))
        t = self._cache.get(ck)
        if t is None:
            t = torch.as_tensor(self.index[key], dtype=torch.long, device=device)
            self._cache[ck] = t
        return t

    def apply(self, batch: Mapping) -> Mapping:
        if self.is_identity:
            return batch                                        # L0: the batch object itself, no op added
        out = dict(batch)
        for ch in self.tokens_dropped + self.extensions_dropped:
            out.pop(ch, None)
        for key in BLOCKS:
            name_key = NAME_KEYS[key]
            on = self.event_blocks_on if key in EVENT_BLOCKS else self.user_blocks_on
            if not on:
                out.pop(key, None)
                out.pop(name_key, None)
                continue
            t = batch.get(key)
            if t is not None:
                if int(t.shape[-1]) != self.base.width_of(key):
                    raise SchemaError(f"{key} has width {int(t.shape[-1])}; the projection expects the full Clean V2 "
                                      f"width {self.base.width_of(key)}")
                if tuple(self.index[key]) != tuple(range(self.base.width_of(key))):
                    out[key] = t.index_select(t.ndim - 1, self._idx(key, t.device))
                ext = self.ext_columns.get(key) or ()
                if ext:                                         # SCHEMA-01R1: decision-level extension columns
                    cols = []
                    for c in ext:
                        v = batch.get(c)
                        if v is None:
                            raise SchemaError(f"the schema keeps {c}, but the batch has no {c!r} key (DATA-05)")
                        if int(v.numel()) != int(t.shape[0]):
                            raise SchemaError(f"{c} must be [B] (got shape {tuple(v.shape)} for B = {t.shape[0]})")
                        cols.append(v.reshape(-1, 1).to(t.dtype))
                    out[key] = torch.cat([out[key]] + cols, dim=-1)
            if name_key in batch:
                out[name_key] = list(self.model.names_of(key))
        return out

    def as_json(self) -> dict:
        return {"index": {k: list(v) for k, v in self.index.items()}, "tokens_dropped": list(self.tokens_dropped),
                "extensions_dropped": list(self.extensions_dropped),
                "ext_columns": {k: list(v) for k, v in self.ext_columns.items() if v},
                "event_blocks_on": self.event_blocks_on, "user_blocks_on": self.user_blocks_on,
                "is_identity": self.is_identity}


@dataclass(frozen=True)
class DerivedSchema:
    schema_id: str
    base: CleanV2Schema
    keep: Mapping                   # KEEP_KEYS -> tuple of names (base order)
    widths: InputWidths             # the MODEL widths (kept names; view id = the base view)
    projection: ChannelProjection
    self_sha256: str
    path: Optional[str] = None
    file_sha256: Optional[str] = None
    raw: Mapping = field(default_factory=dict, repr=False, compare=False)
    is_derived_schema = True

    @property
    def schema_version(self) -> str:
        return f"derived:{self.base.schema_version}"

    @property
    def channel_names(self) -> tuple:
        return tuple(n for k in KEEP_KEYS for n in self.keep[k])

    @property
    def n_information_channels(self) -> int:
        return len(self.channel_names) + 1

    @property
    def extension_tokens(self) -> tuple:
        return tuple(c for c in self.keep["categorical"] if c in EXTENSION_CHANNELS)

    @property
    def extensions(self) -> tuple:
        return tuple(c for k in KEEP_KEYS for c in self.keep[k] if c in EXTENSION_CHANNELS)

    @property
    def is_full(self) -> bool:
        """The full Clean V2 and nothing else (L0; not L0S)."""
        return self.projection.is_identity and not self.extensions

    def record(self) -> dict:
        return {"schema_id": self.schema_id, "self_sha256": self.self_sha256, "base": base_line(self.base),
                "path": self.path, "file_sha256": self.file_sha256, "widths": widths_of_keep(self.keep),
                "n_information_channels": self.n_information_channels, "projection": self.projection.as_json()}


def derived_from_obj(obj: Mapping, base: Optional[CleanV2Schema] = None, *, path: Optional[str] = None,
                     file_sha256: Optional[str] = None) -> DerivedSchema:
    base = base if base is not None else load_schema()
    if obj.get("format") != FORMAT:
        raise SchemaError(f"derived schema format {obj.get('format')!r} != {FORMAT!r}")
    stated = obj.get(SELF_KEY)
    got = self_hash(obj)
    if stated != got:
        raise SchemaError(f"derived schema self-hash mismatch: stated {stated}, recomputed {got}")
    if obj.get("base") != base_line(base):
        raise SchemaError(f"derived schema base {obj.get('base')!r} != the loaded base {base_line(base)!r}")
    keep_raw = obj.get("keep") or {}
    if sorted(keep_raw) != sorted(KEEP_KEYS):
        raise SchemaError(f"keep must have exactly the keys {KEEP_KEYS}")
    names = allowed_names(base)
    keep = {k: _subsequence(keep_raw[k], names[k], f"keep.{k}") for k in KEEP_KEYS}
    ext = tuple(c for k in KEEP_KEYS for c in keep[k] if c in EXTENSION_CHANNELS)
    want_ext = {c: EXTENSION_CHANNELS[c] for c in ext}
    for c in ext:
        m = EXTENSION_CHANNELS[c].get("mask")
        if m is not None and m not in ext:
            raise SchemaError(f"{c} is kept without its availability mask {m}")
    if (obj.get("extension_channels") or {}) != want_ext or (not ext and "extension_channels" in obj):
        raise SchemaError(f"extension_channels {obj.get('extension_channels')} != the kept extensions' contract "
                          f"{want_ext}")
    if "item_tokens" not in keep["categorical"]:
        raise SchemaError("item_tokens is always kept (the task's item-ID sequence)")
    if dict(obj.get("widths") or {}) != widths_of_keep(keep):
        raise SchemaError(f"stated widths {obj.get('widths')} != the keep lists {widths_of_keep(keep)}")
    bw = base.widths
    model = InputWidths(len(keep["event_numeric_features"]), len(keep["event_quality_flags"]),
                        len(keep["user_context"]), len(keep["user_context_masks"]), view_id=bw.view_id,
                        numeric_names=keep["event_numeric_features"], flag_names=keep["event_quality_flags"],
                        user_context_names=keep["user_context"], user_mask_names=keep["user_context_masks"],
                        categorical=keep["categorical"])
    bn = base_names(base)
    index = {k: tuple(bn[k].index(n) for n in keep[k] if n in bn[k]) for k in BLOCKS}
    base_tokens = bn["categorical"]
    proj = ChannelProjection(base=bw, model=model, index=index,
                             tokens_dropped=tuple(c for c in base_tokens if c not in keep["categorical"]),
                             event_blocks_on=(model.numeric_dim + model.flags_dim) > 0,
                             user_blocks_on=(model.user_context_dim + model.user_mask_dim) > 0,
                             extensions_dropped=tuple(c for c in EXTENSION_CHANNELS if c not in ext),
                             ext_columns={k: tuple(c for c in keep[k] if c in EXTENSION_CHANNELS) for k in BLOCKS})
    core = dataclasses.replace(model, categorical=tuple(c for c in model.categorical if c not in EXTENSION_CHANNELS))
    if proj.is_identity and core != bw:
        raise SchemaError("an identity projection must reproduce the base widths exactly")
    return DerivedSchema(schema_id=str(obj["schema_id"]), base=base, keep=keep, widths=model, projection=proj,
                         self_sha256=got, path=path, file_sha256=file_sha256, raw=dict(obj))


def load_derived_schema(path, base: Optional[CleanV2Schema] = None) -> DerivedSchema:
    p = Path(path)
    data = p.read_bytes()
    obj = json.loads(data.decode("utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(SchemaError(c)))
    return derived_from_obj(obj, base, path=str(p), file_sha256=hashlib.sha256(data).hexdigest())


def as_projection(schema) -> Optional[ChannelProjection]:
    """The projection of a derived schema (None for a CleanV2Schema / InputWidths: no projection)."""
    return schema.projection if getattr(schema, "is_derived_schema", False) else None


__all__ = ["FORMAT", "BLOCKS", "KEEP_KEYS", "CATEGORICAL", "EXTENSION_CHANNELS", "EXTENSION_TOKENS",
           "EXTENSION_BY_BLOCK", "allowed_names", "ChannelProjection", "DerivedSchema", "self_hash",
           "base_line", "base_names", "full_keep", "keep_from_names", "widths_of_keep", "make_derived",
           "derived_from_obj", "load_derived_schema", "as_projection"]

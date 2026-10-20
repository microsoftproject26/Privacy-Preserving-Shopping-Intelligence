"""v4fs.schemas — the FEATSTAT ladder as derived schemas, the derived-schema registry and `derive` (SCHEMA-01).

Sources (read-only, sha256-recorded in every derived schema's `source` block):
  featstat_v5_100/FEATSTAT_LADDER.json      the nested rungs L0..L9 and their channel lists (FEATSTAT-01 §8)
  featstat_v5_100/FEATSTAT_INVENTORY.json   the semantic channel groups (FEATSTAT-01 definition; `groups`)
  contracts/FEATURE_SCHEMA_CLEAN_V2.json    the base schema (self-hash 55a244e1…, via v4models.load_schema)

Written (owned): contracts/derived_schemas/<schema_id>.json (L0_CLEAN_V2 and L3_NEAR_DUPLICATE) and REGISTRY.json
  {"format": "PPSI_V4_DERIVED_SCHEMA_REGISTRY_1", "base": ..., "schemas": {schema_id: {"path", "self_sha256",
   "file_sha256", "n_information_channels", "widths"}}, "self_sha256"}  (paths relative to contracts/derived_schemas).
TRAIN-01R2 reads REGISTRY.json: `load_registered(schema_id)` verifies the registry self-hash, the file sha256 and the
schema self-hash, and returns the v4models DerivedSchema.

`derive(base_id, drop_groups)`: base_id's kept channels minus every channel of the named groups — the 14 binding FS
permutation groups G1 … G12c (FS_GROUPS; Fable R2 D3 amendment 3) or a FEATSTAT group — for the S* candidates ("L3
minus the groups within noise for both families"). The id is deterministic: "<base_id>-minus-<G>+<G>..." (groups
sorted). item_tokens (FEATSTAT group ITEM_ID) is in no FS group and can never be dropped.

SCHEMA-01R1 (owner directives DECISION_LOG 47 / 48): L0S_CLEAN_V2_SESSION (= L0 + session_offset_tokens + the G13
SESSION_INTENT channels, 58 channels) is the feature-search REFERENCE and L3S_NEAR_DUPLICATE_SESSION (= L3 + the same
five, 35 channels) the candidate BASE; S* = L3S minus the dropped groups. REQUIRED channels (never dropped):
item_tokens, category_tokens, session_offset_tokens. `derive` refuses any group holding one (FS G1, FEATSTAT ITEM_ID /
ITEM_METADATA_TOKENS); the droppable FS groups are G2 … G12c + G13 (14); G1's importance is still reported,
descriptively. G13 (a candidate, DL 48): session_mode_main_category_tokens (per event), current_matches_session_mode
with its mask current_matches_session_mode_available, session_mode_share (decision level).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import v4models  # noqa: F401  (ppsi_env first)
from v4models.derived import (DerivedSchema, derived_from_obj, full_keep, keep_from_names, load_derived_schema,
                              make_derived, self_hash)
from v4models.schema import CleanV2Schema, SchemaError, load_schema

CODE = Path(__file__).resolve().parents[1]
V4 = CODE.parent
FEATSTAT = V4 / "featstat_v5_100"
LADDER_PATH = FEATSTAT / "FEATSTAT_LADDER.json"
INVENTORY_PATH = FEATSTAT / "FEATSTAT_INVENTORY.json"
DERIVED_DIR = V4 / "contracts" / "derived_schemas"
REGISTRY_PATH = DERIVED_DIR / "REGISTRY.json"
THETA0_PATH = DERIVED_DIR / "THETA0_REGISTRY.json"
REGISTRY_FORMAT = "PPSI_V4_DERIVED_SCHEMA_REGISTRY_1"
THETA0_FORMAT = "PPSI_V4_DERIVED_THETA0_REGISTRY_1"
WRITTEN_RUNGS = ("L0_CLEAN_V2", "L3_NEAR_DUPLICATE")
SESSION_EXTENSION = "session_offset_tokens"
SESSION_INTENT = ("session_mode_main_category_tokens", "current_matches_session_mode",
                  "current_matches_session_mode_available", "session_mode_share")     # G13 (DECISION_LOG 48)
SESSION_SCHEMAS = {"L0S_CLEAN_V2_SESSION": "L0_CLEAN_V2", "L3S_NEAR_DUPLICATE_SESSION": "L3_NEAR_DUPLICATE"}
REFERENCE_ID = "L0S_CLEAN_V2_SESSION"          # the FS reference (DECISION_LOG 47)
CANDIDATE_BASE_ID = "L3S_NEAR_DUPLICATE_SESSION"   # the FS candidate base; S* = CANDIDATE_BASE_ID minus groups
WRITTEN_SCHEMAS = WRITTEN_RUNGS + tuple(SESSION_SCHEMAS)
REQUIRED_CHANNELS = ("item_tokens", "category_tokens", "session_offset_tokens")   # DECISION_LOG 47
COMPOSITE_GROUPS = ("USER_CONTEXT_ALL",)
REQUIRED_GROUPS = ("ITEM_ID",)
# Fable R2 D3 amendment 3 (binding, POPULATION_5_100_SCIENTIFIC_SYNTHESIS §D3 item 4): the 14 groups of L3
FS_GROUPS: Dict[str, tuple] = {
    "G1": ("category_tokens",),
    "G2": ("brand_tokens",),
    "G3": ("weekday_tokens",),
    "G4": ("utc_hour_sin", "utc_hour_cos"),
    "G5": ("log1p_price", "price_missing"),
    "G6": ("log1p_prev_raw_view_gap_s", "prev_gap_unavailable"),
    "G7": ("log1p_item_observed_age_s", "item_age_unavailable"),
    "G8": ("log1p_segment_elapsed_s",),
    "G9": ("prefix_representative_count",),
    "G10": ("prefix_known_category_count", "prefix_category_switch_count", "current_known_category_run_length"),
    "G11": ("log1p_time_since_prev_session_last_event_s", "prev_session_available"),
    "G12a": ("log1p_prev_view_count", "log1p_prev_session_count", "recent_distinct_known_main_categories"),
    "G12b": ("recent_category_match_share", "recent_brand_match_share", "recent_category_match_supported",
             "recent_brand_match_supported"),
    "G12c": ("recent_mean_log_price", "current_minus_recent_mean_log_price", "recent_price_supported"),
    # SCHEMA-01R1 (owner design DECISION_LOG 48): the candidate SESSION_INTENT group of L0S / L3S (droppable)
    "G13": ("session_mode_main_category_tokens", "current_matches_session_mode", "current_matches_session_mode_available",
            "session_mode_share"),
}
LADDER_BLOCKS = {"tokens": "categorical", "numeric": "event_numeric_features", "flags": "event_quality_flags",
                 "user_context": "user_context", "user_masks": "user_context_masks"}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _read(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def ladder(path: Path = LADDER_PATH) -> dict:
    lad = _read(path)
    if not lad.get("rungs_nested"):
        raise SchemaError("FEATSTAT ladder is not nested")
    return lad


def rung(rung_id: str, path: Path = LADDER_PATH) -> dict:
    for r in ladder(path)["rungs"]:
        if r["id"] == rung_id:
            return r
    raise KeyError(f"no rung {rung_id!r} in {path}")


def groups(path: Path = INVENTORY_PATH) -> dict:
    """The FEATSTAT semantic groups (name -> channel list), incl. the composite USER_CONTEXT_ALL."""
    g = _read(path)["groups"]
    return {k: list(v) for k, v in g.items()}


def partition_groups(path: Path = INVENTORY_PATH) -> dict:
    """The non-overlapping FEATSTAT groups (the composite USER_CONTEXT_ALL excluded)."""
    return {k: v for k, v in groups(path).items() if k not in COMPOSITE_GROUPS}


def rung_channel_names(r: Mapping) -> list:
    return [n for blk in LADDER_BLOCKS for n in r["channels"][blk]]


def source_block(rung_id: Optional[str] = None, extra: Optional[dict] = None) -> dict:
    src = {"task": "SCHEMA-01", "ladder": {"path": "featstat_v5_100/FEATSTAT_LADDER.json", "sha256": _sha(LADDER_PATH)},
           "groups": {"path": "featstat_v5_100/FEATSTAT_INVENTORY.json", "sha256": _sha(INVENTORY_PATH)}}
    if rung_id is not None:
        src["rung"] = rung_id
    src.update(extra or {})
    return src


def rung_schema_obj(rung_id: str, base: Optional[CleanV2Schema] = None) -> dict:
    """The derived-schema JSON object of a ladder rung, keep lists exactly the rung's channel lists."""
    base = base or load_schema()
    r = rung(rung_id)
    keep = keep_from_names(base, rung_channel_names(r))
    for blk, key in LADDER_BLOCKS.items():               # the rung lists must already be in the base order
        if list(r["channels"][blk]) != keep[key]:
            raise SchemaError(f"{rung_id}.{blk} is not the base-ordered keep list {keep[key]}")
    obj = make_derived(base, rung_id, keep, title=r["title"], source=source_block(rung_id))
    if obj["n_information_channels"] != r["n_information_channels"]:
        raise SchemaError(f"{rung_id}: {obj['n_information_channels']} channels, the ladder says "
                          f"{r['n_information_channels']}")
    want = {"tokens": r["input_widths"]["tokens"], "numeric_dim": r["input_widths"]["numeric_dim"],
            "flags_dim": r["input_widths"]["flags_dim"], "user_context_dim": r["input_widths"]["user_context_dim"],
            "user_mask_dim": r["input_widths"]["user_mask_dim"]}
    if obj["widths"] != want:
        raise SchemaError(f"{rung_id}: widths {obj['widths']} != ladder {want}")
    if rung_id == "L0_CLEAN_V2" and obj["keep"] != full_keep(base):
        raise SchemaError("L0 must keep every Clean V2 channel")
    return obj


def session_schema_obj(schema_id: str, base: Optional[CleanV2Schema] = None) -> dict:
    """L0S / L3S: the rung's keep lists plus session_offset_tokens and the G13 SESSION_INTENT channels, each after the
    base names of its block (SCHEMA-01R1; DECISION_LOG 47 / 48)."""
    base = base or load_schema()
    rung_id = SESSION_SCHEMAS[schema_id]
    r = rung(rung_id)
    extra = [SESSION_EXTENSION, *SESSION_INTENT]
    keep = keep_from_names(base, rung_channel_names(r) + extra)
    obj = make_derived(base, schema_id, keep,
                       title=f"{r['title']} + session_offset_tokens + G13 SESSION_INTENT (DECISION_LOG 47 / 48)",
                       source=source_block(rung_id, {"extensions": extra, "authority": "owner directives DECISION_LOG "
                                                     "47 (session_offset_tokens) and 48 (G13) (SCHEMA-01R1)"}))
    if obj["n_information_channels"] != r["n_information_channels"] + len(extra) or \
            keep["categorical"][-2:] != [SESSION_EXTENSION, SESSION_INTENT[0]]:
        raise SchemaError(f"{schema_id}: expected the {rung_id} channels + {extra}")
    return obj


def schema_obj(schema_id: str, base: Optional[CleanV2Schema] = None) -> dict:
    return session_schema_obj(schema_id, base) if schema_id in SESSION_SCHEMAS else rung_schema_obj(schema_id, base)


def rung_schema(rung_id: str, base: Optional[CleanV2Schema] = None) -> DerivedSchema:
    base = base or load_schema()
    return derived_from_obj(rung_schema_obj(rung_id, base), base)


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def write_file(path: Path, text: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    tmp.replace(path)
    return _sha(path)


def registry_obj(entries: Mapping[str, Mapping], base: CleanV2Schema) -> dict:
    from v4models.derived import base_line
    obj = {"format": REGISTRY_FORMAT, "base": base_line(base), "task": "SCHEMA-01",
           "note": "schema_id -> derived schema file (relative to contracts/derived_schemas); TRAIN-01R2 reads it",
           "schemas": {k: dict(v) for k, v in sorted(entries.items())},
           "self_sha256_canonicalization": "as the derived schemas (v4models.derived.self_hash)"}
    obj["self_sha256"] = self_hash(obj)
    return obj


def write_registry(out_dir: Path = DERIVED_DIR, rung_ids: Iterable[str] = WRITTEN_SCHEMAS,
                   extra: Optional[Mapping[str, dict]] = None) -> dict:
    """Write the schemas (rungs, L0S / L3S, + any extra derived objects) and REGISTRY.json; returns the registry."""
    base = load_schema()
    out_dir = Path(out_dir)
    objs = {rid: schema_obj(rid, base) for rid in rung_ids}
    objs.update(dict(extra or {}))
    entries = {}
    for sid, obj in objs.items():
        rel = f"{sid}.json"
        fsha = write_file(out_dir / rel, _dump(obj))
        ds = load_derived_schema(out_dir / rel, base)
        entries[sid] = {"path": rel, "self_sha256": ds.self_sha256, "file_sha256": fsha,
                        "n_information_channels": ds.n_information_channels, "widths": obj["widths"]}
    reg = registry_obj(entries, base)
    write_file(out_dir / REGISTRY_PATH.name, _dump(reg))
    return reg


def read_registry(path: Path = REGISTRY_PATH) -> dict:
    reg = _read(path)
    if reg.get("format") != REGISTRY_FORMAT:
        raise SchemaError(f"registry format {reg.get('format')!r} != {REGISTRY_FORMAT!r}")
    if reg.get("self_sha256") != self_hash(reg):
        raise SchemaError("derived-schema registry self-hash mismatch")
    return reg


def load_registered(schema_id: str, registry_path: Path = REGISTRY_PATH,
                    base: Optional[CleanV2Schema] = None) -> DerivedSchema:
    """A registered derived schema: registry self-hash, file sha256 and schema self-hash all verified."""
    reg = read_registry(registry_path)
    ent = reg["schemas"].get(schema_id)
    if ent is None:
        raise SchemaError(f"schema {schema_id!r} is not in {registry_path} ({sorted(reg['schemas'])})")
    p = Path(registry_path).parent / ent["path"]
    if _sha(p) != ent["file_sha256"]:
        raise SchemaError(f"{p}: file sha256 differs from the registry")
    ds = load_derived_schema(p, base)
    if ds.self_sha256 != ent["self_sha256"] or ds.schema_id != schema_id:
        raise SchemaError(f"{p}: schema id / self-hash differ from the registry entry")
    return ds


def schema_groups(schema: DerivedSchema, *, include_composite: bool = False) -> dict:
    """FEATSTAT groups restricted to the schema's kept channels (non-empty groups only), in FEATSTAT order."""
    src = groups() if include_composite else partition_groups()
    kept = set(schema.channel_names)
    out = {}
    for g, mem in src.items():
        m = [c for c in schema.channel_names if c in mem and c in kept]
        if m:
            out[g] = m
    covered = set().union(*map(set, partition_groups().values())) if partition_groups() else set()
    stray = sorted(kept - covered - {SESSION_EXTENSION} - set(SESSION_INTENT))
    if stray:
        raise SchemaError(f"channels outside every FEATSTAT group: {stray}")
    return out


def derive(base_id: str, drop_groups: Iterable[str], *, registry_path: Path = REGISTRY_PATH,
           base_schema: Optional[DerivedSchema] = None) -> DerivedSchema:
    """`base_id` (a registered derived schema, default L3) minus every channel of `drop_groups` (FEATSTAT groups)."""
    drop = sorted(set(str(g) for g in drop_groups))
    all_groups = dict(groups())
    all_groups.update({g: list(m) for g, m in FS_GROUPS.items()})       # the FS groups G1 ... G12c + G13 (DL 48)
    unknown = [g for g in drop if g not in all_groups]
    if unknown:
        raise SchemaError(f"unknown FEATSTAT groups {unknown}; known: {sorted(all_groups)}")
    req = {g: sorted(set(all_groups[g]) & set(REQUIRED_CHANNELS)) for g in drop
           if g in REQUIRED_GROUPS or set(all_groups[g]) & set(REQUIRED_CHANNELS)}
    if req:
        raise SchemaError(f"groups {req} hold REQUIRED channels {REQUIRED_CHANNELS} (DECISION_LOG 47): never dropped")
    src = base_schema if base_schema is not None else load_registered(base_id, registry_path)
    if src.schema_id != base_id:
        raise SchemaError(f"base schema id {src.schema_id!r} != {base_id!r}")
    gone = set().union(*(set(all_groups[g]) for g in drop)) if drop else set()
    not_in = [g for g in drop if not (set(all_groups[g]) & set(src.channel_names))]
    if not_in:
        raise SchemaError(f"groups {not_in} have no channel in {base_id}")
    names = [n for n in src.channel_names if n not in gone]
    lost = [c for c in REQUIRED_CHANNELS if c in src.channel_names and c not in names]
    if lost:                                                       # belt and braces: never reached through `drop`
        raise SchemaError(f"deriving from {base_id} would drop REQUIRED channels {lost}")
    sid = base_id if not drop else f"{base_id}-minus-{'+'.join(drop)}"
    obj = make_derived(src.base, sid, keep_from_names(src.base, names), title=f"{base_id} minus {drop}",
                       source=source_block(None, {"derived_from": {"schema_id": base_id,
                                                                    "self_sha256": src.self_sha256},
                                                  "dropped_groups": drop, "dropped_channels": sorted(gone
                                                  & set(src.channel_names))}))
    return derived_from_obj(obj, src.base)


def derived_obj(ds: DerivedSchema) -> dict:
    return dict(ds.raw)


# ------------------------------------------------------------------------------------------------ theta0 registry
def read_theta0(path: Path = THETA0_PATH) -> dict:
    reg = _read(path)
    if reg.get("format") != THETA0_FORMAT or reg.get("self_sha256") != self_hash(reg):
        raise SchemaError(f"{path}: theta0 registry format / self-hash invalid")
    return reg


def registered_theta0(schema_id: str, schema_self_sha256: str, family: str, seed: int, torch_build: str,
                      path: Path = THETA0_PATH) -> Optional[dict]:
    """{init_sha256, raw_init_sha256, manifest_digest, shared_numel} for (schema, family, seed, torch build) or None."""
    if not Path(path).is_file():
        return None
    reg = read_theta0(path)
    ent = reg["schemas"].get(schema_id)
    if ent is None or ent["self_sha256"] != schema_self_sha256:
        return None
    return ent["builds"].get(torch_build, {}).get(str(family).upper(), {}).get(str(int(seed)))


def theta0_obj(existing: Optional[dict], records: Iterable[Mapping]) -> dict:
    """Merge measured records {schema_id, schema_self_sha256, torch, family, seed, init_sha256, raw_init_sha256,
    manifest_digest, shared_numel, host, utc} into the theta0 registry (a changed value for a key is refused)."""
    reg = {"format": THETA0_FORMAT, "task": "SCHEMA-01",
           "definition": "fl.numerics.state_digest(adapter.broadcast_state()) of v4models.build(family, "
                         "derived schema, seed) at the full catalogue (K = 158,486; CPU build before any device move; "
                         "SASRec after the one C7 recentering)", "schemas": {}}
    if existing:
        reg["schemas"] = json.loads(json.dumps(existing["schemas"]))
    for r in records:
        ent = reg["schemas"].setdefault(r["schema_id"], {"self_sha256": r["schema_self_sha256"], "builds": {}})
        if ent["self_sha256"] != r["schema_self_sha256"]:
            raise SchemaError(f"{r['schema_id']}: schema self-hash changed ({ent['self_sha256']} -> "
                              f"{r['schema_self_sha256']})")
        fam = ent["builds"].setdefault(r["torch"], {}).setdefault(str(r["family"]).upper(), {})
        val = {k: r[k] for k in ("init_sha256", "raw_init_sha256", "manifest_digest", "shared_numel", "K")}
        val.update({"host": r.get("host"), "utc": r.get("utc")})
        old = fam.get(str(int(r["seed"])))
        if old is not None and {k: old[k] for k in val if k not in ("host", "utc")} != \
                {k: val[k] for k in val if k not in ("host", "utc")}:
            raise SchemaError(f"theta0 for {r['schema_id']}/{r['torch']}/{r['family']}/{r['seed']} differs from the "
                              "registered value")
        if old is None:
            fam[str(int(r["seed"]))] = val
    reg["self_sha256"] = self_hash(reg)
    return reg


# ------------------------------------------------------------------------------------------- safe theta0 merging
# FS-02 (Stage B): `v4fs.theta0 --merge-into` has so far only ever been run locally; on the server the SAME
# THETA0_REGISTRY.json is also read (not written) by concurrently running run_fs / theta0 preflights, and a Stage B
# retry could in principle race a second merge. `merge_theta0` makes the read-modify-write safe for that case: a
# plain, portable exclusive lock FILE (os.O_CREAT | O_EXCL; no OS-specific locking API, so it works identically on
# the local Windows CPU build and the server Linux GPU build) serializes concurrent merges into the SAME path, with
# a bounded wait/timeout instead of a silent lost update. `theta0_obj` still refuses a changed value for an existing
# key (the self-hash rule is unchanged); a reader only ever sees the old or the new file (write_file's tmp+rename).
def merge_theta0(path: Path = THETA0_PATH, records: Iterable[Mapping] = (), *, timeout_s: float = 120.0,
                 poll_s: float = 0.2) -> dict:
    """Read-modify-write `path` (THETA0_REGISTRY.json) with `records` merged in (v4fs.theta0.theta0_obj), guarded by
    a short-lived exclusive lock file so a concurrent merge into the SAME path waits instead of racing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    t0 = time.time()
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - t0 > timeout_s:
                raise SchemaError(f"{lock}: another theta0 merge holds the lock (timeout {timeout_s}s); "
                                  "remove it by hand only after confirming no merge is actually running")
            time.sleep(poll_s)
    try:
        os.write(fd, f"{platform.node()}:{os.getpid()}:"
                     f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}".encode("utf-8"))
        os.close(fd)
        fd = None
        old = read_theta0(path) if path.is_file() else None
        reg = theta0_obj(old, records)
        write_file(path, _dump(reg))
        return reg
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.remove(lock)
        except OSError:
            pass


__all__ = ["FS_GROUPS", "REQUIRED_CHANNELS", "REFERENCE_ID", "CANDIDATE_BASE_ID", "SESSION_SCHEMAS",
           "WRITTEN_SCHEMAS", "SESSION_INTENT", "session_schema_obj", "schema_obj", "LADDER_PATH", "INVENTORY_PATH", "DERIVED_DIR", "REGISTRY_PATH", "THETA0_PATH", "WRITTEN_RUNGS",
           "ladder", "rung", "groups", "partition_groups", "rung_channel_names", "rung_schema_obj", "rung_schema",
           "write_registry", "read_registry", "load_registered", "schema_groups", "derive", "read_theta0",
           "registered_theta0", "theta0_obj", "merge_theta0", "write_file"]

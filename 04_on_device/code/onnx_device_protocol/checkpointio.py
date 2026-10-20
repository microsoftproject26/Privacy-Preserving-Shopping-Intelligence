"""v4dev.checkpointio — read-only checkpoint loading + model construction for the ONNX protocol (DEV-01 item 1).

Accepts either a v4train checkpoint record (`v4train.checkpoint`: a dict with `kind` / `meta` / `model`, `model`
being the state dict) or a bare state dict (any other `torch.load`-able mapping of tensors) — DEV-01 never writes a
checkpoint, only reads one Ops points it at (`--ckpt`), so no v4train.checkpoint.CheckpointStore (a live run
directory) is required. `weights_only=True` throughout (never unpickles arbitrary objects).

DEV-04 (the checkpoint's OWN registered feature view). A v4train run can be trained on a registered DERIVED schema
(v4models.derived; e.g. C_SMALL: L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7, which keeps the SCHEMA-01R1
extension side channels and drops main_category / daypart / weekday and most numeric / flag / user-context columns).
Building the default Clean V2 module for such a checkpoint fails the strict state-dict load (INT8_BENCH.log, 2026-10-01).
`resolve_feature_view` therefore builds the module from the view the checkpoint itself records, exactly like the
trainer (v4int.launch_central.load_production_schema) and the re-evaluator (v4int.reeval: RUN_START schema ->
v4fs.schemas.load_registered -> v4models.build):
  1. the view identity (schema_id, schema_self_sha256) is read from the checkpoint meta (`meta.schema`, written by
     v4train.trainer for every record) and from the run's RUN_START.json (`schema`, located by the reeval rule
     <run>/[sealed/]checkpoints/<file>); an explicit `schema_id` may be given too. Every present source must agree;
  2. the schema object is loaded from its registry, never from tensor shapes: the base Clean V2 schema
     (v4models.schema.load_schema) when the id / self-hash are the base's, else v4fs.schemas.load_registered
     (registry self-hash, file sha256 and schema self-hash verified) with the RUN_START-recorded registry, whose file
     sha256 must equal the one RUN_START recorded;
  3. family / model_variant / seed / config_sha256 / init_sha256 are cross-checked between the meta and RUN_START;
     the module is built with the run's own seed and variant and its theta0 digest must reproduce the recorded
     init_sha256 (reeval.load_central's rule); after the strict load the state digest must equal meta.state_digest;
  4. the strict load is the shape assertion: any missing / unexpected key or width difference refuses.
A bare state dict (no meta, no run directory) has no record: it is built with the base Clean V2 schema (the only
default) or an explicit `schema_id`, and the strict load refuses any other layout. A v4train record (non-empty meta)
with no view record anywhere is refused."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch

from v4models import build as build_model    # the build(...) function v4models/__init__.py re-exports
from v4models.schema import CleanV2Schema, InputWidths, load_schema

from .scoring import ScoringModule

RUN_START_NAME = "RUN_START.json"
VIEW_FORMAT = "v4dev.feature_view.v1"


class CheckpointIOError(ValueError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state_dict(ckpt_path) -> tuple:
    """(state_dict, meta) from a checkpoint file. `meta` is `{}` for a bare state dict."""
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(obj, Mapping) and "model" in obj and isinstance(obj["model"], Mapping):
        return dict(obj["model"]), dict(obj.get("meta") or {})
    if isinstance(obj, Mapping):
        return dict(obj), {}
    raise CheckpointIOError(f"{ckpt_path}: not a checkpoint record or a state dict (got {type(obj).__name__})")


# ------------------------------------------------------------------------------------- the registered feature view
def run_dir_of(ckpt_path) -> Optional[Path]:
    """<run>/[sealed/]checkpoints/<file> -> <run> (the v4int.reeval.source_central rule); None if not that layout."""
    p = Path(ckpt_path)
    if p.parent.name != "checkpoints":
        return None
    return p.parent.parent.parent if p.parent.parent.name == "sealed" else p.parent.parent


def read_run_start(ckpt_path) -> Optional[dict]:
    """{"path", "sha256", "start"} of the checkpoint's run RUN_START.json, or None when the checkpoint is not under a
    v4train checkpoints/ directory. A checkpoints/ layout WITHOUT a RUN_START is refused (a run record is expected)."""
    rd = run_dir_of(ckpt_path)
    if rd is None:
        return None
    p = rd / RUN_START_NAME
    if not p.is_file():
        raise CheckpointIOError(f"{ckpt_path} is under a checkpoints/ directory but its run has no {RUN_START_NAME}: {p}")
    raw = p.read_bytes()
    try:
        start = json.loads(raw.decode("utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except ValueError as e:
        raise CheckpointIOError(f"{p}: not strict JSON ({e})") from e
    if not isinstance(start, Mapping):
        raise CheckpointIOError(f"{p}: not a JSON object")
    return {"path": p, "sha256": hashlib.sha256(raw).hexdigest(), "start": dict(start)}


def _view_ids(meta: Mapping, start: Optional[Mapping], schema_id: Optional[str]) -> dict:
    """The (schema_id, schema_self_sha256) claims of every present source; all must agree."""
    claims = {}
    ms = meta.get("schema") if isinstance(meta, Mapping) else None
    if isinstance(ms, Mapping) and ms.get("schema_id"):
        claims["checkpoint_meta"] = (str(ms["schema_id"]), ms.get("schema_self_sha256"))
    if start is not None:
        ss = start.get("schema")
        if isinstance(ss, Mapping) and ss.get("schema_id"):
            claims["RUN_START"] = (str(ss["schema_id"]), ss.get("schema_self_sha256") or start.get("schema_self_sha256"))
    if schema_id is not None:
        claims["explicit"] = (str(schema_id), None)
    ids = {c[0] for c in claims.values()}
    if len(ids) > 1:
        raise CheckpointIOError(f"the feature-view sources disagree on the schema id: {claims}")
    selfs = {c[1] for c in claims.values() if c[1] is not None}
    if len(selfs) > 1:
        raise CheckpointIOError(f"the feature-view sources disagree on the schema self-hash: {claims}")
    return {"claims": claims, "schema_id": next(iter(ids)) if ids else None,
            "schema_self_sha256": next(iter(selfs)) if selfs else None}


def _registry_path(start: Optional[Mapping], schema_registry) -> tuple:
    """(registry path, recorded registry file sha256 or None): explicit > RUN_START-recorded (if present on disk) >
    v4train.registry.DERIVED_SCHEMA_REGISTRY_PATH. The recorded file sha256 (RUN_START) binds whichever is used."""
    rec = ((start or {}).get("schema") or {}).get("registry") or {}
    recorded_sha = rec.get("file_sha256")
    if schema_registry is not None:
        return Path(schema_registry), recorded_sha
    if rec.get("path") and Path(rec["path"]).is_file():
        return Path(rec["path"]), recorded_sha
    from v4train.registry import DERIVED_SCHEMA_REGISTRY_PATH
    return Path(DERIVED_SCHEMA_REGISTRY_PATH), recorded_sha


def load_view_schema(schema_id: str, schema_self_sha256: Optional[str], *, start: Optional[Mapping] = None,
                     schema_registry=None) -> tuple:
    """(schema object, record) for a registered view id: the base Clean V2 schema when the id is the base's, else a
    registered derived schema (v4fs.schemas.load_registered). Refuses a self-hash or registry-file mismatch."""
    base = load_schema()
    if schema_id == base.schema_id:
        if schema_self_sha256 is not None and schema_self_sha256 != base.self_sha256:
            raise CheckpointIOError(f"base schema {schema_id}: recorded self-hash {schema_self_sha256} != the loaded "
                                    f"{base.self_sha256}")
        return base, {"kind": "BASE", "schema_id": base.schema_id, "schema_self_sha256": base.self_sha256,
                      "file": str(base.path), "file_sha256": base.file_sha256, "registry": None}
    from v4fs.schemas import load_registered
    reg_path, recorded_sha = _registry_path(start, schema_registry)
    if not reg_path.is_file():
        raise CheckpointIOError(f"derived-schema registry not found: {reg_path}")
    reg_sha = sha256_file(reg_path)
    if recorded_sha is not None and reg_sha != recorded_sha:
        raise CheckpointIOError(f"{reg_path}: file sha256 {reg_sha} != the RUN_START-recorded {recorded_sha}")
    try:
        ds = load_registered(schema_id, reg_path)
    except Exception as e:                                      # SchemaError / KeyError / OSError: refuse, never guess
        raise CheckpointIOError(f"schema {schema_id!r} could not be loaded from {reg_path}: {type(e).__name__}: {e}") from e
    if schema_self_sha256 is not None and ds.self_sha256 != schema_self_sha256:
        raise CheckpointIOError(f"schema {schema_id}: registered self-hash {ds.self_sha256} != the recorded "
                                f"{schema_self_sha256}")
    return ds, {"kind": "DERIVED", "schema_id": ds.schema_id, "schema_self_sha256": ds.self_sha256, "file": ds.path,
                "file_sha256": ds.file_sha256, "registry": {"path": str(reg_path), "file_sha256": reg_sha}}


def _check_run_identity(family: str, meta: Mapping, start: Optional[Mapping]) -> dict:
    """family / model_variant / seed / config_sha256 / init_sha256: meta vs RUN_START (each where both state it)."""
    fam = str(family).upper()
    got = {}
    for src_name, src in (("checkpoint_meta", meta), ("RUN_START", start or {})):
        cfg = (src.get("config") or {}) if src_name == "RUN_START" else {}
        for key in ("family", "model_variant", "seed", "config_sha256", "init_sha256"):
            v = src.get(key, cfg.get(key))
            if v is None:
                continue
            if key in got and got[key][1] != v:
                raise CheckpointIOError(f"{key}: {got[key][0]} records {got[key][1]!r} but {src_name} records {v!r}")
            got.setdefault(key, (src_name, v))
    if "family" in got and str(got["family"][1]).upper() != fam:
        raise CheckpointIOError(f"the checkpoint is a {got['family'][1]} run, not {fam}")
    return {k: v[1] for k, v in got.items()}


def resolve_feature_view(ckpt_path, family: str, *, meta: Optional[Mapping] = None, schema_id: Optional[str] = None,
                         schema_registry=None) -> dict:
    """The checkpoint's own registered feature view (module docstring, steps 1-3 identity part). Returns
    {"schema": CleanV2Schema | DerivedSchema, "record": {...}, "identity": {...}, "run_start": RUN_START dict or None}.
    `meta` defaults to the checkpoint's own (read here when not given)."""
    if meta is None:
        _, meta = load_state_dict(ckpt_path)
    rs = read_run_start(ckpt_path)
    start = rs["start"] if rs is not None else None
    ids = _view_ids(meta, start, schema_id)
    identity = _check_run_identity(family, meta, start)
    if ids["schema_id"] is None:
        if meta:
            raise CheckpointIOError(f"{ckpt_path}: a v4train checkpoint record with no registered feature view "
                                    f"(no meta.schema, no RUN_START schema); give the registered schema id explicitly")
        base = load_schema()
        ids = {"claims": {"default_bare_state_dict": (base.schema_id, base.self_sha256)},
               "schema_id": base.schema_id, "schema_self_sha256": base.self_sha256}
    schema, srec = load_view_schema(ids["schema_id"], ids["schema_self_sha256"], start=start,
                                    schema_registry=schema_registry)
    if start is not None:
        ss = start.get("schema") or {}
        if ss.get("file_sha256") and srec.get("file_sha256") and ss["file_sha256"] != srec["file_sha256"]:
            raise CheckpointIOError(f"schema {srec['schema_id']}: file sha256 {srec['file_sha256']} != the "
                                    f"RUN_START-recorded {ss['file_sha256']}")
    kb = (start or {}).get("key_base") or {}
    record = {"format": VIEW_FORMAT, **srec, "sources": sorted(ids["claims"]),
              "run_start": ({"path": str(rs["path"]), "sha256": rs["sha256"]} if rs is not None else None),
              "eval_manifest_id": kb.get("eval_manifest_id"), "history_policy": kb.get("history_policy"),
              "identity": {k: v for k, v in identity.items()}}
    return {"schema": schema, "record": record, "identity": identity, "run_start": start}


def base_schema_of(schema) -> CleanV2Schema:
    """The Clean V2 base a view is a projection of (the schema itself for the base)."""
    return schema.base if getattr(schema, "is_derived_schema", False) else schema


def view_required_row_keys(schema) -> tuple:
    """The batch keys a model of this view consumes from a FULL Clean V2 batch (the adapter projects that batch):
    item_tokens / lengths / attention_mask, the kept token channels (base and extension), the float blocks the model
    has a branch for (full Clean V2 width: the projection selects by name index), and the kept decision-level
    extension floats (appended by the projection)."""
    from v4models.derived import EXTENSION_CHANNELS
    w = schema.widths
    keys = ["item_tokens", "lengths", "attention_mask"]
    keys += [c for c in w.categorical if c != "item_tokens"]
    if w.numeric_dim + w.flags_dim > 0:
        keys += ["event_numeric_features", "event_quality_flags"]
    if w.user_context_dim + w.user_mask_dim > 0:
        keys += ["user_context", "user_context_masks"]
    if getattr(schema, "is_derived_schema", False):
        keys += [c for blk in ("user_context", "user_context_masks") for c in schema.keep[blk] if c in EXTENSION_CHANNELS]
    return tuple(dict.fromkeys(keys))


def _resolve_variant(family: str, variant: Optional[str], identity: Mapping) -> Optional[str]:
    from v4models import variants
    rec = identity.get("model_variant")
    if rec is None:
        return variant
    name, _ = variants.resolve(family, variant if variant is not None else rec)
    if name != rec:
        raise CheckpointIOError(f"--variant {variant!r} resolves to {name} but the checkpoint was trained as {rec}")
    return rec


def build_scoring_module(family: str, *, variant: Optional[str] = None, ckpt_path=None, seed: int = 2026,
                         k: Optional[int] = None, poc: Optional[np.ndarray] = None,
                         catalogue: Optional[str] = None, v1_root: Optional[str] = None,
                         schema: Optional[CleanV2Schema | InputWidths] = None, schema_id: Optional[str] = None,
                         schema_registry=None) -> dict:
    """Build the family/variant module (fresh init if `ckpt_path` is None, else the checkpoint's weights loaded
    strictly on top of it) and wrap it as a `ScoringModule` in eval mode.

    Class map (v4models.build.build's own rule): give exactly one of `k` (a small synthetic catalogue override —
    tests only), `poc`, `catalogue` (a local catalogue.parquet) or `v1_root` (the real, hash-checked v1 mapping
    files — the on-server invocation).

    Feature view (DEV-04): with `ckpt_path`, the module is built with the checkpoint's OWN registered view
    (`resolve_feature_view`; `schema` must then be None, `schema_id` may name it explicitly and must agree with the
    record), with the run's recorded seed and variant (a different `--variant` refuses); the rebuilt theta0 must
    reproduce the recorded init_sha256 and the loaded weights the recorded state_digest. Without `ckpt_path`,
    `schema` defaults to `v4models.schema.load_schema()` (or the registered `schema_id`).

    Returns `{"scoring", "built", "checkpoint_meta", "state_dict_keys", "feature_view"}` (`feature_view`: the view
    record, None for a fresh init built from an explicit `schema`).

    SASRec's `impl` is forced to `"padded_reference"` here regardless of what the checkpoint trained with: `impl`
    picks one of two numerically-equivalent forward-compute paths over the SAME parameters (v4models/sasrec_v2.py:
    "Two numerically equivalent implementations (tests compare them)"), never a weight/shape difference, so flipping
    it after loading is always safe. `"padded_reference"` is the mask-based, ONNX-traceable path; the default
    `"unpadded"` builds a Python-level `flat_idx` gather whose shape depends on the traced example's data (not
    ONNX-safe, though it does not crash the exporter the way GRU's pack/unpack does -- see v4dev.scoring)."""
    meta: dict = {}
    view = None
    state = None
    build_seed = seed
    if ckpt_path is not None:
        if schema is not None:
            raise CheckpointIOError("with --ckpt the schema is the checkpoint's own registered view; do not pass one")
        state, meta = load_state_dict(ckpt_path)
        view = resolve_feature_view(ckpt_path, family, meta=meta, schema_id=schema_id, schema_registry=schema_registry)
        given_schema = view["schema"]
        variant = _resolve_variant(family, variant, view["identity"])
        if view["identity"].get("seed") is not None:
            build_seed = int(view["identity"]["seed"])          # theta0 identity: the run's own construction seed
    elif schema is not None:
        given_schema = schema
    elif schema_id is not None:
        given_schema, srec = load_view_schema(schema_id, None, schema_registry=schema_registry)
        view = {"record": {"format": VIEW_FORMAT, **srec, "sources": ["explicit"]}}
    else:
        given_schema = load_schema()
    built = build_model(family, given_schema, build_seed, "cpu", K=k, poc=poc, catalogue=catalogue, v1_root=v1_root,
                        variant=variant)
    if ckpt_path is not None:
        want_init = view["identity"].get("init_sha256")
        if want_init is not None and built.init_sha256 != want_init:
            raise CheckpointIOError(f"the rebuilt theta0 {built.init_sha256} (view {view['record']['schema_id']}, "
                                    f"variant {built.variant}, seed {build_seed}) differs from the run's recorded "
                                    f"init_sha256 {want_init}: not the checkpoint's construction")
        try:
            built.module.load_state_dict(state, strict=True)     # strict=True: torch raises RuntimeError itself
        except RuntimeError as e:                                 # on any missing/unexpected key, never returns
            raise CheckpointIOError(f"checkpoint state dict does not match the module built with its registered view "
                                    f"{view['record']['schema_id']} (variant {built.variant}): {e}") from e
        if meta.get("state_digest") is not None:
            from fl.numerics import state_digest
            d = state_digest(built.adapter.broadcast_state())
            if d != meta["state_digest"]:
                raise CheckpointIOError(f"the loaded weights' state digest {d} != the checkpoint's {meta['state_digest']}")
    if str(built.family).upper() == "SASREC":
        built.module.impl = "padded_reference"
    built.module.eval()
    scoring = ScoringModule(built)
    fv = None
    if view is not None:
        fv = dict(view["record"], variant=built.variant, build_seed=int(build_seed),
                  init_sha256=built.init_sha256, input_keys=list(scoring.input_keys),
                  required_row_keys=list(view_required_row_keys(given_schema)))
    return {"scoring": scoring, "built": built, "checkpoint_meta": meta,
            "state_dict_keys": len(list(built.module.state_dict())), "feature_view": fv,
            "schema": given_schema}


__all__ = ["CheckpointIOError", "RUN_START_NAME", "VIEW_FORMAT", "sha256_file", "load_state_dict", "run_dir_of",
           "read_run_start", "load_view_schema", "resolve_feature_view", "base_schema_of", "view_required_row_keys",
           "build_scoring_module"]

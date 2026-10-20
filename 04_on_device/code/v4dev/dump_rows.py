"""v4dev.dump_rows — materialize a role's evaluation-manifest E2E rows into the `--rows` NPZ `v4dev.bench` reads
(DEV-02; closes DEV-01 open question 2).

    python -B -m v4dev.dump_rows --role MAIN --history-policy USER_HISTORY_L50 --out <dir>/rows.npz
        [--first-users 16384]

Writes, in manifest order, EVERY E2E (`eval_mask`) row of the role's real eval manifest (StoreEvalProvider /
v4int.providers, exactly the object `v4int.launch_central.run` builds for training — same manifest verification,
same Clean V2 store, same history policy), as one array per batch field the provider actually yields (whatever a
real production batch carries: `item_tokens`, `lengths`, `attention_mask`, the categorical/numeric/flags/user-context
channels present for this data version — a superset of any one family's `v4dev.scoring.input_keys`, so the same NPZ
serves GRU and SASREC benches alike) PLUS `manifest_row` (each row's 0-based position in the FULL eval manifest,
`v4eval.predict.stream_rows` order — the index space `v4dev.bench.ranks_from_int8_session` scatters into, sized
`manifest.n`, NOT the E2E-row count). `decision_id` / `target_class` are already among the provider's own batch
fields (checked against the manifest by the provider itself on every batch), so they need no separate handling here.
Also writes `<out's directory>/rows.meta.json`: manifest_id, manifest content_sha256 (`manifest_binding`, the
REGISTRY_FS rule), `manifest_n` (the full-manifest length -- what Ops must pass to `v4dev.bench --manifest-n`),
n_rows / n_users of the rows actually written, history_policy, role, `first_users` (null unless given), and the
written NPZ's own sha256.

`--first-users N` (PREG1_AMENDMENT_06A item 6's registered deterministic fallback, "the first N ... users in
population-selection-key order"; REV-ANA-01 D-F2, option b): the E2E users (users with at least one `eval_mask` row)
are ordered by the POPULATION selection key v2, `sha256(SEL_PREFIX_V2 + digest32)[:8]` big-endian
(`census_v2.core.selection_key_v2`, the key MAIN and the 5-100 population are ordered by), computed from each
user's `client_key_hex` (`v4base.identity.eval_client_key_hex_by_eval_user_key`, read-only import; the eval
manifest's own `users` table), NOT from `eval_user_key < N` (that dense rank is the v1 client-digest order, a
different permutation). The first N in that order are kept (ties on the 64-bit key, which do not occur in practice,
break by eval_user_key). rows.meta.json records `first_users_rule`, `first_users`, `n_e2e_users_available`,
`n_users`, `first_users_selection_key_prefix` and `first_users_set_sha256` (sha256 of the sorted selected
client_key_hex joined by newlines; the ANA `users_subset` must equal this set). When fewer than N users have E2E
rows, all of them are kept and n_users < N is visible in the meta. NOTE: `np.savez` embeds zip timestamps, so
`rows_npz_sha256` is NOT reproducible across runs; it binds the file that was used, nothing more.

Numeric array fields only (GATE-B B2; DEV-04 / DL 209 crash; review 12 [11-R1]): a real StoreEvalProvider batch also
carries provenance scalars / channel-name tuples / dicts (feature_view, clean_v2_rc_counts, ...) that np.concatenate
cannot stack. `dump_from_provider` keeps a batch field only if it is an array with a row axis (a tensor or ndarray,
not str / bytes: the exact `v4int.dump_rows_retro.ArrayFieldsView._is_array` rule) whose dtype is numeric or bool
(`numeric_array_field`); the dropped field names are recorded in the meta (`dropped_non_array_fields`, present only
when a field was dropped, as in dump_rows_retro), so the VALIDATION and the RETRO_TEST NPZ have one schema.

Feature view (DEV-04): providers serve the FULL Clean V2 batch and a derived-schema model's adapter projects it, so
the rows are the full batch for every view (the C_SMALL rows carry all five SCHEMA-01R1 extension keys). What the NPZ
cannot carry is the column ORDER the projection's name index relies on: `dump_from_provider` therefore records the
batch's view provenance in the meta as `batch_view` (feature_view, clean_v2_schema_self_sha256 and the four Clean V2
name arrays; each must be identical in every batch, else refused) and every kept field's dtype / per-row shape
(`fields`). `--ckpt` binds the dump to a checkpoint's own registered view (v4dev.checkpointio.resolve_feature_view):
the role's eval manifest and the history policy must be the run's, and `check_rows_view` must pass (every key the
view consumes present, float blocks at the full Clean V2 width, names / view / base self-hash equal to the view's
base); the binding is recorded as `feature_view_binding`. v4dev.bench runs the same `check_rows_view` before
scoring and refuses rows of another view.

Everything above the CLI (`dump_from_provider`, `population_selection_first_users_mask`, `write_rows`) is pure /
provider-duck-typed and is what `tests_dev/test_dump_rows.py` exercises against synthetic `v4eval.manifest`
fixtures (no Clean V2 store, no server). Building the REAL provider (`build_eval_provider`) reuses
`v4int.launch_central`'s own lock-binding / trainer-registry-view / `StoreEvalProvider` construction verbatim
(read-only import; v4int is not in this package's edit scope) and is exercised only on the server, where the real
data lives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from v4dev.checkpointio import sha256_file

DEFAULT_EVAL_PREFETCH = 0


class DumpRowsError(ValueError):
    pass


# --------------------------------------------------------------------------------------------- pure helpers (local)
def _rows_len(batch: Mapping[str, np.ndarray]) -> int:
    for v in batch.values():
        return int(np.asarray(v).shape[0])
    raise DumpRowsError("a provider batch carried no fields")


NUMERIC_KINDS = "biuf"                                    # bool, signed / unsigned int, float


def numeric_array_field(v) -> bool:
    """True for a batch field `dump_from_provider` keeps: an array with a row axis (tensor / ndarray, not str / bytes;
    the v4int.dump_rows_retro.ArrayFieldsView._is_array rule) whose dtype kind is bool / int / uint / float."""
    if isinstance(v, (str, bytes)) or not hasattr(v, "shape") or len(getattr(v, "shape")) < 1:
        return False
    if not (hasattr(v, "numpy") or isinstance(v, np.ndarray)):
        return False
    return np.asarray(v).dtype.kind in NUMERIC_KINDS


FIRST_USERS_RULE = ("first N E2E users in selection_key_v2 order: sha256(SEL_PREFIX_V2 + digest32)[:8] big-endian "
                    "over client_key_hex (census_v2.core.selection_key_v2), ties by eval_user_key")


def first_users_selection(eval_user_key_stream: np.ndarray, client_key_hex, first_users: int) -> dict:
    """The first `first_users` distinct users (of those present in `eval_user_key_stream`, i.e. with E2E rows) in
    selection_key_v2 order. `client_key_hex[euk]` is the hex client digest of eval_user_key `euk` (64 hex chars).
    Returns {"keys": selected eval_user_keys in selection order, "n_available", "set_sha256"}."""
    from census_v2.core import SEL_PREFIX_V2, selection_key_v2
    n = int(first_users)
    if n < 0:
        raise DumpRowsError("--first-users must be >= 0")
    euk = np.asarray(eval_user_key_stream)
    if euk.ndim != 1:
        raise DumpRowsError("eval_user_key (stream) must be 1-D")
    if client_key_hex is None:
        raise DumpRowsError("--first-users needs client_key_hex per eval_user_key (selection_key_v2 order); "
                            "eval_user_key < N is the v1 digest order and is NOT accepted")
    users = np.unique(euk)
    if users.size and (int(users.min()) < 0 or int(users.max()) >= len(client_key_hex)):
        raise DumpRowsError("an eval_user_key is outside the client_key_hex table")
    keyed = []
    for u in users.tolist():
        h = str(client_key_hex[u])
        try:
            digest = bytes.fromhex(h)
        except ValueError as e:
            raise DumpRowsError(f"client_key_hex of eval_user_key {u} is not hex") from e
        keyed.append((selection_key_v2(digest), u, h))
    keyed.sort(key=lambda t: (t[0], t[1]))
    chosen = keyed[:n]
    set_sha = hashlib.sha256("\n".join(sorted(h for _, _, h in chosen)).encode("ascii")).hexdigest()
    return {"keys": np.asarray([u for _, u, _ in chosen], dtype=np.int64), "n_available": int(users.size),
            "set_sha256": set_sha, "prefix": SEL_PREFIX_V2.decode("ascii")}


def population_selection_first_users_mask(eval_user_key_stream: np.ndarray, first_users: Optional[int],
                                          client_key_hex=None) -> np.ndarray:
    """A boolean mask (stream order) of rows whose user is among the first `first_users` E2E users in
    selection_key_v2 order (module docstring). `None` keeps every row (no client_key_hex needed)."""
    euk = np.asarray(eval_user_key_stream)
    if euk.ndim != 1:
        raise DumpRowsError("eval_user_key (stream) must be 1-D")
    if first_users is None:
        return np.ones(euk.shape[0], dtype=bool)
    sel = first_users_selection(euk, client_key_hex, first_users)
    return np.isin(euk, sel["keys"])


def dump_from_provider(provider, *, first_users: Optional[int] = None, client_key_hex=None) -> tuple:
    """(rows, meta): `rows` is the NPZ-ready dict (module docstring's layout); `meta` is the `rows.meta.json`
    content minus the NPZ's own sha256 (added once the file is actually written, by `write_rows`).

    `provider` is duck-typed to `v4int.providers.StoreEvalProvider`'s public surface: `.manifest` (a
    `v4eval.manifest.EvalManifest`), `.eval_batch` (int), `.batches(bs)` (an iterator of field-name -> array-like,
    batch-of-`bs` rows, in `v4eval.predict.stream_rows` order -- exactly the E2E rows, manifest order), `.manifest_id`,
    `.manifest_binding` (a dict with `content_sha256`), `.history_policy`."""
    from v4eval.predict import stream_rows

    manifest = provider.manifest
    stream = stream_rows(manifest)                            # full-manifest positions of the E2E rows, manifest order
    n_stream = int(stream.shape[0])

    chunks: dict = {}
    keys: Optional[set] = None
    dropped: set = set()
    view: Optional[dict] = None
    seen = 0
    for raw in provider.batches(int(provider.eval_batch)):
        batch = {k: v for k, v in raw.items() if numeric_array_field(v)}      # B2: numeric array fields only
        dropped |= set(raw) - set(batch)
        bview = batch_view_of(raw)                                           # DEV-04: column-order provenance
        if view is None:
            view = bview
        elif bview != view:
            raise DumpRowsError(f"provider batches disagree on the batch view: {view} vs {bview}")
        bs = _rows_len(batch)
        bkeys = set(batch.keys())
        if keys is None:
            keys = bkeys
        elif bkeys != keys:
            raise DumpRowsError(f"provider batches disagree on field names: {sorted(keys)} vs {sorted(bkeys)}")
        for k, v in batch.items():
            chunks.setdefault(k, []).append(np.asarray(v))
        seen += bs
    if seen != n_stream:
        raise DumpRowsError(f"the provider yielded {seen} rows but manifest.eval_mask has {n_stream} E2E rows")
    if "manifest_row" in (keys or ()):
        raise DumpRowsError("a provider batch already carries 'manifest_row' (this function adds it)")

    full = {k: np.concatenate(vs, axis=0) for k, vs in chunks.items()}
    full["manifest_row"] = stream.astype(np.int64)

    euk_stream = np.asarray(manifest.eval_user_key)[stream]
    keep = population_selection_first_users_mask(euk_stream, first_users, client_key_hex)
    rows = {k: v[keep] for k, v in full.items()}

    meta = {"manifest_id": str(manifest.manifest_id), "manifest_content_sha256": provider.manifest_binding["content_sha256"],
            "manifest_n": int(manifest.n), "history_policy": str(provider.history_policy),
            "n_rows": int(rows["manifest_row"].shape[0]), "n_users": int(np.unique(euk_stream[keep]).size),
            "first_users": (int(first_users) if first_users is not None else None),
            "first_users_rule": (FIRST_USERS_RULE if first_users is not None else None)}
    if first_users is not None:
        sel = first_users_selection(euk_stream, client_key_hex, first_users)
        meta.update(n_e2e_users_available=sel["n_available"], first_users_set_sha256=sel["set_sha256"],
                    first_users_selection_key_prefix=sel["prefix"])
    if dropped:
        meta["dropped_non_array_fields"] = sorted(dropped)
    meta["batch_view"] = view or {}
    meta["fields"] = {k: {"dtype": str(v.dtype), "row_shape": [int(x) for x in v.shape[1:]]}
                      for k, v in sorted(rows.items())}
    return rows, meta


BATCH_VIEW_SCALARS = ("feature_view", "clean_v2_schema_self_sha256")


def batch_view_of(raw: Mapping) -> dict:
    """A provider batch's column-order provenance (module docstring, DEV-04): the str scalars `BATCH_VIEW_SCALARS`
    and the Clean V2 name arrays (`v4models.schema.NAME_KEYS`), as JSON-ready str / list-of-str. Absent keys are
    simply not recorded (a synthetic provider carries none)."""
    from v4models.schema import NAME_KEYS
    out: dict = {}
    for k in BATCH_VIEW_SCALARS:
        if k in raw and raw[k] is not None:
            out[k] = str(raw[k])
    for name_key in NAME_KEYS.values():
        if name_key in raw and raw[name_key] is not None:
            v = raw[name_key]
            if isinstance(v, (str, bytes)):
                raise DumpRowsError(f"{name_key} must be a sequence of names, not one string")
            out[name_key] = [x.decode() if isinstance(x, bytes) else str(x) for x in v]
    return out


def check_rows_view(rows: Mapping, meta: Optional[Mapping], schema, *, require_batch_view: bool,
                    run_binding: Optional[Mapping] = None) -> dict:
    """Refuse rows that do not carry `schema`'s registered view (DEV-04). `schema` is a CleanV2Schema or a
    DerivedSchema (v4dev.checkpointio.load_view_schema). Checks, each a refusal:
      * every key `view_required_row_keys(schema)` lists is present, tokens [n, L], float blocks [.., width] at the
        FULL Clean V2 (base) width, decision-level extension floats [n];
      * `meta.batch_view` (required when `require_batch_view`; checked whenever present): feature_view = the base
        view id, clean_v2_schema_self_sha256 = the base self-hash, the name arrays = the base names in order;
      * `meta.feature_view_binding`, when present: the same schema id / self-hash;
      * `run_binding` (the checkpoint run's eval_manifest_id / history_policy), when given: equal to the meta's.
    Returns a small JSON-ready record of what was checked."""
    from v4dev.checkpointio import base_schema_of, view_required_row_keys
    from v4models.schema import NAME_KEYS
    base = base_schema_of(schema)
    bw = base.widths
    meta = dict(meta or {})
    need = view_required_row_keys(schema)
    missing = [k for k in need if k not in rows]
    if missing:
        raise DumpRowsError(f"the rows lack keys the view {schema.schema_id} consumes: {missing}")
    n = int(rows["lengths"].shape[0])
    for k in need:
        shape = tuple(int(x) for x in rows[k].shape)
        if not shape or shape[0] != n:
            raise DumpRowsError(f"rows[{k!r}] has shape {shape}; lengths has {n} rows")
        if k in NAME_KEYS:
            if shape[-1] != bw.width_of(k):
                raise DumpRowsError(f"rows[{k!r}] width {shape[-1]} != the full Clean V2 width {bw.width_of(k)} "
                                    f"the {schema.schema_id} projection selects from")
        elif k.endswith("_tokens") or k == "attention_mask":
            if len(shape) != 2:
                raise DumpRowsError(f"rows[{k!r}] must be [n, L], got {shape}")
        elif len(shape) != 1:
            raise DumpRowsError(f"rows[{k!r}] must be [n], got {shape}")
    bv = meta.get("batch_view")
    if require_batch_view and not bv:
        raise DumpRowsError(f"the rows meta records no batch_view (column-order provenance); the {schema.schema_id} "
                            f"projection selects columns by name index, so rows without it are refused -- re-run "
                            f"v4dev.dump_rows (DEV-04)")
    if bv:
        want = {"feature_view": bw.view_id, "clean_v2_schema_self_sha256": base.self_sha256}
        want.update({NAME_KEYS[k]: list(bw.names_of(k)) for k in NAME_KEYS if bw.names_of(k)})
        if require_batch_view:
            absent = [k for k in want if k not in bv]
            if absent:
                raise DumpRowsError(f"the rows batch_view lacks {absent}")
        bad = sorted(k for k in want if k in bv and bv[k] != want[k])
        if bad:
            raise DumpRowsError(f"the rows batch_view differs from the {schema.schema_id} base view: {bad}")
    fb = meta.get("feature_view_binding")
    if fb and (fb.get("schema_id") != schema.schema_id or fb.get("schema_self_sha256") != schema.self_sha256):
        raise DumpRowsError(f"the rows were dumped for view {fb.get('schema_id')} ({fb.get('schema_self_sha256')}), "
                            f"not {schema.schema_id} ({schema.self_sha256})")
    for key, meta_key in (("eval_manifest_id", "manifest_id"), ("history_policy", "history_policy")):
        want_v = (run_binding or {}).get(key)
        if want_v is not None and meta.get(meta_key) is not None and meta[meta_key] != want_v:
            raise DumpRowsError(f"the rows {meta_key} {meta[meta_key]!r} != the checkpoint run's {key} {want_v!r}")
    return {"schema_id": schema.schema_id, "schema_self_sha256": schema.self_sha256, "required_row_keys": list(need),
            "batch_view_checked": bool(bv), "n_rows": n}


def view_binding_record(view: Mapping, ckpt_path, checked: Mapping, **extra) -> dict:
    """`meta.feature_view_binding` (DEV-04; shared with v4int.dump_rows_retro, DEV-05): the checkpoint's registered
    view, the checkpoint file and its sha256 (v4dev.bench refuses rows bound to another checkpoint), the run record
    and the keys `check_rows_view` verified. `extra` adds path-specific fields (the TEST split / role)."""
    rec = view["record"]
    out = {"schema_id": rec["schema_id"], "schema_self_sha256": rec["schema_self_sha256"], "kind": rec["kind"],
           "checkpoint": str(ckpt_path), "checkpoint_sha256": sha256_file(ckpt_path),
           "run_start": rec.get("run_start"), "required_row_keys": list(checked["required_row_keys"])}
    out.update(extra)
    return out


def write_rows(rows: Mapping[str, np.ndarray], meta: Mapping, out_path) -> dict:
    """Writes `out_path` (the rows NPZ) and `<out_path's directory>/rows.meta.json` (meta + the NPZ's own sha256).
    NOT byte-reproducible: `np.savez` writes zip timestamps, so `rows_npz_sha256` differs between runs of identical
    content (it binds the file actually used; D-F2 disclosure)."""
    from v4train.checkpoint import write_json_atomic

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{k: np.asarray(v) for k, v in rows.items()})
    full_meta = dict(meta)
    full_meta["rows_npz_sha256"] = sha256_file(out_path)
    full_meta["rows_path"] = str(out_path)
    meta_path = out_path.parent / "rows.meta.json"
    write_json_atomic(meta_path, full_meta)
    return {"rows_path": out_path, "meta_path": meta_path, "meta": full_meta}


# ---------------------------------------------------------------------------------------- real provider (server-only)
def build_eval_provider(role: str, history_policy: str, run_dir, *, eval_prefetch: int = DEFAULT_EVAL_PREFETCH):
    """The REAL `StoreEvalProvider` of `role`'s eval manifest -- the exact construction
    `v4int.launch_central.run` uses (lock binding -> the trainer-registry VIEW, since REGISTRY_V2 whole carries
    DATA-06 population-band roles `v4train.registry.load_manifest_registry` refuses -> `StoreEvalProvider`),
    minus everything training-only (no GPU/CUDA, no train provider, no RunConfig). Server-only (needs the real
    DATA_PROTOCOL_LOCK_V2 tree and the Clean V2 store); not exercised by the local test suite."""
    from types import SimpleNamespace

    from v4int.launch_central import G1_GATE, V4 as CENTRAL_V4, bind_data_protocol_lock_v2, build_trainer_registry_view, data_paths
    from v4int.providers import StoreEvalProvider
    from v4train.registry import load_manifest_registry

    args_ns = SimpleNamespace(manifest_dir=None, lock_path=None, manifest_hashes=None, header_list=None,
                             transform_state=None)
    paths = data_paths(args_ns)
    lockctx = bind_data_protocol_lock_v2(paths, smoke=False, gate_path=G1_GATE)
    view = build_trainer_registry_view(lockctx["manifest_registry_path"], Path(run_dir),
                                       registry_rel_path=lockctx["manifest_registry_rel_path"],
                                       registry_file_sha256=lockctx["manifest_registry_file_sha256"])
    trainer_registry = load_manifest_registry(view["path"])
    _, eval_id = trainer_registry.role(role)
    provider = StoreEvalProvider(eval_id, history_policy=history_policy, paths=paths, prefetch=int(eval_prefetch),
                                 manifest_registry=lockctx["manifest_registry_path"], registry_root=CENTRAL_V4)
    provider_meta = {"eval_manifest_id": eval_id, "manifest_dir": str(paths.manifest_dir), "lock_self_sha256": lockctx["lock_self_sha256"],
                     "data_version": lockctx["data_version"]}
    return provider, provider_meta


def main(argv=None) -> int:
    from v4hist.policy import HISTORY_POLICIES                # import-free (DEV-01 pattern: no torch before argparse)

    ap = argparse.ArgumentParser(description="DEV-02: dump a role's eval-manifest E2E rows to a --rows NPZ for v4dev.bench")
    ap.add_argument("--role", required=True, help="a v4train.recipe.ROLES role in the trainer registry view, e.g. MAIN")
    ap.add_argument("--history-policy", required=True, choices=HISTORY_POLICIES)
    ap.add_argument("--out", required=True, help="the rows NPZ path; rows.meta.json is written alongside it")
    ap.add_argument("--first-users", type=int, default=None,
                    help="PREG1_AMENDMENT_06A item 6's fallback: keep only the first N E2E users in "
                         "selection_key_v2 order (module docstring)")
    ap.add_argument("--eval-prefetch", type=int, default=DEFAULT_EVAL_PREFETCH)
    ap.add_argument("--ckpt", default=None, help="DEV-04: bind the dump to this checkpoint's registered feature view")
    ap.add_argument("--family", default=None, choices=["GRU", "SASREC"], help="the --ckpt family (required with it)")
    ap.add_argument("--schema-registry", default=None, dest="schema_registry",
                    help="derived-schema registry override (default: the one the run's RUN_START records)")
    args = ap.parse_args(argv)
    if (args.ckpt is None) != (args.family is None):
        raise SystemExit("--ckpt and --family must be given together (or neither)")

    view = None
    if args.ckpt is not None:                                 # before the dump: refuse a wrong view early
        from v4dev.checkpointio import resolve_feature_view
        view = resolve_feature_view(args.ckpt, args.family, schema_registry=args.schema_registry)
        hp = view["record"].get("history_policy")
        if hp is not None and hp != args.history_policy:
            raise SystemExit(f"--history-policy {args.history_policy} != the checkpoint run's {hp}")

    out_path = Path(args.out)
    provider, provider_meta = build_eval_provider(args.role, args.history_policy, out_path.parent,
                                                  eval_prefetch=args.eval_prefetch)
    if view is not None:
        em = view["record"].get("eval_manifest_id")
        if em is not None and provider_meta["eval_manifest_id"] != em:
            raise SystemExit(f"role {args.role}'s eval manifest {provider_meta['eval_manifest_id']} != the checkpoint "
                             f"run's {em}")
    key_hex = None
    if args.first_users is not None:
        from v4base.identity import eval_client_key_hex_by_eval_user_key
        key_hex = eval_client_key_hex_by_eval_user_key(provider, manifest_dir=provider_meta["manifest_dir"])
    rows, meta = dump_from_provider(provider, first_users=args.first_users, client_key_hex=key_hex)
    meta.update(provider_meta)
    meta["role"] = args.role
    if view is not None:
        rec = view["record"]
        checked = check_rows_view(rows, meta, view["schema"], require_batch_view=True,
                                  run_binding={"eval_manifest_id": rec.get("eval_manifest_id"),
                                               "history_policy": rec.get("history_policy")})
        meta["feature_view_binding"] = view_binding_record(view, args.ckpt, checked)
    result = write_rows(rows, meta, out_path)
    print(json.dumps(result["meta"], indent=1))
    return 0


__all__ = ["DumpRowsError", "FIRST_USERS_RULE", "BATCH_VIEW_SCALARS", "batch_view_of", "check_rows_view", "view_binding_record",
           "numeric_array_field", "first_users_selection", "population_selection_first_users_mask", "dump_from_provider", "write_rows",
           "build_eval_provider", "main"]

if __name__ == "__main__":
    raise SystemExit(main())

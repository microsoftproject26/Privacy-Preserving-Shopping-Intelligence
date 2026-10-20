"""v4base.identity — real per-decision user identity, read from a provider's OWN verified manifest tables, never
from the batch dict: the real Clean V2 batch carries NO `client_ord`/`client_key_hex` column at all (confirmed;
`clean_v2_view` passes every store column through unchanged, and the store's own decision set is built "WITHOUT the
legacy client_idx-based eval_user_key" — `v4data/store.py`). `client_id`/`client_idx` are a DIFFERENT, legacy key
space and are in `v4eval.manifest.FORBIDDEN_COLUMNS`; never read or used here.

Two identities matter, and they are DIFFERENT per-manifest key spaces (contracts/MANIFEST_SPEC_V4.json,
"client_ord_and_eval_user_key": "0-based dense rank of client_key_hex... Named client_ord in TRAIN manifests and
eval_user_key in evaluation manifests... Cross-manifest joins use client_key_hex only."):
  - `client_ord` (a TRAIN manifest's own dense rank, 0..n-1): read straight off that manifest's OWN "decisions"
    table (never the batch) via `provider.verified.canonical_rows`. Used by `v4base.rule`/`v4base.strata` as a
    PURELY WITHIN-ONE-TRAIN-SCAN grouping key for window de-duplication — never persisted, never joined against
    another manifest's key space.
  - `client_key_hex` (the ONLY safe cross-manifest join key): resolved via a TRAIN manifest's own "clients" table
    (client_ord -> client_key_hex, dense) or an evaluation manifest's own "users" table (eval_user_key ->
    client_key_hex, dense). This is what `v4base.strata`'s registry is actually KEYED BY (HIGH review fix: a raw
    `client_ord`/`eval_user_key` integer must never be used as a cross-manifest join key — the same dense integer
    can, and in general does, name a DIFFERENT real user in each manifest).

Every parquet read here is verified against the provider's OWN `.verified.tables[...]["file_sha256"]` (the SAME
record `describe()`/`record()` already exposes) BEFORE a single value is trusted; a mismatch refuses loudly. No
`v4int`/`v4data` file is edited; neither module publishes a bulk client_ord<->client_key_hex reader (searched: only
a single-client, wrong-direction `v4data.store.ClientView` exists), so `pyarrow` is read directly here — already a
dependency of `v4data`/`v4int`, never installed by this task.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _read_parquet_columns(path, expected_sha256: str, columns) -> dict:
    import pyarrow.parquet as pq

    from v4data.common import sha256_file          # read-only: the SAME hash rule every manifest table already uses

    p = Path(path)
    got = sha256_file(p)
    if got != expected_sha256:
        raise ValueError(f"{p}: file_sha256 {got} != the provider's own verified {expected_sha256} — refusing an "
                         "unverified table")
    tbl = pq.ParquetFile(p).read(columns=list(columns))
    return {name: tbl[name].to_numpy(zero_copy_only=False) for name in columns}


def train_client_ord_by_canonical_position(provider, *, manifest_dir) -> np.ndarray:
    """[n_decisions] int64: this TRAIN manifest's OWN `client_ord` (per-manifest dense, NEVER a cross-manifest key)
    for each canonical position `provider.batch([i])` would serve — read from the manifest's "decisions" table via
    `provider.verified.canonical_rows`, verified against the provider's own recorded file_sha256. A purely WITHIN-
    THIS-TRAIN-SCAN grouping key; never write it anywhere, never join it against another manifest's key space."""
    vm = provider.verified
    info = vm.tables["decisions"]
    cols = _read_parquet_columns(Path(manifest_dir) / info["file"], info["file_sha256"], ["client_ord"])
    client_ord_by_decision_row = np.asarray(cols["client_ord"], dtype=np.int64)
    return client_ord_by_decision_row[vm.canonical_rows]


def train_client_key_hex_by_ord(provider, *, manifest_dir) -> np.ndarray:
    """[n_clients] object (str): this TRAIN manifest's "clients" table, `client_key_hex[client_ord]` (dense,
    ascending client_ord 0..n-1 — MANIFEST_SPEC's own dense-rank rule), verified against the provider's own
    recorded file_sha256."""
    vm = provider.verified
    info = vm.tables["clients"]
    cols = _read_parquet_columns(Path(manifest_dir) / info["file"], info["file_sha256"],
                                 ["client_ord", "client_key_hex"])
    ordv = np.asarray(cols["client_ord"], dtype=np.int64)
    if not np.array_equal(ordv, np.arange(ordv.size, dtype=np.int64)):
        raise ValueError("clients table client_ord is not dense ascending 0..n-1 — cannot index directly")
    return np.asarray(cols["client_key_hex"], dtype=object)


def train_client_key_hex_by_canonical_position(provider, *, manifest_dir) -> np.ndarray:
    """[n_decisions] object (str): `client_key_hex` for each canonical TRAIN position — the ONLY safe cross-
    manifest key (module docstring). Composes the two functions above."""
    ord_by_pos = train_client_ord_by_canonical_position(provider, manifest_dir=manifest_dir)
    hex_by_ord = train_client_key_hex_by_ord(provider, manifest_dir=manifest_dir)
    if ord_by_pos.size and (int(ord_by_pos.min()) < 0 or int(ord_by_pos.max()) >= hex_by_ord.size):
        raise ValueError("a decision's client_ord is outside the clients table's range")
    return hex_by_ord[ord_by_pos]


def eval_client_key_hex_by_eval_user_key(eval_provider, *, manifest_dir) -> np.ndarray:
    """[n_eval_users] object (str): an evaluation manifest's "users" table, `client_key_hex[eval_user_key]` (dense,
    ascending eval_user_key 0..n-1), verified against the provider's own recorded file_sha256. Call ONCE per eval
    manifest, then index it by any row's own `eval_user_key` to get the SAME client_key_hex space a TRAIN-built
    strata registry is keyed by (`v4base.strata.assign_stratum_for_user`)."""
    vm = eval_provider.verified
    info = vm.tables["users"]
    cols = _read_parquet_columns(Path(manifest_dir) / info["file"], info["file_sha256"],
                                 ["eval_user_key", "client_key_hex"])
    keyv = np.asarray(cols["eval_user_key"], dtype=np.int64)
    if not np.array_equal(keyv, np.arange(keyv.size, dtype=np.int64)):
        raise ValueError("users table eval_user_key is not dense ascending 0..n-1 — cannot index directly")
    return np.asarray(cols["client_key_hex"], dtype=object)


__all__ = ["train_client_ord_by_canonical_position", "train_client_key_hex_by_ord",
          "train_client_key_hex_by_canonical_position", "eval_client_key_hex_by_eval_user_key"]

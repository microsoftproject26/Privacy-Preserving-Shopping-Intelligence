"""v4int.retro_store (INT-07) - the GATED RETRO_TEST feature-store / history-window path of the single TEST look
(PREG1_AMENDMENT_06D §2; EVALUATION_CONTRACT_V4 E10.1; company/handoffs/DATA-09.md "Not done" items 1-2).

Why: the Clean V2 store refuses TEST by construction (v4data.store.CleanV2Store refuses a TEST split, v4hist.policy
history_splits / store_splits refuse a TEST decision split, v4hist.windows.build_row_index reads through
v4data.manifests.read_non_test). Those locked / frozen paths are NOT edited. This module adds the TEST path beside them,
and every entry point takes a v4data09.d9_gate.GateRecord that is RE-VALIDATED on every call (d9_gate.revalidate: the
ALL_METHODS_LOCK file and the binding MANIFEST_BUILD register entry unchanged). Without an open gate each one refuses
(TestPathRefused, a TestGateError) before any dataset file is opened.

  gated_history_splits(gate, decision_splits)   USER_HISTORY_L50 partitions: TRAIN + VALIDATION_SELECT + VALIDATION_CONFIRM
                                                + TEST in chronological order, up to and including the latest decision
                                                partition (v4hist.policy.history_splits extended by ONE partition; for
                                                non-TEST decision splits it is exactly history_splits, tested)
  gated_store_splits(gate, policy, splits)       SESSION_L50: the decision splits (unchanged v1 rule); USER_HISTORY_L50:
                                                gated_history_splits
  GatedCleanV2Store                              CleanV2Store with the TEST refusal replaced by the gate; batch() /
                                                v1_batch() / positions_for() are the UNMODIFIED parent methods, so every
                                                batch is CompactStore.batch -> clean_v2_view (the frozen normalisers of
                                                the dataset's mappings/normalizers.json and the frozen transform spec,
                                                no refit; E11.5 analogue)
  build_gated_row_index(gate, ds_root, ...)      v4hist.windows.build_row_index with read_with_test (the only TEST scan,
                                                d9_read) and the TEST partition code; the same lexsort, the same
                                                chronology proofs (v4hist.windows._order_checks), the same RowIndex
  gated_path_builders(gate, policy, splits)      the four functions v4int.cache.build_subset_store swaps in on its gated
                                                branch (build_subset_store(..., test_gate=gate)); the default branch
                                                (test_gate=None) is the unchanged VALIDATION / TRAIN code path
  retro_history_record(policy)                   the provider fingerprint's history block for a RETRO_TEST provider
  gate_for_provider(header, gate)               the gate a RETRO_TEST StoreEvalProvider runs under (explicit record,
                                                else the v4int.test_roles install state), bound to the header
  retro_role_key(base_kb, rk, gate, ...)          the TEST comparison-key rule shared by every consumer (reeval /
                                                launch_device / v4base / dump_rows): TEST_KEY_FIELDS replaced from the
                                                gated key (split = "RETRO_TEST"), plus the row identity labels
TEST_DECISION_SPLITS = VALIDATION_SELECT + VALIDATION_CONFIRM + TEST (the task packet's rule: the three decision splits
merged in ascending decision_id, then only the manifest's decisions kept; the gated cache branch additionally refuses a
kept decision that is not a TEST-partition decision).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import v4train  # noqa: F401  (ppsi_env first; the snapshot modules come from code/snap_v1)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from v4data.common import TestGateError, VALIDATION_SPLITS
from v4data.store import CleanV2Store, _runtime_input_schema_sha256
from v4data09.d9_gate import GateRecord, TestGateRefused, check_test_gate, header_matches_gate, revalidate
from v4data09.d9_read import read_with_test
from v4hist.policy import (CONTEXT_LENGTH, RESOLVER_ID, SESSION_L50, USER_HISTORY_L50, HistoryPolicyError,
                           require_policy)
from v4hist.windows import (INDEX_COLUMNS, SESSION_MODE_KEY, SESSION_OFFSET_CAP, SESSION_OFFSET_KEY,
                            SESSION_OFFSET_VOCAB, SPLIT_CODES, HistoryOrderError, RowIndex, _order_checks)

TEST = "TEST"
RETRO_TEST = "RETRO_TEST"
TEST_ORDER = ("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM", TEST)       # chronological, half-open partitions
TEST_DECISION_SPLITS = tuple(VALIDATION_SPLITS) + (TEST,)
TEST_KEY_FIELDS = ("population_hash", "candidate_order_hash", "train_manifest_id", "eval_manifest_id", "split",
                   "manifest_registry_self_sha256")
TEST_RULE = ("USER_HISTORY_L50 over TRAIN + VALIDATION_SELECT + VALIDATION_CONFIRM + TEST (v4hist.policy."
             "history_splits extended by one partition, strictly causal); SESSION_L50 unchanged")
assert tuple(SPLIT_CODES) == TEST_ORDER and [SPLIT_CODES[s] for s in TEST_ORDER] == [0, 1, 2, 3]


class TestPathRefused(TestGateRefused):
    """A TEST store / window / key was requested without an open, unchanged TEST gate (or outside the TEST rule)."""


def require_gate(gate) -> GateRecord:
    """The gate check of every entry point: a GateRecord that still verifies (lock + binding entry unchanged)."""
    if not isinstance(gate, GateRecord):
        raise TestPathRefused(f"the RETRO_TEST store path needs an open TEST gate (v4data09.d9_gate.GateRecord), got "
                              f"{type(gate).__name__}")
    revalidate(gate)
    return gate


# ------------------------------------------------------------------------------------------------ partitions
def gated_history_splits(gate, decision_splits) -> tuple:
    """The partitions a USER_HISTORY window of `decision_splits` may read: every partition of TEST_ORDER up to and
    including the latest decision partition, in chronological order (a causal window never needs a later one)."""
    require_gate(gate)
    ds = tuple(decision_splits)
    if not ds:
        raise HistoryPolicyError("decision_splits must be non-empty")
    for s in ds:
        if s not in TEST_ORDER:
            raise HistoryPolicyError(f"unknown decision split {s!r} (the gated rule knows {TEST_ORDER})")
    return TEST_ORDER[:max(TEST_ORDER.index(s) for s in ds) + 1]


def gated_store_splits(gate, policy, decision_splits) -> tuple:
    p = require_policy(policy)
    hs = gated_history_splits(gate, decision_splits)
    return tuple(decision_splits) if p == SESSION_L50 else hs


# ------------------------------------------------------------------------------------------------ the store
class GatedCleanV2Store(CleanV2Store):
    """CleanV2Store over a snapshot CompactStore that may hold the TEST partition. Only the constructor, build and
    decision_set differ from the locked class (each the same body with refuse_test_split replaced by the gate)."""

    def __init__(self, compact_store, spec, state_sha256: Optional[str] = None, *, gate):
        self.gate = require_gate(gate)
        if getattr(compact_store, "channels", None) != "F4":
            raise ValueError("Clean V2 needs the rich (F4) CompactStore")
        bad = [s for s in compact_store.splits if s not in TEST_ORDER]
        if bad:
            raise TestPathRefused(f"store partitions {bad} are outside {TEST_ORDER}")
        self.cs = compact_store
        self.spec = spec
        self.state_sha256 = state_sha256
        self.runtime_input_schema_sha256 = _runtime_input_schema_sha256()
        if self.runtime_input_schema_sha256 != spec.input_schema_sha256:
            raise ValueError("the vendored loader's v1 column layout differs from the frozen schema's")

    @classmethod
    def build(cls, *, gate, splits: Sequence[str], buckets: Optional[Sequence[int]] = None,
              ds_root: Optional[Path] = None, state_sha256: Optional[str] = None, n_threads: int = 3,
              log=print) -> "GatedCleanV2Store":
        require_gate(gate)
        bad = [s for s in splits if s not in TEST_ORDER]
        if bad or not splits:
            raise TestPathRefused(f"store partitions {list(splits)} must be a non-empty subset of {TEST_ORDER}")
        from full_store import CompactStore  # snapshot copy, via ppsi_env
        from v4data.common import dataset_root
        from v4data.transform import build_spec
        cs = CompactStore.build(Path(ds_root) if ds_root else dataset_root(), splits=tuple(splits), channels="F4",
                                buckets=list(buckets) if buckets is not None else None, n_threads=n_threads, log=log)
        return cls(cs, build_spec(), state_sha256, gate=gate)

    def decision_set(self, split: str, kind: str):
        """v4data.store.CleanV2Store.decision_set (same body) for any partition of this store, TEST under the gate."""
        require_gate(self.gate)
        if split not in self.cs.splits:
            raise ValueError(f"split {split!r} not held by this store {tuple(self.cs.splits)}")
        from full_store import DecisionSet
        arr = self.cs._split_decisions[split]
        mask = arr["loss_mask"] if kind == "loss" else arr["eval_mask"]
        idx = np.flatnonzero(mask)
        order = np.argsort(arr["decision_id"][idx], kind="stable")
        sel = idx[order]
        did = arr["decision_id"][sel].astype(np.int64)
        if did.size > 1 and not np.all(np.diff(did) > 0):
            raise AssertionError("decision_id not strictly ascending")
        return DecisionSet(split=split, kind=kind, decision_id=did, end_row=arr["global_row"][sel].astype(np.int64),
                           length=arr["length"][sel], target_class=arr["target_class"][sel].astype(np.int64),
                           loss_mask=arr["loss_mask"][sel].astype(bool), eval_mask=arr["eval_mask"][sel].astype(bool),
                           client_idx=arr["client_idx"][sel].astype(np.int64),
                           user_bucket=arr["user_bucket"][sel].astype(np.int16), eval_user_key=None,
                           prior_view_count=arr["prior_view_count"][sel], prefix_position=arr["prefix_position"][sel])


# ------------------------------------------------------------------------------------------------ the row index
def _split_codes_test(col) -> np.ndarray:
    idx = pc.index_in(col, value_set=pa.array(list(TEST_ORDER), type=pa.string()))
    if idx.null_count:
        raise HistoryOrderError(f"a row outside {TEST_ORDER} reached the gated index")
    return np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int8)


def _bucket_index_test(ds_root: Path, b: int, splits: tuple, gate) -> dict:
    """v4hist.windows._bucket_index with the gated TEST scan (read_with_test re-validates the gate on every call)."""
    from v4data.manifests import _ts_us
    path = Path(ds_root) / "model_ready" / f"user_bucket={int(b)}" / "features.parquet"
    t = read_with_test(path, list(INDEX_COLUMNS), gate)
    t = t.filter(pc.is_in(t["split"], value_set=pa.array(list(splits), type=pa.string())))
    n = t.num_rows
    if n == 0:
        z = lambda dt: np.zeros(0, dtype=dt)  # noqa: E731
        return {"n": 0, "users": 0, "decision_id": z(np.int64), "user_ord": z(np.int64), "user_rank": z(np.int32),
                "position": z(np.int32), "ts_us": z(np.int64), "split_code": z(np.int8), "session_id": z(np.int32),
                "checks": {"rows": 0}}
    if t["user_id"].null_count:
        raise HistoryOrderError(f"bucket {b}: NULL user_id")
    uid = t["user_id"].to_numpy(zero_copy_only=False)
    seg = t["segment_index"].to_numpy(zero_copy_only=False)
    pos = t["position_in_segment"].to_numpy(zero_copy_only=False)
    if np.any(np.isnan(seg)):
        raise HistoryOrderError(f"bucket {b}: NaN segment_index")
    o = np.lexsort((pos, seg, uid))                  # the store's key (full_store.py L237; loaders.py L96)
    uid, seg, pos = uid[o], seg[o], pos[o]
    newu = np.ones(n, dtype=bool)
    newu[1:] = uid[1:] != uid[:-1]
    newseg = newu.copy()
    newseg[1:] |= seg[1:] != seg[:-1]
    del uid
    ts = _ts_us(t["block_ts"])[o]
    sc = _split_codes_test(t["split"])[o]
    did = np.asarray(t["decision_id"].to_numpy(zero_copy_only=False), dtype=np.int64)[o]
    ses = np.asarray(t["session_index"].to_numpy(zero_copy_only=False), dtype=np.float64)[o]
    del t
    if np.any(~np.isfinite(ses)) or np.any(ses != np.round(ses)) or np.any(ses < 0) or np.any(ses >= 2 ** 31):
        raise HistoryOrderError(f"bucket {b}: session_index is not a non-negative integer")
    sid = ses.astype(np.int32)
    pos32 = np.asarray(pos, dtype=np.int32)
    checks = _order_checks(newu, newseg, pos32, ts, sc, sid)
    starts = np.flatnonzero(newu)
    uof = np.cumsum(newu) - 1
    rank = (np.arange(n) - starts[uof]).astype(np.int32)
    return {"n": int(n), "users": int(starts.size), "decision_id": did, "user_ord": uof.astype(np.int64),
            "user_rank": rank, "position": pos32, "ts_us": ts, "split_code": sc, "session_id": sid, "checks": checks}


def build_gated_row_index(gate, ds_root, buckets: Sequence[int], splits: Sequence[str]) -> RowIndex:
    """The RowIndex of the rows a GatedCleanV2Store built with (`splits`, `buckets`) retains, in its row order
    (v4hist.windows.build_row_index, same aggregation; the TEST partition allowed under the gate)."""
    require_gate(gate)
    sp = tuple(splits)
    for s in sp:
        if s not in TEST_ORDER:
            raise HistoryPolicyError(f"unknown split {s!r}")
    bl = sorted(int(b) for b in buckets)                # CompactStore.build sorts its buckets
    parts, rows, off_u, checks = [], [], 0, {}
    for b in bl:
        r = _bucket_index_test(Path(ds_root), b, sp, gate)
        r["user_ord"] = r["user_ord"] + off_u
        off_u += r["users"]
        parts.append(r)
        rows.append((b, r["n"]))
        checks[str(b)] = r["checks"]
    cat = lambda k, dt: (np.concatenate([p[k] for p in parts]).astype(dt, copy=False) if parts  # noqa: E731
                         else np.zeros(0, dtype=dt))
    if off_u >= 2 ** 31:
        raise HistoryOrderError("more users than an int32 ordinal holds")
    agg = {k: int(sum(c.get(k, 0) for c in checks.values()))
           for k in ("sessions", "session_starts_inside_segment", "session_decreasing", "fence_rows",
                     "fence_rows_same_session", "session_gaps_gt1")}
    agg.update({"rows": int(sum(c.get("rows", 0) for c in checks.values())), "users": int(off_u),
                "segments": int(sum(c.get("segments", 0) for c in checks.values())),
                "user_row_pairs_checked": int(sum(c.get("user_row_pairs_checked", 0) for c in checks.values())),
                "ts_decreasing": int(sum(c.get("ts_decreasing", 0) for c in checks.values())),
                "partition_decreasing": int(sum(c.get("partition_decreasing", 0) for c in checks.values())),
                "segment_spans_partitions": int(sum(c.get("segment_spans_partitions", 0) for c in checks.values()))})
    return RowIndex(splits=sp, buckets=tuple(bl), bucket_rows=tuple(rows), decision_id=cat("decision_id", np.int64),
                    user_ord=cat("user_ord", np.int32), user_rank=cat("user_rank", np.int32),
                    position=cat("position", np.int32), ts_us=cat("ts_us", np.int64),
                    split_code=cat("split_code", np.int8), n_users=int(off_u), order_checks=agg,
                    session_id=cat("session_id", np.int32))


# ------------------------------------------------------------------------------------------------ cache branch
def gated_path_builders(gate, policy, decision_splits) -> tuple:
    """(store partitions, store builder, index builder, store wrapper) for v4int.cache.build_subset_store's gated
    branch. Only the TEST eval rule is served: decision splits == TEST_DECISION_SPLITS."""
    require_gate(gate)
    if tuple(decision_splits) != TEST_DECISION_SPLITS:
        raise TestPathRefused(f"the gated subset store serves decision splits {TEST_DECISION_SPLITS} only, not "
                              f"{tuple(decision_splits)}")
    ss = gated_store_splits(gate, policy, decision_splits)

    def build_store(*, splits, buckets, ds_root, state_sha256, n_threads, log):
        return GatedCleanV2Store.build(gate=gate, splits=splits, buckets=buckets, ds_root=ds_root,
                                       state_sha256=state_sha256, n_threads=n_threads, log=log)

    def build_index(ds_root, buckets, splits):
        return build_gated_row_index(gate, ds_root, buckets, splits)

    def wrap(cs, spec, state_sha256):
        return GatedCleanV2Store(cs, spec, state_sha256, gate=gate)

    return ss, build_store, build_index, wrap


def retro_decision_mask(gate, store, decision_ids) -> np.ndarray:
    """Which of `decision_ids` are TEST-partition decisions of `store` (the gated branch refuses any other)."""
    require_gate(gate)
    arr = store.cs._split_decisions.get(TEST)
    if arr is None:
        return np.zeros(np.asarray(decision_ids).shape, dtype=bool)
    return np.isin(np.asarray(decision_ids, dtype=np.int64), np.asarray(arr["decision_id"], dtype=np.int64))


# ------------------------------------------------------------------------------------------------ provider helpers
def is_retro_test_header(header: Mapping) -> bool:
    return str((header or {}).get("v4_split", "")).upper() == RETRO_TEST


def shim_gate() -> GateRecord:
    """The gate of the in-process v4int.test_roles install (its lock / register paths; the defaults otherwise)."""
    from v4int import test_roles as TR
    st = getattr(TR, "_STATE", {}) or {}
    return check_test_gate(st.get("lock"), st.get("register"))


def gate_for_provider(header: Mapping, gate: Optional[GateRecord] = None) -> GateRecord:
    """The gate a RETRO_TEST StoreEvalProvider runs under: `gate` (re-validated) or the test_roles install's; the
    manifest header must have been built under it (all_methods_lock_sha256 / register entry id)."""
    try:
        g = require_gate(gate) if gate is not None else shim_gate()
    except TestGateError as e:
        raise TestPathRefused(f"RETRO_TEST provider refused: {e}") from e
    if not header_matches_gate(header, g):
        raise TestPathRefused(f"{header.get('manifest_id')}: the header was built under another TEST gate")
    return g


def retro_history_record(gate, policy) -> dict:
    """v4hist.windows.history_record for a RETRO_TEST evaluation provider (decision splits TEST_DECISION_SPLITS)."""
    p = require_policy(policy)
    return {"history_policy": p, "context_length": CONTEXT_LENGTH, "resolver": RESOLVER_ID,
            "decision_splits": list(TEST_DECISION_SPLITS),
            "store_splits": list(gated_store_splits(gate, p, TEST_DECISION_SPLITS)),
            "window_rule": ("rows [max(user_start, r - (L - 1)), r] of the user's retained stream in (user_id, "
                            "segment_index, position_in_segment) order" if p == USER_HISTORY_L50 else
                            "rows [r - min(L, position_in_segment) + 1, r] (v1 loaders.py L263-L270)"),
            "per_event_channels": "stored per-event values (session-relative channels restart at each segment start)",
            "session_offset_tokens": ({"key": SESSION_OFFSET_KEY, "dtype": "int64", "cap": SESSION_OFFSET_CAP,
                                       "vocab": SESSION_OFFSET_VOCAB} if p == USER_HISTORY_L50 else None),
            "session_mode_features": ({"key": SESSION_MODE_KEY} if p == USER_HISTORY_L50 else None),
            "test_rule": TEST_RULE, "resolver_module": "v4int.retro_store (INT-07, gated)"}


def retro_role_key(base_kb: Mapping, rk: Mapping, gate, *, role: str, eval_manifest_content_sha256: str) -> tuple:
    """(key base, row overrides) of a TEST role for every consumer: the fields TEST_KEY_FIELDS come from `rk` (the
    gated comparison_key_base of the role's support / RETRO_TEST pair, split = RETRO_TEST); the rows are labelled
    manifest_role = the TEST role, its manifest ids, the gate entry id and the eval manifest content sha."""
    g = require_gate(gate)
    if rk.get("split") != RETRO_TEST:
        raise TestPathRefused(f"a TEST role key needs split {RETRO_TEST!r}, got {rk.get('split')!r}")
    missing = [f for f in TEST_KEY_FIELDS if f not in rk]
    if missing:
        raise TestPathRefused(f"the TEST role key lacks {missing}")
    kb = dict(base_kb, **{f: rk[f] for f in TEST_KEY_FIELDS})
    over = {"manifest_role": role, "train_manifest_id": rk["train_manifest_id"],
            "eval_manifest_id": rk["eval_manifest_id"], "v4_split": RETRO_TEST,
            "test_exposure_register_entry_id": g.entry_id, "all_methods_lock_sha256": g.lock_sha256,
            "eval_manifest_content_sha256": eval_manifest_content_sha256,
            "test_registry_self_sha256": rk["manifest_registry_self_sha256"]}
    return kb, over


def retro_manifest_dir(ext_path: Path, eval_manifest_id: str, root: Optional[Path] = None) -> Path:
    """The directory of a RETRO_TEST manifest's files (its registry entry's header path; data_test/manifests)."""
    import json
    ext = json.loads(Path(ext_path).read_text(encoding="utf-8"))
    e = (ext.get("manifests") or {}).get(eval_manifest_id) or {}
    if not e.get("path"):
        raise TestPathRefused(f"{eval_manifest_id} has no header path in {ext_path}")
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return (base / str(e["path"])).parent


def train_key_ref(manifest_id: str, *, history_policy: str, paths=None, manifest_registry=None,
                  registry_root=None, log: Callable = lambda *_: None):
    """The train side of comparison_key_base WITHOUT building a store (an eval-only consumer needs the support
    manifest's verified identity only): registry verification + the locked transform state, exactly as
    StoreTrainProvider records them."""
    from types import SimpleNamespace
    from v4data.transform import build_spec
    from .providers import _verify, manifest_binding_of
    from .renorm import resolve_transform_state
    pol = require_policy(history_policy)
    vm = _verify(manifest_id, paths, "TRAIN", manifest_registry, registry_root, log)
    _renorm, ts = resolve_transform_state(None, build_spec(), vm.transform_state_self_sha256)
    vm.columns = None
    return SimpleNamespace(manifest_id=manifest_id, history_policy=pol, transform_state=ts, verified=vm,
                           manifest_binding=manifest_binding_of(vm), fingerprint=None, key_only=True)


__all__ = ["TEST_ORDER", "TEST_DECISION_SPLITS", "TEST_KEY_FIELDS", "RETRO_TEST", "TestPathRefused", "require_gate",
           "gated_history_splits", "gated_store_splits", "GatedCleanV2Store", "build_gated_row_index",
           "gated_path_builders", "retro_decision_mask", "is_retro_test_header", "shim_gate", "gate_for_provider",
           "retro_history_record", "retro_role_key", "retro_manifest_dir", "train_key_ref"]

"""v4int.providers — the real-data providers of the central trainer (INT-01).

Population-agnostic (OWNER_AMENDMENT_POP_5_100): a provider serves ANY manifest registered in DATA_PROTOCOL_LOCK whose
locked table layout has the right role (v4int.verify.manifest_kind); no manifest id and no count is hard-coded here.
(The trainer itself pairs phases with ids through v4train.provider.expected_manifests; see the INT-01 handoff.)

StoreTrainProvider (a TRAIN-layout manifest; today M01_COHORT_TRAIN / M05_CALIB_TRAIN, HISTORICAL after the owner
amendment) implements v4train.provider.BatchProvider exactly:
  manifest_id / split "TRAIN" / fingerprint; n_decisions (the header's and the lock's count, verified against the
  table); decision_ids() = the canonical strictly ascending order (MANIFEST_SPEC central_canonical_order, through the
  manifest's central_order table);
  batch(pos) = CPU tensors in the Clean V2 layout for exactly those canonical positions, in order, produced by the
  unmodified CleanV2Store.batch over the manifest-scoped cache (v4int.cache). Every row is loss-eligible and its
  decision_id / target_class equal the manifest's (asserted on every batch; ProviderIntegrityError otherwise).
  batch() is a pure function of `pos` (no state, no RNG), so a resumed run is served the identical bytes.
StoreEvalProvider (an evaluation-layout manifest; today M02_VALIDATION_WARM / M06_CALIB_VALIDATION_WARM) implements
v4train.provider.EvalProvider:
  manifest = v4eval.manifest.build_eval_manifest(header, columns) (the TEST gate and the ordered-hash checks run);
  batches(bs) yields the manifest's evaluation stream in manifest order, exactly `bs` rows per batch except an
  unpadded tail; `bs` must equal the frozen config's eval_batch. The stream is the frozen config's `eval_stream` when
  v4eval.frozen_config.load() carries one; the loader (EVAL-03, locked) has no such attribute yet, so the stream is
  read from contracts/PREG1_AMENDMENT_01.md item 5 (eval_stream = "EVAL_ROWS_ONLY", PROPOSED, Fable to confirm at
  G1; recorded with the file's sha256 in the fingerprint) and mapped to v4eval's registered name E2E_ROWS (the
  eval_mask rows, OOV included, censored rows excluded). ALL_ROWS is refused: the Clean V2 store does not materialise
  censored decisions. The data path is the T3 GPU receipt's (tests_eval/gpu_t3_receipt.py: VALIDATION splits merged
  in ascending decision_id, the eval_mask stream, batches of eval_batch through CleanV2Store.batch); the target
  classes are checked against the manifest at build time and on every batch. An optional ordered prefetch thread
  (depth 2) builds the next batches while the GPU ranks the current one; it calls the same function in the same order,
  so the bytes and the order are unchanged.
Fingerprints bind the manifest's ordered decision-id sha256 (table order and canonical order), the target classes,
the manifest file hashes (tables + header), the header content hash, the TRANSFORM_STATE self-hash and the
DATA_PROTOCOL_LOCK self-hash (plus, for evaluation, the three ordered hashes and the stream and its source).

History policy (DATA-05; DECISION_LOG 35-36): both providers take a REQUIRED keyword `history_policy` (no default;
v4hist.policy: USER_HISTORY_L50 locked, SESSION_L50 the v1 diagnostic), checked before any file is opened. It selects
the store partitions and the window lengths of the cache (v4int.cache), and it is bound into the fingerprint (with the
resolver record v4hist.windows.history_record), `describe()`, the build record and the comparison key:
comparison_key_base refuses a train / eval pair with different policies, records `history_policy`, and binds the
policy into `feature_contract_hash` (v4hist.windows.input_contract_sha256), because the locked v4eval.rows
COMPARISON_KEY has no separate history field and would otherwise drop it from result rows. Targets, masks, the TEST
gate and every other check are unchanged.

Transform state (DECISION_LOG 42; v4int.renorm): both providers take an optional keyword `transform_state` (a path to a
TRANSFORM_STATE JSON with a verified self-hash). None = the locked Clean V2 state: batches are untouched (bitwise the
previous behaviour). Given: the 17 z-normalized channels are re-normalized with the state's normalizers (NORMALIZERS
ONLY; item-side tables, flags, masks, rezero map unchanged). The applied state's self-hash is bound into the
fingerprint, describe(), the build record and the comparison key (a train / eval pair with different states is
refused; a non-default state is also bound into feature_contract_hash).
"""
from __future__ import annotations

import hashlib
import queue
import re
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional

import v4train  # noqa: F401  (ppsi_env first)

import numpy as np

from v4data.common import VALIDATION_SPLITS, canonical_json_bytes, ordered_ids_sha256, sha256_bytes, sha256_file
from v4data.transform import build_spec
from v4eval import frozen_config as fc
from v4eval.manifest import build_eval_manifest
from v4eval.predict import stream_rows
from v4hist.policy import require_policy
from v4hist.windows import add_session_features, history_record, input_contract_sha256

from .cache import build_subset_store, peak_rss_bytes, rss_bytes
from .renorm import resolve_transform_state
from .verify import (DataPaths, V4, manifest_content_sha256, refuse_test_manifest, verify_lock, verify_manifest,
                     verify_manifest_by_registry)

# the columns the T3 GPU receipt built its EvalManifest from (tests_eval/gpu_t3_receipt.py M02_COLS)
EVAL_MANIFEST_COLUMNS = ("decision_id", "eval_user_key", "eval_mask", "target_class", "target_oov", "censor_class",
                         "subperiod", "old_split", "v4_split", "oov_cause", "rankable")
STREAM_NAMES = {"EVAL_ROWS_ONLY": "E2E_ROWS", "E2E_ROWS": "E2E_ROWS", "ALL_ROWS": "ALL_ROWS"}
STORE_SERVABLE_STREAMS = ("E2E_ROWS",)
AMENDMENT_PATH = V4 / "contracts" / "PREG1_AMENDMENT_01.md"
_AMENDMENT_RE = re.compile(r'`eval_stream\s*=\s*"([A-Z_]+)"`')


class ProviderError(RuntimeError):
    pass


class ProviderIntegrityError(AssertionError):
    """A batch disagrees with the manifest (decision_id, target_class or loss-eligibility): refused."""


def _fp(obj) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def _i64_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<i8").tobytes()).hexdigest()


def resolve_eval_stream(frozen, amendment_path: Optional[Path] = None) -> tuple:
    """(v4eval stream name, source record). The frozen config's `eval_stream` if the loader carries one, else the
    registered PROPOSED value of PREG1_AMENDMENT_01 item 5 read from the contract file (never a code default)."""
    v = getattr(frozen, "eval_stream", None)
    if v is not None:
        if v not in STREAM_NAMES:
            raise ProviderError(f"frozen eval_stream {v!r} is not a registered stream {sorted(STREAM_NAMES)}")
        return STREAM_NAMES[v], {"source": "v4eval.frozen_config.load().eval_stream", "value": v,
                                 "lock_sha256": frozen.lock_sha256}
    p = Path(amendment_path) if amendment_path is not None else AMENDMENT_PATH
    if not p.is_file():
        raise ProviderError(f"the frozen config has no eval_stream and {p} is missing: the stream is unregistered")
    found = sorted(set(_AMENDMENT_RE.findall(p.read_text(encoding="utf-8"))))
    if len(found) != 1 or found[0] not in STREAM_NAMES:
        raise ProviderError(f"{p}: expected exactly one registered eval_stream value, found {found}")
    return STREAM_NAMES[found[0]], {"source": "contracts/PREG1_AMENDMENT_01.md item 5 (PROPOSED; Fable to confirm at "
                                              "G1); v4eval.frozen_config.load() carries no eval_stream",
                                    "value": found[0], "file_sha256": sha256_file(p)}


def ordered_prefetch(fn: Callable, items: Iterable, depth: int) -> Iterator:
    """fn(item) for item in items, in order, computed up to `depth` items ahead in ONE background thread."""
    q: "queue.Queue" = queue.Queue(maxsize=max(1, int(depth)))
    stop = threading.Event()
    done = object()

    def put(x) -> bool:
        while not stop.is_set():
            try:
                q.put(x, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def work():
        try:
            for it in items:
                if stop.is_set() or not put((True, fn(it))):
                    return
            put((True, done))
        except BaseException as e:                  # re-raised in the consumer
            put((False, e))

    th = threading.Thread(target=work, name="v4int-eval-prefetch", daemon=True)
    th.start()
    try:
        while True:
            ok, v = q.get()
            if not ok:
                raise v
            if v is done:
                return
            yield v
    finally:
        stop.set()
        while th.is_alive():
            try:
                q.get(timeout=0.05)
            except queue.Empty:
                pass
        th.join()


MANIFEST_CONTENT_RULE = ("sha256 of the canonical JSON list [{table, row_count, table_content_sha256}] in the header's "
                         "table order (REGISTRY_FS rules.content_sha256 = v4fs_data.fs_common.manifest_content_sha256), "
                         "from the verified tables")


def _verify(manifest_id, paths, kind, manifest_registry, registry_root, log):
    """The lock path (M01-M06) unless a manifest registry is given (FS / v2 manifests): v4int.verify."""
    if manifest_registry is None:
        return verify_manifest(manifest_id, paths, expect_kind=kind, log=log)
    return verify_manifest_by_registry(manifest_id, manifest_registry, paths, root=registry_root, expect_kind=kind,
                                       log=log)


def _registry_fp(vm) -> dict:
    return {} if not getattr(vm, "registry", None) else {
        "manifest_registry_self_sha256": vm.registry["self_sha256"], "manifest_registry_data_version":
        vm.registry["data_version"]}


def add_session_offsets(batch: dict, store, ds, pos) -> dict:
    """DECISION_LOG 47 + 48: USER_HISTORY_L50 batches carry `session_offset_tokens` and the session-mode candidate
    features (v4hist.windows.add_session_features), computed from the cache's per-row behavioural session ids;
    SESSION_L50 batches keep the exact v1 key set."""
    sid = getattr(store, "session_id", None)
    if sid is None:
        return batch
    return add_session_features(batch, sid, ds.end_row[pos], ds.length[pos])


def manifest_binding_of(vm) -> dict:
    """TRAIN-01R2 `manifest_binding` of a VERIFIED manifest (v4int.verify.VerifiedManifest): content_sha256 by the
    REGISTRY_FS rule over the table identities verify_manifest re-read and checked (row counts and
    table_content_sha256), header_sha256 = the sha256 of the header file bytes it verified. The header's own
    header_content_sha256 is carried too (the HISTORICAL registry's older content rule)."""
    try:
        content = manifest_content_sha256(vm.header.get("tables", []), vm.tables)
    except Exception as e:
        raise ProviderError(f"{vm.manifest_id}: {e}") from e
    return {"manifest_id": vm.manifest_id, "content_sha256": content, "header_sha256": vm.header_file_sha256,
            "header_content_sha256": vm.header.get("header_content_sha256"), "content_sha256_rule": MANIFEST_CONTENT_RULE}


class StoreTrainProvider:
    """v4train.provider.BatchProvider over a registered TRAIN-layout manifest (real Clean V2 store)."""

    split = "TRAIN"
    PROVIDER_ID = "v4int.StoreTrainProvider/2"

    def __init__(self, manifest_id: str, *, history_policy: str, paths: Optional[DataPaths] = None,
                 ds_root: Optional[Path] = None, buckets=None, group_size: int = 8, n_threads: int = 3,
                 transform_state: Optional[Path] = None, manifest_registry: Optional[Path] = None,
                 registry_root: Optional[Path] = None, log: Callable = print):
        self.history_policy = require_policy(history_policy)          # before any file is opened
        refuse_test_manifest(manifest_id)
        t0 = time.time()
        rss0 = rss_bytes()
        vm = _verify(manifest_id, paths, "TRAIN", manifest_registry, registry_root, log)
        self.manifest_binding = manifest_binding_of(vm)                # TRAIN-01R2 registry binding
        did = vm.canonical_decision_id
        tgt = np.ascontiguousarray(np.asarray(vm.columns["target_class"], dtype=np.int64)[vm.canonical_rows])
        self.renorm, self.transform_state = resolve_transform_state(transform_state, build_spec(),
                                                                    vm.transform_state_self_sha256)
        t1 = time.time()
        store, ds, info = build_subset_store(did, splits=("TRAIN",), kind="loss",
                                             state_sha256=vm.transform_state_self_sha256,
                                             history_policy=self.history_policy, ds_root=ds_root,
                                             buckets=buckets, group_size=group_size, n_threads=n_threads, log=log)
        if not np.array_equal(ds.target_class, tgt):
            raise ProviderIntegrityError(f"{manifest_id}: store target_class differs from the manifest "
                                         f"({int(np.count_nonzero(ds.target_class != tgt))} rows)")
        if not bool(np.all(ds.loss_mask)) or not bool(np.all(ds.target_class >= 0)):
            raise ProviderIntegrityError(f"{manifest_id}: a manifest decision is not loss-eligible in the store")
        vm.columns = None                                    # the decisions table is no longer needed
        self.manifest_id = manifest_id
        self.verified = vm
        self.store, self.ds, self.cache_info = store, ds, info
        self._did = did
        self._target = tgt
        self._N = int(did.size)
        self.history = history_record(self.history_policy, ("TRAIN",))
        self.fingerprint = _fp({
            "provider": self.PROVIDER_ID, "manifest_id": manifest_id,
            "canonical_ordered_decision_id_sha256": ordered_ids_sha256(did),
            "manifest_ordered_decision_id_sha256": vm.header["ordered_decision_id_sha256"],
            "canonical_target_class_sha256": _i64_sha(tgt), "manifest_file_sha256": vm.file_hashes(),
            "header_content_sha256": vm.header["header_content_sha256"],
            "transform_state_self_sha256": vm.transform_state_self_sha256,
            "data_protocol_lock_sha256": vm.lock_self_sha256,
            "history_policy": self.history_policy, "history": self.history,
            "applied_transform_state_sha256": self.transform_state["transform_state_sha256"], **_registry_fp(vm)})
        self.build_record = {"verify_s": round(t1 - t0, 2), "cache_build_s": info.build_s,
                             "total_s": round(time.time() - t0, 2), "rss_bytes_before": rss0,
                             "rss_bytes_after": rss_bytes(), "peak_rss_bytes_after": peak_rss_bytes(),
                             "cache_nbytes": info.cache_nbytes, "cache_event_rows": info.n_event_rows,
                             "history_policy": self.history_policy, "mean_window_length": info.mean_window_length,
                             "transform_state": self.transform_state}

    @property
    def n_decisions(self) -> int:
        return self._N

    def decision_ids(self) -> np.ndarray:
        return self._did.copy()

    def batch(self, positions) -> dict:
        pos = np.asarray(positions)
        if pos.ndim != 1 or pos.size == 0 or pos.dtype.kind not in "iu":
            raise ProviderError("positions must be a non-empty 1-D integer array")
        pos = pos.astype(np.int64, copy=False)
        if int(pos.min()) < 0 or int(pos.max()) >= self._N:
            raise ProviderError(f"positions outside [0, {self._N})")
        b = add_session_offsets(self.store.batch(self.ds, pos), self.store, self.ds, pos)
        if self.renorm is not None:
            b = self.renorm.apply(b)
        if not (np.array_equal(b["decision_id"].numpy(), self._did[pos])
                and np.array_equal(b["target_class"].numpy(), self._target[pos])
                and bool(b["loss_mask"].all())):
            raise ProviderIntegrityError(f"{self.manifest_id}: a batch disagrees with the manifest (decision_id, "
                                         "target_class or loss_mask)")
        return b

    def describe(self) -> dict:
        return {"provider": self.PROVIDER_ID, "manifest_id": self.manifest_id, "split": self.split,
                "n_decisions": self._N, "fingerprint": self.fingerprint, "history_policy": self.history_policy,
                "history": self.history, "transform_state": self.transform_state,
                "manifest_binding": self.manifest_binding, "verified": self.verified.record(),
                "cache": self.cache_info.as_json(), "build": self.build_record}


class StoreEvalProvider:
    """v4train.provider.EvalProvider over a registered evaluation-layout manifest (real Clean V2 store)."""

    PROVIDER_ID = "v4int.StoreEvalProvider/2"

    def __init__(self, manifest_id: str, *, history_policy: str, paths: Optional[DataPaths] = None, frozen=None,
                 ds_root: Optional[Path] = None, buckets=None, group_size: int = 8, n_threads: int = 3,
                 prefetch: int = 2, amendment_path: Optional[Path] = None, transform_state: Optional[Path] = None,
                 manifest_registry: Optional[Path] = None, registry_root: Optional[Path] = None,
                 log: Callable = print, test_gate=None):
        self.history_policy = require_policy(history_policy)          # before any file is opened
        refuse_test_manifest(manifest_id)       # a TEST id passes only through the v4int.test_roles gate (W4)
        t0 = time.time()
        rss0 = rss_bytes()
        self.frozen = fc.require_frozen(frozen if frozen is not None else fc.load())
        self.eval_batch = int(self.frozen.eval_batch)
        self.eval_stream, self.eval_stream_source = resolve_eval_stream(self.frozen, amendment_path)
        if self.eval_stream not in STORE_SERVABLE_STREAMS:
            raise ProviderError(f"eval_stream {self.eval_stream}: the Clean V2 store materialises eval_mask decisions "
                                f"only, so only {STORE_SERVABLE_STREAMS} can be served")
        vm = _verify(manifest_id, paths, "EVAL", manifest_registry, registry_root, log)
        self.test_gate = None                                          # INT-07: the RETRO_TEST branch (gated)
        if str(vm.header.get("v4_split", "")).upper() == "RETRO_TEST":
            from . import retro_store as RS
            self.test_gate = RS.gate_for_provider(vm.header, test_gate)
        elif test_gate is not None:
            raise ProviderError(f"{manifest_id}: test_gate is for a RETRO_TEST manifest only")
        self.manifest_binding = manifest_binding_of(vm)                # TRAIN-01R2 registry binding
        cols = {c: vm.columns[c] for c in EVAL_MANIFEST_COLUMNS if c in vm.columns}
        self.manifest = build_eval_manifest(vm.header, cols)       # TEST gate + ordered hashes vs the header
        stream = stream_rows(self.manifest)
        ids = np.ascontiguousarray(self.manifest.decision_id[stream], dtype=np.int64)
        tgt = np.ascontiguousarray(self.manifest.target_class[stream], dtype=np.int64)
        self.renorm, self.transform_state = resolve_transform_state(transform_state, build_spec(),
                                                                    vm.transform_state_self_sha256)
        t1 = time.time()
        if self.test_gate is None:
            store, ds, info = build_subset_store(ids, splits=VALIDATION_SPLITS, kind="eval",
                                                 state_sha256=vm.transform_state_self_sha256,
                                                 history_policy=self.history_policy, ds_root=ds_root,
                                                 buckets=buckets, group_size=group_size, n_threads=n_threads, log=log)
        else:                                   # INT-07: TRAIN+VS+VC+TEST windows, gate re-validated per group
            store, ds, info = build_subset_store(ids, splits=RS.TEST_DECISION_SPLITS, kind="eval",
                                                 state_sha256=vm.transform_state_self_sha256,
                                                 history_policy=self.history_policy, ds_root=ds_root,
                                                 buckets=buckets, group_size=group_size, n_threads=n_threads, log=log,
                                                 test_gate=self.test_gate)
        if not np.array_equal(ds.target_class, tgt):             # the T3 receipt's check
            raise ProviderIntegrityError(f"{manifest_id}: store target_class differs from the manifest on the "
                                         f"evaluation stream ({int(np.count_nonzero(ds.target_class != tgt))} rows)")
        vm.columns = None
        self.manifest_id = manifest_id
        self.verified = vm
        self.store, self.ds, self.cache_info = store, ds, info
        self._ids, self._target = ids, tgt
        self.prefetch = int(prefetch)
        self.history = history_record(self.history_policy, VALIDATION_SPLITS) if self.test_gate is None else \
            RS.retro_history_record(self.test_gate, self.history_policy)
        test_fp = {} if self.test_gate is None else {"v4_split": "RETRO_TEST", "test_gate": self.test_gate.binding()}
        self.fingerprint = _fp({
            "provider": self.PROVIDER_ID, "manifest_id": manifest_id, "manifest_hashes": self.manifest.hashes,
            "stream_ordered_decision_id_sha256": ordered_ids_sha256(ids), "stream_target_class_sha256": _i64_sha(tgt),
            "eval_stream": self.eval_stream, "eval_stream_source": self.eval_stream_source,
            "manifest_file_sha256": vm.file_hashes(), "header_content_sha256": vm.header["header_content_sha256"],
            "transform_state_self_sha256": vm.transform_state_self_sha256,
            "data_protocol_lock_sha256": vm.lock_self_sha256,
            "history_policy": self.history_policy, "history": self.history,
            "applied_transform_state_sha256": self.transform_state["transform_state_sha256"], **_registry_fp(vm),
            **test_fp})
        self.build_record = {"verify_s": round(t1 - t0, 2), "cache_build_s": info.build_s,
                             "total_s": round(time.time() - t0, 2), "rss_bytes_before": rss0,
                             "rss_bytes_after": rss_bytes(), "peak_rss_bytes_after": peak_rss_bytes(),
                             "cache_nbytes": info.cache_nbytes, "cache_event_rows": info.n_event_rows,
                             "history_policy": self.history_policy, "mean_window_length": info.mean_window_length,
                             "transform_state": self.transform_state}

    @property
    def n_stream_rows(self) -> int:
        return int(self._ids.size)

    def _make(self, start: int, bs: int) -> dict:
        pos = np.arange(start, min(self._ids.size, start + bs), dtype=np.int64)
        b = add_session_offsets(self.store.batch(self.ds, pos), self.store, self.ds, pos)
        if self.renorm is not None:
            b = self.renorm.apply(b)
        if not (np.array_equal(b["decision_id"].numpy(), self._ids[pos])
                and np.array_equal(b["target_class"].numpy(), self._target[pos])):
            raise ProviderIntegrityError(f"{self.manifest_id}: an evaluation batch disagrees with the manifest")
        return b

    def batches(self, batch_size: int) -> Iterator[dict]:
        bs = int(batch_size)
        if bs != self.eval_batch:
            raise ProviderError(f"batch size {bs} != the frozen eval_batch {self.eval_batch} (PREG1 T3)")
        starts = range(0, int(self._ids.size), bs)
        if self.prefetch > 0:
            return ordered_prefetch(lambda a: self._make(a, bs), starts, self.prefetch)
        return (self._make(a, bs) for a in starts)

    def describe(self) -> dict:
        return {"provider": self.PROVIDER_ID, "manifest_id": self.manifest_id, "fingerprint": self.fingerprint,
                "history_policy": self.history_policy, "history": self.history,
                "transform_state": self.transform_state, "manifest_binding": self.manifest_binding,
                "n_manifest_rows": self.manifest.n, "n_stream_rows": self.n_stream_rows, "eval_batch": self.eval_batch,
                "eval_stream": self.eval_stream, "eval_stream_source": self.eval_stream_source,
                "prefetch_depth": self.prefetch, "manifest_hashes": self.manifest.hashes,
                "verified": self.verified.record(), "cache": self.cache_info.as_json(), "build": self.build_record,
                **({} if self.test_gate is None else {       # INT-07: a RETRO_TEST provider names its gate + content
                    "split": "RETRO_TEST", "test_gate": self.test_gate.binding(),
                    "test_exposure_register_entry_id": self.test_gate.entry_id,
                    "manifest_content_sha256": self.manifest_binding["content_sha256"]})}


POPULATION_HEADER_KEY = "population_sha256_selection_key_v2"           # R4 population_hash (the stable band identity,
                                                                        # shared by a role's train/eval pair; NOT the
                                                                        # per-manifest `population_sha256`)
R4_COMPARISON_KEY_FIELDS = (
    "data_protocol_hash", "data_version", "manifest_registry_self_sha256", "feature_schema_self_sha256",
    "feature_contract_hash", "history_policy", "transform_state_sha256", "split", "label_policy_hash",
    "population_hash", "candidate_order_hash", "train_manifest_id", "eval_manifest_id", "determinism_mode",
    "eval_stream", "exposure_point", "eval_score_block", "eval_batch", "eval_device_class")


def _refuse_non_frozen_eval_stream(eval_stream: Optional[str]) -> None:
    """R2: the R4 key's eval_stream must be the frozen contract name 'EVAL_ROWS_ONLY'; ALL_ROWS (or anything else) is
    refused. A separate, monkeypatchable function so a negative control can remove exactly this guard and nothing
    else (INT-02R1 F7) — the control then fails only if `comparison_key_base` stops calling it."""
    if eval_stream != "EVAL_ROWS_ONLY":
        raise ProviderError(f"R2: eval_stream must be the frozen contract name 'EVAL_ROWS_ONLY', got {eval_stream!r} "
                            "(ALL_ROWS is refused)")


def is_v2_lock(lock: Mapping) -> bool:
    """LOCK_V2-shaped: features_and_transforms carries the production_schema block (the v1 lock has flat fields)."""
    return isinstance((lock.get("features_and_transforms") or {}).get("production_schema"), Mapping)


def population_hash_of(header: Mapping, lock: Mapping) -> str:
    """TRAIN-02R2 F4: R4 population_hash. Under LOCK_V2 the header MUST carry population_sha256_selection_key_v2 (a
    missing key is an error, never a fallback); only a v1-shaped lock (the HISTORICAL route) may fall back to the
    per-manifest population_sha256."""
    if is_v2_lock(lock):
        v = header.get(POPULATION_HEADER_KEY)
        if not v:
            raise ProviderError(f"under DATA_PROTOCOL_LOCK_V2 the training manifest header must carry "
                                f"{POPULATION_HEADER_KEY!r} (R4 population_hash); no fallback")
        return v
    v = header.get(POPULATION_HEADER_KEY) or header.get("population_sha256")
    if not v:
        raise ProviderError(f"the training manifest header lacks {POPULATION_HEADER_KEY!r} and 'population_sha256' "
                            "(R4 population_hash)")
    return v


def _key_split(evalp) -> str:
    """INT-07: the key's split. VALIDATION for every non-TEST evaluation provider (unchanged); RETRO_TEST for a
    RETRO_TEST provider, whose gate is re-validated here; a RETRO_TEST header without an open gate is refused."""
    header = getattr(getattr(evalp, "manifest", None), "header", None) or {}
    is_test = str(header.get("v4_split", "")).upper() == "RETRO_TEST"
    gate = getattr(evalp, "test_gate", None)
    if gate is None and not is_test:
        return "VALIDATION"
    if gate is None or not is_test:
        raise ProviderError("a RETRO_TEST evaluation provider must carry its TEST gate (and only it may)")
    from . import retro_store as RS
    RS.require_gate(gate)
    return RS.RETRO_TEST


def comparison_key_base(paths: Optional[DataPaths], train: StoreTrainProvider, evalp: StoreEvalProvider) -> dict:
    """RunConfig.key for a COHORT run under DATA_PROTOCOL_LOCK_V2 (G1 combined gate, R4): the CLOSED comparison-key
    field list (`R4_COMPARISON_KEY_FIELDS`), every value explicit, read from LOCK_V2 and the two verified providers —
    no code default:
      data_protocol_hash          the DATA_PROTOCOL_LOCK_V2 self-hash `paths` verifies to;
      data_version                the manifest registry's data_version that verified both providers;
      manifest_registry_self_sha256  that registry's self-hash (train and eval must share one registry);
      feature_schema_self_sha256  lock.features_and_transforms.production_schema.self_sha256 (S*);
      feature_contract_hash       v4hist.windows.input_contract_sha256(S* self-hash, history_policy, applied state
                                  or None when it is the locked shared statistics);
      history_policy              DATA-05: identical for TRAIN and evaluation (DECISION_LOG 36);
      transform_state_sha256      the APPLIED transform state's self-hash (DECISION_LOG 42);
      split                       VALIDATION; RETRO_TEST for a gated RETRO_TEST evaluation provider (INT-07);
      label_policy_hash           sha256 of the lock's canonical splits_and_labels block;
      population_hash             the training manifest header's stable population band identity
                                  (`POPULATION_HEADER_KEY`), shared by the role's train/eval pair;
      candidate_order_hash        the evaluation manifest header's;
      train_manifest_id / eval_manifest_id  the two providers' verified manifest ids;
      determinism_mode             the locked mode (v4train.recipe.DETERMINISM_MODE_LOCK, "STRICT");
      eval_stream                  the frozen config's contract name ("EVAL_ROWS_ONLY"; R2 — the internal E2E_ROWS
                                  mapping stays v4eval-internal; ALL_ROWS is refused here too);
      exposure_point                the locked LR exposure point (v4train.schedule.LR_EXPOSURE_POINT, "end");
      eval_score_block / eval_batch / eval_device_class  the T3 additions (frozen.comparison_key_additions).
    Both providers must be registry-verified (TRAIN-01R2): a provider verified only against the historical
    DATA_PROTOCOL_LOCK (no manifest registry) cannot fill manifest_registry_self_sha256 and is refused."""
    from v4train.recipe import DETERMINISM_MODE_LOCK
    from v4train.schedule import LR_EXPOSURE_POINT
    tpol, epol = getattr(train, "history_policy", None), getattr(evalp, "history_policy", None)
    if tpol is None or epol is None:
        raise ProviderError("a provider carries no history_policy (DATA-05: required)")
    if require_policy(tpol) != require_policy(epol):
        raise ProviderError(f"train history_policy {tpol} != eval history_policy {epol}: the input window must be "
                            "identical for TRAIN and evaluation (DECISION_LOG 36)")
    tst = getattr(train, "transform_state", None) or {}
    est = getattr(evalp, "transform_state", None) or {}
    if not tst.get("transform_state_sha256") or tst.get("transform_state_sha256") != est.get("transform_state_sha256"):
        raise ProviderError("train and eval providers must apply the SAME transform state (DECISION_LOG 42)")
    lock, lself, _ = verify_lock(paths or DataPaths.default())
    if train.verified.lock_self_sha256 != lself or evalp.verified.lock_self_sha256 != lself:
        raise ProviderError("the providers were verified against a different DATA_PROTOCOL_LOCK")
    treg, ereg = getattr(train.verified, "registry", None), getattr(evalp.verified, "registry", None)
    if not treg or not ereg:
        raise ProviderError("R4 requires registry-verified providers (TRAIN-01R2): pass manifest_registry to both "
                            "StoreTrainProvider and StoreEvalProvider")
    if treg.get("self_sha256") != ereg.get("self_sha256"):
        raise ProviderError("train and eval providers were verified through different manifest registries")
    if treg.get("data_version") != ereg.get("data_version"):
        raise ProviderError("train and eval providers' manifest registries disagree on data_version")
    # INT-02R1 (OPS-10): comparison_key_base is also called against the HISTORICAL v1-shaped lock directly (its own
    # flat features_and_transforms fields, its own header population_sha256) — e.g. tests_hist's registries, all
    # bound to the v1 lock on an explicit historical route (never LOCK_V2 itself). Both shapes are read; neither is a
    # silent default (a lock/header carrying NEITHER shape still refuses below).
    ft = lock["features_and_transforms"]
    schema = (ft.get("production_schema") or {}).get("self_sha256") or ft.get("schema_self_sha256")
    state = tst["transform_state_sha256"]
    locked_state = (ft.get("shared_statistics") or {}).get("transform_state_sha256") or ft.get("transform_state_self_sha256")
    if not schema or not locked_state:
        raise ProviderError("the lock carries neither the LOCK_V2 (production_schema / shared_statistics) nor the "
                            "v1 (schema_self_sha256 / transform_state_self_sha256) feature/transform-state fields")
    header = train.verified.header
    population_hash = population_hash_of(header, lock)                # TRAIN-02R2 F4: fallback on a v1 lock only
    split = _key_split(evalp)                                         # INT-07: RETRO_TEST only under the gate
    eval_stream = getattr(evalp.frozen, "eval_stream", None)
    _refuse_non_frozen_eval_stream(eval_stream)
    key = {"data_protocol_hash": lself, "data_version": treg["data_version"],
          "manifest_registry_self_sha256": treg["self_sha256"], "feature_schema_self_sha256": schema,
          "feature_contract_hash": input_contract_sha256(schema, tpol, None if state == locked_state else state),
          "history_policy": tpol, "transform_state_sha256": state, "split": split,
          "label_policy_hash": _fp(lock["splits_and_labels"]), "population_hash": population_hash,
          "candidate_order_hash": evalp.manifest.header["candidate_order_hash"], "train_manifest_id": train.manifest_id,
          "eval_manifest_id": evalp.manifest_id, "determinism_mode": DETERMINISM_MODE_LOCK, "eval_stream": eval_stream,
          "exposure_point": LR_EXPOSURE_POINT, **evalp.frozen.comparison_key_additions}
    missing = [f for f in R4_COMPARISON_KEY_FIELDS if f not in key]
    extra = [f for f in key if f not in R4_COMPARISON_KEY_FIELDS]
    if missing or extra:
        raise ProviderError(f"R4 comparison key is not the closed field list: missing {missing}, extra {extra}")
    return key


__all__ = ["StoreTrainProvider", "StoreEvalProvider", "ProviderError", "ProviderIntegrityError", "resolve_eval_stream",
           "ordered_prefetch", "comparison_key_base", "manifest_binding_of", "MANIFEST_CONTENT_RULE",
           "EVAL_MANIFEST_COLUMNS", "STREAM_NAMES", "R4_COMPARISON_KEY_FIELDS", "POPULATION_HEADER_KEY"]

"""v4int.verify — fail-closed verification of a Clean V2 manifest before any row of it is used (INT-01).

Nothing is used on trust. `verify_manifest(manifest_id, paths)` refuses (raises) unless ALL of these hold:
  * the manifest id is not a TEST manifest (M07-M09, or any id containing TEST) — checked before any file is opened;
    after the header is read, a RETRO_TEST header or a TEST / RETRO_TEST row in the decisions table is refused too;
  * DATA_PROTOCOL_LOCK: strict JSON, `document`, and its self-hash recomputed with the lock's own rule;
  * MANIFEST_HASHES.json: file sha256 = the lock's `manifest_hashes_file_sha256`, and its entry for this manifest equals
    the lock's entry exactly;
  * every table file (decisions, clients / users, central_order): file sha256 = the lock's `file_sha256`; re-read, the
    MANIFEST_SPEC `table_content_sha256` (v4data.manifests.reread_columns + v4data.common.table_content_sha256) and
    the row count equal the lock's;
  * the header (DATA-04R1 re-issue): manifest id, `status` BUILT_PRE_LOCK, no ALL_METHODS_LOCK / TEST-exposure entry,
    `data_protocol_lock_sha256` = the lock self-hash, `data_protocol_lock_file_sha256` = the lock file sha256, the
    re-issue block pointing at the lock's pre-lock header (file + content sha256), the header content self-hash
    recomputed with the builder's rule, its tables / ordered hashes / population / counts / builder code / schema /
    dataset / candidate-order / K equal to the lock's, and its file sha256 listed in MANIFEST_FILE_SHA256_R1.txt;
  * the `.OK` success marker lists sha256 values equal to the files on disk;
  * TRANSFORM_STATE_CLEAN_V2.json: file sha256 and self-hash (recomputed) equal the lock's; v4data.transform_state
    .load_state accepts it (schema consistency);
  * the decisions table order hash = the header's; for TRAIN manifests the central_order table is a permutation onto a
    strictly ascending decision_id order (MANIFEST_SPEC central_canonical_order; when the header carries the derived
    value, its hash equals it), every target_class is in [0, K), every target_status is OBSERVED_IN_PARTITION and every
    old_split is TRAIN; for evaluation manifests the table order itself is strictly ascending;
  * the row count equals the header's and the lock's n_decisions (no count is hard-coded anywhere).
Population-agnostic (OWNER_AMENDMENT_POP_5_100): any manifest registered in the lock is served. Its role comes from its
locked table layout (MANIFEST_SPEC: clients + decisions + central_order = a TRAIN manifest; users + decisions = an
evaluation manifest), never from its id; a caller may require a role (expect_kind) and is refused otherwise.
Read-only: nothing is written. Imports v4train first (ppsi_env isolation before any snapshot import).

DATA-05 (Ops, SCHEMA-01 finding): `verify_manifest_by_registry(manifest_id, registry_path, paths)` is the second,
registry-based path for manifests that DATA_PROTOCOL_LOCK does not list (FS-01's FEATURE_SEARCH_SCREEN manifests,
DATA-06's v2 bands). The lock path above is unchanged. The registry path refuses unless ALL of these hold:
  * the manifest id is not a TEST manifest (before any file is opened; the header and rows are checked too);
  * DATA_PROTOCOL_LOCK verifies (its own rule) and the registry's `source_data_protocol_lock` equals its self-hash and
    file sha256;
  * the registry is strict JSON whose `self_sha256` verifies (sha256 of json.dumps(object minus self_sha256,
    sort_keys, ASCII, compact, allow_nan=False); the lock-style variant that also drops `self_sha256_rule` is accepted,
    as v4train.registry does), and it has an entry for the id whose table layout is a served layout;
  * the registry's transform_state names the lock's TRANSFORM_STATE (self-hash and file sha256), which verifies;
  * every table file's sha256 equals the entry's; re-read, its table_content_sha256 and row count equal the entry's;
  * the header file sha256 = the entry's header_sha256; the header's content self-hash (builder rule) verifies and
    equals the entry's header_content_sha256; content_sha256 (the REGISTRY_FS rule over the verified tables) equals
    the entry's; the header's status equals the entry's and is FEATURE_SEARCH_SCREEN or one of the registry's own
    role names (DATA-06 v2 bands; never PARTIAL_TEST_BUILD, never a TEST name); the header's data_version equals the
    registry's; the header binds the lock (self and file sha256); no ALL_METHODS_LOCK / TEST-exposure entry; tables,
    ordered hashes, population and counts equal the entry's; builder code, schema, dataset, candidate order and K
    equal the lock's;
  * the `.OK` marker matches the files on disk; and the same row checks as the lock path (order hash, canonical
    order, targets, statuses, TRAIN split, counts).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import v4train  # noqa: F401  (ppsi_env first: snapshot isolation before any snapshot import)

import numpy as np
import pyarrow.parquet as pq

from v4data.common import TestGateError, canonical_json_bytes, ordered_ids_sha256, self_hash, sha256_bytes, sha256_file
from v4data.common import table_content_sha256
from v4data.manifests import reread_columns

V4 = Path(__file__).resolve().parents[2]
TEST_MANIFEST_PREFIXES = ("M07", "M08", "M09")          # the TEST gate (MANIFEST_SPEC M07-M09), not a population
HEADER_STATUS = "BUILT_PRE_LOCK"
TABLES = {"TRAIN": ("clients", "decisions", "central_order"), "EVAL": ("users", "decisions")}   # MANIFEST_SPEC layouts


def manifest_kind(entry) -> str:
    """TRAIN or EVAL from a lock entry's table set (the MANIFEST_SPEC layout); anything else is refused."""
    got = set((entry or {}).get("tables", {}))
    for kind, tabs in TABLES.items():
        if got == set(tabs):
            return kind
    raise ManifestVerificationError(f"manifest tables {sorted(got)} match no served layout {TABLES}")


class ManifestVerificationError(RuntimeError):
    """A manifest, its header, the lock, MANIFEST_HASHES or the transform state failed verification: refused."""


class TestManifestRefused(TestGateError):
    """A TEST manifest (or TEST rows) was requested before ALL_METHODS_LOCK: refused unconditionally."""


@dataclass(frozen=True)
class DataPaths:
    manifest_dir: Path
    lock_path: Path
    manifest_hashes_path: Path
    header_list_path: Path
    transform_state_path: Path
    production: bool = False

    @classmethod
    def default(cls, v4: Path = V4) -> "DataPaths":
        """B2 (G1 combined gate, INT-02 item 2): the production default binds DATA_PROTOCOL_LOCK_V2 over the data_v3
        tree (the shared TRANSFORM_STATE file is unchanged, unlocked_state.path in both locks). The v1
        DATA_PROTOCOL_LOCK / data_v2 manifest defaults remain reachable, HISTORICAL ONLY, through `.historical()`."""
        d = Path(v4) / "data_v3"
        return cls(manifest_dir=d / "manifests", lock_path=Path(v4) / "contracts" / "DATA_PROTOCOL_LOCK_V2.json",
                   manifest_hashes_path=d / "receipts" / "MANIFEST_HASHES.json",
                   header_list_path=d / "receipts" / "MANIFEST_FILE_SHA256_R1.txt",
                   transform_state_path=Path(v4) / "data_v2" / "TRANSFORM_STATE_CLEAN_V2.json", production=True)

    @classmethod
    def historical(cls, v4: Path = V4) -> "DataPaths":
        """HISTORICAL ONLY (B2): the superseded v1 DATA_PROTOCOL_LOCK over the data_v2 manifest tree (M01/M02/M05/M06).
        Kept available for the v1 lock verification path; never the default for a new run."""
        d = Path(v4) / "data_v2"
        return cls(manifest_dir=d / "manifests", lock_path=Path(v4) / "contracts" / "DATA_PROTOCOL_LOCK.json",
                   manifest_hashes_path=d / "receipts" / "MANIFEST_HASHES.json",
                   header_list_path=d / "receipts" / "MANIFEST_FILE_SHA256_R1.txt",
                   transform_state_path=d / "TRANSFORM_STATE_CLEAN_V2.json", production=True)

    def as_json(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}


@dataclass
class VerifiedManifest:
    manifest_id: str
    kind: str                                   # TRAIN | EVAL
    header: dict
    header_file_sha256: str
    tables: dict                                # table -> {file, file_sha256, row_count, table_content_sha256}
    columns: dict                               # decisions-table columns (numpy; strings as numpy object arrays)
    canonical_decision_id: np.ndarray           # strictly ascending decision_id (central_canonical_order)
    canonical_rows: np.ndarray                  # decisions-table row of each canonical position
    lock_self_sha256: str
    lock_file_sha256: str
    transform_state_self_sha256: str
    transform_state_file_sha256: str
    checks: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    registry: Optional[dict] = None             # the registry record when verified by manifest registry (DATA-05)

    def file_hashes(self) -> dict:
        out = {t: v["file_sha256"] for t, v in sorted(self.tables.items())}
        out["header"] = self.header_file_sha256
        return out

    def record(self) -> dict:
        return {"manifest_id": self.manifest_id, "kind": self.kind, "n_decisions": int(self.canonical_decision_id.size),
                "header_file_sha256": self.header_file_sha256,
                "header_content_sha256": self.header.get("header_content_sha256"),
                "tables": self.tables, "ordered_decision_id_sha256": self.header.get("ordered_decision_id_sha256"),
                "canonical_ordered_decision_id_sha256": ordered_ids_sha256(self.canonical_decision_id),
                "lock_self_sha256": self.lock_self_sha256, "lock_file_sha256": self.lock_file_sha256,
                "transform_state_self_sha256": self.transform_state_self_sha256,
                "transform_state_file_sha256": self.transform_state_file_sha256,
                "checks_passed": len(self.checks), "checks": self.checks, "timings_s": self.timings,
                "verified_by": "MANIFEST_REGISTRY" if self.registry else "DATA_PROTOCOL_LOCK",
                "registry": self.registry}


# ------------------------------------------------------------------------------------------------ helpers
def strict_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"),
                          parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
    except (OSError, ValueError) as e:
        raise ManifestVerificationError(f"{path}: not readable strict JSON ({e})") from e


def lock_self_sha256(lock: Mapping) -> str:
    """The DATA_PROTOCOL_LOCK rule: sha256 of json.dumps(lock without lock_self_sha256 and lock_self_sha256_rule,
    sort_keys=True, ensure_ascii=True, separators=(',', ':'))."""
    d = {k: v for k, v in lock.items() if k not in ("lock_self_sha256", "lock_self_sha256_rule")}
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def header_content_sha256(h: Mapping) -> str:
    """The builder's rule (v4data.manifests._header; DATA-04R1 reissue): canonical JSON of the header without
    build_utc, reissue_utc and header_content_sha256, tables reduced to (table, row_count, table_content_sha256)."""
    stable = {k: v for k, v in h.items() if k not in ("build_utc", "tables", "header_content_sha256", "reissue_utc")}
    stable["tables"] = [{k: t[k] for k in ("table", "row_count", "table_content_sha256")} for t in h["tables"]]
    return sha256_bytes(canonical_json_bytes(stable))


def _canon(obj) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def manifest_content_sha256(header_tables, tables: Mapping) -> str:
    """REGISTRY_FS rules.content_sha256 (= v4fs_data.fs_common.manifest_content_sha256): sha256 of the canonical JSON
    list [{table, row_count, table_content_sha256}] in the header's table order, taken from `tables` (the verified
    identities)."""
    order = [t["table"] for t in header_tables]
    if sorted(order) != sorted(tables) or len(set(order)) != len(order):
        raise ManifestVerificationError("the header's table list differs from the verified tables")
    return _canon([{"table": t, "row_count": int(tables[t]["row_count"]),
                    "table_content_sha256": tables[t]["table_content_sha256"]} for t in order])


def refuse_test_manifest(manifest_id: str, header: Optional[Mapping] = None,
                         columns: Optional[Mapping[str, np.ndarray]] = None) -> None:
    mid = str(manifest_id)
    if mid.startswith(TEST_MANIFEST_PREFIXES) or "TEST" in mid.upper():
        raise TestManifestRefused(f"{mid}: TEST manifests are refused before ALL_METHODS_LOCK (E10.1; INT-01)")
    if header is not None:
        if str(header.get("v4_split", "")).upper() in ("RETRO_TEST", "TEST") or header.get("all_methods_lock_sha256"):
            raise TestManifestRefused(f"{mid}: the header is a TEST / post-ALL_METHODS_LOCK header: refused")
    if columns is not None:
        for c, bad in (("old_split", ("TEST",)), ("v4_split", ("RETRO_TEST",))):
            if c in columns and bool(np.isin(np.asarray(columns[c], dtype=object), bad).any()):
                raise TestManifestRefused(f"{mid}: the decisions table holds {c} in {bad}: refused")


def parse_sha_list(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        sha, name = line.split(None, 1)
        out[name.strip()] = sha
    return out


def _table_numpy(tbl) -> dict:
    return {name: tbl[name].to_numpy(zero_copy_only=False) for name in tbl.column_names}


# ------------------------------------------------------------------------------------------------ verification
def verify_lock(paths: DataPaths) -> tuple:
    lock = strict_json(paths.lock_path)
    got = lock_self_sha256(lock)
    if lock.get("document") != "DATA_PROTOCOL_LOCK" or lock.get("lock_self_sha256") != got:
        raise ManifestVerificationError(f"DATA_PROTOCOL_LOCK self-hash {lock.get('lock_self_sha256')} != recomputed "
                                        f"{got} (or not a DATA_PROTOCOL_LOCK document)")
    return lock, got, sha256_file(paths.lock_path)


def verify_transform_state(paths: DataPaths, lock: Mapping) -> tuple:
    """INT-02R1 F1: DATA_PROTOCOL_LOCK_V2 nests the shared-statistics binding under
    `features_and_transforms.shared_statistics.{transform_state_sha256, file_sha256}`; the v1 (HISTORICAL) lock keeps
    the old flat `transform_state_self_sha256` / `transform_state_file_sha256` keys. Both name the SAME file
    (unchanged, ADR-005 addendum 2 D5), so either shape is accepted, whichever `lock` carries."""
    ft = lock["features_and_transforms"]
    shared = ft.get("shared_statistics")
    if isinstance(shared, Mapping):
        want_self, want_file = shared.get("transform_state_sha256"), shared.get("file_sha256")
    else:
        want_self, want_file = ft.get("transform_state_self_sha256"), ft.get("transform_state_file_sha256")
    fsha = sha256_file(paths.transform_state_path)
    state = strict_json(paths.transform_state_path)
    rec = state.get("transform_state_sha256")
    got = self_hash(state, "transform_state_sha256")
    if not (rec == got == want_self) or fsha != want_file:
        raise ManifestVerificationError(f"TRANSFORM_STATE self-hash recorded {rec} / recomputed {got} / lock "
                                        f"{want_self}; file {fsha} / lock {want_file}")
    from v4data.transform_state import load_state
    try:
        load_state(paths.transform_state_path)
    except ValueError as e:
        raise ManifestVerificationError(f"TRANSFORM_STATE refused by v4data.transform_state.load_state: {e}") from e
    return got, fsha


def verify_manifest(manifest_id: str, paths: Optional[DataPaths] = None, *, expect_kind: Optional[str] = None,
                    log=None) -> VerifiedManifest:
    mid = str(manifest_id)
    refuse_test_manifest(mid)                                 # before any file is opened
    paths = paths or DataPaths.default()
    say = log or (lambda *_: None)
    t0 = time.time()
    checks: dict = {}
    lock, lself, lfile = verify_lock(paths)
    checks["lock_self_sha256_recomputed"] = True
    pm = lock["population_and_manifests"]
    entry = pm["manifest_hashes"]["manifests"].get(mid)
    if entry is None:
        raise ManifestVerificationError(f"{mid}: no entry in DATA_PROTOCOL_LOCK manifest_hashes (not registered)")
    kind = manifest_kind(entry)
    if expect_kind is not None and kind != expect_kind:
        raise ManifestVerificationError(f"{mid}: a {kind} manifest (locked table layout), not {expect_kind}")
    mh_sha = sha256_file(paths.manifest_hashes_path)
    mh = strict_json(paths.manifest_hashes_path)
    checks["manifest_hashes_file_sha256_eq_lock"] = mh_sha == pm["manifest_hashes_file_sha256"]
    checks["manifest_hashes_entry_eq_lock"] = _canon(mh.get("manifests", {}).get(mid)) == _canon(entry)
    ts_self, ts_file = verify_transform_state(paths, lock)
    checks["transform_state_verified"] = True
    t_lock = time.time()

    d = Path(paths.manifest_dir)
    listed = parse_sha_list(paths.header_list_path)
    tables = {}
    for t in TABLES[kind]:
        e = entry["tables"].get(t)
        f = d / f"{mid}.{t}.parquet"
        if e is None or not f.is_file():
            raise ManifestVerificationError(f"{mid}: table {t} missing from the lock entry or from disk ({f})")
        fsha = sha256_file(f)
        checks[f"{t}.file_sha256_eq_lock"] = fsha == e["file_sha256"]
        checks[f"{t}.file_sha256_listed_r1"] = listed.get(f.name) == fsha
        tables[t] = {"file": f.name, "file_sha256": fsha, "row_count": int(e["row_count"]),
                     "table_content_sha256": e["table_content_sha256"]}
    hp = d / f"{mid}.header.json"
    hsha = sha256_file(hp)
    header = strict_json(hp)
    refuse_test_manifest(mid, header=header)
    lk, ftr = lock, lock["features_and_transforms"]
    checks.update({
        "header.file_sha256_listed_r1": listed.get(hp.name) == hsha,
        "header.manifest_id": header.get("manifest_id") == mid,
        "header.status": header.get("status") == HEADER_STATUS,
        "header.no_all_methods_lock": header.get("all_methods_lock_sha256") is None,
        "header.no_test_exposure_entry": header.get("test_exposure_register_entry_id") is None,
        "header.data_protocol_lock_sha256_eq_lock_self": header.get("data_protocol_lock_sha256") == lself,
        "header.data_protocol_lock_file_sha256_eq_lock_file": header.get("data_protocol_lock_file_sha256") == lfile,
        "header.content_self_hash": header.get("header_content_sha256") == header_content_sha256(header),
        "header.reissue_previous_file_eq_lock": (header.get("reissue") or {}).get("previous_header_file_sha256")
        == entry.get("header_file_sha256"),
        "header.reissue_previous_content_eq_lock": (header.get("reissue") or {}).get("previous_header_content_sha256")
        == entry.get("header_content_sha256"),
        "header.tables_eq_lock": {x["table"]: {k: x[k] for k in ("row_count", "table_content_sha256", "file_sha256")}
                                  for x in header.get("tables", [])} == {
            t: {k: v[k] for k in ("row_count", "table_content_sha256", "file_sha256")} for t, v in entry["tables"].items()},
        "header.builder_code_sha256_eq_lock": header.get("builder_code_sha256") == lk["code"]["v4data_builder_code_sha256"],
        "header.schema_self_sha256_eq_lock": header.get("feature_schema_self_sha256") == ftr["schema_self_sha256"],
        "header.dataset_version_eq_lock": header.get("dataset_version_hash") == lk["data_version"]["base_version_hash"],
        "header.candidate_order_eq_lock": header.get("candidate_order_hash") == pm["manifest_hashes"]["candidate_order_hash"],
        "header.catalogue_K_eq_lock": header.get("catalogue_K") == lk["catalogue"]["K"],
    })
    for k in ("ordered_decision_id_sha256", "eval_ordered_decision_id_sha256", "rankable_ordered_decision_id_sha256",
              "population_sha256", "n_users", "n_decisions", "n_eval_rows", "n_rankable_rows"):
        checks[f"header.{k}_eq_lock"] = header.get(k) == entry.get(k)
    marker = d / f"{mid}.OK"
    if marker.is_file():
        mk = strict_json(marker).get("verified_sha256", {})
        checks["ok_marker_matches_disk"] = bool(mk) and all(
            (d.parent / rel).is_file() and sha256_file(d.parent / rel) == s for rel, s in mk.items())
    else:
        checks["ok_marker_matches_disk"] = False
    t_head = time.time()
    _fail_if_bad(mid, checks)
    cols, canon, rows = _verify_tables_and_rows(mid, kind, header, {t: d / tables[t]["file"] for t in TABLES[kind]},
                                                tables, entry, checks, "lock")
    t_end = time.time()
    say(f"[v4int.verify] {mid}: {len(checks)} checks passed ({t_end - t0:.1f} s)")
    return VerifiedManifest(manifest_id=mid, kind=kind, header=header, header_file_sha256=hsha, tables=tables,
                            columns=cols, canonical_decision_id=np.ascontiguousarray(canon, dtype=np.int64),
                            canonical_rows=np.ascontiguousarray(rows, dtype=np.int64), lock_self_sha256=lself,
                            lock_file_sha256=lfile, transform_state_self_sha256=ts_self,
                            transform_state_file_sha256=ts_file, checks=checks,
                            timings={"lock_and_state": round(t_lock - t0, 2), "files_and_header": round(t_head - t_lock, 2),
                                     "tables": round(t_end - t_head, 2), "total": round(t_end - t0, 2)})


def _verify_tables_and_rows(mid: str, kind: str, header: Mapping, files: Mapping, tables: Mapping, entry: Mapping,
                            checks: dict, src: str) -> tuple:
    """The table re-read and row checks shared by both paths (`src` names the authority in the check keys: "lock"
    keeps the INT-01 check names byte for byte). Returns (decisions columns, canonical decision ids, canonical rows)."""
    cols, central = None, None
    for t in TABLES[kind]:
        tbl = pq.read_table(files[t])
        checks[f"{t}.row_count_eq_{src}"] = tbl.num_rows == tables[t]["row_count"]
        checks[f"{t}.table_content_sha256_eq_{src}"] = table_content_sha256(reread_columns(tbl)) == \
            tables[t]["table_content_sha256"]
        if t == "decisions":
            cols = _table_numpy(tbl)
        elif t == "central_order":
            central = tbl["row"].to_numpy(zero_copy_only=False).astype(np.int64)
        del tbl
    refuse_test_manifest(mid, header=header, columns=cols)
    did = np.asarray(cols["decision_id"], dtype=np.int64)
    checks["decisions.order_hash_eq_header"] = ordered_ids_sha256(did) == header.get("ordered_decision_id_sha256")
    if kind == "TRAIN":
        n = did.size
        is_perm = central is not None and central.shape == (n,) and np.array_equal(np.sort(central), np.arange(n))
        checks["central_order.is_permutation"] = bool(is_perm)
        _fail_if_bad(mid, checks)
        canon = did[central]
        checks["central_order.strictly_ascending"] = bool(n <= 1 or np.all(canon[1:] > canon[:-1]))
        derived = (header.get("derived") or {}).get("central_order_ordered_decision_id_sha256")
        if derived is not None:
            checks["central_order.hash_eq_header_derived"] = ordered_ids_sha256(canon) == derived
        tc = np.asarray(cols["target_class"], dtype=np.int64)
        checks["decisions.target_class_in_range"] = bool(n == 0 or (tc.min() >= 0 and tc.max() < int(header["catalogue_K"])))
        checks["decisions.target_status_observed"] = bool(np.all(np.asarray(cols["target_status"], dtype=object)
                                                                 == "OBSERVED_IN_PARTITION"))
        checks["decisions.old_split_train"] = bool(np.all(np.asarray(cols["old_split"], dtype=object) == "TRAIN"))
        rows = central
    else:
        n = did.size
        checks["decisions.strictly_ascending"] = bool(n <= 1 or np.all(did[1:] > did[:-1]))
        rows = np.arange(n, dtype=np.int64)
        canon = did
    checks["n_decisions_eq_header"] = int(canon.size) == int(header.get("n_decisions", -1))
    checks[f"n_decisions_eq_{src}"] = int(canon.size) == int(entry.get("n_decisions", -1))
    _fail_if_bad(mid, checks)
    return cols, canon, rows


# ------------------------------------------------------------------------------------------------ registry path
REGISTRY_SELF_KEY = "self_sha256"
REGISTRY_RULE_KEY = "self_sha256_rule"
FS_SCREEN_STATUS = "FEATURE_SEARCH_SCREEN"


def registry_self_sha256(obj: Mapping, exclude: tuple = ()) -> str:
    body = {k: v for k, v in obj.items() if k != REGISTRY_SELF_KEY and k not in exclude}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
                          .encode("utf-8")).hexdigest()


def _b1_superseded_exception(src: Mapping, stated: str, lock: Optional[Mapping], lock_self: str) -> bool:
    """G1 combined gate B1 (INT-02 item 2): `verify_lock` today refuses a registry whose declared
    `source_data_protocol_lock` is not THIS lock's self-hash — which would refuse REGISTRY_V2 under
    DATA_PROTOCOL_LOCK_V2 (REGISTRY_V2 was built on the superseded v1 lock). Accepted iff BOTH, read from `lock`
    itself (no code default, no hard-coded hash):
      clause 1: registry.source_data_protocol_lock == lock.supersedes (lock_self_sha256 AND file_sha256);
      clause 2: registry.self_sha256 == lock.population_and_manifests.manifest_registry.self_sha256.
    Any other pairing (either clause failing alone) is refused; this is a one-way exception (LOCK_V2 -> REGISTRY_V2 by
    self-hash is the only non-circular direction, per the ruling) and never applies when `lock` carries no
    `supersedes` block (e.g. the v1 lock itself, or a lock that supersedes nothing), nor when `lock` is not actually
    the lock `lock_self` was verified against (belt and braces: `lock_self_sha256(lock)` — RECOMPUTED, never the
    stored field alone (INT-02R1 F4) — must equal `lock_self`).
    This is the CHEAP check (declared hashes only); `_load_superseded_lock` then loads and independently verifies the
    REAL v1 lock file the exception relies on (INT-02R1 F1) before any header is trusted against it."""
    if not isinstance(lock, Mapping) or lock_self_sha256(lock) != lock_self:
        return False
    sup = lock.get("supersedes") or {}
    pinned = ((lock.get("population_and_manifests") or {}).get("manifest_registry") or {})
    sup_self, sup_file, pinned_self = sup.get("lock_self_sha256"), sup.get("file_sha256"), pinned.get("self_sha256")
    clause1 = bool(sup_self and sup_file and src.get("lock_self_sha256") == sup_self and src.get("file_sha256") == sup_file)
    clause2 = bool(pinned_self and stated == pinned_self)
    return clause1 and clause2


def _load_superseded_lock(lock: Mapping, v4: Path = V4) -> tuple:
    """INT-02R1 F1: the B1 exception must not just compare declared hashes — it must load the REAL lock
    `lock.supersedes` names (`supersedes.file`, relative to `v4`), recompute ITS self-hash (`lock_self_sha256`, the
    canonical rule — never trust `v1_lock['lock_self_sha256']` as a stored field alone) and its file sha256, and
    confirm both equal `lock.supersedes`. Returns (v1_lock, v1_lock_self_sha256, v1_lock_file_sha256)."""
    sup = lock.get("supersedes") or {}
    rel = sup.get("file")
    if not rel:
        raise ManifestVerificationError("lock.supersedes.file is required to load the superseded lock (B1)")
    p = Path(v4) / str(rel)
    v1 = strict_json(p)
    got = lock_self_sha256(v1)
    if v1.get("document") != "DATA_PROTOCOL_LOCK" or got != sup.get("lock_self_sha256"):
        raise ManifestVerificationError(f"{p}: recomputed self-hash {got} != lock.supersedes.lock_self_sha256 "
                                        f"{sup.get('lock_self_sha256')} (or not a DATA_PROTOCOL_LOCK document)")
    fsha = sha256_file(p)
    if fsha != sup.get("file_sha256"):
        raise ManifestVerificationError(f"{p}: file sha256 {fsha} != lock.supersedes.file_sha256 "
                                        f"{sup.get('file_sha256')}")
    return v1, got, fsha


def verify_registry(registry_path, lock_self: str, lock_file: str, *, lock: Optional[Mapping] = None,
                    root: Optional[Path] = None) -> tuple:
    """(registry object, file sha256, self sha256, v1_ctx): self-hash verified; built on THIS DATA_PROTOCOL_LOCK
    (`v1_ctx` is then None), OR accepted under the B1 superseded-lock exception (`_b1_superseded_exception`), in
    which case `v1_ctx` = (v1_lock, v1_lock_self_sha256, v1_lock_file_sha256) — the REAL, independently loaded and
    verified superseded lock (`_load_superseded_lock`), for the caller to bind the header against (INT-02R1 F1).
    `root` resolves `lock.supersedes.file` (default V4, the real ppsi_v4 workspace; a test workspace root otherwise
    — the same root the registry's own relative paths resolve against).
    F3: whenever `lock` pins `population_and_manifests.manifest_registry.self_sha256`, the registry's self_sha256
    MUST equal it, on BOTH routes (not just as part of the B1 exception's clause 2)."""
    p = Path(registry_path)
    if not p.is_file():
        raise ManifestVerificationError(f"manifest registry {p} does not exist")
    reg = strict_json(p)
    stated = reg.get(REGISTRY_SELF_KEY)
    ok = {registry_self_sha256(reg)}
    if REGISTRY_RULE_KEY in reg:
        ok.add(registry_self_sha256(reg, (REGISTRY_RULE_KEY,)))
    if not isinstance(stated, str) or stated not in ok:
        raise ManifestVerificationError(f"manifest registry {p}: self_sha256 {stated!r} does not verify")
    pinned_self = ((lock or {}).get("population_and_manifests") or {}).get("manifest_registry", {}).get("self_sha256") \
        if isinstance(lock, Mapping) else None
    if pinned_self and stated != pinned_self:
        raise ManifestVerificationError(f"manifest registry {p}: self_sha256 {stated} != the lock's pinned "
                                        f"population_and_manifests.manifest_registry.self_sha256 {pinned_self} (F3)")
    src = reg.get("source_data_protocol_lock") or {}
    direct = src.get("lock_self_sha256") == lock_self and src.get("file_sha256") == lock_file
    v1_ctx = None
    if not direct:
        if not _b1_superseded_exception(src, stated, lock, lock_self):
            raise ManifestVerificationError(f"manifest registry {p} was built on another DATA_PROTOCOL_LOCK "
                                            f"({src}; current self {lock_self}, file {lock_file}); the B1 "
                                            "superseded-lock exception (registry.source == lock.supersedes AND "
                                            "registry.self_sha256 == lock.population_and_manifests.manifest_registry"
                                            ".self_sha256) does not apply")
        v1_ctx = _load_superseded_lock(lock, v4=root if root is not None else V4)  # F1: load + verify the v1 lock
    if not isinstance(reg.get("manifests"), dict) or not isinstance(reg.get("roles"), dict) or not reg.get("data_version"):
        raise ManifestVerificationError(f"manifest registry {p}: 'data_version', 'roles' and 'manifests' are required")
    return reg, sha256_file(p), stated, v1_ctx


def registry_statuses(reg: Mapping) -> tuple:
    """Header statuses a registry can serve: FEATURE_SEARCH_SCREEN and the registry's own role names (DATA-06 labels a
    v2 band header with its role); never a TEST name and never PARTIAL_TEST_BUILD."""
    names = {FS_SCREEN_STATUS} | {str(r) for r in reg.get("roles", {})}
    return tuple(sorted(n for n in names if "TEST" not in n.upper()))


def verify_manifest_by_registry(manifest_id: str, registry_path, paths: Optional[DataPaths] = None, *,
                                root: Optional[Path] = None, expect_kind: Optional[str] = None,
                                log=None) -> VerifiedManifest:
    """The registry-based verification path (see the module docstring). `root` is the registry's path_root (the
    ppsi_v4 workspace; default V4); `paths` gives the lock and the TRANSFORM_STATE (default the locked data_v2)."""
    mid = str(manifest_id)
    refuse_test_manifest(mid)                                 # before any file is opened
    paths = paths or DataPaths.default()
    root = Path(root) if root is not None else V4
    say = log or (lambda *_: None)
    t0 = time.time()
    checks: dict = {}
    lock, lself, lfile = verify_lock(paths)
    checks["lock_self_sha256_recomputed"] = True
    reg, reg_file, reg_self, v1_ctx = verify_registry(registry_path, lself, lfile, lock=lock, root=root)
    checks["registry.self_sha256_verified"] = True
    checks["registry.source_lock_eq_lock"] = True
    # F1: when the registry was accepted through the B1 superseded-lock exception, the header it names is bound to
    # the REAL v1 lock (`v1_ctx`), not to LOCK_V2 — bind and compare against that lock instead, wherever the header
    # itself was built against it (DATA-06's headers: v4data06/d6_common.py:150).
    ref_lock, ref_self, ref_file = v1_ctx if v1_ctx is not None else (lock, lself, lfile)
    entry = reg["manifests"].get(mid)
    if not isinstance(entry, dict):
        raise ManifestVerificationError(f"{mid}: no entry in the manifest registry {registry_path}")
    kind = manifest_kind(entry)
    if entry.get("kind") not in (None, kind):
        raise ManifestVerificationError(f"{mid}: registry kind {entry.get('kind')} != its table layout {kind}")
    if expect_kind is not None and kind != expect_kind:
        raise ManifestVerificationError(f"{mid}: a {kind} manifest (registered table layout), not {expect_kind}")
    ts_self, ts_file = verify_transform_state(paths, lock)
    rts = reg.get("transform_state") or {}
    checks["registry.transform_state_eq_lock"] = (rts.get("transform_state_sha256") == ts_self
                                                  and rts.get("file_sha256") == ts_file)
    t_lock = time.time()

    tables, files = {}, {}
    for t in TABLES[kind]:
        e = entry["tables"].get(t)
        f = root / str((e or {}).get("file", ""))
        if not e or not f.is_file():
            raise ManifestVerificationError(f"{mid}: table {t} missing from the registry entry or from disk ({f})")
        fsha = sha256_file(f)
        checks[f"{t}.file_sha256_eq_registry"] = fsha == e.get("file_sha256")
        tables[t] = {"file": f.name, "file_sha256": fsha, "row_count": int(e["row_count"]),
                     "table_content_sha256": e["table_content_sha256"]}
        files[t] = f
    hp = root / str(entry.get("path", ""))
    if not hp.is_file():
        raise ManifestVerificationError(f"{mid}: header file {hp} is missing")
    hsha = sha256_file(hp)
    header = strict_json(hp)
    refuse_test_manifest(mid, header=header)
    status = header.get("status")
    # F1: builder code / (BASE) schema / candidate order / K are checked against `ref_lock` — the REAL v1 lock when
    # the registry was accepted via B1 (DATA-06's headers are built against it, never against S* or LOCK_V2's own,
    # absent, flat fields), else the same `lock` the direct route already verified against.
    ref_code = (ref_lock.get("code") or {}).get("v4data_builder_code_sha256")
    ref_schema = (ref_lock.get("features_and_transforms") or {}).get("schema_self_sha256")
    ref_dataset_version = (ref_lock.get("data_version") or {}).get("base_version_hash")
    ref_candidate_order = ((ref_lock.get("population_and_manifests") or {}).get("manifest_hashes") or {}).get(
        "candidate_order_hash")
    ref_K = (ref_lock.get("catalogue") or {}).get("K")
    checks.update({
        "registry.header_sha256": hsha == entry.get("header_sha256"),
        "header.manifest_id": header.get("manifest_id") == mid,
        "header.status_eq_registry_entry": status == entry.get("status", status),
        "header.status_allowed": status in registry_statuses(reg),
        "header.data_version_eq_registry": header.get("data_version") == reg.get("data_version"),
        "header.no_all_methods_lock": header.get("all_methods_lock_sha256") is None,
        "header.no_test_exposure_entry": header.get("test_exposure_register_entry_id") is None,
        "header.data_protocol_lock_sha256_eq_lock_self": header.get("data_protocol_lock_sha256") == ref_self,
        "header.data_protocol_lock_file_sha256_eq_lock_file": header.get("data_protocol_lock_file_sha256") == ref_file,
        "header.content_self_hash": header.get("header_content_sha256") == header_content_sha256(header),
        "registry.header_content_sha256": header.get("header_content_sha256") == entry.get(
            "header_content_sha256", header.get("header_content_sha256")),
        "header.tables_eq_registry": {x["table"]: {k: x[k] for k in ("row_count", "table_content_sha256", "file_sha256")}
                                      for x in header.get("tables", [])} == {
            t: {k: v[k] for k in ("row_count", "table_content_sha256", "file_sha256")} for t, v in entry["tables"].items()},
        "header.builder_code_sha256_eq_lock": bool(ref_code) and header.get("builder_code_sha256") == ref_code,
        "header.schema_self_sha256_eq_lock": bool(ref_schema) and header.get("feature_schema_self_sha256") == ref_schema,
        "header.dataset_version_eq_lock": bool(ref_dataset_version) and header.get("dataset_version_hash") == ref_dataset_version,
        "header.candidate_order_eq_lock": bool(ref_candidate_order) and header.get("candidate_order_hash") == ref_candidate_order,
        "header.catalogue_K_eq_lock": ref_K is not None and header.get("catalogue_K") == ref_K,
    })
    for k in ("ordered_decision_id_sha256", "eval_ordered_decision_id_sha256", "rankable_ordered_decision_id_sha256",
              "population_sha256", "n_users", "n_decisions", "n_eval_rows", "n_rankable_rows"):
        if k in entry:
            checks[f"header.{k}_eq_registry"] = header.get(k) == entry.get(k)
    try:
        checks["registry.content_sha256"] = manifest_content_sha256(header.get("tables", []), tables) == \
            entry.get("content_sha256")
    except ManifestVerificationError:
        checks["registry.content_sha256"] = False
    mk_path = root / str(entry.get("marker", ""))
    if entry.get("marker") and mk_path.is_file():
        mk = strict_json(mk_path).get("verified_sha256", {})
        base = mk_path.parent.parent
        checks["ok_marker_matches_disk"] = bool(mk) and all(
            (base / rel).is_file() and sha256_file(base / rel) == s for rel, s in mk.items())
    else:
        checks["ok_marker_matches_disk"] = False
    t_head = time.time()
    _fail_if_bad(mid, checks)
    cols, canon, rows = _verify_tables_and_rows(mid, kind, header, files, tables, entry, checks, "registry")
    t_end = time.time()
    say(f"[v4int.verify] {mid}: {len(checks)} registry checks passed ({t_end - t0:.1f} s)")
    rec = {"path": str(registry_path), "file_sha256": reg_file, "self_sha256": reg_self,
           "data_version": reg.get("data_version"), "run_label": reg.get("run_label"), "root": str(root),
           "entry_content_sha256": entry.get("content_sha256"), "entry_header_sha256": entry.get("header_sha256")}
    return VerifiedManifest(manifest_id=mid, kind=kind, header=header, header_file_sha256=hsha, tables=tables,
                            columns=cols, canonical_decision_id=np.ascontiguousarray(canon, dtype=np.int64),
                            canonical_rows=np.ascontiguousarray(rows, dtype=np.int64), lock_self_sha256=lself,
                            lock_file_sha256=lfile, transform_state_self_sha256=ts_self,
                            transform_state_file_sha256=ts_file, checks=checks,
                            timings={"lock_state_registry": round(t_lock - t0, 2),
                                     "files_and_header": round(t_head - t_lock, 2),
                                     "tables": round(t_end - t_head, 2), "total": round(t_end - t0, 2)},
                            registry=rec)


def _fail_if_bad(mid: str, checks: Mapping) -> None:
    bad = sorted(k for k, v in checks.items() if v is not True)
    if bad:
        raise ManifestVerificationError(f"{mid}: verification failed: {bad}")


__all__ = ["DataPaths", "VerifiedManifest", "ManifestVerificationError", "TestManifestRefused", "verify_manifest",
           "verify_lock", "verify_transform_state", "lock_self_sha256", "header_content_sha256",
           "refuse_test_manifest", "manifest_kind", "TABLES", "strict_json", "verify_manifest_by_registry",
           "verify_registry", "registry_self_sha256", "registry_statuses", "manifest_content_sha256",
           "_b1_superseded_exception"]

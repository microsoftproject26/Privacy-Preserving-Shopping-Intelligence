"""INT-01 test kit: locked manifest environments (tables -> pre-lock header -> MANIFEST_HASHES -> lock -> DATA-04R1-style
header re-issue -> R1 list -> .OK markers), real-row manifests over the frozen v1 bucket 0, evidence rows.

The manifest chain mirrors production exactly (v4data.manifests writer rules, reissue_headers_r1 rule); the lock is a
TEST lock with the production lock's structure, written into the test's own directory (the real contracts/ lock is
never touched). Store-level tests read the frozen v1 dataset read-only (data_testkit.REAL_DS, bucket 0; TRAIN and
VALIDATION splits only) — the synthetic frozen-layout dataset (data_testkit.write_dataset) carries no F4 feature
columns, so it drives the manifest-format tests through the real v4data ManifestBuilder instead.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE / "tests_data") not in sys.path:
    sys.path.append(str(CODE / "tests_data"))

import v4train  # noqa: E402,F401  (ppsi_env first)

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from data_testkit import REAL_DS, assert_raises, nc  # noqa: E402,F401  (read-only import)
from v4data.common import (K, SCHEMA_SELF_SHA256, TRAIN_START_US, VALIDATION_SPLITS, VERSION_HASH,  # noqa: E402
                           ordered_ids_sha256, sha256_file, table_content_sha256, utc_now)
from v4data.manifests import FIELD_TYPES  # noqa: E402
from v4int.verify import DataPaths, header_content_sha256, lock_self_sha256  # noqa: E402

REAL_LOCK = V4 / "contracts" / "DATA_PROTOCOL_LOCK.json"
CAND = "ab606d5e3728c993d31048f6d613838658d3a5ce8a0d0bd023611ce0ff027476"
_NP = {"int64": np.int64, "uint64": np.uint64, "int32": np.int32, "int8": np.int8}


def real_lock() -> dict:
    return json.loads(REAL_LOCK.read_text(encoding="utf-8"))


def transform_state_source() -> Path:
    """A byte copy of the production TRANSFORM_STATE (its sha256 is the lock's transform_state_file_sha256)."""
    want = real_lock()["features_and_transforms"]["transform_state_file_sha256"]
    for p in (V4 / "data_v2" / "TRANSFORM_STATE_CLEAN_V2.json", V4 / "data_v2" / "server" / "TRANSFORM_STATE_CLEAN_V2.json"):
        if p.is_file() and sha256_file(p) == want:
            return p
    raise FileNotFoundError("no byte copy of the locked TRANSFORM_STATE_CLEAN_V2.json found")


def columns_of(tbl: Mapping) -> list:
    out = []
    for name, v in tbl.items():
        dt = FIELD_TYPES[name][0]
        out.append((name, dt, np.asarray(v, dtype=object) if dt == "string" else np.asarray(v)))
    return out


def arrow_of(cols: list) -> pa.Table:
    arrays = []
    for name, dt, v in cols:
        if dt == "string":
            arrays.append(pa.array([str(x) for x in v.tolist()], pa.string()))
        elif dt == "bool":
            arrays.append(pa.array(np.asarray(v, dtype=bool)))
        else:
            arrays.append(pa.array(np.asarray(v).astype(_NP[dt])))
    return pa.Table.from_arrays(arrays, names=[c[0] for c in cols])


def _json(p: Path, obj) -> None:
    Path(p).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def write_manifest_files(md: Path, mid: str, tables: Mapping, *, header_extra: Optional[Mapping] = None,
                         K_: int = K, cand: str = CAND) -> dict:
    """Tables + the PRE-LOCK header (builder rule). Returns the MANIFEST_HASHES entry."""
    md.mkdir(parents=True, exist_ok=True)
    tmeta = []
    for tname, tbl in tables.items():
        cols = columns_of(tbl)
        f = md / f"{mid}.{tname}.parquet"
        pq.write_table(arrow_of(cols), f)
        tmeta.append({"table": tname, "row_count": int(len(cols[0][2])), "table_content_sha256": table_content_sha256(cols),
                      "file_sha256": sha256_file(f)})
    dec = tables["decisions"]
    did = np.asarray(dec["decision_id"], dtype=np.int64)
    ev = np.asarray(dec.get("eval_mask", np.zeros(did.size, dtype=bool)), dtype=bool)
    rk = np.asarray(dec.get("rankable", np.zeros(did.size, dtype=bool)), dtype=bool)
    lock = real_lock()
    h = {"manifest_id": mid, "spec_id": "MANIFEST_SPEC_V4", "spec_version": "1.2.0", "status": "BUILT_PRE_LOCK",
         "build_utc": utc_now(), "builder_code_sha256": lock["code"]["v4data_builder_code_sha256"],
         "dataset_version_hash": VERSION_HASH, "feature_schema_version": "2.1.0",
         "feature_schema_self_sha256": SCHEMA_SELF_SHA256,
         "data_protocol_lock_sha256": "PENDING until DATA_PROTOCOL_LOCK", "catalogue_K": int(K_),
         "candidate_order_hash": cand, "parent_manifests": [], "tables": tmeta,
         "ordered_decision_id_sha256": ordered_ids_sha256(did), "eval_ordered_decision_id_sha256": ordered_ids_sha256(did[ev]),
         "rankable_ordered_decision_id_sha256": ordered_ids_sha256(did[rk]),
         "population_sha256": hashlib.sha256(f"SYN-POP|{mid}|{did.size}".encode()).hexdigest(),
         "n_users": int(len(next(iter(tables[[t for t in tables if t != "decisions"][0]].values())))),
         "n_decisions": int(did.size), "n_eval_rows": int(ev.sum()), "n_rankable_rows": int(rk.sum()),
         "previously_seen_evaluation_period": True, "all_methods_lock_sha256": None,
         "test_exposure_register_entry_id": None}
    if mid == "M01_COHORT_TRAIN" and "central_order" in tables:
        h["derived"] = {"central_canonical_order": "table central_order", "central_order_ordered_decision_id_sha256":
                        ordered_ids_sha256(np.sort(did))}
    h.update(dict(header_extra or {}))
    h["header_content_sha256"] = header_content_sha256(h)
    hp = md / f"{mid}.header.json"
    _json(hp, h)
    return {"tables": {t["table"]: {k: t[k] for k in ("row_count", "table_content_sha256", "file_sha256")} for t in tmeta},
            "header_file_sha256": sha256_file(hp), "header_content_sha256": h["header_content_sha256"],
            **{k: h[k] for k in ("ordered_decision_id_sha256", "eval_ordered_decision_id_sha256",
                                 "rankable_ordered_decision_id_sha256", "population_sha256", "n_users", "n_decisions",
                                 "n_eval_rows", "n_rankable_rows")}}


def lock_and_reissue(d2: Path, entries: Mapping, *, K_: int = K, cand: str = CAND,
                     lock_mutator: Optional[Callable] = None) -> DataPaths:
    """MANIFEST_HASHES.json -> a TEST DATA_PROTOCOL_LOCK -> headers re-issued with the lock hashes -> R1 list -> .OK."""
    d2 = Path(d2)
    md, rc, ct = d2 / "manifests", d2 / "receipts", d2 / "contracts"
    rc.mkdir(parents=True, exist_ok=True)
    ct.mkdir(parents=True, exist_ok=True)
    ts = d2 / "TRANSFORM_STATE_CLEAN_V2.json"
    if not ts.exists():
        shutil.copyfile(transform_state_source(), ts)
    mhp = rc / "MANIFEST_HASHES.json"
    _json(mhp, {"receipt": "MANIFEST_HASHES", "spec_version": "1.2.0", "manifests": dict(entries)})
    real = real_lock()
    st = json.loads(ts.read_text(encoding="utf-8"))
    lock = {"document": "DATA_PROTOCOL_LOCK", "version": "INT-01-TEST", "data_version": {"base_version_hash": VERSION_HASH},
            "splits_and_labels": real["splits_and_labels"], "catalogue": {"K": int(K_)},
            "features_and_transforms": {"schema_self_sha256": SCHEMA_SELF_SHA256,
                                        "transform_state_self_sha256": st["transform_state_sha256"],
                                        "transform_state_file_sha256": sha256_file(ts)},
            "population_and_manifests": {"manifest_hashes_file_sha256": sha256_file(mhp),
                                         "manifest_hashes": {"candidate_order_hash": cand,
                                                             "manifests": json.loads(json.dumps(dict(entries)))}},
            "code": {"v4data_builder_code_sha256": real["code"]["v4data_builder_code_sha256"]}}
    if lock_mutator is not None:
        lock_mutator(lock)
    lock["lock_self_sha256"] = lock_self_sha256(lock)
    lp = ct / "DATA_PROTOCOL_LOCK.json"
    _json(lp, lock)
    lfile = sha256_file(lp)
    for mid in entries:
        hp = md / f"{mid}.header.json"
        h = json.loads(hp.read_text(encoding="utf-8"))
        prev_file, prev_content = sha256_file(hp), h.pop("header_content_sha256")
        h["data_protocol_lock_sha256"] = lock["lock_self_sha256"]
        h["data_protocol_lock_file_sha256"] = lfile
        h["reissue"] = {"task": "INT-01 test kit", "reason": "header re-issued with the TEST lock hashes",
                        "previous_header_file_sha256": prev_file, "previous_header_content_sha256": prev_content}
        h["reissue_utc"] = utc_now()
        h["header_content_sha256"] = header_content_sha256(h)
        _json(hp, h)
    write_r1_and_markers(d2, entries)
    return DataPaths(manifest_dir=md, lock_path=lp, manifest_hashes_path=mhp,
                     header_list_path=rc / "MANIFEST_FILE_SHA256_R1.txt", transform_state_path=ts, production=False)


def write_r1_and_markers(d2: Path, entries) -> None:
    md, rc = Path(d2) / "manifests", Path(d2) / "receipts"
    files = sorted(p for p in md.iterdir() if p.is_file() and p.suffix in (".parquet", ".json"))
    (rc / "MANIFEST_FILE_SHA256_R1.txt").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files),
                                                    encoding="utf-8")
    for mid in entries:
        mine = {f"manifests/{p.name}": sha256_file(p) for p in files if p.name.startswith(mid + ".")}
        _json(md / f"{mid}.OK", {"verified_sha256": mine, "utc": utc_now()})


def write_env(root: Path, manifests: Mapping, **kw) -> DataPaths:
    """manifests: {mid: tables-dict (TABLES order)} -> a fully locked TEST environment under root/data_v2."""
    d2 = Path(root) / "data_v2"
    entries = {mid: write_manifest_files(d2 / "manifests", mid, tables) for mid, tables in manifests.items()}
    return lock_and_reissue(d2, entries, **kw)


# ------------------------------------------------------------------------------------------ real-row manifests
def real_train_tables(store, n_clients: int = 60, seed: int = 0) -> dict:
    """An M01-format TRAIN manifest over real TRAIN loss decisions of the store (a client = a v1 client_idx)."""
    ds = store.decision_set("TRAIN", "loss")
    cli = np.unique(ds.client_idx[ds.client_idx > 2])
    chosen = np.sort(np.random.default_rng(seed).choice(cli, size=min(n_clients, cli.size), replace=False))
    m = np.isin(ds.client_idx, chosen)
    did, tc, ci = ds.decision_id[m], ds.target_class[m], ds.client_idx[m]
    ordv = np.searchsorted(chosen, ci).astype(np.int64)
    o = np.lexsort((did, ordv))
    n, nc_ = int(did.size), int(chosen.size)
    dec = {"decision_id": did[o], "client_ord": ordv[o], "decision_ts_us": TRAIN_START_US + np.arange(n, dtype=np.int64),
           "target_ts_us": TRAIN_START_US + 1 + np.arange(n, dtype=np.int64), "target_class": tc[o].astype(np.int64),
           "target_status": np.full(n, "OBSERVED_IN_PARTITION", dtype=object), "old_split": np.full(n, "TRAIN", dtype=object)}
    cnt = np.bincount(ordv, minlength=nc_).astype(np.int64)
    clients = {"client_ord": np.arange(nc_, dtype=np.int64),
               "client_key_hex": np.array([hashlib.sha256(f"int01-client-{x}".encode()).hexdigest() for x in chosen],
                                          dtype=object),
               "selection_key": np.arange(1, nc_ + 1, dtype=np.uint64), "n_eligible_train_examples": cnt,
               "row_start": np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64), "row_count": cnt,
               "in_calib_prefix": np.ones(nc_, dtype=bool)}
    return {"clients": clients, "decisions": dec,
            "central_order": {"row": np.argsort(dec["decision_id"], kind="stable").astype(np.int64)}}


def real_eval_tables(vstore, n_users: int = 120, seed: int = 1, censor_every: int = 9) -> dict:
    """An M02-format evaluation manifest over real VALIDATION eval decisions (warm v1 clients), plus fabricated
    censored rows (eval_mask False; never materialised by the store, exactly like the real censored rows)."""
    parts = [(s, vstore.decision_set(s, "eval")) for s in VALIDATION_SPLITS]
    cli = np.unique(np.concatenate([p.client_idx for _, p in parts]))
    cli = cli[cli > 2]
    chosen = np.sort(np.random.default_rng(seed).choice(cli, size=min(n_users, cli.size), replace=False))
    did, tc, ci, sp = [], [], [], []
    for s, p in parts:
        m = np.isin(p.client_idx, chosen)
        did.append(p.decision_id[m]); tc.append(p.target_class[m]); ci.append(p.client_idx[m])  # noqa: E702
        sp.append(np.full(int(m.sum()), s, dtype=object))
    did, tc, ci, sp = (np.concatenate(x) for x in (did, tc, ci, sp))
    ev_ids = set(did.tolist())
    cen = sorted({int(x) + 1 for x in did[::censor_every].tolist() if int(x) + 1 not in ev_ids})
    ci_c = np.repeat(chosen[:1], len(cen))
    all_did = np.concatenate([did, np.asarray(cen, dtype=np.int64)])
    o = np.argsort(all_did, kind="stable")
    n_ev = did.size
    evm = np.concatenate([np.ones(n_ev, dtype=bool), np.zeros(len(cen), dtype=bool)])[o]
    tcl = np.concatenate([tc, np.full(len(cen), -1, dtype=np.int64)])[o].astype(np.int64)
    cia = np.concatenate([ci, ci_c])[o]
    spl = np.concatenate([sp, np.full(len(cen), VALIDATION_SPLITS[0], dtype=object)])[o]
    oov = evm & (tcl == -1)
    euk = np.searchsorted(chosen, cia).astype(np.int64)
    dec = {"decision_id": all_did[o].astype(np.int64), "eval_user_key": euk, "eval_mask": evm, "target_class": tcl,
           "target_oov": oov, "censor_class": np.where(evm, "NONE", "CEN_NO_NEXT").astype(object),
           "subperiod": np.where(spl == VALIDATION_SPLITS[0], "VAL_A_20191116_20191120",
                                 "VAL_B_20191120_20191123").astype(object),
           "old_split": spl.astype(object), "v4_split": np.full(all_did.size, "VALIDATION", dtype=object),
           "oov_cause": np.where(~evm, "NOT_APPLICABLE", np.where(oov, "BELOW_SUPPORT", "NONE")).astype(object),
           "rankable": evm & (tcl >= 0)}
    users = {"eval_user_key": np.arange(chosen.size, dtype=np.int64),
             "n_validation_decisions": np.bincount(euk, minlength=chosen.size).astype(np.int64)}
    return {"users": users, "decisions": dec}


# ------------------------------------------------------------------------------------------ batches
def batches_equal(a: Mapping, b: Mapping) -> tuple:
    import torch
    if set(a) != set(b):
        return False, f"key sets differ: {sorted(set(a) ^ set(b))}"
    for k in a:
        x, y = a[k], b[k]
        if torch.is_tensor(x) or torch.is_tensor(y):
            if not (torch.is_tensor(x) and torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape
                    and torch.equal(x, y)):
                return False, f"tensor {k} differs"
        elif x != y:
            return False, f"field {k} differs"
    return True, "equal"


def evidence_rows() -> list:
    """8 valid cohort evidence rows (2 families x 2 seeds x {mrr@20, ndcg@10}, PRACTICAL_BEST) for select_family."""
    base = {"data_protocol_hash": "D", "feature_contract_hash": "F", "split": "VALIDATION", "label_policy_hash": "L",
            "population_hash": "P", "query_order_hash": "Q", "candidate_order_hash": "C", "objective_id": "O",
            "precision_id": "X", "evaluator_code_sha256": "E", "eval_score_block": 158486, "eval_batch": 1024,
            "eval_device_class": "A16-FP32-DET", "eval_stream": "E2E_ROWS", "aggregation": "macro",
            "target_denominator": "E2E", "method": "C", "selection_rule": "PRACTICAL_BEST", "value_status": "OK",
            "row_status": "CENTRAL_CONTEST", "comparison_status": "SAME_CONDITIONS_BLOCK_CANDIDATE",
            "c_matched_eligible": True, "status": "MEASURED", "efe": 3.0,
            "run_label": "PRODUCTION"}          # TRAIN-01R2 (G1 R3): every row a TRAIN-01R2 trainer would emit carries one
    vals = {("GRU", 2026): 0.170, ("GRU", 2027): 0.171, ("SASREC", 2026): 0.160, ("SASREC", 2027): 0.161}
    rows = []
    for (fam, seed), v in vals.items():
        for metric, k, val in (("mrr", 20, v), ("ndcg", 10, v / 2)):
            rows.append(dict(base, family=fam, seed=seed, metric=metric, k=k, value=val,
                             checkpoint_id=f"{fam}:{seed}:best"))
    return rows


__all__ = ["REAL_DS", "nc", "assert_raises", "write_env", "write_manifest_files", "lock_and_reissue",
           "write_r1_and_markers", "real_train_tables", "real_eval_tables", "batches_equal", "evidence_rows",
           "transform_state_source", "columns_of", "arrow_of", "CAND"]

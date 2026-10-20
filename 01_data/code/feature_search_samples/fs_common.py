"""v4fs_data.fs_common (owner: Data engineer, FS-01) - constants, the rank bands of the final Fable ruling, the FS
selection, hashing rules and the manifest emitter shared by fs_build and fs_finalize.

Nothing here reads the dataset. The LOCKED DATA-04 builder (v4data.manifests.ManifestBuilder) is only called, never
edited: tables come from its _train_manifest / _eval_manifest, rows are checked by its _check_no_raw_ids, typed by
its _columns / _arrow, hashed by v4data.common.table_content_sha256 and re-read by v4data.manifests.reread_columns.
"""
from __future__ import annotations

import io
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent

from census_v2 import core as C2  # noqa: E402
from v4data.common import (atomic_write_bytes, atomic_write_json, builder_code_sha256, canonical_json_bytes,  # noqa: E402
                           code_tree_sha256, digest_rows, json_bytes_strict, population_sha256, self_hash,
                           sha256_bytes, sha256_file, table_content_sha256, write_marker)
from v4data.manifests import BuildBlocked, refuse_test_manifest, reread_columns  # noqa: E402

TASK = "FS-01"
FS_STATUS = "FEATURE_SEARCH_SCREEN"
DATA_VERSION = "FS_SCREEN_5_100_v1"
FS_N = 32_768
FS_TRAIN_ID = f"FS_TRAIN_5_100_{FS_N}"
FS_EVAL_ID = f"FS_VALIDATION_5_100_{FS_N}"
POOL_N = 1_446_986                       # CENSUS-02 stratum C (5 <= n <= 100, catalogue A)
POOL_EXAMPLES = 28_623_386
POOL_SHA_V2 = "e2a082cda5e3b246f1ea85cf1e78368c185e003ce1013572d07bd8ff471fbc70"
# CENSUS-02 prefix hashes (selection_key_v2 rule) - reproduced at build as a check of the pool order
PREFIX_SHA_V2 = {32768: "9be38de146f5069a88d57c867dfa021cf1c67f2d67ae1e58df63f7c6ac9c1f02",
                 65536: "565157638fa27126f2780e5f4312f777243dbb4c40e3b97e65170e13b6b8571e",
                 131072: "563e76637a384c2985fff5e272d4290f4f3e52c99f57eebec4c20cde0628b8b3",
                 262144: "e4cf6887d77ae2c5a12062bcb0438814598a598398b1badcfaf5cecb9e4eb70d"}
# rank bands over the ascending selection_key_v2 order of the pool (rank 0 = smallest key); final Fable ruling as
# relayed by the Operations Director (FS = the top 32,768 = the original FS-01 packet), plus the packet's own
# [0, 294,912) region
BANDS = {"MAIN": (0, 131_072), "CALIB": (0, 32_768), "TIEBREAK": (0, 1_348_682),
         "U_NEW_VAL": (1_348_682, 1_381_450), "U_NEW_TEST": (1_381_450, 1_414_218),
         "PACKET_REGION_0_294912": (0, 294_912), "FS": (1_414_218, 1_446_986)}
DISJOINT_FROM = ("MAIN", "TIEBREAK", "U_NEW_VAL", "U_NEW_TEST", "PACKET_REGION_0_294912")
LOCK_FILE_SHA256 = "ebf4e86f6beb82e98283108a2dad9082727e8559b81301cd46f35f60c79fd7c0"
LOCK_SELF_SHA256 = "da51d271e97d3a681227a7063fd35b429797e52ee25586ba39843a37b7c839e1"
DATA04_BUILDER_CODE_SHA256 = "443254e9ed95e41a48076f91643c83fbc77355dc80bb6b111d8432703d3e34a5"
TRANSFORM_STATE_SELF_SHA256 = "2ff0ff4ec1d27d254972c2f212e81f5fee6277dc9cced31a6a5baf5b4b002727"
OLD_M05_POP_SHA = "3d680aa7222fb5c9246d03fe6088464c78095d90a344e3453acb68e98d53a6f4"
SELECTION_REFUSAL = ("FEATURE_SEARCH_SCREEN: owner-directed screening data (DECISION_LOG 39); every result row is "
                     "refused by architecture selection, the T1 contest, the tie-break and calibration; TEST is never "
                     "touched")


# ------------------------------------------------------------------------------------------------ utilities
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    return o


def with_self_hash(obj: dict, field: str) -> dict:
    d = jsonable(dict(obj))
    d.pop(field, None)
    d[field] = self_hash(d, field)
    return d


def verify_self_hash(obj: dict, field: str) -> bool:
    return obj.get(field) == self_hash(obj, field)


def header_content_sha256(h: dict) -> str:
    """The builder's rule (v4data.manifests._header; DATA-04R1; v4int.verify.header_content_sha256): canonical JSON of
    the header without build_utc, reissue_utc and header_content_sha256, tables reduced to (table, row_count,
    table_content_sha256)."""
    stable = {k: v for k, v in h.items() if k not in ("build_utc", "tables", "header_content_sha256", "reissue_utc")}
    stable["tables"] = [{k: t[k] for k in ("table", "row_count", "table_content_sha256")} for t in h["tables"]]
    return sha256_bytes(canonical_json_bytes(stable))


def manifest_content_sha256(tables_meta: list) -> str:
    """Registry `content_sha256`: sha256 of the canonical JSON list of (table, row_count, table_content_sha256) in the
    header's table order - the writer-independent content identity of all of a manifest's tables."""
    return sha256_bytes(canonical_json_bytes([{k: t[k] for k in ("table", "row_count", "table_content_sha256")}
                                              for t in tables_meta]))


BUILD_FILES = ("__init__.py", "fs_common.py", "fs_build.py")


def fs_code_hashes() -> dict:
    files = [HERE / n for n in BUILD_FILES]
    return {"fs_code_sha256": code_tree_sha256(files, HERE),
            "fs_code_sha256_rule": "v4data.common.code_tree_sha256 over the build files code/v4fs_data/"
                                   "{__init__,fs_common,fs_build}.py (fs_finalize.py is hashed in its own receipt)",
            "fs_code_files_sha256": {p.name: sha256_file(p) for p in files},
            "census_v2_core_sha256": sha256_file(CODE / "census_v2" / "core.py"),
            "v4data_builder_code_sha256": builder_code_sha256()}


def lock_info(v4_root: Path = V4) -> dict:
    """DATA_PROTOCOL_LOCK verification by the lock's own rule (as in tests_data/reissue_headers_r1.py)."""
    import hashlib
    p = Path(v4_root) / "contracts" / "DATA_PROTOCOL_LOCK.json"
    fsha = sha256_file(p)
    d = json.loads(p.read_text(encoding="utf-8"))
    x = {k: v for k, v in d.items() if k not in ("lock_self_sha256", "lock_self_sha256_rule")}
    rec = hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    ok = fsha == LOCK_FILE_SHA256 and d.get("lock_self_sha256") == LOCK_SELF_SHA256 == rec
    if not ok:
        raise BuildBlocked(f"DATA_PROTOCOL_LOCK verification failed: file {fsha}, recorded {d.get('lock_self_sha256')}, "
                           f"recomputed {rec}")
    return {"path": "contracts/DATA_PROTOCOL_LOCK.json", "file_sha256": fsha, "lock_self_sha256": rec, "verified": True}


def host_info() -> dict:
    return {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "pyarrow": pa.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"), "argv": sys.argv[1:]}


# ------------------------------------------------------------------------------------------------ selection
def select_fs(u_nelig: np.ndarray, u_dig: np.ndarray, *, lo: int = C2.C_LO, hi: int = C2.C_HI, n_fs: int = FS_N,
              bands: dict | None = None, disjoint_from=DISJOINT_FROM) -> dict:
    """The FS sample: the n_fs users of the pool (lo <= n_eligible_TRAIN_examples <= hi, the CENSUS-02 definition)
    with the LARGEST selection_key_v2 (census_v2.core.selection_keys_v2 over the C1.2 digest), i.e. the top ranks of
    the ascending key order. Disjointness is checked on user indices and on rank ranges for every named band."""
    bands = BANDS if bands is None else bands
    pool = np.flatnonzero((u_nelig >= lo) & (u_nelig <= hi))
    keys = C2.selection_keys_v2(digest_rows(u_dig[pool]))
    if np.unique(keys).size != keys.size:
        raise BuildBlocked("two pool users share a selection_key_v2: BLOCK and escalate")
    o = np.argsort(keys, kind="stable")
    ps, ks = pool[o], keys[o]
    P = int(ps.size)
    if n_fs > P:
        raise BuildBlocked(f"the pool has {P} users, fewer than the FS size {n_fs}")
    a, b = P - n_fs, P
    fs, fk = ps[a:b], ks[a:b]
    disj = {}
    for name in disjoint_from:
        x, y = bands[name]
        x, y = min(x, P), min(y, P)
        ov = int(np.intersect1d(fs, ps[x:y]).size)
        disj[name] = {"ranks": [x, y], "overlap_users": ov, "rank_ranges_disjoint": bool(b <= x or y <= a),
                      "disjoint": bool(ov == 0 and (b <= x or y <= a))}
    band_hashes = {}
    for name, (x, y) in bands.items():
        if y <= P:
            band_hashes[name] = {"ranks": [x, y], "n_users": y - x,
                                 "population_sha256_selection_key_v2": population_sha256(ks[x:y])}
    return {"pool_sorted": ps, "keys_sorted": ks, "fs": fs, "fs_keys_v2": fk, "fs_ranks": [a, b], "pool_size": P,
            "pool_examples": int(u_nelig[pool].sum()), "pool_population_sha256_v2": population_sha256(ks),
            "fs_population_sha256_v2": population_sha256(fk), "disjointness": disj,
            "all_disjoint": all(v["disjoint"] for v in disj.values()), "band_hashes": band_hashes,
            "fs_key_v2_min_hex": format(int(fk[0]), "016x"), "fs_key_v2_max_hex": format(int(fk[-1]), "016x"),
            "key_v2_below_fs_max_hex": format(int(ks[a - 1]), "016x") if a > 0 else None}


def keys_v2_of_hex(client_key_hex) -> np.ndarray:
    """selection_key_v2 recomputed from client_key_hex (the C1.2 digest in hex)."""
    return C2.selection_keys_v2([bytes.fromhex(h) for h in client_key_hex])


# ------------------------------------------------------------------------------------------------ emitter
def encode_tables(b, mid: str, m: dict) -> list:
    """Every table of a builder manifest dict -> (name, content sha256, parquet bytes, rows); the in-memory Parquet round
    trip must reproduce the content hash (the builder's own write-time check)."""
    refuse_test_manifest(mid)
    out = []
    for tname, tbl in m["tables"].items():
        b._check_no_raw_ids(mid, tname, tbl)
        content = table_content_sha256(b._columns(tbl))
        buf = io.BytesIO()
        pq.write_table(b._arrow(tbl), buf, compression="zstd", compression_level=9)
        data = buf.getvalue()
        back = pq.read_table(pa.BufferReader(data))
        if table_content_sha256(reread_columns(back)) != content:
            raise BuildBlocked(f"{mid}.{tname}: content hash changed after the in-memory Parquet round trip")
        del back
        out.append((tname, content, data, int(len(next(iter(tbl.values()))))))
    return out


def tables_meta_of(enc: list) -> list:
    return [{"table": t, "row_count": n, "table_content_sha256": c, "file_sha256": sha256_bytes(d)} for t, c, d, n in enc]


def hashes_entry(header: dict, tables_meta: list, header_file_sha256: str | None) -> dict:
    """The MANIFEST_HASHES.json entry shape of the DATA-04 builder (write())."""
    return {"tables": {x["table"]: {"row_count": x["row_count"], "table_content_sha256": x["table_content_sha256"],
                                    "file_sha256": x["file_sha256"]} for x in tables_meta},
            "header_file_sha256": header_file_sha256, "header_content_sha256": header["header_content_sha256"],
            **{k: header[k] for k in ("ordered_decision_id_sha256", "eval_ordered_decision_id_sha256",
                                      "rankable_ordered_decision_id_sha256", "population_sha256", "n_users",
                                      "n_decisions", "n_eval_rows", "n_rankable_rows")}}


def write_manifest(out_root: Path, mid: str, enc: list, header: dict, out_dir: str = "manifests") -> dict:
    """Parquet tables, then the header, then (only after every file is re-verified on disk) the .OK marker."""
    refuse_test_manifest(mid)
    files = {}
    for tname, content, data, n in enc:
        rel = f"{out_dir}/{mid}.{tname}.parquet"
        p = atomic_write_bytes(out_root, rel, data)
        fsha = sha256_file(p)
        if fsha != sha256_bytes(data):
            raise BuildBlocked(f"{rel}: file bytes differ from the encoded bytes")
        if table_content_sha256(reread_columns(pq.ParquetFile(p).read())) != content:
            raise BuildBlocked(f"{rel}: content hash changed after the disk round trip")
        files[rel] = fsha
    hb = json_bytes_strict(header)
    back = json.loads(hb, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    if back != header or header_content_sha256(back) != header["header_content_sha256"]:
        raise BuildBlocked(f"{mid}: header does not round-trip or its content hash does not verify")
    hrel = f"{out_dir}/{mid}.header.json"
    hp = atomic_write_bytes(out_root, hrel, hb)
    files[hrel] = sha256_file(hp)
    for rel, fsha in files.items():
        if sha256_file(Path(out_root) / rel) != fsha:
            raise BuildBlocked(f"{rel}: file changed before the marker was written")
    write_marker(out_root, f"{out_dir}/{mid}.OK", files)
    return {"files": files, "header_file_sha256": files[hrel]}


def compare_entries(got: dict, ref: dict, *, informative=("file_sha256", "header_file_sha256")) -> dict:
    """Field-by-field comparison of two MANIFEST_HASHES-shaped entries. Parquet file bytes depend on the writer
    version, so file hashes are reported but are not part of the verdict (MANIFEST_SPEC identity = content)."""
    rows, ok = {}, True
    for t in sorted(set(got.get("tables", {})) | set(ref.get("tables", {}))):
        g, r = got.get("tables", {}).get(t, {}), ref.get("tables", {}).get(t, {})
        for k in ("row_count", "table_content_sha256", "file_sha256"):
            eq = g.get(k) == r.get(k) and g.get(k) is not None
            rows[f"tables.{t}.{k}"] = {"got": g.get(k), "ref": r.get(k), "equal": eq,
                                       "in_verdict": k not in informative}
            ok &= eq or k in informative
    for k in ("header_content_sha256", "ordered_decision_id_sha256", "eval_ordered_decision_id_sha256",
              "rankable_ordered_decision_id_sha256", "population_sha256", "n_users", "n_decisions", "n_eval_rows",
              "n_rankable_rows", "header_file_sha256"):
        if k not in ref and k not in got:
            continue
        eq = got.get(k) == ref.get(k) and got.get(k) is not None
        rows[k] = {"got": got.get(k), "ref": ref.get(k), "equal": eq, "in_verdict": k not in informative}
        ok &= eq or k in informative
    return {"all_equal": bool(ok), "fields": rows}


def write_json_self_hashed(out_root: Path, rel: str, obj: dict, field: str = "receipt_sha256") -> tuple[Path, dict]:
    d = with_self_hash(obj, field)
    p = atomic_write_bytes(out_root, rel, json_bytes_strict(d))
    return p, d


__all__ = [n for n in dir() if not n.startswith("_")] + ["atomic_write_json"]

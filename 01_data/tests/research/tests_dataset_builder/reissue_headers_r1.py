"""DATA-04R1: re-issue every pre-TEST manifest header with the DATA_PROTOCOL_LOCK hashes (header-only).

Run ON THE SERVER with -B, CUDA_VISIBLE_DEVICES="". Lives in tests_data/ on purpose: v4data/ (the builder, whose
builder_code_sha256 is recorded in every header) is not touched.

For each manifest:
  1. re-read every table, recompute table_content_sha256 and compare with MANIFEST_HASHES.json and the header;
     parquet file sha256 and population_sha256 must be unchanged;
  2. build the new header = old header + data_protocol_lock_sha256 (the lock_self_sha256) +
     data_protocol_lock_file_sha256 + a reissue block; header_content_sha256 recomputed with the builder's rule;
  3. serialise strictly, validate the round trip, write atomically, verify the written bytes;
  4. only then replace the .OK marker (atomic) naming the unchanged tables and the new header.
MANIFEST_HASHES.json and MANIFEST_FILE_SHA256.txt (build receipts) are left untouched; the current list is written
as receipts/MANIFEST_FILE_SHA256_R1.txt. Receipt: receipts/DATA04R1_HEADERS_REISSUED.json (self-hash).

Usage: python -B reissue_headers_r1.py --date-start "<date -u>"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

import pyarrow.parquet as pq  # noqa: E402

from v4data.common import (V4, atomic_write_bytes, atomic_write_json, builder_code_sha256, canonical_json_bytes,  # noqa: E402
                           data_v2_root, json_bytes_strict, self_hash, sha256_bytes, sha256_file, table_content_sha256,
                           utc_now, write_marker)
from v4data.manifests import PRE_LOCK_IDS, reread_columns  # noqa: E402

LOCK_FILE_SHA256 = "ebf4e86f6beb82e98283108a2dad9082727e8559b81301cd46f35f60c79fd7c0"
LOCK_SELF_SHA256 = "da51d271e97d3a681227a7063fd35b429797e52ee25586ba39843a37b7c839e1"


def lock_check() -> dict:
    p = V4 / "contracts" / "DATA_PROTOCOL_LOCK.json"
    fsha = sha256_file(p)
    d = json.loads(p.read_text(encoding="utf-8"))
    x = {k: v for k, v in d.items() if k not in ("lock_self_sha256", "lock_self_sha256_rule")}
    recomputed = hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    ok = fsha == LOCK_FILE_SHA256 and d.get("lock_self_sha256") == LOCK_SELF_SHA256 == recomputed
    if not ok:
        raise SystemExit(f"lock verification failed: file {fsha}, recorded {d.get('lock_self_sha256')}, recomputed {recomputed}")
    return {"path": str(p), "file_sha256": fsha, "lock_self_sha256": recomputed, "verified": True}


def header_content_sha256(h: dict) -> str:
    """The builder's rule (manifests._header): everything except build_utc, the table file hashes, the content hash
    itself and the reissue time."""
    stable = {k: v for k, v in h.items() if k not in ("build_utc", "tables", "header_content_sha256", "reissue_utc")}
    stable["tables"] = [{k: t[k] for k in ("table", "row_count", "table_content_sha256")} for t in h["tables"]]
    return sha256_bytes(canonical_json_bytes(stable))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-start", required=True)
    a = ap.parse_args()
    out = data_v2_root()
    mdir = out / "manifests"
    lock = lock_check()
    mh = json.loads((out / "receipts" / "MANIFEST_HASHES.json").read_text())
    listed = {}
    for line in (out / "receipts" / "MANIFEST_FILE_SHA256.txt").read_text().splitlines():
        h, name = line.split(None, 1)
        listed[name.lstrip("*")] = h
    code_now = builder_code_sha256()
    per = {}
    all_ok = True
    for mid in PRE_LOCK_IDS:
        hp, mp = mdir / f"{mid}.header.json", mdir / f"{mid}.OK"
        old_bytes = hp.read_bytes()
        old = json.loads(old_bytes)
        old_file_sha = sha256_bytes(old_bytes)
        r = {"old_header_file_sha256": old_file_sha, "old_header_content_sha256": old["header_content_sha256"],
             "old_marker_sha256": sha256_file(mp), "tables": {}}
        # 1. tables unchanged
        ok = old_file_sha == listed[f"{mid}.header.json"] == mh["manifests"][mid]["header_file_sha256"]
        ok &= old["builder_code_sha256"] == code_now
        ok &= old["header_content_sha256"] == header_content_sha256(old)            # builder rule reproduced
        verified_files = {}
        for t in old["tables"]:
            rel = f"manifests/{mid}.{t['table']}.parquet"
            fsha = sha256_file(out / rel)
            content = table_content_sha256(reread_columns(pq.ParquetFile(out / rel).read()))
            rec = mh["manifests"][mid]["tables"][t["table"]]
            same = (content == rec["table_content_sha256"] == t["table_content_sha256"]
                    and fsha == rec["file_sha256"] == t["file_sha256"] == listed[f"{mid}.{t['table']}.parquet"])
            r["tables"][t["table"]] = {"table_content_sha256": content, "file_sha256": fsha, "unchanged": same}
            ok &= same
            verified_files[rel] = fsha
        ok &= old["population_sha256"] == mh["manifests"][mid]["population_sha256"]
        r["population_sha256"] = old["population_sha256"]
        if not ok:
            raise SystemExit(f"{mid}: content verification failed before reissue: {r}")
        # 2. new header
        new = dict(old)
        new.pop("header_content_sha256")
        new["data_protocol_lock_sha256"] = LOCK_SELF_SHA256
        new["data_protocol_lock_file_sha256"] = LOCK_FILE_SHA256
        new["reissue"] = {"task": "DATA-04R1", "reason": "header re-issued with the DATA_PROTOCOL_LOCK hashes; tables "
                                                          "unchanged", "previous_header_file_sha256": old_file_sha,
                          "previous_header_content_sha256": old["header_content_sha256"]}
        new["reissue_utc"] = utc_now()
        new["header_content_sha256"] = header_content_sha256(new)
        # 3. strict serialisation, validation, atomic write
        data = json_bytes_strict(new)
        back = json.loads(data, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        valid = (back == new and back["tables"] == old["tables"] and back["population_sha256"] == old["population_sha256"]
                 and back["data_protocol_lock_sha256"] == LOCK_SELF_SHA256
                 and header_content_sha256(back) == back["header_content_sha256"]
                 and all(back[k] == old[k] for k in old if k not in ("header_content_sha256", "data_protocol_lock_sha256")))
        if not valid:
            raise SystemExit(f"{mid}: new header failed validation; nothing written")
        atomic_write_bytes(out, f"manifests/{mid}.header.json", data)
        new_file_sha = sha256_file(hp)
        if new_file_sha != sha256_bytes(data):
            raise SystemExit(f"{mid}: written header differs from the validated bytes")
        # 4. marker last
        verified_files[f"manifests/{mid}.header.json"] = new_file_sha
        write_marker(out, f"manifests/{mid}.OK", verified_files)
        r.update({"new_header_file_sha256": new_file_sha, "new_header_content_sha256": new["header_content_sha256"],
                  "new_marker_sha256": sha256_file(mp), "header_validated_before_marker": True,
                  "content_unchanged": all(x["unchanged"] for x in r["tables"].values())})
        all_ok &= r["content_unchanged"]
        per[mid] = r
    lines = [f"{sha256_file(p)}  {p.relative_to(mdir).as_posix()}" for p in sorted(mdir.rglob("*")) if p.is_file()]
    atomic_write_bytes(out, "receipts/MANIFEST_FILE_SHA256_R1.txt", ("\n".join(lines) + "\n").encode())
    rec = {"receipt": "DATA04R1_HEADERS_REISSUED", "task": "DATA-04R1", "date_u_start": a.date_start,
           "written_utc": utc_now(), "lock": lock, "builder_code_sha256_unchanged": code_now,
           "per_manifest": per, "all_content_unchanged": all_ok,
           "unchanged_build_receipts": {"MANIFEST_HASHES.json": sha256_file(out / "receipts" / "MANIFEST_HASHES.json"),
                                        "MANIFEST_FILE_SHA256.txt": sha256_file(out / "receipts" / "MANIFEST_FILE_SHA256.txt")},
           "current_file_list": {"path": "receipts/MANIFEST_FILE_SHA256_R1.txt",
                                 "sha256": sha256_file(out / "receipts" / "MANIFEST_FILE_SHA256_R1.txt"),
                                 "n_files": len(lines)},
           "header_content_sha256_rule": "sha256 of canonical JSON of the header without build_utc, reissue_utc and "
                                         "header_content_sha256, with tables reduced to (table, row_count, "
                                         "table_content_sha256) - the builder's rule"}
    rec["receipt_sha256"] = self_hash(rec, "receipt_sha256")
    atomic_write_json(out, "receipts/DATA04R1_HEADERS_REISSUED.json", rec)
    print(json.dumps({"all_content_unchanged": all_ok, "receipt_sha256": rec["receipt_sha256"],
                      "new_headers": {m: v["new_header_file_sha256"][:16] for m, v in per.items()}}))
    return 0 if all_ok else 5


if __name__ == "__main__":
    sys.exit(main())

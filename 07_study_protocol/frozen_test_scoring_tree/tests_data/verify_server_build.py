"""Final DATA-04 verification of a built data_v2 tree, run ON THE SERVER after the build (Operations Director
directive). Read-only except for one output: data_v2/receipts/DATA04_SERVER_FINAL_RECEIPT.json (strict JSON with a
self-hash). Run with -B / PYTHONDONTWRITEBYTECODE=1, CUDA_VISIBLE_DEVICES="".

Checks: the v4data code that built the manifests equals the code present now (builder_code_sha256 in every header
and in TRANSFORM_STATE) and contains the safe_path fix; every manifest re-read and content-hashed; cohort key hashes
recomputed from the written clients tables; the post-build manifest sha256 list; MANIFEST_HASHES vs headers;
TRANSFORM_STATE self-hash and v1 input hashes; every v1 file the build or tests read vs FILES.json; the dataset
version_hash formula; the schema self-hash; the snapshot; the server test receipt.

Usage: python -B verify_server_build.py --tests-receipt <TESTS_server.json> --date-start "<date -u>" --date-end "<date -u>"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from v4data.build_all import snapshot_check, verify_written  # noqa: E402
from v4data.common import (SCHEMA_SELF_SHA256, V4DATA, VERSION_HASH, atomic_write_json, builder_code_sha256,  # noqa: E402
                           data_v2_root, dataset_root, load_schema, population_sha256, self_hash, sha256_file,
                           utc_now)
from v4data.manifests import PRE_LOCK_IDS  # noqa: E402
from v4data.transform_state import V1_INPUTS, load_state  # noqa: E402

COHORT_KEYS = {"M01_COHORT_TRAIN": "54e1d3c2539fd4cd7fe22d2b8ef91bed1a786d69be27867406bb19141ac48184",
               "M05_CALIB_TRAIN": "3d680aa7222fb5c9246d03fe6088464c78095d90a344e3453acb68e98d53a6f4"}
V1_PREFIXES = ("decisions/", "model_ready/", "mappings/", "splits/", "decisions_index/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-receipt", required=True)
    ap.add_argument("--date-start", required=True)
    ap.add_argument("--date-end", required=True)
    a = ap.parse_args()
    out, ds = data_v2_root(), dataset_root()
    mdir = out / "manifests"
    c: dict = {}
    ev: dict = {}

    # ---- code identity at build time --------------------------------------------------------------
    code_now = builder_code_sha256()
    per_file = {p.relative_to(V4DATA).as_posix(): sha256_file(p) for p in sorted(V4DATA.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts}
    headers = {m: json.loads((mdir / f"{m}.header.json").read_text()) for m in PRE_LOCK_IDS}
    state = load_state(out / "TRANSFORM_STATE_CLEAN_V2.json")               # self-hash + frozen-schema values
    hdr_codes = sorted({h["builder_code_sha256"] for h in headers.values()})
    c["code_that_built_equals_code_now"] = hdr_codes == [code_now] and state["builder_code_sha256"] == code_now
    c["safe_path_fix_present_in_built_code"] = "platform-dependent output path refused" in (V4DATA / "common.py").read_text()
    c["no_pycache_in_v4data"] = not any("__pycache__" in p.parts for p in V4DATA.rglob("*"))
    ev["v4data_builder_code_sha256"] = code_now
    ev["v4data_file_sha256"] = per_file

    # ---- manifests -------------------------------------------------------------------------------
    ver = verify_written(out)
    c["every_manifest_reread_and_content_verified"] = ver["all_verified"]
    ev["manifest_reread"] = ver["per_manifest"]
    for mid, want in COHORT_KEYS.items():
        keys = pq.ParquetFile(mdir / f"{mid}.clients.parquet").read(columns=["selection_key"])["selection_key"]
        rec = population_sha256(np.sort(keys.to_numpy()))
        c[f"{mid}_cohort_key_hash"] = rec == headers[mid]["population_sha256"] == want
        ev[f"{mid}_population_sha256"] = rec
    listed = {}
    for line in (out / "receipts" / "MANIFEST_FILE_SHA256.txt").read_text().splitlines():
        h, name = line.split(None, 1)
        listed[name.lstrip("*")] = h
    now = {name: sha256_file(mdir / name) for name in listed}
    on_disk = sorted(p.relative_to(mdir).as_posix() for p in mdir.rglob("*") if p.is_file())
    c["manifest_sha256_list_matches_disk"] = listed == now and sorted(listed) == on_disk
    ev["n_manifest_files"] = len(listed)
    ev["manifest_sha256_list_sha256"] = sha256_file(out / "receipts" / "MANIFEST_FILE_SHA256.txt")
    mh = json.loads((out / "receipts" / "MANIFEST_HASHES.json").read_text())
    c["MANIFEST_HASHES_matches_headers"] = all(
        mh["manifests"][m]["header_content_sha256"] == headers[m]["header_content_sha256"]
        and all(t["table_content_sha256"] == mh["manifests"][m]["tables"][t["table"]]["table_content_sha256"]
                and t["file_sha256"] == listed[f"{m}.{t['table']}.parquet"] for t in headers[m]["tables"])
        for m in PRE_LOCK_IDS)
    c["no_test_manifest_present"] = not any(n.startswith(("M07", "M08", "M09")) for n in on_disk)
    c["headers_pre_lock_and_no_test_lock"] = all(h["all_methods_lock_sha256"] is None
                                                 and h["status"] == "BUILT_PRE_LOCK" for h in headers.values())
    ev["header_content_sha256"] = {m: h["header_content_sha256"] for m, h in headers.items()}
    ev["population_sha256"] = {m: h["population_sha256"] for m, h in headers.items()}

    # ---- transform state and v1 inputs -----------------------------------------------------------
    c["transform_state_self_hash"] = True                                       # load_state raised otherwise
    ev["transform_state_sha256"] = state["transform_state_sha256"]
    ev["transform_state_file_sha256"] = sha256_file(out / "TRANSFORM_STATE_CLEAN_V2.json")
    v1_now = {rel: sha256_file(ds / rel) for rel in V1_INPUTS}
    c["v1_input_hashes_match_transform_state"] = v1_now == state["v1_input_sha256"]
    files = {e["path"]: e for e in json.loads((ds / "FILES.json").read_text())["files"]}
    bad, n = [], 0
    for rel, e in sorted(files.items()):
        if rel.startswith(V1_PREFIXES):
            n += 1
            if sha256_file(ds / rel) != e["sha256"]:
                bad.append(rel)
    c["v1_files_read_match_FILES_json"] = not bad
    ev["v1_files_checked"] = n
    ev["v1_files_mismatched"] = bad
    ver_json = json.loads((ds / "VERSION.json").read_text())
    recomputed = hashlib.sha256(json.dumps(ver_json["components"], sort_keys=True, default=str).encode()).hexdigest()
    c["dataset_version_hash_recomputed"] = recomputed == ver_json["version_hash"] == VERSION_HASH
    schema = load_schema()
    c["schema_self_sha256"] = schema["schema_self_sha256"] == SCHEMA_SELF_SHA256

    # ---- snapshot ------------------------------------------------------------------------------
    snap = snapshot_check()
    c["snapshot_unchanged"] = snap["unchanged"]
    ev["snapshot"] = snap

    # ---- build and test receipts ---------------------------------------------------------------
    ba = json.loads((out / "receipts" / "BUILD_ALL.json").read_text())
    c["build_all_ok"] = ba["verify"]["all_verified"] and ba["snapshot_after"]["unchanged"] \
        and ba["build_receipt"]["reconciliation_all_match"] and ba["build_receipt"]["full_build"]
    rss = [json.loads(x) for x in (out / "receipts" / "rss.jsonl").read_text().splitlines() if x.strip()][-1]
    c["build_rss_under_8GiB_and_rc0"] = rss["rc"] == 0 and not rss["killed_rss_guard"] and rss["peak_rss_mib"] < 8192
    rec_json = json.loads((out / "receipts" / "CENSUS_RECONCILIATION.json").read_text())
    c["census_reconciliation_all_match"] = rec_json["all_match"]
    ev["build"] = {"build_all_utc": ba["utc"], "wall_s": ba["wall_s"], "peak_rss_mib": rss["peak_rss_mib"],
                   "rss_end_utc": rss["end_utc"], "manifest_bytes_total": ba["build_receipt"]["manifest_bytes_total"],
                   "n_reconciliation_items": len(rec_json["reconciliation"])}
    t = json.loads(Path(a.tests_receipt).read_text())
    outcomes = {x["test"].split("::")[-1]: x["outcome"] for x in t["cases"]}
    c["server_tests_pass"] = t["exit_code"] == 0 and not ({"failure", "error"} & set(t["counts"])) \
        and t["negative_controls"]["all_failed_as_intended"]
    c["symlink_escape_test_passed_on_linux"] = outcomes.get("test_t44_symlink_escape_refused") == "passed"
    c["symlink_negative_control_failed_as_intended"] = outcomes.get("test_nc_t44_symlink_naive") == "xfailed"
    ev["server_tests"] = {"exit_code": t["exit_code"], "counts": t["counts"], "negative_controls": t["negative_controls"],
                          "utc": t["utc"], "platform": t["platform"],
                          "skipped": [k for k, v in outcomes.items() if v == "skipped"],
                          "receipt_sha256": sha256_file(Path(a.tests_receipt))}

    rec = {"receipt": "DATA04_SERVER_FINAL_RECEIPT", "task": "DATA-04",
           "date_u_start": a.date_start, "date_u_end": a.date_end, "written_utc": utc_now(),
           "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
           "data_v2_root": str(out), "dataset_root": str(ds), "checks": c, "evidence": ev,
           "overall_pass": all(c.values()),
           "scope_note": "pre-TEST manifests only; no TEST manifest, row, label or feature was read; the "
                         "DATA_PROTOCOL_LOCK is written by the Operations Director, not by DATA-04"}
    rec["receipt_sha256"] = self_hash(rec, "receipt_sha256")
    atomic_write_json(out, "receipts/DATA04_SERVER_FINAL_RECEIPT.json", rec)
    print(json.dumps({"overall_pass": rec["overall_pass"], "failed": [k for k, v in c.items() if not v],
                      "receipt_sha256": rec["receipt_sha256"]}))
    return 0 if rec["overall_pass"] else 4


if __name__ == "__main__":
    sys.exit(main())

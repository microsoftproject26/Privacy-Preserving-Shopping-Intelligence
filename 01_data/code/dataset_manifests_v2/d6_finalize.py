"""v4data06.d6_finalize (owner: Data engineer, DATA-06) - the manifest-role registry and the final server receipt.

  registry  verifies the DATA-06 build receipt (self-hash, PASS), re-reads every DATA-06 header / table / .OK marker,
            then writes contracts/manifest_registry/REGISTRY_V2.json. Self-hash field name is `self_sha256` (Ops
            correction, DECISION_LOG 45 / v4train.registry convention - see the reissued REGISTRY_FS.json v2), NOT
            `registry_self_sha256`.
  final     adds the RSS-guard record, the job's `date -u` stamps and the test receipt, and writes
            data_v3/receipts/DATA06_SERVER_RECEIPT.json (self-hash `receipt_sha256`).
Reads only data_v3/ and contracts/DATA_PROTOCOL_LOCK.json; never the dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import ppsi_env  # noqa: E402,F401  (import isolation first, T43)

import pyarrow.parquet as pq  # noqa: E402

from v4data.common import atomic_write_bytes, json_bytes_strict, sha256_file, self_hash, table_content_sha256  # noqa: E402
from v4data.manifests import refuse_test_manifest, reread_columns  # noqa: E402
from v4data06 import d6_common as DC  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

V4 = DC.V4
REG_REL = "contracts/manifest_registry/REGISTRY_V2.json"
SELF_HASH_FIELD = "self_sha256"
SELF_HASH_RULE = ("sha256 of json.dumps(object minus self_sha256, sort_keys=True, ensure_ascii=True, "
                 "separators=(',',':'), allow_nan=False) - v4data.common.self_hash, field renamed self_sha256 to "
                 "match v4train.registry (Ops correction, DECISION_LOG 45)")


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def with_self_sha256(obj: dict) -> dict:
    d = FC.jsonable(dict(obj))
    d.pop(SELF_HASH_FIELD, None)
    d[SELF_HASH_FIELD] = self_hash(d, SELF_HASH_FIELD)
    return d


def verify_self_sha256(obj: dict) -> bool:
    return obj.get(SELF_HASH_FIELD) == self_hash(obj, SELF_HASH_FIELD)


def verify_manifest_on_disk(v4: Path, out_rel: str, mid: str) -> dict:
    refuse_test_manifest(mid)
    d = Path(v4) / out_rel / "manifests"
    hp = d / f"{mid}.header.json"
    h = _load(hp)
    allowed_false = DC.KNOWN_ASSERTION_DEVIATIONS.get(mid, set())
    bad_assertions = [k for k, v in h["assertions"].items() if v is not True and k not in allowed_false]
    ok = {"header.content_self_hash": h["header_content_sha256"] == FC.header_content_sha256(h),
          "header.manifest_id": h["manifest_id"] == mid,
          "header.no_all_methods_lock": h.get("all_methods_lock_sha256") is None,
          "header.no_test_exposure_entry": h.get("test_exposure_register_entry_id") is None,
          "header.assertions_all_true_or_known_deviation": not bad_assertions}
    tables = {}
    for t in h["tables"]:
        f = d / f"{mid}.{t['table']}.parquet"
        fsha = sha256_file(f)
        tbl = pq.ParquetFile(f).read()
        c = table_content_sha256(reread_columns(tbl))
        ok[f"{t['table']}.file_sha256"] = fsha == t["file_sha256"]
        ok[f"{t['table']}.content_sha256"] = c == t["table_content_sha256"]
        ok[f"{t['table']}.row_count"] = tbl.num_rows == t["row_count"]
        tables[t["table"]] = {"file": f"{out_rel}/manifests/{f.name}", "file_sha256": fsha,
                              "table_content_sha256": c, "row_count": int(tbl.num_rows)}
        del tbl
    mk = _load(d / f"{mid}.OK")["verified_sha256"]
    ok["ok_marker_matches_disk"] = bool(mk) and all(sha256_file(Path(v4) / out_rel / rel) == s for rel, s in mk.items())
    return {"header": h, "header_file_sha256": sha256_file(hp), "tables": tables, "checks": ok,
            "all_ok": all(ok.values())}


def build_registry(v4: Path = V4, out_rel: str = "data_v3") -> dict:
    root = Path(v4) / out_rel
    brec = _load(root / "receipts" / "DATA06_BUILD_RECEIPT.json")
    pre = {"build_receipt_self_hash": FC.verify_self_hash(brec, "receipt_sha256"),
           "build_receipt_pass": brec.get("status") == "PASS" and brec.get("full_build") is True,
           "build_receipt_n_failed_zero": brec.get("n_failed") == 0}
    if not all(pre.values()):
        raise SystemExit(f"DATA06_BUILD_RECEIPT does not verify: {pre}")
    lock = FC.lock_info(Path(v4))
    mans = {}
    for out_id, definition, kind in (
        (DC.MAIN_TRAIN_ID, "M01_COHORT_TRAIN", "TRAIN"), (DC.MAIN_EVAL_ID, "M02_VALIDATION_WARM", "EVAL"),
        (DC.CALIB_TRAIN_ID, "M05_CALIB_TRAIN", "TRAIN"), (DC.CALIB_EVAL_ID, "M06_CALIB_VALIDATION_WARM", "EVAL"),
        (DC.UNEW_VAL_TRAIN_ID, "M01_COHORT_TRAIN", "TRAIN"), (DC.UNEW_VAL_EVAL_ID, "M02_VALIDATION_WARM", "EVAL"),
        (DC.UNEW_TEST_TRAIN_ID, "M01_COHORT_TRAIN", "TRAIN"), (DC.UNEW_TEST_EVAL_ID, "M02_VALIDATION_WARM", "EVAL"),
        (DC.SPARSE_TRAIN_ID, "M01_COHORT_TRAIN", "TRAIN"), (DC.SPARSE_EVAL_ID, "M02_VALIDATION_WARM", "EVAL"),
        (DC.HEAVY_TRAIN_ID, "M01_COHORT_TRAIN", "TRAIN"), (DC.HEAVY_EVAL_ID, "M02_VALIDATION_WARM", "EVAL"),
        (DC.TRUE_COLD_ID, "M03_COLD_PANEL_VAL", "EVAL"), (DC.COLD_H_ID, "M04_COLD_H_PANEL", "PANEL"),
        (DC.WARM_TRUNCATION_ID, "M04W_WARM_TRUNCATION_PANEL", "PANEL"),
    ):
        v = verify_manifest_on_disk(v4, out_rel, out_id)
        if not v["all_ok"]:
            raise SystemExit(f"{out_id}: on-disk verification failed: {v['checks']}")
        h = v["header"]
        if FC.hashes_entry(h, h["tables"], v["header_file_sha256"]) != brec["manifests"][out_id]:
            raise SystemExit(f"{out_id}: header / files differ from the build receipt")
        mans[out_id] = {"path": f"{out_rel}/manifests/{out_id}.header.json", "manifest_dir": f"{out_rel}/manifests",
                        "kind": kind, "definition": definition, "status": h["status"], "role": h["run_label"],
                        "content_sha256": FC.manifest_content_sha256(h["tables"]), "header_sha256": v["header_file_sha256"],
                        "header_content_sha256": h["header_content_sha256"], "n_users": h["n_users"],
                        "n_decisions": h["n_decisions"], "n_eval_rows": h["n_eval_rows"],
                        "n_rankable_rows": h["n_rankable_rows"], "population_sha256": h["population_sha256"],
                        "population_sha256_selection_key_v2": h["population_sha256_selection_key_v2"],
                        "ordered_decision_id_sha256": h["ordered_decision_id_sha256"],
                        "eval_ordered_decision_id_sha256": h["eval_ordered_decision_id_sha256"],
                        "rankable_ordered_decision_id_sha256": h["rankable_ordered_decision_id_sha256"],
                        "parent_manifests": h["parent_manifests"], "tables": v["tables"],
                        "marker": f"{out_rel}/manifests/{out_id}.OK"}
    reg = {
        "registry": "MANIFEST_ROLE_REGISTRY", "registry_version": 2, "task": DC.TASK, "data_version": DC.DATA_VERSION,
        "roles": {
            "MAIN": {"train": DC.MAIN_TRAIN_ID, "eval": DC.MAIN_EVAL_ID},
            "CALIBRATION": {"train": DC.CALIB_TRAIN_ID, "eval": DC.CALIB_EVAL_ID},
            "UNSEEN_VAL": {"train": DC.UNEW_VAL_TRAIN_ID, "eval": DC.UNEW_VAL_EVAL_ID},
            "UNSEEN_HOLDOUT": {"train": DC.UNEW_TEST_TRAIN_ID, "eval": DC.UNEW_TEST_EVAL_ID},
            "SPARSE_PANEL": {"train": DC.SPARSE_TRAIN_ID, "eval": DC.SPARSE_EVAL_ID},
            "HEAVY_PANEL": {"train": DC.HEAVY_TRAIN_ID, "eval": DC.HEAVY_EVAL_ID},
            "TRUE_COLD": {"eval": DC.TRUE_COLD_ID},
            "COLD_H": {"panel": DC.COLD_H_ID},
            "WARM_TRUNCATION": {"panel": DC.WARM_TRUNCATION_ID},
        },
        "manifests": mans,
        "transform_state": {"path": "data_v2/TRANSFORM_STATE_CLEAN_V2.json",
                            "transform_state_sha256": DC.TRANSFORM_STATE_SELF_SHA256,
                            "file_sha256": sha256_file(V4 / "data_v2" / "TRANSFORM_STATE_CLEAN_V2.json"),
                            "rule": "shared statistics = full TRAIN as fitted (TRANSFORM_STATE_CLEAN_V2, unchanged), "
                                    "identical for every arm (ADR-005 addendum 2, D5); DATA-06 fits no new statistics"},
        "path_root": "the ppsi_v4 workspace root (server: /home/jovyan/ahmedabdelhamed01/"
                     "Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4); every path is relative to it",
        "source_data_protocol_lock": {"lock_self_sha256": lock["lock_self_sha256"], "file_sha256": lock["file_sha256"]},
        "builder_code_sha256": brec["code"]["v4data_builder_code_sha256"], "d6_code_sha256": brec["code"]["d6_code_sha256"],
        "population_bands": {
            "pool_5_100": {"n_users": DC.POOL_N, "n_examples": DC.POOL_EXAMPLES, "population_sha256_selection_key_v2": DC.POOL_SHA_V2},
            "MAIN": {"rank_range": list(DC.BANDS["MAIN"]), "population_sha256_selection_key_v2": DC.MAIN_SHA_V2},
            "CALIB": {"rank_range": list(DC.BANDS["CALIB"]), "population_sha256_selection_key_v2": DC.CALIB_SHA_V2},
            "U_NEW_VAL": {"rank_range": list(DC.BANDS["U_NEW_VAL"])},
            "U_NEW_TEST": {"rank_range": list(DC.BANDS["U_NEW_TEST"])},
        },
        "build_receipt": {"receipt": f"{out_rel}/receipts/DATA06_BUILD_RECEIPT.json",
                          "receipt_file_sha256": sha256_file(root / "receipts" / "DATA06_BUILD_RECEIPT.json"),
                          "receipt_sha256": brec["receipt_sha256"]},
        "naming_decision": DC.NAMING_DECISION,
        "selection_refusal": DC.SELECTION_REFUSAL,
        "rules": {"content_sha256": "sha256 of canonical JSON [{table,row_count,table_content_sha256}] in header "
                                    "order (v4fs_data.fs_common.manifest_content_sha256)",
                  "header_sha256": "sha256 of the header file bytes",
                  "header_content_sha256": "the DATA-04 builder rule (v4int.verify.header_content_sha256)"},
        "self_sha256_rule": SELF_HASH_RULE,
    }
    return with_self_sha256(reg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("registry", "final"))
    ap.add_argument("--v4", default=str(V4))
    ap.add_argument("--out-rel", default="data_v3")
    ap.add_argument("--tests-receipt", default=None)
    ap.add_argument("--rss-log", action="append", default=[])
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args(argv)
    v4 = Path(a.v4)
    if a.step == "registry":
        reg = build_registry(v4, a.out_rel)
        p = atomic_write_bytes(v4, REG_REL, json_bytes_strict(reg))
        back = _load(p)
        if not verify_self_sha256(back):
            raise SystemExit("registry self-hash does not verify after the write")
        print(json.dumps({"registry": REG_REL, "file_sha256": sha256_file(p), "self_sha256": reg["self_sha256"]}))
        return 0
    root = v4 / a.out_rel
    run = Path(a.run_dir)
    brec = _load(root / "receipts" / "DATA06_BUILD_RECEIPT.json")
    reg = _load(v4 / REG_REL)
    tests = _load(Path(a.tests_receipt))
    guards = [[json.loads(x) for x in Path(g).read_text(encoding="utf-8").splitlines() if x.strip()][-1]
              for g in a.rss_log]
    stamps = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(run.glob("*_utc.txt"))}
    df = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(run.glob("df_*.txt"))}
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and "_run" not in p.parts)
    checks = {"build_receipt_pass": brec["status"] == "PASS" and FC.verify_self_hash(brec, "receipt_sha256")
                                    and brec["n_failed"] == 0,
              "registry_self_hash": verify_self_sha256(reg),
              "tests_exit_0": tests.get("exit_code") == 0,
              "tests_negative_controls_failed_as_intended": tests.get("negative_controls", {}).get(
                  "all_failed_as_intended") is True,
              "tests_real_data_ran": tests.get("counts", {}).get("skipped", 0) == 0,
              "rss_guards_rc_0": bool(guards) and all(g["rc"] == 0 and not g["killed_rss_guard"] for g in guards),
              "no_pycache_in_owned_code": not any(True for d in ("v4data06", "tests_data06")
                                                  for _ in (CODE / d).rglob("__pycache__")),
              "no_test_manifest_built": all("TEST" not in mid.upper() for mid in reg["manifests"]),
              "no_test_role": all("TEST" not in role.upper() for role in reg["roles"])}
    rec = {"receipt": "DATA06_SERVER_RECEIPT", "task": DC.TASK, "overall_pass": all(checks.values()), "checks": checks,
           "date_u": stamps, "df": df, "rss_guards": guards, "data_v3_bytes": int(size),
           "registry": {"path": REG_REL, "file_sha256": sha256_file(v4 / REG_REL), "self_sha256": reg["self_sha256"]},
           "build_receipt_sha256": brec["receipt_sha256"],
           "tests": {k: tests.get(k) for k in ("exit_code", "counts", "negative_controls", "summary_line")},
           "tests_receipt_file_sha256": sha256_file(Path(a.tests_receipt)),
           "manifests": {mid: {k: e[k] for k in ("n_users", "n_decisions", "n_eval_rows", "n_rankable_rows",
                                                 "content_sha256", "header_sha256", "header_content_sha256",
                                                 "population_sha256", "population_sha256_selection_key_v2")}
                        for mid, e in reg["manifests"].items()},
           "code": brec["code"], "finalize_code_sha256": sha256_file(HERE / "d6_finalize.py"), "test_split_read": False}
    p, rec = _write_self_hashed(root, "receipts/DATA06_SERVER_RECEIPT.json", rec)
    print(json.dumps({"overall_pass": rec["overall_pass"], "receipt_sha256": rec["receipt_sha256"],
                      "file_sha256": sha256_file(p), "checks": checks}))
    return 0 if rec["overall_pass"] else 4


def _write_self_hashed(out_root: Path, rel: str, obj: dict):
    return FC.write_json_self_hashed(out_root, rel, obj, field="receipt_sha256")


if __name__ == "__main__":
    sys.exit(main())

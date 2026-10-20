"""v4fs_data.fs_finalize (owner: Data engineer, FS-01) - the manifest-role registry and the final server receipt.

  registry  verifies the FS build and equivalence receipts (self-hash, PASS), re-reads every FS header / table / .OK
            marker, verifies TRANSFORM_STATE_CLEAN_V2 (the shared statistics, unchanged), then writes
            contracts/manifest_registry/REGISTRY_FS.json (self-hash `registry_self_sha256`).
  final     adds the RSS-guard record, the job's `date -u` stamps and the test receipt, and writes
            data_fs/receipts/FS01_SERVER_FINAL_RECEIPT.json (self-hash `receipt_sha256`).
Reads only data_fs/, data_v2/TRANSFORM_STATE_CLEAN_V2.json, contracts/DATA_PROTOCOL_LOCK.json; never the dataset.
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

from v4data.common import atomic_write_bytes, json_bytes_strict, sha256_file, table_content_sha256  # noqa: E402
from v4data.manifests import refuse_test_manifest, reread_columns  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

V4 = FC.V4
REG_REL = "contracts/manifest_registry/REGISTRY_FS.json"


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def verify_manifest_on_disk(v4: Path, out_rel: str, mid: str) -> dict:
    refuse_test_manifest(mid)
    d = Path(v4) / out_rel / "manifests"
    hp = d / f"{mid}.header.json"
    h = _load(hp)
    ok = {"header.content_self_hash": h["header_content_sha256"] == FC.header_content_sha256(h),
          "header.manifest_id": h["manifest_id"] == mid, "header.status": h["status"] == FC.FS_STATUS,
          "header.no_all_methods_lock": h.get("all_methods_lock_sha256") is None,
          "header.no_test_exposure_entry": h.get("test_exposure_register_entry_id") is None,
          "header.assertions_all_true": all(v is True for v in h["assertions"].values()),
          "header.expected_all_match": all(e["match"] for e in h["expected_counts_check"])}
    tables = {}
    for t in h["tables"]:
        f = d / f"{mid}.{t['table']}.parquet"
        fsha = sha256_file(f)
        tbl = pq.ParquetFile(f).read()
        c = table_content_sha256(reread_columns(tbl))
        ok[f"{t['table']}.file_sha256"] = fsha == t["file_sha256"]
        ok[f"{t['table']}.content_sha256"] = c == t["table_content_sha256"]
        ok[f"{t['table']}.row_count"] = tbl.num_rows == t["row_count"]
        tables[t["table"]] = {"file": f"{out_rel}/manifests/{f.name}", "file_sha256": fsha, "table_content_sha256": c,
                              "row_count": int(tbl.num_rows)}
        del tbl
    mk = _load(d / f"{mid}.OK")["verified_sha256"]
    ok["ok_marker_matches_disk"] = bool(mk) and all(sha256_file(Path(v4) / out_rel / rel) == s for rel, s in mk.items())
    return {"header": h, "header_file_sha256": sha256_file(hp), "tables": tables, "checks": ok,
            "all_ok": all(ok.values())}


def build_registry(v4: Path = V4, out_rel: str = "data_fs") -> dict:
    root = Path(v4) / out_rel
    brec = _load(root / "receipts" / "FS_BUILD_RECEIPT.json")
    erec = _load(root / "receipts" / "EQUIVALENCE_M05_M06.json")
    pre = {"build_receipt_self_hash": FC.verify_self_hash(brec, "receipt_sha256"),
           "build_receipt_pass": brec.get("status") == "PASS" and brec.get("full_build") is True,
           "equivalence_receipt_self_hash": FC.verify_self_hash(erec, "receipt_sha256"),
           "equivalence_all_equal": erec.get("all_equal") is True and erec.get("full_build") is True}
    if not all(pre.values()):
        raise SystemExit(f"receipts do not verify: {pre}")
    from v4data.transform_state import load_state
    tsp = Path(v4) / "data_v2" / "TRANSFORM_STATE_CLEAN_V2.json"
    st = load_state(tsp)
    if st["transform_state_sha256"] != FC.TRANSFORM_STATE_SELF_SHA256:
        raise SystemExit("TRANSFORM_STATE_CLEAN_V2 self-hash changed")
    lock = FC.lock_info(Path(v4))
    tsf = root / "TRANSFORM_STATE_FS.json"
    fst = _load(tsf)
    if not (FC.verify_self_hash(fst, "transform_state_fs_sha256") and fst["status"] == "OPTIONAL_SENSITIVITY_ONLY"
            and fst["method_proof_all_train"]["all_reproduced"] is True):
        raise SystemExit("TRANSFORM_STATE_FS does not verify (self-hash, status or the all-TRAIN method proof)")
    mans = {}
    for mid, kind, definition in ((FC.FS_TRAIN_ID, "TRAIN", "M01_COHORT_TRAIN"),
                                  (FC.FS_EVAL_ID, "EVAL", "M02_VALIDATION_WARM")):
        v = verify_manifest_on_disk(v4, out_rel, mid)
        if not v["all_ok"]:
            raise SystemExit(f"{mid}: on-disk verification failed: {v['checks']}")
        h = v["header"]
        if FC.hashes_entry(h, h["tables"], v["header_file_sha256"]) != brec["manifests"][mid]:
            raise SystemExit(f"{mid}: header / files differ from the build receipt")
        mans[mid] = {"path": f"{out_rel}/manifests/{mid}.header.json", "manifest_dir": f"{out_rel}/manifests",
                     "kind": kind, "definition": definition, "status": h["status"],
                     "content_sha256": FC.manifest_content_sha256(h["tables"]),
                     "header_sha256": v["header_file_sha256"], "header_content_sha256": h["header_content_sha256"],
                     "n_users": h["n_users"], "n_decisions": h["n_decisions"], "n_eval_rows": h["n_eval_rows"],
                     "n_rankable_rows": h["n_rankable_rows"], "population_sha256": h["population_sha256"],
                     "population_sha256_selection_key_v2": h["population_sha256_selection_key_v2"],
                     "ordered_decision_id_sha256": h["ordered_decision_id_sha256"],
                     "eval_ordered_decision_id_sha256": h["eval_ordered_decision_id_sha256"],
                     "rankable_ordered_decision_id_sha256": h["rankable_ordered_decision_id_sha256"],
                     "parent_manifests": h["parent_manifests"], "tables": v["tables"],
                     "marker": f"{out_rel}/manifests/{mid}.OK"}
    if fst["fit"]["population"]["header_content_sha256"] != mans[FC.FS_TRAIN_ID]["header_content_sha256"]:
        raise SystemExit("TRANSFORM_STATE_FS was fitted on a different FS_TRAIN manifest")
    pop = brec["population"]
    reg = {
        "registry": "MANIFEST_ROLE_REGISTRY", "registry_version": 1, "task": FC.TASK, "data_version": FC.DATA_VERSION,
        "run_label": FC.FS_STATUS,
        "roles": {"FEATURE_SEARCH_SCREEN": {"train": FC.FS_TRAIN_ID, "eval": FC.FS_EVAL_ID}},
        "manifests": mans,
        "transform_state": {"path": "data_v2/TRANSFORM_STATE_CLEAN_V2.json",
                            "transform_state_sha256": st["transform_state_sha256"], "file_sha256": sha256_file(tsp),
                            "rule": "shared statistics = full TRAIN as fitted (TRANSFORM_STATE_CLEAN_V2, unchanged), "
                                    "identical for every arm; the FS screen uses THIS state (DECISION_LOG 42 / 43)"},
        "transform_state_optional_sensitivity": {
            "path": f"{out_rel}/TRANSFORM_STATE_FS.json", "transform_state_fs_sha256": fst["transform_state_fs_sha256"],
            "file_sha256": sha256_file(tsf), "use": fst["use"], "fit_population": FC.FS_TRAIN_ID,
            "fit_population_header_content_sha256": fst["fit"]["population"]["header_content_sha256"],
            "normalizers": sorted(fst["normalized_channels_fs"]),
            "method_proof_all_train_all_reproduced": fst["method_proof_all_train"]["all_reproduced"],
            "self_hash_rule": "v4data.common.self_hash of the state without transform_state_fs_sha256"},
        "path_root": "the ppsi_v4 workspace root (server: /home/jovyan/ahmedabdelhamed01/"
                     "Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4); every path is relative to it",
        "source_data_protocol_lock": {"lock_self_sha256": lock["lock_self_sha256"], "file_sha256": lock["file_sha256"]},
        "builder_code_sha256": brec["code"]["v4data_builder_code_sha256"], "fs_code_sha256": brec["code"]["fs_code_sha256"],
        "fs_population": {k: pop[k] for k in ("definition", "n_fs", "pool_size", "pool_examples",
                                              "pool_population_sha256_selection_key_v2", "fs_ranks",
                                              "fs_population_sha256_selection_key_v2", "disjointness", "bands")},
        "equivalence": {"receipt": f"{out_rel}/receipts/EQUIVALENCE_M05_M06.json",
                        "receipt_file_sha256": sha256_file(root / "receipts" / "EQUIVALENCE_M05_M06.json"),
                        "receipt_sha256": erec["receipt_sha256"], "all_equal": erec["all_equal"]},
        "build_receipt": {"receipt": f"{out_rel}/receipts/FS_BUILD_RECEIPT.json",
                          "receipt_file_sha256": sha256_file(root / "receipts" / "FS_BUILD_RECEIPT.json"),
                          "receipt_sha256": brec["receipt_sha256"]},
        "selection_refusal": FC.SELECTION_REFUSAL,
        "rules": {"content_sha256": "sha256 of canonical JSON [{table,row_count,table_content_sha256}] in header order "
                                    "(v4fs_data.fs_common.manifest_content_sha256)",
                  "header_sha256": "sha256 of the header file bytes",
                  "header_content_sha256": "the DATA-04 builder rule (v4int.verify.header_content_sha256)",
                  "registry_self_sha256": "v4data.common.self_hash: sha256 of the canonical JSON (sort_keys, ASCII, "
                                          "compact separators) of this object without registry_self_sha256"},
    }
    return FC.with_self_hash(reg, "registry_self_sha256")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("registry", "final"))
    ap.add_argument("--v4", default=str(V4))
    ap.add_argument("--out-rel", default="data_fs")
    ap.add_argument("--tests-receipt", default=None)
    ap.add_argument("--rss-log", action="append", default=[])
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args(argv)
    v4 = Path(a.v4)
    if a.step == "registry":
        reg = build_registry(v4, a.out_rel)
        p = atomic_write_bytes(v4, REG_REL, json_bytes_strict(reg))
        back = _load(p)
        if not FC.verify_self_hash(back, "registry_self_sha256"):
            raise SystemExit("registry self-hash does not verify after the write")
        print(json.dumps({"registry": REG_REL, "file_sha256": sha256_file(p),
                          "registry_self_sha256": reg["registry_self_sha256"]}))
        return 0
    root = v4 / a.out_rel
    run = Path(a.run_dir)
    brec = _load(root / "receipts" / "FS_BUILD_RECEIPT.json")
    erec = _load(root / "receipts" / "EQUIVALENCE_M05_M06.json")
    reg = _load(v4 / REG_REL)
    tests = _load(Path(a.tests_receipt))
    guards = [[json.loads(x) for x in Path(g).read_text(encoding="utf-8").splitlines() if x.strip()][-1]
              for g in a.rss_log]
    stamps = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(run.glob("*_utc.txt"))}
    df = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(run.glob("df_*.txt"))}
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and "_run" not in p.parts)
    checks = {"build_receipt_pass": brec["status"] == "PASS" and FC.verify_self_hash(brec, "receipt_sha256"),
              "equivalence_all_equal": erec["all_equal"] is True and FC.verify_self_hash(erec, "receipt_sha256"),
              "registry_self_hash": FC.verify_self_hash(reg, "registry_self_sha256"),
              "tests_exit_0": tests.get("exit_code") == 0,
              "tests_negative_controls_failed_as_intended": tests.get("negative_controls", {}).get(
                  "all_failed_as_intended") is True,
              "tests_real_data_ran": tests.get("counts", {}).get("skipped", 0) == 0,
              "rss_guards_rc_0": bool(guards) and all(g["rc"] == 0 and not g["killed_rss_guard"] for g in guards),
              "transform_state_fs_registered": reg["transform_state_optional_sensitivity"]["file_sha256"]
              == sha256_file(root / "TRANSFORM_STATE_FS.json"),
              "no_pycache_in_owned_code": not any(True for d in ("v4fs_data", "tests_fs_data")
                                                  for _ in (CODE / d).rglob("__pycache__"))}
    rec = {"receipt": "FS01_SERVER_FINAL_RECEIPT", "task": FC.TASK, "overall_pass": all(checks.values()),
           "checks": checks, "date_u": stamps, "df": df, "rss_guards": guards, "data_fs_bytes": int(size),
           "registry": {"path": REG_REL, "file_sha256": sha256_file(v4 / REG_REL),
                        "registry_self_sha256": reg["registry_self_sha256"]},
           "build_receipt_sha256": brec["receipt_sha256"], "equivalence_receipt_sha256": erec["receipt_sha256"],
           "tests": {k: tests.get(k) for k in ("exit_code", "counts", "negative_controls", "summary_line")},
           "tests_receipt_file_sha256": sha256_file(Path(a.tests_receipt)),
           "manifests": {mid: {k: e[k] for k in ("n_users", "n_decisions", "n_eval_rows", "n_rankable_rows",
                                                 "content_sha256", "header_sha256", "header_content_sha256",
                                                 "population_sha256", "population_sha256_selection_key_v2")}
                         for mid, e in reg["manifests"].items()},
           "fs_population_sha256_selection_key_v2": reg["fs_population"]["fs_population_sha256_selection_key_v2"],
           "transform_state": reg["transform_state"],
           "transform_state_optional_sensitivity": reg["transform_state_optional_sensitivity"],
           "code": brec["code"], "finalize_code_sha256": sha256_file(HERE / "fs_finalize.py"),
           "transform_code_sha256": sha256_file(HERE / "fs_transform.py"), "test_split_read": False}
    p, rec = FC.write_json_self_hashed(root, "receipts/FS01_SERVER_FINAL_RECEIPT.json", rec)
    print(json.dumps({"overall_pass": rec["overall_pass"], "receipt_sha256": rec["receipt_sha256"],
                      "file_sha256": sha256_file(p), "checks": checks}))
    return 0 if rec["overall_pass"] else 4


if __name__ == "__main__":
    sys.exit(main())

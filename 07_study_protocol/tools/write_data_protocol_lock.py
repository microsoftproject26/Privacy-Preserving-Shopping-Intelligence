"""write_data_protocol_lock.py (owner: Operations Director) — writes contracts/DATA_PROTOCOL_LOCK.json ONCE (O_EXCL) from the
FINAL SERVER DATA-04 receipt plus the frozen contracts, per the owner directive of 2026-09-26 (DECISION_LOG 18):
the lock is written from server evidence, never from local-build evidence alone.

Fails closed if: the local copy of the server receipt does not hash to the value measured on the server; overall_pass is
not true or any check is false; the server's frozen_inputs copies of the contracts differ from the local contracts; any
listed file is missing. Prints the lock's self-hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
import ppsi_paths

V = ppsi_paths.PROJECT_ROOT.resolve()   # the ppsi_v4 workspace
SERVER_RECEIPT_SHA256_MEASURED_ON_SERVER = "9f3f296e03b55fdb316a9899cd1910ac6cbc8bc2d366ac48575f5391c208a5e0"   # sha256sum on the server, 16:32:46Z


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jl(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    rp = V / "data_v2/server/receipts/DATA04_SERVER_FINAL_RECEIPT.json"
    if sha(rp) != SERVER_RECEIPT_SHA256_MEASURED_ON_SERVER:
        raise SystemExit("local copy of the server receipt does not match the server-measured hash")
    r = jl(rp)
    if r.get("overall_pass") is not True or not all(v is True for v in r["checks"].values()):
        raise SystemExit("server receipt is not a full PASS")
    contracts = {n: V / "contracts" / n for n in ("DATA_CONTRACT_CLEAN_V2.md", "FEATURE_SCHEMA_CLEAN_V2.json", "EVALUATION_CONTRACT_V4.md",
                                                  "CLIENT_CONTRACT_V1.md", "MANIFEST_SPEC_V4.json", "PREG1_LOCK_VALUES.json")}
    csha = {n: sha(p) for n, p in contracts.items()}
    frozen = {k.split("/", 1)[1]: v for k, v in r["evidence"]["v4data_file_sha256"].items() if k.startswith("frozen_inputs/")}
    mismatch = {n: (frozen[n], csha[n]) for n in frozen if n in csha and frozen[n] != csha[n]}
    if mismatch:
        raise SystemExit(f"server frozen_inputs differ from local contracts: {mismatch}")
    schema = jl(contracts["FEATURE_SCHEMA_CLEAN_V2.json"])
    reviews = {n: sha(V / "company/reviews" / n) for n in ("G0_PLAN_FABLE_POSITION.md", "G0B_PROTOCOL_FABLE_POSITION.md", "PREG1_FABLE_POSITION.md",
                                                            "G0_PLAN_SCIENTIFIC_SYNTHESIS.md", "G0B_PROTOCOL_SCIENTIFIC_SYNTHESIS.md", "PREG1_SCIENTIFIC_SYNTHESIS.md")}
    adrs = {p.name: sha(p) for p in sorted((V / "company/decisions").glob("ADR-*.md"))}
    mh = jl(V / "data_v2/server/receipts/MANIFEST_HASHES.json")
    ev = r["evidence"]
    lock = {
        "document": "DATA_PROTOCOL_LOCK", "version": "1.0.0", "project": "PPSI V4 single-backbone",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "written_by": "Operations Director (Claude), after Level-1 review",
        "authority": ["Fable G0 APPROVE_WITH_FIXES (G0_PLAN_FABLE_POSITION.md)", "Fable G0(b) APPROVE_WITH_FIXES (G0B_PROTOCOL_FABLE_POSITION.md), fixes applied",
                      "Fable pre-G1 T1-T3 DECIDED (PREG1_FABLE_POSITION.md)", "Owner directives DECISION_LOG 18-19"],
        "basis": "FINAL SERVER DATA-04 receipt (not local-build evidence)",
        "server_receipt": {"path": "ppsi_v4/data_v2/receipts/DATA04_SERVER_FINAL_RECEIPT.json (server); local copy data_v2/server/receipts/",
                           "sha256_server_measured": SERVER_RECEIPT_SHA256_MEASURED_ON_SERVER, "receipt_self_sha256": r.get("receipt_sha256"),
                           "window_utc": [r["date_u_start"], r["date_u_end"]], "checks_all_true": len(r["checks"]), "host": r.get("host")},
        "data_version": {"name": "rees46_nextitem_v4_clean (Clean V2 sidecar over immutable v1)", "base": "rees46_nextitem_v1",
                         "base_version_hash": "860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd",
                         "v1_files_verified": ev["v1_files_checked"], "v1_files_mismatched": ev["v1_files_mismatched"],
                         "raw_source_sha256": {"2019-10": "fedd938409b5f836ec89b39c861b13dad99fc7cd9beb1fddd97a2d50488b5b80", "2019-11": "addd9a27ed99abdece368019ebfba19568972c7fcbe949db3484aab8a3bcffec"}},
        "splits_and_labels": {"TRAIN": ["2019-10-01T00:00:00Z", "2019-11-16T00:00:00Z"], "VALIDATION": ["2019-11-16T00:00:00Z", "2019-11-23T00:00:00Z"],
                              "RETRO_TEST": ["2019-11-23T00:00:00Z", "2019-11-30T00:00:00Z"], "LABEL_GRACE": ["2019-11-30T00:00:00Z", "2019-12-01T00:00:00Z"],
                              "rule": "half-open UTC; VALIDATION = old VS ∪ VC minimal-diff; TRAIN labels observed before TRAIN_END; GRACE emits 0 decisions",
                              "censor_classes_validation": {"CEN_LEGACY_BARRIER": 170, "CEN_AMBIGUOUS_NEXT_BARRIER": 4489, "CEN_RIGHT_EDGE": 152, "NO_OBSERVED_NEXT": 3064778},
                              "exposure_status": "VALIDATION and RETRO_TEST are previously inspected; all V4 results are retrospective/adaptive"},
        "catalogue": {"support_rule": ">= 5 TRAIN representative views, fitted on full TRAIN", "K": 158486,
                      "catalogue_parquet_sha256": "d99d0ce7dd05473f1fa84459b7e1e868c6f9412747d79e511661801b11268acd", "candidate_order": "v1 output_class order"},
        "features_and_transforms": {"schema_version": schema.get("schema_version") or schema.get("version"), "schema_self_sha256": schema.get("schema_self_sha256"),
                                    "transform_state_self_sha256": ev["transform_state_sha256"], "transform_state_file_sha256": ev["transform_state_file_sha256"],
                                    "widths": {"numeric": 15, "flags": 15, "user_context": 12, "masks": 4, "channels": 53},
                                    "disclosure": "re-encoded columns (M1 x2, F-1) are computed by an exact float64 inverse of the stored v1 z-values (max error 2.4e-7) because the snapshot store cannot take new normalizers"},
        "population_and_manifests": {"cohort_salt": "PPSI_V4_POPULATION_20260926", "cohort_N": 262144, "calibration_prefix_N": 65536,
                                     "M01_population_sha256": ev["M01_COHORT_TRAIN_population_sha256"], "M05_population_sha256": ev["M05_CALIB_TRAIN_population_sha256"],
                                     "population_sha256": ev["population_sha256"], "header_content_sha256_prelock": ev["header_content_sha256"],
                                     "n_manifest_files": ev["n_manifest_files"], "manifest_sha256_list_sha256": ev["manifest_sha256_list_sha256"],
                                     "manifest_hashes_file_sha256": sha(V / "data_v2/server/receipts/MANIFEST_HASHES.json"), "manifest_hashes": mh,
                                     "test_manifests": "NOT BUILT; M07-M09 refused until ALL_METHODS_LOCK",
                                     "notes": ["M06 follows the spec's field definition (cohort-TRAIN positives over M05); the 'equals M02 but one field' sentence conflicts and is recorded in the M06 header",
                                               "manifest headers carry data_protocol_lock_sha256=PENDING; they are re-issued after this lock (header-only; table content hashes unchanged)"]},
        "code": {"v4data_builder_code_sha256": ev["v4data_builder_code_sha256"], "v4data_file_sha256": ev["v4data_file_sha256"],
                 "snapshot_manifest_sha256": ev["snapshot"]["manifest_sha256"], "snapshot_unchanged": ev["snapshot"]["unchanged"],
                 "server_tests": ev["server_tests"]},
        "contracts_sha256": csha, "fable_rulings_and_syntheses_sha256": reviews, "adrs_sha256": adrs,
        "preg1_lock_values": "contracts/PREG1_LOCK_VALUES.json (T1 selection trigger + staged tie-break; T2 exposure_point=end; T3 frozen eval numerics)",
        "not_locked_here": ["model topology hashes and per-seed init hashes (ARCHITECTURE_LOCK after G2; re-measured on the server)",
                            "TEST manifests (after ALL_METHODS_LOCK)", "FL/PFL/LO method settings beyond the G0 values (G3)"],
    }
    body = json.dumps(lock, indent=1, ensure_ascii=False, allow_nan=False)
    canon = json.dumps(lock, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    lock["lock_self_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    lock["lock_self_sha256_rule"] = "sha256 of json.dumps(lock without this key and without lock_self_sha256_rule, sort_keys=True, ensure_ascii=True, separators=(',',':'))"
    out = V / "contracts" / "DATA_PROTOCOL_LOCK.json"
    fd = os.open(str(out), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(lock, indent=1, ensure_ascii=False, allow_nan=False))
    print(json.dumps({"lock": str(out), "lock_self_sha256": lock["lock_self_sha256"], "file_sha256": sha(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

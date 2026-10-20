"""write_data_protocol_lock_v2.py (owner: Operations Director) — writes contracts/DATA_PROTOCOL_LOCK_V2.json ONCE (O_EXCL)
from the SERVER DATA-06 receipt, REGISTRY_V2, the locked feature schema S* and ADR-005 (R2 + R4 + addendum 3).
DATA_PROTOCOL_LOCK.json (v1.0.0) is never modified; its population part is HISTORICAL (DECISION_LOG 31).

Fails closed if: the local copies of the server receipt, REGISTRY_V2, FS_FINAL or the S* schema do not hash to the values
measured on the server; overall_pass is not true or any check is false; the receipt read TEST; the registry self-hash does
not match the receipt; FS_FINAL is not a LOCK on the S* schema id; the S* self-hash differs from the registered one.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
import ppsi_paths

V = ppsi_paths.PROJECT_ROOT.resolve()   # the ppsi_v4 workspace
S_STAR = "L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7"
# sha256sum on the server, 2026-09-27T11:06Z (DECISION_LOG 54)
SERVER_SHA256 = {
    "data_v3/receipts/DATA06_SERVER_RECEIPT.json": "b73ac7624977c05a89ca9dda9c6842f9e1a3c4d67a6d03be215fe2ad2e04151a",
    "contracts/manifest_registry/REGISTRY_V2.json": "56dd0840cb50e89250a8b9a54dc1d4f621e9e6f9500ec44441cf07cb472ef740",
    "runs_fs/FS_FINAL.json": "799590a456b06384302879a99e301669f4abfc7b05bc6ab7625c1752bc33bf4c",
    "runs_fs/stageB/DECISION_A.json": "142a9915502c631bbf4fcab6e641314627c7f8dfd9bc2cb74a0306badf46f374",
    "runs_fs/stageB/DECISION_B.json": "d8257a4f8b38d0c3b9e32d08e6d801d4614fa70a34881d7bdcc09a3bc797b8a8",
    f"contracts/derived_schemas/{S_STAR}.json": "626ca93e7f51606fe4269f4551a65c0246f85f9f199d2af92881dca971e8b759",
    "contracts/derived_schemas/REGISTRY.json": "c23454cb7d34886e0374349b73a4d153907ebc56b17061926e193e221e4a0bc6",
    "contracts/derived_schemas/THETA0_REGISTRY.json": "e4d8eb86bd4badfc3ab7671b132a49e68f8711617e618d6e76ba5e513aee369c",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jl(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    bad = {k: (v, sha(V / k)) for k, v in SERVER_SHA256.items() if sha(V / k) != v}
    if bad:
        raise SystemExit(f"local copies differ from the server-measured hashes: {bad}")
    r = jl(V / "data_v3/receipts/DATA06_SERVER_RECEIPT.json")
    if r.get("overall_pass") is not True or not all(v is True for v in r["checks"].values()):
        raise SystemExit("server receipt is not a full PASS")
    if r.get("test_split_read") is not False:
        raise SystemExit("server receipt does not certify test_split_read = false")
    reg = jl(V / "contracts/manifest_registry/REGISTRY_V2.json")
    if reg["self_sha256"] != r["registry"]["self_sha256"] or SERVER_SHA256["contracts/manifest_registry/REGISTRY_V2.json"] != r["registry"]["file_sha256"]:
        raise SystemExit("REGISTRY_V2 does not match the receipt")
    fs = jl(V / "runs_fs/FS_FINAL.json")
    if fs.get("action") != "LOCK" or fs.get("schema_id") != S_STAR:
        raise SystemExit("FS_FINAL is not a LOCK on S*")
    schema = jl(V / f"contracts/derived_schemas/{S_STAR}.json")
    dreg = jl(V / "contracts/derived_schemas/REGISTRY.json")
    ent = dreg["schemas"].get(S_STAR)
    if ent is None or schema["self_sha256"] not in json.dumps(ent):
        raise SystemExit("S* self-hash is not the registered one")
    v1 = jl(V / "contracts/DATA_PROTOCOL_LOCK.json")
    contracts = {n: V / "contracts" / n for n in ("DATA_CONTRACT_CLEAN_V2.md", "FEATURE_SCHEMA_CLEAN_V2.json", "EVALUATION_CONTRACT_V4.md",
                                                  "CLIENT_CONTRACT_V1.md", "MANIFEST_SPEC_V4.json", "MANIFEST_SPEC_V4_ADDENDUM_01.json",
                                                  "PREG1_LOCK_VALUES.json", "PREG1_AMENDMENT_01.md")}
    csha = {n: sha(p) for n, p in contracts.items()}
    changed_vs_v1 = {n: [v1["contracts_sha256"][n], csha[n]] for n in v1["contracts_sha256"] if n in csha and v1["contracts_sha256"][n] != csha[n]}
    reviews = {p.name: sha(p) for p in sorted((V / "company/reviews").glob("POPULATION_5_100_*.md"))}
    adrs = {p.name: sha(p) for p in sorted((V / "company/decisions").glob("ADR-*.md"))}
    fd_ = fs["final_decision"]
    acc = fd_["acceptance"][S_STAR]
    lock = {
        "document": "DATA_PROTOCOL_LOCK", "version": "2.0.0", "project": "PPSI V4 single-backbone",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "written_by": "Operations Director (Claude), after Level-1 review",
        "supersedes": {"file": "contracts/DATA_PROTOCOL_LOCK.json", "version": v1["version"], "lock_self_sha256": v1["lock_self_sha256"],
                       "file_sha256": sha(V / "contracts/DATA_PROTOCOL_LOCK.json"),
                       "scope": "population_and_manifests (N_MAIN 262,144 / N_CALIB 65,536 / eligibility >= 1) and every manifest built on it are HISTORICAL "
                                "(DECISION_LOG 31); splits, labels, catalogue A, transform state and PREG1 values carry over unchanged",
                       "v1_file_unmodified": True},
        "authority": ["Owner amendment POP 5-100 (DECISION_LOG 31) and owner decisions DECISION_LOG 35-36, 47-48",
                      "Fable POPULATION_5_100 FINAL R2 + R4 (owner-proxy; binding per DECISION_LOG 43) = ADR-005 + addendum 2",
                      "Feature search LOCK, mechanical D3 rule (DECISION_LOG 53) = ADR-005 addendum 3",
                      "[PROXY] items pending owner ratification: fast-search rule; D5 full-TRAIN statistics; unseen-user protocol and sizes; EQUIVALENT -> GRU on cost"],
        "basis": "SERVER DATA-06 receipt + server-registered S* (not local-build evidence)",
        "server_receipt": {"path": "ppsi_v4/data_v3/receipts/DATA06_SERVER_RECEIPT.json",
                           "sha256_server_measured": SERVER_SHA256["data_v3/receipts/DATA06_SERVER_RECEIPT.json"],
                           "receipt_self_sha256": r["receipt_sha256"], "window_utc": r["date_u"], "checks_all_true": len(r["checks"]),
                           "build_receipt_sha256": r["build_receipt_sha256"], "tests": {k: r["tests"][k] for k in ("exit_code", "counts", "negative_controls")},
                           "rss_guard_peak_mib": r["rss_guards"][0]["peak_rss_mib"], "test_split_read": r["test_split_read"]},
        "data_version": {"name": reg["data_version"], "base": v1["data_version"]["base"], "base_version_hash": v1["data_version"]["base_version_hash"],
                         "data_v3_bytes": r["data_v3_bytes"]},
        "splits_and_labels": v1["splits_and_labels"],
        "catalogue": v1["catalogue"],
        "history_policy": {"id": "USER_HISTORY_L50", "rule": "last 50 representative events of the user's whole prior stream, across sessions and partition fences, "
                                                             "strictly causal, identical for TRAIN / VALIDATION / TEST; targets keep the same-partition rule",
                           "diagnostic_only": "SESSION_L50 (v1 session-scoped window)", "authority": "DECISION_LOG 36"},
        "features_and_transforms": {
            "base_schema": {"version": "FEATURE_SCHEMA_CLEAN_V2 2.1.0", "file_sha256": csha["FEATURE_SCHEMA_CLEAN_V2.json"]},
            "production_schema": {"schema_id": S_STAR, "self_sha256": schema["self_sha256"], "file_sha256": SERVER_SHA256[f"contracts/derived_schemas/{S_STAR}.json"],
                                  "n_information_channels": schema["n_information_channels"], "widths": schema["widths"],
                                  "dropped_groups": schema["source"]["dropped_groups"], "dropped_channels": schema["source"]["dropped_channels"],
                                  "required_channels": ["item_tokens", "category_tokens", "session_offset_tokens"],
                                  "derived_schema_registry_self_sha256": dreg["self_sha256"],
                                  "derived_schema_registry_file_sha256": SERVER_SHA256["contracts/derived_schemas/REGISTRY.json"],
                                  "theta0_registry_file_sha256": SERVER_SHA256["contracts/derived_schemas/THETA0_REGISTRY.json"]},
            "feature_search_evidence": {"FS_FINAL_sha256": SERVER_SHA256["runs_fs/FS_FINAL.json"], "FS_FINAL_utc": fs["utc"],
                                        "DECISION_A_sha256": SERVER_SHA256["runs_fs/stageB/DECISION_A.json"],
                                        "DECISION_B_sha256": SERVER_SHA256["runs_fs/stageB/DECISION_B.json"],
                                        "rule": acc["rule"], "worse_family": acc["worse_family"], "per_family_relative_drop_pct": acc["per_family"],
                                        "restored_group": fs["restored_group"], "run_label": fd_["run_label"], "statistic": fd_["statistic"],
                                        "publication": "relative drops only until ARCHITECTURE_LOCK"},
            "shared_statistics": reg["transform_state"],
        },
        "population_and_manifests": {"selection_salt": "PPSI_V4_POP_5_100_SEL_20260926", "eligibility": "5 <= n_eligible_TRAIN_examples <= 100 (TRAIN-only)",
                                     "population_bands": reg["population_bands"], "roles": reg["roles"],
                                     "manifest_registry": {"path": "contracts/manifest_registry/REGISTRY_V2.json", "self_sha256": reg["self_sha256"],
                                                           "file_sha256": SERVER_SHA256["contracts/manifest_registry/REGISTRY_V2.json"]},
                                     "manifests": r["manifests"], "n_manifests": len(r["manifests"]),
                                     "test_manifests": "NOT BUILT; refused until ALL_METHODS_LOCK",
                                     "known_tolerated_deviation": "calib_prefix_is_first_65536 is an inherited historical assertion; CALIB is selected by key v2 (DECISION_LOG 50)"},
        "code": {"data06": r["code"], "finalize_code_sha256": r["finalize_code_sha256"], "registry_builder_code_sha256": reg["builder_code_sha256"]},
        "contracts_sha256": csha, "contracts_changed_since_v1": changed_vs_v1,
        "fable_rulings_and_syntheses_sha256": reviews, "adrs_sha256": adrs,
        "preg1_lock_values": v1["preg1_lock_values"],
        "not_locked_here": ["peak LR (LR calibration on CALIB under S*, before the contest)",
                            "model topology hashes and per-seed init hashes (ARCHITECTURE_LOCK after G2)",
                            "TEST manifests (after ALL_METHODS_LOCK)", "FL/PFL/LO method settings beyond ADR-005 (G3)"],
    }
    canon = json.dumps(lock, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    lock["lock_self_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    lock["lock_self_sha256_rule"] = "sha256 of json.dumps(lock without this key and without lock_self_sha256_rule, sort_keys=True, ensure_ascii=True, separators=(',',':'))"
    out = V / "contracts" / "DATA_PROTOCOL_LOCK_V2.json"
    fd = os.open(str(out), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(lock, indent=1, ensure_ascii=False, allow_nan=False))
    print(json.dumps({"lock": str(out), "lock_self_sha256": lock["lock_self_sha256"], "file_sha256": sha(out), "contracts_changed_since_v1": sorted(changed_vs_v1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

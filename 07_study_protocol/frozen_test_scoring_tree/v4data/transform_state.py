"""v4data.transform_state (owner: Data Lead, DATA-04) - writes and verifies data_v2/TRANSFORM_STATE_CLEAN_V2.json.

Field list: FEATURE_SCHEMA_CLEAN_V2.json `transform_state_v2_fields` (schema 2.1.0) and DATA_CONTRACT_CLEAN_V2 section 10.
The state holds only frozen values copied from the verified schema plus input and runtime-module sha256s; it
fits nothing (every fitted value was produced upstream and is quoted from its receipt).
Self-hash: `transform_state_sha256` = sha256 of the canonical JSON (common.canonical_json_bytes) with that one
field removed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

from .common import (CODE, DATASET_CODE_HASH, SCHEMA_SELF_SHA256, SCHEMA_VERSION, V4, V4DATA, VERSION_HASH,
                     atomic_write_json, builder_code_sha256, contract_path, data_v2_root, dataset_root, load_schema,
                     self_hash, sha256_file, utc_now)
from .transform import build_spec

STATE_REL = "TRANSFORM_STATE_CLEAN_V2.json"

V1_INPUTS = ("VERSION.json", "FILES.json", "mappings/normalizers.json", "model_ready/FEATURE_GROUPS.json",
             "splits/splits.json")
RUNTIME_MODULES = ("snap_v1/reuse/vendor/rees46_eda/loaders.py", "snap_v1/reuse/vendor/rees46_eda/config.py",
                   "snap_v1/reuse/full_store.py", "snap_v1/reuse/index_utils.py", "ppsi_env.py")


def _runtime_module_hashes() -> dict:
    out = {rel: sha256_file(CODE / rel) for rel in RUNTIME_MODULES}
    for p in sorted(V4DATA.glob("*.py")):
        out["v4data/" + p.name] = sha256_file(p)
    return out


def build_state(ds_root: Optional[Path] = None, schema: Optional[Mapping] = None) -> dict:
    s = schema if schema is not None else load_schema()
    spec = build_spec(s)
    ds_root = Path(ds_root) if ds_root is not None else dataset_root()
    version = json.loads((ds_root / "VERSION.json").read_text(encoding="utf-8"))
    if version.get("version_hash") != VERSION_HASH:
        raise ValueError(f"dataset version_hash {version.get('version_hash')} != {VERSION_HASH}")
    files_manifest = {e["path"]: e["sha256"] for e in json.loads(
        (ds_root / "FILES.json").read_text(encoding="utf-8"))["files"]}
    v1_inputs = {}
    for rel in V1_INPUTS:
        got = sha256_file(ds_root / rel)
        rec = files_manifest.get(rel)
        if rec is not None and rec != got:
            raise ValueError(f"v1 input {rel} sha256 {got} != FILES.json {rec}")
        v1_inputs[rel] = got
    m1 = s["m1_reencoding"]
    f1 = s["f1_reencoding"]
    lay = s["tensor_layout_v2"]
    drops = []
    for c in s["channels"]:
        if c["decision"] == "DROP_REDUNDANT":
            drops.append({"name": c["name"], "v1_slot": c["v1_slot"], "reconstruction": c["reconstruction"]["expression"],
                          "full_population_contradictions": c["full_population_contradictions"],
                          "runtime_check": {"log1p_prefix_elapsed_s": "RC1", "prefix_item_count": "RC2",
                                            "local_hour_sin": "RC3", "local_hour_cos": "RC3",
                                            "has_history_mask": "RC4", "prev_user_event_available": "RC5"}[c["name"]]})
    contract = contract_path("DATA_CONTRACT_CLEAN_V2.md")
    state = {
        "schema_version": SCHEMA_VERSION,
        "view_id": "clean_v2_rich",
        "status": "BUILT_PENDING_DATA_PROTOCOL_LOCK",
        "created_utc": utc_now(),
        "created_by_task": "DATA-04",
        "v1_version_hash": VERSION_HASH,
        "v1_dataset_code_hash": DATASET_CODE_HASH,
        "v1_files_manifest_sha256": v1_inputs["FILES.json"],
        "v1_input_sha256": v1_inputs,
        "base_normalizers_sha256": v1_inputs["mappings/normalizers.json"],
        "schema_sha256_file": sha256_file(contract_path("FEATURE_SCHEMA_CLEAN_V2.json")),
        "schema_self_sha256": SCHEMA_SELF_SHA256,
        "contract_sha256": sha256_file(contract) if contract.is_file() else None,
        "runtime_module_sha256": _runtime_module_hashes(),
        "builder_code_sha256": builder_code_sha256(),
        "v1_input_schema_sha256": spec.input_schema_sha256,
        "fit": {"population": "FULL_TRAIN", "split": "TRAIN", "buckets": "0..63",
                "grain": "one unweighted row per TRAIN feature row (canonical representative = decision)",
                "estimator": "clip at recorded bounds; arithmetic mean; population std (ddof 0)",
                "rows_all": m1["features"]["log1p_time_since_prev_user_event_s"]["n_all_train"],
                "source": "fitted upstream and quoted: M1 from feature_opt_v1 M1_FIT.json (reproduced by QA-02 "
                          "OI04_M1_REPRODUCTION.json), F-1 from QA-02 OI03_F1_PRICE_REZERO.json; DATA-04 fits nothing",
                "source_receipt_sha256": {k: v for k, v in s["base"]["receipt_hashes_sha256"].items()
                                          if "OI03" in k or "OI04" in k},
                "m1_fit_json_sha256": s["base"]["input_file_hashes_sha256"]["FOI/audit/M1_FIT.json"]},
        "m1": {name: {"mask": e["governing_mask_stored"], "clip_lo": e["clip_lo"], "clip_hi": e["clip_hi"],
                      "old_mean": e["old_mean"], "old_std": e["old_std"], "new_mean": e["new_mean"],
                      "new_std": e["new_std"], "n_valid": e["n_valid_train"], "unavailable_encoding": 0.0,
                      "rule": m1["formula"],
                      "reproduction_receipt": s["m1_reproduction_receipt"]["result"]}
               for name, e in m1["features"].items()},
        "f1": {"feature": f1["feature"], "gate_predicate": "unknown := recent_price_supported == 0 OR price_missing "
                                                            "OR price_conflict at last_valid_index",
               "clip_lo": f1["clip_lo"], "clip_hi": f1["clip_hi"], "old_mean": f1["old_mean"],
               "old_std": f1["old_std"], "new_mean": f1["new_mean"], "new_std": f1["new_std"],
               "n_valid": f1["n_valid"], "unavailable_encoding": 0.0, "rule": f1["formula"],
               "receipt_sha256": s["base"]["receipt_hashes_sha256"]["PPSI/g0/receipts_qa02/OI03_F1_PRICE_REZERO.json"]},
        "normalized_channels_v2": s["normalized_channels_v2"],
        "rezero_map": s["rezero_map_v2"],
        "drops": drops,
        "layout": {k: {"names": lay[k]["names"], "v1_source_indices": lay[k]["v1_source_indices"]}
                   for k in ("event_numeric_features", "event_quality_flags", "user_context", "user_context_masks")},
        "widths": {"event_numeric_features": 15, "event_quality_flags": 15, "user_context": 12,
                   "user_context_masks": 4, "channels_total": 53},
        "runtime_identity_checks": s["runtime_identity_checks"],
        "shared_train_statistics": [dict(o, identical_across_methods=True) for o in s["shared_train_statistics"]],
        "m1_application_path": "EXACT_INVERSE (snapshot CompactStore encodes with the v1 normalizers and cannot "
                               "inject; the adapter inverts the v1 z of the 3 re-encoded columns in float64); the "
                               "INJECTED reference encoder v4data.transform.encode_rows is the formula of record",
        "equivalence": None,
        "prior_lean_gate": "PARITY_FAILED (unchanged; not re-evaluated)",
        "no_test_rows_read": True,
    }
    state["transform_state_sha256"] = self_hash(state, "transform_state_sha256")
    return state


def write_state(out_root: Optional[Path] = None, ds_root: Optional[Path] = None) -> tuple[Path, dict]:
    out_root = Path(out_root) if out_root is not None else data_v2_root()
    state = build_state(ds_root)
    p = atomic_write_json(out_root, STATE_REL, state)
    back = load_state(p)
    if back["transform_state_sha256"] != state["transform_state_sha256"]:
        raise ValueError("TRANSFORM_STATE round-trip changed its self-hash")
    return p, state


def load_state(path: Path, *, verify_against_schema: bool = True) -> dict:
    st = json.loads(Path(path).read_text(encoding="utf-8"),
                    parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    rec = st.get("transform_state_sha256")
    got = self_hash(st, "transform_state_sha256")
    if rec != got:
        raise ValueError(f"TRANSFORM_STATE self-hash mismatch: recorded {rec}, recomputed {got}")
    if st.get("schema_self_sha256") != SCHEMA_SELF_SHA256 or st.get("v1_version_hash") != VERSION_HASH:
        raise ValueError("TRANSFORM_STATE was built for a different schema or dataset")
    if verify_against_schema:
        spec = build_spec()
        if st["v1_input_schema_sha256"] != spec.input_schema_sha256:
            raise ValueError("TRANSFORM_STATE input schema hash differs from the schema's")
        for name, e in spec.reencode.items():
            blk = st["f1"] if name == st["f1"]["feature"] else st["m1"][name]
            if (blk["new_mean"], blk["new_std"]) != e["new"] or (blk["old_mean"], blk["old_std"]) != e["old"]:
                raise ValueError(f"TRANSFORM_STATE values for {name} differ from the frozen schema")
    return st


assert V4.name == "ppsi_v4"

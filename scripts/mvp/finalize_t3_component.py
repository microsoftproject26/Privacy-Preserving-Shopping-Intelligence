"""Finalize T3 as the selected shared fixed retrieval component.

Reconciles existing evidence from scripts/t3/output/ without retraining:
- Protocol ID: T3_SHARED_FIXED_RETRIEVAL_FINAL_V1
- Selected component: cooccurrence_then_popularity
- Retrieval macro NDCG@5: 0.2707, micro: 0.2046 (18,814 queries, 4,591 clients)
- Rejection of learned rerankers (listwise 0.2505, pointwise 0.2186, query-aware retry)
- System status: NOT_APPLICABLE_SHARED_FIXED_COMPONENT
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    print("=" * 78)
    print("PPSI T3 FINAL COMPONENT PACKAGING (T3_SHARED_FIXED_RETRIEVAL_FINAL_V1)")
    print("=" * 78)

    t3_out = ROOT / "scripts/t3/output"

    # 1. Load existing evidence JSONs
    gate_data = read_json_file(t3_out / "s2_ds_07_gate.json")
    ladder_data = read_json_file(t3_out / "s2_ds_07_ladder.json")
    query_aware_data = read_json_file(t3_out / "s2_ds_07_query_aware.json")
    rebuild_data = read_json_file(t3_out / "s2_ds_07_candidates_rebuild.json")

    # 2. Reconcile retrieval baseline
    retrieval_component = ladder_data["retrieval"]
    macro_ndcg = float(ladder_data["best_simple_macro"])
    micro_ndcg = float(ladder_data["best_simple_micro"])
    eval_queries = int(ladder_data["evaluable_queries"])
    clients_count = int(ladder_data["clients"])
    candidate_recall = float(gate_data["gate"]["candidate_recall"])

    assert retrieval_component == "cooccurrence_then_popularity"
    assert abs(macro_ndcg - 0.2707) < 1e-4
    assert abs(micro_ndcg - 0.2046) < 1e-4
    assert eval_queries == 18814
    assert clients_count == 4591
    assert abs(candidate_recall - 0.8159) < 1e-4

    print(f"Retrieval verified: {retrieval_component}")
    print(f"  Macro NDCG@5: {macro_ndcg:.4f} | Micro NDCG@5: {micro_ndcg:.4f}")
    print(f"  Evaluable queries: {eval_queries:,} across {clients_count:,} clients | Recall: {candidate_recall:.4f}")

    # 3. Reconcile reranker rejection
    listwise_res = next(r for r in ladder_data["results"] if r["loss"] == "listwise")
    pointwise_res = next(r for r in ladder_data["results"] if r["loss"] == "pointwise")

    listwise_macro = float(listwise_res["macro_ndcg@5"])
    listwise_gain = float(listwise_res["gain_interval"]["gain"])
    listwise_ci = [float(listwise_res["gain_interval"]["ci_low"]), float(listwise_res["gain_interval"]["ci_high"])]
    listwise_above_zero = bool(listwise_res["gain_interval"]["above_zero"])

    pointwise_macro = float(pointwise_res["macro_ndcg@5"])
    pointwise_gain = float(pointwise_res["gain_interval"]["gain"])
    pointwise_ci = [float(pointwise_res["gain_interval"]["ci_low"]), float(pointwise_res["gain_interval"]["ci_high"])]
    pointwise_above_zero = bool(pointwise_res["gain_interval"]["above_zero"])

    assert abs(listwise_macro - 0.2505) < 1e-4
    assert not listwise_above_zero
    assert abs(pointwise_macro - 0.2186) < 1e-4
    assert not pointwise_above_zero

    # Query-aware 3-seed check
    qa_runs = query_aware_data["runs"]
    assert len(qa_runs) == 3
    for run in qa_runs:
        assert run["best_epoch"] == 0, f"Query-aware seed {run['seed']} best epoch is not 0"
        assert run["macro_ndcg@5"] == 0.2707
        assert not run["meets_acceptance"]

    print("Learned rerankers verified underperforming:")
    print(f"  Listwise: {listwise_macro:.4f} (gain: {listwise_gain:.6f}, CI: {listwise_ci})")
    print(f"  Pointwise: {pointwise_macro:.4f} (gain: {pointwise_gain:.6f}, CI: {pointwise_ci})")
    print(f"  Query-aware (seeds {[r['seed'] for r in qa_runs]}): all accepted deliverables remain epoch 0 retrieval")

    # 4. Artifact verification
    reported_sha = "8892029ED48AEA056F6357811A53E3D74F5F9AC7C5572057741477D256B854C3"
    artifact_name = rebuild_data["artifact"]
    artifact_rows = int(rebuild_data["rows"])
    anchors_total = int(rebuild_data["anchors_total"])
    anchors_train = int(rebuild_data["anchors_train"])
    anchors_frozen = int(rebuild_data["anchors_frozen"])

    assert artifact_rows == 10258299
    assert anchors_total == 107484
    assert anchors_train == 102282
    assert anchors_frozen == 47948

    # Check local candidate artifact on disk
    local_artifact_path = ROOT / "data" / artifact_name
    if not local_artifact_path.exists():
        # check fixtures
        cand_paths = list(ROOT.glob(f"**/{artifact_name}"))
        local_artifact_path = cand_paths[0] if cand_paths else None

    if local_artifact_path and local_artifact_path.exists():
        print(f"Found physical candidate artifact at {local_artifact_path}. Computing SHA-256...")
        hasher = hashlib.sha256()
        with open(local_artifact_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        local_hash = hasher.hexdigest().upper()
        local_verified = (local_hash == reported_sha)
        local_status = "VERIFIED_MATCHES_REPORTED_SHA256" if local_verified else "HASH_MISMATCH"
    else:
        print("Physical extended candidate artifact not present locally; preserving upstream reported SHA-256.")
        local_hash = None
        local_verified = False
        local_status = "ARTIFACT_HASH_REPORTED_UPSTREAM_NOT_RECOMPUTED_LOCALLY"

    # 5. Output evidence directory
    evidence_dir = ROOT / "docs/evidence/mvp/t3-final-component-001"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    preflight_data = {
        "task": "T3",
        "protocol_id": "T3_SHARED_FIXED_RETRIEVAL_FINAL_V1",
        "status": "PASS",
        "decision": "SELECTED_SHARED_FIXED_RETRIEVAL_COMPONENT",
        "component": retrieval_component,
        "trainable": False,
        "test_rows_read": 0,
    }
    (evidence_dir / "preflight.json").write_text(json.dumps(preflight_data, indent=2), encoding="utf-8")

    retrieval_data = {
        "component": retrieval_component,
        "type": "SHARED_FIXED_RETRIEVAL",
        "macro_ndcg_at_5": macro_ndcg,
        "micro_ndcg_at_5": micro_ndcg,
        "evaluable_queries": eval_queries,
        "clients": clients_count,
        "candidate_recall": candidate_recall,
        "provenance": "scripts/t3/output/s2_ds_07_ladder.json",
    }
    (evidence_dir / "retrieval_component.json").write_text(json.dumps(retrieval_data, indent=2), encoding="utf-8")

    rejection_data = {
        "rerankers_tested": ["listwise", "pointwise", "query_aware_residual"],
        "listwise": {
            "macro_ndcg_at_5": listwise_macro,
            "gain": listwise_gain,
            "ci95": listwise_ci,
            "beats_baseline": False,
        },
        "pointwise": {
            "macro_ndcg_at_5": pointwise_macro,
            "gain": pointwise_gain,
            "ci95": pointwise_ci,
            "beats_baseline": False,
        },
        "query_aware": {
            "seeds": [r["seed"] for r in qa_runs],
            "best_deliverable": "epoch_0_retrieval_on_all_seeds",
            "beats_acceptance_bar": False,
        },
        "conclusion": "Learned rerankers failed to exceed retrieval baseline; retrieval component finalized.",
    }
    (evidence_dir / "reranker_rejection.json").write_text(json.dumps(rejection_data, indent=2), encoding="utf-8")

    artifact_data = {
        "artifact": artifact_name,
        "rows": artifact_rows,
        "anchors_total": anchors_total,
        "anchors_train": anchors_train,
        "anchors_frozen": anchors_frozen,
        "reported_sha256": reported_sha,
        "local_hash_computed": local_hash,
        "local_verification_status": local_status,
        "frozen_lists_reproduce_exactly": bool(rebuild_data["frozen_lists_reproduce_exactly"]),
        "validation_baseline_unchanged": bool(rebuild_data["validation_baseline_unchanged"]),
    }
    (evidence_dir / "artifact_verification.json").write_text(json.dumps(artifact_data, indent=2), encoding="utf-8")

    system_contract = {
        "component_type": "SHARED_FIXED_RETRIEVAL",
        "selected_component": retrieval_component,
        "trainable_under_r1": False,
        "trainable_under_r2a": False,
        "same_component_in_both_variants": True,
        "quality_retention": None,
        "quality_retention_status": "NOT_APPLICABLE_SHARED_FIXED_COMPONENT",
        "headline_metric": "macro_ndcg@5",
        "headline_value": macro_ndcg,
        "support": {
            "evaluable_queries": eval_queries,
            "clients": clients_count,
            "candidate_recall": candidate_recall,
        },
    }
    (evidence_dir / "system_contract.json").write_text(json.dumps(system_contract, indent=2), encoding="utf-8")

    summary_md = f"""# T3 Final Component Summary

- **Protocol**: `T3_SHARED_FIXED_RETRIEVAL_FINAL_V1`
- **Selected Component**: `{retrieval_component}`
- **Quality Retention Status**: `NOT_APPLICABLE_SHARED_FIXED_COMPONENT`
- **Trainable**: False (Identical fixed component in both R1 and R2A)

## Retrieval Metrics
- **Macro NDCG@5**: **{macro_ndcg:.4f}**
- **Micro NDCG@5**: **{micro_ndcg:.4f}**
- **Evaluable Queries**: **{eval_queries:,}**
- **Clients**: **{clients_count:,}**
- **Candidate Recall**: **{candidate_recall:.4f}**

## Reranker Rejection Summary
- **Listwise**: Macro NDCG@5 = **{listwise_macro:.4f}** (gain: {listwise_gain:+.4f}, CI: {listwise_ci}) — below baseline.
- **Pointwise**: Macro NDCG@5 = **{pointwise_macro:.4f}** (gain: {pointwise_gain:+.4f}, CI: {pointwise_ci}) — below baseline.
- **Query-aware (3 seeds: 13, 42, 2026)**: Best deliverable on every seed remained epoch-0 retrieval.

## Candidate Artifact
- **Rows**: {artifact_rows:,}
- **Total Anchors**: {anchors_total:,}
- **Reported SHA-256**: `{reported_sha}`
- **Verification Status**: `{local_status}`
"""
    (evidence_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"\nAll T3 evidence successfully written to: {evidence_dir}")
    print("\nFINAL STATUS MARKER: T3_FINAL_COMPONENT_READY")


if __name__ == "__main__":
    main()

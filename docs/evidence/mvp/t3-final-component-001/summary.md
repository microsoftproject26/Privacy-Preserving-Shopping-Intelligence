# T3 Final Component Summary

- **Protocol**: `T3_SHARED_FIXED_RETRIEVAL_FINAL_V1`
- **Selected Component**: `cooccurrence_then_popularity`
- **Quality Retention Status**: `NOT_APPLICABLE_SHARED_FIXED_COMPONENT`
- **Trainable**: False (Identical fixed component in both R1 and R2A)

## Retrieval Metrics
- **Macro NDCG@5**: **0.2707**
- **Micro NDCG@5**: **0.2046**
- **Evaluable Queries**: **18,814**
- **Clients**: **4,591**
- **Candidate Recall**: **0.8159**

## Reranker Rejection Summary
- **Listwise**: Macro NDCG@5 = **0.2505** (gain: -0.0202, CI: [-0.026522, -0.01418]) — below baseline.
- **Pointwise**: Macro NDCG@5 = **0.2186** (gain: -0.0521, CI: [-0.058424, -0.045774]) — below baseline.
- **Query-aware (3 seeds: 13, 42, 2026)**: Best deliverable on every seed remained epoch-0 retrieval.

## Candidate Artifact
- **Rows**: 10,258,299
- **Total Anchors**: 107,484
- **Reported SHA-256**: `8892029ED48AEA056F6357811A53E3D74F5F9AC7C5572057741477D256B854C3`
- **Verification Status**: `ARTIFACT_HASH_REPORTED_UPSTREAM_NOT_RECOMPUTED_LOCALLY`

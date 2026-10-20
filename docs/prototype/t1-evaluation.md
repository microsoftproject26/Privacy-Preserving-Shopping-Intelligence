# T1 Evaluation Harness (S2-PR-07 / #31)

This document describes the canonical Phase 1 T1 evaluation harness implemented in `ppsi.evaluation.t1`.

## 1. Scope and Purpose

The T1 evaluation harness provides a deterministic, reproducible evaluation pipeline that all downstream Phase 1 models share:
- #32: T1 Baselines (Popularity, First-Order Markov, Last-Category)
- #30 / #33: Phase 1 GRU model and Centralized Baseline Pilot
- #54: Final R1 centralized baseline training and evaluation
- Federated model evaluations (R2A)

The evaluator does **not** train any model, does not access sealed test data, and does not compute unapproved metrics.

## 2. Frozen Upstream Semantics

- **Vocabulary**: Exactly 588 dense categories (`0..587`) defined in `docs/evidence/s1-d1-ds-07/vocabulary_v1.proposed.json`.
- **Label Semantics**: `label_value` in TaskExample parquets is already the dense TRAIN category code (integer in `[0, 587]`). It is never mapped through raw category IDs.
- **Evaluation Population**: Validated on frozen C1 VALIDATION TaskExamples (`438,185` decisions across `31,576` clients).

## 3. Metric Definitions

For each decision with 1-based target rank $r$:
- **Accuracy@1**:
  $$\text{Acc@1}(r) = \mathbb{I}[r = 1]$$
- **MRR@20**:
  $$\text{MRR@20}(r) = \begin{cases} \frac{1}{r} & \text{if } r \le 20 \\ 0 & \text{if } r > 20 \end{cases}$$

Cutoff at 20 is strictly enforced; untruncated MRR is not computed.

## 4. Aggregation Rules

### Micro
Mean over all decisions in the slice:
$$\text{Metric}_{\text{micro}} = \frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \text{contrib}(d)$$
Support is the number of decisions.

### Macro
One vote per client with support in the slice:
1. Average contributions within each client:
   $$\mu_c = \frac{1}{|\mathcal{D}_c|} \sum_{d \in \mathcal{D}_c} \text{contrib}(d)$$
2. Average client-level means across clients present in the slice:
   $$\text{Metric}_{\text{macro}} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \mu_c$$
Support is the number of unique clients. Clients with no decisions in the slice are excluded from the denominator (not assigned zero).

### Empty Slice
If a requested slice has zero decisions, the evaluator returns `ZERO_SUPPORT` and emits no numeric metric.

## 5. Slices and Hierarchy

1. **`next_distinct` (`category_changed == true`)**:
   - **Headline Regime-Comparison Slice** (ADR-001 / Gate G1).
   - Headline metric: `t1.next_distinct.mrr_at_20.macro`.
   - Micro metric `t1.next_distinct.mrr_at_20.micro` is reported alongside.
   - Accuracy@1 (macro and micro) is also emitted.
2. **`overall` (all valid decisions)**:
   - **Product Diagnostic Only**.
   - Emits macro and micro MRR@20 and Accuracy@1.
   - Never used as the primary regime-comparison headline.

## 6. Target-Rank and Tie-Breaking Semantics

- **Scores input**: Tensor `[N, 588]`.
- **Tie Policy**: `ASCENDING_DENSE_CATEGORY_CODE`.
  When scores are equal, lower dense category codes win.
  $$\text{rank}(t) = 1 + \sum_{c} \mathbb{I}[\text{score}_c > \text{score}_t] + \sum_{c < t} \mathbb{I}[\text{score}_c = \text{score}_t]$$
- **Ranked-list input**: Each row must be a strict permutation of `0..587`. Duplicates, partial rankings, or out-of-range categories raise errors.

## 7. History Stratification

Stratification buckets follow ADR-001:
- `10_19` (10–19 TRAIN events)
- `20_49` (20–49 TRAIN events)
- `50_99` (50–99 TRAIN events)
- `100_plus` (100+ TRAIN events)
- `BELOW_10_RETAINED_C1` (0–9 TRAIN events retained post-boundary edge case)

The evaluator requires valid external TRAIN event counts and never infers or substitutes TaskExample row counts. If history counts are absent, history stratification is marked `NOT_AVAILABLE`.

## 8. Validation CLI and Evidence

Run validation CLI:
```powershell
uv run --locked python scripts/evaluation/validate_t1_evaluator.py `
  --config config/evaluation/t1_evaluator.v1.json
```

Generated public evidence files:
- `docs/evidence/s2-pr-07/t1_metric_spec.v1.json`
- `docs/evidence/s2-pr-07/t1_hand_worked_metric_records.v1.json`
- `docs/evidence/s2-pr-07/t1_evaluator_validation.v1.json`

## 9. Decision and Approval State

Edge semantics are recorded in `docs/decisions/t1-evaluator-edge-semantics.md`.
Status is `ACCEPTED`. All three required approvals (Eid Abdelrihem, Ahmed Abdelhameed,
Ahmed Sherif) are recorded in that document.

Approval evidence wording: team approval confirmed by project owner on 2026-09-10; no external
permalink supplied.

The generated evidence carries the same state in `freeze_status`, which the validation CLI derives
from the decision document rather than from a hand-edited constant. If the decision document is ever
returned to a pending state, the CLI falls back to `PROPOSED_PENDING_TEAM_APPROVAL` on the next run.

### History bucket attribution

`ADR-001` freezes exactly four TRAIN-history buckets: `10_19`, `20_49`, `50_99`, `100_plus`.

`BELOW_10_RETAINED_C1` is **not** an `ADR-001` bucket. It is the approved S2-PR-07 edge extension
(rule 12) covering clients retained in `C1` whose supplied post-boundary usable TRAIN-event count is
below 10. Those clients are neither dropped nor folded into `10_19`; they are disclosed separately.
`ADR-001` itself is unchanged.

The metric spec makes this split explicit through `adr_001_history_buckets` and
`approved_edge_history_bucket`, so a downstream consumer never has to infer provenance from the
combined `history_buckets` list.

## 10. Identity and Hash Conventions

Downstream evidence reuses the **owning upstream artifact's established identity convention** rather
than imposing a single new convention across the repository. Within one `source_references` block
this means more than one convention can legitimately appear:

| Artifact kind | Convention | Example |
|---|---|---|
| Frozen vocabulary | Upstream raw-file SHA-256, as first recorded by the owning S1 issue | `docs/evidence/s1-d1-ds-07/vocabulary_v1.proposed.json` |
| Repository text artifacts | Canonical text SHA-256 with normalized line endings | `docs/decisions/ADR-001-evaluation-protocol.md` |
| Private TaskExample Parquet | Existing binary/raw SHA-256 | `data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_t1_v1.proposed.parquet` |

The vocabulary hash recorded here is byte-identical to the value already pinned by S1-PR-07 evidence,
which is the point: one frozen artifact keeps one identity across the whole project. Do not rewrite
upstream artifacts to make these values look uniform, and when verifying a pin, use the convention
that the owning artifact uses.

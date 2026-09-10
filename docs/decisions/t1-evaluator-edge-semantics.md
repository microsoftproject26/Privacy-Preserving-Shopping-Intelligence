# T1 Evaluator Edge Semantics v1

| Field | Value |
|---|---|
| Issue | `S2-PR-07 / #31` |
| Status | `ACCEPTED` |
| Applies to | `T1 evaluator v1` |

## Approved rules

1. Full score universe is exactly the frozen 588 dense category codes.
2. T1 `label_value` is already a dense code in `0..587`.
3. Exact score ties are ordered by ascending dense category code.
4. Non-finite scores/labels are errors.
5. Target outside the vocabulary is an error.
6. Ranked-list inputs must be a full permutation of the frozen vocabulary; duplicates, unknowns, missing categories, and partial rankings are errors.
7. Empty slices return explicit `ZERO_SUPPORT`; no numeric metric is fabricated.
8. Macro means one vote per client with at least one decision in the requested slice.
9. Micro means one vote per decision.
10. History buckets use valid supplied TRAIN event counts only; the evaluator never substitutes TaskExample counts.
11. Default v1 emits MRR@20 and Accuracy@1 only; secondary metrics remain deferred unless separately approved.
12. Clients retained in `C1` whose supplied post-boundary usable TRAIN-event count is below 10 are NOT dropped and are NOT reassigned to `10_19`. When history stratification is available, they are exposed separately as `BELOW_10_RETAINED_C1`. This is an S2-PR-07 edge-case extension; `ADR-001` itself remains unchanged and freezes only `10_19`, `20_49`, `50_99`, and `100_plus`.

## Relationship to ADR-001

`ADR-001` remains the accepted scientific protocol and is not modified by this document. Rule 12
adds an explicit disclosure bucket for a cohort-boundary residue; it does not alter the four
`ADR-001` history buckets, the headline metric, the headline averaging rule, or any metric formula.

## Approval record

Do not edit these rows unless the human team has actually approved the exact rules above.

| Reviewer | Required | Approval | Evidence |
|---|---:|---|---|
| Eid Abdelrihem | yes | APPROVED | Team approval confirmed by project owner on 2026-09-10; no external permalink supplied. |
| Ahmed Abdelhameed | yes | APPROVED | Team approval confirmed by project owner on 2026-09-10; no external permalink supplied. |
| Ahmed Sherif | yes | APPROVED | Team approval confirmed by project owner on 2026-09-10; no external permalink supplied. |

All three required approvals are recorded, so the status above is `ACCEPTED`.

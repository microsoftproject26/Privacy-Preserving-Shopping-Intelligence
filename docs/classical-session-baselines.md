# Classical T2 and session-based T1 baselines

The scope/search/feature/evaluation decisions are in
`decisions/s2-pr-04-05-baselines.md`. They preserve Eid's frozen TaskExamples and T1
protocol, including the upstream T2 metric exception. These are validation baselines,
not the final neural R1 denominator, R2a, or a Quality Retention result.

## Data flow

```
frozen TaskExamples + allowed raw TRAIN/VALIDATION events
    -> exact original global-order anchor binding
    -> causal features / real known session prefix
    -> TRAIN-only LR/LightGBM pipelines or session-kNN index
    -> full frozen VALIDATION membership
    -> existing result schema + public summaries + read-only notebooks
```

Censored T2 rows never become negatives. Category-change labels are used only by the
T1 evaluator, never by its predictor. History buckets use actual TRAIN event counts.

## Commands

Run from the project root on the human-created baseline branch, sequentially:

```powershell
uv run --locked python scripts/baselines/supervise_baselines.py --stage prepare --batch batch-001
uv run --locked python scripts/baselines/run_baselines.py --stage preflight --batch batch-001
uv run --locked python scripts/baselines/supervise_baselines.py --stage classical --batch batch-001
uv run --locked python scripts/baselines/supervise_baselines.py --stage session --batch batch-001
uv run --locked python scripts/baselines/run_baselines.py --stage verify --batch batch-001
```

Preparation uses its documented isolated pandas environment. Root dependencies/lock
remain unchanged. Reuse existing prepared views only after identity verification;
do not rerun preparation blindly over its existing directory. Use a new batch ID for
an explicitly diagnosed fit/evaluation retry, preserving prior attempts and logs.

## Public and private artifacts

Public: `docs/evidence/s2-pr-04-05/data_views.v1.json`, per-batch configurations,
metric summaries, source snapshots and validation results. The validated result JSONs
use the existing `artifacts/experiment-results/` registry; selective force-add may be
needed because this parent directory is ignored.

Private: raw data, prepared views, session membership, index objects, fitted pipelines,
per-decision outputs and detailed execution logs under `artifacts/baselines/`.
Do not stage private model/index files merely to satisfy a public artifact reference.

## Evaluation

T2: pooled average precision and the upstream positive-client top-three recall
companion, with history-stratified diagnostics. No macro AP or calibration claim.
T1: existing #31 evaluator with full 588-category scoring, category-change macro
MRR@20, micro alongside and overall diagnostics. kNN is a recent-session,
category-projected variant with a documented 2,000-session retrieval bound.

The notebook files display only the public evidence; they do not fit another model
or read private parquet. Until real runs complete, there are no model-quality numbers
to report. Low but valid scores are results, not permission to change the search.

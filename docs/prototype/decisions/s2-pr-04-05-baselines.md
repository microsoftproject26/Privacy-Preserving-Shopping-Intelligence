# S2-PR-04/05 — Classical baseline scope and execution decision

Date: 2026-09-11. Scope: issues #48 and #49 only.

**Authority: PROJECT_OWNER_DELEGATION.** The project owner instructed the assistant to
select necessary scoped implementation decisions and record them in the repository.
This document records those choices. It does not invent individual team signatures or
claim that Eid authored these new choices. Human PR acceptance remains separate.

## Unchanged upstream authorities

- `docs/decisions/ADR-001-evaluation-protocol.md` and frozen G1 remain unchanged.
- The current TaskExample files, their membership, masks, labels, exclusions and
  vocabulary remain unchanged. No additional modulo sampling is applied.
- `notebooks/S1_D1_DS_07_T3_Protocol_and_Task_Examples.ipynb` owns the exact pandas
  session key and the **split-global** canonical `decision_order`.
- T1 labels are dense codes, not raw category IDs. Prediction uses 588 categories;
  category-change is an evaluation slice, not information available to the predictor.
- T2 retains the upstream exception: pooled `average_precision_score` (called micro
  PR-AUC upstream) plus the positive-client top-three purchase-recall companion.
  We do not invent per-client macro PR-AUC. The companion's unit is one buyer: the
  two averages shown by the upstream `summarise` call are the same client-unit mean.
- History reporting uses actual TRAIN event counts. Counts of TaskExamples are not
  a substitute. The accepted retained-C1 edge bucket is kept separate from the four
  original ADR buckets.

## Decisions made in this bundle, before its model results

### #48: T2 C1 Logistic Regression and LightGBM

Use all **2,291,753 observed** frozen T2 TRAIN examples and all **322,087 observed**
frozen T2 VALIDATION examples. Preserve and report the masked counts; masked labels
are missing, not negative. Use identical memberships and the same thirteen explicit
features for both estimators. Feature lineage is machine-readable in
`config/baselines/t2_feature_manifest.v1.json`.

Features use the anchor's known category/price/time and strict prefixes of the current
session. They exclude user/session IDs, label maturation, session end, future events,
labels and global `decision_order` as predictors. Cumulative counts exclude the anchor.
The query's category, price and timestamp are known at the anchor.

Fit imputation, missing indicators, scaling and one-hot categories on observed TRAIN
only. Unknown VALIDATION category codes do not extend the fitted encoder. No target
encoding, oversampling, class weights, probability calibration or TEST tuning.
We report AP and client top-three recall, not a calibrated-probability claim.

The predeclared grid has three LR C values and four LightGBM leaf/regularization
settings. Other settings and training budgets are fixed in code/config before runs.
Selection is pooled VALIDATION AP separately per family. Ties within `1e-12` use the
first listed configuration. Rerun each selected configuration once from fresh state;
require full-validation probability agreement within `1e-10` absolute, zero relative.
A convergence failure is a failed attempt, not permission for a hidden larger budget.

Why T2: the issue emphasizes purchase likelihood and these decision-time features
form a valid tabular problem. T1/T3 tabular variants are not silently added. Why no
class weighting: it avoids making class-prior correction/calibration another tuning
axis. The selected validation figures remain development results, not held-out TEST
estimates.

### #49: T1 C1 recent-session cosine kNN, with category projection

Index all **788,317 TRAIN sessions** in the official example-user slice. A session is
represented as a binary item set; the query uses the last twenty actual observed
session events **including the current anchor**, then deduplicates items. Repeated
items receive no extra weight. Unknown query items remain in the cosine denominator.

For a query, retrieve up to 2,000 most recent overlapping TRAIN sessions. Compute
cosine similarity using full binary item membership, then select k from {50,100,200}.
Neighbor ties use newer session end followed by ascending fixed-width frozen session
ID. There is no further position or recency decay. Each neighbor votes its similarity
once for each category present in that session. This is an explicit **category-
projected adaptation**, not a claim to reproduce an item-level paper implementation.

Backoff for empty/no-overlap queries is the frozen T1 TRAIN target-frequency vector.
The universe always contains all 588 categories; do not suppress the current category
or pass `category_changed` to the scorer. Index and postings never learn from queries.

Use all **438,185 T1 VALIDATION decisions**, with the existing frozen T1 evaluator.
Collect ranks for the entire split before computing macro metrics. Selection is
`t1.next_distinct.mrr_at_20.macro`; near-exact ties prefer smaller k via declared list
order. Show overall diagnostics, micro, and actual TRAIN-history buckets alongside.
Share retrieval across all k values. Check saved-index readback on 1,024 fixed queries;
report that exact scope rather than claiming two complete evaluation runs.

Why this variant: transparent item-overlap retrieval, explicit finite resource bound,
full TRAIN history and an existing frozen T1 evaluator. T3 is not added while its
candidate/evaluator integrations are owned elsewhere. The recent shortlist is an
explicit approximation; do not claim exhaustive global nearest neighbors.

Reference: authors' session-rec repository, `algorithms/knn/sknn.py`:
https://github.com/rn5l/session-rec/blob/master/algorithms/knn/sknn.py
The equations/config in this decision define our variant when it differs.

## One preprocessing exception, isolated from the project environment

The upstream session key uses pandas `hash_pandas_object`. Reimplementing it or using
Python `hash()` risks a silent identity mismatch. Therefore one PEP-723 preparation
script uses pinned pandas 2.2.3 / NumPy 2.3.5 / PyArrow 21.0.0 / psutil 7.2.2 in an
**isolated uv script environment**, with Python 3.11.14. This is a deliberate, documented
execution dependency, not a silent edit to `pyproject.toml`, `uv.lock` or `.venv`.
The normal model, evaluator and test commands use the locked project environment.

The complete original user slice is recovered from ALL frozen T2 TRAIN clients,
including censored rows (97,279 users); using T1-only clients would shift global row
orders. Every frozen T1/T2 anchor must bind exactly by split-global order, client,
session, category and, for T2, item and view type. Labels are consumed, never rebuilt.

Raw events are now needed for genuine histories, unlike the old zero-history runtime
smoke. Use the official raw parquet; do not substitute the old smoke slices. The
source's opaque bytes are hashed; TRAIN/VALIDATION predicates restrict semantic row
use before session/feature operations. Mixed row groups are not a license to inspect
TEST. The existing excluded-session artifact is consumed, not recomputed.

The existing registry does not expose an authoritative full raw-Parquet checksum.
Record its canonical Drive identity, actual full local SHA-256 and complete anchor
binding. Do not compare it with the CSV's checksum or claim an upstream checksum
match that was not performed.

## Result/schema convention — no fabricated final R1

Reuse `ExperimentConfig v1` and `ExperimentResult v1`. The fixed schema has no separate
classical regime, so R1 denotes **centralized orchestration** here, with
`VALIDATION_BASELINE_NOT_FINAL_R1`, distinct model/objective/representation identities
and no QR or final-neural-reference claim. No regime enum or schema is edited.

The mandatory initialization reference describes the real **unfitted estimator or
empty index and seed**, not neural weights or an invented CommonInitialization
checkpoint. The mandatory trainer reference points at the actual baseline fitting
entry point, not a false claim that sklearn ran through LocalTrainerCore. These
baseline-specific types/identities intentionally cannot match a neural R1/R2a pair.
Review this convention explicitly before publishing; do not exploit validator
permissiveness to mislabel a trained model as an untrained initialization.

## Resources and attempts

One heavy process at a time; two CPU threads, no GPU or Flower. The inherited process-
tree monitor enforces 6 GiB available before admission, 2 GiB minimum available during
runs and 70% total-RSS guard. Allow at most four hours per stage. A label-blind 256-query
kNN feasibility probe can block an infeasible full evaluation; it cannot change the
candidate limit, k grid or membership. No unreported sample substitution.

Use immutable `batch-001`, `batch-002`, ... directories. Never overwrite a completed
batch or clean failed attempts to obtain PASS. A shared prepared directory may be
reused only after source/input/output hashes validate. Logs, fitted objects, histories,
predictions, indices and identifiers remain private and gitignored. Human publication
stages explicit public evidence and only the newly verified result JSONs.

## Approval Record

Team approval confirmed by project owner on 2026-09-11.

Approved reviewers:
- Eid Abdelrihem
- Ahmed Abdelhameed
- Ahmed Sherif

Approved scope for S2-PR-04:
- T2 / C1 classical baselines
- frozen decision-time feature manifest
- Logistic Regression + LightGBM search space
- VALIDATION-only selection
- pooled PR-AUC as headline selection metric
- buyer top-3 recall as companion metric
- no probability-calibration claim

Approved scope for S2-PR-05:
- T1 / C1 Session-kNN
- category-projected binary-item-set cosine variant
- k in {50, 100, 200}
- TRAIN-only session index
- configured candidate cap and TRAIN-frequency fallback
- deterministic tie ordering
- VALIDATION-only selection by next-distinct macro MRR@20

No external approval permalink supplied.

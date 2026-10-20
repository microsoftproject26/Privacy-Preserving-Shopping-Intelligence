# PPSI V4 — G0 data census (Sonnet A)

Read-only census of `rees46_nextitem_v1` (frozen, version_hash
`860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd`,
`decision_ledger_digest.rows = 75083291` per `VERSION.json`). No writes outside this
directory (`.local_ai/ppsi_v4/g0/data_census/`). No raw `user_id` or `product_id`
string is ever written to any output file — only counts, derived aggregates, and
one-way sha256-derived selection keys.

Python: `next_item_eda/.venv/Scripts/python.exe` (3.13.11) — pandas 2.3.3, pyarrow
22.0.0, numpy 2.4.1 available; polars NOT installed (nothing was installed to get
this result; pandas/pyarrow were sufficient).

## Files

| file | contents |
|---|---|
| `SPLIT_COUNTS.json` | per-old-split decision/mask/status counts and distinct users; the V4 `VALIDATION = VALIDATION_SELECT ∪ VALIDATION_CONFIRM` union and `RETRO_TEST = TEST` relabeling |
| `LABEL_TIME_AUDIT.json` | T04/T05 acceptance-test facts: TRAIN label-time leakage checks, VALIDATION/TEST right-censoring and label-grace maturation counts |
| `CATALOGUE_SUPPORT.json` | support definition + verification against `catalogue.representative_views`, K_s for s∈{1,2,5}, TRAIN/VALIDATION(union) label coverage, GRU/SASRec parameter-byte arithmetic |
| `CLIENT_CENSUS.json` | per-user TRAIN-eligible example counts: totals, quantiles, binned histogram, **exact frequency table**, warm/cold breakdown for VALIDATION(union) and TEST |
| `COHORT_PROJECTIONS.json` | deterministic sha256-keyed cohorts of sizes {16384, 65536, 262144, ALL} over the universe of TRAIN-eligible (support≥5) users, with exact frequency tables for both eligible-example counts and VALIDATION(union) eval-row counts per cohort |
| `scripts/common.py` | shared paths, split boundaries, quantile/histogram bin constants |
| `scripts/pass1_support.py` | Pass 1: per-product TRAIN support counts (2-column read); writes `scripts/_support_counts.parquet` (product_id, support) — an intermediate artifact, not raw-id-sensitive since product_id is not a personal identifier and this file mirrors exactly what `mappings/catalogue.parquet` already publishes for the support≥5 subset |
| `scripts/pass2_census.py` | Pass 2: everything else, single 11-column read per bucket; `--bucket B` mode processes one bucket in isolation for the reproducibility check |
| `scripts/validate_rerun.py` | re-runs one bucket via `--bucket` and diffs against the full run's per-bucket log |
| `scripts/_pass1_manifest.json`, `scripts/_pass2_bucket_log.jsonl`, `scripts/*_stderr.log` | run logs / per-bucket debug trail supporting the reproducibility check below |

### Script sha256 (of the files as committed here)

```
common.py           8823d7e039a6ffb16a13d83e3ef400f3a67f6472979f84654906bb60a02334ab
pass1_support.py    a2f9cf33ef0a2b6ed5befbcae87244e9e3a15f1946453040cecd98b035bf58db
pass2_census.py     3e79b764a43e0616d692ba724226beb2e8ae8212c2a640ad645377c4d38783d0
validate_rerun.py   4605a0dbaaf0465b6f80013c1aee2c4d6bc9351c0afe2d252f716be931ad796a
```

## Runtime

- Pass 1 (support counts, 2 columns × 64 buckets, 75,083,291 rows): **19.4 s**
- Pass 2 (11 columns × 64 buckets, full per-user aggregation + sha256 keys for the
  2,694,998-user universe): **55.9 s**
- Total wall time for the full census (excl. interactive exploration): under 2 minutes.

## Definitions used (with citations)

- **Split boundaries** — `rees46_nextitem_v1/splits/splits.json` `intervals_half_open_utc`
  (half-open, UTC): TRAIN `[2019-10-01T00:00Z, 2019-11-16T00:00Z)`, VALIDATION_SELECT
  `[11-16, 11-20)`, VALIDATION_CONFIRM `[11-20, 11-23)`, TEST `[11-23, 11-30)`,
  LABEL_GRACE `[11-30, 12-01)`. `splits.json` also states `label_grace_emits_decisions: 0`
  and gives `decisions_by_split` — **independently reproduced exactly** by Pass 1
  (TRAIN 52,760,314 / VALIDATION_SELECT 10,930,090 / VALIDATION_CONFIRM 3,427,555 /
  TEST 7,965,332; no `LABEL_GRACE` rows found in the ledger at all — `split` value
  counts across all 64 buckets contain exactly the four values above, confirmed in
  `SPLIT_COUNTS.json.other_split_values_present == {}`).
- **`loss_mask` / `eval_mask`** — DATA_CONTRACT.md:55-56 ("`loss_mask` bool `[B]` TRAIN
  decisions that may contribute to the loss"; "`eval_mask` bool `[B]` future-partition
  decisions that may be scored"). DATA_CONTRACT.md:132-134: an OOV observed target
  "In TRAIN its `loss_mask` is false; in future partitions it stays end-to-end
  eligible and scores zero." protocol.resolved.yaml:115 `train_oov_target:
  loss_mask_false`, :116 `evaluation_oov_target: retain_e2e_zero`.
  **Empirically verified** (row-level, spot-checked on `user_bucket=5`, and
  consistent with exact aggregate totals across all 64 buckets in
  `SPLIT_COUNTS.json`): in TRAIN, `loss_mask == (has_observed_target &
  target_class>=0)` exactly; in VALIDATION_SELECT / VALIDATION_CONFIRM / TEST,
  `eval_mask == has_observed_target` exactly. This is reported as a MEASURED finding
  from this census, not an assertion that it is the documented formula beyond what
  the two citations above already state.
- **`target_class`** — FEATURE_DICTIONARY.md:64: "compact class id ... -1 when
  unobserved/immature; OOV targets keep their raw identity and score zero."
  DATA_CONTRACT.md:130-131: `target_class = -1` means "no rankable target."
- **Catalogue / support** — FEATURE_DICTIONARY.md:25 (`product_idx` fitting scope:
  "TRAIN representatives, support >= 5"). protocol.resolved.yaml:118-124: `catalogue:
  fit_scope: full_valid_train_population`, `rule: one_pass_representative_view_support`,
  `minimum_support: 5`, `singleton_representatives_count_for_item_support: true`.
  Task-specified operational definition used here: `support(product_id) := count of
  ledger rows with split=='TRAIN' for that product_id` (the CONTEXT/current-item
  column, not the target column). **Verified exactly**: for all 158,486 catalogue
  rows, computed support == `catalogue.representative_views` (0 mismatches); the
  support≥5 product set is set-identical to the catalogue product set (0 in either
  symmetric difference). See `CATALOGUE_SUPPORT.json.verification`.
- **Identifiers** — DATA_CONTRACT.md:177-207 (§4b): `user_id`/`product_id` are the
  canonical STRING identifiers; never written to model-facing or (per this task's
  instructions) census outputs. `client_idx`/`product_idx` are the safe integer
  handles; `mappings/clients.parquet` (4,204,379 rows) is confirmed here to equal
  exactly the distinct-user count of TRAIN (`SPLIT_COUNTS.json.per_old_split.TRAIN.
  distinct_users == 4204379`).
- **Selection key** (COHORT_PROJECTIONS.json) — exactly the formula given in the
  task: `int.from_bytes(sha256(b"PPSI_V4_POPULATION_20260926" + b"|" +
  user_id.encode("utf-8")).digest()[:8], "big")`. Cohort(N) = the N smallest-key
  users of the universe (support≥5 TRAIN-eligible, i.e. `CLIENT_CENSUS.json`'s
  `s5.users_with_ge1_example` population, U = 2,694,998). Nested-prefix sanity is
  checked programmatically (`nested_prefix_sanity` in the output), not just asserted.
- **User→bucket partitioning** — verified before writing any aggregation code: for
  `user_bucket=0` and `user_bucket=1`, the `user_bucket` column inside each file is
  constant and equal to the partition value, and the two buckets' `user_id` sets have
  zero overlap (checked directly). This underlies every per-user / distinct-user
  computation in this census: a user's rows never cross a bucket, so per-bucket
  partial aggregates can be summed (or set-unioned for the VALIDATION union) without
  double counting or cross-bucket merging of per-user state.
- **Quantile method** — `numpy.percentile(..., method="linear")` (linear
  interpolation between order statistics) over the per-user counts. Documented
  explicitly in `CLIENT_CENSUS.json._definition.quantile_method` because "exact
  quantiles" does not by itself specify an interpolation rule; the **exact frequency
  tables** (`examples_frequency_table`, `validation_union_eval_rows_frequency_table`)
  are provided alongside so the underlying distribution is fully reconstructable
  regardless of which quantile convention a reader prefers.
- **Parameter-byte arithmetic** (`CATALOGUE_SUPPORT.json.thresholds.*.param_bytes_fp32`)
  — computed literally per the task's formulas (`K_s*128*4` GRU tied table,
  `2*K_s*256*4` SASRec two tables, ×2 for AdamW state). A `note_item_input_table_
  may_need_Ks_plus_3...` field is also included because DATA_CONTRACT.md/
  FEATURE_DICTIONARY.md state the INPUT item vocabulary carries `PAD=0/MISSING=1/
  OOV=2` special rows in addition to the `K_s` output classes (categorical_tokens.md
  §"Special categorical tokens"); if the tied embedding table must also serve as the
  input lookup, its row count may need to be `K_s+3`, not `K_s`. This is reported as
  an arithmetic note only — which convention the training code actually uses is out
  of scope here.

## Flags (not resolved — for the lead / Fable)

1. **`target_status` enum members are not prose-defined** in DATA_CONTRACT.md or
   FEATURE_DICTIONARY.md. Observed values across the full ledger: `OBSERVED_IN_
   PARTITION`, `NO_OBSERVED_NEXT`, `AMBIGUOUS_BOUNDARY`, `BOUNDARY_UNRESOLVED`
   (VALIDATION_SELECT/CONFIRM/TRAIN only, small counts, e.g. TRAIN has exactly 1
   row), `OBSERVED_MATURED_IN_GRACE` (TEST only, 133 rows total — consistent with
   `LABEL_TIME_AUDIT.json.test_eval_and_target_ts_in_grace == 133`). Two literal
   examples of the tuple-shaped values appear in `INTEGRATION_SMOKE.json` (lines
   427, 437) but neither file gives a prose definition of `AMBIGUOUS_BOUNDARY` or
   `BOUNDARY_UNRESOLVED`. Full per-split value counts are in `SPLIT_COUNTS.json`.
2. **`BOUNDARY_UNRESOLVED` appears in VALIDATION_SELECT/CONFIRM and TEST but not in
   TRAIN's per-bucket breakdowns except a single row dataset-wide** (TRAIN total = 1,
   see `SPLIT_COUNTS.json.per_old_split.TRAIN.target_status_counts`). Whether that
   single row is expected or an artifact is left to the lead.
3. **TRAIN `loss_mask`/eval_mask row-level identity** (see Definitions above) was
   spot-checked on one bucket only (`user_bucket=5`), not all 75M rows individually —
   the census instead relies on exact aggregate-total matches across all 64 buckets
   as the full-data evidence. If a row-level counterexample exists it would have to
   be exactly offset by another row-level exception elsewhere to preserve the totals
   observed, which is possible in principle though not something this census ruled
   out by direct full-data row comparison.
4. No `LABEL_GRACE`-split decisions were found anywhere in the ledger (matches
   `splits.json.label_grace_emits_decisions: 0`), and no `decision_ts >= 2019-11-30`
   rows were found either (`LABEL_TIME_AUDIT.json.decisions_with_decision_ts_ge_
   2019_11_30 == 0`) — both checks passed cleanly, nothing to flag, noted here only
   because the task asked to check for LABEL_GRACE rows explicitly.
5. Nothing else surprising encountered: the support definition matched the frozen
   catalogue exactly (0 mismatches), OOV/loss_mask/eval_mask relationships were
   internally consistent everywhere checked, and `clients.parquet` row count matched
   TRAIN's distinct-user count exactly.

## Reproducibility check performed

`scripts/validate_rerun.py 0` and `scripts/validate_rerun.py 37` both re-run that
bucket in total isolation (fresh process, `pass2_census.py --bucket B`) and diff
every aggregate field plus a sample of derived sha256 selection keys against the
same bucket's contribution recorded during the full 64-bucket run
(`scripts/_pass2_bucket_log.jsonl`). Both report `"identical": true` with no
mismatched keys. Cross-bucket totals were checked against two independent external
sources: `VERSION.json.components.decision_ledger_digest.rows == 75083291` (matches
`SPLIT_COUNTS.json.total_decisions_all_splits`) and `splits/splits.json.
decisions_by_split` (matches every per-old-split `decisions` count exactly).

## Headline numbers

- Total decisions: **75,083,291** (TRAIN 52,760,314 / VALIDATION_SELECT 10,930,090 /
  VALIDATION_CONFIRM 3,427,555 / TEST 7,965,332). V4 VALIDATION union: 14,357,645
  decisions, 1,578,912 distinct users (vs. 1,139,657 + 700,093 for the two old splits
  separately — the union removes ~260,838 double-counted users).
- Catalogue support verification: **0 mismatches** against `catalogue.
  representative_views`; support≥5 set is exactly the catalogue set. K_1 = 191,890,
  K_2 = 179,315, K_5 = 158,486.
- TRAIN positive coverage (support≥5): 99.86% (56,560 of 40,237,168 observed
  pre-cutoff TRAIN targets excluded for being outside the catalogue). VALIDATION
  (union) label coverage at support≥5: 99.14% (96,871 excluded).
- Label-time audit: 0 TRAIN loss rows with `target_ts` on/after TRAIN_END or null;
  0 VALIDATION(union) eval rows with `target_ts` on/after VALIDATION_END (old or new
  boundary); 133 TEST eval rows matured in the label-grace window; 0 decisions with
  `decision_ts >= 2019-11-30`.
- Client census (support≥5 definition): 2,694,998 of 4,204,379 TRAIN users have
  ≥1 TRAIN-eligible example (40,180,608 examples total); median 6, p90 35, p99 138,
  max 11,628 examples/user. VALIDATION(union) users with ≥1 eval row: 1,023,523
  (642,228 warm / 381,295 cold). TEST: 746,662 (420,693 warm / 325,969 cold).
- Cohort universe U = 2,694,998; nested-prefix sanity confirmed programmatically for
  all four cohort sizes {16384, 65536, 262144, ALL}.

Not interpreted or recommended here (support thresholds, cohort sizing, coverage
trade-offs) — that is the lead's / Fable's call, per instructions.

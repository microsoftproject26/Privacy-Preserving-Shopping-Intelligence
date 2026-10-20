# Protocol deviations and conflict register

Generated 2026-09-21T14:44:48Z.

This is a **new versioned data specification**. It does not overwrite, supersede or claim
approval from any earlier protocol, and it does not certify that an already-viewed holdout was
historically untouched. Earlier artifacts are preserved.

## A. Deviations from published recipes (intentional, project-specific)

| # | Published / author practice | What we did | Why | Impact |
|---|---|---|---|---|
| 1 | Iterative session+item filtering to a support threshold | **One-pass** representative-view support >= 5, computed once and frozen | Simpler, deterministic, single full-data pass; avoids an iteration whose fixed point depends on ordering | Catalogue = 158,486 items. Some retained sessions keep sub-threshold items as legitimate OOV context instead of being deleted |
| 2 | Supplied `user_session` used as the session (Transformers4Rec REES46 tutorial) | Sessions **rebuilt** from the user's own view stream with a 1-hour gap | The supplied UUIDs span multiple users and run for weeks (section 08) | Our session counts are **not** comparable with numbers computed on the supplied UUIDs |
| 3 | Split by fraction of available data | **Five fixed UTC dates**, frozen in config before any fitting | Reproducible, leakage-proof, independent of which files happen to exist | Split sizes are whatever the calendar gives, not a round percentage |
| 4 | Targets linked after filtering rare items | Target relation **built and frozen before** catalogue filtering; never relinked | Prevents manufactured adjacency (A,Xrare,B never becomes A->B) | See notebook section 16 for the exact count of adjacencies this avoids |
| 5 | ID-only sequence input | **Feature-enriched** input: side categoricals, price, time, bounded user context | The task requires enriched GRU and SASRec variants | These are **not** the unchanged official GRU4Rec or original SASRec recipes |
| 6 | (none) | **Ambiguity barriers** at same-timestamp multi-product blocks | Whole-second timestamps make the order genuinely unknowable | Real browsing information is excluded rather than arbitrarily ordered; the cost is counted |
| 7 | (none) | **As-of metadata recovery** with a masked-validation acceptance gate | Repair must not leak the future and must be measurable | Families that fail the gate are disabled, not loosened |
| 8 | category -> first observed brand (seen in community code) | **Explicitly rejected** | Not an accepted recovery rule; a category does not determine a brand | Brand stays MISSING where unrecoverable |
| 8b | MASTER_PROMPT S13: "zero remains valid with a `price_zero` flag" | **`price <= 0` is INVALID for the numeric price feature** | **Protocol-owner decision superseding the frozen MASTER_PROMPT wording.** A zero price in an e-commerce log is far more likely an unpriced listing than a genuinely free product, so it is not a trustworthy price signal | The interaction is NEVER deleted. `price_raw` is preserved verbatim; `price_zero=1` AND `price_missing=1` are both set; zero rows are excluded from all price normalizer/statistics fitting; the model sees the neutral normalized sentinel ONLY together with the missing indicator |
| 9 | `clean_filled` style pre-imputed datasets | **Never used or downloaded** | A file named `clean_filled` is not certified metadata | All repair is derived locally from original evidence with recorded lineage |

## B. Declared fallbacks actually exercised in this run

| Fallback | Trigger | What changed | Status |
|---|---|---|---|
| Chunked out-of-core aggregation | DuckDB out-of-memory on a grouped COUNT DISTINCT over ~23M session keys under the resolved memory limit | Heavy GROUP BYs run in 32 deterministic hash chunks with on-disk intermediates | Results remain **EXACT full-data**, not sampled |
| External-sort order statistic | `quantile_cont` cannot materialize a column under a tight memory limit | Quantile falls back to `ORDER BY ... LIMIT 1 OFFSET k` | Exact order statistic, never an approximation. **Correction:** an earlier draft said "the method used is recorded per statistic". That holds for 1 of 6 call sites (`profiling.py:264`); `final_gate.py` session percentiles and `profiling.py:292/323/324` discard the method string. The two methods are different *definitions* (interpolated vs lower order statistic), so for those five we cannot tell from the artifact which was used. **No blocking gate and no normalizer constant is affected** — the normalizer fit calls `quantile_cont` directly (`pipeline6.py:361`) and every stage-14 gate is an exact count or join |
| Windowed builds sub-chunked by user hash | DuckDB out-of-memory in the chained window operators of stages 06 and 07 | Each user-hash bucket is processed in 6 sub-chunks; a user is never split, so no session boundary is inferred at a chunk edge | Logical output identical; proven by the repartition-invariant digest |
| Validation folds spilled to Parquet | Tens of millions of per-event validation rows | Fold results written to disk, then aggregated with `arg_max` hash aggregates | No behaviour change |
| Memory-budget retry ladder | `04b_profile`: 2 attempts, passed at fraction 0.55 | The stage was re-run in a fresh process at a smaller budget; one stage per process means each retry re-measures the machine | **Exercised in this run.** Exact-aggregate gates are budget-independent, so the result carries the same meaning; only the physical plan and spilling changed |
| Recovery family disabled | Acceptance gate not met | Field keeps originals + MISSING, marked `RECOVERY_UNVERIFIED` | **Not a pipeline failure.** Currently disabled: none |

## B2. Protocol-owner decisions that SUPERSEDE the frozen MASTER_PROMPT

These are deliberate, recorded departures from the frozen specification text,
made by the protocol owner during review. They are listed separately from the
deviations above because those are choices *within* the spec's latitude,
whereas these contradict its literal wording.

### B2.1 Zero price is invalid for the numeric price feature

| | MASTER_PROMPT S13 (frozen text) | Implemented (protocol-owner decision) |
|---|---|---|
| `price > 0`, finite | valid | valid |
| `price == 0` | **valid**, with `price_zero` flag | **INVALID / numeric missing** |
| `price < 0` | numeric missing | numeric missing |
| non-finite / unparseable | numeric missing | numeric missing |

What the implementation does with a zero-price row:

```text
price_raw     = 0          # preserved verbatim, never rewritten
price_zero    = 1          # audit/context flag: the SOURCE recorded a zero
price_missing = 1          # there is no trustworthy numeric price for the model
price_valid   = NULL       # cannot become price_obs
log1p_price   = NULL       # before tensorization
model value   = 0          # neutral sentinel, ONLY together with...
missing ind.  = 1          # ...the missing indicator
```

`price_zero` and `price_missing` are kept as **separate** flags because they
say different things: `price_missing=1` means *no usable numeric price*, while
`price_zero=1` means *the specific reason is that the source recorded zero*.
Collapsing zero into generic missingness would destroy that distinction, and
`price_zero` may itself turn out to be informative.

**But measure it before claiming it.** On this data the two flags are
**100% co-occurring**: of the feature rows carrying either flag, 198,060 of
198,060 carry both, and no other combination exists (`price_conflict` never
fires). So in *this* dataset `price_zero` carries **no information beyond**
`price_missing`. The separation is implemented, correct and worth keeping for
audit and for future data where the two could diverge — but it must not be
reported as an additional informative signal here, because empirically it is
not one.

**A second limit on what was proven.** `price_status` takes exactly two values
in the real source: `OK` (109,693,982 rows) and `ZERO` (256,761 rows).
`invalid_prices_in_source = 0`, which means the NEGATIVE, NONFINITE and
UNPARSEABLE branches **were never exercised on real data**. They are
implemented and covered by fixtures in `tests/test_sql_parity.py`; they are not
something this run measured clean. The correct statement is "code and fixtures
cover them", not "we verified from the data that they are handled".

**The interaction is never deleted.** The sequence, user, session, target,
category and brand all survive intact; only the price channel is masked.

Scope: 256,761 rows of 109,950,743 (0.234%), so excluding them cannot
meaningfully distort the price statistics of the whole dataset - whereas
admitting them as genuine zeros would bias every fitted price normalizer.

Enforced by blocking gates: `zero_price_never_becomes_price_valid`,
`zero_price_never_becomes_price_obs`, `zero_price_excluded_from_normalizer_fit`,
`invalid_or_zero_price_is_flagged_missing`, plus two negative fixtures in
`tests/test_sql_parity.py` that fail if a zero or negative price ever enters as
a valid numeric value.

### B2.2 Engine memory budget raised, and what the protocol field actually is

**Correction first.** An earlier draft of this section said the protocol
*declares* an operative 0.50 and that the run raised it. The first half is
misleading: `resources.memory_fraction_available: 0.50` in `config/protocol.yaml`
is **read by no code path in this project**. `engine.resolve_settings` and
`run_pipeline.py --memory-fraction` each hard-code their own 0.50 default;
nothing consults `cfg.raw["resources"]["memory_fraction_available"]`. The YAML
field is **descriptive**. The operative value is the CLI / supervisor argument,
and the resolved `EngineSettings` it produces.

That also means the field was never a cache input, so leaving it untouched did
not "preserve" anything: `config_hash` covers the whole YAML and is unchanged
regardless, which is why no stage cache was invalidated.

What genuinely changed is the budget rule in `engine.resolve_settings`,
authorised by the protocol owner mid-run ("use up to 90% of the machine, stop
immediately if it starts to freeze").

| | Before | After |
|---|---|---|
| budget rule | `available x fraction` | `min(available - 1.6 GiB, 90% of total RAM) x fraction` |
| fraction passed | 0.50 | 0.90, stepping down on retry |
| hard cap | 6.0 GiB | 9.0 GiB |
| threads | `min(cpus, memory_gib x 2)` | `min(cpus, 16, memory_gib / 0.40)` |

**There is no single figure for "stages 07-14".** The supervisor runs one stage
per process and every process re-measures free memory, so the budget moves
between stages and between retries of the same stage. An earlier draft quoted
"~4.6 GiB, 11 threads" for that range; that number is not supported by the
ledger and is withdrawn. The per-invocation values actually used are recorded
in section B2.3 below, recovered from the run ledger rather than restated from
memory.

**Unit caveat, stated because it makes the recorded numbers off by 7%.** The
budget is computed in GiB (`psutil` bytes / 2^30) but applied as
`SET memory_limit='<x>GB'`, and DuckDB reads `GB` as 10^9 bytes. The effective
limit is therefore `x * 0.9313` GiB, not `x` GiB. This is why stage 14's
out-of-memory message reports `6.0 GiB/6.0 GiB used` while the ledger records a
resolved 6.47 for that invocation: 6.47 GB = 6.02 GiB. The error is in the
conservative direction and changes no result, but `memory_limit_gib` is a
misnomer for a value handed to DuckDB as GB.

The two subtracted terms are not decoration. DuckDB's `memory_limit` bounds its
buffer manager only; PyArrow tables, numpy arrays and the interpreter allocate
outside that accounting, which is why a budget set to all of free memory pages
the machine to a standstill instead of raising a clean out-of-memory error.

**This is a speed setting, not a semantic one** — but that is a claim the run
*tests*, not one the cache key enforces. Being precise about the difference:
`engine.py` is deliberately outside 15 of the 16 stage code-fingerprints, yet it
can influence bytes on disk, through `preserve_insertion_order=false` (row order
in `COPY` output) and through which branch `engine.exact_quantile` takes. So a
stage manifest can remain "valid" across an `engine.py` edit that would have
produced different bytes. Three things make that benign here: the derived
identities are content-derived strings and never execution-order row numbers;
the reproducibility digests are order-independent by construction; and
`exact_quantile` feeds only descriptive statistics, never a blocking gate and
never a normalizer constant.

The evidence, rather than the assertion: stages 01-05 of the primary run
executed at 2.14 (GB) / 4 threads and the later stages at budgets between 3.72
and 6.65, while the cold rebuild re-derives every stage from the raw CSVs at a
*deliberately different* fraction (0.55). The repartition-invariant digests in
`COLD_REBUILD_REPRODUCIBILITY.json` therefore compare genuinely different
execution configurations. If any stage's output depended on the memory limit or
the thread count, those digests would diverge.

Protection added with the increase:

- `scripts/resource_watchdog.py` samples free memory every 5s and terminates
  the pipeline process tree after 4 consecutive samples below 0.60 GiB.
- `scripts/supervisor.py` runs one stage per process, so each stage re-measures
  the machine, and retries a failed stage down a budget ladder
  (0.90 -> 0.70 -> 0.55 -> 0.45) instead of abandoning it.
- Stopping at any point is safe by construction: stages are checkpointed by
  manifest and `06_sessions` additionally resumes per user bucket. The
  per-bucket guard tests existence and non-zero size, not the Parquet footer,
  so it is not atomic; a truncated file has no footer and fails loudly on the
  next read rather than being accepted as data. See the pending recommendation
  to make that guard positive.

### B2.3 Engine budget actually resolved, per invocation

Recovered from the run ledger, not restated from memory. Only `01_preflight`
writes an engine report into its manifest, so without the ledger the budget
that produced the other stages would not be recoverable from the artifacts at
all. Values are as recorded (see the GB/GiB caveat in B2.2).

| وقت الاستدعاء (UTC) | المراحل | حد الذاكرة (GiB) | الخيوط | الذاكرة المتاحة وقتها (GiB) |
|---|---|---|---|---|
| `2026-09-20T23:48:44Z` | `01_preflight` | 1.5 | 3 | 1.874 |
| `2026-09-20T23:48:56Z` | `02_staging` | 1.5 | 3 | 1.844 |
| `2026-09-20T23:52:23Z` | `03_canonical` | 1.5 | 3 | 2.389 |
| `2026-09-21T00:09:59Z` | `04_profile` | 1.5 | 3 | 2.389 |
| `2026-09-21T00:18:52Z` | `04_views`, `04b_profile` | 2.0 | 4 | 3.788 |
| `2026-09-21T00:22:35Z` | `04b_profile` | 2.0 | 4 | 2.863 |
| `2026-09-21T00:29:54Z` | `05_recovery` | 2.0 | 4 | 2.524 |
| `2026-09-21T00:35:46Z` | `06_sessions` | 2.0 | 4 | 3.152 |
| `2026-09-21T00:36:54Z` | `06_sessions` | 2.0 | 4 | 3.223 |
| `2026-09-21T00:38:06Z` | `06_sessions` | 2.0 | 4 | 3.713 |
| `2026-09-21T00:54:31Z` | `07_ledger`, `08_catalogue` | 2.0 | 4 | 2.269 |
| `2026-09-21T01:02:35Z` | `07_ledger`, `08_catalogue`, `09_features`, `10_index`, `11_loader_checks`, `12_sealed_test` | 2.0 | 4 | 2.686 |
| `2026-09-21T01:10:42Z` | `03_canonical`, `04_views`, `04b_profile`, `05_recovery`, `06_sessions`, `07_ledger`, `08_catalogue`, `09_features`, `10_index`, `11_loader_checks`, `12_sealed_test` | 2.0 | 4 | 2.344 |
| `2026-09-21T01:40:37Z` | `03_canonical`, `03b_roundtrip`, `13_source_coverage` | 2.0 | 4 | 2.161 |
| `2026-09-21T01:56:01Z` | `03_canonical`, `03b_roundtrip`, `13_source_coverage` | 2.0 | 4 | 2.515 |
| `2026-09-21T01:57:57Z` | `03b_roundtrip`, `13_source_coverage` | 2.0 | 4 | 3.668 |
| `2026-09-21T02:00:03Z` | `03b_roundtrip`, `13_source_coverage` | 2.16 | 4 | 4.328 |
| `2026-09-21T02:01:53Z` | `13_source_coverage`, `04_views`, `04b_profile`, `05_recovery`, `06_sessions`, `07_ledger`, `08_catalogue`, `09_features`, `10_index`, `11_loader_checks`, `12_sealed_test`, `03b_roundtrip` | 2.0 | 4 | 3.257 |
| `2026-09-21T02:04:38Z` | `03b_roundtrip` | 2.13 | 4 | 4.27 |
| `2026-09-21T02:06:34Z` | `13_source_coverage`, `04_views`, `04b_profile`, `05_recovery`, `06_sessions`, `07_ledger`, `08_catalogue`, `09_features`, `10_index`, `11_loader_checks`, `12_sealed_test` | 2.11 | 4 | 4.229 |
| `2026-09-21T02:48:12Z` | `09_features`, `10_index`, `11_loader_checks`, `12_sealed_test`, `14_final_gate` | 2.22 | 4 | 4.432 |
| `2026-09-21T02:49:13Z` | `08_catalogue`, `09_features`, `10_index`, `11_loader_checks`, `12_sealed_test`, `14_final_gate` | 2.04 | 4 | 4.077 |
| `2026-09-21T03:03:26Z` | `01_preflight`, `02_staging`, `03_canonical`, `03b_roundtrip`, `13_source_coverage`, `04_views`, `04b_profile`, `05_recovery`, `06_sessions`, `07_ledger`, `08_catalogue`, `09_features`, `10_index`, `11_loader_checks`, `12_sealed_test`, `14_final_gate` | 2.14 | 4 | 4.276 |
| `2026-09-21T03:53:36Z` | `01_preflight` | 5.68 | 14 | 7.911 |
| `2026-09-21T03:53:37Z` | `02_staging` | 5.67 | 14 | 7.896 |
| `2026-09-21T03:53:38Z` | `03_canonical` | 5.66 | 14 | 7.885 |
| `2026-09-21T03:53:39Z` | `03b_roundtrip` | 5.63 | 14 | 7.855 |
| `2026-09-21T03:53:40Z` | `13_source_coverage` | 5.64 | 14 | 7.871 |
| `2026-09-21T03:53:42Z` | `04_views` | 5.64 | 14 | 7.866 |
| `2026-09-21T03:53:43Z` | `04b_profile` | 4.38 | 10 | 7.858 |
| `2026-09-21T03:53:44Z` | `05_recovery` | 5.65 | 14 | 7.874 |
| `2026-09-21T03:53:45Z` | `06_sessions` | 5.63 | 14 | 7.857 |
| `2026-09-21T03:57:13Z` | `07_ledger` | 5.68 | 14 | 7.911 |
| `2026-09-21T04:00:38Z` | `08_catalogue` | 4.9 | 12 | 7.048 |
| `2026-09-21T04:01:48Z` | `09_features` | 3.82 | 9 | 7.058 |
| `2026-09-21T04:21:16Z` | `10_index` | 5.04 | 12 | 7.203 |
| `2026-09-21T04:21:44Z` | `11_loader_checks` | 4.82 | 12 | 6.955 |
| `2026-09-21T04:22:55Z` | `12_sealed_test` | 6.65 | 16 | 8.991 |
| `2026-09-21T04:23:00Z` | `14_final_gate` | 6.47 | 16 | 8.789 |
| `2026-09-21T04:24:36Z` | `14_final_gate` | 5.34 | 13 | 9.223 |
| `2026-09-21T04:26:12Z` | `14_final_gate` | 4.17 | 10 | 9.184 |
| `2026-09-21T05:11:24Z` | `01_preflight` | 4.97 | 12 | 7.122 |
| `2026-09-21T05:11:25Z` | `02_staging` | 5.0 | 12 | 7.156 |
| `2026-09-21T05:11:26Z` | `03_canonical` | 5.0 | 12 | 7.159 |
| `2026-09-21T05:11:28Z` | `03b_roundtrip` | 4.97 | 12 | 7.121 |
| `2026-09-21T05:11:29Z` | `13_source_coverage` | 4.98 | 12 | 7.128 |
| `2026-09-21T05:11:30Z` | `04_views` | 4.96 | 12 | 7.11 |
| `2026-09-21T05:12:35Z` | `04b_profile` | 4.12 | 10 | 7.492 |
| `2026-09-21T05:13:55Z` | `01_preflight` | 5.47 | 13 | 7.681 |
| `2026-09-21T05:13:56Z` | `02_staging` | 5.46 | 13 | 7.662 |
| `2026-09-21T05:13:57Z` | `03_canonical` | 5.45 | 13 | 7.653 |
| `2026-09-21T05:13:58Z` | `03b_roundtrip` | 5.41 | 13 | 7.615 |
| `2026-09-21T05:14:00Z` | `13_source_coverage` | 5.41 | 13 | 7.608 |
| `2026-09-21T05:14:01Z` | `04_views` | 5.44 | 13 | 7.643 |
| `2026-09-21T05:15:05Z` | `04b_profile` | 4.31 | 10 | 7.762 |
| `2026-09-21T05:19:02Z` | `04b_profile` | 3.38 | 8 | 7.748 |
| `2026-09-21T05:19:03Z` | `04c_timezone_audit` | 5.52 | 13 | 7.736 |
| `2026-09-21T05:19:14Z` | `05_recovery` | 5.51 | 13 | 7.719 |
| `2026-09-21T05:19:15Z` | `06_sessions` | 5.52 | 13 | 7.728 |
| `2026-09-21T05:19:16Z` | `07_ledger` | 5.49 | 13 | 7.695 |
| `2026-09-21T05:19:25Z` | `08_catalogue` | 5.4 | 13 | 7.601 |
| `2026-09-21T05:20:46Z` | `09_features` | 4.17 | 10 | 7.556 |
| `2026-09-21T05:41:39Z` | `10_index` | 5.58 | 13 | 7.804 |
| `2026-09-21T05:42:10Z` | `11_loader_checks` | 5.57 | 13 | 7.785 |
| `2026-09-21T05:43:13Z` | `12_sealed_test` | 6.91 | 16 | 9.273 |
| `2026-09-21T05:43:18Z` | `14_final_gate` | 6.69 | 16 | 9.031 |
| `2026-09-21T05:45:08Z` | `10b_local_time_eda`, `10c_drift_eda` | 5.83 | 14 | 8.885 |
| `2026-09-21T05:46:17Z` | `10b_local_time_eda`, `10c_drift_eda` | 5.84 | 14 | 8.906 |
| `2026-09-21T05:47:20Z` | `10c_drift_eda` | 5.77 | 14 | 8.818 |
| `2026-09-21T07:08:24Z` | `14_final_gate` | 3.82 | 9 | 8.552 |
| `2026-09-21T14:08:36Z` | `11_loader_checks`, `14_final_gate` | 2.0 | 5 | 0.264 |
| `2026-09-21T14:08:56Z` | `04c_timezone_audit`, `11_loader_checks`, `12_sealed_test`, `14_final_gate` | 2.0 | 5 | 2.532 |
| `2026-09-21T14:11:16Z` | `11_loader_checks`, `14_final_gate` | 5.24 | 13 | 7.425 |
| `2026-09-21T14:13:12Z` | `14_final_gate` | 4.82 | 12 | 6.957 |
| `2026-09-21T14:40:39Z` | `11_loader_checks`, `14_final_gate` | 4.29 | 10 | 6.369 |

## C. Things we did NOT do, and why

- **No model training of any kind.** Not authorized by this specification.
- **No ML/LLM metadata imputer.** Deferred: it would need its own grouped temporal calibration
  study and could conflict with later data-locality assumptions. Recorded as deferred, not as
  unfinished mandatory EDA.
- **No reference URLs fetched during this run.** Source verification is marked *unavailable*;
  this is a warning, not a blocker, and no quotation or result was inferred.
- **No bot / high-activity / expensive-product removal.** Unusual is not invalid.
- **No global `dropna()` or `drop_duplicates()` on business payloads.**
- **No timezone shift.** Timestamps stay UTC; no shopper timezone is inferred.
- **No fabricated sub-second components** to break same-second ties.

## D. Claims we explicitly do NOT make

1. That this preprocessing maximizes recommendation accuracy. No model was trained.
2. That recovered metadata is true for every repaired product. The acceptance interval is a
   measured bound on a de-duplicated per-product sample, under stated independence assumptions.
3. That the raw source metadata contains no errors.
4. That the TEST partition was never physically read. It was, by the deterministic builder, and
   every access is logged.
5. That this holdout was historically unexposed. That status is **unknown** and is reported as a
   separate research status, not invented.
6. That construction was federated, client-isolated or differentially private. The catalogue and
   vocabularies are platform-wide TRAIN information in a centralized preparation. A later
   reduced-client FL experiment must declare its own scope.
7. That a plain ID hash is anonymity. It is pseudonymization.

## E. Context length

Chosen by the predeclared coverage rule (`use_20_if_train_full_prefix_coverage_at_least_0.95_else_50`), which selected
**L = 50** from candidates [20, 50].
This is an efficiency/coverage decision and is **not** proof of an accuracy optimum.

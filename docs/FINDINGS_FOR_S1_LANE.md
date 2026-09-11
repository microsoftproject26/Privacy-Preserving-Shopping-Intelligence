# Two findings for `S1-D1-DS-07` and `S1-DS-05/06`

From the S2 model lane, after an independent review of our own results. Both artifacts are
still `PROPOSED`, so neither of these is a frozen decision being reopened — they are defects
found before approval, which is when they are cheapest to fix.

Neither finding is a disagreement about a rule. In both cases **the rule you wrote is the
correct one**, and the code implements something narrower.

---

## Finding 1 — T2 censors decisions whose outcome is observed

**Severity: high.** It withholds 549,185 decisions, inflates the reported base rate by 22%,
and inflates every T2 metric built on it.

### The rule, as `task_headroom_v1` states it

> *"a decision with no observable horizon is censored, never negative"*

Correct. What the notebook does instead:

```python
t2["session_end"] = t2["session"].map(validation.groupby("session")["order"].max())
t2["censored"]    = (t2["label"] == 0) & (t2["query_order"] == t2["session_end"])
```

`session_end` is the last event of the **session**, not the end of the observable window.
The comment two lines above says *"a decision taken at the very end of the observable
window"* — so the intent was right and the expression is narrower than the intent.

### Why the two are not the same thing

The T2 label asks: *was this product purchased later **in this same session**?* Once the
session is over, nothing further can belong to it, so the answer is settled: it was not.
There is no unobserved horizon, because the horizon the question is scoped to has closed.

A session's end coincides with the window's end only when the window cut the session short.
And `S1-DS-05/06` already removed exactly those sessions — `ALLOWED_PATTERNS` admits only
`(TRAIN,)`, `(VALIDATION,)`, `(TEST,)`, `(LABEL_GRACE,)` and `(TEST, LABEL_GRACE)`, so a
session with events in both VALIDATION and TEST was excluded as a crossing session.

**Every session that survives into VALIDATION is therefore complete inside it.** The correct
count of censored T2 decisions is zero.

### Measured

Session-end times of the rows currently marked `CENSORED`, against a `[10-22, 10-27)` window:

| day the session ended | rows |
|---|---:|
| 10-22 | 15,034 |
| 10-23 | 14,424 |
| 10-24 | 13,363 |
| 10-25 | 14,714 |
| 10-26 | 12,932 |

Evenly spread — ordinary sessions ending, not a window edge truncating them. `99.918%` ended
more than an hour before the window closed; the latest is `2019-10-26 23:58:54`, and even
that session is complete. TRAIN behaves identically.

### What it costs

| | rows | withheld | share |
|---|---:|---:|---:|
| TRAIN | 2,770,471 | 478,718 | 17.28% |
| VALIDATION | 392,554 | 70,467 | 17.95% |

Because only negatives were ever withheld, the denominator shrank while every positive
stayed:

| | published | correct |
|---|---:|---:|
| VALIDATION prevalence | 0.0351 | **0.0288** |
| TRAIN item-popularity PR-AUC | 0.0831 | **0.0657** |

And 478,718 real training negatives were discarded.

### The fix

```python
# every session is complete inside its split - crossing sessions were already excluded -
# so a terminal negative is observed, not censored
t2["censored"] = False
```

If you prefer to keep the guard explicit rather than deleting it, express it against the
**window**, which is what the comment already says:

```python
t2["censored"] = (t2["label"] == 0) & (t2["session_end_time"] >= WINDOW_END)
```

That evaluates to zero rows today, and it stays correct if crossing sessions are ever kept.

### Until then

`S2-DS-06/labels.py` applies this correction in the model lane, in one place, and reports
both the corrected number and the published-mask number beside it. It is written to become
a pass-through the moment upstream emits zero censored rows, so the two lanes cannot end up
holding two different label sets.

---

## Finding 2 — the T3 candidate lists are selected using VALIDATION outcomes

**Severity: medium.** It does not invalidate the T3 metric. It does limit what the number
can be claimed to mean, and it silently costs 15.1% of TRAIN queries.

```python
# Only anchors with a positive are ever scored, so only those need a list.
anchor_meta = (queries.loc[queries["query_item"].isin(positives["query_item"].unique()), ...
```

`positives` are VALIDATION outcomes, so candidate lists exist only for query items that
turned out to have a positive in VALIDATION. The reasoning in the comment is sound for
**evaluation** — an anchor with no positive contributes nothing to NDCG either way, and we
confirmed every one of the 18,814 evaluable queries has its list, so the model and the
baseline are scored on identical candidate sets and the comparison is fair.

The consequences are elsewhere:

| | |
|---|---|
| TRAIN queries with no candidate list | **263,112 of 1,740,433 — 15.1%** |
| TRAIN *scoring* queries with no list | 14,558 — 11.2% |
| catalogue coverage | 47,948 anchors, **42.1%** of TRAIN catalogue items |

So a sixth of the training signal is unreachable, and the candidate generator cannot serve
an arbitrary query item — which is what an on-device recommender would face. The reported
NDCG is an honest measure of *reranking within these lists*; it is not evidence about a
deployable retrieval stage, and `S2-SE-*` should not read it as one.

### Suggested fix

Build anchors from **TRAIN** query items rather than VALIDATION positives. Co-occurrence is
already TRAIN-only, so nothing else changes, and the artifact grows but stops depending on
the split it is evaluated against. If size is the constraint, cap by TRAIN query frequency —
also a TRAIN-only rule.

If the artifact is left as it is, the scope line needs to say so wherever the T3 number
appears, because a reader will otherwise assume the candidate generator is general.

---

## What we are not asking for

No cohort rework, no split change, no re-derivation of the vocabulary or the price
transform. We checked those and they hold: C1 is TRAIN-fitted, the intervals are half-open
and disjoint, crossing sessions are handled, and the two null-session rows are singletons.

Finding 1 is a one-line change. Finding 2 is a rebuild of one artifact, and only if you
agree it is worth it.

---

## Finding 3 — the TEST seal needs physically split files to be true

**Severity: medium.** No number is contaminated. The provenance claim was simply stronger
than the evidence, and for this project that matters on its own.

`S1-DS-05/06` publishes the split as **logical intervals**. Any consumer therefore has to
open the monolithic Parquet and filter by timestamp, which means TEST rows get decoded on
the way past. Our loader did exactly that, and wrote `test_rows_read: 0` into every result
JSON as a literal rather than a measurement. That claim was not earned.

We have fixed what the model lane can fix. `load_events` now consults the per-row-group
statistics Parquet already carries and never opens a group that cannot intersect the
window, then counts what it actually decoded. Measured on
`Dataset/processed_raw_parquet_v1.parquet`, 85 row groups:

| window | groups opened | rows decoded | **TEST rows decoded** | TEST rows used |
|---|---:|---:|---:|---:|
| TRAIN | 59 | 29,500,000 | **0** | 0 |
| VALIDATION | 15 | 7,500,000 | **384,203** | 0 |

Before the change, every window decoded all 85 groups — the entire 42,448,764-row file,
TEST included.

TRAIN is now sealed by construction. VALIDATION cannot be, because **row group 72 straddles
the `2019-10-27` boundary**, and a group spanning a boundary must be decoded before it can
be filtered. That is a property of one monolithic file, not of our loader.

The full measurement is in `_SEAL/test_seal_measured.json`.

### The fix, which belongs upstream

Emit boundary-aligned physical artifacts once — `events_train_v1.parquet`,
`events_validation_v1.parquet` — validate their min/max timestamps, and let training jobs
mount only those. Then `test_rows_read == 0` is true because the file being read cannot
contain a TEST row, which is a much stronger statement than any amount of filtering.

Until that exists, our result JSONs carry the measured provenance above instead of a
constant, and say plainly that 384,203 TEST rows are decoded and discarded during
VALIDATION loading without reaching an array, a window, a label or a metric.

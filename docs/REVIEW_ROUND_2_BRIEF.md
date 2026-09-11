# Second review — what changed after your first pass, and where we want your judgement

**To the reviewer.** You reviewed the S2 model lane on 2026-09-11 and returned *ADJUST, then
proceed*. This is what we did with it, what it changed, and what is new since you looked.

**How to read this.** Everything below was re-derived here before it was acted on — we did not
take a single number on authority, and we did not reject one on pride. Where your figure and
ours differ in the last decimal, both are shown. Where we declined something, the reason is
stated and you are invited to push back.

**Your brief is still accurate for the setup** — dataset, splits, cohort, task definitions,
contract. Only results and code have moved.

---

## 1. Your six findings, and what happened to each

| # | Your finding | Verified? | Action | Where |
|---|---|---|---|---|
| 1 | T2 censors observed hard negatives | **yes, exactly** | labels corrected, T2 retrained | `S2-DS-06/labels.py` |
| 2 | a stronger simple T2 baseline exists | **yes** | adopted as the number to beat | `S2-DS-06/README.md` |
| 3 | T3 IDCG from retrieved gains | **yes** | full-oracle denominator, now required | `ppsi/models/…/train_t3.py` |
| 4 | T3 reported micro, ADR-001 says macro | **yes** | macro headline, selection on macro | `S2-DS-07/ladder_t3.py` |
| 5 | T3 candidate lists selected by VALIDATION outcomes | **yes** | documented as a scope limit, fix handed upstream | `FINDINGS_FOR_S1_LANE.md` |
| 6 | `test_rows_read: 0` is a constant, not a measurement | **yes** | instrumented and measured | `_SEAL/test_seal_measured.json` |

### Where our measurement matched yours

| quantity | you | us |
|---|---|---|
| T2 corrected prevalence | `0.028778` | `0.028778` |
| T2 smoothed item+category baseline | `0.097269` | `0.097407` |
| T3 correct micro baseline | `0.204637` | `0.204637` |
| T3 macro baseline | `0.270687` | `0.270687` |
| T3 macro ceiling | `0.879421` | `0.879460` |
| clients with a scoring query | `4,591` | `4,591` |
| TRAIN queries with no candidate list | `263,112` (15.12%) | `263,112` (15.12%) |
| full-TRAIN rows / distinct categories | `29,218,702` / `595` | `29,218,702` / `595` |

### Two we pushed further than you did

**The censoring defect is not a judgement call, it is arithmetic.** We measured *when* the
sessions of the 70,467 withheld rows end, against a `[10-22, 10-27)` window:

| 10-22 | 10-23 | 10-24 | 10-25 | 10-26 |
|---:|---:|---:|---:|---:|
| 15,034 | 14,424 | 13,363 | 14,714 | 12,932 |

Evenly spread; `99.918%` end more than an hour before the window closes. Combined with
`S1-DS-05/06`'s `ALLOWED_PATTERNS`, which already excludes every boundary-crossing session,
**every surviving VALIDATION session is complete and the correct censored count is zero** —
not "fewer".

**The TEST seal, measured.** `load_events` now consults row-group statistics before opening
anything:

| window | groups opened | rows decoded | **TEST rows decoded** | TEST rows used |
|---|---:|---:|---:|---:|
| TRAIN | 59 of 85 | 29,500,000 | **0** | 0 |
| VALIDATION | 15 of 85 | 7,500,000 | **384,203** | 0 |

TRAIN is now sealed by construction. VALIDATION cannot be, because **row group 72 straddles
`2019-10-27`** and a straddling group must be decoded to be filtered. That is a property of one
monolithic Parquet, which is exactly your argument for physically split sources.

---

## 2. The numbers now

### T2 — the gain survives, the absolute number does not

| model | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|
| prevalence | 0.0288 | 0.500 | — |
| TRAIN item popularity | 0.0657 | — | — |
| **smoothed item+category (α=100)** | **0.0760** | — | — |
| frozen encoder + T2 head | 0.1104 | 0.7488 | 0.02685 |
| **fine-tuned encoder** | **0.1424** | **0.7923** | **0.02633** |

**Gain `+0.0664`, paired client-cluster bootstrap 95% CI `[+0.0582, +0.0752]`**, 42,312
clients, 1,000 resamples, PR-AUC recomputed for both systems inside each resample.

| | published before | now |
|---|---:|---:|
| model | 0.1797 | 0.1424 |
| baseline | 0.0831 | 0.0760 |
| ratio | 2.16× | **1.87×** |

### T3 — neither loss beats the frozen retrieval order

| | macro | micro | gain | 95% CI | above zero |
|---|---:|---:|---:|---|---|
| **frozen retrieval order** | **0.2707** | **0.2046** | — | — | — |
| listwise | 0.2505 | 0.1891 | −0.0202 | `[−0.0265, −0.0142]` | no |
| pointwise | 0.2186 | 0.1713 | −0.0521 | `[−0.0584, −0.0458]` | no |
| ceiling | 0.8795 | 0.8201 | | | |

Both intervals sit entirely below zero. **The deliverable for T3 is the retrieval ordering**,
and the trained models are reported as losses.

---

## 3. What is new since you looked

### 3.1 We built your residual reranker, and it works exactly as you specified

Your acceptance criterion was *"epoch 0 exactly equals retrieval order"*. Implemented in
`SessionGRU` as a rank prior at `-1` and a ReZero scale at `0`:

```
frozen retrieval order   macro 0.270687   micro 0.204637
untrained model          macro 0.270687   micro 0.204637
difference               0.000000
```

Asserted every run, on every seed. This is the single best idea in your review: it turns *"can
the model beat popularity?"* from a comparison of two independent fits into a measured
departure from one, which is why we can state the negative result with confidence rather than
as a shrug.

Your second criterion — *"final paired macro gain at least about +0.010 with interval above
zero, otherwise use the retrieval baseline"* — is **not met**, by a wide margin and in the
wrong direction. So we use the retrieval baseline, as you specified.

### 3.2 Catastrophic negative transfer — you did not ask for this, and it is the biggest finding

Our own plan required re-measuring T1 after every head. It had never been run. T1 re-scored on
its own frozen VALIDATION cache with `S2-DS-01`'s own evaluator, unchanged:

| encoder | T1 slice macro MRR | T1 overall micro MRR |
|---|---:|---:|
| `S2-DS-01`, T1 only | 0.3479 | 0.8604 |
| after T2, **frozen** | 0.3479 | 0.8604 |
| after T2, **fine-tuned** | **0.1791** | **0.4728** |
| *model-free baseline* | *0.3143* | *0.8232* |

**Fine-tuning for T2 drives T1 below its own model-free baseline on both averages.** The
control is exact — the frozen rung reproduces T1 at `0.000000` drift, and freezing cannot
change an encoder, so the harness measures the encoder and nothing incidental.

This is not a bug: the T2 rung carries no T1 term, so nothing defended T1. But it changes the
shape of the project. `S2-DS-08`'s joint loss is not a refinement on a working arrangement —
it is what makes the arrangement possible at all.

### 3.3 The label defect was a measurement error, not a training error

The two effects separate cleanly:

| model | PR-AUC corrected | PR-AUC on the published mask |
|---|---:|---:|
| trained on the published mask | 0.1411 | **0.1797** |
| retrained on corrected labels | **0.1424** | 0.1785 |

Your rescore of the old checkpoint gave `0.1411`; ours reproduces it exactly. **Retraining on
the 478,718 restored TRAIN negatives moved PR-AUC by `+0.0013`** — inside noise. The model was
always this good; the number was always wrong.

A detail that explains it: mean predicted probability on the restored rows fell from `0.0516`
to `0.0406`, so the retrained model does learn they are negatives — but PR-AUC barely moves
because those rows are genuinely hard. They are end-of-session views: the ones that look most
like purchases and are not.

### 3.4 Seed spread replaced with paired bootstrap

Per your step 4. `paired_client_bootstrap` lives in `ppsi/models/evaluation.py` and is used by
both tasks. The T2 interval is `±0.0085` half-width against the old seed-spread "noise floor"
of `0.0029` — **three times wider**, which is the correct outcome, not a regression.

### 3.5 Contract repairs

* **One encoder loader.** T2 and T3 had grown separate copies; they diverged, and T2's copy
  then refused the checkpoint its own task is built on when `candidate_projection` widened
  40→41. Now `ppsi/models/checkpoint.py`, shared, raising on a reshaped **encoder** rather
  than silently training on random weights.
* **Checkpoints carry the batch spec** they were built under (`spec_fingerprint`).
* **`candidate_continuous_dim` is read from the spec at every call site.** Three builders had
  it as a literal `0`; two surfaced as mid-run shape errors during this work.
* A contract test caught a real defect: a 0-dimensional parameter cannot be hashed by
  `LocalTrainerCore`, so the rank weight is shape `[1]`. 23/23 model tests pass.

### 3.6 Vocabulary scope, quantified

| | |
|---|---:|
| distinct categories in full TRAIN | **595** |
| codes in `vocabulary_v1` (C1, `user_id % 4 == 1`) | **588** |
| our codes absent from full TRAIN | **0** |
| **in full TRAIN, unknown to our vocabulary** | **7** |

Clean subset, seven short. Written into `HANDOFF_TO_LANES.md` so no lane adopts a mixed-scale
comparison by accident.

---

## 4. The one step we declined, and why

**Step 2 — promote pilot artifacts to full-TRAIN artifacts — is declined.** This is a settled
project decision: all tasks continue on the frozen 25% slice.

Your underlying concern is correct and we made it explicit rather than arguing it away (§3.6).
That leaves exactly two consistent positions:

1. **Every regime stays on the slice.** Internally valid; every number carries its scope line.
   *Current decision.*
2. **Refit on full retained TRAIN** and rebuild every dependent artifact together.

The failure mode is mixing them — one lane scaling while another does not — which would let
the regime gap absorb a vocabulary difference that has nothing to do with federation.

**If you think position 1 is untenable for the R1/R2/R4 comparison specifically, say so
plainly and we will take it back to the owner.** We are not defending the decision, we are
reporting it.

---

## 5. Still open

| | status |
|---|---|
| three-seed confirmation, T2 | **done** — mean `0.1418`, spread `0.0008`, gain `+0.0661` (83× the spread) |
| three-seed confirmation, T3 | **running as this is written** — `seeds_t3.py`, listwise only |
| T1 paired client bootstrap of our own | not done; we have seed spread `0.0007` over 3 seeds |
| physically sealed TRAIN/VALIDATION Parquet | upstream; `FINDINGS_FOR_S1_LANE.md` finding 3 |
| T3 candidate lists rebuilt from TRAIN anchors | upstream; finding 2 |
| upstream T2 label fix at source | upstream; finding 1. Our correction is self-retiring |
| `S2-DS-08` joint loss | next, and §3.2 changed its brief |

**One unexplained number, reported rather than smoothed over.** The three-seed run reproduces
seed 13 at `0.1422` where the ladder gave `0.1424`. The model gained two parameters between
the runs (the rank prior and ReZero scale) and T2's loss touches neither. `0.0002` is far
inside the bootstrap interval so no conclusion moves, but we have not identified the cause and
are not going to invent one — we falsified a cuDNN-nondeterminism hypothesis earlier in this
project by measuring a spread of exactly `0.00000`.

---

## 6. Questions we actually want answered

Ranked by how much your answer would change what we do.

**Q1 — Is the T3 negative result the end of T3, or the start of a different T3?**
The learned reranker loses by `0.0202` macro. Our reading is that the frozen order already
encodes TRAIN co-occurrence, only 7.5% of TRAIN queries carry any scoring positive, and 15.1%
have no candidate list at all — so there is very little for a reranker to add. Do you agree
that is the explanation, or do you see a formulation that has a real chance? Specifically:
would a **query-aware** residual (the candidate scored against the query item's embedding
rather than only the session vector) change the picture, or is the signal simply not there?

**Q2 — Given §3.2, what should `S2-DS-08` actually weight?**
Sequential fine-tuning destroys T1. T3 has no learned component worth carrying. That leaves a
joint loss over T1 and T2 where T2's frozen rung already reaches `+0.0347` without touching the
encoder. Is the honest next experiment a joint T1+T2 loss, or is it **T1 encoder frozen with a
T2 adapter head**, with the joint loss tested only as a hypothesis against it?

**Q3 — Does the negative-transfer measurement change your R2/R4 advice?**
You wrote that R4 should mean local fine-tuning of a shared initialization, preferably a small
adapter or head. §3.2 is fairly strong independent support for that. Does it change the *size*
of adapter you would recommend, or how you would detect the same collapse happening federated?

**Q4 — Is our T2 interval the right one?**
We resample clients, recompute PR-AUC for both systems inside each resample, and difference
within the resample. Is a cluster bootstrap on PR-AUC sound as implemented, or would you want
a different estimator given PR-AUC is not a mean?

**Q5 — Anything in §1 you think we got wrong, or fixed in the wrong place?**
Particularly finding 5: we concluded the VALIDATION-derived anchors are **not** a metric leak,
because every evaluable query has its list and both systems are scored on identical candidate
sets, but **are** a deployability limit. If you think that is too generous, we would rather
hear it now.

**Q6 — What should we measure that we still have not?**
The `S2-PR-07` diagnostics — category coverage, popularity correlation, per-client divergence,
confidence reliability, failure by stratum — are specified but unrun. Per-client divergence is
the one we think matters most, because R3 and R4 rest on a client's own history changing what
they are shown, and our windows are session-local (we wrote that scope limit into
`HANDOFF_TO_LANES.md`). Is that your priority ordering too?

---

## 7. Where things are

```
FINDINGS_FOR_S1_LANE.md        three upstream defects, evidence and one-line fixes
REVIEW_RESPONSE.md             per-claim verification, and the declined step
_SEAL/test_seal_measured.json  the TEST-exposure measurement
_SEAL/vocabulary_coverage.json the 588 vs 595 measurement

S2-DS-06_T2_Purchase_Head/
  labels.py                    the censoring correction, one place, self-retiring
  interval.py                  paired client-cluster bootstrap
  compare_labels.py            measurement-error vs training-error ablation
  negative_transfer.py         T1 re-scored after each T2 rung
  seeds.py                     three-seed confirmation of the selected rung
  README.md · EXPLAINED_AR.md
  output/*.json

S2-DS-07_T3_Ranking_Head/
  ladder_t3.py · prepared.py · train_t3.py
  seeds_t3.py                  three-seed confirmation of the better loss
  README.md · EXPLAINED_AR.md
  output/s2_ds_07_ladder.json

Repo_S2DS01/ppsi/models/
  checkpoint.py                one shared encoder loader + spec fingerprint
  session_gru.py               the residual reranker
  evaluation.py                paired_client_bootstrap
```

Everything is committed on `s2-ds-01-gru-t1`. Six commits, not yet pushed.

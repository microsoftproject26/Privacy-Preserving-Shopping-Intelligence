# S2-DS-06 — the T2 purchase-likelihood head

**The decision:** the first time a product is viewed inside a session.
**The label:** whether that same product is purchased later in the same session.

> Rewritten 2026-09-11. An independent review found a label defect upstream; we reproduced it
> exactly and corrected it. The previously published `0.1797` was measured on a population
> that had dropped 70,467 observed negatives while keeping every positive.

---

## Result

All figures on **corrected labels** — every complete session's outcome counted, which is all
of them. See *The censoring defect* below.

| model | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|
| prevalence — no signal | 0.0288 | 0.500 | — |
| TRAIN item popularity | 0.0657 | — | — |
| **smoothed item+category (α=100)** | **0.0760** | — | — |
| frozen encoder + T2 head | 0.1104 | 0.7488 | 0.02685 |
| **fine-tuned encoder** | **0.1424** | **0.7923** | **0.02633** |

**Gain over the strongest simple rule: `+0.0664`, 95% CI `[+0.0582, +0.0752]`** — paired
client-cluster bootstrap over 42,312 clients, entirely above zero. The model scores **1.87×**
the best model-free rule and **4.9×** prevalence.

That interval is **three times wider** than the `0.0029` seed spread this task used to quote
as its noise floor, and that is the correct outcome, not a regression: resampling shoppers is
a harder test than re-running the same shoppers with a different seed. The conclusion is
unchanged and now rests on the right statistic.

### What changed from the first publication, and why

| | published | now | why |
|---|---:|---:|---|
| model PR-AUC | 0.1797 | **0.1424** | scored on all 392,554 decisions, not 322,087 |
| best simple baseline | 0.0831 | **0.0760** | a stronger rule, on the same corrected population |
| prevalence | 0.0351 | **0.0288** | the denominator had dropped negatives only |
| ratio to best simple | 2.16× | **1.87×** | |

Both numbers moved together, which is the point: the **relative** claim barely changed while
the absolute one was clearly inflated.

---

## The censoring defect

Upstream states the rule correctly — *"a decision with no observable horizon is censored,
never negative"* — and then implements a narrower one:

```python
t2["censored"] = (t2["label"] == 0) & (t2["query_order"] == t2["session_end"])
```

`session_end` is the end of the **session**, not of the observable window. But the label asks
*"purchased later in this same session?"*, and once the session is over the answer is settled:
no. There is no unobserved horizon, because the horizon the question is scoped to has closed.

The two coincide only when the window cut a session short — and `S1-DS-05/06` already excluded
every boundary-crossing session. **So no VALIDATION session is truncated, and the correct
count of censored decisions is zero.**

Session-end times of the 70,467 withheld rows, against a `[10-22, 10-27)` window:

| 10-22 | 10-23 | 10-24 | 10-25 | 10-26 |
|---:|---:|---:|---:|---:|
| 15,034 | 14,424 | 13,363 | 14,714 | 12,932 |

Spread evenly — ordinary sessions ending, not a window edge truncating them. `99.918%`
finished more than an hour before the window closed.

| | rows | withheld | |
|---|---:|---:|---|
| TRAIN | 2,770,471 | 478,718 | 17.28% |
| VALIDATION | 392,554 | 70,467 | 17.95% |

The correction lives in `labels.py`, in one place, written to become a pass-through the
moment upstream emits zero censored rows. The upstream fix is in `FINDINGS_FOR_S1_LANE.md`.

### The defect was in the measurement, not in the model

Two separable effects, easy to conflate, so they were measured apart — `compare_labels.py`,
`output/s2_ds_06_label_ablation.json`:

| model | PR-AUC corrected | PR-AUC on the published mask |
|---|---:|---:|
| trained on the published mask | 0.1411 | 0.1797 |
| retrained on corrected labels | **0.1424** | 0.1785 |

**Retraining on 478,718 restored negatives moved PR-AUC by `+0.0013`** — inside the noise.
The model was always this good; the number was always wrong.

One detail explains why the extra data changed so little. Mean predicted probability on the
restored rows fell from `0.0516` to `0.0406` — the retrained model does learn they are
negatives — but PR-AUC barely moves, because those rows are genuinely **hard**: they are
end-of-session views, the ones that look most like purchases and are not.

---

## Negative transfer — the finding this task exists to produce

The plan called for re-measuring T1 after every head and it had not been done. The review did
not ask for it either. It is the single most consequential number here.

T1 re-scored on its own frozen VALIDATION cache with `S2-DS-01`'s own evaluator, unchanged:

| encoder | T1 slice macro MRR | T1 overall micro MRR |
|---|---:|---:|
| `S2-DS-01`, T1 only | 0.3479 | 0.8604 |
| after T2, **frozen** | 0.3479 | 0.8604 |
| after T2, **fine-tuned** | **0.1791** | **0.4728** |
| *model-free baseline* | *0.3143* | *0.8232* |

**Fine-tuning the encoder for T2 drives T1 below its own model-free baseline on both
averages.** The encoder has not been degraded; it has lost the category-transition structure
outright.

The control is exact: the frozen rung reproduces T1 with a drift of `0.000000`. Freezing
cannot change an encoder, so the harness is measuring the encoder and nothing incidental to
how a checkpoint was loaded.

**This is not a bug.** The T2 rung optimises a purchase loss with no T1 term, so nothing was
defending T1. The measurement is of *how much*, not of *whether*. What it changes:

* `S2-DS-08`'s joint loss is not a refinement on top of a working arrangement. **It is the
  only thing that makes the arrangement possible.** Training the heads one after another
  destroys the shared representation.
* `S2-DS-ST1` (shared versus separate models) now has a measured cost of sharing instead of
  an assumption.

**What it does not say:** that a shared encoder cannot serve both. The frozen rung reaches
`+0.0347` over the T2 baseline *with the T1 encoder untouched*, so the T1 representation
demonstrably carries purchase signal. Whether a **jointly** trained encoder can hold both at
once is exactly the open question — and it is now a quantified one.

---

## The prediction, and what it settled

Written into `run_t2.py` before either rung ran:

> *the fine-tuned encoder wins, because the T1 encoder was trained to separate categories and
> purchase intent is a different question. If freezing wins instead, the shared representation
> is already carrying intent and `S2-DS-08` has an easier job than expected.*

Fine-tuning won by `+0.0320`. Both halves proved informative:

| | |
|---|---|
| the T1 representation **already carries** purchase signal | frozen: +0.0347 over baseline |
| it does **not** carry all of it | fine-tuning adds a further +0.0320 |
| **and taking that +0.0320 costs T1 everything** | −0.1688, below baseline |

The third line was not predicted, and it is the one that matters most.

---

## Why PR-AUC and not accuracy

**11,297 positives in 392,554 decisions — a base rate of 2.88%.**

A model that answers "never" scores **97.1% accuracy**. Accuracy here is not a weak metric,
it is an actively misleading one, which is why the protocol fixed PR-AUC as the headline
before any of this was measured.

Brier and Log Loss are reported on the raw probability, uncalibrated, because plain BCE was
enough — `pos_weight` was never needed. `pos_weight` lifts ranking metrics while destroying
probability calibration, and the calibration here is good as it stands.

### The interval

`HALF_WIDTH = 0.0029` was a **seed spread** — max minus min of three runs on the same 42,312
clients. It answers *"would I get this number again?"*, not *"would this gain survive a
different sample of shoppers?"* Those are different questions, and only the second is what a
headline claims.

`interval.py` answers the second directly: clients are resampled with replacement, PR-AUC is
recomputed for **both** systems inside each resample, and the difference is taken within the
resample. Pairing matters — a hard client drags both scores down at once, and differencing
inside the resample removes that shared difficulty instead of counting it as uncertainty.

Clients, not rows, are the unit. A client contributes many correlated decisions; resampling
rows would treat them as independent evidence and produce an interval far too narrow.

---

## T2 overfits fast, and T1 did not

| | best epoch | of |
|---|---:|---:|
| T1 (`S2-DS-01`) | 29 | 30 |
| **T2, frozen** | **2** | 12 |
| **T2, fine-tuned** | **2** | 12 |

By epoch 4 the PR-AUC is falling while the training loss keeps dropping. The reason is the
positives: **11,297 of them**. T1 had 3.1M decisions each carrying a 586-way label; T2 has a
rare binary event. The signal is small and the model memorises it quickly. `S2-DS-07` and
`S2-DS-08` should expect this shape rather than reusing T1's thirty-epoch schedule.

---

## The gate

Everything asserted before a single epoch ran:

| check | measured | expected |
|---|---:|---:|
| TRAIN withheld upstream | 17.28% | 17.28% |
| VALIDATION withheld upstream | 17.95% | 17.95% |
| restored rows that are positives | 0 | 0 |
| decisions, corrected | 392,554 | 392,554 |
| decisions, published mask | 322,087 | 322,087 |
| positives | 11,297 | 11,297 |
| base rate, corrected | 0.0288 | 0.0288 |
| base rate, published mask | 0.0351 | 0.0351 |
| prevalence PR-AUC, published mask | **0.035074** | 0.0351 |

Asserting on the **published** mask is what proves we are on the same rows upstream
published. Without it, a corrected number would be comparable to nothing.

Restored rows are additionally verified present in the batch and carrying an observed zero,
through the **canonical** validator rather than the runtime one — the mirror of the old
check, which asserted the same rows were absent.

### One correction on the way

PR-AUC was hand-written first, to avoid depending on a library's tie convention. That was
wrong twice over: `scikit-learn==1.9.0` is already pinned, and a second implementation of a
metric is exactly the defect that has cost this project three wrong numbers.

The hand-written version agreed with sklearn to six decimal places on any input with distinct
scores, and disagreed only in the degenerate all-tied case — `0.03540` against the true
`0.03507`. A trained model produces distinct scores, so it never mattered in practice. It
mattered in the **gate**, which scores a no-signal baseline where everything ties. Both
metrics now delegate to sklearn.

---

## Scope

| | |
|---|---|
| encoder | `s2_ds_01_gru_t1_seed13.pt`, unchanged from `S2-DS-01` |
| cohort | `C1`, the 25% slice — 97,279 clients |
| TRAIN | 2,770,471 decisions, **all contributing** |
| VALIDATION | 392,554 decisions, **all scored** |
| clients evaluated | 42,312 |
| labels | corrected — see `labels.py` |
| `TEST` | never used; the seal is measured in `_SEAL/test_seal_measured.json` |

**Not yet done:** three-seed confirmation. The gain is far outside any plausible noise, so the
conclusion is not in doubt, but the reported figure is one seed and says so.

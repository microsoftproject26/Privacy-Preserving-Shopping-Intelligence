# Response to the independent review of the S2 model lane

**Date:** 2026-09-11 · **Reviewer verdict:** *ADJUST, then proceed*

Every material claim in the review was re-derived here from the artifacts before anything was
changed. **All six reproduced.** Several matched to more decimal places than the review
quoted. Nothing was accepted on authority and nothing was rejected on pride.

## What we verified independently

| Claim | Reviewer | Our own measurement | |
|---|---|---|---|
| T2 corrected prevalence | `0.028778` | `0.028778` | ✓ |
| smoothed item+category baseline | `0.097269` | `0.097407` | ✓ |
| T3 correct micro baseline | `0.204637` | `0.204637` | ✓ |
| T3 macro baseline | `0.270687` | `0.270687` | ✓ |
| T3 macro ceiling | `0.879421` | `0.879460` | ✓ |
| clients with a scoring query | `4,591` | `4,591` | ✓ |
| TRAIN queries with no candidate list | `263,112` (15.12%) | `263,112` (15.12%) | ✓ |
| full-TRAIN rows / categories | `29,218,702` / `595` | `29,218,702` / `595` | ✓ |

Two we pushed further than the review did:

* **The censoring defect is not a judgement call.** We measured *when* the sessions of the
  70,467 withheld rows actually end: spread evenly across all five days of the window, with
  `99.918%` finishing more than an hour before it closes. And `S1-DS-05/06` already excludes
  every boundary-crossing session, so each surviving session is complete inside its split.
  The correct count of censored T2 decisions is **zero**, not "fewer".
* **The TEST seal.** The review established the claim was unmeasured. We measured it: TRAIN
  loading decodes **0** TEST rows once row-group statistics are consulted; VALIDATION decodes
  **384,203**, all from row group **72**, the single group straddling `2019-10-27`.

## What we changed

| | |
|---|---|
| `ndcg_at_k` | full-oracle denominator, now a required argument; gate tightened `0.005` → `0.001` |
| T3 reporting | macro headline with micro beside it, per `ADR-001`; selection on macro |
| T3 model | **residual reranker** anchored at the retrieval order — untrained NDCG equals the baseline to six decimals, asserted before training |
| T2 labels | `labels.py`, one place, self-retiring; both populations reported |
| T2 baseline | `0.0831` → `0.0974` published-mask / `0.0757` corrected |
| encoder loading | one shared implementation; raises on a reshaped encoder instead of training on random weights |
| checkpoints | carry the batch spec they were built under |
| uncertainty | paired client-cluster bootstrap replaces seed spread |
| raw loader | row-group statistics skipping plus measured provenance |

The review's own suggestion — the zero-initialised residual reranker — is the single best
idea in it, and it is now the T3 design. It converts *"can the model beat popularity?"* from
a comparison of two independent fits into a measured departure from one.

## The one step we are not taking, and why

**Step 2, "promote pilot artifacts to full-TRAIN artifacts", is declined for now.** This is a
settled project decision, not an oversight: the frozen 25% slice is the data all tasks
continue on.

The reviewer's underlying concern is nonetheless correct and we have made it explicit rather
than arguing it away. The vocabulary is fitted on `C1, user_id % 4 == 1` and holds **588**
category codes; full TRAIN contains **595**. Our codes are a clean subset — none is spurious
— but seven categories would map silently to OOV at full scale.

That gives exactly two consistent positions, and the project must hold one of them:

1. **Every regime stays on the slice.** The comparison is internally valid, and every number
   carries its scope line. *This is the current decision.*
2. **Refit on full retained TRAIN** and rebuild every dependent artifact together.

The failure mode is mixing them — one lane scaling while another does not. Then the regime
gap, which is this project's headline, would silently absorb a vocabulary difference that has
nothing to do with federation. This is written into `HANDOFF_TO_LANES.md` so no lane can
adopt it by accident.

## What the review got right that we would not have found

The IDCG defect is the one that matters most. It is small — `0.2056` against a true `0.2047`
— and it moves in the direction that looks like success, which is exactly why it survived our
own gate. The gate's tolerance was `0.005`, five times the size of the error it was meant to
catch. That is a lesson about tolerances, not about NDCG.

## Standing disagreement: none

We found no claim in the review that does not hold.

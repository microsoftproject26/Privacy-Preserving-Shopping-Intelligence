# S2-DS-08 — the joint objective

**The question:** can one encoder serve T1 and T2 at once, or does sharing have to cost?

**The answer: yes, with a joint loss — and only with one.**

---

## Result

| arrangement | T1 slice macro | T2 PR-AUC | T1 cost |
|---|---:|---:|---:|
| *T1 alone — the starting point* | *0.3479* | — | — |
| frozen encoder + T2 head | 0.3479 | 0.1104 | 0.0000 by construction |
| **sequential** — fine-tune the encoder for T2 | **0.1791** | 0.1424 | **−0.1688** |
| joint, `λ_T2 = 0.1` | 0.3457 | 0.1197 | −0.0022 |
| joint, `λ_T2 = 0.3` | 0.3471 | 0.1250 | −0.0008 |
| **joint, `λ_T2 = 1.0`** | **0.3467** | **0.1337** | **−0.0012** |
| *model-free baselines* | *0.3143* | *0.0760* | |

**The joint objective takes 87% of what sequential fine-tuning gets on T2 and pays `0.0012`
of T1 for it.** Sequential pays `0.1688` and lands below T1's own model-free baseline.

That is the whole finding. The shared-backbone premise this project is built on does not
survive training the heads one after another, and it does survive a joint loss.

---

## Why this task changed shape

It was scoped as a refinement: pick weights for three losses that already work. `S2-DS-06`
made it structural. Fine-tuning the encoder for T2 alone takes T1 from `0.3479` to `0.1791`
slice macro and `0.8604` to `0.4728` overall micro — **below the model-free baseline on
both**. The encoder does not degrade, it loses the category-transition structure outright.

So a joint objective is not an optimisation on top of a working arrangement. It is the only
thing that makes the arrangement exist.

---

## Selection: Pareto-constrained, and the margin was fixed first

Picking the epoch with the best T2 is exactly what produced the sequential collapse — nothing
in a T2 metric can see T1 failing. So every epoch is scored on **both** tasks, and the rule
was written before the run, taken from the review that specified this task:

> **maximise T2 PR-AUC, subject to T1 slice macro staying within `0.003` of `0.3479`.**

`0.003` is roughly four times T1's own three-seed spread of `0.0007` — loose enough not to
reject a run over noise, tight enough that a real regression cannot hide inside it. An epoch
that breaks the constraint is printed as `refused`, not dropped.

**No epoch at any weight broke it.** The constraint never had to bite, which is itself the
result: the tension everyone expected between the two tasks did not appear.

---

## Starting from the trained encoder, not from scratch

Training continues from `s2_ds_01_gru_t1_seed13.pt`. If it started cold, T1 would sit below
`0.3479` simply from having had fewer epochs, and nothing could separate that from
interference. Starting at `0.3479` makes every point of T1 loss attributable to the joint
objective and to nothing else.

The learning rate is `0.0003` — lower than either task used alone — for the same reason: this
is a continuation, not a fresh fit.

---

## What the weights show

| `λ_T2` | T2 | T1 cost |
|---:|---:|---:|
| 0.1 | 0.1197 | −0.0022 |
| 0.3 | 0.1250 | −0.0008 |
| 1.0 | 0.1337 | −0.0012 |

T2 rises monotonically with the weight and T1 does not respond. There is no visible trade-off
inside the tested range.

**That is an honest gap in this experiment, not a strength.** The ladder `(0.1, 0.3, 1.0)`
was fixed before the run, and `1.0` — the best point — sits on its edge. **The breaking point
was not found.** Extending the sweep now, having seen that higher is better, would be tuning
until the number improves, which is the deception this project's protocol exists to prevent.

The correct next step is a **separately pre-registered** sweep upward (`3.0`, `10.0`) asking
one question: where does T1 start to pay? Until that runs, the honest claim is *"a joint loss
holds both tasks at every weight we tried"*, not *"λ = 1.0 is optimal"*.

---

## What this hands to the federated lane

1. **Do not fine-tune the full backbone locally.** Measured: it costs `0.1688` of T1. `R4`
   should update a small adapter or head over a frozen base, which is what the review
   recommended and what this measurement independently supports.
2. **A shared encoder is viable**, so `R1` and `R2` can legitimately compare one architecture
   serving both tasks rather than two separate models.
3. **Evaluate both tasks after every federated round**, against a fixed central validation
   set, and roll back on a T1 non-inferiority break. The failure mode here was invisible to
   the task being optimised, and it will be invisible again.
4. **`S2-DS-ST1` (shared versus separate) now has numbers**: sharing costs `0.0012` of T1 and
   buys `+0.0233` of T2 over the frozen control at `λ = 1.0`.

---

## T3 is deliberately absent

No T3 loss enters the shared encoder. `S2-DS-07` has no learned reranker that beats the
frozen retrieval order, so weighting a T3 term would spend encoder capacity on a task with no
demonstrated gain. The review asked for it to stay out until a useful reranker exists, and
that is the right call.

---

## Scope

| | |
|---|---|
| started from | `s2_ds_01_gru_t1_seed13.pt` |
| cohort | `C1`, the 25% slice — 97,279 clients |
| T1 | 3,113,814 TRAIN decisions, 438,185 VALIDATION |
| T2 | 2,770,471 TRAIN decisions (corrected labels), 392,554 VALIDATION |
| epochs | 8 per weight, batch 512 on each task |
| seed | 13 |
| `TEST` | never used; the seal is measured in `_SEAL/test_seal_measured.json` |

**Not yet done:** three seeds, and the upward sweep described above. The T1 cost at every
weight is within two to three times T1's own seed spread, so single-seed numbers carry real
uncertainty about the *size* of the cost — though not about its sign, and not about the
difference from sequential fine-tuning, which is two orders of magnitude larger.

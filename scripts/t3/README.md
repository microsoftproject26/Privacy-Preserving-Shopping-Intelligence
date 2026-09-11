# S2-DS-07 — the T3 candidate-ranking head

> ## ⚠ Two corrections — 2026-09-11, after a second review
>
> **1. The reranker this task first measured could not see the query item.** `t3_scores`
> was `session · candidate`; the query vector was computed and used only by the T2 head.
> Measured by permuting every query tensor across a batch: T3 moved by **exactly `0.0`**
> while the T2 control moved by `0.81`. The frozen retrieval order *is* co-occurrence
> between the query item and each candidate, so that model could not represent what the
> baseline does, let alone beat it. **The `0.2505` result below is evidence about a dot
> product, not about T3.**
>
> A query-aware cross-feature reranker has since been built and run on three seeds with a
> frozen encoder and an acceptance bar fixed beforehand. It does better — peak `0.2599`
> against the query-blind `0.2505` — and **still loses to the frozen order's `0.2707` on
> every seed.** The deliverable is unchanged; the reason for it is not.
>
> **2. The explanation given for that failure was wrong, and the correction matters more
> than the failure.** This file used to say the retrieval order leaves little for a reranker
> to add. Measured: **52.27% of evaluable queries have the engaged product in the candidate
> list but ranked outside the top 5**, and a perfect reranking of those same lists scores
> `0.8782` macro against the frozen order's `0.2707`. **The contest is worth `+0.6075` and
> our models have taken none of it.** The headroom is enormous and the failure is ours.
>
> **Still outstanding before a third attempt:** the candidate lists are built only for query
> items that had a VALIDATION positive, so 6.14% of TRAIN queries can drive a listwise loss.
> The review asks for them to be rebuilt from TRAIN anchors first, and that is upstream
> (`FINDINGS_FOR_S1_LANE.md`). The experiment above ran before that fix, and is caveated by
> it.
>
> `test_t3_uses_the_query_item` and `test_t3_starts_exactly_at_the_retrieval_order` exist so
> the first defect cannot recur.

---

**The task:** given a query item, rank its frozen candidate list so the products the user
actually engages with next come first.

**The result: neither learned loss beats the frozen retrieval ordering.** Both lose, and both
confidence intervals sit entirely below zero. The deliverable for T3 is the retrieval order
itself, and the trained models are reported as losses.

---

## Result

Headline is **macro** — `ADR-001` fixes that for this project and T1 already follows it.
Micro is printed beside it because the two differ a lot here.

| | macro NDCG@5 | micro NDCG@5 | gain over baseline | 95% CI | above zero |
|---|---:|---:|---:|---|---|
| **frozen retrieval order** | **0.2707** | **0.2046** | — | — | — |
| listwise softmax CE | 0.2505 | 0.1891 | **−0.0202** | `[−0.0265, −0.0142]` | no |
| pointwise squared error | 0.2186 | 0.1713 | **−0.0521** | `[−0.0584, −0.0458]` | no |
| *ceiling* | *0.8795* | *0.8201* | | | |

Intervals are paired client bootstraps over the 4,591 clients with a scoring query. Neither
interval touches zero, so these are not ties — the learned rerankers are **measurably worse**
than doing nothing.

---

## Why that conclusion is trustworthy: the model starts at the baseline

The first T3 attempt asked the model to out-rank a popularity-and-co-occurrence ordering
while giving it no way to see that ordering. It scored `0.0996` against `0.2046`. Adding the
retrieval rank as a feature fixed the collapse, but left the model obliged to *rediscover*
the ordering before it could improve on it — so most of its capacity went on reproducing a
number we already had, and any gain was tangled up with how well it managed that.

This version removes the obligation. Two parameters in `SessionGRU`:

```python
self.t3_rank_weight    = nn.Parameter(torch.tensor([-1.0]))   # the frozen ordering
self.t3_residual_scale = nn.Parameter(torch.zeros(1))         # the learned part, off
```

so `t3_scores = t3_rank_weight · rank + t3_residual_scale · (session · candidate)`.

At initialisation the score **is** the negated retrieval rank. Measured, before a single
gradient step:

```
frozen retrieval order   macro 0.270687   micro 0.204637
untrained model          macro 0.270687   micro 0.204637
difference               0.000000
```

The ladder asserts this every run. It means every later point is a **measured departure from
the frozen ordering**, not a difference between two independent fits — so "training made it
worse by 0.0202" is a statement about training, and nothing else. The idea came from the
independent review and it is the best idea in it.

(The zero scale still receives gradient immediately — `d(loss)/d(scale)` is the raw score,
not zero — so this delays learning rather than preventing it. It is ReZero.)

---

## The prediction, and what it settled

Written into `ladder_t3.py` before either loss ran:

> *listwise wins, because NDCG scores an ordering and pointwise never sees one. If pointwise
> wins instead, the gains are separable enough that absolute calibration carries the ranking.*

**Half right, and the wrong half was the important one.** Listwise did beat pointwise, by a
wide margin — `0.2505` against `0.2186`, and the reason given was the right reason: the
pointwise loss drives every score toward its own gain in isolation, and with 98.1% of gains
equal to zero it learns to predict zero everywhere, which carries no ordering at all. Its
first epoch scored `0.0333`.

What the prediction never contemplated was that **both would lose to not training**. That is
the finding.

---

## Why learning fails here — and the first explanation was wrong

The original version of this section said the frozen retrieval order leaves little to add,
and offered sparse training signal as the reason. **Measuring it refutes that**
(`miss_decomposition.py`, `output/s2_ds_07_miss_decomposition.json`). Where the 18,814
evaluable queries actually stand under the frozen ordering:

| | queries | |
|---|---:|---:|
| no candidate list at all | 0 | 0.00% |
| retrieval miss — list exists, engaged item absent | 3,207 | 17.05% |
| **present, but ranked outside the top 5** | **9,835** | **52.27%** |
| already inside the top 5 | 5,772 | 30.68% |

Only the third row is a reranking problem, and it is **over half of all evaluable queries**.
The engaged product is sitting in the candidate list, somewhere between rank 5 and rank 99,
and the frozen order has buried it.

| | macro NDCG@5 |
|---|---:|
| frozen retrieval order | 0.2707 |
| **a perfect reranking of the same lists** | **0.8782** |
| best learned attempt so far | 0.2599 |

**The contest is worth `+0.6075` macro and our rerankers have captured none of it.** That is
a much less comfortable conclusion than "there was nothing to win", and it is the one the
measurement supports: the headroom is enormous, the task is genuinely learnable in principle,
and two architectures have now failed to take any of it.

Note also that `0.8782` and not `0.8795` is the number a reranker should be measured against.
The oracle ceiling includes the 17.05% retrieval never returned, which no ranking of the
returned list can reach.

### What is known about the difficulty

| | |
|---|---:|
| TRAIN queries | 1,740,433 |
| …with a retrieved positive, so a listwise loss can use them | **106,910 — 6.14%** |
| candidates carrying a non-zero gain | 1.9% |
| candidates per query | 100, all same-category and co-occurring |

So the model must pick one product from a hundred plausible ones, trained on 6% of the
available queries, and the candidate set is by construction full of near-misses. Whether
that is the binding constraint has not been established — it is the hypothesis to test next,
not a conclusion.

---

## The metric defect this task was reporting through

`ndcg_at_k` took its ideal ranking from the gain **matrix**, which holds one column per
*retrieved* candidate. A query whose engaged product was never retrieved was therefore judged
against an ideal that had already forgotten it.

Full misses were handled — `idcg == 0` returns zero. **Partial** misses were not, and there
are **485** of them: a query with two positives, one retrieved and one missed, scored a
perfect 1.0 for ranking the retrieved one first.

| | frozen retrieval order, micro |
|---|---:|
| IDCG from retrieved gains (what we had) | 0.2056 |
| IDCG from the full oracle (correct) | **0.2046** |

`0.2046` is exactly what upstream published. Upstream had always computed it correctly; the
drift was ours alone.

**The gate could not have caught it.** Its tolerance was `0.005` — five times the size of the
`0.0010` error. That is a lesson about tolerances, not about NDCG. It is now `0.001`, and the
denominator is a required argument of `ndcg_at_k` rather than something derivable from the
gains, so the mistake cannot be made silently again.

---

## Macro and micro are not the same question here

| | macro | micro |
|---|---:|---:|
| frozen retrieval order | 0.2707 | 0.2046 |
| ceiling | 0.8795 | 0.8201 |

T3 was first reported in micro while `ADR-001` fixes macro as the headline. The gap is large
because micro lets clients with many queries dominate, and those clients have the harder
queries. Both are now reported, selection is on macro, and the gate asserts both.

---

## The ceiling is 0.8795, not 1.0

**18.4% of scoring positives were never retrieved.** No reranking can recover an item that is
not in the list, so end-to-end NDCG gives those zero rather than hiding them. `0.35` against a
ceiling of `0.8795` is a different statement from `0.35` against `1.0`, and only one of them
is true.

---

## Scope limit: the candidate lists were selected using VALIDATION outcomes

From the upstream notebook, stated plainly in its own comment:

```python
# Only anchors with a positive are ever scored, so only those need a list.
anchor_meta = queries.loc[queries["query_item"].isin(positives["query_item"].unique()), ...]
```

`positives` are VALIDATION outcomes, so candidate lists exist only for query items that
turned out to have a positive in VALIDATION.

**This is not a metric leak.** Every one of the 18,814 evaluable queries has its list, and the
model and the baseline are scored on identical candidate sets, so the comparison above is
fair. We checked that specifically.

What it does limit:

| | |
|---|---:|
| TRAIN queries unreachable for training | **263,112 — 15.1%** |
| catalogue coverage | 47,948 anchors — **42.1%** of TRAIN catalogue items |

A sixth of the training signal is out of reach, and the candidate generator cannot serve an
arbitrary query item — which is exactly what an on-device recommender would face. **The number
above is an honest measure of reranking within these lists; it is not evidence about a
deployable retrieval stage,** and `S2-SE-*` must not read it as one. The upstream fix is
written up in `FINDINGS_FOR_S1_LANE.md`.

---

## The gate

Asserted before any training:

| check | measured | expected |
|---|---:|---:|
| evaluable queries | 18,814 | 18,814 |
| clients with a scoring query | 4,591 | 4,591 |
| frozen retrieval order, macro | 0.2707 | 0.2707 |
| frozen retrieval order, micro | 0.2046 | 0.2046 |
| untrained model vs baseline | 0.000000 | 0.000000 |
| canonical validator on a T3 batch | passes | passes |
| distinct candidate categories | 37 | > 2 |

That last pair exists because the first version filled candidate category and price band with
pad ids, and `validate_canonical_phase1_batch` had never been run on a T3 batch. It now runs
before training starts.

---

## What this hands to `S2-DS-08`

1. **T3 has no learned component worth carrying.** Weighting a T3 loss into a joint objective
   currently buys nothing and costs encoder capacity. If T3 is included, it should be to test
   whether *joint* training finds something single-task training could not — stated as that
   hypothesis, not as an assumed gain.
2. **The anchoring pattern transfers.** Any head that has a strong frozen baseline should
   start at it, so that its contribution is a measured departure rather than a comparison of
   two fits.
3. **The bar is public:** `+0.0202` macro is what a T3 model has to recover before it is even
   level with doing nothing.

---

## Scope

| | |
|---|---|
| encoder | `s2_ds_01_gru_t1_seed13.pt`, unchanged from `S2-DS-01` |
| cohort | `C1`, the 25% slice — 97,279 clients |
| TRAIN | 1,740,433 decisions |
| VALIDATION | 18,814 evaluable queries over 4,591 clients |
| retrieval | `cooccurrence_then_popularity`, read from the frozen protocol |
| gains | `INTENT` — purchase 2, cart 1, view 0 |
| `TEST` | never used; the seal is measured in `_SEAL/test_seal_measured.json` |

**Not yet done:** three-seed confirmation. Both intervals are far from zero and on the wrong
side of it, so the conclusion is not in doubt, but the reported figures are one seed and say
so.

**Checkpoints:** `t3_listwise.pt` and `t3_pointwise.pt` hold the best *trained* epoch of each
loss. Both are worse than the frozen ordering and neither is a deliverable; they are kept so
the losing rungs stay reproducible. The selection code now treats epoch 0 as a candidate, so a
future run that never beats its anchor will say so and select the anchor rather than shipping
the least-bad trained epoch.

# S2-DS-01 — the shared session encoder and its T1 head

*Merges `S2-DS-04`, the first centralized R1–T1 pilot. `S2-SMOKE` already delivered a small
training smoke; repeating it would have produced nothing.*

---

## The one number to read correctly

**`0.3143`** is the number to beat: the TRAIN transition table, macro-averaged, on the
category-change slice. It was fixed in `ADR-001` **before any of these results existed**.

Everything below is measured against it. The absolute value of any single figure is not the
result — the distance above `0.3143` is.

| model | slice macro MRR@20 |
|---:|---:|
| repeat the current category | 0.0000 |
| globally most popular (off-diagonal) | 0.1049 |
| **TRAIN transition table** | **0.3143** |
| `S2-SMOKE` GRU | 0.3348 |
| **this model** — mean of seeds 13, 42, 2026 | **0.3474** |

**Gain: `+0.0331`.** The three seeds landed at `0.3479`, `0.3472`, `0.3472` — a spread of
**`0.0007`**, so the gain is **47× the spread**.

That spread is five times tighter than the `0.0033` this task used as its comparison
threshold. The threshold was measured on the *calibration* configuration, which trained for
four epochs at a rate that stopped it early; the selected configuration, annealed over
thirty, is markedly more stable. Reporting the wider figure was the conservative choice at
the time, and it stays as the bar every ablation was judged against.

---

## What was actually learned

Three questions were asked of the data, and two of them came back "no".

### More features did not help

| rung | change | slice macro | step |
|---|---|---:|---:|
| 1 | all 3,113,814 TRAIN decisions | 0.3373 | — |
| 2 | + product identity | 0.3367 | **−0.0006** |
| 3 | + brand | 0.3375 | +0.0008 |
| 4 | + price band | 0.3390 | +0.0015 |
| 5 | two layers | 0.3370 | −0.0020 |

Every step is smaller than the **0.0033** run-to-run spread. Not one of the added channels,
and not the second layer, made a difference that could be told apart from training again.

**A prediction was made before rung 2 and it was wrong.** `S2-SMOKE` measured the product
channel at **−0.0031** while overfitting from epoch 2, on 1.2M decisions. The stated
expectation was that 2.6× the data would turn it positive. It measured **−0.0006**. The
hypothesis is recorded as falsified, as the `modulo → blake2b` hash hypothesis was before
it — that one predicted an improvement and measured −0.0103.

### A bigger model did not help

`hidden 256` scored below `hidden 128`. A second GRU layer scored below one. The final model
is the **smallest configuration tried**: 2,329,789 parameters over two feature channels.

That is the best possible outcome for a project whose premise is a model running on a phone.
Had the gain come from five channels and a wider hidden state, the deployment lane would
have inherited a problem.

### The model was undertrained, and that was the whole story

| learning rate | slice macro | best epoch |
|---|---:|---:|
| 0.002 | 0.3390 | 3 |
| 0.001 | 0.3436 | 9 |
| 0.0005 | 0.3454 | 14 |
| 0.0003 | 0.3470 | 14 |
| **cosine from 0.001** | **0.3478** | 30 |

The best epoch moving from **3 to 30** is the finding, not the rate. The model was reaching
a mediocre optimum quickly and early stopping was calling it finished. Lowering the rate let
it keep going.

Following that one axis bought **+0.0088** — more than every feature channel and the extra
layer combined. Stopping where the coordinate sweep stopped would have closed this task at
`0.3390` and reported that the model had reached its limit.

**Selected:** cosine from `0.001` over 30 epochs, dropout `0.3`. The schedule was chosen
over the constant `0.0003` even though they are indistinguishable, because `0.0003` is a
number arrived at by trial and a schedule is a rule. "Why 0.0003?" has no answer but "it was
tried".

---

## Is it a recommender, or a popularity table with extra steps?

A model that only ever names the few most common categories scores well — those categories
genuinely are common — and recommends nothing. The headline metric cannot see that.

| diagnostic | measured on the final model |
|---|---|
| categories ever placed in a top-5 | **573 of 588** (97.45%) |
| distinct top-1 predictions overall | **547** |
| share taken by the most common top-1 | 23.51% |
| **distinct top-1s per client** | **2.342** |

Tuning did not narrow the model. A configuration that gains on the metric by retreating to
the popular few would show it here — coverage falling, one prediction taking a larger share.
Neither moved: 571 → 573 categories, 540 → 547 distinct top-1s.

**The last row is the one that matters to this project.** `R3` personalizes and `R4` trains
on the device; both assume a client's own history changes what they are shown. At 1.0 the
model would be giving everyone the same answer and neither regime would have anything to
personalize. At 2.34 it is responding to the session.

### The gain reaches the clients with the least history

| TRAIN events | clients | baseline | model | gain |
|---|---:|---:|---:|---:|
| **10–19** | 2,924 | 0.3302 | 0.3632 | **+0.0330** |
| 20–49 | 6,209 | 0.3111 | 0.3435 | +0.0324 |
| 50–99 | 3,946 | 0.3088 | 0.3401 | +0.0313 |
| 100+ | 3,017 | 0.3128 | 0.3489 | **+0.0361** |

Flat. A client with ten to nineteen TRAIN events gains **+0.0330**; one with a hundred or
more gains **+0.0361**. Had the advantage been concentrated in the 100+ bucket, device-local
learning would only have helped power users and `R4` would be answering a much smaller
question than the proposal claims.

---

## What a person actually experiences

MRR is the protocol metric and the right one for comparing regimes. Nobody experiences an
MRR. These are the baseline's hit rates; the model's are in the result file.

| view | HR@1 | HR@5 | HR@10 |
|---|---:|---:|---:|
| **all decisions** — what a user meets | 82.35% | **89.66%** | 92.36% |
| category-change slice — the hard 17.68% | 18.61% | 45.65% | 58.60% |

**About 90% of the time the right category is already in the top five.** The `0.34` figure
describes the hard 17.68% of decisions where the user changes direction — chosen precisely
because the overall number is **82.32% free**: a rule that answers "the same category again"
is right that often having learned nothing.

**`T1` alone is not a recommender.** It narrows to a category. Turning that into specific
items is `T3`, in `S2-DS-07`.

---

## Two defects found, both silent

### The time-gap channel was dead

`S2-SMOKE` converted timestamps with `astype("int64") // 10**9`, correct only for
`datetime64[ns]`. Under pandas 3 the column is `datetime64[us]`, so the divisor is a
thousand times too large and **every inter-event gap floored to zero**.

No exception. No failing shape, dtype or range check. The feature simply never arrived, and
an ablation would have reported that inter-event timing carries no signal.

Fixed with `.dt.as_unit("s")`, which states the unit instead of assuming it. A test now
asserts the channel is **not constant** — the check that would have caught it.

### Two different hashes for the same product

`S2-SMOKE` hashed products with `blake2b(str(id), digest_size=7)`. The frozen `S1-SE-05`
contract is `blake2b-64-v1`: keyed, personalised, over the namespaced `rees46:item:<id>`.
Embedding rows built with one do not correspond to rows built with the other, so a model
trained on the smoke's hash could not be exported against the frozen representation.

This task uses `ppsi.features.hashing.hash_product_id`.

---

## Verification

Before any training, on a six-row fixture — because a bug found after forty minutes of
training cost this project that once.

| check | |
|---|---|
| history right-padded, decision at `lengths-1` | ✅ |
| padding after the real events, per channel | ✅ |
| the window never reaches into a previous session | ✅ |
| a category unseen in TRAIN becomes OOV as an input | ✅ |
| **`gather` equals `pack_padded_sequence`** | ✅ `< 1e-5` |
| the batch passes the **canonical** validator | ✅ |
| the model satisfies `Phase1Model` | ✅ |
| `RawModelOutput` shapes and finiteness | ✅ |
| shared state is all floating, so FedAvg can build | ✅ |
| one `LocalTrainerCore.train_step` completes | ✅ |
| common initialisation reproducible and seed-specific | ✅ |
| building a model does not disturb global RNG | ✅ |
| zero-length history encodes to exactly zero | ✅ |
| the time-gap channel is not constant | ✅ |

**18 of 18 pass.**

### And against upstream, before a single step was trained

| | measured here | upstream |
|---|---:|---:|
| clients | 97,279 | 97,279 |
| TRAIN events | 4,376,137 | 4,376,137 |
| decisions with an unseen current category | 138 | 138 |
| category-change slice | 77,457 | 77,457 |
| same-category share | 82.32% | 0.8232 |
| transition table, slice micro | 0.3085 | 0.3085 |
| transition table, slice macro | 0.3143 | 0.3144 |
| strata client counts | 2,924 / 6,209 / 3,946 / 3,017 | identical |

A separate implementation — different window builder, different rank algorithm, different
hash, different head size — reproducing the published baseline to four decimal places.

`test_rows_read = 0` throughout.

---

## Scope, stated with every number

| | |
|---|---|
| cohort | `C1`, the deterministic 25% slice — `user_id % 4 == 1` |
| clients | 97,279 of 388,789 |
| TRAIN decisions | 3,113,814 · VALIDATION 438,185 |
| averaging | **macro is the headline**, micro beside it — `ADR-001` |
| comparison threshold | **0.0033**, the measured run-to-run spread across seeds 13/42/2026 |
| `TEST` | never opened |

**On the threshold.** `0.0026` is the 95% CI half-width — how precisely this split measures
a fixed model. `0.0033` is what the same configuration actually varied by across three
seeds, and it is the larger of the two, so it governs. A gain below it is not
distinguishable from training again.

**A hypothesis about that number was stated and then falsified.** The ladder and the sweep
trained the same configuration on the same seed and reported `0.3373` against `0.3379`, and
the explanation offered was cuDNN non-determinism — GRU kernels that do not reproduce.
`determinism.py` measured it: the same seed, three times, spread **`0.00000`**, with default
kernels *and* with `torch.use_deterministic_algorithms`. Determinism costs **1.19×** runtime
and buys nothing, because the default is already deterministic.

So the `0.0006` between those two runs has a cause that has **not** been identified. It sits
inside the run-to-run band and changes no conclusion, but it is not explained, and saying
otherwise would be inventing a reason.

**The measurement has a limit worth stating:** the three repeats ran in one process. That
establishes kernel determinism; it does not establish reproducibility across separate
processes, which is what `S2-PR-09` actually needs. That check belongs to whoever makes the
reproducibility claim.

---

## What the other lanes get

| artifact | for |
|---|---|
| `s2_ds_01_gru_t1_seed{13,42,2026}.pt` | `S2-SE-01` — ONNX export |
| `common_initialization_seed*.pt` + sha256 | `S2-PR-06`, `S2-PR-09` — R2a must start where R1 did |
| `ci_parity_fixture.json` | `S2-SE-04` — asked for by name |
| `ppsi.models.batch_spec` | every lane — replaces the fixture spec |
| `ppsi.models.evaluation` | `S2-PR-07`, `S2-PR-08` — so a second evaluator is never written |
| `ppsi.data.sequences` / `.batching` | the real history representation |

Details and the one change the federated adapter needs: `HANDOFF_TO_LANES.md`.

---

## Files

| | |
|---|---|
| `prepare_data.py` | raw events + frozen examples → cached history windows |
| `client_events.py` | TRAIN events per client — the `ADR-001` strata definition |
| `train.py` | the training loop and the scorer |
| `calibrate.py` | rung 0, the gate |
| `ladder.py` | rungs 1–5, one change each |
| `sweep.py` | rung 6, the coordinate sweep |
| `probe_lr.py`, `probe_schedule.py` | following the one axis that moved |
| `determinism.py` | kernel non-determinism vs seed variance |
| `finalize.py` | the final model, three seeds, every artifact |
| `prepare_heads.py` | T2 and T3 windows, for the next two tasks |

The loop is ours rather than `LocalTrainerCore`'s: the core packs and sha256-hashes the
whole state dict on **every step**, and at 6,081 steps an epoch that dominates. The model
still satisfies `Phase1Model` and a test proves a core step works, so the federated lane is
unaffected.

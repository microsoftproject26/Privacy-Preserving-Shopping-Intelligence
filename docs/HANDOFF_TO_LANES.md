# S2-DS-01 handoff — what the other two lanes get, and the one thing each must change

Written for `FEDERATED_TRAINING` and `SYSTEM_DEPLOYMENT_SECURITY`. Everything below is in
the shared `ppsi` package, so it is imported rather than copied.

---

## 1. The evaluator — please do not write a second one

`S2-PR-07` owns the official T1 evaluation harness and `S2-PR-08` the official baselines.
**They are already measured**, on the frozen VALIDATION examples, and the code that
measured them is in `ppsi/models/evaluation.py`:

| baseline | MRR@20 |
|---|---:|
| trivial — repeat the current category | **0.8232** |
| globally most popular category | **0.2841** |
| TRAIN transition table | **0.8560** |
| transition table, category-change slice — micro | **0.3085** |
| transition table, category-change slice — macro | **0.3143** |

Two independent evaluators would disagree by a small amount nobody could explain, and
"one rule written in two places" has already produced three wrong numbers in this project.
If your harness recomputes them, **assert equality against these** before publishing.

Three things in that module are easy to get wrong and each moves the number a lot:

- **Off-diagonal suppression applies to the slice only.** The baseline is evaluated
  off-diagonal, so a model that is not is being compared unfairly. Worth **+0.13 MRR**.
- **Macro is computed after the slice mask**, so the denominator is the 16,096 clients
  with a slice decision, not the 31,576 with any decision.
- **The transition table is built over decision rows only.** Given all events, the `-1`
  that marks "no later different item" competes as a category and the baseline collapses
  from 0.3085 to 0.1866 — which would inflate a model's apparent gain to +0.1454.

`rank_of_truth` counts the rank rather than sorting, with ties broken by lowest class
index. The transition table is mostly ties, so an unspecified tie rule would move the very
number the model is compared against.

---

## 2. The batch spec — `default_batch_spec()` has to be replaced

`ppsi/federated/task_examples.py` currently builds batches against
`ppsi.training.fixtures.default_batch_spec()`, which describes a toy world: 64 items, 16
categories, 8 event types. That was right for a plumbing smoke. Real training needs
`ppsi.models.batch_spec.phase1_batch_spec_v1()`:

| channel | vocab | pad | source |
|---|---:|---:|---|
| `category_id` | 590 | **589** | frozen `vocabulary_v1`: 0–587 real, 588 OOV |
| `product_bucket` | 50,000 | 0 | `S1-SE-05` blake2b over `rees46:item:<id>` |
| `event_type_id` | 4 | 0 | 1 view, 2 cart, 3 purchase |
| `brand_bucket` | 5,000 | 0 | same hash over `rees46:brand:<name>` |
| `price_band` | 6 | **5** | frozen `price_transform_v1`, bands 0–4 |

Pad ids are not all 0 on purpose: the canonical validator forbids a pad id in a valid
position, and both categories and price bands have real values at 0.

### The one change your code needs

Your batch construction is spec-driven and will mostly just work — it already iterates
`spec.history_categorical` and `spec.candidate_categorical`. **The query block is the
exception**, because it names a channel literally:

```python
query_categorical_ids = {"query_context_id": ...}   # fixture-only name
query_continuous = [[f0, f1]]                       # our spec has 0 continuous dims
```

`_row_to_query_features` currently invents `1 + (decision_order % 7)`, which is a
placeholder rather than a feature. The real spec has four query channels, all filled from
the item the decision is actually about:

```python
query_category_id · query_product_bucket · query_brand_bucket · query_price_band
```

`ppsi.data.sequences.build_windows` produces them, and
`ppsi.data.batching.windows_to_batch` assembles a valid batch from them.

---

## 3. Real session history

Your adapter's docstring says it: *"zero-history representation. Sequential GRU history is
deferred to the final model training lane."* That is now paid.

`ppsi/data/sequences.py` builds one right-padded window per decision. Three properties are
load-bearing, and each has already caused a wrong number once:

- real events at columns `0 … lengths-1`, the **decision event itself at `lengths-1`**
- padding strictly after that
- the window never reaches into a previous session

Left-padded instead, the encoder reads pure padding for every window shorter than the
limit — which is most of them, since the median history is **5 events**. That drove MRR@20
to 0.4509, *below* the trivial baseline, while every shape and dtype check passed.

---

## 4. Seeds and the common initialization — `S2-PR-06` and `S2-PR-09`

Both tasks require R1 and R2a to start from *identical* weights. Otherwise part of the
measured centralized-vs-federated gap is just a different random start, and that gap is the
project's headline number.

```python
from ppsi.models.session_gru import common_initialization
state, digest = common_initialization(13)   # also 42, 2026
```

The digest is a sha256 over the state dict, so a run can prove it started where it claimed.
Any other seed is refused.

Client identity comes from **your** module — `ppsi.federated.clients.client_id_from_user` —
so the partitioning is shared by construction, not by agreement.

---

## 5. Deployment — `S2-SE-01`, `S2-SE-02`, `S2-SE-04`, `S2-SE-08`

**The encoder is already export-shaped.** `pack_padded_sequence` obstructs ONNX export,
particularly with `enforce_sorted=False`, so the encoder runs the whole padded sequence and
gathers the state at `clamp(lengths-1, min=0)` — the same shape as
`ppsi/training/stub_model.py`. With right-padded history the two are mathematically
identical, and a test asserts they agree to `1e-5`.

Output names are **frozen** for `S2-SE-08` to verify:

| output | shape |
|---|---|
| `t1_logits` | `[B, 588]` |
| `t2_logit` | `[B, 1]` — 2-D despite the singular name |
| `t3_scores` | `[B, K]` — sized from `batch.candidate_width` |

No activations anywhere in the model: no sigmoid, no softmax, no sorting. The contract
requires raw outputs so loss and metric code owns those choices.

For `S2-SE-02`'s benchmark metadata, the values that must be recorded with any latency
number are in `config.json`: history length, batch size, and the vocabulary sizes above.

There is no `BatchNorm` and no integer buffer anywhere, so
`SharedStateSpec.all_shared_floating` builds — which is what FedAvg needs and what a
`num_batches_tracked` buffer would have quietly broken.

---

## 6. What is frozen, and what is deliberately not

**Frozen — moving any of these forces a re-export or a re-benchmark:** channel names and
pad ids, all five vocabulary sizes, history length, the output names and shapes, the
common-initialization scheme, and the adapter and evaluator APIs.

**Not frozen — and safe, because the interface hides them:** the encoder architecture
(`S2-DS-05` may replace the GRU), the head internals (`S2-DS-06` / `S2-DS-07`), and the
loss weights (`S2-DS-08`).

The board plans two export passes — `S2-SE-01` on the first real model and `S2-SE-08` on
the final multi-task one. A second export is expected. A third, caused by us renaming a
channel, is not.

---

## 7. Two defects worth knowing about, because they are not model-specific

**The timestamp unit.** `S2-SMOKE` converted event times with
`astype("int64") // 10**9`, which is correct only for `datetime64[ns]`. Under pandas 3 the
column is `datetime64[us]`, so the divisor is a thousand times too large and every
inter-event gap floors to zero. No exception, no failing check — the feature is simply
absent. If any of your code converts timestamps this way, it has the same silent hole.
`.dt.as_unit("s")` states the unit instead of assuming it.

**Two hashes for the same thing.** `S2-SMOKE` hashed products with
`blake2b(str(id), digest_size=7)`, which is *not* the frozen `S1-SE-05` contract
(`blake2b-64-v1`, keyed, over the namespaced `rees46:item:<id>`). Embedding rows built with
one do not correspond to the other. This code uses the frozen contract.


---

# What this model is, precisely — read before claiming personalization

Added 2026-09-11 after an independent review raised it. It is a scope statement, not a
defect, but getting it wrong in a write-up would be a serious overclaim.

**The window stops at the session boundary.** `build_windows` clips every history window at
`session_start`:

```python
inside = index >= session_start[decisions][:, None]
```

So at inference the model sees the events of the **current session only**. No state is
carried from a client's earlier sessions, and none from TRAIN into a later VALIDATION
session. Every C1 client has TRAIN history by construction, and the model does not read it.

What the `0.3474` therefore measures:

| | |
|---|---|
| **is** | a session-context model, trained across many users, that reads the session in front of it |
| **is not** | a persistent user-history recommender whose inference state remembers earlier visits |

Both are legitimate systems and session-based recommendation is a standard formulation. The
distinction matters because of what the project claims next:

* **`R3` (personalized) and `R4` (device-local)** rest on the premise that a client's own
  history changes what they are shown. With session-local windows there is no persistent
  per-client state for personalization to act on beyond what the current session already
  supplies. If `R4` fine-tunes the full 2.3M-parameter network per client, note that the
  median VALIDATION client has **7** T1 examples and the p10 has **1** — enough for a small
  adapter or head, not for a network.
* **`S2-PR-07`'s per-client divergence diagnostic** will still show different clients
  getting different answers, because their *sessions* differ. That is evidence the model
  uses context. It is not evidence it uses a client's history, and the two must not be
  reported as the same finding.

**If cross-session state is wanted, it is a design decision that has to be made and frozen
deliberately** — a compact user profile, a carried hidden state, or a per-client adapter —
not something to be assumed from the fact that the data contains prior sessions. It would
change the batch contract, so it cannot be added quietly after `S2-SE-01` exports.


---

# The vocabulary is fitted on the 25% slice — 7 categories short of full TRAIN

Measured 2026-09-11, on TRAIN row groups only (`_SEAL/vocabulary_coverage.json`):

| | |
|---|---:|
| distinct categories in **full** TRAIN | **595** |
| codes in `vocabulary_v1` (cohort C1, `user_id % 4 == 1`) | **588** |
| our codes absent from full TRAIN | **0** |
| **in full TRAIN, unknown to our vocabulary** | **7** |

Our vocabulary is a clean subset — nothing in it is spurious — but it is missing seven
categories that exist in the wider TRAIN data. On the 25% slice this costs nothing, because
those seven never appear there. **Any lane that runs on more than the slice will map them to
`CATEGORY_OOV` silently**, with no error and no warning, and the resulting number will be
slightly and unaccountably different from ours.

`vocabulary_v1` states the consequence itself:

> *"rebuilding this mapping from different data produces different codes for the same
> category; task examples are only meaningful against this file"*

So there are exactly two consistent positions, and the project has to pick one:

1. **Every regime stays on the slice.** R1, R2, R4 and the ONNX export all use these 588
   codes, the comparison is internally valid, and every published number carries the scope
   line. This is the current decision.
2. **Refit on full retained TRAIN** and rebuild every artifact that depends on the codes —
   task examples, caches, checkpoints, the export. Nothing may be mixed across the two.

What must not happen is one lane scaling up while another stays on the slice. The codes
would no longer mean the same thing, and the regime gap — the headline of this whole project
— would absorb a vocabulary difference that has nothing to do with federation.


---

# ⚠ `S2-PR-07`'s harness and ours rank identically — and would report different numbers

Measured 2026-09-11, after `S2-PR-07` landed. **Read this before scoring any model through
either evaluator.**

## The ranks agree exactly

`ppsi.evaluation.t1.rank_targets_from_scores` and `ppsi.models.evaluation.rank_of_truth`
produce identical ranks on distinct scores, on heavy ties, and on the fully-degenerate
all-tied case a no-signal baseline produces. `tests/evaluation/test_evaluators_agree.py`
asserts it so the two cannot drift.

## The slice metric still differs, by a lot

The slice number depends on a step that happens **before** ranking, and the frozen harness
does not perform it: **off-diagonal suppression**, which removes the current category from
the score row.

Measured on `s2_ds_01_gru_t1_seed13.pt`, over the 438,185 VALIDATION decisions:

| | slice macro MRR@20 |
|---|---:|
| with off-diagonal suppression — **what we publish** | **0.3479** |
| without it — what the frozen harness computes from raw scores | **0.2174** |
| the difference | **0.1305** |
| the baseline to beat | 0.3143 |

**Scoring our model through the frozen harness and comparing it to our published `0.3143`
gives `0.2174` — below the baseline.** The reader's conclusion would be that a GRU loses to
a transition table, and it would be wrong.

## Why suppression exists, and why it is a choice rather than a bug

The slice is defined as *decisions where the category changes*. The transition-table baseline
is evaluated **off-diagonal** on that slice: it is handed the fact that the answer is not the
current category. A model scored without the same information is spending rank 1 on an answer
the slice definition has already excluded, and the comparison measures that handicap rather
than the model.

So suppression is not a trick to raise a number. It is what makes the two sides answer the
same question. But it is a **convention**, and the only thing that matters is that the
baseline and the model are scored under the same one.

## What to do

* **Do not mix.** Our `0.3143` / `0.3474` pair is internally consistent and both sides are
  suppressed. The harness's output is internally consistent and neither side is.
* If the frozen harness becomes authoritative, **recompute the baseline through it too**, and
  republish both numbers together. The gap will be smaller and the conclusion should survive
  — but that has not been measured yet and must not be assumed.
* If suppression is adopted into the harness, it belongs in the rank step
  (`rank_targets_from_scores` has no such parameter today), not in each caller. A suppression
  rule implemented per-caller is the same defect this project has already paid for three
  times.

The measurement is in `S2-DS-01_GRU_T1_Model/output/suppression_gap.json`.

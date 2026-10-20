# Data Access

The code in this repository is public. **The data is not.**

Everything with a `user_id` in it lives on the shared drive, not in git. This page tells you
what to download, where to put it, and how to check you got the right file.

> ## 📥 [Download the data](https://drive.google.com/drive/folders/1-_NGI6gIqwR4mlSFXi2APgOqXl4DSBc8?usp=sharing)
>
> `https://drive.google.com/drive/folders/1-_NGI6gIqwR4mlSFXi2APgOqXl4DSBc8`
>
> Ask Eid for access if the folder does not open.

---

## Why the data is not in this repository

| | |
|---|---|
| The repository is **public** | and the project is about privacy-preserving learning |
| The task examples carry **`user_id`** | for 388,789 clients |
| **Git keeps every version forever** | deleting a file later does not remove it from history |

The `.gitignore` blocks these paths so an accidental `git add .` cannot publish them. **Do not
override it.**

---

## What you need, by task

Find your task below and download only what it lists.

### `S2-DS-01` — GRU + T1 · `S2-DS-05` — architecture exploration

| File | MB | Put it in |
|---|---:|---|
| `INTERNAL_DO_NOT_UPLOAD_task_examples_t1_train_v1.proposed.parquet` | 47.90 | `data/examples/` |
| `INTERNAL_DO_NOT_UPLOAD_task_examples_t1_v1.proposed.parquet` | 6.92 | `data/examples/` |
| `processed_raw_parquet_v1.parquet` | 803 | `Dataset/` |

**Why the raw parquet too?** The examples say *what* to predict. The raw events supply the
**history to predict it from**. A sequence model needs both.

The `TEST` file is sealed — see [Sealed until final models](#sealed-until-final-models).

### `S2-DS-06` — T2 purchase-likelihood head

| File | MB | Put it in |
|---|---:|---|
| `INTERNAL_DO_NOT_UPLOAD_task_examples_t2_train_v1.proposed.parquet` | 43.38 | `data/examples/` |
| `INTERNAL_DO_NOT_UPLOAD_task_examples_t2_v1.proposed.parquet` | 6.54 | `data/examples/` |
| `processed_raw_parquet_v1.parquet` | 803 | `Dataset/` |

> ⚠️ **Respect `task_mask`.** 17.95% of `T2` decisions are `CENSORED` — the outcome was never
> observable. Their `label_value` is **null, not zero**. Training on them as negatives is the
> exact failure the field exists to prevent.

### `S2-DS-07` — T3 candidate-ranking head

| File | MB | Put it in |
|---|---:|---|
| `INTERNAL_DO_NOT_UPLOAD_task_examples_t3_train_v1.proposed.parquet` | 77.53 | `data/examples/` |
| `INTERNAL_DO_NOT_UPLOAD_task_examples_t3_v1.proposed.parquet` | 9.17 | `data/examples/` |
| `item_catalog_v1.proposed.parquet` | 2.1 | **already in the repo** — `fixtures/reference/` |
| `t3_candidate_lists_v1.proposed.parquet` | 9.2 | **already in the repo** — `fixtures/reference/` |

**Negatives are derived, not shipped.** Take the candidate list for an anchor and subtract that
query's positives. Measured: **97.4%** of candidates are negatives, and **1,993 of 2,000**
queries have at least 10.

### `S2-DS-08` — joint multi-task loss

Everything from `S2-DS-01`, `S2-DS-06` and `S2-DS-07`.

The three tasks align on **`(client, session, decision_order)`** — verified, 131,978 shared keys
between `T1` and `T2` in a sample. That key is what makes a joint loss possible.

### `S2-PR-*` — federated lane and classical baselines

| File | MB | Put it in |
|---|---:|---|
| `INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet` | 13.42 | `data/protocol/` |
| the task examples for whichever task you are baselining | — | `data/examples/` |

**The cohort manifest is how you partition clients.** `C1` is the trainable cohort — 388,789
users. Do not invent your own partition.

### `S1-SE-05` / `S1-SE-06` — feature modules

| File | MB | Put it in |
|---|---:|---|
| `item_catalog_v1.proposed.parquet` | 2.1 | **already in the repo** |
| `price_transform_v1.proposed.json` | <1 | **already in the repo** |
| `vocabulary_v1.proposed.json` | <1 | **already in the repo** |

**No user-level data needed.** Your modules turn ids into vectors; the catalogue and the
transforms are all the input required.

> The price bands we fitted are in `price_transform_v1.proposed.json` — median TRAIN price,
> `log1p`, TRAIN quartile edges at **29.58 / 77.20 / 194.17**. Reconcile against these rather
> than deriving your own silently, or every `T3` number moves without anyone seeing why.

### Anyone touching boundaries or splits

| File | MB | Put it in |
|---|---:|---|
| `INTERNAL_DO_NOT_UPLOAD_excluded_sessions_v1.proposed.parquet` | 0.26 | `data/protocol/` |

**4,995 sessions cross a split boundary and are excluded everywhere.** Consume this list —
**do not recompute it.** Deciding whether a `VALIDATION` session crosses into `TEST` requires
reading `TEST` rows, which you are not allowed to do.

The file carries `user_id` and `user_session` alongside the hashed key, so you can apply the rule
from any language without reproducing our pandas hashing.

---

## Directory layout after download

```
<repo>/
├── data/
│   ├── examples/      ← the parquet files you downloaded
│   ├── protocol/      ← cohort manifest, excluded sessions
│   └── reference/     ← already in git: catalogue, candidate lists
└── Dataset/
    └── processed_raw_parquet_v1.parquet
```

All of `data/examples/`, `data/protocol/` and `Dataset/` are gitignored.

---

## Verify what you downloaded

Every file is checksummed in `docs/evidence/s1-ds-09/g1_gate_v1.frozen.json`. A truncated
download will not match.

```bash
python - <<'PY'
import hashlib, json, pathlib
gate = json.loads(pathlib.Path("docs/evidence/s1-ds-09/g1_gate_v1.frozen.json").read_text())
expected = {a["file"]: a["sha256"] for a in gate["artifacts"]}
for path in list(pathlib.Path("data").rglob("*.parquet")) + list(pathlib.Path("Dataset").glob("*.parquet")):
    want = expected.get(path.name)
    if want is None:
        print(f"?  {path.name} — not in the manifest")
        continue
    got = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    print(f"{'OK' if got == want else 'BAD'} {path.name}  {got}")
PY
```

### Expected digests

| File | MB | `sha256` (first 16) |
|---|---:|---|
| `..._task_examples_t1_train_v1.proposed.parquet` | 47.90 | `42b1617c1d2f1b5a` |
| `..._task_examples_t1_v1.proposed.parquet` | 6.92 | `e5e8522510371741` |
| `..._task_examples_t2_train_v1.proposed.parquet` | 43.38 | `72379556cc03dcd9` |
| `..._task_examples_t2_v1.proposed.parquet` | 6.54 | `fcb22295c58d70d8` |
| `..._task_examples_t3_train_v1.proposed.parquet` | 77.53 | `e9e8ea9d7a0279f3` |
| `..._task_examples_t3_v1.proposed.parquet` | 9.17 | `018303badbb4fd0c` |
| `..._cohort_manifest_v1.proposed.parquet` | 13.42 | `32d4b8ce4bb84f78` |
| `..._excluded_sessions_v1.proposed.parquet` | 0.26 | `60404af5c89f8d1b` |

---

## Sealed until final models

The `TEST` examples exist and are on the drive, but **nothing reads them until final models
exist**.

| File | MB | `sha256` |
|---|---:|---|
| `..._task_examples_t1_test_v1.frozen.parquet` | 4.79 | `5dccd3e9e617d843` |
| `..._task_examples_t2_test_v1.frozen.parquet` | 4.56 | `d8d2054332531384` |
| `..._task_examples_t3_test_v1.frozen.parquet` | 6.48 | `33519f3d4cffb9e0` |

### Why

`VALIDATION` is a practice exam you can retake. `TEST` is the real one, and you sit it **once**.

If we score `TEST` while still choosing architectures, we end up picking the model that
*happened* to do well on it — and the number stops being an estimate and becomes optimism.

**Develop and tune on `VALIDATION`.** `TEST` is scored once, at the end, on final models.

---

## The contract every example follows

| Column | Meaning |
|---|---|
| `client` | the user — one client, one device |
| `session` | logical session key `(user_id, user_session)` |
| `decision_order` | canonical position, ordered by `(event_time, source_row_number)` |
| `label_value` | the answer — **null when censored** |
| `task_mask` | **false means: do not train on this row** |
| `status` | `OBSERVED` or `CENSORED` |
| `label_matures_at` | when the outcome became observable |
| `cohort` · `split` | `C1` · `TRAIN` / `VALIDATION` / `TEST` |

---

## Three rules that are not negotiable

**1. Read the examples — do not rebuild them.**
Rebuilding is how this project produced **263 decisions** that one notebook had and another did
not. It was caught by an assertion. With a shared file it cannot happen at all.

**2. Respect `task_mask`.**
A masked row has **no label**. Training on it as a zero teaches the model something the data
never said.

**3. Never read `TEST` before final models.**
It is the only clean measurement the project gets.

---

## Scope every result must declare

| | |
|---|---|
| Cohort | `C1`, measured on a deterministic 25% user slice — `user_id % 4 == 1` |
| Evaluation reaches | **33.56%** of `C1` clients have `TEST` data, and they are more active than average |
| `T3` covers | **56.65%** of real next-engagements, not all of them |
| Averaging | **`macro` is the headline**, `micro` printed beside it — see `docs/decisions/ADR-001-evaluation-protocol.md` |
| `TRAIN → TEST` drift | cart share falls **29.5%**; absolute numbers on `TEST` will be lower, and that is drift, not model failure |

---

## If a file is missing or a checksum fails

Ask before improvising. Regenerating an example file yourself produces a **different** file, and
your results stop being comparable to everyone else's — which is the one thing this whole
protocol exists to prevent.

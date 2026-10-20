# DECISIONS.md — Opus synthesis over FABLE_PLAN.md

Fable's verdict was FIX_REQUIRED with 8 corrections. I independently verified the 5 highest-impact claims
against source before accepting them (evidence below). Fable's plan is accepted in substance; the deltas are
recorded here. Anything not listed is accepted as written in FABLE_PLAN.md.

## Verification I performed myself (not taken on faith)

| Fable claim | My check | Result |
|---|---|---|
| `eval_mask ∧ target_class=-1 ⟺ target_oov`, must score 0 not drop | read `pipeline5.py:394-404` | **CONFIRMED** — SQL comment states "OOV targets retained end-to-end and scored zero" |
| `product_idx = output_class + KNOWN_START(3)` by construction | read `pipeline5.py:303-306` | **CONFIRMED** |
| `post_freeze_smoke.py:101` is vacuous | read lines 99-103 | **CONFIRMED** — literal `... or True` |
| `BucketStore` reads all columns, per-row dict, cache never evicted | read `loaders.py:85-115,186-193` | **CONFIRMED** — `bucket()` passes no `columns` projection; `row_of_decision` dict over every row |
| `index()` does not filter `loss_mask` | read `loaders.py:194-215` | **CONFIRMED** |
| `client_id` collapses unseen users to OOV=2 | read `loaders.py:316-321` | **CONFIRMED** (also noted: `d.get("client_idx") or ...` would mis-fall-through on a literal 0, harmless since 0 is PAD) |

Measured facts added by me (Fable listed these as assumptions):

- **`yaml` IS available** in `~/venvs/ppsi-gpu` (6.0.3) → Fable's vendoring risk (a) is **closed**; vendor byte-identical, no lazy-import edit.
- Lobot env: numpy 2.4.6, pyarrow 21.0.0, torch 2.13.0+cu130.
- **Both A16 GPUs idle**, 0 MiB used, no compute processes → no teammate coordination needed right now.
- **Disk: 33 GB free** on `/home/jovyan` → the ~5.9 GB window cache fits with wide margin.

---

## D1 — Workspace location: ALL pilot work under `.local_ai/gru_v1/`, not `ppsi/`

- **Fable:** vendor the loader into `ppsi/data/rees46_nextitem/vendored_rees46_eda/`.
- **Opus decision:** every artifact of this pilot — vendored loader, new model/trainer/evaluator code, tests, caches, experiment runs — lives under `.local_ai/gru_v1/` on Lobot, which is already in `.git/info/exclude` (verified: `git status --short` is empty).
- **Reason:** this is a shared team repo that Eid and Ahmed Sherif also work in. Writing ~20 new files into `ppsi/` leaves a permanently dirty working tree, which (a) invites an accidental `git add -A` commit by any teammate, (b) makes `git status` useless for them, and (c) buys nothing, since no commit is required for this task and pushing is forbidden. Promotion into `ppsi/` is a later, separately-authorised step if the pilot succeeds.
- **Evidence:** repo is shared (`LOBOT_TEAM_SETUP.md`), commits optional per the task brief, `.local_ai/` already excluded.
- **Must benchmark?** no.
- **Consequence:** `sys.path.insert` of the vendored package in the trainer entrypoint; experiment records go to `.local_ai/gru_v1/experiments/gru/<run_id>/` rather than a repo-level `experiments/`.

## D2 — ONE shared window cache serving both arms (not an F0-slim cache)

- **Fable:** notes F0 needs only `item_tokens` (200 B/decision) vs ~5.9 KB for F4.
- **Opus decision:** build a single F4-complete cache; the F0 arm reads the same cache and simply does not consume the extra channels.
- **Reason:** the fairness contract (§14 of the brief) demands identical decision population, identical windows, identical targets across arms. A shared byte-identical cache makes that **structural** rather than procedural — it becomes impossible for the two arms to disagree about which decisions they saw. The disk saving from an F0-slim cache (~5.7 GB) is irrelevant against 33 GB free.
- **Must benchmark?** no.

## D3 — Exact CE thresholds E1/E2 accepted verbatim and pre-registered

- Accepted exactly as written in FABLE_PLAN.md §C, including the "> 13 GiB at microbatch 128 → implementation bug, fix rather than switch objective" clause, which correctly prevents an OOM from silently becoming a science change.
- **Must benchmark?** yes — E1 and E2 are the benchmark's entire purpose. Thresholds are frozen **before** measurement (this file is written pre-benchmark).

## D4 — `eval_user_key` mechanism accepted in full

- Accepted. I verified the structural guarantee myself: `_passthrough` only forwards keys in `OPTIONAL_CHANNELS` (`loaders.py:424-443`), so a key absent from that tuple cannot reach `gru_inputs` output and therefore cannot reach the model.
- Reinforcement I add: the model's `forward` takes **named** tensor arguments only (no `**kwargs`, no dict passthrough), so an accidental extra key raises `TypeError` rather than being silently embedded.
- **Must benchmark?** no. Enforced by test T9 + the T6 exclusion test.

## D5 — Pilot acceptance criteria accepted verbatim, frozen now

- All of FABLE_PLAN.md §F (P1–P5, learning-signal floor, GO-1, GO-2, overfitting check, bootstrap) is adopted **unchanged** and is frozen as of this file's creation, before any pilot data exists. No threshold may be edited after results are seen; if a criterion proves mis-specified, the run is reported as inconclusive against the stated criterion rather than re-scored against a new one.

## D6 — GPU policy

- One A16 only for the pilot, pinned via `CUDA_VISIBLE_DEVICES=0`. Re-check `nvidia-smi` immediately before any heavy job and abort if a teammate process appears. No DDP, no second GPU, per brief §12.
- **Must benchmark?** no.

## D7 — Sequencing: correctness work does not wait on the cache

- Tasks that need no server data (model module, objective, evaluator, their unit tests, the synthetic memorisation test) run first and in parallel with cache construction, since the dataset gate has already PASSED and the cache build is I/O-bound.

## Deltas summary

| # | Fable said | Opus decided | Why |
|---|---|---|---|
| D1 | vendor into `ppsi/` | everything under git-excluded `.local_ai/gru_v1/` | shared repo hygiene; no commit needed |
| D2 | F0 could use a slim cache | one shared F4-complete cache | makes arm fairness structural |
| D3 | E1/E2 thresholds | accepted verbatim | — |
| D4 | `eval_user_key` | accepted + named-args hardening | defence in depth |
| D5 | acceptance criteria | accepted verbatim, frozen pre-run | anti-HARKing |
| D6 | one A16 | accepted, pinned + pre-flight recheck | shared server |
| D7 | task order 1→16 | reorder so data-free work starts immediately | gate already passed |
| — | risk: yaml may be missing | **resolved: yaml 6.0.3 present** | measured |

---

# ADDENDUM — threshold disambiguation, recorded BEFORE the pilot runs

Two terms in FABLE_PLAN.md §F were not defined unambiguously. Both are resolved here, with the
measured baseline numbers, **before any GRU result exists**. Neither may be revisited afterwards.

## A1 — "Client-Macro MRR@20" (the selection metric)
Resolved to: **macro MRR@20 on the END-TO-END denominator** (`eval_mask`), aggregated by `eval_user_key`.
Reason: §E names it the selection metric on VALIDATION_SELECT, and §F's baseline-comparison block
explicitly says "Micro and Macro MRR@20 on the fixed cohort, end-to-end denominator". The end-to-end
denominator is also the honest one — OOV targets are counted with zero credit rather than hidden.
(The Sonnet worker flagged this ambiguity rather than silently choosing; this confirms its resolution.)

## A2 — GO-1's "Recall@20": micro or macro?
Unspecified in §F. Resolved to: **GO-1 must hold for BOTH micro and macro Recall@20.**
Reason: requiring both is strictly harder than either single reading, so the choice cannot be accused of
having picked the convenient one after seeing results.

## Measured baselines (TRAIN-pilot-manifest fit, same evaluator and tie rule, end-to-end denominator)

| Baseline | micro Recall@20 | micro MRR@20 | macro Recall@20 | macro MRR@20 |
|---|---|---|---|---|
| POP   | 0.04890 | 0.01145 | 0.08620 | 0.02189 |
| TRANS | 0.28530 | 0.10639 | 0.34828 | 0.13308 |

(cohort n = 112,365 decisions, 10,000 users)

## Frozen numeric gates for the pilot

- **GO-1 (mandatory sanity floor):**
  micro Recall@20(GRU-F0) ≥ **0.07335** AND macro Recall@20(GRU-F0) ≥ **0.12930**
  AND micro MRR@20(F0) > **0.01145** AND macro MRR@20(F0) > **0.02189**
- **GO-2 (warrants full training):** macro MRR@20(GRU-F0) vs TRANS = **0.13308**
  - ≥ 0.13308 → FULL RUN APPROVED
  - 0.11977 ≤ x < 0.13308 → approved with EXACTLY ONE change (LR 1e-3→5e-4 or 2e-3 chosen from the
    pass-1 loss slope, OR extend to 3 passes — not both)
  - < 0.11977 → NO-GO, engineering review before any further training
- Overfitting check: pass-2 micro MRR@20 ≥ pass-1 − 0.002, else select the best checkpoint.
- F4 vs F0 gain: paired bootstrap over the 10,000 cohort users (1,000 resamples, seed 2026) on
  macro MRR@20 difference. Never gates the GRU decision.

Note for interpretation: TRANS is a genuinely strong baseline here, consistent with Ludewig & Jannach
(arXiv:1803.09587). A GRU that lands just under it at 2M exposures is not obviously "broken" — that is
precisely why GO-2 has a middle band rather than a single pass/fail line.

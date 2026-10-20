# FULL_RUN_PROTOCOL.md — ContextGRU-v1 full-data experiment

**Status: SCIENTIFIC PROTOCOL FROZEN 2026-09-22 ~22:40 UTC.** Frozen before any full-run code existed and before any full-data training or validation number was observed.
Sections 1–12 must not be edited after this point. Section 13 (systems addendum) is filled after the system benchmark and before launch. It may choose only systems parameters that cannot change what is computed scientifically. Any later deviation is appended as a dated, reasoned entry in §14, never edited in place.

Authority: owner authorisation in `START_MESSAGE.txt` / `01_OPUS_MASTER_PROMPT.md`; Fable review `FABLE_FULL_RUN_REVIEW.md` (APPROVE_WITH_FIXES).
The pilot's pre-registered gate result (**NO_GO**, F0 macro MRR@20 0.100795 < 0.11977) stays on record unchanged. This is a separate, owner-authorised experiment and does not re-score the pilot.

## 0. Disposition of the Fable review

| Fable point | Decision |
|---|---|
| Epoch policy 2/6/patience 2/+0.001 with clarifications (a)–(e) | **Accepted verbatim** (§6) |
| LR: single pre-declared plateau step 1e-3 → 1e-4 | **Accepted verbatim** (§5) |
| Fused AdamW acceptable if fixed before launch and identical across arms | **Accepted**. Chosen in §13 only after the correctness suite re-passes with it |
| H512 challenger: SKIP | **Accepted: no challenger arm is run.** F0-d256 is also not run. |
| Data-path gates + edge cases + tolerances | **Accepted** (§10) |
| Full-VS selection; 10k cohort as a diagnostic only | **Accepted** (§6) |
| Two-GPU fairness conditions (pinning, ~3 threads/process, RSS logging) | **Accepted** (§13) |
| Optional shared memory-mapped store | **Declined.** The F0 store is only ~0.6 GB, so sharing saves little. Each process builds its store from the frozen Parquet in RAM, so there is zero disk risk. Gates (i)–(iii) run on the same code path the trainers use. |
| Blockers: pre-registered decision rule, VS-leakage test, correctness re-run, extended parity | **All adopted as launch gates** (§8, §9, §10) |

## 1. Frozen data identity

- Dataset: `/home/jovyan/shared_data/rees46_nextitem_v1`, version hash `860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd`, layout `frozen`. Read-only. Integrity is re-verified (all 409 hash-covered files) before launch and again before TEST.
- Context length L = 50. Catalogue K = 158,486 (TRAIN support ≥ 5, frozen). `product_idx = output_class + 3` (asserted).
- Target and OOV semantics are frozen. OOV targets stay in the denominator and receive zero credit.

## 2. Training population (no new filters)

- **Exactly** the TRAIN decisions with frozen `loss_mask = true`: **40,180,608** decisions. `loss_mask ⇒ target_class ≥ 0` is asserted.
- No user-activity filter of any kind: no minimum events per user, no minimum sessions, no post-hoc exclusions. A decision with one context event and a valid target is a valid example.
- Canonical order: ascending `decision_id`.

## 3. Arms (mandatory, no challenger)

| Arm | Feature group | Model |
|---|---|---|
| **A** `F0_item_only_h256` | `F0_item_only` | ContextGRU-v1, item emb 128, GRU hidden 256, 1 layer, tied head |
| **B** `F4_rich_side_information_h256` | `F4_rich_side_information` | same + all approved side channels (FEATURE_GROUPS.json) |

The architecture is exactly `code/model.py` as frozen for the pilot (params F0 20,791,190 / F4 20,964,062). The only difference between arms is the feature group.

## 4. Optimisation (identical for both arms)

- Exact full-catalogue cross-entropy over one `[B, K]` output per decision (`objective.exact_ce_loss`, sum/n reduction, divide once).
- AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 1e-5 on parameters with ndim ≥ 2, 0 on the rest; global grad-norm clip 1.0 after the 1/n scaling.
- Microbatch = effective batch = **256**, accumulation 1. Dropout 0.20 input / 0.10 readout. Seed **2026**.
- Precision: strict FP32 (TF32 off for matmul and cuDNN), deterministic algorithms (warn-only), cuDNN deterministic, benchmark off.
- AdamW kernel (foreach vs fused) is a §13 systems choice, fixed before launch, identical across arms, never switched mid-run.

## 5. Learning-rate policy (pre-declared)

1. Updates 1..100: linear warmup `lr = 1e-3 · u / 100`.
2. Then constant `lr = 1e-3`.
3. **Plateau step (once):** at the first epoch-end evaluation whose Δ-vs-best < +0.001 (§6), set `lr = 1e-4` for all parameter groups from the first update of the next epoch. Weights continue from the latest state (no restore of best). The LR is never raised again, and no second step is taken. That evaluation still counts toward patience. Each arm records its decay epoch.

## 6. Epochs, evaluation during training, selection and stopping

- Epoch `e` (0-based) = one pass over all 40,180,608 decisions in the order `np.random.default_rng(2026 + e).permutation(40180608)` applied to the canonical order. Consecutive microbatches of 256 are taken from it; the final microbatch of each epoch has 128 decisions. Updates per epoch = **156,956**; exposures per epoch = **40,180,608** exactly. The permutation's sha256 is logged per epoch.
- **Epoch-end evaluation:** full `VALIDATION_SELECT`, all `eval_mask` decisions, **8,826,488** decisions over **790,459** eval users (asserted). Exact full-catalogue ranking, rank `= 1 + #{s_j > s_t} + #{j < t : s_j = s_t}` (ascending class id wins ties; the target never benefits). No candidate filtering. FP32. Fixed eval microbatch 1024 in canonical order.
- **Primary selection metric:** end-to-end client-macro MRR@20 (denominator = `eval_mask`; macro over `eval_user_key` = dense position in the sorted distinct `user_id` list of VS eval rows; never `client_id`).
- Δ-vs-best at an epoch-end eval = metric − (highest metric at any earlier epoch-end eval of this arm); at epoch 1, Δ = +∞.
- **Best checkpoint** = highest raw primary metric over the arm's epoch-end evals (ties → earlier epoch). The +0.001 threshold governs stopping and the LR step only.
- **Patience:** count of consecutive epoch-end evals with Δ-vs-best < +0.001. **Stop when it reaches 2.** Min epochs 2 (redundant but kept). **Max epochs 6.** The budget is in epochs only, never wall-clock.
- **Diagnostic only:** the pilot 10k-user cohort (112,365 decisions, a subset of VS with identical `eval_user_key`s) is evaluated every 20,000 updates and at each epoch end, logged, and never used for selection, stopping, or the LR step.
- TEST and VALIDATION_CONFIRM are never loaded by a training process.

## 7. Model lock

After both arms stop:
1. The **selected arm** is the arm whose best checkpoint has the higher full-VS primary metric. On an exact tie, F0 wins as the simpler model.
2. The lock manifest records: selected arm, both arms' best-checkpoint sha256, epoch, metric, config, code hashes, protocol sha256, dataset hash, and UTC time. It is written before any VC or TEST decision is loaded.
3. After the lock, nothing may be retuned, retrained or reselected.

## 8. Locked evaluation (each exactly once)

1. `VALIDATION_CONFIRM` once: selected GRU, other arm's best, POP-full, TRANS-full.
2. `TEST` once: the same four models.
3. Lock files prevent a second VC or TEST run. Results never feed back into any choice.

Reported metrics per model and split:
- Micro and macro Recall@20, MRR@20, NDCG@20, on both denominators: end-to-end (`eval_mask`) and closed (`eval_mask ∧ target_class ≥ 0`).
- `user_count`, OOV target count and rate, stated explicitly with their zero-credit handling.
- Warm (`client_id ≥ 3`) and unseen (`client_id == 2`) cohorts.
- Diagnostic activity slices, never filters: user prior view count 0 / 1 / 2–4 / 5–9 / 10+ (from the causal `log1p_prev_view_count`), and prefix-length buckets.

## 9. Pre-registered decision rules

- **Primary:** selected GRU vs TRANS-full on **TEST**, end-to-end client-macro MRR@20.
  - Test: paired user-level bootstrap. Resample TEST eval users with replacement, 1,000 resamples, `default_rng(2026)`; each resample recomputes both models' macro MRR@20 on the same user draw; 95% percentile CI of (GRU − TRANS).
  - Claim "GRU beats TRANS" iff CI lower bound > 0. Claim "TRANS beats GRU" iff CI upper bound < 0. Otherwise report "no detectable difference".
  - **Minimum effect of practical interest: +0.001 absolute.** Report whether the point estimate and the CI lower bound exceed it.
- **Secondary:** F4 − F0 (each arm's best checkpoint), same paired bootstrap, on TEST. Also reported on VS at each epoch as a descriptive learning curve. "Side information helps" iff CI lower bound > 0.
- Also reported, descriptively: micro metrics, Recall/NDCG, POP comparisons, cohort and activity slices.
- The pilot NO_GO remains a separate recorded result and is not re-evaluated.

## 10. Launch gates (all must pass on the server before full training)

1. Frozen dataset integrity: 409/409 sha256 match, 0 missing, 0 mismatched.
2. Class map: `product_idx_of_class == arange(K) + 3`.
3. Population counts: TRAIN loss = 40,180,608; VS eval = 8,826,488 decisions / 790,459 users.
4. **VS-leakage test:** the training decision set is disjoint from every VS/VC/TEST decision_id; every training batch is TRAIN with `loss_mask` true; updates × 256 accounting matches 40,180,608 per epoch (156,956 updates, last microbatch 128).
5. Padding/order invariance, OOV zero credit, macro grouping of distinct unseen users, tied head is a single Parameter, no `[B,L,K]` tensor. The existing pilot tests (`test_model.py`, `test_objective.py`, `test_evaluator.py`) are re-run and must pass.
6. Resume determinism under the exact launch configuration (including the §13 AdamW kernel and aggregated logging): interrupted-then-resumed losses match the uninterrupted run to rel 1e-4 (bit-identical expected). Checkpoint rotation never leaves zero valid checkpoints.
7. Synthetic memorisation re-passes (top-1 ≥ 0.95, loss ≤ 0.05 × initial) under the launch optimizer configuration. No non-finite values.
8. **Parity gate (i):** the compact store reproduces the pilot window cache **byte-exactly** (after the documented int32→int64 casts) for all 1,000,000 pilot TRAIN decisions and all 112,365 cohort VS decisions, across all 19 arrays.
9. **Parity gate (ii):** byte-exact vs the authoritative `DecisionWindowLoader.collate()` on ≥ 4,096 fresh random decisions from ≥ 8 buckets, plus targeted edge cases: positions 1 / 49 / 50 / 51+, first and last event of a session, sessions crossing TRAIN→VS, windows containing `loss_mask = false` events, VS decisions whose context starts in TRAIN, and unseen-user decisions.
10. **Parity gate (iii), tolerance frozen:** the pilot best checkpoints (F0, F4), evaluated through the new store and GPU evaluator on the 10k cohort at the pilot's eval microbatch (512), reproduce the recorded pilot cohort metrics with |Δ| ≤ 1e-6 on end-to-end micro and macro MRR@20 and Recall@20, with identical `n` and `user_count`. POP/TRANS fitted through the new baseline code on the pilot manifest reproduce `BASELINES.json` within the same 1e-6.
11. Normaliser/feature invariants on a TRAIN sample, plus a real replacement for the vacuous `post_freeze_smoke.py:101` check: the deterministic cold-user fixture (T11), which must fail when broken.
12. Disk budget re-checked: projected free space ≥ 10 GiB.

## 11. Checkpoints and state

- Per run keep at most `best.pt`, `latest.pt`, `previous.pt`. A replacement is written to `.tmp`, fully validated (loadable, expected keys and update number), and only then rotated. There is never zero valid checkpoints.
- Checkpoints hold model, optimizer, RNG states, epoch, offset, update, LR-step state, patience, eval history, best metric, config, and dataset, protocol and code hashes.
- Run metadata: config, dataset hash, git HEAD + dirty flag, hardware, library versions, seed, start/end UTC, epoch summaries, best metric, checkpoint paths and sha256, resume count.
- Logs never contain raw user IDs.

## 12. Failure policy

- **NaN/Inf:** stop the arm, keep the last valid checkpoint, diagnose before any resume.
- **OOM:** microbatch 128 × accumulation 2 (effective 256 preserved, the sum/n rule makes it exact). Population unchanged.
- **Disk:** no write that would leave < 8 GiB free; stop cleanly.
- **Process death:** the supervisor resumes from `latest.pt` (≤ 5 automatic restarts per arm; each is recorded).
- **One GPU unavailable:** serialise on the other GPU.
- Never kill or touch teammate processes.

## 13. Systems addendum (filled after the system benchmark, before launch)

_Pending. Only these may be set here: AdamW kernel (foreach | fused), concurrent vs serialised arm scheduling, DataLoader/prefetch mechanics, checkpoint cadence, CPU thread counts, the ranking class-chunk width (result-invariant by construction; the eval microbatch itself stays 1024 per §6). Each choice records the measurement that justified it._

### 13.0 Decision rules, pre-registered 2026-09-23 before any system benchmark was run

- **AdamW kernel:** `fused` iff (a) the correctness suite (gates 6–7) passes under `fused` (Sonnet B's suite already covers resume determinism and synthetic memorisation for both kernels), and (b) the single-arm benchmark shows `fused` ≥ 5% faster than `foreach` on F4. Otherwise `foreach`, the pilot kernel.
- **Concurrency:** benchmark F0-alone (GPU0), F4-alone (GPU1), then both together, 3,000 benchmark updates each, same kernel. Run both arms concurrently iff **each** arm's concurrent median dec/s ≥ 0.85 × its alone median dec/s, and host RSS + page-cache headroom leaves ≥ 12 GiB of the 64 GiB pod limit. Otherwise serialise: F4 first on GPU1, then F0. This rule uses system throughput only, never model metrics.
- **Checkpoint cadence:** `latest` every 10,000 updates (≈ 10 min) plus every epoch end. Diagnostic cohort eval every 20,000 updates.

### 13.1 Measurements and decisions (2026-09-23 ~00:55 UTC, before launch)

Source: `bench/BENCH_SUMMARY.json`. 3,000 benchmark updates per phase on the real `CompactStore`, microbatch 256, strict FP32.

| Phase | Wall dec/s | Median step dec/s | Data-wait fraction | Peak GPU reserved | Host RSS |
|---|---|---|---|---|---|
| F4 foreach, alone (GPU1) | 3,989 | 4,058 | 0.0014 | 1.10 GiB | 18.5 GiB |
| F4 fused, alone (GPU1) | **4,482** | 4,566 | 0.0014 | 1.10 GiB | 18.6 GiB |
| F0 fused, alone (GPU0) | 4,660 | 4,725 | 0.0007 | 1.09 GiB | 7.9 GiB |
| F0 fused, concurrent (GPU0) | 4,670 | 4,729 | 0.0007 | 1.09 GiB | 7.8 GiB |
| F4 fused, concurrent (GPU1) | 4,469 | 4,570 | 0.0030 | 1.10 GiB | 18.4 GiB |

- **AdamW kernel = `fused`.** It is 12.4% faster than `foreach` (≥ 5% rule). The correctness suite also passes under `fused`: Sonnet B's resume-determinism and synthetic-memorisation tests, and Opus's production-path resume test on the real store (`code/resume_real_test.sh`), with bit-identical 100-update loss/grad-norm aggregates for updates 500–1,200 after a SIGKILL + resume from update 400.
- **Scheduling = CONCURRENT.** The concurrent/alone ratio is F0 1.002 and F4 0.997 (both ≥ 0.85). Concurrent host RSS is 26.1 GiB, leaving 37.9 GiB of the 64 GiB pod (≥ 12 GiB rule). Aggregate throughput is 9,139 dec/s. Cgroup CPU during concurrency averages 2.6 cores, well inside the 8-core allocation. Assignment: F0 → GPU0, F4 → GPU1.
- **Eval cost:** one full-VS evaluation takes 1,068 s (8,317 rows/s, 2.5 GiB peak GPU; `tests_full/opus/integration.json`). The ranking class-chunk stays at the default 32,768.
- **Expected epoch wall time:** F0 ≈ 2.4 h train + 0.3 h eval; F4 ≈ 2.5 h + 0.3 h. Six epochs is at most ≈ 17 h per arm, running in parallel.
- Threads are capped at 3 per process (OMP/MKL/torch/pyarrow).

## 14. Deviations log

- **2026-09-23 14:10 UTC: additional descriptive metrics (not a change to any rule).** The owner asked mid-training for Recall@2/5/10. The locked VC and TEST evaluations will also report Recall, MRR and NDCG at k ∈ {1, 2, 5, 10} next to the pre-registered @20. They are computed from the same exact ranks, for every model, both denominators, micro and macro. The primary metric, the selection rule, the stopping rule and the decision rules in §6–§9 are unchanged. This is recorded before any VC or TEST decision has been loaded. The mid-training look used only the VS 10k cohort (`reports_full/recall_at_k_cohort.json`, `code/recall_at_k_quick.py`) and feeds no decision.

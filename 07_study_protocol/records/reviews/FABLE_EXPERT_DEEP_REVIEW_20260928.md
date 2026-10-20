# Fable 5.1 expert deep review — PPSI V4 against the proposal question (2026-09-28T21:50Z)

- **Author:** Claude Fable 5.1, chief scientist + Operations Director for this session (owner mandate
  `company/prompts/FABLE_DEEP_EXPERT_SESSION_PROMPT.md`). Stance: hostile examiner toward our own work.
- **Governing question:** *"How much shopping recommendation quality can be retained when moving from cloud-based to on-device or
  federated models?"* Judged on the idea and the question; the agreed substitutions (REES46, next-item over 158,486 items,
  GRU vs SASRec, client-macro MRR@20) are taken as given.
- **Evidence tags:** MEASURED (file:line or a run file), REPORTED (a project document's number), INFERENCE (my estimate),
  OPINION (my judgement), PENDING (not yet done). Every ETA is INFERENCE.
- **What I read:** the proposal PDF (7 pages); RESUME_HERE (all blocks); DECISION_LOG rows 1–158; both 2026-09-28 Fable reviews;
  PREG1_LOCK_VALUES and amendments 01–06C; ARCHITECTURE_LOCK; EVALUATION_CONTRACT §7, §8, §10, §11; REGISTRY_V2 roles;
  MANIFEST_SPEC M07–M09; G2_RESULT; handoffs INT-03/04/05, FL-0304/05/06/07, ANA-01, DATA-07/08, DEV-01/02/03; code
  `fl/{dp,accountant,accountant_wor,participation,runtime,server,aggregate,quant,client,personal,comm,finetune,local_only,
  schedule}.py`, `v4int/{launch_fl,lr_screen}.py` in full, `v4int/{launch_device,c_full}.py`, `v4train/{bootstrap_ci,trainer
  (device check)}.py`, `v4eval/{bootstrap,quality,frozen_config,rows}.py`, `tools/retention_report.py` (header + contrast logic),
  `v4data/manifests.py` (TEST gate), `v4base/{run,strata,knn_emb,rule}.py` docstrings, `v4dev/bench.py` docstring; the thesis
  reference STATE/CHANGELOG/§14; the server state (read-only, 2026-09-28T21:35Z: 2× A40 idle, 384 GiB cgroup, 314 GB free).
- **Companion files written this session:** `contracts/PREG1_AMENDMENT_06D.md` (every decision below, registered before any affected
  read), `company/plans/MASTER_PLAN_V5.md` (the literal plan), task packets `company/tasks/{REV-FL-01,REV-ANA-01,INT-06,DATA-09}.md`.

---
## 0. Verdict (OPINION)

The programme is scientifically sound and, as a procedure, above the federated-recommendation literature it will be examined
against: pre-registration with hashes before every read, sealed selection, a global chronological split, exact full-catalogue
ranking, a hash-chained TEST register, paired user bootstrap on every contrast. **Nothing that has been run needs to be re-run.**
The two earlier Fable reviews were right on the science and on the write-up, and their MUST items (LR screen, contrast registry,
privacy ladder, DP ε=1, C_FULL, the CALIB→UNSEEN_VAL selection fix, Poisson sampling) are registered and coded.

**But the programme cannot currently be executed to its end.** I found two engineering blockers that no earlier review caught,
both on the critical path, both invisible until the first non-smoke launch on the new hardware or the last step of the project:

1. **The frozen evaluation device class refuses the A40.** The trainer refuses to construct on any GPU that is not an A16 unless
   test overrides are on; every downstream launcher builds a trainer. Not one downstream run could have started on the A40.
2. **There is no executable path to the single TEST look.** The locked data builder refuses every manifest id containing "TEST",
   there is no TEST read path, no TEST role in any registry, and no tool evaluates a checkpoint or re-runs a device arm on a TEST
   manifest. The contract's TEST look was a sentence, not a procedure.

Both are fixable without touching any read value (06D §1, §2; tasks INT-06, DATA-09, INT-07), and both are now in the plan.
A third finding is scientific: the DP arms' PRACTICAL_BEST checkpoint is chosen on the users' own validation data, outside the
accountant; K12/K13 are re-registered as confirmatory under CONTROLLED_BUDGET (06D §3). Everything else is disclosure, ordering and
feasibility. With the fixes, the thesis can answer the question honestly at every privacy level; without them it would stall at
the first A40 launch and again at the TEST look.

---
## 1. Findings ranked by severity

Format per finding: evidence → why it matters for the thesis claim → the fix → cost (INFERENCE) → priority → pre-registration status.

### F1 — BLOCKER — the A40 is refused by the frozen device class
- **Evidence (MEASURED):** `code/v4train/trainer.py:302-304` — `if self.eval_device_class != self.frozen.device_class and not
  config.test_overrides: raise TrainerError(...)`; `code/v4eval/frozen_config.py:276-282` maps an A40 to `NVIDIA_A40-FP32-DET`;
  `code/v4eval/rows.py:96` labels non-frozen-device rows `REHEARSAL_NON_FROZEN_DEVICE`. `launch_fl.run` (L780-789) and
  `launch_device.run` (L808-821) construct a `CentralTrainer` with `test_overrides=bool(args.smoke)`; `c_full.py` runs
  `launch_central.main` (same check). Server (MEASURED 21:35Z): the only GPUs are 2× NVIDIA A40.
- **Why it matters:** every FL, device and C_FULL run — the entire answer to the research question — would be refused at launch.
  06A §5 assumed re-evaluation on the A40 "computed inside one comparison block with eval_device_class = A40"; there was no way
  to produce such rows through the frozen path.
- **Fix:** a REGISTERED second frozen class through an O_EXCL pin file written only after the A40 receipts (T3 receipt PASS or a
  recorded diff; STRICT determinism bitwise); trainer/evaluator accept the pinned class; rows labelled `MEASURED_PINNED_DEVICE`
  with the pin hash; rows of the two classes never mix in one block. A re-evaluation tool (`v4int/reeval.py`) for C_BIG/C_SMALL
  checkpoints on the A40 and for every locked checkpoint at the TEST look. Task **INT-06** (Opus, dispatched) + one Opus review.
- **Cost:** ≈ 0.5–1 day of worker time; ≈ 1 h of A40 for receipts and re-evaluations.
- **Priority:** MUST. **Pre-registration:** 06D §1 (registered before any downstream read; the contest rows are untouched).

### F2 — BLOCKER — no executable TEST look
- **Evidence (MEASURED):** `code/v4data/manifests.py:98-105` (`refuse_test_manifest`: any id containing "TEST" is refused
  unconditionally, "this builder has no TEST code path"); L108-115 `read_non_test` filters `split IN (TRAIN, VS, VC)`;
  `v4data06/d6_build.py:18-19` "TEST: refused, unconditionally"; REGISTRY_V2 `roles` has no TEST entry; `v4base/run.py:67`
  `PHASES = ("CALIBRATION", "MAIN")`; `launch_device.SUPPORT_ROLES` has no TEST role; ANA-01 handoff: "a TEST run will refuse until
  the registry ... contains it (decide at ALL_METHODS_LOCK)". MANIFEST_SPEC M07/M08 define the TEST manifests but nothing builds them.
- **Why it matters:** the thesis's only unbiased numbers come from the single TEST look (EVALUATION_CONTRACT E10.1–E10.3; 06A §9g).
  Discovering this at ALL_METHODS_LOCK would have cost days at the worst moment, or tempted a hand-built TEST path without review.
  Also: the device arms keep no per-client weights (`launch_device` docstring: "the dense state is discarded"), so the TEST look must
  RE-RUN their locked procedure on TEST queries; this must be registered as the meaning of "evaluated once".
- **Fix:** DATA-09 (gated builder for `MAIN_RETRO_TEST_5_100_131072`, `UNEW_HOLDOUT_RETRO_TEST_32768`, `TRUE_COLD_RETRO_TEST`,
  the SPARSE/HEAVY TEST panels; `REGISTRY_V2_TEST_EXT.json`; a loader gate that refuses without ALL_METHODS_LOCK + the register
  entry; role shim), INT-06's `reeval.py` for checkpoints, INT-07 wiring of `--role MAIN_TEST` into launch_device, v4base.run,
  dump_rows. Task **DATA-09** (Opus, dispatched) + review; INT-07 (Sonnet) after both land.
- **Cost:** ≈ 1–1.5 days of worker time; the look itself ≈ 1 day of A40 (device arms re-run ×2 seeds) + CPU.
- **Priority:** MUST. **Pre-registration:** 06D §2 (manifests, tools, order, "re-run the locked procedure" wording).
- **F2b (added 2026-09-29T09:01Z from the DATA-09 handoff; MEASURED):** the manifests alone do not suffice — the feature store and
  the history-window policy have no TEST path either: `v4hist/policy.py:41-54` (`history_splits` refuses any TEST split),
  `v4int/cache.py` (`build_subset_store` with `VALIDATION_SPLITS`), `v4int/providers.py` (`comparison_key_base` split hard-coded
  "VALIDATION"), and the CentralTrainer binds the registry at import. A gated TEST store + history path (TRAIN+VS+VC+TEST windows,
  frozen normalisers) and consumer wiring are required: task INT-07 rewritten as an Opus task (after INT-06).

### F3 — MAJOR — the DP arms' PRACTICAL_BEST checkpoint is a data-dependent selection outside the accountant
- **Evidence (MEASURED):** `fl/runtime.py:451-458` `_evaluate` chooses best.pt by the MAIN_VALIDATION metric; `launch_fl.execute`
  L477-483 writes the PRACTICAL_BEST rows from `best_half_efe`; the accountant covers the 384 released rounds only (accountant.py
  docstring L1-35). MAIN_VALIDATION rows belong to the same 131,072 users (REGISTRY_V2 MAIN role). FL-07 raised it (handoff, Open).
- **Why it matters:** an examiner in privacy will say the released "ε = 8 model" was picked using private data. Under user-level
  adjacency the users' validation rows are part of their records.
- **Fix:** K12 = R(DP8/FA_1024) and K13 = R(DP1/FA_1024) confirmatory under CONTROLLED_BUDGET/CONTROLLED_BUDGET (the round-384
  endpoint, data-independent); PRACTICAL_BEST versions descriptive with the disclosure; ANA-01 gains a per-contrast rule map
  (ANA-02) before its freeze. The full list of unaccounted DP hyper-parameters (S, LR multiplier, pooled bootstrap block) is
  registered as a disclosure.
- **Cost:** hours (a small ANA-01 change + tests). **Priority:** MUST. **Pre-registration:** 06D §3.

### F4 — MAJOR — feasibility on ONE A40 is unmeasured and the earlier plans assumed two cards
- **Evidence:** 06B §6.3 "C_FULL ... on one A40 in parallel with the FL queue on the other"; every cost figure in the project is an
  A16 number or an INFERENCE; no A40 job has run; `launch_central` refuses `--allow-shared-gpu` for a non-smoke MAIN run
  (INT-03 handoff), so C_FULL needs the card exclusively for its whole duration (INFERENCE 6–11 h on an A40, more if the automatic
  continuation to 8.0 EFE fires: DL 151). The owner's binding split gives this project one A40, ~8 cores, ~180 GiB.
- **Why it matters:** the SHOULD tier (FP, PF, FA_1024, DP×3, FA_Q8, FT_FA, C_FULL, panels, truncation studies) is ≈ 3× the MUST
  tier in GPU-hours; without measured rates and a binding cut order the project risks finishing with an incomplete core.
- **Fix:** MASTER_PLAN_V5 puts a measurement gate first (receipts + smokes measure visits/s, GPU utilisation and memory per arm
  type), computes the schedule from measured rates with a stated formula, orders everything by 06D §8 tiers, allows up to 3 FL /
  device jobs to share the card (STRICT numerics are co-scheduling independent: Fable staggered ruling) and reserves one exclusive
  window for C_FULL. **Priority:** MUST (planning). **Pre-registration:** 06D §8 (tier order; "nothing cut is replaced").

### F5 — MAJOR (write-up) — the DP design will very likely show a large loss; interpret it before reading it
- **Evidence (INFERENCE from registered values):** d = 20,710,048 shared parameters (ARCHITECTURE_LOCK `parameter_counts`),
  m = 1,024, z(ε=8) = 0.5642, server LR 1, fixed denominator. Per-round noise on the averaged update: expected L2 norm
  z·S·√d/m ≈ 0.5642 × 4,551 / 1,024 × S ≈ 2.51·S; the signal (mean of clipped deltas) has norm ≤ S and, with 1–14 local AdamW
  steps per visit from a fresh optimiser, far below S in practice. Over 384 rounds the noise random-walks (≈ 49·S) while the
  signal is coherent at best. At ε = 1 (z = 1.3094) the ratio is 2.3× worse.
- **Why it matters:** "DP destroyed the model" is a legitimate finding only if the design and its interpretation were written down
  before the value was read; otherwise it reads as a straw man. The cohort/rounds trade-off is also an examiner question: larger
  cohorts reduce noise per round (Charles et al. 2021, NeurIPS, *On Large-Cohort Training for Federated Learning*,
  https://proceedings.neurips.cc/paper/2021/hash/ab9ebd57177b5106ad7879f0896685d4-Abstract.html) but with server LR 1 fewer rounds
  move the model less per client-visit; fixed S is the McMahan-2018 choice, adaptive quantile clipping (Andrew, Thakkar, McMahan &
  Ramaswamy 2021, NeurIPS, https://proceedings.neurips.cc/paper/2021/hash/91cff01af640a24e7f9f7a5ab407889f-Abstract.html) is the
  private alternative not adopted.
- **Fix:** the analytical noise-to-signal note and the (m, T, d) interpretation are registered NOW (06D §3); an m = 4,096 / T = 96
  sensitivity arm is NICE (06D §8). No registered value changes. **Priority:** SHOULD (write-up). **Pre-registration:** 06D §3.

### F6 — MAJOR (statistics) — sixteen confirmatory contrasts with no hierarchy
- **Evidence:** 06B §2 lists K1–K16 as "the only contrasts stated as findings with CIs"; no primary/secondary designation; 1,000
  bootstrap resamples (percentile CI) cannot resolve a Bonferroni-adjusted 99.7% interval.
- **Fix:** K2 primary; K1, K3, K8, K12 secondary; the rest tertiary; unadjusted 95% CIs reported with the hierarchy and the family
  size stated; the §7.6 claim vocabulary applies to K1–K16 only. **Priority:** MUST (doc). **Pre-registration:** 06D §4.

### F7 — MEDIUM — FL under-training is possible at a fixed 6.0-EFE budget
- **Evidence:** FA reaches 6.0 EFE only at round 6,827 with a fresh AdamW per visit and 1–14 local steps (client sizes 5–100,
  B16 × 2); the LR screen (06B §1) protects against a wrong local LR but not against a too-short horizon.
- **Fix:** a registered descriptive flag (best mark ≥ 5.5 ⇒ "still improving") and an optional FA_X2 budget-sensitivity arm
  (12.0 EFE, seed 2026, label BUDGET_SENSITIVITY) so it cannot be a post-hoc addition. **Priority:** SHOULD/NICE.
  **Pre-registration:** 06D §5.

### F8 — MEDIUM — the thesis numbers are produced by an unfrozen script fed by a hand-assembled index
- **Evidence:** `tools/retention_report.py` sha 61c1eafd… not frozen; its input `BLOCK_INDEX.json` (22+ entries with paths, rules,
  seeds, meta) "Ops must provide" (ANA-01 handoff). Transcription of a wrong rule/seed/path is exactly the failure a frozen script
  cannot catch.
- **Fix:** ANA-02: `tools/ana_block_index.py` builds the index deterministically from run directories (reads LAUNCH/RUN_SUMMARY/
  REEVAL_RECORD, never metric values); the per-contrast rule map (F3); the hierarchy labels (F6); the still-improving flag (F7);
  review fixes; then freeze the hash in `company/gates/ANA_SCRIPT_FREEZE.json`. **Priority:** MUST before the first read.

### F9 — MEDIUM — FL-07's out-of-packet edits
- **Evidence (MEASURED):** `fl/dp.py:22-25, 168-181` (Poisson guards: `n_sampled` no longer bounded by m, `allow_empty`),
  `fl/quant.py:11-14, 96-98` (+8-byte header), 4 tests re-pinned (FL-07 handoff). Both changes follow 06C (|C_r| ~ Binomial may exceed
  m; an empty round releases noise only) and FA's accounting (`fl/comm.py:27`).
- **Decision:** APPROVED and registered (06D §6). The 06B §5 byte formula is clarified, not changed in substance (8 bytes of 20.7 MB).

### F10 — MEDIUM — C_FULL: number, exclusivity, continuation
- **Evidence:** DATA-08 corrected the population to 1,348,682 users / 26,677,600 decisions (`v4fs_data/fs_common.py:48`); 06 §5 said
  1,315,914; the trainer's automatic C3 continuation applies (DL 151); C_FULL must run alone on the card (06 §4).
- **Decision:** 06D §7 corrects the number; the plan schedules C_FULL in one exclusive window before ALL_METHODS_LOCK; K15 stays
  seed-2026 only. Its recipe (02B_SASREC unchanged: LR 4e-4, S0, D256_B3, 6.0 EFE) is a registered choice, disclosed as "the MAIN
  recipe applied to 10× the data without re-tuning" (it can only understate the cloud advantage).

### F11 — LOW (disclosed) — the contest is procedure-fair, finding-fragile
- **Evidence (MEASURED):** G2_RESULT: PRACTICAL_BEST SASRec 0.16121 vs GRU 0.15694, Δ −0.00427 CI [−0.00518, −0.00326];
  `runs_p3/g2/cond5_diagnostics.py` + DL 138: at the 6.0 endpoint GRU 0.15829 vs SASRec 0.15391, Δ +0.00421 CI [+0.00328, +0.00522];
  SASRec best at 3.0 EFE; not parameter-matched (83.9M vs 21.0M); both winning LRs at grid edges (DL 92, DL 108).
- **Assessment:** disclosed (06A §9a, 06B §7c); the retention question is within-family; SMALL (20.7M) is size-matched to the GRU
  contender. Nothing to change. The thesis presents both rows and says "selected under rule T1".

### F12 — LOW (disclosed) — external validity and TEST provenance
One retailer, 46 days, users with 5–100 TRAIN examples; RETRO_TEST week contains Black Friday (29 Nov 2019); the TEST period was
inspected by the v1 campaigns (register entry 3). All disclosed (06B §7b, C.10 of the thesis review). No change.

### F13 — LOW — write-up debt
Structured thesis registers PENDING since 2026-09-27 (STATE.md); proposal bibliography errors ([5] title, [6] venue/year, [8]
author/venue, [12] unverified) to fix; the "what changed from the proposal" section not yet drafted. Plan §8 schedules a batch
after the TEST look; never type a number a script can print.

### F14 — LOW — local harness nit
`code/tests_trunc/run_all.py` fails at setup when the `--basetemp` directory does not pre-exist (4 setup errors on my first run,
0 on a re-run with the directory created: MEASURED 2026-09-28T21:45Z / 21:53Z). One-line `mkdir(parents=True, exist_ok=True)`
(Sonnet, with ANA-02 or INT-07).

### What I checked and found sound (MEASURED unless noted)
- **DP mechanism** (`fl/dp.py`): flat L2 clip, unweighted sum, N(0, z²S²) on the sum, fixed denominator m, CPU-seeded noise per
  (seed, round), no side output when z > 0, buffers server-copied; matches McMahan et al. 2018 and the accountant's mechanism.
- **Poisson sampling** (`fl/participation.py:84-91`): `u_i ~ Uniform{0..N-1}`, member iff `u_i < m` — inclusion exactly m/N,
  independent per client and per round, stateless; the accountant's q = m/N is exact (06C). The FL-06 WOR receipt (ε ≈ 1,388 under
  replace-one for the old sampler) is the reason for 06C and is the right number to report in the DP appendix.
- **Accountant** (`fl/accountant.py`): MTZ-2019 integer and fractional orders, standard conversion, z rounded up; independently
  confirmed by mpmath earlier (DL 119); REV-FL-01 recomputes z(8), z(1), z(4).
- **Index-only schedule for Poisson runs** (`fl/runtime.py:230-235, 382-384, 361-363`): LR progress and the 12 marks depend on
  (r+1)/T only — no data-dependent hyper-parameter inside the DP run.
- **FA_Q8** (`fl/quant.py`): unbiased stochastic rounding, per-visit seeded, dequantised server-side, aggregated bitwise as FA,
  `check_round` after every round, refused on the process pool.
- **LR screen** (`v4int/lr_screen.py`): three verified runs, spec equality apart from the multiplier, the 06B §1 rule and tie
  order, O_EXCL output; `require_lr_selection` enforced for every non-smoke MAIN arm including S_CAL (`launch_fl.py:616-621`).
- **INT-05**: `SELECTION_ROLE = "UNSEEN_VAL"`, `--select` refused elsewhere (`launch_device.py:118-126`), SELECTION.json carries the
  role and the eval-manifest content hash, applied-selection role checked.
- **Retention statistics** (`v4eval/bootstrap.py`): one W per user set, ratios bootstrapped per resample, seed-mean deltas on the
  same W, zero-denominator resamples reported not dropped (E7.5).
- **C_FULL shim** (`v4int/c_full.py`): three in-process wrappers; recipe, schema, evaluator and eval manifest untouched; RunConfig
  differs only through derived fields.
- **T41 golden replay** (DATA-07): 173,921 queries bitwise except 7 brand-OOV cells in 48 self-flagged rows; negative controls fail.
  The "price cumsum" flag was resolved as a float-rounding artefact with no cross-user information (DL 157) — I agree with that
  reading (C0 is a prefix constant independent of the user's own events).

---
## 2. The two earlier reviews, tested claim by claim (OPINION)

| Claim | Verdict | Note |
|---|---|---|
| "Nothing done needs a re-run or a scale change" | Agree | Contest, C_SMALL, baselines, locks stand. |
| 06A §5 "re-evaluate C_BIG/C_SMALL on the A40 → one block with eval_device_class = A40" | Wrong as stated | Impossible under the frozen check (F1); now 06D §1 + INT-06. |
| 06B §6.3 "C_FULL on one A40 in parallel with the FL queue on the other" | Obsolete | One card (owner); C_FULL needs an exclusive window (F4, F10). |
| Thesis review C.7 "Poisson bound for fixed-size WOR … conservative-by-convention" | Wrong, caught by the team | FL-06 measured ε ≈ 1,388 under replace-one; 06C switched the sampler. Good. |
| 06B §1 full-6.0-EFE LR screen (Ops raised it from the review's 1.0 EFE) | Agree | Removes short-horizon bias; ≈ ¾ of one FA run in total (INFERENCE). |
| D-6 FA_Q8 as the bits-vs-quality point | Agree | Registered; bytes clarified (F9). |
| "Observational cold start acceptable if worded correctly" | Agree; DATA-07 exists now | Study A is cheap (evaluation-only) → SHOULD-medium. |
| "K1–K16 confirmatory, everything else descriptive" | Agree, incomplete | Needs the hierarchy (F6) and the DP rule (F3). |
| "TEST once at ALL_METHODS_LOCK" | Agree, not executable | F2. |

---
## 3. Does every element serve the question? (proposal → programme)

| Proposal element | Programme | Status |
|---|---|---|
| Central upper bound | C_BIG (same cohort) + C_FULL (10× data, seed 2026) | Covered; C_FULL SHOULD-high |
| FedAvg / FedProx / FedBN / personalised FL | FA / FP / (FedBN inapplicable, LayerNorm) / PF + FT_FA | Covered; disclosed |
| Communication cost, convergence | bytes per visit/client/fleet; 12 marks; FA_Q8 (8-bit) | Covered |
| Privacy levels | ladder L0–L6; DP ε ∈ {∞, 8, 1} (+4 NICE); WOR sensitivity | Covered |
| Local model; pre-trained + local fine-tune; kNN on cached embeddings; rules | LO; FT_C; KNN_EMB; RULE | Covered |
| Cold start: events to parity, few-shot, category diversity | support bins + events-to-90/95 (observational); DATA-07 Study A/B (controlled, SHOULD); strata | Covered |
| Device guidelines (memory, latency, battery) | ONNX FP32/INT8 size, latency, RSS, INT8 retention; browser page (owner's PC); battery not measurable | Covered/disclosed |
| Open benchmark | code trees, contracts, registers, reproducibility package | Pending release (owner) |
| Tasks: next category / purchase AUC / offers; datasets Criteo, MovieLens, … | replaced by agreement | "What changed" section (06D §9g) |

Missing pieces an examiner expects and the programme now has: a real cloud reference (C_FULL), tuned FL local LR (screen),
a privacy curve (ε ladder), a communication curve (FA_Q8 + marks), controlled cold start (DATA-07), a device measurement in a
browser (DEV-03), statistical power (half-widths ≈ ±0.6 retention points at 40,764 users: INFERENCE from the G2 CI), multiplicity
(06D §4), threats to validity (plan §8).

---
## 4. Statistical design
- Unit of inference = user (paired bootstrap, W shared, 1,000 resamples, seed 20260926); 2 seeds report a range, not an SD
  (measured MAIN seed gaps 0.00041 / 0.00002: ARCHITECTURE_LOCK). State it (EVALUATION_CONTRACT §7.7).
- Ratios bootstrapped directly; numerator/denominator same rule and seed; 2-seed mean = mean of per-seed ratios (06A §7).
- Confirmatory set K1–K16 with the 06D §4 hierarchy; DP contrasts under CONTROLLED_BUDGET (06D §3); everything else descriptive
  (strata 3+3+9+4 cells, 5 support bins, 4 panels, 12 marks — hundreds of CIs, all labelled).
- Power (INFERENCE): the G2 contrast half-width was 0.00096 MRR on 40,764 users; a retention ratio's half-width ≈ 0.6 points of
  R at C_BIG ≈ 0.16; every registered arm difference the literature would call interesting (≥ 2 points) is detectable.

---
## 5. Feasibility on one A40 — see MASTER_PLAN_V5 §3 (measure first; INFERENCE tables; cut order 06D §8)

---
## 6. Thesis — structure, ladder, figures, "what changed", reproducibility, ethics
The chapter plan of `FABLE_THESIS_RESEARCH_REVIEW_20260928.md` §E.2 stands; MASTER_PLAN_V5 §8 maps every table/figure to its
artefact and adds: the "what changed from the proposal and why" half-page, the reproducibility package (code_ds3/code_ds4 hashes,
contracts, registers, `verify_reference.py`, exact commands), the ethics/data statement (public Kaggle REES46, pseudonymous ids,
salted hashing, no raw ids written, retention of artefacts), the DP appendix (mechanism, accountant, WOR sensitivity, the
unaccounted-hyper-parameter list, the noise-to-signal note), and the device-class sensitivity table (A16 vs A40 values of the same
checkpoints).

---
## 7. Decisions registered this session
All in `contracts/PREG1_AMENDMENT_06D.md`: §1 A40 pinned class; §2 TEST manifests/tools/order; §3 K12/K13 under
CONTROLLED_BUDGET + DP disclosure list + noise note; §4 hierarchy/multiplicity; §5 still-improving flag + FA_X2; §6 FA_Q8 bytes and
FL-07 approval; §7 C_FULL number; §8 tiers and cut order; §9 disclosures.

---
## 8. Outcome of the independent reviews (updated 2026-09-29T08:50Z)
- **REV-FL-01 (Opus, read-only; `company/reviews/REV-FL-01_OPUS_REVIEW.md`): ACCEPT_WITH_FIXES.** Poisson sampling, the DP mechanism,
  the accountant (z(8) = 0.5642, z(1) = 1.3094, z(4) = 0.7244 re-derived with mpmath; max RDP difference 2.6e-11), the WOR bound,
  the round-count LR schedule and marks, FA_Q8, the LR screen and INT-05 are CORRECT. **Blockers (PF only):** B1 a resumed PF run
  evaluates and saves an EMPTY personal store (launcher-level; reproduced); B2 the personal store of the PRACTICAL_BEST mark is never
  persisted, so PF's best model cannot be re-evaluated at TEST. **MAJOR:** M1 device rows on the pinned A40 are labelled
  REHEARSAL_NON_FROZEN_DEVICE (the pin sha is missing from `launch_device.assemble`; numerics correct); M2/M3 device-class continuity
  guards for FL resume legs and device shards; D13 LO had no registered MAIN cell. Deviations accepted and registered in 06E.
- **REV-ANA-01 (Opus, read-only; `company/reviews/REV-ANA-01_OPUS_REVIEW.md`): ACCEPT_WITH_FIXES.** The ratio/bootstrap/pairing
  core, the rank-to-row checks, the contrast registry, bins, bytes and determinism are CORRECT. **9 MAJOR fixes in
  retention_report.py** (trunc-row field names; K16 needs an ONNX-CPU sub-block; RULE_FREE crash on device arms; registered seed
  sets; comparison key too narrow + pin status; rank coverage; arm identity; TEST gate; the 06D rule map/hierarchy/flag),
  **3 MAJOR in DATA-07** (a smoke T41 receipt passes as the gate; the registered panel is never re-derived; WARM rows labelled
  COLD_START), **2 MAJOR in DEV** (the INT8 bench crashes on OOV targets; `--first-users` uses digest order, not the v2 key).
- **Actions:** amendment 06E registered (08:42Z) for every review-driven rule (LO cell, single-seed K13/K15/K16, K16 ONNX sub-block
  + v2-key subset, ANA key/evidence rules, sanity trigger, PF best store, device continuity, disclosures). Packets ANA-02 (rewritten
  with the FIX list), DATA-07R1, DEV-04 dispatched; FL-08 written, sequenced after the Ops session's INT-06 completion (same files).
  PF lanes wait for FL-08; LO/FT_C lanes wait for M1 (or a re-finalize); SEL_LO before any MAIN LO read.
- **Phase-1 artefacts accepted (MEASURED 08:33Z):** T3 A40 receipt COMPLETE with all checks (rank diffs vs A16 12,635 / 14,063 rows →
  the pinned class is necessary), STRICT r2 PASS, pin written 00:37:27Z, 8 A40 re-evaluations present (values unread), LR screen
  x0.5/x1 done and x2 running. Local suites before the reviews (MEASURED 2026-09-28): tests_int 142 / 37, tests_fl 223 / 68,
  tests_ana 39 / 7, tests_dev 63 / 5 skipped / 11, tests_data08 21 / 8 skipped / 12, tests_base 0, tests_trunc 31 / 9 — all green.

---
## 9. Addendum 2026-09-29T09:26Z — the FL local-LR screen result and its consequences (MEASURED / registered)
- The registered 06B §1 screen (CALIB, FA, 6.0 EFE) gave PRACTICAL_BEST client-macro MRR@20 of 0.020263 (m 0.5), 0.023473 (m 1),
  0.025562 (m 2): the best at the top edge with a monotone rising curve, and an FA level ≈ 20–25 % of the matched central CALIB value
  (INFERENCE from the ladder era). This is the untuned-optimiser artefact flagged in §1 F7 and by the thesis review (D-2), now
  observed. Registered before any MAIN FL value: **06F** (grid extension {4, 8, 16[, 32]}, rule over the screened set,
  LR_SELECTION_EXT.json; the m = 2 MAIN runs of 09:12Z superseded), **06G** (Ops, owner decision: FedAdam server optimiser,
  Reddi 2021 Alg. 2, β1 0.9 / β2 0.99 / τ 1e-3, η_s ∈ {0.01, 0.1} × m ∈ {1, m*}; joint selection → FL_TUNING_SELECTION.json),
  **06H** (this reviewer: 06G approved; DP-arm server-optimiser sensitivity DP_OPT_SENS if FedAdam wins, with the
  noise-normalisation mechanism written down before any DP value is read).
- Consequence for the thesis: the FL chapter gains a tuning figure (FA quality vs local LR × server optimiser on CALIB) and a
  paragraph explaining that the federated arms were tuned as the field guide (Wang et al. 2021) and Reddi et al. 2021 prescribe;
  if the tuned optimum still retains far less than the central model, that is a finding about dense full-catalogue FedAvg at this
  client scale, supported by the curve, not an LR artefact.
- Process incident recorded (DL 185–188): a cross-session overwrite of 06F, resolved by keeping the Fable 06F text and registering
  FedAdam as 06G; coordination is now DECISION_LOG-only.

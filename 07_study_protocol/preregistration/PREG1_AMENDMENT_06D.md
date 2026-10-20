# PREG1 amendment 06D — Fable deep-review decisions (registration only)

- **Status:** REGISTERED 2026-09-28T21:50Z by Claude Fable 5.1 acting as chief scientist + Operations Director (owner
  mandate: `company/prompts/FABLE_DEEP_EXPERT_SESSION_PROMPT.md`). Review: `company/reviews/FABLE_EXPERT_DEEP_REVIEW_20260928.md`.
  Plan: `company/plans/MASTER_PLAN_V5.md`.
- **Registered BEFORE any affected read.** Unread at registration (MEASURED: no RESULT_ROWS of these runs has been opened by any
  decision-maker): C_SMALL s2026/s2027, BASELINE_RULE, KNN_EMB_*, the SMALL ladder values (used only mechanically by the
  registered selection), and every FL / device / ONNX / C_FULL / truncation arm (none exists yet). Read and final: G2
  (ARCHITECTURE_LOCK fd201013…), the MAIN baselines POP/LAST/SKNN (DL 132), the GRU P2 and SASRec P2 tables (G1b).
- **Unchanged:** every registered value of amendments 03–06C except where a numbered item below says "replaces" or "adds".
  TEST stays sealed until ALL_METHODS_LOCK and gets exactly one look.

## 1. Evaluation device class: a REGISTERED second frozen class for the A40 (adds to PREG1 T3; replaces nothing)
- **Finding (MEASURED):** `code/v4train/trainer.py:302-304` refuses to construct a CentralTrainer whose runtime device class
  differs from the frozen `A16-FP32-DET` unless `test_overrides`; `v4eval/frozen_config.py:276-282` maps an A40 to
  `NVIDIA_A40-FP32-DET`; `v4eval/rows.py:96` labels rows on any non-frozen device `REHEARSAL_NON_FROZEN_DEVICE`. Every
  downstream launcher (launch_central for C_FULL, launch_fl and launch_device through their evaluation harness) would therefore
  REFUSE a non-smoke run on the A40. Amendment 06A §5 assumed re-evaluation on the A40 was possible; it was not.
- **Decision:** a second frozen class `NVIDIA_A40-FP32-DET` is REGISTERED by an O_EXCL pin file
  `contracts/eval_device/EVAL_DEVICE_PIN_A40.json` (self-hash; fields: the exact `torch.cuda.get_device_name` string, driver and
  torch/CUDA versions, the T3 GPU receipt sha256 and its PASS verdict, the STRICT determinism receipt sha256, the C_BIG / C_SMALL
  re-evaluation rows sha256, the registration UTC). The trainer and the evaluator accept a runtime device whose class equals the
  frozen class OR the pinned class; rows produced under the pinned class carry `eval_device_class = NVIDIA_A40-FP32-DET`,
  `status = MEASURED_PINNED_DEVICE` and `device_pin_sha256`. Rows of the two classes never mix inside one retention block
  (ANA-01 refuses); the contest rows (A16) stay untouched. Code task **INT-06** (Opus, one independent review); the pin file is
  written by Ops ONLY after the A40 receipts (T3 receipt PASS on the A40, STRICT determinism two identical smoke trainings) exist.
- **Consequence for the comparison key:** all downstream retention is computed inside ONE block with
  `eval_device_class = NVIDIA_A40-FP32-DET`, from C_BIG (best.pt @3.0, endpoint_6.0.pt) and C_SMALL (best, endpoint) re-evaluated
  on the A40 through the frozen T3 path (evaluation only, no training; tool `v4int/reeval.py`, INT-06). The A16 values of the same
  checkpoints are reported beside them as a device-numerics sensitivity (descriptive).

## 2. The single TEST look: registered manifests, tools and order (adds to EVALUATION_CONTRACT §10; replaces nothing)
- **Finding (MEASURED):** `v4data/manifests.py:98-105` refuses every manifest id containing "TEST" unconditionally and has no TEST
  read path (`read_non_test`, L108-115); REGISTRY_V2 has no TEST role; no launcher, baseline or ONNX tool can address a TEST
  manifest. The contract's single TEST look (E10.1) therefore has NO executable path today.
- **Decision — manifests built at the look (task DATA-09, Opus, reviewed; built only after ALL_METHODS_LOCK exists, its sha256 is
  in TEST_EXPOSURE_REGISTER with access_purpose MANIFEST_BUILD, and the register entry id is in every manifest header):**
  `MAIN_RETRO_TEST_5_100_131072` (M07: every RETRO_TEST decision of the MAIN users, grace-matured targets included, same schema as
  MAIN_VALIDATION); `UNEW_HOLDOUT_RETRO_TEST_32768` (support = UNEW_HOLDOUT_SUPPORT_32768 unchanged, eval = the RETRO_TEST rows of
  the same users); `TRUE_COLD_RETRO_TEST` (M08); `SPARSE_PANEL_16384_RETRO_TEST`, `HEAVY_PANEL_16384_RETRO_TEST` (eval rows of the
  registered panel users). M09 (TEST cold-h) is NOT registered. Registry extension `REGISTRY_V2_TEST_EXT.json` (REGISTRY_V2 entries
  byte-identical; roles `MAIN_TEST`, `UNSEEN_HOLDOUT_TEST`, `TRUE_COLD_TEST`, `SPARSE_PANEL_TEST`, `HEAVY_PANEL_TEST`).
- **What the look evaluates, once each:** every locked central / FL checkpoint by `reeval.py` (C_BIG, C_FULL, C_SMALL, FA, FP, PF
  (with its PF_STORE), FA_1024, DP8, DP1, [DP4], FA_Q8: both PRACTICAL_BEST and CONTROLLED_BUDGET checkpoints); the device arms
  LO / FT_C / FT_FA re-executed on the `MAIN_TEST` role with the SAME locked SELECTION.json and the SAME support rows (their
  per-client models are never stored, so the TEST look re-runs the registered procedure; the procedure, not the weights, is what is
  locked); baselines POP / LAST / SKNN / KNN_EMB / RULE on `MAIN_TEST` (CPU); C_SMALL ONNX FP32/INT8 rows on `MAIN_TEST`
  (`dump_rows --role MAIN_TEST`); panels (UNSEEN_HOLDOUT_TEST, TRUE_COLD_TEST, SPARSE/HEAVY_TEST) for the arms that have VALIDATION
  panel rows. Then ANA-01 (frozen) on the TEST block with `records.VALIDATION_RESULT` for the shift indicator. Nothing is re-run
  after the look except an interrupted computation with identical hashes (E10.3).
- **Order:** DATA-09 code + review + deployment happen DURING the downstream runs; the builder itself runs only after the lock.

## 3. Confirmatory rule for the DP contrasts (clarifies 06B §2; replaces the rule for K12 and K13 only)
- **Finding (FL-07, OPINION confirmed):** the DP guarantee (ε, δ) covers the 384 released rounds; the PRACTICAL_BEST checkpoint of
  a DP arm is chosen by the best of 12 marks on MAIN_VALIDATION rows of the SAME users — a data-dependent selection outside the
  accountant. Under user-level adjacency the users' VALIDATION rows are part of their records.
- **Decision:** K12 = R(DP8 / FA_1024) and K13 = R(DP1 / FA_1024) are CONFIRMATORY under the pairing
  CONTROLLED_BUDGET / CONTROLLED_BUDGET (the round-384 endpoint of both arms: a data-independent release). Their PRACTICAL_BEST
  ratios are reported as DESCRIPTIVE with the disclosure "best-of-12 selection on the users' own validation data is not covered
  by the accountant". Every other contrast keeps PRACTICAL_BEST primary with CONTROLLED_BUDGET reported (06B §2). The
  ANA-01 script gains a per-contrast confirmatory-rule map (task ANA-02) BEFORE its freeze.
- **Registered DP disclosure list (thesis §5.5 and §9):** the following data-dependent choices are NOT accounted: the clip norm S
  (median of unclipped norms, S_CAL, seed 2026); the local-LR multiplier (CALIB screen: CALIB users ⊂ MAIN); the pooled bootstrap
  block (catalogue A, full-TRAIN normalisers, θ0; CLIENT_CONTRACT C5.6); the checkpoint choice for the descriptive
  PRACTICAL_BEST rows. The LR schedule and the 12 evaluation marks of DP runs depend on the round index only (FL-07). Drop-out is
  0 for DP arms. δ = 5e-6 < 1/N. The server sees clipped per-client deltas (no secure aggregation).
- **Registered analytical note for the write-up (no run, no read):** with d = 20,710,048 shared parameters, m = 1,024 and z = 0.5642
  the per-round noise added to the averaged update has expected L2 norm z·S·√d/m ≈ 2.51·S, against a signal whose norm is at
  most S (the mean of clipped deltas). This is stated before any DP value is read, so a large DP cost is interpreted as a property of
  the registered (m, T, d) design, not as a post-hoc excuse. The cohort/rounds trade-off under server LR 1 (larger m with fewer
  rounds moves the model less per client-visit) is why m = 1,024 / T = 384 stays as registered; an m = 4,096 / T = 96 sensitivity
  arm is NICE (§8).

## 4. Confirmatory hierarchy and multiplicity (adds to 06B §2)
- Primary headline: **K2**. Secondary: K1, K3, K8, K12. Tertiary: the rest. All sixteen are reported with unadjusted 95% paired
  user-bootstrap CIs (1,000 resamples; a Holm/Bonferroni-adjusted 99.7% percentile is not resolvable with 1,000 resamples). The
  thesis states the hierarchy, the number of confirmatory contrasts (16), and that every other number is descriptive. The claim
  vocabulary of EVALUATION_CONTRACT §7.6 applies to K1–K16 only.

## 5. FL "still improving" flag and the optional budget-sensitivity arm (adds; descriptive only)
- For every FL arm and seed, ANA-01 prints `still_improving_at_6.0 = (PRACTICAL_BEST mark ≥ 5.5)`. If FA (2-seed) is flagged on
  VALIDATION, an arm `FA_X2` (FA exactly with endpoint_rounds = 2 × 6,827 = 13,654; 12 marks over 12.0 EFE; seed 2026) MAY be run
  as a DESCRIPTIVE budget-sensitivity run (cut list tier NICE; label BUDGET_SENSITIVITY; never a confirmatory row; registered here
  so it cannot be a post-hoc addition).

## 6. FA_Q8 upload bytes (clarifies 06B §5; ratifies FL-07 item 5b)
- Upload bytes per visit = Σ numel (int8) + 4 × n_tensors (FP32 scales) + the same 8-byte n_consumed header FA's FP32 upload
  carries (`fl/comm.py:27`). Download unchanged (dense FP32). FL-07's out-of-packet edits (`fl/dp.py` Poisson guards,
  `fl/quant.py` header, 4 re-pinned tests) are APPROVED as consistent with 06C and this item.

## 7. C_FULL population number (corrects amendment 06 §5; definition unchanged)
- C_FULL users = the registered TIEBREAK band = ranks [0, 1,348,682) of the 5–100 pool (`code/v4fs_data/fs_common.py:48`);
  expected decisions 26,677,600 (DATA-08). The figure 1,315,914 in 06 §5 was a superseded pre-DL-40 number. The automatic C3
  continuation to 8.0 EFE applies to C_FULL exactly as to C_BIG (DL 151). C_FULL runs alone on the card (06 §4).

## 8. Priority tiers for the ONE-A40 schedule (binding order; replaces 06A §4 and 06B §6.2 ordering)
MUST (the answer to the question): A40 receipts + pin → C_BIG/C_SMALL re-evaluation → LR screen → FA ×2 → LO ×2 → FT_C
(selection + ×2) → C_SMALL INT8 (CPU) → K1–K16 preflight on VALIDATION → ALL_METHODS_LOCK → the TEST look.
SHOULD-high: FP ×2, PF ×2, S_CAL, FA_1024 ×2, DP8 ×2, DP1 ×1, FT_FA (selection + ×2), FA_Q8 ×2, C_FULL s2026, KNN_EMB SMALL/BIG,
UNSEEN_HOLDOUT / TRUE_COLD / SPARSE / HEAVY panels (seed 2026), browser ONNX (owner's PC).
SHOULD-medium: DATA-07 T41 on both panels + Study A (evaluation-only) on WARM_TRUNCATION; Study B (FT_C, LO) on COLD_H seed 2026.
NICE: DP4, DP1 s2027, C_FULL s2027, FA_X2, CPF, KD_SMALL/FT_KD, m = 4,096 DP sensitivity, μ sensitivity, seed 2028.
Anything cut is reported as "not run" in the thesis; nothing cut is replaced by a value from another population.

## 9. Disclosures added to the thesis list (06A §9, 06B §7)
(a) The contest was evaluated on A16 GPUs and the deployment study on an A40; the C_BIG/C_SMALL checkpoints were re-evaluated on
the A40 and every retention ratio uses A40 rows only; the A16 values are shown as a device sensitivity. (b) The TEST look re-runs
the device arms' locked procedure (per-client weights are not stored). (c) K12/K13 confirmatory under CONTROLLED_BUDGET (§3).
(d) The DP noise-to-signal note (§3). (e) The exact list of unaccounted DP hyper-parameters (§3). (f) Multiplicity policy (§4).
(g) The proposal's tasks (next category, purchase AUC, comparable offers) and datasets were replaced by agreement with the supervisor;
the "what changed from the proposal" section lists each substitution with its reason.

# PREG1 amendment 06B — adoption of the Fable thesis-level review (registration only)

- **Status:** REGISTERED 2026-09-28 by the Operations Director (Claude Opus 5.5) on the owner's instruction "do everything
  you see as right" (after the quota reset). Source: `company/reviews/FABLE_THESIS_RESEARCH_REVIEW_20260928.md` (§C, §D, §E).
  Registered BEFORE any downstream value is read: C_SMALL rows, BASELINE_RULE, KNN_EMB, and every FL/device arm are
  UNREAD (the server is down; no downstream run exists except C_SMALL and RULE, both sealed/unread).
- **Unchanged:** amendments 03–06A except where a numbered item below says "replaces".

## 1. Federated local-LR screen (review D-2, MUST)
- **What:** FA only, CALIBRATION role (clients = CALIB_TRAIN_5_100_32768 users keyed by client_key_hex; evaluation on
  CALIB_VALIDATION_5_100_32768 via the frozen T3 path), seed 2026, local-LR multipliers **m ∈ {0.5, 1, 2}** × the SMALL
  peak (0.000711). **Full 6.0-EFE schedule** on CALIB (rounds = ceil(3 × 32,768 / 64 / 0.9) = 1,707; same knobs as FA:
  64/round sweep, 10% seeded drop-out, B16 × 2, FedAvg weighting, 12 marks). Ops change vs the review's "1.0 EFE"
  proposal: the full schedule removes short-horizon bias and mirrors the SMALL central ladder (amendment 03 §3); cost
  ≈ 1/4 of a MAIN FA run each (INFERENCE).
- **Rule:** select the m with the highest CALIB_VALIDATION client-macro MRR@20 at PRACTICAL_BEST (best of the 12 marks);
  tie (|Δ| ≤ 0.0005) → the m closest to 1 (1 before 0.5 before 2). Written O_EXCL to
  `contracts/fl/LR_SELECTION.json` with the three values and a DECISION_LOG row. These three CALIB values are the only
  downstream values read before the VALIDATION preflight (§6).
- **Application:** the selected m multiplies the local shared-model LR of **FA, FP, PF, FA_1024, DP (all ε), S_CAL,
  FA_Q8**. Unchanged: PF personal LR (= SMALL peak × round factor, PREG1 T2), FedProx μ = 0.01, the FT grid (fractions of
  the SMALL peak), LO (lo_step schedule at the SMALL peak). S_CAL must run with the selected m (S depends on it).
- **Disclosure:** FL local LR screened on CALIB (3 points); μ and server LR (1.0) untuned.

## 2. Confirmatory contrast registry (review D-5, MUST)
Confirmatory = the only contrasts stated as findings with CIs; everything else (strata, bins, panels, curves, marks,
NDCG/HR, per-seed) is DESCRIPTIVE. Unit: client-macro MRR@20, 2-seed mean (1 seed where only one exists), PRACTICAL_BEST
primary and CONTROLLED_BUDGET reported, MAIN population, paired user bootstrap 1,000 (seed 20260926), ratios bootstrapped
directly. Primary on MAIN TEST at the single look; VALIDATION interim.
| # | Contrast | Question |
|---|---|---|
| K1 | R(C_SMALL / C_BIG) | size cost |
| K2 | R(FA / C_BIG) | federated retention (headline) |
| K3 | R(FA / C_SMALL) | federation cost at matched size |
| K4 | R(FP / C_BIG) | FedProx retention |
| K5 | R(PF / C_BIG) | personalised-FL retention |
| K6 | R(FT_FA / C_BIG) | deployed on-device system (FL + local fine-tune) |
| K7 | R(FT_C / C_BIG) | cloud-trained + local fine-tune |
| K8 | R(LO / C_BIG) | device-only retention |
| K9 | FT_C − C_SMALL | local fine-tuning gain (paired Δ) |
| K10 | FT_FA − FA | local fine-tuning gain after FL (paired Δ) |
| K11 | FA − LO | collaboration value (paired Δ) |
| K12 | R(DP8 / FA_1024) | privacy cost at ε = 8 |
| K13 | R(DP1 / FA_1024) | privacy cost at ε = 1 (seed 2026) |
| K14 | R(FA_Q8 / FA) | compression cost (8-bit uploads) |
| K15 | R(C_BIG / C_FULL) | cloud data advantage (seed 2026; if C_FULL runs) |
| K16 | R(INT8 / FP32) for C_SMALL | on-device quantisation cost (seed 2026) |
The list is copied verbatim into `ALL_METHODS_LOCK.json` and read by ANA-01 (`contrasts` input); ANA_SCRIPT_FREEZE pins it.

## 3. Privacy-level ladder (review §0.5, MUST — thesis framing, no run)
| Level | Arms | Leaves the device (training) | Leaves the device (inference) | Formal guarantee |
|---|---|---|---|---|
| L0 cloud-train / cloud-infer | C_BIG, C_FULL, C_SMALL | raw events | query context | none |
| L1 cloud-train / device-infer | C_SMALL shipped (ONNX FP32/INT8), KNN_EMB | raw events (model trained centrally once) | nothing | none |
| L2 federated | FA, FP, FA_1024, FA_Q8, PF (shared part) | model deltas (dense FP32; 8-bit for FA_Q8) | nothing | none (server sees clipped/unclipped deltas) |
| L3 federated + DP | DP ε = 8, ε = 1 | clipped deltas + Gaussian noise on the sum | nothing | user-level (ε, δ = 5e-6)-DP of the released model |
| L4 federated + local personalisation | PF (p_u), FT_FA | as L2; personal parameters never leave | nothing | none beyond L2 |
| L5 cloud model + local personalisation | FT_C | nothing after the download | nothing | none (download only) |
| L6 device-only | LO, RULE (own history) | nothing | nothing | data never leaves (shared catalogue/θ0 only) |
Disclosed (CLIENT_CONTRACT C5.6): every arm shares the pooled bootstrap block (catalogue, transform statistics, θ0).

## 4. Privacy curve (review D-3, D-4)
- **DP ε = 1 promoted to the core tier:** seed 2026 (seed 2027 if time), z = 1.3094 (registered, amendment 06 §2 Q5).
- **ε = 4 (optional, seed 2026):** z computed by `fl.accountant` with the same (m, N, T, δ) and rounded up to 4 dp; the
  z value is recorded in `contracts/fl/DP_Z_EPS4.json` + DL row BEFORE the run.
- **WOR sensitivity (analysis only):** ε at the registered z under the fixed-size sampling-without-replacement RDP bound
  (Wang, Balle & Kasiviswanathan 2019, AISTATS) is reported beside the registered Poisson-bound ε. Registered z unchanged.

## 5. New arm FA_Q8 (review D-6)
- FA exactly (knobs, rounds, drop-out, selected local LR), except each surviving client's upload = its model delta
  quantised per tensor to int8 with a symmetric uniform quantiser (scale = max|Δ| / 127 per tensor, FP32 scale sent) and
  **stochastic rounding** seeded by derive_seed(seed, "q8", round, client); the server dequantises and aggregates as FA.
  Download stays dense FP32. Bytes: up = Σ(numel) + 4 × n_tensors per visit; recorded by fl.comm. Seeds 2026, 2027.
- Method code "FA" with arm label FA_Q8 (ARCHITECTURE_LOCK allows FA); rows labelled FA_Q8.

## 6. Order and reads
1. INT-05, DEV-02, FL-05 (lr-mult, CALIB role, FA_Q8), FL-06 (WOR accountant), ANA-01 (+ contrasts), DATA-07, DATA-08,
   DEV-03 (browser page) — code, local.
2. Server back → A40 receipts + C_BIG/C_SMALL re-evaluation (unread) → **LR screen (§1)** → FL smokes → core FL (FA, FP,
   PF × 2) → S_CAL → FA_1024 × 2 → DP8 × 2 → DP1 → FA_Q8 × 2 → device (SEL on UNSEEN_VAL, MAIN × 2, panels) → ONNX.
3. **C_FULL seed 2026 moves BEFORE ALL_METHODS_LOCK** (amendment 06 §5 spec; manifest by DATA-08), on one A40 in
   parallel with the FL queue on the other.
4. **Controlled cold start (DATA-07) moves to SHOULD:** if the truncated-history builder passes its T41 golden-replay
   negative control, Study A on WARM_TRUNCATION_16384 and Study B on COLD_H_16384 (h ∈ {0,1,5,10,20,50}; contract §11)
   are run before ALL_METHODS_LOCK; the observational bins of 06A §2 stay as registered (both reported).
5. **First read = ANA-01 on VALIDATION**, all arms at once, after the ANA freeze; then ALL_METHODS_LOCK; then the single
   TEST look.
6. Browser measurement (ONNX Runtime Web on the owner's PC) after the ONNX export; label HISTORICAL_REFERENCE.

## 7. Disclosures added
(a) The review's proposal-bibliography corrections ([5] title; [6] IEEE Intelligent Systems 36(5) 2021; [8] Luo et al.
CIKM 2022 vs Zhang et al. IJCAI 2023 PFedRec; [12] unverified) go into the thesis bibliography. (b) RETRO_TEST contains
Black Friday 29 Nov 2019 — the VALIDATION→TEST change of C_BIG is reported as a shift indicator. (c) The contest selects
the backbone under rule T1; never phrased as "SASRec beats GRU". (d) Downstream arms trained on A40; the contest on A16.
(e) UNSEEN_VAL is a selection panel, never a headline panel; applying the FT recipe to TRUE_COLD/SPARSE is an extrapolation.

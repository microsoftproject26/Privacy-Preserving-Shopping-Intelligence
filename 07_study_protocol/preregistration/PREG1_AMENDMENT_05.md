# PREG1 amendment 05 — MAIN-scale candidate selection (registration only)

- **Status:** REGISTERED by the Operations Director on 2026-09-27, BEFORE any MAIN run has been launched and before any
  MAIN or SASRec-P2 value has been read (GRU P2 values were read for G1b part 1 as registered). Authority: owner
  directive DECISION_LOG 93 ("do it right — ALL training, all models, on MAIN") and Fable's ruling
  `company/reviews/MAIN_SCALE_FABLE_RULING.md` (design A modified).
- **Supersedes:** amendment 02 §2 (single tuned recipe per family in P3) and the single-recipe-per-family 02b registry
  format. **Unchanged:** P2 (02R1/02R2) rules and procedure (SASRec P2 finishes as registered, hash unchanged), all
  data/protocol locks, the staggered-P3 seal rules, amendments 03 and 04, TEST sealing.
- **Changes after the first MAIN launch are refused** (a new amendment and a Fable call are needed).

## 1. Role of P2
P2 on CALIBRATION (the nested first 32,768 MAIN users) is a **screen** that fixes the within-size hyperparameters.
MAIN_VALIDATION is never read in P2.

## 2. Candidates (fixed count, mechanical, no extension)
Per family, **exactly two candidates** = the P2 stage-2 configuration of each of the **top-2 size levels** ranked by
their CALIB two-seed 6.0-endpoint mean (the p2_driver stage-2 table). No grid extension, no new configuration.
- GRU (from P2_GRU_STAGE2, self da08395b…): **C1 = P2_GRU_D128_S1_s4** (recipe id `02B_GRU`, already registered, sha
  bbe3b88f…; must stay byte-identical) and **C2 = P2_GRU_D256_S1_s2**.
- SASRec: computed mechanically from P2_SASREC_STAGE2 when it ends (nobody reads earlier); its C1 is the p2_driver
  'tuned' recipe (`02B_SASREC`).
- A family whose stage 2 has only one valid size level has one candidate; this is recorded, not repaired.

## 3. P3 contest (MAIN only)
Every candidate × seeds 2026 and 2027 = **8 MAIN runs** (MAIN_TRAIN_5_100_131072), 6.0 EFE, marks every 0.5 EFE,
automatic continuation 6 → 8 on STILL_IMPROVING per the staggered ruling (its rows never selected). GRU runs
sequentially on GPU0; SASRec runs in parallel. INFERENCE ≈ 35–45 GPU-h.

## 4. G2 — one script, one unsealing, two steps
(a) **Within family:** the candidate with the higher two-seed mean of the 6.0 endpoint macro MRR@20 on
MAIN_VALIDATION; |difference| ≤ 0.0005 → the smaller model (parameter count), then the lower peak LR. Exclusions as
02R1 §3; if both candidates of a family are excluded → Fable.
(b) **Between families:** the registered T1 rule on the two selected recipes (DECISIVE_STABLE only);
ARCHITECTURE_EQUIVALENT → HOLD (owner-pending GRU-on-cost).
The CALIB-disjoint MAIN_VALIDATION diagnostic (38,496 users) is printed for both steps (never selection); ranking flips
are reported. Non-selected candidates are labelled **CANDIDATE_NOT_SELECTED** and never enter P4, P5, SYSTEM or the
downstream study.

## 5. Unsealing preconditions (replaces the 4-run form)
All 8 runs complete; `company/gates/G1B_APPROVED.json` holds part 1 (byte-identical), part 1R (the two GRU recipes)
and part 2 (the SASRec recipes), append-only; all 8 RunConfig hashes match the pinned recipes; the G2 script hash is
frozen before unsealing.

## 6. Code
Prefer no trainer change. Any change is limited to the recipe registry (a second recipe id per family), the G1B part
format, the unseal gate and the G2 selection; it ships as a NEW frozen tree (`code_p3b/`), reviewed, and its dry-run
RunConfig hashes for the GRU D128 runs must equal s2026 d9469319446d4a4aaf50729320572a3e79820d4083e52dc5986f6f777bc2cbd5
and s2027 4ed9a13a96f59edeb3a8de4ed09b2ca5e8ad2b9ef658dade775c7d222823fc36. `code_p3/` stays frozen for the GRU D128
runs (G1b C3).

## 7. Downstream and reporting
C_BEST, SYSTEM and TEST (P4/P5) apply to the selected recipe of each family only; the tuning ledger publishes all
candidates and all P2 trials. Thesis disclosures (Fable): within-size hyperparameters chosen on the nested 25% subset,
size and final recipe chosen at MAIN scale, every contest model trained on MAIN only; the GRU LR edge (0.001687) was
not extended; family MAIN_VALIDATION values are max-of-2 (selection-biased, symmetric) — TEST is the only unbiased
number; 24.8% user overlap CALIB_VALIDATION / MAIN_VALIDATION; SASRec's dropped size chosen on CALIB only.

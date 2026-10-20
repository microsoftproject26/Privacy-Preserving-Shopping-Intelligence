# PREG1 amendment 06 — downstream execution decisions after G2 (registration only; no Fable — owner directive DL 132+)

- **Status:** REGISTERED by the Operations Director on 2026-09-28 after G2 (ARCHITECTURE_LOCK fd201013…), BEFORE any
  downstream run or read. Authority: owner decisions 2026-09-28 (review of company/plans/REST_OF_PROJECT_PLAN.md:
  "your suggestions are fine"; no more Fable calls; ~10% Claude quota left). Builds on amendments 03, 04, 05.
- **Disclosed:** these decisions were taken by Ops without a Fable review (owner directive), using prior Fable
  rulings and the independent Opus code review (reviews/DOWNSTREAM_CODE_OPUS_REVIEW.md) as precedent.

1. **SMALL = SASREC_D64_B2** (amendment 03 §1): 20,710,048 shared parameters, 165,680,392 B per visit (down+up FP32).
   Recipe per amendment 03 §3 (4-LR mini-ladder on CALIB at the four SASRec 02R2 §1 midpoints, S0 + dropout [0.1]
   inherited from 02B_SASREC, seed 2026, 6.0 EFE; best re-run at seed 2027; 02R1 §3 selection).
2. **FL decisions** (FL-0304 open questions): (Q1) fixed-round runs use the normalised schedule s(6·E/E_total); raw EFE
   reported. (Q2) dropout_p = 0 for FA_1024 and DP arms (T = 3N/m = 384; no privacy credit from drop-out); FA/FP/PF keep
   10% seeded drop-out. (Q3) S = median client-delta norm of an UNCLIPPED calibration pass with the exact FA_1024 config,
   seed 2026, rounds 1–20; disclosed that it is non-private and follows the unclipped trajectory. (Q4) FA/FP/PF keep the
   sweep sampling (partial 4th visit disclosed); FA_1024 and DP use uniform sampling without replacement (the
   FA → FA_1024 comparison changes sampling; disclosed). (Q5) the standard RDP → (ε, δ) conversion is registered; z(ε=8)
   = 0.5642 (δ = 5e-6, m = 1,024, N = 131,072, T = 384; code/tests_fl/receipts/DP_Z_RECORD_FL04.json); the improved
   conversion is reported for information only.
3. **Deferred** (owner tiering DL 89, confirmed): CPF, DP ε = 1, KD_SMALL, FT_KD — run at the end only if time allows.
4. **GPU sharing:** two FL/device jobs may share one A16 when each under-uses it (measured before use); STRICT
   numerics are independent of co-scheduling (Fable staggered-P3 ruling). Central/MAIN-scale training keeps one job per card.
5. **C_FULL (full 5–100 'cloud' reference):** approved by the owner as a secondary reference (retention also reported
   vs C_FULL); DEFERRED until the downstream phase is done (needs a new TIEBREAK-band training manifest, a registry/lock
   extension and a FULL population definition — the existing FULL_TRAIN path is the v1 40.2M union). When run: users =
   the 5–100 pool minus the U_NEW bands (1,315,914), TRAIN-period events only, S* columns, recipe 02B_SASREC unchanged,
   6.0 EFE, marks + PRACTICAL_BEST, evaluated on the SAME MAIN_VALIDATION rows (and MAIN TEST at ALL_METHODS_LOCK),
   seed 2026 (seed 2027 if time). Disclosed as added after G2. U_NEW bands stay held out (cold-start study).
6. **Strata:** cut points from MAIN TRAIN (v4base.strata, keyed by client_key_hex, NO_TRAIN stratum) written O_EXCL
   before any downstream value is read.
7. **Primary answer:** R = 100 × arm / C_BIG (client-macro MRR@20, 2-seed mean, paired user bootstrap 1,000, seed
   20260926); C_BIG = the 02B_SASREC runs at PRACTICAL_BEST (best@3.0) per the lock; CONTROLLED_BUDGET (6.0) reported too.

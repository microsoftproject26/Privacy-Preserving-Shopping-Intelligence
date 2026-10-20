# PREG1 amendment 06G — FedAdam server optimizer added to the federated tuning screen (registration only)

- **Status:** REGISTERED 2026-09-29 (~09:35Z) by the Operations Director (Claude Opus 5.5, session eda-7b) on the OWNER's explicit
  decision of ~09:17Z ("try higher rates ×4 and ×8 AND add a server optimizer (FedAdam) as real systems do"). Builds on 06F (the
  chief-scientist session's extension of the local-LR grid to {4, 8, 16[, 32]}), which stays in force unchanged. Registered BEFORE
  any MAIN FL value exists or is read; the only FL values read are the registered 06B §1 CALIB screen values.
- **Why (owner + realism):** production cross-device FL uses adaptive server optimizers; Reddi et al. 2021 (ICLR, "Adaptive
  Federated Optimization", Alg. 2) show FedAdam outperforming FedAvg on sparse/embedding-heavy tasks. Screening only the client LR
  would leave the "untuned FedAvg" objection partly open.

1. **FedAdam** (Reddi et al. 2021 Alg. 2): pseudo-gradient Δ_r = the round's aggregated model delta exactly as FedAvg forms it
   (for DP arms: the noisy fixed-denominator mean — post-processing, accounting unchanged); m_r = β1 m_{r−1} + (1−β1) Δ_r;
   v_r = β2 v_{r−1} + (1−β2) Δ_r²; θ_{r+1} = θ_r + η_s m_r / (√v_r + τ); β1 = 0.9, β2 = 0.99, τ = 1e-3, m_0 = v_0 = 0, no bias
   correction; server state checkpointed and resumed.
2. **Additional screen configurations** (identical protocol to 06F §1: FA, CALIBRATION, seed 2026, 1,707 rounds / 6.0 EFE, 12 marks,
   PRACTICAL_BEST): FedAdam × η_s ∈ {0.01, 0.1} × local multiplier m ∈ {1, m*}, where m* = the 06F-selected FedAvg multiplier
   (if m* = 1, use m ∈ {1, 4}). Four runs, launched as soon as the FedAdam code (task FL-10) is reviewed; they may run in parallel
   with MAIN device lanes.
3. **Final selection (supersedes LR_SELECTION_EXT.json for launches once written):** over the union of the 06F FedAvg runs and the
   four FedAdam runs, the highest CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST; ties (|Δ| ≤ 0.0005) → FedAvg before
   FedAdam, then smallest |log2 m|, then smaller η_s. Written O_EXCL to `contracts/fl/FL_TUNING_SELECTION.json` (every value, the
   rule, the winner: optimizer, η_s, m) + DECISION_LOG row. Every MAIN FL run (FA, FP, PF, FA_Q8, FA_1024, S_CAL, DP) uses it.
4. **Order / cost:** FedAvg screen (06F) first; FedAdam screen after FL-10; MAIN FL starts only after FL_TUNING_SELECTION exists.
   INFERENCE: +3 h wall for the FedAdam screen if FL-10 lands while the 06F screen runs; the GPUs meanwhile run C_FULL ×2 and the
   device arms (which do not depend on the FL tuning, except FT_FA).
5. **Co-scheduling (owner directive "100 % utilisation"):** FL / device jobs may share a card with a central run (C_FULL); STRICT
   numerics are independent of co-scheduling (STAGGERED_P3 ruling); only wall time changes.
6. **Disclosure:** the thesis shows the full FedAvg LR curve and the FedAdam points, and states that β1, β2, τ were fixed to Reddi et
   al.'s defaults (not tuned) and that the local optimiser (fresh AdamW per visit), passes and μ remain untuned.

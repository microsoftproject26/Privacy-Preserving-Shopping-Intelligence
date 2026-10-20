# PREG1 amendment 06N — the pretrained (warm) line becomes primary; the random-start line becomes the ablation (registration only)

- **Status:** REGISTERED 2026-09-30 ~21:00Z (server clock) by the Operations Director on the OWNER's decision ("do the pretrained line
  first, complete and with good results, then the random line on top of it to measure the improvement"), after the chief-scientist
  memo `company/reviews/FABLE_RUN_ALLOCATION_20260930.md` (option B) and the practice audit
  `company/reviews/REAL_WORLD_FL_PRACTICE_AUDIT_20260930.md`. **No MAIN federated value has been read** (DL 250–281); the S7/S8
  confirmatory screens are running and unread. Amends 06D (K2 primary), 06K, 06L_C, 06M_B and ANA_CONTRASTS_v3 as stated below.
- **Why:** deployed systems and the FL initialisation literature start federated training from a pretrained model; a random start is
  the ablation (Nguyen et al. ICLR 2023, arXiv 2206.15387; Chen et al. ICLR 2023, arXiv 2206.11488; Xu et al. 2023, arXiv 2305.18465).

## N1. Primary and secondary contrasts
- **Primary:** K20b = R(FA_WARM* / C_FULL) with its required companion **K21 = FA_WARM* − PRE** (paired, 2 seeds), where FA_WARM* is the
  registered warm variant chosen in N2.
- **Secondary:** K20 = R(FA_WARM* / C_BIG); S8 − S7 (frozen vs full-model warm); FT_FA_WARM* − FA_WARM*; the warm-line method contrasts of N3;
  **K2 (cold FA vs cloud) = the random-start ablation**, together with the new **K_INIT = FA_WARM* − FA (cold)** ("value of the pretrained
  start"), both 2 seeds.
- Everything else in ANA_CONTRASTS_v3 keeps its tier unless listed here. The analysis script is updated before the ANA freeze.

## N2. The warm variant FA_WARM*
After the S7 and S8 selections (06M_B B2) exist: FA_WARM* = the variant (S7 full-model or S8 frozen tables) with the higher CALIB value of
its selected cell; ties within 0.0005 → S8 (the phone-class payload). Both S7 and S8 winners run on MAIN with seeds 2026 and 2027.

## N3. Order of MAIN work (priority)
1. Warm: FA_WARM (S7 winner) ×2 seeds, FA_WARM_FROZEN (S8 winner) ×2 seeds; then FT_FA_WARM* ×2 (and FT for the other variant if idle).
2. Warm method set on FA_WARM*: FP, PF, FA_Q8 (2 seeds each), then DP: S_CAL on the warm trajectory → S_RECORD_WARM → FA_1024 / DP8 (2 seeds)
   → DP1 (1 seed). DP8_FROZEN (06K) is this DP row when FA_WARM* = S8. Needs code task FL-16 (warm starts for these arms) + review.
3. Random-start line after the warm line (or when a card is idle): the running cold FA ×2 seeds continue (they are the K2 / K_INIT
   ablation); cold FP, PF, FA_Q8, FA_1024 / DP8 / DP1 are re-run only if time allows (single seed acceptable, labelled).
- Fallback: if FL-16 is not deployed by 2026-10-02 12:00Z, the warm DP row is not run and the DP result is reported on the cold line
  (FA_1024 / DP8 s2026, run then), labelled.

## N4. Stopped now (no value read; their partial runs are kept, labelled STOPPED_REALLOCATED)
Cold FP s2026 / s2027 (and the queued FA_Q8), cold FA_1024 s2026 (and the queued DP8 / DP1), the queued cold FA_1024 / DP8 s2027, and the
queued cold PF after cold FA. They may be re-launched later under N3.3 from scratch.

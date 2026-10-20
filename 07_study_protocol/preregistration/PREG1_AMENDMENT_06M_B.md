# PREG1 amendment 06M (part B) — warm-start server-step axis, late-phase contingency S3b, FT_FA_WARM (registration only)

- **Status:** REGISTERED 2026-09-30 ~08:25Z (server clock) by the Operations Director, from
  `company/reviews/REAL_WORLD_FL_PRACTICE_AUDIT_20260930.md` (R1–R4, R6). Before: any S3 selection read, any S3 cell value, any
  WARM_PREVIEW value, any S7 / S8 value, any FA_WARM MAIN run. Adds to 06K, 06J, 06L, 06M; nothing already registered changes except
  the S7 / S8 screen grids named in B2.

## B1. Preview extension (adds to 06M M1; same status: DESCRIPTIVE, never in any selection)
Three more CALIB runs from θ_pre at the same S2-leader point: FA_WARM at **η_s 0.001** × f ∈ {1, 0.25}, and one plain-FedAvg warm
cell (server LR 1, m 1, f 0.25). Run dirs `runs_ds3/WARM_PREVIEW/*`. All 7 preview runs are read together, after all have finished.

## B2. Confirmatory S7 / S8 screen grids (amends 06K §2–§3, before any S7 / S8 run exists)
- S7 (FA_WARM) at the S4 winner: {m*, m*/4} × {η*, η*/10} when the selected server optimiser is FedAdam; {m*, m*/4} × {η*, η*/4} for
  FedAvgM / FedAvg (η = server LR). 4 runs. Rule: the 06J rule restricted to these runs; ties → (m*, η*).
- S8 (FA_WARM_FROZEN): {m*, 4·m*} × {η*, η*/10} (FedAdam) or × {η*, η*/4}. 4 runs. Same rule; S8 screened in parallel with S7.
- Code: select-warm must accept the 2-D grid (task FL-15). Until FL-15 is reviewed, no S7 / S8 selection file is written.

## B3. Late-phase contingency S3b (before the S3 selection is read)
- Decline index D = (PRACTICAL_BEST value − mark-6.0 value) / PRACTICAL_BEST value, computed for the S3 winner.
- Trigger (mechanical): D ≥ 0.10 and peak mark ≤ 4.5 → S3b runs before S4; otherwise recorded "not triggered".
- Candidates at the S3 winner (2 CALIB runs): (a) SERVER_ANNEAL: η_s,r = η_s × the S3 winner's client-LR round factor; (b) SERVER_EMA:
  θ̄_r = 0.99 θ̄_{r−1} + 0.01 θ_r (bias-corrected, checkpointed), marks and PRACTICAL_BEST evaluated on θ̄, trajectory unchanged.
- Rule: 06J (≥ 0.002 over the incumbent, 0.0005 band, raw values); ties → incumbent → SERVER_ANNEAL → SERVER_EMA.
- Both are server-side and data-independent (clients stateless, bytes and DP accounting unchanged). Code: task FL-15, written only if
  triggered.

## B4. FT_FA_WARM (+ FT_FA_WARM_FROZEN): the full production stack
θ_pre → FA_WARM (or S8) MAIN → the registered FT_C local fine-tune (SELECTION_FT_C cell FT_C_f0.3_P4) → on-device inference. Seeds
2026 / 2027, DESCRIPTIVE, paired Δ FT_FA_WARM − FA_WARM reported with the harmed-client share (06L_C C13–C15). Code: a registered
FA_WARM checkpoint as a launch_device start kind (task FL-15 / DEV).

## B5. Analysis note (into the ANA freeze, DESCRIPTIVE)
"Quality during rollout": for FA, FA_WARM and S8, the area under the per-mark curve and the first mark reaching PRE-alone / LAST.

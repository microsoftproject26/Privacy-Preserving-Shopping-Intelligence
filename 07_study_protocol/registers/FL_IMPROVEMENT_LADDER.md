# FL improvement ladder — thesis register (append-only; newest last)
Purpose (owner, 2026-09-29): record every step taken to make the federated arm as strong as realistically possible, so the thesis
can state "when we did X, FL improved by Y". Values here are CALIB screen reads (registered, allowed) unless marked MAIN; MAIN values
are added only after the pre-declared analysis (never used to select). Each row: stage, what changed, registration, runs, value(s),
Δ vs the previous stage's winner, decision. Metric: CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST (FA, seed 2026,
full 6.0-EFE schedule = 1,707 rounds on 32,768 CALIB clients) unless stated.

| Stage | Change | Registration | Runs | Value(s) | Δ vs previous winner | Decision |
|---|---|---|---|---|---|---|
| S0 | FedAvg as registered (server LR 1, fresh client AdamW, B16×2, central LR shape over rounds), local LR = SMALL peak ×1 | 06, 06B §1 | LRSCREEN/x1 | 0.023473 | — (reference) | as-registered lower bound |
| S1a | local-LR multiplier grid {0.5, 1, 2} | 06B §1 | LRSCREEN/x0.5, x1, x2 | 0.020263 / 0.023473 / 0.025562 | ×2 vs ×1: +0.002089 (+8.9 %) | winner ×2 at the grid EDGE → extend |
| S1a-MAIN | FA on MAIN at ×2 (FedAvg), seed 2026 — descriptive ladder row | DL (12:13Z) | FL/LADDER_S1a_FA_s2026 (running since 12:13Z) | PENDING (read only at the §9 preflight) | — | descriptive only |
| S1b | local-LR multiplier grid extended {4, 8, 16} (×32 not needed: winner interior) | 06F | LRSCREEN_EXT/x4, x8, x16 | ×4 0.027570 / **×8 0.028796** / ×16 0.022909 | ×8 vs S1a winner ×2: +0.003234 (+12.7 %); vs S0: +0.005323 (+22.7 %) | **winner ×8** (contracts/fl/LR_SELECTION_EXT.json sha 7d922e4a…) |
| S2 | server optimizer: FedAdam η_s {0.01, 0.1} × m {1, 8}; FedAvgM β 0.9, η 1 at m {1, 2, 8} (+ edge rules incl. FedAvgM m 0.5) | 06G, 06J §1, 06L A1, 06L_D D1 | LRSCREEN_S2/* (5 runs tree code_ds3s2 since 14:44Z; FEDAVGM_x1/x2 tree code_ds3r3 since 17:56Z) | FedAdam η_s 0.01: ×1 **0.034247** / ×8 0.016630; η_s 0.1: ×1 0.015632 / ×8 0.016130; edge η_s 0.003 ×1 0.027238, 0.001 ×1 0.021876; FedAvgM ×0.5 0.024236 / ×1 0.027541 / ×2 0.022626 / ×8 0.008489; τ 1e-5 edge running (read 07:55Z 2026-09-30, Ops, CALIB) | FedAdam η0.01 ×1 vs S1b ×8: **+0.005451 (+18.9 %)**; vs S0: +0.010774 (+45.9 %) | leading = FedAdam η0.01 τ1e-3 ×1 (≥ 0.002 rule met); FL_TUNING_S2.json after the τ edge |
| S3 (A2 cells) | client SGD-m 0.9 × FedAdam η_s 0.01 τ 1e-3 × FL_CONST, client lr {0.1, 1.0} — screened whatever S2 selects | 06L A2, 06L_D D2 | LRSCREEN_S3/A2_* (launched ~22:00Z on GPU1, tree code_ds3r3) | PENDING | PENDING | read only at the S3 selection |
| S7-PRE | pretraining model on 1,217,610 non-MAIN users (θ_pre = endpoint_6.0) for FA_WARM | 06K | runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026 (build started 12:39Z) | PENDING | — | init for S7/S8 |
| next | further stages per company/reviews/FL_STRONGEST_LADDER_RESEARCH_20260929.md (research running) | to register | — | — | — | — |

Context (INFERENCE, not a result): a central D64-class model on the same CALIB users reaches ≈ 0.11–0.12 (P2/ladder era), so S0 retains
≈ 20 % of the matched central value on CALIB.

**Training curves (CALIB E2E macro MRR@20 per mark, read 2026-09-30 ~08:10Z; MEASURED):**
- S0 x1: rises to 0.0235 at mark 2.5, then flat (0.0233–0.0235) to 6.0.
- S1b x8: noisy rise, peak 0.0288 at 3.5, 0.0261 at 6.0.
- S2 FedAdam η0.01 ×1: 0.0218 → 0.0342 at mark 3.0 (peak), then steady decline to 0.0251 at 6.0 (−27 % from peak).
- FedAvgM ×1: peak 0.0275 at 2.5, ≈ 0.026 after.
INFERENCE: past mid-budget the stronger FL recipes overfit the 32,768 CALIB clients (same pattern as central SASRec, which peaks early — G2 contest). Selection already uses PRACTICAL_BEST (the peak), so reported values are peaks. Training longer alone is not expected to help (relevant to S5 FA_X2); more data is — MAIN has 4× the clients, and S7 starts from θ_pre trained on 1.2M non-MAIN users. Candidate later rungs if still weak (need registration first): stronger regularisation (dropout / weight decay), an LR decay after the peak.
**Correction to the INFERENCE above (2026-09-30 ~08:30Z, after re-reading FL_STRONGEST_LADDER_RESEARCH §1.1):** the registered FA runs use
the CENTRAL LR shape mapped onto rounds — peak to EFE 2.0, linear decay to 0.1× over 2.0–3.0, 0.1× for 3.0–5.5. The FedAdam decline
starts exactly at mark 3.0, when the client LR reaches 0.1×. A second candidate mechanism: FedAdam's per-coordinate normalisation
(β2 0.99) re-amplifies the 10× smaller client deltas to ≈ η_s-sized steps once v forgets the early large deltas → noise-dominated steps.
Overfitting and this LR-shape × FedAdam interaction are both INFERENCE; the S3 FL_CONST cells (constant peak + short cool-down; A2
and W1/W2, running or done) directly test it.

| S7/S8 preview (06M) | warm start from θ_pre (PRE endpoint_6.0, 1.2M opted-in non-MAIN users) at the S2 leader point; FA_WARM f {1, 0.25}; FA_WARM_FROZEN (item tables frozen, not uploaded) f {1, 4} | 06M (descriptive) | WARM_PREVIEW/* (launched 08:09Z 2026-09-30, r4) | PENDING | vs S2 FedAdam cold 0.034247 (same point) | descriptive; registered S7/S8 run at the S4 point |
| S2 FINAL | τ edge added: FedAdam η0.01 τ1e-5 ×1 0.024374 | 06J §1 | LRSCREEN_S2/FEDADAM_eta0.01_tau1e-5_x1 | — | — | **winner FedAdam η_s 0.01 τ 1e-3 ×1 = 0.034247** (gain +0.005451 over S1b; contracts/fl/FL_TUNING_S2.json sha 24080b1e…) |
| S3 | LR shape FL_CONST × client optimiser: AdamW 0.033104; client SGD-m 0.9 lr 0.01 0.058775 / **0.1 0.094936** / 1.0 0.082381 (all FedAdam η0.01, m 1) | 06I §3, 06J §2, 06L A2 | LRSCREEN_S3/* | see left | **+0.060689 vs S2 (+177 %)**; vs S0 0.023473: **+304 % (×4.04)** | **winner FedAdam η0.01 + FL_CONST + client SGD-m lr 0.1** (FL_TUNING_S3.json sha 32decc6a…); SGD edge not triggered (0.1 interior); **S3b not triggered: D = 0** (the winner peaks at mark 6.0 = 0.0949, still rising) |

**Curves after S3 (MEASURED, CALIB per mark):** S3 winner 0.0285 → 0.0758 (3.0) → 0.0876 (5.0) → **0.0949 (6.0), still rising**; AdamW × FL_CONST still peaks at 2.5 and declines (0.0331 → 0.0295); τ 1e-5 flat ≈ 0.023. INFERENCE: the post-peak decline was caused by the fresh client AdamW (sign-like dense updates on all 158k output rows — research cause 2), not by the LR shape alone and not mainly by overfitting; with client SGD the model is now UNDER-trained at 6.0 EFE, so a longer budget (S5 FA_X2) is expected to help.

**S7/S8 preview results (06M/06M_B, DESCRIPTIVE, all at the S2 point = client AdamW-fresh, central LR shape; read 2026-09-30 ~14:25Z; CALIB PRACTICAL_BEST):**
| Run | best (mark) | mark 6.0 | curve |
|---|---|---|---|
| **FA_WARM_FROZEN f1** (item tables frozen, ≈1 MB/visit) | **0.1673 (0.5)** | 0.166 | flat 0.164–0.167 across all marks |
| FA_WARM_FROZEN f4 | 0.1624 (5.0) | 0.162 | flat 0.157–0.162 |
| FA_WARM η_s 0.001 f0.25 | 0.1582 (0.5) | 0.038 | collapses from 0.158 to 0.04 by mark 2.5 |
| warm FedAvg f0.25 | 0.1288 (0.5) | 0.030 | collapses |
| FA_WARM η_s 0.001 f1 | 0.0706 (0.5) | 0.030 | collapses |
| FA_WARM f0.25 (η_s 0.01) | 0.0301 (0.5) | 0.026 | destroyed by mark 0.5 |
| FA_WARM f1 (η_s 0.01) | 0.0293 (3.5) | 0.025 | destroyed |
Comparison at the same point: cold FedAdam (S2) 0.034247; cold S3 winner (client SGD) 0.094936.
INFERENCE: (1) with the old client (fresh AdamW) every full-model warm run DESTROYS θ_pre — the gentler the step, the later — the same sign-noise on the 158k output rows found at S3 (research cause 2); (2) freezing the item tables protects the pretrained structure: FROZEN f1 stays ≈ 0.166 throughout; (3) the frozen-table value is already reached at mark 0.5, so the federated gain over θ_pre alone may be small — that is K21 = FA_WARM − PRE, measured on MAIN at the §9 read (PRE on CALIB is not read: partial MAIN read); (4) the registered S7/S8 screens run at the S4 point with client SGD, where full-model warm runs are not expected to collapse.
| S4 FINAL | local computation at the S3 winner: passes 1 **0.107562** / P2 B16 (incumbent) 0.094936 / batch 8 0.088351 / passes 4 0.077878 | 06J §3 | LRSCREEN_S4/* | see left | **+0.012626 (+13.3 %) vs S3; ×4.58 vs S0 (0.023473)** | **winner: FedAdam η_s 0.01 τ 1e-3 + FL_CONST + client SGD-m 0.9 lr 0.1 + 1 local pass, batch 16** = FL_TUNING_SELECTION.json (sha 9bca186e…); DP / FA_1024 / S_CAL keep 2 passes (DL 219 exception) |

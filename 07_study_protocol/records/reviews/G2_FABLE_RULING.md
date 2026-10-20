# G2 — Fable 5.1 ruling (verbatim; fresh agent, one message; received 2026-09-28 ~11:33Z; packet G2_PACKET.md)

**VERDICT: CONFIRM_WITH_CONDITIONS**

**LOCK:** Write ARCHITECTURE_LOCK.json from the draft (sha 3a0db5b3…) with its values unchanged, status LOCKED, and add: this verdict's path; recipe sha 9a253087… and registry self-sha 0b0bc3b4…; `topology_hash` for 02B_SASREC (EVALUATION_CONTRACT_V4 topology row); per-seed θ0 hashes **re-measured on the server now** (DATA_PROTOCOL_LOCK L528; the draft's own `requires`), equal to fe075edd… / b6e22c58…; C_BIG checkpoint identity — 02B_SASREC s2026/s2027 best@3.0 (PRACTICAL_BEST) and endpoint_6.0 (CONTROLLED_BUDGET) with file hashes; parameter counts 83,881,764 / 20,962,210. Keep the registered `disclosure` string verbatim.

**CONDITIONS:**
1. Re-measured θ0 hashes match the RUN_START receipts; mismatch → HOLD.
2. 02R1 §4: winner payload bytes and SYS-02b FL forecast registered with the lock.
3. E11.9a recipe grid (PF/CPF p_u-only, LO full, passes {1,2,4}, personal LR, λ) registered at the lock, before any M04S read.
4. Checkpoints above and 02B_GRU best@6.0 (P5 singles) never deleted; C2 candidates stay CANDIDATE_NOT_SELECTED.
5. Diagnostics from existing eval_6.0 ranks, labelled REPORTED/never-selection: CONTROLLED_BUDGET between-family delta with paired-bootstrap CI; GRU 7.0 PRACTICAL_BEST_EXTENDED values.

**DISCLOSURES:** T1 = PRACTICAL_BEST was frozen in PREG1_LOCK_VALUES before any central result; amendment 05 §4a's endpoint rule was registered before the first MAIN launch — so the mixed row types are pre-registered, not chosen. Under CONTROLLED_BUDGET the order reverses (GRU 0.1569 > SASRec 0.1527): the verdict is "best validated checkpoint within a 6.0-EFE budget". SASRec peaks at 3.0 and loses ≈0.0085 by 6.0; GRU was STILL_IMPROVING at 6.0. Not parameter-matched (83.9M vs 21.0M; SMALL D64_B2 is the matched size). MAIN seed gaps are now measured (0.00041 / 0.00002), superseding the "unmeasured" string. VALIDATION values are max-of-marks (symmetric bias); TEST is the only unbiased number. 40,764 of 51,211 users carry eval rows. SASRec's within-family pick used the endpoint; the disjoint PRACTICAL_BEST diagnostic agrees.

**DOWNSTREAM:** Now (no value reads): SMALL registry entry, θ0 seeds 2026/2027, SYS-04c bench. After the lock is written under 1–3: the CALIB 4-LR mini-ladder. C_BIG = the existing 02B_SASREC runs, no re-run. GRU: P5 singles only. No TEST.

**BASIS:** Amendment 05 §5 preconditions verified: 8 runs final, GRU D128 config hashes equal the pins, script sha frozen 04:28Z before the 11:25Z run, library shas match. Both steps applied registered rules mechanically; all T1 triggers false, CI [−0.00518, −0.00326] excludes 0 widely, disjoint diagnostic agrees without flip. The endpoint reversal must be published but cannot alter a pre-registered outcome; the conditions concern what the contracts require the lock to carry, not the result.

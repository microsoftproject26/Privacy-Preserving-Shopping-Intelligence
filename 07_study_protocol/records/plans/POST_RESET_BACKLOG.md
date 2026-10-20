# Post-reset backlog (owner decisions 2026-10-08; DL 553)

Rules for every item:
- All work is POST_HOC_EXPLORATORY.
- Selection uses inner validation / CALIB only.
- A TEST look needs a Fable gate, a register entry, and the owner's yes.
- No confirmatory artifact is ever touched.

## A. New experiments (owner: "اه نعمله")

**A1. Benchmarks — PRE on disjoint users (a real server-side data advantage, as in REES46).**
- Split users into a server band S and FL clients F, with no overlap. Train PRE on S only; run FL on F only, cold and warm-from-PRE. Evaluate on F's users.
- Use the fixed recipe: tau 1e-6, B24, unfrozen.
- Try 2 band sizes, e.g. 15 % and 50 %, so we can see how the warm-start value grows with the server data.
- Datasets: s3_beauty, s3_sports, s3_toys, ml_1m.

**A2. Benchmarks — remove the capacity confound (C_FULL D256 vs FL D64).**
- (a) C_SMALL: central D64_B2 on all users, at B6 / B24, s2026 + s2027. This gives the pure federation cost at matched size.
- (b) FL at D256_B3 (cold and warm), fixed recipe, s2026 + s2027. This gives the pure size cost under FL.
- Report the decomposition: total gap = size cost + federation cost.
- REES46 already has K1 R(C_SMALL / C_BIG) and K3 R(FA / C_SMALL): reuse them in the write-up.

## B. Running now (finish automatically)

- **B1. REES46 cold FA, tau-fixed (C1):** MAIN s2026 + s2027 → then DP8 (C2) and E1 (C3).
- **B2. PB100 strongest-FL grid:** S2 / S3 (fixed, DL 551) → final selection → MAIN s2026 + s2027 → lock v2 candidate.
- **B3. REES46 warm, tau-fixed (Part B):** MAIN done; proofB done.

## C. TEST looks to gate (after the reset; one Fable call can cover several)

- **C1.** Fixed benchmarks (6 datasets, FA / FA_WARM winners, 2 seeds): exploratory TEST + the usual analysis.
- **C2.** REES46 Part B (warm, tau 1e-6) and C1 cold FA: exploratory TEST with contrasts vs PRE, W*, C_FULL.
- **C3.** PB100 v2 (the grid winner) + its DP8 / E1.
- **C4.** FT-on-device on top of the best moving FL models: needs a v2 lock (Fable condition, DL 540).

## D. Write-up items (no compute)

- **D1.** The tau finding: MEASURED evidence, why it hid FL learning, and the corrected results beside the confirmatory ones.
- **D2.** FL value depends on the server-side data: REES46 1.2M θ_pre (K21 ≈ 0) vs PB100 (+0.0074) vs the benchmark A1 results.
- **D3.** FedProx inert at 1 local step (94–97 % single-step visits); θ_pre single seed; FA_WARM_FROZEN 0.0 for out-of-band targets.
- **D4.** Published comparison:
  - the Spotify Table 1 (full ranking);
  - a note that most FL papers use sampled metrics;
  - our ML-1M C_FULL 0.1936 is above the published eSASRec (0.1767).
- **D5.** Incidents and fixes: I055, the step-7 parallelisation (DL 535), the code_p2r4 read-only refusal (DL 536), the E12 / S3 wrapper bugs (DL 548–551), the lost channel (DL 541).

## E. Wrap-up

- Final report; thesis_reference capture (DL 442–553+); STATE / CHANGELOG; RESUME_HERE.
- GitHub branch (identity checked with `git var`, no AI attribution).
- Handover ZIP.
- Phone: restore the screen timeout, disable USB debugging.

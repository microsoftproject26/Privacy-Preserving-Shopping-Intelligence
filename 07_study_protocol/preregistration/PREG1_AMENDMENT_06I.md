# PREG1 amendment 06I — realism-audit registrations: contrasts vs C_FULL, FL tuning wave 3, interpretations, disclosures, DP at scale

- **Status:** REGISTERED 2026-09-29T09:41Z by Claude Fable 5.1 (chief scientist) on the owner's request that the realism audit's MUST
  items be registered (DL 192; `company/reviews/REALISM_AUDIT_20260929.md` §0, §G). Registered BEFORE any MAIN FL, device, C_FULL or
  panel value exists or is read (the only reads so far: the 06B §1 CALIB screen values). Builds on 06D–06H; nothing in them changes.

## 1. Confirmatory contrasts against the cloud reference (audit M3; adds to 06B §2 and 06E §2)
| # | Contrast | Question | Seeds | Rule | Hierarchy |
|---|---|---|---|---|---|
| K17 | R(FA / C_FULL) | federated retention vs the full-data cloud model | 2026 | PRACTICAL_BEST / PRACTICAL_BEST | secondary |
| K18 | R(FT_C / C_FULL) | cloud-trained + local fine-tune vs the full-data cloud model | 2026 | PRACTICAL_BEST / PRACTICAL_BEST | tertiary |
| K19 | R(LO / C_FULL) | device-only vs the full-data cloud model | 2026 | PRACTICAL_BEST / PRACTICAL_BEST | tertiary |
K2 stays the primary headline (R(FA / C_BIG) = the matched-data federation cost); K17 is its "cloud-data-advantage" co-headline.
The pre-lock registry `company/gates/ANA_CONTRASTS_v2.json` carries K1–K19 (K1–K16 verbatim from 06B §2); ALL_METHODS_LOCK copies it.
Code: ANA-03 (after REV-DELTA-01), same pairing rules (06A §7).

## 2. "FA" denotes the 06F/06G/06I-selected federated configuration (clarifies 06B §2, 06G §3)
Every FL arm (FA, FP, PF, FA_Q8, FA_1024, DP, S_CAL) runs the ONE configuration selected on CALIB by the registered tuning waves
(server optimiser, local multiplier, LR shape, client optimiser). There is no separate "FA_M" arm and no headline swap: the arm
names and the contrasts are fixed; only their tuned recipe is chosen before any MAIN run. The untuned FedAvg-η1 configuration is run
at MAIN only as the 06H DP_OPT_SENS sensitivity.

## 3. FL tuning wave 3 — LR shape and client optimiser on CALIB (audit B7/B8; SHOULD-high; registered rule)
- **Why:** the registered FL recipe maps the central warm-up/decay shape onto rounds (≈ 2/3 of rounds at ≤ 0.1 × peak) and re-creates
  AdamW every visit with ≈ 4 steps (sign-like updates on all 158,486 item rows of the full-K softmax). Both are structural handicaps
  a deployment would not have; a CALIB screen is cheap (≈ 2.7 h per run, in parallel).
- **Runs (4, same protocol as 06F §1, at the best-known (server optimiser, m) when launched):**
  (a) AdamW-fresh × **FL_CONST** shape; (b) client **SGD** (momentum 0.9) lr 0.01 × FL_CONST; (c) SGD lr 0.1 × FL_CONST;
  (d) SGD lr 1.0 × FL_CONST. FL_CONST = linear warm-up over the first 1 % of rounds to the peak, constant peak, linear cool-down over
  the last 10 % of rounds to 0.1 × peak (rounds-indexed; T2's exposure point "end" kept; the central schedule and every central run
  are untouched; T2 pin P3's equality of s(·) between fl.schedule and v4train.schedule still holds for the registered shape —
  FL_CONST is a second, FL-only shape registered here).
- **Selection (joint with 06F/06G):** the highest CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST over EVERY screened
  configuration; ties (|Δ| ≤ 0.0005) → the configuration with fewer deviations from the registered default (FedAvg, m = 1, S0 shape,
  AdamW), then FedAvg before FedAdam, then smallest |log2 m|, then smaller η_s, then AdamW before SGD, then smaller SGD lr.
  If the best (optimiser, shape, m, client optimiser) combination was never run jointly, ONE combination run is made and the
  selection is re-applied over the set including it. Output `contracts/fl/FL_TUNING_SELECTION.json` (06G §3) records every run.
- **Concurrency:** waves 2 (06G) and 3 may run concurrently with the tail of wave 1 using the best-known m at launch time; the
  combination run closes the gap. Code: FL-10 (Ops) gains the FL_CONST shape and the client-SGD-momentum option (bitwise resume
  tests; the registered defaults byte-identical); if FL-10 cannot carry it, FL-11 follows immediately.
- **HOLD:** S_CAL and every FA_1024 / DP / MAIN FL launch wait for FL_TUNING_SELECTION.json (S depends on the whole recipe).

## 4. FA_X2 promoted to SHOULD-high (audit S3; 06D §5 unchanged otherwise)
Run unconditionally after the core FL (seed 2026, 12.0 EFE, the selected configuration), DESCRIPTIVE budget sensitivity.

## 5. DP at deployment scale — DP8_FULL (audit S2; SHOULD; run if time; registered now so nothing is post hoc)
DP-FedAvg with the selected configuration on the TIEBREAK band (N = 1,348,682 users, the C_FULL population; support = their TRAIN
rows), Poisson q = m / N with **m = 8,192**, **T = ceil(3 N / m) = 494** rounds (≈ 3 expected visits per client, the MAIN
convention), **δ = 5e-7** (< 1/N), z(ε = 8) from `fl.accountant` for (m, N, T, δ) written to `contracts/fl/DP_Z_FULL.json` + DL row
BEFORE the run, S from an S_CAL_FULL pass (20 rounds, same rule), evaluated on the SAME MAIN_VALIDATION rows (and MAIN TEST at the
look), seed 2026, the `fl/pool.py` multi-worker driver. Reported as the "Edge-scale" privacy point: R(DP8_FULL / C_FULL) and
R(DP8_FULL / FA) DESCRIPTIVE (a different population than FA's); per-round noise ≈ z S √d / m stated beside 06D §3's MAIN value.
If time is short it replaces DP ε = 1 at MAIN scale (06D §8 cut order amended accordingly).

## 6. Interpretations registered before any read (audit M4, B9/B10; adds to 06D §3)
(a) The registered FL arms use an equal-exposure budget (6.0 EFE ≈ 3 visits per client, 6,827 averaged updates): their retention is
a LOWER BOUND of what a deployment with unconstrained rounds would reach; FA_X2 and the still-improving flag quantify the slack.
(b) FA_1024 / DP use 384 rounds of 1,024 by the same budget; the DP result is a property of the simulated scale (N = 131,072,
d = 20.7 M, m = 1,024 → per-round noise ≈ 2.5 S vs ≈ 0.23 S in McMahan et al. 2018 at C̃ = 5,000, d = 1.35 M); DP8_FULL is the
scale-realistic point. (c) The thesis reports "FedAvg/FedAdam-with-<client optimiser> at a 6-EFE budget retains X %", never
"FL retains X %" without the recipe.

## 7. Disclosures registered as thesis text (audit M2; zero cost)
(a) In every non-DP federated arm the uploaded delta reveals the user's viewed items to the server (the two 158,486-row item tables
are 98 % of the parameters; the non-zero rows of the input table are exactly the visit's items; no secure aggregation): "only
weights leave the device" is data minimisation, not privacy; DP is the only formal guarantee (ladder L2 vs L3). (b) Bytes per
client are also reported with dropped clients' downloads charged (audit S4; ANA-03 column). (c) Uniform / sweep participation
(no availability bias), FP32 downloads in FA_Q8, PF LR and μ untuned, the server sees clipped deltas under DP, the snapshot week,
Black Friday, the previously inspected TEST, the 6.0-endpoint contest reversal, battery, the display-catalogue leak of the SMALL
model on device (C2), on-device training availability in a browser (C3), LO as the no-collaboration control (C4), one-shot FT (C5).
(d) The FL tuning waves (06F/06G/06I) are reported in full as a figure and a table; the first grid ended at its edge.

## 8. Cut order (amends 06D §8 for the FL tier)
MUST unchanged. SHOULD-high: FP ×2, PF ×2, S_CAL, FA_1024 ×2, DP8 ×2, FT_FA, FA_Q8 ×2, C_FULL s2026, FA_X2, DP8_FULL (if the pool
driver receipt exists), panels, browser. SHOULD-medium: DP1 s2026, DATA-07 Study A/B. NICE: DP4, DP1 s2027, C_FULL s2027 (running),
CPF, KD, μ sensitivity, seed 2028, FA_WARM (owner decision pending; if adopted, registered as a separate amendment with its ladder
placement: L2 with public pre-training on non-MAIN users).

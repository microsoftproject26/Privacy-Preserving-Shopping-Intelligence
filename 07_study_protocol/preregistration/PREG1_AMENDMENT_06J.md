# PREG1 amendment 06J — the federated tuning LADDER, stages S2–S4 (registration only)

- **Status:** REGISTERED 2026-09-29 (~12:33Z) by Claude Fable 5.1 (chief scientist), claimed in DL 202. Source:
  `company/reviews/FL_STRONGEST_LADDER_RESEARCH_20260929.md` §1–§5 (diagnosis, catalogue, ladder, stopping rules) and the owner's
  instruction of 09:38Z (DL 195: "strongest realistic FL, staged for the thesis: when we did X, FL improved by Y; ordered so we do not
  go back"). Registered BEFORE any S1b, S2, S3 or S4 value is read and before any MAIN FL value is read. The only FL values read so far
  are S0/S1a (06B §1 screen). Builds on 06F (S1b), 06G (S2 FedAdam), 06H (DP server-optimiser sensitivity), 06I §3 (S3); where this
  amendment is stricter it supersedes them; nothing else changes.

## 1. Stage S2 — server optimiser (adds to 06G §2)
- Candidates at the S1b winner m* (06F selection, `contracts/fl/LR_SELECTION_EXT.json`): FedAdam η_s ∈ {0.01, 0.1} × m ∈ {1, m*}
  (06G §2, 4 runs) **plus FedAvgM** (server momentum β = 0.9, server LR η = 1, at m*; Hsu, Qi & Brown 2019) = 5 runs.
- **Edge rules (one step each, registered; no further extension):** if the best FedAdam run has η_s = 0.01 → two runs η_s ∈ {0.003,
  0.001} at that run's m; if a FedAdam run is the S2 winner → one run with τ = 1e-5 at the winner's (η_s, m).
- **S2 winner:** the highest CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST over the S2 set and the S1b winner (the
  incumbent); ties |Δ| ≤ 0.0005 → the simpler configuration first: FedAvg (incumbent) → FedAvgM → FedAdam, then smallest |log2 m|,
  then smaller η_s, then larger τ.
- Code: FL-10 (FedAdam) gains FedAvgM (a server momentum buffer, checkpointed; FedAvgM at β = 0 must equal FedAvg bitwise).

## 2. Stage S3 — LR shape and client optimiser (tightens 06I §3)
- Built ON the S2 winner (server optimiser, η_s, τ, m*): the four 06I §3 runs (AdamW-fresh × FL_CONST; SGD-m0.9 × FL_CONST at lr
  {0.01, 0.1, 1.0}).
- **Edge rules:** if SGD lr 1.0 is the raw best of the SGD runs → two runs lr ∈ {3.0, 10.0}; if SGD lr 0.01 is the raw best of the
  SGD runs → one run lr 0.003. Disclosed: gradient clip 1.0 binds for SGD (max step = lr × 1).
- **S3 winner:** best over the S3 set and the S2 winner (incumbent); ties → incumbent, then AdamW before SGD, then smaller SGD lr.
- **Sequential, not concurrent (supersedes the concurrency allowance of 06I §3):** S3 is launched only after the S2 winner is
  written (DL row); S4 only after the S3 winner. The 06I §3 "combination run" is therefore not needed and is withdrawn. Reason:
  "built on the previous winner" is what makes the thesis ladder read as "when we did X, FL improved by Y" without backtracking.

## 3. Stage S4 — local computation (adds)
- At the S3 winner, one-at-a-time: local passes 1 (instead of 2), local passes 4, local batch 8 (instead of 16) = 3 runs. No LR
  re-screen inside S4 (disclosed).
- **S4 winner:** best over the S4 set and the S3 winner; ties → incumbent. The S4 winner is written as
  `contracts/fl/FL_TUNING_SELECTION.json` (06G §3 format extended with the stage history) + DL row, and is "FA" for every FL arm
  (06I §2). S_CAL is re-run under it before any FA_1024 / DP run (06G §3, 06H §3).
- **Exposure accounting:** S4 changes the per-visit exposure (passes); the round count keeps the registered 6.0-EFE budget
  (rounds = ceil(6.0 × N_EFE / (expected survivor exposures per round))) so every stage stays at equal exposure; the launcher
  records the recomputed round count.

## 4. Stopping rules (register; apply to S1b–S4)
(i) The tie band 0.0005 stands. (ii) A stage whose best candidate gains < 0.002 over its incumbent (≈ the paired CALIB half-width,
INFERENCE) is recorded "no measurable gain" and the incumbent carries on. (iii) No further optimiser stage is added after S4 whatever
S4 shows. (iv) A candidate that breaks the realism budget (bytes per visit > 170 MB, clients stateful across rounds, server access to
MAIN users' data, a model other than SASREC_D64_B2) is out of scope. (v) A run with a non-finite loss or metric is excluded and
recorded. (vi) Every stage's values go into `company/registers/FL_IMPROVEMENT_LADDER.md` (the thesis ladder table) the moment the
stage selection is written, never before.

## 5. What this amendment does NOT do
No change to S0/S1a/S1b, to any central run, to the evaluator, to K1–K19, to the DP accountant, to the TEST rules. The descriptive
MAIN ladder row LADDER_S1a (DL 198; FA MAIN s2026 at ×2 FedAvg, launched 12:13Z from the interim tree code_ds3r2s) is ACCEPTED as a
DESCRIPTIVE ladder point only; disclosed as produced by an interim tree whose evaluator bytes equal the A40 block (v4eval frozen);
never confirmatory, never selects anything. Warm start (S7) and frozen item tables (S8) wait for amendment 06K (owner decision).

## 6. Cost (INFERENCE, ≈ 2.7 h per CALIB run, runs of a stage in parallel)
S2 5 (+ ≤ 3) runs ≈ 3–6 h wall; S3 4 (+ ≤ 2) ≈ 3 h; S4 3 ≈ 3 h → ≈ 9–12 h wall after S1b, then MAIN FL (≈ 14–18 h on 8–10 lanes).

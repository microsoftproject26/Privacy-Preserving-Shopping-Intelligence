# 06 — External benchmarks (amendment 07 + addenda B12–B16)

*Binding texts (copies in `HANDOVER/reference/`): `PREG1_AMENDMENT_07.md` (REGISTERED DL 341, sha `29beb5e3…`; sections
B1–B11, later sections supersede earlier ones), `PREG1_AMENDMENT_07_ADDENDUM_B12…B16.md`, and the procedure
`LOCK_FINAL_RUNBOOK.md`. Background: `company/plans/BENCHMARK_PLAN_V1.md`, `company/research/BM-00_PROTOCOL_AND_COMPARISONS.md`
(published numbers, evaluator rules), handoffs `BM-02.md` (runner), `BM-W2.md`, `BM-03.md`–`BM-05.md`, reviews
`REV-BM-02…05.md`, `FABLE_GATE_AMENDMENT_07.md`, `FABLE_AUDIT_BENCHMARKS_20261001.md`, `FABLE_DECISION_ML1M_TRACKB_20261003.md`,
`FABLE_DECISION_WAVE2_20261003.md`.*

## 1. Purpose and rules

- Show that the REES46 recipe transfers to standard public datasets, and compare against published numbers where the split
  is byte-identical. **Owner decision (DL 302): only the REES46 recipe is run** — no new method (eSASRec, PCTM, KD, text …
  are not run by us).
- Benchmark values are **DESCRIPTIVE** with respect to every PREG1 claim and never enter a REES46 selection, register or
  lock (R8). Retention ratios on leave-one-out data are not comparable in magnitude to REES46's.
- Claim rule against a published single run: a difference is stated only if |Δ NDCG@10| > 0.005 **and** every seed lies on
  the same side; otherwise "on par". Our arm-vs-arm contrasts: paired user bootstrap (1,000, seed 20260926) per seed, then
  the seed mean (R8).
- One LOCK and one TEST look (`--final`) per dataset; nothing is run, added or re-selected afterwards (B4, R7, B11).

## 2. Datasets (MEASURED counts)

| Wave | Dataset | Users / items / interactions | Status of the data |
|---|---|---|---|
| 1 | S3-Beauty | 22,363 / 12,101 / 198,502 | byte-identical to the published eSASRec release (DL 302) |
| 1 | S3-Sports | 35,598 / 18,357 / 296,337 | same |
| 1 | S3-Toys | 19,412 / 11,924 / 167,597 | same |
| 1 | MovieLens-1M | 6,040 / 3,706 / 1,000,209 | same; 2,483 users have timestamp ties (the first row in file order is the validation target) |
| 2 | Amazon 2023 Video_Games (official 5-core `last_out_w_his`) | 94,762 / 25,612 / 814,586 | converted by BM-W2; 69 cold holdout items count as misses |
| 2 | Amazon 2023 Baby_Products | 150,777 / 36,013 / 1,241,083 | converted; 59 cold holdout items |
| dropped | Amazon 2023 All_Beauty (253 users), Taobao full (1.7 M items), MBHT Tmall (782,694 items), MBHT Taobao (training-instance dump, not users), Amazon 2023 Beauty_and_Personal_Care (FL infeasible: 75,998 rounds) | — | DL 297, 325, 329, 377 |

Our copy of the Wave-1 data and protocol reproduced the published MC / SeqRules / PCTM NDCG@10 of Spotify Table 1 in 12 / 12
cells to 4 decimals (DL 303). Published comparison set (BM-00, VERIFIED, single runs, spread ≈ ±0.003): SAS+ NDCG@10 .0531 /
.0314 / .0580 / .1684 and eSAS .0523 / .0323 / .0538 / .1772 (Beauty / Sports / Toys / ML-1M); ML-1M HSTU .1719, FuXi-α
.1813. Wave 2 has no published numbers (ours-vs-ours only).

## 3. Protocol (B1, R2, B11)

Leave-one-out with the eSASRec construction: inner validation = the user's last train row (eSASRec code rule); training =
train.csv minus that row for every arm; TEST = the holdout row, scored from the full train history with no retrain;
full-catalogue ranking with seen-item filtering; ties by ascending item id. Scored users 22,311 / 35,539 / 19,365 / 6,038;
inner-validation populations 22,308 / 35,475 / 19,312 / 6,038. Primary metric NDCG@10; secondary HR@10, NDCG@20, HR@20,
MRR@20 (ours only). Populations: all holdout users, the 15 % band users and the 85 % others separately.

## 4. Arms, tracks, seeds (B2, B3, B7–B10, B12)

- **Arms (REES46 recipe):** C_FULL (SASREC_D256_B3, all users, 6.0 EFE); PRE (SASREC_D64_B2 on a seeded 15 % user band =
  the consenting-data assumption; θ_pre = endpoint 6.0); FA (cold); FA_WARM (S7 factors) and FA_WARM_FROZEN (S8 factors) from
  PRE; FT_FA_WARM\* (FT_C_f0.3_P4); FP / PF / FA_Q8 warm on FA_WARM\*; S_CAL → FA_1024 ×2 → DP8 ×2 (DP8_FROZEN if FA_WARM\* = S8,
  DP_WARM if S7). FL clients = **all users**; rounds T = ceil(60·N / (576·P)) = 630 / 2,330 / 2,023 / 3,709 (ML-1M / Beauty /
  Toys / Sports, P = 1); Poisson arms q = 2⁻⁷ exactly, T 384, δ 5e-6, z 0.5642, m = round(qN) = 47 / 175 / 152 / 278,
  P = 2 / B16. Pinned: FL_TUNING_SELECTION `9bca186e…`, S7 `d67a0a85…`, S8 `8b79d305…`, SELECTION_FT_C cell.
- **FA_WARM\*** = the variant with the higher inner-validation NDCG@10 (mean over seeds); ties within 0.0005 → S8.
- **B12 (DL 357):** S_CAL once **per PRE seed**; FA_1024 s / DP8 s bind their own seed's S record. Disclosed: the benchmark
  warm-arm seed spread includes the pretraining seed (REES46 has one θ_pre).
- **Track A (headline):** the recipe unchanged, seeds 2026 / 2027; a third seed 2028 for C_FULL and FA_WARM\* if the two
  seeds differ by > 0.002 NDCG@10 (decided before the LOCK).
- **Track B (Wave 1 only, secondary):** max_len {50; 200 on ML-1M} × dropout {0.1, 0.2, 0.5}, for C_FULL, PRE, FA and
  FA_WARM\*; central arms also over a budget axis {6, 24, 96} EFE (ML-1M C_FULL {6, 24} only); seed-2026 grid, the winner per
  arm re-run at 2027. Track A and Track B values are never mixed.
- Device class: CPU-FP32 (16 threads fixed, deterministic, nice 15) for S3 and Wave 2; **NVIDIA_A40-FP32-DET for ML-1M
  (B16)**. One device class per dataset for training **and** the TEST look; mixing is refused.

## 5. Status per dataset (as of DL 435)

| Dataset | Code tree / runs root | State |
|---|---|---|
| S3-Beauty, S3-Sports, S3-Toys | code_bench3 (Wave-1 driver) + code_bench4 (B12 resume) / `runs_bench/<ds>` | **every Wave-1 job done** incl. the s2027 DP re-runs (DL 369); decisions (FA_WARM_DECISION, SEED3_DECISION, Track-B winners) written by the runner; **no LOCK, no `--final` yet** |
| ML-1M | **code_bench5b**, driver `run_ml1m_gpu.sh` (`1a623877…`) / `runs_bench_gpu/ml_1m` | **restarted from scratch on both A40s at DL 433** (73 jobs, weight 75). The 46 CPU ML-1M dirs in `runs_bench/ml_1m` are sealed `SUPERSEDED_DEVICE`, never opened. Driver yields to any `laneP2*.sh` or H process. INFERENCE: final ≈ 10-05/06 |
| Video_Games, Baby_Products | code_bench4, `run_wave2.sh` (Track A only, 54 jobs, 3 slots × 16 threads) / `runs_bench/<ds>` | running on CPU since DL 377 (8 Video_Games results, 0 Baby at DL 414) |

History to disclose: B13 (DL 368) added a clock backstop for ML-1M Track B at 2026-10-06 00:00Z; **B15 (DL 401) rescinded
it** before the trigger (owner wanted the full grid; the stopper was disarmed at 15:26:18Z). Then **B16 (DL 433)** moved
ML-1M to the GPU after a Fable re-review requested by the owner (DL 404): timing gate MEASURED on smoke (C_FULL L200 B24
≈ 34 min on GPU vs ≈ 5.4 days on CPU; two-run determinism identical), BM-03 (`--device`), BM-04 (driver), BM-05 (`final
--device cuda`) all reviewed PASS_WITH_MINORS; hard switch with a relaunch guard. Disclosure sentence (B16.6): "device class
switched before any ML-1M value was read, on schedule grounds; the CPU runs are sealed SUPERSEDED_DEVICE and were never
opened".

## 6. How to lock and run the single TEST look per dataset

Follow `LOCK_FINAL_RUNBOOK.md` (§0 rules, §2 preconditions P1–P9, §3 value-free run list, §4 `lock`, §5 pre-final checks,
§6 `final`, §7 the TEST_ATTEMPTS rule). **Corrections the runbook needs before use (it was written 10-02, before B12 / B16):**
1. **Code tree.** The runbook says `cd $V/code_bench3`. After B12 the S3 LOCK must be written by the tree that knows
   per-seed S records (**code_bench4**; its `make_lock` requires S_CAL at both seeds and pins both records, DL 360–361).
   Verify `code_bench4/v4bench/*` against `company/handoffs/BM-02.sha256.txt` + the B12 delta before use.
2. **ML-1M** uses **code_bench5b** and its runs root `runs_bench_gpu`; `final` must run with `--device cuda --gpu N`
   (B16.4); the LOCK pins code_bench5b; never mix with the CPU dirs.
3. **Wave 2** locks follow B14.7 (code_bench4; cold holdout items are misses and the Wave-1 cold-count refusal does not
   apply).
4. Blocker §8.1 of the runbook (s2027 DP refusals) is solved by B12 (DL 369, 373).
5. Before each `final`: the REES46 K16 pass-2 latency bench must not be running (quiet CPU); the owner is told "single TEST
   look of <dataset> now"; afterwards read only the count keys first, write the DL row, then use values for the write-up.

These corrections are INFERENCE from the DL rows: **have an independent reviewer confirm the exact lock / final commands for
each tree before the first lock** (file 08 §6).

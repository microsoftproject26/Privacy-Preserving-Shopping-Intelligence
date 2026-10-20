# PREG1 amendment 07 — external benchmarks (DRAFT; becomes binding when renamed PREG1_AMENDMENT_07.md and logged in a DL row, before any benchmark run)

- **Status:** DRAFT, 2026-10-01 ~04:10Z, Operations Director. Owner decisions: DL 296, 302 and 303; BENCHMARK_PLAN_V1 §14.
- **No benchmark result exists.** The only benchmark numbers read so far are the published-baseline reproductions (DL 303),
  which are not our method.
- **REES46 is unaffected.** This amendment changes nothing in PREG1 / 06N / 06O or in any REES46 code path, lock or read.

## B1. Datasets and protocol (fixed)
- **Wave 1 — the eSASRec standard splits:**
  - S3-Beauty, S3-Sports, S3-Toys and MovieLens-1M;
  - built by spotify-research/sequential-capacity-probes @705710f7 with eSASRec @2b039927;
  - verified byte-identical with the published release (DL 302).
- **Protocol:**
  - leave-one-out; the holdout (one item per user) = TEST;
  - full-catalogue ranking, seen-item filtering, no sampled negatives (the eSASRec protocol).
- **Inner validation** = built exactly as the eSASRec code builds it, including its sort and timestamp-tie rule. ML-1M has
  timestamp ties for 2,483 users (BM-00), so the rule is the code, not "the last row". Every selection uses only this.
- **Evaluator rules (BM-00, verified):**
  - users scored = those with a holdout row: 22,311 / 35,539 / 19,365 / 6,038;
  - candidates = every train item minus the user's whole train history;
  - HR@10 = Recall@10; NDCG@10 = 1/log2(rank + 1) for rank ≤ 10;
  - ties between equal scores break by ascending item ID;
  - no retrain: train once, early-stop on inner validation, score TEST from the full train history.
- **Wave 2 (later, by a short addendum before its runs):** Amazon Reviews 2023 official splits and the MBHT Tmall / Taobao
  versions.
  - Amazon 2023: no published number exists on the official 5-core files (BM-00), so it is ours-vs-ours only.
  - MBHT: tmall_beh = Taobao and ijcai_beh = Tmall. MBHT ranks against 100 fixed popular items, so its numbers are sampled and
    context only.

## B2. Systems (the REES46 recipe only; no new method)
| Arm | Definition |
|---|---|
| C_FULL | central SASREC_D256_B3, all users |
| PRE | central SASREC_D64_B2 on the TRAIN part of a seeded 15 % user band (seed 2026), server-side, consenting-data assumption |
| FA | cold FL, D64_B2 |
| FA_WARM | S7 factors from FL_WARM_SELECTION_S7.json, from PRE |
| FA_WARM_FROZEN | S8 factors from FL_WARM_SELECTION_S8.json, frozen item tables, from PRE |
| FT_FA_WARM* | the FT_C / SELECTION_FT_C recipe on the FA_WARM* base |
| Phase 2 | FP / PF / FA_Q8 (warm on FA_WARM*), DP8 — see B7 R4 (S_CAL on the FL clients = all users, the REES46 rule; NOT on the band) |
- **FL tuning:** FL_TUNING_SELECTION.json (FedAdam η_s 0.01, τ 1e-3, FL_CONST, client SGD-m 0.9 lr 0.1, 1 local pass, B16).
- **Rounds:** the REES46 exposure rule (fa_rounds on the dataset's client count).
- **FA_WARM\*:** the variant (S7 or S8) with the higher inner-validation NDCG@10 at PRACTICAL_BEST, mean over seeds; ties
  within 0.0005 → S8. This replaces REES46's CALIB, which has no counterpart here.

## B3. Tracks
- **Track A (headline):** the recipe above, unchanged on every dataset. Claim: transfer without retuning.
- **Track B (secondary, Wave-1 small datasets):** a validation-only grid of max_len {50, 200 (ML-1M only)} × dropout
  {0.1, 0.2, 0.5}, applied identically to C_FULL, PRE and the FL arms. Its winner is chosen per dataset on inner validation.
  Claim: the best each system reaches. Track A and Track B claims are never mixed.

## B4. Metrics, seeds, selection, TEST look
- **Primary:** full-catalogue NDCG@10.
- **Secondary:** HR@10, NDCG@20, HR@20, MRR@20.
- **Populations:** all users; separately the 15 % band users and the 85 % others.
- **Seeds:** 2026 and 2027 for every arm; a third seed (2028) for the Track-A FA_WARM* and C_FULL if the seed gap exceeds 0.002
  NDCG@10.
- **Checkpoints:** PRACTICAL_BEST (best inner-validation NDCG@10 over marks) and CONTROLLED_BUDGET (the exposure endpoint).
- **Locks and TEST:**
  - per dataset, `runs_bench/<dataset>/LOCK.json` lists every run and checkpoint (sha256) after all runs finish;
  - then ONE `--final` holdout evaluation per dataset;
  - no change after the TEST look.

## B5. Comparisons
- Published numbers are compared directly only on the byte-identical eSASRec splits: Spotify Table 1 (MC, FMC, FMC+, SAS+,
  eSAS, SeqRules, PCTM) and the eSASRec paper's tables.
  - Our CPU reproduction of MC / SeqRules / PCTM matched 12 / 12 (DL 303).
- Directly comparable (BM-00, VERIFIED, single runs; run-to-run spread ≈ ±0.003):
  - Spotify Table 1;
  - the eSASRec README runs: SAS+ .0531 / .0314 / .0580 / .1684 and eSAS .0523 / .0323 / .0538 / .1772 NDCG@10
    (Beauty / Sports / Toys / ML-1M), with Recall@10;
  - ML-1M only: HSTU .1719, FuXi-α .1813.
- Numbers on other versions of the same dataset name are context only. This covers all seven FL / on-device papers checked;
  none uses these splits.
- No other method is run by us.

## B6. Device and disclosure
- Benchmarks run on CPU (registered device class CPU-FP32, deterministic, nice 15). REES46 stays on its A40 pin.
- **Deviations from REES46, disclosed:**
  - leave-one-out instead of temporal multi-decision evaluation;
  - ID-only inputs;
  - the 15 % band replaces REES46's non-MAIN pretraining band;
  - inner validation replaces CALIB.
- **MEASURED and reported:** on S3-Beauty a 15 % band sees 66.9 % of items and 74.8 % of test targets (DL 301). S8's frozen
  tables are therefore expected to be at a disadvantage there. The registered N2-type rule above decides FA_WARM\*, not this
  expectation.

## B7. Gate fixes (Fable REGISTER_WITH_FIXES, company/reviews/FABLE_GATE_AMENDMENT_07.md) — these SUPERSEDE any conflicting line in B1–B6

**R1. Who the FL clients are, and the rounds rule written out.**
- **FL clients = ALL users of the dataset** (band users keep a device; their consent covers the server-side PRE copy only). The band is the same set for every run seed (band seed 2026 ≠ run seed). **Rounds:** the REES46 exposure rule `launch_fl.fa_rounds(N_clients, P)` = ceil(60 · N / (576 · P)) with the FL_TUNING_SELECTION local-pass count P, 64 clients per round, sweep sampling, 10 % seeded drop-out — i.e. T = 630 (ML-1M, N 6,040), 2,330 (Beauty, 22,363), 2,023 (Toys, 19,412), 3,709 (Sports, 35,598) for P = 1; the 12 evaluation marks fall at every 0.5 EFE (round ceil(k · T / 12)). The exact T per dataset is written in each dataset manifest before the first run.

**R2. Training data vs inner validation for every arm.**
- **Every training set is train.csv minus the user's inner-validation row:** C_FULL (all users), PRE (band users), every FL client's local data, and the FT_FA_WARM\* on-device fine-tune. Inner-validation candidates = the train.csv catalogue minus the user's inner-train items. **TEST inference** uses the full train.csv history (including the inner-validation item), no retrain, as eSASRec does (BM-00 §2.2). The ML-1M tie rule is stated literally: rows are ranked by datetime **descending with a stable sort over file order**; the rank-0 row is the validation target, so among rows tied at the user's maximum timestamp the **first row in file order** is the target (2,483 users differ from "last row"; BM-00). Before the first run, the dataset manifest records that our construction equals eSASRec's target for 6,040 / 6,040 ML-1M users under our pandas, and the inner-validation populations (22,308 / 35,475 / 19,312 / 6,039). We rank the **full train.csv catalogue** (12,092 / 18,347 / 11,917 / 3,704) for every arm and do not truncate it; published SAS+ / eSAS could not reach 89 / 265 / 232 / 0 holdout targets (BM-00 §2.3) — disclosed beside every direct comparison.

**R3. Central budget, θ_pre and PRE's selection population.**
| C_FULL | central SASREC_D256_B3 on all users; the registered central recipe (S0 schedule on the 6.0-EFE clock, registered peak LR, AdamW, grad-clip), 6.0 EFE, 12 marks; PRACTICAL_BEST selected on ALL users' inner-validation NDCG@10 |
| PRE | central SASREC_D64_B2 (the registered SMALL recipe), same clock, trained on the band users' train rows minus their inner-validation row; **θ_pre = the CONTROLLED_BUDGET (endpoint 6.0) checkpoint, no selection** (06K §1); PRE's own PRACTICAL_BEST is selected on the **band** users' inner validation only (the server has only their consent) and is reported as the PRE-alone row on all / band / others |

**R4. DP8 parameters and S calibration.**
DP8: the REES46 mechanism unchanged — Poisson sampling with **q = 2⁻⁷, T = 384, δ = 5e-6, z = 0.5642** (identical (q, T, δ), so the registered accountant value is exact), fixed denominator **m = round(q · N)** = 47 (ML-1M) / 175 (Beauty) / 152 (Toys) / 278 (Sports); warm start from FA_WARM\*'s θ_pre as in 06K §3 when FA_WARM\* = S8 (noise dimension = the uploaded parameter count). **S** = the REES46 rule (S_RECORD: median client-delta L2 norm over rounds 1–20 of a z = 0 S_CAL run with the same configuration on the FL clients; non-private, disclosed as in 06 §2 Q3), written to `runs_bench/<dataset>/dp/S_RECORD.json` before the DP8 run. Disclosed: per-round noise ≈ z · S · √d / m is 1,024 / m ≈ 4–22 × REES46's; DP8 is reported at CONTROLLED_BUDGET as its headline (06D §3), PRACTICAL_BEST secondary.

**R5. Track B made executable.**
- **Track B (secondary, Wave-1 only):** grid max_len {50; 200 on ML-1M only} × dropout {0.1, 0.2, 0.5} (3 cells; 6 on ML-1M). Arms: C_FULL, PRE, FA, and FA_WARM\* (the variant decided in Track A; not re-decided). Grid cells run with seed 2026 only; **the winner is chosen per dataset and per arm** on inner-validation NDCG@10 at PRACTICAL_BEST (ties within 0.0005 → the Track-A cell, then smaller max_len, then larger dropout); the winning cell of each arm is then run with seed 2027. A warm cell uses the PRE trained in the same (max_len, dropout) cell. Track A and Track B values are never mixed, pooled or compared against each other's published row.

**R6. Third-seed trigger and the 3-seed rule.**
- **Seeds:** 2026 and 2027 for every arm. A third seed (2028) is run for Track-A C_FULL and FA_WARM\* on a dataset when the two seeds' **inner-validation PRACTICAL_BEST NDCG@10 (all users)** differ by more than 0.002; the decision is taken and logged (DL row) before that dataset's LOCK.json is written, never after a TEST look. Where a third seed exists the reported value is the 3-seed mean (all seeds listed).

**R7. Populations and the LOCK / TEST gate.**
- **Populations:** all holdout users (= the published denominators 22,311 / 35,539 / 19,365 / 6,038); separately the band ∩ holdout and the others ∩ holdout, with their counts printed. The 52 / 59 / 47 / 2 users without a holdout row stay in training and inner validation only.
- **Locks and TEST:** per dataset, after every Track-A and Track-B run (and any third seed) has a RUN_SUMMARY, `runs_bench/<dataset>/LOCK.json` lists every run, both checkpoints (sha256), the FA_WARM\* decision, the Track-B winners and the code-tree digest; then ONE `--final` invocation scores every locked checkpoint for every population and metric in a single pass (holdout users only); nothing is run, added or re-selected afterwards. Wave 2's addendum cannot touch a Wave-1 lock.

**R8. Claim strength against published single runs, and the REES46 firewall.**
- **Claim rule:** our value = the 2-seed (or 3-seed) mean with both seed values printed. Against a published single-run cell, a difference is stated only when |Δ NDCG@10| > 0.005 (the Spotify SAS+ / eSAS acceptance tolerance; the observed reproduction spread is ≈ ±0.003) **and** every seed lies on the same side; otherwise "on par". No significance test is computed against a published number (no paired data). Our own arm-vs-arm contrasts use the REES46 unit: paired user bootstrap (1,000, seed 20260926) per seed, then the seed mean. Benchmark retention ratios (LOO, one query per user, ID-only) are **not comparable in magnitude** to REES46's temporal multi-decision K-contrasts; they are DESCRIPTIVE with respect to every PREG1 confirmatory claim, and no benchmark value enters any REES46 selection, register or lock.

**R9. Pins and device class.**
- **Pinned inputs (sha256 in this file and in every LOCK):** FL_TUNING_SELECTION.json (S4, 9bca186e…), FL_WARM_SELECTION_S7.json (d67a0a85…), FL_WARM_SELECTION_S8.json (8b79d305…), SELECTION_FT_C (cell FT_C_f0.3_P4), the r6 code tree digest; the item vocabulary of every arm = the full train.csv catalogue; S8's frozen rows for items absent from the band's inner-train stay at θ_pre's values (disclosed). Band coverage (items seen / validation targets covered) is MEASURED for all four datasets (data manifest) before the first run; today only Beauty and ML-1M are measured.
- **Device class CPU-FP32:** torch deterministic algorithms, **16 threads fixed** (reduction order changes with the thread count), nice 15; FA_Q8 stochastic rounding keeps its seeded generator. REES46 stays on the A40 pin; `code/v4bench` imports v4models / v4train / code/fl read-only and no REES46 file, lock or run directory is written.

**R3 (B6 addition).** the central budget is 6.0 EFE (≈ 6 passes) vs eSASRec's up-to-200 epochs with patience 50; this is the Track-A 'no retuning' condition and is disclosed beside C_FULL.

**R9 coverage (MEASURED 10-01, on the eSASRec splits).** Band = seeded 15 % of users (numpy default_rng(2026)); inner-train
items only. The runner's own band file is the binding one and re-measures these figures before the first run.
| Dataset | Items seen | Holdout targets covered |
|---|---|---|
| S3-Beauty | 67.2 % | 75.2 % |
| S3-Sports | 66.0 % | 74.5 % |
| S3-Toys | 65.9 % | 71.9 % |
| ML-1M | 92.7 % | 99.8 % |

## B8. Optional items adopted (Ops + Fable concurrence, DL 318). B8 supersedes B3 / R5 where they conflict.
- **O1 — exposure-budget axis, Track B, central arms only.**
  - Budgets {6, 24, 96} EFE, crossed with the R5 grid. Applies to C_FULL and PRE only; the FL arms keep the R1 exposure rule.
  - The S0 schedule is stretched to each cell's clock.
  - C_FULL early-stops after 6 marks without an inner-validation improvement. PRE never early-stops (θ_pre = its endpoint).
  - Tie order: the Track-A cell → smaller budget → smaller max_len → larger dropout.
  - Disclosed: this budget asymmetry favours the central arms. Track-A retention ratios use the 6-EFE ceiling and are labelled so.
  - R3 / R7 wording is read with this axis; R1, R2, R4, R6, R8 and R9 are unchanged.
- **O2.** On every dataset, FA_1024 (the z = 0 clipped twin: same q / T / m, 2 seeds) runs beside DP8. Chain: S_CAL → S_RECORD →
  FA_1024 ×2 → DP8 ×2.
- **O3.** A DESCRIPTIVE sub-population row "holdout target covered by the band" for S7 vs S8 on every dataset, which separates
  coverage from freezing.
- **O4.** HSTU / FuXi-α Amazon rows are context only (split not verified; BM-00 §1b).
- **O5.** MRR@20 appears only in our own tables; it has no published counterpart.

## B9. Fable benchmark-audit fixes (company/reviews/FABLE_AUDIT_BENCHMARKS_20261001.md; DL 329). Supersedes conflicting lines above.
- **M1a.** DP8's S calibration is defined only by R4 (S_CAL on the FL clients, i.e. all users; the REES46 rule). The B2 "on the
  band" wording is withdrawn.
- **M1b.** A Track-B warm cell (FA_WARM*) starts from the **6-EFE PRE of the same (max_len, dropout) cell**. The 24 / 96-EFE PREs
  are PRE-alone rows only.
- **M1c.** The Poisson arms (FA_1024, DP8, S_CAL) keep the DL 219 rule: P = 2 local passes, B16. The sweep arms use FL_TUNING_SELECTION
  P = 1.
- **M1d (minor).** If FA_WARM* = S7 on a dataset, its DP row is DP_WARM (not frozen), with S from that variant's S_CAL_WARM, as in
  FL-16. If FA_WARM* = S8, it is DP8_FROZEN as in B2 / R4.
- **M2.** mbht_taobao is DROPPED from Wave 2. Its "sessions" are MBHT training-instance dumps (108,437 exact duplicate rows; about
  407 k heads vs 147,892 paper users; MEASURED by Fable), not users, so they cannot be FL clients. mbht_tmall was already dropped
  (DL 325).
- **M3.** Wave-2 FL feasibility must be measured before the Wave-2 addendum. R1 gives T = 9,872 / 15,707 / 75,998 rounds for
  Video_Games / Baby_Products / Beauty_and_Personal_Care; the addendum states the measured time or a changed client rule.
- **Minors:**
  - leave-one-out leakage is disclosed (inner validation is a train item);
  - the BENCHMARK_PLAN §14 "RICH" applies to Wave 2 only (Wave 1 is ID-only, DL 324);
  - Wave-2 cold holdout items count as misses, never silently dropped;
  - the B6 coverage figure is superseded by the R9 table;
  - K&R is cited as Krichene & Rendle (KDD 2020), with the sampled_eval hypergeometric form as its without-replacement analogue.

## B10. Runner questions resolved (Ops + Fable; reviews/FABLE_GATE_AMENDMENT_07.md thread; DL 333). Supersedes conflicting lines above.
- **Q1.** In every Poisson arm (S_CAL, FA_1024, DP8 and the warm variants) each client is included independently with probability
  **exactly q = 2⁻⁷**. m = round(q·N) is used only as the fixed denominator. With (q, T, δ) = (2⁻⁷, 384, 5e-6), z = 0.5642 is the
  exact accountant value. R4 is read this way. Sampling at m/N is not allowed.
- **Q2.** PF is implemented and run as in REES46: the 06E θ plus the per-client store checkpoint, warm on FA_WARM*. It is a phase-2
  arm with 2 seeds.
- **Q3.** The on-device fine-tune (FT) trains on train minus the inner-validation row. The validation row is TEST-time context only,
  inside the full train history.
- **Q4.**
  - The ML-1M C_FULL Track-B budget axis is {6, 24} EFE only; the 96-EFE cell is not run.
  - Any run that hits a wall cap is INCOMPLETE, reported as such, and NOT eligible as a Track-B winner.
  - Every other dataset / arm keeps {6, 24, 96}.
- **Q5.** Cold holdout items count as misses in the denominator. Wave 1 has none: the evaluator refuses a Wave-1 --final if the
  cold count is non-zero.

## B11. Correction before registration (REV-BM-02 delta review; DL 338)
- The ML-1M inner-validation population is **6,038**, not 6,039 (R2). eSASRec / RecTools keep the last session_max_len + 1 = 201
  inner events (RecTools data_preparator `_filter_train_interactions`; eSASRec ml_1m.yaml session_max_len 200).
- Beauty / Sports / Toys keep tail 51 → 22,308 / 35,475 / 19,312, unchanged.
- Every TEST attempt (start / end / error) is appended to runs_bench/<dataset>/TEST_ATTEMPTS.jsonl. The single TEST look is the one
  whose TEST_LOOK.json is written. A crashed attempt is disclosed and may be retried only with the identical LOCK.

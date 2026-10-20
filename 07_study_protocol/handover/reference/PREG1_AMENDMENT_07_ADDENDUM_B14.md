# PREG1 amendment 07 — addendum §B14: Wave 2 (REGISTERED 2026-10-03, DL 377; Fable decision company/reviews/FABLE_DECISION_WAVE2_20261003.md)

- **Status:** registered before any Wave-2 run. No benchmark holdout / TEST value and no REES46 MAIN / VALIDATION / TEST value has
  been read. Amendment 07 B1–B11 and addenda B12 / B13 are unchanged; B14 specialises them to Wave 2 and supersedes only where it
  says so. Wave 2 cannot touch a Wave-1 lock (R7). REES46 is unaffected.

**B14.1 Datasets (fixed).** Wave 2 = Amazon Reviews 2023 official 5-core `last_out_w_his` splits for **Video_Games** and
**Baby_Products** only, as converted by BM-W2 into `bench_proc_w2/amazon2023_video_games` (94,762 users / 25,612 items / 814,586
interactions; holdout.csv 45212128e455badf…, train.csv 7e616074de56e97c…, statistics.csv 426d4fa56d5f49b6…) and
`bench_proc_w2/amazon2023_baby_products` (150,777 / 36,013 / 1,241,083; 43c0ce1ad1c027fb…, 9445cbbe995d7269…, d897c1357eba3d03…);
the full sha256 values in each dataset's SHA256SUMS are pinned in its LOCK. train.csv = official train + official valid rows;
holdout.csv = the official test row (one per user, all users). **Beauty_and_Personal_Care is DROPPED** (B9 M3 MEASURED: FA 20 rounds
1,164 s and C_FULL 0.1 epoch 2,986 s on 16 threads; T = 75,998 rounds → weeks per FL run); its converted files and timing stay as
evidence; it returns only by a new addendum registered before any run on it. mbht_tmall (DL 325) and mbht_taobao (B9 M2) stay
dropped. The `timestamp_w_his` variant is not run (not one holdout per user). The Wave-2 "RICH" variant (B9 minor) is **not run**;
Wave 2 is ID-only like Wave 1; the rating column is ignored.

**B14.2 Protocol.** Leave-one-out exactly as B1 / R2: inner-validation target = the user's last train.csv row under the registered
rule (datetime descending, stable over file order) = the official valid row; training = train.csv minus that row for every arm;
TEST inference from the full train.csv history, no retrain; inner-validation candidates = the train.csv catalogue minus the user's
inner-train items; full-catalogue ranking (25,612 / 36,013), seen-item filtering, ascending-ID ties. Before the first run of a
dataset its manifest records: the inner-validation population and the check that our validation row equals the official valid row
for every user (Video_Games MEASURED equal, DL 363; Baby_Products to be recorded), N FL clients, T, m, and the R9 band coverage
(items seen / holdout targets covered). **Cold holdout items** (items absent from train.csv: 69 / 59; BM-W2) count as misses in
every metric's denominator (B9 minor, B10 Q5); their count and the affected user count are printed in the manifest, the LOCK and the
TEST_LOOK; the Wave-1 rule "a --final refuses a non-zero cold count" does **not** apply to Wave 2.

**B14.3 Scope and arms.** **Track A only** (B3; Track B is Wave 1 only; no grid, no budget axis, no tuning of any kind). Arms as
B2 / B7 R3–R4 / B8 O2 / B9 M1c–M1d / B10 Q2 / B12: C_FULL (SASREC_D256_B3, all users, 6.0 EFE, 12 marks, PRACTICAL_BEST on all users'
inner validation); PRE (SASREC_D64_B2 on the seeded 15 % band, band seed 2026, θ_pre = the 6.0-EFE endpoint, PRE-alone row selected
on the band); FA; FA_WARM (S7) and FA_WARM_FROZEN (S8) with the B2 FA_WARM\* rule; FT_FA_WARM\* (FT_C_f0.3_P4); FP; PF; FA_Q8 warm on
FA_WARM\*; S_CAL per PRE seed (B12); FA_1024 ×2; DP8 ×2 (DP8_FROZEN if FA_WARM\* = S8, DP_WARM if S7; M1d). **FL clients = all users**;
T = `fa_rounds(N, P = 1)` as written in the manifest (Video_Games 9,872 MEASURED; Baby_Products' manifest value is binding, M3
estimated 15,707); 64 clients per round, sweep sampling, 10 % seeded drop-out; 12 marks at ceil(k·T/12). Poisson arms: q = 2⁻⁷
exactly (B10 Q1), T = 384, δ = 5e-6, z = 0.5642, P = 2, B16, **m = round(q·N) = 740 (Video_Games) / 1,178 (Baby_Products)**; S =
the R4 / B12 rule per PRE seed, written to `runs_bench/<dataset>/dp/S_RECORD_<VARIANT>_s<S>.json` before the matching DP run.
Disclosed: per-round noise ≈ z·S·√d/m is 1,024/m ≈ 1.4× (Video_Games) and ≈ 0.9× (Baby_Products) REES46's — closer to REES46 than
any Wave-1 dataset. Pins as R9 (FL_TUNING_SELECTION 9bca186e…, S7 d67a0a85…, S8 8b79d305…, SELECTION_FT_C cell FT_C_f0.3_P4).

**B14.4 Seeds, selection, populations, metrics.** Seeds 2026 and 2027 for every arm; the R6 third-seed rule (2028 for C_FULL and
FA_WARM\* when the two seeds' inner-validation PRACTICAL_BEST NDCG@10 on all users differ by > 0.002; decided and logged before the
dataset's LOCK). Checkpoints PRACTICAL_BEST and CONTROLLED_BUDGET (DP8 headline at CONTROLLED_BUDGET, R4). Metrics as B4; MRR@20 in
our tables only (O5). Populations: all holdout users (94,762 / 150,777, cold-item users included as misses), band ∩ holdout,
others ∩ holdout, and the O3 descriptive "holdout target covered by the band" row for S7 vs S8.

**B14.5 Comparisons and claims.** No published number exists on these files: Wave 2 is **ours-vs-ours only**. No published row, no
"on par" statement, no external tolerance test. Arm-vs-arm contrasts use the R8 unit (paired user bootstrap, 1,000 resamples, seed
20260926, per seed, then the seed mean) and are DESCRIPTIVE with respect to every PREG1 confirmatory claim; retention ratios carry
the 6-EFE-ceiling label (O1 disclosure) and are never pooled with Wave 1 or REES46. HSTU / FuXi-α Amazon rows stay context only (O4).

**B14.6 Device, runner, scheduling.** Device class **CPU-FP32 exactly as B6 / R9** (torch deterministic, 16 threads fixed, nice 15),
runner **code_bench4 (the B12 tree) unchanged**, its code-tree digest pinned in each LOCK; any runner code change for Wave 2 needs
a delta review before the first run that uses it. The device class of a dataset is fixed at its first run; no device switch
inside a dataset; **no GPU for Wave 2** (a GPU class could be registered only by a new addendum for a dataset with no run yet).
Driver: the registered Wave-1 driver pattern restricted to `amazon2023_video_games amazon2023_baby_products`, Track A only; its
plan (rc 0, job list) is logged in the registering DL row before launch; a shell-only driver change is verified by Ops (plan mode),
as DL 341 did; one driver per dataset (DL 360); separate state root `runs_bench/<dataset>` under the Wave-1 layout. **Slots: 3 × 16
threads now**; raised to at most 6 only after the ML-1M code_bench3 driver and laneBENCH_RESUME have exited. **Precedence:** (i) the
REES46 endgame (FA_Q8, D–H) is never delayed by Wave 2; (ii) **K16 INT8 pass 2 (DL 368) has precedence over Wave 2**: when every
Wave-1 driver has exited, Ops drains Wave 2 (no new job launched; running jobs may be SIGSTOPped for the pass-2 window and
SIGCONTed after — this changes wall time only, never a value) and the pass-2 host load is written beside its result as pass 1 did;
(iii) Wave 2 never delays a Wave-1 LOCK or `--final`.

**B14.7 Locks, TEST, stopping.** Per dataset, after every Track-A run (and any third seed) has a RUN_SUMMARY: `runs_bench/<dataset>/
LOCK.json` (every run, both checkpoints, the FA_WARM\* decision, both S records, the dataset shas, the manifest, the code digest);
then ONE `--final` (R7, B11: TEST_ATTEMPTS.jsonl append-only; TEST_LOOK.json atomic; a crashed attempt is retried only with the
identical LOCK). Nothing is run, added or re-selected after the look. **Stopping rule (clock, never value):** if the thesis schedule
requires it, Ops may stop a Wave-2 dataset by a DL row **before its LOCK**; a stopped dataset has no LOCK and no TEST look and is
reported as "converted, not run" with the measured reason — no partial Track-A table is ever locked. Nothing else changes.

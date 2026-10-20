# Fable audit — PPSI V4 external-benchmark track (DL 296–325)

- **Auditor:** Claude Fable 5.1, read-only, web research allowed. 2026-10-02. No Codex, no git, no file edited except this one.
- **Evidence classes:** MEASURED (I computed it on the server, read-only), VERIFIED (I read the primary source), INFERENCE.
- **Verdict:** no BLOCKER. 3 MAJOR, 7 MINOR. Amendment 07 is registrable after the MAJOR cleanups M1–M3 below.

## Summary table

| # | Item | Verdict |
|---|---|---|
| 1 | Dataset choices and drops | OK, with M3 (Wave-2 round counts) and m1–m2 |
| 2a | Wave-1 integrity | OK (MEASURED 8 / 8 file hashes = repo manifest; 12 / 12 reproduction values = Table 1) |
| 2b | Wave-2 integrity | OK (counts re-read from STATS.json = BM-W2); m3 cold holdout items |
| 2c | MBHT pseudo-user ≈ 2× | DIAGNOSED: duplicated + overlapping instance rows, not users → M2 |
| 3 | Amendment 07 B1–B8 consistency | ISSUE MAJOR M1 (B2 vs R4 S-calibration population; O1 vs R5 warm-cell PRE); m4–m5 |
| 4 | BM-00 spot checks + eSASRec attribution | OK (all checked numbers VERIFIED; "Tikhonovich 2025" correct) |
| 5 | Tooling (site dir, CPU plan, sampled_eval math) | OK; m6 |
| 6 | Thesis chapter 4 benchmark facts | ISSUE MINOR m7 (repeats the B2 "band" S-calibration wording) |

## MAJOR

**M1 — Amendment 07 internal conflict (B2 vs R4; O1 vs R5) — fix before registration.**
- B2 (Phase 2 row): "DP8 … S calibrated on the band". R4: S = median client-delta L2 norm over rounds 1–20 of a z = 0 S_CAL run "on the FL clients" (= all users). REES46's S_CAL ran on MAIN, the FL population (DL rows on S_CAL_T (MAIN)), so R4 is the faithful port and B2 is wrong. B7 says it supersedes, but B2 still reads the other way and chapter 4 §4.4.6 copied the B2 wording. **Fix:** strike "S calibrated on the band" in B2, write "S from a z = 0 S_CAL run on the FL clients (all users), R4"; same in chapter 4 line "S calibrated on the band with the REES46 rule".
- O1 adds budgets {6, 24, 96} EFE for PRE in Track B; R5 says "a warm cell uses the PRE trained in the same (max_len, dropout) cell"; R3 says θ_pre = the 6.0-EFE endpoint. With O1 there are three PREs per cell and the amendment does not say which one seeds the warm FA cell. **Fix (one sentence in B8 O1):** "Warm Track-B cells always start from the 6-EFE PRE of the same (max_len, dropout) cell; the 24 / 96-EFE PREs are PRE-alone rows only." (Any other choice must be stated, since it changes the FL arms' budget.)
- R4 omits that the Poisson arms (S_CAL, FA_1024, DP8) keep local passes 2 / batch 16 by DL 219 while the sweep arms use P = 1; code/v4bench/recipe.py implements it (`poisson_local_rule`), the amendment does not state it. **Fix:** add the DL 219 sentence to R4.

**M2 — MBHT "Taobao" (tmall_beh) is not a user population (MEASURED on `MBHT_dataset/tmall_beh.train.inter`).**
- 462,945 rows / 437,186 session_ids (mean 1.06 rows per id, so the "prefix rows share a session_id" model in mbht.py covers only 6 % of the augmentation).
- **108,437 of the 437,186 longest rows are exact duplicates** (identical history + target; 328,749 unique sequences; events 14.11 M → 10.36 M after exact de-duplication vs the paper's 7.09 M).
- 186,103 rows share their first-20-item prefix with another row; 29,708 rows' history equals another row's history + target (same real user, consecutive purchase targets, different session_ids); 407,478 "chain heads" vs 147,892 paper users.
- Test file: 0 test targets leak into train (test history + target never equals a train row); 348 test histories equal a train row's full sequence (the test user's earlier purchase is a train instance), 13,850 train rows share a test row's first-20 prefix.
- **Diagnosis:** the RecBole file is MBHT's training-instance dump (one row per purchase target, duplicated), with ids that are instance ids. The ≈ 2× events = duplicates + overlapping sub-sequences of the same user. Real users cannot be recovered from these files (BM-00 is right), so **a session cannot be a federated client** (one real user would sit on several devices, 25 % of clients are exact copies, and the PRE band / "others" populations are undefined).
- **Fix:** keep mbht_taobao OUT of any FL arm; at most a central sampled-metric context row, labelled "instance file, duplicates kept as published". If Taobao is wanted as an FL dataset, rebuild from the raw UserBehavior (bench_raw/Taobao, official counts) with a documented user-level filter; that is a new Wave-2 decision, not a converter fix. Record item_type 4 as "target marker written by the MBHT dump" (463,690 occurrences = 437,186 train targets + 9,999 holdout + 16,505 in-list), not as a behaviour.

**M3 — Wave-2 Amazon 2023 round counts are not CPU-feasible under R1 as written (INFERENCE from R1).**
- T = ceil(60 N / 576) with FL clients = all users gives Video_Games 9,872, Baby_Products 15,707, **Beauty_and_Personal_Care 75,998 rounds** per FL run (× arms × 2 seeds), vs 630–3,709 on Wave 1. The Wave-2 addendum must either measure the CPU round time first or register a different client rule (e.g. a seeded client subsample) before Beauty_and_Personal_Care is promised. Not a Wave-1 problem.

## MINOR

- **m1 (item 1, missed disclosure).** Leave-one-out shares users across train and test and is known to leak future information at the population level (Ji et al. 2023, "A critical study on data leakage in recommender system offline evaluation"); REES46's temporal protocol avoids it. State this once as the reason the benchmarks are "comparison only" (chapter 4 §4.4.1).
- **m2 (item 1).** BENCHMARK_PLAN §14 keeps "ID-only + RICH"; amendment B6 is ID-only and DL 324 shows the eSASRec splits carry item ids only. Write "RICH not applicable on Wave 1 (no metadata in the release splits)" into B6 so §14 and B6 stop disagreeing.
- **m3 (item 2b).** Cold holdout items (69 / 59 / 143 Amazon, 1 / 98 MBHT) are dropped by the Wave-1 adapter (BM-W2 open item 4). For Wave 2 the denominators must stay "all holdout users" with cold targets scored as misses, or the count must be disclosed next to every number; register one choice in the addendum.
- **m4 (item 3).** B6 quotes Beauty coverage 66.9 % / 74.8 % (bench_proc 5-core band, DL 301) while R9 measured 67.2 % / 75.2 % on the eSASRec split. Mark the B6 figure superseded.
- **m5 (item 3).** R4 defines the DP8 warm start only "when FA_WARM* = S8". State the S7 case (θ_pre with trainable tables; noise dimension = full uploaded vector) or "DP8 cold if S7 wins".
- **m6 (item 5).** sampled_eval.py `expected` mode is correct for negatives drawn **without replacement** (hypergeometric: M = candidates − 1, s = rank − 1, N draws; HR@k = P(X ≤ k−1), NDCG@k = Σ_{j<k} P(X=j)/log2(j+2), MRR = Σ P(X=j)/(j+1), checked against the code lines 194–215). Krichene & Rendle's closed forms use sampling with replacement (binomial); cite it as "the exact without-replacement analogue of Krichene & Rendle 2020" so the label is not challenged. `draw_fixed` can draw the target or a duplicate as a replacement (MBHT does the same); already disclosed in the docstring, keep it in the output JSON.
- **m7 (item 6).** Chapter 4 §4.4.6 "S calibrated on the band" → see M1. All other benchmark facts in §4.4 re-checked and correct (counts, holdout denominators, train catalogues, inner-validation populations, tie rule, T values, m values, Table 1 cells, README cells, drops table).

## OK items with evidence

- **Item 1.** Wave 1 is the right choice for a thesis on central vs federated vs on-device: it is the only public release with byte-verifiable splits, full-catalogue published numbers for 7 systems and a 2026 independent reproduction; users have ≥ 5 events so every user is a plausible device. Drops are justified: Taobao full (1,708,530 items, 875 MB tables), mbht_tmall (782,694 items), All_Beauty (253 users), Tmall-EODRec (sessions, no users), Amazon 2023 timestamp variant (0–818 test rows per user, MEASURED by BM-W2; a one-per-user version would be a new protocol). Deferral of ML-20M is fine.
- **Item 2a (MEASURED).** `data/manifest.json["processed"]` sha256 = on-disk files for all 8 Wave-1 csvs (rows 176,139 / 22,311; 260,739 / 35,539; 148,185 / 19,365; 994,169 / 6,038). `results_cpu/*/{mc,seqrules,pctm}.json` official `ndcg@10` = Spotify Table 1 in 12 / 12 cells; the repo checkout is clean except the four untracked CPU-run paths (matches DL 303).
- **Item 2b (MEASURED).** bench_proc_w2 STATS.json: VG 94,762 / 25,612 / 814,586 (train 719,824, holdout 94,762); Baby 150,777 / 36,013 / 1,241,083; Beauty_and_Personal_Care 729,576 / 207,649 / 6,624,441; mbht_taobao 447,185 / 99,037 / 14,622,365; mbht_tmall 453,732 / 782,694 / 28,855,872 — equal to BM-W2 and DL 325. Leave-one-out construction (train = official train + valid, holdout = official test, one row per user, no holdout item in the user's train) is correct in amazon2023.py.
- **Item 4 (VERIFIED on the web today).** Spotify Table 1 (arXiv 2608.19833): all 28 Wave-1 cells and the caption "All values are our matched-protocol runs" equal BM-00 §1a; authors Petrov, Chandar, Bennett, Bouchard, Lalmas (Spotify), RecSys 2026. eSASRec README (blondered/transformer_benchmark): SAS+ .0531 / .0314 / .0580 / .1684, eSAS .0523 / .0323 / .0538 / .1772, HSTU ML-1M .1719, FuXi-α .1813 and the "recalculated … on the identical training and test data" sentence equal BM-00 §1b. MBHT Table 1 (arXiv 2207.05584): 147,892 / 99,038 / 7,092,362; 200,000 / 808,354 / 13,072,940; 100 popularity negatives — equal BM-00 §4b. **eSASRec first author = Daria Tikhonovich** (Tikhonovich, Zelinskiy, Petrov, Spirina, Semenov, Savchenko, Kuliev; RecSys 2025): "Tikhonovich 2025" is correct; outline.md's "verify" note can be closed.
- **Item 5.** /home/jovyan/venvs/bench-probes-site holds rectools 0.13.0, implicit 0.7.3, typeguard 4.6.0, tqdm 4.70.0 + wheels only (separate PYTHONPATH; ppsi-gpu untouched by construction). CPU plan matches practice: a BM-02 smoke (`v4bench.run … --threads 16`, nice 15) was running at audit time. fa_rounds in v4bench/recipe.py = launch_fl's formula; T and m values in R1 / R4 recomputed and correct.

# BENCHMARK PLAN V1 — external benchmarks for PPSI V4 (PLAN, not registered)

- **Status:** PLAN written 2026-10-01 ~01:45Z by the Operations Director at the owner's request. Nothing downloaded, registered or run.
  Becomes binding only as PREG1_AMENDMENT_07 (BM), registered before any benchmark run.
- **Owner goal:** benchmark results as strong as possible, without weakening the REES46 study.
- **Not touched:** the REES46 study (06N plan, sealed MAIN, single TEST look).

## 0. What "higher than every paper" can honestly mean
| Claim | Achievable? (INFERENCE) |
|---|---|
| Best federated / on-device sequential recommender under full ranking on the same split | likely; our warm start + FedAdam + FT target the known FL–central gap (e.g. PTF-FSR: Fed-SASRec HR@20 0.052 vs central 0.087) |
| Central model at or above the strongest ID-only baseline (SASRec+ full-CE, e.g. Beauty NDCG@10 0.0327) | plausible; same loss family, plus side channels |
| Above text / LLM-based centralized methods on Amazon | unlikely without text features; out of scope |
- **Rules:**
  - Every number is compared only with papers that use the same protocol: full ranking (or the same sampled scheme), the same split
    and the same filtering.
  - All results are reported, not only the favourable ones.

## 1. Facts from our code (MEASURED 2026-10-01)
- **Loss:** `v4train/objective.py` OBJECTIVE_ID = EXACT_FULL_K_CE (full-catalogue cross-entropy, the SASRec+ setting).
- **Side channels:** `v4models/sasrec_v2.py` has category / brand / main_category tokens, event numeric features, quality flags and
  user context.
- **Variants:** max_len 50, dropout 0.2. The federated model is SASREC_D64_B2 (20.7 M params, 98 % in the item tables); the central
  model is SASREC_D256_B3 (83.9 M).

## 2. Datasets and protocols
| Dataset | Source | Protocol (exact text checked against each compared paper before registration) | Side channels mapped |
|---|---|---|---|
| Amazon Reviews 2023 (All_Beauty + one larger category, e.g. Video_Games) | McAuley-Lab official benchmark files | official splits: leave-one-out AND timestamp 8:1:1 | category, brand, price bucket |
| Amazon Beauty (2014, the SASRec / S3-Rec version) | standard 5-core | leave-one-out; the most-cited numbers (SASRec+, S3-Rec, DIF-SR) | category, brand |
| MovieLens-1M | GroupLens | leave-one-out, standard filtering | genre |
| Taobao UserBehavior | Tianchi official (owner download) or a mirror that matches the official counts exactly | 5-core + leave-one-out (Taobao has no canonical version: disclosed) | category, behavior type |

**Pretraining band for PRE (owner decision 10-01: strongest + comparable):** a fixed, seeded 15 % of users (server-side,
consenting). PRE trains on the TRAINING part of these users only; their validation / test items are never seen. Evaluation covers
ALL users, as in the papers. A secondary row reports the 85 % federated-only users.

## 3. Systems run on each dataset
- **Central:**
  - C_FULL (the central ceiling: all data in one place);
  - SASRec ID-only (the standard SASRec+ baseline);
  - Popularity and GRU (lower anchors).
- **Federated (our line):**
  - PRE (server pretraining only);
  - FA_WARM* (S8 frozen tables, the REES46 winner);
  - FT_FA_WARM* (on-device fine-tuning);
  - FP / PF / FA_Q8 warm;
  - DP8 warm frozen.
- **Federated reference:** FA cold start (= vanilla Fed-SASRec in the literature).
- **Taobao:** C_FULL, PRE, FA_WARM*, FT and FA cold get 2 seeds; the other methods get 1 seed (labelled).

## 4. Two tracks
- **Track A — fixed recipe.**
  - The REES46-locked recipe is transferred unchanged: FL_TUNING_SELECTION, the S8 factors, the FT selection and the model sizes.
  - Claim: the recipe generalizes without retuning.
  - Runs on all datasets.
- **Track B — tuned on validation.**
  - A small grid per dataset, chosen on that dataset's VALIDATION split only:
    - max_len {50, 200} (200 only for MovieLens-1M);
    - dropout {0.2, 0.5};
    - early stopping on validation instead of the fixed exposure budget.
  - Claim: the best numbers each method reaches.
  - Runs on Amazon and MovieLens-1M only (small; hours). Taobao is Track A only.
- **Feature variants** in both tracks:
  - ID-only: the main comparison, comparable with papers;
  - RICH: our side channels mapped from each dataset's metadata (secondary).

## 5. Metrics
- **Main (full ranking):**
  - HR@10, NDCG@10, HR@20, NDCG@20 (the papers' metrics);
  - MRR@20 (our REES46 headline);
  - bootstrap CIs over users, paired against the baselines.
- **Secondary:**
  - sampled-100 metrics with the exact scheme of the compared paper (uniform as in SASRec; popularity as in BERT4Rec);
  - computed by a separate benchmark-only tool from the saved models (v4eval untouched; no retraining);
  - labelled "for comparison with sampled-metric papers only" (Krichene & Rendle 2020: sampled metrics can reverse rankings).
- **Deployment columns:**
  - bytes up / down per round;
  - phone-model size;
  - DP ε;
  - measured CPU inference latency and memory (ONNX on CPU as a phone-class proxy).

## 6. Integrity rules
1. PREG1_AMENDMENT_07 is registered before any benchmark run. It fixes datasets, protocols, systems, tracks, grids, seeds and the
   comparison papers with their published numbers.
2. All selection (Track B, PB / CB) is on VALIDATION only. Each dataset has one TEST look, after its lock.
3. No dataset, protocol or metric is added or dropped after seeing results.
4. The REES46 code paths are untouched. Benchmark code is new modules plus a converter; v4eval's evaluator sha is unchanged.

## 7. Work packages (agents only after the 10-02 quota reset; owner rule: minimal agents)
| ID | Work | Who | Estimate |
|---|---|---|---|
| BM-00 | Verify each compared paper's exact protocol + published numbers → table in the amendment | Ops (web) | 0.5 day |
| BM-01 | Download + verify (official counts, sha) the 4 datasets | Ops (scripts, server CPU) | 0.5 day, can start now |
| BM-02 | Converter → our manifests / registry format, protocol splits, pretraining band, side-channel mapping + tests | 1 Opus worker | 1 day |
| BM-03 | Sampled-metric tool (separate module) + tests | 1 Sonnet worker | 0.5 day |
| BM-04 | Amendment 07 text + DL row | Ops | 0.5 day |
| BM-05 | Track A + Track B runs (small datasets beside REES46; Taobao after the warm line) | lanes (scripts) | Amazon / ML-1M ≈ 1 day, Taobao ≈ 2.5 days |
| BM-06 | Lock → one TEST look per dataset → results tables | Ops | 0.5 day |
| BM-07 | ONNX CPU latency / memory measurement | Ops | 0.5 day |

## 8. Schedule (INFERENCE)
- **10-01 (now):** BM-01 downloads (no GPU, no agents).
- **10-02:** BM-00, BM-02, BM-03 and BM-04 → register amendment 07.
- **10-03:** Amazon + ML-1M Track A / B runs, in parallel with the REES46 runs.
- **10-03 → 10-06:** Taobao Track A, after the REES46 warm line frees the cards.
- **10-06 → 10-07:** locks, TEST looks, tables.

## 9. Risks
- **Our recipe may not be best on tiny datasets.** ML-1M has 6 k clients, so FL has few clients per round. Mitigation: Track B,
  and an honest report.
- **Taobao's 4 M items are too many.** Mitigation: the item filter of the standard protocol (5-core); the table size is reported.
- **Published numbers may use undisclosed protocol details.** Mitigation: compare only where the protocol is verified; otherwise
  re-run the baseline ourselves under our protocol.

## 10. Owner decisions (10-01 ~01:50Z: "the strongest results")
- Both tracks, RICH variants on; the 15 % band as in §2; all 4 datasets; Taobao from a mirror only if it matches the official
  counts exactly (100,150,807 rows / 987,994 users / 4,162,024 items), else the owner downloads it from Tianchi.

### Original questions
1. Taobao source: Tianchi official download (recommended) or a verified mirror.
2. Approve the two tracks and the 15 % pretraining band.
3. Approve the dataset list: Amazon 2023 + Amazon Beauty 2014 + ML-1M + Taobao.

## 11. UPDATE 10-01 ~02:15Z — data in hand, final dataset set, per-dataset settings (owner: "strongest metrics; drop any dataset that breaks something")
**Raw data (MEASURED):** `ppsi_v4/bench_raw/<Dataset>/`, each folder with its own SHA256SUMS.
- Taobao UserBehavior (Kaggle copy marwa80/userbehavior) matches the official counts exactly: 100,150,807 rows, 987,994 users,
  4,162,024 items, 9,439 categories, 4 behaviors; csv sha 46fdd7d3….

**Standard processing (MEASURED):** `bench_proc/<Dataset>/` (bm02_standard_prep.py: sequences, items + side channels, STATS,
split.json, SHA256SUMS).
- MovieLens-1M = 6,040 / 3,416 / 999,611: exact match with SASRec / SASRec+.
- Amazon-Beauty-2014 = 22,363 / 12,101 / 198,502: exact match with S3-Rec.
- Amazon2023 Video_Games = 94,762 / 25,612 / 814,586.
- Amazon2023 Baby_Products = 150,777 / 36,013 / 1,241,083.
- The two Amazon 2023 sets are the official files, kept as given; their published counts are to be checked in BM-00.
- Side-channel coverage: Beauty categories 100 %, brand 83 %; Amazon 2023 main_category / categories / store ≥ 96 %;
  ML-1M genres 100 %.

**Dropped (would break the study):**
- Amazon2023 All_Beauty: only 253 users, too few federated clients.
- Taobao: the ComiRec setting (clicks, items ≥ 5) leaves 1,708,530 items (MEASURED; equals ComiRec's published count).
  - Two item tables at d 64 ≈ 219 M params ≈ 875 MB FP32 on the phone (REES46: 83 MB).
  - Full-catalogue CE over 1.7 M items.
  - Kept on disk; possible later as a central-only scale test (not in the comparison set).

**Per-dataset settings.** Strong points taken from the strongest published setups; every value is to be verified in BM-00 before
amendment 07. All datasets use our full-catalogue CE (the SASRec+ setting).
| Dataset | max_len | dropout | Side channels (RICH variant) | Split | Paper metrics (+ our MRR@20) | Compare with |
|---|---|---|---|---|---|---|
| MovieLens-1M | 200 | 0.2 | genres | leave-one-out | HR / NDCG@10, @20 | SASRec+ (full ranking); SASRec / BERT4Rec (sampled-100, uniform / popularity) |
| Amazon-Beauty-2014 | 50 | 0.5 | categories, brand (the DIF-SR setting) | leave-one-out | HR / NDCG@5, @10, @20 | S3-Rec, CL4SRec, DuoRec, ICLRec, DIF-SR, BSARec (protocol per paper); SASRec (sampled-100) |
| Amazon2023 Video_Games / Baby_Products | 50 | 0.5 | main_category, categories, store | official leave-one-out AND timestamp | Recall / NDCG@10, @50 (official benchmark convention, to verify) + @20 | Hou et al. 2024 benchmark + our own SASRec+ ID-only |
- **Shared strong points:**
  - full CE;
  - side-channel fusion (RICH);
  - early stopping on VALIDATION (Track B);
  - PRE warm start + on-device FT (our line);
  - 2 seeds plus bootstrap CIs.
- **Note:** SASRec+ reports Beauty on the 40,226-user version. Its Beauty number is therefore not comparable with the S3-Rec
  version; our own ID-only C_FULL there plays the SASRec+ role.

## 12. UPDATE 10-01 ~02:40Z — external review adopted (owner pasted it; corrections checked where possible)
**Corrections accepted:**
1. PTF-FSR uses Amazon Cell Phone / Baby / MIND. It is not a direct comparison on Beauty / ML-1M (it was cited only as FL–central
   gap context).
2. ODUpdate (CIKM'23) is not a Tmall study (the review says Gowalla + LastFM; my Tmall attribution came from an unverified
   search summary).
3. Federated work on Tmall exists: FCUCR (WWW 2026; Tmall, RetailRocket, XING, LastFM; code public; verified). Its Tmall version is
   user-centric but small (1,746 users per the review).
4. The same counts do not mean the same protocol.
5. "Beauty" has several versions (S3-Rec 22,363 / 12,101; FedAttack 40,226 / 54,542; FELLAS 20,574 / 10,020; CF-FedSR
   52,204 / 57,289 — per the review). Published numbers from another version are contextual only.

**Final dataset set:**
- **Main study:** REES46.
- **Benchmarks:** MovieLens-1M (B1) and Amazon Beauty 2014, S3-Rec version (B2). Both are ready and match exactly (§11).
- **Optional specialized on-device experiment:** Tmall, EODRec version (session data from the EODRec repo). It covers
  compression / inference only, not FL, unless a user-centric version (e.g. the FCUCR protocol) is used.
- **Deferred (kept on disk):** Amazon 2023 Video_Games / Baby_Products / Beauty_and_Personal_Care, Taobao, and the MBHT versions.

**Comparison rule:** either we re-run the reference code on our version and protocol (labelled Re-run), or we run our method on the
reference's own version and protocol. Published numbers from a different version are context only.

**Reporting rules:**
- PRE uses consenting server-side data, an assumption different from pure FL. It is reported as a system option, and its gain is
  not attributed to the FL algorithm.
- Results are shown for ALL users AND separately for the 15 % pretraining users and the 85 % others.
- No REES46-specific rules are ported (5–100 TRAIN examples, the next-distinct-item rule, session gaps).
- Field semantics are respected: Amazon timestamp = review time; store ≠ brand; price = at crawl time.

**Roles:**
- Benchmarks (single last-item query per user) = comparison with the literature.
- REES46 (temporal, multi-decision) = realism and deeper analysis.

**Time (INFERENCE):** B1 + B2 ≈ 1.5–2 days of extra work. The Tmall-EODRec experiment ≈ +1 day if run.

## 13. UPDATE 10-01 ~03:30Z — second external review (owner pasted it): verified parts, own analysis, priorities
**Verified (web):**
- Petrov et al. (Spotify), "Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?", RecSys 2026,
  arXiv 2608.19833. The recency-weighted pairwise probes SeqRules / PCTM beat the eSASRec reproduction by 15–38 % on three Amazon
  sets and by 4.4 % on ML-1M, and trail it by 27.3 % on ML-20M. Code public.
- eSASRec (RecSys 2025, arXiv 2508.06450) and Forward-OFA (KDD 2025, arXiv 2501.02837) exist.
- Exact table values quoted by the review (e.g. PCTM 0.0635 / 0.1815, Forward-OFA 0.1182) are NOT yet verified: read the papers in
  BM-00 before citing.

**MEASURED (bench_proc; seeded 15 % / 30 % user band; training part only):**
| Dataset | Band | Items seen | Test targets covered |
|---|---|---|---|
| MovieLens-1M | 15 % | 97.2 % | 99.9 % |
| MovieLens-1M | 30 % | 99.3 % | 100 % |
| Amazon-Beauty-2014 | 15 % | 66.9 % | 74.8 % |
| Amazon-Beauty-2014 | 30 % | 85.9 % | 87.7 % |
- **Consequence:** S8 (frozen item tables) caps Beauty, because a quarter of the test targets would keep untrained embeddings.
  - Beauty therefore needs trainable tables (S7-type) or a frozen base plus a trainable residual.
  - The item tables are tiny there (ML-1M ≈ 1.75 MB, Beauty ≈ 6.2 MB for two d64 FP32 tables), so freezing saves little
    communication.

**Own analysis (beyond the review):**
1. Any transition expert or KD teacher in the private FL line must be built from the 15 % consenting band only. Built from all
   users centrally, it is a different (non-private) system. On Beauty the band-built transition model sees only 67 % of items.
2. The Spotify result means these benchmarks reward pairwise transitions. The thesis contribution should rest on the
   privacy / deployment trade-off, not on neural sophistication. Strong non-neural baselines are mandatory.
3. Scope: the review lists about 10 components, which is weeks of work. They are prioritized below; KD, adapters and prototypes
   are not in the first wave.

**Priority order (PLAN):**
| # | Work | Cost |
|---|---|---|
| P1 | PCTM + SeqRules (authors' code) + ItemKNN on ML-1M / Beauty, on our versions and protocol | CPU |
| P2 | Our model with trainable tables (S7) vs frozen (S8) vs frozen + residual | the S7 / S8 arms exist; residual needs code |
| P3 | Validation-only grid: blocks {1, 2}, d {64}, dropout {0.1, 0.2, 0.5}, max_len {50 Beauty, 200 ML-1M} | the eSASRec-style settings |
| P4 | Score mixture p = α·p_neural + (1−α)·p_transition, α chosen on VALIDATION | the transition model built from the allowed data only |
| P5 | eSASRec baseline (RecTools implementation) | medium |
| Later | text embeddings, DIF-SR fusion, DuoRec loss, KD student, adapters, prototypes | expensive |
- **Primary benchmark metric (to register):** full-catalogue NDCG@10. Secondary: HR@10 / @20, NDCG@20, MRR@20.
- **Seeds:** 3 for the final candidates.

## 14. OWNER DECISION 10-01 ~04:00Z — supersedes the method proposals of §12–§13
- **Owner:** "I work with your plan and opinion from before the external reviews; they added many unnecessary things, and things we
  did not do in our work must not be added to the benchmark."
- **Methods on the benchmarks = only the REES46 recipe:** C_FULL, PRE, FA_WARM S7 / S8 (FA_WARM* by the same N2 rule), FT on the
  FA_WARM* base, FP / PF / FA_Q8, DP8, FA cold, the REES46 baselines; Track A (fixed recipe) + Track B (validation-only max_len /
  dropout grid); ID-only + RICH.
  - Removed (not in our work): eSASRec, PCTM / SeqRules runs, ItemKNN, the score mixture, KD, text embeddings, DIF-SR fusion,
    DuoRec, adapters, prototypes, frozen + residual.
  - Kept: facts about the data and our own recipe. These are the dataset versions, hashes and published counts, and the MEASURED
    band coverage (Beauty: a 15 % band sees 67 % of items; this is reported, and S7 vs S8 is decided by the registered N2 rule).
- **Comparison numbers:** taken from published tables only where the split is byte-identical (eSASRec / Spotify Table 1:
  SAS+, eSAS, MC, FMC, FMC+, SeqRules, PCTM). Elsewhere they are context only. No other method is run by us.

**Benchmark data (MEASURED):**
- **Wave 1 — the eSASRec standard splits, built and verified byte-identical with the published release:**
  - built by the Spotify probes repo, commit 705710f7, pinned eSASRec 2b039927; bench_raw/spotify-sequential-capacity-probes/
    data/processed/;
  - S3-Beauty 22,363 / 12,101 / 198,502; S3-Sports 35,598 / 18,357 / 296,337; S3-Toys 19,412 / 11,924 / 167,597;
    ML-1M 6,040 / 3,706 / 1,000,209;
  - leave-one-out, full-catalogue ranking, seen-item filtering;
  - ML-20M (138,493 / 26,744 / 20.0 M) is deferred.
  - Tooling: RecTools 0.13.0 + typeguard / implicit / tqdm wheels unpacked into /home/jovyan/venvs/bench-probes-site (separate
    PYTHONPATH). The ppsi-gpu environment was not modified.
- **Wave 2 (my earlier plan, after Wave 1):**
  - Amazon 2023 official splits: Video_Games, Baby_Products, Beauty_and_Personal_Care;
  - the MBHT multi-behavior Tmall / Taobao versions (verify counts and protocol first).
- My earlier ML-1M 5-core version (3,416 items) and Beauty version remain in bench_proc/ as cross-checks.

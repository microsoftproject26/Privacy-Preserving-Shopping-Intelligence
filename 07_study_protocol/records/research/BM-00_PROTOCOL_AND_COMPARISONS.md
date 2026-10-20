# BM-00 — Protocol and published comparisons for amendment 07

- **Task:** BM-00 (research worker, Claude Opus). Read-only work plus web research. No training, no server writes, no git.
- **Date:** 2026-10-01.
- **Inputs read:** BENCHMARK_PLAN_V1 §14; contracts/PREG1_AMENDMENT_07_DRAFT.md; DECISION_LOG rows 296–308.
- **Code read** (server, read-only): `bench_raw/spotify-sequential-capacity-probes` @705710f and `.external/transformer_benchmark` @2b03992. RecTools 0.13.0 is in `/home/jovyan/venvs/bench-probes-site/rectools`.
- **Flags:**
  - **VERIFIED** = I read it in the primary source (paper HTML/PDF, the official repo file, or the published artifact). Where a number is involved, I checked it against our measured data.
  - **MEASURED** = I computed it in this task from the frozen split files (read-only).
  - **INFERENCE** = a reasoned conclusion that has not been tested directly.
  - **UNVERIFIED** = could not be confirmed.

---

## 1. Wave-1 published numbers on the exact eSASRec release splits

All numbers below use the same protocol:
- the eSASRec leave-one-out (LOO) release splits, which we verified byte-identical (DL 302);
- full-catalogue ranking;
- seen-item filtering (every train.csv item of the user is removed before top-k);
- no sampled negatives;
- RecTools `NDCG(k=10, divide_by_achievable=True)` and `Recall(k=10)`, averaged over the holdout users (§2.4).

With one holdout item per user, **Recall@10 = HR@10** exactly. This is VERIFIED: in every probes result JSON, `hit_rate10 == recall10`.

### 1a. Spotify RecSys'26 Table 1 (arXiv 2608.19833v1): full-catalogue NDCG@10, 4 decimals
Source: https://arxiv.org/html/2608.19833 (Table 1). The same values appear in the repo README at https://github.com/spotify-research/sequential-capacity-probes.

- Table 1 caption: "All values are our matched-protocol runs." The paper reproduced SAS+ and eSAS itself.
- **Runs: 1 per cell.**
  - The table config has a single neural seed, 32.
  - MC, SeqRules and PCTM are deterministic.
  - Neither the paper nor the repo reports a mean or standard deviation.
- Flag: VERIFIED for all 28 Wave-1 cells (paper HTML and repo README agree).

| Data | MC | FMC | FMC+ | SAS+ | eSAS | SeqRules | PCTM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Beauty | .0492 | .0481 | .0531 | .0537 | .0524 | .0605 | **.0635** |
| Sports | .0251 | .0244 | .0271 | .0315 | .0324 | **.0371** | .0368 |
| Toys | .0548 | .0554 | .0588 | .0575 | .0533 | .0730 | **.0738** |
| ML-1M | .1240 | .1124 | .1206 | .1662 | .1739 | .1505 | **.1815** |

**Our CPU reproduction (DL 303) is VERIFIED here.** In `results_cpu/*/{mc,seqrules,pctm}.json`, all 12 values equal Table 1 to 4 decimals.

The full-precision values are listed in §1c.

### 1b. eSASRec paper (arXiv 2508.06450 / RecSys'25) and its repo README: Recall@10 and NDCG@10
Sources:
- paper appendix Tables 4–5 (3 decimals): https://arxiv.org/html/2508.06450;
- repo README "Preliminary study results" (4 decimals): https://github.com/blondered/transformer_benchmark, read at the pinned commit 2b03992 on the server.

**Runs:** one value per cell. The code uses `setup_deterministic(random_seed=32)`, and no variance is reported. That this is a single run is an INFERENCE.

| Data | Model | R@10 (README) | N@10 (README) | Paper Table 4/5 (R / N) | Flag | Same split? |
|---|---|---:|---:|---|---|---|
| Beauty | SASRec+SS (= SAS+) | 0.0921 | **0.0531** | 0.092 / 0.053 | VERIFIED | yes |
| Beauty | eSASRec | **0.0928** | 0.0523 | 0.093 / 0.052 | VERIFIED | yes |
| Sports | SASRec+SS | **0.0569** | 0.0314 | 0.057 / 0.031 | VERIFIED | yes |
| Sports | eSASRec | 0.0560 | **0.0323** | 0.056 / 0.032 | VERIFIED | yes |
| Toys | SASRec+SS | **0.0970** | **0.0580** | 0.097 / 0.058 | VERIFIED | yes |
| Toys | eSASRec | 0.0939 | 0.0538 | 0.094 / 0.054 | VERIFIED | yes |
| ML-1M | SASRec+SS | 0.2929 | 0.1684 | 0.293 / 0.168 | VERIFIED | yes |
| ML-1M | eSASRec | 0.3133 | 0.1772 | 0.313 / 0.177 | VERIFIED | yes |
| ML-1M | HSTU (recalculated by eSASRec authors) | 0.3037 | 0.1719 | 0.304 / 0.172 | VERIFIED | yes: README says "recalculated … on the identical training and test data" |
| ML-1M | FuXi-α (recalculated) | **0.3164** | **0.1813** | 0.316 / 0.181 | VERIFIED | yes, same statement |
| Beauty / Sports / Toys | HSTU | 0.0788 / 0.0437 / 0.0778 | 0.0439 / 0.0243 / 0.0450 | 0.079/0.044 · 0.044/0.024 · 0.078/0.045 | VERIFIED values | **UNVERIFIED split** (see note) |
| Beauty / Sports / Toys | FuXi-α | 0.0809 / 0.0437 / 0.0765 | 0.0451 / 0.0244 / 0.0437 | 0.081/0.045 · 0.044/0.024 · 0.077/0.044 | VERIFIED values | **UNVERIFIED split** (see note) |

- **Note on the Amazon HSTU / FuXi-α rows.** The README says HSTU and FuXi-α need `s3_repro_w_time.py`. That builder rebuilds Beauty / Sports / Toys from the P5 raw review dumps with real timestamps, not from S3Rec's `.txt` files. The README links the separate HSTU repo to "our data splits", but I could not confirm byte identity. Treat these rows as context-only.
- **"Reported in literature" rows** (SASRec, S3Rec, LSAN, DuoRec, CL4SRec, TIGER, ActionPiece, published HSTU): these are numbers copied from other papers, on their own pipelines, with mixed full and sampled protocols. Context-only; never compare directly.
- **Consistency check (VERIFIED).** Spotify Table 1 SAS+ / eSAS vs the eSASRec README N@10:
  - |Δ| ≤ 0.0033 everywhere;
  - the maximum is ML-1M eSAS, .1739 vs .1772;
  - this matches the paper's "within 0.0033" claim.

  The SAS+ / eSAS cells are therefore single-run reproductions with roughly ±0.003 run-to-run spread.

### 1c. Spotify clean-room re-run artifact (repo `artifacts/table1-clean-room/`, H100, 16 Aug 2026): NDCG@10 and Recall@10, full precision
Source: `artifacts/table1-clean-room/table1.csv` and `model-metadata/*/*.json`, in the published repo. Flag: VERIFIED (read from the published files).

This is a second, independent run. For SAS+ / eSAS it differs from Table 1 by up to 0.0017. The Table 1 acceptance tolerance for SAS+ / eSAS is 0.005.

| Data (holdout users) | MC N / R | FMC+ N / R | SAS+ N / R | eSAS N / R | SeqRules N / R | PCTM N / R |
|---|---|---|---|---|---|---|
| Beauty (22,311) | .049177 / .072296 | .053071 / .082471 | .053440 / .092107 | .051084 / .090987 | .060477 / .095558 | .063517 / .099816 |
| Sports (35,539) | .025126 / .038999 | .027099 / .043895 | .032140 / .056952 | .031993 / .056614 | .037126 / .062551 | .036795 / .061566 |
| Toys (19,365) | .054779 / .079577 | .058793 / .088510 | .058661 / .097495 | .051633 / .090008 | .072997 / .114227 | .073770 / .113039 |
| ML-1M (6,038) | .123956 / .211163 | .120593 / .206194 | .165425 / .291156 | .176674 / .310699 | .150494 / .257536 | .181516 / .305730 |

FMC rows: Beauty .048107 / .072431, Sports .024419 / .038409, Toys .055875 / .082985, ML-1M .112437 / .199569.

### 1d. Other papers on the eSASRec release splits
- **None found.** I searched the web for "eSASRec" together with split, LOO, Beauty, ML-1M and 2026.
- I opened the two hits that cite eSASRec:
  - arXiv 2604.21536: cites eSASRec in §3, but no release-split use was found;
  - arXiv 2607.11168 (SISA-Rec): quotes eSASRec's ML-20M temporal number 0.1563.
- Neither uses the release splits. Flag: VERIFIED that no use was found; absence is not proven.

**Directly comparable set for B5 (recommended):**
- Spotify Table 1 (all 7 columns) as the primary reference;
- eSASRec README SASRec+SS / eSASRec R@10 and N@10 on all four sets;
- HSTU / FuXi-α on ML-1M only.

Everything else is context-only.

---

## 2. Protocol our evaluator must match (with code citations)

Paths:
- `P/` = `spotify-sequential-capacity-probes/`;
- `E/` = `P/.external/transformer_benchmark/`;
- `R/` = `/home/jovyan/venvs/bench-probes-site/rectools/`.

### 2.1 How the splits were built (already frozen; do not rebuild)
- **Holdout splitter** (`E/src/val_schemes/leave_one_out.py:14-20`): `LastNSplitter(n=1, filter_cold_users=True, filter_cold_items=True, filter_already_seen=True)`.
  - The holdout is each user's last interaction.
  - It is dropped if the user is absent from train, if the item is absent from train, or if the user already has that item in train.
- **Tie rule inside the splitter** (`E/src/evaluation/splitter.py:92-105`): `groupby(user).datetime.rank(method="first")`. Among equal timestamps, the **row that appears last in the file** is the holdout.
- **Amazon S3 timestamps are synthetic.** In `E/src/datasets/s3_repro.py:47-52`, the timestamp is the 1..n position in the S3Rec `.txt` line.
  - MEASURED: 0 timestamp ties.
  - MEASURED: train.csv is already sorted by (user, datetime).
- **ML-1M** (`E/src/datasets/ml_1m.py:14,53`): all ratings ≥ 1 are kept, and `datetime = to_datetime(timestamp, unit="s")`.
  - MEASURED: train.csv is in raw `ratings.dat` order, **not** time-sorted within a user.
  - MEASURED: 2,483 / 6,040 users have ties at their maximum train timestamp.
  - MEASURED: 2,544 holdout rows share their timestamp with the user's last train timestamp.
- **`loo_correction`** (`statistics.csv`, written by `E/src/datasets/common.py:153-198`) = holdout rows / users.
  - It is **never applied to any metric.** No reader of it exists in `E/src` or `P/src`.
  - The published numbers are therefore averages over holdout users only (VERIFIED, §2.4).

### 2.2 Inner validation and the "final retrain"
**eSASRec neural (SAS+ / eSAS): there is no retrain.** It is one `fit` on the full train.csv, with an inner LOO validation:
- `get_val_mask_func_all` (`E/src/models/transformers/trainer.py:217-229, 250-252`) masks each user's rank-0 row after `sort_values(datetime, ascending=False, kind="stable")`.
- RecTools removes those rows from training (`R/models/nn/transformers/data_preparator.py:194-199`). It keeps them as validation targets only if the user and item survive in the inner train (`:224-232`).
- The inner train also drops users with fewer than 2 events and keeps only the last `session_max_len + 1` events per user (`:181-187`; `R/models/nn/transformers/sasrec.py:77` sets the addition to 1).
- Early stopping and checkpointing (`trainer.py:120-183`):
  - monitor = validation Recall@10, seen-filtered (`:80-91`);
  - patience 50;
  - max epochs 200 for Amazon (`get_trainer_200_epochs`) and 100 for ML-1M;
  - the best-Recall@10 checkpoint is reloaded at fit end (`BestModelLoad`, `:30-47`).
- Training settings: lr 1e-3 and batch 128 (`P/src/capacity_probes/neural.py:101-110`). The seed is 32 (`E/src/utils.py:30`). The probes contract checks all of these (`P/src/capacity_probes/neural_contract.py:70-71`).
- TEST scoring uses `model.recommend(users=holdout users, dataset=full train.csv, k=10, filter_viewed=True)` (`E/src/evaluation/common.py:254-277`).
  - The history is the full train.csv sequence, **including** the inner-validation item.
  - Sequences are sorted by datetime, stable (`data_preparator.py:184-185`).

**Tie surprise on ML-1M (MEASURED):**
- eSASRec's inner-validation target is the **first row in file order** among the tied max-timestamp rows. It equals the first-in-file tied row for 6,040 / 6,040 users.
- It is the last element of the (datetime, file-order) sequence for only **3,557 / 6,040** users.
- This was tested with pandas 3.0.3. Under eSASRec's pandas 2.3.3, the same `nargsort` stable-descending logic makes it an INFERENCE.
- Amazon sets are unaffected (no ties).

**Inner-validation population (MEASURED, replicating the RecTools filters):**
- S3-Beauty 22,308; S3-Sports 35,475; S3-Toys 19,312.
- ML-1M **6,039**. With the RecTools 201-event truncation this becomes 6,038 (INFERENCE), because one validation item falls outside the truncated catalogue.

**Probes (MC / FMC / FMC+ / SeqRules / PCTM):**
- Hyperparameters were selected earlier on an inner LOO of train.csv, then **refit on the full train.csv** (`P/docs/PROTOCOL.md`, "Data boundary").
- The paper text says the same: "selected on an inner training holdout before retraining each model on the full training split" (arXiv 2608.19833 §3).
- FMC / FMC+ are refit for the frozen epoch count with seed 20270710 (`P/configs/table1.json` `refit_seed`).

### 2.3 Seen-item filtering and candidate set
- **Candidates:** the full catalogue, i.e. the items in train.csv. This is the `catalog=train.item.unique()` used in `E/src/evaluation/common.py:271-277` and `P/src/capacity_probes/metrics.py:46-60`.
  - MEASURED train catalogues: Beauty 12,092; Sports 18,347; Toys 11,917; ML-1M 3,704.
  - The 12,101 / 18,357 / 11,924 / 3,706 counts are whole-dataset item counts.
- **Filter:** remove **every item in the user's train.csv history** (including the inner-validation item) before top-k.
  - eSASRec: `filter_viewed=True`.
  - Probes: `blocked[seen] = True` in `P/src/capacity_probes/models/mc.py:97-98`, `pctm.py:153-154`, `seqrules.py:243-244`, and `-inf` masking in `fmc.py:157-170`.
- **No repeat targets are possible.** The splitter's `filter_already_seen` guarantees that no holdout target is in that user's train history.
- **Model-catalogue caveat (INFERENCE from replicating RecTools).**
  - The eSASRec SAS+ / eSAS item table is built from the inner train *after* removing the validation item and truncating to max_len + 1.
  - It covers 12,066 / 18,281 / 11,876 / 3,646 items.
  - So 89 / 265 / 232 / 0 holdout targets were unreachable for SAS+ / eSAS (about 0.4–1.2 % of Amazon users). Probes and our models rank the full train catalogue.
  - Our evaluator must rank the full train.csv catalogue; that is the protocol of record. Disclose that published SAS+ / eSAS carry this small handicap.

### 2.4 Which users are evaluated (6,038 / 6,039 / 6,040 on ML-1M)
**TEST population = holdout rows** (VERIFIED from the published artifacts):

| Data | Users (n_users) | Holdout rows = evaluated users | Dropped by splitter filters |
|---|---:|---:|---:|
| S3-Beauty | 22,363 | **22,311** | 52 |
| S3-Sports | 35,598 | **35,539** | 59 |
| S3-Toys | 19,412 | **19,365** | 47 |
| ML-1M | 6,040 | **6,038** (holdout.csv has 6,039 lines, including the header) | 2 (users 46 and 1812) |

- The published means divide by these counts. VERIFIED: `users` in the probes JSONs = 22,311 / 35,539 / 19,365 / 6,038, and `calc_metrics` merges left on the holdout (`R/metrics/base.py:81-103`).
- All holdout users and items are in train (MEASURED: 0 cold users, 0 cold items).
- **"6,039" is the inner-validation count, not the TEST count** (§2.2).
- Users without a holdout row (52 / 59 / 47 / 2) are **not** in the TEST denominator.
  - They stay in training and in inner validation.
  - Our band / others splits must report over holdout users, so that "all users" equals the published denominator.
  - Recommendation: report the band population counts explicitly.

### 2.5 Tie handling in ranking
| Model | Equal-score order | Citation |
|---|---|---|
| MC, SeqRules | ascending external item ID | `P/configs/table1.json`; `P/src/capacity_probes/ranking.py:27-35` (`tie_ranks("external_id")`), `:38-50` (`topk` = lexsort(−score, tie rank)) |
| PCTM | ascending SHA-256 of UTF-8 external ID, then the ID string | `ranking.py:19-24` (`stable_hash_ranks`); arXiv 2608.19833 suppl. "Computation" |
| FMC, FMC+, SAS+, eSAS | `torch.topk`; backend-dependent, no guarantee | `P/docs/PROTOCOL.md`; `R/models/rank/rank_torch.py:147` |

**Rule for our evaluator:**
- sort by score descending;
- break exact ties by ascending external item ID (the MC / SeqRules convention, deterministic);
- log the number of exact ties at rank ≤ 20.

For float32 neural scores, ties are practically absent, so this cannot move a published-comparison number.

### 2.6 Metric formulas (one target per user, rank r ≥ 1 among the filtered candidates)
Source: RecTools `NDCG(k, divide_by_achievable=True)`, `R/metrics/ranking.py:444-465`. With a single target, IDCG = 1 (`log_base` 2). Recall = TP / liked (`R/metrics/classification.py`, `Recall`).

| Metric | Formula | Used where |
|---|---|---|
| NDCG@k | 1/log2(r+1) if r ≤ k, else 0 | primary (k = 10), published |
| Recall@k = HR@k | 1 if r ≤ k, else 0 | published R@10 |
| MRR@20 | 1/r if r ≤ 20, else 0 | ours only (no published counterpart) |
| NDCG@20, HR@20 | as above with k = 20 | ours only |

- **Aggregation:** the mean over holdout users (§2.4).
- **Parity test:** the probes' independent implementation (`P/src/capacity_probes/metrics.py:22-43`, `ranking.py:81-88`) matches the RecTools value. Our evaluator should reproduce PCTM / SeqRules / MC from their published recommendation lists, or re-score them, to 1e-9. Our CPU outputs in `results_cpu/` can serve as fixtures.

### 2.7 Implications for PREG1 amendment 07 (PLAN, not decided here)
1. **TEST:** one holdout item per user from holdout.csv; denominators 22,311 / 35,539 / 19,365 / 6,038; candidates = the train.csv catalogue; filter the full train.csv history; tie rule as in §2.5.
2. **Inner validation:**
   - Amendment B1 says "each user's last TRAIN item". This needs a tie convention on ML-1M.
   - Option (a): the last element of the train sequence sorted by (datetime, source row order). This is consistent with the splitter's holdout convention and with the probes' `SequenceStore` order (`P/src/capacity_probes/sequences.py:21-36`).
   - Option (b): eSASRec's convention, the first-in-file among tied max-timestamp rows.
   - They differ for 2,483 ML-1M users.
   - Recommendation: (a), written into the amendment, with eSASRec's convention disclosed. Selection is ours, so either is legitimate, but it must be fixed before any run.
3. **TEST inference history** = the full train.csv sequence, including the inner-validation item. This is eSASRec practice: no retrain, best checkpoint, then recommend from full history. Our PRACTICAL_BEST checkpoint matches it. A post-selection retrain would match the probes instead; either way, register one choice.
4. **Comparisons:** Table 1 cells are single runs. Report our 2–3-seed mean against them, and state the roughly 0.003 reproduction spread of SAS+ / eSAS (§1b–1c).

---

## 3. Federated / on-device papers: dataset version and protocol (all context-only)

| Paper | Datasets (version, counts) | Split | Ranking protocol | On eSASRec splits? | Flag |
|---|---|---|---|---|---|
| **FELLAS** (Yuan et al., TOIS 2025; arXiv 2410.04927) | Amazon **2018** (Ni et al. 2019) Beauty 20,574 u / 10,020 i / 164,268; Office 4,619 / 1,939 / 42,268; PLG 1,518 / 751 / 10,439; max len 50 | last 2 items = valid / test | full ranking over all non-interacted items (cites Krichene & Rendle); HR / NDCG@10/20. Fed SASRec Beauty H@10 .01769, N@10 .00839; FELLAS .02532 / .01264; central SASRec .02289 / .01133 (d = 8 models) | **No** (different Beauty version) | VERIFIED |
| **FedSeqRec** | Not a single paper. FELLAS (Table 3) uses "FedSeqRec" for the vanilla federated SASRec / GRU4Rec baseline. Nearest titled paper: LUMOS (arXiv 2602.09306): Amazon Cell Phone, Amazon Baby, MIND; 5-core; last 2 items valid / test; max len 50 | LOO | full ranking, HR@20 / NDCG@20 | **No** | VERIFIED (FELLAS usage, LUMOS protocol); the exact paper the plan meant is UNVERIFIED |
| **CF-FedSR** (Luo et al., arXiv 2208.10692) | Amazon Beauty 52,204 u / 57,289 i / 394,908; Video 31,013 / 23,715 / 287,107; Wikipedia 8,227 / 1,000 / 157,474 | LOO (last = test, 2nd-last = valid) | **sampled: 100 random negatives** + target; HR / NDCG@5/10 (Beauty central HR@10 .4482) | **No** | VERIFIED |
| **FCUCR** (WWW 2026; arXiv 2603.17315) | XING 877 u / 29,366 i; RetailRocket 391 / 14,051; LastFM 940 / 145,469; Tmall 1,746 / 123,405 (HRNN session preprocessing, 30-min gap) | last session = test, 2nd-last = valid | HR@10 / NDCG@10; full vs sampled ranking not stated in the text | **No** | VERIFIED datasets; ranking protocol UNVERIFIED |
| **Forward-OFA** (KDD 2025; arXiv 2501.02837) | ML-1M 6,040 u / **3,012** i / 994,852 (20-core); ML-10M 69,878 / 8,882; Amazon-2018 Food 27,149 / 51,885; Game 11,292 / 68,308 | last interaction = test | NDCG / Hit (cutoff and sampling not stated) | **No** (ML-1M 3,012 items ≠ 3,706) | VERIFIED datasets; metric details UNVERIFIED |
| **DUET** (WWW 2023; arXiv 2209.05227) | ML-1M, ML-100k (all pairs positive) | — | **AUC** with sampled negatives (1:4 train, 1:100 test) | **No** | VERIFIED |
| **EODRec** (TOIS; arXiv 2209.13422) | Tmall (IJCAI-15) 351,268 train / 25,898 test sessions / 40,728 items; RetailRocket 433,643 / 15,132 / 36,968; session length 1 removed, items with fewer than 5 occurrences removed, prefix augmentation | last item of each session = test; 10 % random valid | P@K / NDCG@K, K = 5 / 10 (session-based) | **No** | VERIFIED |

**Conclusion:**
- None of these is on the eSASRec splits; they differ in dataset version, filtering, split and/or sampled metrics.
- None is directly comparable. All are context-only, as expected.
- Forward-OFA's ML-1M (3,012 items, 20-core) is the closest. It still differs.

---

## 4. Wave-2 checks

### 4a. Amazon Reviews 2023 official 5-core benchmark (McAuley Lab)
Sources:
- https://amazon-reviews-2023.github.io/data_processing/5core.html (official benchmark files; rounded "K" / "M" counts);
- Hou et al. 2024 (BLaIR), arXiv 2403.03952.

| Category | Official #User / #Item / #Rating (5-core) | Our MEASURED | Leave-last-out split (official) | Flag |
|---|---|---|---|---|
| Video_Games | 94.8K / 25.6K / 814.6K | 94,762 / 25,612 / 814,586 | 625.1K / 94.8K / 94.8K | VERIFIED (consistent with rounding) |
| Baby_Products | 150.8K / 36.0K / 1.2M | 150,777 / 36,013 / 1,241,083 | 939.5K / 150.8K / 150.8K | VERIFIED (consistent with rounding) |
| Beauty_and_Personal_Care | 729.6K / 207.6K / 6.6M | 729,576 test users (DL 298); items and ratings not measured | 5.2M / 729.6K / 729.6K | users VERIFIED; items and ratings UNVERIFIED (exact) |
| All_Beauty (dropped) | 253 / 356 / 2.5K | 253 users | — | VERIFIED (confirms the drop) |

**Surprise:** Hou et al. (BLaIR) do **not** benchmark on these 5-core files.
- BLaIR v1 Table 3 and v2 (2026) both use **raw (0-core) data with absolute-timestamp splits**: Games 115,815 items / 2,550,503 interactions; Baby 179,133 / 3,599,935.
- The website says "Researchers are encouraged" to use the 5-core / leave-last-out files; it does not publish model numbers for them.
- So no published table exists on our exact Wave-2 Amazon version. Every Amazon-2023 number is context-only, unless a paper states it uses the official 5-core leave-last-out files (none found).

### 4b. MBHT (KDD'22; arXiv 2207.05584; repo yuh-yang/MBHT-KDD22)
**Paper Table 1** (VERIFIED, https://arxiv.org/html/2207.05584):

| Paper name | Users | Items | Interactions | Behaviours |
|---|---:|---:|---:|---|
| Taobao | 147,892 | 99,038 | 7,092,362 | buy, cart, fav, pv |
| Retailrocket | 11,649 | 36,223 | 87,822 | buy, cart, pv |
| IJCAI (= Tmall, IJCAI-15) | 200,000 | 808,354 | 13,072,940 | buy, cart, fav, pv |

**Folder ↔ paper mapping** (MEASURED on `bench_raw/MBHT-KDD22-processed/MBHT_dataset`, from the item-ID range):

| Folder | Max item ID / distinct items seen | Train rows / test rows | → Paper name | Flag |
|---|---|---|---|---|
| `tmall_beh` | 99,038 / 99,037 | 462,945 / **9,999** | **Taobao** (UserBehavior, Tianchi 649); *not* Tmall despite the name | MEASURED mapping, consistent with paper items |
| `ijcai_beh` | 808,355 / 782,694 | 558,250 / **10,000** | **IJCAI-15 = Tmall** (808,354 items) | MEASURED, consistent |
| `retail_beh` | 36,224 / 31,239 | 30,690 / **2,000** | **Retailrocket** (36,223 items) | MEASURED, consistent |

- The DL 298 lower bounds (≥ 235,710 and ≥ 132,351 items) are superseded by the counts above.
- **Users and interactions cannot be reconstructed** from the RecBole augmented files: `session_id` is a row ID, not a user ID, and sequences are capped at 199. Paper users and interactions are therefore UNVERIFIED against the files.
- The test files hold only a 10,000 / 10,000 / 2,000-user **sample**, not all users.
- **Protocol (VERIFIED, paper §4.1.2 plus repo code):**
  - LOO on the last purchase; earlier purchases are validation.
  - Each positive is ranked against **100 negatives sampled by popularity**. In code (`recbole/model/sequential_recommender/mbht.py:363-385`, `customized_eval: 1`, `topk [5, 10, 101]`), the **100 popular negatives are a fixed hard-coded list per dataset**; any already-seen candidate is replaced by a random unseen item.
  - Metrics: HR / NDCG@5/10 and MRR.
  - This is **sampled, not full-catalogue**, so MBHT numbers are not comparable with our full-ranking protocol. Context-only. We would need to re-run baselines ourselves, and the owner's rule forbids that.

---

## 5. Sources
- Spotify probes paper: https://arxiv.org/html/2608.19833 (v1, 20 Aug 2026). Repo: https://github.com/spotify-research/sequential-capacity-probes (server clone @705710f; `artifacts/table1-clean-room/`, `docs/PROTOCOL.md`, `configs/table1.json`).
- eSASRec: https://arxiv.org/html/2508.06450 (Tables 4–5). Repo: https://github.com/blondered/transformer_benchmark @2b039927 (README, `src/`).
- RecTools 0.13.0 source (server site dir).
- FELLAS https://arxiv.org/html/2410.04927 · CF-FedSR https://arxiv.org/html/2208.10692 · FCUCR https://arxiv.org/html/2603.17315 · Forward-OFA https://arxiv.org/html/2501.02837 · DUET https://arxiv.org/html/2209.05227 · EODRec https://arxiv.org/html/2209.13422 · LUMOS https://arxiv.org/html/2602.09306.
- Amazon Reviews 2023: https://amazon-reviews-2023.github.io/data_processing/5core.html · BLaIR https://arxiv.org/html/2403.03952 (v2) and /2403.03952v1.
- MBHT: https://arxiv.org/html/2207.05584 · https://github.com/yuh-yang/MBHT-KDD22 (`MBHT.yaml`, `run_MBHT.py`, `mbht.py`).

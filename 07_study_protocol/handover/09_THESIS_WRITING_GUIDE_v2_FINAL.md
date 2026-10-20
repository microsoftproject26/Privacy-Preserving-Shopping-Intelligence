# 09 v2 FINAL — Thesis writing guide

*Replaces `09_THESIS_WRITING_GUIDE.md` (2026-10-03, written before the final results; kept unchanged). Written 2026-10-10 for the colleagues who write the MSc thesis. All numbers are copied from `14_UPDATE_v2_20261008.md` ("file 14"). Where this guide says "see file 14 §x", look the value up there; do not estimate it. Next steps and rules: `17_NEXT_STEPS_START_HERE.md`.*

**Two rules for every sentence with a number.** (1) Give the source (file 14 section, DL row, or result file) and the label: CONFIRMATORY or POST_HOC_EXPLORATORY. (2) Confirmatory numbers are quoted exactly as printed. Exploratory numbers go beside them, never instead of them.

**Note on older numbers.** The 2026-10-03 guide quotes K20b 88.81 % and K21 −0.000433. Those were VALIDATION values. Use the TEST values below: K20b 88.49, K21 −0.000508. Keep the VALIDATION values only if you show a VALIDATION column.

---

## 1. The main argument (4–5 sentences)

The value of federated learning (FL) depends on how strong the server-side pretraining is. On REES46, with a strong server pretrain on 1.2M users, FL adds nothing: the registered result is K21 = −0.000508, and the tau-fixed re-run is also indistinguishable from zero. The registered FedAdam setting (tau = 1e-3) silently froze FL, because almost all per-round updates were below tau; with tau = 1e-6, FL learns, and it closes most of the gap to the cloud model when the pretrain is weak or absent (PB100, the benchmarks, the disjoint server band). FL that really learns also leaks membership (E1 AUC 0.747 on cold FL); DP at epsilon = 8 removes the leak, at a small utility cost on a warm start and a large cost on a cold start. The realistic Edge system, warm-started FL, is feasible on devices (INT8 and 8-bit uploads cost nothing measurable; see file 14 §1 and DEVICES.json), and it keeps 88.5 % of the full-data cloud model.

---

## 2. Chapter-by-chapter outline with the results to report

Structure follows the old guide (§1 of file 09). Only chapters that carry results are expanded; the others keep the old sources.

### Ch 1 Introduction
- Motivation (browser-side shopping help, privacy), the proposal question verbatim, Edge framing (cloud only / FL with weights only / small local model).
- State the contribution as answers. Use the four-point story of file 14 §2c ("Thesis story").
- List the departures from the proposal (old guide, Ch 1 row).
- Source: proposal; `14` §2c; `17` §1.

### Ch 2 Background
- FedAvg, FedProx, FedAdam and FedAvgM; why FedBN does not apply (LayerNorm); SASRec and GRU4Rec; full-ranking vs sampled metrics; DP-FedAvg and the RDP accountant; INT8 / ONNX.
- **Explain FedAdam's tau here** (update divided by sqrt(v) + tau). It is needed to understand the tau finding. Source: `fix/fedadam-tau` PR text in `17` §3 and `docs/fedadam-tau.md` on GitHub.
- Sources: as in the old guide, Ch 2 row.

### Ch 3 Data and task
- REES46 frozen v1, the task, splits, populations, CALIB inside MAIN; the six benchmarks (S3 Beauty, Sports, Toys; Amazon-2023 Video Games, Baby Products; ML-1M).
- Sources: files 02, 06; `DATA_PROTOCOL_LOCK_V2.json`.

### Ch 4 Methods
- Models, FL recipe, warm start (PRE, FA_WARM_FROZEN), DP design, device arms.
- **The exploratory recipe change** (tau 1e-3 to 1e-6, larger budget, unfrozen items) must be described here as a separate, later recipe. Sources: files 01, 05; file 14 §2.

### Ch 5 Pre-registration and evaluation protocol
- Amendments, locks, the T0a sitting (2026-10-05T21:03Z to 2026-10-07T10:09Z, DL 511, 535–538), register entries 10 and 11 (confirmatory looks).
- **The exploratory looks**: register entries 12–19. Show them in a table with the date, the lock and what each look covered (file 14 §2, §2b, §2c).
- Statistics: bootstrap CIs, Holm family, TOST, claim vocabulary (old guide §4).

### Ch 6 Results: see the Results chapter plan in section 3 of this guide.

### Ch 7 Discussion
- What a strong pretrain changes; why the registered FL recipe was weak (tau); the privacy and utility trade-off; deployment advice (frozen tables, 8-bit uploads, INT8, on-device fine-tuning); capacity: the federation cost is the real gap, not size.
- Explain the practical message: with a strong server pretrain, on-device fine-tuning (K9c +0.002283) is the deployable gain and needs no federated traffic.
- Sources: file 14 §1–2c.

### Ch 8 Conclusion and future work
- Answer the four-point story. Future work: a large teacher on opted-in users distilled into the small model (DL 276); DP at full-pool scale; real secure aggregation; the optional TEST looks for A1, A2 and the DP studies (`17` §2 step 3); more than one seed for θ_pre.

### Appendices
- Amendments and locks, register entries 10–19, arm list, incident log (I055, DL 535 / 536 / 541 / 548–551), declared-not-run list, AI-use note (section 6 below).

---

## 3. Results chapter plan

### Part A. Confirmatory results (report first, exactly as printed)

**A1. REES46: Phase-1 K-contrasts** (MRR@20, TEST, PRACTICAL_BEST). Source: file 14 §1; `SITTING_T0a_20261007/`; Phase-1 RETENTION_RESULT `28b76492…`. Report in this order:

| Contrast | Meaning | Value |
|---|---|---|
| K2 R(FA / C_BIG) | federated vs central model | 85.11 [84.30, 85.90] |
| K3 R(FA / C_SMALL) | vs central model of same size | 89.88 |
| K17 R(FA / C_FULL) | vs full-data central | 70.10 |
| **K20b** R(FA_WARM_FROZEN / C_FULL) | the realistic Edge system | **88.49** [87.73, 89.18] |
| K20c R(PRE / C_FULL) | θ_pre alone | 88.77 |
| **K21** FA_WARM_FROZEN − PRE | value of FL over θ_pre | **−0.000508** [−0.000933, −0.000037] |
| K9c FT_PRE − PRE | on-device fine-tune gain | +0.002283 |
| K11 FA − LO | collaboration vs device only | +0.1234 |
| K8 R(LO / C_BIG) | device only | 2.10 |
| K12 / K13 | DP at epsilon = 8 / 1 | 98.52 / 94.38 |
| K14 / K16 | 8-bit uploads / INT8 on device | no measurable cost |

CIs are given only where file 14 gives them. For the others write "see file 14 §1" or the RETENTION_RESULT file. K21 is slightly negative with a CI that excludes 0: write "FL does not add value over θ_pre, and the point estimate is slightly below it". Do not write "FL hurts a lot".

**A2. Phase-2, Q1–Q9.** Source: file 14 §1; Phase-2 RETENTION_RESULT `c44382c5…`.
- Q1 P2_WSTAR − PRE = −0.000081 [−0.000179, 0.000024], read as **EQUIVALENT**.
- Holm family: nothing rejected.
- Q9 R(W* / C_FULL) = 88.72. DEPLOYMENT-GRADE = False.
- Q2–Q8: the values are not in file 14. Take them from the Phase-2 RETENTION_RESULT and `P2_CONTRASTS_v3`; do not copy from memory.

**A3. The six benchmarks, confirmatory TEST** (one look each, LOCK_FINAL_RUNBOOK, DL 521, 523, 542). Source: file 14 §1; GitHub `results/benchmarks/`.
- Five CPU datasets (Track A, NDCG@10): C_FULL wins everywhere; every FL arm is far below it. The confirmatory FA values are in the table of section B2 (last column).
- ML-1M (GPU, A40-FP32-DET): C_FULL 0.1936 (published eSASRec 0.1767), FA 0.1572, PRE 0.1092.
- Confirmatory retention ranged 13–81 % (DL 551–552).
- Say that s3_toys has 2 TEST_ATTEMPTS lines: 1 look plus 1 post-look gate refusal (incident I055, DL 528).
- Compare with Spotify Table 1 (full ranking) in a separate table, descriptive only.

**Other confirmatory-adjacent items:** E1 MIA audit VALID; E4 SecAgg 0 violations, full-model run 26 / 26 cells bit-identical (DL 525); devices in DEVICES.json (C_FULL device run: 31 sessions, one hung session excluded, DL 520).

### Part B. Exploratory analyses after the test look

Start the section with this paragraph: *"The analyses below were designed after the confirmatory TEST results were seen. They are POST_HOC_EXPLORATORY. Their TEST looks are register entries 12–19. They are reported to explain the confirmatory results and are not confirmatory evidence."* Give each sub-section a small "label / lock / register entry" line.

**B1. The tau finding** (file 14 §2; DL 528, 540, 546).
- Mechanism: about 100 % of per-round update values were below tau = 1e-3; the server step was 20–100x too small; W* MAIN moved about 0.06 % from θ_pre over 13,654 rounds (MEASURED).
- Show the unit-test demonstration from the `fix/fedadam-tau` PR (pseudo-gradients near 1e-7).

**B2. Fixed-recipe benchmarks** (T1; lock `EXP_BENCH_X_TEST_LOCK.json` 025a59dc…, X_RESULT e1b6368d…; register entries 14–17). TEST NDCG@10, mean of 2 seeds:

| Dataset | C_FULL | C_SMALL | FA_X | FA_WARM_X (% of C_FULL) | confirmatory FA |
|---|---|---|---|---|---|
| s3_beauty | 0.0437 | 0.0449 | 0.0366 | 0.0366 (84 %) | 0.0253 |
| s3_sports | 0.0222 | 0.0213 | 0.0161 | 0.0167 (75 %) | 0.0078 |
| s3_toys | 0.0408 | 0.0449 | 0.0372 | 0.0377 (92 %, R8 "on par") | 0.0054 |
| video_games | 0.0503 | 0.0452 | 0.0368 | 0.0381 (76 %) | 0.0240 |
| baby_products | 0.0182 | 0.0174 | 0.0139 | 0.0146 (80 %) | 0.0099 |
| ml_1m | 0.1951 | 0.1867 | 0.1462 | 0.1613 (83 %) | 0.1504 |

Message: the fix works on TEST; C_SMALL is 96–110 % of C_FULL; the federation cost at matched size is FL / C_SMALL about 78–86 %. Amazon-2023 is descriptive. The validation-selection retentions (beauty 79.5, sports 79.6, toys 91.3, video_games 75.9, baby_products 82.2, ML-1M 89.4) may be shown as a VALIDATION column, labelled as such.

**B3. REES46 tau-fixed** (T2; lock `EXP_XREES46_TEST_LOCK.json` a67b19ca…, X_RESULT df9ad2c6…). MRR@20:

| Contrast | Value | Reading |
|---|---|---|
| X_FA_WARM − PRE | +0.00022 [−0.00011, 0.00057] | includes 0 |
| X_FA_WARM − W* | +0.00030 [−0.00004, 0.00068] | includes 0 |
| R(X_FA_WARM / C_FULL) | 89.05 | — |
| X_FA_COLD − confirmatory FA | −0.0068 [−0.0078, −0.0059] | worse; tau and eta changed together |
| R(X_FA_COLD / C_FULL) | 66.5 | — |
| R(X_FA_COLD / C_SMALL) | 85.0 | — |

Also report (validation, DL 542): larger steps on top of the strong θ_pre hurt CALIB (eta 3e-5: 0.1705 to 0.1663). Conclusion: K21 and Q1 are robust, not a tau artefact.

**B4. PB100: FL value depends on server data** (θ_pre trained on 100k users only).
- v1 (DL 542, register entries 12, 13): FA_WARM_PB100K − PRE100K = +0.0074 [0.0064, 0.0084]; retention 73.8 to 77.9; the 1.2M θ_pre is 88.8.
- v2 (T3, register entries 18 / 19, X_RESULT 200532ec…): X2 − PRE100K = +0.0091 [0.0081, 0.0101]; X2 − v1 = +0.0017; X2 − PRE (1.2M) = −0.0179; R(X2 / C_FULL) = 79.0.
- Message: the less server data, the more FL adds; it still stays below the strong pretrain.

**B5. Disjoint server band (A1)** — validation only, NDCG@10 on the FL users only. On S3 with a 50 % band, FA_WARM beats FA cold by +26–53 % and matches or beats a central model trained on the FL users alone. On ML-1M (6,040 users) the warm start does not beat cold, even at 50 %. Report the ML-1M exception; do not hide it. Per-dataset numbers: GitHub `results/benchmarks/` (not in file 14).

**B6. Capacity (A2)** — validation and TEST (C_SMALL on TEST). C_SMALL (central D64) reaches 96–110 % of C_FULL. FL at D256 is no better than at D64, on S3 and ML-1M. The federation cost at matched size is about 15–22 %; model size is not the bottleneck.

**B7. Privacy redo** (validation only for DP; E1 reads TRAIN rows).

| Model | DP8 MRR@20 vs non-private | E1 leakage (calibrated AUC) |
|---|---|---|
| Cold FL, tau-fixed | 0.038 vs 0.12: **large cost** | **MATERIAL_LEAKAGE** (0.747); with DP8: PASS (0.502) |
| PB100 warm FL | 0.146 vs 0.153: **small cost** | NO_MATERIAL_LEAKAGE (0.537); with DP8: PASS (0.504) |

Message: E1 leakage appears once FL learns; DP8 closes it; the DP8 cost is small on warm and large on cold. Include the confirmatory K12 / K13 (DP on the frozen recipe, 98.52 / 94.38) beside it, and explain that those were near-free partly because the frozen FL barely learned.

---

## 4. Disclosures and limitations checklist

Write each item in the thesis (tick when done). Items 1–8 are the required new ones; 9 keeps the older mandatory list.

- [ ] **1. Single seed for θ_pre (2026).** Both warm seeds share it; their spread measures FL variance only.
- [ ] **2. FedProx is inert at 1 local step.** The proximal term has no effect when there is one local step.
- [ ] **3. FROZEN arms score 0.0 for out-of-band targets.** Items outside the pretrain band cannot be predicted by arms with frozen item tables. State which tables this affects.
- [ ] **4. Extra exploratory TEST exposures.** Register entries 12–19 are looks at TEST after the confirmatory sitting (entries 10 and 11). Selection pressure on TEST is possible; list each look.
- [ ] **5. ML-1M device-class difference.** ML-1M ran on GPU (A40-FP32-DET); the five other datasets ran on CPU. Ratios are within-dataset only (B16).
- [ ] **6. D1 / B2 confounds.** In the REES46 cold re-run, tau changed together with eta, so "X_FA_COLD vs confirmatory FA" cannot be attributed to tau alone. In the benchmark re-runs, tau changed together with unfrozen items (and a larger budget). Do not say "tau alone explains the gain" without this caveat.
- [ ] **7. amazon2023 is descriptive only** (video_games, baby_products).
- [ ] **8. Sampled vs full-ranking metrics.** Most FL papers use sampled metrics; Spotify Table 1 uses full ranking. Compare only like with like.
- [ ] **9. Older mandatory list** (`09_THESIS_WRITING_GUIDE.md` §2, items 1–17): threat model, DP statement (trusted aggregator, unaccounted tuning), non-DP updates reveal viewed items, PRE's data advantage, C_FULL larger model, CALIB inside MAIN, backbone wording, recipe-qualified wording, VALIDATION to TEST order, post-validation labels, benchmark amendments, C2 history, freeze refusals, K16.

Further points to state in one line each:
- The DP8 / E1 / A1 / A2 results are validation-only unless the optional TEST looks are done.
- The A1 and A2 conclusions rest on validation, not TEST (except C_SMALL's TEST numbers).
- The tau default stays 1e-3 in the code for reproducibility; the registered recipe was not wrong by the rules, it was badly scaled.
- Wording: say "the registered recipe", never "FL retains X %" alone.

---

## 5. Suggested figures and tables

Source CSVs: GitHub `results/rees46/` and `results/benchmarks/` (branches `results/rees46-final`, `results/benchmarks-final`; see `results/README`). Notebook: `notebooks/final_results.ipynb` (branch `docs/final-results-report`; it reads those CSVs). Check the exact CSV file names in `results/README`; I have not listed them here to avoid guessing. Merge the PRs first (`17` §2 step 1).

| # | Figure or table | Content | Source |
|---|---|---|---|
| T1 | Confirmatory REES46 contrasts | table of section A1 | `results/rees46/` retention CSV; notebook |
| T2 | Phase-2 Q1–Q9 | Q values, CIs, reading | `results/rees46/` Phase-2 CSV |
| F1 | Retention by system | K2, K3, K17, K20b, K20c as bars with CIs | notebook |
| T3 | Confirmatory benchmarks | six datasets, C_FULL, FA, PRE | `results/benchmarks/` |
| T4 | Published comparison | our C_FULL vs Spotify Table 1 and eSASRec | `results/benchmarks/` comparison CSV |
| F2 | Tau mechanism | share of updates below tau; drift over rounds | `docs/fedadam-tau.md`; notebook if present |
| F3 | Confirmatory vs fixed-recipe benchmarks | FA vs FA_X vs FA_WARM_X, as % of C_FULL | `results/benchmarks/` fixed-recipe CSV |
| T5 | REES46 tau-fixed contrasts | table of section B3 | `results/rees46/` |
| F4 | Value of FL vs server data | PRE100K, PB100 v1, v2, PRE 1.2M against C_FULL | `results/rees46/` |
| F5 | Disjoint band | warm vs cold by band size (15 %, 50 %), S3 and ML-1M | `results/benchmarks/` band CSV |
| T6 | Capacity decomposition | C_FULL, C_SMALL, FA D64, FA D256 | `results/benchmarks/` |
| T7 | Privacy matrix | DP8 cost and E1 AUC for cold and warm | `results/rees46/` privacy summary |
| T8 | Device summary | INT8, latency, payload | DEVICES.json; device CSV |
| T9 | Exposure register | entries 10–19 with dates and locks | register file; Ch 5 |

If a CSV is missing a column you need, recompute nothing: quote file 14 and say so.

---

## 6. AI-use disclosure note (owner's choice: keep or edit)

> *AI tools (Claude models, and OpenAI Codex for some early code) assisted with code and data analysis during this project, under the author's supervision. The author made all design decisions and is responsible for the content of this thesis.*

Queen's requires a "Statement of Originality and Disclosure of Use of AI". Confirm the final wording, and whether supervisor approval is needed, with the supervisor. The owner may shorten or change this text; it is a suggestion only.

---

## 7. Final checks before submission

- Every confirmatory number matches file 14 or the RETENTION_RESULT file exactly.
- Every exploratory number is labelled and sits in Part B.
- No sentence says "FL cannot help": the supported claim is "FL adds nothing over a strong 1.2M-user pretrain on REES46, and adds a lot when the pretrain is weak or absent".
- The open citation checks from the old guide (§4) are closed: eSASRec author list, proposal FedRec "95 % / 60 %" claim, corrected proposal references.

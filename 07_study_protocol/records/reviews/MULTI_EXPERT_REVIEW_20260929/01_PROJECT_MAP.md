# 01 — PROJECT MAP (multi-expert review, 2026-09-29)

Lead reviewer: Claude Opus 5.5. READ-ONLY review; this folder is the only thing written. No server contact, no git, no RETRO_TEST
row and no downstream RESULT_ROWS value read. Tags: MEASURED (path:line), REPORTED (a project document's own number),
INFERENCE, OPINION. Paths relative to `.local_ai/ppsi_v4/` unless absolute.

## 1. The idea and the question
- **Question (proposal §1, `eda/Privacy-Preserving Shopping Intelligence - Research Proposal 1.pdf`, p.1):** "How much shopping
  recommendation quality can be retained when moving from cloud-based to on-device or federated models?" — framed for
  Microsoft Edge ("how much recommendation quality is lost when shopping data never leaves the device?").
- **Proposal method (§3):** Step 1 centralized upper bound (next category / purchase likelihood / comparable offers; NDCG@10, HR@10,
  AUC); Step 2 federated variants (FedAvg, FedProx, FedBN, personalised FL; communication cost, convergence, quality vs upper bound);
  Step 3 on-device-only (local MF, pre-trained embeddings + local fine-tune, kNN on cached embeddings, rule-based); Step 4 cold start
  (events to parity, few-shot adaptation, category diversity). Contributions (§5): privacy-utility curves, persona analysis, cold-start
  characterisation, bits-vs-quality, browser deployment guidelines (memory, latency, battery), open benchmark. Datasets (§4): Criteo,
  FedScale, MovieLens/Amazon, Taobao, RetailRocket.
- **What the project actually does (agreed deviations, REPORTED in amendment 06D §9g / MASTER_PLAN_V5 §12.2 ch.1):** one task — next-item
  prediction over the full catalogue K = 158,486 (`contracts/DATA_PROTOCOL_LOCK_V2.json` "catalogue.K"); one dataset — REES46
  e-commerce events Oct–Nov 2019 (frozen `rees46_nextitem_v1`, hash 860e5a1c…); primary metric client-macro MRR@20 (NDCG@10, HR@10
  secondary, amendment 04 §4); FedBN excluded by the owner (`contracts/ARCHITECTURE_LOCK.json` "FedBN": "excluded_by_owner"); no AUC
  / purchase task.
- **Owner's top priority (memory + REALISM_AUDIT_20260929 header):** simulate an Edge deployment realistically — (1) cloud/server-only,
  (2) federated (data stays, only model updates leave), (3) small local model on the device — and answer the proposal's questions.

## 2. The three deployments as built (privacy ladder, `contracts/PREG1_AMENDMENT_06B.md` §3)
| Level | Arms | What leaves the device |
|---|---|---|
| L0 cloud train + infer | C_BIG (SASREC_D256_B3, ≈ 84 M params, MAIN 131,072 users), C_FULL (same recipe on 1,348,682 users), C_SMALL (SASREC_D64_B2 ≈ 20.4 M) | raw events |
| L1 cloud train, device infer | C_SMALL shipped as ONNX FP32/INT8; KNN_EMB on cached item table | raw events once (training) |
| L2 federated | FA (FedAvg as tuned by the ladder), FP (FedProx μ 0.01), FA_1024, FA_Q8, PF shared part | model deltas |
| L3 federated + DP | DP ε = 8, ε = 1 (DP-FedAvg, Poisson sampling, RDP, δ 5e-6), DP8_FULL (SHOULD) | clipped + noised deltas |
| L4 FL + local personalisation | PF (personal params), FT_FA | as L2 |
| L5 cloud model + local fine-tune | FT_C | nothing after download |
| L6 device-only | LO, RULE | nothing |
Deployment branch added 2026-09-29 (amendment 06K, DL 206–207): PRE (SMALL recipe on 1,217,610 non-MAIN users) → FA_WARM / FA_WARM_FROZEN.

## 3. Pipeline end to end
1. **Data (CLEAN_V2 / POP 5–100):** REES46 events → representative events, near-duplicate-session schema, censoring classes; splits
   TRAIN 2019-10-01→11-16, VALIDATION 11-16→11-23, RETRO_TEST 11-23→11-30, GRACE 11-30→12-01 (DATA_PROTOCOL_LOCK_V2.json L55-80;
   "VALIDATION and RETRO_TEST are previously inspected", L80). Population: users with 5–100 eligible TRAIN examples = 1,446,986 users /
   28,623,386 examples (L174-178); MAIN = rank band [0, 131,072) (≈ 2.59 M TRAIN decisions), CALIB = [0, 32,768) ⊂ MAIN, U_NEW
   (unseen) bands [1,348,682, …), SPARSE/HEAVY panels 16,384, TRUE_COLD, C_FULL band = first 1,348,682 ranks.
2. **Features/history:** schema S* (`FEATURE_SCHEMA_CLEAN_V2.json`), history policy USER_HISTORY_L50 (last 50 events across sessions,
   strictly causal; L90), pooled bootstrap block (catalogue, transform statistics, θ0) shared by every arm (CLIENT_CONTRACT C5.6).
3. **Backbone contest (G0→G2):** GRU vs SASRec, P2 tuning on CALIB, P3 = 2 candidates × 2 seeds on MAIN, rule T1 on MAIN_VALIDATION
   PRACTICAL_BEST; SASRec won DECISIVE_STABLE (Δ GRU−SASRec −0.00427, CI [−0.00518, −0.00326]); at the 6.0 endpoint GRU > SASRec
   (disclosed) — RESUME_HERE §2e; ARCHITECTURE_LOCK fd201013….
4. **Central references:** C_BIG = the P3 winner runs (best@3.0 and endpoint_6.0); C_SMALL (≤ 170 MB per-visit dense payload,
   amendment 03 §1) via 4-LR mini-ladder on CALIB; C_FULL (06, 06B §6.3, DATA-08).
5. **Federated:** 64 clients/round, 10 % drop-out, B16 × 2 passes, 6,827 rounds = 6.0 EFE ≈ 3 visits/client (amendment 04 §3);
   improvement ladder S0→S1b (local-LR ×8 selected, CALIB 0.028796 vs central ≈ 0.11–0.12) → S2 server optimiser (FedAdam/FedAvgM,
   running) → S3 LR shape + client SGD → S4 passes/batch → S5 FA_X2 → S7 warm start → S8 frozen tables (06F/06G/06I/06J/06K;
   `company/registers/FL_IMPROVEMENT_LADDER.md`).
6. **DP:** S_CAL (median clip norm) → FA_1024 (no noise) → DP8 ×2, DP1; K12/K13 on CONTROLLED_BUDGET (06D §3); DP8_FULL at scale (06I §5).
7. **Communication:** FA_Q8 (int8 stochastic-rounding uploads), bytes accounting (surviving visits charged down + up; dropped visits add their download only — `tools/retention_report.py:1023-1044`, DL 218/221; corrected by expert 8), S8 encoder-only.
8. **On-device:** LO, FT_C, FT_FA (grid selected on UNSEEN_VAL, 06A/06E), PF, KNN_EMB (BIG/SMALL), RULE; panels UNSEEN_HOLDOUT,
   SPARSE, HEAVY, TRUE_COLD; DATA-07 truncated-history Study A/B (T41 negative control).
9. **Device measurements:** ONNX FP32/INT8 of C_SMALL (server-CPU single-thread proxy), INT8 retention K16, browser ONNX Runtime Web
   bench on the owner's PC (HISTORICAL_REFERENCE).
10. **Evaluation:** frozen T3 evaluator (full-K score block, strict FP32, FP64 accumulation), A40 pinned device class (06D §1), client-
    macro MRR@20, retention ratios with paired user bootstrap (1,000, seed 20260926), contrasts K1–K21 (hierarchy K2 primary), VALIDATION
    first read → ALL_METHODS_LOCK → ONE TEST look (06D §2, MASTER_PLAN_V5 §9–§11).
11. **Engineering:** STRICT FP32 determinism, immutable code trees (code_p3 … code_ds3r3), O_EXCL registration files, hashes, suites,
    DECISION_LOG coordination between concurrent sessions.
12. **Write-up:** MASTER_PLAN_V5 §12.2 chapter → artefact map; thesis living reference `.local_ai/thesis_reference/`.

## 4. State at review time (REPORTED, DL 218–225, RESUME_HERE §2n)
S1b selected (×8); S2 CALIB screens running; C_FULL s2026/s2027 at ≈ 2.0 EFE; PRE training; LADDER_S1a MAIN running; device lanes
SEL_FT_C / SEL_LO launched 16:20Z from code_ds3r3; ANA freeze pending; TEST sealed.

## 5. Parts (one expert each)
| # | Part | File |
|---|---|---|
| 1 | Research question, proposal alignment, realistic Edge framing (privacy ladder) | 02_question_framing.md |
| 2 | Data: cleaning, sessions, targets, censoring, 5–100 population, bands, chronological splits, leakage | 03_data.md |
| 3 | Features and history policy (S*, USER_HISTORY_L50, device-computability, pooled statistics) | 04_features_history.md |
| 4 | Backbones and the GRU vs SASRec contest | 05_backbone_contest.md |
| 5 | Central references C_BIG / C_SMALL / C_FULL | 06_central_references.md |
| 6 | Federated training and the improvement ladder | 07_federated.md |
| 7 | Differential privacy | 08_dp.md |
| 8 | Communication efficiency | 09_communication.md |
| 9 | On-device arms and selection panel | 10_on_device.md |
| 10 | Cold start and user strata | 11_cold_start_strata.md |
| 11 | Device deployment measurements (ONNX, browser) | 12_device_measurement.md |
| 12 | Evaluation and statistics | 13_evaluation_statistics.md |
| 13 | Engineering and reproducibility | 14_engineering_repro.md |
| 14 | Thesis write-up, threats, disclosures, bibliography | 15_thesis_writeup.md |

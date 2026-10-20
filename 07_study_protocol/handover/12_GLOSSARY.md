# 12 — Glossary

*Arms are defined in file 01 §5, contrasts in file 01 §6 and file 05 §3. This list gives short meanings and pointers.*

## Project, roles, records

| Term | Meaning |
|---|---|
| PPSI V4 | Privacy-Preserving Shopping Intelligence, version 4 — the single-backbone deployment study (scope `OWNER_SCOPE_V4.md`) |
| Phase 1 / Phase 2 | the pre-registered deployment study (PREG1 + amendments) / the registered follow-up on K21 (PREG2) |
| PREG1, PREG2 | the pre-registration of Phase 1 (amendments 01–07) / of Phase 2 (`contracts/PREG2.md`) |
| DL n | row n of `company/DECISION_LOG.md`; 169b = the second row 169 |
| Ops | the operations lead (an AI agent session run by the owner) |
| Fable | a Claude Fable 5.1 instance used as chief scientist / gate reviewer / owner-proxy |
| REV-… | an independent read-only code review |
| Handoff / packet | a worker's report / its instructions (`company/handoffs`, `company/tasks`) |
| G0, G1, G1b, G2 | early gates: data protocol, code readiness, contest launches, contest selection |
| ARCHITECTURE_LOCK | the backbone lock after G2 (`fd201013…`) |
| ANA freeze | `ANA_SCRIPT_FREEZE_v3.json`: the analysis script + inputs pinned before the VALIDATION read |
| ALL_METHODS_LOCK | arms, checkpoints, contrasts and the TEST plan locked after the VALIDATION read (`04a3cd19…`) |
| P2_METHODS_LOCK / P2_SCRIPT_FREEZE | the Phase-2 equivalents (to be written) |
| TEST_EXPOSURE_REGISTER | hash-chained log of every TEST access (6 entries) |
| O_EXCL | written once with exclusive create; then read-only |
| Endgame phases B, B2, P, D, E, F, G, H | panel reevals, C_FULL s2027 reevals, park smoke dirs, freeze, check-only, the VALIDATION read, the lock, the TEST look (gate / build / final / reeval_dry / reeval / dev / base / onnx / analysis) |
| T0a | one TEST sitting for Phase 1 + Phase 2 (T0b and TEST2-only were refused) |
| S1–S8 (PREG2) | the binding conditions of that sitting (file 03 §6) — not to be confused with ladder stages S0–S8 |
| TEST2 | an optional December-2019 replication; declined (O2 = NO) |
| O1–O9 (PREG2) | Phase-2 decisions: O1 T0a; O2 December data NO; O3 run C2 YES (after YES → NO → YES); O4 grids / ties / cap; O5 second θ_pre seed NO; O6 second dataset NO; O7 DP T = 384 + registered z; O8 devices; O9 second study on the built TEST manifests YES. (Amendment 07 also had options O1–O5, and MASTER_PLAN §0 had O1 / O2 for the GPU index — different lists.) |
| B1–B16 | sections of amendment 07 (B1–B11) and its addenda (B12–B16) |
| Lane | a detached bash job queue on the server; `LANE_STOP` = it stopped on a failure / refusal |

## Data and evaluation

| Term | Meaning |
|---|---|
| REES46 | e-commerce behaviour logs (Oct–Nov 2019) of one multi-category store |
| `rees46_nextitem_v1` | the frozen dataset (hash `860e5a1c…`) |
| MAIN | the 131,072 benchmark users (5–100 eligible TRAIN examples, key ranks [0, 131,072)) |
| CALIB | the first 32,768 MAIN users; the only users whose values are read for tuning |
| PRE band | 1,217,610 non-MAIN pool users assumed to consent to server training |
| TIEBREAK band | MAIN + PRE band = the C_FULL population (1,348,682) |
| UNSEEN_VAL / UNSEEN_HOLDOUT (U_NEW_VAL / U_NEW_TEST, UNEW_*) | never-trained user bands: the device-recipe selection panel / the new-user panel |
| TRUE_COLD, SPARSE, HEAVY | users with 0 / 1–4 / > 100 eligible TRAIN examples (panels) |
| FS band | users of the feature search |
| E2E / RANKABLE | denominator with out-of-vocabulary targets as misses (registered) / only catalogue targets |
| client-macro MRR@20 | per-user mean reciprocal rank within the top 20, then mean over users (primary metric) |
| HR@k / NDCG@k | hit rate (= recall with one target) / normalised discounted cumulative gain |
| Catalogue A | the 158,486 products with global TRAIN support ≥ 5 |
| USER_HISTORY_L50 | the model sees the last 50 events of the user's whole history, across sessions |
| S* | the locked 24-column feature schema (+ output catalogue = 25 channels) |
| EFE | the training-budget unit: 1 EFE = one exposure of every eligible TRAIN example; all arms use 6.0 EFE; 12 marks every 0.5 EFE |
| PB / PRACTICAL_BEST | the strictly best of the 12 marks (gain > 0.0002) |
| CB / CONTROLLED_BUDGET | the 6.0-EFE endpoint (DP: round 384) |
| RULE_FREE | baselines without checkpoints |
| Retention R | 100 × arm / reference, mean of per-seed ratios |
| Paired user bootstrap | resample users with replacement (1,000×; seed 20260926, Phase 2 20261004) |
| Holm band | Holm step-down interval beside the unadjusted CI (06L_C C13) |
| SEED_SENSITIVE | seed spread > half the CI width |
| Harmed share | share of users whose difference is < 0 |
| TOST | two one-sided tests: "equivalent" if the 90 % CI lies inside the margin (K14 / K14w ±1 %; Q1 ±0.0005; Q13 ±1 %) |
| BELOW_DEVICE_HEURISTIC | the CI of Δ(arm − LAST) lies below 0 |
| R* | headroom (arm − LAST) / (C_ref − LAST) |
| REPEAT / NOVEL | targets inside / outside the user's 50-event window |
| CALIB-disjoint | MAIN users not in CALIB (diagnostic) |
| Manifest users vs E2E users | users listed in a VALIDATION manifest (51,211) vs users with ≥ 1 evaluation row (40,764) |
| Register-before-read | every rule hashed before the data it governs is read |
| POST_VALIDATION_EXPLORATORY | anything added after the VALIDATION read |
| CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP), TERTIARY_P2 | Phase-2 claim labels |

## Models, FL and privacy

| Term | Meaning |
|---|---|
| SASREC_D256_B3 / SASREC_D64_B2 | the cloud model (83.9 M params) / the device model (20.7 M) |
| θ0 / θ_pre | registered random initial weights / the pretrained PRE endpoint (`cf08588c…`) |
| C_BIG, C_FULL, C_SMALL, PRE | cloud references (file 01 §5) |
| FA, FP, PF, FA_Q8 | FedAvg-type federated training with the tuned recipe / + FedProx / + personal vector / + 8-bit uploads |
| FA_1024 | the z = 0 twin of DP: Poisson sampling (q = 2⁻⁷, ≈ 1,024 clients / round), 384 rounds, clipping |
| DP8 / DP1 | DP-FedAvg at ε = 8 / 1 (z 0.5642 / 1.3094, δ 5e-6) |
| S_CAL / S_RECORD | clip-norm calibration run / its median client-delta norm S (cold 0.284363) |
| S7 / FA_WARM | warm start from θ_pre, all weights trained |
| S8 / FA_WARM_FROZEN / FA_WARM* | warm start with the item tables frozen and not sent (the chosen warm variant) |
| *_WARM_FROZEN arms | the method set on S8 (FP, PF, FA_Q8, FA_1024, DP8_FROZEN, DP1_WARM_FROZEN) |
| FT_C, FT_PRE, FT_FA, FT_FA_WARM(_FROZEN) | a base model downloaded then fine-tuned on each device (cell FT_C_f0.3_P4) |
| LO | local-only training from θ0 (cell LO_P4) |
| LAST, POP, SKNN, RULE, KNN_EMB | baselines |
| N33 | the cold re-runs of 06N §N3.3 (`FL_T/N33_*`) |
| STOPPED_REALLOCATED | cold runs stopped by 06N, never evidence |
| Ladder S0–S8 | the FL improvement stages (S0 FedAvg … S4 1 pass; S5 FA_X2; S6 personalisation; S7 / S8 warm) |
| FedAdam / FedAvgM / FL_CONST | adaptive server optimiser / server momentum / constant-peak LR shape with warm-up and cool-down |
| m, η_s, τ | client-LR multiplier / server step / FedAdam stabiliser |
| Incumbent / ≥ 0.002 rule / tie band | ladder selection rules (06J) |
| Edge run / EdgeRunsMissing | the extra run when a winner sits on a grid edge |
| K-contrasts | K1–K21, K9b, K9c, K11b, K20b, K20c, K_INIT, P5, K4w, K5w, K12w, K13w, K14w |
| Q-contrasts | Phase-2 Q1–Q14 |
| W-grid (WA / WB / WC / WD), A*, W* | Phase-2 recipe stages and their winners |
| C0 / C1 / C2 | Phase-2 controls: θ_pre / central fine-tune on MAIN / D64 central on PRE + MAIN users |
| P2-T3 | the check that the Phase-2 code reproduces S8's incumbent run bit-identically |
| D1 / D1-ext | the θ-distance diagnostic (DL 407) / per-round update-norm logging in Phase 2 |
| NO_OP / MOVED / DRIFT | Phase-2 classification of a warm run |
| L0–L6 | privacy levels (cloud … device-only) |
| Trusted aggregator | the server that adds the DP noise (central DP) |
| SecAgg, DP-FTRL, LDP | secure aggregation, tree-aggregation DP, local DP (Phase-2 Part B) |
| K16 pass 1 / pass 2 | INT8 accuracy + loaded-host timing / latency-only re-run on a quiet CPU |

## Infrastructure

| Term | Meaning |
|---|---|
| Lobot | the Queen's JupyterHub group server (`group-digi2026-g8`) |
| `lobot.py` | the local helper that runs commands on Lobot over a Jupyter kernel websocket |
| A40-FP32-DET / `NVIDIA_A40-FP32-DET` | the registered deterministic A40 device class (pin `db744ca3…`) |
| A16-FP32-DET | the contest device class |
| CPU-FP32 | benchmark device class (16 threads fixed) |
| SERVER_BROWSER | headless Edge versions on the server (not a consumer device) |
| STRICT | deterministic FP32 mode, independent of co-scheduling |
| Frozen tree `code_*` | an immutable deployed copy of `code/` |
| SYNC_MANIFEST / aggregate | per-file sha list of a tree / its combined hash |
| Tiny-run digests | 7 FL-arm fingerprints used to prove two trees behave identically |
| SMOKE_NOT_EVIDENCE | smoke outputs, never used |
| SUPERSEDED_DEVICE | the CPU ML-1M runs sealed after B16 |
| Wave 1 / Wave 2 | benchmark batches (S3 + ML-1M / Amazon 2023) |
| Track A / Track B | benchmark recipe unchanged / small validation-only grid |
| eSASRec splits | the published leave-one-out splits used for Wave 1 |

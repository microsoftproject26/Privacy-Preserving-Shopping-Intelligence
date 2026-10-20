# Supervisor deck — data and sources

- **Deck:** `company/presentations/PPSI_Supervisor_Deck.pptx` (31 slides: 25 main and 6 appendix). Speaker notes are in Arabic.
- **Generator:** `company/presentations/build_supervisor_deck.js` (pptxgenjs). To rebuild:
  - `npm install pptxgenjs` in a scratch folder;
  - `node build_supervisor_deck.js <out.pptx>`.
- **Written:** 2026-09-29, for the supervisor meeting on 2026-09-30.
- **State captured:** DECISION_LOG up to row 228 (2026-09-29T18:36Z). RESUME_HERE top block §2n. Ladder register as of S1b.
- **Allowed-number policy:**
  - Allowed: the contest (G2) values, the baselines of DL row 132, the registered calibration (CALIB) ladder values, run status and progress, and sizes and costs.
  - Not shown: any downstream main-cohort (MAIN) value, i.e. C_SMALL, federated, device, C_FULL, PRE and ONNX. Not shown: anything from the test week. These appear only as "running — expected <date>".
- **Label key:** MEASURED, REPORTED, REGISTERED-CALIB, ESTIMATE (inference), STATUS.

Paths below are relative to `.local_ai/ppsi_v4/` unless noted. All values were read on 2026-09-29.

## 1. Proposal, framing, scope (slides 1–3)
| Item on deck | Value | Label | Source |
|---|---|---|---|
| Research question (quote) | "How much shopping recommendation quality can be retained when moving from cloud-based to on-device or federated models?" | — | `Privacy-Preserving Shopping Intelligence - Research Proposal 1.pdf` §1 p.3 |
| Proposal datasets | Criteo, FedScale, MovieLens 25M, Amazon Reviews, Taobao, RetailRocket | — | Proposal §4 |
| Proposal tasks and metrics | next category, purchase likelihood, best offers; NDCG@10, HR@10, purchase AUC | — | Proposal §3, Step 1 |
| Proposal FL variants | FedAvg, FedProx, FedBN, personalised FL | — | Proposal §3, Step 2 |
| FedBN not applicable (LayerNorm) | — | — | `company/reviews/REALISM_AUDIT_20260929.md` §F |
| Retention definition | deployment ÷ cloud, with a paired CI | — | GitHub repo `docs/federated-deployment-study.md` "The question" |

## 2. Data and task (slide 4; appendix E)
| Item | Value | Label | Source |
|---|---|---|---|
| Catalogue size | 158,486 products | MEASURED | study description "Data and task"; `runs_p3/g2/G2_RESULT.json` comparison_key.eval_score_block |
| Main cohort | 131,072 users; 2,587,776 training decisions (≈ 2.59 M) | MEASURED | `REALISM_AUDIT_20260929.md` A1 (citing DATA_PROTOCOL_LOCK_V2 L247–256); `MULTI_EXPERT_REVIEW_20260929/01_PROJECT_MAP.md` §3 |
| Full pool for the realistic cloud | 1,348,682 users / 26,677,600 decisions (shown as 1.35 M) | MEASURED | DL row 162 area; MASTER_PLAN_V5 addendum A |
| 5–100 population | 1,446,986 users | MEASURED | `REALISM_AUDIT` A1 |
| Calibration users | 32,768 (subset of main) | — | project map §3; ladder register header |
| Split dates | train to 15 Nov; validation 16–22 Nov; test 23–29 Nov 2019 | — | study description "Data and task" |
| History window | at most 50 prior events | — | MASTER_PLAN_V5 (USER_HISTORY_L50); review synthesis E wording |
| Black Friday in the test week | disclosed; sub-period reported | REGISTERED | `contracts/PREG1_AMENDMENT_06L_C.md` D4 |

## 3. Fairness (slide 5)
| Item | Value | Source |
|---|---|---|
| Bootstrap | 1,000 user-level paired resamples | `G2_RESULT.json` delta_bar_ci.n_resamples; study description |
| Seeds | 2 for the main methods (DP ε = 1 and the full-pool cloud: 1 seed) | study description "How fairness is kept"; DL 225 |

## 4. Backbone contest (slides 7–8; appendices A and C)
| Item | Value | Label | Source |
|---|---|---|---|
| SASRec (D256, 3 blocks), best checkpoint, 2-seed mean | 0.16121 (deck 0.1612); per seed 0.161196 / 0.161221 | REPORTED (G2) | `G2_RESULT.json` between_families.cohort.family_mean / per_seed |
| GRU (D128), best checkpoint, 2-seed mean | 0.15694 (deck 0.1569); per seed 0.157141 / 0.156733 | REPORTED (G2) | same |
| Δ GRU − SASRec | −0.004271, CI95 [−0.005180, −0.003261] | REPORTED (G2) | `G2_RESULT.json` delta_bar_ci |
| NDCG@10 | SASRec 0.19668, GRU 0.19105 | REPORTED | `G2_RESULT.json` ndcg10_family_mean |
| Validation size | 40,764 users, 519,714 decisions | MEASURED | `G2_RESULT.json` delta_bar_ci.n_users / n_decisions |
| CALIB-disjoint check agrees | −0.004103, no flip | REPORTED | `G2_RESULT.json` diagnostic; DL 129 |
| Fixed endpoint (6.0) | GRU 0.156937 vs SASRec 0.152724 | REPORTED | DL 129 (within-family endpoint means) |
| Endpoint Δ GRU − SASRec | +0.004213, CI95 [+0.003283, +0.005216] | REPORTED | DL 138. `runs_p3/g2/G2_CONDITION5_DIAGNOSTICS.json` is not in the local copy; the DL row is the source |
| Per-run values (appendix C) | GRU128 0.15714 / 0.15673; GRU256 0.15412 / 0.15345 (best = endpoint); SASRec B3 0.16120 / 0.16122 (best) and 0.15287 / 0.15258 (endpoint); SASRec B2 0.16059 / 0.16006 and 0.15086 / 0.15011 | REPORTED | DL 132; G2_RESULT per_seed; DL 129 |
| Parameters | cloud 83,881,764; GRU128 20,962,210; device 20,710,048 | MEASURED | DL 131 (ARCHITECTURE_LOCK) |

## 5. Sizes and bytes (slide 9)
| Item | Value | Label | Source |
|---|---|---|---|
| Cloud-size model per FL visit | 671,054,112 B ≈ 671.1 MB | MEASURED | DL 131 |
| Device model per FL visit | 165,680,392 B ≈ 165.7 MB (82.84 down + 82.84 up) | MEASURED | DL 131; `REALISM_AUDIT` B12 |
| 8-bit uploads per visit | ≈ 103.6 MB (82.84 down + 20.71 up) | computed from the parameter count | `REALISM_AUDIT` B12 / project map; my arithmetic |
| Frozen item tables | 265,162 uploaded params ≈ 1.06 MB each way (deck bar 2.1 MB = both ways); 81.78 MB one-time table download | to be re-measured | `PREG1_AMENDMENT_06L_C.md` C8; review synthesis §3.2, B3 |
| Device download | 82.8 MB FP32; ≈ 21 MB INT8 | ESTIMATE until benched | `REALISM_AUDIT` C1; review synthesis E9 |
| Item tables share | 98 % of parameters | — | `REALISM_AUDIT` / DL 192 |
| Device cap | 170 MB per visit | registered | study description "Models" |

## 6. Comparison matrix (slide 10)
- Levels L1–L6: `contracts/PREG1_AMENDMENT_06B.md` §3.
- Placement corrections (the fine-tune rows, the pretrained branch, LAST as device-only, RULE as published aggregate): `PREG1_AMENDMENT_06L_C.md` C1.
- Deployment branch: `PREG1_AMENDMENT_06K.md`; DL 206–207.

## 7. Federated learning (slides 12–14; appendix B)
| Item | Value | Label | Source |
|---|---|---|---|
| S0 (as registered, ×1) | 0.023473 | REGISTERED-CALIB | `company/registers/FL_IMPROVEMENT_LADDER.md` |
| S1a grid ×0.5 / ×1 / ×2 | 0.020263 / 0.023473 / 0.025562; ×2 vs ×1 +8.9 % | REGISTERED-CALIB | ladder register; DL 181 |
| S1b grid ×4 / ×8 / ×16 | 0.027570 / **0.028796** / 0.022909; +12.7 % vs S1a; +22.7 % vs S0 | REGISTERED-CALIB | ladder register; DL 218 |
| Matched central on CALIB | ≈ 0.11–0.12 | ESTIMATE | ladder register "Context"; DL 181 |
| FL ≈ 25 % of the cloud (CALIB) | 0.0288 / ≈ 0.115 | ESTIMATE | review synthesis §1 |
| LAST ≈ 44 % of the cloud | 0.0711 / 0.1612 | REPORTED (baseline + contest) | DL 132; review synthesis §1 |
| Diagnosis items | no server optimiser; fresh client AdamW (≈ 4 steps); LR shape wastes ≈ half the budget; LR too small; random-init tables | — | `REALISM_AUDIT` §0; `FL_STRONGEST_LADDER_RESEARCH_20260929.md` §1.2; 06L_C C6 wording |
| S2 status | 5 screens launched 2026-09-29 14:44Z, plus 2 FedAvgM screens at 17:56Z; expected done ≈ 30 Sep | STATUS / ESTIMATE | DL 218, 221 (finish ≈ 21–23Z), 226 |
| Stop rules | gain ≥ 0.002; tie 0.0005 → simpler; sequential S2 → S3 → S4 | REGISTERED | `PREG1_AMENDMENT_06J.md`; DL 204, 222 |
| Research estimates (notes only) | S2 0.045–0.065; S3 0.055–0.075 | ESTIMATE | ladder research §3 |
| PRE cohort | 1,217,610 users / 24,089,824 decisions | MEASURED | DL 207 |

## 8. Privacy (slide 16; appendix D)
| Item | Value | Source |
|---|---|---|
| ε levels and noise multipliers | ε 8 → z 0.5642; ε 1 → 1.3094; ε 4 → 0.7244 (optional) | DL 169b / 170 (independently confirmed) |
| δ; adjacency; threat model | 5e-6; client-level add/remove; honest-but-curious browser vendor | `PREG1_AMENDMENT_06L_C.md` C2; 06B §3 |
| Sampling | Poisson, ≈ 1,024 clients per round, 384 rounds | project map (via review summary); DL 219 (T = 384) |
| Leak statement; no secure aggregation | — | 06L_C C1; DL 192 |

## 9. Deployment branch, device, cold start (slides 17, 19, 20)
| Item | Value | Label | Source |
|---|---|---|---|
| PRE running | since 2026-09-29 13:03Z | STATUS | DL 216 |
| Device lanes launched | 2026-09-29 16:20Z | STATUS | DL 224 |
| Unseen holdout | 12,862 users | MEASURED | `REALISM_AUDIT` D2 |
| True cold | 381,295 users | MEASURED | `REALISM_AUDIT` D2 |
| Panels | 16,384 each (sparse, heavy) | — | project map §3; 06D |
| Truncation h | {0, 1, 5, 10, 20, 50}; persistence rule | REGISTERED | project map; 06L_C C10 |
| ONNX training packages deprecated | inference only in the browser | — | `REALISM_AUDIT` (on-device) |

## 10. Reviews (slide 21)
| Item | Value | Source |
|---|---|---|
| Overall | 7/10 (mean 6.96); 8–8.5 if the MUST list is done before the first read | review synthesis §1 item 9; DL 226 |
| Part scores | Data 8, Eval 8, Features 7, Contest 7, Central refs 7, DP 7, Engineering 7, Framing 6.5, FL 6, Comms 6, On-device 6, Cold start 6, Write-up 6, Device measurement 5 | review synthesis §2 |
| Fixes | Gate A registered (DL 226); Gate B done (DL 228); Gate C registered (06L_C, DL 227) | DL 226–228 |

## 11. Timeline (slides 22, 25) — all future dates are ESTIMATES
| Item | Value | Source / basis |
|---|---|---|
| C_FULL s2026 / s2027 | both at 2.0 of 6.0 EFE at 16:05Z → end ≈ 30 Sep | DL 221; rate from DL 175 (start 08:45Z) |
| PRE | 1.0 EFE at 16:05Z → end ≈ 30 Sep | DL 221 |
| One FA-class MAIN run | ≈ 10.7 h per process | MASTER_PLAN_V5 addendum B (MEASURED rate) |
| First validation read | ≈ 2–3 Oct | ladder research §6 planned 1–2 Oct; slipped ≈ 1 day because S2 runs slower co-scheduled (DL 221: planned S2 end 18Z, now ≈ 21–23Z, plus 2 extra runs from 17:56Z) |
| Lock → single test look | ≈ 4–5 Oct | MASTER_PLAN_V5 §11 (TEST look ≈ 1–1.5 days) |
| Final tables | ≈ 6–7 Oct | MASTER_PLAN_V5 §12 (≈ 1 day) |
| Cut order | as listed | MASTER_PLAN_V5 §13 |

## 12. Earlier phase (context only; not on slides)
- README G1 gate: 388,789 clients; sequence model gain +0.0204 macro over the model-free rule.
- Source: GitHub repo `README.md`.
- Earlier-phase test-week values are deliberately left out of the deck.

## Update procedure (when the owner says "update it")
1. Re-read `company/RESUME_HERE.md` (top block), the `company/DECISION_LOG.md` tail after row 228, and `company/registers/FL_IMPROVEMENT_LADDER.md`.
2. Update only what changed:
   - new ladder rows (slide 13 chart and cards, slide 14 status chips);
   - run status (slides 14, 17, 19, 22, 25);
   - dates (slides 22, 25).
   Edit the values in `build_supervisor_deck.js`. Rebuild to the SAME file name.
3. Add a line to the update log in the notes of the last slide (glossary, appendix E), and a row below.
4. Never add a MAIN downstream or test-week value unless the pre-registered read has happened and is recorded in the DL.

| Version | Date | Changes |
|---|---|---|
| v1 | 2026-09-29 | First version (DL ≤ 228; ladder ≤ S1b; S2 running) |
| v1-simple | 2026-09-29 | Short, comparison-focused version added on the owner's request (same facts, same state) |

## Short version (comparison-focused)
- **Files:** `PPSI_Supervisor_Deck_Simple.pptx` (+ `.pdf`), 15 slides. Generator: `build_supervisor_deck_simple.js`.
- **Numbers:** it reuses only numbers already listed above. There are no new sources.
- **Where each number sits:**

| Slide | Content | Numbers used (see sections) |
|---|---|---|
| 2 | three deployments side by side | parameters 83.9M / 20.7M (§4) |
| 3 | four headline comparisons + the floor | 131k / 1.35M users (§2) |
| 4 | comparison board: every method, compared with what, status | cloud reference 0.1612 (§4); baselines 0.0711 / 0.1114 / 0.0194 (§4, appendix A of the full deck); run status (§7, §9, §11) |
| 6 | SASRec vs GRU under both checkpoint rules | §4 |
| 7 | models vs simple rules (44 %, 69 %) | §4; 0.0711 / 0.1612 and 0.11138 / 0.1612 |
| 8 | megabytes per visit | §5 |
| 9 | federated vs cloud early warning (≈ 25 % / ≈ 44 %) | §7 |
| 10 | federated before vs after tuning | §7 |
| 11 | no DP vs ε = 8 vs ε = 1 | §8 |
| 13 | proposal vs now | §1 |
| 14 | four milestones | §11 |

- **Updating:** update both generators when the owner says "update it", unless he names only one version.

## Numbers version (work done, earlier phases, why we changed)
- **Files:** `PPSI_Supervisor_Deck_Numbers.pptx` (+ `.pdf`), 23 slides. Generator: `build_supervisor_deck_numbers.js`.
- **Owner request (2026-09-29):** mostly numbers; earlier experiments; why we changed; next-category results; our numbers against published numbers, with a reason beside each; feature engineering.
- **Current-study numbers:** unchanged from the sections above.
- **Earlier-phase numbers:** shown only as validation-week values, never TEST-week values. The old test week is the same calendar week as the sealed test of the current study.
- **Earlier-phase test values** (SASRec 0.1705 vs GRU 0.1687) appear only in the speaker notes, and are marked "read once in that phase".

| Item | Value | Source |
|---|---|---|
| Next-category data | REES46 Oct 2019; slice of 97,279 users; 588 categories | GitHub `docs/MVP_REPORT_AR.md` §1; `docs/evidence/s2-ds-01/s2_ds_01_result.json` |
| Next-category MRR@20 (category-change decisions, validation, macro) | popularity 0.1049; transition table 0.3143; GRU 0.3474 (3 seeds; spread 0.0007); gain +0.0331 | `MVP_REPORT_AR.md` §2 |
| No-learning share of the overall metric | ≈ 82 % | GitHub `docs/decisions/ADR-001-evaluation-protocol.md` (Decision 3) |
| Federated pilot, next category (1,000 clients, raw convention) | centralised 0.11268; FedAvg 0.09959; retention 88.38 % | GitHub `docs/evidence/mvp/mvp-t1-003/comparison.v1.json`, `pilot_quality_retention.v1.json` |
| Purchase PR-AUC | rule 0.0760 → model 0.1418 (+0.0661) | `MVP_REPORT_AR.md` |
| Why we moved to next product | No published next-category results on REES46 to compare with. This is the owner's statement; no file says it verbatim. The next-item task is "more compatible with an official baseline". | owner (2026-09-29); `thesis_reference/sources/local/S13_06_PROPOSAL_AMENDMENT_AND_THESIS_EVIDENCE_AR.md` table A; `PREG1_AMENDMENT_06D.md` §9(g) |
| Published benchmark | GRU4Rec official REES46 page (Hidasi & Czapp, RecSys 2023): Recall@20 0.5291, MRR@20 0.2003, per-event | `thesis_reference/sources/local/S270_BENCHMARK_COMPARABILITY.md`; `S298_LITERATURE_REGISTER.json` card R05; `thesis_reference/chapters/21_research_comparison.md` §21.3 |
| Ours, per-event, validation-confirm week, full data | SASRec micro R@20 0.431504, micro MRR@20 0.154173; GRU 0.424205 / 0.153140 | `.local_ai/sasrec_v1/FINAL_SASREC_REPORT_EN.md` §6.1 |
| Gaps shown | R@20 −0.098 (0.5291 − 0.4315); MRR@20 −0.046 (0.2003 − 0.1542) | computed |
| Reasons for the gap | full 158,486-item catalogue, with out-of-vocabulary targets counted as misses; next-week vs last-day split; less tuning; population and filters. The gap was not decomposed by ablation. | S270; chapter 21 §21.4 |
| Earlier compute | 2× A16 (15.4 GiB each); 12.8 GiB free disk | `company/reviews/G0_EVIDENCE_PACKET.md` §B |
| SASRec training cost | 5 epochs, ≈ 20.7 GPU-h | `FINAL_SASREC_REPORT_EN.md` §9 |
| Now | 2× A40 server | MASTER_PLAN_V5 addendum V5.1; DL 169 |
| Earlier-phase central models (validation-confirm, macro) | POP 0.020168; transition 0.148740; GRU 0.168988; SASRec 0.171339 | `FINAL_SASREC_REPORT_EN.md` §6.1 |
| SASRec repair (validation-selection week, seed 2026) | failed 0.11791 → repaired 0.181421 → cooldown 0.186298; seed gap 0.000216 | same, §§3–4 |
| Full-data size | 52.76 M training decisions; 40,180,608 loss examples | `G0_EVIDENCE_PACKET.md` §C |
| Raw users | 2,694,998 warm; median 6; max 11,628; 473,402 with 1 example | `G0_EVIDENCE_PACKET.md` §C |
| Earlier features: 58 inputs | 6 tokens, 19 numeric, 15 quality flags, 12 user-context, 6 masks | `thesis_reference/chapters/15_feature_dictionary.md` §15.1; execution reference §2.6 |
| Current features: 24 inputs | 5 token, 7 numeric, 2 flag, 6 user-context, 4 mask | DL rows 39, 52–54; review part `04_features_history.md` §1.1 |
| Feature selection path | 53 → 30 (statistics) → grouped search | same |
| Computable on the device | 21 of 24 (3 local event + 18 local history) | review part 04 |
| Feature probes (Δ macro MRR@20; 10k users, 112,365 decisions) | all side −0.046669; item age −0.024654; price −0.009761; main category −0.006262; brand −0.005687; cold user −0.004068; previous gap −0.003619; calendar +0.000660 | `.local_ai/feature_opt_v1/FINAL_FEATURE_OPT_REPORT_EN.md` §3 |
| Feature decision | best candidate +0.000262 vs threshold 0.002 → KEEP_CURRENT; lean parity 0.998683 < 0.999 | same; thesis reference STATE |

| Version | Date | Changes |
|---|---|---|
| v1-numbers | 2026-09-29 | First numbers version, with the next-category, paper-comparison and feature slides |

## FINAL version (the one to present)
- **Files:** `PPSI_Supervisor_Deck_Final.pptx` (+ `.pdf`): 24 main slides, then a divider and 2 backup slides (27 slides in all). Generator: `build_supervisor_deck_final.js`.
- **Owner feedback (2026-09-29):** too many numbers in the other versions. Base it on the full deck (the best one), trimmed. Keep only numbers that help. Say plainly that the weak federated result came from our own setup: federated training started from an untrained model with untuned settings, not from a pretrained one.
- **Numbers kept on the main slides:**
  - next category: 0.10 / 0.31 / 0.35, and ≈ 82 %;
  - paper vs ours: 0.53 / 0.43 and 0.20 / 0.15;
  - features: 58 → 24, and 21 of 24 on device;
  - data: 158,486, 131,072 and 1.35 M;
  - contest: 0.161 / 0.157 / 0.153;
  - sizes: 84 M / 21 M, and 671 / 166 / 104 / 2 MB;
  - federated: ≈ 25 % and +22.7 %, with 0.023 → 0.026 → 0.029 in the ladder chart;
  - DP: ε 8 and 1;
  - realistic branch: ≈ 1.2 M opted-in users, ≈ 1 MB each way;
  - review: 7/10 → 8–8.5;
  - timeline dates (estimates).

  All come from the sections above.
- **Rounding:** values are rounded for readability (for example 0.1612 → 0.161, and 165.7 MB → 166). Exact values are in the sections above.
- **Federated framing** ("our first setup was wrong"):
  - Causes: random initialisation of the item tables; the cloud's LR schedule reused; no server optimizer; the client optimizer reset every visit.
  - Sources: `REALISM_AUDIT_20260929.md` §0; `FL_STRONGEST_LADDER_RESEARCH_20260929.md` §1.2 (warm start is what production systems do).
  - Review synthesis §1: "cause is the recipe, not a bug".

| Version | Date | Changes |
|---|---|---|
| final v1 | 2026-09-29 | Built from the full deck plus the owner's additions; fewer numbers; federated framed as our setup mistake |
| final v2 | 2026-09-30 08:40Z | Update from DECISION_LOG rows 229–251 and the ladder register; details below |

### final v2 — what changed, with sources (all read 2026-09-30)
| Slide | Change | Source |
|---|---|---|
| 15 | "What it caused": ≈ 25 % → **≈ 20 %**. This describes the FIRST setup (S0 ≈ 20 % of matched central, INFERENCE; the old 25 % was the state after S1b). "What we changed": tuning **+46 %** so far (learning rate, then server optimizer); pretrained model ready | ladder register (S0 context line; S2 row); DL 249, 251 |
| 16 | Chart is now start 0.023 → client LR 0.029 → **+ server optimizer (FedAdam) 0.034**; title "+46 % so far". Status: server optimizer done; LR-shape × client optimizer running; pretrained start next | ladder register S2 row (FedAdam η0.01 ×1 **0.034247**, +18.9 % vs S1b, +45.9 % vs S0); DL 247 (the τ-edge confirmation run is still running before FL_TUNING_S2.json); DL 238, 247 (S3 cells running or done) |
| 16 notes | Training-curve answer: FedAdam peaks mid-budget, then declines as the client LR decays. S3 tests a constant LR. Labelled INFERENCE | ladder register "Training curves" and its correction; DL 250–251 |
| 18, 22, 24 | Pretraining finished its budget | DL 251 ("PRE finished its budget"); DL 249 (STATUS efe 6.5) |
| 22, 24 | Full-pool cloud: seed 2027 done 07:15Z; seed 2026 at 6.0 EFE → "finishing" | DL 248, 249 |
| 22, 24 | Device selections registered (FT_C cell, LO cell); LO MAIN s2026/s2027 running since 20:51Z | DL 231, 232 |
| 22 | Read dates unchanged (estimates) | no new schedule information in DL 229–251 |

**Consistency notes for the owner:**
- **Large-model depth:**
  - DL 236 (GitHub doc correction) says the large model has 2 blocks, not 3.
  - DL 124 and 129 call the recipe "D256_B3".
  - The FINAL deck gives no block count, so it is unaffected. The older full, simple and numbers versions say "3 blocks"; check this before using them.
- **Deck guides:** `DECK_GUIDE_AR.md` and `DECK_SCRIPT_EN.md` (written by the Ops session, DL 233) describe final v1. Slides 15, 16, 18, 22 and 24 changed in v2.

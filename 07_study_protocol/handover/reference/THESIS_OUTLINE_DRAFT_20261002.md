# Thesis outline: Privacy-Preserving Shopping Intelligence via Federated / On-Device Learning

Draft 2026-10-02. For each section: what it will contain, and the source that backs it (DL = decision-log row; "06x" = PREG1 amendment; "07" = amendment 07 DRAFT; "BM-00" = company/research; "EXEC" = CLAUDE_EXECUTION_REFERENCE_EN section; "HO" = team handover section). Sections marked [DRAFTED] exist in this folder. Sections that need results are marked [RESULT PENDING]; nothing in this outline states a MAIN, VALIDATION or TEST value.

## Chapter 1. Introduction
- 1.1 Motivation: on-device shopping assistance in a browser, privacy as a product feature; the gap between generic FL benchmarks and real shopping telemetry. Source: proposal sections 1 and 7.
- 1.2 Research question, verbatim: "How much shopping recommendation quality can be retained when moving from cloud-based to on-device or federated models?" and the decision that there is no pass/fail threshold. Source: proposal section 1; DL 86; 04 section 1.
- 1.3 The deployment framing: cloud-only, federated (weights only), small local model; the pretrained (warm) line as primary and the random-start line as ablation. Source: 06N; DL 281-282; EXEC 14.22.
- 1.4 Contributions (stated as questions answered, not as results): a retention ladder from cloud to device, a tuned federated recipe for sequential recommendation, the warm-start deployment branch with frozen item tables, a privacy ladder with a correct accountant, a pre-registered evaluation protocol with a sealed test. Source: proposal section 5; HO section 1.
- 1.5 Departures from the proposal (task, datasets, metrics, FedBN dropped, thresholds dropped) and the reasons. Source: HO section 2; 06A section 9. (Table drafted in Chapter 3, Section 3.2.)
- 1.6 Thesis structure and a note on the AI-use statement (pending: supervisor pre-approval, HO section 3 question 4; review item E).

## Chapter 2. Background and Related Work
- 2.1 Federated learning fundamentals: FedAvg [McMahan 2017], FedProx [Li 2020], SCAFFOLD [Karimireddy 2020], server optimisers FedAdam / FedAvgM [Reddi 2021; Hsu 2019], why FedBN does not apply to layer-normalised models [Li 2021]. Source: proposal section 2.1; 06A section 9(c); 06G.
- 2.2 Federated recommendation: FCF [Ammad-ud-din 2019], FedRec [Lin 2020], FedeRank [Anelli 2021], PFedRec [Zhang 2023], federated sequential recommendation (FELLAS [Yuan 2025], CF-FedSR [Luo 2022], FCUCR, LUMOS); their datasets and protocols as context-only comparisons. Source: proposal section 2.2; BM-00 section 3.
- 2.3 Sequential recommendation: GRU4Rec [Hidasi 2016], SASRec [Kang 2018], the strong SASRec reproductions with full cross-entropy and their splits [Tikhonovich 2025], the finding that simple transition models are competitive on standard benchmarks [Petrov 2026], and the limits of sampled metrics [Krichene 2020]. Source: BM-00 sections 1 and 3; BENCHMARK_PLAN sections 13-14.
- 2.4 Federated initialisation and pretraining: Nguyen 2023; Chen 2023; Xu 2023; practice in deployed systems (real-world practice audit). Source: 06N header; EXEC 14.21 (audit, REPORTED).
- 2.5 Differential privacy for federated learning: DP-FedAvg [McMahan 2018], moments / Renyi accountants [Abadi 2016; Mironov 2017], sampling and adjacency [Wang 2019]; why weights-only is data minimisation and not privacy. Source: HO section 10; 06C; DL 150, 192.
- 2.6 On-device machine learning: compression and quantisation, ONNX in the browser, transfer from pretrained embeddings, cold-start. Source: proposal section 2.3; 04 section 3; 06E section 3.
- 2.7 Gap statement: evaluation on real shopping logs, full-catalogue ranking, a pre-registered protocol, the deployment (pretrained) branch.

## Chapter 3. Methodology [DRAFTED: chapter3_methodology.md]
- 3.1 Problem framing: three architectures, retention, LAST anchoring. Source: 06N; DL 86, 132, 192, 281-282.
- 3.2 Departures from the proposal (table). Source: HO section 2.
- 3.3 Dataset (REES46 v1 frozen), cohorts, chronological split, roles. Source: HO section 4; EXEC 2; DL 31, 50, 151, 162, 207, 234.
- 3.4 Models: SASRec family sizes, GRU briefly, payloads. Source: ARCHITECTURE_LOCK; HO section 5; DL 129-131, 138, 257, 276.
- 3.5 Federated setup: shared knobs; ladder S0 to S4 and its selected point; warm start from theta_pre; S7 / S8; FA_WARM* rule; warm method set (FP / PF / FA_Q8). Source: DL 249, 256, 267, 271, 285-287, 292; 06N; 06P.
- 3.6 Differential privacy: accountant, Poisson sampling correction, clipping S record, epsilon 8 and 1, scale caveat, optimiser sensitivity, threat model, logging rules. Source: 06C, 06D section 3, 06L_C, 06L_E; DL 119, 150, 170, 279, 300.
- 3.7 On-device fine-tuning and device-only arms: FT_C, LO, FT_PRE, FT_FA, FT_FA_WARM*; cold start; device measurement. Source: HO section 11; DL 231, 232, 255.
- 3.8 Pre-registration discipline: amendments, sealed MAIN, one TEST look; the 06O descriptive table. Source: HO section 7; 06D section 2; 06O; DL 283-286, 293.
- 3.9 Metrics and the contrast hierarchy: primary pair {K20b, K21}; Holm family; tertiary; descriptive. Source: 06N N1; 06P P1-P7; 06L_C C13; DL 317.

## Chapter 4. Experimental Setup [DRAFTED: chapter4_experimental_setup.md]
- 4.1 Compute: two A40, device classes, determinism, rates, CPU benchmarks. Source: HO section 8; DL 139, 158, 163, 169, 281, 294, 304.
- 4.2 Arm list and seeds; execution order; analysis chain. Source: HO section 6.2; 06N N3; DL 292-319.
- 4.3 Metrics summary. Source: Chapter 3.9.
- 4.4 External benchmarks: eSASRec splits, inner validation, evaluator rules, published comparison set, claim rule, Tracks A / B, locks, dropped datasets. Source: 07 DRAFT B1-B8; BM-00; BENCHMARK_PLAN sections 11-14; DL 296-313, 318.
- 4.5 Planned tables and figures with [RESULT PENDING] slots.
- 4.6 Threats to validity registered before the reads. Source: 06E section 2; review disclosures; 06D section 3.

## Chapter 5. Results [RESULT PENDING; one section per registered question]
- 5.1 Context already published or descriptive: the backbone contest (DL 129, 138), baselines (DL 132), the CALIB ladder (DL 249, 256, 271), the S7 / S8 selection (DL 292), the 06O descriptive table (DL 293).
- 5.2 Primary pair: K20b and K21 for the deployment branch. [RESULT PENDING] Source: 06P P1.
- 5.3 Retention across architectures: C_SMALL, FA, LO, FT_C, FT_PRE, FT_FA, FT_FA_WARM*, against C_BIG and C_FULL, anchored on LAST. [RESULT PENDING] Source: 06P; 06L_C C3.
- 5.4 The value of the pretrained start: K_INIT, K2, S8 - S7. [RESULT PENDING] Source: 06P P2-P3.
- 5.5 Federated variants and communication: FP, PF, FA_Q8, bytes per visit, quality-bytes frontier. [RESULT PENDING] Source: 06P P4; 06D section 6.
- 5.6 Privacy cost: FA_1024, DP8, DP1 at CONTROLLED_BUDGET; clipped-fraction proxy; DP_OPT_SENS. [RESULT PENDING] Source: 06D section 3; 06P P4; 06L_E.
- 5.7 On-device adaptation and cold start: FT gain, harmed-client share, support bins, events to 90 %/95 %. [RESULT PENDING] Source: 06P P5; 06A section 2; DATA-07.
- 5.8 Device cost: ONNX latency, memory, INT8. [RESULT PENDING] Source: 06E section 3; handover section 11.3.
- 5.9 External benchmarks (Track A and Track B), with the claim rule. [RESULT PENDING] Source: 07 B3-B8; BM-00 section 1.
- 5.10 The single TEST look, and the shift indicator (Black Friday). [RESULT PENDING] Source: 06D section 2; 06B section 7(b).

## Chapter 6. Discussion
- 6.1 What the pretrained line changes and what it assumes (consent for a server-side band). Source: 06N; BENCHMARK_PLAN section 12.
- 6.2 What federated weights reveal: data minimisation against privacy; the role of DP and its simulated scale. Source: 06I section 7(a); 06D section 3; 06L_C C2.
- 6.3 Why the as-registered federated recipe understated federated quality (the five diagnosed mistakes) and what the ladder teaches. Source: HO section 9.1; DL 192.
- 6.4 Where the benchmarks agree with, and differ from, REES46. Source: 07 R8; BM-00.
- 6.5 Practical guidelines for a browser deployment: payload cap, frozen tables, INT8, fine-tuning. Source: proposal section 5; HO section 5.2.
- 6.6 Limitations: user-level restriction (5 to 100 events), cloud recipe not re-tuned at scale, no secure aggregation, CALIB subset of MAIN, public deterministic DP seed, single dataset for the main study, partition fence at TEST. Source: HO sections 4.3 and 10; 06D section 3, 06E.

## Chapter 7. Conclusion and Future Work
- 7.1 Answer to the research question, stated with retention and its intervals. [RESULT PENDING]
- 7.2 Future work: a large teacher on opted-in users distilled into the small model before federated training (FedDF / FedGKT style; DL 276); DP at the full-pool scale (06I section 5); secure aggregation; Wave-2 benchmarks; a December holdout; real-device measurement beyond the browser proxy. Source: DL 276; 06L_C C16-C17; 07 Wave 2.
- 7.3 Bibliography and appendices: the amendment list; the contrast file; the arm list; code-tree digests; incident log (HO section 13); AI-use statement.

---

## Gaps and inconsistencies found in the sources (to resolve before the draft is final)

1. **Holm family size.** 06P P6 states "Holm family (secondary, 19)" but names 18 identifiers (K1, K2, K3, K8, K9b, K9c, K11b, K12, K17, K20, K20c, K_INIT, P5, K4w, K5w, K12w, K13w, K14w). The count or the list needs to be reconciled when ANA_CONTRASTS_v4.json is built and its sha256 registered.
2. **Amendment 07 is a DRAFT.** Its B1 to B8 sections have overlaps (B7 supersedes B1 to B6; B8 supersedes B3 and R5); the benchmark chapters are PLAN until registration (DL 318 states registration waits for the BM-02 review).
3. **RICH variants and Track-B scope.** BENCHMARK_PLAN sections 4 and 11 include RICH (side-channel) variants and Taobao; the owner's decision at section 14 (DL 302) removes everything not done on REES46, and the 07 draft is ID-only. The thesis must state ID-only for benchmarks unless a later amendment adds RICH.
4. **S7 / S8 on small benchmarks.** The 15 % band covers only 66 to 67 % of items on Beauty, Sports and Toys (BM-00 / 07 R9). The expected S8 disadvantage is a PLAN-time expectation; the registered rule decides.
5. **Selection-panel values** for FT_C and LO (UNSEEN_VAL) are deliberately not quoted anywhere (HO section 11.1); the thesis can quote them only from the selection files after the owner agrees.
6. **Server ONNX latency run and the cold-start Study A / B server receipts** are not clearly recorded in the decision-log rows read (HO section 11.3); verify on the server before writing Chapter 5.7 and 5.8.
7. **Heading of the proposal reference list.** The handover records that the proposal's reference [5] conflates two papers, [6] has a wrong volume, [8] is cited as Luo et al. CIKM 2022 against PFedRec (Zhang et al. IJCAI 2023), and [12] was not found (HO section 2; 06B section 7(a)). The thesis bibliography must use the corrected entries.
8. **The proposal's citation of FedRec "95 % of centralized accuracy with 60 % of the communication cost"** (Lin et al. 2020) is a claim in the proposal that was not re-verified in the sources read; do not repeat it without checking.
9. **FL-16 / fallback date.** 06N N3 sets a fallback date (2026-10-02 12:00Z) for the warm DP row; the decision log shows the warm DP chain started on 2026-10-01 (DL 300), so the fallback is not triggered. The statement is drawn from DL 287 and 300 and should be re-checked against the final run summaries.
10. **Not yet captured in the reading list given to this worker:** the full ANA_CONTRASTS_v4.json, FL_TUNING_SELECTION.json and the ladder register values beyond DL 249 / 256 / 271 / 292 (the proposal asked only for these rows). Tables 4.3 and 4.4 use the logged values.

## Citation checklist (author-year keys used in Chapters 3 and 4)

Keys appear as [Author Year] and need a final bibliographic check, with the venue and a verified author list. Keys taken from the proposal: McMahan 2017, Li 2020, Li 2021, Karimireddy 2020, Ammad-ud-din 2019, Lin 2020, Anelli 2021, Zhang 2023. Keys taken from the sources: Nguyen 2023 (ICLR 2023, arXiv 2206.15387), Chen 2023 (ICLR 2023, arXiv 2206.11488), Xu 2023 (arXiv 2305.18465), Wang 2019 (Wang, Balle and Kasiviswanathan, Theorem 9), McMahan 2018 (DP-FedAvg), Krichene 2020, Petrov 2026 (RecSys 2026, arXiv 2608.19833), Yuan 2025 (FELLAS, TOIS, arXiv 2410.04927), Luo 2022 (CF-FedSR, arXiv 2208.10692). Keys added by the writer and not backed by a source in the list: Kang 2018 (SASRec), Hidasi 2016 (GRU4Rec), Reddi 2021 (adaptive federated optimisation), Hsu 2019 (server momentum), Abadi 2016 (deep learning with differential privacy), Mironov 2017 (Renyi differential privacy), and **Tikhonovich 2025 for the eSASRec paper (arXiv 2508.06450): the first-author attribution is from memory and must be verified.** BM-00 gives the eSASRec arXiv and repository addresses but no author list.

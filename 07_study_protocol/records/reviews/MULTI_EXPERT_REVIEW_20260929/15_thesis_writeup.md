# 15 — Thesis write-up: chapter structure, figures/tables, threats to validity, disclosures, bibliography

Expert part 14 of the multi-expert review (2026-09-29). Reviewer: Claude Opus 5.5, acting as an MSc/PhD examiner in ML, recommender
systems and privacy. This is a READ-ONLY review. It is the only file I wrote. I did not contact the server, use git, run any project
script or test, or open any RETRO_TEST row, RESULT_ROWS, ranks, metrics or runs_* file. The only values quoted are CALIB screen reads that
already appear in the DECISION_LOG and the ladder register.

Tags: **MEASURED** (read from a file, cited as path:line; paths are relative to `.local_ai/ppsi_v4/` unless absolute), **REPORTED** (a
number or claim stated in a project document), **INFERENCE**, **OPINION**, **UNVERIFIED**. A citation is marked "verified" only when I
confirmed it by web search or fetch today (title, authors, year, venue, URL/DOI).

---

## 1. What the project plans for the write-up (MEASURED)

### 1.1 The question and the headline
- **Research question** (proposal p.1, `scratchpad/proposal.txt:19-20`): "How much shopping recommendation quality can be retained when
  moving from cloud-based to on-device or federated models?" The product framing is at `proposal.txt:24-26` ("how much recommendation
  quality is lost when shopping data never leaves the device?").
- **Registered headline:** K2 = R(FA / C_BIG) is primary. K1, K3, K8 and K12 are secondary; the rest are tertiary
  (`contracts/PREG1_AMENDMENT_06D.md:74-78`). K17 = R(FA / C_FULL) was later added as the "cloud-data-advantage co-headline"
  (`PREG1_AMENDMENT_06I.md:10,13`). K20 and K21 cover the deployment branch (FA_WARM) (`PREG1_AMENDMENT_06K.md:22-24`). With these
  additions the confirmatory family is **K1–K21** (DL 218, `company/DECISION_LOG.md:223`: "ANA_CONTRASTS_v3 (K1–K21) to be written
  before the ANA freeze").
- **Registered interpretation texts:**
  - FL retention is a **lower bound** at equal exposure (`06I.md:57-58`);
  - the thesis says "FedAvg/FedAdam-with-<client optimiser> at a 6-EFE budget retains X %", never "FL retains X %" (`06I.md:60-61`);
  - "only weights leave the device" is data minimisation, not privacy (`06I.md:64-67`).

### 1.2 Chapter → artefact map
`company/plans/MASTER_PLAN_V5.md:203-221` (§12.2) adopts the outline of `company/reviews/FABLE_THESIS_RESEARCH_REVIEW_20260928.md:272-296`
(§E.2):

| Chapter | Content |
|---|---|
| 1 Intro | includes "What changed from the proposal and why" |
| 2 Background | |
| 3 Related work | |
| 4 Data | Tables 4.1–4.3, Fig 4.1 |
| 5 Methods | Table 5.1 ladder; DP; K1–K16 + hierarchy |
| 6 Setup | run inventory; A16/A40 |
| 7 Results | 7.1 contest; 7.2 main retention table (TEST); 7.3 curves vs bytes / ε / size / rounds; 7.4 personalisation; 7.5 persona heat-map; 7.6 cold start; 7.7 device; 7.8 baselines |
| 8 Discussion | |
| 9 Threats | |
| 10 Conclusion | |
| Appendices A–H | incl. B "pre-registration record (all locks/amendments with hashes, the TEST register)" and H "incident summary" |

The rule is "every number printed by a script" (`MASTER_PLAN_V5.md:203`).

### 1.3 Planned threats list (`MASTER_PLAN_V5.md:218`)
| Category | Threats listed |
|---|---|
| Internal | screened LR only; μ untuned; checkpoint-rule dependence; A16→A40; max-of-marks on VALIDATION |
| External | one retailer; 46 days; Black Friday; 5–100 users; cohort size |
| Construct | MRR@20; privacy by construction; no attack; battery |
| Statistical | 2 seeds; user-bootstrap scope; **16** confirmatory contrasts; previously inspected TEST |

The sources named for this list are 06A §9, 06B §7, 06D §9 and CLIENT_CONTRACT C5.6.

### 1.4 Registered disclosure lists (where they live)
The disclosures are spread across at least 14 locations:
- 06A §9 (a)–(g) (`06A.md:42-47`); 06B §7 (a)–(e) (`06B.md:90-95`); 06C "Thesis" (`06C.md:24-25`);
- 06D §3 DP list and noise note (`06D.md:62-72`); 06D §9 (a)–(g) (`06D.md:105-111`);
- 06E §9 (a)–(i) (`06E.md:55-66`); 06F §1 disclosure (`06F.md:29-33`); 06G §6 (`06G.md:28-29`); 06H §4;
- 06I §6–§7 (`06I.md:55-71`); 06J §3/§5 (S4 has no LR re-screen; LADDER_S1a came from an interim tree; `06J.md:31-32,50-52`);
- 06K §1–§3 ("opted-in" interpretation; frozen tables come from the opted-in cohort; `06K.md:11,32-33`);
- DL 206 (PRE was trained on 9× MAIN's users; one PRE seed; SMALL recipe not re-tuned; `DECISION_LOG.md:211`);
- DL 218–219 (charged bytes = downloads only; DP arms keep 2 passes / B16 whatever S4 selects; `DECISION_LOG.md:223-224`);
- CLIENT_CONTRACT C5.6 (`contracts/CLIENT_CONTRACT_V1.md:201-213`).

The §12.2 map points only at 06A §9, 06B §7, 06D §9 and C5.6 (`MASTER_PLAN_V5.md:218`).

### 1.5 Pre-registration material the appendix must present
- 19 PREG1 amendment files: 01, 02, 02R1, 02R2, 03, 04, 05, 06, 06A–06K (directory listing of `contracts/`).
- The locks (PREG1_LOCK_VALUES, DATA_PROTOCOL_LOCK_V2, ARCHITECTURE_LOCK, EVAL_DEVICE_PIN_A40, and the planned ALL_METHODS_LOCK).
- 225 DECISION_LOG rows (`DECISION_LOG.md:230`, row 225) and a hash-chained TEST_EXPOSURE_REGISTER (`company/registers/TEST_EXPOSURE_REGISTER.json`,
  entries with `prev_entry_sha256`/`entry_sha256`).
- The data lock labels the whole V4 programme retrospective: "VALIDATION and RETRO_TEST are previously inspected; all V4 results are
  retrospective/adaptive" (`contracts/DATA_PROTOCOL_LOCK_V2.json:80`).
- **No external timestamp exists:** a grep for OSF / OpenTimestamps / "external timestamp" over `company/` and `contracts/` returns
  nothing. The workspace is not a git repository (`eda/CLAUDE.md` header).

### 1.6 Bibliography plan
- `MASTER_PLAN_V5.md:223-224` (§12.4) plans to fix proposal refs [5], [6], [8], [12] and to add Charles 2021, Andrew 2021,
  Wang–Balle–Kasiviswanathan 2019, McMahan 2018, Mironov 2017 and MTZ 2019.
- 06B §7(a) registers the same corrections (`06B.md:91-92`).
- The FL-optimisation citations the later amendments rely on are **not** in the §12.4 list: Reddi 2021 for FedAdam (`06G.md:11`) and
  Hsu–Qi–Brown 2019 for FedAvgM (`06J.md:12`).

### 1.7 Supporting stores
- **Literature register** (`company/research/LITERATURE_REGISTER.json`): 34 entries, generated 2026-09-26. Every entry has
  `authors = null` and `venue = null`; the README explains that the source package never recorded them
  (`LITERATURE_REGISTER_README.md`, "What this is not"). It holds none of the DP or FL-optimisation papers the amendments cite.
- **Thesis living reference** (`eda/.local_ai/thesis_reference/`):
  - STATE.md header "Last update: 2026-09-26T19:27Z" (`STATE.md:3`), but it has appended sections up to 2026-09-28T22:25Z
    (`STATE.md:156-163`);
  - the last CHANGELOG entry is REF-1.1-claude-v4-deep-06d, 2026-09-28T22:25Z;
  - the EN reference stops at §14.17 (`CLAUDE_EXECUTION_REFERENCE_EN.md:1243`);
  - the structured registers have been "PENDING since 2026-09-27" (`STATE.md:163`);
  - `bibliography/references.bib` has 48 entries, none of them a DP or federated-recommendation paper (Abadi, McMahan 2018, Mironov,
    Ammad-ud-din, Lin, PFedRec…).
  - The next source id is S509 (DL 214, `DECISION_LOG.md:219`).
- **Public document** (pushed to GitHub, DL 217 and 225): `F:\Queens University\Microsoft Project\github repo\Privacy-Preserving-Shopping-Intelligence\docs\federated-deployment-study.md`
  (79 lines; audited in §3.6).
- **Prior-work material:**
  - `../sasrec_v1/FINAL_SASREC_REPORT_EN.md` (headed "PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK", L3);
  - `../gru_v1/FINAL_FULL_GRU_REPORT_AR.md` (**Arabic only**; there is no English GRU final report, only `FINAL_FULL_GRU_RESULTS.json`).

---

## 2. State of the art: how strong theses and papers in this area write this up

### 2.1 Validity-threat practice
- **Wohlin et al.'s four-category scheme** (conclusion, internal, construct, external validity) is the standard checklist for
  empirical-methods chapters.
  - Source: C. Wohlin, P. Runeson, M. Höst, M. C. Ohlsson, B. Regnell, A. Wesslén, *Experimentation in Software Engineering*,
    Springer, 2012, DOI 10.1007/978-3-642-29044-2 (verified).
  - The project's "statistical" category corresponds to Wohlin's *conclusion validity*. The thesis should use Wohlin's names so an
    examiner can map it.
- **ML reproducibility and leakage checklists:**
  - J. Pineau, P. Vincent-Lamarre, K. Sinha, V. Larivière, A. Beygelzimer, F. d'Alché-Buc, E. Fox, H. Larochelle, "Improving
    Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)", *JMLR* 22 (2021)
    1–20, https://www.jmlr.org/papers/v22/20-303.html (verified). This is the NeurIPS checklist.
  - S. Kapoor, A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science", *Patterns* 4(9) (2023),
    DOI 10.1016/j.patter.2023.100804 (verified). It gives an 8-type leakage taxonomy and "model info sheets".
  - Either can serve as an appendix checklist that maps each item to an artefact.
- **Recommender-system rigour:** these define the evaluation bar the thesis will be judged against. All were already cited and
  verified in `FABLE_THESIS_RESEARCH_REVIEW_20260928.md:81` §B.1; I did not re-verify them today.
  - Dacrema et al. RecSys 2019; Krichene & Rendle KDD 2020; Ji et al. TOIS 2023; Gusak et al. RecSys 2025 ("Time to Split");
    Hidasi & Czapp RecSys 2023.
- **Institutional requirement (verified by text extraction today):**
  - Queen's SGSPA *Standards for Graduate Theses* (updated July 2025,
    https://www.queensu.ca/grad-postdoc/sites/sgswww/files/uploaded_files/Thesis%20Completion/SGSPA_General_Forms_of_Theses_REVISED_2025.pdf)
    lists as **Required** a "Statement of Originality and Disclosure of Use of Artificial Intelligence". In it, students "must
    declare how AI was used in the research and writing process, including the specific tools used, the reasons for their use, and
    how the generated content was reviewed" (§2.2 and the front-matter table). Unacknowledged use of generative AI is named a breach
    of academic integrity (§2.2).
  - REPORTED (search summary, not full-text-checked): the Queen's *Guidelines for AI use in Graduate Research*
    (https://www.queensu.ca/grad-postdoc/sites/sgswww/files/uploaded_files/Guidelines%20for%20AI%20use%20in%20Graduate%20Research.pdf)
    require that uses be **agreed in advance** with the supervisor or committee and documented.

### 2.2 How FL-recsys and privacy-utility papers report "retention vs centralised"
- **Metric tables with a centralised row, not ratios with CIs:**
  - FedNewsRec compares with "CenNewsRec", "the same news recommendation model … trained on the centralized user behavior data". It
    reports AUC/MRR/nDCG averaged over 5 repetitions ± SE and says only that "the gap … is not very big" (T. Qi, F. Wu, C. Wu,
    Y. Huang, X. Xie, "Privacy-Preserving News Recommendation Model Learning", *Findings of EMNLP 2020*, arXiv 2003.09592; full text
    read today). This paper is from **Microsoft Research Asia on MSN-News** — the closest Microsoft-product precedent for this thesis.
    The project does not cite it.
  - FCF, Lin et al.'s FedRec, FedFast and PFedRec all follow the same pattern (`FABLE_THESIS_RESEARCH_REVIEW_20260928.md:74-79`).
    Ratios with paired CIs are **above** the norm; state this as a contribution.
- **User-level DP papers state the privacy–utility trade-off as "cost at ε".** McMahan, Ramage, Talwar & Zhang, "Learning
  Differentially Private Recurrent Language Models", ICLR 2018, https://research.google/pubs/learning-differentially-private-recurrent-language-models/
  (verified), claims user-level DP with "only a negligible cost in predictive accuracy" at deployment-scale cohorts. The project's
  registered scale argument (`06I.md:58-59`) is the right way to explain why its DP cost will be larger.
- **Leakage from federated recommendation updates is well established.** Chai, Wang, Chen & Yang, "Secure Federated Matrix
  Factorization", IEEE Intelligent Systems 36(5) 11–20, 2021 (arXiv 2019; verified via Semantic Scholar/ResearchGate listing), shows
  that gradients leak the rated items. This supports 06I §7a's disclosure with a citation.
- **What strong FL papers disclose:**
  - the tuned hyper-parameter grids per method (Reddi et al. 2021);
  - cohort size and rounds (Charles et al. 2021);
  - the clipping strategy (Andrew et al. 2021);
  - the accountant and sampling scheme (Mironov 2017; MTZ 2019; WBK 2019).

  All are verified in §2.3 below. The project's amendments already contain this material; the write-up task is to present it
  compactly.

### 2.3 Verified entries for the citations the plan names (all verified today unless marked)
| Key | Correct entry |
|---|---|
| Charles et al. 2021 | Z. Charles, Z. Garrett, Z. Huo, S. Shmulyian, V. Smith, "On Large-Cohort Training for Federated Learning", NeurIPS 34 (2021), arXiv 2106.07820, https://proceedings.neurips.cc/paper/2021/file/ab9ebd57177b5106ad7879f0896685d4-Paper.pdf |
| Andrew et al. 2021 | G. Andrew, O. Thakkar, H. B. McMahan, S. Ramaswamy, "Differentially Private Learning with Adaptive Clipping", NeurIPS 34 (2021) 17455–17466, https://proceedings.neurips.cc/paper/2021/hash/91cff01af640a24e7f9f7a5ab407889f-Abstract.html |
| WBK 2019 | Y.-X. Wang, B. Balle, S. P. Kasiviswanathan, "Subsampled Rényi Differential Privacy and Analytical Moments Accountant", AISTATS 2019, PMLR 89:1226–1235, https://proceedings.mlr.press/v89/wang19b.html (arXiv 1808.00087) |
| McMahan et al. 2018 | H. B. McMahan, D. Ramage, K. Talwar, L. Zhang, "Learning Differentially Private Recurrent Language Models", ICLR 2018 (dblp conf/iclr/McMahanRT018; arXiv 1710.06963) |
| Mironov 2017 | I. Mironov, "Rényi Differential Privacy", IEEE 30th Computer Security Foundations Symposium (CSF 2017), pp. 263–275, DOI 10.1109/CSF.2017.11 (arXiv 1702.07476) |
| MTZ 2019 | I. Mironov, K. Talwar, L. Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism", arXiv 1908.10530 (2019); preprint, no venue found |
| Reddi et al. 2021 | S. J. Reddi, Z. Charles, M. Zaheer, Z. Garrett, K. Rush, J. Konečný, S. Kumar, H. B. McMahan, "Adaptive Federated Optimization", ICLR 2021 (arXiv 2003.00295) |
| Hsu et al. 2019 | T.-M. H. Hsu, H. Qi, M. Brown, "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification", arXiv 1909.06335 (2019); preprint (the FedAvgM / server-momentum reference) |

### 2.4 The proposal's reference errors (verified today)
| Proposal ref (`proposal.txt`) | Problem | Correct entry |
|---|---|---|
| [5] L211 "Ammad-ud-din et al. (2019) Federated Collaborative Filtering for On-Device Next App Prediction. arXiv" | **Title is wrong: two papers are conflated.** The 2019 paper is M. Ammad-ud-din, E. Ivannikova, S. A. Khan, W. Oyomno, Q. Fu, K. E. Tan, A. Flanagan, "Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System", arXiv 1901.09888 (2019). The "on-device next app prediction" title belongs to a different, later paper: A. Sayapin, G. Balitskiy, D. Bershatsky, A. Katrutsa, E. Frolov, A. Frolov, I. Oseledets, V. Kharin, "Federated Privacy-preserving Collaborative Filtering for On-Device Next App Prediction", arXiv 2303.04744 (2023); *User Modeling and User-Adapted Interaction* (2024), DOI 10.1007/s11257-024-09395-0. | Cite 1901.09888 for FCF (the text at L63-65 describes FCF correctly). |
| [6] L212 "Lin, G., et al. (2020). FedRec: Federated Recommendation with Explicit Feedback. AAAI." | **Venue and year are wrong.** | G. Lin, F. Liang, W. Pan, Z. Ming, "FedRec: Federated Recommendation With Explicit Feedback", *IEEE Intelligent Systems* 36(5) (2021) 21–30 (IEEE document 9170754; published online 2020). DOI not confirmed by me: UNVERIFIED. |
| [6] body text L67-68 "FedRec … federated news recommendation … achieving 95% of centralized accuracy with 60% of the communication cost" | **Two errors.** (i) Lin et al.'s FedRec is explicit-feedback rating prediction, not news; the *news* "FedRec" is Qi et al. (arXiv 2003.09592 v1/v2 title "FedRec: Privacy-Preserving News Recommendation with Federated Learning"; published as "Privacy-Preserving News Recommendation Model Learning", Findings of EMNLP 2020). (ii) **The "95 % / 60 %" figures appear in neither.** I searched the full Qi et al. text for "95", "60%", "centraliz" and "communication cost" and found no such claim; its only statement is qualitative ("the gap … is not very big"). The Lin et al. abstract (search summary) mentions reducing computation/communication cost with no numbers. | **Delete the claim** or mark it unsupported. Never repeat it in the thesis. |
| [8] L214 "Li, Z., et al. (2023). Personalized Federated Recommendation via Joint Representation Learning. SIGIR." | **Author, venue and title fragment are wrong.** The title fragment belongs to S. Luo, Y. Xiao, L. Song, "Personalized Federated Recommendation via Joint Representation Learning, User Clustering, and Model Adaptation", CIKM 2022 (arXiv 2208.09375). The name "PFedRec" (text L73) belongs to C. Zhang, G. Long, T. Zhou, P. Yan, Z. Zhang, C. Zhang, B. Yang, "Dual Personalization on Federated Recommendation", IJCAI 2023, pp. 4558–4566, https://www.ijcai.org/proceedings/2023/507 (arXiv 2301.08143). The author list is as I recall it; the search confirmed title, venue and pages, not every author. INFERENCE: "Li, Z. 2023" may come from Z. Li, G. Long, T. Zhou, "Federated Recommendation with Additive Personalization" (FedRAP), arXiv 2301.09109 (register R34). | Cite Zhang et al. IJCAI 2023 for PFedRec, and cite Luo et al. CIKM 2022 separately only if its content is used. |
| [12] L218 "Zhu, W., et al. (2020). Federated Learning for E-Commerce Recommendation. RecSys Workshop." | **Not found.** Two targeted searches today (and the prior review, `FABLE_THESIS_RESEARCH_REVIEW_20260928.md:83`) found no such paper. | UNVERIFIED: drop it unless the owner can produce the PDF. |
| [7] FedeRank; [11] Yang 2019; [13] FedScale; [2] FedProx; [3] FedBN; [4] SCAFFOLD | [7] verified: Anelli, Deldjoo, Di Noia, Ferrara, Narducci, ECIR 2021, LNCS 12656 pp. 32–47, DOI 10.1007/978-3-030-72113-8_3. [11] verified: Q. Yang, Y. Liu, T. Chen, Y. Tong, ACM TIST 10(2) Art. 12 (2019), DOI 10.1145/3298981. [2]–[4] and [13]: venues as in the proposal, consistent with the prior review (not re-checked). | keep |

---

## 3. Comparison: the planned write-up against that standard

### 3.1 Is the chapter → artefact map a coherent MSc thesis with a clear headline? (OPINION)
- **Coherent: yes.** The organising device — a privacy ladder in which every arm sits (`06B.md:50-60`) — turns many arms into one
  answer to the proposal's question. The retention ratio with a paired user-bootstrap CI is exactly "how much is retained". The
  evaluation (full-catalogue exact ranking, global temporal split, hashed pre-registration) is above what the FedRec papers in §2.2
  report.
- **The headline is not yet clear, for three reasons:**
  1. **The §12.2 map predates the whole FL improvement ladder.** It was written 2026-09-28T22:05Z, before 06E–06K (registered
     2026-09-29). It has no row for:
     - the LR curve, which 06F requires as a figure (`06F.md:29`);
     - the FedAdam/FedAvgM points (`06G.md:28`);
     - the ladder table (`06I.md:71`; `company/registers/FL_IMPROVEMENT_LADDER.md`);
     - DP_OPT_SENS, FA_X2, DP8_FULL, PRE / FA_WARM / FA_WARM_FROZEN, or K17–K21.

     The FL story is now the largest part of the results, and it is missing from the map.
  2. **Two denominators and two FL headlines.** K2 (vs C_BIG) is primary. K17 (vs C_FULL) is a co-headline. K20 (FA_WARM / C_BIG)
     "can exceed 100 and mixes a data advantage" (DL 206, `DECISION_LOG.md:211`), so the text must lead with the *descriptive*
     R(FA_WARM / C_FULL). A reader will meet K2, K17, K20 and a descriptive ratio for "the federated answer". **Without a single
     pre-declared headline sentence template, the abstract will be written after the numbers are seen.** That is a HARKing risk in
     presentation, even though the contrasts themselves are registered.
  3. **The CALIB context signals a stark result.** REPORTED/INFERENCE: S0 reaches ≈ 20 % of the matched central value on CALIB, and
     S1b 0.028796 vs ≈ 0.11–0.12 central (`FL_IMPROVEMENT_LADDER.md:13,18-19`; DL 218). The thesis must be written so that a low FL
     number reads as a registered, mechanism-explained *finding* (`06F.md:31-33`, `06I.md:55-61`), not as a failure. That text
     belongs in the thesis draft **before** the first read.
- **Overload (OPINION):** there are 21 confirmatory contrasts, 7 ladder levels plus a deployment branch, ≈ 25 arm labels, 19 amendment
  files and ≈ 20 CALIB tuning configurations. An MSc committee can absorb about 5 main figures and 4 main tables. §E.2 already plans
  ≈ 9 figures and ≈ 12 tables in chapters 4–7, before the ladder material. The fix is not to cut runs but to **tier the reporting**:
  main body = the answer; appendices = everything else, generated by the same script.

### 3.2 Recommended main-body figures and tables (OPINION; each from `retention_report.py` output, TEST unless stated)
| # | Item | Content | Why a committee needs it |
|---|---|---|---|
| **F1** | Retention by privacy level (the headline figure) | Dot-and-95 %-CI chart, grouped by ladder level L0 → L6 (+ deployment branch). Two panels: R vs C_BIG (matched data) and R vs C_FULL (Edge cloud). Arms: C_SMALL (L0/L1), INT8 (L1), FA / FP / PF / FA_Q8 (L2), DP8 / DP1 (L3, CONTROLLED_BUDGET), FT_FA (L4), FT_C (L5), LO / RULE (L6), FA_WARM and PRE (branch). Confirmatory markers on K-rows; descriptive rows hollow. | It *is* the answer to the proposal's question and to its "quantified tradeoff data for various privacy levels" (`proposal.txt:190-191`) |
| **F2** | The FL improvement ladder | Waterfall of CALIB values S0 → S1a → S1b → S2 → S3 → S4 (→ S7 / S8), each bar labelled with the change; the MAIN endpoints of LADDER_S1a (descriptive) and of the final FA beside it | Answers "did you tune FedAvg?" and shows that the federation cost is not an LR artefact (06F §1). It is also the owner's requested "when we did X, FL improved by Y" (`06J.md:5-6`) |
| **F3** | Quality vs communicated bytes | R vs cumulative bytes per client (the 12 marks) for FA, FA_Q8, FA_WARM_FROZEN. A horizontal reference for C_SMALL downloaded once (≈ 82.8 MB) and FT_C. The x-axis states "downloads charged" (DL 218) | The proposal's "bits transmitted vs quality retained" contribution (`proposal.txt:164-165`) |
| **F4** | Privacy–utility curve | R vs ε ∈ {∞ (FA_1024), 8, 1 [, 4]} at MAIN scale (CONTROLLED_BUDGET, K12 / K13), plus the DP8_FULL "Edge-scale" point. Per-round noise-to-signal z·S·√d/m annotated per point (`06D.md:67-72`, `06I.md:58-59`) | Turns "one DP point" into a curve and interprets it by mechanism |
| **F5** | Cold start | R vs support bin {0, 5–10, 11–20, 21–50, 51–100} for FT_C, LO, FA, C_SMALL vs C_BIG (observational, 06A §2); the truncation Study A/B overlay if run; events-to-90/95 % marked | Proposal Steps 3–4 (`proposal.txt:113-119`) |
| **T1** | What changed from the proposal and why | One row each: tasks (next category / purchase AUC / offers → next-item), datasets (5 → REES46), local MF → LO SASRec, FedBN (n/a), SCAFFOLD (not run), battery (not measured), venues | Required by 06D §9g (`06D.md:110-111`) |
| **T2** | The privacy ladder | Arm × what leaves the device at training / at inference × formal guarantee × bytes per visit × what the server can infer (item set, activity via n_consumed) | Operationalises "privacy-preserving" (06B §3 + 06I §7a) |
| **T3** | The main retention table | K1–K21 with hierarchy tier, seeds, rule, value [CI], confirmatory/descriptive flag, INCOMPLETE_SEEDS flag; NDCG@10 / HR@10 columns | The registered confirmatory family in one place |
| **T4** | Device measurements | ONNX FP32 / INT8 size, single-thread latency p50 / p95, RSS, K16 INT8 retention, the browser numbers (HISTORICAL_REFERENCE label) | "Practical guidelines" contribution (`proposal.txt:167-168`) |

To appendices (generated, not hand-typed):
- the contest (Table 7.1 / Fig 7.1): one summary row in Methods ("backbone selection"), full detail in an appendix;
- the persona 3×3 heat-map and strata tables;
- the SPARSE / HEAVY / UNSEEN / TRUE_COLD panels;
- per-seed rows;
- the A16 vs A40 sensitivity;
- the DP_OPT_SENS pair;
- FA_X2; the WOR history of 06C;
- the S7 / S8 CALIB screens;
- run inventory and hashes.

### 3.3 Threats to validity: coverage check
Planned list: `MASTER_PLAN_V5.md:218`. Status: **Listed** / **Registered elsewhere, not in the map** / **Missing everywhere**.

| # | Threat (Wohlin category) | Status | Evidence / note |
|---|---|---|---|
| 1 | Single retailer, 46 days, one task (next different viewed item), one dataset (external) | Listed (task only implicitly) | Add "one task" explicitly; the purchase / offer tasks were dropped (06A §9d) |
| 2 | 5–100 population; MAIN = 131,072 of 1,446,986 (external) | Listed | `DATA_PROTOCOL_LOCK_V2.json:174-178` per the project map |
| 3 | Black Friday in RETRO_TEST (external) | Listed | 06B §7b |
| 4 | Previously inspected TEST → "retrospective/adaptive" (conclusion) | Listed | **Wording:** use the lock's own phrase (`DATA_PROTOCOL_LOCK_V2.json:80`); never "held-out TEST" |
| 5 | **CALIB ⊂ MAIN for the FL tuning ladder (internal / conclusion)** | **Missing** | 06A §1 moved FT / LO selection off CALIB *because* CALIB is "25 % of the report users" (`06A.md:12-13`). The FL recipe, however, is selected over ≈ 20 CALIB configurations (S1a–S4, S7, S8 screens) on CALIB_VALIDATION rows of MAIN users (`06B.md:10-11`; `06J.md`). Consequence (INFERENCE): FL retention on **VALIDATION** is optimistically biased; TEST is not (fixed recipe, one look). C_BIG / C_SMALL were also tuned on CALIB (P2, SMALL ladder), so the bias is partly symmetric; say so. For DP it is listed as unaccounted (`06D.md:63-64`), but not as a validity threat |
| 6 | **Tuning-budget asymmetry now favours FL (internal)** | **Missing** | C_SMALL had a 4-LR ladder (amendment 03 §3). FA gets ≈ 20 configurations over 6 dimensions. The earlier concern ("FedAvg is a straw man") is now reversed ("the central SMALL model is less tuned than FL"). Report a per-arm tuning-budget table (configurations screened, dimensions, panel) |
| 7 | **FL equal-exposure budget = lower bound (internal / construct)** | Registered (`06I.md:57-58`), not in the map | Carry into ch.9 with FA_X2 and the still-improving flag |
| 8 | **Uploaded deltas reveal the viewed-item set; n_consumed reveals activity; no secure aggregation (construct)** | Registered (`06I.md:64-67`; REALISM_AUDIT B2 / B3), not in the map | Needs a citation (Chai et al. 2021, §2.2) and a row in T2 |
| 9 | Pooled bootstrap block (catalogue, full-TRAIN normalisers, θ0) shared by every arm (construct) | Implied ("reproduce C5.6") | C5.6(b) says "no differential privacy" (`CLIENT_CONTRACT_V1.md:205`), stale since the DP arms exist; reword in the thesis |
| 10 | Simulated time: FL "finished instantly" after TRAIN; snapshot model for a whole week; no online adaptation (construct / external) | Registered (06I §7c "snapshot week"; REALISM_AUDIT B13, D4), not in the map | Add |
| 11 | Uniform / sweep participation, no availability bias (external) | Registered (06I §7c) | Add |
| 12 | A16 → A40 device classes (internal) | Listed | 06D §9a; 06E §9h (12,635 / 14,063 differing rank rows) |
| 13 | Checkpoint rule / max-of-marks on VALIDATION; DP best-of-12 outside the accountant (conclusion) | Listed / registered | 06A §9g; 06D §3 |
| 14 | Multiplicity: **21**, not 16, confirmatory contrasts; K13 / K15–K19 single-seed (conclusion) | Listed as "16", **stale** | `06D.md:75-77` and `MASTER_PLAN_V5.md:218` say 16; K17–K21 were added later (06I §1, 06K §2). The thesis must state the family size actually frozen in ALL_METHODS_LOCK |
| 15 | Two seeds; the bootstrap CI is conditional on the trained models (conclusion) | Listed | Already stated in the sasrec_v1 report (L105); reuse that sentence |
| 16 | PRE "opted-in users" interpretation; PRE trained on 9× MAIN's users with an un-retuned recipe; one θ_pre seed (construct / internal) | Registered (DL 206; `06K.md:11`), not in the map | Needed wherever K20 / K21 appear |
| 17 | DP arms keep 2 passes / B16 whatever S4 selects → FA vs FA_1024 / DP differ in more than noise (internal) | Registered (DL 219), not in the map | Affects the reading of K12 / K13 as "pure noise cost" |
| 18 | Interim code trees for descriptive rows (code_ds3r2, r2s, r2p, s2, r3) (internal / reliability) | Registered (`06J.md:50-52`; DL 223) | Put the tree per run in the run inventory |
| 19 | **Pre-registration is internal and not externally time-stamped; amendments registered by AI agents (reliability of the pre-registration claim)** | **Missing** | §1.5. An examiner can ask "how do we know these files predate the reads?" The DL rows, O_EXCL files and hashes are strong internal evidence but have no third-party timestamp. See recommendation W5 |
| 20 | **AI-agent-driven research process (construct / reliability, and an integrity requirement)** | **Missing** from the thesis map | Code, amendments, reviews, gate verdicts and operations were produced by Claude models in named roles ("chief scientist", "Operations Director"; e.g. `06D.md:3-4`, `06K.md:3-5`, `06F.md:3`). Queen's requires the originality / AI statement (§2.1). Research-validity angle: AI-written evaluation code was reviewed by other AI instances (REV-DELTA-01/02/02b); human review is not documented. Thesis-reference chapter `thesis_reference/chapters/22_agents_and_delegation.md` exists (Arabic) and is a starting point |
| 21 | Battery not measured; latency on a server-CPU proxy; browser on-device *training* not available in ORT ≥ 1.20 (construct / external) | Listed (battery) / registered (REALISM_AUDIT C3) | Add C3 |
| 22 | K16 computed in an ONNX-CPU sub-block, not against A40 PyTorch rows (internal) | Registered (`06E.md:23-28`) | Appendix note |

**Verdict (OPINION):** the planned list covers about half of what the project itself has registered. The missing items (5, 6, 19, 20)
are the ones an examiner is most likely to raise, because they concern the *credibility of the process*, not a single arm.

### 3.4 Presenting the amendments honestly
Current plan: "B. Pre-registration record (all locks/amendments with hashes, the TEST register)" (`MASTER_PLAN_V5.md:219`). This is
necessary but not sufficient. With 19 amendment files, the committee needs to see **what was known when each change was made**.
Recommended format for Appendix B (OPINION), one generated row per amendment:

| Column | Content |
|---|---|
| id | amendment id |
| UTC | registration time |
| sha256 | file hash |
| registrant | human, or AI model + role |
| trigger | which review or event prompted it |
| values read before it | registered reads only (e.g. 06F follows the 06B CALIB screen values: 0.020263 / 0.023473 / 0.025562, `06F.md:6-9`; 06J follows S1a; the S2 choices follow S1b) |
| what it changed | |
| direction of possible bias | e.g. "can only strengthen FL" (06F §1b) |
| arms affected | |
| superseded by | |

Two further points:
- Add a one-page narrative in Methods: "The design was registered in stages. After each registered calibration read, the grid was
  extended by rule; no MAIN, panel or TEST value of a downstream arm had been read when any amendment was written, except where
  stated."
- Keep the three incident items (F3 unseal token; DL 105 STATUS.json read; the venv incident) in Appendix H with their impact
  assessments.

### 3.5 Disclosure-list completeness
The §12.2 source list (06A §9, 06B §7, 06D §9, C5.6) misses at least these disclosures, each already registered:
- 06C (Poisson switch; the FL-06 WOR numbers ε ≈ 1,388 at z 0.5642, `06C.md:9-11`);
- 06D §3 (the unaccounted DP list + the noise note, `06D.md:62-72`);
- 06E §9 (a)–(i) (non-cryptographic PRNG; ε covers θ_1..θ_384 only; the PRACTICAL_BEST 0.0002 rule; mean of per-seed ratios;
  bytes prorated; truncation details, `06E.md:55-66`);
- 06F §1 (the grid ended at its edge and was extended once; server LR, μ, local optimiser and passes untuned at that point);
- 06G §6 (Adam β1, β2, τ at Reddi defaults);
- 06H §2 / §4 (FedAdam noise normalisation);
- 06I §6–§7 (lower bound; scale argument; item-set leak; bytes; uniform participation; snapshot week; C2–C5);
- 06J §2 / §3 / §5 (SGD clip binds; S4 without LR re-screen; interim tree);
- 06K / DL 206 (PRE interpretation and data advantage; one θ_pre seed; un-retuned recipe);
- DL 218 / 219 (download-only charged bytes; DP arms fixed at 2 passes / B16);
- 06D §9b (TEST re-runs the device arms' locked procedure).

Several texts contradict later decisions and must be reconciled in the thesis (MEASURED):
- 06B §4 "WOR sensitivity (analysis only)" (`06B.md:66-67`) vs 06C's switch to Poisson, where the WOR result became the *reason* to
  switch (`06C.md:24-25`). `FABLE_EXPERT_DEEP_REVIEW_20260928.md:260` still lists "WOR sensitivity" in the DP appendix.
- CLIENT_CONTRACT C5.6(b) "no differential privacy" (`CLIENT_CONTRACT_V1.md:205`).
- "16 confirmatory contrasts" (`06D.md:75-77`; `MASTER_PLAN_V5.md:218`) vs K1–K21.
- `MASTER_PLAN_V5.md:210` "K1–K16".

A generated single **DISCLOSURES register** (id, text, category, source path:line, the chapter it goes in) removes this failure mode.

### 3.6 The public GitHub document (read-only audit of `docs/federated-deployment-study.md`)
- **Accurate (MEASURED against the contracts):**
  - K = 158,486 with no sampled negatives (L26; `DATA_PROTOCOL_LOCK_V2.json:84`);
  - 5–100 users (L27);
  - split dates as half-open intervals (L28 vs `DATA_PROTOCOL_LOCK_V2.json:57-72`);
  - Large = D256 × 3 blocks, Small ≈ 21 M params, < 170 MB per visit (L37-39; 20.71 M params and 165.68 MB per REALISM_AUDIT B2 /
    B12);
  - "about ten times more users" (L42: 1,348,682 / 131,072 = 10.3);
  - the 8 ladder stages (L62-69 = 06F / 06G / 06I / 06J / 06K);
  - the seed caveat added in the DL 225 fix (L78-79).
- **Inaccurate or incomplete (OPINION: fix at the next owner-approved push; each is a claim, not a result):**
  1. L28 "The test week is evaluated once, after every method is locked." This is true for V4, but it omits that the test week was
     **already inspected** by the v1 campaigns (`DATA_PROTOCOL_LOCK_V2.json:80`; `FINAL_SASREC_REPORT_EN.md:3`). A public methods
     note should say "retrospective".
  2. L17 "the device … sends only model updates". It gives no hint that updates reveal the viewed-item set (06I §7a says this must
     be stated). Add one sentence: "model updates reveal which items were viewed unless differential privacy is added".
  3. L42 "full user pool". C_FULL is the TIEBREAK band, 1,348,682 of 1,446,986 eligible users (≈ 93 %) (`06D.md:92-93`). Say
     "almost the full pool".
  4. L35 "each tuned with the same budget". This is defensible (equal trial count and EFE, per `05_backbone_contest.md:137,191`),
     but the note omits the registered 6.0-endpoint reversal (06A §9a). Acceptable for a plan note; the thesis must carry it.
  5. L50 "ε = 8 and ε = 1": ε = 1 is now SHOULD-medium and may be cut (`06I.md:75`). The doc should say "planned".

### 3.7 Literature register accuracy (sample of 12 of 34, checked against arXiv / proceedings today)
- **Correct title, year and URL:** R02, R03, R04, R06, R07 (Klenitskiy & Vasilev, RecSys 2023; the bib key `klenitskiy_dross_2023`
  agrees), R09, R12, R14, R15, R20, R34 (FedRAP; arXiv 2023; the arXiv page lists it as a conference paper, revised 2024).
- **Errors:**
  - **R24** title is "A Survey on Federated Sequential Recommendation"; arXiv 2504.05313 is **"A Systematic Survey on Federated
    Sequential Recommendation"** (Y. Li, Q. Qin, G. Zhu, W. Xu, H. Wang, Y. Li, R. Zhang, R. Li; submitted 2025-02-19).
  - **R30** "Feature-Indexed Federated Recommendation with Residual-Quantized Codebooks — RQFedRec": arXiv 2601.18570 v2 (revised
    2026-09-10) is titled **"Dynamic Feature-Embedding Communication via Codebook Distillation for Federated Recommendation"**
    (M. Han et al.). Cite the version used, with its date.
  - R13 is dated 2020 (arXiv); the published version is TOIS 2023 (per the prior review).
- **Systemic gaps:**
  - 34 / 34 entries have no authors and no venue, so the register cannot produce a bibliography without a manual pass;
  - it holds none of the ≈ 20 papers the V4 amendments rely on (§2.3 plus Abadi 2016, Bonawitz 2017/2019, Kairouz 2021,
    Konečný 2016, Hard 2018, Yuan 2022, Yu 2020, Zhang 2021 CCS);
  - R32 is Claude Code documentation: evidence that the AI tooling is part of the method, relevant to §3.3 item 20.

### 3.8 Is the thesis living reference ready to feed a thesis? (MEASURED)
**No, not yet.** Six problems:
1. **Stale by more than a day of decisive events:** nothing from 06E–06K, the S1b selection (DL 218), C_FULL / PRE launches, the
   ladder, REV-DELTA-01/02/02b, code_ds3r3, or the GitHub push (DL 217, 225). The last entry is 2026-09-28T22:25Z.
2. **Inconsistent header:** STATE.md says "Last update: 2026-09-26T19:27Z" (`STATE.md:3`) above later appended sections (`STATE.md:129-163`).
3. **Structured registers (DECISION / EXPERIMENT / INCIDENT / CLAIM) PENDING since 2026-09-27** (`STATE.md:134,163`). The plan
   defers them to after the TEST look (`MASTER_PLAN_V5.md:201-202`). The CLAIM register is exactly what should constrain the
   thesis text *before* the TEST numbers exist.
4. **Language:** the thesis-writing chapters are in Arabic (`chapters/22_agents_and_delegation.md`, `25_thesis_writing.md`), and
   `25_thesis_writing.md` describes a pre-V4 thesis (GRU / SASRec contributions, "Privacy/Deployment: proposed contracts").
5. **The bibliography file lacks the whole FL / DP / FedRec literature** (§1.7).
6. On the positive side:
   - the MEASURED / REPORTED / PLAN discipline and the source IDs are strong;
   - the SASRec v1 English report is thesis-grade, including its conditional-CI sentence (L105);
   - `24_claims_and_limits.md` exists.

---

## 4. Recommendations

Class = MUST / SHOULD / NICE / DISCLOSE-ONLY; action = KEEP / CHANGE / REDO / ADD. Costs are INFERENCE. "Prereg?" = whether a new
registration is needed before the first VALIDATION read.

| ID | Recommendation | Class | Action | Cost | Affects | Prereg? |
|---|---|---|---|---|---|---|
| W1 | **Write the Queen's "Statement of Originality and Disclosure of Use of AI"** (required front matter, §2.1). Name the models and roles (Fable / Opus / Sonnet; chief scientist, Ops, reviewers, workers), the parts they produced (code, amendments, reviews, operations, drafting), and how the owner reviewed them. Confirm the supervisor approved these uses **in advance** (Queen's guidelines, REPORTED). Add a Methods §"Research process and tooling" and a threats item (§3.3 #20). Keep it consistent with the public repo; the owner's "no AI attribution in commits" rule (DL 217) is a git-style choice and must not become a thesis omission | MUST | ADD | 0.5–1 day writing + a supervisor conversation | Front matter, ch.5 / 9, the defence | No |
| W2 | **Pre-write the headline.** A number-free abstract and RQ-answer template: one sentence per deployment (cloud: C_FULL vs C_BIG; federated: "FedAvg-family recipe <final> at 6 EFE retains X % [CI] of C_BIG and Y % of C_FULL, a lower bound"; FL + DP; cloud model shipped + fine-tuned; device-only). State which K is primary and how K17 / K20 / descriptive rows are phrased. Freeze it with the ANA freeze | MUST | ADD | 2–3 h | Abstract, ch.1 / 8 / 10 | Text frozen before the read (recommended; no rule change) |
| W3 | **Consolidated DISCLOSURES register** generated from every source in §3.5 (id, text, Wohlin category, source path:line, target chapter). Ch.9 and the DP appendix are rendered from it | MUST | ADD | 0.5 day (Sonnet, mechanical) | ch.5.5, ch.9, App. D | No |
| W4 | **Complete the threats chapter** with the missing items of §3.3: #5 CALIB ⊂ MAIN for FL tuning (VALIDATION FL retention optimistic; TEST not); #6 tuning-budget asymmetry + a per-arm tuning-budget table; #7–#11, #16–#18; #19 internal pre-registration; #20 AI process. Use Wohlin's four category names. Change "16" to the frozen family size | MUST | CHANGE | 0.5 day | ch.9 | No |
| W5 | **Externally time-stamp the registration record before the TEST look:** a hash manifest (sha256 of every contract, amendment, lock, ANA freeze and ALL_METHODS_LOCK) emailed to the supervisor, and/or an owner-approved commit of that manifest. The email route needs no GitHub mutation. Report the timestamp in Appendix B | SHOULD (high) | ADD | 1 h | The credibility of "pre-registered"; App. B | No |
| W6 | **Appendix B as a generated timeline** (columns in §3.4), plus the one-page Methods narrative | MUST | CHANGE | 0.5 day (script over contracts / + DL) | App. B, ch.5.11 | No |
| W7 | **Reporting budget: 5 main figures + 4 main tables** (§3.2, F1–F5, T1–T4); everything else to generated appendices. Update the §12.2 map to include the ladder (F2), K17–K21, DP8_FULL, FA_WARM / PRE | MUST | CHANGE | 2 h plan edit (by the plan owner) + ANA output mapping | ch.7, ANA-03 output list | No (presentation), but the ANA script must emit the needed CSVs before its freeze |
| W8 | **Fix the bibliography** with the verified entries of §2.3 / §2.4. Delete the "95 % / 60 %" claim. Separate Lin et al. FedRec (explicit feedback, IEEE IS 2021) from Qi et al. (news, Findings of EMNLP 2020); cite Qi et al. as the Microsoft precedent. Drop [12] unless produced. Add Reddi 2021 and Hsu 2019 to §12.4 | MUST | CHANGE | 2–3 h | Bibliography, ch.2 / 3 | No |
| W9 | **Repair the literature register:** fill authors and venues for all 34; fix R24 and R30; add the ≈ 20 missing V4 citations with URL / DOI and a "verified on" date; keep UNVERIFIED flags | SHOULD (high) | CHANGE / ADD | 0.5–1 day (Sonnet + web checks) | Bibliography pipeline | No |
| W10 | **Bring the thesis living reference up to date before the ANA freeze, not after TEST:** sources S509+ for DL 181–225 and 06E–06K; EN §14.18+; STATE header fix; **the CLAIM register now** (allowed and forbidden claims per contrast), so the text is constrained before numbers arrive | MUST | CHANGE | 0.5–1 day | Everything downstream of the first read | No |
| W11 | **Public doc corrections** (§3.6 items 1–3, 5) at the next owner-approved push | SHOULD | CHANGE | 30 min | Public accuracy | No; needs owner approval (GitHub) |
| W12 | **Restructure the chapters:** merge Background + Related work; move the backbone contest into Methods as "backbone selection" (one table; detail in an appendix); add a short Phase-1 chapter or section (GRU v1 / SASRec v1 central benchmark, feature study KEEP_CURRENT), labelled retrospective and **not comparable with V4** (different population; `STATE.md:131`); produce an English GRU final report (only the Arabic one exists) | SHOULD | CHANGE / ADD | 1 day | ch.2–3, 5, 7.1 | No |
| W13 | Reconcile the stale texts listed in §3.5 (WOR vs Poisson; C5.6(b); 16 vs 21; K1–K16 in the plan) in the thesis wording and in the DISCLOSURES register | MUST | CHANGE | 1 h | ch.5.5, 9 | No |
| W14 | Add an appendix mapping the NeurIPS reproducibility checklist (Pineau et al. 2021) and a Kapoor–Narayanan leakage "model info sheet" to artefacts | NICE | ADD | 2–3 h | App. F | No |
| W15 | State multiplicity for the actual family (K1–K21), with the hierarchy. Note that a Holm step on bootstrap tail probabilities is computable down to p = 0.001 with 1,000 resamples (INFERENCE; defer to the statistics expert, part 12) | SHOULD | CHANGE | 1 h text | ch.5.8, 9 | Only if a Holm rule is adopted (then before the read) |
| W16 | Present the FL tuning waves in full (table + F2), as 06F §1 / 06I §7d require, **including configurations that lost** | DISCLOSE-ONLY | KEEP | — | ch.7 | No |
| W17 | Keep "the contest selects the backbone under T1, never 'SASRec beats GRU'" (06B §7c) and the conditional-CI sentence from the sasrec_v1 report | DISCLOSE-ONLY | KEEP | — | ch.5 / 7 | No |

---

## 5. Score: 6 / 10 for the write-up part (OPINION)

The raw material for a strong MSc thesis is present and in some respects exceeds the field:
- a privacy ladder that makes the proposal's question answerable;
- ratio-with-CI reporting that FedRec papers do not attempt;
- hashed, staged registrations;
- a hash-chained TEST register;
- an explicit, mechanism-based interpretation of the likely-large FL and DP costs written before any read.

The *write-up plan*, however, has not kept pace with the project:
- the chapter map predates the FL ladder, K17–K21 and the deployment branch, so the largest part of the federated story has no place
  in it;
- the threats list covers about half of what the project itself registered, and omits the four items an examiner will press
  hardest: CALIB ⊂ MAIN for FL tuning, the reversed tuning asymmetry, internal-only pre-registration, and the AI-agent-driven process;
- disclosures are scattered over at least 14 files, several contradicting later decisions;
- the literature register has no authors or venues and none of the DP / FL-optimisation papers;
- the living reference is more than a day stale, with the CLAIM register deferred until after TEST;
- the Queen's-required AI disclosure is not planned at all.

None of this needs a run or changes a registered rule, and most of it is one or two days of writing. Done before the first
VALIDATION read, it would raise this part to 8–9.

---

## 6. UNVERIFIED items and open questions
1. **Proposal ref [12]** (Zhu et al. 2020, "Federated Learning for E-Commerce Recommendation", RecSys workshop): not found by two
   searches today or by the prior review. UNVERIFIED; treat it as non-existent unless the owner supplies the PDF.
2. **Lin et al. FedRec DOI:** IEEE document 9170754 was seen in the URL; the DOI string was not confirmed. The Lin et al. full text
   (checking for any "95 % / 60 %" statement) could not be fetched (403 / empty page). Only the abstract, via a search summary, was
   checked.
3. **PFedRec author list:** title, venue and pages verified; the full author list is not verified character by character.
4. **Queen's Guidelines for AI use in Graduate Research** ("agree in advance"): REPORTED by search summaries; I did not
   full-text-extract that PDF. The SGSPA Standards requirement (Statement of Originality and Disclosure of AI) was extracted and
   verified.
5. Hard et al. 2018 (Gboard), Bonawitz et al. 2019, Kairouz et al. 2021, Konečný et al. 2016, Yuan et al. 2022: cited in project
   reviews as verified there; **not re-verified by me today**.
6. **Open question for the owner and supervisor:** was the AI-agent operating model (AI "chief scientist" writing amendments and
   verdicts) approved in advance by the supervisor, as the Queen's guidelines require? If not, this is the single largest
   non-scientific risk to the thesis and should be raised with the supervisor now.
7. **Open question:** which K-family is frozen in ALL_METHODS_LOCK (K1–K19 per `06I.md:14`, or K1–K21 per DL 218)? The thesis must
   quote the frozen file, not the amendments.
8. **Open question:** will the CALIB-tuned FL recipe's optimism on VALIDATION (§3.3 #5) be quantified? For example, report the FL
   VALIDATION retention on the non-CALIB 75 % of MAIN as a descriptive check. This is analysis-only and would need to be written into
   the ANA outputs before the freeze to count as pre-declared.

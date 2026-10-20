# 03 — Part 2: Data (REES46 cleaning, sessions, targets, censoring, the 5–100 population, bands, chronological splits, leakage)

Reviewer: expert reviewer for part 2 (Claude Opus 5.5), multi-expert audit 2026-09-29. READ-ONLY. This file is the only thing I wrote.
I did not contact the server, use git, run any test or project script, or open any RETRO_TEST row, rank file, RESULT_ROWS file or
data parquet. The only numbers used are counts from contracts, receipts, handoffs, the census JSON summaries, the DECISION_LOG and the
published thesis reference.
Tags: **MEASURED** (read from a file, path:line), **REPORTED** (a project document's own claim, not re-checked), **INFERENCE** (my
reasoning or arithmetic on quoted counts), **OPINION**. Paths are relative to `.local_ai/ppsi_v4/` unless they start with `V1/`
(= `eda/rees46_nextitem_v1/`), `EDA/` (= `eda/next_item_eda/src/rees46_eda/`) or `TREF/` (= `eda/.local_ai/thesis_reference/`).

**Correction to the task framing (MEASURED).** The project has no "near-duplicate session" concept. `L3S_NEAR_DUPLICATE_SESSION` is the
name of a **feature-schema** candidate: L3 is the ladder step that removes near-duplicate feature *channels* (|ρ| ≥ 0.95), and "S"
adds the session-offset and session-mode channels (`contracts/derived_schemas/L3S_NEAR_DUPLICATE_SESSION.json` keys `extension_channels`
and `keep`; `company/reviews/POPULATION_5_100_FABLE_POSITION_FINAL.md:23`). The pipeline does **not** deduplicate sessions and does
not remove bots. v1 states "No bot / high-activity / expensive-product removal. Unusual is not invalid" (`V1/PROTOCOL_DEVIATIONS.md`
§C). Session de-duplication and bot handling are reviewed below as a *missing* piece, not as an existing step.

---

## 1. What the project does (MEASURED)

### 1.1 Source, cleaning and representative events (frozen v1, `rees46_nextitem_v1`, version_hash 860e5a1c…)
- **Source.** The REES46 "eCommerce behavior data from multi category store" files for 2019-10 and 2019-11 (Kaggle): 109,950,743 rows,
  of which 104,335,509 are views, 3,955,446 cart events and 1,659,788 purchases (`TREF/CLAUDE_EXECUTION_REFERENCE_EN.md:104-106`).
  The raw sha256 values are in `contracts/DATA_PROTOCOL_LOCK.json:31-34`.
- **Views only.** The task uses views only (`V1/protocol.yaml:41` `next_item_event_type: view`). Cart and purchase events stay in the
  canonical layer but are never an input or a target (`TREF/...:106`).
- **Identifiers are raw strings end to end** (round-trip proven). The integer indices are fitted on TRAIN only
  (`V1/DATA_CONTRACT.md:177-207`).
- **Price.** A price ≤ 0 or a non-finite price is treated as missing. The row is kept and only the price channel is masked
  (`V1/OWNER_DECISIONS.md:47-52`; `V1/PROTOCOL_DEVIATIONS.md` B2.1). 256,761 rows (0.234%) have a zero price.
- **Metadata recovery.** Recovery is as-of only (earlier evidence of the same entity, no future back-fill). It is accurate (Wilson
  lower bound ≥ 0.9995) but yields almost nothing: 1,149 brand values and 0 category_code values recovered in TRAIN
  (`TREF/...:117-127`).
- **Sessions are rebuilt** from each user's own stream. A new session starts when the gap is > 3600 s, measured in UTC before
  repeat compression (`V1/protocol.yaml:69-81`). The supplied `user_session` is kept for audit only, because 932 of those sessions
  span more than one user (`TREF/...:128-133`). There are 17,984,974 sessions with median length 3, and 30.29% are singletons.
  Sensitivity of the break rate: 13.57% / 12.79% / 12.02% at gaps of 1800 / 3600 / 7200 s (`TREF/...:130-135`).
- **Representative events.** Consecutive repeats of the same product are compressed to the first event; a revisit A,B,A is kept
  (`V1/protocol.yaml:82-86`). In TRAIN, 71,785,832 blocks become 52,760,314 representatives (`TREF/...:136-137`). A same-second block
  with several products becomes an ambiguity barrier: 25,317 blocks and 19,365 blocked transitions (`TREF/...:138-139`).

### 1.2 Targets and censoring
- **What the target is.** The target is the **next representative in the same model segment**: `lead(product_id) OVER (user,
  segment_index)`. A model segment is a run of clean blocks inside one behavioural session and one split (`EDA/pipeline5.py:79-92,
  107-132`; `contracts/EVALUATION_CONTRACT_V4.md:242-258`). Because repeats are compressed, the task is "next *different* viewed item
  **within the current session**" (`V1/protocol.yaml:88`). The first item of a later session is never a target (INFERENCE from the
  CASE branch "next clean block starts a new session → NO_OBSERVED_NEXT", `EVALUATION_CONTRACT_V4.md:255`).
- **Targets are built before the catalogue filter** and never relinked. Filtering first would have manufactured 38,858 A→C
  adjacencies; the executed recipe creates 0 (`TREF/...:142-144`).
- **TRAIN labels.** A TRAIN label must be observed strictly before TRAIN_END. 40,180,608 loss rows; 0 with target_ts ≥ TRAIN_END; 0 null
  (`EVALUATION_CONTRACT_V4.md:113-118`). TRAIN status counts: NO_OBSERVED_NEXT 12,512,673, AB 10,472, BU 1
  (`code/v4data/reference_counts.py:16-19`). So 23.7% of TRAIN decisions have no target (INFERENCE: 12,512,673 / 52,760,314).
- **Censoring classes on VALIDATION** (mutually exclusive, fixed precedence, exhaustive; `EVALUATION_CONTRACT_V4.md:173-231`;
  `contracts/DATA_PROTOCOL_LOCK_V2.json:74-79`):

  | class | count | meaning |
  |---|---|---|
  | CEN_NO_NEXT | 3,064,778 | session end or end of stream (21.3% of VALIDATION decisions, INFERENCE) |
  | CEN_AMBIGUOUS_NEXT_BARRIER | 4,489 | the next raw block is a same-second multi-product barrier |
  | CEN_LEGACY_BARRIER | 170 | a session crossing the preserved 2019-11-20 VS/VC fence |
  | CEN_RIGHT_EDGE | 152 | a session continuing into RETRO_TEST |

  The builder refuses any row that matches no rule and asserts `eval_mask == (status == OBSERVED_IN_PARTITION)`
  (`code/v4data/manifests.py:254-272`).
- **LABEL_GRACE** (11-30 → 12-01) only lets RETRO_TEST tail targets mature. It emits 0 decisions (133 targets matured in v1;
  `EVALUATION_CONTRACT_V4.md:130-139`; `TREF/...:145-149`).
- **OOV targets.** In TRAIN they get `loss_mask = false`. In evaluation they stay in the E2E denominator and score 0. RANKABLE
  (target_class ≥ 0) is reported separately; the macro identity and the invalid factorisation are both tested
  (`EVALUATION_CONTRACT_V4.md:358-440`). Target OOV rates: TRAIN 0.141%, VS 0.561%, VC 1.923%; the v1 structural TEST rate is
  3.383% (`TREF/...:175-180`).
- **Catalogue.** Support ≥ 5 TRAIN representative views, fitted once on full TRAIN: K = 158,486 (`DATA_PROTOCOL_LOCK_V2.json:82-87`).
  The build asserts support = catalogue with 0 mismatches (`code/v4data/manifests.py:210-225`).

### 1.3 Splits (global, half-open UTC; `DATA_PROTOCOL_LOCK_V2.json:56-80`)
- **The four windows.** TRAIN [10-01, 11-16), VALIDATION [11-16, 11-23) = old VS [11-16, 11-20) ∪ old VC [11-20, 11-23) merged
  "minimal-diff", RETRO_TEST [11-23, 11-30), GRACE [11-30, 12-01). The status line reads "VALIDATION and RETRO_TEST are previously
  inspected; all V4 results are retrospective/adaptive" (L80).
- **Minimal-diff merge.** The merge keeps the v1 segments. The 2019-11-20 barrier therefore still splits sessions (170 censored rows,
  §1.2), and the recent-100 user context of a VC decision omits that user's VS events (`contracts/DATA_CONTRACT_CLEAN_V2.md:592-594`, B9).
- **Decision volume per window** (INFERENCE from `reference_counts.py:12-24`):

  | window | decisions per day |
  |---|---|
  | TRAIN | 1.15 M |
  | VAL_A (old VS, 4 days) | 2.73 M |
  | VAL_B (old VC, 3 days) | 1.14 M |

  VAL_A holds 78.2% of the VALIDATION eval rows. The v1 structural TEST count is 7,965,332 decisions over 7 days, about 1.14 M per
  day (`TREF/...:158`).

### 1.4 History policy and features that reach an evaluation decision
- **The sequence window.** USER_HISTORY_L50 = the last 50 representatives of the user's whole prior stream, across sessions and
  fences, strictly causal; the target keeps the same-partition rule (`DATA_PROTOCOL_LOCK_V2.json:88-93`; DL 36). For VALIDATION the
  window reads TRAIN + VS + VC, never TEST (`company/handoffs/DATA-05.md:115`). For RETRO_TEST, the INT-07 gated store reads
  TRAIN + VS + VC + TEST (`company/handoffs/INT-07.md:30`).
- **Decision-row user context is unchanged from v1.** It is proven bitwise equal across the two history policies (`DATA-05.md:110`).
  v1 computes it by a per-partition replay: `TRAIN → TRAIN`, `VS → TRAIN+VS`, `VC → TRAIN+VC`, `TEST → TRAIN+TEST`. "TEST is therefore
  never initialised from the final VALIDATION state, and the observations between fences are intentionally omitted"
  (`EDA/pipeline6.py:1-13,42-47`; `V1/protocol.yaml:127-128`).
- **The INT-07 TEST path uses the unmodified store path.** "The batch path is the unmodified parent path (CompactStore.batch →
  clean_v2_view)… Nothing is refitted" (`INT-07.md:32-33`).
- **The stored inter-view gap is partition-scoped.** At a VS→VC fence it measured 624,176 s where the real gap was 160 s
  (`DATA-05.md:46-47`).
- **S\* keeps four partition-scoped recent-100 aggregates** plus three of their masks: `recent_category_match_share`,
  `recent_brand_match_share`, `recent_mean_log_price` and `current_minus_recent_mean_log_price`. It also keeps
  `log1p_prev_raw_view_gap_s` (`contracts/derived_schemas/L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7.json` `keep`).
- **Shared statistics.** The 17 normalizers, the catalogue, the vocabularies and the recovery tables are fitted on **full TRAIN**
  (all users, including U_NEW, FS, HEAVY and SPARSE) and given identically to every arm (`DATA_PROTOCOL_LOCK_V2.json:163-168`;
  `CLIENT_CONTRACT_V1.md:215-244`). Strict-unseen is disclosed as "BOUNDED ≈ 0.01 z" (`POPULATION_5_100_FABLE_POSITION_FINAL.md:36`).

### 1.5 Population and bands (CENSUS-02 and DATA-06/08/10)
- **Strata**, with n = eligible TRAIN examples (`company/handoffs/CENSUS-02.md` §3 table). Shares are of all 4,804,933 users with a
  TRAIN or VALIDATION decision.

  | stratum | users (share) | share of TRAIN supervision | share of VAL eval users | share of VAL eval rows |
  |---|---|---|---|---|
  | A0: no TRAIN event | 600,554 (12.5%) | 0% | 29.2% | 20.1% |
  | A1: TRAIN events, n = 0 | 1,509,381 (31.4%) | 0% | 8.0% | 5.2% |
  | B: 1–4 | 1,197,003 (24.9%) | 6.2% | 15.2% | 11.0% |
  | **C: 5–100** | **1,446,986 (30.1%)** | **71.2%** | **44.0%** | **51.6%** |
  | D: > 100 | 51,009 (1.1%) | 22.5% | 3.6% | 12.1% |

  In C: n p50 13, p90 45, max 100. In D: max 11,628.
- **Selection.** `selection_key_v2` = the first 8 bytes of sha256(salt | digest), ascending rank r; the keys are asserted unique
  (`POPULATION_5_100_FABLE_POSITION_FINAL.md:16-17`).

  | band | rank range | contents |
  |---|---|---|
  | MAIN | [0, 131,072) | 2,587,776 TRAIN decisions; 51,211 VALIDATION users (40,764 with ≥ 1 eval row); 519,714 eval rows |
  | CALIB | [0, 32,768) ⊂ MAIN | 12,715 VALIDATION users (10,070 with eval rows) |
  | U_NEW_VAL / U_NEW_TEST | [1,348,682, 1,381,450) / [1,381,450, 1,414,218) | never trained on |
  | FS | [1,414,218, 1,446,986) | feature-search cohort |
  | TIEBREAK = C_FULL | [0, 1,348,682) | 1,348,682 users / 26,677,600 decisions |
  | PRE | [131,072, 1,348,682) = TIEBREAK − MAIN | 1,217,610 users / 24,089,824 decisions |

  Sources: `DATA_PROTOCOL_LOCK_V2.json:170-289`; DL 162; DL 207; `company/handoffs/DATA-10.md:7-9`.
- **Panels** (never trained on; users selected by smallest key among those with ≥ 1 E2E VALIDATION row):
  - TRUE_COLD = 381,295 users (A0 299,035 / A1 82,260);
  - COLD_H 16,384 ⊂ TRUE_COLD;
  - SPARSE_PANEL 16,384 from B;
  - HEAVY_PANEL 16,384 from D;
  - WARM_TRUNCATION 16,384 ⊂ MAIN.

  Sources: `POPULATION_5_100_FABLE_POSITION_FINAL.md:17`; `DATA_PROTOCOL_LOCK_V2.json:335-411`.
- **Disjointness verified** by rank arithmetic and direct set intersection. MAIN, U_NEW_VAL, U_NEW_HOLDOUT and the panels are
  pairwise disjoint. CALIB ⊂ MAIN, WARM_TRUNCATION ⊂ MAIN and COLD_H ⊂ TRUE_COLD are checked directly, and FS is disjoint from
  everything (`company/handoffs/DATA-06.md:83-88`). PRE is disjoint from MAIN, CALIB, U_NEW, FS and the panels (`DATA-10.md:9`).
  C_FULL is an exact cover check of the pool minus U_NEW and FS (`DATA-08.md:8`).
- **CALIB ⊂ MAIN, and what was done about it.** CALIB is 24.7% of MAIN's evaluable VALIDATION users (10,070 / 40,764, INFERENCE).
  FT/LO recipe selection was therefore moved from CALIB to UNSEEN_VAL (`contracts/PREG1_AMENDMENT_06A.md:10-17`; DL 146, 151).
  The FL local-LR ladder (S1b and later stages) and the C_SMALL LR ladder are still selected on CALIB_VALIDATION
  (`PREG1_AMENDMENT_06F.md:14-17`; `PREG1_AMENDMENT_06J.md:15`; DL 143, 218).

---

## 2. State of the art (literature and real systems; web-verified unless marked UNVERIFIED)

**Leakage and splitting in recommender evaluation**
- **[L1] Global timeline.** Y. Ji, A. Sun, J. Zhang, C. Li, "A Critical Study on Data Leakage in Recommender System Offline
  Evaluation", ACM TOIS 2023 (arXiv 2010.11060; https://dl.acm.org/doi/fullHtml/10.1145/3569930). Leakage arises when the global
  timeline is not observed, i.e. a model learns from interactions that would not exist at prediction time. The paper proposes a
  timeline evaluation scheme.
- **[L2] Splits and popularity.** A. Sun, "Take a Fresh Look at Recommender Systems from an Evaluation Standpoint", SIGIR 2023,
  DOI 10.1145/3539618.3591931. Random and leave-one-out splits break the global timeline; the popularity baseline is ill-defined
  under them.
- **[L3] The split is a confounder.** Z. Meng, R. McCreadie, C. Macdonald, I. Ounis, "Exploring Data Splitting Strategies for the
  Evaluation of Recommendation Models", RecSys 2020, DOI 10.1145/3383313.3418479. The splitting strategy can reorder systems.
- **[L4] Sequential recommenders.** D. Gusak, A. Volodkevich, A. Klenitskiy, A. Vasilev, E. Frolov, "Time to Split: Exploring Data
  Splitting Strategies for Offline Evaluation of Sequential Recommenders", RecSys 2025, DOI 10.1145/3705328.3748164 (arXiv 2507.16289).
  Leave-one-out permits train/test time overlap (temporal leakage). Global temporal splits are more realistic, and the choice reorders
  models.
- **[L5] Common flaws.** B. Hidasi, Á. T. Czapp, "Widespread Flaws in Offline Evaluation of Recommender Systems", RecSys 2023,
  DOI 10.1145/3604915.3608839. It lists unsuitable datasets, questionable preprocessing, leaking splits and sampled negatives.
- **[L6] Session-based practice.** M. Ludewig, D. Jannach, "Evaluation of session-based recommendation algorithms", UMUAI 28:331–390,
  2018, DOI 10.1007/s11257-018-9209-6. This is the standard session-based protocol family (rebuilt sessions, item-support filtering,
  next-item in session, strong simple baselines).
- **[L7] Strong generalisation.** D. Liang, R. G. Krishnan, M. D. Hoffman, T. Jebara, "Variational Autoencoders for Collaborative
  Filtering", WWW 2018, DOI 10.1145/3178876.3186150. It evaluates on held-out users.
- **[L8] The same dataset in the literature.** G. de Souza P. Moreira et al., "Transformers4Rec: Bridging the Gap between NLP and
  Sequential/Session-Based Recommendation", RecSys 2021, DOI 10.1145/3460231.3474255. It uses REES46 with the supplied session ids.
  The project rebuilds sessions instead, and documents why (`V1/PROTOCOL_DEVIATIONS.md` A#2). Dataset page:
  https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store.

**Federated data partitioning and real systems**
- **[F1] LEAF.** S. Caldas et al., "LEAF: A Benchmark for Federated Settings", arXiv 1812.01097 (2018/2019). Natural per-user partitions
  and a rigorous evaluation framework.
- **[F2] FedScale.** F. Lai et al., "FedScale: Benchmarking Model and System Performance of Federated Learning at Scale", ICML 2022,
  PMLR 162:11814-11827. Real-world client partitions and a unified evaluation protocol per dataset.
- **[F3] FLUTE.** M. Hipolito Garcia et al. (Microsoft), "FLUTE: A Scalable, Extensible Framework for High-Performance Federated
  Learning Simulations", arXiv 2203.13789 (2022). A simulation framework (as used here).
- **[F4] Participation gap.** H. Yuan, W. Morningstar, L. Ning, K. Singhal, "What Do We Mean by Generalization in Federated
  Learning?", ICLR 2022 (arXiv 2110.14216). It separates the out-of-sample gap from the participation gap; the project's U_NEW design
  follows it.
- **[F5] Gboard.** A. Hard et al., "Federated Learning for Mobile Keyboard Prediction", arXiv 1811.03604 (2018). A production Gboard
  model trained on natural user data.
- **[F6] Production system design.** K. Bonawitz et al., "Towards Federated Learning at Scale: System Design", MLSys 2019
  (arXiv 1902.01046). Device eligibility is idle, charging and unmetered network. Which devices participate is decided by the
  system, not by per-user activity in a later period.
- **[F7] Apple.** M. Paulik et al., "Federated Evaluation and Tuning for On-Device Personalization: System Design & Applications",
  arXiv 2102.08503 (2021). Federated evaluation and tuning on held-out devices.
- **[F8] Federated Reconstruction.** K. Singhal et al., "Federated Reconstruction: Partially Local Federated Learning", NeurIPS 2021
  (arXiv 2102.03448). Deployed for Gboard collaborative filtering, with evaluation on unseen clients.
- **[F9] Open problems.** P. Kairouz, H. B. McMahan et al., "Advances and Open Problems in Federated Learning", FnTML 14(1–2), 2021,
  DOI 10.1561/2200000083.

**UNVERIFIED (not checked in this session; do not cite as fact):**
- that production FL systems impose a *minimum example count* per client, or cap examples per client (e.g. DP-FL Gboard pipelines);
- the cause of the 16–19 Nov traffic spike in REES46 (a sale event);
- any published rule that 30-minute versus 1-hour session gaps matter for REES46.

---

## 3. Comparison with the state of the art

### 3.1 Strengths (well above typical thesis and much conference practice)
1. **Global temporal split with successive targets.** No model or personal update ever uses VALIDATION or TEST labels
   (`EVALUATION_CONTRACT_V4.md:142-151`). This is exactly what [L1][L2][L4] ask for, and avoids the leave-one-out leakage they
   document. Label-time audits are MEASURED zeros (§1.2).
2. **Targets built before the catalogue filter.** 0 manufactured adjacencies versus 38,858. OOV targets are kept end to end at zero
   credit, and the macro and micro identities are proven. This avoids the "filter rare items then relink" flaw common in session-based
   benchmarks ([L5][L6]).
3. **Censoring is explicit, code-derived, exhaustive and fail-closed**, with a reporting rule (E2.5). The AB wording caveat is honest
   ("next raw block is a barrier", `EVALUATION_CONTRACT_V4.md:273-275`).
4. **Sessions rebuilt from each user's own stream**, with a documented reason (the supplied UUIDs span users). A gap sensitivity is
   reported, and same-second multi-product ambiguity is handled by barriers instead of an arbitrary order.
5. **The population selection is outcome-blind and verifiable.** TRAIN-only eligibility and a hash-rank sample. The nested prefixes
   keep the n distribution identical at every N (p50 13 at every N, CENSUS-02). Band disjointness is proven by arithmetic and by set
   intersection. MAIN, CALIB, U_NEW, FS, TIEBREAK and PRE are exact rank slices of one order, and the band sums reconcile exactly
   (INFERENCE: 1,348,682 + 3 × 32,768 = 1,446,986; 2,587,776 + 24,089,824 = 26,677,600).
6. **Natural partitions: one persistent user = one client**, with no merging of histories (`CLIENT_CONTRACT_V1.md:22-28`). This is
   the LEAF/FedScale practice ([F1][F2]) rather than a synthetic Dirichlet split.
7. **A user-disjoint unseen-client panel** (U_NEW_VAL for selection, U_NEW_HOLDOUT for reporting), evaluated in the same future
   windows. This measures the participation gap of [F4] and the strong generalisation of [L7]. Moving FT/LO selection off CALIB was
   the right call (06A §1).
8. **Pooled bootstrap statistics are TRAIN-only**, classified per owner §9, delivered identically to every arm and disclosed
   (B1–B9, C5.6). No temporal leakage enters through them; the cross-user component is bounded and stated.
9. **Exposure honesty.** RETRO_TEST is labelled previously inspected and retrospective. A gated TEST builder exists, with a hash-chained
   exposure register. The CENSUS-02 local TEST-metadata slip was self-reported (`CENSUS-02.md` Incidents 1).

### 3.2 Weaknesses and errors

**W1 — The TEST-period decision features omit the VALIDATION week (not leakage; an undisclosed train/eval inconsistency).**
- **What is MEASURED:** the v1 replay seeds TEST user context from end-of-TRAIN and skips VALIDATION (`EDA/pipeline6.py:1-13,42-47`;
  `V1/protocol.yaml:127-128`). The decision-row user_context is carried unchanged into V4 (`DATA-05.md:110`). The INT-07 TEST path
  uses that unmodified store (`INT-07.md:32-33`).
- **What follows (INFERENCE):** for every RETRO_TEST decision, the four kept recent-100 aggregates of S\* and their masks are
  computed from TRAIN + earlier TEST events only.
  - A user's last up to 7 days (the whole VALIDATION week) are missing from them.
  - Meanwhile the L50 sequence window *does* contain those events.
  - The first TEST events' `log1p_prev_raw_view_gap_s` is very likely the gap to the last TRAIN view, not to the last VALIDATION
    view. This is INFERENCE by analogy with DATA-05's VS→VC measurement and needs a count-only check.
- **Why it matters:**
  - TRAIN decisions never have this hole. VALIDATION has it only for VC decisions (3 days, disclosed as B9). RETRO_TEST has it for
    all 7 days.
  - The headline split is therefore evaluated under a feature-staleness regime that no training or selection data had.
  - It breaks the owner requirement DL 35(d): "all of a user's previous events must enter VALIDATION/TEST prediction".
  - It breaks the lock's wording "identical for TRAIN / VALIDATION / TEST" (`DATA_PROTOCOL_LOCK_V2.json:90`) for these channels.
  - It is unrealistic for Edge: a device always has its last week.
- **What limits the damage:** it is the same input for every arm, so paired retention is protected to first order. Arms can still
  lean on user_context to different degrees. I found no V4 disclosure of it for RETRO_TEST: the only V4 mention is the VC→VS case
  (DL 156; B9).

**W2 — The "cloud" references are cloud-on-the-5–100-slice, not the cloud.**
- C_BIG trains on 131,072 users. C_FULL trains on 1,348,682 users / 26.68 M decisions, which is **66.4%** of the 40.18 M eligible
  TRAIN examples (INFERENCE).
- Heavy users (1.1% of users, **22.5%** of supervision) and sparse users (6.2%) are trained on by no arm (CENSUS-02).
- A real Edge/cloud service would train its cloud model on everyone. In a real FL deployment, heavy users would participate, probably
  with a per-client cap (UNVERIFIED as production practice).
- The matched design is scientifically clean: it isolates the federation factor on identical users. But "retains X% of cloud quality"
  is relative to a population-restricted cloud and must be worded that way.
- The v1 campaigns trained on all users (`V1/protocol.yaml` `population.main: all_available_valid_train_users`). They are not
  comparable: they use other features and SESSION_L50.

**W3 — Only within-session next-item targets.**
- The task never scores the first item of a new session. 21–24% of decisions are session ends with no target (§1.2).
- For a browser, the start of a visit (homepage or new-tab recommendations) is arguably the most valuable prediction moment. That is
  exactly where cross-session history (USER_HISTORY_L50, FL/PF personal state) should matter most.
- The design is standard in session-based benchmarks ([L6][L8]), so this is a scope limitation, not an error. But it is under-stated
  relative to the Edge framing. In addition, views only: carts and purchases (5.6 M events) are unused as inputs or targets.

**W4 — Non-stationary evaluation windows; the selection window is dominated by an atypical surge.**
- VAL_A (11-16 → 11-20) runs at 2.4× the TRAIN decision rate and holds 78% of VALIDATION eval rows. VAL_B and the v1 TEST structural
  count run at the TRAIN rate (INFERENCE, §1.3).
- RETRO_TEST contains Black Friday (2019-11-29) and has 2–6× the OOV rate of VALIDATION (`TREF/...:175-180`).
- Every model is frozen at 11-16, so at TEST it is 7–14 days stale.
- The on-device arms re-run on TEST with "the SAME support rows" (`PREG1_AMENDMENT_06D.md:45`), i.e. TRAIN labels only, although on a
  device the VALIDATION-week labels would be locally available.
- Consequence: the offline protocol structurally under-represents the one advantage on-device and federated learning has in
  deployment, continual local adaptation. Symmetrically, it omits cloud retraining. VAL_A/VAL_B subperiods are registered as
  descriptive (E1.4/E1.5), which is good.

**W5 — FL selection intensity on a subset of the reporting users.**
- The FL ladder is screened on CALIB_VALIDATION, which is 25% of MAIN's evaluable users in the same week: S1b uses 6 multipliers,
  S2 a grid, and S3/S4 further candidates.
- 06A made a CALIB-disjoint MAIN_VALIDATION diagnostic mandatory only as the fallback for FT/LO. G2 had one for the contest
  (`PREG1_AMENDMENT_05.md:36`). I found **none registered for the FL ladder, C_SMALL or DP S_CAL arms** (no "disjoint" in
  `company/gates/ANA_CONTRASTS_v2.json`, 06F or 06J).
- The winner's curse is small relative to the reported FL/central gap (CALIB FA 0.0288 vs central ≈ 0.11–0.12, DL 218 / 06F L7-9).
  RETRO_TEST removes the time overlap. So this is a SHOULD, not a MUST.

**W6 — The panels condition on future activity.**
- SPARSE, HEAVY, WARM_TRUNCATION and TRUE_COLD users are drawn among those with ≥ 1 VALIDATION eval row
  (`FABLE_POSITION_FINAL.md:17`). Their RETRO_TEST versions (DATA-09) are therefore returning-user panels selected on VALIDATION
  presence.
- MAIN is not selected this way, so panel results versus MAIN results at TEST mix a selection difference. The owner's "zero influence
  of VALIDATION/TEST on selection" rule (§1) literally covers only MAIN.
- Similarly, only 31% of MAIN users are in any VALIDATION macro. The metric is "quality for users who return next week". This is
  standard, and it should be stated.

**W7 — "TRUE_COLD" is not history-cold for A1.**
- The 82,260 A1 users have TRAIN events, just no eligible labelled transition. Under USER_HISTORY_L50 their queries carry TRAIN
  context.
- A0/A1 are reported separately: registered in Fable D1, and the evaluator emits a `cold_stratum` stratum
  (`code/v4eval/evaluate.py:25`).
- The thesis text must never pool them as "new users". KEEP, with a wording guard.

**W8 — Stale binding text in the locked contracts.**
- DATA_PROTOCOL_LOCK_V2 declares `contracts_changed_since_v1: {}` (L443). The contracts it hashes therefore still say:
  - "There is no generic '≥ 5 interactions' filter" (`CLIENT_CONTRACT_V1.md:28`), the 262,144 cohort and M02 as the primary
    (`CLIENT_CONTRACT_V1.md:52-111`; `EVALUATION_CONTRACT_V4.md:316-324`);
  - "L = 50 is the maximum same-derived-session prefix" (`EVALUATION_CONTRACT_V4.md:150`).
- These are superseded by the POP 5–100 amendment, DL 36 and LOCK_V2, but nowhere is this recorded as an erratum next to the
  contracts. A thesis methods chapter that cites the contracts would be wrong.

**W9 — Missing descriptive cuts that matter for this dataset.**
1. **Repeat versus novel targets.** Revisits are allowed and nothing is filtered (`EVALUATION_CONTRACT_V4.md:507`), but there is no
   breakdown by whether the target is already in the user's L50 window. On session e-commerce data, repeats carry a large share of
   MRR, and on-device, LAST and RULE arms depend on them.
2. **No bot or anomaly audit on D.** Max n is 11,628. D is excluded from training anyway, but the HEAVY panel is reported.

**W10 — Pooled catalogue and normalizers include every excluded user**, including U_NEW (≈ 3% of TRAIN).
- This is correctly classified and bounded. It is realistic for Edge in one respect: a retailer's catalogue is public.
- It is not realistic for the support ≥ 5 popularity gate or for the count-type normalizers, which are cross-user statistics a
  browser vendor would need a federated-analytics round to obtain. These are disclosed (Fable D5). No action beyond wording.

### 3.3 Unrealistic choices relative to an Edge deployment (OPINION, with the evidence above)
- One REES46 `user_id` = one client, but a browser client holds a user's history across many retailers. The client's data here is a
  single-site slice (framing expert's area; noted for completeness).
- Participation is decided by an activity band over the same 46 days used for training. Production FL selects devices by system
  eligibility ([F6]), not by the number of labelled transitions. The 5–100 band is best described as "the owner-fixed activity band of
  the matched benchmark" and not as "Edge users".
- Static offline evaluation with models frozen at 11-16 and no local adaptation on the week before TEST (W4).

### 3.4 Leakage verdict
- **No temporal label leakage found.** TRAIN-only labels, eligibility, catalogue, normalizers, vocabularies and selection keys. There
  is a global split; the censoring is fail-closed; there are no updates from future labels; targets are built before filtering.
- **The cross-user pooled statistics are bootstrap information, not leakage.** They are disclosed.
- **Adaptive reuse remains.** It comes from the v1 TEST inspection (for the backbone and feature claims), from CALIB ⊂ MAIN and from
  the reuse of the VALIDATION week by FS, the contest and the ladders. It is properly labelled "retrospective". For the thesis's
  *new* quantity (FL/central retention) no RETRO_TEST value has ever been seen, so the one-look TEST stays a valid confirmatory read.
- **W1 is the only data issue that changes what the TEST inputs are, relative to TRAIN and VALIDATION**, and it is currently
  undisclosed.

---

## 4. Recommendations

**Governed-value status (MEASURED from DL 218–229 and the project map):**
- RETRO_TEST has not been read.
- ALL_METHODS_LOCK is not written and the ANA freeze is pending.
- CALIB FL screen values *have* been read (DL 181, 218).
- MAIN FL values are still being produced.

A **descriptive** addition that does not alter a registered rule may still be registered before the ANA freeze. Nothing below touches
a rule whose governed value has already been read.

| ID | item | class | action | cost (INFERENCE) | affects | pre-registration needed + status |
|---|---|---|---|---|---|---|
| D1 | Disclose W1: at RETRO_TEST, 4 user_context channels + 3 masks of S\* (and very likely the first-event inter-view gap) are computed without the user's VALIDATION-week events, while the L50 sequence includes them. Add a **count-only receipt** before the look: the share of MAIN_TEST / UNSEEN_HOLDOUT_TEST decisions whose user has ≥ 1 VALIDATION event, the hole length distribution, and the share of first-TEST-event gaps that skip a VALIDATION view. Correct the lock wording "identical for TRAIN/VALIDATION/TEST" to "sequence window identical; decision-level aggregates partition-scoped (v1)" | **MUST** (disclosure + counts) | ADD | ≈ 1–2 h CPU, counts from ledger timestamps only, no targets | TEST look (all arms equally); thesis methods and threats | Disclosure: no pre-registration. The count receipt must be registered as COUNTS_ONLY in the exposure register if it touches TEST rows (decision timestamps) — do it inside the gated DATA-09 build, before any metric. Status: not done |
| D2 | Optional sensitivity for W1: a deployment-consistent TEST user-context sidecar (replay TRAIN+VS+VC+TEST) and a second DESCRIPTIVE TEST evaluation of the locked checkpoints on it. The headline stays on the registered path | NICE | ADD | ≈ 1–2 days of engineering + one extra re-eval pass | TEST look only | YES, before ALL_METHODS_LOCK, labelled DESCRIPTIVE / never selection. Not registered |
| D3 | Word every retention claim as "relative to a cloud model trained on the same 5–100 users (C_BIG) / on 1.35 M 5–100 users (C_FULL, 66% of TRAIN supervision)". State that heavy (22.5% of supervision) and sparse users are trained on by no arm | **MUST** (wording) | CHANGE (text) | 0 | thesis text; K17–K21 interpretation | No. Status: partly (Fable D10 limitation "claims cover users with 5–100") |
| D4 | CALIB-disjoint MAIN_VALIDATION diagnostic (the 38,496 users with decisions, as in amendment 05 / G2 condition 5) for every CALIB-tuned arm (FA ladder winner, FP, FA_Q8, DP arms via S_CAL, C_SMALL, FA_WARM) at VALIDATION. Report only, never selection | SHOULD | ADD | minutes (re-aggregation of existing rows via decision_id) | VALIDATION tables of finished and running FL arms | YES (descriptive), register before the ANA freeze. Not registered |
| D5 | Repeat / novel target breakdown: for each E2E row, flag whether target ∈ the user's L50 window (a pure function of the inputs); report MRR@20 per arm on both subsets, VALIDATION and TEST | SHOULD | ADD | ≈ 0.5 day (a flag column + an ANA stratum) | ANA only | YES (descriptive stratum), before the ANA freeze. Not registered |
| D6 | Disclose the evaluation-window non-stationarity: VAL_A surge (2.4× TRAIN rate, 78% of VAL rows), Black Friday in RETRO_TEST, TEST OOV 3.4% versus VAL 0.6–1.9%, models 7–14 days stale. Keep VAL_A/VAL_B as registered descriptive subperiods, and add the TEST daily breakdown (Black Friday day separately) as descriptive | SHOULD (disclose) / NICE (per-day TEST) | KEEP + ADD | 0 / minutes | thesis; TEST tables | Per-day TEST stratum: YES (descriptive), before ALL_METHODS_LOCK. Not registered |
| D7 | State that the protocol is static: no arm uses VALIDATION-week data at TEST (no cloud retrain, no FL rounds, no local LO/FT adaptation), so the on-device advantage of continual local learning is not measured. Optional NICE: an "LO/FT with support = TRAIN+VALIDATION labels" TEST variant, paired with C retrained on TRAIN+VAL — too costly now | DISCLOSE-ONLY (NICE variant) | ADD (text) | 0 (text); variant ≫ 1 day | thesis framing (Edge realism) | Variant: YES + a new amendment; recommended NOT to run. Text: none |
| D8 | Disclose that the panels (SPARSE, HEAVY, TRUE_COLD, WARM_TRUNCATION) are selected on ≥ 1 VALIDATION eval row, so their TEST results are "returning-user" panels, unlike MAIN; and that the VALIDATION macro covers the 31% of MAIN users who return | DISCLOSE-ONLY | ADD (text) | 0 | panel tables | No |
| D9 | Wording guard: TRUE_COLD = A0 (no TRAIN events) ∪ A1 (TRAIN events, no eligible label). Always report A0 and A1 separately; never call A1 "new users"; state that A1 queries carry TRAIN context under USER_HISTORY_L50 | SHOULD | KEEP (registered) + text | 0 | cold-start chapter | No (already registered in Fable D1; the evaluator emits cold_stratum) |
| D10 | Erratum note (a new file, not an edit of locked contracts) listing the superseded clauses: CLIENT_CONTRACT C1.1 "no ≥ 5 filter", C2 262,144 cohort / M02 primary, E1.12 session-prefix L, and M04S retired → UNSEEN_VAL; cross-reference DATA_PROTOCOL_LOCK_V2 + POP amendment + DL 36 + 06A | SHOULD | ADD | ≈ 1 h | thesis methods; reproducibility | No (documentation) |
| D11 | Describe the task precisely in the thesis: "next different viewed item within the current (1-hour-gap) session, given up to 50 prior events across sessions"; views only; first-item-of-session and purchase prediction out of scope | **MUST** (wording; the Edge framing otherwise over-claims) | CHANGE (text) | 0 | thesis ch. 3 / threats | No |
| D12 | HEAVY stratum sanity: count-only audit of D users (events per minute, sessions per day, distinct products per session) to flag likely bots before reporting HEAVY_PANEL; report the panel with and without flagged users only as DESCRIPTIVE | NICE | ADD | ≈ 0.5 day CPU (TRAIN-only counts) | HEAVY panel reporting | If used as a filter at TEST: YES before the look. Not registered |
| D13 | Keep: the global temporal split, targets before filter, censor classes, OOV E2E, catalogue A, full-TRAIN statistics (disclosed), hash-rank bands, the U_NEW user-disjoint panels, FT/LO selection on UNSEEN_VAL, PRE = TIEBREAK − MAIN, and C_FULL as the matched reference for the deployment branch | — | KEEP | 0 | — | — |

**Things I recommend NOT doing (OPINION):**
- Do not change the 5–100 band, the catalogue, the normalizer population or the split dates. All are owner-fixed or locked, and their
  governed values have been used.
- Do not rebuild sessions (the raw CSVs are not on the server; E1.3).
- Do not add a rolling-origin backtest (out of scope per Fable D7; it would reopen the timeline).

---

## 5. Score: **8 / 10**

**Why it is high.** The data pipeline is the strongest part of this project that I have seen, and it is better than most published
sequential-recommendation evaluations:
- a global temporal split with successive targets and no future updates ([L1][L2][L4]);
- targets built before the catalogue filter;
- exhaustive, code-derived, fail-closed censoring;
- OOV kept end to end, with a proven macro identity;
- sessions rebuilt from each user's own stream, with a sensitivity check;
- outcome-blind hash-rank populations with proven disjointness and exact reconciliation;
- natural one-user-one-client partitions ([F1][F2]);
- a user-disjoint unseen panel for the participation gap ([F4]);
- candid exposure labelling.

**Why it is not 9–10.**
- (i) W1: at RETRO_TEST the decision-level aggregates silently omit the VALIDATION week. That contradicts the project's own "identical
  for TRAIN/VALIDATION/TEST" and "all previous events" requirements on the headline split, and it is not disclosed.
- (ii) The "cloud" reference is a cloud on the 5–100 slice (66% of supervision at most). That must shape every retention sentence.
- (iii) Within-session, views-only targets and a static, frozen-at-11-16 protocol structurally cannot show the deployment advantages of
  on-device and federated learning (continual adaptation, visit-start prediction).
- (iv) Smaller issues: CALIB-overlap selection for the FL ladder without a disjoint diagnostic, panels conditioned on VALIDATION
  presence, stale locked contract text, and no repeat/novel breakdown.

None of these is temporal label leakage, and all except D2/D7 can be fixed or disclosed in hours before ALL_METHODS_LOCK.

---

## 6. UNVERIFIED items and open questions
1. **W1 gap feature (INFERENCE).** That the first RETRO_TEST event's `log1p_prev_raw_view_gap_s` skips VALIDATION views. I verified
   this only for VS→VC (DATA-05:46-47) and from the per-partition STREAMS design; the TEST case needs the D1 count.
2. **W1 magnitude.** Unknown until D1. At VALIDATION, 39% of MAIN users have any VALIDATION decision (51,211 / 131,072), so a large
   share of MAIN_TEST decisions is likely affected (INFERENCE, not a TEST count).
3. **Cause of the VAL_A surge** (a sale around 16–17 Nov 2019?): UNVERIFIED. Only the decision-rate arithmetic is MEASURED.
4. **Production minimum/maximum example counts per client** (as an external justification of 5–100): UNVERIFIED. I found no verified
   source this session, and it is owner-fixed regardless.
5. **Repeat share of targets.** Not measured anywhere I could find. D5 would measure it.
6. **Does ANA-01 intersect user sets** for VALIDATION-vs-TEST per-user comparisons, as DATA-09 judgement 3 asks? Evaluation expert to
   confirm.
7. **The v1 TEST structural counts** (7,965,332 decisions, 3.383% OOV) are quoted from the published thesis reference, not from any
   RETRO_TEST row. Leads may drop them if they judge them out of bounds.

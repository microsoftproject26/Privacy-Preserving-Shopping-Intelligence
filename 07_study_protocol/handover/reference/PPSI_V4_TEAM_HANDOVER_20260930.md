# PPSI V4 — Team Handover: the federated / on-device deployment study

*Privacy-Preserving Shopping Intelligence (PPSI), MSc project, Queen's University (with a Microsoft supervisor).*
*Written 2026-09-30 from the project records. State as of decision-log row 268 (2026-09-30 14:50Z, server clock).*

---

## 0. How to read this document

**Who it is for.** Teammates who know the project idea and are ML-literate, but did not follow the last days. After reading
it you should be able to: (a) explain every design decision of the deployment study and why it was taken, (b) find the
evidence for each fact, (c) know exactly what is running, what is finished, what is sealed, and (d) continue the work without
breaking the pre-registration.

**Evidence tags.** Every number comes from a project file, cited next to it.
- **MEASURED**: read from a run record, a selection file or a receipt.
- **REPORTED**: a number stated in a project document (a review, the supervisor deck) that I did not re-derive.
- **INFERENCE**: an interpretation or estimate by the project team or a reviewer. It is not a result.
- **PLAN / ESTIMATE**: something that has not happened yet. **An ETA is never a result.**

**What you will NOT find here.**
- No MAIN_VALIDATION value and no TEST value of the deployment study. None has been read by anyone; they are sealed.
- No value from the UNSEEN_VAL selection panel. The selected cells are named, but the panel values are kept out on purpose.
- The only model-quality values in this document are:
  - **CALIBRATION (CALIB) values**: registered tuning reads on the 32,768 tuning users. They are labelled `CALIB` every time.
  - **Phase-2 contest values**: opened once at the registered G2 gate (DL 129) and final since then. They were already shown to
    the supervisor. They are labelled `contest, MAIN_VALIDATION, final`.

**Citation codes.** Keep them; they are how you trace evidence.

| Code | Meaning | Where |
|---|---|---|
| `DL n` | Row `n` of the decision log. Use the number in the `#` column, not the file line number (the line is about row + 5). Row "169" exists twice; the second one is cited as **169b** (DL 171). | `company/DECISION_LOG.md` |
| `06A` … `06M_B` | Pre-registration amendments. Each is written and hashed **before** the data it governs is read. | `contracts/PREG1_AMENDMENT_<id>.md` |
| `FL-14`, `GATE-B`, … | Task ids. Each task has a packet (the instructions) and a handoff (what the worker did, with file hashes and test counts). | `company/tasks/<id>.md`, `company/handoffs/<id>.md` |
| `REV-…` | Independent read-only code reviews of a task. | `company/reviews/REV-*.md` |
| `K1` … `K21`, `K9b`, … | Registered confirmatory contrasts (§7.3). | `company/gates/ANA_CONTRASTS_v3.json` |
| `S0` … `S8` | Stages of the federated improvement ladder (§9). | `company/registers/FL_IMPROVEMENT_LADDER.md` |

Paths are relative to `F:\Queens University\Microsoft Project\eda\.local_ai\ppsi_v4\` (called `V4/`) unless stated.
Some review and reference file names contain internal tool labels. They are cited here with a `*` wildcard, e.g.
`company/reviews/REV-DELTA-01_*_REVIEW.md`.

**Roles you will see in the records** (described neutrally here):
- **The owner**: the student who owns the thesis and takes the binding decisions.
- **The operations lead**: runs the server, writes most amendments, dispatches work, keeps the decision log.
- **The chief-scientist session**: a second operating session. For about a day it co-owned amendments and scientific
  acceptance (DL 169–215), then handed its duties back to the operations lead (DL 214–216).
- **Workers**: implement one task packet each, locally, with tests, and write a handoff.
- **Independent reviewers**: read-only; they return a verdict (ACCEPT / ACCEPT_WITH_FIXES / PASS / REJECT).
- **Research audits**: read-only literature and realism studies.

---

## 1. One-page summary

**Research question** (the proposal's question, unchanged; amendment 04 header, DL 86):
> *"How much shopping recommendation quality can be retained when moving from cloud-based to on-device or federated models?"*

**How it is answered.** Retention R = 100 × (arm quality) / (C_BIG quality).
- Quality = client-macro MRR@20 (per user, then averaged over users).
- Uncertainty: paired user bootstrap, 1,000 resamples, seed 20260926.
- Seeds: 2026 and 2027, mean of the per-seed ratios.
- **There is no pass/fail threshold** (amendment 04 §1, amendment 06 §7, 06A §7). The owner asked "how much is retained",
  not "is it acceptable" (DL 86).

**What is compared** (all at the same data, features, evaluator and exposure budget unless stated):

| Group | Arms | What leaves the device |
|---|---|---|
| Cloud | C_BIG (131,072 MAIN users, 83.9 M params), C_FULL (1,348,682 users), C_SMALL (device-size model trained centrally) | raw events |
| Federated | FA (the tuned federated recipe), FP (FedProx), PF (personalised FL), FA_Q8 (8-bit uploads), FA_1024 (1,024 clients / round, clipped), DP ε = 8 and ε = 1 | model updates (noised for DP) |
| Deployment branch | PRE (pretrained on 1,217,610 opted-in non-MAIN users) → FA_WARM → FA_WARM_FROZEN (S8), FT_PRE, FT_FA_WARM | the opted-in users' raw events; model updates from the rest |
| On-device | LO (local only), FT_C (cloud model fine-tuned locally), FT_FA (FL model fine-tuned locally) | nothing after the download |
| Baselines | LAST (repeat the last viewed item), POP, SKNN, RULE, KNN_EMB | — |

**Headline status (DL 268, 2026-09-30 14:50Z).**
- **Phase-2 contest: DONE.** SASRec D256_B3 was selected under the registered rule; ARCHITECTURE_LOCK written (DL 129–131).
- **Cloud references: DONE, sealed.** C_BIG checkpoints re-evaluated on the A40 (DL 169). C_FULL seeds 2026 and 2027 finished
  (DL 248, 259). The C_FULL s2026 and PRE re-evaluations finished, 13/13 jobs (DL 268).
- **PRE (pretraining for the deployment branch): DONE.** θ_pre = endpoint_6.0, sha cf08588c… (DL 252, 254).
- **Federated ladder, measured on CALIB:** 0.023473 (S0) → 0.028796 (S1b) → 0.034247 (S2) → **0.094936 (S3)**, i.e. ×4.04
  (DL 218, 249, 256).
  - S4 is RUNNING (DL 268).
  - The early S7 / S8 warm-start screens are RUNNING or queued (DL 268).
- **Warm-start preview on CALIB: DONE, descriptive.** Frozen item tables reached 0.1673. Full-model warm runs with the old
  client optimiser collapsed (DL 267).
- **On-device:**
  - LO: both seeds DONE (sealed).
  - FT_C and FT_PRE: RUNNING on MAIN (DL 268).
- **Not started:** the main federated lanes (FA, FP, PF, FA_Q8, S_CAL → FA_1024 → DP8 → DP1). They wait for the S4 selection
  file.

**What is sealed.**
- Every MAIN_VALIDATION value of the deployment study. No decision-maker has opened one.
- Every RETRO_TEST (TEST) row. It gets **one look**, after ALL_METHODS_LOCK (06D §2).
- PRE's values on CALIB. They were not read, because that would be a partial MAIN read (DL 253).
- The UNSEEN_VAL selection-panel values of the device selections. They are not quoted.

**What comes next** (PLAN, details in §14):
1. S4 selection, which writes FL_TUNING_SELECTION.
2. The MAIN federated lanes.
3. The S7 / S8 selection, then FA_WARM MAIN and FT_FA_WARM.
4. Fix the C_FULL s2027 re-evaluation gap.
5. The ANA freeze, then the single VALIDATION read.
6. ALL_METHODS_LOCK, then the single TEST look.
7. The thesis tables.

The supervisor deck estimates the VALIDATION read around 2–3 Oct, the TEST look around 4–5 Oct and the final tables around
6–7 Oct. **These are estimates only** (`company/presentations/DECK_SCRIPT_EN.md`, slide 22).

---

## 2. The research question and what changed from the proposal (and why)

The owner told the team that the **question** matters, not the specific tasks, models or data of the proposal (DL 86). Every
substitution below was agreed with the supervisor or taken by the owner. 06D §9(g) registers that the thesis must list each
substitution with its reason.

| Proposal item | What the project does | Why | Source |
|---|---|---|---|
| Tasks: next category, purchase AUC, comparable offers | **Next product viewed**: "the next different viewed item within the session, views only, given ≤ 50 prior events across sessions" | Next category was too easy to judge: ≈ 82 % of the metric is solved by repeating the current category, and there was no published baseline. Next product on REES46 has a published benchmark. Purchase-AUC and comparable offers are out of scope by agreement. | deck slides 4, 7; 06A §9(d); multi-expert review E5 |
| Datasets named in the proposal | **REES46** e-commerce behaviour logs, Oct–Nov 2019. Every user is simulated as one device (one persistent user = one logical client). | One realistic shopping log in which each user can be a device | deck slide 7; CLIENT_CONTRACT §1 C1.1 |
| Metric | **Client-macro MRR@20** over the **full catalogue** of 158,486 products (per user, then averaged). Secondary: NDCG@10 and HR@10 from the same frozen evaluator. | A browser serves one user, so per-user quality is the right unit. Full ranking avoids the known bias of sampled metrics. | amendment 04 §4; deck slides 7, 8 |
| FL methods: FedAvg, FedProx, personalised FL, FedBN | FedAvg (FA) with a **tuned** recipe, FedProx (FP, μ 0.01), personalised FL (PF), plus server optimisers FedAdam / FedAvgM, 8-bit uploads (FA_Q8) and a warm-started branch. **FedBN is dropped.** | The model uses LayerNorm, not BatchNorm, so FedBN does not apply (06A §9(c)). The owner declined FedLN (DL 86). Server optimisers are standard production practice (06G). | DL 86; 06A §9(c); 06G |
| "Various privacy levels" | **DP-FedAvg at ε = 8 and ε = 1** (ε = 4 optional), plus a privacy matrix L0–L6 | Makes privacy measurable instead of descriptive | 06B §3–§4; 06L_C C1 |
| An acceptability threshold (early design) | **Dropped.** Retention is reported with CIs and curves; there is no pass / fail verdict. | The question asks how much is retained | DL 86; amendment 04 §1 |
| Proposal bibliography | Corrected: [5] conflates two papers; [6] = IEEE Intelligent Systems 36(5) 2021; [8] Luo et al. CIKM 2022 vs PFedRec (Zhang et al. IJCAI 2023); [12] not found | Found by a research review | 06B §7(a); multi-expert review E7 |

---

## 3. The four project phases in brief

The deck presents four phases (`company/presentations/DECK_SCRIPT_EN.md`, slides 3–6; `DECK_GUIDE_AR.md`). Phases 1–3 are
context only. Their numbers are REPORTED from the deck and from the living thesis reference; they are not part of the
deployment study.

### Phase 1 — next category
- A GRU predicted the next category: 0.35, against 0.31 for a transition table and 0.10 for popularity (REPORTED, deck
  guide slide 4).
- About 82 % of the metric is solved by "repeat the current category", and nothing had been published for the task on this
  data.
- With the supervisor's agreement, the project moved to next product (deck slide 4).

### Phase 2 — cloud next-product model vs the published benchmark
- Against the published GRU4Rec results on REES46 (RecSys 2023), the project's model was close but lower (REPORTED, deck
  slide 5):
  - Recall@20: 0.43 vs 0.53;
  - MRR@20: 0.15 vs 0.20.
- The project's protocol is harder:
  - it ranks the whole catalogue;
  - new products always count as misses;
  - it predicts the following week, not the last day;
  - it used much less compute.
- It is **not** a same-protocol re-run. Whether to re-run the benchmark protocol is an open question to the supervisor
  (deck slide 10).
- The v1 GRU and v1 SASRec campaigns were evaluated retrospectively on TEST (≈ 0.169 / 0.170; `company/RESUME_HERE.md`
  §1 owner Q&A; thesis reference §4–§5).
  - **Consequence: the TEST week was already inspected in earlier phases.** This is disclosed (TEST_EXPOSURE_REGISTER
    entry 3, DL 33; 06B §7; 06I §7(c)).

### Phase 3 — which input columns help
- The v1-era feature study ended KEEP_CURRENT: no feature change passed its gates (thesis reference §6.4b).
- The V4 feature search reduced the schema from 58 input columns to 24. 21 of the 24 can be computed on the device itself
  (REPORTED, deck slide 6; multi-expert review §5 item 9).
- The locked schema **S\*** has 25 information channels = 24 input columns + the output catalogue (schema self c53d47fe…,
  DL 53–54).
- Registered acceptance rule (DL 41–43): the worse of GRU and SASRec may lose at most 1.0 % relative on the seed mean, with
  UB95 ≤ 2.0 %.
  - Result on the separate feature-search cohort (relative drop vs the full schema, not MAIN): GRU −0.90 %, SASRec −0.13 %.
    S* passed with no fallback (DL 53).
- Owner additions that shape every later model:
  - **USER_HISTORY_L50** (DL 36): the model input is the last 50 events of the user's whole prior stream, across sessions,
    strictly causal, identical for TRAIN / VALIDATION / TEST. Before this change the window was session-scoped.
  - **session-offset channel** (DL 47): the model knows which events belong to the current session and which to earlier
    sessions.
  - **SESSION_INTENT** (DL 48): the session's dominant main category, a droppable group. It survived the search (DL 56).

### Phase 4 — the deployment study (V4; this document)
**Scope** (DL 1, `OWNER_SCOPE_V4.md`):
- a GRU vs SASRec contest picks **one** backbone;
- the winner alone is used for the cloud, federated and on-device arms plus cold start and device measurement.

**Contest procedure.**
- **P2 — equal-effort tuning on CALIB** (amendments 02 → 02R1 → 02R2).
  - Model size was part of the search (owner, DL 60).
  - GRU {D128, D256}; SASRec {(256, 2), (256, 3), (384, 2)}.
  - 12 trials per family, then the best of each size re-run at a second seed (DL 61, 70).
- **P3 — 8 MAIN runs** (amendment 05): 2 candidates per family × 2 seeds, 6.0 EFE.
- **G2 — one sealed read** (DL 94–95).

**P2 results** (CALIB, two-seed means):

| Family | Configuration | CALIB MRR@20 | Source |
|---|---|---|---|
| GRU | D128 | 0.124844 | DL 91 |
| GRU | D256 | 0.115286 | DL 91 |
| SASRec | D256_B3 | 0.132402 | DL 108 |
| SASRec | D256_B2 | 0.130120 | DL 108 |
| SASRec | D384_B2 | 0.127326 | DL 108 |

**G2 result** (contest, MAIN_VALIDATION, final; DL 129):

| Rule | GRU D128 (2 seeds) | SASRec D256_B3 (2 seeds) |
|---|---|---|
| Registered rule T1 = PRACTICAL_BEST (best validated mark within 6.0 EFE) | 0.157141 / 0.156733 (best mark 6.0) | **0.161196 / 0.161221** (best mark 3.0) |

- delta_bar GRU − SASRec = −0.004271, CI95 [−0.005180, −0.003261] → **DECISIVE_STABLE, winner SASRec** (recipe 02B_SASREC).
- NDCG@10: 0.19105 (GRU) vs 0.19668 (SASRec).
- The CALIB-disjoint diagnostic agrees (−0.004103).

**Mandatory disclosure (DL 138).** At the fixed 6.0 endpoint (CONTROLLED_BUDGET) the order **reverses**:
- GRU − SASRec = +0.004213, CI [+0.003283, +0.005216].
- SASRec peaks early and then overfits; GRU was still improving at 6.0.
- The thesis says "SASRec was selected under the registered rule T1", **never** "SASRec beats GRU" (06B §7(c); multi-expert
  review E6).

**Baselines on MAIN** (post-G2, REPORTED, never used for selection; DL 132):

| Baseline | MRR@20 |
|---|---|
| POP | 0.01943 |
| LAST | 0.07111 |
| SKNN | 0.11138 |

LAST reaches about 44 % of C_BIG (0.0711 / 0.1612; multi-expert review §1 item 2). This drives the "anchor on LAST" rule
(§6.4).

**Supervisor meeting (30 Sep 2026).** The deck and the speaker script are in `company/presentations/`:
- `PPSI_Supervisor_Deck_Final.pptx`;
- `DECK_SCRIPT_EN.md`;
- the Arabic slide-by-slide guide `DECK_GUIDE_AR.md` (DL 233).

The deck asks the supervisor four groups of questions:
1. Is the task right? Should the benchmark protocol be re-run? Which cloud should lead the retention figures?
2. Which deployment should be the Edge headline? Should LAST be the floor?
3. How much tuning is fair for FL? Are ε = 8 and 1 enough? Is pretraining on opted-in users realistic?
4. What should be cut first if time runs short? Is one browser run enough? Can the supervisor pre-approve the AI-use statement?

The meeting outcome is not in the records I read.

---

## 4. Data, cohorts and splits

### 4.1 Source and frozen data
- **Raw data.** REES46 October 2019 raw file `data/raw/2019-Oct.csv`: 5,668,612,855 bytes, 42,448,764 events (DL 234).
  - The mean line size, 133.54 bytes per raw event, is the "raw events" level of the bytes axis.
  - That measurement rule was registered before the measurement (DL 229, 234).
- **Frozen v1 dataset.** `rees46_nextitem_v1`, hash 860e5a1c…, 409 files verified. Frozen preprocessing is immutable
  (thesis reference §2.1).
- **V4 data version.** CLEAN_V2_POP5100_V2, 15 manifests (DL 50).
  - Data lock: `contracts/DATA_PROTOCOL_LOCK_V2.json`, lock_self bc699a97… (DL 54).
- **Catalogue A.** Global support ≥ 5: **158,486 products**. Every prediction ranks all of them (CLIENT_CONTRACT §2;
  deck slide 8).

### 4.2 Population and cohorts

| Population | Definition | Users | TRAIN decisions / examples | Source |
|---|---|---|---|---|
| Universe U | users with ≥ 1 TRAIN-eligible example (support ≥ 5 catalogue) | 2,694,998 | — | CLIENT_CONTRACT §2 C2.1 |
| All strata | TRUE COLD (0) + SPARSE (1–4) + MAIN (5–100) + HEAVY (> 100) | 4,804,933 | 40,180,608 | DL 33 |
| **5–100 pool** | owner amendment POP 5–100: 5 ≤ n_eligible_TRAIN_examples ≤ 100, from TRAIN only | **1,446,986** | **28,623,386** (71.24 % of supervision, median 13) | DL 31; thesis reference §14.10 |
| **MAIN** | the first 131,072 pool users by selection key v2 (ranks [0, 131,072)) | **131,072** | 2,587,776 | DL 41–42, 50 |
| **CALIB** | nested prefix of MAIN (ranks [0, 32,768)); the only users used for tuning | **32,768** | 646,067 | DL 42, 50 |
| **TIEBREAK band = C_FULL population** | ranks [0, 1,348,682) = the pool minus the unseen bands and the feature-search band | **1,348,682** | 26,677,600 | DL 151, 162; 06D §7 |
| **PRE (pretraining) cohort** | TIEBREAK minus MAIN = ranks [131,072, 1,348,682); disjoint from MAIN, CALIB, U_NEW, FS and every panel (band hash 716beeb9…) | **1,217,610** | 24,089,824 | DL 207; DATA-10 handoff |
| U_NEW_VAL (role UNSEEN_VAL) | ranks [1,348,682, 1,381,450); never trained on; the **selection panel** for device recipes | 32,768 (12,792 evaluable) | — | DL 42; 06A §1 |
| U_NEW_TEST (role UNSEEN_HOLDOUT) | ranks [1,381,450, 1,414,218); never trained on; participation-gap / cold-start panel | 32,768 | — | DL 42 |
| FS | ranks ≥ 1,414,218, the top 32,768; used only by the feature search | 32,768 | — | DL 42 |
| Panels | TRUE_COLD 381,295; SPARSE and HEAVY panels (16,384-user adaptive subsets); WARM_TRUNCATION 16,384; COLD_H 16,384 | see left | — | DL 40–43, 50 |

Key points:
- **Selection key v2.** Each user gets a pseudonymous key, sha256 of a salted user id (CLIENT_CONTRACT §1 C1.2). Ranks by that
  key define every band, so every cohort is a deterministic, outcome-free prefix or range. The salt is
  `PPSI_V4_POP_5_100_SEL_20260926` (DL 40).
- **"Full pool" is not 100 % of the pool.** The C_FULL "full-pool cloud" uses ≈ 93 % of the 5–100 pool (1,348,682 of
  1,446,986), because the unseen and feature-search bands are held out. The public doc was corrected for this (DL 236).
- **Why 5–100 events per user.**
  - Every simulated device needs a realistic history: users with one event have nothing to train on.
  - Users with thousands of events are often bots.
  - Source: deck guide slide 8; owner amendment DL 31.
- **MAIN_VALIDATION and CALIB_VALIDATION.**
  - MAIN_VALIDATION: 51,211 users with evaluation rows.
  - CALIB_VALIDATION: 12,715 users.
  - CALIB-disjoint MAIN users: 38,496 (enforced counts, DL 115).
  - CALIB_VALIDATION overlaps MAIN_VALIDATION by 24.8 % of users (thesis reference §14.11). This is why selecting device
    recipes on CALIB was abandoned (06A §1).

### 4.3 Chronological split
- **TRAIN:** 1 Oct – 15 Nov 2019 (target_ts < 2019-11-16).
- **VALIDATION:** 16–22 Nov.
- **RETRO_TEST:** 23–29 Nov.

Sources: thesis reference §14.11; CLIENT_CONTRACT §2 C2.1; deck guide slide 8.

- The split is **global** (one timeline for every user), so there is no future leakage (multi-expert review §5 item 3).
- **TEST is sealed until ALL_METHODS_LOCK and gets one look.** It was previously inspected by the v1 campaigns (§3).
- **TEST contains Black Friday (29 Nov 2019).** The VALIDATION→TEST change of C_BIG is reported as a shift indicator
  (06B §7(b)).
- **A known TEST detail to disclose.** At TEST, some user-context features replay TRAIN then TEST and skip the VALIDATION week
  (the "partition fence"). Found by the multi-expert review (§3.1); disclosure item D2 before the TEST look.

---

## 5. Models and payload sizes

### 5.1 Model inventory

| Name | Variant / recipe | Params | Trained on | Notes | Source |
|---|---|---|---|---|---|
| **C_BIG** (the cloud model) | `02B_SASREC` = **SASREC_D256_B3** (d 256, **3 blocks**) | **83,881,764** | MAIN, 6.0 EFE, seeds 2026 / 2027 | These are the P3 contest runs themselves (no re-run). PRACTICAL_BEST = best.pt at mark 3.0; CONTROLLED_BUDGET = endpoint_6.0.pt. Re-evaluated on the A40 for the deployment block. | ARCHITECTURE_LOCK `parameter_counts`, `c_big.checkpoints`; DL 131, 169, 257 |
| **C_SMALL** (device-size model trained centrally) | `SMALL_SASREC_D64_B2` = **SASREC_D64_B2** (d 64, 2 blocks, 4 heads of dim 16) | **20,710,048** | MAIN, 6.0 EFE, seeds 2026 / 2027 | Recipe from a 4-LR mini-ladder on CALIB: peak LR 0.000711, S0 schedule, dropout inherited from 02B_SASREC | amendment 03 §1, §3; amendment 06 §1; DL 143, 145 |
| **C_FULL** (full-pool cloud) | same as C_BIG (02B_SASREC) | 83,881,764 | TIEBREAK band (1,348,682 users), seeds 2026 (confirmatory) and 2027 (descriptive) | Automatic continuation up to 8.0 EFE allowed, as for C_BIG. Recipe not re-tuned at this scale, a disclosed upward bias of the vs-C_FULL ratios. | amendment 06 §5; 06D §7; 06E §2; DL 151, 176; review E9 |
| **PRE** (θ_pre) | SASREC_D64_B2 with the SMALL recipe | 20,710,048 | PRE cohort (1,217,610 users), seed 2026 | θ_pre = endpoint_6.0 (no checkpoint selection). sha cf08588c…, 82,853,197 bytes. | 06K §1; DL 252, 254 |
| GRU_D128 (contest loser) | `02B_GRU` | 20,962,210 | MAIN | Kept only as a descriptive size-matched family comparison against C_SMALL (evaluation only) | ARCHITECTURE_LOCK; 06L_C C11 |

Initial weights θ0 per seed are registered.
- SASREC_D64_B2: s2026 e9eb0bb9…, s2027 a5d1e683… (DL 134).
- The FL server starts from the registered θ0 and asserts its hash (DL 17).

**Two model facts matter for privacy and communication:**
1. The input and the output item tables are **untied**, and together they are ≈ 98 % of the parameters. So a non-DP
   update's non-zero input-table rows reveal exactly which items the user viewed (06I §7(a); multi-expert review §3.1).
2. The device model differs from C_BIG in depth as well as width (B3 → B2). This is disclosed, because the size step
   K1 = R(C_SMALL / C_BIG) is not a pure width change (multi-expert review, DISCLOSE list).

### 5.2 Payload per federated visit and other byte levels

| Quantity | Bytes | Source |
|---|---|---|
| Cloud-size model, one FL visit (FP32 download + upload) | 671,054,112 (≈ 671 MB) | ARCHITECTURE_LOCK `winner_payload` |
| Device model, one FL visit (FP32 download + upload) | 165,680,392 (≈ 166 MB) | ARCHITECTURE_LOCK; amendment 06 §1 |
| **Registered device cap** | **≤ 170 MB per visit** (so only the device-size model is federated) | amendment 03 §1 |
| FA_Q8 visit (8-bit uploads, FP32 download) | ≈ 104 MB (REPORTED). Upload = Σ numel (int8) + 4 × n_tensors (scales) + the 8-byte header | deck slide 12; 06D §6 |
| S8 (frozen item tables) upload | 265,162 parameters ≈ 1.06 MB per direction (to be re-measured). The frozen tables (81.78 MB) are downloaded **once**, at the first visit. | multi-expert review §3.2; 06L_C C8; DL 228 |
| Inference-only download (no training) | 82.8 MB FP32 / ≈ 21 MB INT8 (REPORTED) | multi-expert review E9 |
| One raw event (the "raw events" privacy level) | 133.54 bytes | DL 234 |

**Two things to state separately in the thesis** (multi-expert review E9):
- The inference download and the FL payload are different quantities.
- The origin of the 170 MB cap must be stated. It came from the GRU_D128 payload (≈ 168 MB) at design time (amendment 03 §1).

---

## 6. The methods matrix

### 6.1 Federated knobs shared by all FL arms
Registered knobs (amendment 04 §3; 06B §1):
- 64 clients per round, sweep sampling, 10 % seeded drop-out.
- Local batch 16 × 2 passes.
- n_consumed-weighted aggregation.
- Budget 6.0 EFE ≈ 3 visits per client:
  - MAIN: **6,827 rounds**;
  - CALIB: **1,707 rounds** = ceil(3 × 32,768 / 64 / 0.9).
- 12 evaluation marks, one every 0.5 EFE.
- FedProx μ = 0.01, not tuned.

Stateless clients: nothing is kept on a device between rounds, except PF's personal vector, which never leaves the device.

**The federated recipe.** Since 06I §2, "FA" means the **one recipe selected on CALIB by the ladder**: server optimiser, local
LR, LR shape, client optimiser, passes / batch. Every FL arm (FA, FP, PF, FA_Q8, FA_1024, S_CAL, DP) uses it.
- There is no separate "FA_M" arm and no headline swap.
- DP arms are the exception: they keep 2 passes / batch 16 whatever S4 selects (DL 219), because their T and z are
  registered.

### 6.2 Arms, what they are, and what leaves the device

| Arm | What it is | Seeds | What leaves the device (training / inference) | Privacy level (06L_C C1) |
|---|---|---|---|---|
| C_BIG | cloud SASREC_D256_B3 on MAIN | 2026, 2027 | raw events / query context | L0 |
| C_FULL | the same recipe on 1.35 M users | 2026 (+2027 descriptive) | raw events | L0 |
| C_SMALL | device-size model trained centrally | 2026, 2027 | raw events once; can be shipped for on-device inference | L0 / L1 when shipped |
| **FA** | federated, the tuned recipe | 2026, 2027 | model deltas (dense FP32); no secure aggregation | L2 |
| FP | FA + FedProx μ 0.01 | 2026, 2027 | deltas | L2 |
| PF | FA + a personal vector p_u kept on the device (λ = 1e-4) | 2026, 2027 | shared-part deltas | L2 (shared) / L4 |
| FA_Q8 | FA with int8 stochastic-rounded uploads | 2026, 2027 | 8-bit deltas | L2 |
| S_CAL | clip-norm calibration: median client-delta norm of an **unclipped** pass with the FA_1024 configuration, seed 2026, rounds 1–20 → S_RECORD | 2026 | deltas | (tool run) |
| FA_1024 | 1,024 clients per round with Poisson sampling, clipping on, no noise; the matched non-DP twin of DP | 2026, 2027 | clipped deltas | L2 |
| DP8 / DP1 | DP-FedAvg, ε = 8 / ε = 1, δ = 5e-6 | DP8: 2026, 2027; DP1: 2026 | clipped deltas + Gaussian noise on the sum | L3 |
| DP_OPT_SENS | FA_1024 s2026 + DP8 s2026 re-run under plain FedAvg (sensitivity). **Triggered**, because the S2 / S3 winner is FedAdam. | 2026 | as DP | L3 |
| FA_X2 | FA at 12.0 EFE; descriptive budget sensitivity | 2026 | deltas | L2 |
| PRE | θ_pre evaluated alone (no training on MAIN users) | 2026 | opted-in users' raw events only | L1 on opted-in data |
| **FA_WARM** (S7) | FA started from θ_pre instead of θ0 | 2026, 2027 | deltas | L2 with pretraining on opted-in users |
| **FA_WARM_FROZEN** (S8) | FA_WARM with both item tables frozen and not transmitted; only the encoder is trained / uploaded | 2026, 2027 (06L_C C8) | encoder deltas; brand / category deltas may still leak | L2 |
| DP8_FROZEN | S8 + DP ε = 8 | 2026 | noised encoder deltas | L3 |
| LO | local only: the device trains its own model from θ0; the no-collaboration control | 2026, 2027 | nothing | L6 |
| FT_C | C_SMALL downloaded, then fine-tuned on the device | 2026, 2027 | nothing after the download (but the base was trained on raw events of the same users) | L1 base + local FT |
| FT_FA | FA downloaded, then fine-tuned on the device | 2026, 2027 | as FA | L4 |
| **FT_PRE** | θ_pre + the FT_C local procedure: the cleanest download-only option | 2026, 2027 | nothing (base from opted-in users only) | **L5** |
| FT_FA_WARM (+ _FROZEN) | θ_pre → FA_WARM (or S8) → local FT: the full production stack; descriptive | 2026, 2027 | as FA_WARM | L4 |
| LAST | repeat the last viewed item | — | nothing | L6 |
| RULE / POP | category / brand affinity / popularity (uses pooled MAIN-TRAIN statistics) | — | device + published aggregate | — |
| SKNN, KNN_EMB (BIG / SMALL) | session kNN; kNN on a cached item-embedding table | — | — | — |

Sources: amendment 04 §2; 06 §2; 06B §3, §5; 06C; 06D §5; 06E §1–§2; 06H §3; 06I §4–§5; 06K §1–§3; 06L A3; 06L_C C1, C7,
C8, C16–C17; 06M_B B4; DL 219.

**Deferred or not adopted:**
- CPF, KD_SMALL, FT_KD, DP ε = 4 and seed 2028 are NICE tier (06D §8; 06I §8).
- DP8_FULL (DP at 1.35 M-user scale: m 8,192, T 494, δ 5e-7) is registered in 06I §5 but stays NICE, with no launch path
  written (06L_C C16–C17).
- C_SMALL_FULL was not adopted (owner, 06L_C). Neither was a December-2019 holdout (it would need a new data version).

### 6.3 Selection panels for the device arms
- Device recipes (FT grid: passes {1, 2, 4} × LR fraction {0.1, 0.3} of the SMALL peak; LO passes {1, 2, 4}) are
  selected **once**, on the **UNSEEN_VAL** role.
- UNSEEN_VAL users are never trained on and are disjoint from MAIN, UNSEEN_HOLDOUT and every report panel.
- The selected cell goes to `contracts/device/SELECTION_*.json` (O_EXCL, i.e. written once and never overwritten) with a DL
  row, before any MAIN application (06A §1; 06E §1).
- An earlier plan selected on CALIB. It was replaced because CALIB is the rank-0 prefix of MAIN, so 25 % of the report users
  would have taken part in the selection (06A §1).

### 6.4 Anchoring on LAST
LAST needs no training and reaches ≈ 44 % of C_BIG (DL 132). So every answer is also reported against LAST (06L_C C3):
- the paired Δ(arm − LAST) with its CI;
- the headroom R\* = (arm − LAST) / (C_ref − LAST);
- a flag BELOW_DEVICE_HEURISTIC when the CI lies below 0;
- a new confirmatory contrast **K11b = FA − LAST**;
- the device-only envelope {LO, LAST, RULE}.

LAST is evaluated on the CPU and paired with A40 rows as a registered exception: it is model-free, so the device class cannot
change its ranks (DL 230).

---

## 7. The pre-registration discipline

### 7.1 The rules
1. **Register first, read later.** A rule, grid, contrast or arm is written into an amendment, hashed, and logged in the DL
   **before** any value it governs is read. "No rule change after its value is read" is the core invariant. An amendment
   that arrives after a read is refused, or explicitly labelled POST_VALIDATION_EXPLORATORY and never confirmatory
   (06L_C C5).
2. **Amendments are never overwritten.** Corrections go into new files or new DL rows. The one overwrite accident (06F, DL
   185) was reverted byte-identically (DL 188).
3. **Tuning happens only on CALIB.** Device-recipe selection happens only on UNSEEN_VAL.
   - MAIN_VALIDATION is read **once**, at the frozen analysis preflight.
   - TEST is read **once**, after ALL_METHODS_LOCK (06B §6; 06D §2).
4. **Selections are mechanical.** A selection file is written by a reviewed tool (`v4int/lr_screen.py`, `launch_device
   --select`), not by hand. The operations lead copies it O_EXCL into `contracts/` and writes a DL row.
5. **The "no read before selection" discipline.** Values of a stage enter the ladder register only after that stage's
   selection file exists (06J §4(vi)). Before the S2 read, only refusal texts of the selector were looked at (DL 245, 247).
6. **The analysis script is frozen by hash** (`company/gates/ANA_SCRIPT_FREEZE.json`, not yet written) before the
   VALIDATION read (06A §8).

### 7.2 The amendments (one line each)
The earlier amendments 01–06 set the contest and the downstream design. The 06A–06M_B series (the last days) governs the
deployment study.

| Id | Date (UTC) | What it registered | DL |
|---|---|---|---|
| 01 | 2026-09-26 | Clarifications: tie chain, manifest ids, endpoint tolerance, training determinism, eval_stream = EVAL_ROWS_ONLY | 26–27, 55 |
| 02 | 09-27 | First P2 contest design (superseded before use) | 59 |
| 02R1 | 09-27 | Model size in the P2 search, full 6 EFE, no legacy anchoring | 60–62 |
| 02R2 | 09-27 | LR grid stratified within each size (fixes size / LR confounding) | 69–70, 74 |
| 03 | 09-27 | BIG / SMALL: one family, two sizes; device cap ≤ 170 MB per visit; SMALL = SASREC_D64_B2 if SASRec wins | 83–84 |
| 04 | 09-27 | Realistic downstream plan: retention with no threshold; arms; DP-FedAvg; baselines; ONNX; strata | 86–87 |
| 05 | 09-27 | CALIB P2 is a screen only; 2 candidates × 2 seeds = 8 MAIN runs; one sealed two-step G2 | 93–95 |
| 06 | 09-28 | Downstream execution after G2: SMALL recipe; FL open questions Q1–Q5 (schedule, drop-out 0 for DP, S rule, sampling, RDP conversion; z(ε 8) = 0.5642); deferred arms; GPU sharing; C_FULL spec; strata; retention rule | 133 |
| **06A** | 09-28 ~17:10Z | Device selection panel = UNSEEN_VAL; cold start primary = observational support-count bins {0, 5–10, 11–20, 21–50, 51–100}; communication curves from the 12 marks; deferred-tier order; A40 block; ONNX INT8 row subset; retention pairing rule; ANA script freeze; disclosures | 146 |
| **06B** | 09-28 | FL local-LR screen m ∈ {0.5, 1, 2} on CALIB at the full 6.0 EFE; contrasts K1–K16; privacy ladder L0–L6; DP ε = 1 promoted to core; FA_Q8 arm; C_FULL s2026 before the lock; DATA-07 cold start SHOULD; browser measurement | 147–148 |
| **06C** | 09-28 ~20:45Z | DP sampling corrected to **Poisson** with a fixed denominator (the fixed-size sampler did not match the accountant) | 150 |
| **06D** | 09-28 21:50Z | A40 as a second registered device class (pin file); an executable TEST look (manifests, tools, order); K12 / K13 confirmatory at CONTROLLED_BUDGET; hierarchy (K2 primary); still-improving flag + FA_X2; FA_Q8 bytes; C_FULL = 1,348,682 users; priority tiers MUST / SHOULD / NICE; DP noise-to-signal note | 159 |
| **06E** | 09-29 08:42Z | LO cell selected once on UNSEEN_VAL; registered seed sets (K13 / K15 / K16 single-seed); K16 in an ONNX-CPU sub-block; comparison-key and evidence-integrity rules; sanity trigger; PF best-mark store; device-class continuity on resume | 169b, 170, 171 |
| **06F** | 09-29 09:20Z | LR screen extended to m ∈ {4, 8, 16} (+32 only if 16 wins); tie → closest to 1 by \|log2 m\|; the m = 2 MAIN runs started at 09:12Z are superseded | 184, 185, 188 |
| **06G** | 09-29 (DL row 09:22Z; header ~09:35Z) | **FedAdam** server optimiser (Reddi et al. 2021, Alg. 2; β1 0.9, β2 0.99, τ 1e-3; η_s ∈ {0.01, 0.1}); joint selection → FL_TUNING_SELECTION; co-scheduling with central runs allowed. Clarification: v0 = τ² (DL 193) | 183, 186, 193 |
| **06H** | 09-29 09:25Z | Review of 06G: under DP, FedAdam may turn noise into a random walk of size ≈ η_s → conditional sensitivity DP_OPT_SENS (FA_1024 + DP8 under FedAvg) | 189 |
| **06I** | 09-29 (header 09:41Z; logged 12:12Z) | K17–K19 = R(FA / FT_C / LO over C_FULL); "FA" = the tuned recipe; tuning wave 3 = FL_CONST LR shape × client optimiser; FA_X2 SHOULD-high; DP8_FULL registered; interpretations (equal-exposure, DP scale, recipe-qualified claims); realism disclosures; cut order | 196, 199 |
| **06J** | 09-29 ~12:33Z | The **FL ladder** S2 → S3 → S4, strictly sequential; FedAvgM added to S2; edge rules; **≥ 0.002 incumbent rule**; 0.0005 tie band; stopping rules | 202, 204 |
| **06K** | 09-29 12:39Z | S7 = PRE + FA_WARM (warm start), S8 = FA_WARM_FROZEN (frozen tables); K20 = R(FA_WARM / C_BIG), K21 = FA_WARM − PRE; never warm-start from a MAIN-trained model | 206, 207 |
| **06L** (part A) | 09-29 ~18:00Z | FedAvgM at m ∈ {1, 2} (+ m 0.5 edge) added to S2; S3 combination cells (client SGD × FedAdam η 0.01 × FL_CONST); DP_OPT_SENS for any non-FedAvg winner; clipped-fraction reporting; PF and FedProx interpretations | 226 |
| **06L_C** | 09-29 ~18:10Z | Privacy matrix; threat model; LAST anchoring + K11b; contrast registry v3; frozen arm set; deployment branch secondary-confirmatory (K9b, K9c, K20b, K20c); S8 two seeds; K16 quantiser cell; cold-start minimal set; GRU_D128 descriptive; claim register; statistics additions; DP8_FROZEN | 227 |
| **06L_D** | 09-29 ~22:00Z | Implementation reading of 06L (exact S2 / S3 run sets, edge order, dedupe, tie order) so the selector can execute it | 237 |
| **06L_E** | 09-29 ~22:55Z | Under DP (z > 0) **nothing** about clipping is recorded; the z = 0 twin's clipped fraction is the labelled proxy | 240 |
| **06M** | 09-30 ~08:15Z | Early warm-start **preview** on CALIB at the S2 point; descriptive only, never in any selection | 252 |
| **06M_B** | 09-30 ~08:25Z | Preview extension (η_s 0.001; warm FedAvg); 2-D S7 / S8 grids (local × server step); S3b late-phase contingency; FT_FA_WARM; rollout-quality note | 253 |

**Registrations made in DL rows** (same force as amendments; each one made before the affected read):

| DL | What it registered |
|---|---|
| 198 | LADDER_S1a MAIN row, descriptive only |
| 219 | DP arms keep P 2 / B 16 |
| 222 | The ≥ 0.002 rule reads raw, unrounded values; a gain within float noise of 0.002 counts as < 0.002 |
| 229 | Raw-event bytes rule |
| 230 | Ratification of ANA_CONTRASTS_v3 and of the LAST exception |
| 242 | Tie-case disclosures m2–m4 |
| 261 | S7 / S8 in-band tie order; no TRAIN_LOSS log for z > 0 |
| 263 | The winner's own gain is reported beside the file's gain |

### 7.3 The confirmatory contrasts (the K list)
The registered family has **26 confirmatory contrasts**. The file is `company/gates/ANA_CONTRASTS_v3.json`, sha a3d1456d…,
ratified in DL 230. Everything else is DESCRIPTIVE: strata, bins, panels, curves, marks, NDCG / HR, per-seed values.

Unit of every contrast:
- client-macro MRR@20;
- PRACTICAL_BEST primary, CONTROLLED_BUDGET reported;
- paired user bootstrap;
- ratios bootstrapped directly.

Sources: 06B §2, 06D §3–§4, 06I §1, 06K §2, 06L_C C3, C4, C7.

| K | Contrast | Question | Tier |
|---|---|---|---|
| K1 | R(C_SMALL / C_BIG) | size cost | secondary |
| **K2** | R(FA / C_BIG) | **federated retention (headline)** | **primary** |
| K3 | R(FA / C_SMALL) | federation cost at matched size | secondary |
| K4 | R(FP / C_BIG) | FedProx retention (interpretation fixed by 06L A7) | tertiary |
| K5 | R(PF / C_BIG) | personalised-FL retention ("PF ≈ FA by construction" is a registered possible outcome, 06L A5) | tertiary |
| K6 | R(FT_FA / C_BIG) | FL + local fine-tune | tertiary |
| K7 | R(FT_C / C_BIG) | cloud model + local fine-tune | tertiary |
| K8 | R(LO / C_BIG) | device-only retention | secondary |
| K9 | FT_C − C_SMALL | local fine-tuning gain | tertiary |
| K9b | FT_C − C_SMALL on UNSEEN_HOLDOUT (seed 2026) | fine-tuning gain on never-seen users | secondary (deployment branch) |
| K9c | FT_PRE − PRE | fine-tuning gain on the opted-in model | secondary (deployment branch) |
| K10 | FT_FA − FA | local fine-tuning gain after FL | tertiary |
| K11 | FA − LO | collaboration value | tertiary |
| K11b | FA − LAST | FL vs the free device heuristic | secondary |
| K12 | R(DP8 / FA_1024) at CONTROLLED_BUDGET | privacy cost at ε = 8 | secondary |
| K13 | R(DP1 / FA_1024) at CONTROLLED_BUDGET (seed 2026) | privacy cost at ε = 1 | tertiary |
| K14 | R(FA_Q8 / FA) | compression cost; equivalence margin ±1 % (90 % CI inside [99, 101]) | tertiary |
| K15 | R(C_BIG / C_FULL) (seed 2026) | cloud data advantage | tertiary |
| K16 | R(INT8 / FP32) for C_SMALL, inside an ONNX-CPU sub-block (seed 2026) | on-device quantisation cost | tertiary |
| K17 | R(FA / C_FULL) | FL vs the full-data cloud (an algebraic composition, K2 × K15) | secondary |
| K18 | R(FT_C / C_FULL) | | tertiary |
| K19 | R(LO / C_FULL) | | tertiary |
| K20 | R(FA_WARM / C_BIG) | warm-started FL retention | secondary |
| K20b | R(FA_WARM / C_FULL) | deployment-branch retention vs the full-data cloud | secondary (deployment branch) |
| K20c | R(PRE / C_FULL) | the opted-in cloud model vs the full-data cloud | secondary (deployment branch) |
| K21 | FA_WARM − PRE | what federation adds on top of the opted-in model | secondary |

Also registered: S8 − S7 (FA_WARM_FROZEN − FA_WARM) is DESCRIPTIVE, reported with its bytes ratio (06L_C C8).

**Multiplicity:**
- Unadjusted 95 % CIs are reported.
- A Holm band and a Bonferroni band are added beside them, with the family size = the registered 26 (06L_C C13; ANA-04;
  FIX-DELTA-03 M3).
- The hierarchy (K2 primary) is stated in the thesis.

### 7.4 Selection rules used by the ladder (all on CALIB_VALIDATION, E2E client-macro MRR@20 at PRACTICAL_BEST)
- **06B (S1a):**
  - pick the highest value;
  - |Δ| ≤ 0.0005 counts as a tie;
  - a tie goes to the m closest to 1 (order 1, 0.5, 2).
- **06F (S1b):** the same rule over the whole screened set; ties go to the smallest |log2 m|, then the smaller m. m = 32 was
  screened only if 16 won (it did not).
- **06J §4(ii) incumbent rule (S2, S3, S4, S7 / S8):**
  - The previous stage's winner is the incumbent.
  - The stage's best candidate replaces the incumbent **only if it gains ≥ 0.002**. Otherwise the stage is recorded "no
    measurable gain" and the incumbent carries on.
  - 0.002 is about the paired CALIB half-width (INFERENCE, ladder research §3).
  - Values are compared raw, unrounded (DL 222).
  - Then the 0.0005 tie band applies, with a registered tie order: simpler first, i.e. FedAvg → FedAvgM → FedAdam, then
    smaller |log2 m|, then smaller η_s, then larger τ.
- **Edge rules (one step each, no further extension).** When the best cell sits at the edge of a registered grid, one extra
  registered run is required before the stage file can be written:
  - FedAdam η_s 0.01 best → η_s {0.003, 0.001};
  - FedAvgM m 1 best → m 0.5;
  - FedAdam wins → τ 1e-5;
  - SGD lr 1.0 best → {3, 10};
  - SGD lr 0.01 best → 0.003.

  The selector refuses with `EdgeRunsMissing` until the edge runs exist (FIX-DELTA-02 M3; DL 245, 247).
- **Disclosed consequence of the literal rule** (DL 263). The ≥ 0.002 test is on the stage's best candidate, and the winner
  is the first cell of the 0.0005 band under the tie order. So the winner's own gain can lie in [0.0015, 0.002). The ladder
  register reports the winner's own gain beside the file's gain.

### 7.5 The single TEST look (06D §2)
**Building the TEST manifests.** They are built only after ALL_METHODS_LOCK exists:
- builder DATA-09;
- the lock's sha is recorded in a TEST_EXPOSURE_REGISTER entry;
- the gate is re-validated on every call.

The manifests are MAIN_RETRO_TEST_5_100_131072, UNEW_HOLDOUT_RETRO_TEST_32768, TRUE_COLD_RETRO_TEST and the SPARSE / HEAVY
panel TEST manifests.

**What happens at the look:**
- Every locked central / FL checkpoint is re-evaluated once by `v4int/reeval.py`.
- The device arms **re-run their locked procedure**, because per-client weights are not stored.
- The baselines and the ONNX rows run on MAIN_TEST.
- Then the frozen analysis script runs.

**Rules around it:**
- Nothing is re-run after the look, except an interrupted computation with identical hashes.
- The review's item D1 asks to restore the registered TEST contest singles (02B_GRU and 02B_SASREC on the A40), or to
  withdraw them in a dated amendment (multi-expert review Gate D).
- The code path is DATA-09 + INT-07. The analysis script refuses TEST for any contrast file other than v3 (FIX-DELTA-03 M5).

---

## 8. Infrastructure

### 8.1 Hardware and access
- **Server.** "Lobot", the university's shared Jupyter-based group server. Our group was granted up to 2 × A40 (DL 139). The
  disk went from 50 to 350 GB after a shutdown for the upgrade (DL 139–145).
  - After the upgrade the server first came back with 2 × A16 (DL 149). It was restarted on the A40 profile (DL 158).
  - Measured then: 2 × NVIDIA A40, 46,068 MiB each; CPU guarantee 16; cgroup memory limit 384 GiB; 345 G disk; torch
    2.13.0+cu130.
  - Later: nproc 128 and cpu.max unlimited (DL 168).
  - Environment lock archived: `deploy_archive/ENV_LOCK_A40.txt`, `ENV_PKGS_A40.txt` (106 packages; DL 226).
- **GPU allocation.** The owner first gave the project GPU 0 only (DL 161), then the whole server, both A40s (DL 168). He then
  asked for full utilisation (DL 176–177).
- **Access.** A remote execution channel to the Lobot server (no SSH). A local helper script sends shell scripts and fetches
  files. Credentials are never printed or stored in project files.
- **Local machine.** Windows. The project venv is under `next_item_eda/.venv`. Test temp dirs live on F: because C: is nearly
  full (DL 140).

### 8.2 Two device classes and the frozen evaluator
- **Two evaluation blocks.**
  - The contest was evaluated on A16 GPUs (`A16-FP32-DET`).
  - The deployment study runs on the A40, registered as a second frozen class `NVIDIA_A40-FP32-DET` by an O_EXCL pin file
    `contracts/eval_device/EVAL_DEVICE_PIN_A40.json` (self db744ca3…, written 2026-09-29 00:37:27Z; 06D §1; DL 169).
  - Why a separate class is needed: the A40 and A16 ranks of the same random-init model differ on 12,635 (GRU) / 14,063
    (SASRec) rows (T3 A40 receipt, DL 163).
  - Rows of the two classes never mix in one retention block. C_BIG and C_SMALL were re-evaluated on the A40, 8 checkpoint
    re-evaluations (DL 169).
- **Evaluator frozen.** `code/v4eval/*.py` hashes to evaluator_code_sha256 **61b64e61…**, which is part of the comparison key.
  Any edit to `v4eval` would split the A40 block. It is therefore frozen for the whole study (DL 180).
- **Determinism.** Training and evaluation run in STRICT deterministic FP32 (DL 29). Results do not depend on which other jobs
  share the GPU (co-scheduling independent; DL 82, 179); only wall time changes. This is why several jobs may share a card.

### 8.3 Lanes (how jobs run)
- A **lane** is a detached bash queue on the server (started with `nohup setsid`). It runs one or more jobs in sequence and
  logs to `runs_ds3/queues/<lane>.log`.
  - Some lanes wait for a file before starting the next job, e.g. "wait for `contracts/device/SELECTION_LO.json`" or "wait
    for a QUEUE_DONE line".
  - Lane helpers `_common_code_ds3r{3,4,5}.sh` bind a lane to one frozen code tree (DL 223, 244, 264).
  - Lane layouts: `company/plans/ops_scripts/setup_lanes_v1.sh` (DL 169b) and `split_fl_lanes.sh` (DL 177).
  - Read-only status scripts: `ops_scripts/status_*.sh`.
- **Throughput** (MEASURED rates, INFERENCE durations):
  - A federated process is CPU-bound: ≈ 1 core and 18–30 % of an A40 (DL 176), ≈ 10.2 client visits per second (DL 169).
  - A MAIN FA run is therefore ≈ 10.7 h per process when unshared (INFERENCE, DL 169; 06J §6).
  - A CALIB screen run takes ≈ 2.7 h unshared and ≈ 5–7 h on a shared A40 (REPORTED, practice audit header).
  - 5–8 FL processes fit on one card. Ten processes pushed the load to 17.7 on a 16-core guarantee (DL 253).
- **Co-scheduling rules.**
  - Amendment 06 §4 kept central runs alone on a card. 06G §5 and DL 179 relaxed this: it is a throughput rule, not a
    numeric one.
  - The central launcher needs `--allow-shared-gpu --coschedule-06g` to share a card (DL 207–208).
  - The re-evaluation tool needs `--allow-shared-gpu` (DL 266).

### 8.4 The frozen code trees
Each server tree is an immutable, read-only copy of the local `code/` at one moment. A run always resumes on the tree it
started on, because the code identity (`fl_code_sha256`, `device_code_sha256`) is recorded and checked on resume (06E §8).

| Tree | Deployed | What it added | Hashes | Used for | DL |
|---|---|---|---|---|---|
| code_ds3pre | 09-28 night | interim measurement tree (all suites green); **not for production** | — | short measurement steps | 161 |
| code_ds3r1 | 09-29 00:05Z | A40 device pin, provenance, T3 comparison path, reeval preflight (overnight session) | tar b707868b…, 583 files | superseded **before** the pin was written | 164, 166 |
| **code_ds3r2** | 09-29 00:13Z | corrected pin hash rule (ASCII JSON escaping) | tar 217d1c9d…, 583 files; server aggregate 48f6117b | A40 pin, T3 / STRICT receipts, 8 C_BIG / C_SMALL re-evaluations, LR screen ×0.5 / ×1 / ×2, **C_FULL s2026 / s2027** | 166, 181, 226 |
| code_ds3r2s | 09-29 09:35Z | = the full local tree with FL-09 (LR multipliers up to 32) | aggregate 27e99e92; launch_fl ca93ceea…, lr_screen 651d1c84… | **S1b screens** ×4 / ×8 / ×16; LADDER_S1a MAIN (descriptive) | 193, 194, 198, 226 |
| code_ds3r2p | 09-29 12:39Z | = r2s + DATA-10 (pretraining cohort) + `--coschedule-06g` + the server-side pretrain_small patch | aggregate a9b09086; pretrain_small a5550479…, launch_central a4ded7da… | **PRE** | 207, 212, 226 |
| code_ds3s2 | 09-29 14:44Z | snapshot after FL-10 (FedAdam / FedAvgM / FL_CONST / client SGD) + FIX-DELTA-01, before FL-12/13 | tar ccda9d64…; aggregate 172f76b6; launch_fl f2840329, fl/server 8c94ee53 | first 5 **S2** runs | 218, 220, 226 |
| **code_ds3r3** | 09-29 16:19Z | the first reviewed production tree: INT-06/07, DATA-09/10, FL-09/10/12/13, FIX-DELTA-01/02, ANA-02/03 (reviews REV-DELTA-01/02/02b) | tar 15d8c994…; 659 files; aggregate 5bef821a; lr_screen b1e243f4, launch_fl 1ef778d2 | S2 FedAvgM m 1 / m 2; S3 A2 cells; device lanes (SEL_FT_C, SEL_LO, LO, FT_C) | 223, 224, 226, 238 |
| **code_ds3r4** | 09-30 ~00:40Z | + FL-14 (+ review fixes), DEV-FTPRE, FIX-DEV-FTPRE | tar 357fa3a8…; 667 files; aggregate 77b488d7 | S2 edge runs + S2 / S3 selections; S3 W cells; **S4**; warm preview; PRE init record; FT_PRE; baselines BASE_R4 | 244, 245, 247, 252 |
| **code_ds3r5** | 09-30 14:20Z | + FIX-REEVAL + FL-15 (+ review fixes) | tar 8654dbc1…; 675 files; aggregate 66fbb9fb | C_FULL s2026 and PRE re-evaluations; **early S7 / S8 screens** | 264, 265, 268 |

Older trees from the contest phase: `code_p3` (GRU D128 contest runs), `code_p3b` (all other contest runs), `code_ds` /
`code_ds2` (SMALL ladder, C_SMALL, baselines) (DL 91, 102, 133, 137).

**Archives.** `deploy_archive/` holds:
- code_ds3r3.tar and code_ds3s2.tar with `SHA256SUMS.txt`;
- `r4/` and `r5/` (tar + `SYNC_MANIFEST.json` with per-file sha256 and the aggregate).

The r2 / r2s / r2p / ds3pre server trees are recorded by aggregate hash only (DL 226).

### 8.5 Deployment and verification procedure (as used for r3–r5)
1. Freeze the local code. Every suite exits 0, and every negative control (NC) fails when its guard is removed
   (`--runxfail -k test_nc_`).
2. Pack the local `code/` (excluding caches and temp dirs) into a tar and record its sha256.
3. Upload the tar through the remote channel, extract it into a **new** tree directory, verify every file's sha256 against
   the sync manifest, and make the tree read-only.
4. Run the targeted server tests.
   - Known noise: some pinned trajectory digests are Windows-CPU values and fail identically on every tree on the server
     (DL 245).
5. Run a **tiny-run digest comparison** with the previous tree for all 7 FL arms.
   - r3 == r4 (DL 245): DP 08383b6f, FA 807eadfb, FA_1024 878fc357, FA_Q8 651f0eb0, FP 825b27f8, PF 991bd6a4, S_CAL ff1b5150.
   - r4 == r5 (DL 264).
6. Write a lane helper for the tree, archive the tar under `deploy_archive/`, and add a DL row.

### 8.6 Watchers
- **Wait-then-act lanes on the server.** Examples:
  - `laneSEL2r4` → `r4b` → `r4c` → `r4d`: wait for the S2 runs, then run select-joint (DL 244–247).
  - `laneSEL4`: waits for S4, then writes FL_TUNING_S4.json (DL 256).
- **Local watchers.** For example, the shutdown watcher before the disk upgrade (DL 144). A session watcher was restarted
  after a session restart (DL 239).
- **Rule.** A watcher may launch only what is already registered. It never makes a discretionary decision.

---

## 9. The federated improvement ladder — full detail

### 9.1 Why FL was weak: the diagnosis (the five mistakes)
The first registered federated recipe reached only ≈ 20–25 % of a matched central model on CALIB. The comparison was
0.0288 at S1b against ≈ 0.11–0.12 for a central D64-class model on the same users; the ≈ 0.11–0.12 is itself INFERENCE from
the P2 / ladder era (ladder register; multi-expert review §1 item 2). It was even below LAST's share of C_BIG on MAIN (≈ 44 %; different populations, so this comparison is INFERENCE).

A research audit and a realism audit found that the gap came from the recipe, not from a bug:
- `company/reviews/REALISM_AUDIT_20260929.md` §0;
- `company/reviews/FL_STRONGEST_LADDER_RESEARCH_20260929.md` §1.

The operations lead summarised the ranked causes in DL 251. The deck calls it "our first federated setup was wrong"
(slide 15).

| # | Mistake | Mechanism (MEASURED from code + arithmetic, research §1.1) | Fixed by |
|---|---|---|---|
| 1 | **No server optimiser** (server LR 1, no momentum) | One server step can never move a coordinate further than the average client moved it locally. The still-rising LR curve is the signature of a server step that is too small. | S2 (FedAdam) |
| 2 | **Client AdamW re-created at every visit, with ≈ 4 steps** | A client holds 5–100 examples (mean 19.7) → 1–14 steps per visit. A fresh Adam's first step is ≈ lr · sign(g) per coordinate. With the full-catalogue softmax every one of the 158,486 output rows gets a gradient, so each visit moves all ≈ 10.1 M output-table coordinates by ≈ ±lr: dense sign noise. | S3 (client SGD with momentum) |
| 3 | **Local LR too small** for FedAvg with server LR 1 | The 06B winner sat at the top edge of the grid and the curve was still rising | S1b (×8) |
| 4 | **The central LR shape mapped onto rounds** | Peak only for EFE 0.02–2.0 (33 % of rounds); 0.1 × peak for EFE 3.0–5.5 (42 %). Only ≈ 3,200 peak-equivalent rounds out of 6,827. | S3 (FL_CONST) |
| 5 | **Random initialisation** of two 158k-row tables with ≈ 3 visits per client | Every later production system warm-starts from permitted data (Gboard, Firefox, Apple; practice audit §1) | S7 / S8 (PRE warm start, frozen tables) |

Two further causes are handled elsewhere:
- **Sparse-row dilution.** An item row is updated only by the k clients who viewed it and averaged with ≈ 57 zero deltas, so
  it moves ≈ k/57 of a local step. This is handled by S2's per-coordinate adaptivity and by S8.
- **The equal-exposure budget.** 6.0 EFE = 3 visits per client. It is measured by S5 = FA_X2.

The realism audit phrased the same diagnosis as four choices no production system uses together: no server optimiser, fresh
client AdamW, the central LR shape on rounds, and the equal-exposure budget (REALISM_AUDIT §0).

**Framing correction (06L_C C6; multi-expert review §3.2).** The best marks of the early runs sat well before 6.0, at 1.5–3.5.
So the as-registered row is called "matched-exposure retention of the registered recipe", not "a lower bound".

### 9.2 Protocol of every ladder screen (06F §1; ladder research §3)
- FA arm, CALIBRATION role (32,768 clients), seed 2026.
- 1,707 rounds = the full 6.0-EFE schedule; 64 clients per round, sweep sampling, 10 % drop-out.
- 12 marks. Value = CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST.
- Run directories: `runs_ds3/LRSCREEN*`, `LRSCREEN_EXT`, `LRSCREEN_S2`, `LRSCREEN_S3`, `LRSCREEN_S4`, `WARM_PREVIEW`, `S7S8`
  (server).

### 9.3 Stage by stage (all values CALIB, MEASURED from the selection files; "mark" = the PRACTICAL_BEST mark)

**S0 — FedAvg as registered, local LR = SMALL peak × 1** (06; 06B §1)

| Run | Value | Mark |
|---|---|---|
| ×1 | **0.023473** | 2.5 |

Source: `contracts/fl/LR_SELECTION.json`.

**S1a — local-LR multiplier m ∈ {0.5, 1, 2}** (06B §1; DL 181)

| m | Value | Mark |
|---|---|---|
| 0.5 | 0.020263 | 5.0 |
| 1 | 0.023473 | 2.5 |
| 2 | **0.025562** | 1.5 |

- Winner ×2, +0.002089 (+8.9 %), but at the **top edge** of the grid.
- File: `contracts/fl/LR_SELECTION.json` sha cdf6663d…, self 1e944cd8….
- The MAIN FL lanes started at ×2 at 09:12Z were stopped at ≈ 09:17Z, before any mark. They are labelled
  `SUPERSEDED_LR_SCREEN_EXTENSION` and are never read (06F §2; DL 183).
- **S1a-MAIN (descriptive):** LADDER_S1a = FA on MAIN, seed 2026, ×2 FedAvg, from the interim tree code_ds3r2s (DL 198;
  accepted as descriptive by 06J §5). It finished 2026-09-30 08:19:31Z and is **sealed** (DL 253).

**S1b — m ∈ {4, 8, 16}** (06F; DL 218)

| m | Value | Mark |
|---|---|---|
| 4 | 0.027570 | 2.5 |
| **8** | **0.028796** | 3.5 |
| 16 | 0.022909 | 3.5 |

- The winner ×8 is interior (the curve rises, then falls), so ×32 was not run.
- Δ vs S1a +0.003234 (+12.7 %); Δ vs S0 +0.005323 (+22.7 %).
- File: `contracts/fl/LR_SELECTION_EXT.json` sha 7d922e4a…, self e23431af….

**S2 — server optimiser** (06G, 06H, 06J §1, 06L A1, 06L_D D1)

Run sequence:

| When | What | Tree | DL |
|---|---|---|---|
| 09-29 14:44Z | 5 runs: FedAdam η_s {0.01, 0.1} × m {1, 8}, and FedAvgM (β 0.9, η 1) at m 8 | code_ds3s2 | 218 |
| 09-29 17:56Z | FedAvgM at m 1 and m 2 added by 06L A1 (the m = 8 cell was an effective step ≈ 10× past the FedAvg optimum; review [6-F1]) | code_ds3r3 | 226 |
| 09-29 23:39Z | all 7 done; select-joint **refused** (EdgeRunsMissing): FedAdam η edge + FedAvgM m 0.5 edge; the three edge runs launched | code_ds3r4 | 245 |
| 09-30 05:37Z | edges done; refused again: τ edge (a FedAdam run is the winner); τ 1e-5 launched | code_ds3r4 | 247 |
| 09-30 11:03Z | **S2 file written** | code_ds3r4 | 256 |

| Configuration | Value | Mark |
|---|---|---|
| incumbent: FedAvg ×8 (S1b) | 0.028796 | 3.5 |
| **FedAdam η_s 0.01, τ 1e-3, m 1** | **0.034247** | 3.0 |
| FedAdam η_s 0.003, m 1 (edge) | 0.027238 | 2.5 |
| FedAdam η_s 0.001, m 1 (edge) | 0.021876 | 6.0 |
| FedAdam η_s 0.01, τ 1e-5, m 1 (edge) | 0.024374 | 3.5 |
| FedAdam η_s 0.1, m 1 | 0.015632 | 6.0 |
| FedAdam η_s 0.01, m 8 | 0.016630 | 6.0 |
| FedAdam η_s 0.1, m 8 | 0.016130 | 6.0 |
| FedAvgM m 0.5 (edge) | 0.024236 | 5.0 |
| FedAvgM m 1 | 0.027541 | 2.5 |
| FedAvgM m 2 | 0.022626 | 6.0 |
| FedAvgM m 8 | 0.008489 | 4.5 |

- Winner **FedAdam η_s 0.01, τ 1e-3, m 1 = 0.034247**: gain +0.005451 (+18.9 %) over S1b; +45.9 % over S0.
- File: `contracts/fl/FL_TUNING_S2.json` sha 24080b1e…, written 2026-09-30 11:03Z.
- FedAdam details: Reddi et al. 2021 Alg. 2, no bias correction, m_0 = 0, **v_0 = τ²**. The registered clarification of DL 193
  supersedes 06G's "v_0 = 0".
- INFERENCE (DL 245): the edges fired because the best cell sat at the smallest registered server step. Note that m multiplies
  the client LR; with FedAdam the server step η_s is the effective step.
- **Reading note.** The operations lead read the S2 values at the owner's request at 07:56Z. That was before the S2 file
  existed (the τ edge run was still running). The DL records the justification: no discretionary decision depended on them,
  because the τ edge and the S3 cells were already mechanically fixed (DL 249). Disclose this.

**S3 — LR shape and client optimiser** (06I §3, 06J §2, 06L A2, 06L_D D2)

What changes:
- **FL_CONST shape:** linear warm-up over the first 1 % of rounds, constant peak, linear cool-down over the last 10 % to
  0.1 × peak (06I §3; FL-10).
- **Client SGD:** momentum 0.9, fresh per visit (clients stay stateless), no weight decay. The gradient clip 1.0 binds for
  SGD: the maximum step is lr × 1 (06J §2; `FL_TUNING_S3.json` client_sgd).

How the cells were run:
- The two 06L A2 cells (FedAdam η 0.01 × FL_CONST × SGD lr {0.1, 1.0}) were launched early, at 2026-09-29 21:58:49Z. They are
  screened whatever S2 selects, so no S2 value was needed (DL 238).
- Once S2 = FedAdam η 0.01 τ 1e-3 was mechanically fixed, the 06J cells coincided with the A2 cells (dedupe, 06L_D D2). The two
  missing cells (AdamW × FL_CONST, SGD lr 0.01) were launched at 05:35Z (DL 247).
- None was unused (DL 256).

| Configuration (all FedAdam η_s 0.01, τ 1e-3, m 1) | Value | Mark |
|---|---|---|
| incumbent: S2 winner (central LR shape, fresh AdamW) | 0.034247 | 3.0 |
| FL_CONST × fresh AdamW | 0.033104 | 2.5 |
| FL_CONST × SGD-m lr 0.01 | 0.058775 | 6.0 |
| **FL_CONST × SGD-m lr 0.1** | **0.094936** | **6.0** |
| FL_CONST × SGD-m lr 1.0 | 0.082381 | 6.0 |

- Winner **FedAdam η 0.01 + FL_CONST + client SGD-m lr 0.1 = 0.094936**: gain +0.060689 (+177 %) over S2; ×4.04 over S0.
- The SGD edge did not fire: lr 0.1 is interior.
- File: `contracts/fl/FL_TUNING_S3.json` sha 32decc6a…, written 11:05:36Z (DL 256).
- **S3b not triggered.** 06M_B B3 defines the decline index D = (PRACTICAL_BEST − mark-6.0 value) / PRACTICAL_BEST. The trigger
  was D ≥ 0.10 with a peak ≤ 4.5; for the S3 winner D = 0, because it peaks at 6.0. So the server-side SERVER_ANNEAL /
  SERVER_EMA cells were not run (DL 256).

**Per-mark curves** (MEASURED, CALIB; ladder register, DL 250, 256):

| Run | Curve |
|---|---|
| S0 ×1 | rises to 0.0235 at mark 2.5, then flat (0.0233–0.0235) to 6.0 |
| S1b ×8 | noisy; peak 0.0288 at 3.5; 0.0261 at 6.0 |
| S2 FedAdam | 0.0218 → **0.0342 at 3.0** → steady decline to 0.0251 at 6.0 (−27 %) |
| FedAvgM ×1 | peak 0.0275 at 2.5, ≈ 0.026 after |
| **S3 winner** | 0.0285 → 0.0758 (3.0) → 0.0876 (5.0) → **0.0949 (6.0), still rising** |
| FL_CONST × AdamW | still peaks at 2.5 and declines (0.0331 → 0.0295) |
| τ 1e-5 | flat ≈ 0.023 |

**How the interpretation of the S2 decline evolved** (all INFERENCE):
1. DL 250: first read as overfitting of the 32,768 CALIB clients.
2. DL 251: corrected. The decline starts exactly at mark 3.0, when the central LR shape reaches 0.1 ×; an LR-shape × FedAdam
   interaction was suspected.
3. Practice audit §4.2: argued a partial, uneven anneal plus a v-lag transient.
4. After S3 (DL 256; register): the decline came from the **fresh client AdamW**, not mainly from the LR shape or from
   overfitting. With client SGD the curve rises to the end.

This last attribution is itself an inference. Only the AdamW × FL_CONST cell separates the two factors (thesis reference
§14.21). **Consequence (INFERENCE):** the model is now *under*-trained at 6.0 EFE, so S5 (FA_X2, 12 EFE) is expected to help.

**S4 — local computation** (06J §3; FL-12/13)
- Three runs at the S3 winner, one change at a time: passes 1 (S4a), passes 4 (S4b), batch 8 (S4c). There is no LR re-screen
  (disclosed).
- The round count is recomputed so every run keeps 6.0 EFE: rounds = ceil(60 · N_clients / (576 · P)).
  - P 1 → 13,654 MAIN / 3,414 CALIB rounds.
  - P 4 → 3,414 / 854.
  - Batch changes no exposure (FL-12-13 handoff).
- **Status: RUNNING**, launched 11:06:38Z on code_ds3r4. Progress at DL 268: P1 11/12 marks, P4 8/12, B8 4/12; B8 needs
  ≈ 7 h more (INFERENCE).
- laneSEL4 writes `FL_TUNING_S4.json` when all three finish. The operations lead then copies it to
  `contracts/fl/FL_TUNING_SELECTION.json`, which is the file every MAIN FL launch requires (FIX-DELTA-02 M2).
- Tie order: incumbent first, then B8, P1, P4. This is a worker's reading of "ties → incumbent", recorded for the DL
  (FIX-DELTA-02 m9).
- **The DP arms do not take S4's passes / batch** (DL 219).

**S5 — FA_X2** (PLAN): FA at 12.0 EFE, MAIN seed 2026, descriptive budget sensitivity. SHOULD-high (06D §5; 06I §4).

**S6 — personalisation rungs on FA** (PLAN): FT_FA (selection on UNSEEN_VAL, then ×2 seeds) and PF. These are registered
arms, not screens.

**S7 / S8 — the deployment branch** (06K as amended by 06M_B B2).

Registered 2-D grids at the S4 winner:
- S7 (FA_WARM): local factor {1, 1/4} × server factor {1, 1/10}, because the selected optimiser is FedAdam.
- S8 (FA_WARM_FROZEN): local factor {1, 4} × server factor {1, 1/10}.
- 4 runs each, S7 and S8 in parallel.
- Rule: 06J restricted to the 4 runs. The incumbent is (m\*, η\*) = factors (1, 1). The in-band tie order was registered
  before any read: (m\*, η\*) → fewer changed axes → local factor closest to 1 → server factor closest to 1 (DL 261).
- `select-warm` accepts only the **final S4** tuning file (REV-FL-15 m2).
- Output: `contracts/fl/FL_WARM_SELECTION_S7|S8.json`, which binds the MAIN warm runs.

Status (DL 268):
- An **early** S7 screen (S7a–d) was launched at 14:45:47Z on code_ds3r5, **at the S3 point**. S8a–d are queued behind S4a
  and S4b.
- These specs are identical to the registered screen **if S4 keeps the incumbent**, which the ≥ 0.002 rule makes the default.
- If S4 selects a change, these 8 runs are recorded "unused" and re-run at the S4 point.

### 9.4 The warm-start preview (06M, 06M_B B1; DESCRIPTIVE, never in any selection)
**Why a preview.** The owner said: "FL starts from a random model; train the model first, as real systems do" (DL 252). The
preview measured the warm-start effect early, at the **S2 point**: FedAdam η 0.01, τ 1e-3, m 1, central LR shape, fresh client
AdamW, P 2 / B 16.
- θ_pre = PRE endpoint_6.0, sha cf08588c….
- Seven CALIB runs, WP1–WP7, launched 08:09Z and 08:20Z on code_ds3r4 (DL 252, 253). All seven were read together after all had
  finished (DL 267).

| Run | Best (mark) | Mark 6.0 | Curve |
|---|---|---|---|
| **FA_WARM_FROZEN f1** (item tables frozen, ≈ 1 MB per visit) | **0.1673** (0.5) | 0.166 | flat 0.164–0.167 across all marks |
| FA_WARM_FROZEN f4 | 0.1624 (5.0) | 0.162 | flat 0.157–0.162 |
| FA_WARM η_s 0.001, f 0.25 | 0.1582 (0.5) | 0.038 | collapses to ≈ 0.04 by mark 2.5 |
| warm FedAvg, f 0.25 | 0.1288 (0.5) | 0.030 | collapses |
| FA_WARM η_s 0.001, f 1 | 0.0706 (0.5) | 0.030 | collapses |
| FA_WARM η_s 0.01, f 0.25 | 0.0301 (0.5) | 0.026 | destroyed by mark 0.5 |
| FA_WARM η_s 0.01, f 1 | 0.0293 (3.5) | 0.025 | destroyed |

For comparison at the same point: cold FedAdam (S2) 0.034247; the cold S3 winner (client SGD) 0.094936.

**Interpretation** (INFERENCE; ladder register; DL 267):
1. With the old client (fresh AdamW), every full-model warm run **destroys θ_pre**. The gentler the step, the later it
   collapses. This is the same sign-noise mechanism S3 found.
2. Freezing the item tables protects the pretrained structure: FROZEN f1 stays ≈ 0.166 at every mark.
3. The frozen-table value is already reached at mark 0.5. So the federated gain over θ_pre alone may be small. That is exactly
   K21 = FA_WARM − PRE, which is measured only on MAIN at the preflight. PRE on CALIB is deliberately **not** read, because it
   would be a partial MAIN read.
4. The registered S7 / S8 screens run with client SGD, where full-model warm runs are not expected to collapse.

---

## 10. Privacy

### 10.1 DP-FedAvg as registered (amendment 04 §3; 06 §2; 06C; 06D §3)

| Item | Value | Source |
|---|---|---|
| Mechanism | DP-FedAvg: flat L2 clipping of each client delta to norm S; Gaussian noise z·S added to the **sum**; divided by a **fixed** denominator m (McMahan, Ramage, Talwar & Zhang 2018) | 04 §3; 06C |
| Sampling | **Poisson**: each of the N = 131,072 MAIN clients joins a round independently with q = 1,024 / 131,072 = 2⁻⁷ (seeded); empty or small rounds are still released | 06C |
| Rounds, δ | T = 384 rounds (= 3N/m); δ = 5e-6 (< 1/N) | 06 §2; 06C |
| Noise multipliers | z(ε = 8) = **0.5642**; z(ε = 1) = **1.3094**; z(ε = 4) = 0.7244 (optional arm) | 06 §2 Q5; 06C |
| Accountant | RDP, Poisson-subsampled Gaussian, add / remove adjacency, **user level**; z values independently re-derived with mpmath to ≈ 2.6e-11 | DL 170, 119 |
| Clip norm S | median client-delta norm of an **unclipped** calibration pass (S_CAL) with the FA_1024 configuration, seed 2026, rounds 1–20 → S_RECORD. S_CAL is re-run under the selected FL recipe before any FA_1024 / DP run. | 06 §2 Q3; 06G §3; 06J §3 |
| Drop-out | 0 for FA_1024 / DP (no privacy credit is taken for drop-out) | 06 §2 Q2 |
| Local work | P 2 / B 16, whatever S4 selects. Optimiser, LR and shape come from the tuning file. | DL 219; FIX-DELTA-02 M1 |
| Confirmatory pairing | K12 / K13 at **CONTROLLED_BUDGET** (the round-384 endpoint: a data-independent release). PRACTICAL_BEST ratios are descriptive, because choosing the best of 12 marks on the users' own validation rows is outside the accountant. | 06D §3 |

**Why Poisson (the 06C correction, DL 150).**
- The first registered sampler was fixed-size sampling without replacement. Its natural adjacency is replace-one, with
  sensitivity 2S.
- The registered accountant assumes Poisson sampling with add / remove adjacency.
- Under the matching bound (Wang, Balle & Kasiviswanathan 2019, Thm 9), the fixed-size sampler at z = 0.5642 would give
  ε ≈ 1,388 (lower bound ≈ 1,133), not 8. At z = 1.3094 it would give ≈ 6.7, not 1.
- Switching to Poisson makes the registered z values exact for the mechanism actually run.
- Evidence: `code/tests_fl/receipts/DP_WOR_SENSITIVITY_FL06.json`.

**Analytical note, registered before any DP value (06D §3; 06I §6(b)).**
- With d = 20,710,048 parameters, m = 1,024 and z = 0.5642, the per-round noise on the averaged update has expected norm
  z·S·√d / m ≈ 2.51·S. The signal norm is at most S.
- Production DP (McMahan et al. 2018) is at ≈ 0.23·S.
- So a large DP cost is to be read as a property of the simulated scale (N, d, m), not as a post-hoc excuse.
- DP8_FULL (N = 1,348,682, m 8,192, T 494, δ 5e-7) would be the scale-realistic point. It stays NICE, with no launch path
  (06I §5; 06L_C C16–C17).

**Server optimiser under DP (06H; 06L A3).**
- FedAdam divides each coordinate by √v + τ. In noise-dominated coordinates this turns the noise into a random walk of size
  ≈ η_s per round, so the measured privacy cost could become a property of the optimiser.
- Server momentum carries the noise forward with weight ≈ 1/(1−β).
- **DP_OPT_SENS** is therefore registered: FA_1024 s2026 + DP8 s2026 re-run under plain FedAvg with the same m\* and the same
  S_RECORD. It is reported beside K12, with a flag if the CONTROLLED_BUDGET ratio moves by more than 5 retention points.
- It fires for any non-FedAvg winner. **The S2 / S3 winner is FedAdam, so it applies.**

**What the guarantee does NOT cover** (registered disclosure list, 06D §3; 06L_C C2):
- the clip norm S;
- the local-LR multiplier and the whole FL tuning on CALIB (CALIB ⊂ MAIN);
- the public bootstrap block (catalogue, transform statistics, θ0 / θ_pre from opted-in users);
- the PRACTICAL_BEST checkpoint choice for the descriptive rows.

Also (06E §9(b)): the DP noise is drawn from a public deterministic seed with a non-cryptographic FP32 PRNG. The ε is the
guarantee of the mathematical mechanism in simulation, not of a deployable implementation.

### 10.2 Threat model and secure aggregation (06L_C C1, C2; 06I §7(a))
- **Adversary:** an honest-but-curious browser vendor (the server). It follows the protocol but may try to infer information
  from what it receives. The retailer's own logs are out of scope.
- **DP statement:** central DP with a trusted aggregator; client-level add / remove adjacency; (ε, δ = 5e-6); conditional on the
  public bootstrap block.
- **Secure aggregation is not simulated** in any FL arm. The server sees every client's (clipped) delta.
- **Without DP, an update reveals the viewed items.**
  - The non-zero rows of the untied input item table are exactly the items the user viewed in the visit.
  - So "only weights leave the device" is **data minimisation, not privacy**. DP is the only formal guarantee (L2 vs L3).
  - S8 (frozen tables) removes the input-table leak only partly: brand / category deltas may still leak, and off-support rows
    move ≈ 1e-9, not exactly 0 (06L_C C8).

### 10.3 Logging rules that protect the guarantee
- **Clipping log (06L_E; FL-14).**
  - z = 0 runs (FA_1024) write `CLIP_LOG.jsonl` per round: round, n_clipped, n_deltas, clipped_fraction; never per-client norms.
  - **z > 0 runs record nothing about clipping.** A raw clipped count is an un-noised, data-dependent output outside the
    accountant.
  - The thesis reports the z = 0 twin's fraction as a labelled proxy.
  - S_CAL does no clipping, so it writes no log; 06L_E's mention of S_CAL is void (DL 243).
- **Training-loss log (FL-15; DL 261).** Every non-DP FL run writes `TRAIN_LOSS.jsonl` per round, for monitoring only. It is
  **not** written for z > 0 runs, by the same principle.

### 10.4 Privacy levels (06B §3 ladder, replaced by the 06L_C C1 matrix)
Every arm is described by four columns:
- the data seen at training by the server;
- the data seen at inference;
- the formal guarantee;
- against whom.

The levels: L0 cloud; L1 cloud-trained / device inference (incl. FT_C's base and PRE-alone); L2 federated; L3 federated + DP;
L4 FL + local personalisation; L5 opted-in base + local fine-tune (FT_PRE); L6 device-only (LO, LAST). RULE / POP = "device +
published aggregate". The table in §6.2 gives each arm's level.

---

## 11. The on-device branch

### 11.1 Registered selections (UNSEEN_VAL role, seed 2026, base C_SMALL s2026)

| Arm | Grid | Selected cell | File (sha) | Registered | DL |
|---|---|---|---|---|---|
| FT_C | LR fraction {0.1, 0.3} × SMALL peak × passes {1, 2, 4} | **FT_C_f0.3_P4** (0.3 × peak, 4 passes) | `contracts/device/SELECTION_FT_C.json` (e47965d2…) | 2026-09-29 20:42Z | 231 |
| LO | passes {1, 2, 4} | **LO_P4** (4 local passes) | `contracts/device/SELECTION_LO.json` (744e6a41…) | 2026-09-29 20:52Z | 232 |

- Both use the eval manifest UNEW_VAL_VALIDATION (content 6802cc6c…).
- The selection-panel values are inside the files and are deliberately not quoted.
- LO's unselected cells are reported as `LO_CELL_NOT_SELECTED` (06E §1).
- FIX-DELTA-01 F7 makes a non-smoke LO run off UNSEEN_VAL refuse without `--selection`.

### 11.2 Arms and status (DL 224, 232, 255, 268)
- **LO** (local only from θ0, the no-collaboration control):
  - MAIN s2026 and s2027 started 20:51Z (09-29) on code_ds3r3. Both **DONE** (sealed).
- **FT_C:**
  - Lanes 7 / 8 were first gated on SELECTION_LO; they are now running FT_C s2026 / s2027 on MAIN (RUNNING).
- **FT_PRE** (θ_pre + the registered FT_C procedure with SELECTION_FT_C, pinned by sha e47965d2…):
  - The review made a smoke run mandatory before production (REV-DEV-FTPRE M2). The smoke passed at 08:55Z: LAUNCH
    start.kind = PRE, checkpoint cf08588c = θ_pre, selection sha e47965d2 (DL 255).
  - Full runs on MAIN: s2026 on GPU0 since 12:51Z and s2027 on GPU1 since 12:47Z (RUNNING, DL 268).
- **FT_FA** (FA + local fine-tune): selection on UNSEEN_VAL, then × 2 seeds. PLAN, after FA MAIN exists.
- **FT_FA_WARM / FT_FA_WARM_FROZEN** (the full production stack: θ_pre → FA_WARM / S8 MAIN → FT_C procedure → on-device
  inference):
  - descriptive; seeds 2026 / 2027; paired with FA_WARM per seed (base seed must equal `--seed`) (06M_B B4; FL-15; DL 261).
  - PLAN, after FA_WARM MAIN.
- **No-FT references:** C_SMALL and cold FA re-evaluated on UNSEEN_VAL, descriptive (06L_C C16; DEV-FTPRE §3). PLAN.
- **At the TEST look** the device arms re-run their locked procedure: same SELECTION file, same support rows (06D §2).
- **Disclosure** (06A §9(f)): per-client batching reorders 6 of 1,251 rows by one rank position, all at ranks > 20. There is no
  effect on @≤ 20 metrics.

### 11.3 Device measurement and cold start (registered; server status to verify)
**ONNX / INT8** (amendment 04 §3; 06E §3; 06L_C C9):
- C_SMALL seed 2026, opset 17, onnxruntime CPU single-thread batch 1; latency median / p95 over 1,000 queries; peak RSS.
- K16 = dynamic INT8 (QInt8 weights / QUInt8 activations, per tensor, reduce_range False, onnxruntime 1.24.1), computed inside
  an ONNX-CPU sub-block.
- The rows subset is the first 16,384 E2E MAIN_VALIDATION users in selection-key-v2 order.
- `dump_rows` crashed on real text fields; GATE-B B2 fixed it.
- A browser run (ONNX Runtime Web 1.24.1 on the owner's PC; DEV-03) is planned.
- **The DL does not record the server ONNX latency run.** The deck says "latency on a single CPU thread is running" (slide 20).
  Verify on the server.

**Cold start:**
- Primary analysis (06A §2): observational bins of TRAIN support {0, 5–10, 11–20, 21–50, 51–100} on MAIN and UNSEEN_HOLDOUT,
  with events-to-90 % / 95 %.
- Controlled study (DATA-07; 06L_C C10 minimal set = SHOULD-high):
  - the truncated-history builder passed its T41 golden replay on the full frozen bucket 0 (TRAIN / VALIDATION only):
    173,921 / 173,921 VALIDATION queries (7 admitted brand-OOV cells) and 622,168 / 622,168 TRAIN decisions (DL 178);
  - Study A = WARM_TRUNCATION, context-only arms; Study B = COLD_H, fine-tune arms;
  - DEV-FTPRE added `TRUNC_PER_USER.npz` for CIs;
  - 06L_C C10 added matched-h parity and a persistence rule.
- The server T41 receipts and the Study A / B runs are not clearly recorded in the DL rows I read. DL 244 mentions an "old
  Study B" in flight. Verify on the server.

---

## 12. Code changes of the last days (2026-09-28 night → 2026-09-30)

**How every change was made:**
1. A packet with a closed scope was written.
2. A worker implemented it locally with tests and negative controls, and wrote a handoff with before / after sha256s.
3. The operations lead re-ran the suites.
4. An independent read-only review followed (except where noted).
5. Review fixes were made and checked.
6. The change was deployed only inside a new frozen tree.

The shas below are sha256 prefixes from the handoffs. `v4eval` (61b64e61…) was never changed.

### 12.1 Overview table

| Task | Purpose | Main files (before → after) | Tests (worker / operations re-run) | Review | DL |
|---|---|---|---|---|---|
| INT-06 | Hostile review and completion of the overnight session's partial A40 work. PF blockers B1 (resumed PF evaluated an empty store) and B2 (best-mark store never saved). Device rows labelled with the pin. Resume continuity guards. LO selection on UNSEEN_VAL. `reeval.py` rewritten (central / FL / PF, panel roles). | `v4eval/device_pin.py` **reverted** to the deployed bytes (a post-pack edit would have split the evaluator hash); `launch_fl`, `launch_device`, `reeval` | tests_int 168 / 43 NC (verified) | covered by REV-DELTA-01 | 180–182 |
| DATA-09 | Gated RETRO_TEST manifest builder (5 TEST manifests), REGISTRY_V2_TEST_EXT, lock-chained gate; found the blocker F2b (no TEST path in the feature store) | `code/v4data09`, `v4int/test_roles.py` | tests_data09 63 / 24 NC | REV-DELTA-01 | 174 |
| DEV-04 | INT8 ranks skip out-of-vocabulary targets; FP32 ONNX ranks for K16; `--first-users` in selection-key-v2 order; browser page SRI | `v4dev`, `v4data08/d8_registry.py`, `c_full.py` | tests_dev 73 / 13 NC (verified) | REV-DELTA-01 | 173 |
| DATA-07R1 | Truncation study fixes (T41 receipt binding, panel re-derivation, row labels) + the full-bucket T41 replay | `v4trunc` (new code sha 4d766d5b…), `launch_trunc` | tests_trunc 42 / 11 NC (verified) | REV-DELTA-01 | 178 |
| ANA-02 | Analysis script: every REV-ANA-01 item (ONNX-CPU block for K16, registered seeds, full comparison key, rank coverage, TEST gate, rule map) + `tools/ana_block_index.py` | retention_report → 1e93a819…, ana_block_index f6bb6f48… | tests_ana 75 / 18 NC (verified) | REV-DELTA-01 | 186–191 |
| **FL-09** | Local-LR multipliers up to 32; `lr_screen select` writes LR_SELECTION_EXT (06F) | launch_fl 2ab8c9bf → ca93ceea; lr_screen → 651d1c84 | 28 passed / 9 NC; 96 (arm-spec, RunConfig) digest pairs byte-identical | REV-DELTA-01 | 193–194 |
| INT-07 | The gated TEST evaluation path: TEST store + history windows over TRAIN + VS + VC + TEST with frozen normalisers; TEST roles in reeval / launch_device / baselines; `dump_rows_retro` | `v4int/retro_store.py` (new), `cache.py`, `providers.py`, `reeval.py`, `launch_device.py`, `v4base/run.py` | tests_int 204 / 55 NC | REV-DELTA-02 | 209 |
| DATA-10 | Pretraining cohort manifest (TIEBREAK minus MAIN, 1,217,610 users), REGISTRY_V2_PRETRAIN_EXT, `pretrain_small.py` shim | `code/v4data10`, `v4int/pretrain_small.py` (9cb58162…) | tests_data10 33 / 18 NC | REV-DELTA-02 | 207 |
| **FL-10** | FedAdam (v0 = τ²), FedAvgM (β 0 = FedAvg bitwise), FL_CONST shape, client SGD-m, `select-joint --stage S2 / S3` | fl/server 63ccf395 → 8c94ee53; fl/runtime d51e2dc0 → c4e1db6d; fl/client 8aabe73c → 8a41e6d6; launch_fl → bb011490; lr_screen → e9418f64 | tests_int 215 / 57; tests_fl 239 / 73 + 1 stale-helper failure fixed by the operations lead; 224 default digest pairs byte-identical; server moments resume bitwise | REV-DELTA-02 | 211–212 |
| FIX-DELTA-01 | REV-DELTA-01 fixes: F1 top-level result / rank hashes in REEVAL records; **F2 MAIN FL launch bound to the registered selection file** (path + sha + provenance); F5 test-isolation dirs; F6 selection manifest check; F7 LO needs a selection. Also reviewed the operations lead's `--coschedule-06g` and pretrain_small patches. | reeval → 4744a48d; launch_fl → f2840329; launch_device → 48eee710; launch_central → 2d56c222; pretrain_small → 26d8004a | tests_int 225 / 64, tests_fl 240 / 73, tests_train 214 / 61, tests_data10 34 / 19 | REV-DELTA-02 | 216, 218 |
| ANA-03 | K17–K21, contrast file v2 / v3 loader, charged bytes, F9 (C_BIG `…:best` id bound to the lock sha), F1 index side | retention_report → 10bc046a; ana_block_index → bfc3fc72 | tests_ana 100 / 24 NC | REV-DELTA-02 | 218 |
| **FL-12/13** | S4 `--local-passes / --local-batch` with a recomputed round count; S7 `--init-from` θ_pre (refuses any MAIN-trained start); S8 `--freeze-item-tables` (tables are fixed buffers, not uploaded; bytes and the DP dimension = uploaded params); `select-warm` | launch_fl → e5015be4; lr_screen → 08d92b4a; fl/runtime → 443fdfc6; fl/comm → 2f8bd172 | tests_fl 248 / 75; tests_int 238 / 67 (+1 stale test fixed) | REV-DELTA-02 | 219 |
| FIX-DELTA-02 | **B1: the 06J ≥ 0.002 incumbent rule** in lr_screen and in the launcher re-check; M1 DP arms fixed at P 2 / B 16; M2 MAIN requires the final S4 file; M3 edge runs required (`EdgeRunsMissing`); M4 charged bytes = download only (up + down descriptive) | lr_screen → b1e243f4; launch_fl → 1ef778d2; retention_report → da394e6d | tests_int 253 / 76, tests_fl 248 / 75, tests_ana 101 / 25 | **REV-DELTA-02b PASS** | 221–222 |
| GATE-B | Execution blockers from the multi-expert review: B1 reeval accepts the registered C_FULL / PRE sources; B2 `dump_rows` keeps numeric arrays only; B3 S8 admitted in the analysis (81.78 MB tables charged once, at the first visit); B4 NO_C_REF status + the Study B events-to-X path | reeval 4744a48d → 7a86b07d; dump_rows 1bde41bd → 4696c4af; retention_report da394e6d → 94b778e3; ana_block_index → bc6129be; launch_trunc 1f6304df → de009063 | tests_int 262, dev 77, ana 109, trunc 48 (all exit 0) | REV-DELTA-03 | 228 |
| ANA-04 | 06L_C analysis items: LAST anchoring + baseline rank files (C3), **ANA_CONTRASTS_v3** (26 confirmatory), bytes axis + privacy matrix (C8 / C1), GRU_D128 comparison (C11), claim register + number-free headline template (C12), seed stats / Holm band / harmed share / K14 TOST / CALIB-disjoint (C13–C15) | retention_report → 41f2ab49; ana_block_index → 6584a029; v4base/run.py → 400d20a4; ANA_CONTRASTS_v3 a3d1456d… | tests_ana 142 / 43, base 64 / 10, int 262 / 78 | REV-DELTA-03 | 229–230 |
| FIX-DELTA-03 | REV-DELTA-03 fixes: M1 PRE / C_FULL identity checks; M2 warm-arm init record; M3 Holm band tail; M5 TEST refuses non-v3 contrasts; C_FULL reeval restricted to best / endpoint; C10 persistence rule | retention_report → d48692b3; ana_block_index → cede8b98; reeval → ab0f65c1 | tests_ana 156 / 52, int 264 / 81, trunc 48 / 13 | (fix of a review) | 233 |
| **FL-14** | Makes 06L A1 / A2 / A4 executable (06L_D): S2 set + FedAvgM m 0.5 edge + edge order; S3 A2 cells + dedupe + tie order; clip logging for z = 0 | lr_screen b1e243f4 → **e81667d3**; launch_fl 1ef778d2 → a23d5b4f → (review fixes) **5652e4c7**; new test_fl14_06l | tests_int 113 / 41 NC (subset), tests_fl 21 / 6; pinned digests unchanged | **REV-FL-14 PASS-WITH-FIXES** (M1: z > 0 must never log a count; m1: incomplete clip log flagged) | 240, 242, 243 |
| **DEV-FTPRE** | FT_PRE arm (θ_pre + FT_C procedure; MAIN-trained bases refused); deterministic `TRUNC_PER_USER.npz`; no-FT UNSEEN_VAL references | launch_device 48eee710 → cc7aba48; launch_trunc de009063 → c7a7e7b0; reeval ab0f65c1 → 6b76e9d2; retention_report d48692b3 → **c699b3bf**; ana_block_index → 9c4954df | tests_int 275 / 86, ana 160 / 53, trunc 52 / 15 | **REV-DEV-FTPRE PASS-WITH-FIXES** (0 blocker / 0 major / 5 minor) | 239, 241 |
| **FIX-DEV-FTPRE** | M1 helpers so the NCs act on real wiring; **M3 PRETRAIN_FOR_FL_INIT.json write-once**; M4 index checks the TRUNC_PER_USER sha; M5 FT_PRE selection pinned by sha e47965d2… | launch_device → 0ae48c72; reeval → 777959f9; pretrain_small 26d8004a → 1f5630c6; ana_block_index → 780b3ef9 | tests_int 298 / 91, ana 161 / 54, trunc 52 / 15, data10 17 / 13 | (fix of a review) → deployed as **code_ds3r4** | 244 |
| **FIX-REEVAL** | The re-evaluation tool refused the REAL C_FULL / PRE runs (GATE-B's check compared the wrong registry hash). New `check_source_binding`; TEST eval-only binding made fail-closed; real run records used as test fixtures | reeval 777959f9 → **adfba078**; test_gate_b_reeval → 48e4218a; new test_fix_reeval_real_records | tests_int 327 / 110, data09 73 / 28, ana 161 / 54 | **REV-FIX-REEVAL PASS** (0 blocker / 0 major / 3 minor) | 260–262 |
| **FL-15** | 06M_B B2: 2-D S7 / S8 screens (`--warm-server-lr-factor`, `select-warm` with the 4 registered runs); B4: FT_FA_WARM start kind; R5: per-round `TRAIN_LOSS.jsonl` (not for z > 0) | lr_screen e81667d3 → 860ec751 → (fixes) **1a391b1e**; launch_fl 5652e4c7 → 5facf3a7 → **3c0d91b5**; launch_device 0ae48c72 → 9106f67c → **1968ca95**; fl/runtime 443fdfc6 → e627e943 | tests_int 330 / 113, tests_fl 248 / 75; 75 + 11 NCs fail as intended | **REV-FL-15 PASS-WITH-FIXES** (m2 select-warm refuses a non-S4 tuning file; m3 FT_FA / FT_C refuse warm bases; m4 warm dry run binds like the real launch) → deployed as **code_ds3r5** | 261, 263, 264 |

Operations-lead edits that were reviewed afterwards:
- `launch_central --coschedule-06g`: lets a central MAIN run share a card; only valid together with `--allow-shared-gpu`
  (DL 207, 208).
- The server-side `pretrain_small.py` patch, so the pretraining shim accepts the co-scheduling flag (DL 212).
- Both were first flagged as unreviewed (DL 213), then reviewed in FIX-DELTA-01 (DL 216, 218). Their server shas were matched
  to the reviewed versions (a5550479… / a4ded7da…) in Gate B item B6 (DL 226).
- Small test-only fixes: the `_old_digest` helper (DL 212) and the "S4 not implemented" pin (DL 219).

Earlier on 2026-09-28 (before the A40 move; for completeness):
- FL-05 (lr-mult, CALIB role, FA_Q8);
- FL-06 (WOR accountant finding → 06C);
- FL-07 (Poisson sampling);
- INT-05 (selection role UNSEEN_VAL);
- DEV-02 / DEV-03 (rows dump, browser page);
- DATA-07 (truncation builder);
- DATA-08 (C_FULL manifest);
- ANA-01 (the retention / analysis script).

See DL 148–160. **FL-08** (PF store fixes) was never run as its own task; its content was folded into INT-06 (DL 170, 180).

### 12.2 Reviews and their verdicts (read-only; `company/reviews/REV-*.md`)

| Review | Scope | Verdict / key findings | DL |
|---|---|---|---|
| REV-FL-01 | FL correctness incl. the overnight changes | ACCEPT_WITH_FIXES. Poisson q, DP mechanism, accountant, LR schedule, FA_Q8, LR screen all correct. **PF blockers B1 / B2.** Device rows mislabelled (M1). | 169b, 170 |
| REV-ANA-01 | analysis script, DATA-07, DEV | ACCEPT_WITH_FIXES: 9 major in retention_report, 3 in DATA-07, 2 in DEV | 170 |
| REV-DELTA-01 | INT-06, DATA-09, ANA-02, DATA-07R1, DEV-04, FL-09, INT-07 | ACCEPT_WITH_FIXES, **no blocker, no server artefact invalidated**. F1 reeval hashes; F2 MAIN selection binding | 210 |
| REV-DELTA-02 | FIX-DELTA-01, ANA-03, FL-10, FL-12/13, INT-07, DATA-10, the operations lead's edits | ACCEPT_WITH_FIXES. **Blocker B1: the ≥ 0.002 rule was missing** (must be fixed before the S2 selection). M1–M4. The S2 values from code_ds3s2 stay comparable. | 220 |
| REV-DELTA-02b | FIX-DELTA-02 | **PASS** (probes: +0.0012 / +0.0013 keep the incumbent, +0.0025 wins) | 222 |
| REV-DELTA-03 | GATE-B, ANA-04 | ACCEPT_WITH_FIXES, no blocker; statistics judged correct; M1–M5 | 230 |
| REV-DEV-FTPRE | DEV-FTPRE | PASS-WITH-FIXES, 5 minor; **M2: FT_PRE smoke mandatory** | 241 |
| REV-FL-14 | FL-14 | PASS-WITH-FIXES; cross-tree check clean (select-joint never compares code hashes); tie disclosures m2–m4 | 242 |
| REV-FIX-REEVAL | FIX-REEVAL | PASS | 262 |
| REV-FL-15 | FL-15 | PASS-WITH-FIXES; m1 winner's-own-gain disclosure; m2–m4 code fixes | 263 |

Earlier reviews, not code:
- the full review and plan (DL 146);
- the thesis-level research review (DL 147);
- the deep expert review that found the A40 and TEST-path blockers (DL 159);
- the realism audit (DL 192);
- the ladder research (DL 203);
- the 14-part multi-expert review (DL 226; score 7/10, "8–8.5 if the MUST list is done", OPINION);
- the real-world FL practice audit (DL 253).

---

## 13. Incidents, errors and errata (honestly)

Rule of the project: an error is recorded in the DL, corrected in a new file or row, and **disclosed**. It is never silently
fixed. None of the incidents below changed a registered rule after its value was read, and none exposed a sealed value.

### 13.1 The last days (2026-09-28 night → 2026-09-30)

| # | What happened | Effect / resolution | DL |
|---|---|---|---|
| 1 | The server came back from the disk upgrade on 2 × A16, not the promised A40 | Nothing was launched. The owner had the A40 profile enabled and the server restarted. | 149, 152, 153, 158 |
| 2 | **Two blockers** found before any run: the trainer refused every non-A16 device (so the A40 plan was not executable), and there was no executable TEST path at all | Registered 06D (A40 pin, TEST manifests); tasks INT-06, DATA-09, INT-07 | 159 |
| 3 | The first T3 A40 receipt attempt refused because of a stale bundle manifest | Refused before any computation; the r2 bundle passed | 163 |
| 4 | The pin code hashed JSON with a different escaping rule than the launcher | Found **before** the pin was written. The queue was stopped (SUPERSEDED_BEFORE_PIN); code_ds3r2 deployed. | 166 |
| 5 | The overnight session's plan ran everything **serially** (LR screen ≈ 8 h instead of ≈ 2.7 h in parallel), and its FL queue departed from the registered MUST order (it omitted FP, LO, FT_C) | Queue stopped while still waiting; no run had started. Code reviewed and completed in INT-06. | 168, 180 |
| 6 | Two sessions both wrote a DL row "169"; several rows and document headers carried time stamps typed ahead of the clock | The second row is cited as 169b. Stamps were corrected before any use. Rule: take stamps from `date -u`. | 171, 179 |
| 7 | Duplicate workers across the two sessions (ANA-02 dispatched twice; overlapping fix packets) | Rule "one owner per file set"; the duplicates were stopped | 172 |
| 8 | Blocker F2b: the feature store and the history policy had no TEST path | INT-07 rewritten into a full TEST-path task | 174 |
| 9 | **HIGH:** a post-pack edit in `v4eval/device_pin.py` changed the evaluator hash (f1d4c717 vs the deployed 61b64e61). That would have split the A40 block. | Reverted byte-identically. `v4eval` frozen for the whole study; a guard test was added. | 180 |
| 10 | Three FL lanes started on GPU 1 next to C_FULL s2026 before they could be moved (amendment 06 §4 kept central runs alone on a card). C_FULL s2027 was refused by the launcher's GPU ownership check. | Kept: numerics are STRICT and independent of co-scheduling. The one-central-job-per-card rule was later registered as a throughput rule only. Disclosed. | 181, 179 |
| 11 | MAIN FL lanes started at ×2 (09:12Z) just before the LR screen was extended | Stopped ≈ 09:17Z, before any mark; labelled SUPERSEDED_LR_SCREEN_EXTENSION; never read | 183, 06F §2 |
| 12 | **The 06F overwrite.** The chief-scientist session's write of 06F overwrote the operations lead's 06F, registered one minute earlier on the owner's decision. Cross-session messages had expired. | Resolved: the chief-scientist 06F (LR grid) was kept and restored byte-identically; FedAdam was registered separately as 06G. Lesson: check the DL tail and the file's existence before writing under `contracts/`; amendments are never overwritten; coordination only through DL rows. | 183–188 |
| 13 | A service usage limit interrupted a review and four workers mid-task | No partial output used; resumed from their instructions after the reset. Rule: no production MAIN launch before the interrupted review returned. | 197, 198, 200 |
| 14 | A regression run failed because two concurrent test runs shared a fixture directory, and one failure was misread as "the default digest changed" | The review showed it was a stale test helper; the real digest was unchanged (DL 210 corrects DL 205). F5 gave each run its own directory. | 201, 205, 210 |
| 15 | `v4dev.dump_rows` would crash on real provider batches (text fields) | Found on the synthetic TEST provider; fixed in GATE-B B2 | 209, 228 |
| 16 | The PRE dry run was refused by the shim's own shared-GPU check. The operations lead patched `pretrain_small.py` **on the server**, an unreviewed change. | Mirrored locally, reviewed (FIX-DELTA-01), server sha matched to the reviewed version (B6) | 212, 213, 216, 226 |
| 17 | The ≥ 0.002 incumbent rule of 06J was missing from the selector | Found by REV-DELTA-02 (blocker B1) **before** the S2 selection. Fixed and re-checked (PASS). | 220–222 |
| 18 | The multi-expert review found four execution blockers: reeval refused C_FULL / PRE; dump_rows crash; S8 not admitted by the analysis; the production tree existed only in an OS temp folder | GATE-B + `deploy_archive/`. The 21.8 MB `STRATA_CUTPOINTS_MAIN.json` could not be mirrored (large-file fetch fails); the server copy is authoritative. | 226, 228 |
| 19 | **Public-doc erratum.** A correction pushed to the public branch said "the large model has two blocks, not three". That was wrong: C_BIG / C_FULL is SASREC_D256_B3 (3 blocks). The 2-block entry that was checked belonged to an older model family. | Fixed and pushed. Lesson: verify architecture facts from the V4 recipe / variant registry, not from older entries. | 236, 257 |
| 20 | The frozen selector in code_ds3r3 implemented 06J only, so it would refuse the 06L A1 runs | Found before any S2 value was opened (5 of 7 runs done); the waiting select lane was withdrawn before it ran. Registered 06L_D; FL-14. | 237 |
| 21 | A4 privacy issue: logging a raw clipped count under DP would release an un-noised output | Registered 06L_E before any DP / FA_1024 MAIN run: z > 0 records nothing | 240 |
| 22 | Server tests on code_ds3r4: 5 failures + 3 errors | All explained. Some trajectory digests are Windows-CPU pins and fail identically on r3. The analysis tools were not yet deployed. A direct tiny-run comparison gave r3 == r4 for all 7 arms. | 245 |
| 23 | S2 values were read (owner's request) before the S2 selection file existed (τ edge pending) | Justification recorded: no discretionary decision depended on them; the remaining steps were mechanical. Disclose. | 249 |
| 24 | The explanation of the S2 post-peak decline changed twice (overfitting → LR shape × FedAdam → fresh client AdamW) | Each version is labelled INFERENCE; the last attribution is not an isolated measurement | 250, 251, 256 |
| 25 | **Init-record rewrite.** PRE's continuation, running on the old tree, rewrote `PRETRAIN_FOR_FL_INIT.json` at completion with a new timestamp. The record sha changed c8d644a6 → **5ed989f6**. | θ_pre bytes (cf08588c…) unchanged. The warm-preview runs were launched with the old record (same θ_pre). Every later warm / FT_PRE run binds 5ed989f6. The write-once fix existed only in r4. | 255 |
| 26 | **Reeval binding bug found on real records.** GATE-B's check compared the key base's registry hash with the extension's hash. The trainer always rewrites that field with its per-run trainer view, so every real C_FULL and PRE run was refused. GATE-B's tests had used synthetic records. | FIX-REEVAL: new binding check; real records as fixtures; a second fail-open (TEST eval-only binding) closed | 260, 261 |
| 27 | Reeval refused C_FULL **s2027**: the source is registered for seed 2026 only, although s2027 is a registered NICE seed | Open gap: allow s2027, labelled DESCRIPTIVE, in the next tree | 265 |
| 28 | The first real reeval was refused by the GPU ownership check (GPU 0 shared with S4a) | Relaunched with the registered `--allow-shared-gpu` flag (results are co-scheduling independent) | 266 |

### 13.2 Earlier incidents (2026-09-26 → 28), for context

| Incident | DL |
|---|---|
| Hand-typed time stamps ran ahead of the clock; rule: take stamps from `date -u` | 8, 92 |
| A local schema check read TEST status-column **counts**. This created the hash-chained TEST_EXPOSURE_REGISTER. A superseded census run containing v1-model levels was quarantined, not deleted. | 33 |
| A service usage limit stopped two workers; they were resumed | 38 |
| Failed launches were kept as evidence (fail-closed): a manifest registry not passed; a missing CUDA_VISIBLE_DEVICES | 49, 52 |
| An independent review **rejected** INT-02 (every production launch would have failed) → fixed in INT-02R1 | 65 |
| A prose erratum transcribed LR midpoints wrongly; replaced by computed values (the registry itself was always right) | 74 |
| G1_APPROVED was first written with a null evaluator fingerprint; removed before any use | 80 |
| One SASRec P2 intermediate value was printed incidentally; disclosed; the selection was mechanical | 105 |
| Lesson: the GPU index is part of the RunConfig, so dry runs must use the target GPU | 110 |
| A worker installed packages into the local venv and later removed them; rule: never install | 118, 121 |
| The server was shut down for the disk upgrade earlier than announced; one CPU baseline was cut and re-run | 145 |
| A price-feature flag was checked and resolved as a float-rounding artefact, not leakage | 156, 157 |

---

## 14. Current state (as of DL 268, 2026-09-30 14:50Z server clock) and next steps

### 14.1 Finished

| Item | Status | DL |
|---|---|---|
| G2 contest + ARCHITECTURE_LOCK (sha fd201013…) | final | 129–131 |
| C_SMALL both seeds | DONE_MAIN, sealed | 145 |
| A40 receipts, pin, 8 C_BIG / C_SMALL A40 re-evaluations | done, sealed | 167, 169 |
| C_FULL s2026 and s2027 | DONE rc 0, sealed | 248, 259 |
| C_FULL s2026 (8 jobs) + PRE (5 jobs) re-evaluations on code_ds3r5 | 13 / 13 rc 0, sealed | 265, 268 |
| PRE training + init record (θ_pre cf08588c…; final record 5ed989f6…) | done | 252, 254, 255 |
| Ladder S0 → S3 selections (LR_SELECTION, LR_SELECTION_EXT, FL_TUNING_S2, FL_TUNING_S3) | written | 181, 218, 256 |
| LADDER_S1a MAIN (descriptive) | done, sealed | 253 |
| Warm-start preview WP1–WP7 (CALIB, descriptive) | done, read | 267 |
| Device selections SELECTION_FT_C, SELECTION_LO | registered | 231, 232 |
| LO MAIN s2026 / s2027 | done, sealed | 268 |
| Baselines re-run with rank files (POP, LAST, SKNN, RULE, KNN_EMB BIG / SMALL) on MAIN | done rc 0, unread | 246, 249 |
| Raw-event bytes measured (133.54 B / event) | done | 234 |
| Code trees r3, r4, r5 deployed and archived | done | 223, 244, 264 |

### 14.2 Running at DL 268
- **S4:** P1 11 / 12 marks, P4 8 / 12, B8 4 / 12 (code_ds3r4). laneSEL4 will write FL_TUNING_S4.json.
- **Early S7 screen S7a–d** (FA_WARM, 2-D grid, at the S3 point) on GPU 0 since 14:45:47Z (code_ds3r5).
  - S8a–d are queued behind S4a / S4b.
- **FT_PRE MAIN** s2026 (GPU 0) and s2027 (GPU 1).
- **FT_C MAIN** s2026 / s2027 (lanes 7 / 8).

### 14.3 Not started (waiting)
- **MAIN federated arms.** FA ×2, FP ×2, PF ×2, FA_Q8 ×2, S_CAL → FA_1024 ×2 → DP8 ×2 → DP1, DP_OPT_SENS, FA_X2, FT_FA,
  DP8_FROZEN.
  - They need `contracts/fl/FL_TUNING_SELECTION.json` (= the final S4 file). The launcher refuses anything else for MAIN
    (FIX-DELTA-02 M2).
- FA_WARM / FA_WARM_FROZEN MAIN and FT_FA_WARM. They need the S7 / S8 selection files.
- C_FULL s2027 re-evaluation (the reeval gap, DL 265). The GRU_D128 descriptive re-evaluation (06L_C C11). The no-FT UNSEEN_VAL
  references (C16).
- ONNX / INT8 bench and K16 rows, the browser run, the DATA-07 minimal set. Their server status is not clearly in the DL;
  verify.
- ANA freeze, VALIDATION read, ALL_METHODS_LOCK, TEST manifests, TEST look.

### 14.4 Sealed
- All MAIN_VALIDATION values of the deployment study (C_SMALL, C_FULL, PRE, LO, re-evaluations, LADDER_S1a, baselines).
- All TEST rows.
- PRE on CALIB.
- UNSEEN_VAL selection-panel values (not quoted).

### 14.5 Next steps, in order (PLAN; every duration is an ESTIMATE)
1. **S4 selection.**
   - When S4a–c finish, laneSEL4 runs `select-joint --stage S4`.
   - The operations lead copies FL_TUNING_S4.json **O_EXCL** to `contracts/fl/FL_TUNING_SELECTION.json`, writes a DL row, and
     only then enters the S4 values in the ladder register, with the winner's own gain beside the file's gain (DL 263).
   - Estimate at DL 268: B8 needs ≈ 7 more hours.
2. **Decide the fate of the early S7 / S8 screens.**
   - If S4 keeps the incumbent, S7a–d / S8a–d are the registered screen.
   - Otherwise record them "unused" and launch the 8 runs at the S4 point (DL 268).
3. **MAIN FL lanes** from the newest reviewed tree, with `--fl-tuning contracts/fl/FL_TUNING_SELECTION.json`:
   - FA s2026 / s2027, FP ×2, PF ×2, FA_Q8 ×2;
   - **S_CAL re-run under the selected recipe → S_RECORD → FA_1024 ×2 → DP8 ×2 → DP1 s2026**, with DP arms at P 2 / B 16;
   - **DP_OPT_SENS** (FA_1024 s2026 + DP8 s2026 under plain FedAvg), triggered because the winner is FedAdam;
   - FA_X2 (s2026, 12 EFE);
   - FT_FA: selection on UNSEEN_VAL, then ×2;
   - DP8_FROZEN (SHOULD-high).
   - Estimates: ≈ 10.7 h per MAIN FL process unshared (DL 169); ≈ 14–18 h wall on 8–10 lanes (06J §6). The deck placed the
     main FL and DP runs around 1–2 Oct.
4. **S7 / S8 selection** → `FL_WARM_SELECTION_S7|S8.json` (O_EXCL + DL row) → FA_WARM MAIN ×2 and FA_WARM_FROZEN MAIN ×2
   (`--warm-selection`; dry run first) → FT_FA_WARM (+ _FROZEN) ×2 (smoke first).
5. **Close the tool gaps before the freeze:**
   - reeval must accept C_FULL s2027 (DESCRIPTIVE), then run its 8 re-evaluations;
   - the ANA tools must learn the FT_FA_WARM arms and the B4 / K20 / K21 rows (FL-15 "not in scope" note);
   - run the GRU_D128 and no-FT re-evaluations;
   - deploy `tools/` to the server (the tools are not deployed yet, DL 245).
   - Each change needs a review and a new tree.
6. **Device / CPU items:**
   - finish FT_C and FT_PRE;
   - ONNX FP32 / INT8 bench (K16 cell, 16,384-user subset);
   - the browser run;
   - the DATA-07 minimal set (T41 on both panels; Study A for C_BIG / C_SMALL / FA on COLD_H; Study B for FT_C on COLD_H).
7. **ANA freeze.** Write `company/gates/ANA_SCRIPT_FREEZE.json`:
   - retention_report sha is currently **c699b3bf…** and ana_block_index **780b3ef9…**; this changes if step 5 edits the tools;
   - the contrast file v3 is a3d1456d…;
   - freeze the `--claims-only` output (claim register + number-free headline template);
   - fix the MASTER_PLAN §9 contrast path to v3 (ANA-04 note).
8. **The single VALIDATION read (preflight).**
   - `ana_block_index` builds the block, then `retention_report --check-only` runs on the real block, then the full VALIDATION
     report runs on all arms at once.
   - Anything added after this read is POST_VALIDATION_EXPLORATORY.
   - Deck estimate: around 2–3 Oct.
9. **Gate D before the lock** (multi-expert review):
   - D1: restore or withdraw the registered TEST contest singles;
   - D2: disclose the TEST partition fence and add a counts-only receipt;
   - D5 (SHOULD-high): e-mail a hash manifest of `contracts/` + `gates/` + the DL to the supervisor as an external timestamp.
10. **ALL_METHODS_LOCK.json.** It copies the arm set, checkpoints, the v3 contrasts and the privacy ladder.
11. **The TEST look.**
    - Build the TEST manifests (DATA-09) only after the lock, with a TEST_EXPOSURE_REGISTER entry.
    - Then run, once: reeval of every locked checkpoint, re-execution of the device procedures, baselines and ONNX rows on
      MAIN_TEST; then the frozen analysis on the TEST block.
    - Deck estimate: around 4–5 Oct.
12. **Thesis:**
    - 5 figures + 4 tables (multi-expert review E4);
    - the disclosures register (E3);
    - the Queen's "Statement of Originality and Disclosure of Use of AI" with the supervisor's approval (E1);
    - the reproducibility package (E8);
    - the living thesis reference and its structured registers (pending since 27 Sep).
    - Deck estimate for the final tables: around 6–7 Oct.

---

## 15. Rules for whoever continues

1. **Sealed data stays sealed.**
   - Do not open any MAIN_VALIDATION RESULT_ROWS, `sealed/` directory, TEST manifest or TEST row before the registered step.
   - The unseal gate is **not** a security boundary: the shared server account can open `sealed/` (DL 90 F3). The discipline
     is personal.
   - Never `cat` a STATUS.json that may carry a metric; grep only status / efe fields (DL 105).
2. **Register first.** Any new arm, grid, rule, contrast or interpretation goes into an amendment (or a DL row) **before** the
   value it governs exists or is read. After a read the item is exploratory. Never edit an amendment; write a new one.
3. **Check the DL tail and the target file before writing under `contracts/`** (lesson of 06F). Use the next free DL row
   number. Take stamps from `date -u`.
4. **Frozen trees.**
   - Never edit a deployed tree.
   - A run resumes only on its own tree.
   - Every code change goes to local `code/`, gets tests + NCs + an independent review, and is deployed as a **new** tree with
     per-file verification and a tiny-run digest comparison.
   - `code/v4eval/*.py` must stay byte-identical (61b64e61…).
5. **Never rewrite a running lane script.** Bash reads scripts incrementally. To change a plan, stop or withdraw the lane (and
   record it), then start a new one. Withdrawn lanes are logged (e.g. DL 237, 245).
6. **Selections are mechanical.** Use the tools (`lr_screen.py select / select-joint / select-warm`, `launch_device --select`).
   Copy the output O_EXCL into `contracts/`. Do not hand-edit a selection.
7. **Keep every checkpoint, receipt, log and failed attempt.**
   - Deletions only with the owner's approval and a verified sha256 manifest (DL 19, 44, 46).
   - Probes and smokes delete their own weights and keep digests (DL 30).
8. **Verify identities.**
   - Before any commit, check your own git identity (`git var GIT_AUTHOR_IDENT`).
   - Never commit under someone else's identity.
   - **No push or remote mutation without the owner's explicit approval.** The owner's own branch
     `research/federated-deployment-study` has a daily-push routine run by him (DL 217, 225, 236).
   - Verify every public claim against the V4 registries (DL 257).
9. **Do not install packages** into the shared venv or on the server (DL 118).
10. **GPU hygiene.**
    - Name the target GPU explicitly; the GPU index is in the RunConfig (DL 110).
    - Launchers refuse an occupied card unless the registered sharing flag is passed.
    - The server is shared with teammates' processes: never touch them (DL 37).
11. **Separate tags.** MEASURED / REPORTED / INFERENCE / PLAN, always. An ETA is never a result.

---

## 16. File map and glossary

### 16.1 Where the evidence lives (`V4/` = `.local_ai/ppsi_v4/`)

| What | Where |
|---|---|
| Decision log (every decision, incident, registration) | `company/DECISION_LOG.md` |
| Resume notes (handover blocks, newest on top; §2n = the chief-scientist handover) | `company/RESUME_HERE.md` |
| Amendments | `contracts/PREG1_AMENDMENT_*.md` |
| Locks | `contracts/ARCHITECTURE_LOCK.json`, `contracts/DATA_PROTOCOL_LOCK_V2.json`, `contracts/PREG1_LOCK_VALUES.json` |
| Client / evaluation contracts | `contracts/CLIENT_CONTRACT_V1.md`, `contracts/EVALUATION_CONTRACT_V4.md` |
| FL selection files | `contracts/fl/LR_SELECTION.json`, `LR_SELECTION_EXT.json`, `FL_TUNING_S2.json`, `FL_TUNING_S3.json` (next: `FL_TUNING_S4.json` / `FL_TUNING_SELECTION.json`, `FL_WARM_SELECTION_S7|S8.json`) |
| Device selection files | `contracts/device/SELECTION_FT_C.json`, `SELECTION_LO.json` |
| A40 pin, manifest registries, model variants, strata | `contracts/eval_device/`, `contracts/manifest_registry/` (REGISTRY_V2 + FULL_EXT / PRETRAIN_EXT / TEST_EXT), `contracts/model_variants/`, `contracts/strata/` (server copy authoritative) |
| Gates (contrast files, G1 / G1B / G2 approvals; ANA freeze to come) | `company/gates/` (ANA_CONTRASTS_v3.json is the registered contrast file) |
| FL ladder register (thesis table + curves) | `company/registers/FL_IMPROVEMENT_LADDER.md` |
| TEST exposure register | `company/registers/TEST_EXPOSURE_REGISTER.json` |
| Task packets / worker handoffs | `company/tasks/`, `company/handoffs/` |
| Reviews and audits | `company/reviews/` (REV-*.md; `REALISM_AUDIT_20260929.md`; `FL_STRONGEST_LADDER_RESEARCH_20260929.md`; `REAL_WORLD_FL_PRACTICE_AUDIT_20260930.md`; `MULTI_EXPERT_REVIEW_20260929/00_SYNTHESIS.md` + 14 part files) |
| Real run records used as reeval test fixtures | `company/evidence/reeval_records/` |
| Plans and server scripts | `company/plans/` (MASTER_PLAN_V5.md, REST_OF_PROJECT_PLAN.md), `company/plans/ops_scripts/` |
| Supervisor deck, script and guide | `company/presentations/` |
| Code (local, the source of every tree) | `code/` (fl, v4int, v4eval [frozen], v4train, v4data*, v4dev, v4trunc, v4base, tests_*), `tools/` (retention_report.py, ana_block_index.py, g2_run.py) |
| Deployed tree archives + checksums + environment | `deploy_archive/` |
| Server run roots (on Lobot) | `runs_ds3/` (LRSCREEN*, LRSCREEN_S2 / S3 / S4, WARM_PREVIEW, S7S8, CFULL, REEVAL_A40, DEV, BASE_R4, queues/), `runs_pretrain/`, `runs_p3/` (contest), `runs_ds/` (C_SMALL, ladder) |
| Living thesis reference (cross-check) | `.local_ai/thesis_reference/*_EXECUTION_REFERENCE_EN.md` §14.x, `STATE.md`, `CHANGELOG.md` |

### 16.2 Glossary

| Term | Meaning |
|---|---|
| **EFE** | The training-budget unit. 1 EFE = as many training-example exposures as the cohort has eligible TRAIN examples (N_EFE; one epoch's worth). All arms use 6.0 EFE (≈ 3 client visits in FL). Marks are every 0.5 EFE. (The expansion of the acronym is not spelled out in the files I read.) |
| **Mark** | One of the 12 in-run evaluations (0.5, 1.0, …, 6.0 EFE) |
| **PRACTICAL_BEST (PB)** | The checkpoint at the strictly best mark (improvement > 0.0002) within the 6.0 budget. The registered primary rule (T1). |
| **CONTROLLED_BUDGET (CB)** | The fixed 6.0-EFE endpoint checkpoint; reported beside PB; confirmatory for K12 / K13 |
| **CALIB** | The 32,768 tuning users (nested prefix of MAIN). The only users on which tuning values are read. |
| **MAIN** | The 131,072-user benchmark cohort (5–100 pool, selection key v2 prefix) |
| **UNSEEN_VAL / UNSEEN_HOLDOUT** | Two user-disjoint bands never trained on: device-recipe selection panel / participation-gap and cold-start panel |
| **TIEBREAK band** | Pool ranks [0, 1,348,682): the C_FULL population. Minus MAIN it is the PRE cohort. |
| **Retention (R)** | 100 × arm / reference (C_BIG unless stated), same checkpoint rule and seed, mean of per-seed ratios, paired user bootstrap |
| **K-contrasts** | The registered confirmatory comparisons K1–K21 + K9b, K9c, K11b, K20b, K20c (26); everything else is descriptive |
| **E2E / client-macro MRR@20** | Reciprocal rank of the true next item within the top 20 of all 158,486 items (0 beyond 20). Averaged over a user's rows, then over users. E2E counts out-of-vocabulary targets as misses. |
| **m (local-LR multiplier)** | The client LR = SMALL peak (0.000711) × m |
| **η_s, τ** | FedAdam server step size and its stabiliser |
| **FedAvgM** | FedAvg with server momentum (β 0.9, server LR 1) |
| **FL_CONST** | FL LR shape: warm-up 1 % of rounds → constant peak → cool-down over the last 10 % to 0.1 × |
| **Incumbent / ≥ 0.002 rule** | The previous stage's winner; a challenger replaces it only with a raw gain ≥ 0.002 |
| **Tie band** | Values within 0.0005 count as tied; a registered tie order decides |
| **Edge run / EdgeRunsMissing** | The extra registered run when the best cell sits at a grid edge; the selector refuses until it exists |
| **θ0 / θ_pre** | Registered random initial weights / the pretrained PRE endpoint_6.0 weights |
| **S_CAL / S_RECORD** | The clip-norm calibration run / its recorded clip norm S |
| **z, ε, δ, q** | DP noise multiplier, privacy budget, failure probability, Poisson sampling rate |
| **T3 evaluator** | The frozen evaluation configuration (full-catalogue scoring, pessimistic ties, fixed score block) |
| **STRICT** | Deterministic FP32 mode, bitwise reproducible, independent of co-scheduling |
| **A40 pin** | The registered second device class `NVIDIA_A40-FP32-DET`; rows are labelled MEASURED_PINNED_DEVICE |
| **O_EXCL** | Written once with exclusive-create; never overwritten |
| **NC (negative control)** | A test marked xfail that must fail when its guard is removed (checked with `--runxfail`) |
| **Lane** | A detached bash job queue on the server |
| **Frozen tree** | An immutable deployed copy of `code/` (code_ds3r2 … r5) |
| **ALL_METHODS_LOCK** | The lock of arms, checkpoints and contrasts written after the VALIDATION read and before TEST |
| **ANA freeze** | Pinning the analysis script's sha in `company/gates/ANA_SCRIPT_FREEZE.json` before the VALIDATION read |
| **RETRO_TEST** | The sealed TEST week (23–29 Nov 2019) |
| **L0 … L6** | The privacy levels (§10.4) |
| **Visit / round** | One client's download–train–upload / one server aggregation over ≈ 57 surviving clients (64 sampled, 10 % drop-out) |

---
*End of handover. Where this document and a primary file disagree, the primary file (contract, selection file, DL row)
wins. Please record the correction as a new DL row.*

# 02 — Part 1: research question, proposal alignment, and the "realistic Edge deployment" framing (privacy-level ladder)

- **Reviewer:** Claude Opus 5.5, acting as a senior production-FL / recommender scientist (multi-expert audit, 2026-09-29).
  READ-ONLY. This file is the only thing I wrote. No server contact, no git, no pytest, no project script run. I read no
  RETRO_TEST row, no RESULT_ROWS, no ranks or metrics file, and nothing under `runs_*/`.
- **Downstream values used:** only (a) the registered CALIB LR-screen values quoted in the DECISION_LOG (DL 181, DL 218) and
  (b) the MAIN baseline and contest values that DL 132 already reports as read and final (the analysis is post-G2).
- **Tags:** MEASURED (path:line) / REPORTED (a project document's own statement or number) / INFERENCE / OPINION /
  UNVERIFIED. Paths are relative to `.local_ai/ppsi_v4/` unless absolute. "proposal.txt:N" means line N of the proposal
  text extracted to the session scratchpad. The PDF is
  `eda/Privacy-Preserving Shopping Intelligence - Research Proposal 1.pdf`.
- **Verified the lead's project map (01_PROJECT_MAP.md):** its §1–§2 match the sources I read (proposal.txt:19–20, 95–170;
  06B:50–60; 06K:8–35). One nuance the map leaves out: its row "L5 … FT_C | nothing after download" copies 06B's wording,
  and I argue below (F1) that this wording is wrong.

---------------------------------------------------------------------------------------------------------------------
## 1. What the project does for this part (MEASURED)

### 1.1 The question and how the answer is formatted
- **Question (verbatim):** "How much shopping recommendation quality can be retained when moving from cloud-based to
  on-device or federated models?" (proposal.txt:19–20). The Edge framing is "how much recommendation quality is lost when
  shopping data never leaves the device?" (proposal.txt:24–26). The same question is re-registered in
  PREG1_AMENDMENT_04.md:9–10.
- **Answer format:** R = 100 × arm / C_BIG on client-macro MRR@20. It is the mean over seeds 2026 and 2027 with a paired
  user bootstrap (1,000 resamples, seed 20260926), and there is no pass/fail threshold (04:12–16; 06:31–32; 06A §7). The
  pairing rule is the same checkpoint rule and the same seed for numerator and denominator (06A §7).
- **Confirmatory contrasts:**
  - K1–K16: 06B:30–47.
  - K17–K19 = R(FA | FT_C | LO / C_FULL), seed 2026: 06I:7–15.
  - Hierarchy: K2 is primary. K1, K3, K8, K12 and K17 are secondary. The rest are tertiary.
    (`company/gates/ANA_CONTRASTS_v2.json:116–141`.)
- **Deployment branch (06K):**
  - PRE is the SMALL recipe trained on 1,217,610 non-MAIN users. The project interprets them as "users who opted in to
    share data (or public data of the same store); MAIN users' data never leaves their devices" (06K:8–11).
  - FA_WARM is FL on MAIN, warm-started from θ_pre. K20 = R(FA_WARM / C_BIG) and K21 = FA_WARM − PRE are secondary
    (06K:18–24).
  - PRE-alone and FT_PRE are DESCRIPTIVE references (06K:25–26).
  - S8 (FA_WARM_FROZEN) freezes the item tables. It is DESCRIPTIVE unless it beats S7 on CALIB by ≥ 0.002 (06K:29–35).
  - DL 206 adds R(FA_WARM / C_FULL) and R(PRE / C_FULL) as DESCRIPTIVE, and says "the thesis text for the deployment
    branch leads with them".

### 1.2 The privacy-level ladder as registered (06B:50–60)
| Level | Arms | Training exposure | Inference exposure | Guarantee |
|---|---|---|---|---|
| L0 | C_BIG, C_FULL, C_SMALL | raw events | query context | none |
| L1 | C_SMALL shipped (ONNX), KNN_EMB | raw events (trained centrally once) | nothing | none |
| L2 | FA, FP, FA_1024, FA_Q8, PF-shared | model deltas | nothing | none |
| L3 | DP ε = 8, 1 | clipped deltas + Gaussian noise on the sum | nothing | user-level (ε, δ = 5e-6) DP of the released model |
| L4 | PF (p_u), FT_FA | as L2 | nothing | none beyond L2 |
| L5 | FT_C | "nothing after the download" (06B:58) | nothing | none |
| L6 | LO, RULE "(own history)" (06B:59) | nothing | nothing | "data never leaves" |

Registered later:
- 06I:64–66 (disclosure 7a): non-DP deltas reveal the viewed items, because the item tables are 98 % of the parameters
  and there is no secure aggregation. "Only weights leave" is therefore data minimisation, not privacy.
- 06D:62–66: DP does not protect against the server, which sees the clipped per-client deltas.
- 06I:76–77: the placement of FA_WARM is "L2 with public pre-training on non-MAIN users".

### 1.3 Scale and participation knobs relevant to realism (REPORTED in the registrations)
- MAIN has 131,072 clients (≈ 2.59 M TRAIN decisions). C_FULL (TIEBREAK) has 1,348,682 users / 26.68 M decisions
  (06D §7).
- FL uses 64 clients per round with 10 % seeded drop-out, B16 × 2 passes and 6,827 rounds ≈ 3 visits per client
  (04 §3).
- The per-visit payload is 165.68 MB, FP32, down + up (06 §1).
- FL tuning ladder so far: S1b selected ×8, with CALIB FA = 0.028796 at PRACTICAL_BEST, against central CALIB ≈ 0.11–0.12
  (INFERENCE from the P2 era). Source: DL 218; `company/registers/FL_IMPROVEMENT_LADDER.md:18`.
- MAIN baselines (DL 132, read and final): C_BIG SASRec D256_B3 = 0.16120 (s2026 PRACTICAL_BEST), SKNN 0.11138,
  **LAST 0.07111**, POP 0.01943.

### 1.4 Proposal deviations as the project records them
- **Registered disclosures:** 06A:45–46 lists FedBN as inapplicable (LayerNorm backbone), purchase-AUC and
  comparable-offer tasks as out of scope, and battery as not measured. 06D:110–111 (§9g) says the tasks and datasets were
  replaced "by agreement with the supervisor" (REPORTED; I cannot verify that agreement).
- **Owner scope:** OWNER_SCOPE_V4.md:21–22 puts FedBN and Purchase Intent out of scope.
- **Code:** `code/v4base/rule.py:1–10` builds RULE from "item i's OWN MAIN-TRAIN target popularity". `code/v4int/launch_device.py:20`
  starts FT_C from "the base (C_SMALL central checkpoint …)", and C_SMALL is trained on MAIN (03 §2). DL 224 records the
  base as C_SMALL s2026 best.pt.
- **Public document (DL 217):** the pushed document
  `F:\Queens University\Microsoft Project\github repo\Privacy-Preserving-Shopping-Intelligence\docs\federated-deployment-study.md:17–18`
  describes Federated as "events never leave the device … sends only model updates" and On-device as "a small model is
  shipped once". It gives neither the item-set-leak caveat nor the fact that the shipped model was trained on the same
  users' raw events.

---------------------------------------------------------------------------------------------------------------------
## 2. State of the art — literature and real systems (verified today by web search unless marked)

### 2.1 Production FL and on-device systems (what "realistic" means)
| System / paper | Verified facts used here | Source |
|---|---|---|
| Gboard FL: Hard, Rao, Mathews, Ramaswamy, Beaufays, Augenstein, Eichner, Kiddon, Ramage, *Federated Learning for Mobile Keyboard Prediction*, arXiv 1811.03604 (2018/2019) | FedAvg next-word model trained on phones; beats server-trained recall. Eligibility (charging, unmetered, idle), 1.4 M params and 100–500 clients per round are per REALISM_AUDIT B6/B9/B11, which read the PDF; I did not re-read it. | https://arxiv.org/abs/1811.03604 |
| Bonawitz et al., *Towards Federated Learning at Scale: System Design*, MLSys (SysML) 2019, arXiv 1902.01046 | Google's production cross-device FL system: device check-in, per-round selection, drop-out tolerance, secure aggregation | https://arxiv.org/abs/1902.01046 |
| Xu, Zhang, Andrew, Choquette, Kairouz, McMahan, Rosenstock, Zhang, *Federated Learning of Gboard Language Models with Differential Privacy*, ACL 2023 Industry, pp. 629–639 | > 15 production LMs with DP-FTRL, ρ-zCDP ρ ∈ (0.3, 2); no uniform-sampling assumption; adaptive (quantile) clipping | https://aclanthology.org/2023.acl-industry.60/ |
| Singhal, Sidahmed, Garrett, Wu, Rush, Prakash, *Federated Reconstruction: Partially Local Federated Learning*, NeurIPS 2021 | Recommender (MF) with user embeddings that never leave the device; deployed in Gboard for "hundreds of millions" of users | https://arxiv.org/abs/2102.03448 ; Google Research blog 2021-12 |
| Paulik et al. (Apple), *Federated Evaluation and Tuning for On-Device Personalization: System Design & Applications*, arXiv 2102.08503 (2021) | Apple's production system is built mainly for federated *evaluation and tuning* of on-device personalisation, with FL added later | https://arxiv.org/abs/2102.08503 |
| **Browser FL precedent:** Hartmann, Suh, Komarzewski, Smith, Segall (Mozilla), *Federated Learning for Ranking Browser History Suggestions*, arXiv 1911.11807 (2019) | Firefox URL-bar ranker trained by FL on ≈ **360,000 Firefox Beta users in under four days**; a tiny model (handcrafted-heuristic weights, numerical gradients, an Rprop variant) | https://arxiv.org/abs/1911.11807 |
| **Chrome Topics API** (Privacy Sandbox docs) | Interest topics are computed on the device from browsing history by a shipped classifier; "no … external servers" | https://developer.chrome.com/docs/privacy-sandbox/topics/latest/ |
| **Privacy Sandbox retired** (Google announcement, Oct 2025) | Topics, Protected Audience, Attribution Reporting and **On-Device Personalization** retired on Chrome and Android for "low levels of adoption"; deprecation from Chrome 144, removal targeted at Chrome 150 | https://privacysandbox.google.com/blog/update-on-plans-for-privacy-sandbox-technologies ; https://www.adexchanger.com/privacy/google-pulls-the-plug-on-topics-paapi-and-other-major-privacy-sandbox-apis-as-the-cma-says-cheerio/ |
| **Microsoft Edge today (privacy statement)** | "If you allow personalization, the Microsoft Edge team collects and uses your Microsoft Edge browsing history to personalize experiences …"; only when signed in, permission given, and not Guest/InPrivate; browsing history "stored on secure Microsoft servers for up to 180 days" | https://learn.microsoft.com/en-us/legal/microsoft-edge/privacy |
| **Edge Shopping personalisation (Microsoft Support)** | "when we personalize Microsoft Edge Shopping, if we infer from your browsing activity that you prefer shopping at a particular store, the coupons or advertising you see might be customized"; the data includes browsing history and "the pages you view" | https://support.microsoft.com/en-us/microsoft-edge/microsoft-edge-browsing-activity-for-personalized-advertising-and-experiences-37aa831e-6372-238e-f33f-7cd3f0e53679 |
| **Edge on-device models** (Edge developer docs; Edge blog 2025-05-19) | Phi-4-mini built into Edge through the Prompt API; the model is "downloaded the first time a website or extension starts using" it; hardware-gated. A smaller CPU-capable Aion-1.0-Instruct appears from Edge 150 | https://learn.microsoft.com/en-us/microsoft-edge/web-platform/prompt-api ; https://blogs.windows.com/msedgedev/2025/05/19/introducing-the-prompt-and-writing-assistance-apis/ |
| EdgeRec: Gong, Jiang, Feng, Hu, Zhao, Liu, Ou, *EdgeRec: Recommender System on Edge in Mobile Taobao*, CIKM 2020 | Production on-device **re-ranking of cloud-computed candidates** with real-time on-device behaviour features | https://arxiv.org/abs/2005.08416 |
| Yao, Wang, Jia, Han, Zhou, Yang, *Device-Cloud Collaborative Learning for Recommendation*, KDD 2021 | Device-side personalisation patches over a cloud model, with distillation back to the cloud; helps long-tail users | https://arxiv.org/abs/2104.06624 |

### 2.2 Privacy of what leaves the device
- **Melis, Song, De Cristofaro, Shmatikov, *Exploiting Unintended Feature Leakage in Collaborative Learning*, IEEE S&P 2019.**
  Model updates in collaborative/FL training leak membership and properties of participants' data.
  https://discovery.ucl.ac.uk/id/eprint/10061777/. The specific result that non-zero embedding-gradient rows reveal which
  words are in a batch is my recollection of that paper: **UNVERIFIED** at that level of detail.
- **Chai, Wang, Chen, Yang, *Secure Federated Matrix Factorization*.** Gradients reveal ratings. Cited through
  REALISM_AUDIT B2; year and venue **UNVERIFIED** (the same caveat as there).
- **Proposal refs [9] Abadi 2016 (DP) and [10] Bonawitz 2017 (secure aggregation)** (proposal.txt:215–216). The proposal
  itself lists SecAgg as part of the privacy toolkit.

### 2.3 Evaluation framing
- **Yuan, Morningstar, Ning, Singhal, *What Do We Mean by Generalization in Federated Learning?*, ICLR 2022.** Separates
  the out-of-sample gap from the participation gap (https://arxiv.org/abs/2110.14216). The project's UNSEEN panels follow
  it.
- The other benchmark and rigour references (LEAF, FedScale, FLUTE, Krichene & Rendle 2020, Dacrema et al. 2019,
  Reddi et al. 2021, McMahan et al. 2018, Charles et al. 2021) were verified in FABLE_THESIS_RESEARCH_REVIEW §B.1 and
  REALISM_AUDIT. I did **not re-verify them today**, so cite them from those files' URLs only.

### 2.4 What a Google/Apple/Microsoft production reviewer would say the Edge system looks like (INFERENCE / OPINION on top of the verified facts)
1. **Status quo = L0 with consent.** Edge's shopping personalisation is cloud-side today. It is opt-in, needs a signed-in
   account, and history is retained up to 180 days (verified above). So "cloud" is the real baseline, and the real
   privacy question is: **what can Edge offer the users who do not opt in** (or who opt out)? The population is split
   into a consenting cloud cohort and a private cohort. That is exactly the structure of 06K's PRE / FA_WARM branch.
   OPINION: this is the project's most realistic design element, and it is currently only a DESCRIPTIVE side branch.
2. **Who the adversary is.** A browser-level assistant sees shopping across many retailers. The retailer already logs
   every on-site event. Moving to the device therefore protects the user from the **browser vendor** (cross-site profile
   aggregation), not from the retailer. REES46 is one retailer's server log. A realistic reading is: "the browser observes
   these events as the user browses, and Microsoft is the party that would or would not collect them". Nothing in the
   registrations states this threat model (MEASURED: 06B §3 and CLIENT_CONTRACT C5.6 name "the server", never whose server).
3. **Scale and participation.**
   - Browser FL has been shipped at ≈ 10⁵ participants (Firefox, 360 k users in < 4 days, verified). MAIN's
     N = 131,072 is therefore a realistic *participant* scale, though with a model about 10⁶–10⁷× larger than Firefox's.
   - Production cohorts per round are hundreds to thousands (Gboard: 100–500 per REALISM_AUDIT; DP needs a report goal
     of thousands). The simulated 64 per round sits below that range.
   - Eligibility (charging, idle, unmetered) and diurnal availability are standard in production. The simulation samples
     uniformly (disclosed, 06I §7c).
4. **Models browsers actually ship.** Browsers ship large one-time downloads (Edge's Phi-4-mini, hardware-gated). So
   L1 = "download a 21–83 MB model once and infer locally" is a shipped pattern.
   - Recurrent **83 MB uploads per training visit** are not a shipped browser pattern. The largest documented browser FL
     (Firefox) used a tiny model.
   - A production design would federate a small encoder over frozen or public item tables. That is S8 (06K §3), which
     the project keeps DESCRIPTIVE.
5. **Production on-device recommenders are hybrids.** They re-rank cloud candidates (EdgeRec) or patch a cloud model
   (DCCL, Federated Reconstruction). The project's on-device arms score the full 158,486-item catalogue locally. That is
   computationally fine (≈ 10 M MACs per query, REALISM_AUDIT C1), but it is not how deployed e-commerce edge
   recommenders work. The hybrid leaks the candidate request, which is itself a privacy point worth one sentence.
6. **On-device personalisation infrastructure is fragile.** Google's retirement of Topics and On-Device Personalization
   for "low adoption" (verified) is the real-world counterpart of the project's guideline question. Delivering an
   on-device system is an ecosystem/product question, not only a question of retained quality.

---------------------------------------------------------------------------------------------------------------------
## 3. Comparison — strengths, weaknesses, errors, unrealistic choices, missing pieces

### 3.1 Strengths (OPINION, evidence cited)
- **S1. The answer format fits the question.** A retention ratio with paired user-bootstrap CIs, registered before the
  reads (04 §1, 06B §2), is the direct quantitative form of "how much can be retained".
- **S2. Explicit decomposition.** Size (K1), federation at matched size (K3), collaboration (K11), privacy (K12/K13),
  compression (K14), data advantage (K15) and quantisation (K16). This is richer than any federated-recommendation paper
  in the prior literature table (FABLE_THESIS_RESEARCH_REVIEW §B.1).
- **S3. The realism audit was adopted before any governed read.**
  - The item-set-leak disclosure is registered (06I:64–66).
  - The "equal-exposure lower bound" and "recipe-qualified claims" interpretations are registered (06I:55–61).
  - The FL tuning ladder was added (06F–06J). The public doc lists its stages.
  - This is uncommon discipline.
- **S4. The deployment branch matches Edge's real consent structure.** PRE on consenting users, FL on private users
  (06K §1–§2), with the rule "never warm-start from C_FULL/C_BIG" (06K:27). With PRE-alone and FT_PRE as references, the
  project can answer the product question in §2.4(1). No prior review framed it this way.
- **S5. Participation gap and cold users are represented** (UNSEEN_HOLDOUT, TRUE_COLD; Yuan et al. 2022).
- **S6. The trust boundary is honestly disclosed** (CLIENT_CONTRACT_V1.md:201–213: "comparisons of parameter learning
  under a shared information block, not privacy claims").

### 3.2 Errors (the claim would be wrong as registered)
- **F1 — L5 is misplaced: FT_C is not "nothing leaves", it is L1 + local fine-tuning.**
  - FT_C's base is the C_SMALL central checkpoint (`code/v4int/launch_device.py:20`; DL 224). C_SMALL is trained
    centrally on the **same MAIN users' raw events** (03 §2).
  - So, for the evaluated users, FT_C's training exposure equals L0/L1 (raw events). 06B:58 says "nothing after the
    download", which hides that the download is a model of the user's own uploaded data.
  - Placed above L2–L4, L5 suggests FT_C is *more* private than FL. It is less private than FL for the same users.
  - The realistic L5 arm, "cloud model trained on others, fine-tuned locally", is **FT_PRE** (06K:25–26). It is only
    DESCRIPTIVE today.
  - K7 and K18 are labelled correctly ("cloud-trained + local fine-tune", ANA_CONTRASTS_v2.json:41–43, 96–98). The
    ladder level is what is wrong.
- **F2 — RULE is not "own history" only.** RULE multiplies the user's own shares by "item i's OWN MAIN-TRAIN target
  popularity" (`code/v4base/rule.py:8–10`). That popularity is an aggregate of the evaluated users' raw events, the same
  kind of pooled cloud statistic as POP.
  - It is a realistic ingredient: the retailer or browser vendor can publish popularity, ideally from consenting users or
    with DP counts.
  - But 06B:59's "data never leaves (shared catalogue/θ0 only)" understates it.
  - LAST (repeat the most recent item) is the only genuinely history-only device baseline, and it is **absent from the
    ladder**.
- **F3 — The ladder is presented as one ordered scale, but privacy has at least three axes.** The axes are: training
  exposure, inference exposure, and *against whom* the guarantee holds (server vs model consumers).
  - Against an honest-but-curious aggregation server without SecAgg, L2 ≈ L1. Both reveal what the user viewed: L1 as raw
    events, L2 as per-visit item sets plus the activity count (06I:64–66; REALISM_AUDIT B2–B3).
  - L3 protects only against consumers of the released model (06D:66).
  - L4 = L2. L5 (FT_C) = L1.
  - For the private user, the only rows that protect against the vendor are **L6 (LAST / LO; RULE with a published
    table), PRE-alone and FT_PRE**, plus inference privacy in L1.
  - An examiner at PoPETs or USENIX (the proposal's venues, proposal.txt:178–179) will reject a linear L0→L6 scale.
- **F4 — K17–K19 add no new information about deployment.**
  - For seed 2026 under the same PRACTICAL_BEST/PRACTICAL_BEST pairing on the same user set, R(FA / C_FULL) ≡
    R(FA / C_BIG) · R(C_BIG / C_FULL). That is, the point estimate of **K17 = K2(s2026) × K15**. Likewise K18 = K7 × K15
    and K19 = K8 × K15 (INFERENCE from the definitions, 06I:7–15, 06A §7).
  - They also compare FL on 131 k users with a cloud model on 1.35 M users. That is "the federation cost *plus* 90 % of
    the users' data thrown away", a scenario no deployment would choose.
  - The realistic counterparts are (i) FL at the same population as C_FULL and (ii) the hybrid "consenting cloud + FL on
    private users" (FA_WARM vs C_FULL). Neither is confirmatory: (i) does not exist without DP (DP8_FULL only, 06I:46–52);
    (ii) is DESCRIPTIVE (DL 206).
  - As registered, K17 is a *co-headline* (06I:13) that is algebraically redundant and biased against FL.
- **F5 — Retention has no floor, and the likely FL numbers make this decisive.**
  - LAST, a zero-learning device-only heuristic, reaches 0.07111 on MAIN, i.e. 44 % of C_BIG's 0.16120 (DL 132).
  - Cold-start FA on CALIB is 0.0288 at S1b against ≈ 0.11–0.12 central (DL 218), i.e. ≈ 25 % retention.
  - INFERENCE: unless S2–S4 lift FA about 2.5×, **the registered headline FL arm will be below the trivial on-device LAST
    baseline**. Then "FL retains X %" without the floor misleads a product reader; the rational Edge decision would be to
    ship LAST/RULE.
  - K11 = FA − LO uses LO (a 20.7 M model trained from θ0 on ≤ 100 events: "a control, not a product", REALISM_AUDIT C4)
    as the no-collaboration reference. That flatters FL.
  - What is missing is a "headroom closed" reading, (arm − best device-only baseline) / (C_ref − best device-only
    baseline), and FA − LAST.

### 3.3 Weaknesses and unrealistic choices (already disclosed or partly addressed)
- **W1 — No threat model.** The adversary (the browser vendor's aggregator) and the out-of-scope parties (the retailer,
  which sees its own logs; the display-catalogue fetch, REALISM_AUDIT C2) are never stated. See §2.4(2). The proposal's
  question only makes sense once this is fixed.
- **W2 — The deployment branch is secondary or DESCRIPTIVE, though it is the Edge answer.**
  - K20 (vs C_BIG) is secondary and mixes a 9× data advantage (DL 206).
  - The contrasts that match the data (FA_WARM / C_FULL, PRE / C_FULL, FT_PRE / C_FULL) are DESCRIPTIVE.
  - OPINION: for the owner's top priority, these are the numbers a Microsoft PM would ask for first.
- **W3 — The opt-in share is fixed at 90 % / 10 %.** PRE uses 1,217,610 of 1,348,682 users (06K:8–10). Real opt-in
  rates for signed-in personalisation are unknown to me (UNVERIFIED) but are unlikely to be 90 %. The deployment answer
  is conditional on this share. That is a disclosed point, and a cheap sensitivity (see R7).
- **W4 — PRE's recipe and the pooled block touch MAIN.**
  - The SMALL recipe was selected on CALIB ⊂ MAIN (03 §3).
  - The normalisers, catalogue and item age are pooled over full TRAIN, which includes MAIN (CLIENT_CONTRACT_V1.md:223,
    "34 of the 53 kept channels depend on other users' pooled interactions").
  - So "MAIN users' data never leaves their devices" (06K:11) holds for training examples, not for the bootstrap block.
    This is already disclosed generically (06B:60) and needs one sentence in the deployment-branch text.
- **W5 — Payload, participation and optimiser realism.** 165.7 MB per visit; 64 per round, uniform; 6.0-EFE budget.
  These are disclosed (06I §6–§7) and belong to Parts 6 and 8. For framing, only S8 has a production-shaped payload.
  Keeping it DESCRIPTIVE and single-seed (06K:34) leaves the realistic bits-vs-quality point weakest.
- **W6 — One task and one retailer.**
  - "Comparable offers" (proposal Step 1c, proposal.txt:99) is the actual Edge Shopping product feature (coupons, price
    comparison). It is the dropped task most relevant to Edge.
  - Its removal is justified by the data: a single retailer has no cross-retailer offers. The thesis must state that
    reason, not only "by agreement".
  - Next-category (Step 1a) could be derived cheaply from item scores, but v4eval is frozen (DL 180), so it would need a
    separate descriptive tool.
- **W7 — The public document overstates privacy** (github repo `docs/federated-deployment-study.md:17–18`). It repeats
  exactly the "only model updates leave" sentence that 06I §7a says must not appear unqualified.

### 3.4 Proposal coverage (status now; builds on FABLE_THESIS_RESEARCH_REVIEW §A, updated for 06F–06K)
| Proposal item (proposal.txt line) | Status | Comment |
|---|---|---|
| RQ (19–20) | **Covered (algorithmic)**, **Partial (deployment)** | K2 answers matched-cohort retention; the realistic Edge answer depends on the DESCRIPTIVE branch (F4, W2) |
| Step 1 centralized upper bound, tasks (a)(b)(c), NDCG@10/HR@10/AUC (95–100) | Partial | Next-item replaces (a)–(c). NDCG@10/HR@10 secondary. AUC dropped with Purchase Intent (owner scope). C_FULL is the true "maximum achievable" (a single seed) |
| Step 2 FedAvg / FedProx / FedBN / PFL; comm cost, convergence (102–107) | Covered except FedBN | FedBN inapplicable (LayerNorm): justified. FedProx μ untuned (disclosed). SCAFFOLD appears only in the proposal background (55–57), is not run, and is not listed as a deviation (add one line) |
| Step 3 local MF / pretrained + fine-tune / kNN cached / rules (109–114) | Covered, but mislabelled levels | LO replaces local MF (fine). FT_C is cloud-trained on the same users (F1); FT_PRE is the faithful "pre-trained embeddings + local fine-tuning". KNN_EMB uses MAIN-trained tables (L1; correct) |
| Step 4 cold start (116–119) | Partial → good if DATA-07 runs | Observational bins registered; Study A/B SHOULD-medium (06I §8) |
| Contribution: privacy-utility curves (156–157) | Partial | ε ∈ {∞, 8, 1, [4]} at MAIN scale plus DP8_FULL at scale. The curve's x-axis (the ladder) is mis-ordered (F3) |
| Persona analysis (159–160) | Covered (descriptive) | — |
| Cold-start "events to parity by user type" (161–162) | Partial | Bins × strata; parity is defined against C_BIG, which trained on those same users (fine for warm users; say so) |
| Bits vs quality for various protocols (164–165) | Partial → OK if S8 has 2 seeds | FA (165.7 MB), FA_Q8 (≈ 103.6 MB), S8 (encoder-only, ≈ 0.4 M params). S8 is the only point with a production-sized payload |
| Browser guidelines: memory, latency, battery (167–168) | Partial | ONNX size, latency and RSS (server CPU) plus an ORT-Web page (inference). On-device *training* time and memory are not measured. Battery is not measured (disclosed). No eligibility or policy guidance |
| Open-source benchmark (170) | Pending | Owner-gated; a public doc already exists (W7) |
| Datasets (123–152) | Replaced by REES46 | Justified by the proposal's own "architecture advantage" argument (199–202). External validity is the cost |
| Privacy toolkit refs [9] DP, [10] SecAgg (215–216) | DP yes; SecAgg no | SecAgg is the missing ingredient that would make L2 meaningful against the server (F3). Name it as the deployment requirement |

---------------------------------------------------------------------------------------------------------------------
## 4. Recommendations

**Read status:** no FL MAIN, device-arm, C_FULL, PRE, FA_WARM or S8 value has been read.
- The only downstream reads are the registered CALIB screens (DL 181, 218). MEASURED from DL 183–224 and RESUME_HERE §2n.
- The ANA freeze and the VALIDATION preflight are still pending (RESUME_HERE:15–16).
- So every item below can still be registered before its governed value is read. **Deadline: before the ANA freeze.**

| ID | Item | Class | Action | Cost (INFERENCE) | Affects | Pre-registration needed + status |
|---|---|---|---|---|---|---|
| R1 | **Re-issue the privacy ladder as a matrix**: columns = training exposure / inference exposure / guarantee / *against whom* (vendor server vs model consumers); rows re-placed. FT_C → "L1 + local FT (same users' raw data in the cloud)". FT_PRE → true L5. PRE-alone → "cloud-trained on consenting users, device inference" (private-user exposure: none). RULE/POP → "device + published aggregate". Add LAST to L6. FA_WARM → "L2 + consenting-cohort pretraining". Add a SecAgg column marked "not simulated". Drop the implied monotone order (F1–F3) | MUST | CHANGE | 0 GPU; ≈ 1 h text | thesis Table 5.1 (MASTER_PLAN_V5.md:203–206), curves' x-axis; no run | Yes, text amendment replacing 06B §3 table; no governed value involved; status: OPEN, do before ANA freeze |
| R2 | **State the threat model** in ch.1/§5 and in the ladder header: adversary = browser vendor's aggregation/serving infrastructure (honest-but-curious); the retailer sees its own logs and is out of scope; the display-catalogue fetch leaks unless local (REALISM_AUDIT C2); simulation = one orchestrator (C5.6) | MUST | ADD | 0; ≈ 1 h | all privacy claims | Text only (disclosure list, like 06I §7); OPEN |
| R3 | **Report a floor with every retention**: add DESCRIPTIVE columns "headroom closed" H = (arm − LAST) / (C_ref − LAST) and the paired Δ FA − LAST (and FT_C − LAST, LO − LAST). The headline sentence must say whether FL beats the best history-only device baseline (F5). Floor fixed as LAST (the only history-only baseline; value already public in DL 132, arms unread) | MUST | ADD | ANA delta + test ≈ 2–3 h (Sonnet/Opus) | ANA-01/ANA-03 before freeze; VALIDATION + TEST reports | Yes (analysis registration, DESCRIPTIVE); floor choice made now before any arm read; OPEN |
| R4 | **Re-label K17–K19 in the thesis as "K2·K15 compositions" (FL on the cohort vs cloud on the pool = federation cost × data deficit)** and stop calling K17 the deployment co-headline (F4). Keep them (they are registered; deleting them is unnecessary) | SHOULD | CHANGE (interpretation) | 0 | thesis text, hierarchy statement | Interpretation note registered before the read (as 06D §3 / 06I §6 did); OPEN. Hierarchy unchanged unless R5 adopted |
| R5 | **Promote the data-matched deployment contrasts to confirmatory (secondary tier)**: D1 = R(FA_WARM / C_FULL), D2 = R(FT_PRE / C_FULL), D3 = R(PRE / C_FULL) (seed 2026 for C_FULL; FA_WARM 2 seeds against the one C_FULL seed, as 06E did for K15), and keep K21. State them as *the* answer to the Edge question ("what Edge can give non-consenting users relative to today's cloud personalisation"); K2 stays the algorithmic primary | SHOULD (strong; MUST if the owner's "realistic Edge" priority is the thesis headline) | CHANGE | 0 GPU (arms already registered); ANA-03 delta ≈ 2 h; multiplicity statement 19 → 22 | ANA_CONTRASTS_v2 → v3 before freeze; ALL_METHODS_LOCK | Yes, new amendment; PRE training, FA_WARM/FT_PRE not run, C_FULL unread → registrable; OPEN |
| R6 | **Add the non-DP counterpart of DP8_FULL** ("FA_FULL": DP8_FULL's exact config with z = 0, clipping as DP8_FULL, N = 1,348,682, m = 8,192, T = 494, seed 2026). It gives the only same-population federation retention R(FA_FULL / C_FULL) and an at-scale privacy cost R(DP8_FULL / FA_FULL). Without it, DP8_FULL's result mixes DP, federation and population (06I:51–52) | SHOULD | ADD | ≈ DP8_FULL's cost (REALISM_AUDIT S2: ≈ 110 process-h, ≈ 14 h on 8 pool workers) | new run; lanes after S4 selection | Yes, before any DP8_FULL/C_FULL read; OPEN. If compute forbids, register that DP8_FULL is DESCRIPTIVE only and not "Edge-scale privacy cost" |
| R7 | **Opt-in-share sensitivity for the deployment branch** (W3): PRE_50 on 131,072 non-MAIN users (a 50/50 split; C_SMALL cost) → FA_WARM_50 on MAIN, seed 2026, DESCRIPTIVE; disclose the 90/10 split in every deployment sentence regardless | NICE (disclosure part: MUST) | ADD | PRE_50 ≈ C_SMALL cost (≈ 1.3 GPU-h per REPORTED 03 §5 scale) + one FA_WARM run ≈ 10.7 process-h | new runs | Yes if run; disclosure text now |
| R8 | **Run S8 (FA_WARM_FROZEN) with 2 seeds unconditionally and present it as the production-shaped FL arm** (encoder-only uploads; input-table item-set leak removed; production-scale payload), not as a conditional side row (06K:34) | SHOULD | CHANGE | + 1 MAIN run ≈ 10.7 process-h | S8 lane | Yes (scheduling/tier change before any S8 read); OPEN |
| R9 | **Fix the ladder text of the deployment branch**: "MAIN users' training examples never leave; the bootstrap block (normalisers, catalogue, item age) and the SMALL recipe were fitted on pooled data including MAIN" (W4) | MUST (one sentence) | CHANGE | 0 | thesis text | Text; OPEN |
| R10 | **Correct the public document** `docs/federated-deployment-study.md:17–18` at the owner's next approved push: add "model updates reveal which items were viewed unless secure aggregation or DP is used" and "the shipped small model is trained in the cloud on the same users' data" (W7) | SHOULD | CHANGE | 10 min + owner-approved push | public repo only | No registration; GitHub mutation needs the owner's explicit approval; OPEN |
| R11 | **Complete the "what changed from the proposal" table**: add SCAFFOLD (background only, not run), SecAgg (cited [10], not simulated), "comparable offers" dropped *because a single-retailer log has no cross-retailer offers* (not only "by agreement"), next-category not reported (evaluator frozen), local MF → LO, datasets. Give the supervisor-agreement evidence a source ID (06D §9g is REPORTED only) | MUST (text) | ADD | ≈ 1 h | thesis ch.1 | No; OPEN |
| R12 | **Device guideline gap**: time and peak-RAM of one FT_C / FL client visit (≈ 4 steps, 20.7 M params, full-K softmax) on the owner's laptop CPU (PyTorch CPU or ORT), label HISTORICAL_REFERENCE; state device eligibility (charging/idle/unmetered as in production) and the one-time download size as guidelines; battery stays "not measured" | SHOULD | ADD | owner ≈ 30 min + a small script | ch.7.7 guidelines | No (descriptive measurement); OPEN |
| R13 | **Name the hybrid device-cloud re-ranker (EdgeRec/DCCL pattern) as the unmodelled production alternative** and state its privacy cost (the candidate request leaks context) | DISCLOSE-ONLY | ADD | 0 | discussion | No |
| R14 | **Browser-ecosystem context**: Chrome Topics was on-device and was retired in 2025 for low adoption; Edge already ships on-device SLMs; Firefox shipped browser FL at 360 k users with a tiny model. Use these as the realism anchors for the guidelines chapter | NICE | ADD | 0 | discussion | No |
| R15 | Stale command: MASTER_PLAN_V5.md:170–171 passes `ANA_CONTRASTS.json` (v1, K1–K16), while the binding set is v2 (06I §1). ANA-03 fails closed without `--allow-v1-contrasts` (company/tasks/ANA-03.md:14), so no silent omission, but the plan text should point to v2 (or v3 after R5) | NICE | CHANGE | 5 min | plan text | No |
| R16 | KEEP: K2 as the algorithmic primary (matched cohort, matched θ0/recipe/evaluator); C_BIG as the "no privacy, same data" control; the recipe-qualified claim rule (06I §6c); the equal-exposure lower-bound interpretation | — | KEEP | 0 | — | Registered (06D §4, 06I §6) |

**The denominator question (asked explicitly):**
1. **C_BIG is the right denominator for the *algorithmic* claim.** It holds the same users, data, θ0 family, recipe and
   evaluator fixed. It includes the size step, which K3 removes.
2. **C_FULL is the right denominator for the *deployment* claim**, but only when the numerator also has access to the
   same population's data: FA_WARM, PRE, FT_PRE, or an FA_FULL that does not exist yet.
3. **C_FULL paired with cohort-only FL (K17)** is a mismatched pair, and it is algebraically redundant (F4).
4. **Retention for a product decision** should also be read against the best history-only device baseline (R3), because
   that is the real alternative to "cloud" for a private user.

---------------------------------------------------------------------------------------------------------------------
## 5. Score: **6.5 / 10** for Part 1

**Justification (OPINION).** As an *algorithmic* answer to the proposal's question, the design is excellent (8–9/10): a
pre-registered retention ratio with paired CIs, a clean decomposition (K1–K16), an honest trust-boundary disclosure, a
participation-gap panel, and a realism audit adopted before any governed read. As a *realistic Edge deployment* answer,
which is the owner's stated top priority, it is weaker (≈ 5/10), for four reasons:
1. The privacy ladder, which is the proposal's "privacy levels" axis, misplaces FT_C (F1) and RULE (F2). It presents a
   multi-axis notion as one order in which L2 ≈ L1 against the server (F3).
2. There is no threat model, even though the dataset is a retailer's own log (W1).
3. The contrasts registered as the "cloud-data" co-headline (K17–K19) are compositions K2·K15 and describe no deployment
   anyone would run (F4). Meanwhile the project's best idea, the consenting-cohort PRE / FA_WARM / FT_PRE branch, which
   mirrors Edge's actual opt-in personalisation, is DESCRIPTIVE (W2).
4. Retention is reported without a device-only floor, while the evidence so far (CALIB FA ≈ 25 % vs LAST ≈ 44 % of
   C_BIG, INFERENCE) suggests that floor will decide the story (F5).

All of these are cheap to fix and none has been read yet. With R1–R3, R5, R9 and R11 applied before the ANA freeze, I
would score this part 8.5/10.

---------------------------------------------------------------------------------------------------------------------
## 6. UNVERIFIED items and open questions
- **Unverified facts**
  - Real opt-in rates for Edge personalisation or shopping data sharing: not found. The 90/10 split's realism is unknown.
  - Edge's active-user count ("hundreds of millions"): not verified. Do not cite.
  - The Melis et al. 2019 detail that non-zero embedding-gradient rows reveal batch words: the paper and venue are
    verified, the specific result is from memory.
  - Chai et al. FedMF year/venue: as in REALISM_AUDIT, unverified.
  - Gboard's per-round cohort (100–500), 1.4 M-parameter model and eligibility rules: taken from REALISM_AUDIT's PDF
    reading. I verified only the paper's existence and abstract today.
- **REPORTED only:** "replaced by agreement with the supervisor" (06D:110–111). No source document was found in the
  files I read.
- **Open questions**
  - **Is LAST computed on exactly the MAIN_VALIDATION user set and rows of the A40 block?** DL 155 says baseline rows are
    CPU, device-independent point estimates. R3 needs identical user sets for the paired Δ.
  - **Does the ANA script pair a 2-seed numerator with a 1-seed C_FULL denominator** for D1/D2 (R5), or must FA_WARM
    also be reported per seed? Follow the 06E K15 precedent.
  - **Does C_FULL's seed 2027 (running, NICE) become a registered second denominator seed?** If so, register it before
    either C_FULL value is read.
  - **Should RULE's popularity table for the private-user scenario come from PRE's consenting cohort** (non-MAIN TRAIN)
    rather than MAIN TRAIN? That is the realistic "published aggregate". It is cheap on CPU, and a DESCRIPTIVE variant is
    possible before any RULE read (RULE is sealed/unread per 06D).

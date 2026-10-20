# 10 — On-device arms (LO, FT_C, FT_FA, PF personal part, KNN_EMB, RULE), selection panel, causality of support

Expert: on-device learning / personalisation reviewer (Claude Opus 5.5 acting as reviewer). READ-ONLY. Written 2026-09-29.
Tags: MEASURED (file:line), REPORTED (a project document's own statement or number), INFERENCE, OPINION.
Paths are relative to `.local_ai/ppsi_v4/` unless absolute.
What I did not read: no RETRO_TEST row, no downstream RESULT_ROWS, no SELECTION.json, no ranks or metric files, nothing under runs_*,
and no BASELINE_RULE or KNN_EMB value. The only numbers I quote are ones the DECISION_LOG itself reports: the contest and MAIN
baseline values in DL 132, and the SMALL peak LR in 06K §1.

---------------------------------------------------------------------------------------------------------------------
## 1. What the project does for this part (MEASURED)

### 1.1 Arms and what each one uses
| Arm | Start state | What adapts on the device | Data used | Bytes | Evidence |
|---|---|---|---|---|---|
| LO | θ0 of the seed (random, recentered SASREC_D64_B2, ≈ 20.7 M params) | every parameter of the SMALL model | the client's own support rows | 0 | `code/fl/local_only.py:1-21, 117-152`; `code/v4int/launch_device.py:13-17, 207-215` |
| FT_C | C_SMALL `best.pt` (central, trained on MAIN TRAIN) | every parameter (`fl/finetune.py:7`) | own support | down = one base copy, up = 0 | `fl/finetune.py:1-33, 140-210`; `launch_device.py:17-20` |
| FT_FA | FA global checkpoint `best.pt` | every parameter | own support | as FT_C | same |
| FT_PRE | θ_pre (PRE, 1,217,610 non-MAIN users) + the FT_C recipe | every parameter | own support | as FT_C | `contracts/PREG1_AMENDMENT_06K.md:25-26`. DESCRIPTIVE only. No code, task or label found (`grep FT_PRE` hits only DL and the ladder research) |
| PF | FA shared model + a personal query offset p_u ∈ R⁶⁴ (starts at 0) | p_u, with its own persistent AdamW, during the ≈ 3 FL visits | own rows, inside FL | as FA; p_u is never uploaded | `fl/personal.py:1-16, 150-155`; `v4int/launch_fl.py:16, 164, 656` (personal LR = SMALL peak × round factor, "never multiplied"; λ = 1e-4) |
| KNN_EMB_SMALL / _BIG | a cached item table: SASRec `output_embed`, from the SMALL or BIG checkpoint | nothing | the last 10 valid history items | a 40.6 MB table for SMALL (158,486 × 64 FP32, INFERENCE) | `code/v4base/knn_emb.py:1-6, 43, 119-145`. Empty history falls back to MAIN-TRAIN popularity (L136, L141) |
| RULE | category/brand affinity × popularity within the group | nothing | own share of views over the causal window | small tables | `code/v4base/rule.py:3-13, 116-169`. Popularity comes from MAIN-TRAIN pooled counts. Category = `main_category` (vocab 16, L63) |
| LAST (baseline, not in the ladder) | none | none | own history ranked by recency | 0 | `code/v4base/last.py:1-14` |

### 1.2 Recipes
- **FT grid:** passes {1, 2, 4} × LR fraction {0.1, 0.3} × SMALL peak (`fl/finetune.py:50-51`; `launch_device.py:103`). The LR is
  constant. Each client gets a fresh AdamW (β 0.9/0.999, eps 1e-8, wd 1e-5 on ndim ≥ 2), B16, clip 1.0, no early stopping
  (`fl/finetune.py:63-97, 193`; `fl/client.py:51-60, 160-168`).
  - SMALL peak = 0.000711 (`contracts/PREG1_AMENDMENT_06K.md:12`), so the FT LRs are 7.1e-5 and 2.1e-4.
  - Each budget is read off one trajectory, evaluated after passes 1, 2 and 4 (nested; `fl/finetune.py:13-16, 197-204`).
  - The grid has **no 0-pass cell**. `eval_base` exists (`fl/finetune.py:76, 183`), but `make_runners` never sets it
    (`launch_device.py:216-223`), so selection cannot choose "do not fine-tune".
- **LO grid:** passes {1, 2, 4} at the SMALL peak, with the `s_efe` shape compressed onto the client's own steps. Step t of
  T = P·⌈n/16⌉ uses peak × s(6(t+1)/T) (`fl/schedule.py:106-117`, knots at `fl/schedule.py:38`; `fl/local_only.py:141-149`).
  - Consequence (MEASURED from the formula): if a client has n ≤ 16 support rows, then T = P.
    - LO_P1 is **one step at s(6.0) = 0.01 × peak ≈ 7e-6**.
    - LO_P2 is two steps at 0.1× and 0.01× peak.
    - LO_P4 has exactly one step at the peak (s(1.5) = 1), then three steps at 0.1×, 0.1× and 0.01×.
  - The module docstring discloses only that tiny clients "skip the warmup" (`fl/schedule.py:23-24`). The LR itself is never
    screened for LO (06B §1 L21-22: "LO (lo_step schedule at the SMALL peak)" is unchanged).
- **LO cell selection:** selected once on UNSEEN_VAL by the same rule as FT. The two cells not selected are DESCRIPTIVE
  (`contracts/PREG1_AMENDMENT_06E.md:11-16`; `launch_device.py:160-166, 670-672`).

### 1.3 Selection panel
- UNSEEN_VAL = support UNEW_VAL_SUPPORT_32768 → eval UNEW_VAL_VALIDATION_32768, 12,792 evaluable users, disjoint from MAIN,
  UNSEEN_HOLDOUT and every report panel (`contracts/PREG1_AMENDMENT_06A.md:9-17`; `launch_device.py:112, 602`).
- `--select` is refused on any other role (`launch_device.py:139-156`).
- SELECTION.json is written O_EXCL. It carries the content sha256 of the eval manifest it selected on and is verified against
  the registry before any GPU work (FIX-DELTA-01 F6).
- Non-smoke LO runs off UNSEEN_VAL need `--selection` (F7, `launch_device.py:160-166`).
- Rule: the best E2E client-macro MRR@20 wins; ties go to fewer passes, then the smaller LR (`launch_device.py:517-524`).
- The selection is made at seed 2026 and applied to both seeds and to every panel (INT-04 handoff; 06E §1).
- Status (REPORTED, DL 224): SEL_FT_C and SEL_LO were launched at 16:20Z from code_ds3r3. No selection row exists in the DL yet,
  so **no device value has been read**.

### 1.4 Causality of support
- `build_client_plans` sets the cutoff to the earliest decision_ts among the client's evaluation rows, and to min(that,
  TRAIN_END) for the warm roles MAIN, CALIBRATION and MAIN_TEST (`launch_device.py:238-271`, L265).
- `execute` **refuses** the run if any valid support row has target_ts ≥ cutoff. It never filters those rows out. This check
  applies to LO as well as FT (`launch_device.py:430-436`; `fl/finetune.py:131-137, 160-168`).
- A client without support is scored with the start state (`launch_device.py:437-445`).
- Per-client models are discarded (`fl/finetune.py:208`; `fl/local_only.py:151`).
- The TEST look re-executes the locked procedure on MAIN_TEST with the same SELECTION and the same support rows
  (`contracts/PREG1_AMENDMENT_06D.md:43-49`, disclosure 06D §9b).
- Every shard record already stores the per-client `final_digest` of each trajectory (`launch_device.py:447-458`).
- **No check found** that the TEST re-execution reproduces the VALIDATION digests. `grep final_digest` finds no consumer in
  `v4int`, `tools`, the contracts or the INT-07 handoff.

### 1.5 Contrasts that use these arms
- K5 R(PF/C_BIG), K6 R(FT_FA/C_BIG), K7 R(FT_C/C_BIG), K8 R(LO/C_BIG), K9 FT_C − C_SMALL, K10 FT_FA − FA and K11 FA − LO are
  confirmatory (`contracts/PREG1_AMENDMENT_06B.md:36-42`).
- K18 R(FT_C/C_FULL) and K19 R(LO/C_FULL) are tertiary confirmatory (`contracts/PREG1_AMENDMENT_06I.md:11-12`).
- **Harmed-client share** = the share of users whose per-user macro delta (mean over seeds) is < 0. It is computed for delta
  contrasts, with the improved share beside it (`tools/retention_report.py:1550-1581`).
- **Privacy ladder** (`contracts/PREG1_AMENDMENT_06B.md:51-60`):
  - L5 = FT_C, "nothing after the download".
  - L6 = "LO, RULE (own history)", "shared catalogue/θ0 only".
  - LAST is not in the ladder.
  - KNN_EMB is L1.

### 1.6 Disclosures already registered
- LO is "the no-collaboration control"; FT is one-shot; browser training is not available; the display catalogue leaks the
  recommendations (`contracts/PREG1_AMENDMENT_06I.md:67-70`; REALISM_AUDIT §C4-C5).
- UNSEEN_VAL is a selection panel, never a headline panel (06B §7e).

---------------------------------------------------------------------------------------------------------------------
## 2. State of the art (literature + real systems; verified unless marked)

**Evaluating on-device personalisation**
- Wang, Mathews, Kiddon, Eichner, Beaufays, Ramage, "Federated Evaluation of On-device Personalization", arXiv:1910.10252
  (2019), https://arxiv.org/abs/1910.10252.
  - Gboard, tens of millions of users. The global model is fine-tuned on each device's own data and evaluated on later local
    data.
  - Results are reported as the distribution of per-user deltas: "a significant fraction of users benefit".
  - The unit of analysis is the per-user change, including users who are harmed.
- Yu, Bagdasaryan, Shmatikov, "Salvaging Federated Learning by Local Adaptation", arXiv:2002.04758 (2020),
  https://arxiv.org/abs/2002.04758.
  - Many participants do better with a model trained locally than with the federated model, and DP or robust aggregation make
    this worse.
  - Fine-tuning, multi-task learning and distillation are the local-adaptation remedies.
  - This makes the **per-client local baseline an honest, tuned competitor**, not a floor.
- Matsuda, Sasaki, Xiao, Onizuka, "An Empirical Study of Personalized Federated Learning", arXiv:2206.13190 (2022),
  https://arxiv.org/abs/2206.13190. FedAvg + fine-tuning often beats dedicated personalised-FL methods, so FT_FA is the right
  strong personalisation baseline.

**Personalised FL: which parameters are local**
- Jiang, Konečný, Rush, Kannan, "Improving Federated Learning Personalization via Model Agnostic Meta Learning",
  arXiv:1909.12488 (2019), https://arxiv.org/abs/1909.12488.
- Fallah, Mokhtari, Ozdaglar, "Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning
  Approach", NeurIPS 33 (2020), https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html.
- These two train the global model to be *fine-tunable*. The project fine-tunes a model that was not trained for it (OPINION:
  acceptable, but it understates FT).
- Collins, Hassani, Mokhtari, Shakkottai, "Exploiting Shared Representations for Personalized Federated Learning", ICML 2021
  (PMLR 139), https://proceedings.mlr.press/v139/collins21a/collins21a.pdf. Shared representation, local low-dimensional head.
- Singhal, Sidahmed, Garrett, Wu, Rush, Prakash, "Federated Reconstruction: Partially Local Federated Learning", NeurIPS 2021,
  https://arxiv.org/abs/2102.03448.
  - User-specific parameters are **never aggregated and are re-fitted on the device from local data**, including for users who
    never took part in training.
  - Deployed in Gboard. This is the closest production analogue to PF's p_u.
- Li, Hu, Beirami, Smith, "Ditto: Fair and Robust Federated Learning Through Personalization", ICML 2021 (PMLR 139:6357-6368),
  https://proceedings.mlr.press/v139/li21h/li21h.pdf. A local model regularised toward the global one. This controls the harm
  that fine-tuning can cause.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022, https://arxiv.org/abs/2106.09685. The standard
  parameter-efficient alternative to full fine-tuning.

**On-device and device-cloud recommenders (real systems)**
- Gong et al., "EdgeRec: Recommender System on Edge in Mobile Taobao", CIKM 2020, https://arxiv.org/abs/2005.08416. The device
  runs **re-ranking / behaviour modelling for inference**; training stays in the cloud.
- Yao, Wang, Jia, Han, Zhou, Yang, "Device-Cloud Collaborative Learning for Recommendation", KDD 2021 (pp. 3865-3874),
  https://doi.org/10.1145/3447548.3467097.
  - On-device personalisation by a small generated "MetaPatch" on top of a frozen cloud model.
  - This is **parameter-efficient, not full fine-tuning**.
- Han, Ma, Mei, Liu, "DeepRec: On-device Deep Learning for Privacy-Preserving Sequential Recommendation in Mobile Commerce",
  WWW 2021, https://doi.org/10.1145/3442381.3449942.
  - A global model is trained on data that was allowed to be collected (pre-GDPR). The personal model is fine-tuned on the
    device on post-GDPR private data, with pruning and sparse embeddings.
  - This is exactly the realistic L5 design. It maps onto **FT_PRE**, not onto FT_C on MAIN.
- Ammad-ud-din et al., "Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System",
  arXiv:1901.09888 (2019). User factors stay local and item factors are federated. This is the proposal's "local MF" in its
  realistic form.
- Yin et al., "On-Device Recommender Systems: A Comprehensive Survey", arXiv:2401.11441 (2024; Data Science and Engineering
  2025), https://arxiv.org/abs/2401.11441. It separates on-device *inference* (compressed models) from on-device *training and
  update*.

**Baselines**
- Ludewig & Jannach, "Evaluation of session-based recommendation algorithms", UMUAI 28:331-390 (2018),
  https://doi.org/10.1007/s11257-018-9209-6. Nearest-neighbour methods are often as good as or better than neural models. The
  project's SKNN at 0.111 vs C_BIG 0.161 (REPORTED DL 132) fits this.

**Optimiser mechanics**
- Balles & Hennig, "Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients", ICML 2018,
  https://proceedings.mlr.press/v80/balles18a.html. Adam's direction is the gradient sign. With a fresh state and few steps,
  every coordinate with a non-negligible gradient moves by about ±lr.

**Production practice (what "device-only" means)**
- The device-side systems above (EdgeRec, DCCL, DeepRec, Gboard FedRecon) all ship a model or a table that was **trained
  elsewhere** and adapt a small part locally.
- None trains a large catalogue model from random initialisation on one user's data.
- A realistic "device-only" system is therefore one of:
  - heuristics over the user's own history plus public catalogue metadata (recently viewed, more from this brand or category);
  - a downloaded, cloud- or opt-in-trained model, which is L1/L5, not L6.
- (OPINION, consistent with the sources above.)

---------------------------------------------------------------------------------------------------------------------
## 3. Comparison

### 3.1 Strengths
1. **Causality is the best I have seen in a thesis codebase.**
   - A required cutoff; refuse, never filter; applied to LO as well as FT.
   - A conservative earliest-query cutoff for clients with several queries.
   - TRAIN_END cap on warm roles; own rows only (CLIENT_CONTRACT C5.1-C5.5).
   - Tested with negative controls (INT-04 / INT-05 / FIX-DELTA-01).
   - The REALISM_AUDIT D3 reached the same conclusion independently.
2. **The selection is honest.**
   - Selection moved off CALIB ⊂ MAIN onto a disjoint unseen-user panel before any read.
   - One selection per arm; O_EXCL; bound to the manifest's content sha; mechanical tie rule; refused on every report role.
   - This matches E11.9a's intent better than the original M04S substitute.
3. **Paired design.** The FT batch stream deliberately excludes the arm and the LR (`fl/finetune.py:18-19`), so FT_C, FT_FA
   and both LR fractions see the same batch order.
4. **Per-client ranking uses the frozen T3 kernel** and is assembled into one full-manifest artifact. Deviations are disclosed
   (6/1,251 rows at ranks > 20).
5. **The TEST look re-executes the procedure.**
   - This is sound in principle: a deterministic STRICT FP32 procedure with a per-(seed, client) RNG and support frozen at
     TRAIN_END yields the same weights.
   - It is also the only storage-feasible option (a 20.7 M-param model per user × 51k+ users).
6. **FT_FA on MAIN is the canonical "FL + local adaptation" evaluation** of Wang 2019 and Yu 2020: clients take part in FL,
   adapt on their own training data and are evaluated on later data.
7. **The harmed-client share is implemented** (`tools/retention_report.py:1569-1581`), with the improved share beside it.
8. **KNN_EMB and RULE exist and are cheap.** The kNN table is realistic in size (SMALL ≈ 40.6 MB FP32; INFERENCE).

### 3.2 Weaknesses and errors

**E1. FT_C on MAIN is mislabelled as L5, and its gain is not "local personalisation of private data"** (MEASURED + INFERENCE).
- The FT_C base is C_SMALL `best.pt`, trained centrally on the **MAIN users' raw TRAIN events** for ~6 EFE.
- On MAIN, FT_C then re-fits the **same rows** that the base already consumed about 6 times (support = MAIN TRAIN,
  `launch_device.py:23-24, 238-271`).
- So for MAIN users the privacy level is L1 (the raw events left the device once), not L5 "nothing after the download"
  (06B:58).
- K9 = FT_C − C_SMALL on MAIN therefore measures "extra per-user epochs on data the model has already fitted". It does not
  measure what DeepRec / FedRecon-style personalisation measures.
- The realistic L5 contrasts are:
  - FT_C − C_SMALL on **UNSEEN_HOLDOUT**: the base never saw these users. The panel is planned for seed 2026, but only as a
    DESCRIPTIVE panel.
  - **FT_PRE − PRE on MAIN**: the base was trained on the non-MAIN opt-in cohort. Registered DESCRIPTIVE only in 06K §2, and
    not implemented.
- No project document raises this. `grep` over the reviews, plans, contracts and DL finds no statement that FT_C on MAIN re-uses
  the base's training data. I rate it the most important framing error of this part.

**E2. Selection population ≠ application population for FT** (INFERENCE).
- UNSEEN_VAL users are *base-unseen*. MAIN users are *base-seen*.
- The best budget for re-fitting already-fitted data (MAIN) differs systematically from the best budget for adapting to new data
  (UNSEEN_VAL).
- The selected cell is applied to MAIN, where it may be too aggressive or too weak. The direction is unknown.
- This is structural: no base-seen, report-disjoint panel exists, because C_SMALL was trained on all of MAIN. It must therefore be
  disclosed.
- It does *not* affect FT_PRE, or FT_C on UNSEEN_HOLDOUT. There the selection and the application are both base-unseen, which is
  another reason to make those the realistic headline.

**E3. LO is closer to a strawman than a control as registered** (MEASURED schedule + INFERENCE on effect).
- LO is a 20.7 M-param model from random init, with a 158,486-way softmax, trained on 5–100 rows, at an **untuned** LR, with the
  central warmup-decay shape compressed onto 1–28 steps. For n ≤ 16, LO_P1 is a single step at 0.01 × peak (§1.2).
  - MAIN averages ≈ 19.7 TRAIN decisions per user (2,587,776 / 131,072; DL 50).
  - So a large share of clients get ≤ 2 steps at ≥ 0.1 × peak (INFERENCE).
- At init, output_embed ~ N(0, 0.02) and the query is post-LayerNorm (‖q‖ ≈ 8) (`code/v4models/sasrec_v2.py:255`). Logit noise is
  therefore ≈ 0.16 (INFERENCE).
- A handful of Adam steps of ≤ 7.1e-4 moves a target's logit by roughly 0.03–0.1 (INFERENCE). That is not enough to lift it into
  the top 20.
- Expected LO MRR@20 is therefore near chance (INFERENCE; not read). This drives:
  - K8 ≈ 0;
  - K11 = FA − LO ≈ FA, so the "collaboration value" is inflated;
  - K19 ≈ 0.
- Meanwhile **LAST, a zero-parameter device-only heuristic, already scores 0.0711 ≈ 44 % of C_BIG's 0.161** (REPORTED DL 132).
  LAST is absent from the ladder and from every K contrast.
- The owner asked for LO in this form (DL 83: "a SMALLER model of the SAME architecture fully on the device"). Keeping it is
  right, but the answer to "device-only retention" must not be K8 alone.
- The 06I §7c disclosure ("LO as the no-collaboration control") is necessary but not sufficient. The confirmatory K8/K11/K19
  still carry the LO number.

**E4. The ladder mislabels RULE and omits LAST** (MEASURED).
- RULE is listed at L6 "(own history)… shared catalogue/θ0 only" (06B:59).
- But `rule.py:9-12, 116-138` uses **MAIN-TRAIN pooled popularity** and a MAIN-TRAIN item → category/brand map built from all
  users' windows.
- That is a population aggregate. It is benign and publishable, but it is not "own history only". The KNN_EMB fallback is the
  same kind of aggregate (`knn_emb.py:136, 141`).
- The only strictly-L6 arms are LAST and LO (plus RULE if its popularity table is treated as a published catalogue statistic).

**E5. No parameter-efficient or partial on-device adaptation arm** (MEASURED absence).
- Production and the literature (DCCL MetaPatch, FedRecon, FedRep, LoRA) adapt a small part: a user vector, the bias, a head or an
  adapter.
- The project's only small-part arm is PF's p_u, and p_u:
  - is learned *inside FL*, at the un-multiplied SMALL peak × round factor (`launch_fl.py:656`), while the shared LR uses m* = 8
    (map §3.5);
  - gets only ≈ 3 visits;
  - is **0 for every unseen / TRUE_COLD user** (`fl/personal.py:3-4`; E11.6), so PF on those panels equals the shared model.
- Neither FedRecon-style reconstruction nor a "fit only p_u on a frozen C_SMALL / PRE" device arm exists.
- p_u is a rank-64 user factor against the shared item table (logit_k += w_k·p_u). In other words it is the proposal's **"local
  matrix factorisation" with shared item factors**, and it is almost free on a device: 64 floats plus 128 of optimiser state.
- This is the missing piece for proposal Step 3a ("local MF"). It is also the realistic answer to "is full fine-tuning
  necessary?".

**E6. Full fine-tuning with a fresh AdamW on a 158k-row softmax may add dense sign noise** (INFERENCE; not measured).
- On step 1, Adam moves every one of the ≈ 10.1 M output_embed coordinates by ≈ ±lr (Balles & Hennig 2018), because their
  softmax gradients (∝ p_k·q ≈ 1e-6 to 1e-5) are far above eps = 1e-8.
- The recentering post-step (`sasrec_v2.py:404-410`) removes only the common mode across rows.
- The FL ladder research reached the same diagnosis for FL (DL 203: "dense ±lr noise on the 158k output rows").
- For FT this is a plausible source of harm that neither the grid nor the selection can separate. The grid only lowers the
  LR; it has no SGD, frozen-table or 0-pass option.
- The harmed-client share will show the effect but not its cause.

**E7. The FT grid has no "don't fine-tune" cell** (MEASURED: `eval_base` unused).
- A deployment would skip FT if it hurt on validation. The registered grid forces fine-tuning.
- This is acceptable for K9/K10, which by definition measure FT's effect, but it should be disclosed.
- The base value on UNSEEN_VAL is not produced alongside the selection.

**E8. Harmed-client share has no noise floor** (MEASURED `retention_report.py:1569`: `mean(du < 0)`).
- Most users have 1–3 eval rows.
- Two arms with identical expected quality would still show a large "harmed" share from rank jitter. Wang 2019 read per-user
  deltas as distributions.
- A seed-vs-seed reference (for example C_SMALL s2026 vs s2027, or FA s2026 vs s2027, per user) is needed to interpret "x % of
  users are harmed".

**E9. The TEST re-execution identity is assumed, not checked** (MEASURED absence).
- The shard JSONs already carry `final_digest` per client and trajectory.
- One comparison of MAIN_TEST against MAIN digests for the overlapping clients would turn "re-executes the locked procedure" into
  a measured fact at zero GPU cost. It reads no metric.

**E10. Device-cost evidence for FT/LO is thin** (MEASURED).
- 04 §4 promises a "device-time proxy" and "local updates".
- Shard records carry the step count but no per-client wall time or peak memory (`launch_device.py:447-458`).
- INFERENCE for full FT of SMALL:
  - AdamW training state ≈ 4 × 82.8 MB ≈ 0.33 GB plus B16 × 158k logits and their gradients (≈ 20 MB);
  - storage = base plus a personal copy ≈ 166 MB FP32.
- This is fine on a laptop and heavy for a phone or a browser tab.
- Browser training is not available in ORT ≥ 1.20 (REALISM_AUDIT C3, already disclosed).

### 3.3 Unrealistic choices (OPINION)
- LO's recipe: the central LR shape compressed onto ≤ 28 steps, and no LR screen (E3).
- One-shot FT at TRAIN_END, never refreshed during the evaluation week. It is symmetric with the snapshot central models and
  disclosed (06I §7c). DeepRec and Gboard fine-tune continuously.
- FT of all 20.7 M parameters as the *only* on-device adaptation (E5).

### 3.4 Missing pieces against proposal Step 3
| Proposal Step 3 item | Project coverage | Gap |
|---|---|---|
| Local matrix factorisation | PF p_u, federated only (user factor × shared item table) | no device-side fit of p_u on a frozen downloaded model; unseen users get p_u = 0 (E5) |
| Pre-trained embeddings + local fine-tune | FT_C (base contains MAIN data, E1), FT_FA, FT_PRE (06K, descriptive, not implemented) | the privacy-consistent version (FT_PRE / UNSEEN_HOLDOUT) is not confirmatory |
| kNN on cached product embeddings | KNN_EMB_SMALL / _BIG | both tables are trained on MAIN raw data (L1). A PRE- or FA-trained table is the privacy-consistent cached-embedding arm. Also no cached item-to-item neighbour list (IKNN co-visitation) |
| Rule-based systems | RULE (+ LAST, POP as baselines) | RULE has no repeat/recency term and uses the coarse main category (vocab 16). LAST is not in the ladder (E4) |

---------------------------------------------------------------------------------------------------------------------
## 4. Recommendations

**Pre-registration status (MEASURED, DL 224-225, map §4):**
- No device value (SEL_FT_C, SEL_LO, LO/FT MAIN), no KNN_EMB and no RULE value has been read.
- The ANA freeze is pending. FA and PF values are unread.
- So every item below can still be registered legitimately, in a new amendment written before the reads it governs.
- Items that change the *running* SEL grids would invalidate SEL_FT_C / SEL_LO; I avoid proposing those.
- Costs are INFERENCE. The MASTER_PLAN §6 estimate is 4–10 h per device run per seed.

| ID | Item | Class | Action | Cost (INFERENCE) | Affects | Pre-registration needed / status |
|---|---|---|---|---|---|---|
| R1 | **Device-only answer = envelope, not LO.** (a) Add LAST to ladder L6. (b) Report a registered DESCRIPTIVE "device-only envelope" row {LO_selected, LAST, RULE}. (c) Add a secondary contrast K11b = FA − LAST ("collaboration value over the best own-history heuristic") beside K11. (d) Register the text: "K8/K19 = an untuned from-θ0 control, not the device-only ceiling", including the LO step arithmetic of §1.2. | MUST | ADD + CHANGE (text / analysis) | 0 GPU; ANA-03 delta + test ≈ 0.5 day | ANA script before freeze; no run | Yes, before the ANA freeze and before FA/LO values are read. **Status: permissible.** LAST's MAIN value is already read (DL 132), but it governs no rule. FA/LO are unread |
| R2 | **Relabel FT_C on MAIN and register the realistic L5 contrasts.** (a) Ladder: FT_C on MAIN = "L1 base (the user's raw events trained the base) + local FT"; L5 is valid only for base-unseen users. (b) Promote to secondary confirmatory: K9b = FT_C − C_SMALL on UNSEEN_HOLDOUT (seed 2026, panel already planned; C_SMALL reeval there ≈ 3 min). (c) K9c = FT_PRE − PRE on MAIN (2 seeds if time, else 2026), using the SELECTION_FT_C recipe as 06K already states. (d) Thesis text leads the L5 answer with K9b/K9c. | MUST | CHANGE (label) + ADD (contrasts; FT_PRE run) | FT_PRE ≈ one FT_C MAIN run per seed (4–10 h) + PRE reeval; small launcher change for a distinct row label FT_PRE (or FT_C rows relabelled by the index) + review | Running FT_C lanes unchanged; adds FT_PRE after PRE finishes | Yes, before any FT_C/PRE value is read. **Status: permissible** (PRE training; FT unread). FT_PRE must use the already-locked SELECTION_FT_C, so no new selection is needed |
| R3 | Disclose the selection/application mismatch (base-unseen UNSEEN_VAL vs base-seen MAIN) and the missing 0-pass cell. Add a DESCRIPTIVE C_SMALL / FA reeval on UNSEEN_VAL so the selected FT cell can be compared with "no FT" on the selection panel. | SHOULD | ADD (reeval) + DISCLOSE | 2 × 3 min reeval | none running | Register as descriptive before the SEL values are read. Status: permissible |
| R4 | **Parameter-efficient on-device adaptation arm** ("FT_PU"): fit only p_u (64-d query offset, the PF forward, start 0; optionally the output bias) on a frozen base (C_SMALL → FT_C_PU, PRE → FT_PRE_PU, FA → FA_PU). Use a persistent AdamW or SGD with its own small grid (LR {1e-3, 1e-2, 1e-1}, passes {1, 2, 4, 8}), selected on UNSEEN_VAL by the same rule. It also gives PF a FedRecon-style re-fit for unseen / TRUE_COLD users. | SHOULD | ADD | code ≈ 0.5–1 day (reuse `PersonalStore` / `pf_step_loss` in `launch_device`) + review; GPU ≈ eval-dominated, ≈ an LO run per population | new arm only | Yes, a new amendment with grid and rule before its first value. Status: permissible |
| R5 | Disclose that PF's personal part is structurally weak (personal LR = SMALL peak × round factor, not × m*; ≈ 3 visits; λ untuned; p_u = 0 off MAIN), so K5 ≈ K2 is expected and "PF − FA ≈ 0" does not mean personalisation is useless. | SHOULD | DISCLOSE | 0 | none | Text before the PF read. Status: permissible (PF unread) |
| R6 | Harmed-client share with a **noise floor**: per-user delta between the two seeds of the same arm (C_SMALL, FA, FT_C) as a reference distribution. Also report the share of "unchanged / both zero" users and the harmed share per n_support bin (06A §2 bins). | SHOULD | ADD (analysis) | 0 GPU; ANA delta | ANA script | Before the ANA freeze. Status: permissible |
| R7 | **TEST re-execution identity check:** for clients present in both runs, MAIN_TEST `final_digest` == MAIN `final_digest` per trajectory (and UNSEEN_HOLDOUT_TEST vs UNSEEN_HOLDOUT). A mismatch means STOP and report. Reads digests only, no metrics. | SHOULD | ADD (integrity check) | ≈ 50 lines + test | TEST look | Register before the TEST look (ALL_METHODS_LOCK). Status: permissible |
| R8 | Correct the ladder text for RULE (and the KNN_EMB fallback): "own history + a published MAIN-TRAIN popularity / category aggregate". KNN_EMB stays L1 (a cloud-trained table). | SHOULD | CHANGE (text) | 0 | none | Text; before the RULE/KNN read |
| R9 | **FT harm diagnostic** (E6): one DESCRIPTIVE FT_C variant on UNSEEN_VAL / UNSEEN_HOLDOUT with the item tables frozen (encoder only ≈ 0.4 M params, the S8 code path of 06K §3) or with SGD, at the selected cell's passes. It separates "fine-tuning hurts" from "fresh AdamW on 10 M softmax weights hurts". | NICE | ADD | code small if FL-12b frozen tables are reusable; 1 run ≈ 1–3 h | new descriptive arm | Register as descriptive before its value. Status: permissible |
| R10 | Privacy-consistent cached-embedding arms: KNN_EMB_PRE (θ_pre table, 1.2 M non-MAIN users) and KNN_EMB_FA (FA table). Both are CPU and evaluation-only. | NICE | ADD | CPU hours; no code change (`--item-table` path; needs a table-role label) | none | Descriptive, before read. Status: permissible |
| R11 | RULE_PLUS: rank the items in the user's own history by recency first (LAST), then fill with RULE's category/brand affinity; optionally use the fine `category_tokens`. This is the production "recently viewed + more from brands you like" rule. | NICE | ADD | CPU; ≈ 30 lines + test | none | Descriptive, before the RULE read. Status: permissible |
| R12 | A stronger learned L6 arm: LO_TUNED with a constant LR grid {1×, 10×, 30×} peak × passes {4, 8, 16}, selected on UNSEEN_VAL. As an alternative, a tiny own-history transition model. It shows whether any learned device-only model beats LAST. | NICE | ADD | ≈ 3 LO-sized runs on UNSEEN_VAL + 1–2 MAIN | new arm | Register before its value. Status: permissible |
| R13 | Device-cost evidence for FT / LO / FT_PU: per-client steps (already recorded), wall time and peak memory on the A40 as a proxy, plus one CPU single-thread timing on ≈ 200 clients (the proxy promised in 04 §4). | SHOULD | ADD (measurement) | a small launcher field or a separate bench; ≈ 1 h CPU | none | No rule is governed; register as descriptive |
| R14 | Keep: causality (refuse, not filter), UNSEEN_VAL selection panel, O_EXCL + manifest-sha binding, per-client T3 ranking, paired batch streams, TEST re-execution design. | — | KEEP | 0 | — | — |
| R15 | Disclose: one-shot FT at TRAIN_END (at TEST, support is one week staler than history; symmetric with the central snapshot); the tiny-client LR compression of LO; storage of a personal full copy (≈ 83 MB FP32 each). | DISCLOSE-ONLY | DISCLOSE | 0 | none | Text |

**Minimum set to make this part's answer convincing:** R1 + R2 (both zero or low GPU), with R5/R8 as text.

---------------------------------------------------------------------------------------------------------------------
## 5. Score: **6 / 10**

- **Engineering and inferential hygiene: 9/10.**
  - Strict support causality that refuses rather than filters; a disjoint, sha-bound, one-shot selection panel.
  - Deterministic per-client procedures; the frozen T3 ranking path; paired batch streams; a harmed-client share already in the
    analysis.
  - This is better than most published FL-personalisation evaluations.
- **Scientific design of the on-device answer: weaker, on three counts.**
  - The headline "device-only" contrasts (K8, K11, K19) rest on LO. LO is an untuned from-random 20.7 M-param model whose
    tiny-client schedule reduces to one or two meaningful steps. A zero-parameter own-history heuristic (LAST, ≈ 44 % of C_BIG,
    REPORTED) is left out of the ladder and the contrasts. The collaboration value will therefore be overstated.
  - The confirmatory local-fine-tuning gain (K7, K9, K18) is measured on users whose raw data trained the base. That is neither
    the L5 privacy level the ladder claims nor the DeepRec / FedRecon setting the literature means by on-device personalisation.
    The realistic variants (FT_C on UNSEEN_HOLDOUT, FT_PRE on MAIN) exist or are cheap, but are only descriptive or not yet
    implemented.
  - There is no parameter-efficient or reconstruction-style adaptation, which is what production device-cloud recommenders
    actually do.
- All of this is fixable before any governed value is read, mostly at zero or low GPU cost, which is why the score is not lower.

---------------------------------------------------------------------------------------------------------------------
## 6. UNVERIFIED items / open questions
1. **LO's expected quality near chance** (E3) is INFERENCE from the init scales and the schedule. It is not measured, and must not
   be quoted as a result.
2. **Dense Adam sign noise as a cause of FT harm** (E6) is INFERENCE. The recentering removes only the common mode. R9 would test
   it.
3. **Memory / storage figures for full FT** (§3.2 E10) are INFERENCE from the parameter count (20,710,048, REALISM_AUDIT C1).
4. **UNSEEN_VAL composition.** UNEW_VAL_SUPPORT_32768 is assumed to hold TRAIN-period support rows of users with 5–100 TRAIN
   examples. I did not open the manifest. The band positions in DL 40 vs the map's "U_NEW bands [1,348,682, …)" should be
   reconciled by the data expert. The disjointness of UNSEEN_VAL from C_FULL / PRE is asserted in 06A §1 and 06K §1, not checked
   here.
5. **Whether FT_PRE can run as `--arm FT_C --base <PRE ckpt>` without a mislabelled row** was not checked. The rows would say
   FT_C; the ANA index needs a distinct label (R2).
6. **Share of MAIN clients with ≤ 16 support rows.** The mean is ≈ 19.7 (DL 50). The median was not read (census files not
   opened).
7. **Brave or other browser on-device ranking as an analogue** stays UNVERIFIED (as in REALISM_AUDIT §I). I did not cite it.
8. **Open question for the lead:** should the owner be asked whether "device-only" in the thesis headline means strict L6
   (own history only) or "device inference with a downloaded model" (L1/L5)? R1/R2 give both. The owner's DL 83 wording
   ("smaller model of the same architecture fully on the device") fits FT_PRE / C_SMALL-on-device better than LO.

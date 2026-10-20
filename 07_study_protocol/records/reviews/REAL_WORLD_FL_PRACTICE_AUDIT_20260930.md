# REAL-WORLD FL PRACTICE AUDIT: what deployed systems actually do, where our plan stands, what to register next (2026-09-30)

- **Author:** Claude Opus 5.5, acting as a senior researcher in production federated learning and on-device recommendation. This was a
  READ-ONLY research and audit session. This file is the only artefact written. No server access, no code, contract or register edits,
  no Codex, no git.
- **Values used:** only CALIB screen values that the ladder register already records (S0 0.023473, S1b ×8 0.028796, S2 FedAdam η_s 0.01
  ×1 0.034247 and the per-mark curves, READ 2026-09-30 about 08:10Z, DL 249–250). **No MAIN, RETRO_TEST, PRE, S3, τ-edge or panel
  value was read.** The central CALIB figure (≈ 0.11–0.12) is INFERENCE, as the register says.
- **Project files read:** FL_STRONGEST_LADDER_RESEARCH_20260929 (all), REALISM_AUDIT_20260929 (all), registers/FL_IMPROVEMENT_LADDER.md,
  amendments 06G, 06I, 06J, 06K, 06L, 06L_C, DECISION_LOG rows 230–251, MULTI_EXPERT_REVIEW_20260929/07_federated.md §2–§4 and
  02_question_framing.md (hybrid notes), code `fl/server.py` (FedAdam, L150–190), `fl/schedule.py` (KNOTS L38), `fl/runtime.py`
  (FL_CONST L108–113, L196–198), `fl/client.py` (VisitRecord.loss_sum).
- **Tags:** MEASURED (read today from a primary source or code, with a location), REPORTED (a project document or a source I did not
  re-open today), INFERENCE (my estimate or arithmetic), PLAN, UNVERIFIED. Every cost is INFERENCE. The owner gave the cost basis: a CALIB
  screen run takes ≈ 5–7 h wall on a shared A40. One FL process uses about 20 % of a card, so a run is ≈ 1–1.5 GPU-h. A MAIN FL run is
  ≈ 10.7 process-h unshared (REPORTED, MASTER_PLAN V5.1) and longer when the card is shared.

---------------------------------------------------------------------------------------------------------------------
## 0. Answer to the owner's question in one paragraph

Real systems almost never start federated training from a random model **when the per-user data are small**. Gboard's first FL language
model did start from random weights, but it had 1.5 M clients, ≈ 400 sentences per client per epoch and 3,000 rounds. Every later Gboard
model starts from a model pretrained on public data. Firefox started from its existing hand-tuned weights, explicitly so that users would
not see a bad model while training ran. Apple tunes a handful of parameters of an on-device algorithm that already works. **Prediction
runs on the device in all of these systems.** The big or general part is trained centrally on data the company may legitimately hold
(public, curated or opted-in data). Federated training and on-device personalisation add only the part that needs private data: a head,
a few parameters or a fine-tune. Our plan already registers this "deployment branch": PRE on 1.2 M opted-in non-MAIN users → FA_WARM
(S7) → frozen item tables (S8) → FT_PRE, with contrasts K20, K20b, K20c, K21 and K9c (06K, 06L_C C7/C8). However, **it runs last, it
waits for S3 and S4 to finish, and its registered screen varies the wrong knob** (the client multiplier m instead of the FedAdam server
step). The largest realistic improvement available now is to screen the warm start early, with a server-step axis. The on-device
personalisation of the warm FL model (the full production stack) should be registered as well.

---------------------------------------------------------------------------------------------------------------------
## 1. How deployed systems do it

### 1.1 System-by-system evidence

| # | System (source opened today) | Initialisation | Where inference runs | What is trained federatedly / on device | Rounds, cohort, optimisers | Tag |
|---|---|---|---|---|---|---|
| P1 | **Gboard next-word prediction, first FL model** (Hard et al. 2018, [arXiv 1811.03604](https://arxiv.org/pdf/1811.03604)) | "starting from random weight initializations" (§5) for both the server and the FL model. The FL-process section says the initial model has "either been randomly initialized or pre-trained on proxy data" (§4) | On device: "On-device inference is supported by TensorFlow Lite" (§3). The shipped model is 1.4 M params, "1.4 megabytes" after quantisation, and two thirds of it is the embedding | The full model (tied input/output embeddings, 10k vocabulary) | "3000 training rounds … 600 million sentences … 1.5 million clients … 4-5 days"; "Between 100 and 500 client updates … each round"; server "Momentum optimizer, using Nesterov accelerated gradient … 0.9, and a server learning rate of 1.0"; client SGD; ≈ 400 sentences per client per epoch; eligible devices must be charging, idle, unmetered and have ≥ 2 GB RAM. The FL model matched the server model on server logs (top-1 recall 16.4 % vs 16.5 %) and beat it on live traffic (impression recall 5.82 % vs 5.76 %) | MEASURED (PDF text) |
| P2 | **Gboard LMs with DP, production** (Xu et al. 2023, [arXiv 2305.18465](https://arxiv.org/pdf/2305.18465)) | **Public pretraining on multilingual C4 for every model**. "pretraining can reduce ∼1000 rounds to reach a given utility threshold". They also note "the accuracy may decrease in the first a few rounds of fine-tuning", "diminishing marginal returns" of more pretraining, and fine-tuning is "challenging … when the word embeddings are shared", so they use **separate input and output embeddings** | On device | The full model: 2.4 M / 4.4 M / 6.4 M params for 10k / 20k / 30k vocabularies | DP-FTRL; server momentum 0.9 (Alg. 1); report goal 6,500; "several thousands of rounds"; populations 0.8–16.6 M | MEASURED (PDF text) |
| P3 | **Gboard privacy white paper** (Zhang et al. 2023, [arXiv 2306.14793](https://arxiv.org/pdf/2306.14793)) | "model pre-training on public C4 corpus … combined with training configuration tuning, and achieved almost neutral utility change" compared with the non-DP model | On device ("rolled out for use in inference") | The full model | Each round samples "from hundreds to thousands" of available devices; clients run SGD. Improving pretraining practice moved ρ-zCDP from 1.31 to 0.25 at the same utility | MEASURED (PDF text) |
| P4 | **Gboard query suggestions** (Yang et al. 2018, [arXiv 1812.02903](https://arxiv.org/pdf/1812.02903)) | The candidate generator is "a traditionally server-side trained baseline model" | "Inference and training are on-device" (Fig. 1). The baseline "generates candidates" on device from "an on-device subset" of the knowledge graph | **Only a small logistic-regression "triggering model"** that "decides whether to show the candidate". This is a two-stage hybrid: a big, centrally trained model plus a tiny FL head | FedAvg | MEASURED (PDF text) |
| P5 | **Gboard on-device personalisation** (Wang et al. 2019, [arXiv 1910.10252](https://arxiv.org/abs/1910.10252)) | Starts from the FL-trained global model | On device | Local fine-tuning of the global model per user, evaluated federatedly; "a significant fraction of users benefit" (tens of millions of users) | — | MEASURED (abstract) |
| P6 | **Federated Reconstruction** (Singhal et al. NeurIPS 2021, [arXiv 2102.03448](https://arxiv.org/abs/2102.03448)) | — | On device | User-specific parameters stay local and are reconstructed per visit (clients stay stateless). Global parameters are federated. "deployment of this approach at scale for federated collaborative filtering in a mobile keyboard application" | Stateless clients | MEASURED (abstract) |
| P7 | **Apple: federated evaluation and tuning** (Paulik et al. 2021, [arXiv 2102.08503](https://arxiv.org/pdf/2102.08503)) | News uses an existing on-device personalisation algorithm. FL is used for keyword-trigger and keyboard models (§4) | On device | **News: only a few global parameters of the on-device algorithm (e.g. a time-decay half-life) are tuned federatedly** by randomised grid search on device, with server-side smoothing. Result: "+1.98 % … daily article views" (run 1) and +1.87 % views, +0.90 % time spent (run 2). ASR: per-device personalised language models | Federated tuning iterations over thousands of devices | MEASURED (PDF text §4–5) |
| P8 | **Firefox URL-bar ranking** (Hartmann et al. 2019, [arXiv 1911.11807](https://arxiv.org/pdf/1911.11807)) | **From the current hand-chosen weights**: "Starting off from a decent initial model also helps to ensure that users do not interact with a bad model during the first few iterations of training"; "If some data is already available on the server, it can be used to pretrain the model" | In the browser | 22 ranking weights; numerical gradients | 137 iterations of 30 min; 360,518 users sent updates; the server optimiser is a Rprop variant (adaptive, per weight, bounded steps). Result: users type "over half a character less" | MEASURED (PDF text) |
| P9 | **Chrome Topics** ([privacysandbox.google.com](https://privacysandbox.google.com/private-advertising/topics/web/)) | Centrally trained on a curated list of 50,000 sites | On device: "downloads a TensorFlow Lite file representing the model so it can be used locally" | Nothing is trained on device; the model ships with browser updates | — | MEASURED (page). Current status of the API: contested (the multi-expert review says retired in 2025; the page still documents it) → UNVERIFIED |
| P10 | **Taobao EdgeRec** (Gong et al. CIKM 2020, [arXiv 2005.08416](https://arxiv.org/abs/2005.08416)) | Cloud models | **Hybrid:** the cloud recommender returns 100 candidates and the device re-ranks them with real-time behaviour features | On-device re-ranking model (trained in the cloud per the review's reading) | Reported "9 % lift in IPV and 11 % lift in GMV" | MEASURED (abstract/comments); training location REPORTED (02_question_framing) |
| P11 | **DCCL device-cloud** (Yao et al. KDD 2021, [arXiv 2104.06624](https://arxiv.org/abs/2104.06624)) | A central cloud model | On device | A small on-device "MetaPatch" per user over the frozen cloud model, distilled back into the cloud model; helps long-tail users | — | MEASURED (abstract) |
| P12 | **Billion-scale submodel FL, Alibaba** (Niu et al. MobiCom 2020; [dblp](https://dblp.org/rec/conf/mobicom/NiuWTHJLWC20.html)) | — | On device | Clients download and upload only the embedding rows they touch (the submodel), with a private set union and secure aggregation (Taobao CTR, DIN) | — | REPORTED (search summary + dblp record) |
| P13 | **FedSubAvg** (Ding et al. NeurIPS 2022, [arXiv 2109.07704](https://arxiv.org/abs/2109.07704)) | — | — | Scale each embedding row's aggregated update by the share of clients that touched it: FedAvg's zero updates are "inaccurately aggregated"; FedSubAvg "significantly outperforms FedAvg" on industrial data | — | MEASURED (abstract) |
| P14 | **Pretraining in FL** (Nguyen et al. ICLR 2023, [arXiv 2206.15387](https://arxiv.org/abs/2206.15387)) | Pretrained vs random | — | — | Pretraining gives "more accurate models (up to 40 %)" and "reduces the effect of both data and system heterogeneity" | MEASURED (abstract) |
| P15 | **Public data before private cross-device FL** (Wang, …, McMahan, Xu, Findings NAACL 2024, [arXiv 2305.12132](https://arxiv.org/abs/2305.12132)) | Pretraining and distillation on public data close to the private distribution, then FL | On device (small models) | — | — | MEASURED (abstract) |
| P16 | **Client LR matters in FedAvg** (Charles & Konečný 2020, [arXiv 2007.00878](https://arxiv.org/abs/2007.00878)) | — | — | — | The client LR "controls the condition number of that surrogate loss" and the distance between surrogate and true minimisers; they propose automatic client-LR decay | MEASURED (abstract) |
| P17 | **Meta PAPAYA** (Huba et al. MLSys 2022, [arXiv 2111.04877](https://arxiv.org/pdf/2111.04877)) | — | On device | — | A production asynchronous FL system for millions of devices; asynchronous training converged ≈ 5× faster than synchronous | REPORTED (search summary; PDF not opened) |
| P18 | **Adaptive server optimisers** (Reddi et al. ICLR 2021) | — | — | — | Client SGD + server Adam/Yogi. On sparse Stack Overflow NWP, FedAdam ≈ 25.7 vs FedAvg ≈ 20.5; optimum η_s ≈ 1e-2, τ ≈ 1e-5; staircase client-LR decay gives modest gains (App. E.6) | REPORTED (FL_STRONGEST_LADDER §1–2, from the PDF read on 2026-09-29) |

### 1.2 Practice consensus, by question

**Initialisation.** MEASURED from P1–P3, P8, P14; INFERENCE for the rule.
- Random init works in production only at huge scale. Gboard 2018 had 1.5 M clients, ≈ 400 examples per client per epoch and 3,000
  rounds.
- Every later production system starts from something good:
  - public-corpus pretraining (Gboard since 2023);
  - existing heuristic weights (Firefox);
  - an existing on-device algorithm (Apple News);
  - a centrally trained model (Topics, query suggestions, EdgeRec, DCCL).
- There is also a product reason, beyond accuracy: **while FL trains, users are served the current model** (P8). A random model would
  serve useless recommendations for days.
- Our setting sits at the opposite extreme from Gboard 2018: 19.7 examples per client, ≈ 3 visits, and 158,486-row tables. Random init
  is therefore the least realistic starting point (INFERENCE).

**Where inference runs.**
- On device in every FL/personalisation system above (P1–P9).
- Large-catalogue e-commerce uses a **hybrid**: cloud retrieval of N ≈ 100 candidates, then on-device re-ranking (P10), or a cloud model
  plus an on-device patch (P11).
- The hybrid exists because catalogues have ~10⁸ items and because of real-time features. With 158,486 items, full-catalogue scoring on
  device is ≈ 10 M MAC per query (REPORTED, REALISM_AUDIT C1), so our on-device full ranking is realistic for Edge.
- The hybrid's privacy cost: the candidate request reveals context (REPORTED, 02_question_framing R13).

**What is federated.**
- The whole model only when it is small: Gboard ships 1.4–6.4 M params.
- Otherwise a small head (P4), a few parameters (P7, P8), local user parameters (P6), a patch (P11), or touched rows only (P12, which
  needs SecAgg to avoid the item-set leak).
- Nobody federates a 20 M-parameter model whose 98 % is two item tables on 20-example clients (INFERENCE from the list above).

**Huge item/embedding tables.** In production and the literature (MEASURED P2, P6, P12, P13):
- separate input and output embeddings when warm-starting (P2);
- submodel / touched-row updates with private set union (P12);
- per-row sparse-aware aggregation (P13);
- local user embeddings (P6);
- a frozen or server-side item side (P4-style two-stage; our S8).

**Personalisation.** Local fine-tuning of the global model (P5), local parameters (P6), federated tuning of personalisation
hyper-parameters (P7), on-device patches (P11). It is always on top of a good global model.

**Rounds and cohorts.**
- Rounds: thousands (P1: 3,000; P2: "several thousands").
- Cohort per round: 100–500 (P1), "hundreds to thousands" (P3), report goal 6,500 under DP (P2).
- Eligibility: charging, idle, unmetered (P1).
- Firefox: 137 iterations of 30 min (P8).

**LR schedules and optimisers.**
- Clients run SGD (P1, P3). The server uses momentum 0.9 at LR 1 (P1, P2) or adaptive per-coordinate optimisers (P8's Rprop; P18's
  FedAdam).
- Client LR is constant or has a staircase decay (P18). The client LR changes the objective being optimised (P16).
- No production system maps a central warm-up/decay schedule onto rounds (INFERENCE: none of P1–P8 describes one).

---------------------------------------------------------------------------------------------------------------------
## 2. Gap table: real-world practice → our plan

Status legend: REGISTERED+DONE, RUNNING, REGISTERED (waiting), PARTIAL, MISSING, NOT ADOPTED (a decision was made), DISCLOSE-ONLY.

| # | Real-world practice | Our plan | Status | Covering file |
|---|---|---|---|---|
| G1 | Start FL from a centrally pretrained model on permitted data | PRE = SMALL recipe on 1,217,610 non-MAIN users, θ_pre = endpoint_6.0; FA_WARM = S7 | PRE DONE (value unread); S7 REGISTERED, **waits for FL_TUNING_SELECTION (S4)** | 06K §1–2; 06L_C C7 (secondary confirmatory K20b, K20c, K21, K9c) |
| G2 | The warm-start screen tunes the step that fine-tuning needs (smaller server step) | S7 screen m ∈ {m*, m*/4} only. Under FedAdam the client multiplier is partly normalised away (§4.2), so η_s is the effective step, and η_s is not screened | **GAP** | 06K §2 (grid). Proposal R1 |
| G3 | Serve a good model during rollout (P8) | Cold FA is K2's algorithmic contrast (random θ0, same as central). The deployment rows lead the Edge answer (06L_C C7) | PARTIAL: registered framing; the "quality during rollout" view is not analysed | 06L_C C7; proposal R6 |
| G4 | Inference on device with a centrally trained model and no FL (Topics / query-suggestion pattern) | PRE-alone on MAIN (L1 on opted-in data), FT_PRE | REGISTERED (FT_PRE code done, smoke run mandatory, DL 241) | 06K §2; 06L_C C1, C7 |
| G5 | Personalise the global model locally (P5, P6, P11) | FT_C (on C_SMALL), FT_FA (on cold FA), FT_PRE (on θ_pre), PF | REGISTERED. **FT on FA_WARM (the full production stack) is MISSING** | 06, 06A §1, 06K §2, 06L A5; proposal R3 |
| G6 | Small federated part, big part frozen or central (P4, P11) | S8 FA_WARM_FROZEN: encoder only (265,162 params, ≈ 1 MB per direction), 2 seeds unconditional; DP8_FROZEN SHOULD-high | REGISTERED (waits for S7) | 06K §3; 06L_C C8, C16 |
| G7 | Separate input/output embeddings when warm-starting (P2) | SASRec untied tables (`sasrec_v2.py:202`) | ALREADY TRUE | ARCHITECTURE_LOCK |
| G8 | Server momentum / adaptive server optimiser | S2 selected FedAdam η_s 0.01 τ 1e-3 ×1 (0.034247); FedAvgM screened | DONE (τ 1e-5 edge pending → FL_TUNING_S2.json) | 06G, 06J §1, 06L A1, 06L_D |
| G9 | Client SGD, constant client LR | S3: FL_CONST × {AdamW, SGD-m 0.01 / 0.1 / 1.0}, A2 cells | RUNNING (A2 done, W1/W2 running; unread) | 06I §3, 06J §2, 06L A2, 06L_D D2 |
| G10 | Local computation tuning | S4: passes 1 / 4, batch 8 | REGISTERED | 06J §3 |
| G11 | Thousands of rounds, cohorts of hundreds or more | 1,707 CALIB / 6,827 MAIN rounds of 64 (≈ 57 survivors) at an equal-exposure budget; FA_X2 descriptive | REGISTERED (framing: matched-exposure retention) | 06I §4, §6; 06L_C C6 |
| G12 | Sparse-aware aggregation for embedding tables (P12, P13) | FedSubAvg-style | **NOT ADOPTED** (disclosed; possible NICE rung after S4) | 06L A6 |
| G13 | Touched-row upload + SecAgg (P12) | Not simulated; the item-set leak is disclosed | DISCLOSE-ONLY | 06I §7a; 06L_C C1 |
| G14 | DP with pretraining and large report goals (P2, P3) | DP8 / DP1 at MAIN (m 1,024); DP8_FULL at N 1.35 M (m 8,192); DP8_FROZEN (warm + frozen) | REGISTERED (the DP arms wait for the FL selection; DP8_FULL "if time") | 06D §3, 06I §5, 06K §3, 06L_C C16 |
| G15 | Hybrid: cloud candidates + on-device re-rank (P10) | Not modelled (full on-device ranking is feasible at 158k items) | DISCLOSE-ONLY (R13) | MULTI_EXPERT_REVIEW 02 R13 |
| G16 | Federated evaluation to choose the checkpoint to deploy | PRACTICAL_BEST (best mark on validation) and CONTROLLED_BUDGET (6.0) | ALREADY TRUE (equivalent) | EVALUATION_CONTRACT_V4 L845 |
| G17 | Availability-biased participation (charging/idle/Wi-Fi, diurnal) | Uniform sweep + 10 % uniform drop-out | DISCLOSE-ONLY (NICE sensitivity) | 06I §7c; 07_federated F12 |
| G18 | A late-phase schedule suited to the server optimiser (annealing applied where it acts) | Central LR shape on the client LR only; η_s constant (`fl/server.py:167–177`; `fl/schedule.py:38`) | **GAP** if S3 does not fix the decline (§4) | proposal R2 |
| G19 | A central small model on all users (C_SMALL_FULL) | — | NOT ADOPTED by the owner (respected here) | 06L_C (owner decisions) |

---------------------------------------------------------------------------------------------------------------------
## 3. Ranked improvement proposals (realistic for Edge, implementable here)

Constraints respected in every proposal:
- no C_SMALL_FULL;
- TEST stays sealed;
- every rule is registered before its governed read;
- no rule changes after its value is read;
- no early MAIN read;
- no warm start from C_FULL or C_BIG (06K §2).

"New registration" means an amendment (06M suggested) written BEFORE the first value it governs.

| Rank | Proposal | Why (evidence) | Expected effect (INFERENCE) | Cost | Registration |
|---|---|---|---|---|---|
| **R1** | **Screen the warm start NOW, with a server-step axis.** Register S7-EARLY: FA from θ_pre on CALIB at the current S2 incumbent (FedAdam τ 1e-3, ×1; LR shape = the central shape, as S2 ran), η_s ∈ {0.01, 0.001} × m ∈ {1, 1/4}, plus one plain-FedAvg warm cell (server LR 1, m = 1/4) = **5 runs**. They are labelled DESCRIPTIVE, feed no selection, and are read after all 5 finish. **Amend 06K §2 for the confirmatory S7:** at the S4 winner, grid = {m*, m*/4} × {η_s*, η_s*/10} when the selected server optimiser is FedAdam, or {m*, m*/4} × {η*, η*/4} for FedAvgM/FedAvg. Same rule, ties → (m*, η_s*) | Production fine-tunes from pretrained models with small steps; Xu 2023 saw accuracy "decrease in the first a few rounds of fine-tuning". Under FedAdam, η_s sets the per-coordinate step (≈ 0.007 per round on consensus coordinates at the peak, §4.2), about a third of the 0.02 init std per round. That can erase θ_pre's table structure within tens of rounds. The registered m-only grid may not reach a gentle step at all | S7 becomes the strongest realistic FL row: PRE-level quality (INFERENCE ≈ 0.10–0.13, ladder research §2.6) instead of ≈ 0.034. The risk it removes: a registered S7 screen that only shows FL *degrading* θ_pre because the server step is too large | 5 runs ≈ 5–7 h wall in parallel, ≈ 6–8 GPU-h; the confirmatory S7 grid grows from 2 to 4 runs (+2 × 5–7 h). Code: none (FL-12a exists; η_s is a launch flag) | **Yes**: 06M §1 before any S7 or S7-EARLY value. Do not read PRE on CALIB or MAIN directly (§5 note) |
| **R2** | **Register the S3-failure contingency now**, before the S3 read: the "late-phase" rung S3b (details §4.4). Trigger = decline index D ≥ 0.10 on the S3 winner's curve. Candidates: (a) SERVER_ANNEAL (η_s × the same round factor as the client LR); (b) SERVER_EMA (evaluate an EMA of the server weights, β 0.99 per round) | The FedAdam curve loses 27 % after mark 3.0. PRACTICAL_BEST hides this on CALIB. But CONTROLLED_BUDGET (6.0) is reported, the DP contrasts K12/K13 are CONTROLLED_BUDGET (06D §3), and DP runs have no mark selection, so a late decline goes straight into the DP and endpoint rows | If H2 (§4) holds: CONTROLLED_BUDGET +20–35 % relative and PRACTICAL_BEST +0–10 %. If pure overfitting: EMA +0–5 %, anneal ≈ 0 | 2 runs ≈ 5–7 h wall, ≈ 2–3 GPU-h. Code: η_s round factor ≈ 20 lines + tests; EMA ≈ 60 lines (server-side buffer, checkpointed, evaluated at marks) | **Yes**: 06M §2 before the S3 selection is read |
| **R3** | **FT_FA_WARM (and FT_FA_WARM_FROZEN): on-device personalisation of the warm FL model**, the full production stack (pretrain on permitted data → FL on private users → local fine-tune → on-device inference). Same FT_C procedure and SELECTION_FT_C cell (FT_C_f0.3_P4), seeds 2026/2027, DESCRIPTIVE, with the paired contrast FT_FA_WARM − FA_WARM | This is what Gboard (P5, P6) and DCCL (P11) do. Currently FT exists only on C_SMALL, cold FA and θ_pre (G5) | FT is usually positive for heavy users. The harmed-client share is already reported (06L_C C13–15). It gives the thesis its "best realistic Edge system" row | Device lanes; INFERENCE ≈ the FT_FA cost per seed (a few GPU-h); code ≈ 0 (launch_device already takes `--init-from` for FT_PRE; add a registered FA_WARM checkpoint as a start kind, ≈ 0.5 day incl. tests) | **Yes**: 06M §3 before any FA_WARM value is read |
| **R4** | **Run S8 alongside S7, not after it**, and give S8 the same server-step axis: S8 grid {m*, 4m*} × {η_s*, η_s*/10} | S8 is the most production-like configuration: the big item side is centrally trained and frozen; FL covers a ≈ 1 MB encoder (P4, P11 pattern). It is also the only phone-class payload and the one with ≈ 7× lower DP noise | S8 ≈ S7 minus a little quality; bytes ÷ 100; DP8_FROZEN far better than DP8 (INFERENCE) | +2 runs on the S8 screen ≈ 5–7 h wall; MAIN as registered (2 seeds) | **Yes** for the η_s axis (06M §4). Scheduling in parallel needs no registration |
| **R5** | **Read the zero-GPU diagnostics that separate overfitting from the LR-shape × FedAdam interaction** (§4.3): per-mark curves of every S2 cell already read (η_s 0.003 / 0.001, FedAdam ×8, FedAvgM ×0.5 / ×2 / ×8), later τ 1e-5 once the S2 selection is written; server-step norms ‖θ_mark(i+1) − θ_mark(i)‖ from saved checkpoints; `embedding_stats` on θ0 / S0 / S2 best / endpoint (07_federated F5). Persist `VisitRecord.loss_sum` (`fl/client.py:107`, `fl/server.py:382`) as a per-round mean client training loss in future runs | This decides R2 and informs S7's step. The loss signal already exists in memory but is not written to disk (INFERENCE from `fl/runtime.py`, no persistence found) | Explanation only; selects nothing | CPU < 1 h; loss logging ≈ 20 lines + digest note (numerics unchanged) | DL row "descriptive, selects nothing". The loss logging is code only, no rule |
| **R6** | **"Quality during rollout" analysis**: for FA, FA_WARM and S8, the area under the per-mark MAIN curve and the mark at which each first reaches PRE-alone / LAST. It uses marks the runs already produce | Production serves the current model during training (P8). The thesis can then say how long users would see a worse model under cold vs warm FL | Framing: the realistic reason for the warm start, beyond accuracy | Analysis-script change only (ANA) | **Yes**: into the ANA freeze before the VALIDATION read (DESCRIPTIVE family) |
| **R7** | **Sparse-aware aggregation (FedSubAvg-style per-row scaling of `item_embed`)** as a NICE rung for cold FA/FP only | P12 and P13 are the production and literature answers to row dilution. Not DP-compatible as-is | Cold FA +10–30 % relative (INFERENCE; the effect shrinks with FedAdam, which already rescales rare coordinates by 1/τ, §4.2). Irrelevant for S8 | Code ≈ 0.5 day + review; 1–2 runs | Yes (lifts 06L A6), before its read. **Lower priority than R1–R4:** the realistic Edge line (S7/S8) does not need it |
| **R8** | **Hybrid retrieval → on-device re-rank**: keep as DISCLOSE-ONLY | Needed for 10⁸-item catalogues, not for 158k items. Its candidate request leaks context | — | 0 | No |

Not proposed, with reasons:
- Reading PRE's value on CALIB_VALIDATION: that is the same model on ≈ 25 % of the rows of its confirmatory MAIN evaluation, i.e. an
  early partial MAIN read. Forbidden in spirit. The PRE run's own UNEW_VAL marks are non-MAIN and DESCRIPTIVE (06K §1); reading them is
  the safe sanity check if the owner wants one number, and a DL row should say it selects nothing.
- Larger cohorts at equal exposure (fewer rounds; Charles et al. 2021).
- MIME at full tables (bytes > 170 MB).
- SCAFFOLD / stateful clients.
- Sampled softmax, tied tables, KD into SMALL (they change the model class against C_SMALL).
- Any central small model on MAIN users (C_SMALL_FULL, owner).

**Recommended order (PLAN).**
1. Register 06M now (§1 R1, §2 R2, §3 R3, §4 R4, plus the R6 analysis note) before the S3 selection and before any S7 value.
2. Launch S7-EARLY (5 runs) on whichever card has room; PRE is done, so nothing blocks it.
3. S3 selection → (S3b if triggered) → S4 → confirmatory S7/S8 (extended grids) → MAIN.
4. FT_FA_WARM follows FA_WARM MAIN.

S7-EARLY adds ≈ 7 h wall in parallel and moves the owner's key evidence (does training on private users improve a pretrained model?)
forward by ≈ 20 h.

---------------------------------------------------------------------------------------------------------------------
## 4. Diagnosis of the post-peak decline of FedAdam (η_s 0.01, τ 1e-3, ×1)

### 4.1 Facts (MEASURED)
- **The curve.** 0.0218 → **0.0342 at mark 3.0** → steady decline to **0.0251 at 6.0** (−27 %). S0 ×1 is flat after 2.5. S1b ×8
  (FedAvg) peaks at 3.5 and ends at 0.0261 (−9 %). FedAvgM ×1 peaks at 2.5 and stays ≈ 0.026 after (≈ −5 %). Source: register, curves
  read 2026-09-30 about 08:10Z.
- **The schedule.** The client LR follows the central shape on rounds: KNOTS ((0, 0), (0.02, 1), (2.0, 1), (3.0, 0.1), (5.5, 0.1),
  (6.0, 0.01)) (`fl/schedule.py:38`). The **server LR η_s is constant** (`fl/server.py:167–177`).
- **FedAdam.** m ← 0.9 m + 0.1 Δ; v ← 0.99 v + 0.01 Δ²; θ ← θ + η_s · m / (√v + τ); no bias correction, v₀ = τ²
  (`fl/server.py:171–176`).
- **Delta sizes** (REPORTED, ladder research §1.1): client AdamW, fresh each visit, ≈ 4 steps, peak lr 7.11e-4 at ×1. The per-client
  displacement is ≤ ≈ 2.8e-3 per coordinate. There are ≈ 57 survivors per round. An input-table row touched by k clients moves
  ≈ k/57 of one local update.

### 4.2 What the FedAdam normalisation really does to the LR decay (INFERENCE, arithmetic)
In steady state, for a coordinate whose aggregated delta has magnitude |Δ|, the step is u ≈ η_s · |Δ| / (|Δ| + τ).

| Coordinate class | Δ at peak (×1) | u at peak | Δ at 0.1× | u at 0.1× | Decay of u | FedAdam / FedAvg ratio (peak → 0.1×) |
|---|---|---|---|---|---|---|
| consensus (encoder, popular-item output rows, the common "push non-targets away from the mean state" direction) | ≈ 2.8e-3 | ≈ 7.4e-3 | ≈ 2.8e-4 | ≈ 2.2e-3 | **only ≈ 3.4×** | 2.6 → 7.8 |
| non-consensus / noise (≈ 1/√57 of a client step) | ≈ 3.8e-4 | τ-dominated ≈ 10 · Δ | ≈ 3.8e-5 | ≈ 10 · Δ | ≈ 10× | ≈ 10 (like FedAvgM with an effective LR of 10) |
| rare input-table rows (k = 1) | ≈ 5e-5 | ≈ 10 · Δ | ≈ 5e-6 | ≈ 10 · Δ | 10× | 10 (a 1/τ amplification of rare rows at all times) |

Consequences:
1. **The register's wording** ("FedAdam re-amplifies the 10× smaller deltas to ≈ η_s-sized steps once v forgets") holds only when
   |Δ| ≫ τ. At τ = 1e-3 most aggregated coordinates sit at or below τ, so the normalisation cannot fully undo the decay. It would hold
   strongly at τ = 1e-5, so **the S2 τ 1e-5 edge run is a free test of it** (prediction: a steeper, earlier post-3.0 decline).
2. **What does happen at τ = 1e-3 is a *partial and uneven* anneal.** The strongest (consensus) directions keep ≈ 30 % of their peak
   step, while fine-grained, item-specific and noise directions anneal fully (10×). After mark 3.0 the model keeps being pushed along
   its dominant directions at roughly 7.8× the FedAvg rate, while the fine-grained learning that would counter-balance them is annealed.
   With a full softmax and fresh client AdamW, those dominant directions include the sign-like, all-rows "push every non-target item away
   from the mean state" component (a popularity-prior drift). Continued drift along it plausibly degrades ranking (INFERENCE, unproven).
3. **The v-lag transient explains the exact timing.** The decay runs over marks 2.0–3.0 (≈ 285 CALIB rounds), while v's memory is
   ≈ 1/(1 − β2) = 100 rounds. During the decay, √v overestimates the shrinking |Δ|, so steps shrink faster than the client LR: an extra
   anneal that peaks when the decay ends (mark 3.0 = the observed peak). Once v catches up (≈ 100–200 rounds, by ≈ mark 3.4–3.7), the
   step rises back to the steady-state ratio in the table, and part of the anneal gain is lost. **FedAvg has no such rebound**, which
   fits S0 staying flat after the decay.
4. **Overfitting is also present.** Every stronger recipe declines after its peak: S1b −9 % with no normalisation at all, FedAvgM −5 %.
   CALIB has ≈ 650k examples for a 20.7 M-parameter model, and each client is revisited ≈ 1.5 more times after mark 3.0. Central SASRec
   on this data also peaks early (REPORTED: the G2 / "6.0-endpoint reversal"). FedAdam's 1/τ amplification of rare-row coordinates
   (table rows 2–3) is also a memorisation channel: it is exactly the mechanism that helps sparse features and can over-fit them.
5. **Working decomposition (INFERENCE, to be tested):** of FedAdam's −27 %, ≈ −5 to −9 % is the overfitting all strong recipes share,
   and the remaining ≈ −18 to −22 % is the FedAdam-specific late-phase effect (points 2–3).

### 4.3 Tests that separate the hypotheses (predictions registered here as expectations, not rules)

| Observation | H1 overfitting predicts | H2 LR-shape × FedAdam predicts | Available |
|---|---|---|---|
| S2 τ 1e-5 curve | same decline as τ 1e-3, scaled by its peak | a **steeper**, earlier post-3.0 decline | after the S2 selection is written (then allowed) |
| S2 η_s 0.003 / 0.001 curves | similar relative decline, perhaps later | relative decline shrinks with η_s | now (values already read, DL 249) |
| S3 W1 (AdamW, FedAdam, FL_CONST) | peaks at a similar or earlier mark (larger steps), then declines | smooth rise with no 3.0 transient; the peak moves late (≥ 4.5) or no decline until the last-10 % cool-down | at the S3 selection |
| S3 SGD cells (A2, W2) | decline, if present, similar to W1's | little late decline: SGD deltas on rare rows are their true tiny gradients, so the table-2 rows shrink; the all-rows sign drift disappears | at the S3 selection |
| Server-step norm per mark (checkpoint differences) | falls ≈ 10× after 3.0 like the LR | falls only ≈ 3–4× after 3.0 | now if mark checkpoints are kept (CPU) |
| Mean client training loss per round | keeps falling while validation falls | flat or rising after 3.0 (noise-dominated steps) | only after the logging fix (R5) |
| S4 passes 1 | reduces the decline | little effect | at the S4 selection |

### 4.4 What to register if S3 does not fix it (06M §2, before the S3 selection is read)
- **Decline index (descriptive, computed by the selection tool):** D = (PRACTICAL_BEST value − mark-6.0 value) / PRACTICAL_BEST value,
  reported for every screen.
- **Trigger (mechanical):** the S3 winner has D ≥ 0.10 and its peak mark ≤ 4.5 → stage S3b runs before S4. Otherwise S3b is recorded
  "not triggered".
- **S3b candidates at the S3 winner (2 runs):**
  - (a) **SERVER_ANNEAL:** η_s,r = η_s × f(r), where f is the same round factor that the S3 winner applies to the client LR (for
    FL_CONST, only the last-10 % cool-down; for the central shape, the full decay). Rationale: under a normalising server optimiser the
    anneal must act where the step is formed (§4.2).
  - (b) **SERVER_EMA:** the server keeps θ̄_r = 0.99 θ̄_{r−1} + 0.01 θ_r (bias-corrected, checkpointed). Marks and PRACTICAL_BEST are
    evaluated on θ̄. The training trajectory is unchanged.
  - Both are server-side and data-independent: clients stay stateless, bytes are unchanged, and DP accounting is unchanged
    (post-processing).
- **Selection:** 06J rule: best over S3b ∪ the S3 winner; the ≥ 0.002 incumbent rule; ties → incumbent, then SERVER_ANNEAL before
  SERVER_EMA.
- **Cost:** 2 × 5–7 h wall in parallel. Code ≈ 1 day incl. tests (FedAvg digests unchanged at defaults).
- **Why before the S3 read:** the trigger reads S3 curve values, so the rule must exist first (the no-change-after-read rule).
- **If D < 0.10 on the S3 winner (FL_CONST cured it):** H2 is supported. Record it in the ladder register as "the central LR shape
  interacted with FedAdam's normalisation", with the §4.3 evidence.
- **If D ≥ 0.10 even under FL_CONST and SGD:** overfitting dominates. S4 passes 1 and the S3b EMA are the remedies, and MAIN's 4× clients
  should shift the peak later (INFERENCE). Disclose that CONTROLLED_BUDGET understates the recipe.

---------------------------------------------------------------------------------------------------------------------
## 5. Plain-language summary for the owner (10 lines)

1. Real systems (Google keyboard, Apple, Firefox, Chrome, Taobao) run the prediction on the phone or in the browser, not on the server.
2. They almost never start federated training from a random model. They start from a model trained on data they may use (public data, opted-in users) or from settings that already work.
3. Then federated training or on-device fine-tuning improves only a small part with the private data (a head, a few settings, a personal fine-tune).
4. Our plan already has this realistic line: PRE (trained on 1.2 M opted-in users, finished) → warm federated training (S7) → frozen catalogue tables (S8) → personal fine-tune (FT_PRE).
5. The problem: this line runs last, after two more tuning stages, and its screen changes the wrong knob.
6. Fix 1: test the warm start now, on 5 small runs (about 7 hours in parallel), with smaller server steps, because a pretrained model is easily damaged by big steps.
7. Fix 2: add "warm federated model + personal fine-tune on the device", the complete production recipe.
8. The FedAdam curve falls after mark 3.0 for two likely reasons: some overfitting (all strong recipes show it), and the learning-rate decay not being fully passed through FedAdam.
9. The running S3 runs (constant learning rate) test this. We should register now what to do if they do not fix it (cool the server step, or average the server model).
10. None of this reads MAIN or TEST values early, and none uses a central small model on the private users.

---------------------------------------------------------------------------------------------------------------------
## 6. Sources opened today (URLs)
- Hard et al. 2018: https://arxiv.org/pdf/1811.03604 (full text)
- Xu et al. 2023: https://arxiv.org/pdf/2305.18465 (full text)
- Zhang et al. 2023 (Gboard white paper): https://arxiv.org/pdf/2306.14793 (full text)
- Yang et al. 2018: https://arxiv.org/pdf/1812.02903 (full text)
- Paulik et al. 2021: https://arxiv.org/pdf/2102.08503 (full text)
- Hartmann et al. 2019: https://arxiv.org/pdf/1911.11807 (full text)
- Wang et al. 2019: https://arxiv.org/abs/1910.10252 (abstract)
- Singhal et al. 2021: https://arxiv.org/abs/2102.03448 (abstract)
- Ding et al. (FedSubAvg): https://arxiv.org/abs/2109.07704 (abstract)
- Charles & Konečný 2020: https://arxiv.org/abs/2007.00878 (abstract)
- Nguyen et al. 2023: https://arxiv.org/abs/2206.15387 (abstract)
- Wang et al. 2023/24: https://arxiv.org/abs/2305.12132 (abstract)
- Gong et al. (EdgeRec): https://arxiv.org/abs/2005.08416 (abstract)
- Yao et al. (DCCL): https://arxiv.org/abs/2104.06624 (abstract)
- On-device recommender tutorial: https://arxiv.org/abs/2312.10864 (abstract; used only for the three-way categorisation)
- Chrome Topics: https://privacysandbox.google.com/private-advertising/topics/web/

From search summaries only (REPORTED, pages not opened):
- Niu et al. MobiCom 2020: https://dblp.org/rec/conf/mobicom/NiuWTHJLWC20.html
- PAPAYA: https://arxiv.org/pdf/2111.04877

Reddi et al. 2021 figures are REPORTED from FL_STRONGEST_LADDER_RESEARCH_20260929 (PDF read there on 2026-09-29).

UNVERIFIED:
- the current status of Chrome Topics;
- where EdgeRec's re-ranker is trained;
- Granqvist et al. 2020 (arXiv 2008.02651): its abstract does not state the initialisation, so it is not used.

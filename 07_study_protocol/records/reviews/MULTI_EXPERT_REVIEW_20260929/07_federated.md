# 07 — Federated training: FedAvg / FedProx / PF, participation, drop-out, rounds, local optimiser, LR screen, the S0→S8 ladder, realism

Expert 6 of the multi-expert review, 2026-09-29. Reviewer: Claude Opus 5.5, acting as a senior production-FL / recommender scientist.
READ-ONLY. The only file written is this one. I did not contact the server, run git, run tests or any project script, or open any
RESULT_ROWS, ranks, eval-row, RETRO_* or runs_*/ file. The only downstream numbers used are the registered CALIB screen values
(DECISION_LOG rows 181 and 218, `company/registers/FL_IMPROVEMENT_LADDER.md`) and the registered CALIB selection record
`contracts/fl/LR_SELECTION_EXT.json`. That record is a contract file, not a RESULT_ROWS file. It holds the same six values plus the
index of each run's PRACTICAL_BEST mark.

Tags: **MEASURED** (from a file, cited as path:line; paths are relative to `.local_ai/ppsi_v4/`), **REPORTED** (a number stated in a
project document), **INFERENCE** (my own derivation or estimate), **OPINION**. Literature is cited only where I checked it by web
search today; everything else is marked UNVERIFIED.

---------------------------------------------------------------------------------------------------------------------
## 1. What the project does (MEASURED)

### 1.1 The federated recipe as registered (S0)
- **Knobs** (`code/v4int/launch_fl.py:163`, KNOBS):
  - 64 clients per round, sweep sampling;
  - 10 % seeded drop-out;
  - 6,827 rounds on MAIN;
  - local batch 16, 2 passes;
  - clip 1.0, weight decay 1e-5;
  - FedProx μ 0.01;
  - PF λ 1e-4, PF clip 1.0;
  - FA_1024 / DP: m = 1,024 clients and 384 rounds.
- **Round counts.** The CALIB screen runs `ceil(3·32,768/64/0.9) = 1,707` rounds (`contracts/PREG1_AMENDMENT_06B.md:12`). S4 recomputes
  the count as `ceil(60 N/(576 P))` (`launch_fl.py:228-237`).
- **Sampling** (`code/fl/participation.py:1-40, 277-288`). Each sweep is a seeded permutation without replacement, cut into fixed groups
  of 64, so every client is visited exactly once per sweep. A 1,707-round CALIB run covers 3 sweeps plus a partial 4th (512 rounds per
  sweep).
- **Drop-out** (`participation.py:211-221`). It is drawn before training, independently per sampled client, from (seed, round). Only
  survivors train and are aggregated (`fl/runtime.py:517-551`).
- **Client visit** (`code/fl/client.py:265-307`):
  - the client receives θ_r and makes a **fresh AdamW every visit** (`client.py:290`, `make_optimizer` `client.py:160-168`);
  - it runs 2 passes of B16 over its own valid rows;
  - the loss is full-K CE over 158,486 classes (`client.py:186-188`);
  - the gradient is clipped to a total L2 norm of 1.0 (`client.py:222-223`), then AdamW steps;
  - SASRec is recentred after every step (`client.py:229`; `code/fl/families.py:104-108`);
  - the upload is the full shared state, with weight n_consumed = passes × n_valid (`client.py:306`).
- **Aggregation** (`code/fl/aggregate.py:3`, `aggregate.py:486` add, `aggregate.py:526-531` finalize). The server takes the
  n_consumed-weighted mean of the uploaded *states*, which equals θ_r plus the weighted mean of the deltas. Buffers are FIXED or
  SERVER_COPY and never averaged (`aggregate.py:532-536`).
- **Server step.** Registered: server LR 1, no momentum (`code/fl/server.py:296-301`, `server_opt=None`). FL-10 adds:
  - FedAdam per Reddi et al. Alg. 2: β1 0.9, β2 0.99, τ 1e-3, v0 = τ², no bias correction (`server.py:152-187`);
  - FedAvgM with β 0.9, η 1 (`server.py:213-248`).
  In both cases SASRec recentring follows the step (`server.py:299-300`).
- **LR over rounds** (`code/fl/schedule.py:35-38`; `runtime.py:351-395`). The central s(EFE) shape is applied through normalised
  exposure progress:
  - warm-up to 0.02;
  - peak to 2.0;
  - linear decay to 0.1× by 3.0;
  - 0.1× until 5.5;
  - linear to 0.01× at 6.0.
  The LR is constant within a round. FL_CONST (`runtime.py:112-125`) warms up over 1 % of rounds, holds the peak, then cools over the
  last 10 % of rounds to 0.1×.
- **Client optimisers.** Options are AdamW-fresh (default) and SGD with momentum 0.9 (fresh per visit, no weight decay, clip kept)
  (`client.py:164-166`).
- **Local LR.** SMALL peak 0.000711412 × m (`contracts/fl/LR_SELECTION_EXT.json` "lr_peak_base"). The S1b winner is m = 8, i.e. a peak
  local LR of 0.00569.
- **PF** (`client.py:239-255`). The personal vector p_u has the query dimension and its own persistent AdamW. Its LR is the **SMALL
  peak × round factor and is never multiplied by m** (`launch_fl.py:656`; 06B §1 L21-22).
- **FedProx** (`client.py:171-178, 220-221`). The exact proximal gradient μ(θ−θ_r) is added before clipping.
- **Evaluation of FL weights.**
  - MAIN: the server adapter *is* the locked CentralTrainer's harness model (`launch_fl.py:1928, 1966-1971`, calling
    `CentralTrainer._evaluate`, `code/v4train/trainer.py:972-994`).
  - CALIB: `role_evaluator` mirrors `_evaluate` line for line, with the CALIB manifests (`launch_fl.py:1700-1735`).
  - `predict_adapter` sets `module.eval()` (`code/v4eval/predict.py:146`).
  - Clients and the harness use the same `StoreTrainProvider` / `StoreEvalProvider` classes with the same `history_policy`
    (`launch_fl.py:1912-1915, 1935-1938`).
  - The worker model is built with the same variant and dropout as the harness (`launch_fl.py:1929-1930`).

### 1.2 The ladder (registered)
| Stage | Registration | State (REPORTED, DL 218-223) |
|---|---|---|
| S0 | FedAvg, m = 1 | CALIB 0.023473 |
| S1a | m ∈ {0.5, 1, 2} (06B §1) | 0.020263 / 0.023473 / 0.025562 |
| S1b | m ∈ {4, 8, 16} (06F) | 0.027570 / **0.028796** / 0.022909 → ×8 (DL 218) |
| S2 | FedAdam η_s {0.01, 0.1} × m {1, 8} + FedAvgM β 0.9, η 1 at m 8; edge rules (06G, 06J §1) | 5 runs launched 14:44Z; **values not read** (DL 221: "at marks 1.0–1.5") |
| S3 | on the S2 winner: AdamW-fresh×FL_CONST; SGD-m lr {0.01, 0.1, 1}×FL_CONST; edges (06I §3, 06J §2); sequential, combination run withdrawn (06J L26-28) | not started |
| S4 | passes 1 / 4, batch 8, one at a time, no LR re-screen (06J §3) | not started |
| Stopping rule | tie 0.0005; gain < 0.002 → incumbent; nothing after S4 (06J §4) | registered; raw-value reading DL 222 |
| S5 | FA_X2 12 EFE, MAIN, descriptive (06I §4) | — |
| S7 / S8 | PRE on 1,217,610 non-MAIN users → FA_WARM (m ∈ {m*, m*/4}); frozen item tables (06K) | PRE training (DL 221) |

**Best-mark positions** (MEASURED, `contracts/fl/LR_SELECTION_EXT.json` runs[].practical_best_mark):

| m | best mark (EFE) |
|---|---|
| ×0.5 | 5.0 |
| ×1 | 2.5 |
| ×2 | **1.5** |
| ×4 | 2.5 |
| ×8 | 3.5 |
| ×16 | 3.5 |

For ×1, the max of the 12 marks (0.023507) exceeds the PRACTICAL_BEST value (0.023473) by less than 0.0002, so the curve is flat after
2.5 EFE.

---------------------------------------------------------------------------------------------------------------------
## 2. State of the art (literature and real systems; verified today unless marked)

**FedAvg and server optimisers**
- **FedAvg.** McMahan, Moore, Ramage, Hampson & Agüera y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized
  Data", AISTATS 2017 ([PMLR v54](https://proceedings.mlr.press/v54/mcmahan17a.html); arXiv 1602.05629). Clients run local SGD and the
  server averages weighted by n_k.
- **FedOpt / adaptive servers.** Reddi, Charles, Zaheer, Garrett, Rush, Konečný, Kumar & McMahan, "Adaptive Federated Optimization",
  ICLR 2021 ([arXiv 2003.00295](https://arxiv.org/abs/2003.00295)). The canonical configuration is **client SGD + server
  Adam/Yogi/Adagrad**. The largest gains are on sparse, large-vocabulary tasks (Stack Overflow NWP). The table values quoted by the
  project research (FedAvg 19.5–20.5 vs FedAdam 25.2–25.7) are REPORTED from that PDF; I did not re-read them.
- **Server momentum.** Hsu, Qi & Brown, "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification",
  2019 ([arXiv 1909.06335](https://arxiv.org/abs/1909.06335)) introduces FedAvgM. Hard et al., "Federated Learning for Mobile Keyboard
  Prediction", 2018 ([arXiv 1811.03604](https://arxiv.org/abs/1811.03604)) is the Gboard next-word model with client SGD and server
  momentum. The "Nesterov 0.9, server LR 1" detail is REPORTED by REALISM_AUDIT B6; I did not re-read it.
- **Field guide.** Wang, Charles, Xu, Joshi, McMahan et al., "A Field Guide to Federated Optimization", 2021
  ([arXiv 2107.06917](https://arxiv.org/abs/2107.06917)). It frames ClientOpt/ServerOpt and recommends server-side adaptivity with a
  cheap client (SGD) and joint tuning of client and server LR.
- **Cohort size.** Charles, Garrett, Huo, Shmulyian & Smith, "On Large-Cohort Training for Federated Learning", NeurIPS 2021
  ([proceedings](https://proceedings.neurips.cc/paper/2021/hash/ab9ebd57177b5106ad7879f0896685d4-Abstract.html)). Larger cohorts have
  diminishing returns per round and behave like large batches. At a fixed number of client visits, bigger cohorts mean fewer server
  steps.
- **Stateless adaptive clients.** Karimireddy, Jaggi, Kale, Mohri, Reddi, Stich & Suresh, "Breaking the centralized barrier for
  cross-device federated learning" (Mime), NeurIPS 2021
  ([proceedings](https://proceedings.neurips.cc/paper/2021/hash/f0e6be4ce76ccfa73c5a540d992d0756-Abstract.html)). Clients reuse the
  server's optimiser state inside local steps instead of starting a fresh optimiser, which is exactly the remedy for the per-visit Adam
  reset. It costs extra download bytes.

**Production systems**
- **Gboard with DP.** Xu, Zhang, Andrew, Choquette, Kairouz, McMahan, Rosenstock & Zhang, "Federated Learning of Gboard Language Models
  with Differential Privacy", ACL 2023 Industry ([ACL Anthology](https://aclanthology.org/2023.acl-industry.60/)). Production uses
  DP-FTRL, server momentum and public pre-training.
- **System design.** Bonawitz et al., "Towards Federated Learning at Scale: System Design", MLSys 2019
  ([arXiv 1902.01046](https://arxiv.org/abs/1902.01046)). The "few hundred devices per round" and drop-out-rate figures are REPORTED by
  REALISM_AUDIT B11; I could not confirm the drop-out percentage today (UNVERIFIED).

**Initialisation and pre-training**
- Nguyen, Wang, Malik, Sanjabi & Rabbat, "Where to Begin? On the Impact of Pre-Training and Initialization in Federated Learning",
  ICLR 2023 ([code](https://github.com/facebookresearch/where_to_begin)). Pre-training gives up to 40 % better accuracy and less
  sensitivity to heterogeneity.

**Sparse embeddings in FL — the key literature for this project's architecture**
- **FedSubAvg.** Ding, Niu, Wu, Tang, Lyu, Feng & Chen, "Federated Submodel Optimization for Hot and Cold Data Features", NeurIPS 2022
  ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/002262941c9edfd472a79298b2ac5e17-Abstract-Conference.html);
  arXiv 2109.07704). FedAvg with large sparse embedding tables is "severely slowed down": cold-feature rows are optimised far more
  slowly than hot ones. FedSubAvg scales each parameter's aggregated update by (#clients)/(#clients involving it), which is shown to
  act as a diagonal preconditioner of an ill-conditioned objective.
- **Submodels at billion scale.** Niu, Wu, Tang, Hua, Jia, Lv, Wu & Chen, "Billion-scale federated learning on mobile clients: a
  submodel design with tunable privacy", MobiCom 2020 ([dblp](https://dblp.org/rec/conf/mobicom/NiuWTHJLWC20.html)). Clients download
  and upload only the embedding rows they touch, with a private set union and secure aggregation.
- **FedFast.** Muhammad et al., "FedFast: Going Beyond Average for Faster Training of Federated Recommender Systems", KDD 2020
  ([ACM](https://dl.acm.org/doi/10.1145/3394486.3403176)). Better client sampling and aggregation are ≥ 4× faster than FedAvg for
  federated recommenders.
- **Federated Reconstruction.** Singhal, Sidahmed, Garrett, Wu, Rush & Prakash, "Federated Reconstruction: Partially Local Federated
  Learning", NeurIPS 2021 ([proceedings](https://proceedings.neurips.cc/paper/2021/hash/5d44a2b0d85aa1a4dd3f218be6422c66-Abstract.html)).
  Local parameters are reconstructed per visit (stateless) and the method is deployed in Gboard.
- **Federated sequential recommendation.**
  - "A Systematic Survey on Federated Sequential Recommendation" ([arXiv 2504.05313](https://arxiv.org/abs/2504.05313); venue
    UNVERIFIED).
  - PTF-FSR ([arXiv 2406.05387](https://arxiv.org/abs/2406.05387); venue UNVERIFIED).
  - SecEmb, sparsity-aware secure FL for large embeddings, evaluated on SASRec ([arXiv 2505.12453](https://arxiv.org/abs/2505.12453);
    venue UNVERIFIED).
  I found none that trains full-catalogue SASRec over ~10⁵ clients with a temporal split, so this project's setting is harder and
  larger than the published federated SASRec work (INFERENCE from the survey's summary).

**Simulators and benchmarks**
- FedScale, Lai et al., ICML 2022 ([PMLR v162](https://proceedings.mlr.press/v162/lai22a.html)): realistic client-availability traces.
- FLUTE, Dimitriadis et al., Microsoft 2022 ([arXiv 2203.13789](https://arxiv.org/abs/2203.13789)): simulation at millions of clients.
- LEAF: not re-verified today (UNVERIFIED).

**Practice consensus (OPINION, supported by the sources above)**
1. Client SGD with server momentum or adaptivity.
2. Constant or staircase client LR over rounds.
3. Tune client and server LR jointly.
4. Warm-start from public or consenting-user data.
5. For huge sparse tables: sparse-aware aggregation or submodel updates, sampled or partial softmax, or frozen or server-side tables.

---------------------------------------------------------------------------------------------------------------------
## 3. Comparison

### 3.1 Why FA sits at ≈ 20–25 % of central on CALIB — first principles and code (INFERENCE unless tagged)
Numbers: CALIB ≈ 32,768 users × 19.7 decisions ≈ 0.65 M decisions (19.7 per user is REPORTED in the ladder research §1.1).

**(a) Nine times fewer server steps.**
- Central CALIB at 6 EFE and batch 256 makes ≈ 6 × 0.65 M / 256 ≈ **15 k persistent-Adam steps**. The FL run makes **1,707** averaged
  updates. The ratio is ≈ 9.
- With server LR 1 and no momentum, one round can move a coordinate by at most the survivors' mean local displacement: ≈ 4 local steps
  × lr × (the coherence across ≈ 57 clients).
- Matching central's total path therefore needs m ≈ 9/(4 × coherence). With coherence ≈ 0.25–0.3 that gives **m ≈ 8**, which is where
  the S1b optimum lies (MEASURED 0.028796 at ×8).
- The LR screen behaves as this displacement argument predicts. ×16 falls (0.0229) because the incoherent part grows with it.

**(b) Adam reset every visit plus full softmax gives dense, sign-like noise on the output table.**
- With a fresh Adam, the first step is lr·g/|g| per coordinate, and the next ≈ 3 steps stay close to that. Every one of the 158,486
  output rows gets a non-zero CE gradient (≈ p_i·h, with p_i ≈ 1/K for tail items). Adam's scale invariance lifts those ~10⁻⁶ gradients
  to full ±lr steps.
- At ×8, one step is 0.0057 ≈ **0.28 × the output-table init std 0.02** (MEASURED `code/v4models/sasrec_v2.py:255`). Four steps per
  visit are therefore ≈ 1 init-std of perturbation on every tail row.
- The common mode (identical across tail items) is removed exactly by recentring (`sasrec_v2.py:405-410`). The item-specific residual
  (the sign of Σ_b p_{b,i} h_b differs by item) is not removed.
- Averaged over ≈ 57 clients, it leaves ≈ 4·lr/√57 ≈ 0.003 per coordinate per round of incoherent drift. Over ~600 peak-equivalent
  rounds that random-walks to ≈ 0.07 ≈ 3–4 init-std on tail rows (order of magnitude only).
- Central persistent Adam does not do this: v accumulates the rare large target gradients, so m̂/√v̂ stays ≪ 1 on the tail.
- Random logit offsets on tail items put irrelevant items into the top 20, which is exactly what MRR@20 punishes.

**(c) Sparse-row dilution.**
- An input-table row is updated only by the clients whose histories contain that item. The server averages their deltas with everyone
  else's zeros (`aggregate.py:526-531`), so the row moves ≈ (n_u/Σn) ≈ 1/57 of its local update per round.
- Central Adam gives a full per-coordinate step on every batch that contains the item.
- Worked example: a user-specific item seen by one client in its 3 visits moves ≈ 3 × 4.5 × 0.0057/57 ≈ 1.3 × 10⁻³ in FL, against
  ≈ 120 × 0.0007 ≈ 0.08 centrally (≈ 60× less, INFERENCE).
- This is the hot/cold-feature pathology of Ding et al. 2022. FedAvg can learn head items and the encoder, but hardly the long tail.

**(d) The LR shape.** Normalised progress puts **50 %** of the budget (EFE 3.0–6.0) at ≤ 0.1× peak (MEASURED knots `schedule.py:38`).
06I §3 L24 says "≈ 2/3", which is an overstatement; the ladder research's "≈ 53 % lost / 0.47 peak-equivalent" is right.

**(e) Heavy-user dominance.** Under Adam, a client's displacement grows with its step count (∝ n), and its weight is also ∝ n. So a
client's influence grows ≈ n² (1–14 steps per visit). The model is pulled towards heavy users while the metric is client-macro, where
light users are the majority. This is small but systematic.

**(f) Early plateau.** PRACTICAL_BEST marks of 1.5–3.5 EFE for m ≥ 1 (§1.2, MEASURED) mean the runs reach their best early and then stay
flat or decline.
- The FL run is **not** clearly budget-limited. It saturates, which fits (b) and (c): noise accumulates as fast as signal.
- Central C_BIG also peaks at 3.0 (best@3.0, PROJECT_MAP), so part of this is ordinary SASRec overfitting.
- I cannot separate the two without the per-mark curves, which I did not read.

**Net (OPINION).** 20–25 % is what (a)–(c) predict for FedAvg + fresh-Adam + full softmax + random 158k-row tables at ~3 visits per
client. It is real FL difficulty *of this recipe*, not a measurement artefact. It is not "FL difficulty" in general: every production
remedy listed in §2 targets one of (a)–(c).

### 3.2 Bug and evaluation-mismatch check (MEASURED unless noted)
| Check | Finding |
|---|---|
| Same evaluator, features and history for FL and central | Same harness object and T3 path (`launch_fl.py:1928, 1966-1971`; `trainer.py:972-994`); CALIB `role_evaluator` is a line-for-line mirror (`launch_fl.py:1700-1735`); same provider classes and `history_policy` (`launch_fl.py:1912-1915, 1935-1938`); `eval()` set (`predict.py:146`). **No mismatch.** I did not verify row-level identity of the client batches with central batches (INFERENCE: same class and args → same rows). |
| Dropout / model config | Worker built with the recipe's variant and dropout (`launch_fl.py:1929-1930`), as the harness. OK. |
| SASRec gauge / recentring | Row-mean subtraction is logit-shift invariant (`sasrec_v2.py:405-410`), so it is loss- and rank-neutral. Clients recentre per step, the server after every apply (`server.py:299-300`). The average of recentred states is recentred. FedAdam/FedAvgM steps are followed by recentring, which is loss-neutral. **No gauge bug.** |
| Aggregation | n_consumed-weighted state mean (`aggregate.py:486, 526-531`), exactly-once, planned == actual (`runtime.py:549-551`). FP32 rounding of summed states is ~10⁻⁹ against deltas of ~10⁻³–10⁻⁵ (INFERENCE). OK. |
| Buffers | FIXED verified bit-identical per visit (`fl/adapter.py:232-243`); no averaged buffers; FedBN out of scope. OK. |
| n_consumed weighting | Standard FedAvg n_k weighting; central is also example-weighted. Not a bug; see §3.1(e). |
| Weight decay on untouched rows | Decoupled lr·wd ≈ 6×10⁻⁸ per step. Negligible (agrees with the ladder research, rank 11). |
| Clip 1.0 | Irrelevant for Adam (scale-invariant). Binding for SGD-m, where max step = lr (disclosed, 06J §2). |
| Cosmetic | `fl_rows` hard-codes `optimizer_lifecycle="FRESH_ADAMW_PER_VISIT"` (`launch_fl.py:862, 871`) even for SGD-m rows. `fl_client_opt` is added by tuning_labels, so the information is present but the label is wrong for S3 SGD rows. |

**Verdict: no bug and no evaluation mismatch found.** The gap is algorithmic.

### 3.3 Strengths
- Exceptionally careful simulation engineering:
  - deterministic shard-ordered reduction;
  - bitwise resume, including server moments;
  - planned-LR table hashed before training;
  - exactly-once aggregation;
  - O_EXCL selection files.
  This is better than typical FedScale, Flower or FLUTE research code (OPINION).
- The honest *mechanical* diagnosis already exists (ladder research §1; realism audit B5-B10). The ladder is pre-registered stage by
  stage before each read, with tie and no-gain rules.
- Realism guards: stateless clients, a 170 MB per-visit cap, no server access to MAIN data, and a byte ledger that charges dropped
  clients' downloads.
- The warm-start branch (S7/S8) keeps PRE-alone and K21 = FA_WARM − PRE beside it. This is the correct way to avoid crediting FL with
  the pre-training (06K §2; DL 206).

### 3.4 Weaknesses, errors, unrealistic choices, missing pieces
1. **S2 grid for FedAvgM is mis-specified (design error, INFERENCE).**
   - FedAvgM with β 0.9 and η 1 has a steady-state step of Δ/(1−β) = 10Δ. Run only at m = 8 (06J L12), that is ≈ ×80 FedAvg-equivalent
     in the consistent direction.
   - FedAvg already collapses at ×16 (0.0229).
   - The production choice (server momentum) could therefore "lose" to FedAvg by artefact.
   - Reddi et al. and the field guide both say client and server LR must be tuned *jointly* for FedAvgM.
   - FedAdam got an m ∈ {1, m*} grid and edge rules; FedAvgM got neither.
2. **S2 → S3 order hides the canonical configuration.**
   - S2 evaluates server optimisers only with AdamW-fresh clients. Their deltas are sign-like and larger than FedAdam's τ = 1e-3, so
     FedAdam re-normalises the tail-row noise to ≈ η_s per coordinate per round (the same mechanism 06H registered for DP). S2 is thus
     biased *against* FedAdam.
   - S3 then tests client SGD only on the S2 winner, and 06J L26-28 withdrew the combination run.
   - If S2 picks FedAvg or FedAvgM, **client SGD + FedAdam — the Reddi et al. recipe for exactly this sparse-gradient regime — is never
     run.** Coordinate-wise ascent over two strongly interacting factors is the textbook way to miss the optimum. "No backtracking" is
     a presentation goal and should not decide which configuration gets tested.
3. **The direct remedy for dilution is missing.** Sparse-aware / submodel aggregation (FedSubAvg: per-row division by the share of
   clients that touched it) is not in the ladder or the catalogue. The ladder research names dilution but maps it only to FedAdam and
   S8.
   - Realism: the server already sees each client's item set in non-DP arms (06I §7a). Per-row counts can be computed under secure
     aggregation (Niu et al. 2020). It is stateless and adds zero bytes.
   - Constraint: it is not DP-compatible as-is, because scaling by 1/count changes the sensitivity. So it is for FA/FP only.
4. **The "lower bound" framing (06I §6(a)) is not supported as stated.**
   - The best marks are at 1.5–3.5 of 6.0 EFE (§1.2) and ×1 is flat after 2.5, so more rounds of the same recipe need not help.
   - Also, "lower bound" is conditional on the recipe. A deployment with better aggregation would be higher; one with availability bias
     or client heterogeneity would be lower (the uniform, uncorrelated drop-out here is optimistic).
   - Better wording: "retention under a matched exposure budget, with FA_X2 measuring budget sensitivity".
5. **PF's personal component is inert by construction (INFERENCE).**
   - p_u gets ≈ 3 visits × ≈ 4.5 steps at LR 0.000711 × s̄ (≈ 0.5). That is ≤ ≈ 0.005 per coordinate in total, ≈ 0.04 L2 against a query
     norm of O(8).
   - Meanwhile the shared LR is ×8. So PF ≈ FA, and K5 (and PF − FA) would measure "a personal vector that could not move".
   - PF MAIN has not launched and no PF value is read, so it is registrable now.
6. **FedProx under fresh Adam is not "mild".**
   - After step 1, μ(θ−θ_r) ≈ 0.01 × 0.0057 = 5.7×10⁻⁵ exceeds the tail-row CE gradient (≈10⁻⁶) by ~50×. Adam then steps those
     coordinates back toward θ_r at ≈ lr.
   - So FP with μ = 0.01 is a strong anchor on low-gradient coordinates. It may *suppress* the tail sign noise of §3.1(b), and FP > FA is
     plausible for a reason unrelated to client drift.
   - This interpretation should be registered before the FP read.
7. **Selection on CALIB ⊂ MAIN (25 % of MAIN users), with ≈ 25 configurations by the end of S8, all max-of-marks.** This has a winner's
   curse on 25 % of the K2 evaluation users. The effect is small relative to a 75–80 % gap (INFERENCE ≈ +1–2 % of FA's MAIN value), but
   it is undisclosed and asymmetric: central SMALL was tuned over 4 LRs.
   - 06A §1 already fixed this for FT/LO (UNSEEN_VAL).
   - For FL, a CALIB-disjoint diagnostic on MAIN_VALIDATION (the G2 condition-5 precedent) is the cheap fix.
8. **Transfer from CALIB to MAIN.** Hyperparameters are tuned at N = 32,768 / 1,707 rounds and applied at 131,072 / 6,827 rounds with
   the same cohort. The optimal local LR usually falls with 4× more server steps (INFERENCE). LADDER_S1a (×2, MAIN) gives one
   descriptive point.
9. **The central CALIB reference is soft.** "≈ 0.11–0.12" is INFERENCE from the P2 era, and 0.124844 is GRU D128. The matched
   denominator is the registered SMALL mini-ladder on CALIB (same D64, same CALIB users and manifest, 6.0 EFE; DL 143). The ladder
   register should quote that record (I did not open it).
10. **Stopping rule 0.002** ≈ 7 % of FA's CALIB value (OPINION). It is honest against noise (≈ 0.0017 half-width, REPORTED), but four
    stages each gaining 0.0015 (≈ 5 %) would all be discarded, a possible ≈ 20 % cumulative loss. It is registered and the S1b value
    has been read, so it must stay. Disclose the tension.
11. **Participation realism.**
    - Sweep sampling (everyone exactly once per sweep) and uniform drop-out, drawn before training and uncorrelated with activity or
      device, are more favourable than production diurnal, availability-biased participation (FedScale traces).
    - Here drop-out only shrinks the cohort to ≈ 57.6 and charges bytes. It has no model effect.
    - 64 per round is below production's hundreds. At equal exposure that is the accuracy-favourable direction (Charles et al. 2021).
12. **Methods excluded, with correct reasons (KEEP, DISCLOSE).**
    - SCAFFOLD (in the proposal): stateful, with ~3 visits so control variates go stale.
    - FedBN: owner-excluded, and there is no BN (LayerNorm).
    - Mime: breaks the 170 MB cap for full tables, but is feasible under S8.
    - FedYogi/FedAdagrad: ≈ FedAdam in Reddi et al.
    - Sampled softmax: changes the objective relative to C_SMALL.
13. **Contract text.** CLIENT_CONTRACT_V1.md L240 ("Centrally trained winner weights never initialize FL or LO") should cite 06K as the
    governing exception for θ_pre. PRE is not a winner and is not trained on MAIN users; this is a wording issue only.

---------------------------------------------------------------------------------------------------------------------
## 4. Recommendations

Governed-value status (MEASURED, DL 218-225):
- **read:** S0, S1a, S1b;
- **not read:** S2 (running), S3, S4, S7, S8, every MAIN FL value, PF, FP.

A new registration is therefore still legitimate for anything that touches S2 onward, PF, FP or the analysis, **provided it is written
before the S2 selection is read.**

| ID | Item | Class | Action | Cost (INFERENCE) | Affects | Pre-registration needed / status |
|---|---|---|---|---|---|---|
| F1 | Add FedAvgM at m ∈ {1, 2} (β 0.9, η 1) to S2, plus an edge rule (if m = 1 is the best FedAvgM → one run at m = 0.5). Rationale §3.4-1 | **MUST** | ADD | 2 CALIB runs ≈ 2 × 2.7 process-h, in parallel with the running S2 | S2 (running); every MAIN FL arm via FL_TUNING_SELECTION | Yes — amendment before any S2 value is read. **Feasible now** (S2 unread, DL 221) |
| F2 | Guarantee the canonical cross cell: register S3 candidates **client SGD-m × FedAdam** (η_s 0.01, τ 1e-3, lr ∈ {0.1, 1.0}, FL_CONST) whatever the S2 winner; S3 winner = best over S3 ∪ incumbent as registered | **MUST** | ADD (re-instates a combination run) | 2 CALIB runs | S3, final FA | Yes — must be registered BEFORE the S2 read so it cannot look outcome-driven. Feasible now |
| F3 | Sparse-aware aggregation candidate (FedSubAvg-style per-row count correction on `item_embed` rows; output rows are dense, no change) as an S2b/S3b rung for FA/FP; DP arms keep plain aggregation (disclosed) | SHOULD (high) | ADD | code ≈ 0.5 d + tests (`aggregate.py` row-count accumulator; bitwise default unchanged); 1–2 CALIB runs | FA/FP; not DP | Yes — new amendment before its first read; before S3 selection if it is to enter FL_TUNING_SELECTION |
| F4 | Reword 06I §6(a) "LOWER BOUND" → "matched-exposure retention of recipe R; FA_X2 bounds the budget effect"; correct "≈ 2/3 of rounds at ≤ 0.1×" (06I §3 L24) → "50 % of the exposure budget"; cite the best-mark positions | **MUST** (wrong claim otherwise) | CHANGE (text) | 0 | thesis text, ANA labels | Interpretation text — register before the VALIDATION read (ANA freeze pending). Not a governed value |
| F5 | Mechanism diagnostics, no selection: `embedding_stats` (`sasrec_v2.py:413-421`) on θ0, S0/S1b screen best.pt and C_SMALL best.pt (head vs tail row norms, drift from θ0, output-row dispersion); per-mark CALIB curves + train loss of the screens. Separates dilution / sign-noise / overfitting | SHOULD | ADD | CPU < 1 h, weights only | thesis explanation of K2 | Descriptive; state in a DL row that it selects nothing |
| F6 | PF personal LR: either scale p_u's LR by m* (symmetry with the shared LR) or a 2-point CALIB screen {×1, ×m*} for PF; otherwise register "PF ≈ FA by construction" as K5's interpretation | SHOULD (high) | CHANGE or DISCLOSE | 0–2 CALIB runs | PF ×2, K5, CPF | Yes — before any PF run/read. Feasible (PF not launched) |
| F7 | Register the FedProx-under-Adam mechanism (§3.4-6) as the interpretation of K4 / FP − FA before the FP read | SHOULD | DISCLOSE | 0 | K4 | Text registration before the FP read |
| F8 | CALIB-disjoint MAIN diagnostic for K2/K17/K20 (MAIN_VALIDATION users outside ranks [0, 32,768)) beside the full-MAIN value | SHOULD | ADD (analysis) | ANA-script change only | K2, K17, K20 | Yes — into ANA_CONTRASTS / freeze before the VALIDATION read |
| F9 | Quote the matched central CALIB reference from the registered SMALL mini-ladder record (DL 143) in the ladder register instead of "≈ 0.11–0.12 INFERENCE" | SHOULD | CHANGE (text) | 0 | ladder figure / text | No (a reference already read at registration) |
| F10 | Disclose the CALIB→MAIN transfer (N ×4, rounds ×4, same cohort) and the n²-influence of heavy users; NICE: a per-client-uniform-weight (or FedNova) FA sensitivity on CALIB | DISCLOSE-ONLY (+ NICE run) | ADD (text) | 0 (+1 CALIB run) | thesis | Run: register first |
| F11 | Disclose the 0.002 stopping-rule tension (≈ 7 % relative; cumulative sub-threshold gains discarded); keep the rule | DISCLOSE-ONLY | KEEP | 0 | ladder text | Already read (S1b) — cannot change |
| F12 | Participation realism: disclose sweep + uniform pre-training drop-out as optimistic; NICE: availability-biased (activity-weighted or FedScale-trace) sampling sensitivity on CALIB | DISCLOSE-ONLY (+ NICE) | ADD | 0 (+1 run) | thesis | Run: register first |
| F13 | Keep S7/S8 as the deployment branch; FA_WARM re-screens only m. NICE: one FedAvgM-at-m*/4 warm cell (warm starts favour momentum at small LR) | NICE | KEEP / ADD | 1 CALIB run | FA_WARM | Register before the S7 read |
| F14 | Exclusions (SCAFFOLD, FedBN, Mime at full tables, FedYogi, sampled softmax) with the reasons of §3.4-12 in one thesis table; Mime noted as feasible under S8 | DISCLOSE-ONLY | KEEP | 0 | thesis | No |
| F15 | Label fix: `fl_rows` `optimizer_lifecycle` should follow the client optimiser (SGD_M_FRESH_PER_VISIT) | NICE | CHANGE (code, labels only) | 15 min + test | S3 SGD rows (metadata only) | No numeric effect; code review only |
| F16 | CLIENT_CONTRACT L240 wording vs 06K (θ_pre is from non-MAIN users) | DISCLOSE-ONLY | CHANGE (text via amendment note) | 0 | contract consistency | Note only |
| F17 | Implementation: aggregation, recentring, evaluation identity, buffers, determinism | — | **KEEP** | — | — | — |

Priority if only three things can be done before the S2 read (OPINION): **F1, F2, F4**. They are cheap, they must be registered before
S2 is read, and without them the thesis sentence "server momentum / adaptive optimisation did not help FL here" could be an artefact of
the grid.

---------------------------------------------------------------------------------------------------------------------
## 5. Score: **6 / 10**

The simulation engineering is excellent: deterministic, auditable, with no bug and no evaluation mismatch found. The ladder is
pre-registered with unusual discipline, and the project diagnosed its own under-tuning and is fixing it in a principled order. That is
worth 8–9 on its own.

The registered S0 recipe, however, is a pre-2019 configuration placed on the hardest possible parameterisation: fresh client Adam,
server LR 1, the central LR shape on rounds, random 158k-row tables and full softmax. At 20–25 % retention the headline would
currently be about that recipe more than about federation.

The ladder has three gaps that could leave the answer unconvincing:
1. FedAvgM is screened only at an effective LR ≈ 10× past the FedAvg optimum.
2. The canonical client-SGD × FedAdam cell can be skipped by the sequential order.
3. The literature's direct remedy for sparse-table dilution (FedSubAvg-style aggregation) is absent.

In addition, the "lower bound" framing conflicts with the early best marks, and PF is inert by construction. All of these are cheap and
still registrable, because S2 is unread. With F1–F4 and F6 adopted the part would rate 8.

---------------------------------------------------------------------------------------------------------------------
## 6. UNVERIFIED items and open questions
- **Unconfirmed citation details.**
  - The Bonawitz et al. 2019 drop-out percentage and devices-per-round figures (REPORTED by REALISM_AUDIT; not confirmed today).
  - Gboard's "Nesterov 0.9, server LR 1" (REPORTED; not re-read).
  - The Reddi et al. table values (REPORTED).
  - LEAF citation; venues of arXiv 2504.05313, 2406.05387 and 2505.12453.
- **Estimates I could not measure.**
  - The 0.65 M CALIB decision count.
  - The coherence ≈ 0.25–0.3.
  - The random-walk magnitude on tail rows and the PF displacement bound.
  All are INFERENCE and order-of-magnitude only; F5 would measure the mechanism.
- **Open questions.**
  - Did the early best marks come from a declining curve or a flat one? I did not read the per-mark values. The ×1 max-of-12 exceeding
    PRACTICAL_BEST by < 0.0002 shows a flat tail for ×1 only.
  - Row-level identity of the FL client batches with the central training batches (same provider class and `history_policy`; not
    byte-checked by me).
  - Whether the SMALL mini-ladder's CALIB value is indeed ≈ 0.11–0.12 (F9).
  - Whether the S2 runs launched from `code_ds3s2` include `server.py` 8c94ee53 (REPORTED DL 218), i.e. v0 = τ² (06G said v0 = 0;
    DL 220 says the correction was registered before any S2 read).

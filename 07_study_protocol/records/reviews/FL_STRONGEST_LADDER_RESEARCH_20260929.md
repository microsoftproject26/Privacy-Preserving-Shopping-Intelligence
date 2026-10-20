# The strongest REALISTIC federated result — diagnosis, method catalogue, no-backtrack LADDER, thesis design, 48–72 h order

- **Author:** Claude Fable 5.1 (federated-optimisation / production cross-device FL / federated recommendation / thesis design), READ-ONLY
  research session; this file is the only artefact written. No server access, no installs, no git. **No RETRO_TEST row and no MAIN
  RESULT_ROWS value was read.** The only downstream values used are the three registered CALIB screen reads (06B §1): FA on CALIB,
  full 6.0 EFE, CALIB_VALIDATION client-macro MRR@20 at PRACTICAL_BEST = **×0.5 0.020263, ×1 0.023473, ×2 0.025562**.
- **Written:** 2026-09-29, 12:20Z–13:00Z. Server state at 12:12Z (DL 198, coordinator): C_FULL s2026 at 1.0 EFE (GPU1, shared),
  C_FULL s2027 at 0.5 EFE (GPU0, 35 % util) → each ≈ 30 h; the 06F screen ×4/×8/×16 RUNNING on GPU1 (results ≈ 13:00–14:00Z);
  LADDER_S1a FA MAIN s2026 (FedAvg, ×2) running on GPU0 since 12:13Z; GPU0 has spare capacity.
- **Owner's request:** the strongest federated result with a small accuracy loss; every method researched; staged for the thesis ("when we
  did X, FL improved by Y"); an order that never backtracks; the FL must stay REALISTIC for a Microsoft Edge deployment (data never
  leaves the device; only model updates leave; ≤ 170 MB per visit; realistic participation) — realism is the top priority.
- **Tags:** MEASURED (file:line, a run record, or a number read from a cited paper today), REPORTED (a project document's number),
  INFERENCE (my estimate), OPINION, PLAN, UNVERIFIED (not confirmed on the web today). Every ETA is INFERENCE.
- **Read today:** REALISM_AUDIT_20260929 (fully), FABLE_THESIS_RESEARCH_REVIEW_20260928 (§C.6, §D, §E), FABLE_EXPERT_DEEP_REVIEW_20260928,
  amendments 06F, 06G, 06H, 06I, PREG1_LOCK_VALUES T2, `code/fl/{runtime,client,server,schedule,participation,aggregate,pool}.py`,
  `v4int/launch_fl.py` (KNOBS, recipe path), `v4models/sasrec_v2.py` (tables, recentering), `fl/adapter.py` (param groups),
  MASTER_PLAN_V5 (+V5.1), DECISION_LOG 180–200, RESUME_HERE §2l–§2m, `company/registers/FL_IMPROVEMENT_LADDER.md`, task packets
  FL-10 and DATA-10. Web: ~25 searches + the Reddi et al. 2021 PDF appendix (pages 26–38 read as images).
- **Stage names** follow the existing register `company/registers/FL_IMPROVEMENT_LADDER.md` and DL 199: S0 (as registered, ×1), S1a
  (06B grid), S1b (06F), S2 (06G), S3 (06I §3). This file adds S4–S8 and the extension rules; everything new must be registered
  (amendment 06J/06K) BEFORE its first CALIB read.

---------------------------------------------------------------------------------------------------------------------
## 1. Diagnosis — why FA sits at ≈ 20–23 % of the matched central value on CALIB

Ranked by (size of effect × certainty). Evidence: code lines, the registered CALIB curve, and verified literature.

### 1.1 The mechanics that fix the ceiling (MEASURED from code + arithmetic)
- A client holds 5–100 TRAIN examples (mean 19.7 = 2,587,776 / 131,072); `B16 × 2 passes` → **1–14 AdamW steps per visit, ≈ 4 typical**
  (`launch_fl.py:97` KNOBS; `fl/client.py:185–225`). The optimiser is **re-created every visit** (`fl/client.py:279`), so every visit
  starts from Adam's bias-corrected first step = **lr · sign(g)** per coordinate (Balles & Hennig 2018 — Adam's direction is the sign,
  its magnitude a variance estimate; verified: [PMLR 80](https://proceedings.mlr.press/v80/balles18a/balles18a.pdf)). With the
  full-K softmax, **every one of the 158,486 output rows receives a non-zero gradient at every step**, so each visit moves all
  10.1 M output-table coordinates by ≈ ±lr·steps, regardless of how tiny the true gradient on an irrelevant item is.
- The server applies the n_consumed-weighted mean of the uploaded states with **server LR 1 and no momentum** (`fl/server.py:113–116`,
  `fl/aggregate.py:3, 129–130`). Therefore **one server step can never move a coordinate further than the average client moved it
  locally** (≈ 4 · lr = 0.0028 · m at the SMALL peak 0.000711 × multiplier m; embedding init std 0.02, `sasrec_v2.py:243, 255`).
  The n-weighted mean of ≈ 57 sign-like vectors from heterogeneous clients keeps only their consensus component; on non-consensus
  coordinates the mean shrinks to ≈ 1/√57 ≈ 0.13 of one client's step and its direction is noise.
- **Sparse-row dilution.** An input-table row of item i is updated only by the k clients who viewed i; the server averages their
  deltas with the other clients' zero deltas: the row moves ≈ k/57 of one local update per round. With 158,486 items and 57 clients
  × 20 items per round, most rows move at most 1/57 of a local step per round they are touched at all.
- **The LR shape.** The central warm-up/decay shape is mapped onto rounds (`fl/schedule.py:38` KNOTS; `fl/runtime.py:236–243`): peak
  only for EFE 0.02–2.0 = rounds 23–2,276 of 6,827 (33 %); linear decay to 0.1× over 2.0–3.0; **0.1× for 3.0–5.5 (42 % of rounds)**;
  cool-down to 0.01×. Summing the fractions, the run has ≈ 0.47 × 6,827 ≈ **3,200 peak-equivalent rounds**, not 6,827.
- **Budget.** 6.0 EFE = 3 visits per client = 6,827 averaged updates (`launch_fl.py:125–127`). A deployment is not bounded this way
  (Gboard: 3,000 rounds × 100–500 devices over 4–5 days, ≈ 400 sentences per client per epoch; Hard et al. 2018, verified in the
  realism audit). 06I §6(a) already registers the equal-exposure result as a LOWER BOUND.

### 1.2 Ranked causes
| Rank | Cause | Evidence | Certainty | Fixed by stage |
|---|---|---|---|---|
| 1 | **No server optimiser (server LR 1, no momentum)** — the per-round progress is capped at the mean local displacement; the still-rising LR curve (+16 %, +9 % per doubling) is the signature of a server step that is too small | code above; CALIB curve (DL 181); Reddi et al. 2021 Table 10 on the sparse-gradient Stack Overflow NWP task: **FedAvg 20.5, FedAvgM 24.5, FedAdagrad 24.4, FedAdam 25.7, FedYogi 25.7** (test accuracy, MEASURED from the PDF p.30); Hsu, Qi & Brown 2019 introduce server momentum FedAvgM for non-IID (verified [dblp](https://dblp.org/rec/journals/corr/abs-1909-06335.html)); Gboard uses Nesterov momentum 0.9 / server LR 1 (Hard 2018 §5.2, verified in the audit) | HIGH | S2 |
| 2 | **Fresh client AdamW with ≈ 4 steps → sign-like updates on all 158k output rows** (noise floor, dense ±lr on 10 M coordinates per visit); production clients run SGD (Hard 2018; McMahan et al. 2018; Reddi 2021 App. D.1 "mini-batch SGD throughout", MEASURED p.27); local adaptivity without correction causes objective inconsistency (Wang, Xu, Garrett, Charles, Liu & Joshi 2021, [arXiv 2106.02305](https://arxiv.org/pdf/2106.02305), verified) | HIGH | S3 |
| 3 | **Local LR too small for FedAvg-η1** (winner at the top of the 06B grid) | DL 181; Reddi 2021 App. E.4: for FedAvg/FedAvgM client and server LR are inversely related and "must be tuned simultaneously"; for adaptive server optimisers "tuning the client learning rate is more important" (MEASURED p.35) | HIGH (already observed) | S1b (running) |
| 4 | **Central LR shape on rounds** — 53 % of the rounds are effectively lost | `fl/schedule.py:38`; Reddi 2021 App. E.6: a staircase client-LR decay improves every federated optimiser modestly "in communication-limited settings" (MEASURED p.36) — a constant peak with a short cool-down is the production shape | HIGH | S3 (FL_CONST) |
| 5 | **Random initialisation of two 158k-row tables with 3 visits per client** — every production system warm-starts (Gboard pre-trains on public data before FL/DP-FL: Xu et al. 2023, verified in the audit; "Where to Begin?" Nguyen et al., ICLR 2023: pre-training "enables the training of more accurate models (up to 40 %) than is possible with random initialization" and shrinks the FedAvg–central gap, [arXiv 2206.15387](https://arxiv.org/abs/2206.15387) verified; Stremmel & Singh 2020 pre-train federated NWP models centrally, [arXiv 2005.04828](https://arxiv.org/abs/2005.04828) verified) | HIGH for the effect; it changes the research contrast (see §3 S7) | S7 |
| 6 | **Equal-exposure budget** (3 visits per client) | 06I §6(a); Hard 2018 | HIGH (framing) | S5 (FA_X2, descriptive) |
| 7 | **Sparse item tables + dilution** (rank-1.1 above) — per-coordinate server adaptivity is exactly the remedy Reddi et al. give for sparse gradients | Reddi 2021 §5 / Table 10 | HIGH | S2 (FedAdam/Yogi) and S8 (freeze the tables) |
| 8 | **Unequal local steps (1–14) with n-weighting** → the aggregate is biased toward heavy users' objectives (objective inconsistency; Wang, Liu, Liang, Joshi & Poor 2020 FedNova, NeurIPS, [arXiv 2007.07481](https://arxiv.org/abs/2007.07481) verified) | code | MEDIUM, small effect here | not staged (see §2.8) |
| 9 | Non-IID / client drift | bounded by the tiny local displacement (4 steps) — second order here; FedProx μ = 0.01 is not expected to help (INFERENCE; the FP arm measures it) | MEDIUM | — |
| 10 | Cohort 64 with 10 % drop-out (≈ 57 survivors) | more clients per round at equal EFE = fewer rounds; with a server optimiser per-round progress is bounded, so fewer rounds lose (Charles et al. 2021, NeurIPS, diminishing returns of large cohorts; verified in the deep review) | LOW gain at equal EFE | not staged (§2.9) |
| 11 | Weight decay on untouched rows | AdamW's decoupled decay hits every row of a table that has a (dense) grad tensor: p ← p·(1 − lr·wd) = 1 − 7e-9·m per step; ≈ 12 steps per client over the run → ≈ 1e-7·m relative. **Negligible — NOT a cause** (arithmetic from `fl/client.py:156`, `fl/adapter.py:225–228`) | — | — |
| 12 | Gradient clip 1.0 | irrelevant for Adam (scale-invariant up to eps); BINDING for client SGD (max step = lr × 1.0) — the S3 SGD grid must be read with that in mind (disclose) | — | S3 note |
| 13 | D64 capacity (C_SMALL ≈ 0.11–0.12 on CALIB, INFERENCE) | the FL ceiling is C_SMALL, not C_BIG; the retention headline R(FA/C_BIG) carries the size cost too (K1 isolates it) | — | — |

**Bottom line (OPINION):** the registered FA is a correct FedAvg but a *pre-2019* recipe on the hardest possible parameterisation
(two 158k-row tables, full softmax, random init, 3 visits per client). Causes 1–4 are optimiser/schedule choices that production FL
made a decade ago; fixing them is legitimate tuning on CALIB. Cause 5 (warm start) is what actually makes the loss "small" in
production, and it must be presented as a separate ladder branch because it adds server-side data (from non-MAIN users).

---------------------------------------------------------------------------------------------------------------------
## 2. Catalogue of candidate methods

Per method: what / verified citation / expected effect HERE / Edge realism / privacy / implementation cost in `code/fl` / GPU cost.
CALIB screen run = 1,707 rounds ≈ **2.7 h in one process at ≈ 20 % of an A40 and one CPU core** (MEASURED, MASTER_PLAN V5.1 §B);
MAIN run ≈ 10.7 h per process; 5–8 processes per card (INFERENCE; the loop is CPU-bound).

### 2.1 Client LR and its schedule over rounds
- **Multiplier m** (S1a/S1b, registered): m ∈ {0.5 … 32}. Expected: still rising to ×4–×16, divergence possible at ×16–×32 with fresh
  Adam (INFERENCE). Cost: running. Realism: neutral.
- **FL_CONST shape** (S3, 06I §3): warm-up 1 % of rounds → constant peak → cool-down over the last 10 % to 0.1×. Reddi 2021 use constant
  client LRs and show modest gains from a staircase decay (E.6). Expected here: recovers the ≈ 53 % of rounds spent at ≤ 0.1× → +20–50 %
  relative in this under-trained regime (INFERENCE). Cost: in FL-10 per DL 200; one CALIB run per client optimiser.
- **Cosine per round**: no advantage over FL_CONST + cool-down at this budget; excluded (one shape keeps the ladder small).

### 2.2 Client optimiser
- **SGD (+ momentum 0.9, fresh per visit — stateless)** (S3, 06I §3; lr ∈ {0.01, 0.1, 1.0}, clip 1.0 binding): the production client
  (Hard 2018; McMahan 2018 "fixed learning rate of 6.0" for the LSTM; Reddi 2021 D.1). Removes the dense sign noise on the output table
  (SGD moves an irrelevant item row by its true tiny gradient). Expected: the largest single client-side gain once a server optimiser
  exists (INFERENCE). Realism: high. Privacy: unchanged.
- **Adam with restarts (current)**: keep as the incumbent in S3 (one run AdamW-fresh × FL_CONST) so the client-optimiser effect is
  measured cleanly.
- **Persistent client Adam state across visits**: stateful clients → **excluded** (cross-device clients are stateless: Kairouz et al. 2021;
  Reddi 2021 §1; SCAFFOLD's control variates are the textbook example — Karimireddy et al., ICML 2020, [arXiv 1910.06378](https://arxiv.org/abs/1910.06378) verified;
  Reddi's App. E.1 note that SCAFFOLD is "slightly worse than FedAvg" with stale control variates, MEASURED p.30).
- **MIME** (server-side momentum/Adam statistics broadcast and applied inside every local step; stateless clients; Karimireddy et al.
  2020, [arXiv 2008.03606](https://arxiv.org/abs/2008.03606) verified): realistic in principle, but the download doubles (a 20.7 M
  momentum vector) → ≈ 250 MB per visit > the 170 MB cap → **excluded by the realism budget** unless S8 (frozen tables) is adopted.

### 2.3 Server optimiser
- **FedAvgM** (server momentum β 0.9, server LR 1; Hsu et al. 2019; Gboard's Nesterov 0.9 / LR 1, Hard 2018; Xu et al. 2023 Alg. 1):
  ≈ 60–100 lines in `fl/server.py` (buffer v = βv + Δ̄; θ ← θ − v; checkpointed), β = 0 must equal FedAvg bitwise. Reddi Table 10:
  +4.0 points over FedAvg on SO NWP (20.5 → 24.5). **DP-safe** (post-processing; no per-coordinate noise normalisation — the 06H concern
  does not arise). Realism: the most production-proven choice. Cost: 1 CALIB run. **Add to S2 (not in 06G).**
- **FedAdam** (S2, 06G registered: β1 0.9, β2 0.99, τ 1e-3, η_s ∈ {0.01, 0.1}, m ∈ {1, m*}): per-coordinate normalisation is exactly what
  the sparse tables need (Reddi §5). **Warning from Reddi's own grids (MEASURED, Table 8/9 p.29 and Fig. 11 p.34):** on SO NWP the optimum
  is server LR **10⁻²**, client SGD LR 10⁻⁰·⁵, **τ = 10⁻⁵**; server LR ≥ 10⁻¹ gives ≈ 0 accuracy (the top rows of the FedAdam heat-map are
  0.0–3.x). So η_s = 0.1 will very likely fail here and 0.01 will sit at the LOWER edge of 06G's grid. **Register now (06J), before the
  S2 read: if the best FedAdam η_s is 0.01, extend once with η_s ∈ {0.003, 0.001} at the same m; if FedAdam wins, one run at τ = 1e-5.**
  06H's DP risk (noise-dominated coordinates random-walk by ≈ η_s per round) is real; FedAvgM is the fallback for the DP arms
  (06H DP_OPT_SENS already registered).
- **FedYogi / FedAdagrad**: FedYogi ≈ FedAdam on SO NWP (25.7 / 25.7), Adagrad lower (24.4) → not worth separate runs; excluded with this
  reason.
- Server LR > 1 for plain FedAvg (η ∈ {2, 4}): a cheaper version of the same idea, but momentum dominates it in Reddi's grids; excluded.

### 2.4 Local computation and cohort
- **Passes ∈ {1, 2, 4}, local batch ∈ {8, 16}** (S4): with SGD, steps × lr is what matters, so passes act like an LR change; expected small
  (±0.005 absolute, INFERENCE). Bounded: screened at the fixed selected LR, no LR re-screen (disclosed). Field-guide advice is to tune
  jointly (Wang et al. 2021, arXiv 2107.06917, verified in the thesis review) — the bounded version is the price of no backtracking.
  Cost: 3 CALIB runs; a small KNOBS-override flag (FL-13).
- **Clients per round 64 → 256** at equal EFE = 4× fewer rounds; per-round progress is bounded by the server optimiser → expected to LOSE
  at equal EFE (Charles et al. 2021). Excluded from the ladder; a descriptive sensitivity if time (NICE).
- **Drop-out 10 %**: negligible; keep (realism).

### 2.5 Training budget
- **FA_X2 = 12.0 EFE** (S5; 06D §5, promoted by 06I §4 and the owner): descriptive; ≈ 21 h per process on MAIN. Expected +10–25 % relative
  if the still-improving flag fires (INFERENCE). Realism: production runs far more rounds; disclose as "rounds are not the scarce resource".

### 2.6 Warm start from a model pre-trained on OTHER users (Gboard-style) — S7
- **What:** pre-train SASREC_D64_B2 centrally (the unchanged SMALL recipe) on the NON-MAIN users of the 5–100 pool (TIEBREAK band minus
  MAIN = 1,217,610 users expected; DATA-10 packet, running), take the **endpoint_6.0** checkpoint (no selection, DATA-10 §3), start FL on
  MAIN from it (θ_pre replaces θ0; `Server(init_sha256=…)` accepts a registered init hash). Also evaluate **PRE alone** on MAIN_VALIDATION
  (the "cloud model of consenting users" serving everybody) and **FT_PRE** (PRE + on-device fine-tune, no FL) as the device alternative.
- **Citations (verified):** Nguyen et al. ICLR 2023 (above); Xu et al. 2023 Gboard DP LMs pre-trained on public data (audit); Stremmel &
  Singh 2020; Chen, Tu, Li, Shen & Chao 2022 "On Pre-Training for Federated Learning" ([arXiv 2206.11488](https://arxiv.org/abs/2206.11488),
  arXiv verified, venue UNVERIFIED) find pre-training closes much of the FedAvg–central gap.
- **Expected here (INFERENCE):** PRE alone ≈ 0.10–0.13 on MAIN_VALIDATION (a D64 model on 10× C_SMALL's data); FA_WARM ≥ PRE by a
  small Δ; DP arms from θ_pre lose far less (the model must move less under noise). This is the stage that makes the loss "small".
- **Realism: HIGH and the honest Edge story** — Edge has consenting (optional-diagnostic-data) users; the cloud model trains on them; the
  non-consenting users contribute only through FL. **It changes the contrast:** most of FA_WARM's quality is PRE's. The thesis therefore
  reports the PRE row, FA_WARM − PRE (paired CI) as "what federation adds on top of the consenting-user model", and keeps the from-scratch
  ladder (S0–S5) as the algorithmic answer. Never warm-start from C_FULL (it contains the MAIN users' own data → the FL contribution
  would be confounded).
- **Privacy:** the pre-training data are other users' data on the server (the L0 tier of the ladder); MAIN users' data still never leave.
- **Cost:** DATA-10 manifest (CPU 1–2 h) + PRE run (INFERENCE 8–12 h on a shared A40: D64 is ≈ 3–4× cheaper per exposure than the D256
  C_FULL that takes ≈ 30 h) + a 2-run CALIB LR check {m*, m*/4} (fine-tuning regimes prefer smaller LRs; INFERENCE) + MAIN FA_WARM 10.7 h
  + FT_PRE (device lane). Code: FL-12a `--init-ckpt` (registered PRE hash) ≈ 0.5 day Opus.

### 2.7 Embedding-table strategies (the 98 % of the payload)
- **Frozen pre-trained item tables + FL on the encoder only** (S8; needs S7): shared set = blocks, norms, pos_embed, side embeds
  (≈ 0.42 M params incl. or excl. `output_bias`; freeze the bias too, or disclose its leak) → **≈ 1.7 MB per direction per visit**
  (from 165.7 MB), the item-set leak of the input table disappears (only encoder deltas leave), DP noise norm falls by √(20.7/0.42) ≈ 7×
  (d = 0.42 M), compute per visit falls (no table gradients). Analogous designs: Efficient-FedRec keeps the large news model on the server
  and federates a light user model (Yi et al., EMNLP 2021, [ACL](https://aclanthology.org/2021.emnlp-main.223/) verified); FedRecon keeps
  user embeddings local and item embeddings global, stateless, deployed in Gboard (Singhal et al., NeurIPS 2021, [arXiv 2102.03448](https://arxiv.org/abs/2102.03448) verified).
  Expected quality: slightly below FA_WARM (INFERENCE; the encoder adapts, the catalogue stays PRE's). Realism: the highest of any stage —
  it is the only configuration with a phone-class payload. Code: FL-12b (a FROZEN manifest rule: not trained, not uploaded, downloaded
  once; byte accounting; tests). Cost: 1–2 CALIB runs + 1 MAIN run.
- **Low-rank / LoRA / FedPara on the tables** (Hyeon-Woo et al., ICLR 2022, [arXiv 2108.06098](https://arxiv.org/abs/2108.06098) verified):
  at d = 64 a rank-8 factorisation of a 158k × 64 table saves only ≈ 5× and changes the model class → **excluded** (future work).
- **Tied input/output tables**: halves the sparse parameters but is a different model from C_SMALL (untied, `sasrec_v2.py:202`) → excluded
  (would break the same-model retention contrast).
- **Sampled softmax / negative sampling on the client** (gSASRec, Petrov & Macdonald RecSys 2023; Klenitskiy & Vasilev RecSys 2023 show
  full CE ≥ sampled for SASRec centrally; both verified): sparse output gradients → sparse deltas → fewer bytes and no dense sign noise,
  BUT a different objective from C_SMALL's full CE → **excluded from the ladder** (future work; if ever adopted, apply to C_SMALL too).
  FedAwS (Yu, Rawat, Menon & Kumar, ICML 2020, [PMLR](https://proceedings.mlr.press/v119/yu20f.html) verified) is the reference for
  clients that only hold positives — not our case (clients have the full output table).
- **Sending only touched rows**: cuts bytes 100× but leaks the item set by construction (already leaks: 06I §7a) — only with secure
  aggregation (Bonawitz 2017) — future work.
- **Compositional codes / hashing for the tables** (Shu & Nakayama, ICLR 2018, [arXiv 1711.01068](https://arxiv.org/abs/1711.01068);
  Xia et al. 2023 on-device session rec, verified in the thesis review): a different model → future work / device-size discussion.
- **FA_Q8** (registered): orthogonal compression point; keep.

### 2.8 FedProx / SCAFFOLD / FedNova / MIME
FedProx (registered FP, μ 0.01): stateless, realistic, expected no gain here (drift is second order). SCAFFOLD: stateful → excluded
(§2.2). FedNova normalised averaging (unequal local steps): cheap, stateless, but the server optimiser already normalises per coordinate
→ NICE at most. MIME: excluded by bytes unless S8.

### 2.9 Personalisation (rungs on top of the selected FA — registered arms, no new screens)
FT_FA (local adaptation after download; Yu, Bagdasaryan & Shmatikov 2020), PF (personal vector p_u with device-local persistent state —
the FedRecon/FedRep family: Collins et al., ICML 2021, [arXiv 2102.07078](https://arxiv.org/abs/2102.07078) verified; Pillutla 2022
partial personalisation), and FT_PRE (S7). All realistic (nothing uploaded). They are reported as the last rungs: "FA_final → +FT".

### 2.10 Distillation
FedDF (server-side ensemble distillation on unlabelled data; Lin, Kong, Stich & Jaggi, NeurIPS 2020, [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html) verified) and KD of C_FULL into SMALL (KD_SMALL, deferred):
realistic with the non-MAIN population as the server's data, but they raise C_SMALL as well and add code → future work (06D tiers).

### 2.11 Recent federated sequential recommendation (2024–2026)
The survey arXiv 2504.05313 (verified) and FELLAS/PTF-FSR-class methods federate SASRec/GRU4Rec on ≤ 10k clients with sampled metrics;
PTF-FSR reports 1.2–2.4 KB uploads vs 1.6–14.8 MB for Fed-SASRec by keeping the item side on the server — the same direction as S8.
None reports a full-catalogue, temporally split, 131k-client result; the ladder above is what a production team would run.

---------------------------------------------------------------------------------------------------------------------
## 3. THE LADDER — ordered, no backtracking, each stage built on the previous winner

Protocol of every CALIB screen (unchanged from 06F §1): FA arm, CALIBRATION role (32,768 users), seed 2026, 1,707 rounds (= 6.0 EFE),
64/round sweep, 10 % drop-out, 12 marks, PRACTICAL_BEST on CALIB_VALIDATION E2E client-macro MRR@20. Selection = mechanical, registered
before the read; ties within |Δ| ≤ 0.0005 go to the configuration with fewer deviations from the registered default (06I §3 order).
**The joint winner of S1b–S4 is written to `contracts/fl/FL_TUNING_SELECTION.json` and becomes "FA" for every FL arm (06I §2).**
Where a stage changes a component that interacts with the LR, the LR re-check is INSIDE the stage's grid (S2: m ∈ {1, m*}; S3: the SGD
LR grid is a fresh grid by construction; S7: {m*, m*/4}); S4 deliberately has no LR re-check (bounded, disclosed).

| Stage | Change (built on the previous winner) | Screen on CALIB (runs) | Selection rule | Expected gain (INFERENCE) | Cost (process-h / wall on 2× A40, screens in parallel) | Registration |
|---|---|---|---|---|---|---|
| **S0** | FedAvg as registered, m = 1 | done: 0.023473 | — | reference | 0 | 06/06B |
| **S1a** | m ∈ {0.5, 1, 2} | done: 0.020263 / 0.023473 / **0.025562** | 06B rule | +8.9 % (MEASURED) | 0; S1a-MAIN descriptive running | 06B; DL 198 |
| **S1b** | m ∈ {4, 8, 16} (+32 if 16 is the raw best) | RUNNING, values ≈ 13–14Z | 06F §1 | 0.026 → 0.030–0.040 | 3 × 2.7 h (+2.7 h) / ≈ 0–3 h left | 06F |
| **S2** | server optimiser at m*: FedAdam η_s ∈ {0.01, 0.1} × m ∈ {1, m*} (06G) **+ FedAvgM β 0.9, η 1 at m*** (new) **+ edge rule:** if the best FedAdam is η_s = 0.01 → {0.003, 0.001}; if FedAdam wins → one run τ = 1e-5 | 5 runs (+ ≤ 3) | 06G §3 order; ties FedAvg → FedAvgM → FedAdam (simpler first) | → 0.045–0.065 (Reddi: FedAvg→FedAdam +25 % rel. from a tuned start; here from an untuned start, more) | 5–8 × 2.7 h / ≈ 3 h (+3 h if extended) | 06G + **06J §1** |
| **S3** | LR shape FL_CONST × client optimiser {AdamW-fresh, SGD-m0.9 lr 0.01 / 0.1 / 1.0} at the S2 winner (server opt, m*) (06I §3) **+ edge rule:** if SGD lr 1.0 is the raw best → {3.0, 10.0}; if 0.01 → {0.003} | 4 runs (+ ≤ 2) | 06I §3 joint rule | → 0.055–0.075 | 4–6 × 2.7 h / ≈ 3 h | 06I §3 + **06J §2** |
| **S4** | local computation at the S3 winner: passes 1, passes 4, batch 8 (one-at-a-time) | 3 runs | best vs incumbent; ties → incumbent; **no LR re-screen (disclosed)** | ±0.005 | 3 × 2.7 h / ≈ 3 h | **06J §3** |
| → | **FL_TUNING_SELECTION.json** = the S4 winner → MAIN FA ×2 seeds + every FL arm; S_CAL → FA_1024 → DP8 → DP1 | — | 06G §3 / 06I §3 | — | ≈ 13 × 10.7 h / ≈ 14–18 h on 8–10 lanes | registered |
| **S5** | budget: FA_X2 (12 EFE) at the selected recipe, MAIN s2026 | none (MAIN descriptive) | none (framing) | +10–25 % rel. if still improving | 21 h process | 06D §5 / 06I §4 |
| **S6** | personalisation rungs on FA: FT_FA (selected on UNSEEN_VAL), PF | none (registered) | 06A §1 | FT typically +; harmed-client share reported | registered lanes | 06/06A |
| **S7** | **warm start (deployment branch):** PRE = SMALL recipe on 1,217,610 non-MAIN users (endpoint_6.0); FA_WARM = the S4 recipe from θ_pre; PRE-alone and FT_PRE as references | 2 runs: m ∈ {m*, m*/4} from θ_pre | best; ties → m* | PRE ≈ 0.10–0.13 on MAIN_VALIDATION; FA_WARM − PRE small positive; the "small loss" result | PRE 8–12 h; 2 × 2.7 h; MAIN 10.7 h; FT_PRE device lane | **06K** (DATA-10 running; owner decision pending per 06I §8) |
| **S8** | **frozen item tables (realism branch):** FA_WARM with item_embed / output_embed (/bias) FROZEN at θ_pre; encoder-only FL; 1.7 MB per direction; DP8_WARM_FROZEN if time | 1–2 runs (m from S7; optionally m*/4) | best; ties → S7's m | ≈ S7 − small; bytes ÷ 100; DP noise ÷ 7 | 1–2 × 2.7 h; MAIN 10.7 h (+ DP 10.7 h) | **06K** |

**Why this order.** (expected gain × certainty) / cost: S2 (server optimiser) is the largest certain gain per 2.7 h; S3 depends on S2
(the client optimiser's right LR depends on the server step); S4 is small and must come after the LR/optimiser are fixed; S5/S6 are
framing and registered arms; S7/S8 add server-side data (a different privacy tier) and code, so they come last and live on their own
branch with their own references — but their pre-training run can start as soon as DATA-10 lands, in parallel with S2–S4, because it
does not depend on the FL recipe. Running S3 concurrently with S2 (06I allows it with one combination run) would save ≈ 3 h of wall but
break "built on the previous winner"; with both cards able to take 5–8 processes, the sequential waves cost nothing in GPU-hours and
only ≈ 6 h of wall. **Recommendation: sequential waves S2 → S3 → S4.**

**Stopping rules (register in 06J):** (i) the registered tie band 0.0005 stands for S1b–S4; (ii) the paired CALIB half-width is ≈ 0.0017
MRR (INFERENCE: 0.00096 on 40,764 MAIN users scaled to ≈ 12.8k CALIB eval users), so a stage whose best candidate gains < 0.002 over the
incumbent is recorded as "no measurable gain" and the ladder moves on; no further optimiser stage is added after S4 whatever S4 shows;
(iii) any candidate that breaks the realism budget (bytes per visit > 170 MB, stateful clients across rounds, server access to MAIN
users' data, a model different from C_SMALL) is out of scope regardless of its CALIB value; (iv) a run with a non-finite loss/metric is
excluded and recorded (06F).

**Realism check of every stage:** all of S1–S6 keep 165.7 MB per visit, stateless clients (FedAdam/FedAvgM state is server-side; SGD
momentum is per visit), no server access to client data; S7 adds server-side training on OTHER users (disclosed tier L0→L2); S8 is the
only stage with a phone-class payload.

---------------------------------------------------------------------------------------------------------------------
## 4. Thesis design of the ladder

1. **Two evidence layers.** (a) The CALIB tuning curve — every screened point of S1a–S4 (and S7/S8) as small markers, the stage winners as
   large markers, joined as a staircase; labelled "tuning on CALIB (max-of-grid, descriptive)". (b) The MAIN staircase — one descriptive
   MAIN run (seed 2026) per stage winner that differs from the previous stage's winner, plus the confirmatory final FA (2 seeds).
2. **MAIN ladder rows (register in 06J as DESCRIPTIVE, never headline, never used to select):** LADDER_S1a (running), LADDER_S1b (FedAvg,
   m*), LADDER_S2 (server optimiser winner, m*), LADDER_S3 (S3 winner) — only if it differs from FA; FA = the S4 winner (2 seeds,
   confirmatory K2/K17); FA_X2; PRE, FA_WARM, FT_PRE, FA_WARM_FROZEN (seed 2026). ≈ 3 extra MAIN FL runs ≈ 32 process-hours.
3. **"Improvement from stage i to i+1":** Δ_i = MRR(stage i+1) − MRR(stage i) on the same MAIN_VALIDATION users, paired user bootstrap
   (1,000 resamples, the shared W), under BOTH rules (PRACTICAL_BEST and CONTROLLED_BUDGET 6.0), unadjusted 95 % CIs, reported as the
   **ladder contrast family L1…Lk** (descriptive; family size stated; 06D §4 hierarchy untouched: K2 primary stays R(FA/C_BIG)). Also
   report each stage's R vs C_SMALL and vs C_BIG so the reader sees "retained quality" climb. ANA-04 adds the family and the figure.
4. **Figure "FL improvement ladder"** (two panels): left CALIB curve (x = stage, y = MRR@20, central D64 CALIB reference as a line, the
   under-tuned start marked); right MAIN staircase with CIs and, beneath each stage, the realism annotations: bytes per visit, client
   state, server-side data used (none / non-MAIN users), formal guarantee. **Table:** stage | change | CALIB value | Δ CALIB | MAIN value(s)
   | Δ MAIN [CI] | R vs C_SMALL | R vs C_BIG | bytes/visit | realism notes.
5. **Text pattern the owner asked for:** "Extending the client LR (S1b) raised CALIB MRR from 0.0256 to X; adding a server optimiser (S2)
   to Y; replacing the per-visit Adam by SGD with a constant LR (S3) to Z; on MAIN the same steps gave …[CI]; warm-starting from a model of
   consenting users (S7) brought the federated model to W = V % of the cloud model, and freezing the catalogue tables (S8) kept U % at
   1.7 MB per visit."
6. **Pre-registration discipline.** Every stage's grid, rule and edge-extension is in an amendment BEFORE that stage's first CALIB read
   (06F/06G/06I exist; **06J for S2-extension, S3-extension, S4, MAIN ladder rows, the contrast family, the stopping rules — written
   before the S2 read, i.e. before ≈ 17Z today**; **06K for S7/S8 — before the PRE launch**). MAIN runs only of pre-declared stage winners.
   The headline FL row is the pre-declared final stage (06I §2); S0/S1a rows are kept as the as-registered lower bound; the TEST look
   re-evaluates every locked ladder checkpoint once (reeval, 06D §2) and the staircase is shown on TEST as descriptive.
7. **Disclosures:** tuning on CALIB ⊂ MAIN; max-of-grid on CALIB_VALIDATION; one seed for intermediate rungs; the equal-exposure framing;
   S7 uses other users' data (tier); the item-set leak persists in S1–S7; the "FL retains X %" sentence always carries the recipe (06I §6c).

---------------------------------------------------------------------------------------------------------------------
## 5. What NOT to do
- No server-side access to MAIN users' data in any FL arm (the only server-side data is the registered non-MAIN pre-training cohort).
  Never warm-start from C_FULL or C_BIG (both contain MAIN users' data).
- No stateful clients across rounds on 131k devices (SCAFFOLD control variates, persistent client Adam moments, MIME with a 20.7 M
  download beyond the cap). PF's p_u is device-local — fine.
- No tuning or selection on MAIN_VALIDATION, no reading MAIN values before ANA freeze, no grid extension after a MAIN read, no S_CAL
  re-calibration after any DP value is read, no changing the evaluator (hash-pinned), the metric, the split or the checkpoint rule.
- No budget longer than central without the disclosure (FA_X2 is descriptive; never a headline).
- No model changes disguised as FL tuning (tied tables, sampled softmax, hashed embeddings, KD into SMALL) — they break the same-model
  retention contrast; future work.
- No cohort-size stage at equal EFE (it trades rounds for cohort and will read as a loss that is really a budget artefact).
- No concurrent S2/S3 launch "to save wall time" (it forces a combination run and reintroduces backtracking); no reading the S1b value
  before LR_SELECTION_EXT is written by the tool; no interim-tree MAIN runs (DL 197: CALIB screens may use an interim tree, MAIN may not).
- Do not present S7/S8 as "FL from scratch": the PRE row must always stand beside them.

---------------------------------------------------------------------------------------------------------------------
## 6. Literal execution order for the next 48–72 h (T0 = 2026-09-29 12:20Z; all times INFERENCE)

Capacity: GPU1 = C_FULL s2026 (≈ 30 h, ends ≈ 30 Sep 15–18Z) + 3 screens until ≈ 14Z; GPU0 = C_FULL s2027 (35 %, ends ≈ 30 Sep 18Z) +
LADDER_S1a MAIN (ends ≈ 23Z). Each FL process ≈ 20 % GPU + 1 core → plan ≤ 5 extra processes per card while the C_FULLs run, 8 after.

| When | Action | Where | Depends on |
|---|---|---|---|
| T0–T0+1 h | Ops writes **06J** from §3/§4 (S2 FedAvgM run + η_s/τ edge rules; S3 edge rule; S4; MAIN ladder rows; contrast family; stopping rules) — must exist before the first S2 read | CPU | this file |
| T0 | Workers (already resumed, DL 200): FL-10 (FedAdam + select-joint + 06I §3 FL_CONST/client-SGD **+ FedAvgM: `Server.apply` momentum buffer, β = 0 bitwise = FedAvg, resume bitwise, digest invariance**), DATA-10 (PRE manifest + `pretrain_small.py`), INT-07 (TEST path). New small packets: **FL-13** (`--passes {1,2,4} --batch {8,16}` CALIB-only overrides; MAIN refuses unless FL_TUNING_SELECTION carries them; Sonnet, 2 h), **FL-12a** (`--init-ckpt` warm start: Server accepts a registered PRE endpoint hash; run label carries `init=PRE`; Opus, 0.5 d), **FL-12b** (FROZEN manifest rule + byte accounting; Opus, 0.5 d), **ANA-04** (ladder family + figure; Sonnet after ANA-03) | CPU | — |
| ≈ 13–14Z | S1b values → `lr_screen select` → LR_SELECTION_EXT (DL row); if 16 is the raw best → launch ×32 (GPU1, 2.7 h) | GPU1 | 06F |
| ≈ 15Z (when FL-10 lands + local suites; interim tree allowed for CALIB, DL 197) | **Launch S2 wave: FedAdam 4 runs + FedAvgM 1 run at m*** (split 3 on GPU0, 2 on GPU1) | both | LR_SELECTION_EXT, 06J |
| ≈ 16–17Z | DATA-10 manifest build + receipt (CPU); **06K** registered (PRE recipe, endpoint_6.0 init, S7 grid {m*, m*/4}, S8 frozen set, references PRE/FT_PRE, descriptive contrasts) — owner's go on FA_WARM (06I §8) | CPU | DATA-10 |
| ≈ 17Z | **Launch PRE** (pretrain_small, seed 2026) on GPU0 (shares with C_FULL s2027 + S2 runs) → ends ≈ 30 Sep 01–05Z | GPU0 | 06K, DATA-10 |
| ≈ 18Z | S2 values → interim winner; if η_s = 0.01 is the FedAdam best → launch {0.003, 0.001} (+ τ 1e-5 if FedAdam wins) concurrently with S3 | both | 06J |
| ≈ 18Z | **Launch S3 wave (4 runs) at the S2 winner**; S1a-MAIN finishes ≈ 23Z (not read) | both | S2 read |
| ≈ 21Z | S3 values (+ S2 extension values) → if the S2 extension changed the server LR, ONE combination run (06I §3) | both | — |
| ≈ 21Z | **Launch S4 wave (3 runs)** at the S3 winner (FL-13 landed) | both | S3 read |
| ≈ 24Z (30 Sep 00Z) | S4 values → `select-joint` → **FL_TUNING_SELECTION.json** + DL row → **MAIN queue** on 8–10 lanes: FA s2026, FA s2027, LADDER_S1b, LADDER_S2, (LADDER_S3), FP ×2, PF ×2 (after FL-08), FA_Q8 ×2, **S_CAL → S_RECORD → FA_1024 ×2 → DP8 ×2 → DP1**; FT_FA after FA s2026; FA_X2 after FA s2026 (21 h) | both | selection |
| 30 Sep 01–05Z | PRE done → reeval PRE on MAIN_VALIDATION (3 min; not read) → **S7 screen** (2 CALIB runs, 2.7 h) → **FA_WARM MAIN** (10.7 h) + FT_PRE selection on UNSEEN_VAL → MAIN; **S8 screen** (FL-12b) → **FA_WARM_FROZEN MAIN**; DP8_WARM_FROZEN if time | GPU0 | 06K, FL-12a/b |
| 30 Sep 15–18Z | C_FULLs end → lanes expand to 8 per card; device arms LO / FT_C (their lanes), panels, DP8_FULL (pool driver) | both | — |
| 30 Sep–1 Oct | REV-DELTA-01 / one delta review of FL-10/12/13; ANA-03/04; ANA freeze; ALL_METHODS_LOCK candidates; TEST manifests (INT-07/DATA-09) | CPU | — |
| 1–2 Oct | first read = ANA on VALIDATION (all arms + ladder rows) → lock → the single TEST look | — | — |

Totals (INFERENCE): screens S1b–S4 + extensions ≈ 20 × 2.7 h ≈ 54 process-hours (≈ 11 GPU-hours), wall ≈ 12 h sequentially; MAIN FL
incl. ladder rows, FA_X2, S7/S8 ≈ 20 runs ≈ 240 process-hours ≈ 30 h wall on 8 lanes; PRE ≈ 8–12 h. Everything fits in 48–72 h beside
the C_FULLs and the device lanes if the packets land today.

**Acceptance tests per packet (short):** FL-10 — FedAdam 3-round hand reference; FedAvgM β = 0 bitwise = FedAvg; η_s = 0 leaves θ;
DP + server opt uses the noisy mean; resume bitwise; all existing digests byte-identical at defaults; NCs. FL-13 — KNOBS override only
with `--role CALIBRATION`; MAIN refuses unless FL_TUNING_SELECTION carries passes/batch; digest changes recorded. FL-12a — Server accepts
init_sha256 = the PRE endpoint digest from a registered record; refuses any other; run label and arm spec carry the init. FL-12b —
FROZEN keys receive no gradient (requires_grad False), are absent from uploads, present once in the download ledger; bytes per visit
≈ 1.7 MB per direction on the SMALL manifest; aggregator still refuses non-shared keys. ANA-04 — ladder family from run records, never
from typed numbers; figure regenerates bitwise from the same inputs.

---------------------------------------------------------------------------------------------------------------------
## 7. Verified citations used (today unless marked)
Reddi et al. 2021 ICLR, [arXiv 2003.00295](https://arxiv.org/pdf/2003.00295) (Tables 8–10, App. D–E read from the PDF) · Hsu, Qi & Brown
2019, [arXiv 1909.06335](https://dblp.org/rec/journals/corr/abs-1909-06335.html) · Nguyen et al. ICLR 2023, [arXiv 2206.15387](https://arxiv.org/abs/2206.15387) ·
Chen et al. 2022, [arXiv 2206.11488](https://arxiv.org/abs/2206.11488) (venue UNVERIFIED) · Stremmel & Singh 2020, [arXiv 2005.04828](https://arxiv.org/abs/2005.04828) ·
Singhal et al. NeurIPS 2021, [arXiv 2102.03448](https://arxiv.org/abs/2102.03448) · Wang, Xu, Garrett, Charles, Liu & Joshi 2021,
[arXiv 2106.02305](https://arxiv.org/pdf/2106.02305) · Karimireddy et al. MIME, [arXiv 2008.03606](https://arxiv.org/abs/2008.03606) ·
Karimireddy et al. SCAFFOLD ICML 2020, [arXiv 1910.06378](https://arxiv.org/abs/1910.06378) · Wang, Liu, Liang, Joshi & Poor FedNova NeurIPS
2020, [arXiv 2007.07481](https://arxiv.org/abs/2007.07481) · Yu, Rawat, Menon & Kumar FedAwS ICML 2020, [PMLR](https://proceedings.mlr.press/v119/yu20f.html) ·
Yi et al. Efficient-FedRec EMNLP 2021, [ACL](https://aclanthology.org/2021.emnlp-main.223/) · Petrov & Macdonald gSASRec RecSys 2023,
[arXiv 2308.07192](https://arxiv.org/abs/2308.07192) · Klenitskiy & Vasilev RecSys 2023, [arXiv 2309.07602](https://arxiv.org/abs/2309.07602) ·
Hyeon-Woo et al. FedPara ICLR 2022, [arXiv 2108.06098](https://arxiv.org/abs/2108.06098) · Balles & Hennig ICML 2018, [PMLR](https://proceedings.mlr.press/v80/balles18a/balles18a.pdf) ·
Collins et al. FedRep ICML 2021, [arXiv 2102.07078](https://arxiv.org/abs/2102.07078) · Paulik et al. (Apple) 2021, [arXiv 2102.08503](https://arxiv.org/abs/2102.08503) ·
Lin et al. FedDF NeurIPS 2020 · Li et al. ICLR 2020 "On the Convergence of FedAvg on Non-IID Data", arXiv 1907.02189 · Kairouz et al.
DP-FTRL ICML 2021, [PMLR](https://proceedings.mlr.press/v139/kairouz21b.html) · Shu & Nakayama ICLR 2018, [arXiv 1711.01068](https://arxiv.org/abs/1711.01068) ·
survey arXiv 2504.05313. Verified in earlier sessions (cited from those files): Hard et al. 2018; McMahan et al. 2018; Xu et al. 2023;
Charles et al. 2021; Wang et al. 2021 field guide; Kairouz et al. 2021 FnT; Bonawitz 2017/2019; Xia et al. 2022/2023; Pillutla 2022;
Yu, Bagdasaryan & Shmatikov 2020. UNVERIFIED today: FELLAS/PTF-FSR byte figures (taken from the survey's summary), Chen 2022 venue.

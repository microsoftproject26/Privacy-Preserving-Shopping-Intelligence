# 05 — Part 4: Model backbones and the GRU vs SASRec contest (P2 / P3 / G2, rule T1, the checkpoint-rule reversal, fairness)

Reviewer: expert 4 (Claude Opus 5.5, acting as a senior recommender / production-FL scientist). READ-ONLY review; this file is
the only thing written. No server contact, no git, no test runs, no RETRO_TEST row, no downstream RESULT_ROWS value read.
Paths are relative to `.local_ai/ppsi_v4/` unless stated. Tags: **MEASURED** (read in a file, cited path:line), **REPORTED**
(a project document's own number, quoted), **INFERENCE** (my reasoning), **OPINION** (my judgement).

Values used: only the G2 result and the P2/P3 values already quoted in DECISION_LOG (DL) rows 92–138, the G1B packets and
ARCHITECTURE_LOCK, plus the published v1 final reports. None of them is a downstream or TEST value.

---

## 1. What the project does (MEASURED)

### 1.1 The two backbones as built
- **GRU ("ContextGRU-v1", not a literal GRU4Rec).** Per-decision encoder: concatenated side-channel embeddings →
  `input_proj` 128 → LayerNorm → dropout → one `nn.GRU(128, 256)` over the ≤ 50-event window, zero initial state, final
  hidden state `h_n[-1]` → readout 128 → **tied** item table + output bias (`code/v4models/gru_v2.py:239`, `:379`, `:458`).
  D128 = 20,962,210 parameters (`contracts/ARCHITECTURE_LOCK.json:169-173`). GRU_D256 = 256-d embedding/readout, hidden still
  256, tied (`contracts/model_variants/MODEL_VARIANT_REGISTRY.json`, entry GRU_D256). The thesis chapter already names it
  ContextGRU-v1, not GRU4Rec (`../thesis_reference/chapters/08_gru_design.md:5`).
- **SASRec (v3 recipe of the v1 campaign).** Pre-LN causal transformer, input LayerNorm (no √d), learned absolute position
  embeddings, gated side-information residual, **untied** output table `output_embed [K, d]` with per-step recentring
  (`code/v4models/sasrec_v2.py:202`, docstring L21-29). Windows are **right-padded and positions are counted from the
  oldest retained event** (`sasrec_v2.py:7-8`), and the query is the hidden state of the last real event (`:362`).
  D256_B3 = 83,881,764 parameters (`ARCHITECTURE_LOCK.json:169-173`); ≈ 81 M of them are the two 158,486 × 256 item tables
  (INFERENCE from the shapes).
- **Objective, identical for both:** exact full-catalogue softmax cross-entropy, one [B, K = 158,486] row per decision,
  label smoothing 0 (`code/v4train/objective.py:22-33`; comparison key `objective_id`, `ARCHITECTURE_LOCK.json:57`). Every
  decision is its own example (prefix-per-decision), so SASRec is trained on the last position of each window, not with the
  usual all-positions sequence loss (INFERENCE from `objective.py:25-33` + `sasrec_v2.py:362`).
- **Evaluation:** full-K ranking, frozen T3 evaluator, strict FP32 (`contracts/PREG1_LOCK_VALUES.json:47-66`). No sampled
  metrics.
- **Data order:** per-seed permutation independent of family, so both families see identical batches for the same seed
  (common random numbers) (`code/v4train/order.py:4-11`).

### 1.2 The contest procedure
| Stage | What | Where |
|---|---|---|
| T1 rule (frozen before any central result) | family = higher 2-seed mean of **PRACTICAL_BEST** (best validated mark in [0.5, 6.0]) client-macro MRR@20; triggers T1a \|Δ̄\| < 0.0020, T1b sign flip / \|Δs\| ≤ 0.0002, T1c \|Δ̄\| < 2·max seed gap, T1d paired-user bootstrap CI ∋ 0; else DECISIVE_STABLE | `contracts/PREG1_LOCK_VALUES.json:7-17` |
| Stopping/marks | eval every 0.5 EFE, no early stop; strictly-best needs +0.0002; STILL_IMPROVING iff best at 5.5 or 6.0; continuation 6→8 stops after 2 non-improving marks | `code/v4train/schedule.py:21-28` |
| Schedules | S0 = (0,0),(0.02,1),(2,1),(3,0.1),(5.5,0.1),(6,0.01); S1 (WSD) = (0,0),(0.02,1),(4.2,1),(6,0.01); continuation knots (7.5,0.1),(8,0.01), applied as S0 for every run | `schedule.py:64-67`; code_p3 `v4train/trainer.py:356` (in `runs_p3/deploy_code_p3/code_sync.tar`) |
| P2 (screen, CALIB 32,768 users / 646,067 decisions) | fixed AdamW(0.9,0.999), wd 1e-5, clip 1.0, B 256, 6.0 EFE, **endpoint read only**; GRU {D128, D256} × {S0,S1} × p{0.1,0.2,0.3} = 12 trials, 6 LR strata per size; SASRec {D256_B2, D256_B3, D384_B2} × {S0,S1} × 2 = 12 trials, 4 LR strata per size; best per size re-run at seed 2027 | `contracts/PREG1_AMENDMENT_02R1.md:22-33`; `PREG1_AMENDMENT_02R2.md:10-20` |
| Candidates | top-2 size levels per family by CALIB 2-seed endpoint mean → 2 candidates × 2 seeds × 2 families = 8 MAIN runs | `contracts/PREG1_AMENDMENT_05.md:16-28` |
| G2 step (a) | within family: higher 2-seed **6.0-endpoint** mean on MAIN_VALIDATION | `PREG1_AMENDMENT_05.md:30-34` |
| G2 step (b) | between families: T1 on **PRACTICAL_BEST** rows | `PREG1_AMENDMENT_05.md:35-36` |
| P4/P5 (registered) | C_BEST, SYSTEM ensemble, and **contest singles of both families at the one TEST look** | `contracts/PREG1_AMENDMENT_02.md:44-60`; `PREG1_AMENDMENT_05.md:52-57` |

### 1.3 What happened (REPORTED, all already-read values)
- **GRU P2:** best D128_S1_s4 (lr 0.001687 = top of the S1 sub-grid) CALIB 2-seed 0.124844; D256_S1_s2 0.115286
  (`company/reviews/G1B_PART1_PACKET.md:10-26`).
- **SASRec P2:** the **lowest grid LR 4.0e-4 wins for all three sizes**, always in the S0/p = 0.1 cell; S0 curves peak at 4.0
  and fall by 6.0 (e.g. D256_B3 0.13553 → 0.13323); 2-seed means B3 0.132402, B2 0.130120, D384 0.127326
  (`G1B_PART2_PACKET.md:6-25`, `:32-35`).
- **MAIN (8 runs):** GRU D128 PB 0.15714/0.15673 at 6.0 (= endpoint); GRU D256 0.15412/0.15345; SASRec B3 PB **0.16120/0.16122
  at 3.0**, endpoint 0.15287/0.15258; SASRec B2 PB 0.16059/0.16006 at 3.0, endpoint 0.15086/0.15011 (DL 132,
  `company/DECISION_LOG.md:136`).
- **G2:** Δ̄(GRU−SASRec) at PRACTICAL_BEST = −0.004271, CI [−0.005180, −0.003261], no trigger → DECISIVE_STABLE SASREC
  (`ARCHITECTURE_LOCK.json:215-250`). At the 6.0 endpoint: **+0.004213, CI [+0.003283, +0.005216]** (GRU better) (DL 138,
  `DECISION_LOG.md:142`). Seed gaps GRU 0.00041, SASRec 0.000025 (`ARCHITECTURE_LOCK.json:144-148`).
- **GRU continuation:** all four GRU runs `DONE_CONTINUATION` at 7.0 (DL 100/106/108, `G2_PACKET.md:4`).
- **Throughput:** GRU_D128 ≈ 19.5 upd/s (P2) / 25.3 upd/s (MAIN), SASRec D256_B2 ≈ 8.6, D256_B3 6.19 upd/s (DL 79/81/105).
- **Downstream consequence:** C_BIG = SASRec B3 best@3.0 and endpoint_6.0; SMALL = SASREC_D64_B2 (20,710,048 params,
  165.7 MB/visit, **2 blocks**), LR from a 4-LR CALIB mini-ladder (0.000711), schedule S0 and dropout 0.1 inherited
  (`ARCHITECTURE_LOCK.json:15-43`, `:257-261`; DL 143). GRU is `forbidden_downstream_families` (`ARCHITECTURE_LOCK.json:107-109`).

---

## 2. State of the art (literature and real systems; all citations verified by web search on 2026-09-29)

**Fair comparison / reproducibility of sequential recommenders**
- Petrov, A. & Macdonald, C. (2022). *A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation.*
  RecSys '22. https://doi.org/10.1145/3523227.3548487 — published comparisons depend heavily on implementation and training
  time; under-trained baselines invert conclusions.
- Klenitskiy, A. & Vasilev, A. (2023). *Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?* RecSys '23.
  https://doi.org/10.1145/3604915.3610644 — SASRec trained with full-softmax CE beats BERT4Rec; the loss, not the
  architecture, drove earlier rankings. **The project's exact full-K CE is exactly this recommendation.**
- Petrov, A. & Macdonald, C. (2023). *gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative
  Sampling.* RecSys '23. https://doi.org/10.1145/3604915.3608783 — negative sampling causes overconfidence; full-K CE avoids
  the issue.
- Hidasi, B. & Czapp, Á. T. (2023). *The Effect of Third Party Implementations on Reproducibility.* RecSys '23.
  https://doi.org/10.1145/3604915.3609487 — third-party GRU4Rec implementations miss features and differ from the official
  one; GRU results in comparisons are often understated.
- Hidasi, B. & Karatzoglou, A. (2018). *Recurrent Neural Networks with Top-k Gains for Session-based Recommendations.*
  CIKM '18. https://doi.org/10.1145/3269206.3271761 — the official GRU4Rec recipe (session-parallel batches, sampled
  negatives, BPR-max/CE with extra samples).
- Hidasi, B. & Czapp, Á. T. (2023). *Widespread Flaws in Offline Evaluation of Recommender Systems.* RecSys '23.
  https://doi.org/10.1145/3604915.3608839 — leakage from splits, unsuitable datasets, sampled test negatives.
- Shehzad, F. & Jannach, D. (2023). *Everyone's a Winner! On Hyperparameter Tuning of Recommendation Models.* RecSys '23.
  https://doi.org/10.1145/3604915.3609488 — the ranking of models flips with the tuning effort given to each.
- Ferrari Dacrema, M., Cremonesi, P. & Jannach, D. (2019). *Are We Really Making Much Progress?* RecSys '19.
  https://doi.org/10.1145/3298689.3347058 — tuned simple baselines beat most neural methods.
- Ludewig, M., Mauro, N., Latifi, S. & Jannach, D. (2021). *Empirical analysis of session-based recommendation algorithms.*
  User Modeling and User-Adapted Interaction 31(1):149–181 (arXiv 1910.12781) — kNN heuristics are hard to beat; GRU4Rec is
  competitive when tuned.
- Krichene, W. & Rendle, S. (2020). *On Sampled Metrics for Item Recommendation.* KDD '20.
  https://doi.org/10.1145/3394486.3403226 — sampled metrics can reverse rankings; full-K evaluation is the correct choice.
- Kang, W.-C. & McAuley, J. (2018). *Self-Attentive Sequential Recommendation.* ICDM 2018, pp. 197–206.
  https://doi.org/10.1109/ICDM.2018.00035 — the original SASRec.

**Variance and seeds**
- Bouthillier, X. et al. (2021). *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021.
  https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html — comparisons should
  randomise data sampling, initialisation *and hyperparameter choice*; one seed pair captures only part of the variance.

**Architecture × federated optimisation**
- Qu, L. et al. (2022). *Rethinking Architecture Design for Tackling Data Heterogeneity in Federated Learning.* CVPR 2022
  (arXiv 2106.06047) — self-attention models hold up better than CNNs under non-IID FL. (Vision, not recommendation.)
- Zhang, J. et al. (2020). *Why are Adaptive Methods Good for Attention Models?* NeurIPS 2020 — attention models need
  adaptive optimisers and clipping (heavy-tailed gradient noise). This matters when FedAvg local/server optimisers are chosen for a
  transformer backbone.

**Production on-device FL (real system)**
- Hard, A., Rao, K., Mathews, R. et al. (2018). *Federated Learning for Mobile Keyboard Prediction.* arXiv 1811.03604.
  https://arxiv.org/abs/1811.03604 — Gboard's next-word model is a **single-layer CIFG RNN with tied input/output embeddings**
  (V = 10,000, D = 96, 670 units, **1.4 M parameters, 1.4 MB after quantisation**; §3, verified in the PDF). "Language models are
  usually limited to tens of megabytes" on device (§1).

**What recommender practice uses (OPINION, not a citation):** offline comparisons normally select the epoch by early stopping on a
validation split. That is PRACTICAL_BEST. A fixed-epoch endpoint comparison is less common and is used when a later process
(here FL) must share the budget. Production next-item retrieval often uses two-tower or sequential encoders with sampled
softmax or logQ correction over very large catalogues. Full softmax at K ≈ 160 k is feasible and is the cleaner research
choice.

---

## 3. Comparison with the state of the art

### 3.1 Strengths (what a RecSys/KDD reviewer would praise)
1. **Pre-registration and sealing are better than normal practice.** T1 (rule, triggers, bootstrap seed) was frozen before any
   central result (`PREG1_LOCK_VALUES.json:4`, `:7-17`). The G2 read was a single unsealing by a frozen script
   (`G2_PACKET.md:3-7`). Both checkpoint rules were registered, and the reversal was published instead of hidden
   (DL 138; `G2_FABLE_RULING.md:14`). Few published SASRec-vs-GRU comparisons go this far.
2. **Correct loss and metric choices.** Exact full-K CE and full-K ranking for both families avoid the sampled-loss and
   sampled-metric artefacts that inverted earlier rankings (Klenitskiy & Vasilev 2023; Petrov & Macdonald 2023; Krichene &
   Rendle 2020).
3. **The same search budget for both families.** Both got 12 stage-1 trials plus per-size stage-2 seeds, the same LR range,
   the same schedules, rungs and selection rule, and all trials are in the ledger (`PREG1_AMENDMENT_02R1.md:22-33`). The
   02R2 fix removed a size×LR confound before launch (`PREG1_AMENDMENT_02R2.md:1-12`). This answers Shehzad & Jannach 2023.
4. **Paired design.** The same data order per seed for both families (`order.py:4-11`), a shared bootstrap resample matrix W
   (`code/v4train/bootstrap_ci.py:3-6`), and a CALIB-disjoint diagnostic that agrees (−0.004103, CI [−0.005255, −0.002908]).
5. **The GRU is taken seriously** (Hidasi & Czapp 2023): it gets its own size and dropout search, a separate LR grid with 6 strata,
   and continuation. It is honestly named ContextGRU-v1.
6. **Tuning at the target scale.** Within-size hyperparameters are screened on CALIB, then size is selected on MAIN, so all
   contest models are MAIN-trained (`PREG1_AMENDMENT_05.md:57-61`). This fixes the classic "tuned on a subset" objection.

### 3.2 Weaknesses and errors

**W1 — The verdict depends on the checkpoint rule, and the cause is SASRec over-fitting, not noise (INFERENCE from MEASURED
values).** SASRec B3 loses 0.0085 between 3.0 and 6.0 EFE (0.1612 → 0.1527), with the learning rate held at 0.1× peak
(= 4e-5) over [3.0, 5.5] (`schedule.py:64`). That drop is 2× the family gap (0.0043) and more than 300× the SASRec seed gap
(0.000025). So this is systematic over-fitting, not selection noise. The likely mechanism: 84 M parameters, ≈ 81 M of them in
two untied item tables, on 2.59 M training decisions (≈ 32 parameters per decision), with practically no weight regularisation.
wd 1e-5 is decoupled, so the shrink per update is lr·wd ≈ 4e-9 (`PREG1_AMENDMENT_02R1.md:22`), dropout sits at the grid
minimum 0.1, and weight decay and label smoothing were excluded from the search (`BEST_BACKBONE_FABLE_POSITION.md`, Q2).
The GRU (tied, 21 M) does not over-fit. **Meaning:** under early stopping (the common RecSys protocol), SASRec is the stronger
predictor; under a fixed 6-EFE budget, GRU is. Both statements are true and both are pre-registered. The honest claim is
"SASRec's best validated checkpoint beats GRU's best, within a 6-EFE budget". It is not "SASRec is the better architecture".
The project already words it this way (`PREG1_AMENDMENT_06B.md:94` "never phrased as 'SASRec beats GRU'").

**W2 — The GRU "still improving" story is not established, and the continuation could not have established it (MEASURED
design + INFERENCE).** STILL_IMPROVING fires whenever the best mark is 5.5 or 6.0 (`schedule.py:25`). Under any end-of-run
cooldown this is almost automatic (Fable's own disclosure, `STAGGERED_P3_FABLE_RULING.md`). The continuation then jumps the
LR **10×**, from 0.01× to 0.1× peak (CONT_KNOTS `schedule.py:66`, applied as S0 even to S1 recipes, code_p3
`trainer.py:356`). It stops after two non-improving marks, i.e. at 7.0 (`schedule.py:27`), **before** the second cooldown at
7.5–8.0 could show any gain. All four GRU runs stopped exactly at 7.0 (DL 100/106/108). So "GRU would gain from more
budget" remains **untested, not refuted**. The prior project review repeats "GRU D128 is still improving at 6.0"
(`FABLE_THESIS_RESEARCH_REVIEW_20260928.md:149`). That sentence should read "the GRU's best mark is its endpoint (a cooldown
artefact); a longer budget was not tested in a way that could show a gain".

**W3 — Three different selection criteria in one pipeline (MEASURED).** P2 picks hyperparameters by the **CALIB 6.0 endpoint**
(`02R1.md:22-26`). G2 step (a) picks size by the **MAIN 6.0 endpoint** (`05.md:30-34`). G2 step (b) picks the family by
**PRACTICAL_BEST** (`05.md:35`). All three were registered in advance, so this is not p-hacking (`G2_FABLE_RULING.md:14`), but
it is internally incoherent. The SASRec recipe was tuned to *survive to 6.0*, which pushed it to the lowest LR, and was then
judged at its 3.0 peak. The direction of the bias is probably **against** SASRec (a recipe tuned for peak value might peak
higher), so the verdict is conservative for the winner (INFERENCE). The thesis must still explain it. The step-(a) choice agrees
under PB (B3 0.1612 > B2 0.1603; D128 0.1571 > D256 0.1541; DL 132), so no selection outcome depends on the mismatch.

**W4 — Thin tuning with edge optima in both families (MEASURED).** Each (size, LR) point got exactly one trial with one fixed
schedule/dropout pairing (`02R2.md:15-20`), so schedule and dropout are confounded with LR (`02R2.md:34-38`, disclosed).
SASRec's winner is the **only** cell at LR 4e-4 at each size, always S0/p 0.1. S1 or higher dropout was never tried at that LR.
It sits on a steep edge: B3 0.1332 at 4.0e-4 vs 0.1208 at 7.1e-4 (`G1B_PART2_PACKET.md:15-18`). GRU's winner is the top of
the S1 sub-grid (`G1B_PART1_PACKET.md:34`), and S1 was never tried at 2.476e-3. "Symmetric non-extension" (G1b-2 ruling) is
symmetric in procedure, but the truncation loss may differ between families. SASRec's steeper edge suggests it is the more
truncated family, which again is conservative for the winner (INFERENCE). CALIB optima also shift at MAIN, with 4× the
updates per EFE (disclosed, `MAIN_SCALE_FABLE_RULING.md`).

**W5 — The families never overlap in size, and equal EFE is not equal compute (MEASURED).** GRU candidates are 21 M and
≈ 41.5 M; SASRec candidates are ≥ 83 M. No parameter-matched pair was contested. SASRec costs ≈ 3–4× GPU time per EFE
(6.19 vs 19.5–25 upd/s). Equal *EFE* is the RecSys "same number of epochs" convention (acceptable, disclosed by Fable:
"budget equality is in EFE and trial count, not GPU-h"). GRU D256 did worse than D128 on MAIN (0.1541 < 0.1571), which is
partial evidence that the GRU is not capacity-limited along the embedding-width axis. But an untied GRU, a deeper GRU and a
compute-matched GRU were never tried.

**W6 — The family was chosen at BIG size, but every Edge arm runs at SMALL size (MEASURED + INFERENCE). This is the most
important point for the research question.** All FL, device and DP arms run SASREC_D64_B2 (≈ 20.7 M) (`PREG1_AMENDMENT_03.md:21-24`).
At that size a parameter-matched GRU exists and is already trained on MAIN: 02B_GRU = GRU_D128, 20.96 M, retained as the
"P5 singles" (`ARCHITECTURE_LOCK.json:151-168`). The contest answered "which family is best at ≈ 84 M vs 21 M". The Edge
question needs "which ≈ 21 M on-device model is best". If GRU_D128 ≥ C_SMALL, then K1, the size cost R(C_SMALL/C_BIG),
overstates the minimum cost of going small, and every device/FL retention ratio carries a family penalty that is unmeasured.
Nothing in the frozen contrast list (`company/gates/ANA_CONTRASTS_v2.json`, K1–K19) compares GRU_D128 with C_SMALL. The owner's
"same architecture" directive (DL 83) makes the design internally valid. But without this descriptive comparison, the external
answer to "how much quality is lost on the device" is incomplete.

**W7 — P5/P4 were registered but have dropped out of the TEST plan (MEASURED; protocol deviation unless withdrawn).**
Amendment 02 §5 registers "One TEST look at ALL_METHODS_LOCK: contest singles, C_BEST, SYSTEM, BASELINE"
(`PREG1_AMENDMENT_02.md:58`). Amendment 05 §7 keeps P4/P5 for the selected recipe of each family
(`PREG1_AMENDMENT_05.md:53`). ARCHITECTURE_LOCK keeps the GRU P5 checkpoints "never deleted" (`:151-168`). The executable TEST
plan re-evaluates C_BIG, C_FULL, C_SMALL and the FL/DP arms, and **no GRU single, no C_BEST, no SYSTEM**
(`company/plans/MASTER_PLAN_V5.md:190`). I found no amendment withdrawing them (grep of `contracts/PREG1_AMENDMENT_06*.md`). The v1
campaign's lesson makes the omission matter: the SASRec-over-GRU margin **shrank from +0.0126 on VS to +0.0018 on TEST**
(`../sasrec_v1/FINAL_SASREC_REPORT_EN.md:21`). The ID-only SASRec was *below* GRU on TEST (`:22`). A VALIDATION-only family
verdict on previously inspected weeks, with no TEST confirmation, is exactly what that lesson warns against. The GRU
checkpoints were evaluated on the A16, and the TEST look runs on the pinned A40 class (06D §1). So the GRU singles need an
A40 re-evaluation, which is evaluation only.

**W8 — PRACTICAL_BEST is chosen on the same VALIDATION split that the downstream first read uses (INFERENCE; small, symmetric,
disclosed).** C_BIG's PB is a max over 12 marks on MAIN_VALIDATION. The recipe is also a max over 2 candidates and 2 families
(max-of-marks disclosure, `G2_FABLE_RULING.md:14`; `05.md:101`). The downstream K-contrasts use PRACTICAL_BEST by default
(`ANA_CONTRASTS_v2.json` `default_confirmatory_rule`), with PB for the FL arms selected on the same split. Numerator and
denominator are both max-of-marks, so the bias mostly cancels in the ratios. The residual bias makes the denominator slightly
optimistic, so VALIDATION retention is slightly conservative. The size of the bias is set by mark-to-mark noise: seed gaps
are 2.5e-5 to 4e-4 and the curves are smooth. It is ≪ 0.0043 and ≪ the 0.0085 PB–CB gap. TEST, a later week that includes
Black Friday (06B §7b), is the only unbiased number. For DP arms the project already switched to CONTROLLED_BUDGET (06D §3).
The **checkpoint rule of the denominator** matters more than the family: C_BIG's PB–CB gap (≈ 5.3 % relative) is larger than
many expected retention losses. Every K ratio should therefore be read with both rules, and the project has both rows.

**W9 — SMALL is not "BIG with a smaller d" (MEASURED).** Amendment 03 §1 defines SMALL as "everything else identical to BIG
(blocks, heads, ff, tying, dropout)" (`PREG1_AMENDMENT_03.md:12`). It also names SASREC_D64_B2, registered when BIG was assumed to
be D256_B2. BIG turned out to be **B3**, so the size step K1 changes both width (256 → 64) and depth (3 → 2). The LR also
changes (4e-4 → 7.1e-4, DL 143). This is a disclosure item, not an error: the variant was named explicitly before any read.

**W10 — Two seeds (MEASURED; adequate for this effect, not in general).** With 2 seeds the between-seed variance has 1 degree
of freedom. T1c compares the gap with 2× the observed range; the gap is 0.0043 vs a range of ≤ 0.0004, a ratio above 10. The
bootstrap covers user sampling but not training randomness or the hyperparameter draw (Bouthillier et al. 2021). For a 2.7 %
relative effect with a 10× margin, a flip from a third seed is implausible (INFERENCE). The rule-dependence (W1) and the untested
regularisation axis (W4) are larger threats than seed count. The SASRec seed gap of 2.5e-5 is remarkably small next to
GRU's 4e-4. That is plausible because the same data order and the full-K CE smooth out differences, but it deserves one
sentence (INFERENCE).

**W11 — Minor SASRec fidelity points (MEASURED; OPINION on impact).** (a) Absolute positions are aligned to the *oldest*
retained event (`sasrec_v2.py:7-8`). Canonical SASRec left-pads, so the most recent item always has the same position
(Kang & McAuley 2018). Here recency must be inferred from varying indices, which may slightly handicap SASRec. The GRU's
final hidden state is recency-aligned by construction. (b) Last-position-only training (prefix-per-decision) instead of the
all-positions sequence loss is symmetric across families, but it is not how either model is usually trained. Both points are
inherited from v1, where this SASRec reached 0.1705 on TEST (`../sasrec_v1/FINAL_SASREC_REPORT_EN.md`), so they are not
defects. They are deviations to list in the "not a literal reproduction" table.

### 3.3 Unrealistic for an Edge deployment (INFERENCE / OPINION)
- **C_BIG (84 M, 336 MB FP32 checkpoint, `ARCHITECTURE_LOCK.json:18-24`; 671 MB per visit, `:258`)** is a cloud model only.
  It is correctly used only as the L0 reference.
- **SMALL (20.7 M, ≈ 83 MB FP32, 165.7 MB per visit down+up)** is ≈ 15× Gboard's production FL model (1.4 M params / 1.4 MB,
  Hard et al. 2018 §3). About 98 % of it is the two item tables (158,486 × 64 × 2 ≈ 20.3 M). For a browser this is plausible
  as a download (≈ 21 MB INT8), but it is heavy as a per-round FL payload. That is the communication expert's part. The
  backbone-specific point: the **untied** output table doubles the item bytes. At the same ≈ 21 M budget, GRU_D128 gets a
  128-d tied item space where SASRec D64 gets two 64-d tables. Gboard's production choice (a tied-embedding RNN) is exactly
  the GRU design. A tied SASRec at SMALL was not explored: the v1 tied variant failed from common-mode drift, and the v3
  recentring might have fixed it, but this was never tested.
- **Compute/latency:** D64/2-block attention over 50 events plus a 158 k × 64 scoring matmul (≈ 10 M MAC) is cheap on any
  device (INFERENCE). Latency does not decide the family at SMALL size; bytes and trainability under FedAvg do.
- **Trainability under FL:** attention models need adaptive optimisers and clipping (Zhang et al. 2020). The FL client uses a
  fresh AdamW each visit (`code/fl/__init__.py:10`). The very low FL CALIB values (S0 0.0235 vs central ≈ 0.11–0.12,
  `company/registers/FL_IMPROVEMENT_LADDER.md:10`) may be partly a family effect: the fresh-AdamW / few-step regime is harsh on
  transformers. This cannot be separated without a GRU FL arm (OPINION; flag as a threat, not a claim).

### 3.4 Missing pieces
- A parameter-matched GRU_D128 vs SASREC_D64_B2 comparison at the deployed size (W6), which is cheap.
- The registered P5 TEST confirmation of the family verdict (W7).
- A stated mechanism for the reversal (W1). The project has the mark curves; a one-page diagnostic of VALIDATION MRR vs
  EFE for all 8 runs (already read) would show the over-fitting directly.
- A regularisation axis for SASRec (weight decay, dropout > 0.1 at LR 4e-4) and a fair longer-budget test for GRU (W2, W4).
  These are exploratory only and can never change G2.

---

## 4. Recommendations

Pre-registration status check (DECISION_LOG): the G2 governed value **has been read** (DL 129–131); T1, the P2/P3 rules and
ARCHITECTURE_LOCK can therefore **never** change. The C_SMALL values and all downstream arms are **unread** (06D preamble;
DL 146 "unread"). ANA/ALL_METHODS_LOCK is **not yet frozen** (project map §4 "ANA freeze pending"). TEST is **sealed**. So a
descriptive row or contrast can still be *added* before ALL_METHODS_LOCK. Nothing may alter G2.

| ID | Item | Class | Action | Cost (INFERENCE) | Affects | Pre-registration needed + status |
|---|---|---|---|---|---|---|
| B1 | Keep the G2 verdict and ARCHITECTURE_LOCK exactly as they are; no re-run of the contest | MUST | KEEP | 0 | finished P3/G2 | none — any change is forbidden (governed value read) |
| B2 | Restore the registered P5 contest singles at the TEST look: re-evaluate 02B_GRU (D128) s2026/s2027 best@6.0 and 02B_SASREC PB **on the pinned A40** for MAIN_VALIDATION and MAIN_RETRO_TEST, and report the family contrast at TEST with the paired CI next to the VALIDATION one (v1 shrinkage lesson). Or formally withdraw P4/P5 in a dated amendment with the reason | MUST | ADD (to MASTER_PLAN_V5 §11.2) | ≈ 0.2–0.5 GPU-h of evaluation; ≈ 1 h Ops | TEST look (not yet run) | yes: an addendum listing the rows before ALL_METHODS_LOCK. It restores amendment 02 §5 / 05 §7, so it changes no read rule. Status: OPEN |
| B3 | Add a **descriptive, parameter-matched deployed-size comparison**: GRU_D128 (20.96 M) vs C_SMALL SASREC_D64_B2 (20.71 M), PB and CB, same A40 block, VALIDATION and TEST, paired CI; label it "family at device size — descriptive, never selection". Report K1 alongside "best-small / C_BIG" if GRU_D128 > C_SMALL | MUST (for the Edge RQ's external validity) | ADD | evaluation only, included in B2's cost | analysis only; C_SMALL unread | yes: add as a registered descriptive contrast (e.g. K20) before the VALIDATION first read / ALL_METHODS_LOCK. Status: OPEN, still allowed |
| B4 | Thesis wording of the contest. Use "selected under T1 (best validated checkpoint within 6.0 EFE)". Show both rules in one figure (PB/CB × family × seed). Explain the reversal as SASRec over-fitting after its S0 decay (0.0085 loss at LR 0.1× peak; 84 M params vs 2.59 M decisions; wd effectively 0; dropout at the grid minimum). Never write "SASRec beats GRU" or "SASRec is the better architecture" | MUST | CHANGE (text) | ≈ 2 h writing | write-up | no |
| B5 | Correct the "GRU was still improving at 6.0" wording in all documents: it is a cooldown artefact; the continuation (10× LR re-warm, patience 2, stop at 7.0 before the 7.5–8.0 cooldown) could not show a gain, so a longer-budget GRU is **untested** | SHOULD | CHANGE (text) | ≈ 0.5 h | write-up, RESUME/thesis reference | no |
| B6 | Disclose the three-criterion mismatch (P2 endpoint on CALIB → size endpoint on MAIN → family PB on MAIN) with the likely direction (conservative for SASRec) and the evidence that step (a) agrees under PB (DL 132) | SHOULD | DISCLOSE | ≈ 0.5 h | write-up | no |
| B7 | Report every K ratio with both checkpoint rules for the denominator (C_BIG PB and CB), stating that C_BIG's PB–CB gap (≈ 5 %) is larger than the family gap | SHOULD | ADD (report column) | minimal; rows exist | analysis (unread) | only if a rule is promoted to confirmatory; as descriptive, add before ANA freeze. Status: OPEN |
| B8 | Disclose the tuning thinness: one trial per (size, LR), confounded schedule and dropout, both optima on grid edges (SASRec lowest LR at all sizes, steep edge; GRU top of the S1 sub-grid), unextended; regularisation (wd, label smoothing) excluded, although SASRec's over-fitting suggests it was the missing axis | SHOULD | DISCLOSE | ≈ 0.5 h | write-up | no |
| B9 | Disclose non-overlapping sizes and compute: 84 M vs 21 M, ≈ 3–4× GPU-time per EFE for SASRec; GRU D256 < D128 on MAIN as partial evidence against GRU capacity limits; untied/deeper/compute-matched GRU untested | SHOULD | DISCLOSE | ≈ 0.5 h | write-up | no |
| B10 | Disclose that SMALL differs from BIG in depth (B3 → B2) as well as width, plus LR (4e-4 → 7.1e-4), so K1 is a "recipe-size" step, not a pure width step | SHOULD | DISCLOSE | ≈ 0.2 h | write-up | no |
| B11 | State that the retention results are **within the SASRec family**, and that the FL gap may include a family × optimiser interaction (transformer + fresh AdamW per visit; Zhang et al. 2020); a GRU FL arm was not run | SHOULD | DISCLOSE | ≈ 0.3 h | write-up | no |
| B12 | Exploratory only (label EXPLORATORY_POST_G2, never selection, never C_MATCHED): SASRec D256_B3 at LR 4e-4 with (i) S1 and (ii) dropout 0.3 / wd 0.05, one seed on MAIN; GRU D128 with a proper second cooldown to 8.0 (no patience stop). Answers "was the reversal a regularisation artefact?" | NICE | ADD | ≈ 5 GPU-h per SASRec run, ≈ 2.5 per GRU run (A16 rates) | new runs only; nothing finished | yes: a new registration before launch, stating it cannot alter G2/ARCHITECTURE_LOCK. Status: not registered |
| B13 | Add a "deviations from canonical SASRec / GRU4Rec" table: oldest-aligned positions, right padding, last-position-only loss, untied head + recentring, input LN, AdamW; GRU = ContextGRU (no session-parallel batches, no sampled BPR-max) — cite Hidasi & Czapp 2023 and Kang & McAuley 2018 | NICE | ADD (text) | ≈ 1 h | write-up | no |
| B14 | One-page mechanism figure: VALIDATION macro MRR@20 vs EFE for all 8 contest runs (already read at G2/condition 5), with the S0/S1 LR curve overlaid | NICE | ADD | ≈ 1 h (plot from already-read rows) | write-up | no (values already read) |
| B15 | Two seeds: keep, and state that the unit of inference is the user and that training variance is bounded by the observed seed range (≤ 4e-4 ≪ 0.0043). A third seed is not needed for the family verdict | DISCLOSE-ONLY | KEEP | 0 | — | no |
| B16 | Max-of-marks optimism on VALIDATION (C_BIG PB selected on the same split as the downstream first read): state it is symmetric in PB ratios, small (mark noise ≪ gaps), and that TEST is the only unbiased number | DISCLOSE-ONLY | KEEP | 0 | — | no |
| B17 | Edge realism of the backbone: C_BIG is cloud-only; SMALL ≈ 15× Gboard's 1.4 M-parameter tied RNN, 98 % item tables; the untied head doubles item bytes | DISCLOSE-ONLY | KEEP | 0 | write-up | no |

---

## 5. Score: **7 / 10**

Justification. As a *procedure*, this contest is better than most published SASRec-vs-GRU comparisons. The rule was frozen
before any result. Search budgets, the grid, seeds and data order are identical. The loss is exact full-K CE and the
evaluation is full-K. The G2 read was a single sealed one, a CALIB-disjoint diagnostic agrees, and the endpoint reversal was
honestly published. On process alone it would score 9. It loses points because, as a *finding about backbones*, it is fragile
and partly off-target for the research question. (i) The verdict flips with the checkpoint rule. The flip is explained by an
under-regularised 84 M-parameter SASRec over-fitting a 2.59 M-decision cohort, while regularisation was excluded from the
search and both LR optima sit on grid edges. (ii) The families never overlap in size or compute. (iii) The family was chosen at
BIG size, but every Edge arm runs at SMALL size, where a parameter-matched GRU exists and has never been compared. (iv) The
registered TEST confirmation of the family verdict (P5) has silently dropped out of the TEST plan, despite the v1 evidence that
the SASRec margin shrinks roughly 7× from VALIDATION to TEST. Items (iii) and (iv) are cheap to fix (evaluation only) and still
allowed before ALL_METHODS_LOCK. With B2 and B3 done and B4 to B6 written, I would score this part 8.

---

## 6. UNVERIFIED items and open questions
- UNVERIFIED: that early stopping on validation is the "standard" protocol of RecBole or of the original SASRec code (stated as
  OPINION in §2, not cited).
- UNVERIFIED: whether code_p3's `planned_lr_table(..., schedule="S0")` for CONTINUATION (`trainer.py:356` in the code_p3 tar)
  matches the reading of the Fable staggered ruling for S1 recipes. It matches the knots; it is the 10× re-warm plus patience-2
  interaction that is the issue (W2), not a code bug.
- OPEN: the GRU 7.0 PRACTICAL_BEST_EXTENDED values required by G2 condition 5 (`G2_FABLE_RULING.md:12`) do not appear in
  DL 138 (only the CONTROLLED_BUDGET delta is quoted). I did not open `runs_p3/g2/G2_CONDITION5_DIAGNOSTICS.json` (under
  `runs_*`, outside my read permission), so I cannot say whether condition 5 is fully closed.
- OPEN: whether an amendment I did not find withdraws P4 (C_BEST, SYSTEM) and P5 contest singles. The grep of
  `contracts/PREG1_AMENDMENT_06*.md` and `company/plans/*.md` found none; the lead should confirm (W7/B2).
- OPEN: the SMALL mini-ladder CALIB values are not quoted in the DL (only the chosen LR 0.000711). So a CALIB-level
  GRU_D128 (0.124844) vs SASREC_D64_B2 comparison cannot be made from allowed sources. B3 makes it properly on MAIN.
- OPEN (INFERENCE): the SASRec seed gap (2.5e-5) is 16× smaller than GRU's. It is plausible under shared data order, but a
  one-line check that seeds 2026/2027 differ in θ0 **and** order **and** dropout stream for SASRec (`order.py`, trainer.py
  `torch.manual_seed(seed)`) would pre-empt an examiner's question. The code indicates they do.

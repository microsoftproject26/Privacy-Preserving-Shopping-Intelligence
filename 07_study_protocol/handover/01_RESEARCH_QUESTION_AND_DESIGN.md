# 01 — Research question and study design

*Paths are relative to `.local_ai/ppsi_v4/` ("V4/"). "DL n" = decision-log row n. "06x" = `contracts/PREG1_AMENDMENT_06x.md`.
Labels: MEASURED, REPORTED, INFERENCE, PLAN, PENDING.*

## 1. The question and how it is answered

**Question (proposal, unchanged; amendment 04 header; DL 86):** *How much shopping recommendation quality can be retained
when moving from cloud-based to on-device or federated models?*

- The owner asked "how much is retained", not "is it acceptable". **There is no pass / fail threshold** (amendment 04 §1;
  06 §7; 06A §7).
- **Quality:** client-macro MRR@20 over the full catalogue of 158,486 products (each user's reciprocal rank averaged over
  their decisions, then averaged over users; 0 beyond rank 20; out-of-vocabulary targets count as misses = the "E2E"
  denominator). NDCG@10 and HR@10 are secondary; HR / NDCG / MRR at 1, 5, 10, 20 are reported descriptively (DL 385).
- **Retention:** R = 100 × arm / reference, same checkpoint rule, same seed, then the mean of the per-seed ratios
  (06A §7; 06E §9(e)).
- **Uncertainty:** paired user bootstrap, 1,000 resamples, seed 20260926; ratios are bootstrapped directly; percentile
  95 % CIs (06B §2; REPORT header).
- **Seeds:** 2026 and 2027 for almost every arm (single-seed exceptions are registered, §6).
- **Reference models:** C_BIG (cloud model trained on the same 131,072 users) for the random-start ("cold") line; C_FULL
  (cloud model trained on 1,348,682 users) for the pretrained ("warm") line, because the pretrained model saw 9× more users
  than C_BIG (owner decision DL 273; 06K disclosure (1)).
- **Anchoring on LAST:** LAST ("repeat the last viewed item") needs no training and reaches about 44 % of C_BIG, so every
  arm is also reported against LAST: Δ(arm − LAST), headroom R* = (arm − LAST) / (C_ref − LAST), and a flag
  BELOW_DEVICE_HEURISTIC when the CI of Δ lies below 0 (06L_C C3).

## 2. The realistic Edge-deployment framing

The owner's top priority is that every experiment simulates what a browser vendor could actually deploy: a big model on
the server with the data, federated learning where devices send weights only, and a smaller model of the same
architecture on the device (owner directive DL 83; 06N; "realism first", RESUME_HERE §2o-ter). The study maps onto
these systems plus references:

| System | What leaves the device | Arms | Privacy level (06L_C C1) |
|---|---|---|---|
| **Cloud-only** (server trains on raw events, serves queries) | raw events, query context | C_BIG, C_FULL, C_SMALL | L0 |
| **Cloud-trained model shipped to the device** (inference on device) | raw events once (or only opted-in users' events) | C_SMALL_FP32 / _INT8 (ONNX), PRE alone, KNN_EMB | L1 |
| **Federated, weights only** (devices train, send updates) | model updates (dense FP32, 8-bit, or encoder-only) | FA, FP, PF, FA_Q8; FA_WARM, FA_WARM_FROZEN and its method set | L2 (no formal guarantee) |
| **Federated + differential privacy** | clipped updates + Gaussian noise on the sum | FA_1024 (z = 0 twin), DP8, DP1; DP8_FROZEN, DP1_WARM_FROZEN | L3 (central DP, trusted aggregator) |
| **FL + local personalisation** | as FL; personal parameters never leave | PF, FT_FA, FT_FA_WARM(_FROZEN) | L4 |
| **Small local model** (shipped base + local fine-tune) | nothing after the download | FT_C (L1 base), **FT_PRE** (base from opted-in users only = L5) | L1 / L5 |
| **Device-only** | nothing | LO, LAST | L6 |
| Device + published aggregate | nothing of the user | RULE, POP | — |

Two facts shape the privacy reading (06I §7(a); 06L_C C1–C2):
- **Without DP, an update reveals the viewed items.** The input and output item tables of the model are untied and hold
  about 98 % of the parameters; the non-zero rows of a non-DP update identify the items the user saw. "Only weights leave
  the device" is data minimisation, not privacy. Secure aggregation is **not simulated** in any arm.
- **Threat model:** an honest-but-curious browser vendor (the server). DP = central DP with a trusted aggregator,
  client-level add / remove adjacency, (ε, δ = 5e-6), conditional on the public bootstrap block (catalogue, transform
  statistics, θ0 / θ_pre); the FL tuning on CALIB is not accounted.

## 3. Models

| Name | Architecture | Parameters | Trained on | Source |
|---|---|---|---|---|
| C_BIG | SASREC_D256_B3 (width 256, 3 blocks) = recipe `02B_SASREC` | 83,881,764 | MAIN, 6.0 EFE, the contest runs themselves | ARCHITECTURE_LOCK; DL 129–131 |
| C_FULL | same as C_BIG | 83,881,764 | TIEBREAK band, 1,348,682 users | 06 §5; 06D §7; DL 151, 248, 259 |
| C_SMALL | SASREC_D64_B2 (width 64, 2 blocks, 4 heads) = `SMALL_SASREC_D64_B2`, peak LR 0.000711 | 20,710,048 | MAIN | amendment 03; DL 125, 143, 145 |
| PRE / θ_pre | SASREC_D64_B2, SMALL recipe unchanged | 20,710,048 | PRE band, 1,217,610 non-MAIN users, seed 2026; θ_pre = endpoint 6.0 (sha `cf08588c…`) | 06K §1; DL 252, 254–255 |
| Every FL / device model | SASREC_D64_B2 (the server's global model = the device model) | 20,708,708 shared parameters in the FL rows (MEASURED, REPORT §2) | — | DL 276 |

- The device model differs from C_BIG in depth as well as width (B3 → B2), so K1 is not a pure width effect (disclosure).
- The registered device payload cap is ≤ 170 MB per visit (amendment 03 §1); the cap came from the GRU_D128 payload at
  design time. FA moves 165,669,672 bytes per visit (FP32 down + up; MEASURED, REPORT §2); S8 moves 2,110,584 bytes per
  visit plus a one-time 81,779,544-byte download of the frozen tables (MEASURED, REPORT §12b).
- A larger server model trained alone is future work, not part of the design (owner decision DL 276): FedAvg needs one
  shape on server and device.

## 4. The federated recipe and the warm selections (all CALIB values)

The first registered FL recipe reached only about 20–25 % of a matched central model on CALIB. A research audit found five
recipe causes (no server optimiser; a fresh client AdamW each visit; local LR too small; the central LR shape mapped onto
rounds; a random start), fixed by a registered, strictly sequential ladder (06F, 06G, 06J, 06L; details in
`company/registers/FL_IMPROVEMENT_LADDER.md` and the 09-30 handover §9). MEASURED, CALIB_VALIDATION E2E client-macro
MRR@20 at PRACTICAL_BEST:

| Stage | Change | Value | Source |
|---|---|---|---|
| S0 | FedAvg as registered | 0.023473 | `contracts/fl/LR_SELECTION.json`; DL 181 |
| S1a / S1b | local LR × 2, then × 8 | 0.025562 / 0.028796 | `LR_SELECTION_EXT.json`; DL 218 |
| S2 | server optimiser FedAdam η_s 0.01, τ 1e-3 | 0.034247 | `FL_TUNING_S2.json`; DL 256 |
| S3 | FL_CONST LR shape + client SGD-momentum 0.9, LR 0.1 | 0.094936 | `FL_TUNING_S3.json`; DL 256 |
| **S4** | **1 local pass** (instead of 2), batch 16 | **0.107562** (× 4.58 over S0) | `contracts/fl/FL_TUNING_SELECTION.json` sha `9bca186e…`; DL 271 |

Every FL arm uses the S4 recipe ("FA" = this recipe, 06I §2). Exception: the Poisson DP-type arms (S_CAL, FA_1024, DP8, DP1
and their warm variants) keep 2 local passes / batch 16 because their T and z are registered (DL 219).

**Warm start (S7 / S8; 06K, 06M_B B2; DL 292).** Two registered 2 × 2 screens at the S4 point, from θ_pre:

| Screen | Cells (local LR factor × server LR factor) | Winner | CALIB |
|---|---|---|---|
| S7 = FA_WARM (full model trained) | {1, 0.25} × {1, 0.1} | f 0.25 (client LR 0.025) / sf 0.1 (FedAdam η_s 0.001) | 0.170582 |
| S8 = FA_WARM_FROZEN (item tables frozen, not sent) | {1, 4} × {1, 0.1} | f 1 (client LR 0.1) / sf 0.1 (η_s 0.001) | 0.170709 |

FA_WARM* = the higher-CALIB winner = **S8** (06N N2; 06P P1). Files: `FL_WARM_SELECTION_S7.json` (`d67a0a85…`) and
`_S8.json` (`8b79d305…`), server-only (copies in `../thesis_reference/sources/local/S579_*`, `S580_*`). INFERENCE recorded
later (PREG2 §1 M5): the screen only moved the client LR upward and its winner sat at the smallest server LR (a grid edge);
this is the starting point of Phase 2.

## 5. All arms, in plain words

Seeds are 2026 and 2027 unless stated. Every MAIN arm is evaluated on MAIN_VALIDATION (40,764 users with evaluation rows,
519,714 decisions).

**Cloud references**
- **C_BIG** — the cloud model on the 131,072 MAIN users. PRACTICAL_BEST = the contest's best checkpoint (mark 3.0),
  CONTROLLED_BUDGET = endpoint 6.0.
- **C_FULL** — the same model trained on 1,348,682 users (= the PRE users + the MAIN users). Seed 2026 is confirmatory,
  2027 descriptive. Its recipe was not re-tuned at that scale (an upward bias of ratios against C_FULL, disclosed).
- **C_SMALL** — the device-size model trained centrally on MAIN. Shows the size cost.
- **PRE (θ_pre)** — the device-size model pretrained centrally on 1,217,610 *other* users who are assumed to have opted in to
  server-side training. Evaluated alone, it is the "cloud-only, shipped pretrained model" system. One seed (2026).

**Pretrained ("warm") federated line — PRIMARY since 06N (DL 282)**
- **FA_WARM (S7)** — federated training of all weights, starting from θ_pre.
- **FA_WARM_FROZEN (S8) = FA_WARM\*** — federated training from θ_pre with the three item tables frozen (not trained,
  never uploaded); only the encoder (263,822 parameters, MEASURED) is trained and sent. Phone-class payload.
- **FP_WARM_FROZEN / PF_WARM_FROZEN / FA_Q8_WARM_FROZEN** — FedProx (μ 0.01), personalised FL (a personal vector p_u kept
  on the device, λ 1e-4) and 8-bit stochastic-rounding uploads, each on top of S8.
- **FA_1024_WARM_FROZEN** — the non-private twin of the DP arms: Poisson sampling (each client joins a round with
  q = 2⁻⁷, about 1,024 clients per round), 384 rounds, clipping on, no noise.
- **DP8_FROZEN / DP1_WARM_FROZEN** — the same with Gaussian noise for ε = 8 (two seeds) / ε = 1 (seed 2026). Clip norm
  from `S_RECORD_WARM_FROZEN.json` (`2e47ba39…`, DL 300).
- **FT_FA_WARM_FROZEN / FT_FA_WARM** — the S8 / S7 model downloaded, then fine-tuned on each device with the registered
  FT_C procedure (cell FT_C_f0.3_P4). "The full production stack".
- **FT_PRE** — θ_pre downloaded and fine-tuned on each device (FT_C procedure). No MAIN user's data ever leaves the device:
  the cleanest download-only option (L5).

**Random-start ("cold") federated line — the ablation**
- **FA** — the S4 recipe started from random weights θ0, 13,654 MAIN rounds (`FL_T/FA_s2026|s2027`).
- **FP, PF, FA_Q8** — the cold method set. They are the **"N33" re-runs** (06N §N3.3): after the warm line had priority,
  these arms were re-run from scratch at the final S4 recipe in `FL_T/N33_<ARM>_s<S>`. The earlier partial runs
  `FL_T/FP_s2026`, `FP_s2027`, `FA_1024_s2026` were stopped for reallocation (STOPPED_REALLOCATED, 06N N4) and are never
  evidence (DL 317 D2; DL 331 M5).
- **FA_1024, DP8, DP1** — the cold DP chain (N33 runs), clip norm S = 0.284363 (`contracts/dp/S_RECORD.json`, DL 279).
  Noise multipliers z(ε 8) = 0.5642, z(ε 1) = 1.3094, δ = 5e-6, T = 384, user-level RDP accountant, Poisson sampling
  (06 §2; 06C).
- **FT_FA** — cold FA downloaded + local fine-tune, with its own selection `SELECTION_FT_FA.json` (`a645edaf…`, chosen on
  UNSEEN_VAL; DL 351).

**On-device and device-only**
- **FT_C** — C_SMALL downloaded + local fine-tune (cell FT_C_f0.3_P4 = 0.3 × the SMALL peak LR, 4 passes; selected on
  UNSEEN_VAL, `SELECTION_FT_C.json` `e47965d2…`, DL 231). Its base was trained on the same users' raw events (L1 base).
- **LO** — each device trains its own model from θ0 with no collaboration (cell LO_P4, DL 232). An untuned control, not
  the device-only ceiling.
- **C_SMALL_FP32 / C_SMALL_INT8** — C_SMALL exported to ONNX and run on CPU, FP32 vs dynamic INT8 (K16), on the first
  16,384 E2E users in selection-key order (fallback, disclosed; DL 371).

**Baselines** (CPU rank files, seed-free): **LAST**; **POP** (popularity); **SKNN** (session kNN, k 500, sample 5,000);
**RULE** (category / brand affinity + popularity); **KNN_EMB_BIG / _SMALL** (kNN on cached item-embedding tables of
C_BIG / C_SMALL).

**Declared not run (in ALL_METHODS_LOCK):** DP_OPT_SENS (superseded by the 06N reallocation); FA_X2 (12-EFE budget);
device panels on SPARSE / HEAVY / TRUE_COLD; TRUNC Study A / B and T41; FA_WARM panel rows; GRU_D128 re-evaluation (the
reeval tool accepts SASRec variants only); the raw-event bytes level (no registered EVENT_SIZE). Deferred or not adopted
earlier: CPF, KD_SMALL, FT_KD, DP ε = 4, seed 2028, DP8_FULL, C_SMALL_FULL, a December-2019 holdout.

## 6. The registered contrasts (ANA_CONTRASTS_v4, `f1769f0b…`, DL 336)

33 confirmatory contrast items. "R(a / b)" = retention in %; "a − b" = paired Δ in MRR@20. PB = PRACTICAL_BEST,
CB = CONTROLLED_BUDGET.

| Id | Contrast | Plain question | Tier | Rule | Seeds |
|---|---|---|---|---|---|
| **K20b** | R(FA_WARM_FROZEN / C_FULL) | How much of the matched cloud model does the deployable warm federated system keep? | **primary** | PB | 2 vs C_FULL s2026 |
| **K21** | FA_WARM_FROZEN − PRE | Does federated training on the private users add anything over the shipped pretrained model? | **primary** | PB | 2 |
| K1 | R(C_SMALL / C_BIG) | Cost of the smaller model | secondary | PB | 2 |
| K2 | R(FA / C_BIG) | Cold (random-start) federated retention — the ablation | secondary | PB | 2 |
| K3 | R(FA / C_SMALL) | Cost of federation at matched size (= K2 / K1) | secondary | PB | 2 |
| K8 | R(LO / C_BIG) | Device-only retention | secondary | PB | 2 |
| K9b | FT_C − C_SMALL on UNSEEN_HOLDOUT | Local fine-tuning gain for users the base never saw | secondary | PB | 2026 |
| K9c | FT_PRE − PRE | Local fine-tuning gain over θ_pre (the L5 answer) | secondary | PB | 2 |
| K11b | FA − LAST | Does cold FL beat the free device heuristic? | secondary | PB | 2 |
| K12 | R(DP8 / FA_1024) | Privacy cost at ε = 8 (cold) | secondary | **CB** | 2 |
| K17 | R(FA / C_FULL) | Cold FL vs the full-data cloud (= K2 × K15) | secondary | PB | 2026 |
| K20 | R(FA_WARM_FROZEN / C_BIG) | Warm FL vs C_BIG (mixes a data advantage) | secondary | PB | 2 |
| K20c | R(PRE / C_FULL) | The pretrained model alone vs C_FULL | secondary | PB | 2026 |
| K_INIT | FA_WARM_FROZEN − FA | Value of the pretrained start | secondary | PB | 2 |
| P5 | FT_FA_WARM_FROZEN − FA_WARM_FROZEN | Local fine-tuning gain after warm FL | secondary | PB | 2 |
| K4w | R(FP_WARM_FROZEN / C_FULL) | FedProx on the warm line | secondary | PB | 2 |
| K5w | R(PF_WARM_FROZEN / C_FULL) | Personalised FL on the warm line | secondary | PB | 2 |
| K12w | R(DP8_FROZEN / FA_1024_WARM_FROZEN) | Privacy cost at ε = 8 (warm) | secondary | **CB** | 2 |
| K13w | R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN) | Privacy cost at ε = 1 (warm) | secondary | **CB** | 2026 |
| K14w | R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN) | 8-bit uploads (warm); TOST ±1 % | secondary | PB | 2 |
| K4 | R(FP / C_BIG) | FedProx (cold) | tertiary | PB | 2 |
| K5 | R(PF / C_BIG) | Personalised FL (cold); "PF ≈ FA" is a registered possible outcome (06L A5) | tertiary | PB | 2 |
| K6 | R(FT_FA / C_BIG) | FL + local fine-tune (cold) | tertiary | PB | 2 |
| K7 | R(FT_C / C_BIG) | Cloud model + local fine-tune | tertiary | PB | 2 |
| K9 | FT_C − C_SMALL | Local fine-tuning gain | tertiary | PB | 2 |
| K10 | FT_FA − FA | Local fine-tuning gain after cold FL | tertiary | PB | 2 |
| K11 | FA − LO | Collaboration value | tertiary | PB | 2 |
| K13 | R(DP1 / FA_1024) | Privacy cost at ε = 1 (cold) | tertiary | **CB** | 2026 |
| K14 | R(FA_Q8 / FA) | 8-bit uploads (cold); TOST ±1 % on the 90 % CI | tertiary | PB | 2 |
| K15 | R(C_BIG / C_FULL) | Cloud data advantage | tertiary | PB | 2026 |
| K16 | R(INT8 / FP32), C_SMALL, ONNX-CPU block | On-device quantisation cost | tertiary | PB | 2026 |
| K18 | R(FT_C / C_FULL) | — | tertiary | PB | 2026 |
| K19 | R(LO / C_FULL) | — | tertiary | PB | 2026 |

Registered DESCRIPTIVE contrasts (no claim): S8 − S7 with the bytes ratio; R(FA_WARM / C_FULL); FA_WARM − PRE;
PF_WARM_FROZEN − FA_WARM_FROZEN; R(FA_1024_WARM_FROZEN / FA_WARM_FROZEN); FT_FA_WARM − FA_WARM; the PB ratios of K12 /
K13 / K12w / K13w (best-of-12 selection on the users' own validation data is outside the DP accountant, 06D §3).

**Algebraic dependencies (never independent evidence):** K3 = K2 / K1; K17–K19 = the matching K × K15;
K20b_s = K20_s × C_BIG_s / C_FULL; K20b − K20c = 100 · K21 / C_FULL (06L_C C4; 06P P7).

## 7. Hierarchy and multiplicity (how it evolved)

1. 06D §4 (DL 159): K2 primary; K1, K3, K8, K12 secondary; the rest tertiary.
2. 06L_C (DL 227): adds K11b, K9b, K9c, K20b, K20c (secondary) → ANA_CONTRASTS_v3, 26 contrasts (DL 229–230).
3. 06N (DL 282): the owner made the pretrained line primary — **K20b + K21**; K2 becomes the random-start ablation; new
   K_INIT.
4. 06P (DL 317; ERRATUM 1 DL 324; post-erratum sha `75f8dbad…`, DL 331): FA_WARM* = S8 in K20 / K20b / K21; the primary
   pair is read **unadjusted, outside Holm**, and a claim about the deployment branch requires both; **Holm step-down over 18
   secondary contrasts** (K1, K2, K3, K8, K9b, K9c, K11b, K12, K17, K20, K20c, K_INIT, P5, K4w, K5w, K12w, K13w, K14w);
   13 tertiary contrasts unadjusted, with a DESCRIPTIVE Holm band beside them (DL 336).

Other statistics reported beside every K (06L_C C13–C15): per-seed points, seed spread and a SEED_SENSITIVE flag (spread
> half the CI width), a user + seed interval, the harmed-client share (users whose difference is < 0) with a seed-vs-seed
null, a CALIB-disjoint diagnostic, REPEAT vs NOVEL target strata, and the K14 / K14w equivalence test (90 % CI inside
[99, 101]).

**Claim vocabulary (frozen with the script; REPORT §12):** "A exceeds B" only when the paired 95 % CI lies above the
reference (0 or 100); an interval that includes it reads "no detectable difference", never "equivalent" (only K14 / K14w
TOST can say "equivalent"); every retention sentence names the recipe and the reference ("matched-exposure retention of
the registered FL recipe"); a SEED_SENSITIVE flag or BELOW_DEVICE_HEURISTIC flag goes in the same sentence; non-DP FL is
never called private.

## 8. Checkpoint rules

- **PRACTICAL_BEST (PB):** the strictly best of the 12 evaluation marks (every 0.5 EFE within 6.0 EFE; a new best needs an
  improvement > 0.0002), evaluated on the users' own validation rows. The registered primary rule for every contrast except
  the DP ones. VALIDATION PB values are max-of-marks, so they are optimistic; TEST is the only unbiased number (06A §9(g)).
  Source: `contracts/EVALUATION_CONTRACT_V4.md` §12 L844–845; 06E §9(d).
- **CONTROLLED_BUDGET (CB):** the fixed 6.0-EFE endpoint (for DP arms the round-384 endpoint, a data-independent release).
  Confirmatory for K12, K13, K12w, K13w (06D §3); reported beside PB elsewhere. Device arms (LO, FT_*) and ONNX rows exist
  under PB only.
- **EFE** is the project's budget unit: 1 EFE = one exposure of every eligible TRAIN example of the cohort (`N_EFE`). All arms
  use 6.0 EFE (about 3 client visits in FL). The records never spell out the acronym.

## 9. Phase 2 in one line

PREG2 (`contracts/PREG2.md`, DL 411) is a separate registered follow-up study that tests whether a fine-tuning-specific
warm-FL recipe (W*) adds value over θ_pre (primary Q1 = P2_WSTAR − PRE), with central controls (C1, C2) that tell "MAIN
data adds nothing" apart from "the FL recipe failed". It is scored in the same single TEST sitting as Phase 1. Details in
file 05.

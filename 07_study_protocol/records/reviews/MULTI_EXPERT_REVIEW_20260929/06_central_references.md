# 06 — Central references: C_BIG, C_SMALL (170 MB device cap), C_FULL (full-data cloud), and PRE

Expert reviewer, part 5 (multi-expert review 2026-09-29). READ-ONLY. No server contact, no git, no test suite, no project script.
I opened no RESULT_ROWS, ranks, RETRO_TEST or runs_*/ value file. C_SMALL, C_FULL, PRE, the SMALL-ladder values and every
downstream value stay UNREAD by me. The only numbers I quote are ones that are already final and read: the G2 contest values in
`ARCHITECTURE_LOCK.json`, the P2 recipe metadata, and the CALIB FL-screen values that DECISION_LOG row 181 quotes.
Tags: **MEASURED** (from a file, path:line; paths are relative to `.local_ai/ppsi_v4/`), **REPORTED** (a number a project
document states), **INFERENCE**, **OPINION**. Citations are marked verified (checked by web search in this session) or UNVERIFIED.

---------------------------------------------------------------------------------------------------------------------
## 1. What the project does (MEASURED)

### 1.1 Denominators and the retention rule
- **Retention rule.** R = 100 × arm / C_BIG on client-macro MRR@20. It is the 2-seed mean of per-seed ratios, with a paired
  user bootstrap (1,000 resamples, seed 20260926). There is no pass/fail threshold (`contracts/PREG1_AMENDMENT_04.md:12-16`;
  `06E.md:61` item e).
- **Pairing rule.** The numerator and the denominator use the same checkpoint rule (PRACTICAL_BEST/PRACTICAL_BEST, or
  CONTROLLED_BUDGET/CONTROLLED_BUDGET) and the same seed (`06A.md:36-38`).
- **Confirmatory contrasts.** K1 is R(C_SMALL/C_BIG), the size cost. K2 is R(FA/C_BIG), the headline. K3 is R(FA/C_SMALL).
  K15 is R(C_BIG/C_FULL) (`06B.md:30-47`). K17–K19 are R(FA | FT_C | LO / C_FULL), seed 2026, PB/PB (`06I.md:7-15`).
  K13, K15 and K16 are single-seed by registration (`06E.md:18-21`). The contrast registry is
  `company/gates/ANA_CONTRASTS_v2.json`, which lists K15 as "if C_FULL runs".

### 1.2 C_BIG
- **Definition.** C_BIG is the existing 02B_SASREC contest runs, not a re-run. PRACTICAL_BEST = best.pt (the 3.0-EFE mark) and
  CONTROLLED_BUDGET = endpoint_6.0.pt. The checkpoints are pinned by sha256 (`contracts/ARCHITECTURE_LOCK.json:15-43`).
- **Model.** SASREC_D256_B3, 83,881,764 parameters (`ARCHITECTURE_LOCK.json:169-173`, `:183-189`).
- **Recipe.** It comes from P2 trial `P2_SASREC_D256_B3_S0_s0`: peak LR 4.0e-4, which is the lowest of the four SASRec LR strata
  (`contracts/hp_registry/HP_REGISTRY_P2R2.json` lr_grid.SASREC, stratum 0). It uses schedule S0 (peak until 2.0 EFE, 0.1× by
  3.0, 0.01× at 6.0), dropout 0.1, effective batch 256, and AdamW (same file, `schedules`/`fixed`).
- **How much tuning.** At its own size, P2 ran four confounded (schedule, LR, dropout) cells on CALIB, and the 6.0-endpoint value
  selected among them (`HP_REGISTRY_P2R2.json` cells.SASREC, selection). The size was then chosen on MAIN as max-of-2 on
  MAIN_VALIDATION (`05.md:30-37, 52-57`). The winning LR sits at the lower edge of the grid and was not extended. This is
  disclosed (`company/reviews/G1B_PART2_FABLE_RULING.md:20`).
- **Values.** MAIN_VALIDATION 0.16120 (2-seed mean). The seed gap is 2.5e-5 (`ARCHITECTURE_LOCK.json:92-105, 144-148`). These
  values are final and were read at G2.

### 1.3 C_SMALL
- **Registered rule.** SMALL is "the winner's family with the model dimension d reduced until the per-visit dense payload
  (down + up, FP32) is ≤ 170 MB (today's GRU_D128 is 168 MB); everything else identical to BIG (blocks, heads, ff, tying,
  dropout)" (`03.md:11-13`).
- **Named variant.** The same amendment names SASRec → **SASREC_D64_B2** (`03.md:16`).
- **Registered values.** SMALL = SASREC_D64_B2, 20,710,048 parameters, 165,680,392 B per visit (`06.md:9-11`;
  `ARCHITECTURE_LOCK.json:172, 257-261`). The variant is "2 blocks … everything else identical to SASREC_D256_B2"
  (`contracts/model_variants/MODEL_VARIANT_REGISTRY.json` variants.SASREC_D64_B2). BIG = D256_**B3** (variants.SASREC_D256_B3:
  "one more causal block (3 blocks)").
- **Recipe ladder.** Four LR trials on CALIB at the four SASRec midpoints. Schedule S0 and dropout [0.1] are inherited from
  02B_SASREC. Seed 2026 ran all four, the best was re-run at seed 2027, and the 02R1 §3 rule selected
  (`03.md:25-29`; `code/v4int/small_recipe.py:10-15, 52`).
- **Selected recipe and runs.** Peak 0.000711 (stratum 1) (DECISION_LOG row 143). C_SMALL was trained on the **A16** on
  2026-09-28 from 14:51Z to 16:09Z. It had no pre-launch dry run (quota), and its config hashes come from LAUNCH_0 (DL 143–145).
- **Sanity trigger.** C_SMALL CONTROLLED_BUDGET < 0.5 × C_BIG PRACTICAL_BEST (`06E.md:42-43`).

### 1.4 C_FULL
- **Definition.** The same recipe, 02B_SASREC, unchanged. The population is the TIEBREAK band, ranks [0, 1,348,682) = 1,348,682
  users / 26,677,600 TRAIN decisions. That is ≈ 10.3× MAIN, and it includes MAIN. It is evaluated on the same MAIN_VALIDATION rows
  and gets automatic C3 continuation up to 8.0 EFE (`06.md:23-28`; `06D.md:91-94`; `company/handoffs/DATA-08.md`).
- **Launcher.** A shim that runs the unedited launch_central (`code/v4int/c_full.py:1-22, 58-60`).
- **Order and tier.** It runs before ALL_METHODS_LOCK (`06B.md:81-82`) and is SHOULD-high (`06I.md:74`).
- **Runs.** Seed 2026 has run on the A40 since 08:45Z and seed 2027 (NICE) since 09:23Z, both from code_ds3r2. They share cards
  with other jobs, a registered deviation from "alone on the card" (DL 175, 176, 179). At 16:05Z both were at 2.0 EFE (DL 221).
  The plan's estimate was "6–11 h" (`company/plans/MASTER_PLAN_V5.md:149`). REPORTED later: ≈ 25–30 h per seed under contention
  (DL 198, 213).

### 1.5 PRE and the deployment branch
- **PRE.** The SMALL recipe on the non-MAIN part of TIEBREAK: 1,217,610 users / 24,089,824 decisions. θ_pre = endpoint_6.0 with no
  selection. Its marks on UNEW_VAL are descriptive (`06K.md:8-16`; `company/handoffs/DATA-10.md`).
- **DL 206 rule.** R(FA_WARM/C_FULL) and R(PRE/C_FULL) are reported as DESCRIPTIVE beside K20 = R(FA_WARM/C_BIG), and the
  deployment-branch text "leads with them".
- **Other disclosures in DL 206.** One θ_pre seed. The SMALL recipe is applied at 9× data without re-tuning ("can only
  understate PRE").

### 1.6 A16 → A40
- **The pin.** A second frozen device class, NVIDIA_A40-FP32-DET, has an O_EXCL pin file (`06D.md:13-30`). The pin binds the GPU
  name and the torch/CUDA versions. It records the driver but does not check it at run time (`code/v4eval/device_pin.py:54-70,
  112-118`). It needs a passing T3 receipt and a STRICT receipt with two bitwise-identical reps for D64_B2 and D256_B3
  (`device_pin.py:81-105`).
- **Re-evaluation.** C_BIG and C_SMALL (PB and CB, both seeds = 8 re-evaluations) were re-evaluated on A40 GPU 0 through the frozen
  T3 path, with values unread (DL 163, 169). T3 rank differences between A40 and A16 for a random-init model are 12,635 / 14,063
  rows, which is why the class is separate (`06E.md:62-63`).
- **One block.** All retention is computed inside the A40 block (`06E.md:30-34`). v4eval is frozen at hash 61b64e61 for the whole
  block (DL 180).

---------------------------------------------------------------------------------------------------------------------
## 2. State of the art: how FL papers and real systems pick the "cloud" and size the device model

### 2.1 Centralised baselines in FL research
- **McMahan et al. 2017**, *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS, PMLR 54:1273–1282
  (https://proceedings.mlr.press/v54/mcmahan17a.html, verified). This is the FedAvg reference practice: the model is trained
  centrally on the pooled data of the same clients. It is a matched-data reference, not a stronger system.
- **Hard et al. 2018/2019**, *Federated Learning for Mobile Keyboard Prediction*, arXiv:1811.03604
  (https://arxiv.org/abs/1811.03604, verified; I read pages 1–4 of the PDF).
  - The production "cloud" was a server model trained on opted-in logs (≈ 7.5 B sentences), a different population than the FL
    clients. The **FL model beat the server model** in recall.
  - The model was a 1.4 M-parameter CIFG with V = 10,000 and tied embeddings, shipped at **1.4 MB** after quantisation. The paper
    says keyboard LMs "are usually limited to tens of megabytes".
  - Clients needed ≥ 2 GB of memory, charging, an unmetered network and idle time. Each round used 100–500 clients, with a
    Nesterov-momentum server optimiser.
  - Lessons for this project: (a) "the cloud" in production is not necessarily an upper bound; (b) the opt-in-logs server model is
    exactly what the PRE branch simulates; (c) the production FL payload was about 100× smaller than here.
- **Xu et al. 2023**, *Federated Learning of Gboard Language Models with Differential Privacy*, ACL 2023 Industry Track
  (https://aclanthology.org/2023.acl-industry.60/, verified). This is production DP-FL at scale. The model sizes are not in the
  abstract (UNVERIFIED here).
- **Bonawitz et al. 2019**, *Towards Federated Learning at Scale: System Design*, MLSys (https://arxiv.org/abs/1902.01046,
  verified). The production FL system: eligibility, pace steering, bounded model/plan sizes.
- **Wang et al. 2021**, *A Field Guide to Federated Optimization*, arXiv:2107.06917 (https://arxiv.org/abs/2107.06917, verified).
  It recommends simulating realistically and tuning both sides of a comparison.

### 2.2 FL recommender papers
- Centralised versions of the same model are used as references: **FCF**, Ammad-ud-din et al. 2019, arXiv:1901.09888
  (https://arxiv.org/abs/1901.09888, verified); **FedNCF**, Perifanis & Efraimidis 2022, Knowledge-Based Systems 242:108441
  (doi 10.1016/j.knosys.2022.108441, verified).
- These papers usually train on the same users and call the result an "upper bound" or "centralised baseline" (OPINION from
  familiarity; the exact wording per paper is UNVERIFIED).
- **Efficient-FedRec**, Yi et al. 2021, EMNLP, pp. 2814–2824 (https://aclanthology.org/2021.emnlp-main.223/, verified). It keeps
  the large item/news model on the server and ships a light user model to clients. It says outright that federating the whole
  recommender is too costly in computation and communication for clients. This is the standard way around full item tables on
  the device.

### 2.3 Distillation for small or on-device recommenders
- **Tang & Wang 2018**, *Ranking Distillation*, KDD (https://arxiv.org/abs/1809.07428, doi 10.1145/3219819.3220021, verified). A
  student under half the teacher's size approaches the teacher's ranking quality. Production practice ships a distilled student,
  not a small model trained from scratch.

### 2.4 Scale and re-tuning
- **Zhang et al. 2024**, *Scaling Law of Large Sequential Recommendation Models*, RecSys (doi 10.1145/3640457.3688129, verified).
  Quality follows power laws in both data and model size. A recipe tuned at one data scale is not assumed optimal at another
  (OPINION; standard scaling-law practice re-tunes the LR and the budget per scale).
- **Rendle, Zhang & Koren 2019**, *On the Difficulty of Evaluating Baselines*, arXiv:1905.01395 (verified). Under-tuned references
  distort every ratio built on them. This applies here to C_BIG (its LR is at the grid edge), C_SMALL (inherited
  schedule/dropout) and C_FULL (untuned at 10× data).

### 2.5 Device and browser limits (verified)
- **ONNX Runtime Web.** A protobuf model is at most 2 GB; WebAssembly memory is at most 4 GB; "no way … to run models larger than
  4 GB" (https://onnxruntime.ai/docs/tutorials/web/large-models.html).
- **ORT Web training.** Introduced in 1.17 for in-browser FL and personalisation
  (https://opensource.microsoft.com/blog/2024/02/06/on-device-training-training-a-model-in-browser/). All ORT training packages
  are deprecated, and 1.19.2 was the last release (https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0).
- **Chrome built-in AI (Gemini Nano).** It needs ≥ 22 GB free, > 4 GB VRAM, or 16 GB RAM with 4 cores
  (https://developer.chrome.com/docs/ai/get-started).
- **Apple Intelligence.** It uses a ≈ 3 B-parameter on-device model (Gunter et al. 2024, arXiv:2407.21075,
  https://arxiv.org/abs/2407.21075, verified; the quantisation bits per weight are UNVERIFIED here).
- What this means: browsers already download multi-GB models for **inference**. FL **per-round payloads** in production are
  MB-scale (Gboard 1.4 MB).

### 2.6 Benchmarks
- **FedScale**, Lai et al. 2022, ICML, PMLR 162 (https://proceedings.mlr.press/v162/lai22a.html, verified). It ships device and
  bandwidth traces that could anchor a per-visit cap. I did not check their exact numbers (UNVERIFIED).
- **FLUTE**, Dimitriadis et al. 2022, arXiv:2203.13789 (verified). Microsoft's simulation platform for millions of clients.
- **LEAF**, Caldas et al. 2018, arXiv:1812.01097 (from memory; UNVERIFIED this session).

---------------------------------------------------------------------------------------------------------------------
## 3. Comparison with the state of the art

### 3.1 Strengths (MEASURED unless marked)
- **S1. Correct denominators for the paradigm question.** C_BIG is trained on exactly the users FL trains on. It shares θ0, the
  features, the evaluator, the rows and the checkpoint rule, so K2 isolates the paradigm cost. This is the FedAvg-paper practice.
- **S2. Two references, both registered before any read.** C_BIG is the matched reference and C_FULL is the Edge-scale cloud
  (K15, K17–K19). C_FULL is realistic in data scale and serving: all opted-in users, with the last 50 events sent at query time
  (`REALISM_AUDIT_20260929.md:A2`). Few MSc or even published FL-recsys studies separate these two.
- **S3. The pairing is done right.** Same rule, same seed, mean of per-seed ratios, and a paired bootstrap over the same users
  (`06A.md:36-38`). For the DP contrasts, CONTROLLED_BUDGET is confirmatory because PB selection happens outside the accountant
  (`06D.md:53-60`).
- **S4. The device pin is sound.** It was recognised as necessary (T3 rank differences), receipted, and O_EXCL. Retention is never
  mixed across device classes (`06D.md:13-30`; `06E.md:30-34, 50-53`). The C_BIG/C_SMALL re-evaluation is evaluation only, with
  identity checks on the checkpoint, RUN_START, θ0 and the provider fingerprint (`code/v4int/reeval.py:173-195, 252-268, 601-604`).
- **S5. The DL 206 rule is the right correction.** PRE saw 9× MAIN, so R(FA_WARM/C_BIG) can exceed 100. Reporting R(·/C_FULL)
  beside it, where C_FULL = the PRE users plus the MAIN users, is the data-matched reading.
- **S6. The SMALL recipe has a clean four-point LR sweep, the same count as BIG's four P2 cells at its size.** The trigger
  "C_SMALL < 0.5 × C_BIG" guards against a broken SMALL.
- **S7. Single-seed C_FULL is defensible.** The SASRec MAIN seed gap is 2.5e-5 (`ARCHITECTURE_LOCK.json:144-148`), far below any
  plausible retention effect.

### 3.2 Errors
- **E1 (MUST, code). Neither C_FULL nor PRE can be re-evaluated by the re-evaluation tool.**
  - `reeval.py:557-559` refuses every central checkpoint whose source run's (train, eval) manifests are not REGISTRY_V2's MAIN pair.
    `base_reg` is the lock-bound REGISTRY_V2 (`reeval.py:524-526`).
  - C_FULL trains on `FULL_TRAIN_5_100_TIEBREAK` (`c_full.py:1-2, 58`), so reeval refuses it for every role, including
    `MAIN_TEST`: the check sits in the central branch before the role branch.
  - PRE trains on `PRETRAIN_TRAIN_5_100_NONMAIN` and evaluates on UNEW_VAL (DATA-10 handoff). It is refused too. Even with the pair
    check relaxed, the MAIN-role provider-fingerprint check at `reeval.py:603-604` would refuse it.
  - What depends on this path:
    - C_FULL PB/CB on MAIN_TEST at the single TEST look (`06D.md:43`; `MASTER_PLAN_V5.md:190`);
    - the planned uniform C_FULL VALIDATION re-evaluation (`MASTER_PLAN_V5.md:155`);
    - PRE-alone on MAIN_VALIDATION, which is registered as always reported and feeds K21 = FA_WARM − PRE and R(PRE/C_FULL)
      (`06K.md:25-26`).
  - K15 and K17–K19 are confirmatory. Without a C_FULL MAIN_TEST row they cannot be computed on TEST.
  - The INT-07 TEST-look commands show only the C_BIG central example (`company/handoffs/INT-07.md:122-129`). REV-DELTA-02 flagged
    the PRE case only as "unverified" MINOR (`company/reviews/REV-DELTA-02_OPUS_REVIEW.md:294`).
  - Status: INFERENCE from reading the code, high confidence. A server `--dry-run` would confirm it in seconds.
  - VALIDATION is not blocked for C_FULL, because its sealed in-run rows are already A40 pinned rows (`MASTER_PLAN_V5.md:155`).
  - Fix: a code-only path with no rule change. Add registered source pairs: C_FULL = FULL_EXT role C_FULL; PRE = PRETRAIN_EXT
    role PRETRAIN evaluated on the MAIN role. Rebuild the key base from the source run, keep the fingerprint check for the eval
    provider actually used, then review and dry-run on the server before ALL_METHODS_LOCK.
- **E2 (DISCLOSE). SMALL does not follow the registered "everything else identical to BIG (blocks …)" rule.** BIG = D256_B3 has 3
  blocks and SMALL = D64_B2 has 2 (`03.md:12` vs `MODEL_VARIANT_REGISTRY.json`). The amendment contradicts itself: it named
  D64_B2 before the size was known. Under the rule, SMALL would be D64_B3, only about 0.05 M more parameters and still ≤ 170 MB
  (INFERENCE). So K1 measures d 256 → 64 **and** depth 3 → 2 **and** an LR re-selection together. It is not "the cost of d".
  Redoing SMALL now would invalidate C_SMALL, the running FL ladder (S1b/S2 CALIB, LADDER_S1a), PRE and the device lanes. That is
  not proportionate. Disclose it, and call K1 "BIG→SMALL model cost".
- **E3 (DISCLOSE / wording). Amendment 03 §3 says the mini-ladder asymmetry "only guards against overstating the size cost". Only
  half of that holds.**
  - BIG's LR was the lowest grid point and was not extended (§1.2). An under-tuned C_BIG **understates** the size cost.
  - SMALL inherits a schedule and dropout that were chosen for a model 4× larger. An under-tuned C_SMALL **overstates** it.
  - The net direction of K1's bias is unknown. The sentence should be replaced by that two-sided statement.

### 3.3 Weaknesses and missing pieces
- **W1. K17–K19 (and R(FA_WARM/C_FULL), R(PRE/C_FULL)) mix three effects.**
  - K17 = R(FA/C_FULL) combines (i) data scale (FL on 131 k users vs a cloud on 1.35 M), (ii) model size (D64_B2 vs D256_B3) and
    (iii) federation.
  - In a real Edge deployment FL would run over all 1.35 M devices too. So K17 is **not** "FL vs a realistic cloud". It compares a
    MAIN-scale FL run with an Edge-scale cloud, and it is a lower bound on FL at deployment scale. 06I §6a says FL is a lower bound
    only for the round budget. The same holds for scale.
  - The only full-scale FL point is DP8_FULL (`06I.md:46-52`, SHOULD, and only if the pool-driver receipt exists).
  - Exact identity for seed 2026 under one pairing rule: R(FA/C_FULL) = R(FA/C_SMALL) · R(C_SMALL/C_BIG) · R(C_BIG/C_FULL). The
    thesis should show this decomposition table beside K17 (zero cost).
- **W2. There is no size-matched full-data cloud reference.**
  - R(PRE/C_FULL) is meant to answer "what does it cost to leave the private users' data out of training?" As defined it also
    carries the D256_B3 → D64_B2 step. R(FA_WARM/C_FULL) is data-matched but not size-matched.
  - A **C_SMALL_FULL** run would complete a clean chain at deployment scale:
    C_FULL (BIG, full) → C_SMALL_FULL (size cost at full scale) → PRE (cost of excluding MAIN's data) → FA_WARM (value of FL on the
    private users).
  - That run is the SMALL recipe on TIEBREAK, with the same shim pattern as PRE and ≈ PRE's cost: ≈ 8.5–12 h alone (DATA-10
    estimate) and more under contention (INFERENCE).
  - It also answers a question an examiner will ask: does the size cost shrink or grow with 10× data? (Zhang et al. 2024 scaling
    laws.)
- **W3. C_FULL's recipe is untuned at 10× data, and the bias runs in the direction that flatters FL for every "vs C_FULL" ratio.**
  - The recipe was tuned on CALIB, which is about 1/40 of TIEBREAK. Its LR is at the grid's low edge. Its S0 decay (0.1× by 3.0
    EFE) was shaped by over-fitting on 2.6 M decisions: C_BIG peaks at 3.0 and then loses about 0.0085 by 6.0 (REPORTED,
    `FABLE_THESIS_RESEARCH_REVIEW_20260928.md:149`).
  - At 26.7 M decisions a 6.0-EFE budget is about 625 k steps. Over-fitting is weaker there, so a higher LR or a longer constant
    phase is plausibly better (OPINION).
  - Tuning cannot lower the validation optimum. So an untuned C_FULL understates the cloud, and R(·/C_FULL) is biased **upward**.
    That flatters FL, FT_C and LO.
  - The realism audit (A2) and deep review F10 phrase this as "can only understate the cloud advantage". That is true, but the
    thesis must add what follows from it: K15 and K17–K19 are upper bounds with respect to C_FULL's tuning, while FL at MAIN scale
    is a lower bound with respect to scale. The two biases run in opposite directions for K17.
  - Report C_FULL's PB mark position and its learning curve, and whether C3 continuation triggered. If PB = 6.0 or later, label
    C_FULL "undertrained".
- **W4. The 170 MB cap is set from an internal model, not an external constraint.** It was fixed as "today's GRU_D128 is 168 MB"
  (`03.md:12`) and justified only by the owner's remark that ≈ 1 GB per visit is unrealistic (DL 83). Measured against real
  systems:
  - **Inference download.** D64_B2 is 82.8 MB FP32 or about 21 MB INT8 (INFERENCE). That is realistic for a browser feature. ORT
    Web allows up to 4 GB, and Chrome already downloads multi-GB Gemini Nano. By the same standard, BIG itself (≈ 335 MB FP32,
    ≈ 84 MB INT8; the checkpoint bytes are at `ARCHITECTURE_LOCK.json:19`) would also be deployable for **inference**. The size
    step (K1) is therefore a choice about communication and training cost, not an inference limit.
  - **FL per-visit payload.** 165.7 MB down+up per visit is about 100× the Gboard FL model (1.4 MB), and it only fits with the
    Gboard-style "unmetered network, charging, idle" eligibility. It is plausible for laptops on Wi-Fi and heavy for mobile.
  - **On-device training memory.** Parameters + gradients + AdamW m, v ≈ 4 × 82.8 MB ≈ 331 MB plus activations. That is feasible
    on desktops, and the WASM 4 GB limit is not binding. But browser training tooling (ORT Web training) is deprecated
    (`REALISM_AUDIT_20260929.md:C3`).
  - The cap is dominated by the dense full-catalogue item tables (≈ 98 % of the parameters). Production FL recommenders send item
    rows sparsely, keep item models on the server (Efficient-FedRec) or freeze them (the project's S8). So the cap mostly limits
    the dense-table design, not model capacity.
  - Required text: state the cap's origin; separate "device inference footprint" (download once, cacheable) from "per-visit FL
    payload"; cite Hard et al. 2018 and the ORT Web and Chrome requirements; note that the optional 85 MB tier was not taken
    (`03.md:19`).
- **W5. KD_SMALL, the production way to build a small model, is deferred** (DL 89; `06.md:20`; `06D.md:102` NICE). Without it, K1
  overstates the size cost a real Edge team would pay: a cloud teacher distilled into the on-device student (Tang & Wang 2018).
  This matters most for the L1 row (cloud-train / device-infer). It is already registered (`04.md:21`), so running it needs no new
  pre-registration, only a DL row before launch.
- **W6. "Upper bound" wording.** The proposal's Step 1 is a "centralized upper bound", but C_BIG and C_FULL are not bounds. Hard
  et al. show FL beating the server model, and K20 can exceed 100 by construction (DL 206). The thesis should say "matched central
  reference" (C_BIG) and "Edge-scale cloud reference" (C_FULL).
- **W7. Device-class continuity across the two physical A40s is not receipted.**
  - The T3 and STRICT receipts and all 8 re-evaluations ran on GPU 0 (DL 161–169). C_FULL s2026, the FL lanes and the device lanes
    also run on GPU 1.
  - The pin matches by SKU name and torch/CUDA only (`device_pin.py:63-69`). The driver is recorded but not checked.
  - Same SKU, driver and wheel should give identical kernels (INFERENCE), but there is no receipt.
  - Cheap check: re-evaluate one C_BIG checkpoint on GPU 1 (about 3 min) and compare the ranks sha256 with the GPU 0 re-evaluation,
    bitwise. No value is read, only a hash equality.
- **W8. The A16-vs-A40 "device sensitivity" also spans an evaluator-code revision.** The A16 block has
  `evaluator_code_sha256 6cbc3e6f…` (`ARCHITECTURE_LOCK.json:53`) and the A40 block 61b64e61 (DL 180). INT-06 says the change is
  metadata only. Disclose that the sensitivity mixes device and code-revision metadata.
- **W9. C_BIG/C_SMALL were trained on A16, but their PB checkpoints were chosen by A16 marks and evaluated on A40.** The effect is
  negligible (INFERENCE), but it belongs in 06D §9a's disclosure.
- **W10. The C_FULL schedule is at risk.** The plan assumed 6–11 h; the measured rate implies about 25–30 h per seed under
  contention, plus up to 33 % more for continuation (REPORTED, DL 198/213). K15 and K17–K19 hold ALL_METHODS_LOCK until C_FULL
  s2026 finishes. DL 213 advice (pause s2027, NICE, which resumes bitwise) should be applied whenever C_FULL s2026 or the critical
  FL screens compete.
- **W11. VALIDATION interim ratios are biased in known directions.** C_BIG's VALIDATION value is max-of-2 candidates and max-of-12
  marks on the same rows. Central and FL arms are max-of-marks, but LO, FT_C and FT_FA are selected on UNSEEN_VAL, so they get no
  selection on MAIN rows. The interim device-arm retentions are therefore biased downward relative to C_BIG. TEST removes this
  (the checkpoints are fixed). It is disclosed generally (`06A.md:47` item g). The thesis should say which arms are affected.
- **W12. C_BIG's CONTROLLED_BUDGET checkpoint is past its optimum** (best at 3.0, falling by 6.0). CB retentions vs C_BIG are
  therefore inflated for arms that do not over-fit. Label them so, and never headline them.

### 3.4 Unrealistic choices (OPINION, graded)
- A 90 % opt-in cohort for PRE (1,217,610 of 1,348,682 users). Real data-sharing opt-in rates are far lower (UNVERIFIED; no
  source checked). DL 206 treats PRE as opt-in or public data. The thesis should call it "the most favourable opt-in fraction".
  If time permits, add a PRE on a 10 % or 1 % slice of the non-MAIN band (NICE, new registration).
- No daily retraining, and the cloud sees only 50 events of history (REALISM_AUDIT A2). This is symmetric across arms: disclose,
  no change.

---------------------------------------------------------------------------------------------------------------------
## 4. Recommendations

Pre-registration state (MEASURED, DL 181–224; `06I.md:4-5`; `06K.md:5-6`): **no** C_SMALL, C_FULL, PRE, FL-MAIN or device value
has been read. The only reads are the CALIB FL screens. So an addition registered now, before the VALIDATION preflight and the ANA
freeze, is legitimate. Changing a registered rule that governs an unread value is still allowed by the project's own rules
(amendment + DL row), but I recommend none except R4.

| ID | Item | Class | Action | Cost (INFERENCE) | Affects | Pre-registration needed / status |
|---|---|---|---|---|---|---|
| R1 | Make `reeval.py` accept the registered C_FULL (FULL_EXT) and PRE (PRETRAIN_EXT → MAIN role) central sources. Review it and dry-run it on the server for C_FULL PB/CB and θ_pre before ALL_METHODS_LOCK. Add the C_FULL MAIN_TEST commands to INT-07's list | **MUST** | CHANGE (code) | 2–4 h Opus + review; minutes of GPU | TEST look (K15, K17–K19); PRE-alone, K21, DL 206 rows; no finished run | No (a tool, not a rule); DL row + REV entry. v4eval stays untouched (the block hash is safe) |
| R2 | Register, before any C_FULL or PRE read: (a) the pairing for R(FA_WARM/C_FULL) and R(PRE/C_FULL) (seed 2026, PB/PB, as K17); (b) K20/K21 in ANA_CONTRASTS_v3 (REV-DELTA-02 m11) | **MUST** | ADD | 30 min | ANA freeze, deployment-branch table | Yes; not yet done (m11 open). C_FULL and PRE unread, so it is legitimate |
| R3 | Thesis text: K17–K19 = "MAIN-scale FL / device arm vs Edge-scale cloud" = a lower bound for FL at deployment scale; show the exact identity K17 = K3(s2026)·K1(s2026)·K15; name the three effects | **MUST** | ADD (analysis/text) | 0–1 h (ANA descriptive table) | write-up, ANA (before the freeze) | Registration of the descriptive decomposition before the freeze is recommended (no value read) |
| R4 | Add **C_SMALL_FULL** (SMALL recipe on TIEBREAK, seed 2026, endpoint and PB, same shim as PRE) so that size cost at full scale, PRE cost and FA_WARM value are size-matched | SHOULD | ADD | ≈ 9–20 GPU-h A40 (co-scheduled) + a small shim | deployment-branch decomposition; none finished | Yes: new amendment + DL row before launch and before any C_FULL/PRE read (both unread now) |
| R5 | Disclose C_FULL's untuned recipe and its **direction** (R(·/C_FULL) biased upward → flatters FL, FT_C, LO); report the PB mark position, the curve, and whether continuation triggered | **MUST** (disclosure) | DISCLOSE-ONLY | 0 | write-up | No |
| R6 | Disclose SMALL = D64_**B2** vs the rule's "blocks identical to BIG" (BIG = B3); rename K1 to "BIG→SMALL model cost (d, depth, LR)" | SHOULD | DISCLOSE-ONLY (KEEP the runs) | 0 | write-up | No (the governing rule is not changed; the deviation is recorded) |
| R7 | Replace 03 §3's "only guards against overstating" with the two-sided statement (BIG LR at the grid edge vs SMALL's inherited schedule/dropout) | SHOULD | DISCLOSE-ONLY | 0 | write-up | No |
| R8 | 170 MB cap: state its origin (= GRU_D128); separate the inference footprint (82.8 MB FP32 / ≈ 21 MB INT8, download once) from the per-visit FL payload (165.7 MB); compare with Gboard 1.4 MB, the ORT Web 4 GB limit and Chrome Gemini Nano; say that BIG would also be inference-deployable, so K1 is a communication/training choice | **MUST** (disclosure) | DISCLOSE-ONLY | 0 | write-up, "device guidelines" | No |
| R9 | Run KD_SMALL (registered in 04 §2; deferred) if any GPU slot opens after the MUST tier; report it on the L1 row | SHOULD | KEEP registration, run | ≈ 4–8 GPU-h (the 04 §2 estimate was 4 on A16 rates) | K1 interpretation, L1 row | No new registration (04 §2); DL row before launch |
| R10 | Cross-GPU check: re-evaluate one C_BIG checkpoint on GPU 1 and compare the ranks sha256 with GPU 0 (hash only) | SHOULD | ADD | ≈ 5 min | the validity of every GPU-1 row in the A40 block | No; DL row with the hash equality |
| R11 | Terminology: "matched central reference" / "Edge-scale cloud reference", never "upper bound"; cite Hard et al. 2018 (FL beat the server model) | SHOULD | CHANGE (text) | 0 | write-up | No |
| R12 | Give C_FULL s2026 GPU priority over C_FULL s2027 and the non-critical lanes (DL 213) | SHOULD | KEEP / operate | 0 | schedule to ALL_METHODS_LOCK | No |
| R13 | Disclose: the A16/A40 sensitivity also spans the evaluator hash (6cbc3e6f → 61b64e61); the PB checkpoints of C_BIG/C_SMALL were chosen on A16 marks | DISCLOSE-ONLY | DISCLOSE | 0 | write-up | No |
| R14 | Label CB retentions vs C_BIG as "C_BIG past its optimum (best@3.0)"; say which interim VALIDATION ratios are biased by the device arms' different selection | DISCLOSE-ONLY | DISCLOSE | 0 | write-up | No |
| R15 | PRE opt-in realism: call 90 % "most favourable"; optionally a PRE on a 10 % non-MAIN slice | NICE | ADD | ≈ 1–2 GPU-h + a manifest | deployment branch | Yes, a new amendment before any PRE read |
| R16 | Optional FA_FULL (non-DP FL at TIEBREAK scale) to give K17 a scale-matched FL numerator | NICE (cost too high for the deadline) | DISCLOSE if not run | ≈ 10× an FA run (≈ 100 h single-process; the pool driver may cut this) | K17 interpretation | Yes, if run |

---------------------------------------------------------------------------------------------------------------------
## 5. Score: 7 / 10

The central references are well designed for a pre-registered study.
- C_BIG is a correctly matched denominator. The pairing rules are right.
- C_FULL was added before any read and gives the Edge-scale view, and DL 206 correctly re-bases the deployment branch on it.
- The A40 pin is careful: receipted, O_EXCL, one block.
- Single-seed C_FULL is justified by a measured seed gap of 2.5e-5.

Three things keep the score from 8–9:
1. **One real execution gap (MUST).** The re-evaluation tool refuses C_FULL and PRE checkpoints (`reeval.py:557-559`), yet the TEST
   look and the deployment branch depend on them.
2. **The "vs C_FULL" contrasts mix three effects and are biased in stated but unquantified directions.** They combine scale, size
   and federation. The C_FULL recipe is untuned, so they are biased up; FL is not scaled, so it is biased down. No size-matched
   full-data reference exists.
3. **The SMALL definition is less principled than presented.** The 170 MB cap is anchored to an internal model. It conflates
   inference footprint with FL payload. It deviates from its own "blocks identical" rule. And its tuning-asymmetry claim is
   one-sided.

All of these are fixable before the first read, with a small code task, registrations, disclosures and optionally one run
(C_SMALL_FULL). None needs a finished run to be redone.

---------------------------------------------------------------------------------------------------------------------
## 6. UNVERIFIED items and open questions
- **Q1.** Does any server-side patch after the local `reeval.py` (mtime 2026-09-29 17:02 local) already handle FULL_EXT/PRETRAIN
  central sources? The local code refuses them. A `reeval --dry-run` of C_FULL best.pt and θ_pre on the server settles it (Ops;
  I did not touch the server).
- **Q2.** Does `ana_block_index.py` accept C_FULL's sealed in-run rows as the VALIDATION C_FULL row (06E §5 identifies them by
  train_manifest_id)? I did not trace this; it is part 12's scope.
- **Q3.** Did C3 continuation trigger for C_BIG, and at which mark is C_SMALL's PB? Both are sealed or unread for me, so I did not
  check.
- **Q4.** The model sizes in Xu et al. 2023 (Gboard DP LMs), the FedScale bandwidth traces, and Apple's on-device bits per weight
  were not extracted (UNVERIFIED). Real opt-in rates for browser data sharing: no source checked (UNVERIFIED).
- **Q5.** LEAF (Caldas et al. 2018, arXiv:1812.01097) is cited from memory (UNVERIFIED).
- **Q6.** Whether the C_FULL runs on code_ds3r2 carry the same `evaluator_code_sha256` (61b64e61) as the eight re-evaluations.
  This is expected per INT-06 (DL 180); it is a hash check for ANA-01's comparison key.
- **Q7.** The INT8 size of D64_B2 (≈ 21 MB) is an INFERENCE until the DEV bench records it. It depends on whether
  `quantize_dynamic` quantises the Gather tables (REALISM_AUDIT §I).

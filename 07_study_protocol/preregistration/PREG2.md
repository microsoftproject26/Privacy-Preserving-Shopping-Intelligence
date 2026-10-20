# PREG2 — Phase-2 pre-registration (REGISTERED 2026-10-03, DL 411)

- Status: **REGISTERED** by DECISION_LOG row 411, which records this file's sha256 (written O_EXCL, read-only).
  Built from contracts/PREG2_FOLLOWUP_DRAFT.md (kept unchanged so the two can be diffed) by applying every edit and
  condition of company/reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md §9 (GO-WITH-EDITS), then the five textual edits
  of §10 (E1 §6.3 S5; E2 §6.3 S8; E3 §12 deadlines; E4 §13 item 6; E5 P2_CONTRASTS_v1 registered in the same row).
  The pre-§10 candidate (3958e365…) is kept as contracts/_drafts_PREG2_candidate_3958e365.md.
- Written 2026-10-03 by the PH2-DESIGN worker (Opus; documents only; no server access, no run, no TEST file read).
- Label legend: **MEASURED**, **REPORTED** (someone else's statement, not re-checked), **PLAN**, **INFERENCE**,
  **TO-VERIFY** (Ops checks before the step it gates), **PENDING**.
- State at writing (DL 402–407): ALL_METHODS_LOCK registered (04a3cd19…). TEST manifests built (COUNTS_ONLY; register
  entries 5–6). The TEST look is ON HOLD. No model has been scored on TEST and no TEST value exists. VALIDATION has been
  read, so every new VALIDATION number is POST_VALIDATION_EXPLORATORY.

## Changelog: Fable §9 items → where applied
| §9 item | content | applied in |
|---|---|---|
| 9.1 O1 | O1 = **T0a**: one TEST sitting for Phase 1 + Phase 2 | §6.3, §13 |
| 9.1 O9 | YES: a second registered study may score on the RETRO_TEST manifests built under ALL_METHODS_LOCK 04a3cd19, under S1–S7 | §6.3 |
| 9.1 T0b | **refused** as a confirmation route; removed from every option table | §6.3 (stated as refused), §13 |
| 9.1 TEST2 | an optional pre-registered **replication** (owner O2; Fable approves the manifest spec), never the Q1 confirmation | §6.2, §6.4, §9.3 |
| 9.1 S1 | Phase-1 lock and H plan run exactly as locked; Phase 2 adds P2_METHODS_LOCK (contents listed) | §6.3 S1, §13 |
| 9.1 S2 | P2_METHODS_LOCK + P2_SCRIPT_FREEZE registered and on the server before the first TEST reeval of ANY arm; two LOCKED_EVALUATION entries | §6.3 S2, §13 |
| 9.1 S3 | order inside the sitting; nothing under $R/TEST, out_test_*, p2_out_test_* opened before both RETENTION_RESULT files exist | §6.3 S3 |
| 9.1 S4 | an arm without a sealed checkpoint is declared not run; **no second RETRO_TEST sitting, ever** | §6.3 S4, §9.5 |
| 9.1 S5 | PRE / FA_WARM_FROZEN TEST rows produced once and hashed into both blocks | §6.3 S5, §9.1 |
| 9.1 S6 | labels CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) with D-a verbatim; "_T0" suffix dropped; TEST2 labels; FB label kept | §6.4, §11 |
| 9.1 S7 | GPU precedence: never share with H; Phase 2 has priority over the ML-1M GPU move | §9.5 |
| 9.2 grids | WA / WA-ext / incumbent / anchor / mark-0 rule / WB / WC accepted as written | §5.2–§5.4 (unchanged) |
| 9.2 E-O4a | CALIB tie band **0.001**; TEST equivalence margin stays **±0.0005** | §5.6, §9.4 |
| 9.2 E-O4b | futility threshold **0.001**; skips WB / WC only | §5.8 |
| 9.2 edge | edge rule accepted | §5.7 (unchanged) |
| 9.2 E-O4c | cap 250 GPU-h; drop order by tier: tertiary → exploration → C2 last; dropped = declared not run | §5.9 |
| 9.2 E-O4d (O7) | DP arms keep T = 384 and the registered z; W*'s server / client recipe; passes 2 / batch 16 for FA1024 and DP; disclosure | §5.5, §11 D-i |
| 9.3 E-S1 | power disclosure **D-h** and the §9.4 reading grid registered as the reading | §9.4, §9.6, §11 D-h |
| 9.3 E-S2 | futility headroom caveat (CALIB 32,768 vs MAIN 131,072 users) disclosed beside FUTILITY_SKIP_WB_WC | §5.8, §11 D-j |
| 9.3 E-S3 | D1-ext bit-identity test (θ, ledger, checkpoint digests on / off); DP path logs post-noise only; E1 PRE-null check before any ε_LB | §3.3, §7.1 |
| 9.3 E-S4 | Part A (critical path) / Part B (parallel; declared not run if unfinished; never delays the sitting) | §2.1, §9.5 |
| 9.3 O3 / O5 / O6 / O8 | C2 recommended, conditional on owner O3; O5 = NO; O6 = NO; O8 owner's devices, thresholds DESCRIPTIVE | §4.3, §8, §9.5, §10, §12 |
| 9.4 | reviewer assignment; Fable's five touch points | §13 |
| consequential (not in §9; Fable to confirm) | the NO_MEASURABLE_CALIB_GAIN flag threshold set to 0.001, so it matches the new CALIB tie band | §5.6 |
| consequential (not in §9; Fable to confirm) | Q9–Q14 labelled TERTIARY_P2 (S6 names Q1…Q8 only) | §6.4, §9.1 |

---

## 0. Firewall: Phase 2 cannot contaminate Phase 1

F1. Phase-1 confirmatory claims stay exactly as registered in company/gates/ALL_METHODS_LOCK.json (sha 04a3cd19…, DL
    402). K20b / K21 are reported as "under the registered recipe". No Phase-1 arm, contrast, rule, seed, lock, freeze,
    H plan or report changes. ALL_METHODS_LOCK `post_validation_rule` still applies **inside Phase 1**. The lock's
    "no change" clause governs Phase-1 claims, not the TEST data (Fable §9.1).
F2. Phase 2 is a **separate registered study** (PREG2). It has its own arm prefix `P2_`, its own contrast ids `Q1…`, its
    own registry (company/gates/P2_CONTRASTS_v1.json), its own lock (company/gates/P2_METHODS_LOCK.json), its own
    analysis freeze (company/gates/P2_SCRIPT_FREEZE.json) and its own bootstrap seed (20261004). It never writes into
    Phase-1 files.
F3. Phase-2 selection uses **CALIB only** (CALIB_TRAIN_5_100_32768 → CALIB_VALIDATION_5_100_32768). It never uses MAIN
    VALIDATION, RETRO_TEST or TEST2 for any choice. Disclosed limit (D-b): CALIB users are a subset of MAIN (ranks
    [0, 32,768)) and the CALIB_VALIDATION rows lie in the VALIDATION window that has been read. Phase 1 selected on CALIB
    under the same condition. The selection tool refuses MAIN / TEST manifests (test).
F4. Phase-1 RETRO_TEST is never used for Phase-2 selection. Phase-2 arms touch RETRO_TEST only in the single T0a sitting
    (§6.3), after P2_METHODS_LOCK and P2_SCRIPT_FREEZE are registered.
F5. Phase-2 runs read Phase-1 checkpoints read-only (θ_pre cf08588c…, sealed FL / central checkpoints). They never
    re-run, re-select or re-seal a Phase-1 arm.
F6. Phase-2 code keeps every Phase-1 code path byte-identical with all new options off. Tests compare arm-spec digests,
    RunConfig digests and short trajectories against the Phase-1 code. Frozen Phase-1 tools (retention_report sha
    9a3b2c2a, ana_block_index, reeval, v4eval) are imported or wrapped, never edited.

---

## 1. Why Phase 2 (MEASURED facts, with sources)

| # | fact | source |
|---|---|---|
| M1 | Warm FL arms land at about PRE on MAIN VALIDATION (PRACTICAL_BEST MRR@20): FA_WARM_FROZEN 0.172465 / 0.172461, FA_WARM 0.171550 / 0.171669, FA_1024_WARM_FROZEN 0.172743 / 0.172713, DP1_WARM_FROZEN 0.172734; PRE 0.172896 | REPORT §2 |
| M2 | K21 = FA_WARM_FROZEN − PRE = −0.000433 [−0.000758, −0.000101] (primary; BELOW). D_S7_PRE = FA_WARM − PRE = −0.001287 | REPORT §1, §1e |
| M3 | K12w R(DP8_FROZEN / FA_1024_WARM_FROZEN) = 99.96 [99.93, 100.00] (CB); K13w 99.88: DP noise barely changes the result | REPORT §1 |
| M4 | FA_WARM_FROZEN marks are flat from mark 0.5 (0.171976) to 6.0 (0.172499); "still improving" = False | REPORT §5c, §2 |
| M5 | The warm screen was 2×2 on top of the cold-start S4 point (FedAdam η 0.01, τ 1e-3, client SGD-m LR 0.1, 1 pass, FL_CONST): S8 local factor {1, 4} (client LR {0.1, 0.4}, i.e. only upward) × server factor {1, 0.1}. CALIB: f1/sf1 0.168461 (PB mark 0.5), **f1/sf0.1 0.170709** (PB mark 6.0), f4/sf0.1 0.170351, f4/sf1 0.165227 (PB mark 0.5). Winner = the smallest server LR 0.001 = a grid edge | FL_WARM_SELECTION_S8.json; DL 292 |
| M6 | D1 (run by Ops, result in DL 407): relative distance ‖θ − θ_pre‖ / ‖θ_pre‖ over all 50 tensors: S8 best 0.53 % / endpoint 0.67 %; S7 0.09 % / 0.26 %; DP8_FROZEN 0.013 % / 0.077 %; FA_1024_WARM_FROZEN 0.004 % / 0.030 %. Change concentrates in small gate / bias / LayerNorm tensors (up to 20–45 %); large attention / FFN matrices barely move; S8 item tables unchanged | DL 407 |
| M7 | S8 freezes item_embed.weight, output_embed, output_bias (20.44 M of 20.71 M parameters); 263,822 parameters are trained and uploaded | REPORT §2; launch_fl FROZEN_ITEM_KEYS |
| M8 | C_FULL (central, PRE + MAIN users) 0.194192 / 0.194716 vs PRE 0.172896. Confounds: C_FULL is SASREC_D256_B3 with recipe 02B (c_full.py); PRE is SASREC_D64_B2 with the SMALL recipe reused at 9× data (REPORT §1b note 3) | REPORT §2; code/v4int/c_full.py; pretrain_small.py |
| M9 | K9c FT_PRE − PRE = +0.003976 [0.003329, 0.004640]: per-user local fine-tuning of θ_pre helps | REPORT §1 |
| M10 | Full-model warm FL with fresh client AdamW destroys θ_pre's tables (0.158 at mark 0.5 → 0.03 at 6.0); frozen tables protect it | DL 267 (DESCRIPTIVE preview) |

INFERENCE (mechanism, to be tested, not claimed): FedAdam normalises each coordinate, so its step is about η_s whatever
the signal. At η_s = 1e-3, small tensors move a lot relative to their size and large matrices hardly move (M6). The
client recipe (SGD-m 0.1, tuned for a random start) is too aggressive for fine-tuning. The client deltas are mostly
noise, and the server step either drifts (η_s = 1e-2) or is too small to matter (η_s = 1e-3). The search went the wrong
way on the client axis and stopped at the edge of the server axis. So Phase 1 does not show that FL cannot add value over
θ_pre. It shows that this recipe did not.

INFERENCE (headroom bound): PRE gained +0.0186 MRR@20 over C_SMALL with about 9.3× more users. Adding MAIN's 131,072
users (+10.8 %) under log-linear scaling suggests about +0.0009 at D64 capacity. The same-user, same-period effect
behind K9c could add more. A perfect FL recipe may still gain only about 0.001. This is why C1 / C2 (§4) are required
and why D-h (§11) is registered.

---

## 2. Questions and the Edge-deployment mapping

The question, in its deployment form: **once a pretrained model is shipped to the device, does federated training on the
users' private data add recommendation quality, and at what communication / privacy / device cost?**

| system | Phase-2 arm(s) | role |
|---|---|---|
| cloud-only, shipped pretrained model | PRE (θ_pre, Phase-1 arm, read-only) | anchor |
| cloud with access to the private data (not deployable; upper reference) | P2_C1_FT_CENTRAL_FULL / _FROZEN; P2_C2_CFULL_SMALL (conditional on O3) | headroom |
| federated, weights only | P2_WSTAR, P2_WSTAR_FA1024 / _DP8 / _DP1 (Part A); P2_WSTAR_DPFTRL8, P2_WSTAR_SECAGG (Part B) | the system under test |
| small local model / local adaptation | FT_PRE (Phase-1 arm), FT_P2_WSTAR (the registered FT_C procedure on top of W*) | device stack |

### 2.1 Scope split (Fable E-S4)
- **Part A, the critical path to the sitting:** C0, C1, C2 (if O3 = yes), WA–WC, W* ×2, FT_P2_WSTAR ×2, the DP chain
  (S_CAL, FA1024 ×2, DP8 ×2, DP1), and contrasts Q1–Q11. Work packets P2-00…P2-07, P2-08a, P2-12, P2-13.
- **Part B, in parallel:** SecAgg (E4 / Q13), DP-FTRL (Q12), LDP, privacy audits E1–E3, devices (§8). Work packets
  P2-08b, P2-09, P2-10, P2-14. Part B **never delays the sitting**. A Part-B arm unfinished at lock time is declared not
  run in P2_METHODS_LOCK. A Part-B analysis that needs no TEST row (audits, devices, SecAgg cost) is reported whenever it
  finishes, labelled DESCRIPTIVE.

---

## 3. Diagnostics D1 (value-free with respect to evaluation)

### 3.1 D1 — run by Ops, result in DL 407
D1 is not repeated. Status: EXPLORATORY (checkpoints only, no evaluation, run after the VALIDATION read). Phase 2 cites
it as motivation only.

### 3.2 No-op definition (registered for Phase-2 runs; applied to D1 only as a post-hoc reading)
A warm-FL run is a **NO_OP** iff both hold at its PRACTICAL_BEST checkpoint and at endpoint_6.0:
- (a) parameter criterion: on the **trained subset** T (the uploaded keys = the manifest's shared keys),
  ‖θ_T − θ_pre,T‖ / ‖θ_pre,T‖ < 1 %, AND no tensor group with ≥ 4,096 elements (qkv, out, ff1, ff2 of each block;
  side_proj; user_mlp / user_proj) moved by ≥ 1 % relative;
- (b) functional criterion (value-free: inputs only, no labels): on a fixed probe set of 4,096 CALIB_TRAIN query rows
  (seeded selection, seed 20261004), the mean top-20 Jaccard overlap between θ and θ_pre is ≥ 0.95, and the mean
  |Δscore| of the top-1 item is below the median top-1 / top-2 score gap of θ_pre on the same probe.

A run that fails (a) or (b) is **MOVED**. A MOVED run whose CALIB quality is ≤ PRE_CALIB is **DRIFT**. D1's "over all 50
tensors" figure is dominated by the 20.44 M frozen item-table elements, so (a) is defined on T. INFERENCE: D1's
per-tensor note (gates up to 20–45 %) would make S8 MOVED under (a). (b) is not measured.

### 3.3 D1-ext — instrumentation of every Phase-2 FL run
Each Phase-2 FL run writes UPDATE_NORM_LOG.jsonl, one line per round, computed from values the round already has (no
extra forward pass, no RNG use): round; ‖Δ_agg‖₂ per tensor group (aggregate − θ_r); ‖θ_{r+1} − θ_r‖₂ (server step) per
group; ‖θ_{r+1} − θ_pre‖₂ on T; cos(Δ_agg,r, Δ_agg,r−1); the number of clients aggregated.
- **Bit-identity requirement (Fable E-S3):** a registered test proves that the θ trajectory, the comm ledger and every
  checkpoint digest are bit-identical with the observer on and off.
- **DP runs (z > 0): post-noise, server-side quantities only** (θ is the released model). No pre-noise aggregate norm and
  no per-client value of any kind is logged (06L_E E1). A test asserts this.
- Reading rule: DESCRIPTIVE. It explains the selection outcome. It never selects.

---

## 4. Controls (separate "MAIN adds no information" from "the FL recipe failed")

### 4.1 C0 — the PRE anchor
θ_pre itself: server LR 0 gives exactly θ_pre (fl.server.ServerOptConfig accepts lr = 0). PRE_CALIB = θ_pre evaluated on
CALIB_VALIDATION through the frozen T3 path (a selection value). PRE's MAIN / TEST values are Phase-1 values. Code test:
an FL run with η_s = 0 leaves θ bit-identical after 3 rounds.

### 4.2 C1 — central fine-tune of θ_pre on MAIN (required; Part A)
- Arms: **P2_C1_FT_CENTRAL_FULL** (every parameter trainable) and **P2_C1_FT_CENTRAL_FROZEN** (item tables frozen = the
  central analogue of S8 / W*'s trained subset).
- Start θ_pre (cf08588c…, state digest ee33345b…). Data: MAIN_TRAIN_5_100_131072 (2,587,776 decisions). Objective: the
  locked SMALL objective. Fresh AdamW. Schedule shape: the SMALL recipe shape. **Same budget axis as the FL arms:** 6.0
  EFE on MAIN decisions, 12 half-EFE marks, PRACTICAL_BEST = the strictly-best mark among marks 0.5 … 6.0 (θ_pre at mark
  0 is **not** a candidate), CONTROLLED_BUDGET = 6.0. No continuation.
- LR screen on CALIB (seed 2026): peak LR ∈ {1e-5, 3e-5, 1e-4, 3e-4} × {FULL, FROZEN} = 8 runs. Selection §5.6 within each
  variant. Edge rule §5.7 (extension values 3e-6 / 1e-3).
- MAIN: each variant × seeds {2026, 2027}.
- Interpretation: C1 − PRE measures how much MAIN data can add to θ_pre when privacy is ignored.

### 4.3 C2 — matched-size central reference with MAIN data (**conditional on owner decision O3**; Fable recommends YES)
- If O3 = yes: **P2_C2_CFULL_SMALL** = SASREC_D64_B2, registered θ0 of seed 2026, the SMALL recipe exactly as PRE, on
  FULL_TRAIN_5_100_TIEBREAK (PRE users + MAIN users), 6.0 EFE, with PRE's automatic C3 continuation. Seed 2026. Q8 is
  in the secondary family.
- If O3 = no: C2 and Q8 are removed from P2_CONTRASTS_v1 and P2_METHODS_LOCK **before the lock**, and the Holm family
  becomes Q2–Q7.
- C2 − PRE is the value of the MAIN users' data at matched size and recipe. It removes the size / recipe confounds of
  C_FULL vs PRE (M8). Phase-1 K20b keeps C_FULL.

---

## 5. Warm-FL recipe for fine-tuning (W-grid), selection, ties, edges

### 5.1 Fixed parts (all W runs)
θ_pre start (as S7 / S8: launch_fl --init-from, identity checks unchanged); SASREC_D64_B2; FA method; 64 clients per
round, seeded sweep, 10 % seeded drop-out; local passes 1, batch 16 (the S4 point); FL_CONST LR shape; budget 6.0 EFE
(CALIB 3,414 rounds, MAIN 13,654 rounds at passes 1); 12 marks; PRACTICAL_BEST over marks 0.5 … 6.0 (**mark 0 = θ_pre is
never a candidate**); CALIB screens seed 2026; client clip 1.0; fresh client optimizer per visit.

### 5.2 Stage WA — server optimiser × server LR × client LR (item tables frozen = the deployed S8 structure)
Client: SGD-m 0.9, client LR c ∈ {0.003, 0.01, 0.03}.

| server point | definition (fl.server) | values |
|---|---|---|
| FedAvg | θ + η_s · (aggregate − θ); η_s < 1 = FedAvgM β 0 at LR η_s (launch_fl.warm_server_opt path) | η_s ∈ {0.3, 1.0} |
| FedAvgM | momentum β 0.9 | η_s ∈ {0.03, 0.1} |
| FedAdam | β1 0.9, β2 0.99, τ 1e-3, v0 τ² | η_s ∈ {1e-4, 3e-4} |

WA core = 6 server points × 3 client LRs = **18 runs**. WA-ext: FedAdam η_s ∈ {1e-5, 1e-4} at c 0.1 = **2 runs**.
Incumbent: S8's f1/sf0.1 (FedAdam 1e-3, c 0.1) = 0.170709. It is reused **only if** the Phase-2 code reproduces run
S8Tc's CALIB trajectory bit-identically on the first 64 rounds (test P2-T3). Otherwise it is re-run (1 run). Anchor
η_s = 0: PRE_CALIB (C0). WA total: 20 new runs (+1).

### 5.3 Stage WB — what is trained and sent (at the WA winner A*)
| id | item tables | upload | runs |
|---|---|---|---|
| WB1 | unfrozen; client LR on the three item tensors × 0.1 (param-group multiplier) | dense FP32 | 1 |
| WB2 | unfrozen; × 0.01 | dense FP32 | 1 |
| WB3 | unfrozen; × 0.1 | **delta-compressed**: item_embed rows the client touched (history and target tokens); output_embed / output_bias top-k rows by ‖row delta‖₂, k = 3,170 (2 % of K); non-item keys dense; rows not sent = zero delta; no error feedback | 1 |

INFERENCE: WB3 upload ≈ 2.7 MB vs 82.8 MB dense full-model and 1.05 MB for S8. Bytes come from the fl.comm ledger.

### 5.4 Stage WC — parameter-efficient variants (at A*'s recipe; base frozen including item tables)
| id | variant | runs |
|---|---|---|
| WC1 / WC2 | LoRA r ∈ {4, 16} on qkv, out, ff1, ff2 of both CausalBlocks, α = r, A seeded Kaiming, B = 0; client LR = c* × 10 | 2 |
| WC3 / WC4 | the same at client LR = c* × 1 | 2 |
| WC5 | personalised head: A* plus the registered PF personal vector p_u (fl/personal.py) | 1 |

The trained subset of S8 is only 263,822 parameters, so LoRA's value is regularisation, not bytes. LoRA is merged into the
base weights for scoring / export (same architecture).

### 5.5 Stage WD — DP variants of W* (MAIN only; not selected; Part A) — Fable E-O4d (O7)
- P2_WSTAR_FA1024 (z = 0), P2_WSTAR_DP8 (ε 8), P2_WSTAR_DP1 (ε 1): DP-FedAvg with the **registered** machinery: Poisson,
  m = 1,024, **T = 384**, δ = 5e-6, RDP accountant, **the registered z** (0.5642 / 1.3094 at N = 131,072). No new
  accountant. This keeps them comparable with K12 / K12w.
- W*'s server optimiser, server LR and client LR. **Local passes 2 / batch 16 for BOTH the FA1024 and the DP arms** (a
  matched pair for Q6; the DL 219 rule).
- Clip S from a fresh S_CAL of this configuration (20 rounds; S_RECORD_P2.json with the checks of
  launch_fl.check_s_content).
- Disclosure D-i: a small-step recipe over 384 rounds may move θ little. That is the registered mechanism's answer, not a
  defect to tune away.
- P2_WSTAR_DPFTRL8 (Part B): §7.5.

### 5.6 Selection rule (CALIB only) — Fable E-O4a
- Metric: CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST.
- Candidates: every finite WA, WA-ext, WB, WC run (non-finite runs excluded and recorded).
- Order: A* = argmax over WA ∪ WA-ext ∪ {incumbent}. W* = argmax over {A*} ∪ WB ∪ WC.
- **CALIB tie band 0.001** (INFERENCE ≈ 3 paired-bootstrap SE at about 10,070 CALIB users; Phase 1's 0.0005 is below CALIB
  resolution). Inside the band, prefer: (1) fewer upload bytes per visit, (2) fewer trained parameters, (3) the smaller
  effective server step (FedAdam η_s; FedAvgM η_s / (1 − β); FedAvg η_s), (4) the smaller client LR, (5) the lexically
  first label.
- Anchor flag: CALIB_GAIN = W*_CALIB − PRE_CALIB is printed beside every Q result. NO_MEASURABLE_CALIB_GAIN iff
  CALIB_GAIN < 0.001 (= the CALIB tie band; consequential edit). **W* is still carried to MAIN** (a confirmed null is a
  result).
- The same tie band and preference order (bytes do not differ; then the smaller LR) apply to the C1 screen.
- The **TEST equivalence margin is a different role and stays ±0.0005** (§9.4).
- Output: write-once contracts/fl/P2_WARM_SELECTION_A.json, P2_WARM_SELECTION_FINAL.json, P2_C1_SELECTION.json
  (self-sha256, every value, mark, bytes, D1-ext summary). The selection tool's sha is in P2_METHODS_LOCK.

### 5.7 Grid-edge rule (one extension only)
If the winner sits on the edge of a numeric axis, that axis is extended **once**, by one step outward, at the winner's
other coordinates: client LR 0.003 → 0.001, 0.03 → 0.1; FedAvg η_s 0.3 → 0.1, 1.0 → 2.0; FedAvgM 0.03 → 0.01, 0.1 → 0.3;
FedAdam 1e-4 → 3e-5 (at c*), 3e-4 → 1e-3 (at c*); WA-ext 1e-5 → 3e-6 (at c 0.1); WB item-LR multiplier 0.01 → 0.001;
LoRA rank 16 → 64, 4 → 1; C1 peak LR 1e-5 → 3e-6, 3e-4 → 1e-3. At most 3 extension runs per stage. The rule is
re-applied over the union. If the winner is again on the new edge: EDGE_UNRESOLVED, disclosed beside every Q result. No
second extension.

### 5.8 Futility — Fable E-O4b, E-S2
If C1_CALIB (best of the C1 screen) − PRE_CALIB < **0.001** **and** the best WA value − PRE_CALIB < **0.001**, then WB
and WC are skipped. A* = W* goes to MAIN. Recorded as FUTILITY_SKIP_WB_WC. The primary path is unaffected. Caveat
printed beside the flag (D-j): C1's CALIB screen measures headroom with 32,768 users, while MAIN has 131,072, so the rule
may under-read the headroom available on MAIN.

### 5.9 Budget cap and drop order — Fable E-O4c
Cap: **250 GPU-h** for Phase 2. If the projected total exceeds it, drop by tier, in this order:
1. tertiary-only arms: P2_WSTAR_DP1 → P2_WSTAR_SECAGG on MAIN → P2_WSTAR_DPFTRL8 s2027;
2. exploration cells: WC3 / WC4 → WB2 → WC1 / WC2 → WB3;
3. C2 last (secondary Q8).
Everything dropped is **declared not run** in P2_METHODS_LOCK. The primary path (WA, C1, W* ×2, FT_P2_WSTAR ×2,
FA1024 ×2, DP8 ×2) is never dropped.

---

## 6. The confirmation split

### 6.1 Finding: REES46 data after 2019-11-30 (MEASURED locally; TO-VERIFY on the server)
- rees46_nextitem_v1/VERSION.json lists source months **2019-10 and 2019-11 only**. Local raw: data/raw/2019-Oct.csv and
  2019-Nov.csv only. Splits: TRAIN [10-01, 11-16), VALIDATION [11-16, 11-23), RETRO_TEST [11-23, 11-30), LABEL_GRACE
  [11-30, 12-01) with zero decisions. Nothing on or after 2019-12-01 was ever loaded.
- REPORTED (unverified): REES46 publishes Oct 2019 – Apr 2020 for this store (MULTI_EXPERT_REVIEW 13, E10 and §6).

### 6.2 TEST2 — an optional pre-registered replication (owner O2; Fable approves the manifest spec)
TEST2 is **never the Q1 confirmation**. It is a replication of Q1 (and of Q2–Q8 as DESCRIPTIVE replications) on a
never-seen period.
- Window: decisions with decision_ts ∈ [2019-12-01T00:00Z, 2019-12-08T00:00Z); LABEL_GRACE [12-08, 12-09) for targets
  only.
- Users: MAIN users with ≥ 1 rankable decision in the window (a different estimand: users active in December; the
  post-Black-Friday / Cyber-Monday shift; wider CI). Optional panel: UNEW_HOLDOUT users (Q14).
- History: causal USER_HISTORY_L50 over every event before the decision. Transforms: TRANSFORM_STATE_CLEAN_V2 unchanged.
  Catalogue K = 158,486 unchanged; OOV targets get credit 0 and stay in the E2E denominator. Frozen checkpoints only.
- Data version: a new immutable version (proposal rees46_nextitem_v2_dec2019) built with the v1 pipeline (code_hash
  02fbb071…); v1 untouched.
- Scoring: in the same sitting if the TEST2 manifests and gate are ready by then, otherwise later under its own register
  (tools/test2_exposure_register.py). Only arms sealed in P2_METHODS_LOCK (and Phase-1 locked arms as
  FRESH_PERIOD_AUDIT) are scored.
- TO-VERIFY before the build (COUNTS_ONLY, no model scored): (1) the December 2019 file of the **same multi-category
  store**: URL, name, bytes, sha256, rows; (2) schema identical to 2019-Nov.csv (event_time, event_type, product_id,
  category_id, category_code, brand, price, user_id, user_session; UTC); (3) event-type set = view / cart / purchase;
  (4) the share of December events with product_id in catalogue.parquet, and their category / brand agreement with the
  v1 vocabularies; (5) first / last event_time, no overlap or duplicates at the month boundary; (6) counts of MAIN and
  UNEW_HOLDOUT users with ≥ 1 decision in [12-01, 12-08), rankable decisions, OOV share; (7) the owner approves the new
  data version; Fable approves the manifest spec; (8) the TEST2 register records the build as MANIFEST_BUILD /
  COUNTS_ONLY.

### 6.3 T0a — the one TEST sitting (O1 = T0a; O9 = YES under S1–S7)
T0b (a procedure lock, Phase-1 look first) is **refused** as a confirmation route: PRE's and FA_WARM_FROZEN's TEST values
would be known while Phase-2 selection, edge rules, futility and bug fixes are still being decided. TEST2-only is
**refused** as the primary route. The guarantee T0a must give: **no human sees any TEST value before every arm of both
studies is locked and scored.**

Binding conditions of the sitting:
- **S1.** ALL_METHODS_LOCK and its H plan (109 reevals, 14 device, 39 panels, 6 baselines, onnx) execute exactly as
  locked. Phase 2 adds its own plan under company/gates/P2_METHODS_LOCK.json: arms → checkpoint shas, rows shas,
  P2_CONTRASTS_v1 sha, the code-tree digest, the selection-file shas (P2_WARM_SELECTION_A / _FINAL, P2_C1_SELECTION,
  S_RECORD_P2), bootstrap seed 20261004, and the declared-not-run list.
- **S2.** P2_METHODS_LOCK and P2_SCRIPT_FREEZE (the shas of tools/p2_report.py and tools/p2_block_index.py) are registered
  and on the server **before the first TEST reeval of ANY arm, Phase 1 included**. TEST_EXPOSURE_REGISTER entries (tool
  unchanged): one LOCKED_EVALUATION for the Phase-1 lock and a second LOCKED_EVALUATION whose all_methods_lock_sha256
  binds the P2 lock's sha.
- **S3.** Order inside the sitting: all Phase-1 scoring → all Phase-2 scoring → Phase-1 `analysis` → Phase-2 analysis.
  **Nothing under $R/TEST, out_test_* or p2_out_test_* is fetched or opened until both RETENTION_RESULT files exist.**
  Ops writes the DL row first and fetches second.
- **S4.** A Phase-2 arm without a sealed checkpoint at lock time is declared not run in the P2 lock (06D §8 wording).
  **There is no second sitting on RETRO_TEST, ever.** A Phase-2 arm missing from the sitting is scored on TEST2 only, or
  not at all.
- **S5.** The Phase-1 arms whose TEST rows Phase 2 reuses are **PRE, FA_WARM_FROZEN, FT_PRE (Q5) and C_FULL (Q9)**:
  produced once by the Phase-1 plan and hash-bound into both analysis blocks. No arm is scored twice on the same role.
- **S6.** Labels: §6.4.
- **S7.** GPU precedence: §9.5.
- **S8 (sealing of Phase-2 MAIN values; Fable §10 E2).** No Phase-2 MAIN VALIDATION value (RESULT_ROWS, metrics.jsonl,
  best_metric, REPORT lines) is opened by a human before both sitting analyses are written. The PRACTICAL_BEST checkpoint
  is the run's own mechanical rule. The selection tools read CALIB rows only (a test pins it). After the sitting these
  values are reported under the label POST_VALIDATION_EXPLORATORY.

Owner consequence, stated plainly: the Phase-1 TEST result waits for Phase 2 (INFERENCE ≈ 10-17…10-21: code 3–4 days +
runs 8–11 days). The owner said he prefers that (Fable §9.1).

### 6.4 Labels (Fable S6)
| claim | label |
|---|---|
| Phase-1 K1…K21 (all registered K) | CONFIRMATORY (PREG1), unchanged |
| Phase-2 Q1…Q8 on RETRO_TEST (the sitting) | **CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP)**, always printed with disclosure D-a verbatim |
| Phase-2 Q9…Q14 on RETRO_TEST | TERTIARY_P2 (unadjusted; DESCRIPTIVE Holm band) (consequential) |
| Phase-2 arms on MAIN VALIDATION | POST_VALIDATION_EXPLORATORY |
| Q1 on TEST2 | REPLICATED_FRESH (same sign, 95 % CI excludes 0) / NOT_REPLICATED / TEST2_NOT_AVAILABLE |
| Phase-1 locked arms on TEST2 | FRESH_PERIOD_AUDIT (DESCRIPTIVE) |
| Fallback FB (§6.5) | HOLDOUT_USERS_SAME_PERIOD, never a Q1 confirmation |
| audits E1–E5, devices §8, D1-ext | DESCRIPTIVE (tertiary) |

### 6.5 Fallback FB (limits as drafted)
No band of the 5–100 pool is unused (MAIN [0, 131,072); PRE [131,072, 1,348,682); U_NEW_VAL [1,348,682, 1,381,450);
U_NEW_TEST [1,381,450, 1,414,218) = UNEW_HOLDOUT; FS [1,414,218, 1,446,986)). FB = the RETRO_TEST window of the U_NEW_VAL
band. It is a different estimand (non-participants), in the same week, and U_NEW_VAL was a selection panel. Label:
HOLDOUT_USERS_SAME_PERIOD. TO-VERIFY: U_NEW_VAL's RETRO_TEST rows are in no scored v1 / V4 TEST artefact. Under T0a FB
is used only if the owner asks for the new-user estimand. Its manifest must then be in P2_METHODS_LOCK and scored in the
sitting (S4).

---

## 7. Privacy: audit, secure aggregation, untrusted-server DP (Part B except where stated)

All audits are separate audit runs or read-only analyses. None selects an arm. None reads VALIDATION / TEST labels.
Members / non-members use TRAIN-window rows only. UNEW_HOLDOUT TRAIN-window rows as non-members touch no TEST row.

### 7.1 E1 — user-level membership inference on final models
- Targets: PRE (null control), FA_WARM_FROZEN s2026 (read-only), P2_WSTAR s2026, P2_WSTAR_DP8 s2026, P2_WSTAR_DP1,
  P2_C1_FT_CENTRAL_FULL s2026.
- Members: 10,000 MAIN users (seeded, seed 20261004). Non-members: 10,000 UNEW_HOLDOUT users. Each user's TRAIN rows.
- Score: s(u) = mean CE_θ(u) − mean CE_θpre(u) (reference-calibrated loss); the raw loss is reported beside it.
- Protocol: users split 50/50; the threshold is set on half A at FPR 1 % and evaluated on half B.
- Metrics: ROC AUC with a user-bootstrap 95 % CI; TPR at FPR ∈ {0.1 %, 1 %}; ε_LB = ln(TPR_lo / FPR_hi), Clopper–Pearson
  95 %.
- Criteria: (a) **validity first (Fable E-S3):** PRE's AUC CI must include 0.5 **before any ε_LB is reported**; otherwise
  the audit is invalid and stops; (b) DP arms PASS iff ε_LB ≤ the accountant ε (8 / 1); FAIL triggers an investigation;
  (c) non-DP arms: MATERIAL_LEAKAGE iff the AUC CI lower bound > 0.55 or the TPR@1 % lower bound > 3 %.

### 7.2 E2 — canary exposure and one-run audit on FL training
CALIB-scale audit runs (never evaluated): W*'s recipe on CALIB_TRAIN plus 1,000 synthetic canary clients (random
length-20 long-tail sequences, fixed random target, each included with probability ½, seeded). Runs: W*; W* with WB1
tables unfrozen; W*-DP8 (z from fl.accountant for N = 32,768 + canaries, T = 384, m = 1,024). Score:
log p_θ(target | history) − log p_θpre(…). Metrics: exposure against 10,000 reference sequences (non-DP); the one-run
ε_LB (Steinke, Nasr & Jagielski 2023) at 95 %. Criteria: DP PASS iff ε_LB ≤ 8; non-DP DESCRIPTIVE.

### 7.3 E3 — leakage from individual FL updates
CALIB replay (rounds 0–49) of W* and WB1 with an upload-capture hook for 512 seeded clients (audit-only; captured
uploads deleted after the analysis). Attacks: item-set recovery from item_embed deltas (unfrozen); a linear probe for the
client's top main category (frozen / LoRA); both on 64-client SecAgg aggregates. SECAGG_NEEDED iff the advantage over
baseline is > 10 points on single uploads and ≤ 2 points on aggregates.

### 7.4 E4 — secure-aggregation simulation (Q13)
- Protocol: Bonawitz et al. 2017 with the SecAgg+ k-regular option (Bell et al. 2020). Cryptography is simulated. Sizes
  are counted at real values.
- Fixed-point mod 2^32 with a common scale from the clip bound. Dropout p ∈ {0, 0.1, 0.3}, n = 64, threshold ⌈2n/3⌉.
- Metrics: bytes per client per round (overhead ratio), communication rounds, unmasking CPU time, success under dropout.
- Criteria: the decoded sum is bit-identical to the plain fixed-point sum (test); quality TOST on CALIB with the 90 % CI
  inside [99, 101] % (DESCRIPTIVE); optional MAIN s2026 (Q13). Limit: central DP still trusts the server with the noise;
  distributed DP is out of scope.

### 7.5 E5 — DP-FTRL and LDP
- **DP-FTRL** (Kairouz et al. 2021): tree aggregation, restart per epoch, at most one participation per sweep. Clip S from
  S_CAL of W*. z from a tree-aggregation zCDP accountant for ε 8, δ 5e-6, N 131,072 (new code; **Fable concurrence on the
  accountant**). Arm P2_WSTAR_DPFTRL8, MAIN s2026 (s2027 only within the cap). Q12a / Q12b tertiary. Audited by E1 / E2.
- **LDP feasibility:** client-side clip + Gaussian noise, per-upload ε_L ∈ {8, 32}, composition over about 6 uploads
  reported. One CALIB run each. Label LDP_FEASIBILITY (DESCRIPTIVE).

---

## 8. Real-device measurement protocol (Part B; DESCRIPTIVE)
- Models: SASREC_D64_B2 FP32 / INT8 ONNX (opset 17, batch 1) of PRE and P2_WSTAR (LoRA merged if a WC cell wins), plus
  C_SMALL. Export / parity with v4dev.export / v4dev.quantize / v4dev.bench (FP32 top-20 set identity ≥ 0.98 on 50
  reference queries).
- Devices (O8: the owner's devices; TO-VERIFY which he has): D1 Windows 11 PC, Edge (wasm-1T, wasm-MT with COOP / COEP,
  WebGPU); D2 Android phone, Chrome (wasm); D3 the same phone native (onnxruntime-android bench or Termux; TO-VERIFY);
  D4 low-end Android (nice); D5 iPhone Safari (nice); REF server CPU (K16 pass 1 / pass 2; pass 2 PENDING in Phase 1).
- Metrics: latency p50 / p95 / p99 over 1,000 timed batch-1 queries after 20 warm-ups; cold start; peak memory (Android
  PSS; browser measureUserAgentSpecificMemory or "n/a"); model size (disk, gzip, brotli); energy proxy (Android
  batterystats, 30 min run vs 30 min idle, ΔmWh per 1,000 inferences; CPU-seconds per 1,000 inferences); on-device
  training cost as a PC proxy (PC_PROXY) unless ORT training works on D3.
- Runs: 3 sessions per configuration on separate reboots / days; plugged in for latency, on battery for energy; 2 min
  cool-down; battery 80–95 %; temperatures recorded; the median over sessions and the range reported.
- Deployable thresholds (DESCRIPTIVE design targets, accepted by Fable): D2 INT8 p95 ≤ 100 ms; cold start ≤ 3 s; peak
  memory ≤ 300 MB; brotli download ≤ 25 MB.

---

## 9. Hypotheses, contrasts, statistics, stopping, success / failure

### 9.1 Contrasts (paired user-level on RETRO_TEST in the sitting; MAIN users; PRACTICAL_BEST unless stated)
| id | contrast | tier | seeds | part |
|---|---|---|---|---|
| **Q1** | P2_WSTAR − PRE | **primary** | W* {2026, 2027} vs PRE s2026 (one θ_pre) | A |
| Q2 | P2_C1_FT_CENTRAL_FULL − PRE | secondary | {2026, 2027} | A |
| Q3 | P2_WSTAR − P2_C1_FT_CENTRAL_FROZEN | secondary | {2026, 2027} | A |
| Q4 | P2_WSTAR − FA_WARM_FROZEN | secondary | {2026, 2027} same-seed pairs | A |
| Q5 | FT_P2_WSTAR − FT_PRE | secondary | {2026, 2027} | A |
| Q6 | R(P2_WSTAR_DP8 / P2_WSTAR_FA1024) | secondary, CONTROLLED_BUDGET | {2026, 2027} | A |
| Q7 | P2_WSTAR_DP8 − PRE | secondary, CONTROLLED_BUDGET | {2026, 2027} | A |
| Q8 | P2_C2_CFULL_SMALL − PRE | secondary (only if O3 = yes) | {2026} | A |
| Q9 | R(P2_WSTAR / C_FULL) | tertiary | {2026, 2027} vs C_FULL s2026 | A |
| Q10 | Q1 / Q2 (headroom capture ratio) | tertiary, derived, DESCRIPTIVE | — | A |
| Q11 | R(P2_WSTAR_DP1 / P2_WSTAR_FA1024) | tertiary, CONTROLLED_BUDGET | {2026} | A |
| Q12a / b | R(P2_WSTAR_DPFTRL8 / P2_WSTAR), P2_WSTAR_DPFTRL8 − PRE | tertiary | {2026} | B |
| Q13 | R(P2_WSTAR_SECAGG / P2_WSTAR), TOST [99, 101] on the 90 % CI | tertiary | {2026} | B |
| Q14 | P2_WSTAR − PRE on the TEST2 UNEW_HOLDOUT panel (new-user estimand) | tertiary | {2026, 2027} | TEST2 |

S5: PRE and FA_WARM_FROZEN rows come from the Phase-1 scoring (hash-bound). DP arms' PRACTICAL_BEST ratios are
DESCRIPTIVE (best-of-12 selection is outside the accountant). Q6 / Q7 / Q11 read CONTROLLED_BUDGET.

### 9.2 Statistics
Paired user bootstrap, 1,000 resamples, seed **20261004**, one shared resample matrix over the sitting's MAIN users.
Percentile 95 % CIs, ratios bootstrapped directly, seed mean on the same W, per-seed points, seed spread, SEED_SENSITIVE,
harmed-client share with the seed-vs-seed null (as Phase 1, 06L-C C13–C15). E2E client-macro MRR@20 is primary; NDCG@10
/ HR@10 DESCRIPTIVE. Implemented by tools/p2_report.py importing retention_report (unchanged).

### 9.3 Multiplicity
Q1 alone is primary, unadjusted at α = 0.05. Secondary family: Holm step-down at α = 0.05 over {Q2…Q7} (+Q8 if O3 = yes).
Tertiary: unadjusted with a DESCRIPTIVE Holm band. TEST2 replication is a separate family (Q1-rep at α = 0.05 with the
same rule). Never pooled with Phase 1. Dependencies: Q10 = Q1 / Q2; Q3 = Q1 − (C1_FROZEN − PRE).

### 9.4 Registered reading of Q1 (Fable E-S1) and the interpretation grid
- **SUCCESS:** Q1 95 % CI lower bound > 0 **and** both per-seed points > 0.
- **EQUIVALENT:** Q1's 90 % CI lies inside **±0.0005** MRR@20 (TOST; the practical-relevance margin, distinct from the
  CALIB tie band).
- **FAILURE:** Q1's 95 % CI upper bound < 0.
- **INCONCLUSIVE:** anything else.
- DEPLOYMENT-GRADE statement (not a separate test): SUCCESS + Q5 > 0 (Holm-rejected) + Q7 > 0 (Holm-rejected).

| Q2 (central headroom) | Q1 (FL) | reading |
|---|---|---|
| > 0 | > 0 | FL captures part of the available headroom (Q10 = how much) |
| > 0 | ≤ 0 or EQUIVALENT | the FL recipe fails to capture available headroom (an FL problem) |
| EQUIVALENT to 0 | EQUIVALENT to 0 | MAIN data adds no measurable information over θ_pre at D64; FL cannot help here (a data / capacity limit) |
| EQUIVALENT to 0 | > 0 | flagged for review (check for a central-recipe problem) |

No positive result is promised (D-h).

### 9.5 Seeds, budgets, stopping, GPU precedence
- CALIB screens: seed 2026. MAIN: {2026, 2027} for W*, C1 (both variants), FT_P2_WSTAR, FA1024, DP8; {2026} for DP1,
  DPFTRL8, SECAGG, C2. θ_pre is one seed shared by every warm arm. **O5 (a second θ_pre seed) = NO**, disclosed by D-c.
- Each run has a fixed 6.0 EFE budget (DP: T = 384). No early stopping. No continuation for FL or C1 (C2 has PRE's C3
  rule).
- Run-level stopping: non-finite runs are excluded from selection and recorded. Mechanical failures resume under
  identical hashes. A second failure with the same signature → INCOMPLETE (DL row) → declared not run.
- Study-level stopping: futility §5.8; cap and drop order §5.9; Part B never delays the sitting (§2.1). No interim look at
  any confirmation split. No second RETRO_TEST sitting (S4).
- **S7 GPU precedence:** Phase-2 lanes never share a GPU with H steps. Phase 2 has priority over the ML-1M GPU move (Fable
  §8). §8's timing gate must include Phase-2 demand (≈ 190 GPU-h on 2 A40 ≈ 4–6 days, INFERENCE). If Phase 2 takes both
  A40s, ML-1M stays on CPU (§8 option A). One DL row before BM-03 effort is spent.

### 9.6 Power (INFERENCE)
K21's VALIDATION CI half-width 0.00033 → SE ≈ 0.000168 at 40,764 users → MDE (80 % power, α 0.05) ≈ 0.00047. The
plausible ceiling is about +0.001 (§1). The study is powered for roughly the full ceiling, not for half of it (D-h).
RETRO_TEST user counts: TO-VERIFY from register entry 6's COUNTS_ONLY record. TEST2's MDE scales with
√(40,764 / n_TEST2-users).

---

## 10. External validity and novelty (stated limits)
- One main dataset and event type. **O6 (second-dataset replication) = NO** (scope). A DESCRIPTIVE stratum by target event
  type on RETRO_TEST is reported if the evaluation rows carry the event type (TO-VERIFY in DATA_CONTRACT_CLEAN_V2).
- Novelty: the methods are known. The contribution is the evaluation design: a fine-tuning-specific warm-FL recipe for a
  full-catalogue sequential recommender, the headroom decomposition (C1 / C2 vs FL, Q10), an empirical audit of
  item-table leakage with and without SecAgg, and real-device numbers, all pre-registered.

## 11. Disclosures (verbatim)
D-a. **"Phase-2 hypotheses were generated after the Phase-1 VALIDATION read; arms were selected on CALIB; everything was
     locked before any TEST value existed."** (Printed with every CONFIRMATORY_P2 claim.)
D-b. CALIB selection data are a subset of the MAIN users and of the VALIDATION window that was read.
D-c. θ_pre is one seed (no second seed was run). The PRE recipe was not re-tuned at its scale (Phase-1 note 3).
D-d. C_FULL differs from PRE in size and recipe. C2 (if run) is the matched reference.
D-e. Central DP trusts the aggregator. SecAgg is simulated (costs counted, cryptography not executed). DP-FTRL / LDP are
     the untrusted-server variants.
D-f. Device numbers come from the listed devices only. On-device training cost is a PC proxy unless stated otherwise.
D-g. RETRO_TEST was inspected at period level by earlier v1 campaigns (Phase-1 E12(d)).
D-h. **"Phase 2 can confirm a gain ≥ ≈ 0.0005 MRR@20 or equivalence within ±0.0005; a true gain between them is likely
     to read INCONCLUSIVE."**
D-i. The DP arms keep T = 384 and the registered z. A small-step recipe over 384 rounds may move θ little; that is the
     registered mechanism's answer, not a defect to tune away.
D-j. Futility uses CALIB headroom (32,768 users); MAIN has 131,072 users, so FUTILITY_SKIP_WB_WC may under-read the
     headroom available on MAIN.
D-k. The Phase-1 TEST result was held for the Phase-2 runs (owner decision, DL 406). No Phase-1 arm, contrast or rule
     changed.

## 12. Decisions
| id | decision | status |
|---|---|---|
| O1 | confirmation route | **T0a** (Fable §9.1) |
| O9 | second study on the RETRO_TEST manifests | **YES under S1–S7** (Fable §9.1) |
| O4 | grids, ties, edges, futility, cap | **accepted with E-O4a–c** (Fable §9.2) |
| O7 | DP T and z | **T = 384, registered z, passes 2 / batch 16** (Fable E-O4d) |
| O2 | December 2019 data version (TEST2 replication) | PENDING — owner; then Fable for the manifest spec. **Deadline (Fable §10 E3):** TEST2 joins the sitting only if its manifests, gate and register exist before P2_METHODS_LOCK; otherwise it is a later replication |
| O3 | run C2 | PENDING — owner (Fable recommends YES). **Deadline (Fable §10 E3):** decided by a DL row before the first Phase-2 MAIN run |
| O5 | second θ_pre seed | **NO** (Fable §9.3) |
| O6 | second-dataset replication | **NO** (Fable §9.3) |
| O8 | devices; thresholds | PENDING — owner's devices; thresholds DESCRIPTIVE (accepted). **Deadline (Fable §10 E3):** before P2-14 starts |

## 13. Registration and the road to the sitting
1. Fable delta re-review of this file (text only) → edits → this file is written read-only and its sha256 recorded in a DL
   row = REGISTERED. company/gates/P2_CONTRASTS_v1.json goes from draft to final with its self-sha in the same or the
   next DL row (after O3 decides Q8).
2. Code packets (PHASE2_IMPLEMENTATION_PLAN) with the assigned reviewers (Fable §9.4): Opus + Opus review for P2-02,
   P2-04, P2-06, P2-09, P2-10, P2-11, P2-12; Opus + Sonnet review for P2-01, P2-03, P2-05, P2-13; Sonnet for P2-07,
   P2-14 and the lock tool.
3. Runs (Part A critical path; Part B parallel). Every run wave gets a DL row. The thesis reference is updated after
   registration and after each selection.
4. Fable touch points (exactly five): (1) the delta re-review of this file; (2) P2_METHODS_LOCK before the sitting;
   (3) P2_SCRIPT_FREEZE; (4) the DP-FTRL accountant concurrence; (5) the TEST2 manifest spec if O2 = yes.
5. P2_METHODS_LOCK + P2_SCRIPT_FREEZE registered and uploaded → the two LOCKED_EVALUATION register entries → the single
   sitting in the S3 order → both RETENTION_RESULT files → DL row → fetch.
6. **Frozen register tool (Fable §10 E4).** tools/test_exposure_register.py (7d59e828…) is in ANA_SCRIPT_FREEZE_v3 `scripts`
   and is never edited. MEASURED by Ops at registration: its PURPOSES enum contains LOCKED_EVALUATION and it imposes no
   uniqueness rule per purpose (append checks fields, enums, the lock-sha requirement and the hash chain only), so two
   LOCKED_EVALUATION entries with different all_methods_lock_sha256 values are admitted. The Phase-2 entry names the P2
   lock in observations_description.
6. Every deviation is an amendment (PREG2_AMENDMENT_xx.md) with a DL row, written before the affected value exists. No
   amendment is possible after the sitting for any Q scored in it.

# PREG2_FOLLOWUP_DRAFT — Phase-2 pre-registration (DRAFT, NOT REGISTERED)

- Status: **DRAFT for the Fable gate.** Nothing here is registered, locked or authorised to run until a DECISION_LOG row
  records this file's sha256 as REGISTERED (§13). Written 2026-10-03 by the PH2-DESIGN worker (Opus; documents only; no
  server access, no run, no TEST file read).
- Inputs read: results/VALIDATION_v3/REPORT.md §1, §1b, §1e, §2 (warm rows), §5c (FA_WARM_FROZEN marks), §6;
  contracts/fl/FL_WARM_SELECTION_S8.json (local copy), contracts/fl/FL_TUNING_SELECTION.json, DATA_PROTOCOL_LOCK_V2.json,
  manifest_registry/REGISTRY_V2_*_EXT.json, company/gates/ALL_METHODS_LOCK.json, DECISION_LOG rows 267, 292, 400–407,
  rees46_nextitem_v1/VERSION.json, code under code/fl, code/v4int, code/v4dev, tools/.
- Label legend: **MEASURED** (in a file or DL row), **REPORTED** (someone else's statement, not re-checked),
  **PLAN**, **INFERENCE** (my estimate or reasoning), **TO-VERIFY** (Ops must check before registration), **PENDING**.
- Ops update 2026-10-03 15:45Z (DL 406): the TEST look is **ON HOLD**. TEST manifests are built (H build + final, data
  only). No model has been scored on TEST. Diagnostics and new runs on CALIB / VALIDATION are allowed now. VALIDATION has
  been read, so every new VALIDATION number is POST_VALIDATION_EXPLORATORY.

---

## 0. Firewall: Phase 2 cannot contaminate Phase 1

F1. Phase-1 confirmatory claims stay exactly as registered in company/gates/ALL_METHODS_LOCK.json (sha 04a3cd19…, DL
    402). K20b / K21 are reported as "under the registered recipe". No Phase-1 arm, contrast, rule, seed, lock, freeze or
    report changes. ALL_METHODS_LOCK `post_validation_rule` still applies **inside Phase 1**: nothing added after the
    VALIDATION read becomes a Phase-1 claim.
F2. Phase 2 is a **separate registered study** (PREG2). It has its own arm prefix `P2_`, its own contrast ids `Q1…`, its
    own registry (company/gates/P2_CONTRASTS_v1.json), its own lock (company/gates/P2_METHODS_LOCK.json) and its own
    bootstrap seed (20261004). It never writes into Phase-1 files.
F3. Phase-2 selection uses **CALIB only** (CALIB_TRAIN_5_100_32768 → CALIB_VALIDATION_5_100_32768). It never uses MAIN
    VALIDATION, RETRO_TEST or TEST2 for any choice. Disclosed limit: CALIB users are a subset of MAIN (ranks [0, 32,768))
    and the CALIB_VALIDATION rows lie in the VALIDATION window that has been read. Phase 1 already selected on CALIB
    under the same condition.
F4. Phase-1 TEST (RETRO_TEST) is never used for Phase-2 selection. The only way Phase-2 arms may touch RETRO_TEST is
    option T0 (§6.3): scoring of **locked** Phase-2 arms, decided by Fable before any TEST value exists.
F5. Phase-2 runs read Phase-1 checkpoints read-only (θ_pre cf08588c…, sealed FL / central checkpoints). They never
    re-run, re-select or re-seal a Phase-1 arm.
F6. Phase-2 code changes keep every Phase-1 code path byte-identical with all new options off. Tests compare arm-spec
    digests, RunConfig digests and short trajectories against the Phase-1 code (same pattern as FL-10 / FL-12 / FL-15).

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
| M7 | S8 freezes item_embed.weight, output_embed, output_bias (20.44 M of 20.71 M parameters); 263,822 parameters are trained and uploaded | REPORT §2 (params column); launch_fl FROZEN_ITEM_KEYS |
| M8 | C_FULL (central, PRE + MAIN users) 0.194192 / 0.194716 vs PRE 0.172896. Confounds: C_FULL is SASREC_D256_B3 with recipe 02B (c_full.py); PRE is SASREC_D64_B2 with the SMALL recipe reused at 9× data (REPORT §1b note 3) | REPORT §2; code/v4int/c_full.py; pretrain_small.py |
| M9 | K9c FT_PRE − PRE = +0.003976 [0.003329, 0.004640]: per-user local fine-tuning of θ_pre helps | REPORT §1 |
| M10 | Earlier finding: full-model warm FL with fresh client AdamW destroys θ_pre's tables (0.158 at mark 0.5 → 0.03 at 6.0); frozen tables protect it | DL 267 (DESCRIPTIVE preview) |

INFERENCE (mechanism, to be tested, not claimed): FedAdam normalises each coordinate, so its step is about η_s whatever
the signal. At η_s = 1e-3, small tensors move a lot relative to their size and large matrices hardly move (M6). The
client recipe (SGD-m 0.1, tuned for a random start) is far too aggressive for fine-tuning. The resulting client deltas
are mostly noise, and the server step either drifts (η_s = 1e-2: quality falls, PB at mark 0.5) or is too small to
matter (η_s = 1e-3: flat). The search went the wrong way on the client axis and stopped at the edge of the server axis.
So Phase 1 does not show that FL cannot add value over θ_pre. It shows that this recipe did not.

INFERENCE (headroom bound, for power and expectations): PRE gained +0.0186 MRR@20 over C_SMALL with about 9.3× more
users. Adding MAIN's 131,072 users (+10.8 %) under log-linear scaling suggests about +0.0009 at D64 capacity. The
same-user, same-period effect behind K9c could add more. So even a perfect FL recipe may gain only about 0.001. This is
why the controls C1 / C2 (§4) are required to interpret any null.

---

## 2. Questions and the Edge-deployment mapping

The owner's top priority is the realistic Edge system: a cloud-only model vs federated learning (only weights leave the
device) vs a small local model. Phase 2 asks the proposal's question in its deployment form: **once a pretrained model is
shipped to the device, does federated training on the users' private data add recommendation quality, at what
communication / privacy / device cost?**

| system | Phase-2 arm(s) | role |
|---|---|---|
| cloud-only, shipped pretrained model | PRE (θ_pre, Phase-1 arm, read-only) | anchor |
| cloud with access to the private data (not deployable; upper reference) | P2_C1_FT_CENTRAL (θ_pre fine-tuned centrally on MAIN), P2_C2_CFULL_SMALL (optional) | headroom |
| federated, weights only | P2_WSTAR (selected warm-FL recipe), P2_WSTAR_DP8 / DP1, P2_WSTAR_DPFTRL8, P2_WSTAR_SECAGG | the system under test |
| small local model / local adaptation | FT_PRE (Phase-1 arm), FT_P2_WSTAR (registered FT_C procedure on top of W*) | device stack |

---

## 3. Diagnostics D1 (value-free with respect to evaluation)

### 3.1 D1 — already run by Ops (result in DL 407)
D1 is not repeated here (DL 406–407). Its status is EXPLORATORY: checkpoints only, no evaluation, run after the
VALIDATION read. Phase 2 cites it as motivation only.

### 3.2 No-op definition (registered for Phase-2 runs; applied to D1 only as a post-hoc reading)
A warm-FL run is a **NO_OP** iff both hold at its PRACTICAL_BEST checkpoint and at endpoint_6.0:
- (a) parameter criterion: the relative distance on the **trained subset** T (the uploaded keys, i.e. the manifest's
  shared keys), ‖θ_T − θ_pre,T‖ / ‖θ_pre,T‖ < 1 %, AND no tensor group with ≥ 4,096 elements (qkv, out, ff1, ff2 of each
  block; side_proj; user_mlp / user_proj) moved by ≥ 1 % relative;
- (b) functional criterion (value-free: inputs only, no labels): on a fixed probe set of 4,096 CALIB_TRAIN query rows
  (seeded selection, seed 20261004), the mean top-20 Jaccard overlap between θ and θ_pre is ≥ 0.95, and the mean
  |Δscore| of the top-1 item is below the median top-1/top-2 score gap of θ_pre on the same probe.

A run that fails (a) or (b) is **MOVED**. A MOVED run whose CALIB quality is ≤ PRE_CALIB is **DRIFT** (FL moved θ, but
not usefully). D1's "over all 50 tensors" figure is dominated by the 20.44 M frozen item-table elements, so (a) is
defined on T. INFERENCE: D1's per-tensor note (gates up to 20–45 %) would make S8 MOVED under (a). (b) is not measured.

### 3.3 D1-ext — instrumentation of every Phase-2 FL run (new code; not a re-run of D1)
Each Phase-2 FL run writes UPDATE_NORM_LOG.jsonl, one line per round, computed from values the round already has (no
extra forward pass, no RNG use, trajectory unchanged):
round; ‖Δ_agg‖₂ per tensor group (aggregate − θ_r); ‖θ_{r+1} − θ_r‖₂ (server step) per group; ‖θ_{r+1} − θ_pre‖₂ on T;
cos(Δ_agg,r, Δ_agg,r−1); the number of clients aggregated. For DP runs with z > 0 only post-noise, server-side
quantities are logged (θ is the released model). Per-client pre-clip norms are never logged (06L_E E1).
Reading rule: DESCRIPTIVE. It explains the selection outcome. It never selects.

---

## 4. Controls (separate "MAIN adds no information" from "the FL recipe failed")

### 4.1 C0 — the PRE anchor
θ_pre itself: server LR 0 is exactly θ_pre (fl.server.ServerOptConfig accepts lr = 0). PRE_CALIB = θ_pre evaluated on
CALIB_VALIDATION through the frozen T3 path. This is a selection value. PRE's MAIN values are Phase-1 values. Code test:
an FL run with η_s = 0 leaves θ bit-identical after 3 rounds.

### 4.2 C1 — central fine-tune of θ_pre on MAIN (required)
- Arms: **P2_C1_FT_CENTRAL_FULL** (every parameter trainable) and **P2_C1_FT_CENTRAL_FROZEN** (item tables frozen,
  i.e. the central analogue of S8 / W*'s trained subset).
- Start θ_pre (cf08588c…, state digest ee33345b…). Data: MAIN_TRAIN_5_100_131072 (2,587,776 decisions). Objective: the
  locked SMALL objective (EXACT_FULL_K_CE…). Fresh AdamW. Schedule shape: the SMALL recipe shape. **Same budget axis as
  the FL arms:** 6.0 EFE on MAIN decisions, 12 half-EFE marks, PRACTICAL_BEST = the strictly-best mark among marks
  0.5 … 6.0 (θ_pre at mark 0 is **not** a candidate), CONTROLLED_BUDGET = 6.0. No continuation.
- LR screen on CALIB (CALIB_TRAIN → CALIB_VALIDATION, seed 2026): peak LR ∈ {1e-5, 3e-5, 1e-4, 3e-4} × {FULL, FROZEN}
  = 8 runs. Selection rule §5.6 within each variant. Edge rule §5.7 (extension values 3e-6 / 1e-3).
- MAIN: each variant × seeds {2026, 2027}.
- Interpretation: C1 − PRE measures how much MAIN data can add to θ_pre when privacy is ignored. It bounds what FL can
  capture with the same data.

### 4.3 C2 — matched-size central reference with MAIN data (optional; recommended; owner decision O3)
- **P2_C2_CFULL_SMALL**: SASREC_D64_B2, registered θ0 of seed 2026, the SMALL recipe exactly as PRE, on
  FULL_TRAIN_5_100_TIEBREAK (PRE users + MAIN users), 6.0 EFE, automatic C3 continuation as PRE had.
- C2 − PRE is the value of the MAIN users' data at matched size and recipe. It removes the size (D256 vs D64) and recipe
  confounds of C_FULL vs PRE (M8). K20b's denominator stays C_FULL in Phase 1. C2 is a Phase-2 reference only.

---

## 5. Warm-FL recipe for fine-tuning (W-grid), selection, ties, edges

### 5.1 Fixed parts (all W runs)
θ_pre start (as S7 / S8: launch_fl --init-from, identity checks unchanged); SASREC_D64_B2; FA method; 64 clients per
round, seeded sweep, 10 % seeded drop-out; local passes 1, batch 16 (the S4 point); FL_CONST LR shape; budget 6.0 EFE
(CALIB 3,414 rounds, MAIN 13,654 rounds at passes 1); 12 marks; PRACTICAL_BEST over marks 0.5 … 6.0 (**mark 0 = θ_pre is
never a candidate**, so "FL ≥ PRE" cannot hold by construction); CALIB screens seed 2026; client clip 1.0; fresh client
optimizer per visit.

### 5.2 Stage WA — server optimiser × server LR × client LR (item tables frozen = the deployed S8 structure)
Client: SGD-m 0.9 (the registered client optimizer), client LR c ∈ {0.003, 0.01, 0.03}. Cold S4 used 0.1; S8 searched
0.1 and 0.4.

| server point | definition (fl.server) | values |
|---|---|---|
| FedAvg | θ + η_s · (aggregate − θ); η_s = 1 is plain FedAvg; η_s < 1 = FedAvgM β 0 at LR η_s (launch_fl.warm_server_opt path) | η_s ∈ {0.3, 1.0} |
| FedAvgM | momentum β 0.9 | η_s ∈ {0.03, 0.1} |
| FedAdam | β1 0.9, β2 0.99, τ 1e-3, v0 τ² | η_s ∈ {1e-4, 3e-4} |

WA core = 6 server points × 3 client LRs = **18 runs**.
WA-ext (extends the S8 axis below 1e-3 at the S8 client LR 0.1): FedAdam η_s ∈ {1e-5, 1e-4} = **2 runs**.
Incumbent cell: S8's f1/sf0.1 (FedAdam 1e-3, c 0.1) = 0.170709. It may be reused **only if** the Phase-2 code reproduces
run S8Tc's CALIB trajectory bit-identically on the first 64 rounds (test P2-T3). Otherwise it is re-run (1 run).
Anchor η_s = 0: PRE_CALIB (C0; evaluation, no training).
WA total: 20 new runs (+1 if the incumbent must be re-run).

### 5.3 Stage WB — what is trained and sent (at the WA winner A*)
| id | item tables | upload | runs |
|---|---|---|---|
| WB1 | unfrozen; client LR on the three item tensors × 0.1 (param-group multiplier) | dense FP32 (S7 bytes) | 1 |
| WB2 | unfrozen; × 0.01 | dense FP32 | 1 |
| WB3 | unfrozen; × 0.1 | **delta-compressed**: for item_embed.weight the rows the client touched (its history and target tokens); for output_embed / output_bias the top-k rows by ‖row delta‖₂, k = 3,170 (2 % of K = 158,486); non-item keys dense; the server treats rows not sent as zero delta; no error feedback (a client is visited about 6 times per run) | 1 |

Bytes per visit are recorded by the fl.comm ledger (the sparse codec reports its exact payload). INFERENCE: WB3 upload is
about 2.7 MB vs 82.8 MB for dense full-model uploads and 1.05 MB for S8.

### 5.4 Stage WC — parameter-efficient variants (at A*'s server / client recipe; base frozen including item tables)
| id | variant | runs |
|---|---|---|
| WC1 / WC2 | LoRA rank r ∈ {4, 16} on qkv, out, ff1, ff2 of both CausalBlocks (v4models/sasrec_v2.py), α = r, LoRA A ~ seeded Kaiming, B = 0; client LR = c* × 10 | 2 |
| WC3 / WC4 | the same at client LR = c* × 1 | 2 |
| WC5 | personalised head: A* plus the registered PF personal vector p_u (fl/personal.py; λ 1e-4; personal LR = the registered PF rule) | 1 |

Honest note: the trained subset of S8 is only 263,822 parameters (1.05 MB). LoRA's value here is regularisation (a
low-rank update anchored at θ_pre), not bytes. For export, LoRA is merged into the base weights (same architecture, same
ONNX graph).

### 5.5 Stage WD — private variants of the final winner W* (MAIN only, not selected)
- P2_WSTAR_FA1024 (z = 0), P2_WSTAR_DP8 (ε 8), P2_WSTAR_DP1 (ε 1): DP-FedAvg with the registered machinery (Poisson,
  m = 1,024, T = 384, δ = 5e-6, RDP accountant, z 0.5642 / 1.3094 at N = 131,072). Clip S from a fresh S_CAL of W*'s
  configuration (20 rounds; record S_RECORD_P2.json, format as S_RECORD_WARM). Local passes 2 / batch 16 by the DL 219
  rule. **Open decision O7 (Fable):** keep T = 384 (registered accountant; with a small server LR the model may barely
  move, which partly explains M3) or register a new z for a longer T.
- P2_WSTAR_DPFTRL8: §7.5.

### 5.6 Selection rule (CALIB only; registered)
- Metric: CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST (the Phase-1 metric).
- Candidate set: every finite WA, WA-ext, WB, WC run (non-finite runs are excluded and recorded).
- Order: stage WA picks A* = the argmax over WA ∪ WA-ext ∪ {incumbent}. Then W* = the argmax over {A*} ∪ WB ∪ WC.
- Tie band 0.0005 (as Phase 1). Inside the band, prefer in order: (1) fewer upload bytes per visit, (2) fewer trained
  parameters, (3) the smaller effective server step (FedAdam η_s; FedAvgM η_s / (1 − β); FedAvg η_s), (4) the smaller
  client LR, (5) the lexically first label.
- Anchor flag: CALIB_GAIN = W*_CALIB − PRE_CALIB is printed beside every Q result. If CALIB_GAIN < 0.0005 the flag
  NO_MEASURABLE_CALIB_GAIN is set. **W* is still carried to MAIN**, so the confirmatory question gets an answer (a
  confirmed null is a result).
- Output: contracts/fl/P2_WARM_SELECTION_A.json and P2_WARM_SELECTION_FINAL.json (write-once, self-sha256, every run's
  value, mark, bytes, D1-ext summary), via the selection tool's code sha recorded in the lock.

### 5.7 Grid-edge rule (registered; one extension only)
If A* (or a WB / WC winner) sits on the edge of a numeric axis, that axis is extended **once**, by one step outward, at
the winner's other coordinates:
- client LR: 0.003 → 0.001; 0.03 → 0.1;
- FedAvg η_s: 0.3 → 0.1; 1.0 → 2.0; FedAvgM η_s: 0.03 → 0.01; 0.1 → 0.3; FedAdam η_s: 1e-4 → 3e-5 (at c*); 3e-4 → 1e-3
  (at c*); WA-ext 1e-5 → 3e-6 (at c = 0.1);
- WB item-LR multiplier: 0.01 → 0.001; LoRA rank 16 → 64; 4 → 1.
At most 3 extension runs per stage (two axes plus their corner). The rule is re-applied over the union. If the winner is
again on the new edge, the search stops with EDGE_UNRESOLVED, disclosed beside every Q result. No second extension.

### 5.8 Futility (saves compute; registered)
If C1_CALIB (best of the C1 screen) − PRE_CALIB < 0.0005 **and** the best WA value − PRE_CALIB < 0.0005, then WB and WC
are skipped. A* = W* goes to MAIN for the confirmatory null. Recorded as FUTILITY_SKIP_WB_WC.

### 5.9 Budget cap and drop order (registered)
Phase-2 GPU cap: 250 GPU-h (INFERENCE sizing, PHASE2_IMPLEMENTATION_PLAN §4). If the projected total exceeds the cap,
drop in this order: WC3/WC4 → WB2 → C2 → DP1 → WC1/WC2 → WB3. The primary path (WA, C1, W* ×2, FT_P2_WSTAR ×2, DP8 ×2)
is never dropped.

---

## 6. The confirmation split: TEST2 finding, options, labels

### 6.1 Finding: is REES46 data after 2019-11-30 available? (MEASURED locally; TO-VERIFY on the server)
- rees46_nextitem_v1/VERSION.json lists source months **2019-10 and 2019-11 only** (source_sha256 keys "2019-10",
  "2019-11").
- The local raw folder holds only data/raw/2019-Oct.csv and data/raw/2019-Nov.csv.
- DATA_PROTOCOL_LOCK_V2 splits: TRAIN [2019-10-01, 11-16), VALIDATION [11-16, 11-23), RETRO_TEST [11-23, 11-30),
  LABEL_GRACE [11-30, 12-01) with **zero decisions** (EVALUATION_CONTRACT_V4 L76). Nothing on or after 2019-12-01 was
  ever loaded into any V4 or v1 artefact found locally.
- REPORTED (not verified): the REES46 dataset page publishes Oct 2019 – Apr 2020 for this store (MULTI_EXPERT_REVIEW
  13_evaluation_statistics E10 and §6, "UNVERIFIED" there too). FABLE_THESIS_RESEARCH_REVIEW D-11 already proposed a
  "December 2019 fresh-period audit" (NICE, after the TEST look).
- So December 2019 is a genuinely **never-seen window**, if it can be obtained and is compatible.

### 6.2 TEST2 definition (PLAN; needs owner approval of a new data version)
- Window: decisions with decision_ts ∈ [2019-12-01T00:00Z, 2019-12-08T00:00Z) plus a LABEL_GRACE for targets
  [12-08, 12-09) with zero decisions. INFERENCE: one week, like RETRO_TEST, with no Black-Friday overlap.
- Users: MAIN users (client keys via v4base.identity) with ≥ 1 rankable decision in the window (the participating-client
  estimand). Optional panel: UNEW_HOLDOUT users (new-user estimand).
- History: causal, USER_HISTORY_L50 over every event before the decision (Oct 1 → decision time), as retro_store does
  for RETRO_TEST.
- Features and transforms: TRANSFORM_STATE_CLEAN_V2 unchanged (TRAIN-fitted). Item first_seen / age from TRAIN only.
  Catalogue K = 158,486 unchanged. A target outside the catalogue is OOV: credit 0, kept in the E2E denominator.
- Models: frozen checkpoints only. No model is retrained with December data.
- Data version: a new immutable version (proposal rees46_nextitem_v2_dec2019) built with the v1 pipeline code
  (code_hash 02fbb071…). v1 stays untouched.

TO-VERIFY (exactly what Ops must check, in order; COUNTS_ONLY, no model scored):
1. Obtain the December 2019 file of **the same multi-category store** (the Oct / Nov files' source, not the cosmetics
   store). Record URL, file name, bytes, sha256, row count.
2. Schema identical to 2019-Nov.csv: columns event_time, event_type, product_id, category_id, category_code, brand,
   price, user_id, user_session; same timestamp format; UTC.
3. Event-type set equal to Oct / Nov (view, cart, purchase). Any extra type is reported.
4. Store identity: the share of December events whose product_id is in catalogue.parquet (expect a majority), and the
   share of those whose category_id / brand match the v1 vocabularies (vocab_*.parquet).
5. Boundaries: first / last event_time; no overlap with 2019-11-30 rows; no duplicate rows across the month boundary.
6. Overlap counts: the number of MAIN users (and UNEW_HOLDOUT users) with ≥ 1 decision in [12-01, 12-08), the number
   of rankable decisions, and the OOV share of targets. Power follows from these counts (§9.6).
7. The owner approves the new data version (CLAUDE.md: frozen data is never modified; a new version is a separate
   artefact). Fable approves the TEST2 manifests' spec before the build.
8. A TEST2 exposure register (separate file, the same tool format) records the build as MANIFEST_BUILD / COUNTS_ONLY.

### 6.3 Options for Fable (T0 added on Ops' request, DL 406)
| option | what happens | when Phase-1 TEST is scored | Phase-2 confirmation split |
|---|---|---|---|
| **T0a** (full lock before the look) | Register PREG2 now; run CALIB selection and MAIN training; write P2_METHODS_LOCK (arms with checkpoint shas, contrasts, code shas); then **one TEST sitting** scores Phase-1 and Phase-2 arms together | after Phase-2 MAIN runs finish (INFERENCE 6–9 days of compute after code is ready) | RETRO_TEST |
| **T0b** (procedure lock before the look) | Register PREG2; finish and review Phase-2 code; write P2_PROCEDURE_LOCK (selection rules, grids, analysis code sha, contrasts; W* defined as the output of the registered procedure); then the Phase-1 TEST look runs; Phase-2 arms are scored on RETRO_TEST later, once each | after the Phase-2 code is ready and locked (INFERENCE 4–6 days) | RETRO_TEST (scored after Phase-1 values were seen) |
| **TEST2 only** | Phase-1 TEST look runs now; Phase 2 is confirmed on December 2019 only | now | TEST2 |
| **T0b + TEST2** | T0b, then a pre-registered replication on TEST2 | after the code lock | RETRO_TEST + TEST2 replication |
| **fallback FB** (if December is unavailable or incompatible) | Phase-2 arms scored on the RETRO_TEST window of a user band that no Phase-1 TEST manifest contains | n/a | see §6.5 |

### 6.4 Honest comparison and the label each claim would carry
| claim | T0a | T0b | TEST2 only | T0b + TEST2 |
|---|---|---|---|---|
| Phase-1 K1…K21 | CONFIRMATORY (PREG1), unchanged | same | same | same |
| Phase-2 arms on MAIN VALIDATION | POST_VALIDATION_EXPLORATORY | same | same | same |
| Q1 "FL adds value over PRE" | **CONFIRMATORY_P2_T0**: hypothesis generated after the VALIDATION read; arms, selection and analysis locked before any TEST value; CALIB selection | **CONFIRMATORY_P2_T0B**: procedure locked before TEST; values scored after the Phase-1 TEST values were seen. Any amendment made after the Phase-1 look downgrades the affected Q to EXPLORATORY | **CONFIRMATORY_P2_FRESH** (new period, nothing seen) | CONFIRMATORY_P2_T0B, then **REPLICATED_FRESH** if TEST2 gives the same sign with CI excluding 0 |
| Phase-1 arms on TEST2 | n/a | n/a | FRESH_PERIOD_AUDIT (DESCRIPTIVE) | FRESH_PERIOD_AUDIT (DESCRIPTIVE) |

Risks and costs, stated plainly:
- All T0 variants: RETRO_TEST was inspected earlier by the v1 central campaigns (period-level; Phase-1 disclosure
  E12(d)). The Phase-2 hypotheses were motivated by K21 < 0 on VALIDATION. The direction of the test was chosen after
  seeing VALIDATION, but no Phase-2 arm value on TEST can have been seen. One TEST sitting (T0a) or one score per arm
  (T0b) keeps the single-look principle per arm. Both need Fable's ruling that a second registered study may use the
  same TEST manifests (built under ALL_METHODS_LOCK 04a3cd19…, register entries 5–6).
- T0a delays the Phase-1 result by the whole Phase-2 compute. T0b delays it only by the code-and-lock time, but it
  weakens protection: after the Phase-1 look, a bug fix in Phase-2 code would be an amendment made with TEST knowledge
  (PRE's TEST value is a Phase-1 value).
- TEST2 needs new data engineering (INFERENCE 2–4 days incl. review), owner approval, and carries a distribution shift
  (December, two weeks after TRAIN ends, more OOV), fewer active MAIN users (wider CI). It is the only option that
  removes the "previously inspected period" and the "hypothesis after VALIDATION" concerns together.
- Recommendation (INFERENCE, for Fable to decide): **T0b + TEST2 replication** if TO-VERIFY items 1–6 pass within about
  5 days; **T0a** if December fails, and only if the owner accepts the Phase-1 delay; T0b alone otherwise.

### 6.5 Fallback FB (registered limits)
User bands of the 5–100 pool (REGISTRY_V2_PRETRAIN_EXT): MAIN [0, 131,072) (CALIB [0, 32,768) inside it); PRE
[131,072, 1,348,682); U_NEW_VAL [1,348,682, 1,381,450); U_NEW_TEST [1,381,450, 1,414,218) (UNEW_HOLDOUT); FS
[1,414,218, 1,446,986). **No band of the pool is unused**: FS served the feature search, U_NEW_VAL the selection panel,
UNEW_HOLDOUT is a Phase-1 TEST role.
FB = the RETRO_TEST window of the **U_NEW_VAL band** (not in any Phase-1 TEST manifest; never in PRE or FL training).
Its limits: (i) a different estimand: these users are not FL clients, so FB measures transfer of the federated model to
non-participants, not the benefit to participants (Q1's estimand); (ii) same week as RETRO_TEST (Black-Friday week),
inspected at period level by v1; (iii) U_NEW_VAL was a selection panel (VALIDATION window). Label:
HOLDOUT_USERS_SAME_PERIOD, never CONFIRMATORY for Q1. TO-VERIFY: that U_NEW_VAL users' RETRO_TEST rows are in no v1 or
V4 TEST artefact that was scored. Conclusion: for Q1, FB is weaker than T0. Its value is the new-user estimand.

---

## 7. Privacy: audit, secure aggregation, untrusted-server DP

All audits are separate audit runs or read-only analyses. None selects an arm. None reads VALIDATION / TEST labels.
Members / non-members use TRAIN-window rows only.

### 7.1 E1 — user-level membership inference on final models
- Targets: PRE (null control), FA_WARM_FROZEN s2026 (Phase-1, read-only), P2_WSTAR s2026, P2_WSTAR_DP8 s2026,
  P2_WSTAR_DP1, P2_C1_FT_CENTRAL_FULL s2026 (the central worst case).
- Members: 10,000 MAIN users (seeded draw, seed 20261004). Non-members: 10,000 UNEW_HOLDOUT users (never in PRE or FL;
  same pool and eligibility). Each user's TRAIN-window rows.
- Score: s(u) = mean CE_θ(u) − mean CE_θpre(u) (reference-calibrated loss, LiRA-style with θ_pre as the OUT model, since
  θ_pre never saw MAIN). The raw loss is reported beside it.
- Protocol: users split 50/50 (seeded). The threshold is chosen on half A at FPR 1 % and evaluated on half B.
- Metrics: ROC AUC with a user-bootstrap 95 % CI; TPR at FPR ∈ {0.1 %, 1 %}; ε_LB = ln(TPR_lo / FPR_hi) with
  Clopper–Pearson 95 % bounds.
- Criteria: (a) validity: PRE's AUC CI must include 0.5, else the audit is invalid (population mismatch) and stops;
  (b) DP arms: **PASS iff ε_LB ≤ the accountant ε** (8 / 1); FAIL triggers an investigation (accountant or
  implementation); (c) non-DP arms: MATERIAL_LEAKAGE iff the AUC CI lower bound > 0.55 or the TPR@1 % lower bound > 3 %;
  otherwise reported as measured.

### 7.2 E2 — canary exposure and one-run audit on FL training
- Audit runs (CALIB scale, never evaluated): W*'s recipe on CALIB_TRAIN clients plus 1,000 synthetic canary clients.
  Each canary has a random length-20 sequence of long-tail catalogue items and a fixed random target. Each canary is
  included with probability ½ (seeded). Runs: W* (non-DP), W* with the WB1 tables unfrozen (non-DP), W*-DP8 (z computed
  by fl.accountant for N = 32,768 + canaries at T = 384, m = 1,024).
- Score: log p_θ(canary target | canary history) − log p_θpre(…).
- Metrics: exposure (Carlini et al. 2019) against 10,000 random reference sequences (non-DP runs); the one-run audit
  lower bound ε_LB (Steinke, Nasr & Jagielski 2023) from inclusion guesses at 95 % confidence.
- Criteria: DP run **PASS iff ε_LB ≤ 8**; non-DP runs reported (DESCRIPTIVE), with frozen vs unfrozen side by side.

### 7.3 E3 — leakage from individual FL updates
- Replay (CALIB, rounds 0–49) of W* and of WB1 with an upload-capture hook for 512 seeded clients (audit-only flag;
  captured uploads deleted after the analysis).
- Attacks: (i) unfrozen tables: item-set recovery = recall of the client's TRAIN items among the nonzero / top rows of
  its item_embed delta; (ii) frozen tables / LoRA: a linear probe on the upload predicting the client's top main
  category (5-fold CV) vs the majority baseline; (iii) both attacks on the SecAgg aggregate of 64 clients (per-client
  advantage).
- Criterion: SECAGG_NEEDED iff some attack's advantage over its baseline is > 10 percentage points on single uploads and
  ≤ 2 points on 64-client aggregates. Otherwise reported as measured.

### 7.4 E4 — secure-aggregation simulation
- Protocol: Bonawitz et al. 2017 (SecAgg) with the SecAgg+ k-regular neighbourhood option (Bell et al. 2020).
  Cryptography is simulated: pairwise seeds from a seeded PRG per pair. Sizes are counted at real values: X25519 public
  keys 32 B, AES-GCM-wrapped Shamir shares, 16 B tags.
- Fixed-point encoding: each client's clipped delta is encoded mod 2^32 with a shared scale from the clip bound. A
  per-client scale (as fl.quant Q8) cannot be summed, so SecAgg uses a common scale.
- Dropout: p ∈ {0, 0.1, 0.3} at group size 64; threshold t = ⌈2n/3⌉.
- Metrics: per-client upload / download bytes per round (overhead ratio vs the plain upload), communication rounds (4),
  server unmasking CPU time, success rate under dropout.
- Criteria: (a) the decoded sum is bit-identical to the plain fixed-point sum on every simulated round (test);
  (b) quality: P2_WSTAR_SECAGG vs P2_WSTAR on CALIB, TOST with 90 % CI inside [99, 101] % (DESCRIPTIVE; CALIB) and, if
  budget allows, MAIN s2026 (tertiary Q13); (c) the overhead ratio is reported (INFERENCE: below 2× at n = 64 for a 1 MB
  upload).
- Stated limit: SecAgg plus central DP still trusts the server with the noise. Distributed DP (DDGauss / Skellam) is out
  of scope and disclosed.

### 7.5 E5 — DP without a trusted aggregator / without sampling amplification
- **DP-FTRL** (Kairouz et al. 2021; the Gboard production mechanism): tree-aggregation noise, restart per epoch. It fits
  the sweep plan (each client participates at most once per sweep, about 6 sweeps per run). Clip S from S_CAL of W*.
  The noise multiplier comes from a tree-aggregation zCDP accountant for ε = 8, δ = 5e-6, N = 131,072 (new code,
  reviewed). Arm P2_WSTAR_DPFTRL8, MAIN seed 2026 (plus 2027 if the budget allows). Contrasts Q12a
  R(DPFTRL8 / W*) and Q12b DPFTRL8 − PRE (tertiary). Audit by E1 / E2 with the same PASS rule.
- **Local DP feasibility** (untrusted server): each client clips to S and adds Gaussian noise σ_L·S before upload;
  per-upload ε_L ∈ {8, 32} via the Gaussian mechanism; composition over about 6 uploads reported. One CALIB run per
  ε_L. DESCRIPTIVE only (INFERENCE: large utility loss at 64 clients per round). Label LDP_FEASIBILITY.

---

## 8. Real-device measurement protocol

- Models: SASREC_D64_B2 FP32 and INT8 ONNX (opset 17, batch 1) of PRE and of P2_WSTAR (LoRA merged if WC wins; the
  architecture is identical), plus C_SMALL (Phase-1 reference). Exported with the existing v4dev.export / v4dev.quantize;
  parity by v4dev.bench (top-20 set identity on 50 reference queries ≥ 0.98 for FP32; INT8 reported).
- Devices (TO-VERIFY with the owner: which devices he has):

| tier | device | runtime / EP | status |
|---|---|---|---|
| D1 | owner's Windows 11 PC, Microsoft Edge | onnxruntime-web: wasm-1T, wasm-MT (served with COOP/COEP), WebGPU | must (page exists: code/v4dev/browser) |
| D2 | owner's Android phone, Chrome | onnxruntime-web wasm-1T / MT | must if available |
| D3 | the same phone, native | onnxruntime-android (minimal bench app) **or** Termux + onnxruntime wheel | should (TO-VERIFY feasibility) |
| D4 | a low-end Android (≤ 4 GB RAM) | onnxruntime-web wasm-1T | nice |
| D5 | iPhone, Safari | onnxruntime-web wasm | nice |
| REF | server CPU (v4dev.bench, K16 pass 1 / pass 2) | onnxruntime CPU, 1 thread | Phase-1 reference (K16 pass 2 PENDING) |

- Metrics per device × model × EP: latency p50 / p95 / p99 (ms) over 1,000 timed batch-1 queries after 20 untimed
  warm-ups (the existing discipline); cold start (model fetch + session create + first inference); peak memory (Android
  `dumpsys meminfo <pkg>` PSS peak; browser `performance.measureUserAgentSpecificMemory()` where available, else
  "n/a"); model size (on disk, gzip, brotli); energy proxy (Android only: `dumpsys batterystats --reset`, screen on at
  fixed brightness, airplane mode, 30 min continuous inference vs 30 min idle with the page open; ΔmWh per 1,000
  inferences from the battery-level / estimated-power delta; CPU-seconds per 1,000 inferences from /proc/<pid>/stat for
  the native runner; D1: CPU-seconds from a native onnxruntime Python run as a proxy).
- On-device training cost (one FL visit / one FT step): measured with PyTorch CPU on D1 as a proxy (label PC_PROXY).
  Measured on D3 only if ORT training works there (TO-VERIFY).
- Runs: 3 sessions per configuration on separate reboots or days. Device plugged in for latency, on battery for energy.
  2 min cool-down between configurations. Battery 80–95 % at the start. Ambient temperature and device temperature
  recorded. Report the median over sessions of each percentile and the between-session range.
- Registered "deployable" thresholds (INFERENCE design targets; Fable may edit): D2 INT8 p95 ≤ 100 ms; cold start ≤ 3 s;
  peak memory ≤ 300 MB; brotli download ≤ 25 MB. Each device / model gets DEPLOYABLE or NOT_DEPLOYABLE per threshold.
  DESCRIPTIVE, no inference.
- Output: DEVICE_BENCH_P2.json per device (device identity: model, SoC, RAM, OS, browser and ORT versions) and a
  summary table. Never mixed with the A40 retention key.

---

## 9. Hypotheses, contrasts, statistics, stopping, success / failure

### 9.1 Contrasts (paired user-level on the confirmation split; MAIN users; PRACTICAL_BEST unless stated)
| id | contrast | tier | seeds | reading |
|---|---|---|---|---|
| **Q1** | P2_WSTAR − PRE | **primary** | W* {2026, 2027} vs PRE s2026 (one θ_pre, shared) | ABOVE / INCLUDES / BELOW 0 |
| Q2 | P2_C1_FT_CENTRAL_FULL − PRE | secondary | {2026, 2027} | headroom by central fine-tune |
| Q3 | P2_WSTAR − P2_C1_FT_CENTRAL_FROZEN | secondary | {2026, 2027} | FL vs central at the same trained subset |
| Q4 | P2_WSTAR − FA_WARM_FROZEN | secondary | {2026, 2027} (same-seed pairs) | effect of the recipe fix |
| Q5 | FT_P2_WSTAR − FT_PRE | secondary | {2026, 2027} | value persists after local fine-tune (deployed stack) |
| Q6 | R(P2_WSTAR_DP8 / P2_WSTAR_FA1024) | secondary | {2026, 2027} | privacy cost at ε 8 (CONTROLLED_BUDGET, as K12) |
| Q7 | P2_WSTAR_DP8 − PRE | secondary | {2026, 2027} | private FL over PRE (CONTROLLED_BUDGET) |
| Q8 | P2_C2_CFULL_SMALL − PRE | secondary (only if C2 runs; else removed before the lock) | {2026} | matched-size value of MAIN data |
| Q9 | R(P2_WSTAR / C_FULL) | tertiary | {2026, 2027} vs C_FULL s2026 | beside K20b |
| Q10 | Q1 / Q2 (headroom capture ratio) | tertiary, DESCRIPTIVE | — | share of available headroom FL captures |
| Q11 | R(P2_WSTAR_DP1 / P2_WSTAR_FA1024) | tertiary | {2026} | ε 1 |
| Q12a/b | R(DPFTRL8 / W*), DPFTRL8 − PRE | tertiary | {2026} | trusted-aggregator-free DP |
| Q13 | R(P2_WSTAR_SECAGG / P2_WSTAR) | tertiary, TOST [99, 101] on 90 % CI | {2026} | SecAgg equivalence |
| Q14 | P2_WSTAR (TEST2 users) − PRE, new-user panel | tertiary | {2026, 2027} | new-user estimand |

DP arms' PRACTICAL_BEST ratios are DESCRIPTIVE (best-of-12 selection on users' validation data is outside the
accountant), as in Phase 1. Q6 / Q7 / Q11 / Q12 read CONTROLLED_BUDGET.

### 9.2 Statistics
Paired user bootstrap, 1,000 resamples, seed **20261004**, one shared resample matrix over the confirmation split's
users. Percentile 95 % CIs. Ratios bootstrapped directly. Seed mean on the same W. Per-seed points, seed spread and the
SEED_SENSITIVE flag as Phase 1 (06L-C C13–C15). Harmed-client share with the seed-vs-seed null beside every Q. E2E
client-macro MRR@20 is primary; NDCG@10 and HR@10 are DESCRIPTIVE. Implemented by reusing tools/retention_report.py's
Engine / ratio / delta / holm_band under a P2 registry profile (plan P2-12).

### 9.3 Multiplicity
Q1 alone is primary, read unadjusted at α = 0.05 (two-sided CI). The secondary family {Q2…Q7 (+Q8)} uses Holm
step-down at α = 0.05, with Holm bands as in Phase 1. Tertiary contrasts are unadjusted with a DESCRIPTIVE Holm band.
TEST2 replication (if chosen) is a separate family: Q1-rep is read at α = 0.05 with the same rule. The Phase-1 family is
untouched and never pooled with Phase 2. Algebraic dependencies are listed (Q10 = Q1 / Q2; Q3 = Q1 − (C1_FROZEN − PRE)).

### 9.4 Success / failure for "FL adds value over PRE"
- SUCCESS: Q1 95 % CI lower bound > 0 **and** both per-seed points > 0 (no SEED_SENSITIVE flip). The claim wording
  carries the confirmation label of §6.4.
- DEPLOYMENT-GRADE SUCCESS (statement, not a separate test): SUCCESS plus Q5 > 0 (Holm-rejected) plus Q7 > 0
  (Holm-rejected).
- EQUIVALENT TO PRE: Q1's 90 % CI lies inside ±0.0005 MRR@20 (TOST; margin = the registered tie band). The claim is
  "no practically relevant gain".
- FAILURE: Q1's CI upper bound < 0 ("FL with the selected fine-tuning recipe hurts").
- INCONCLUSIVE: anything else.

Interpretation grid (registered wording):
| Q2 (central headroom) | Q1 (FL) | reading |
|---|---|---|
| > 0 | > 0 | FL captures part of the available headroom (Q10 = how much) |
| > 0 | ≤ 0 or equivalent | the FL recipe fails to capture available headroom (an FL problem) |
| equivalent to 0 | equivalent to 0 | MAIN data adds no measurable information over θ_pre at D64; FL cannot help here (a data / capacity limit, not an FL failure) |
| equivalent to 0 | > 0 | flagged for review (FL beats central fine-tune; check for a central-recipe problem) |

### 9.5 Seeds, budgets, stopping
- CALIB screens: seed 2026. MAIN: seeds {2026, 2027} for W*, C1 (both variants), FT_P2_WSTAR, DP8; {2026} for DP1,
  DPFTRL8, SECAGG, C2. θ_pre is one seed shared by every warm arm; seed spread measures FL variance only (disclosed).
  Option O5: a second θ_pre seed (PRE s2027) to measure pretraining variance (owner / Fable decision; cost in the plan).
- Budget per run: fixed 6.0 EFE; no early stopping; no continuation for FL or C1 runs (C2 has PRE's automatic C3 rule so
  that it matches PRE).
- Run-level stopping: a non-finite value excludes the run from selection (recorded). A mechanical failure resumes under
  identical hashes. A second failure with the same signature → INCOMPLETE by a DL row.
- Study-level stopping: the futility rule §5.8; the budget cap and drop order §5.9. No interim look at any confirmation
  split.

### 9.6 Power (INFERENCE)
K21's VALIDATION CI half-width 0.00033 → SE ≈ 0.000168 at 40,764 users. MDE (80 % power, α 0.05) ≈ 2.8 × SE ≈ 0.00047
for a split of VALIDATION size (RETRO_TEST counts: TO-VERIFY from register entry 6's COUNTS_ONLY record). The plausible
FL ceiling is about +0.001 (§1). A W* that captures half of it is near the MDE. TEST2's MDE scales with
√(40,764 / n_TEST2-users) (TO-VERIFY item 6).

---

## 10. External validity and novelty (stated limits)
- One main dataset and event type. Optional (owner decision O6): replicate the PRE / C1 / W* triangle on one benchmark
  dataset already prepared by BM-02 (Track B), with users split into a pretraining cohort and a federated cohort.
  DESCRIPTIVE, separate report. A DESCRIPTIVE stratum by target event type (view / cart / purchase) on the confirmation
  split, if the evaluation rows carry the event type (TO-VERIFY in DATA_CONTRACT_CLEAN_V2).
- Novelty: the methods are known (FedAvg / FedAvgM / FedAdam, LoRA, DP-FedAvg, DP-FTRL, SecAgg, LiRA / one-run audits).
  The contribution is the evaluation design: a fine-tuning-specific warm-FL recipe for a full-catalogue sequential
  recommender, the headroom decomposition (C1 / C2 vs FL, Q10), an empirical privacy audit of item-table leakage with and
  without SecAgg, and real-device numbers, all under pre-registration.

## 11. Disclosures (verbatim-ready)
D-a. Phase 2 was designed after the Phase-1 VALIDATION read (K21 < 0). Its hypotheses are post hoc. Its confirmation
     uses only data no Phase-2 arm was ever scored on (§6).
D-b. CALIB selection data are a subset of the MAIN users and of the VALIDATION window that was read.
D-c. θ_pre is one seed. The PRE recipe was not re-tuned at its scale (Phase-1 note 3).
D-d. C_FULL differs from PRE in size and recipe. C2 (if run) is the matched reference.
D-e. Central DP trusts the aggregator. SecAgg is simulated (costs counted, cryptography not executed). DP-FTRL / LDP are
     the untrusted-server variants.
D-f. Device numbers come from the listed devices only. On-device training cost is a PC proxy unless stated otherwise.
D-g. RETRO_TEST was inspected at period level by earlier v1 campaigns (Phase-1 E12(d)).

## 12. Open decisions
| id | decision | who |
|---|---|---|
| O1 | Confirmation option T0a / T0b / TEST2 / T0b + TEST2 (§6.3) and the labels (§6.4) | Fable (owner informed) |
| O2 | New data version with December 2019 (§6.2) | owner, then Fable for the manifests' spec |
| O3 | Run C2 (≈ 10–12 GPU-h per seed, INFERENCE) | owner |
| O4 | Grid values, tie band, edge steps, futility threshold, cap 250 GPU-h | Fable |
| O5 | Second θ_pre seed | owner / Fable |
| O6 | Second-dataset replication | owner |
| O7 | DP variant at T = 384 (registered accountant) vs a new z for a longer T | Fable |
| O8 | Device list and the deployable thresholds | owner (devices), Fable (thresholds) |
| O9 | Whether TEST manifests built under ALL_METHODS_LOCK may serve a second registered study (T0) | Fable |

## 13. Registration procedure
1. Fable gate on this draft → edits applied → contracts/PREG2.md written O_EXCL, its sha256 in a DL row = REGISTERED.
2. Code packets of PHASE2_IMPLEMENTATION_PLAN with reviews. Their code shas are recorded.
3. T0b: P2_PROCEDURE_LOCK.json (O_EXCL) before the Phase-1 TEST look. T0a: P2_METHODS_LOCK.json after the MAIN runs,
   before the single TEST sitting.
4. TEST access for Phase 2 only via a TEST_EXPOSURE_REGISTER entry (LOCKED_EVALUATION) whose
   all_methods_lock_sha256 field binds the P2 lock. TEST2 access via its own register.
5. Every deviation is an amendment file (PREG2_AMENDMENT_xx.md) with a DL row, written before the affected value is read.

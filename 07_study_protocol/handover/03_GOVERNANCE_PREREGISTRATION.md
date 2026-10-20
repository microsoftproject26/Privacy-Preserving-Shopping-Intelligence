# 03 — Governance and pre-registration

*Paths relative to `.local_ai/ppsi_v4/`. "DL n" = decision-log row. Every rule here is binding unless a later registered
file supersedes it. If you are unsure whether something is allowed, it is not: register it first.*

## 1. The core rules

1. **Register first, read later.** Any rule, grid, arm, contrast, seed set or interpretation is written into an amendment
   (`contracts/PREG1_AMENDMENT_*.md`, `contracts/PREG2.md`) or a DL row, hashed, and logged **before** a value it governs
   exists or is read. "No rule change after its value is read" is the invariant.
2. **Amendments are never edited.** Corrections go into a new file, an appended ERRATUM (only before any read, e.g. 06P
   ERRATUM 1, DL 324) or a new DL row. The one accidental overwrite (06F, DL 185) was restored byte-identically (DL 188).
   Before writing under `contracts/`, check the DL tail and whether the file exists (the 06F lesson).
3. **Tuning only on CALIB; device-recipe selection only on UNSEEN_VAL.** MAIN_VALIDATION was read **once** (§4). TEST is
   looked at **once** (§6).
4. **Selections are mechanical.** A selection file is written by a reviewed tool (`v4int/lr_screen.py select…`,
   `launch_device --select`, `v4int/p2_select.py`, the benchmark runner) and copied write-once (O_EXCL, read-only) into
   `contracts/` with a DL row. Never hand-edit a selection.
5. **Everything written once.** Locks, freezes, selections, S records, register entries, run outputs: O_EXCL + `chmod a-w`.
   The endgame scripts refuse when their output exists (`once`) or their predecessor's marker is missing (`need`).
6. **Frozen code.** `code/v4eval/*.py` (evaluator hash `61b64e61…`) has not changed since the A40 block began (DL 180);
   any edit would split the comparison block. Deployed server trees are read-only; a run resumes only on its own tree
   (06E §8). New code → local `code/` → tests + negative controls (NCs) → independent review → a **new** tree with per-file
   sha verification and a tiny-run digest comparison of the Phase-1 FL arms (`traj_cmp*.sh`).
7. **Never rewrite a running lane script** (bash reads scripts incrementally). Stop or withdraw the lane, log it, start a new
   one.
8. **Separate the evidence labels** (§10). An ETA is never a result.

## 2. Amendment timeline (one line each)

| Id | DL | What it registered | Why |
|---|---|---|---|
| 01 | 26–27, 55 | Clarifications: tie chain, manifest ids, endpoint tolerance, STRICT determinism, eval_stream = EVAL_ROWS_ONLY | make the contest rules executable |
| 02 | 59 | First P2 contest design | superseded before use |
| 02R1 | 60–62 | Model size inside the contest search; full 6 EFE; no legacy anchoring (sha `0afab4a0…` was 02) | owner: fair, non-legacy contest (DL 60) |
| 02R2 | 69–70 | LR grid stratified within each size (sha `1fe3258a…`) | a review found size confounded with LR (DL 69 F5) |
| 03 | 84 | BIG / SMALL = one family, two sizes; device cap ≤ 170 MB per visit (sha `23600aac…`) | owner's downstream design (DL 83) |
| 04 | 87 | Realistic downstream plan: retention vs C_BIG, **no threshold**; arms; DP-FedAvg; baselines; ONNX; strata (sha `cb94d3e3…`) | answer "how much is retained" (DL 86) |
| 05 | 95 | CALIB P2 = screen only; 2 candidates × 2 seeds = 8 MAIN contest runs; one sealed two-step G2 (sha `35d42070…`) | owner objection to CALIB-only selection |
| 06 | 133 | Downstream execution after G2: SMALL recipe, FL questions Q1–Q5 (z values), deferred arms, GPU sharing, C_FULL spec, strata | after ARCHITECTURE_LOCK |
| 06A | 146 | Device selection on UNSEEN_VAL; observational cold-start bins; communication curves; A40 block; ONNX INT8 subset; ANA freeze; disclosures | full review (FABLE_FULL_REVIEW_AND_PLAN) |
| 06B | 147–148 | FL local-LR screen; contrasts K1–K16; privacy ladder; DP ε = 1 core; FA_Q8; browser measurement (sha `96cf91c8…`) | thesis-level research review |
| 06C | 150 | DP sampling → **Poisson** with a fixed denominator | the fixed-size sampler did not match the accountant (ε would be ≈ 1,388, not 8) |
| 06D | 159 | A40 = second registered device class (pin); an executable TEST look; K12 / K13 at CONTROLLED_BUDGET; K2 primary; tiers MUST / SHOULD / NICE (sha `fac97ce4…`) | deep review found two blockers (no A40 path, no TEST path) |
| 06E | 169b–171 | LO selection on UNSEEN_VAL; registered seed sets; K16 in an ONNX-CPU block; comparison-key and evidence rules | reviews REV-FL-01 / REV-ANA-01 |
| 06F | 184–188 | FL LR screen extended to ×4 / ×8 / ×16; the m = 2 MAIN runs superseded (sha `182f879b…`) | S1a winner at the grid edge |
| 06G | 183, 186, 193 | FedAdam server optimiser grid; v0 = τ² | standard production practice |
| 06H | 189 | DP server-optimiser sensitivity (DP_OPT_SENS) (sha `9c10f132…`) | FedAdam can turn DP noise into a random walk |
| 06I | 196, 199 | K17–K19 vs C_FULL; "FA" = the tuned recipe; FL_CONST × client optimiser; FA_X2; DP8_FULL; realism disclosures (sha `1b627f22…`) | realism audit |
| 06J | 202, 204 | FL ladder S2 → S3 → S4 strictly sequential; ≥ 0.002 incumbent rule; 0.0005 tie band; edge rules (sha `162cc01f…`) | ladder research |
| 06K | 206–207 | S7 = PRE + FA_WARM; S8 = frozen item tables; K20, K21; never warm-start from a MAIN-trained model | real systems warm-start from permitted data |
| 06L (A) | 226 | FedAvgM cells, S3 combination cells, DP_OPT_SENS for non-FedAvg winners, clipped-fraction reporting | multi-expert review gates |
| 06L_C | 227 | Privacy matrix, threat model, LAST anchoring + K11b, contrasts v3, deployment-branch contrasts K9b / K9c / K20b / K20c, S8 two seeds, claim register, DP8_FROZEN | multi-expert review |
| 06L_D | 237 | Executable reading of 06L for the selector | frozen selector would refuse the 06L runs |
| 06L_E | 240 | Under DP (z > 0) nothing about clipping is recorded | a raw clipped count would be an un-noised output |
| 06M | 252 | Early warm-start preview on CALIB, descriptive only | owner: "train the model first, as real systems do" |
| 06M_B | 253 | Preview extension; 2-D S7 / S8 grids; S3b contingency; FT_FA_WARM | — |
| 06N | 282 | **Pretrained line primary** (K20b + K21); cold line = ablation + K_INIT; cold FP / FA_1024 / DP chains stopped (STOPPED_REALLOCATED); order of work | owner decision after the run-allocation memo |
| 06O | 286 | Descriptive C_FULL / PRE table on the non-MAIN UNSEEN_VAL panel | owner wanted a picture without opening MAIN |
| 06P | 317 (+ ERRATUM 1 DL 324; final sha `75f8dbad…` DL 331) | FA_WARM* = S8; primary pair {K20b, K21} unadjusted; Holm family of 18; warm-method contrasts K4w / K5w / K12w / K13w / K14w; K_INIT; P5; ANA_CONTRASTS_v4 | Fable end-game review D1 (owner delegated to Ops + Fable, DL 316) |
| 07 | 341 | External benchmarks, sections B1–B11 (sha `29beb5e3…`) | owner: external benchmarks (DL 296, 302) |
| 07 B12 | 357 | DP S record per PRE seed on the benchmarks | s2027 DP jobs refused by design |
| 07 B13 | 368 | ML-1M Track-B clock backstop at 2026-10-06 00:00Z (sha `9baa0f4e…`) | schedule |
| 07 B14 | 377 (Wave 2) | Wave 2 = Amazon 2023 Video_Games + Baby_Products, Track A only, CPU (sha `d5b062f1…`) | measured timing; Beauty_and_Personal_Care infeasible |
| 07 B15 | 401 | **B13 rescinded**: every ML-1M Track-B cell runs to completion (sha `6044dd04…`) | owner preferred the full grid |
| 07 B16 | 433 | ML-1M moves to the deterministic A40 class, from scratch; hard switch; CPU dirs sealed SUPERSEDED_DEVICE (sha `85ba5e18…`) | Fable re-review at the owner's request (DL 404) |
| PREG2 | 411 | Phase-2 follow-up study (sha `be0561ab…`); P2_CONTRASTS v1 → v2 (DL 424) → **v3** (DL 427) | K21 ≈ 0 and D1 (DL 405–407) |

**Registrations made in DL rows** (same force as an amendment): 198 (LADDER_S1a descriptive), 219 (DP arms keep 2 passes /
batch 16), 222 (the ≥ 0.002 rule reads raw values), 229–230 (raw-event bytes rule; ANA_CONTRASTS_v3 ratified; LAST CPU
exception), 242 / 261 / 263 (tie-order and disclosure rules), 279 (S_RECORD 0.284363), 300 (S_RECORD_WARM_FROZEN), 317
(06N execution D2, run-or-declare D3, command fixes D4), 336 (ANA_CONTRASTS_v4 sha `f1769f0b…`), 342 (frozen tool shas),
351 (SELECTION_FT_FA), 368 (K16 pass-1 / pass-2 timing), 385 (the descriptive metric table, copy-only extractor
`metric_table.py` `561d758a…`).

## 3. Gates and locks, in order

| Step | DL | File / sha |
|---|---|---|
| G0: data protocol lock v1 | 23 | `contracts/DATA_PROTOCOL_LOCK.json` (historical) |
| Owner amendment POP 5–100 | 31 | `company/owner/OWNER_AMENDMENT_POP_5_100_2026-09-26.md` |
| Feature search locked (S*) | 53 | `runs_fs/FS_FINAL.json`; schema self `c53d47fe…` |
| Data protocol lock v2 | 54–55 | `contracts/DATA_PROTOCOL_LOCK_V2.json` (`bc699a97…`) |
| G1 code readiness | 55, 80 | `company/gates/G1_APPROVED.json` |
| G1b parts 1 / 2 (contest launches) | 92, 109 | `company/gates/G1B_APPROVED.json` |
| G2 contest (the single unsealing) | 129–130 | `runs_p3/g2/`; `company/gates/G2_SCRIPT_FREEZE.json` (`tools/g2_run.py` `3893a49a…`) |
| ARCHITECTURE_LOCK | 131 | `contracts/ARCHITECTURE_LOCK.json` (`fd201013…`) |
| A40 device pin | 167, 169 | `contracts/eval_device/EVAL_DEVICE_PIN_A40.json` (`db744ca3…`) |
| FL tuning selection (S4) | 271 | `contracts/fl/FL_TUNING_SELECTION.json` (`9bca186e…`) |
| Warm selections S7 / S8 | 292 | server `contracts/fl/FL_WARM_SELECTION_S7|S8.json` (`d67a0a85…` / `8b79d305…`) |
| Contrast registry v4 | 336 | `company/gates/ANA_CONTRASTS_v4.json` (`f1769f0b…`) |
| Analysis freeze v1 → v2 → **v3** | 388, 393, 397 | `company/gates/ANA_SCRIPT_FREEZE{,_v2,_v3}.json`; v3 = `59c83c86…` |
| VALIDATION check-only PASS | 398 | `runs_ds3/ANA/E_CHECK_PASS_v3.json` (value_read false) |
| **The single VALIDATION read** | 400 | `runs_ds3/ANA/out_val_v3/` → local `results/VALIDATION_v3/` (RETENTION_RESULT `0d765880…`) |
| **ALL_METHODS_LOCK** | 402 | `company/gates/ALL_METHODS_LOCK.json` (`04a3cd19…`) |
| TEST manifests built (counts only) | 403, 406–407 | `company/gates/test_build_v3/` (TEST_EXT `bfaa2118…`) |
| PREG2 + P2_CONTRASTS_v3 | 411, 427 | `contracts/PREG2.md`; `company/gates/P2_CONTRASTS_v3.json` (`673b547d…`) |

## 4. The single VALIDATION read and the three freezes

The end-game ran D → E → F → G → H with scripts in `company/plans/ops_scripts/endgame/` (server copies `runs_ds3/queues/endgame/`),
reviewed by Fable (GO-WITH-FIXES, then GO after FIX-1..FIX-5; `company/reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md` §1–§4).
The owner authorised Ops to run F / G / H without waiting for him, on condition of a Fable GO and Ops' own checks (DL 384).

- **D (freeze).** First run refused by a script bug (a one-line importlib fix, DL 387). Re-run → freeze **v1** `03028475…`
  (DL 388).
- **E (check-only, value-free) refused twice**, both times on analysis-side definitions that could never pass on real data:
  CALIB manifest users vs E2E users (DL 389) and the S-record `n_norms` field (DL 394). Each refusal was diagnosed from code
  only, ruled on by Fable (§5, §6 of the review file), fixed by a small packet (ANA-07, ANA-08) with an independent review,
  deployed as a new tools bundle (tools_r8, tools_r9) and followed by a **new freeze** (v2 `d96116d8…`, v3 `59c83c86…`).
  For the second fix Fable permitted a value-free "pre-freeze dry check" of the candidate tool on the read-only block, so
  that every remaining refusal would surface in one cycle; it passed (DL 396). Freezes v1 and v2 are SUPERSEDED and were
  never read. Details in file 02 §6.
- **E v3 PASS** (DL 398) → owner informed → **F, the single §9 VALIDATION read** (DL 399–400), plus the registered
  copy-only descriptive metric table (DL 385).
- From that moment **everything new is POST_VALIDATION_EXPLORATORY**: no re-run, re-selection or recipe change for Phase 1
  (DL 400; ALL_METHODS_LOCK `post_validation_rule`).

## 5. ALL_METHODS_LOCK (`company/gates/ALL_METHODS_LOCK.json`, `04a3cd19…`, DL 402)

It copies the v4 contrasts verbatim and adds `all_methods_lock` with: the analysis-freeze sha (v3), retention_report
`9a3b2c2a…`, the VALIDATION result and block-index shas, 131 arms, 56 sealed FL checkpoints, the code trees r2–r7b with
aggregates, the warm selection files and S records, `fa_warm_star` = S8, the N33 mapping, `exclusions_by_name` (10, e.g. the
STOPPED_REALLOCATED dirs, `runs_ds3/FL/`, WARM_PREVIEW, S7S8_T, smoke dirs), `declared_not_run` (9 items; file 01 §5),
`parked_smoke` (5 dirs, manifest `ec779fd4…`), the K16 pass-1 artifacts with `k16_latency = PASS2_PENDING`, the
`descriptive_metric_table` rule, and the **TEST plan**: 109 reevaluations (each locked checkpoint, PB + CB, on MAIN_TEST),
39 panel reevaluations (only where a VALIDATION panel dir exists; TRUE_COLD only for 6 admitted arms), 14 device runs (the
same locked procedures and bases as on VALIDATION), 6 baselines and the ONNX K16 step on tree `code_ds3r7b`.

## 6. The single TEST look

**State (MEASURED):** H `gate` rc 0 (DL 403); H `build` + `final` rc 0 — 5 RETRO_TEST manifests built, n_failed 0, no
model scored (DL 406); then **the owner put the TEST look ON HOLD** (DL 406) to solve the K21 weakness first.

**T0a (Fable gate §9; PREG2 §6.3).** Phase 1 and Phase 2 share ONE sitting on the RETRO_TEST manifests built under
ALL_METHODS_LOCK. Binding conditions:
- **S1** the Phase-1 lock and H plan run exactly as locked; Phase 2 adds `company/gates/P2_METHODS_LOCK.json`.
- **S2** P2_METHODS_LOCK and P2_SCRIPT_FREEZE are registered and on the server **before the first TEST reeval of any arm**;
  two `LOCKED_EVALUATION` register entries are appended first (one binding `04a3cd19…`, one binding the P2 lock sha). The
  frozen register tool admits two such entries (MEASURED, DL 411 E4).
- **S3** order: all Phase-1 scoring → all Phase-2 scoring → Phase-1 `analysis` → Phase-2 analysis; nothing under
  `runs_ds3/TEST`, `out_test_*` or `p2_out_test_*` is fetched or opened before both RETENTION_RESULT files exist; Ops writes
  the DL row first, fetches second.
- **S4** a Phase-2 arm without a sealed checkpoint at lock time is declared not run; **no second RETRO_TEST sitting, ever**.
- **S5** PRE, FA_WARM_FROZEN, FT_PRE and C_FULL TEST rows are produced once by the Phase-1 plan and hash-bound into both blocks.
- **S6** labels (§10); **S7** GPU precedence: Phase-2 lanes never share a GPU with H steps; Phase 2 before the ML-1M GPU move.
- **S8** no Phase-2 MAIN VALIDATION value is opened by a human before both sitting analyses are written.
- T0b (a Phase-1 look first) and a TEST2-only route were **refused** (Fable §9.1). TEST2 (December 2019) was an optional
  replication; the owner declined it (O2 = NO, DL 415).

**TEST_EXPOSURE_REGISTER** (`company/registers/TEST_EXPOSURE_REGISTER.json`, hash-chained, tool `tools/test_exposure_register.py`
`7d59e828…`, never edited). Current head after entry 6: `644f6efd…` (verify {6, []}, DL 407).

| # | Date (UTC) | Purpose / observations | What |
|---|---|---|---|
| 1 | 2026-09-26 | STRUCTURAL_COUNT / COUNTS_ONLY | G0 census of TEST-window user and row counts; no targets |
| 2 | 2026-09-26 | STRUCTURAL_COUNT / COUNTS_ONLY | a local schema check printed TEST status counts (incident, NO_FEEDBACK) |
| 3 | 2026-09-26 | HISTORICAL_PRIOR_EXPOSURE / METRICS | the v1 GRU and SASRec campaigns looked at the legacy TEST window: all V4 TEST results are retrospective |
| 4 | 2026-09-26 | STRUCTURAL_COUNT / COUNTS_ONLY | DATA-05 printed TEST counts of one bucket (incident, NO_FEEDBACK) |
| 5 | 2026-10-03 15:28Z | MANIFEST_BUILD / NONE | binds ALL_METHODS_LOCK `04a3cd19…`; written before any TEST manifest existed (DL 402) |
| 6 | 2026-10-03 15:43Z | INTEGRITY_CHECK / COUNTS_ONLY | the H build + final (manifest header `bfaa2118…`); no model scored (DL 407) |
| 7, 8 (PLAN) | before the sitting | LOCKED_EVALUATION | Phase-1 lock; P2_METHODS_LOCK (PREG2 S2) |

The endgame README's line "After H: register entry 6 (RESULT)" predates entry 6 (INTEGRITY_CHECK); the post-sitting
record takes the next free number (OPEN ITEM, file 13).

## 7. What may and may not be read

| Item | Rule |
|---|---|
| Phase-1 MAIN VALIDATION | read once (DL 400); now public inside the project; anything new on it is POST_VALIDATION_EXPLORATORY |
| RETRO_TEST (any arm, any study) | sealed until the T0a sitting; afterwards no re-run except an interrupted computation with identical hashes |
| Phase-2 MAIN VALIDATION | sealed until both sitting analyses are written (PREG2 S8); selection tools read CALIB rows only |
| Phase-2 CALIB screens | read only through `p2_select` after the stage's runs finish (mechanical) |
| Benchmark inner validation | read only by the runner's registered decisions (decide-warm, seed3, trackb-select) |
| Benchmark holdout (TEST) | sealed until that dataset's LOCK + single `--final`; afterwards counts first, values for the write-up |
| Run status | Ops reads only status / progress fields (`efe`, `round`, `status`), never a metric field; never `cat` a STATUS file that may carry a metric (DL 105) |
| UNSEEN_VAL selection-panel values | inside the selection files; deliberately not quoted (09-30 handover §11.1) |

## 8. How independent review was done (honest account)

- Every code change went through: a task packet with a closed file set (`company/tasks/`) → a worker that implemented it
  with tests and negative controls and wrote a handoff with before / after sha256 (`company/handoffs/`) → the operations
  lead re-ran the suites → an **independent read-only review** (`company/reviews/REV-*.md`) → fixes → (for blocking issues)
  a delta review → deployment in a new tree. Verdicts: ACCEPT / ACCEPT_WITH_FIXES / PASS / PASS_WITH_MINORS /
  PASS-WITH-FIXES / FAIL / REJECT.
- **Who reviewed.** Reviewers were separate AI agent instances (Claude Opus or Sonnet), started fresh with read-only
  access and a review packet, never the agent that wrote the code. Gate decisions, protocol rulings and audits were made by
  a Claude Fable 5.1 instance acting as chief scientist (files `company/reviews/FABLE_*`, `*_FABLE_RULING.md`); for a period
  the owner let Fable act as owner-proxy for blocking decisions, with hard limits kept by the owner (DL 37). Earlier,
  material disagreements went through independent Opus / Fable position memos and a synthesis (`*_POSITION.md`,
  `*_SCIENTIFIC_SYNTHESIS.md`). A 14-part "multi-expert" review (`company/reviews/MULTI_EXPERT_REVIEW_20260929/`; overall
  7/10, OPINION) was also produced by AI agents. **No human expert reviewed the code.** The owner read and approved the
  decisions. Say this plainly in the thesis AI-use statement.
- Reviews that failed and were fixed (examples): INT-02 REJECT (every production launch would have failed, DL 65);
  REV-DELTA-02 blocker (the ≥ 0.002 rule missing, DL 220); REV-BM-02 FAIL (1 blocker that would have lost a benchmark TEST
  look, DL 334); REV-P2-06 FAIL (wrong anchor threshold; every real P2 run would have been refused, DL 417).
- Phase-2 reviewer assignments (Fable §9.4): Opus worker + Opus review for P2-02, P2-04, P2-06, P2-09, P2-10, P2-11, P2-12;
  Opus worker + Sonnet review for P2-01, P2-03, P2-05, P2-13; Sonnet for P2-07, P2-14 and the lock tool. Fable touches
  exactly five points: the PREG2 delta review (done, DL 411), P2_METHODS_LOCK, P2_SCRIPT_FREEZE, the DP-FTRL accountant, and
  the TEST2 spec (moot: O2 = NO).

## 9. Owner decisions (chronological, with rows)

| DL | Decision |
|---|---|
| 1–3 | Adopt V4 (one backbone, winner-only downstream); the "company" operating model; local commits only by the operations lead, no push |
| 31, 35–36, 47–48 | 5–100 population; unseen-user panels; USER_HISTORY_L50; session-offset and SESSION_INTENT channels |
| 37 | Fable acts as owner-proxy on blockers while the owner is away (hard limits kept) |
| 60 | Model size inside the contest search; full 6 EFE; no legacy anchoring |
| 83, 86 | Downstream = big server model / FL with weights / small same-architecture device model; the question is "how much is retained", no threshold |
| RESUME §0 | "All training on MAIN"; Codex revoked (2026-09-28); use the whole server |
| 141, 161, 168 | Move downstream to the A40s; GPU 0 first, then both GPUs |
| 273 | Pretraining option 1: PRE on 1.2 M separate opted-in users; warm line compared with C_FULL |
| 276 | A larger server model is future work only |
| 282 | **Pretrained line primary**, random-start line afterwards (06N) |
| 286 | A descriptive C_FULL picture on non-MAIN users (06O) |
| 296, 302 | External benchmarks; methods = the REES46 recipe only |
| 316 | D1–D4 and amendment-07 options O1–O5 delegated to Ops + Fable jointly |
| 384 | Authorisation to run D → E → F → G → H without waiting, under Fable GO + Ops checks |
| 401 | ML-1M Track B runs to completion: **B13 rescinded by B15** |
| 404, 433 | ML-1M moved to the A40 GPU (**B16**) after a Fable re-review requested by the owner |
| 405–406 | "Solve every weakness by design and code, run nothing now"; **TEST look ON HOLD** |
| 410 (Fable) | PREG2 decisions O1 = **T0a**, O9 = **YES** (second study on the built TEST manifests), O4 grids / ties / futility / cap accepted with edits, O7 DP T = 384 + registered z, O5 (second θ_pre seed) = NO, O6 (second dataset) = NO |
| 415 | **O2 = NO** (no December 2019 data, no TEST2); **O8 devices** = the owner's laptop + headless Edge versions on the server (SERVER_BROWSER) |
| 415 → 424 → 427 | **O3 (run C2): YES → NO → YES (final)**, all before any Phase-2 run; contrasts v1 → v2 → v3; disclose the history |
| 435 | Push the code to GitHub under the owner's identity (separate task); build this handover; keep ML-1M GPU and Phase 2 running for the final handover after TEST |

## 10. Labels

**Evidence labels:** MEASURED (read from a record), REPORTED (someone else's statement, not re-checked), INFERENCE,
PLAN, PENDING, TO-VERIFY (Ops checks before the step it gates).

**Result labels:**
- CONFIRMATORY — a registered contrast under its registered rule and seeds (REPORT §1, bold rows).
- DESCRIPTIVE — everything else (panels, curves, strata, per-seed values, the descriptive metric table, 06O previews).
- POST_VALIDATION_EXPLORATORY — anything added after the 2026-10-03 VALIDATION read (e.g. D1, Phase-2 MAIN VALIDATION values).
- CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) — Phase-2 Q1–Q8 on RETRO_TEST, always printed with disclosure D-a.
- TERTIARY_P2 — Phase-2 Q9–Q14.
- REPLICATED_FRESH / NOT_REPLICATED / TEST2_NOT_AVAILABLE — TEST2 (now always TEST2_NOT_AVAILABLE).

**Run / file status labels:** SMOKE_NOT_EVIDENCE; STOPPED_REALLOCATED; SUPERSEDED_LR_SCREEN_EXTENSION; SUPERSEDED_DEVICE;
SUPERSEDED (freezes v1 / v2); INCOMPLETE (benchmarks, by DL row only); HISTORICAL; `preview_reference NON_MAIN_UNSEEN_VAL`.
**Claim keys:** ABOVE / BELOW / INCLUDES (whether the CI lies above, below or includes the reference); flags SEED_SENSITIVE,
BELOW_DEVICE_HEURISTIC, NO_MEASURABLE_CALIB_GAIN, EDGE_UNRESOLVED, FUTILITY_SKIP_WB_WC.

## 11. Record keeping

- One writer at a time on the decision log; a new row = the tail's max number + 1; time stamps from the server clock or
  `date -u` (local time is UTC+3).
- After every event: a DL row; at milestones: a RESUME_HERE top block and a thesis-reference capture (file 08 §9).
- Keep every checkpoint, receipt, log and failed attempt. Deletion only with the owner's approval and a verified sha256
  manifest (DL 19, 44, 46). Probes and smokes delete their own weights and keep digests (DL 30).
- No git push or remote mutation without the owner's explicit approval; before any commit verify your own identity
  (`git var GIT_AUTHOR_IDENT`) and never commit under someone else's name.

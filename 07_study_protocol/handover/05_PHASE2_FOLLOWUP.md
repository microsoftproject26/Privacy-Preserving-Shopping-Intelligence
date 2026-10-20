# 05 — Phase 2: the registered follow-up on K21 (PREG2)

*Primary texts (read them, this file only guides): `HANDOVER/reference/PREG2.md` (= `contracts/PREG2.md`, REGISTERED DL 411,
sha `be0561abe5ce…`), `HANDOVER/reference/PHASE2_IMPLEMENTATION_PLAN.md` (work packets, costs, dependency graph),
`HANDOVER/reference/P2_CONTRASTS_v3.json` (registered contrasts, DL 427), and `HANDOVER/reference/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md`
§9–§10 (the gate rulings that shaped PREG2). Status below is as of DL 435 (2026-10-03).*

## 1. Why Phase 2 exists

- The VALIDATION read gave K21 = FA_WARM_FROZEN − PRE = −0.000433 [−0.000758, −0.000101] (DL 400): federated rounds did not
  add anything over the shipped pretrained model.
- The owner asked whether the warm FL had been done badly (DL 405). Two facts (PREG2 §1, MEASURED): the S7 / S8 screen only
  moved the client LR **upward** (factors {1, 4} for S8) on top of a **cold-start** recipe and its winner sat at the
  **smallest server LR** (a grid edge); and D1 showed the warm FL moved θ by 0.5–0.7 % (S8) — a near no-op (DL 407).
- Owner decision: solve the weaknesses by design and code first, **run nothing until the design is reviewed**, and hold the
  TEST look so that Phase 1 and Phase 2 share one sitting (DL 405–406).
- Phase-1 claims are untouched: K20b / K21 stay "under the registered recipe" (PREG2 §0 F1). Phase 2 is a separate study with
  its own prefix `P2_`, contrast ids `Q…`, lock, freeze and bootstrap seed **20261004** (F2).

**Question (PREG2 §2):** once a pretrained model is shipped, does federated training on the users' private data add quality,
and at what communication / privacy / device cost? INFERENCE in PREG2 §1: the plausible ceiling is about +0.001 MRR@20
(log-linear scaling of the data effect), close to the minimum detectable effect ≈ 0.00047; **no positive result is promised**
(disclosure D-h).

## 2. Design in brief (PREG2 sections in brackets)

- **Firewall (§0).** Selection on CALIB only; Phase-1 code paths byte-identical with every new option off; frozen tools
  imported, never edited; Phase-1 checkpoints read-only.
- **No-op definition (§3.2)** on the trained subset T: < 1 % relative distance and no large tensor group moved ≥ 1 %, plus a
  value-free functional probe (top-20 Jaccard ≥ 0.95). **D1-ext (§3.3):** every Phase-2 FL run logs per-round update norms
  (`UPDATE_NORM_LOG.jsonl`); bit-identical with the observer on / off; DP runs log post-noise quantities only.
- **Controls (§4).** C0 = θ_pre itself (server LR 0 gives exactly θ_pre); **C1** = central fine-tune of θ_pre on MAIN,
  FULL and FROZEN variants, LR screen {1e-5, 3e-5, 1e-4, 3e-4} on CALIB, then 2 seeds on MAIN — "how much MAIN data can add
  when privacy is ignored"; **C2** = `P2_C2_CFULL_SMALL`, the D64 model trained centrally on PRE + MAIN users (matched size and
  recipe; removes the size confound of C_FULL vs PRE), seed 2026, **run (O3 = YES, DL 427)**.
- **W-grid (§5).** Fixed: θ_pre start, 64 clients / round, sweep, 10 % drop-out, 1 local pass, batch 16, FL_CONST, 6.0 EFE
  (CALIB 3,414 rounds; MAIN 13,654), mark 0 is never a PRACTICAL_BEST candidate.
  - **WA** (item tables frozen): server {FedAvg η 0.3 / 1.0, FedAvgM 0.03 / 0.1, FedAdam 1e-4 / 3e-4} × client SGD-m LR
    {0.003, 0.01, 0.03} = 18 runs, + WA-ext FedAdam {1e-5, 1e-4} at c 0.1 = 2 runs, + the incumbent (S8's FedAdam 1e-3,
    c 0.1) re-used only if the P2-T3 check reproduces run S8Tc's first 64 rounds bit-identically, else re-run.
  - **WB** at the WA winner A*: unfrozen tables with item-LR × 0.1 (WB1) / × 0.01 (WB2), dense; WB3 = × 0.1 with
    delta-compressed sparse item uploads (k = 3,170 rows) — WB3 reveals the set of item IDs a client touched (disclosure, DL 418).
  - **WC** (base frozen): LoRA r {4, 16} on qkv / out / ff1 / ff2 at client LR × 10 (WC1 / WC2) and × 1 (WC3 / WC4); WC5 = A*
    plus the PF personal vector.
  - **WD** (MAIN only, not selected): W*'s recipe under DP-FedAvg with the registered machinery — Poisson, m 1,024,
    **T = 384**, registered z (0.5642 / 1.3094), passes 2 / batch 16 for FA1024 and DP alike, S from a fresh S_CAL
    (`S_RECORD_P2.json`). DP-FTRL, SecAgg and LDP are Part B.
- **Selection (§5.6).** CALIB_VALIDATION E2E macro MRR@20 at PB; A* = argmax(WA ∪ WA-ext ∪ incumbent); W* = argmax({A*} ∪ WB ∪
  WC); **tie band 0.001** (≈ 3 paired SE at about 10,070 CALIB users) with preference: fewer upload bytes → fewer trained
  parameters → smaller effective server step → smaller client LR → label; flag NO_MEASURABLE_CALIB_GAIN if W* − PRE_CALIB
  < 0.001 (W* still goes to MAIN). Outputs write-once: `contracts/fl/P2_WARM_SELECTION_A.json`, `P2_WARM_SELECTION_FINAL.json`,
  `P2_C1_SELECTION.json`.
- **Edge rule (§5.7):** one outward extension per axis, ≤ 3 runs per stage, EDGE_UNRESOLVED disclosed if the winner is
  again at the edge. **Futility (§5.8):** if C1 and the best WA both gain < 0.001 over PRE_CALIB, WB / WC are skipped
  (caveat D-j: CALIB has 32,768 users, MAIN 131,072). **Cap (§5.9):** 250 GPU-h; drop order tertiary → exploration → C2 last;
  dropped = declared not run.
- **Part A / Part B (§2.1).** Part A (critical path): C0, C1, C2, WA–WC, W* ×2, FT_P2_WSTAR ×2, the DP chain; Q1–Q11. Part B
  (never delays the sitting): SecAgg simulation (Q13), DP-FTRL (Q12), LDP, privacy audits E1–E3 (membership inference,
  canaries, update leakage), devices (§8).
- **The sitting (T0a, §6.3 S1–S8):** see file 03 §6. Key points: P2_METHODS_LOCK + P2_SCRIPT_FREEZE + two LOCKED_EVALUATION
  register entries **before the first TEST reeval of any arm**; S3 order; S4 no second sitting; S5 PRE / FA_WARM_FROZEN /
  FT_PRE / C_FULL TEST rows produced once by Phase 1 and hash-bound into both blocks; **S8 Phase-2 MAIN VALIDATION values
  stay sealed** until both analyses are written.
- **Devices (§8, Part B).** O8 (DL 415): D1 = the owner's laptop (Lenovo 82RC, i7-12700H, 15.8 GB, RTX 3050 Ti, Windows 11,
  Edge 154.0.4258.53, MEASURED) + headless Edge versions on the server labelled **SERVER_BROWSER** (154.0.4258.53, 152.0.4191.66,
  150.0.4078.105, 146.0.3856.97; DL 416). No phone. Latency / cold start / memory / model bytes only; thresholds DESCRIPTIVE.

## 3. Contrasts (P2_CONTRASTS_v3, file `673b547d…`, self `e98580564b61…`)

| Id | Contrast | Tier | Rule / seeds |
|---|---|---|---|
| **Q1** | P2_WSTAR − PRE | **primary** (unadjusted) | PB; W* {2026, 2027} vs PRE s2026 |
| Q2 | P2_C1_FT_CENTRAL_FULL − PRE | secondary (Holm) | PB; 2 |
| Q3 | P2_WSTAR − P2_C1_FT_CENTRAL_FROZEN | secondary | PB; same-seed pairs |
| Q4 | P2_WSTAR − FA_WARM_FROZEN | secondary | PB; same-seed pairs |
| Q5 | FT_P2_WSTAR − FT_PRE | secondary | PB; 2 |
| Q6 | R(P2_WSTAR_DP8 / P2_WSTAR_FA1024) | secondary | CB; 2 |
| Q7 | P2_WSTAR_DP8 − PRE | secondary | CB; 2 |
| Q8 | P2_C2_CFULL_SMALL − PRE | secondary | PB; 2026 |
| Q9 | R(P2_WSTAR / C_FULL) | tertiary | PB |
| Q10 | Q1 / Q2 (headroom capture) | derived, DESCRIPTIVE | — |
| Q11 | R(P2_WSTAR_DP1 / P2_WSTAR_FA1024) | tertiary | CB; 2026 |
| Q12a / Q12b | DP-FTRL ratio / DP-FTRL − PRE | tertiary (Part B) | CB; 2026 |
| Q13 | R(P2_WSTAR_SECAGG / P2_WSTAR), TOST [99, 101] | tertiary (Part B) | 2026 |
| Q14 | P2_WSTAR − PRE on a TEST2 panel | tertiary | **TEST2_NOT_AVAILABLE** (O2 = NO) |

Holm family Q2–Q8. **Reading of Q1 (§9.4):** SUCCESS = 95 % CI lower bound > 0 and both seeds > 0; EQUIVALENT = 90 % CI
inside ±0.0005 (TOST); FAILURE = upper bound < 0; else INCONCLUSIVE. Interpretation grid with Q2 (FL captures headroom /
FL recipe fails / MAIN data adds nothing at D64 / review flag). Labels: Q1–Q8 **CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP)**
printed with disclosure D-a verbatim; Q9–Q14 TERTIARY_P2; Phase-2 MAIN VALIDATION values POST_VALIDATION_EXPLORATORY.

**O3 history (disclose):** DL 415 YES (v1 kept) → DL 424 NO (v2 `4194d937…` registered, Q8 removed) → DL 427 YES (v3, content
identical to v1), all before any Phase-2 run and value-free. An earlier v2 draft that was never registered is kept as
`company/gates/_withdrawn_P2_CONTRASTS_v2_never_registered.json`. **P2_METHODS_LOCK must bind v3.**

## 4. Code status (local shas MEASURED 2026-10-03; deployed tree `code_p2r1`, DL 435)

| Packet | Files (sha prefix) | Review | State |
|---|---|---|---|
| P2-00 governance | PREG2 registered (DL 411); P2_CONTRASTS v1 → v2 → v3 | Fable delta GO (§10) | done; **lock tool `tools/p2_lock.py` NOT written** (PENDING) |
| P2-01 + P2-02 D1-ext + recipe plumbing (`P2_W` arm) | `code/v4int/launch_fl.py` `f4e56b80`; `code/fl/update_log.py` `42449044`; `code/v4int/p2_probe.py` `c25bd351`; `tools/p2_noop_check.py` `4074e4e8`; `tools/p2_t3_check.py` `921320e9`; handoff `company/handoffs/P2-01_02.md` | REV-P2-01_02 PASS_WITH_MINORS (DL 433) + delta PASS (DL 435) | deployed in code_p2r1; 56 Phase-1 (arm spec, RunConfig) pairs and 2,548 configurations byte-identical |
| P2-03 sparse item uploads (WB3) | `code/fl/sparse_rows.py` `0d8235e2`; handoff `P2-03_04.md` | REV-P2-03_04 PASS_WITH_MINORS (DL 418), minors closed (DL 429) | module done; **not wired into launch_fl** (`P2_UPLOADS = ("dense",)`; HOOK 2 pending) |
| P2-04 LoRA (WC1–WC4) | `code/fl/peft.py` `bc1012b0` | same review | module done; **launch_fl refuses LoRA cells** ("work packet P2-04"; HOOK 1 pending) |
| P2-05 C1 / C2 controls | `code/v4int/central_ft.py` `4d95d12c`; `code/v4int/c_full_small.py` `c9960d4c`; handoff `P2-05.md` | REV-P2-05 PASS_WITH_MINORS (DL 425); minor fixed (DL 426) | deployed |
| P2-06 selection tool | `code/v4int/p2_select.py` `7a05afcd`; handoff `P2-06.md` | REV-P2-06 **FAIL** (DL 417) → delta PASS_WITH_MINORS (DL 428) → closed by Ops diff check (DL 431) | deployed |
| P2-07 + P2-13 anchor PRE_CALIB + scoring of P2 checkpoints (`v4int/reeval_p2.py`) | not in `code/` yet | — | dispatched (DL 434), **PENDING** |
| P2-08a DP chain for W* | — (`launch_fl` refuses `--epsilon` / `--clip-record` for P2) | — | **not dispatched** (PENDING) |
| P2-08b / P2-09 / P2-10 (Part B: DP-FTRL + LDP, SecAgg, audits) | — | — | not dispatched (optional; declared not run if unfinished) |
| P2-11 TEST2 | — | — | not needed (O2 = NO) |
| P2-12 analysis (`tools/p2_report.py`, `tools/p2_block_index.py`) | not in `tools/` yet | Opus review + Fable freeze check | dispatched (DL 434), **PENDING** |
| P2-14 devices | `code/v4dev/browser/p2_bench.html`, `p2_bench.js`, `serve_coi.py`, `README_P2_DEVICE.md`; `tools/p2_edge_server.sh` `6f16d440`; `tools/p2_packages_sha.py` `9b6b38e5` | REV-P2-14 PASS_WITH_MINORS (DL 419); fixes done (DL 420) | done; server Edge probe works (DL 420); no benchmark run |
| P2-RUN1 lane | `company/plans/ops_scripts/p2/laneP2_RUN1.sh` `e0289882`; test `code/tests_int/test_p2_run1_lane.py` | **no review recorded** | written (dispatched DL 434), **PENDING review**, not launched |
| FT_P2_WSTAR launch path | `code/v4int/launch_device.py` has no P2 support (MEASURED grep) | — | **gap: no packet assigned** (file 13) |

`code_p2r1` (DL 435): tar `54fbd95b…`, 710 files, aggregate `4268be26…`; change set vs r7b = 4 core files (fl/client.py,
fl/runtime.py, fl/server.py, v4int/launch_fl.py) + new P2 modules and tests; tools_p2r1 `e79bfeff…`; frozen tools and the
evaluator unchanged. Any further code (reeval_p2, p2_report, hooks, P2-08a, FT path, p2_lock) needs a **new tree**
(e.g. `code_p2r2`) with the same byte-identity checks, because `fl_code_sha256` changes with every fl/ edit.

## 5. What has run, what remains

**Has run (MEASURED):** nothing of the Phase-2 W / C runs. Only: D1 (exploratory, DL 407), the P2 code's local test suites,
the server Edge probe (no model, DL 420), and the deploy checks of code_p2r1 (DL 435).

**Remains (PLAN; ordered runbook in file 08 §2):**
1. Finish Part-A code: P2-07/13, P2-12, P2-RUN1 review, WB3 + LoRA wiring, P2-08a, FT_P2_WSTAR path, P2-00 lock tool;
   reviews per Fable §9.4; deploy a new tree.
2. RUN-1 (lane): S0 firewall checks → P2-T3 → WA (20 + conditional incumbent) → C1 CALIB screen (8) → C2 MAIN (1); plus the
   P2-07 anchor PRE_CALIB.
3. Select A* (+ ≤ 3 edge runs) and C1; futility check; RUN-2 WB (3) + WC (5) unless FUTILITY_SKIP_WB_WC; select W*.
4. RUN-3 MAIN: W* ×2, C1 FULL / FROZEN ×2, S_CAL(W*) → FA1024 ×2 → DP8 ×2 → DP1; then FT_P2_WSTAR ×2.
5. Part B as time allows; device measurements on the exported W* (PREG2 §8; `README_P2_DEVICE.md`).
6. P2_METHODS_LOCK + P2_SCRIPT_FREEZE (Fable touch points 2–3) → register entries 7 and 8 → the single sitting.

**Estimates (INFERENCE):** about 190 GPU-h for Part A + B (cap 250); code 3–4 days, RUN-1 ≈ 1.5 days, RUN-2 ≈ 1 day, RUN-3
≈ 2–3 days, FT ≈ 1 day; sitting ≈ 2026-10-17 … 10-21 (PHASE2_IMPLEMENTATION_PLAN §4).

## 6. Open questions recorded in the handoffs (resolve before the stage that needs them)

- `p2_probe` is untested on real data; run it once on a finished CALIB run first (P2-01_02 §8 Q1).
- Criterion (b) "|Δscore| of the top-1 item" is implemented with θ_pre's top-1 (Q2); LoRA no-op measurement on merged weights
  (Q4); whether "clients aggregated" may be logged for DP runs (Q5) — to settle at P2_SCRIPT_FREEZE.
- The 265,162 vs measured 263,822 uploaded-parameter count of S8 (registered "to be re-measured").
- `runs_p2` on the server already holds the 2026-09-27 contest P2 tuning runs, while the lane calls it a "fresh root" — check
  for collisions before launch (file 13).

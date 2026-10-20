# G0(b) protocol approval — Fable position (verbatim; claude-fable-5-1; sasrec-fable-reviewer, read-only)

Packet: `G0B_PROTOCOL_PACKET.md`. The Opus memo `G0B_PROTOCOL_OPUS_POSITION.md` was withheld. AgentId ae62475f42758d97d.

ROUTING: claude-fable-5-1

**VERDICT: APPROVE_WITH_FIXES** — freeze Clean V2 and write DATA_PROTOCOL_LOCK once the four lock-text fixes in BLOCKERS(A) are applied. No new science, no re-measurement needed. Coverage disclosure: I read all four contracts, the schema (rows 1–37 in full, rows 38–59 and the normalizer/re-zero/OI blocks by targeted extraction), all seven blocker receipts, the four QA-02 receipts, FL-01 plus its TEST_RUN/LO receipts and the grepped assertion lines of the T26/T27–29/T28/T34 tests and `aggregate.py`/`client.py`/`personal.py`; EXECUTED_RECIPES via its .md transcript, not the JSON. No TEST outcome appears anywhere I read; TEST *counts* appear only in COLD_STRATA L19-28 and BOUNDARY_DEFINITIONS L164, which the contract correctly registers as STRUCTURAL_COUNT.

## DECISIONS

**C1 — Contracts: APPROVE_WITH_FIXES.**
- PROJECT_MEASURED: every DROP_REDUNDANT column has a code identity plus 0 contradictions on TRAIN/VS (DUP) and 0 on merged VALIDATION incl. the 3,427,555 VC rows (`g0/receipts_qa02/OI06_RC_VALIDATION.json`, RC1–RC6 all 0; dead flags 0 non-zero on VC); M1 refit reproduced with abs diff 0.0 (`OI04_M1_REPRODUCTION.json`); 73/73 input hashes match (`OI07_FILE_INTEGRITY.json`); code_hash matches the freeze (`CODE_AND_FLAG_CHECKS.json` L4-9).
- The contracts implement G0 decisions 1–5 and 8 as written: cohort/hashes (CLIENT C2.4), s=5/K=158,486, merged VALIDATION with descriptive subperiods (E1.5), cold strata split (C3.2), FP32+TF32-off in the comparison key (§9), tie rule quoted (E6.1), full-K candidates, user bootstrap 1000/20260926 with shared W and direct ratio bootstrap (E7.2–7.5), macro identity with T36 counterexamples (E4.6), TEST gate before ALL_METHODS_LOCK (E10.1). INFERENCE: the censoring/OOV rules are correct and exhaustive (§2.2 consistency sums verified against BOUNDARY_DEFINITIONS L159-165).
- Fixes (lock text only): (a) EVALUATION §11.4 still cites v1 counts 27/26, `log1p_prefix_elapsed_s` and `user_context_masks[6]` — must be the V2 table (25/22, masks[4]); (b) CLIENT §6 item 4 says "19 clip/z-scored channels" — V2 has 17 normalized channels plus the F-1 refit; (c) P12 below; (d) F-1 values into schema `normalized_channels_v2` and `rezero_map_v2` (schema version bump, self-hash).

**C2 — F-1: ADOPT, inside scope.** PROJECT_MEASURED: 117,315/52,760,314 TRAIN rows (0.22%) and 64,201 VALIDATION rows carry a raw 0 that v1 encodes as z=−0.00207 (`OI03_F1_PRICE_REZERO.json`). Refit on valid rows: mean 0.002065693626402541, std 0.9945344391661521, n_valid 48,434,230, recorded clip bounds. Consistency (INFERENCE, arithmetic): 48,434,230+117,315 = 48,551,545 ≤ 48,555,935 has-history rows, and shrinking the variance by (1−117,315/48,551,545) reproduces the old std 0.99333 exactly — the receipt is internally coherent. It is the same defect class as M1 (unknown ≠ 0 after normalization; PKG 03 §6 L35 is literal) and G0 decision 4 said "03 §7 applied literally". Rule: extend the re-zero gate (recent_price_supported=0 OR price_missing OR price_conflict at last_valid_index) AND use the valid-only refit. Add disclosure B7′ (n_valid) and update the T09 negative control (a fit including the 117,315 zeros must reproduce 0.99333 and be rejected).

**C3 — P3 stopping rule (binding).** No early stop before 6.0 EFE for either family; the "3.0-EFE reference" in G0 D6 is void. Evaluate every 0.5 EFE on M02 (M06 for calibration); save strictly-best (Δ > 0.0002 over the running best) and `endpoint_6.0`. PRACTICAL_BEST = strictly-best in [0.5, 6.0]; CONTROLLED_BUDGET = endpoint 6.0. Family selection and every matched block use [0, 6] values only. At 6.0 the run is STILL_IMPROVING iff the strictly-best checkpoint is at 5.5 or 6.0 (patience 2); only then, and only with a pre-recorded budget, may it continue from the 6.0 resumable to 8.0 (LR 0.1×peak on (6.0, 7.5], linear to 0.01 at 8.0; stop after 2 non-improving evaluations). Continuation rows are labelled PRACTICAL_BEST_EXTENDED, secondary; the resumable is deleted at 6.0 unless STILL_IMPROVING (disk plan rule 2).

**C4 — P11: Candidate A.** M04 = the 16,384 M03 users with the smallest population keys (all cold; stratum mix and FIXED_50/FIXED_20/ALL_AVAILABLE eligibility reported, never used to select). INFERENCE: B qualifies on ≥50 VALIDATION rows, i.e. selection on evaluation-period activity, and 84% of it is warm — it measures truncation of trained history, not cold start. Add a separately labelled WARM_TRUNCATION panel = the 16,384 smallest-key M02 users (nested, no new salt) for T41 and the PKG 09 §5 L27 warm-p_u evaluation. B's receipt is retained as historical only.

**C5 — P12: move the 140 AB VS→VC rows to CEN_AMBIGUOUS_NEXT_BARRIER.** PROJECT_MEASURED: the CASE assigns AB before the session/split tests (`pipeline5.py:122-132`, BOUNDARY_DEFINITIONS L49-60), so they are ambiguous regardless of the fence. CEN_LEGACY_BARRIER = 170 BU rows only; this corrects my G0 "310" wording. Totals unchanged; precedence unchanged.

**C6 — P4/P5/P6.** C_ref(seed, panel, sub-panel) = central C, same seed, PRACTICAL_BEST checkpoint, CONTEXT_ONLY, same queries, full available prior history (per-h for ALL_AVAILABLE); events-to-X% = smallest registered h with M(h) ≥ X·C_ref, no interpolation; also report M's own full-history value. M04S = M03 users at population-key ranks 16,384–32,767; the recipe grid (p_u-only for PF/CPF, full local model for LO, passes {1,2,4}, locked personal LR and λ) is registered at ARCHITECTURE_LOCK and selected once on M04S FIXED_20 before any M04 outcome. Confirm h=0 = query event only (context length 1), FIXED_20 added, one query per user per sub-panel.

**C7 — FL-01 choices: all four ACCEPTED.** (1) init recentering: register the *recentered* θ0 as the per-seed I_META_RANDOM init for every method incl. C and LO (SASRec only), so the init hash is one tensor; PROJECT_MEASURED: fixed-query scores shift by a per-query constant, top-20 and CE equal (`test_recenter`, 4 negative controls). (2) separate p clip 1.0 — it is what makes PF(lr_p=0, λ=0) ≡ FA bitwise (`test_t27_pf_zero_lr_zero_lambda_equals_fa`). (3) CPF penalty over distinct users (`personal.py:114-115`) matches PF's once-per-step penalty. (4) n_shards = 2 frozen; `combine_shards` sums in shard-index order (`aggregate.py:100-113`), T34 verified bitwise for 2 and 4 shards on two spawned PIDs. Also frozen: max_attempts 2; prox gradient before clip.

**C8 — Other lock items.** Blockers 1–5 closed (blocker 1 conditionally: server-only, contest fits; owner decision needed only if SASRec wins). Blocker 2: transcribe AdamW fused, betas (0.9, 0.999), eps 1e-8, wd 1e-5 on ndim≥2, clip 1.0, peak 1e-3 both families, B=256, GRU dropout 0.20/0.10, SASRec 0.2 untied head with per-step recentering, label smoothing 0 (`EXECUTED_RECIPES.md`). Blockers 6–7: closed on CPU — 67 passed / 26 strict xfails all FAILED_AS_INTENDED (`TEST_RUN.json`), μ=0 ≡ FA bitwise (`test_t26` L43), p=0 bitwise for both families (L34, L96), reused-worker digest equality (`test_t28` L70), LO 1,024 clients REPRODUCED_EXACTLY across interpreters/orders. LIMITATION: synthetic data, K=64 families, one full-K visit; the GPU receipt is a G1 blocker.

## LOCK_VALUES delta
- FEATURES: Clean V2 = 53 channels (numeric 15 / flags 15 / uctx 12 / masks 4); 6 DROP_REDUNDANT, 0 DROP_SPEC_DEAD, 2 RE_ENCODE_M1, F-1 re-zero + refit (values in C2).
- CENSOR: LEGACY_BARRIER 170; AMBIGUOUS_NEXT_BARRIER 4,489; RIGHT_EDGE 152; NO_NEXT 3,064,778.
- CENTRAL: executed values as C8; stopping rule as C3.
- COLD: M04 = candidate A; WARM_TRUNCATION 16,384 (M02 prefix); M04S ranks 16,384–32,767 of M03; h=0/FIXED_20/one-query confirmed; C_ref as C6.
- FL: n_shards 2; separate p clip 1.0; CPF distinct-user penalty; max_attempts 2; per-seed θ0 = recentered init for all methods (SASRec).

## BLOCKERS
A. Before writing the lock: fixes C1(a)–(d) and C5; transcribe C8 values.
B. Before G1/G2 (not the lock): GPU determinism receipt (T34 + LO replay on 2×A16, `use_deterministic_algorithms`, TF32 off); OI-05; FL-02 participation plan and s(EFE) hook; disk guard arithmetic — with 12.81 GiB free the "free − peak < 10 GiB refuses" rule (`DISK_RETENTION_PLAN.md` rule 5) blocks the planned concurrent contest (12.81 − 3.4 = 9.4): reset the refuse threshold to the 8 GiB floor plus temp and keep 10 GiB as the pause target; obtain the owner's conditional pre-approval of deleting `gru_v1/cache/` now.

## OPEN_DISAGREEMENT_RISKS
- A vs B for M04: A gives small FIXED_50 support; ALL_AVAILABLE and FIXED_20 carry the study.
- F-1 is a protocol extension beyond the literal M1 wording; numerically ≈0.1% std change.
- Relabelling the 140 AB rows changes my G0 "310" figure to 170.
- Registering a recentered θ0 for C/LO touches the executed C recipe's init gauge (prediction-invariant; only weight decay on the gauge component differs).
- Six DEAD_IN_TRAIN flags keep untrained weight columns; RC9 discloses.
- No early stop before 6.0 spends the full budget on a converged run.

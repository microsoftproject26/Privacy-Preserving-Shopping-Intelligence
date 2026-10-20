# Pre-G1 deliberations — Fable position (verbatim; claude-fable-5-1; sasrec-fable-reviewer, read-only; agentId adf3823511e6808a7)

Packet: `PREG1_DELIBERATION_PACKET.md`. The Opus memo `PREG1_OPUS_POSITION.md` was withheld until this answer.

ROUTING: claude-fable-5-1

**VERDICT: DECIDED — lock T1–T3 below before any central result.** I verified the code facts: `code\v4train\schedule.py` L12–14 and L87–91 read s at the END of the update's exposures; `code\fl\schedule.py` L14–29 and L76–107 implement both points with `DEFAULT_EXPOSURE_POINT = "start"` (L40); `code\v4eval\ranking.py` L13–19, L33, L125–152 register `score_block`, gather s_t from its own block and refuse a non-reproducing kernel. G1 stays blocked until the pins named under each topic exist. No TEST table was read.

## T1 — Full-TRAIN tie-break trigger

**DECISION.** The cohort contest stays the selection instrument (G0 D1, G0B C3). Compute, from the four cohort rows only, Δ̄ = mean_GRU − mean_SASRec of PRACTICAL_BEST client-macro MRR@20 on M02, per-seed δ_s, within-family seed gaps g_f. TRIGGER the full-TRAIN tie-break iff ANY holds:
- T1a |Δ̄| < 0.0020;
- T1b sign(δ_2026) ≠ sign(δ_2027), or any |δ_s| ≤ 0.0002;
- T1c |Δ̄| < 2·max(|g_GRU|, |g_SASRec|);
- T1d the paired user bootstrap 95% CI of Δ̄ (1000 resamples, seed 20260926, shared W) contains 0.
Otherwise DECISIVE_STABLE: select the higher mean, skip the tie-break.

**Rationale.** PROJECT_MEASURED: v1 full-data two-seed gaps were ≤ 0.000869 (packet §Facts, living reference §5.6e); tolerance 0.0002 is the registered resolution (schedule.py L40). INFERENCE: cohort seed variance is unknown and likely larger (9.7% of examples), so an absolute floor (10× tolerance, ≈2.3× the largest v1 gap, above PKG 05 §B's "0.001 is not a tie") plus a self-scaling seed-spread rule (T1c) plus sign consistency is the minimal predeclarable set. T1d is reported evidence in every case; with 62,368 users it will rarely be the only trigger. PKG 05 §B: two seeds show replication, not variance — hence OR logic, not a formal test.

**Tie-break LOCK_VALUES.** Population = full TRAIN comparison union (2,694,998 clients, 40,180,608 examples, N for EFE = 40,180,608); families both; seeds 2026 and 2027; recipe identical to the cohort contest (cohort-calibrated peak LR, same schedule, exposure point "end", 6.0 EFE, stopping rule C3, no continuation, no new calibration) so population is the only changed factor; evaluation every 0.5 EFE on the full warm VALIDATION manifest (642,228 users, 8,430,391 rows; registered and hashed before launch, M02 reported as a subset descriptively); same frozen evaluator (T3); budget cap 125 GPU-h (PROJECT_MEASURED-derived: 2×(14.70+37.36) = 104.1 h training, COST_FORECAST L21/L101, plus ≈10 h evaluation, INFERENCE); checkpoints strictly-best + 6.0 only, deleted after ARCHITECTURE_LOCK.

**Decision rule.** Winner = higher full-data seed-mean macro MRR@20 iff |Δ̄_full| ≥ 0.0020 and both full per-seed deltas share its sign. If inconclusive: if all four per-seed deltas (two cohort, two full) share one sign with |δ| > 0.0002, select that family (CONSISTENT_SMALL_ADVANTAGE); else ARCHITECTURE_EQUIVALENT_WITHIN_RESOLUTION → select GRU on cost (downstream core 88 vs 263 GPU-h, COST_FORECAST L89/L169), never on export feasibility (PKG 05 §D). This fallback needs the owner's ratification as the amendment PKG 05 §B requires.

**Relation to C.** Downstream C = the selected family's cohort contest runs, reused verbatim (PRACTICAL_BEST and CONTROLLED_BUDGET rows). Full-TRAIN runs are ARCHITECTURE_TIEBREAK rows, population ALL, comparison_status HISTORICAL_REFERENCE; they never enter a matched block, never become C_MATCHED, never seed a downstream run. If the tie-break reverses the cohort order, ARCHITECTURE_LOCK records selection_basis = FULL_TRAIN_TIEBREAK and discloses the cohort deficit (bounded by T1a/T1c).

## T2 — LR exposure point

**DECISION: `exposure_point = "end"` for every method.** The LR of a unit of work is peak × s(EFE at the end of the exposures it consumes): central update (a, a+n]/N (already coded); FA/FP/PF round r = lr_peak_local × s((E_r + n_r)/N) with n_r the planned n_consumed, constant across the round's local steps; PF personal LR = shared peak × the same round factor; LO step t of T = lr_peak × s(6(t+1)/T).

**Rationale (INFERENCE from PROJECT_MEASURED code).** s is defined on exposures consumed (PKG 08 §1), so the matched reading is the 1-based ordinal already used by central. Under "start": FA/FP/PF round 0 runs at LR 0 with ~1,900 counted exposures (≈1/41 of the warmup; 4,096 rounds per visit, 2 passes → ~2,048 rounds/EFE); LO's median client (T = 6) gets s = 0, 1, 1, 0.55, 0.1, 0.1 — one of six updates wasted, AdamW moments initialised from a zero-LR step, and the cooldown floor never reached. Under "end" it gets 1, 1, 0.55, 0.1, 0.1, 0.01: same shape, same endpoint 0.01·peak as central. LITERATURE_SUPPORTED caveat: HF-style linear warmup also starts at 0 per step, harmless at step granularity, not at round/client granularity. Disclose that tiny LO clients skip the warmup segment (first step lands at s(6/T)).

**Pins (LOCK tests before G1).** (1) The lock and every run config carry `exposure_point="end"` explicitly; a launcher refusing an absent key (the code default "start" must not be relied on). (2) Planned round-LR table generated before training and hashed; min LR > 0; final main-phase unit ends at integer exposures = 6N, LR = 0.01·peak; planned vs actual n_consumed equality enforced per round (fl/schedule.py L83 check retained). (3) `fl.schedule.s_of_efe` equals `v4train.schedule.s_main/s_continuation` on the knots plus 10,000 grid points, bit-for-bit; FL 6.0 endpoint comparison gets the same 1e-12 tolerance as central (fl L64 has none). (4) Synthetic one-client, one-round-per-update fixture: FL round LRs == central `lr_for_update` exactly. (5) LO: for T ∈ {1, 2, 6, 7, 300, 10^4}, lr(0) > 0, lr(T−1) = 0.01·peak, T effective updates. (6) T26/T27 (μ=0 ≡ FA, PF lr_p=0/λ=0 ≡ FA) re-run with "end". (7) Continuation rounds use `continuation=True` only after a STILL_IMPROVING receipt.

## T3 — Frozen evaluation numerics

**LOCK_VALUES (identical for C/FA/FP/PF/CPF/LO, in-training marks, final rows, and the tie-break).** `score_block = 158,486` (= K, one addmm, bit-identical to `BaseAdapter.logits`; ranking.py L19); `class_chunk = 32,768` (exact invariance, recorded only); `eval_batch = 1,024` query rows in manifest order (strictly ascending decision_id), tail batch unpadded; `cache_bytes = 1,073,741,824` (≥ B·K·4 = 649,158,656, so pass 2 re-reads, EVAL-02 §2); device = A16 CUDA, strict FP32, `matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `cudnn.benchmark=False`, `cudnn.deterministic=True`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; metrics accumulated FP64; comparison key adds `eval_score_block`, `eval_batch`, `eval_device_class = "A16-FP32-DET"`; tie diagnostic (n_tied, pessimistic ranks) reported per row.

**Rationale.** PROJECT_MEASURED: block width changes FP32 scores by ≤ 5.3e-5 and can move near-tie ranks by 1 (EVAL-02 §1); one block removes the block dependence entirely. INFERENCE: batch shape can also change the GEMM kernel, so a fixed batch and fixed order make every row's score computed under identical shapes for every method; 649 MB per block fits a 16 GB A16.

**G1 GPU receipt (blocking).** On the 2×A16: pass-2 bitwise reproduction; single block equals `BaseAdapter.logits` bitwise; two invocations of the full M02 rank vector for one checkpoint identical; wall time and memory at K = 158,486.

**ONNX device study.** SERVER-CPU PROXY (EPYC 7H12, 4 threads, ORT 1.24.1; G0 D9) is a deployment/latency study only. Its scores never produce a headline row: it yields a parity receipt (max |score diff|, top-20 agreement, rank-agreement rate on the 16,384-query WARM_TRUNCATION panel) against the frozen evaluator's scores for the same checkpoint, labelled DEVICE_PARITY / HISTORICAL_REFERENCE. Rank differences within the 5e-5 near-tie band are disclosed, not corrected.

## OPEN_DISAGREEMENT_RISKS
- 0.0020 absolute is a judgment; a family gap of 0.0015 with consistent seeds still triggers ~104 GPU-h (cost-of-certainty trade-off).
- Two tie-break seeds vs one staged seed: I chose replication (PKG 08 §5); a one-seed stage would halve the cost at weaker evidence.
- Tie-break recipe reuses the cohort-calibrated peak, not the v1 executed 1e-3; single-factor design vs proven-on-full-data recipe.
- Cost fallback to GRU when equivalent needs an explicit owner amendment of PKG 05 §B.
- Round-constant LR under "end" is a look-ahead of ≈0.0005 EFE; negligible except the first warmup rounds.
- Single-block evaluation costs 649 MB per batch; if GPU memory contention appears at G1, the only permitted change is `eval_batch`, never `score_block`.
- Disk: full-TRAIN checkpoints (~2 GB) must be reconciled with the 12.81 GiB free / 8 GiB floor plan before the tie-break launches.

---

## Step B — Fable ruling on the Opus response (verbatim; the same agent, resumed)

ROUTING: claude-fable-5-1

**Step B ruling (read-only; I did not open PREG1_OPUS_POSITION.md — the coordinator's message states the position, which suffices).**

**1. Staged tie-break: ACCEPTED, stage-1 threshold 0.0040.**

LOCK_VALUES:
- STAGE 1: both families, seed 2026 only, full TRAIN, everything else as my T1 lock (recipe, 6.0 EFE, "end", full 642,228-user manifest, frozen evaluator). Both runs must complete 6.0 EFE with the C3 stopping rule before either number is read.
- DECISIVE_STAGE1 iff |Δ_full,2026| ≥ 0.0040 AND the paired user bootstrap 95% CI (1000, seed 20260926, shared W) of that single-seed delta excludes 0. Then select, record selection_basis = FULL_TRAIN_TIEBREAK_STAGE1, skip seed 2027.
- Otherwise STAGE 2: seed 2027 for both families; my full rule applies unchanged (|Δ̄_full| ≥ 0.0020 with both per-seed deltas sharing its sign; then CONSISTENT_SMALL_ADVANTAGE / EQUIVALENT).
- Budget cap: 65 GPU-h for stage 1, 125 GPU-h total.

Rationale. PROJECT_MEASURED: full-data two-seed gaps ≤ 0.000869 (SASRec v1, packet §Facts). INFERENCE: a single-seed cross-family delta carries roughly √2× one family's seed spread (two independent draws), still ≈ 3× below 0.0040, and the tie-break only runs when the cohort was inconclusive, so a 0.004 full-data gap is a strong signal relative to everything measured. Disclosure required in the lock: the 0.000869 figure is SASRec-only; GRU's full-data seed spread under the common schedule is not in the packet, which is why stage 1 also demands the CI condition and why the stage-1 threshold is 2× the stage-2 floor rather than equal to it. Replication is preserved wherever the evidence is not decisive, which is what PKG 08 §5 protects.

**2. (a) EQUIVALENT fallback — status and interim default.**

Yes: "EQUIVALENT → GRU on cost" is submitted to the owner as the PKG 05 §B amendment (it replaces the NDCG@10 / latency / lexical chain for this case, because those share the same noise).

Until ratified, the default if EQUIVALENT occurs is: HOLD — ARCHITECTURE_LOCK is not written, no downstream run of either family starts, and the four (or six) rows go to the owner with one question. If the owner is unreachable in an autonomous run, the owner-proxy default is GRU on cost, recorded as PROXY_DECISION_PENDING_RATIFICATION; GRU downstream calibration may begin under it because the exposure is bounded (88 GPU-h core, COST_FORECAST L89) and nothing irreversible (no TEST, no ALL_METHODS_LOCK) happens before ratification.

Everything else in my T1–T3 ruling stands unchanged.


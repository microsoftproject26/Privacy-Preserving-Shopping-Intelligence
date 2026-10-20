# G0 — Fable position (verbatim; claude-fable-5-1; sasrec-fable-reviewer agent, read-only; returned ≈2026-09-26T14:06Z, saved 14:07:15Z)

Packet: `G0_EVIDENCE_PACKET.md`. The Opus position memo was withheld (independence rule). AgentId a87443e5f0cc5bcca.

ROUTING: claude-fable-5-1

**VERDICT: APPROVE_WITH_FIXES** — proceed to Wave A (DATA_PROTOCOL_LOCK, POPULATION_PLAN, code tests, FL/LO smoke, central contest) with the decisions below, after the seven blockers are closed with receipts. None requires new science.

Path roots used below: `G0` = `F:\Queens University\Microsoft Project\eda\.local_ai\ppsi_v4\g0`; `PKG` = `F:\Queens University\Microsoft Project\eda\PPSI_SINGLE_BACKBONE_V4_2026_09_26\PPSI_SINGLE_BACKBONE_V4_2026_09_26`; `HO` = `F:\Queens University\Microsoft Project\eda\.local_ai\ppsi_v4\company\handoffs`. I verified the packet's numbers against the cited files; no discrepancies found.

## DECISIONS

**1. Population — locked cohort N = 262,144; family contest on the same cohort; cold users as a separate panel.**
- PROJECT_MEASURED: ALL downstream core = 908 / 2,701 GPU-h (GRU/SASRec) excluding loading, eval, calibration (`G0\COST_FORECAST.json` L29, L109); N=262,144 = 88 / 263 GPU-h (L89, L169) with 62,368 warm users holding 808,610 VALIDATION rows (`G0\data_census\COHORT_PROJECTIONS.json` L6056-6057); N=65,536 gives only 15,487 evaluable users (L3085). Selection is salted-hash, uniform, nested (`COHORT_PROJECTIONS.json` L4, keys hash L7188).
- INFERENCE: ALL is months on 2 A16s for either family; 262,144 is the largest cohort that finishes (SASRec worst case ~11 days wall at my local-batch choice, see 7), with bootstrap support well above 65,536. Cohort is 9.72% of examples and warm users; warm-query rate 23.8% equals the population rate (unbiased).
- Contest on the cohort, not ALL: the thesis contract is "common data"; a winner chosen on ALL and re-trained as C_MATCHED (`PKG\04_CATALOGUE_OOV_AND_POPULATION.md` §3) would select on a different data regime than the comparator and cost ~104 GPU-h more. With a common schedule (6), the contest's C runs are reused verbatim as the downstream C (`PKG\05_CENTRAL_SELECTION_AND_WINNER.md` §E). Defensible alternative noted in risks.
- Cold users: not in the matched block. COLD_PANEL_VAL = all 381,295 VALIDATION users with zero eligible TRAIN examples (`G0\data_census\CLIENT_CENSUS.json` L3971), secondary metric, every method. COLD_H_PANEL = 16,384 salted-hash subset for the h-truncation study (`PKG\09_EVALUATION_AND_COLD_START.md` §5). Fix: the census "cold" = no eligible example, not no TRAIN events (1,509,381 TRAIN users have zero eligible examples, L3967); the manifest must split COLD_NO_TRAIN_EVENTS vs COLD_TRAIN_EVENTS_NO_LABEL.

**2. Catalogue — keep support ≥ 5, K = 158,486, fitted on full TRAIN.**
- PROJECT_MEASURED: s=1 raises VALIDATION label coverage 99.142% → 99.556% (+50,124 targets) at +21% K; s=2 → 99.464% at +13% K (`G0\data_census\CATALOGUE_SUPPORT.json` L17-80). Step cost floor is the full-K head + AdamW (`HO\SYS-01.md` L28-29).
- INFERENCE: +0.4 pp coverage for ~+20% cost on every run and 33k near-untrained classes is not a meaningful gain; s=5 stays (`PKG\04` §1 permits 1/2 only if meaningful). Support fitted on full TRAIN: it is declared SHARED_TRAIN_STATISTIC bootstrap information common to C/FA/FP/PF/CPF/LO (`PKG\04` §4; `G0\feature_map\FEATURE_LOCALITY_MAP.md` L154-160); a cohort-fitted K would change the task. Receipt required: count of classes with ≥1 cohort-TRAIN positive and cohort VALIDATION coverage at s=5.

**3. VALIDATION — merged [11-16, 11-23), minimal-diff, subperiods reported.**
- PROJECT_MEASURED: union = 14,357,645 decisions, 11,288,056 eval rows, 0 rows with target past the end (`G0\data_census\SPLIT_COUNTS.json` L72-85; `LABEL_TIME_AUDIT.json` L19-27); only 170 VS rows are censored purely by the Nov-20 barrier, plus 140 AMBIGUOUS VS→VC (`G0\BARRIER_COUNT.json` L19-23); raw CSVs are not on the server (`HO\OPS-02.md` L11).
- INFERENCE: 310 of 11.3M rows (0.003%) does not justify a raw rebuild that is not even executable server-side. Report Nov16-20 / Nov20-23 descriptively and the 310 legacy-barrier rows as a named censoring class. Fix: `AMBIGUOUS_BOUNDARY` and `BOUNDARY_UNRESOLVED` need code-derived prose definitions in the lock (`G0\data_census\CENSUS_README.md` L121-133).

**4. Features — clean Rich as proposed, with 03 §7 applied literally.**
- PROJECT_MEASURED: M1 root cause pinned (`FEATURE_LOCALITY_MAP.md` L189-212); 36/59 channels carry pooled information (L31); redundancy/dead-flag full-population checks reported 0 (packet §E, feature study).
- Decision: v1 rich minus columns whose deterministic reconstruction is proven per `PKG\03_DATA_PROTOCOL.md` §7 (a)-(d); dead flags removed only where impossibility follows from spec/code, otherwise kept (an all-zero TRAIN input leaves a random untrained weight — disclose). M1: valid-only refit of the two time-gap normalizers plus entries in the load-time re-zero map; fit population full TRAIN; fit grain = executed convention, stated explicitly. From scratch; no D3, no scaler sweep (D7). Exact column list is a DATA_PROTOCOL_LOCK deliverable with the per-column (a)-(d) table.

**5. Precision — strict FP32 everywhere; no TF32 branch.**
- PROJECT_MEASURED: benchmarks/forecast are strict FP32 (`G0\BENCH_SASREC.json` L5); `tf32_matmul_default=false` (`G0\SERVER_CENSUS.json` L372).
- INFERENCE: μ=0≡FA, p=0 recovery, aggregation-order and recentering-invariance tests need reproducible numerics. Lock both `torch.backends.cuda.matmul.allow_tf32=False` and `torch.backends.cudnn.allow_tf32=False` (cuDNN GRU otherwise uses TF32). LITERATURE_SUPPORTED: precision is part of the comparison key (R26).

**6. Central recipes — common EFE schedule, both families to 6 EFE, bounded symmetric LR calibration.**
- INFERENCE: cohort 6 EFE at B=256 is ~91.6k steps (~0.6 of one v1 epoch); the executed epoch-step timings do not transfer (`PKG\08_TRAINING_BUDGET_WITHOUT_WASTE.md` §3 warning). Schedule (fraction of peak vs EFE): linear warmup 0→0.02; peak to 2.0; linear decay to 0.1 by 3.0; low phase 0.1 to 5.5; cooldown to 0.01 at 6.0. Both families run to 6.0 EFE so the C row is reused as CONTROLLED_BUDGET; PRACTICAL_BEST = best 0.5-EFE checkpoint. Continuation to 8 EFE only under BUDGET_LIMITED_STILL_IMPROVING with a recorded budget.
- Calibration (adaptive, symmetric): per family, peak LR ∈ {executed, 0.5×executed}, 2.0 EFE, seed 2026, chosen by cohort VALIDATION MRR@20 at 2.0 EFE; main runs restart fresh. Cost ≈ 3.4 GPU-h. Departure from the package's trigger-based "if" is deliberate: the population changed 10×.
- Stopping: eval every 0.5 EFE (SASRec ~95 s per eval, `BENCH_SASREC.json` L90-94); tolerance 0.0002; patience 2 evaluations after the 3.0-EFE reference; strictly-best saved.

**7. FL / PF / LO.**
- clients_per_round 64 (2 GPU workers × 32); local passes 2; visits 3 (= 6 EFE, 12,288 rounds, `COST_FORECAST.json` L170); participation = seeded permutation sweeps, fixed groups.
- Local batch 16, not 64: PROJECT_MEASURED per-user median 6, p75 15, p90 35 (`CLIENT_CENSUS.json` L11-20); at B=64, 90% of clients do exactly 1 step per pass. B=16 gives the top decile 3+ steps/pass so the proximal term can act (`PKG\06_FEDERATED_AND_PERSONALIZED.md` §2, §4). Cost SASRec FA 44.4 h vs 31.5 h per method-seed (L163-165); core ≈ 366 (SASRec) / 123 (GRU) GPU-h. INFERENCE.
- Solver: AdamW, winner's betas/eps/clip; shared state reset each round for FA/FP/PF; PF personal state persists; LO/C persist; no server momentum; n_consumed weighting; label "FedAvg with AdamW local solver". Local LR at round r = lr_peak_local × s(EFE(r)); lr_peak_local ∈ {peak_C, 0.5×peak_C}.
- CALIB_PARTITION = nested cohort 65,536 (prefix of the main cohort, keys hash `COHORT_PROJECTIONS.json` L3880), 2 EFE, seed 2026, its warm manifest (15,487 users). FP μ ∈ {0.001, 0.01, 0.1}; μ=0 fixture must equal FA. PF λ ∈ {0, 1e-4, 1e-3}, default 1e-4; p_u dim = query dim (128/256), init 0, personal LR = shared peak; CPF matched. ~26 GPU-h SASRec total calibration.
- Recentering (if SASRec): server recenters after aggregation; anchor = broadcast θ_r; per-step local recentering as executed; invariance test at fixed queries.
- LO: all 262,144 cohort clients, procedural (train, evaluate at 2/4/6 passes, discard), AdamW persistent, B=16, same schedule over its own 6 passes, same per-seed init. TEST-time LO is a deterministic retrain with a VALIDATION-rank identity receipt.
- Endpoint 6 EFE controlled; 8 EFE practical ceiling.

**8. Evaluation.** Primary: client-macro MRR@20, end-to-end (OOV=0), over the registered warm-cohort VALIDATION manifest (62,368 users). Prominent secondary: NDCG@10, HR@10; also micro/macro MRR, Recall, NDCG at k∈{1,5,10,20}; conditional-rankable variants and coverage (`PKG\04` §2 identity tests). Bootstrap unit = user, paired, seed 20260926, 1000 resamples, ratios bootstrapped directly; per-seed values + paired per-seed deltas + mean (`PKG\09` §4). Tie rule = v1 streaming rank rule, quoted in the lock. LITERATURE_SUPPORTED: exhaustive candidates, no sampled metrics (R14).

**9. Device target — declared proxy only.** PROJECT_MEASURED: SASRec exports (legacy exporter, opset 18, parity ≤2.1e-6, top-20 identical 8/8; `G0\export_smoke\SASREC_REPORT.json` L268-332); GRU blocked in torch 2.13 (`GRU_REPORT.json` L92-97). Lock proxy = AMD EPYC 7H12, 4 threads, onnxruntime 1.24.1 CPU EP, labelled SERVER-CPU PROXY. No phone/browser target is invented; owner hardware is a later optional slot. Export feasibility never influences family choice (`PKG\05` §D). EDGE-02 adapter proceeds only if GRU wins.

**10.** See BLOCKERS and OPTIONAL_WORK_SKIPPED.

## LOCK_VALUES
- POPULATION: cohort N=262,144, salt `PPSI_V4_POPULATION_20260926`, keys hash `54e1d3c2…48184`; calib prefix N=65,536 hash `3d680aa7…3a6f4`; cold panel 381,295; cold-h panel 16,384.
- CATALOGUE: support ≥5 on full TRAIN, K=158,486, input table K+3.
- SPLITS: TRAIN [10-01,11-16); VALIDATION [11-16,11-23) minimal-diff; RETRO_TEST [11-23,11-30); LABEL_GRACE [11-30,12-01); subperiods 11-16/11-20/11-23.
- FEATURES: clean Rich per 03 §7; M1 valid-only refit + re-zero; full-TRAIN fit; from scratch.
- PRECISION: FP32, matmul and cudnn TF32 off.
- CENTRAL: B=256; AdamW executed betas/eps/wd/clip; schedule 0.02/2.0/3.0/5.5/6.0 as above; 6 EFE endpoint, 8 ceiling; eval every 0.5 EFE; tol 0.0002, patience 2; seeds 2026/2027; LR calibration {1.0, 0.5}×executed at 2 EFE.
- FL: 64 clients/round; 2 passes; 3 visits; local B=16; AdamW reset per round; n_consumed weights; μ grid {0.001,0.01,0.1}; λ grid {0,1e-4,1e-3}; p dim = query dim; personal LR = shared peak.
- LO: 6 passes, B=16, persistent AdamW, procedural.
- EVAL: macro MRR@20 primary; user bootstrap 1000, seed 20260926.
- DEVICE: SERVER-CPU PROXY (EPYC 7H12, 4 threads, ORT 1.24.1).

## BLOCKERS (close before the lock)
1. Disk: 12.81 GiB free, floor 8 (`SERVER_CENSUS.json` L300). SASRec weights-only checkpoint 332 MB; 12 downstream runs × best-only = 4 GB — over the margin. Written retention/sync plan with numbers required.
2. Executed peak LR, betas, eps, weight decay, clip for GRU F4 and SASRec A1 must be transcribed into the lock with source path; I could not verify them from the cited files.
3. Prose definitions of `AMBIGUOUS_BOUNDARY` / `BOUNDARY_UNRESOLVED` derived from code.
4. Cohort receipts: VALIDATION coverage at s=5 on the cohort manifest; classes with ≥1 cohort positive; cold strata split (1c).
5. `userctx.py` byte-hash vs frozen `code_hash` and the `category_code_missing ≠ main_category_missing` count (`G0\feature_map\OPEN_ITEMS.md` §2, §4).
6. FL/LO mechanics smoke before central training (scope §"Gate order"): μ=0≡FA, p=0 recovery, worker p_u restore, deterministic aggregation order.
7. Determinism receipt for LO retrain (deterministic algorithms flag or a 1,024-client reproduction check).

## OPTIONAL_WORK_SKIPPED
TF32 branch; K=1/2 catalogues; raw resessionization; December audit source check; FedPEFT/RQFedRec efficiency branches; onnxscript install (not permitted); loser-family anything; seed 2028.

## OPEN_DISAGREEMENT_RISKS
- Contest on cohort vs ALL (+C_MATCHED): ALL selects on the strongest data; cohort keeps the contract common. I chose cohort.
- Local batch 16 vs 64: +41% FL cost for meaningful local work on the top decile.
- Unconditional 2-point LR calibration vs package trigger-based.
- Common EFE schedule vs executed per-family epoch-step timings (05 §C/08 §3 wording).
- Primary metric over warm-cohort users only vs all VALIDATION users.
- Support ≥5 vs 1 (0.4 pp coverage).
- LO retrain-at-TEST vs sealed prediction writes.
- Absolute scores on 9.7% data will be below v1; that is a protocol change, not a regression.

# Fable decision — Wave-2 registration (dataset set, scope, device, spare-resource use) — 2026-10-03, read-only

**Scope read:** contracts/PREG1_AMENDMENT_07.md (B1 Wave-2 note, B3, B6, B7 R1 / R4 / R9, B9 M2 / M3, B10, B11), addenda B12 and
B13, company/handoffs/BM-W2.md, company/reviews/FABLE_DECISION_ML1M_TRACKB_20261003.md, DECISION_LOG rows 324–376. No benchmark
holdout / TEST value and no REES46 MAIN / VALIDATION / TEST value was read. No Codex, no git, no file other than this one written.
Facts below marked MEASURED come from DL 325 / 363 / 364 / 365 / 376 and the Ops brief; everything derived from them is INFERENCE.

## Decision 1 — Datasets: Video_Games + Baby_Products. Beauty_and_Personal_Care is DROPPED (M3, measured).
- MEASURED (16 threads, incl. setup): FA 20 rounds = 43 s (VG) / 60 s (Baby) / **1,164 s (BPC)**; C_FULL 0.1 epoch = 126 s / 205 s /
  **2,986 s**. R1 gives T = 9,872 (VG, manifest-confirmed) / ≈ 15.7 k (Baby) / 75,998 (BPC) rounds.
- INFERENCE: ≤ 2.2 s / round → one VG FL run ≤ 6 h; ≤ 3 s / round → one Baby FL run ≤ 13 h; BPC ≥ 5–58 s / round × 75,998 rounds =
  **days to weeks per FL run, ≈ 16 FL-type runs per dataset**; BPC C_FULL ≈ 8.3 h / epoch → ≈ 2 days per 6-EFE run. Not feasible
  before any plausible thesis date on either device class (GPU would still cost ≈ 10 FL runs × many hours and the REES46 endgame
  needs the A40).
- A "different client rule" for BPC only (sub-sampled device population, fewer rounds, more clients per round) is REFUSED: it
  changes R1 for one dataset inside the same wave, so BPC would no longer be the "recipe unchanged" Track-A claim and could not
  sit in the same table; a central-only BPC row answers no FL question. Owner rule "drop what breaks" (DL 325) applies. The
  converted files and the measured timing stay as evidence; BPC can only return by a new addendum registered before any run on it.

## Decision 2 — Scope: Track A only, full arm set, seeds 2026 / 2027, R6 third-seed rule, B12 per-seed S records.
- B3 limits Track B to Wave 1; nothing in Wave 2 is tuned, so Wave 2 is the purest "transfer without retuning" test.
- Arms exactly as B2 / B7 R4 / B8 O2 / B9 M1c–M1d / B10 Q2 / B12: C_FULL, PRE, FA, FA_WARM (S7), FA_WARM_FROZEN (S8) → FA_WARM\*
  rule (B2), FT_FA_WARM\*, FP, PF, FA_Q8 (warm on FA_WARM\*), S_CAL per PRE seed, FA_1024 ×2, DP8 ×2 (DP8_FROZEN or DP_WARM per M1d).
- ID-only (B6 disclosure); the rating column is ignored; the "RICH" variant (B9 minor) is NOT run in Wave 2 — B14 closes Wave 2 to
  Track A ID-only. The unconverted `timestamp_w_his` split is closed (not one holdout per user; BM-W2 Q2).
- Ours-vs-ours only (no published number on the official 2023 5-core files; B1, DL 329): R8's paired user bootstrap for arm-vs-arm
  contrasts, DESCRIPTIVE with respect to every PREG1 claim; no published row, no "on par" statement.

## Decision 3 — Device: (a) CPU-FP32 as registered, on the spare CPU now, 3 slots. (b) A40-FP32 for Wave 2 is REFUSED.
- Time-to-finish (INFERENCE): CPU at 3 slots × 16 threads (measured load ≈ 50 / 128 → ≈ 100 / 128), both datasets in one driver:
  VG ≈ 10-05, Baby ≈ 10-07; 6 slots after the ML-1M driver exits (≈ 10-06 / 07) pull Baby in, not out. GPU: 6–8 h of runner code
  (device selection, deterministic CUDA, TF32 off) + tests + an independent review, which the agent budget (≈ 10 % until the weekly
  reset ≈ 10-08; the same fact DL 368 relied on) cannot pay before the CPU path is already finished; run time after that is hours,
  but the start is ≈ 10-08. CPU finishes first and spends no review risk.
- Integrity: B6 / R9 register one benchmark device class. A second class for a new dataset family is legal (no within-grid mix), but
  it is one more disclosure line in a table that carries no published comparison — no claim gain. The device class of a dataset is
  fixed at its first run; no mid-dataset switch (same reasoning as DL 368).
- GPU1 is idle, but the REES46 endgame (FA_Q8 s2026 → D freeze → E check-only → F §9 read → G lock → H TEST builds) takes the A40 on
  demand with --i-confirm steps; a Wave-2 GPU driver on GPU1 could not be guaranteed to be out of the way at those moments without a
  stop-and-resume mechanism nobody has written or reviewed. Idle is cheaper than invented work.
- INT8 pass 2 (DL 368) needs a quiet host: Wave 2 yields to it (B14 §6), so pass 2 is not delayed by this decision.

## Decision 4 — Other work that may use GPU1 / the spare CPU now without breaking a rule
- **GPU1: nothing is eligible.** Every registered GPU-eligible REES46 job is done or running (FA_Q8 s2026 alone on GPU0; endgame B
  and B2 done, DL 376); ML-1M Track B is CPU by DL 368; the K16 device number is a CPU measurement by definition; Wave 2 is CPU
  class by this decision. An idle GPU breaks no rule; adding unregistered work would. The endgame uses it next (D–H), and the
  owner's --i-confirm on F / G / H is the only thing that can shorten that wait.
- **Spare CPU, all registered or required, in this order:** (1) Wave 2 at 3 slots (this decision); (2) Wave-1 S3-Beauty / Sports /
  Toys LOCK + single `--final` per LOCK_FINAL_RUNBOOK (every job done, DL 369; R7 / B11; Ops, light CPU); (3) the B13 stopper
  mechanism and the INT8 pass-2 latency-only mode (DL 368 TODOs i / ii — small code + review, needed before Wave 1 ends; this is
  the one place a small agent spend is justified); (4) D-freeze preparation (A6 inventory, E `--check-only` dry runs) while FA_Q8
  finishes — gated steps prepared, not run.

## Exact text — contracts/PREG1_AMENDMENT_07_ADDENDUM_B14.md (REGISTERED by Ops with a DL row, before the first Wave-2 run)

> # PREG1 amendment 07 — addendum §B14: Wave 2 (REGISTERED 2026-10-03, DL ___; Fable decision company/reviews/FABLE_DECISION_WAVE2_20261003.md)
>
> - **Status:** registered before any Wave-2 run. No benchmark holdout / TEST value and no REES46 MAIN / VALIDATION / TEST value has
>   been read. Amendment 07 B1–B11 and addenda B12 / B13 are unchanged; B14 specialises them to Wave 2 and supersedes only where it
>   says so. Wave 2 cannot touch a Wave-1 lock (R7). REES46 is unaffected.
>
> **B14.1 Datasets (fixed).** Wave 2 = Amazon Reviews 2023 official 5-core `last_out_w_his` splits for **Video_Games** and
> **Baby_Products** only, as converted by BM-W2 into `bench_proc_w2/amazon2023_video_games` (94,762 users / 25,612 items / 814,586
> interactions; holdout.csv 45212128e455badf…, train.csv 7e616074de56e97c…, statistics.csv 426d4fa56d5f49b6…) and
> `bench_proc_w2/amazon2023_baby_products` (150,777 / 36,013 / 1,241,083; 43c0ce1ad1c027fb…, 9445cbbe995d7269…, d897c1357eba3d03…);
> the full sha256 values in each dataset's SHA256SUMS are pinned in its LOCK. train.csv = official train + official valid rows;
> holdout.csv = the official test row (one per user, all users). **Beauty_and_Personal_Care is DROPPED** (B9 M3 MEASURED: FA 20 rounds
> 1,164 s and C_FULL 0.1 epoch 2,986 s on 16 threads; T = 75,998 rounds → weeks per FL run); its converted files and timing stay as
> evidence; it returns only by a new addendum registered before any run on it. mbht_tmall (DL 325) and mbht_taobao (B9 M2) stay
> dropped. The `timestamp_w_his` variant is not run (not one holdout per user). The Wave-2 "RICH" variant (B9 minor) is **not run**;
> Wave 2 is ID-only like Wave 1; the rating column is ignored.
>
> **B14.2 Protocol.** Leave-one-out exactly as B1 / R2: inner-validation target = the user's last train.csv row under the registered
> rule (datetime descending, stable over file order) = the official valid row; training = train.csv minus that row for every arm;
> TEST inference from the full train.csv history, no retrain; inner-validation candidates = the train.csv catalogue minus the user's
> inner-train items; full-catalogue ranking (25,612 / 36,013), seen-item filtering, ascending-ID ties. Before the first run of a
> dataset its manifest records: the inner-validation population and the check that our validation row equals the official valid row
> for every user (Video_Games MEASURED equal, DL 363; Baby_Products to be recorded), N FL clients, T, m, and the R9 band coverage
> (items seen / holdout targets covered). **Cold holdout items** (items absent from train.csv: 69 / 59; BM-W2) count as misses in
> every metric's denominator (B9 minor, B10 Q5); their count and the affected user count are printed in the manifest, the LOCK and the
> TEST_LOOK; the Wave-1 rule "a --final refuses a non-zero cold count" does **not** apply to Wave 2.
>
> **B14.3 Scope and arms.** **Track A only** (B3; Track B is Wave 1 only; no grid, no budget axis, no tuning of any kind). Arms as
> B2 / B7 R3–R4 / B8 O2 / B9 M1c–M1d / B10 Q2 / B12: C_FULL (SASREC_D256_B3, all users, 6.0 EFE, 12 marks, PRACTICAL_BEST on all users'
> inner validation); PRE (SASREC_D64_B2 on the seeded 15 % band, band seed 2026, θ_pre = the 6.0-EFE endpoint, PRE-alone row selected
> on the band); FA; FA_WARM (S7) and FA_WARM_FROZEN (S8) with the B2 FA_WARM\* rule; FT_FA_WARM\* (FT_C_f0.3_P4); FP; PF; FA_Q8 warm on
> FA_WARM\*; S_CAL per PRE seed (B12); FA_1024 ×2; DP8 ×2 (DP8_FROZEN if FA_WARM\* = S8, DP_WARM if S7; M1d). **FL clients = all users**;
> T = `fa_rounds(N, P = 1)` as written in the manifest (Video_Games 9,872 MEASURED; Baby_Products' manifest value is binding, M3
> estimated 15,707); 64 clients per round, sweep sampling, 10 % seeded drop-out; 12 marks at ceil(k·T/12). Poisson arms: q = 2⁻⁷
> exactly (B10 Q1), T = 384, δ = 5e-6, z = 0.5642, P = 2, B16, **m = round(q·N) = 740 (Video_Games) / 1,178 (Baby_Products)**; S =
> the R4 / B12 rule per PRE seed, written to `runs_bench/<dataset>/dp/S_RECORD_<VARIANT>_s<S>.json` before the matching DP run.
> Disclosed: per-round noise ≈ z·S·√d/m is 1,024/m ≈ 1.4× (Video_Games) and ≈ 0.9× (Baby_Products) REES46's — closer to REES46 than
> any Wave-1 dataset. Pins as R9 (FL_TUNING_SELECTION 9bca186e…, S7 d67a0a85…, S8 8b79d305…, SELECTION_FT_C cell FT_C_f0.3_P4).
>
> **B14.4 Seeds, selection, populations, metrics.** Seeds 2026 and 2027 for every arm; the R6 third-seed rule (2028 for C_FULL and
> FA_WARM\* when the two seeds' inner-validation PRACTICAL_BEST NDCG@10 on all users differ by > 0.002; decided and logged before the
> dataset's LOCK). Checkpoints PRACTICAL_BEST and CONTROLLED_BUDGET (DP8 headline at CONTROLLED_BUDGET, R4). Metrics as B4; MRR@20 in
> our tables only (O5). Populations: all holdout users (94,762 / 150,777, cold-item users included as misses), band ∩ holdout,
> others ∩ holdout, and the O3 descriptive "holdout target covered by the band" row for S7 vs S8.
>
> **B14.5 Comparisons and claims.** No published number exists on these files: Wave 2 is **ours-vs-ours only**. No published row, no
> "on par" statement, no external tolerance test. Arm-vs-arm contrasts use the R8 unit (paired user bootstrap, 1,000 resamples, seed
> 20260926, per seed, then the seed mean) and are DESCRIPTIVE with respect to every PREG1 confirmatory claim; retention ratios carry
> the 6-EFE-ceiling label (O1 disclosure) and are never pooled with Wave 1 or REES46. HSTU / FuXi-α Amazon rows stay context only (O4).
>
> **B14.6 Device, runner, scheduling.** Device class **CPU-FP32 exactly as B6 / R9** (torch deterministic, 16 threads fixed, nice 15),
> runner **code_bench4 (the B12 tree) unchanged**, its code-tree digest pinned in each LOCK; any runner code change for Wave 2 needs
> a delta review before the first run that uses it. The device class of a dataset is fixed at its first run; no device switch
> inside a dataset; **no GPU for Wave 2** (a GPU class could be registered only by a new addendum for a dataset with no run yet).
> Driver: the registered Wave-1 driver pattern restricted to `amazon2023_video_games amazon2023_baby_products`, Track A only; its
> plan (rc 0, job list) is logged in the registering DL row before launch; a shell-only driver change is verified by Ops (plan mode),
> as DL 341 did; one driver per dataset (DL 360); separate state root `runs_bench/<dataset>` under the Wave-1 layout. **Slots: 3 × 16
> threads now**; raised to at most 6 only after the ML-1M code_bench3 driver and laneBENCH_RESUME have exited. **Precedence:** (i) the
> REES46 endgame (FA_Q8, D–H) is never delayed by Wave 2; (ii) **K16 INT8 pass 2 (DL 368) has precedence over Wave 2**: when every
> Wave-1 driver has exited, Ops drains Wave 2 (no new job launched; running jobs may be SIGSTOPped for the pass-2 window and
> SIGCONTed after — this changes wall time only, never a value) and the pass-2 host load is written beside its result as pass 1 did;
> (iii) Wave 2 never delays a Wave-1 LOCK or `--final`.
>
> **B14.7 Locks, TEST, stopping.** Per dataset, after every Track-A run (and any third seed) has a RUN_SUMMARY: `runs_bench/<dataset>/
> LOCK.json` (every run, both checkpoints, the FA_WARM\* decision, both S records, the dataset shas, the manifest, the code digest);
> then ONE `--final` (R7, B11: TEST_ATTEMPTS.jsonl append-only; TEST_LOOK.json atomic; a crashed attempt is retried only with the
> identical LOCK). Nothing is run, added or re-selected after the look. **Stopping rule (clock, never value):** if the thesis schedule
> requires it, Ops may stop a Wave-2 dataset by a DL row **before its LOCK**; a stopped dataset has no LOCK and no TEST look and is
> reported as "converted, not run" with the measured reason — no partial Track-A table is ever locked. Nothing else changes.

## Integrity rules kept
No value read; Wave 2 adds no tuning, no new method, no published comparison; one device class, one LOCK and one TEST look per
dataset; every rule above is registered before the first Wave-2 run and is clock- or artifact-bound, never value-bound; REES46 locks,
runs and the A40 pin are untouched; Wave-1 locks are untouchable from Wave 2 (R7).

INFERENCE on this basis (not a result): Video_Games LOCK + look ≈ 10-05, Baby_Products ≈ 10-07; both before the weekly agent reset,
with zero agent spend on runner code.

---

## Re-review (2026-10-03 ≈ 12:30Z, owner request; read-only) — Wave 2 on GPU1? **KEEP CPU. The GPU1 move is REFUSED.**

**New facts (MEASURED, Ops):** REES46 has one MAIN run left (FA_Q8 s2026, GPU0 21 %, 11,732 / 13,654, ≈ 14:00Z); GPU1 idle (0 %,
0 MiB); CPU 91 / 128 (ML-1M driver 6 slots + Wave 2 3 slots, 10 jobs); Wave 2 (code_bench4, CPU) launched 11:25Z, early jobs only;
B13 stopper armed; the weekly agent quota reset on 10-02 (my earlier budget note is therefore weaker, not void).

**Proposal weighed:** stop the CPU Wave-2 driver, restart Wave 2 from scratch on GPU1 (A40-FP32, TF32 off, deterministic CUDA),
REES46 keeps GPU0; cost = runner GPU path + tests + independent review (≈ 6–8 h) as addendum B15; claimed gain = Wave 2 ends ≈ 10-04
instead of ≈ 10-05 / 07.

1. **The gain is unmeasured and probably small.** No GPU per-round time exists for the benchmark runner. The one GPU FL datum we
   have is REES46's FA_Q8 alone on GPU0: 11,189 → 11,732 rounds between 11:15Z (DL 376) and ≈ 12:00Z ≈ 4–5 s / round at 21 %
   utilisation — INFERENCE: FL rounds (64 small clients, one local pass) are host- and launch-bound, not GPU-bound. The Wave-2 CPU
   figure is ≤ 2.2 s / round (VG). A 2–4× per-job speed-up is plausible, 10× is not; and 6–8 feeder processes land on a host that is
   already at 91 / 128. C_FULL (the one arm a GPU helps a lot) is 2 of ≈ 24 runs per dataset and ≈ 2 h on CPU anyway.
2. **The schedule arithmetic does not reach 10-04.** Code 4–6 h, independent review 2–3 h, fixes + delta review 2–3 h (REV-BM-02
   found 1 BLOCKER + 7 MAJOR first time; REV-BM-02 delta then B12 delta each needed a cycle): realistic first GPU run ≈ 10-04 00:00Z,
   restart from scratch, ≈ 15–25 h of wall for ≈ 110 job-hours of FL at 6–8 concurrent → ≈ 10-05. CPU (INFERENCE, B14): VG ≈ 10-05,
   Baby ≈ 10-07, pulled in when the ML-1M driver exits (6 slots). Expected gain ≈ 0–2 days on a DESCRIPTIVE, ours-vs-ours track
   nothing downstream waits for (R7 locks per dataset; REES46 and the thesis chapter order do not depend on it).
3. **The cost is the scarce resource, even after the reset.** 8–12 h of Opus / Fable time is exactly what the pending required
   items need: the Fable review of the endgame scripts before F / G / H (DL 354), the B13 stopper, the INT8 pass-2 latency-only
   mode (DL 368 ii), thesis capture, and the Wave-1 S3 locks / finals. A GPU port of a CPU-only-by-design runner touches device
   selection, FL engine, DP noise / FA_Q8 stochastic-rounding generators, the evaluator's stable sort and the `--final` path —
   REV-BM-02-sized review surface for a secondary track.
4. **Contention is real on the host, and the "A40 pin" has used both GPUs.** REES46 ran on GPU0 and GPU1 (DL 346 / 363: 28 / 22 GB
   and 28 / 18 GB of 46 on the two cards); the endgame TEST-build jobs are not known to be pinned to GPU0 only. Wave-2 GPU jobs would
   need a yield / drain rule to steps that the owner confirms interactively (F / G / H) — one more mechanism nobody has written.
5. **Integrity is fine either way** (restart from scratch avoids a device mix; one class per dataset) — this is not the reason.
   The reason is value for cost: the move buys at most a day or two of calendar on a descriptive table, for ≈ a working day of the
   review capacity the confirmatory REES46 endgame needs this week.

**Decision:** B14.6 stands — Wave 2 CPU-FP32 on code_bench4, 3 slots now, ≤ 6 after the ML-1M driver exits; no B15. The CPU Wave-2
driver is not stopped. **What would change this answer:** a MEASURED benchmark-runner GPU round time (not a REES46 one) under
≈ 0.5 s on a Wave-2 manifest, obtained without touching Wave 2 — which itself needs the port. So: closed for Wave 2.

**The idle GPU, honestly:** GPU1 is idle because every registered GPU-eligible job is done. The fastest legitimate GPU use is the
REES46 endgame itself from ≈ 14:00Z (D freeze → E check-only → F §9 read → G lock → H TEST builds) — its waits are the owner's
`--i-confirm` steps and the pending Fable review of the endgame scripts, not hardware. Nothing else may be invented for it.

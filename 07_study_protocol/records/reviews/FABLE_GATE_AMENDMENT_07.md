# Fable gate review — PREG1 amendment 07 (external benchmarks), before registration

- **Reviewer:** Fable 5.1 gate reviewer, READ-ONLY, 2026-10-01. No Codex, no git, no server write. Only this file was written.
- **Read:** `contracts/PREG1_AMENDMENT_07_DRAFT.md`; `company/plans/BENCHMARK_PLAN_V1.md` §14 (and §1, §2, §9); `company/research/BM-00_PROTOCOL_AND_COMPARISONS.md`; DECISION_LOG rows 292, 296–311; `contracts/PREG1_AMENDMENT_06N.md`; for cross-checks 06A, 06C, 06D, 06I, 06K, 06M, 06M_B, `contracts/dp/S_RECORD.json`, `EVALUATION_CONTRACT_V4.md` §12, and the unreviewed draft code `code/v4bench/{recipe,central,data}.py` (read only to see which rule the code currently assumes; the code is not under review here).
- **Owner constraints checked against:** only REES46 methods; no new method; published numbers only on byte-identical splits; REES46 unaffected.

## VERDICT: REGISTER_WITH_FIXES

The draft is faithful to the owner's decisions and to BM-00, and the overall design is sound (LOO protocol verified 12 / 12, one TEST look per dataset behind a LOCK, inner validation for every selection, the N2 rule mirrored, deviations disclosed). It is **not yet unambiguous enough to run**: nine points leave a run-time choice to the implementer, and one of them (DP8 "the registered z") is wrong as written — the registered z = 0.5642 is exact only for q = 2⁻⁷, T = 384, δ = 5e-6; the draft code's "client count" rounds rule changes neither, but the draft text fixes neither q nor m for N = 6 k–35 k. None of the fixes adds a method; every replacement text below reuses a rule already registered in the REES46 line (06A, 06C, 06D, 06I §5, 06K, 06N, EVALUATION_CONTRACT §12).

---

## Required fixes (register only after all nine are in the file)

**R1. Who the FL clients are, and the rounds rule written out.** B2 "Rounds" is silent on whether the 15 % band users are also FL clients, so `fa_rounds(n_clients)` has no defined input. Replace the B2 "Rounds" line with:

> - **FL clients = ALL users of the dataset** (band users keep a device; their consent covers the server-side PRE copy only). The band is the same set for every run seed (band seed 2026 ≠ run seed). **Rounds:** the REES46 exposure rule `launch_fl.fa_rounds(N_clients, P)` = ceil(60 · N / (576 · P)) with the FL_TUNING_SELECTION local-pass count P, 64 clients per round, sweep sampling, 10 % seeded drop-out — i.e. T = 630 (ML-1M, N 6,040), 2,330 (Beauty, 22,363), 2,023 (Toys, 19,412), 3,709 (Sports, 35,598) for P = 1; the 12 evaluation marks fall at every 0.5 EFE (round ceil(k · T / 12)). The exact T per dataset is written in each dataset manifest before the first run.

**R2. Training data vs inner validation for every arm.** The draft says "every selection uses only this" but never says that training excludes the validation row. Add to B1 after the "Inner validation" bullet:

> - **Every training set is train.csv minus the user's inner-validation row:** C_FULL (all users), PRE (band users), every FL client's local data, and the FT_FA_WARM\* on-device fine-tune. Inner-validation candidates = the train.csv catalogue minus the user's inner-train items. **TEST inference** uses the full train.csv history (including the inner-validation item), no retrain, as eSASRec does (BM-00 §2.2). The ML-1M tie rule is stated literally: rows are ranked by datetime **descending with a stable sort over file order**; the rank-0 row is the validation target, so among rows tied at the user's maximum timestamp the **first row in file order** is the target (2,483 users differ from "last row"; BM-00). Before the first run, the dataset manifest records that our construction equals eSASRec's target for 6,040 / 6,040 ML-1M users under our pandas, and the inner-validation populations (22,308 / 35,475 / 19,312 / 6,039). We rank the **full train.csv catalogue** (12,092 / 18,347 / 11,917 / 3,704) for every arm and do not truncate it; published SAS+ / eSAS could not reach 89 / 265 / 232 / 0 holdout targets (BM-00 §2.3) — disclosed beside every direct comparison.

**R3. Central budget, θ_pre and PRE's selection population.** B2 defines C_FULL and PRE by architecture only. Replace the C_FULL and PRE rows with:

> | C_FULL | central SASREC_D256_B3 on all users; the registered central recipe (S0 schedule on the 6.0-EFE clock, registered peak LR, AdamW, grad-clip), 6.0 EFE, 12 marks; PRACTICAL_BEST selected on ALL users' inner-validation NDCG@10 |
> | PRE | central SASREC_D64_B2 (the registered SMALL recipe), same clock, trained on the band users' train rows minus their inner-validation row; **θ_pre = the CONTROLLED_BUDGET (endpoint 6.0) checkpoint, no selection** (06K §1); PRE's own PRACTICAL_BEST is selected on the **band** users' inner validation only (the server has only their consent) and is reported as the PRE-alone row on all / band / others |

and add to B6 "Deviations": "the central budget is 6.0 EFE (≈ 6 passes) vs eSASRec's up-to-200 epochs with patience 50; this is the Track-A 'no retuning' condition and is disclosed beside C_FULL."

**R4. DP8 parameters and S calibration.** "the registered z for ε 8; S calibrated on the band" is invalid: z = 0.5642 is the accountant's value for (q = 2⁻⁷, T = 384, δ = 5e-6) and nothing else (06C), and "S on the band" is not the REES46 rule. Replace the Phase 2 row text for DP8 with:

> DP8: the REES46 mechanism unchanged — Poisson sampling with **q = 2⁻⁷, T = 384, δ = 5e-6, z = 0.5642** (identical (q, T, δ), so the registered accountant value is exact), fixed denominator **m = round(q · N)** = 47 (ML-1M) / 175 (Beauty) / 152 (Toys) / 278 (Sports); warm start from FA_WARM\*'s θ_pre as in 06K §3 when FA_WARM\* = S8 (noise dimension = the uploaded parameter count). **S** = the REES46 rule (S_RECORD: median client-delta L2 norm over rounds 1–20 of a z = 0 S_CAL run with the same configuration on the FL clients; non-private, disclosed as in 06 §2 Q3), written to `runs_bench/<dataset>/dp/S_RECORD.json` before the DP8 run. Disclosed: per-round noise ≈ z · S · √d / m is 1,024 / m ≈ 4–22 × REES46's; DP8 is reported at CONTROLLED_BUDGET as its headline (06D §3), PRACTICAL_BEST secondary.

**R5. Track B made executable.** B3 does not say which arms run the grid, whether the winner is per system or shared, how many seeds the grid uses, or which PRE a warm cell starts from. Replace the Track B paragraph with:

> - **Track B (secondary, Wave-1 only):** grid max_len {50; 200 on ML-1M only} × dropout {0.1, 0.2, 0.5} (3 cells; 6 on ML-1M). Arms: C_FULL, PRE, FA, and FA_WARM\* (the variant decided in Track A; not re-decided). Grid cells run with seed 2026 only; **the winner is chosen per dataset and per arm** on inner-validation NDCG@10 at PRACTICAL_BEST (ties within 0.0005 → the Track-A cell, then smaller max_len, then larger dropout); the winning cell of each arm is then run with seed 2027. A warm cell uses the PRE trained in the same (max_len, dropout) cell. Track A and Track B values are never mixed, pooled or compared against each other's published row.

**R6. Third-seed trigger and the 3-seed rule.** Replace the B4 "Seeds" bullet with:

> - **Seeds:** 2026 and 2027 for every arm. A third seed (2028) is run for Track-A C_FULL and FA_WARM\* on a dataset when the two seeds' **inner-validation PRACTICAL_BEST NDCG@10 (all users)** differ by more than 0.002; the decision is taken and logged (DL row) before that dataset's LOCK.json is written, never after a TEST look. Where a third seed exists the reported value is the 3-seed mean (all seeds listed).

**R7. Populations and the LOCK / TEST gate.** Replace the B4 "Populations" bullet and the "Locks and TEST" bullet with:

> - **Populations:** all holdout users (= the published denominators 22,311 / 35,539 / 19,365 / 6,038); separately the band ∩ holdout and the others ∩ holdout, with their counts printed. The 52 / 59 / 47 / 2 users without a holdout row stay in training and inner validation only.
> - **Locks and TEST:** per dataset, after every Track-A and Track-B run (and any third seed) has a RUN_SUMMARY, `runs_bench/<dataset>/LOCK.json` lists every run, both checkpoints (sha256), the FA_WARM\* decision, the Track-B winners and the code-tree digest; then ONE `--final` invocation scores every locked checkpoint for every population and metric in a single pass (holdout users only); nothing is run, added or re-selected afterwards. Wave 2's addendum cannot touch a Wave-1 lock.

**R8. Claim strength against published single runs, and the REES46 firewall.** Add to B5:

> - **Claim rule:** our value = the 2-seed (or 3-seed) mean with both seed values printed. Against a published single-run cell, a difference is stated only when |Δ NDCG@10| > 0.005 (the Spotify SAS+ / eSAS acceptance tolerance; the observed reproduction spread is ≈ ±0.003) **and** every seed lies on the same side; otherwise "on par". No significance test is computed against a published number (no paired data). Our own arm-vs-arm contrasts use the REES46 unit: paired user bootstrap (1,000, seed 20260926) per seed, then the seed mean. Benchmark retention ratios (LOO, one query per user, ID-only) are **not comparable in magnitude** to REES46's temporal multi-decision K-contrasts; they are DESCRIPTIVE with respect to every PREG1 confirmatory claim, and no benchmark value enters any REES46 selection, register or lock.

**R9. Pins and device class.** Add to B2 / B6:

> - **Pinned inputs (sha256 in this file and in every LOCK):** FL_TUNING_SELECTION.json (S4, 9bca186e…), FL_WARM_SELECTION_S7.json (d67a0a85…), FL_WARM_SELECTION_S8.json (8b79d305…), SELECTION_FT_C (cell FT_C_f0.3_P4), the r6 code tree digest; the item vocabulary of every arm = the full train.csv catalogue; S8's frozen rows for items absent from the band's inner-train stay at θ_pre's values (disclosed). Band coverage (items seen / validation targets covered) is MEASURED for all four datasets (data manifest) before the first run; today only Beauty and ML-1M are measured.
> - **Device class CPU-FP32:** torch deterministic algorithms, **16 threads fixed** (reduction order changes with the thread count), nice 15; FA_Q8 stochastic rounding keeps its seeded generator. REES46 stays on the A40 pin; `code/v4bench` imports v4models / v4train / code/fl read-only and no REES46 file, lock or run directory is written.

---

## Scientific-validity judgement (for the record)

1. **LOO single query vs REES46 temporal multi-decision.** Legitimate as a *separate* study answering "does the recipe transfer and where does it land against published full-catalogue numbers". It cannot support or weaken any REES46 K-contrast; R8's firewall sentence must be in the file so that nobody later pools "retention" across the two evaluations. Amazon S3 timestamps are synthetic positions (BM-00 §2.1), so "temporal" has no meaning there anyway.
2. **Single-run published cells (≈ ±0.003) vs our 2 seeds.** Only an "on par / above / below by > 0.005" statement is defensible; no p-value, no CI against a published point. R8 registers exactly this. Our own contrasts keep the REES46 bootstrap unit.
3. **Beauty band coverage (66.9 % of items, 74.8 % of targets).** S8 (frozen tables) is structurally capped on Amazon: ~25 % of test targets have embeddings that never train. The N2-type rule will therefore (correctly) tend to pick S7 on Amazon and possibly S8 on ML-1M (97.2 % coverage). This is not a bias in the decision rule, but it does make "S8 − S7" on Amazon a joint coverage-plus-freezing effect, not the freezing effect the REES46 study isolates (its band sees far more of the catalogue). Report it as such (R9 coverage line + B6 already does half of this). Sports / Toys coverage must be measured before registration (R9).
4. **Rounds / exposure transfer to 6 k–35 k clients.** The sweep rule transfers mechanically (≈ 6 visits per client at P = 1, same as REES46's 6.0 EFE), so FA / FA_WARM are fine. **DP does not transfer unchanged:** with 1,024 ≤ N/6 the REES46 cohort is 17–3 % of the whole population, and the same (q, T, z) at m = 47–278 gives 4–22 × the relative noise. That is a real property of DP-FL at small N and a legitimate finding, but it must be registered as in R4, not improvised.

## What could invalidate the results or the REES46 study
- Running DP8 with "the registered z" at a different (q, T): the stated ε = 8 would be false. (R4.)
- Training any arm on the inner-validation row, or selecting on anything but inner validation: the single TEST look would be contaminated. (R2.)
- A third seed or a Track-B winner decided after the TEST look. (R6, R7.)
- Any benchmark value flowing into a REES46 selection, register or contrast; any write under runs_ds3 / contracts/fl / contracts/dp. The draft already forbids this in spirit; R8–R9 make it a rule and pin the inputs (the local tree lacks the S7 / S8 selection copies, DL 310 — the LOCK must carry the hashes).

## Top risk (not a fix; an owner decision)
Track A's central budget is 6.0 EFE ≈ 6 epochs with the REES46 LR shape. eSASRec's SAS+ ran up to 200 epochs with patience 50 on these same small sets. C_FULL may land well below SAS+ .0531 / .0314 / .0580 / .1684 for budget reasons alone, and every retention ratio is then measured against an under-trained ceiling. Track A reports this honestly ("transfer without retuning"), but the thesis reader will read it as "the recipe is weak".

## Optional improvements (not required for registration)
- O1. Add an exposure-budget axis to Track B only ({6, 24, 96} EFE, early stop on inner validation; ≈ 1 min / epoch on CPU, DL 304). It is a hyper-parameter, not a method, but it changes the owner's §14 grid, so it needs the owner's word.
- O2. Run FA_1024 (the z = 0 clipped twin, 2 seeds) beside DP8 so that K12-style "noise only" is readable on the benchmarks; without it DP8 can only be compared with FA_WARM\*, which also changes cohort and sampling.
- O3. A DESCRIPTIVE "target covered by the band" sub-population row for S7 vs S8 on Amazon, to separate coverage from freezing.
- O4. Record in B5 that HSTU / FuXi-α Amazon rows are context-only (unverified split; BM-00 §1b) so nobody promotes them later.
- O5. Report MRR@20 only in our own tables (no published counterpart) — the draft does this implicitly; one sentence makes it explicit.

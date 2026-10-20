# 13 — Open issues, limitations and risks

*As of DL 435 (2026-10-03). "R" = risk / open item to act on; "L" = limitation to disclose. Each item names where it
comes from. Items marked GAP were found while writing this handover and are not yet in the decision log — record them in a
DL row when you start.*

## 1. Open items that block or gate the next steps

| # | Item | Why it matters | Action |
|---|---|---|---|
| R1 | **GAP — `runs_p2` is not a fresh root.** `laneP2_RUN1.sh` calls `$V/runs_p2` "the FRESH root", but on the server that directory already holds the 2026-09-27 contest P2 tuning runs and logs (RESUME_HERE_prev_20260927_1730Z §1; DL 78). Sub-dir names differ (WA/, C1/, C2/, P2T3/, recipes/ …), so a collision is unlikely but unverified | mixing two studies in one root confuses provenance and the "fresh" claim is false | in the lane review: list the server dir (names only), decide keep vs a new root (e.g. `runs_ph2`) by a DL row before launch |
| R2 | **GAP — no packet for the FT_P2_WSTAR launch path.** `launch_device.py` has no P2 support (MEASURED grep); Q5 needs FT on W* (and on MAIN_TEST in the sitting) | Q5 (secondary) cannot run | write a packet + review before RUN-3 |
| R3 | WB3 / LoRA cells not wired into `launch_fl` (`P2_UPLOADS = ("dense",)`; LoRA refused) | RUN-2 cannot run WB3 / WC1–WC4 | hooks per HANDOFF_P2-01_02 §6 + Opus review + new tree |
| R4 | P2-08a (DP chain for W*) not dispatched; `launch_fl` refuses `--epsilon` / `--clip-record` for P2 arms | Q6 / Q7 / Q11 cannot run | packet + Opus review before RUN-3 |
| R5 | P2-07/13 (`reeval_p2.py`), P2-12 (`p2_report.py`, `p2_block_index.py`), the lock tool (`p2_lock.py`) not yet in `code/` / `tools/`; the RUN-1 lane has no recorded review | no anchor, no TEST scoring of P2 arms, no lock | finish + review (file 08 §1) |
| R6 | **The sitting procedure is not reviewed as a whole.** H_test.sh does not know about Phase 2; nothing technical stops someone from running H `reeval` before P2_METHODS_LOCK and register entries 7 / 8 exist | would break T0a S2 (one-sitting guarantee) | add a guard or a reviewed wrapper; one Fable / independent review of the sitting script |
| R7 | Register numbering: the endgame README plans "entry 6 (RESULT)" after H, but entry 6 is the INTEGRITY_CHECK of the build | wrong entry text | use the next free numbers; describe in a DL row |
| R8 | **LOCK_FINAL_RUNBOOK predates B12 and B16** (it says `cd $V/code_bench3`) | an S3 lock from code_bench3 would not know per-seed S records; ML-1M must use code_bench5b + `--device cuda` | confirm the commands per tree with a review before the first lock (file 06 §6) |
| R9 | K16 pass-2 lane: REV-OPS-B13-P2 minor (PASS2_HOST_LOAD written after the bench; LOAD0 before the sleep) and `W1_BUSY_RE` lacks `run_ml1m_gpu.sh` (DL 380, 430) | a failed bench blocks a rerun; pass 2 could start during ML-1M GPU jobs | fix + short re-review before launch |
| R10 | Open questions in the P2 handoffs: `p2_probe` untested on real data; the top-1 definition of criterion (b); LoRA no-op measurement; logging the client count under DP (HANDOFF_P2-01_02 §8) | P2_SCRIPT_FREEZE needs them settled | decide before the freeze (Fable touch point 3) |
| R11 | The thesis reference is behind (last capture DL 353; next source S603); structured registers pending since 09-27 | the thesis loses sourced history | capture DL 354 → latest (file 08 §9) |
| R12 | The old `tests_fl/run_all.py` pytest (pid 1073) alive for days on the server (DL 399) | resource use; may be a teammate's | identify before touching |

## 2. Scientific limitations (disclose)

- L1 One main dataset (REES46, one store, views only); external benchmarks are leave-one-out, ID-only and DESCRIPTIVE.
- L2 CALIB ⊂ MAIN, and CALIB_VALIDATION lies in the read VALIDATION window; Phase 2 selects on CALIB after that read (D-a, D-b).
- L3 VALIDATION PB values are max-of-marks; TEST was inspected at period level by earlier v1 campaigns; TEST holds Black Friday.
- L4 One θ_pre seed; PRE recipe not re-tuned at 9× data; C_FULL recipe not re-tuned at 1.35 M users; C_FULL is a larger model.
- L5 DP is central (trusted aggregator); secure aggregation not simulated (Phase 2 Part B only simulates its costs); DP noise from
  a public deterministic PRNG; the DP noise-to-signal ratio reflects the simulated scale (131,072 clients, 20.7 M parameters).
- L6 Non-DP federated updates reveal the viewed items; S8 only partly removes the leak; WB3 reveals touched item IDs.
- L7 The cold DP-compatible schedule (FA_1024) is weak (≈ 21.7 % of C_BIG), so K12 / K13 measure DP cost relative to a weak twin;
  K12 is SEED_SENSITIVE (100.70 vs 96.72).
- L8 The warm FL arms barely moved θ_pre (D1): K21 and the warm DP / Q8 / PF / FP contrasts mostly reflect a near no-op recipe.
- L9 Equal-exposure budget (6.0 EFE ≈ 3 visits per client); cold FL was still improving at 6.0 (FA_X2 declared not run).
- L10 Device-only LO is an untuned control; device panels on SPARSE / HEAVY / TRUE_COLD and the controlled cold-start studies
  (TRUNC) were declared not run; no "events to parity" claim is possible.
- L11 K16 on a 16,384-user subset (fallback); device latency only from pass 2; no phone measured (O8); battery not measured.
- L12 GRU_D128 vs C_SMALL comparison not run (the A40 reeval tool accepts SASRec variants only).
- L13 Phase 2 power: the plausible gain (~0.001) is near the MDE (~0.00047); INCONCLUSIVE is a likely outcome (D-h).
- L14 Benchmarks: device class differs by dataset (CPU vs A40 for ML-1M); benchmark θ_pre is per seed (B12); Track-A central
  budget 6 EFE vs published long training; published SAS+ / eSAS could not reach some holdout targets.

## 3. Process risks and what to watch

- **Accidental reads.** Opening a TEST output, a Phase-2 MAIN VALIDATION value or a benchmark holdout before its step would
  break the design irreversibly. Read only status fields; never `cat` result files; write the DL row before fetching.
- **Second sitting.** There is no second RETRO_TEST sitting, ever (S4). An arm not sealed at lock time is declared not run.
- **Editing frozen things.** v4eval, the frozen tools, deployed trees, running lane scripts, registered contracts. Every edit
  needs a new file / tree and a review; the byte-identity checks exist to catch silent drift (e.g. the 06F overwrite, DL 185;
  the post-pack `device_pin.py` edit, DL 180).
- **Tool definitions vs registered definitions.** Two value-free refusals at E (DL 389, 394) came from analysis code that
  disagreed with the registered writers; test fixtures with unrealistic values masked them. Use real records as fixtures
  (as FIX-REEVAL did) and dry-check value-free before freezing.
- **Concurrent writers.** Two sessions once wrote the same DL numbers and overlapping files (DL 171, 290, 377 erratum). Keep one
  DL writer and one owner per file set; check the tail before writing.
- **Time stamps.** Local time is UTC+3; several rows were mislabelled (DL 392, 414). Take UTC from the server.
- **GPU contention.** H > Phase 2 > ML-1M GPU driver. The ML-1M driver occupies both A40s until it yields; check that it
  yielded before starting a Phase-2 or H GPU job.
- **Quota / session cuts.** Agents and sessions were cut by spend limits several times (DL 38, 270, 414, 422); work survived
  because state lives in files and packets. Keep doing that; resume interrupted workers from their packets, verify shas.
- **Shared account.** Teammates' processes and files on `jovyan` are off-limits; identity is never inferred from files.
- **Schedule (INFERENCE).** Phase-2 code + runs ≈ 8–11 days before the sitting (≈ 10-17 … 10-21); ML-1M final ≈ 10-05/06;
  Wave 2 ≈ a week on CPU; K16 pass 2 after all Wave-1 drivers. If time runs short, Phase-2 drops follow the registered order
  (tertiary → exploration → C2 last), Part B is declared not run, a Wave-2 dataset can be stopped whole before its LOCK.

## 4. Gaps in the records (could not be resolved from the files)

- The outcome of the 30 Sep supervisor meeting is not recorded.
- The exact status after DL 435 of the dispatched workers (P2-07/13, P2-12, the RUN-1 lane review) and of the GitHub port task
  is unknown here; check `company/handoffs/` and the DL tail.
- The expansion of the acronym "EFE" is never written in the records (its meaning as a budget unit is clear).
- Live server state (job progress, disk) is only known up to the last DL row; verify on the server.

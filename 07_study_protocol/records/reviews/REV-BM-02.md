# REV-BM-02 — independent read-only review of the BM-02 benchmark runner (Claude Opus, 2026-10-02)

- **Scope:** `code/v4bench` (13 files), `code/tests_bench`, handoff `company/handoffs/BM-02.md`, and the server copy `$V/code_bench1`.
- **Spec:** PREG1_AMENDMENT_07_DRAFT B1–B9 (later sections win) and BM-00.
- **Hashes:** the local files equal `BM-02.sha256.txt` (25 / 25). The server `code_bench1` equals local (aggregate sha `6d512253…`).
- **Server state:** `runs_bench/<ds>/` holds only MANIFEST.json and SMOKE. No LOCK, TEST_LOOK or TEST result exists. None was opened.
- **Local tests (MEASURED):** next_item_eda venv, basetemp `.local_ai/_pytest_tmp/REV-BM-02/run`. Result: 62 passed, 17 skipped (102 s), the same as the handoff. Output is in `SUMMARY.txt`.

## Verdict: FAIL (1 BLOCKER; the fixes are small and local; re-review the delta only)

## BLOCKER
**B1. FT TEST scoring reads the wrong sequence, and it can crash after the one TEST look is burned.**
- **Where:** `device.py:93` sends `(split, users, ends, targets, rows)` without `view.source`. Then `device.py:66` rebuilds `EvalView(d, split, users, ends, targets)` with the default `source="inner"`.
- **Effect in `final_eval`:** the TEST ends are full lengths L, but they index the inner arrays of length L-1. As a result:
  - the context includes the next user's first item;
  - the validation item is missing from the context;
  - the seen filter is wrong.
- **Effect on the last user:** if the last user has a holdout row, the run raises IndexError.
- **Why this is a blocker:** the crash comes after `TEST_LOOK.json` has been written O_EXCL (`final.py:139`). The single look is then lost and only part of the results is written.
- **Reproduced (MEASURED):** a scratch script on the synthetic fixture gave:
  - an FT context ending in `28`, which is the next user's first item, where the correct context ends in `29`;
  - `IndexError` for the last user.
- **Why the tests miss it:** `test_pipeline.py:177` passes because the fixture drops the last users' holdout rows (`conftest.py`, `drop_holdout`). It also never checks the context.
- **Fix:**
  - pass `view.source` through and build `EvalView(..., source=source)`;
  - add a test that the FT TEST window and seen pairs equal `view.batch` / `seen_pairs` of the full view, with the last user in the holdout.

## MAJOR
- **M1. A crash during `final` loses the TEST look** (`final.py:139` vs `final.py:160-190`). Fix: run a preflight before `open_test_gate`.
  - Preflight: build every locked model, load each checkpoint, check the FT base paths, and score one validation row per run.
  - Then open the gate.
- **M2. Track-B max_len 200 is scored on TEST with a 50-item context.** The cause is `run.py:108`, `D.load(a.dataset, root)`. Fix: load the data per run with `max_len = spec["max_len"]`, or derive the window from the model.
- **M3. The strict LOCK conflicts with R7 and R5.**
  - `final.py:81-92` requires seeds 2026 and 2027 for every label. The non-winning Track-B cells have seed 2026 only, so a LOCK that "lists every run" is impossible.
  - S_CAL runs (`DONE_S_CAL`) are refused by `final.py:38`, and the S_RECORD files are not pinned.
  - Fix:
    - exempt the grid cells: seed 2026 only, plus seed 2027 for each `WINNER_*` cell, and verify that the winner run exists;
    - pin `dp/S_RECORD*.json` and the S_CAL RESULT files in the LOCK.
- **M4. The inner-validation population differs from R2.**
  - The manifest scores 22,363 / 35,598 / 19,412 / 6,040 users (`data.py:206-207`).
  - R2 and BM-00 register 22,308 / 35,475 / 19,312 / 6,039, after the RecTools filters.
  - Fix: apply the RecTools validation filter (the target must be in the inner-train item set), or get an Ops amendment before the first run.
- **M5. DP q = m/N, not 2⁻⁷.**
  - The spec writes `q_actual` (`recipe.py:172`), and `fl/participation.py:157` uses q = m/N.
  - The values are ML-1M .007781, Beauty .007825, Toys .007830 and Sports .007809.
  - For Beauty and Toys q_actual is larger than 2⁻⁷, so the registered ε is slightly understated.
  - Fix (Ops ruling): either recompute ε at q_actual with `fl.accountant` and report it, or use Poisson inclusion exactly 2⁻⁷ with the denominator m fixed.
- **M6. The S7 / S8 / tuning / FT factors are hard-coded mirrors, not read from the pinned files.**
  - `recipe.check_warm_winners` (`recipe.py:108`) is never called.
  - `run.py` does not check the pins at run time; the check happens only at LOCK.
  - Fix: in `_run` (non-smoke), call `check_pins(required=True)` and `check_warm_winners()`. Also compare TUNING_WINNER and FT_CELL with the files, and record the result in ARM_SPEC.
- **M7. Outputs can be overwritten.** Rerunning a finished run without `--resume` retrains it and rewrites RESULT.json and the checkpoints:
  - central (`central.py:125-127`);
  - FL (`fl_engine.py:164-169`);
  - FT (`device.py:146`, which has no guard at all).
  - Fix: refuse when RESULT.json exists with status DONE_*.

## MINOR
- **The warm PRE seed is not checked** (`fl_engine.py:78-84`), so FA_WARM s2027 can use PRE_s2026. Enforce or record the REES46 rule.
- **`decide_warm` does not check the cell** (`selection.py:84-102`). It and `seed3` do not check that the runs are the Track-A cell (6, 50, 0.1).
- **The early-stop test is tautological** (`test_dp_budget.py:76`). Assert a stop after exactly 6 flat marks.
- **`test_recipe` compares only 5 sweep arms** with `launch_fl.arm_spec`. Add the Poisson and warm-DP arms.
- **Three items are disclosed but need an Ops ruling:**
  - the mark rule floor(12E/E_tot) vs ceil(kT/12);
  - the FT TEST support excluding the validation item;
  - the C_FULL CONTROLLED_BUDGET being the early-stop mark.
- **PF / PF_WARM\* are not implemented** (`recipe.py:51`), although B2 Phase 2 lists PF.

## Verified correct
- **Evaluator:**
  - holdout users only, with 22,311 / 35,539 / 19,365 / 6,038 scored;
  - candidates = the train.csv catalogue minus the whole history;
  - ties broken by ascending id (`metrics.py:40-44`);
  - HR / NDCG / MRR formulas (`metrics.py:55-60`);
  - cold items counted as misses.
- **eSASRec validation row:** stable-descending first-in-file (`data.py:248-262`). The manifest shows it equal for all users, with 2,483 ML-1M users whose row is not the last.
- **Training excludes the validation row** for every arm, including FT (`data.py:290`, `device.py:80`).
- **The `--final` gate** requires the LOCK and the gate token, and is single-pass.
- **Recipe settings:**
  - FedAdam .01 / τ 1e-3;
  - P1 sweep and P2 Poisson, B16;
  - T 630 / 2,330 / 2,023 / 3,709;
  - m 47 / 175 / 152 / 278, z .5642, T 384;
  - S_CAL on all clients, FA_1024 twin;
  - frozen S8 tables;
  - θ_pre = PRE endpoint 6.0.
- **Track B and selection:**
  - Track-B tie order;
  - budget axis on the central arms only;
  - PRE never early-stops;
  - warm cells use the 6-EFE PRE of the same cell;
  - the FA_WARM\* rule;
  - the seed-3 trigger before the LOCK.
- **Threads fixed at 16**, and CUDA hidden.

---

# Delta review — BM-02 Revision 2 (Claude Opus, 2026-10-02, read-only)

**Inputs checked:**
- **Files:** the 26 local files equal `BM-02.sha256.txt`, and the server `$V/code_bench2` equals local (aggregate `569aeee4…`).
- **Server state:** `runs_bench` holds only MANIFEST and SMOKE. No LOCK, TEST_LOOK or TEST result exists.
- **Beauty smoke:** `smoke3.log` shows 23 / 23 commands with rc=0 and "ALL DONE".
- **Local tests:** `tests_bench`: **85 passed, 22 skipped** (135 s; basetemp `_pytest_tmp/REV-BM-02/delta`).

## Verdict: PASS-WITH-FIXES (one required fix: R1 below; the rest are minor)

| finding | status | evidence |
|---|---|---|
| B1 | FIXED | `device.py:58-71,98` passes `view.source`. `test_review.py:34` keeps the last user's holdout and compares every FT context and seen set with the full view |
| M1 | FIXED | `final.py:210-273`: verify → preflight → TEST scored in memory → `commit_look` (atomic exclusive) → writes. Tests `test_review.py:83,105` |
| M2 | FIXED | `final.py:223-229`: data / view are loaded per max_len (`_ml_of`); there is a test |
| M3 | FIXED | `final.py:82-127`: grid cells need s2026, the winner also needs s2027, S_CAL runs are accepted with the S_RECORD sha, and `_pinned_files` pins the decisions / WINNER / S_RECORD / MANIFEST; INCOMPLETE runs are refused |
| M4 | PARTLY FIXED (see R1) | `data.py:206-218` |
| M5 / Q1 | FIXED | `fl_engine.py:341-369`: u ~ U{0..127}, member iff u = 0. `build` returns `cls` (`participation.py:133`); m is the denominator only. Test `test_dp_budget.py:130` |
| M6 | FIXED | `recipe.check_mirrors` (`recipe.py:109-124`) is called for non-smoke runs (`run.py:187`) and recorded as provenance |
| M7 | FIXED | `common.refuse_overwrite` covers DONE* / INCOMPLETE* in central, FL and FT |
| PF / Q2 | DONE | the store is pinned in RESULT / LOCK (`final.py:49-53`); parity with `launch_fl` (`test_recipe.py:89`) |
| Q4 | DONE | ML-1M C_FULL B96 is refused (`run.py:183`); `--wall-cap-hours` → INCOMPLETE_WALL_CAP, which cannot be locked |
| Q5 | DONE | `final.py:231-232`: a Wave-1 run refuses a non-zero cold count before any write |
| Ops rulings | DONE | marks k·budget/12 (`central.py:108-112`); FT TEST context = full history; an early stop gives PB only with `controlled_budget: "N/A (early stopped)"` (`central.py:247-257`) |
| minors | DONE | the PRE seed equals the run seed (`fl_engine.py:84`); the Track-A cell is asserted; the early-stop test runs k = 3 and 6 |

## R1 (required): the M4 ML-1M rule is not the eSASRec / RecTools rule
**The code (MEASURED, read on the server):**
- RecTools `_filter_train_interactions`:
  - keeps users with ≥ `train_min_user_interactions` (default 2, `base.py:212`);
  - then keeps `tail(session_max_len + train_session_max_len_addition)`.
- `SASRecDataPreparator` sets `train_session_max_len_addition = 1` (`sasrec.py:77`).
- A validation target is kept only if its user and its item are in the truncated inner-train maps (`data_preparator.py:224-232`).
- eSASRec `configs/paper/loo/ml_1m.yaml:9,20-22` uses `cls: SASRecModel` with `session_max_len: 200`.
- `s3_vanilla.yaml:13,22-24` uses `SASRecModel` with `session_max_len: 50`.

**The faithful rule:**
- tail 51 for S3: correct, giving 22,308 / 35,475 / 19,312;
- **tail 201 for ML-1M, giving 6,038, not 6,039.**

**What happened:**
- R2's 6,039 is BM-00's untruncated figure, and BM-00 itself flagged 6,038 under truncation.
- "No truncation for ML-1M" fits the registered number, not the code. That conflicts with B1, "the rule is the code".

**Fix:**
- set `tail = 201` for ml_1m;
- change `VAL_POP["ml_1m"]` to 6,038;
- have Ops log a one-line R2 correction (6,039 → 6,038) before the first ML-1M run.

**Impact:** one user, so it is MINOR in size but required for fidelity.

## Minor
- **N1:** `final.py:246` calls `get_view` inside the preflight loop. A preflight failure on run k > 1 then happens after the holdout has been read. Fix: move every `get_view` after the preflight loop.
- **N2:** a TEST-pass crash can be retried, by design (`test_review.py:83`). Fix: append a `TEST_ATTEMPTS.jsonl` line before the first holdout read, so that retries can be audited.

---

# Delta review 2 (B12) — BM-02 Revision 4 (Claude Opus, 2026-10-02, read-only)

**Scope:**
- the server `$V/code_bench4` vs `code_bench3`;
- `company/plans/ops_scripts/bench/run_wave1.sh` = the deployed `runs_bench/run_wave1_r4.sh` (sha `f8f0d81b…`, equal after LF normalisation).

**Spec:** `PREG1_AMENDMENT_07_ADDENDUM_B12.md`.

## Verdict: PASS (no BLOCKER; no STOP)

| check | result | evidence |
|---|---|---|
| The only diff is B12 | YES | Only `final.py`, `fl_engine.py` and `recipe.py`, plus 3 tests, differ. The r6 tree (`fl`, `v4models`, `v4train`, `v4sup`, `snap_v1`, `ppsi_env`) is identical |
| S_CAL per PRE seed | YES | `recipe.py:176` allows seeds 2026 and 2027. `fl_engine.s_record_path` writes `dp/S_RECORD_<VARIANT>_s<seed>.json` (write-once), bound by init_digest = θ_pre of the same seed (the PRE seed must equal the run seed, `fl_engine.py:84`) |
| Own-seed binding; a foreign seed is refused | YES | `load_s_record(..., seed=)` refuses `obj.seed != run seed` on top of the init_digest check. The pre-B12 s2026 record has `seed` 2026 (MEASURED, Beauty) |
| z / q / T / δ / m unchanged | YES | `recipe.DP` and `_exact_q_plan_cls` are not in the diff |
| LOCK | YES | `final.py:115`: an S_CAL label needs seeds {2026, 2027}. `_pinned_files` globs `dp/S_RECORD*.json`, so both records (old s2026 name + `_s2027`) are pinned. The S_CAL RESULT points to its record and that sha is in the run record |
| `--resume` | YES | Same `STATE=$RUNS/_wave1_state` as the code_bench3 driver. The `.fail` markers are cleared, so the 4 failed jobs (fa1024 / dp8 s2027 on Beauty and Sports) and their dependents are retried. Jobs with a DONE file are skipped. `refuse_overwrite` blocks any rerun of a finished run. The refused run dirs are empty, so the run starts fresh. `scal.<D>.2026` is skipped via the DONE_S_CAL RESULT. `srec` falls back to the old s2026 name |

## Minor (no action needed before the lane)
- An `INCOMPLETE_*` run is retried by `--resume` and then refused by `refuse_overwrite`. Its dependents are marked SKIPPED. This is safe but noisy. Ops should decide each such case explicitly.
- `--resume` is injected into every `run` command. A partial run left by the code_bench3 code with a different spec would be refused and not restarted. This is safe, because the lane starts only after the code_bench3 driver exits.
- Disclose that S_CAL s2027 also changes the Poisson draw seed of the calibration, not only θ_pre. This is consistent with "same configuration".

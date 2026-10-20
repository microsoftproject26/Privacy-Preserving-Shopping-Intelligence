# REV-FL-16 — independent read-only review of FL-16 (the warm method set FP / PF / FA_Q8 / S_CAL / FA_1024 / DP from theta_pre, S7- and S8-bound)

Reviewer: Claude Opus 5.5 (read-only agent), 2026-10-01.
- I edited no code, contract or test. I used no server, GPU, real data, pip, git or Codex.
- Temporary files went to the session scratchpad and to `code/_pytest_tmp/REV-FL-16/` (removed afterwards).

Authority read:
- 06N N1–N4 (N2, N3.2)
- 06K §1–§5 (S7 / S8, DP8_FROZEN)
- 06M_B B2 / B4
- 06L_E E1
- 06C (Poisson DP accounting)
- 06L_C C8
- DECISION_LOG rows 261–283
- handoff `company/handoffs/FL-16.md`

Baseline: `deploy_archive/r5/code_ds3r5.tar`. I read it with `tar -xOf` into the scratchpad only.

Reviewed (sha256 MEASURED, equal to the handoff):
- `code/v4int/launch_fl.py`: 3c0d91b5… (r5) → **5b885b22…**
- `code/tests_int/test_fl16.py`: **60a38bbe…** (new)

## VERDICT: **PASS-WITH-FIXES**

- There is no BLOCKER and no MAJOR. The 12 new arms run their cold arm's semantics from theta_pre, with their variant's registered factors and their own S record. Every existing arm is unchanged: MEASURED independently against the r5 file, see §4.
- There are six MINORs.
  - None blocks packing r6 or the first warm-line smokes.
  - m1 and m3 are operational gates Ops should apply before the first MAIN FA_1024_WARM / DP_WARM launch and before any FT_FA / FT_C launch after r6.
  - m4 must be done before any warm-line row is indexed (ANA).

## 1. The 12 warm arms = the cold arm + θ_pre + the variant's factors (checked)
- **Label table.**
  - `WARM_LINE` (launch_fl.py:207-213) maps each label to (cold arm, variant).
  - The helpers are `base_arm` / `warm_variant` / `is_warm_arm` / `is_frozen_arm` / `is_sweep_arm` / `is_poisson_arm` (:279-304).
  - Every cold-arm branch of `arm_spec` (:857-885), `execute` (:1441-1466, :1563) and `preflight` / `resolve_spec` / `run` switches on `base_arm`.
  - `METHOD_CODE` takes the cold code (:216), so the ARCHITECTURE_LOCK check is unchanged.
  - CALIBRATION is refused for every new arm, because `CALIB["arms"]` is unchanged (:236, :845).
- **θ_pre.**
  - These functions are AST-identical to r5 (MEASURED with an `ast.dump` comparison): `load_pretrain_init`, `apply_init_state`, `check_init_identity`, `pretrain_identity`, `s_record_obj`, `load_s_record`, `load_warm_selection(_2d)`, `load_fl_tuning`, `bind_main_selection`, `spec_tuning`, `spec_digest`, `clip_log_applies`, `train_loss_applies`, `clip_round_record`, `load_z`, `fl_rows`, `warm_row_labels` and `warm_server_opt`.
  - Only the membership tests changed: `is_warm_arm` at :835, :890, :1741, :2238, :2484 and :2613.
- **Factors.**
  - `bind_warm_selection` (:2405) binds the variant's registered file: S7 → `FL_WARM_SELECTION_S7.json`, S8 → `_S8.json` (:2412-2424). The file's arm must be the variant.
  - `warm_factor_value` (:904-914) uses the variant's grid.
  - `warm_preflight` (:1732-1764) refuses these on a non-smoke MAIN run: a missing selection, CLI factors, and a selection on a smoke run.
  - Tested for all 12 arms, including the other variant's file, another θ_pre and the dry-run digest.
- **Frozen variants.**
  - `is_frozen_arm` → `frozen_item_tables` (:899-900); `--freeze-item-tables` is required iff the arm is frozen (:1746).
  - `execute` freezes before `Server` reads the manifest (:1431-1439). fl.dp clips and noises the manifest's shared keys only (fl/runtime docstring; spy test test_fl16:380-396).
  - Upload, clip and noise therefore exclude the tables. The bytes count uploaded params only, plus the tables once per client.
- **DP accounting is unchanged.**
  - Same `load_z` (ε8 → 0.5642, ε1 → 1.3094).
  - Poisson plan: group = denominator = 1,024 → q = 1,024 / 131,072; T = 384; δ 5e-6.
  - The N check now also covers DP_WARM* (:2538). The DL 219 P2 / B16 rule applies through `is_poisson_arm` (:489, :568, :582).
  - The noise dimension does not enter the RDP accountant, which is correct for the Gaussian mechanism.
  - `RUN_SUMMARY.dp_noise_dimension` is recorded for warm DP-configured runs (:1540-1545). It is structural, not data-dependent, so it is compatible with 06L_E.
- **PF.**
  - `cfg["pf"]["lr"] = base` (:865) is never multiplied: the warm factor scales `lr_peak_local` / the solver LR only (:892-895).
  - PF_WARM* uses the PF evaluator (:2564).

## 2. Warm S record (checked; the plan binding is unit-tested only → m1)
- **Writing.**
  - S_CAL_WARM* writes `S_RECORD_WARM[_FROZEN].json` write-once in MAIN (:1563-1584) and the smoke record in `_finish_smoke` (:1634-1641). Neither ever writes `S_RECORD.json`.
  - The cold summary is unchanged (tested).
- **What `load_warm_s_record` (:746) refuses:**
  - the cold format ("cold-start");
  - an unverified self-sha;
  - a smoke record or seed ≠ 2026 for a real run;
  - the other variant's arm;
  - another θ_pre (`check_s_record_init`, :776);
  - another frozen set, spec sha or RunConfig digest (`check_s_record_config`, :783). `ckpt_every` is outside `RunConfig.digest` (fl/runtime.py:289-297), so recomputing at 64 is correct.
- **`warm_s_cal_spec`** (:734) = the variant's S_CAL arm at seed 2026 with the run's tuning point, DL 219 record, warm factors and θ_pre.
  - Reviewer probe (MEASURED): the spec that `resolve_spec` gives a real S_CAL_WARM / S_CAL_WARM_FROZEN launch (`--fl-tuning` with the DL 219 rule, non-default factors) **equals** `warm_s_cal_spec` of the DP / FA_1024 run with the same flags.
  - A record written from that spec binds FA_1024_WARM* and DP_WARM* at seeds 2026 and 2027 through `resolve_spec`, and the other variant's DP arm refuses it.
- **Plan binding.** `check_warm_s_plan` (:798) rebuilds the seed-2026 Poisson plan from this run's keys and `ctp.fingerprint`, the same arguments `execute` uses (:1429, :2601). It is correct by reading and in unit tests, but it is called only in `run` (:2540-2542). → m1.
- **The cold side.** The cold FA_1024 / DP refuse a warm record through the unchanged `load_s_record` format check (tested).

## 3. PF / Q8 / 06L_E (checked)
- **Q8 × frozen.**
  - `bytes_record` merges fl.comm's frozen fields into `q8_visit_bytes` (:1142-1145); the old KeyError path is gone.
  - Probe (MEASURED): the frozen fields and download equal `fl.comm.visit_bytes(frozen manifest)`. `send_plus_receive = shared_bytes + shared_numel + 4·n_tensors + 8`, over the uploaded keys.
  - `install_q8` (FA / no DP / no PF / mu 0) is unchanged and admits FA_Q8_WARM*.
- **06L_E E1.** The gates are unchanged. DP_WARM* write no CLIP_LOG and no TRAIN_LOSS; FA_1024_WARM* write both; S_CAL_WARM* write TRAIN_LOSS only (tested).

## 4. Existing arms unchanged (MEASURED)
- **Old-vs-new probe** (the r5 `launch_fl` loaded from the tar copy with importlib, 3c0d91b5):
  - The 145-pair table computed by the **old** module equals the new one and the pinned sha 2f4d8990….
  - I ran the 7 cold arms and FA_WARM / FA_WARM_FROZEN, each at the default point and at the S4 point (FedAdam + FL_CONST + client SGD), through old and new `execute`. The results are identical:
    - the whole RUN_SUMMARY (except wall time);
    - TRAIN_LOSS.jsonl, CLIP_LOG.jsonl, S_RECORD.json and RESULT_ROWS.json, byte for byte.
- **Unchanged files.** lr_screen 1a391b1e, launch_device 1968ca95, fl/runtime e627e943, dp 516301a9, quant 8f375501, comm 2f8bd172.
  - Every file of the r5 tar equals the tree except launch_fl.py and the ANA-05 files: reeval 615f1c60 and 5 tests_ana files (DL 272 / 280, reviewed separately).
  - The new files are test_fl16.py plus the ANA-05 tests.
- **Identities.** v4eval `evaluator_code_sha256` = 61b64e61…; `device_code_sha256` = 4783f58f…. `fl_code_sha256` = 2caff93c… (changed as expected; see m6).
- **Other readers of launch_fl constants.** lr_screen reads ARMS / WARM_ARMS / WARM_STAGE / WARM_FACTORS / POISSON_ARMS (all unchanged). The only launch_fl definition-level change is `ALL_ARMS`, which no other module reads.

## 5. Tests (next_item_eda venv, `-B -p no:cacheprovider -q`, basetemp on F:; suites run one at a time)
| suite | result (MEASURED) |
|---|---|
| tests_int (full) | **350 passed, 123 xfailed**, exit 0 (= handoff) |
| tests_fl (full, alone) | **248 passed, 75 xfailed**, exit 0 (= handoff / FL-15 baseline) |
| `test_fl16.py --runxfail -k test_nc_` | **7 / 7 fail on AssertionError** (junit messages read: each is "did not raise", or the mutated file / keys observed) — none vacuous |
| reviewer probes (scratchpad, 4 tests) | 4 passed: warm smoke S record, S_CAL spec equivalence, Q8 frozen bytes, old-vs-new invariance |

- An earlier tests_int run with basetemp on C: failed in 124 tests. The cause was environmental: `fl.checkpoint.DiskRefused` (C: has 7.9 GiB free, below the 8 GiB floor). It is not a code defect; the rerun on F: is the result above.
- **Receipts.**
  - Before my tests_fl run, three receipts already differed from r5: LO_DETERMINISM_1024, LO_DETERMINISM_1024_reverse_subprocess and T34_POOL_RUNS. The likely cause is a test run during the session restart.
  - After tests_fl, I restored all differing `code/tests_fl/receipts/*.json` from `deploy_archive/r5/code_ds3r5.tar` (`tar -xOf`). Now 0 / 8 differ from r5 (MEASURED).
- **Quality of the NCs.**
  - NCs 1–6 mutate the new FL-16 wiring.
  - NC 7 (`train_loss_applies` → True) mutates an unchanged FL-15 gate. It proves the gate covers the warm DP arms, but it is not a new FL-16 path.
- **Coverage gaps:** m1 and m2.

## Findings

### BLOCKER
None.

### MAJOR
None.

### MINOR
**m1 — The plan-digest binding of the warm S record is exercised only in `run` and is unit-tested only.**
- Where: `launch_fl.py:2540-2542` (the call); `:798-807` (`check_warm_s_plan`).
- Scenario: a wiring error at the call site would show up only on the server. For example:
  - a wrong manifest hash argument;
  - the `format == S_FORMAT_WARM` gate never true.
- The dry run does not check the plan either (it has no client index).
- Minimal fix (Ops, no code change): before any MAIN FA_1024_WARM* / DP_WARM*, accept the first server smoke of FA_1024_WARM only if its `LAUNCH_1.json` has `s_record.plan_digest_checked` == `s_record.plan_digest`.
- Optional (next tree): a test that drives `run` up to `check_warm_s_plan` with the provider stubbed.

**m2 — The smoke branch of the warm S record in `_finish_smoke` is not covered by test_fl16.**
- Where: `launch_fl.py:1634-1641`.
- My probe (MEASURED) shows it works:
  - `SMOKE_S_CAL_WARM[_FROZEN]` writes `S_RECORD_WARM[_FROZEN].json` labelled SMOKE_NOT_EVIDENCE, and no `S_RECORD.json`;
  - it binds FA_1024_WARM / DP_WARM_FROZEN with `smoke=True` and is refused with `smoke=False`.
- Fix (next edit): add that probe as a test.

**m3 — `launch_device.refuse_warm_base` recognises only FA_WARM*. An FT_FA / FT_C base that is a warm-line checkpoint (FP_WARM, PF_WARM, FA_Q8_WARM, DP_WARM, …) is not refused.**
- Where: `launch_device.py:1029` (`WARM_RUN_ID_PREFIX = "FL_FA_WARM"`) and `:1052` (`arms & set(LF.WARM_ARMS)`).
- This reopens REV-FL-15 m3 for the new labels: a warm-started model could enter the FT_FA contrast.
- `load_warm_base` (FT_FA_WARM*) is safe, because it requires arm == FA_WARM / FA_WARM_FROZEN.
- No such run is registered. The handoff disclosed this (open item 2) and kept `device_code_sha256` unchanged.
- Fix:
  - now (Ops rule): no FT_FA / FT_C `--base` from any `*_WARM*` run;
  - next device tree: `warm = … or any(LF.is_warm_arm(a) for a in arms) or "_WARM" in rid.upper()`.

**m4 — The downstream tools do not yet know the 12 labels, and DP8_FROZEN is named DP_WARM_FROZEN + ε.**
- Where: `default_run_id` (`launch_fl.py:2159`) gives `FL_DP_WARM_FROZEN_eps8_SMALL_s2026`.
- ana_block_index / retention_report expect a `DP8_FROZEN` run with arm `DP` + frozen tables. Their `WARM_INIT_ARMS` lacks the new labels, and reeval has no source for them.
- This fails closed (refusal), so nothing is silently mis-indexed. But no warm-line row can be analysed until an ANA task adds this mapping:
  - DP_WARM_FROZEN + ε8 → DP8_FROZEN, ε1 → DP1_FROZEN;
  - DP_WARM + ε → DP8_WARM / DP1_WARM;
  - the 10 others as they are;
  - N1 K-contrasts;
  - reeval sources.
- Ops should also use the handoff's explicit `--run-dir` names. Disclosed as open item 1.

**m5 — `load_warm_s_record` does not validate the record's S content (S finite > 0, n_norms > 0, rounds_0based = the registered rounds, rule).**
- Where: `launch_fl.py:746-773`.
- It mirrors the cold `load_s_record` (:697-706), and the self-sha only guards integrity. A hand-edited, re-signed record with the right digests but a wrong S would bind.
- The risk is low: Ops copies the file write-once and LAUNCH records `file_sha256`.
- Optional hardening (both loaders, next tree): check the S fields against the registered S rule.

**m6 — Cosmetic / operational notes.**
- `launch_fl.py:307`: `PRETRAIN ={` lost a space (AST-neutral).
- `fl_code_sha256` changes (2caff93c…), so any in-flight r5 FL run refuses `--resume` under r6 (fail-closed, 06E §8). Deploy r6 beside r5, and keep the r5 lanes (FA_WARM* MAIN, cold FA, S7 / S8 screens) on their own tree.

## What Ops can do next
1. Pack r6 with launch_fl 5b885b22 + test_fl16 60a38bbe (receipts = r5).
2. Apply the m1 smoke-LAUNCH gate and the m3 rule.
3. Dispatch the ANA mapping task (m4) before any warm-line row is indexed.
4. Add the m2 test (and optionally the m5 check) in the next launch_fl edit.

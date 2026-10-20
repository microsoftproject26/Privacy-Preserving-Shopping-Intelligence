# REV-FL-14 — independent read-only review of FL-14 (06L A1 / A2 / A4 in the FL tuning selector)

Reviewer: Claude Opus 5.5 (read-only agent), 2026-09-30. No code or contract edited; no server, GPU, real data, pip, git or Codex.
Authority read: 06J §1–§4, 06L A1–A4, 06L_D D1–D4, 06L_E E1, task `company/tasks/FL-14.md`, handoff `company/handoffs/FL-14.md`.
Pre-task baselines: `deploy_archive/code_ds3r3.tar` (sha 15d8c994…, = SHA256SUMS) gives lr_screen b1e243f4 / launch_fl 1ef778d2 /
the 3 old fixtures (hashes match the handoff "old" values). `deploy_archive/code_ds3s2.tar` (ccda9d64…) = the older S2 screen tree
(launch_fl f2840329, lr_screen e9418f64). Both extracted into the session scratchpad only, never over the tree.
Reviewed files (sha256 MEASURED): lr_screen.py e81667d3…, launch_fl.py a23d5b4f…, test_fl14_06l.py 6c9570bf…,
test_fl10_fedadam_launch.py 0b241836…, test_fix_delta01_fl_binding.py 49c3ff55…, test_fix_delta02.py af0f88ba… (all = handoff).

## VERDICT: **PASS-WITH-FIXES**

There are no BLOCKERs. S2 / S3 / S4 selection and `load_fl_tuning` follow 06L_D D1–D3. **Deployment question 6 is clean**: select-joint
on the FL-14 tree will not refuse the live runs because they came from different trees, and it should not refuse them (evidence in
§6). There is one MAJOR: the z > 0 clip-count path is not 06L_E-conformant, though today it is dormant. It is a 2-line fix and should
go in before the new tree is packed. There are four MINORs.

## Findings

### BLOCKER
None.

### MAJOR
**M1 — The launcher is built to record a clipped count at z > 0, against 06L_E E1 ("z > 0 runs record nothing about clipping").**
- `launch_fl.py:973-978` (`clip_round_record`): when `noise_multiplier != 0` and the round info holds `n_clipped`, the function
  returns `{"round", "n_clipped"}`. `execute` then appends that to CLIP_LOG.jsonl (`:1178-1180`) and to RUN_SUMMARY (`:1197-1199`).
  The module docstring (`:125-130`) says "z > 0 -> round + n_clipped only". `test_fl14_06l.py::test_clip_round_record_units`
  asserts `clip_round_record(z = 0.56, n_clipped = 2) == {"round": 4, "n_clipped": 2}`. That test pins the non-conformant behaviour.
- Today nothing is written, because `fl/dp.py:204-206` releases `n_clipped` only for z = 0 (MEASURED:
  `test_dp_and_fa_write_no_clip_log` passes). The failure scenario: `fl/dp.py` is changed later (the handoff itself filed a
  "REQUEST to Ops" for exactly that change). A DP run would then silently write an un-noised, data-dependent count outside
  the RDP accountant, which is the thing 06L_E forbids.
- Timing: 06L_E was registered at about 22:55Z, after the FL-14 code (mtime about 22:04Z), so this is not a worker error. It is
  still a mismatch between the frozen code and the registered rule.
- Minimal fix:
  - In `clip_round_record`, return `None` unless `float(info["noise_multiplier"]) == 0.0`, and drop the z > 0 branch.
  - Change the docstring at `:125-130` and the RUN_SUMMARY `authority` string at `:1010` to cite 06L_E.
  - Change the unit test to assert that `clip_round_record(z > 0 with n_clipped present) is None`. Add an NC ("a z > 0 round with a
    released count is logged") as a strict xfail.
  - The change touches launch_fl, so fl_code_sha256 changes. Apply it **before** the FL-14 tree is packed, so the S3 cells are not
    split across two more trees. Selection validity does not depend on it either way (§6).

### MINOR
**m1 — A crash between the in-step checkpoint and the log append can drop one CLIP_LOG line without detection.**
- `fl/runtime.py:580-581` saves `latest` inside `step()`, at cursor % ckpt_every == 0, before `execute` appends the round's
  line (`launch_fl.py:1178-1180`).
- If the process dies in that window, the resume keeps rounds < cursor (`:1154`, `trim_clip_log`), so round cursor − 1 is never
  re-run and its line is missing. `clip_log_summary` (`:1001-1010`) checks only strictly increasing rounds, not completeness.
- Fix: in `clip_log_summary`, pass `run` and require the logged rounds to equal
  `[p["round"] for p in run.participation_log if p["n_survived"] > 0]`. The participation log is checkpointed. Raise on mismatch
  (fail closed) or record `complete: false`.
- Low probability. It is descriptive data (06L_E uses it as a proxy), so no selection is affected.

**m2 — Residual S3 tie between two FedAdam cells falls to the label (FL-14 interpretation 4).**
- `lr_screen.py:219-221`. When the S2 winner is FedAdam with η_s ≠ 0.01 or τ ≠ 1e-3, a 06J SGD cell on the S2 winner and an A2
  cell at the same client lr can tie inside the band.
- Probe (MEASURED): S2 winner FedAdam η 0.1 ×8, SGD lr 0.1 tie → band `[A2 η 0.01, 06J η 0.1]`, so the A2 cell wins.
- 06L_D D2 gives no rule past "FedAvg family before FedAdam". The label order happens to match the 06J §1 order (smaller η_s,
  then larger τ). A different reading ("06J cells on the winner before A2 cells", i.e. the ladder's build-on-the-winner
  principle) is equally possible.
- Fix: none in code. Register or disclose the residual order (label = smaller η_s, then larger τ) in a DL row **before the S3 read**.

**m3 — S2: FedAvgM m 0.5 against FedAvgM m 2 inside the band is decided by the pre-existing "smaller m" key.**
- `lr_screen.py:213-215`. Both have |log2 m| = 1, and 06J §1 / 06L A1 give no order between them. The code uses the pre-existing
  key element `float(lr_mult)`, so m 0.5 comes before m 2 (probe MEASURED). That matches the 06F §3 precedent ("then the smaller
  m"), but it is not written in 06J or 06L.
- The case is reachable only when the A1 edge fires. Fix: disclose it in the same DL row as m2. No code change.

**m4 — Exact raw ties in the A1 condition (interpretation 1).**
- `lr_screen.py:616-619` uses `_raw_best`, which returns the first maximum in required order (m*, 1, 2). An exact float tie
  between FedAvgM m* and m 1 therefore gives no A1 edge (probe MEASURED).
- This is consistent with the existing 06J η edge (same helper) and practically unreachable. A more conservative reading would
  run m 0.5 whenever m 1 is among the raw best.
- Fix: disclosure only. Optionally, `EdgeRunsMissing` could fire on an exact tie.

### Deployment notes (not defects)
- Once an S2 / S3 file is written by the FL-14 tree, **every consumer of FL tuning files must run on the FL-14 tree or later**. This
  covers select-joint S3 / S4, `--fl-tuning` / MAIN binding, S_CAL / FA_1024 / DP, and select-warm. `code_ds3r3`'s
  `load_fl_tuning` would refuse these files: required / edges do not match its 06J-only sets. That is correct fail-closed
  behaviour.
- The current local `code/` also carries changes from other workers (v4base/run.py, v4dev/dump_rows.py, launch_device.py,
  launch_trunc.py, reeval.py). None of these is imported by the FL CALIB path (launch_fl → fl/*, v4int.launch_central). fl/* is
  byte-identical to code_ds3r3. Check the pack's file list anyway.
- 06L A3 (DP_OPT_SENS for any non-FedAvg winner) has no code. DL 237 records it as an Ops launch rule, so it was not reviewed here.

## Check-by-check evidence

### 1. S2 set, A1 edge, order, EdgeRunsMissing: CORRECT
- REQUIRED = the 06G / 06J FedAdam grid + FedAvgM at m*, then `_append_new` FedAvgM m {1, 2} (dedupe by label)
  (`lr_screen.py:563-566`, `:541-549`). Tested for m* = 8 / 1 / 2 / 0.5 (`test_s2_required_sets`).
- A1 edge (`:616-620`):
  - It considers only the required FedAvgM cells, finite ones only (`_raw_best` `:523-525`), and fires iff the raw best is m 1.
  - No group is formed when m 0.5 is already required (m* = 0.5).
  - Raw values are used: the edge fires even when m 1 is below the incumbent (`test_s2_a1_fedavgm_m1_best_requires_m05`).
- Order: η group appended (`:612-615`), then A1 (`:619-620`). τ waits (pending) until every condition-true edge point is present
  (`:621-624`). The τ pool is incumbent + required + those edge runs, non-excluded, with 06J §4(ii) via `stage_decision`
  (`:625-631`).
- The η condition and the A1 condition each read only their own base cells, independent of each other, as D1 requires.
- Probe with m* = 1 and FedAvgM m 1 best: needed `fedavgm|m=0.5`.
- Flags (`launch_flags` `:650-669`) come out as `--server-opt fedavgm --lr-mult 0.5`. `server_opt_value` (`launch_fl.py:280-283`)
  accepts fedavgm without `--server-lr`, and 0.5 ∈ LR_MULTS.
- An unrequested m 0.5 / η / τ run is refused as "not allowed" (`:703-707`, `test_s2_live_set_all_edges_false`).

### 2. S3 set, SGD edge, tie order, interpretations: CORRECT (m2 residual)
- REQUIRED = 06J cells on the S2 winner (AdamW × FL_CONST at the winner's m; SGD m 1 × {0.01, 0.1, 1.0}), then `a2_cells()`
  (FedAdam η 0.01, τ = FEDADAM_HYPER tau 1e-3, m 1, fl_const, sgd_m, lr {0.1, 1.0}) deduped (`:533-538`, `:570-574`).
- Dedupe happens exactly when the winner is FedAdam η 0.01 τ 1e-3 (at any m, because the SGD cells are at m 1). A τ 1e-5 winner
  is not deduped (`test_s3_required_sets_per_s2_winner`: 6/6/4/4/6/6 cells).
- SGD edge (`:633-646`): it takes the required sgd_m cells whose (optimizer, FedAdam η, τ) equals the incumbent's. These are
  exactly the 06J SGD cells, because an A2 cell can share the server point only when it IS a 06J cell.
- Tie key (`:219-221`): incumbent → AdamW → client lr → FedAvg family → label, the D2 chain in order. For 06J-only sets the new
  element is constant across non-incumbents, so the old behaviour is unchanged.
- Interpretations (handoff §"Interpretations"):
  - (1) Excluded cells are not ranked: literal (06J §4 v). The exact-tie part is not the only reading: see m4.
  - (2) m* = 0.5 forms no group: literal ("a cell that coincides … is one run").
  - (3) "06J SGD cells = those on the S2 winner's server optimiser": literal. It is the D2 parenthetical, and in the dedupe case
    the shared cell is a 06J cell.
  - (4) FedAvg family = FedAvg + FedAvgM: literal. Moot in practice, because FedAvg and FedAvgM can never both be non-incumbent
    S3 cells. The residual label order is unregistered: see m2.
  - (5) The edge rule strings are unchanged, so 06J-only situations write identical `edges` records.
  - None of these departs from the most literal reading. Only the unregistered residuals in m2 / m3 / m4 need disclosure.

### 3. S4 on an A2 winner; `load_fl_tuning`: CORRECT
- `stage_required("S4", …)` (`:568-569`) builds on any point. Test: S4 on `A2 lr 0.1` gives `|passes=1`, `|passes=4`, `|batch=8`.
  `poisson_tuning("DP", …)` returns the A2 point with 2 passes / batch 16.
- `load_fl_tuning` (`launch_fl.py:1474-1479`) recomputes `required` via `stage_required` and `edges` via `check_stage_runs`, the
  same functions select-joint uses. The only change to it is its docstring (AST diff against r3).
- Tests confirm the refusals:
  - a 06J-only S2 file ("required");
  - a dropped A1 edge ("edge") and an empty edges record;
  - a 06J-only S3 file.
- The 06G grid check (`:1486-1494`) is unaffected: FedAvgM cells do not enter it.

### 4. Nothing else changed: CONFIRMED
- lr_screen: the full diff shows only the hunks above. MIN_GAIN 0.002 / TIE 0.0005 / `stage_decision` / `_strip_lr` / `read_run`
  / write paths (O_EXCL) / `_canon` self_sha256 are untouched.
- launch_fl: the AST comparison against r3 shows `arm_spec`, `spec_tuning`, `tuning_config`, `tuning_label`, `row_labels`, `fl_rows`,
  `spec_digest`, `digest_body`, `fl_run_config`, `device_record`, `load_lr_selection`, `check_tuning_for_main`,
  `bind_main_selection`, `resolve_spec`, `run`, `_finish_smoke`, `warm_row_labels`, `check_warm_spec`, `pretrain_identity`,
  **`load_pretrain_init`, `apply_init_state`, `check_init_identity`** all byte-identical. Only `execute` (A4 hooks) and
  `load_fl_tuning` (docstring) differ. No constant changed.
- The pinned default digests and the trajectories of FA / DP / FA_1024 pass (see §7).

### 5. CLIP_LOG: CORRECT for z = 0; z > 0 see M1
- Round index: `out["round"] = r` = the cursor before the increment (`fl/runtime.py:557`, `:570`). Resume keeps rounds < cursor,
  which is correct.
- Counts: `n_clipped` = the count of factor < 1 over `agg.norms`, which holds only clients with n_consumed > 0 (`fl/dp.py:118-129`,
  `:205`). `n_deltas = n_survived − n_noop = len(norms)` (`launch_fl.py:975-977`), which is consistent.
- Fields are {round, n_clipped, n_deltas, clipped_fraction}; no norms. S_CAL (clip_norm None) writes nothing. A fresh start clears
  the file (`:1160`), resume truncates it (`:1154`), and RUN_SUMMARY gets the `clip_log` sha (`:1197-1199`).
- Test: a resumed run with a stale line gives a file byte-identical to the uninterrupted one. Gap detection: m1.

### 6. CRITICAL: cross-tree select-joint: **NO REFUSAL, and none is warranted**
- The code hash is not part of anything select-joint compares:
  - `fl_code_sha256` is written only to ARM_SPEC.json's side record `code_sha256` (`launch_fl.py:1133-1135`). It is checked only
    on `--resume` (`:1148-1150`).
  - `read_run` reads from ARM_SPEC.json only `arm_spec_sha256`, `arm` and `device` (`lr_screen.py:320`, `:356`).
  - The spec itself (`arm_spec` `launch_fl.py:681-696`) carries no code or tree hash.
  - `selection_stage` compares `_strip_lr(spec)` across runs (`lr_screen.py:744-748`), plus manifest, lr_peak_base and device.
- MEASURED:
  - I built the FA / CALIBRATION / seed 2026 arm spec for all 13 relevant points (incumbent ×8; FedAdam η {0.01, 0.1} × m {1, 8};
    FedAvgM m {0.5, 1, 2, 8}; both A2 cells; 06J S3 AdamW / SGD samples), with fused None and True.
  - Each was built with `arm_spec` of code_ds3s2, of code_ds3r3 and of the FL-14 tree. **All 26 specs are identical dicts with
    identical `spec_digest` across the three trees.**
  - `_strip_lr` (FL-14) maps every point, mixed across trees, to one reference per fused class.
  - The current `spec_tuning` / `check_warm_spec` accept every old spec.
  - `fl_rows` and `device_record` are AST-identical across all three trees. `row_labels` / `tuning_label` differ from ds3s2 only in
    the FL-13 lines that fire for non-default passes / batch, which no S2 / S3 cell uses.
- Trajectory: FL-14 changes no file under fl/ (diff against r3: none). The `execute` changes are logging only (`dp` is None for
  sweep arms). FA / DP / FA_1024 digests are equal to PRE_FL10_TRAJ.
- ds3s2 → r3 comparability (fl/comm.py, fl/runtime.py FL-12/13 at defaults) was established by REV-DELTA-02 (DL 220) and is not
  re-opened here.
- Conclusion: the 5 ds3s2 S2 runs, the 2 ds3r3 S2 runs, the 2 ds3r3 A2 runs and later FL-14-tree S3 cells are accepted together,
  and correctly so. The selection must be *written* by the FL-14 tree (ds3r3's lr_screen refuses the A1 / A2 cells).

### 7. Tests: fixtures not weakened; NCs not vacuous; counts reproduced (MEASURED)
- Fixture diffs against r3:
  - The additions are the 06L cells: FedAvgM m {1, 2}, A2 cells, and `A1_LOW` values that keep the A1 condition false where the old
    scenario had no A1 edge.
  - The hard-coded counts `6 + edges` / `5 + edges` / `5 + len(_e)` became `1 + len(required) + edges`. This is self-referential,
    but `test_s2_required_sets` pins the lists explicitly.
  - The S3 SGD probe in test_fl10 is restricted to 06J cells (D2).
  - `test_nc_b1_tau_edge_raw_best` gained `A1_LOW` so that an A1 group cannot make its `assert groups` pass vacuously. That is
    correct.
- New NCs: each would pass if the rule were broken (06J-only set accepted, A1 dropped, A2 triggers the edge, no clip log).
- Commands (venv `next_item_eda/.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider -q`, basetemp in the scratchpad):
  - tests_int (fl05_launch, fl05_lr_screen, fl07_launch, fl09_lr_ext, fl10_fedadam_launch, fl12_launch,
    fix_delta01_fl_binding, fix_delta02, int03_launch_fl, fl14_06l): **111 passed, 39 xfailed** (183 s).
  - tests_fl (test_fl10_ladder, test_fl04_dp): **21 passed, 6 xfailed**.
  - `--runxfail -k test_nc_` over the same tests_int set: **39 failed / 111 deselected**. All 39 failures are AssertionError
    (junit checked).
  - All counts match the handoff.

## Required before use
1. M1 fix (with its test and NC), preferably before packing the FL-14 tree.
2. One DL row, before the S3 read, disclosing the residual tie orders m2 / m3 and the exact-tie reading m4.
3. m1 is optional; do it before any FA_1024 MAIN run if you take it.

The S2 selection may be written with the FL-14 lr_screen (e81667d3) as it stands. M1 does not touch lr_screen.

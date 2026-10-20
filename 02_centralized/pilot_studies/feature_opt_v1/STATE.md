# SASREC_FEATURE_OPT_V1 — study state

Add-on study, lower priority than the running SASRec v3b campaign, which is never touched.
Package: `F:\Queens University\Microsoft Project\eda\SASREC_FEATURE_OPT_V1` (validate_package PASS).
Workspace: server `$R/.local_ai/feature_opt_v1/`, mirrored locally at `.local_ai\feature_opt_v1\`.
`.local_ai/` is already in `.git/info/exclude`.

## Status: DONE — KEEP_CURRENT (2026-09-26T08:28Z; fo_queue done, FO_QUEUE_DONE written)

- [x] 2026-09-25T10:03Z workspace + source snapshot (evidence/snap: dataset docs + reuse code, byte copies)
- [x] 10:10Z protection snapshot T0 (state/PROTECT_T0_study_start.json): 185 active code/config/lock
      hashes, dataset metadata hashes + tree size, process identities, disk 12.37 GiB, cgroup mem 61.3/64 GiB
- [x] 10:15Z baseline rule registered (config/BASELINE_RULE.md), before any study score
- [x] 10:22Z Sonnet A inventory: FEATURE_INVENTORY.csv (69 rows), TRANSFORM_LINEAGE.json, FEATURE_DEPENDENCIES.json, REDUNDANCY_FINDINGS.md, OPEN_QUESTIONS.md, profiler (11 tests pass)
- [x] 10:25Z bounded sample profile (3/64 buckets, 1 core, peak 537 MiB) + 10:36Z duplicate/fit check -> audit/MEASURED_FINDINGS.md (M1 fit-population defect; 9 exact duplicates; 6 dead flags; drift)
- [x] 10:55Z Fable scope decision (owner-delegated): STAGED_WITH_EVIDENCE_GATES -> review/AUTO_DECISION_SCOPE.md (routing claude-fable-5-1 from transcript)
- [x] 11:05Z protocol + registry frozen (FEATURE_EXPERIMENT_PROTOCOL.md, config/candidate_registry.json; manifest sha 15cbb263… identical local/server)
- [x] study code (snapshot-isolated): fo_env, feature_view, fo_models, fo_eval, pin_baseline, stream_fit, probe_runner, screen, finalize_fo, fo_queue, protect_diff; smoke: lean fold parity 1.9e-7, D3 init exact
- [x] Sonnet B independent tests (found 2 real defects, fixed) -> Fable Review C pass 1 FIX_REQUIRED (2 high, 4 medium, 2 low; all fixed) -> +37 tests -> 125/125 pass -> pass 2 APPROVE_FOR_ISOLATED_EXPERIMENTS (review/FABLE_CODE_REVIEW.md, review/FEATURE_CODE_REVIEW_PASS.json; server files == approved hashes)
- [x] 11:44Z armed tmux fo-queue (waits for campaign FINAL_STAGE_DONE, no campaign process, GPU with zero compute apps, disk >= 8 GiB + peak)
- [x] 07:24–07:34Z T1_protect, T1_diff, stream_fit, full_profile, pin_baseline DONE (pin A1_RICH256_s2027 e5 0.186513; sigma_hat 0.000869 → gate 2(b) threshold 0.002)
- [x] 07:34–07:36Z probes FAILED ×2: isolation breach (reused GRU train.py puts gru_v1/code at sys.path[0]; server-only) → FO-02 fix in fo_env.py (pre-import reuse modules from the snapshot, eval_full before train; scrub foreign dirs; wider assert_isolated; +3 tests, 128/128 local + server) → Fable APPROVE_RESUME (review/fo02/, review/AMENDMENT_FO_02_ISOLATION_FIX.md); DONE stages valid (14/14 byte-identical; train never imported there)
- [x] 07:58Z probes DONE: no removal candidate (all 7 coherent-absence probes hurt; P8 +0.00066, CI crosses 0) → C2 skipped; lean parity rank identity 0.998683 < 0.999 → PARITY_FAILED
- [x] 08:28Z screen DONE: CONTROL 0.185454, C1_D3 0.185716 (pin 0.187117); C1−CONTROL +0.000262 CI [+0.000040, +0.000475] < 0.002 → gate 2(b) FAIL → finalize KEEP_CURRENT; T2 protect/diff DONE (dataset tree equal)
- [x] FINAL_FEATURE_OPT_REPORT_AR.md + _EN (lead, report_tools/write_fo_report.py, from result files only); FINAL_FEATURE_OPT_RESULTS.json, SELECTED_FEATURES.json, TRANSFORM_STATE.json (PARITY_FAILED_NOT_ADOPTED), FEATURE_SELECTION_RESULTS.csv

## Resource notes (measured, not historical)
- Disk free 12.37 GiB (10:10Z). A future full pair from scratch = 2 runs x (latest+previous with
  optimizer ~1.9 GiB + epoch weights ~0.33 GiB each) ≈ 5–7 GiB: must be budgeted against the 10 GiB target.
- RAM: two campaign trainers ~18 GB RSS each; study audit processes must stay tiny while they run.
- Compute cost of this study is recorded separately from the campaign ETA.

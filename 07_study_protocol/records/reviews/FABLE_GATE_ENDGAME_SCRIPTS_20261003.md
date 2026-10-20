# FABLE GATE REVIEW — final REES46 endgame scripts D / E / F / G / H (2026-10-03, read-only)

- **Reviewer:** Claude Fable 5.1, read-only, 45-minute cap. No Codex, no git, no phase run. No MAIN / VALIDATION / TEST value read
  (only paths, shas, statuses and script text were opened; no results / ranks / metrics of any MAIN run).
- **Read (local):** `company/plans/ops_scripts/endgame/{README.md, _endgame_lib.sh, _endgame_ana_args.sh, _endgame_freeze_check.sh,
  D_freeze.sh, E_check_only.sh, F_validation_read.sh, G_lock.sh, H_test.sh}`; FABLE_ENDGAME_REVIEW_20261001 §1 / §3 / §5; PREG1_AMENDMENT_06P
  + ERRATUM 1; MASTER_PLAN_V5 §9–§11; DL 317, 331, 336, 342, 343, 351, 354–360, 368, 371, 372, 376, 378–382; lanes_r6w FTW / D3 lane
  forms; `laneD3_INT8_pass2.sh`; `tools/ana_block_index.py` (ALLOWED_JSON, `onnx()`), `tools/retention_report.py` (METRICS), `v4int/launch_device.py`
  (selection defaults), `v4dev/bench.py` (`--latency-only`), `v4data09/d9_gate.py`, `tools/test_exposure_register.py` (path roots).
- **Read (server, paths only, via lobot):** `$Q/endgame/*` shas + `ok/`, `$Q/_common_code_ds3r7.sh` wrappers, `ONNX_R7` / `ONNX_R7_U16384` file
  lists, `company/gates/`, `company/registers/` (absent), `code_r7b_*_cand`, `_PARKED_SMOKE/DEV`, `FL_T` / `DEV` dir names.
- **Labels:** MEASURED = read from a file / DL row; INFERENCE = my reading; FIX = required change.

## 0. Verdict: **GO-WITH-FIXES** — D / E / F / G may run as written once FIX-3 / FIX-4 / FIX-5 are applied (D-level additions);
## **H is NO-GO until FIX-1 and FIX-2** (its `onnx` step cannot succeed on the deployed tree, so `analysis` would never open).

## 1. What is correct (MEASURED against the scripts and the authorities)
- **Order and gates.** D → D-local → E → F → G → G-local → H(gate → build → final → reeval_dry → reeval → dev; base / onnx after final; analysis
  after reeval + dev + base + onnx) is the §2o / 06D §2 order. `need` / `once` make every output write-once; F, G, H require `--i-confirm`
  as the LAST argument (`confirm "$2"`, `confirm "$5"`); `ejob` stops on rc ≠ 0 or any `REFUSED`; no log is printed. `dev()` passes `--gpu "$G"`.
- **D_freeze.** Refuses unless the contrasts argument AND the server file sha = f1769f0b (DL 336). Records: tools shas = the r7 TOOLS_MANIFEST
  (retention_report / ana_block_index / test_exposure_register) + g2_run; `code_ds3r7` = SYNC_MANIFEST per file; `evaluator_code_sha256` 61b64e61;
  `reeval_sha256`, `launch_fl_sha256`, `launch_device_sha256`, `fl_code_sha256`, `library_files_sha256` (= `retention_report.bind(CR)`); trees r2–r7 and
  each run's LAUNCH code identity; contrasts v4 fields (hierarchy, Holm family, registered seeds, rule map, `seed_cuts []`, `allow_single_seed []`);
  record prefixes S7 d67a0a85 / S8 8b79d305 / S_RECORD_WARM_FROZEN 2e47ba39 / SELECTION_FT_FA a645edaf / S_RECORD 7ab91eb2 / SELECTION_FT_C
  e47965d2 / tuning 9bca186e / pin db744ca3; the `--claims-only` v4 output hashes; the §9 command text + sha; `declared_not_run` (DL 317 D3 + Q15 /
  Q16 + K16 / TRUNC if absent); N33 mapping; the STOPPED tuple with the M5 marker required (and any marker-carrying dir skipped); S_CAL* skipped;
  all 11 N33 dirs + FT_FA ×2 + every warm run required DONE_MAIN / DONE (WANT_FL / WANT_DEV); refuses on any smoke-named dir in an indexed root.
- **E_check_only is value-free.** The index parses only LAUNCH / RUN_SUMMARY / STATUS / REEVAL_RECORD / BENCH_RESULT / rows.meta (ALLOWED_JSON, index
  L147 / L54) and copies / hashes rows + ranks; `--check-only` binds and checks without printing a value. E writes the PASS record with the block tree
  digest; `freeze_check` compares the uploaded freeze to the DL sha and every frozen identity (tools, reeval, contrasts, code_root, §9 text) to the deployed files.
- **F is the single §9 read.** Same block (tree digest must equal E's), same freeze, command text frozen; D4 arguments exactly: `--fl $R/FL_T`, `--pin`,
  `--contrasts` v4, `--records` incl. S_RECORD_WARM_FROZEN, FL_WARM_SELECTION_S7 / S8, SELECTION_FT_FA, DP_WOR receipt; **no `--allow-single-seed`**;
  no LR_SELECTION / EVENT_SIZE (Q6 / Q15); arm set = the D inventory (N33 dirs as the cold arms, FT_FA ×2, the warm line). Output O_EXCL, a-w.
- **G_lock.** Re-seals every FL best + endpoint (28 dirs) against events.jsonl + DEV `start.file_sha256` (the DL 356 fix), records central shas from
  sealed CHECKPOINTS.json (reeval re-hashes with `--expect-sha256`), PRE cf08588c (fl_init key = what reeval checks), PF stores vs RUN_SUMMARY; hashes
  (never parses) every block entry's rows / ranks; TEST plan = C_BIG ×2, C_SMALL ×2, C_FULL s2026 + s2027 (DESCRIPTIVE), PRE PB + CB, cold FA ×2, all
  N33 arms, the warm line, each PB + CB; panels only where a VALIDATION panel dir exists (TRUE_COLD only where B admitted); device arms with the
  locked selections / bases (forms = the VALIDATION lanes, MEASURED: FT_FA_WARM(_FROZEN) `--base-kind fa_warm` without `--selection` → launch_device's
  registered SELECTION_FT_C default; FT_FA with SELECTION_FT_FA; FT_PRE `--init-from`), base sha cross-checked against the VALIDATION LAUNCH `start`;
  six baselines (SKNN k 500 / sample 5000); lock = v4 verbatim + `all_methods_lock`, self sha recomputed (Q7); REGISTER_ENTRY_5 = MANIFEST_BUILD /
  RETRO_TEST / NONE / NO_FEEDBACK / lock sha / r7 aggregate. H `gate` requires a register entry binding exactly that lock sha and d9_gate OPEN.
- **H resume rule.** reeval: COMPLETE → skip; incomplete dir → LANE_STOP (manual, E10.3). dev: existing dir without RUN_SUMMARY → `--resume` only
  after the base sha = lock record; with RUN_SUMMARY → skip. base: RECEIPT.json → skip. Nothing re-selects or edits a flag.
- `test_exposure_register.py` and `d9_gate` resolve `company/` from `__file__` (V = tools/..; V4 = code/..) — correct with cwd `$CR`.

## 2. Findings and FIXES

**FIX-1 (BLOCKER, H).** H's `onnx` step is bound to r7 and cannot succeed. MEASURED: `_endgame_lib.sh` L7 refuses `CR ≠ $V/code_ds3r7`;
`_common_code_ds3r7.sh` cd's into r7; `freeze_check` refuses `code_root ≠` the freeze's (D writes `code_ds3r7`); the dump runs `v4int/dump_rows_retro.py`
from `$CR`. The r7 dump has no `--ckpt` / view binding (DL 358), so the guard at H L131 LANE_STOPs (exit 6) whenever the lock says `onnx: true` —
which it will, because K16 pass 1 is DONE on `ONNX_R7_U16384` (DL 372). Without `ok/H_onnx_v1.ok`, `analysis` never runs. No r7b tree is deployed
(MEASURED: only `code_r7b_dev05_cand`, `code_r7b_p2_cand`; no `deploy_r7b`). **Fix:** pack + deploy `$V/code_ds3r7b` (= r7 + DEV-05 + the P2
`--latency-only` bench, the REV-DEV-05 / REV-OPS-B13-P2 shas) with its SYNC_MANIFEST, BESIDE r7; assert its `v4eval/*`, `v4int/reeval.py`, `v4int/launch_*.py`
= r7 bytes. G adds to `test_plan`: `onnx_tree: code_ds3r7b`, its aggregate sha, and the file shas of `v4int/dump_rows_retro.py` + `v4dev/bench.py`; adds
r7b to `code_trees`. H `onnx` verifies the deployed r7b against those lock fields, then runs the dump and the bench with `cd $V/code_ds3r7b` +
`PYTHONPATH=$V/code_ds3r7b` (only these two commands); every other H step stays on r7 and the lib guard stays. If Ops prefers not to deploy r7b, the
lock must set `onnx: false` with TEST K16 in `declared_not_run`, and that asymmetry (K16 on VALIDATION, not on TEST) needs its own DL decision before G.

**FIX-2 (BLOCKER, H).** The server copy of `H_test.sh` is stale. MEASURED: server sha 94d5edd3… vs local 4b0cde99…; the diff is exactly the DL 358
edit (L128 `--ckpt "${P[1]}" --family SASREC`, L129 comment). The other 8 endgame files match (CRLF-stripped). After FIX-1 / 3 / 4 / 5, re-upload every
changed file, sha-check, make read-only, and record the shas in the DL row.

**FIX-3 (MAJOR, D / G — K16 pass 2, DL 368).** The freeze records only `k16_included / k16_onnx_dir / k16_first_users`. Pass 2 (latency only) runs later
on the identical pinned artifact and writes `LATENCY_PASS2.json` + `PASS2_HOST_LOAD.json` into the same `ONNX_R7_U16384` (lane + bench, MEASURED). The index
never parses those names (ALLOWED_JSON), so E / F are not disturbed, but nothing freezes the identity pass 2 must assert. **Fix:** D adds `k16_pass1_artifacts`
= shas of `ONNX_R7_U16384/{BENCH_RESULT.json, rows.meta.json, rows.npz, model_fp32.onnx, model_int8.onnx}` and `ONNX_R7/PASS1_HOST_LOAD.json`, plus
`k16_latency: "PASS2_PENDING — the device latency is LATENCY_PASS2.json (DL 368); pass-1 latency is loaded-host, never the device number"`. G copies
these into the lock and, if `LATENCY_PASS2.json` exists by then, its sha; the TEST `onnx` plan states the same rule for the TEST bench (its in-run
latency is loaded-host unless a TEST pass 2 is run after Wave 1; the retention / accuracy rows are unaffected).

**FIX-4 (MINOR, D).** The parked smoke dirs are not in the freeze. MEASURED: `_PARKED_SMOKE/DEV/{FT_PRE_SMOKE_s2026, FT_FA_WARM_SMOKE_s2026/7,
FT_FA_WARM_FROZEN_SMOKE_s2026/7}`, `ok/P_PARK.ok`, PARK_MANIFEST sha ec779fd4… (DL 356). **Fix:** D requires `ok/P_PARK.ok`, hashes
`_PARKED_SMOKE/DEV/PARK_MANIFEST.json`, and lists the five names under `parked_smoke` (beside `exclusions_by_name`).

**FIX-5 (MINOR, D / F — the descriptive metric set).** `retention_report` METRICS = MRR@20 (primary) + NDCG@10 + HR@10 only (L211). The owner's
timeline (DL 382) promises HR / NDCG / MRR @ 1, 5, 10, 20, micro + macro beside the primary. F as written does not produce them. **Fix (register before D):**
either (a) a small extraction script (reads only the block's RESULT_ROWS / evaluator rows, no selection, no new contrast) is hashed into the freeze
`scripts` map and run inside F right after the report, or (b) the freeze declares the extra metrics a POST_VALIDATION descriptive table (06L-C C5),
read after F from the already-disclosed rows. Either is fine; the choice must be written before the read.

**Notes (no fix required).** (i) `onnx_val_dir` is re-evaluated in every phase and is part of the frozen §9 text; pass 2 writes no BENCH_RESULT into
`ONNX_R7`, so the text stays byte-identical — keep it that way (no `BENCH_RESULT.json` may ever appear in `ONNX_R7`). (ii) G does not re-hash the central
checkpoint files against CHECKPOINTS.json; the reeval's `--expect-sha256` does, acceptable. (iii) H `gate` needs `company/registers/` (absent on the server,
MEASURED) and `g0/data_census/*` uploaded first — already in README §7.4. (iv) ERRATUM 1 (Holm 18) is in the v4 file sha f1769f0b, which D pins.

## 3. Pre-run checklist for Ops (after N33_FA_Q8_s2026 is DONE_MAIN)
1. Apply FIX-3 / 4 / 5 to D (and 3 to G); FIX-1 to G + H; re-upload per FIX-2; DL row with the new shas; upload `ANA_CONTRASTS_v4.json` (f1769f0b).
2. D → fetch → O_EXCL copy → DL row → upload → E → tell the owner → F (`--i-confirm`) → G → entry 5 → upload lock + register → deploy r7b → H.
3. Never run pass 2 so that it writes anything but `LATENCY_PASS2.json` / `PASS2_HOST_LOAD.json` into `ONNX_R7_U16384`.

## 4. Delta re-review (2026-10-03, after OPS-ENDGAME-3; read-only, no value read, no phase run) — **GO for D → E → F → G → H**

Verified MEASURED (local files CRLF-stripped vs server `sha256sum`; server trees and dirs by path / sha only):
- **FIX-1 (r7b).** `$V/code_ds3r7b` deployed read-only beside r7 (0 writable files; r7 dir untouched); `deploy_r7b/SYNC_MANIFEST.json` 9c5783da…
  = aggregate 1cb1e706…, tar 3b766594…, 688 files, `base` = r7 d76b264e…; `v4dev/bench.py` 9662cba5…, `v4int/dump_rows_retro.py` 83f5f2f7…,
  `v4int/reeval.py` 833f0c98 = r7; `R7B_DEPLOYED.ok` / `R7B_VERIFIED.ok` present. G re-hashes all r7b files against the manifest, requires its
  base = the frozen r7 aggregate, and pins `code_trees.code_ds3r7b` + `test_plan.{onnx_tree, onnx_tree_aggregate_sha256, onnx_dump_rows_retro_sha256,
  onnx_bench_sha256, onnx_latency_rule}`. H `onnx` reads those four lock fields (`plan onnx` lines 3–6), refuses unless `onnx_tree = code_ds3r7b`, re-hashes
  all 688 files + the aggregate + the two file shas against the lock, then runs ONLY the dump (`--ckpt … --family SASREC`) and the bench through
  `r7b_run` (subshell `cd` + `PYTHONPATH`). Every other step stays on r7; `_endgame_lib.sh` (186f8337) and the CR guard are unchanged. The batch_view guard stays.
- **FIX-2.** Server = local for all 15 files: D 2fa8d837, F 7f98b65a, G 372e9e51, H 11c04583, metric_table.py 561d758a; E ba2a37fd, lib 186f8337,
  args 7a4071f7, freeze_check 97efd361, P 0878d704, B ad549671, B2 d9af27c6 unchanged from the first review; all read-only on the server.
  Full old-vs-new H diff = exactly the FIX-1 block, the `plan onnx` print, the `--ckpt` dump line and the FIX-5 block; nothing else changed.
- **FIX-3.** D pins `k16_pass1_artifacts` = every file of `ONNX_R7_U16384` except `LATENCY_PASS2.json` / `PASS2_HOST_LOAD.json` (8 files, incl. both
  models, rows.npz, rows.meta, BENCH_RESULT, both rank files) + `ONNX_R7/PASS1_HOST_LOAD.json` (present), `k16_latency` = PASS2_PENDING, and refuses a
  BENCH_RESULT in `ONNX_R7` (none today). G re-hashes them (refuses on change), copies them, adds `k16_latency_pass2_sha256` (sha if present, else
  PASS2_PENDING) and the TEST latency rule.
- **FIX-4.** D requires `ok/P_PARK.ok` (present) and `_PARKED_SMOKE/PARK_MANIFEST.json` = ec779fd4ea9dbe51… (MEASURED equal; a 5-entry list) and
  writes `parked_smoke {manifest, manifest_sha256, dirs[5], label}`; G copies it.
- **FIX-5.** DL 385 registered (text = REG_DESCRIPTIVE_METRIC_TABLE.md, sha 561d758a). `metric_table.py` parses (`--help` on the server), is copy-only
  (five fields verbatim; no mean / CI / rounding), binds BLOCK_INDEX.json by sha and each rows file by the index `sources` sha, writes O_EXCL, prints counts
  only. D freezes it in `scripts` (key `runs_ds3/queues/endgame/metric_table.py`, so `freeze_check` re-hashes it in E / F / G / H) and in
  `descriptive_metric_table` with the DL row (`MT_REG_DL`, refused if unset). F runs it AFTER the F marker on the same block with E's
  `block_index_sha256`; H `analysis` runs it after the TEST report. A refusal stops only the table.
- `ANA_CONTRASTS_v4.json` on the server = f1769f0b (DL 336). README §8 and the OPS-ENDGAME-3 handoff match the scripts.

Residual notes (no fix needed): (a) `metric_table.pick` is slightly laxer than `retention_report.select_rows` (no `_is_evidence` filter; RULE_FREE rows
carrying a selection_rule are not refused) — the worst case is a value-free "two rows" refusal of the table, never a wrong read; record it in the DL and
re-run into a new `--out` only per DL 385 (4). (b) G takes the r7b dump / bench shas from the manifest rather than asserting the reviewed values; the
deployed files equal them today (83f5f2f7 / 9662cba5).

**Run-time preconditions for Ops (DL 384 condition 4):**
1. D only after `FL_T/N33_FA_Q8_s2026` is DONE_MAIN (D refuses otherwise); run `MT_REG_DL=385 D_freeze.sh f1769f0b…`.
2. Before the local O_EXCL copy of the freeze candidate: `k16_onnx_dir` ends `ONNX_R7_U16384`, `k16_first_users` 16384, `k16_pass1_artifacts` has 9 keys,
   `parked_smoke.dirs` has 5, `scripts` has 5 keys incl. metric_table 561d758a, `declared_not_run` has no K16 line, `allow_single_seed` is `[]`.
3. Before the local copy of the lock candidate: `onnx_tree_aggregate_sha256` 1cb1e706…, `onnx_bench_sha256` 9662cba5…, `onnx_dump_rows_retro_sha256` 83f5f2f7…,
   `onnx: true`; then entry 5, upload lock + register, `g0/data_census/*`, all sha-checked.
4. Owner informed (logged) before F and before H `gate`, per DL 384. Any REFUSED / LANE_STOP stops the phase; no flag is edited (a refusal in E = new freeze).
5. The pass-2 lane may run at any time but must write only `LATENCY_PASS2.json` / `PASS2_HOST_LOAD.json` into `ONNX_R7_U16384`, never a BENCH_RESULT into `ONNX_R7`.

## 5. Decision on the E refusal (2026-10-03, DL 389; read-only; counts only, no metric value) — **option (b), the registered manifest semantics**

**Finding (MEASURED).** `tools/g2_run.py` L235–247 enforced amendment 02 §2 / G2 condition 5 on **manifest users** (`np.unique(main.eval_user_key)`,
no eval_mask): 51,211 / 12,715 / 38,496. `retention_report.load_calib_context` L2632–2637 reuses the same constant but counts **E2E users**
(`_euks` = `np.unique(eval_user_key[eval_mask])`, L4314–4317) and builds `calib_hex` from the CALIB **E2E** users. The two definitions were never
reconciled; `tests_ana/test_ana04.py` L430 only checks the synthetic `{48, 16, 32}` and no NC pins the production counts, so the check could not pass on
real data. The data are as registered (DATA-06 L46 / L48 / L70: MAIN 51,211 users, 40,764 with ≥ 1 eval row; CALIB 12,715 / 10,070). Tool bug, not a data problem.

**Decision.** Fix the tool to the registered semantics, not the constant to the data:
1. In `load_calib_context`: build `calib_hex` from **all** CALIB manifest users (`np.unique(cm.eval_user_key)`), compute `counts_manifest` on all MAIN
   manifest users (`np.unique(main.manifest.eval_user_key)`) and enforce `counts_manifest == CALIB_REGISTERED_COUNTS` {51211, 12715, 38496} (unchanged
   constant, amendment 02 §2) plus the subset rule (every CALIB manifest user is a MAIN manifest user) — identical to `g2_run.check_counts`.
2. Keep the E2E counts as the disclosed diagnostic (`counts_e2e`: MAIN E2E, CALIB-member E2E, disjoint E2E) and pin the two DATA-06 exact values
   `CALIB_E2E_REGISTERED = {"main_users": 40764, "calib_users": 10070}` (DATA-06 L46 / L48) as a second refusal; the E2E disjoint count is reported, not pinned
   (it is derived). The disjoint view stays on MAIN E2E users whose hex is not a CALIB **manifest** user (the amendment's "CALIB-disjoint").
3. Output: `counts` → `{"manifest": …, "e2e": …}`; REPORT.md states "38,496 CALIB-disjoint users, of whom N have ≥ 1 eval row".
4. Tests: update the synthetic known-answer (both dicts), add NCs: wrong manifest counts refuse; wrong E2E pins refuse; a CALIB user missing from MAIN refuses.
   Acceptance: `tests_ana` all pass; `--claims-only` under v4 byte-identical to the r7 output (the fix cannot touch claims); diff confined to
   `load_calib_context`, the two constants and tests.
5. Process: Sonnet worker (ANA-07) + one independent short review (REV-ANA-07; Opus or Sonnet) — required, the tool is frozen. Then pack `tools_r8`
   (retention_report only; ana_block_index 6e7043bb and test_exposure_register 7d59e828 unchanged) → `deploy_r8/TOOLS_MANIFEST.json`, archive the r7 copy
   (`tools/_archive_r7`), D_freeze.sh L79 reads `deploy_r8/TOOLS_MANIFEST.json` (new D sha, re-upload, sha-check). Run D → E → … with `ANA_TAG=v2` exported
   in every phase (freeze file `ANA_SCRIPT_FREEZE_v2.json`, dirs `*_v2`, per Q19; the lock name stays). Freeze v1 (03028475) is SUPERSEDED, never used for a
   read; `block_val_v1` stays read-only as the record of the refused check. No Fable re-review needed if REV-ANA-07 confirms the diff scope; anything beyond it → one Fable delta.

**DL text.** "FABLE DECISION on the E refusal (DL 389): tool bug confirmed — retention_report.load_calib_context compares E2E user counts to the amendment 02
§2 manifest counts (51,211 / 12,715 / 38,496), which g2_run enforced on manifest users; no data problem; no MAIN / VALIDATION / TEST value read. FIX (option b):
enforce the registered manifest counts exactly as g2_run.check_counts (all CALIB manifest users mapped by client_key_hex into all MAIN manifest users), report the E2E
counts beside them and pin MAIN 40,764 / CALIB 10,070 E2E users (DATA-06 exact), disjoint view = MAIN E2E users not CALIB manifest users; tests + 3 NCs; claims-only byte-identical.
Worker ANA-07 (Sonnet) + REV-ANA-07 (independent, short) required. Then tools_r8 (retention_report only) → deploy_r8 → D_freeze L79 → D/E/F/G/H with ANA_TAG=v2.
Freeze v1 (03028475, DL 388) SUPERSEDED, never read. The single read remains unspent."

## 6. Decision on the second E refusal (2026-10-03, E v2 14:30:35Z; read-only, no value read) — **fix (a) as proposed; dry check PERMITTED under conditions**

**Finding (MEASURED).** `retention_report.s_record_warm` L4829–4831 requires `n_norms == len(rounds_0based)`. The registered writer / validator
`launch_fl.check_s_content` L709–732 (FL-16b, REV-FL-16 m5) defines `n_norms` as any int > 0 (the client-delta norms behind the median) and
`rounds_0based == list(range(KNOBS["s_cal_rounds"]))` = 0..19 for `S_CALIBRATION`; the ANA fixture (anakit L734, n_norms 20480) and test_ana06 L87–88 (n = 3,
3 rounds) never exercised n ≠ rounds. Same class as DL 389: an ANA-side definition drift from the registered rule. The record (20,625 norms / 20 rounds) is the registered shape.

**Fix (ANA-08, Sonnet; minimal diff).** `s_record_warm` mirrors `check_s_content` exactly: S finite > 0; `n_norms` int > 0 (not bool); `rule` str starting
`median`; `label == "S_CALIBRATION"` → `rounds_0based == list(range(S_CAL_ROUNDS))` with ints (not bools), where `S_CAL_ROUNDS` is imported from
`v4int.launch_fl.KNOBS["s_cal_rounds"]` through the bound code root, or pinned to 20 with a test asserting equality to `launch_fl.KNOBS`. Drop the
`n == len(rounds)` rule. Fixture → n_norms ≠ len(rounds) (e.g. 20,625 / 20). NCs: rounds ≠ 0..19; `n_norms` 0 / −1 / `True`; non-median rule; a bool in rounds.
Acceptance: tests_ana pass; `--claims-only` byte-identical to r8; diff confined to `s_record_warm`, the constant and tests. **No collect-all mode** (keep the diff minimal; the dry check below replaces it). REV-ANA-08 (independent, short) → tools_r9 (retention_report only) → deploy_r9 → D_freeze L79 → D v3 → E v3 (`ANA_TAG=v3`); v1 / v2 SUPERSEDED, never read.

**Dry check — PERMITTED.** `retention_report.py --check-only` is the registered E step; it binds, checks and prints no value (L5963, L6070–), so running it
with a CANDIDATE tool on the already-built `block_val_v2` (read-only, index tool unchanged) spends nothing: the freeze protects the READ, not the check.
Conditions (all mandatory):
1. Exactly the E command with `--check-only`, `--contrasts` v4, `--code-root $CR`, `--expect-script-sha256 <candidate sha>`, `--out $R/ANA/_dry_check_v2/<utc>_<sha12>/`
   — never `out_val*`, never without `--check-only`, never `--claims-only` into `company/gates`, never on a TEST block.
2. Candidate tool placed at `$R/ANA/_dry_check_v2/tools_cand/retention_report.py` (read-only), NOT in `$V/tools/` (the deployed r8 stays untouched until tools_r9).
3. Iterate fix → dry check until `ANA CHECK-ONLY: PASS`; each iteration logged to `$Q/EGE_DRY_<utc>.log`; only the REFUSED line / the PASS line (no numbers beyond
   counts) is copied into the DL. Logs and dirs kept read-only as the record (not deleted).
4. The sha that passed the dry check must equal the sha reviewed by REV-ANA-08 and packed in tools_r9; D v3 then pins it. Any later edit = a new dry check.
5. One DL row per iteration series: candidate shas, every refusal text found, final PASS sha.
6. ana_block_index is NOT re-run (unchanged 6e7043bb); if the index ever changes, the dry check moves to a fresh `block_dry_*` dir, never `block_val_v2`.

**DL text.** "FABLE DECISION on the second E refusal (E v2 14:30Z): ANA-side definition drift confirmed — retention_report.s_record_warm L4829–4831 requires
n_norms == len(rounds_0based), while the registered validator launch_fl.check_s_content (FL-16b, REV-FL-16 m5) defines n_norms as any int > 0 and
rounds_0based = 0..19 (KNOBS s_cal_rounds); the record (20,625 / 20) is the registered shape; no data problem; no value read. FIX (ANA-08, Sonnet): s_record_warm
mirrors check_s_content exactly (S > 0; n_norms int > 0; rule startswith 'median'; S_CALIBRATION → rounds == list(range(20)) pinned to launch_fl.KNOBS by a test;
ints not bools); fixture n ≠ rounds; NCs rounds ≠ 0..19 / n_norms ≤ 0 or bool / non-median rule; claims-only byte-identical; no collect-all mode. REV-ANA-08 →
tools_r9 → deploy_r9 → D_freeze L79 → D v3 → E v3 (ANA_TAG=v3); freezes v1 / v2 SUPERSEDED. PRE-FREEZE DRY CHECK PERMITTED: retention_report --check-only with the
candidate tool (kept outside $V/tools) on the read-only block_val_v2 into $R/ANA/_dry_check_v2/<utc>_<sha12>/, iterate until PASS, logs read-only, only refusal / PASS
lines in the DL, the PASS sha = the reviewed and packed sha. The single read remains unspent."

## 7. Review of PREG1_AMENDMENT_07_ADDENDUM_B15_DRAFT (ML-1M Track B runs to completion; B13 rescinded) — **GO-WITH-EDITS**

Read: the B15 draft, B13 (DL 368), DL 381, 400. Read-only; no ML-1M value read. (For the record: the DL 400 row surfaced in a grep and shows the
already-disclosed post-F REES46 VALIDATION headline; it was not sought and plays no part here.)

**Why GO.** B13 was a clock-only backstop registered on schedule grounds; rescinding it before its trigger, again on schedule grounds, with no Track-B
inner-validation value read, cannot steer a selection: the full grid is the pre-B13 registered design (B10), and trackb-select has not run. The owner
decision is his to make. The INFERENCE schedule is labelled. The resume / INCOMPLETE mechanics are correct and value-free.

**Edits required before registration (textual; no mechanics change):**
- **E1 (Status).** Add, MEASURED by Ops at registration: "trackb-select has not run on ML-1M Track B; no Track-B RESULT.json or inner-validation value has
  been opened; the decision is schedule-only." (The draft asserts it; the DL row must measure it: no `trackb-select` state file, no `.ok` of that step.)
- **E2 (B15.2).** Record the disarm: "b13_stopper disarmed at <UTC>, before 2026-10-06T00:00Z; pid <n>; evidence `<paths of .pid/.log/.nohup>`; the script
  (sha 3e4aa02b…) stays read-only in `plans/ops_scripts/bench/` and on the server and is never run for ML-1M."
- **E3 (B15.2).** Add: "A cell that cannot be resumed with identical hashes is INCOMPLETE by DL row (cause recorded), listed in the ML-1M LOCK disclosure and the
  Track-B table; the per-arm winner is then chosen among completed cells (B10 Q4). No cell is re-run with changed hashes."
- **E4 (new B15.5, consequence for REES46).** "K16 INT8 pass 2 (latency only, DL 368) waits for every Wave-1 driver to exit, so it now follows the ML-1M grid
  (≈ 10-09/10, INFERENCE). The REES46 TEST look (H) is not gated on pass 2; the TEST bench's in-run latency stays loaded-host per the lock's
  `onnx_latency_rule`; the thesis device-latency number is pass 2. H's CPU steps (base, onnx) run beside the 6-slot Wave-1 load as registered."
- **E5 (B15.4).** Add: the ML-1M LOCK records the B15 file sha and the sentence "B13 rescinded before its trigger on schedule grounds; no value read."
- **E6 (header).** DRAFT → REGISTERED with the DL row number and the file sha; filename `..._B15.md`, O_EXCL, read-only.

**DL text.** "REGISTERED contracts/PREG1_AMENDMENT_07_ADDENDUM_B15.md (sha <…>; owner decision in chat 2026-10-03; Fable GO-WITH-EDITS §7 applied): B13 (DL 368)
rescinded before its 2026-10-06T00:00Z trigger on schedule grounds; every ML-1M Track-B seed-2026 cell runs to completion; trackb-select over the full grid;
winners' s2027 complete; mechanical failures resume under identical hashes; INCOMPLETE only by DL row. MEASURED: trackb-select not run, no Track-B value opened;
b13_stopper disarmed <UTC> (pid <n>, evidence kept). Consequence: K16 pass 2 follows the Wave-1 end (≈ 10-09/10, INFERENCE); H not gated on it. ML-1M final
≈ 10-11..10-16 (INFERENCE). Other datasets, Wave 2 and REES46 unaffected."

## 8. Owner-requested re-review of DL 368 Decision 1 (ML-1M Track B device) with the 10-03 evening facts — **B, in the minimal form (ML-1M on GPU; Wave 2 stays CPU unless the timing gate says otherwise); hard switch at a registered milestone, no two-complete-sets fallback**

**The DL 368 refusal was right on its premises and its premises are gone.** It rested on (i) the A40s being the REES46 endgame's resource — now only
H reeval/dev (≈ 10-04 evening) remains; (ii) ≈ 10 % agent budget until ≈ 10-08 — reset on 10-02; (iii) B13 bounding ML-1M at ≈ 10-07/08 — rescinded by B15,
so CPU ML-1M ends ≈ 10-11..16 with 18 mostly L200 cells unstarted, and the slow cells (C_FULL L200 B24, ≈ 6 days each on CPU) are exactly the GPU-friendly
central arm. I reverse Decision 1 for ML-1M. Rules that still hold: no value read before registration (MEASURED (7): trackb-select not run); one device per
dataset; one LOCK, one TEST look; device class disclosed per dataset. Integrity is not touched by a device change registered before any look, so this is a
schedule / cost decision, which is the owner's — my part is the conditions.

**Why not C.** It discards ≈ 140 finished S3 jobs for symmetry only; cross-dataset comparability was never numeric (claims are within-dataset ratios). S3 ×3
lock and look on CPU-FP32 as registered. **Why Wave 2 stays CPU by default.** Its CPU ETA (≈ 10-08..10) improves by itself once the ML-1M driver leaves the
host (6 slots freed), and FL rounds are host-bound (DL 381: GPU 4–5 s / round at 21 %); moving it costs a from-scratch restart for ≈ 1–2 days. Extend to Wave 2
only if the timing gate below measures ≥ 3× per FL job and Baby_Products is < 1/3 done.

**Conditions (all before the first registered GPU run):**
1. **Timing gate (≈ 10 min on the A40 freed by H reeval):** ML-1M C_FULL L200 0.1 epoch and FA 20 rounds, GPU vs the CPU figures; register B16 only if the
   projected ML-1M final on GPU is ≤ 10-08 (≥ 3 days gained); otherwise option A stands and the DL records the measurement.
2. **BM-03 (Opus worker; Sonnet independent review; one Fable concurrence on the addendum text).** `code_bench5` beside bench4 (S3 identities untouched):
   `--device {cpu, cuda}`; for cuda: `torch.use_deterministic_algorithms(True)`, cudnn deterministic / benchmark off, TF32 off (matmul + cudnn), `CUBLAS_WORKSPACE_CONFIG`,
   same seeds and data order as CPU; device class `NVIDIA_A40-FP32-DET` written into every RUN_SUMMARY / RESULT.json and the LOCK, which refuses a mixed
   device inside a dataset; **determinism test** = a smoke cell run twice → identical digests (both arms: C_FULL and FA); CPU path byte-identical (digest test);
   NCs for a TF32-on run and a mixed-device LOCK. No `--final` / no look code touched.
3. **Addendum B16, registered before the first GPU run (O_EXCL, DL row):** ML-1M device class = NVIDIA_A40-FP32-DET, run FROM SCRATCH; S3 ×3 = CPU-FP32 (done);
   Wave 2 = CPU-FP32 unless extended under condition 1; the CPU ML-1M dirs are sealed read-only, labelled SUPERSEDED_DEVICE, never opened, never evidence;
   disclosed in the ML-1M LOCK and the cross-dataset table ("device class differs by dataset; ratios are within-dataset").
4. **Switch rule = HARD SWITCH at the milestone** (REV PASS + determinism test + timing gate + B16 registered): stop the CPU ML-1M driver (state kept, resumable).
   **A two-device fallback that lets both sets complete is NOT acceptable**: two complete result sets for one dataset make the choice between them a
   value-shaped decision however the clock is written. The only fallback: if the GPU set is abandoned by a DL row for a mechanical cause before ANY ML-1M
   result is opened, the CPU driver resumes and the GPU dirs become SUPERSEDED_DEVICE. Never both; never after a value is seen.
5. **GPU precedence:** the bench GPU driver starts only after `ok/H_dev_v3.ok` (H reeval/dev own the A40s); H base / onnx / analysis are CPU and unaffected.
6. **K16 pass 2:** its wait pattern must also treat the GPU bench driver as a Wave-1 blocker (host feeders); otherwise unchanged (DL 368 Decision 4).
7. ML-1M final estimate after the switch is INFERENCE until the gate measures it; the DL carries the measured rates.

**DL text.** "Fable re-review of DL 368 Decision 1 (owner request): REVERSED for ML-1M — premises gone (A40s free after H ≈ 10-04; budget reset 10-02; B13 rescinded,
CPU ML-1M ≈ 10-11..16). Option B minimal: ML-1M from scratch on NVIDIA_A40-FP32-DET; S3 ×3 stay CPU (done); Wave 2 stays CPU unless the timing gate shows ≥ 3× per FL
job; C refused. Conditions: timing gate (≥ 3 days gained) → BM-03 code_bench5 (Opus worker, Sonnet review, Fable concurrence on B16): --device, deterministic CUDA,
TF32 off, device class in every record + LOCK (mixed device refused), two-run identical-digest test, CPU path byte-identical → addendum B16 registered before the
first GPU run → HARD SWITCH (CPU ML-1M driver stopped, dirs sealed SUPERSEDED_DEVICE, never opened). No two complete sets; fallback only by DL row for a mechanical
cause before any ML-1M result is opened. GPU driver starts after ok/H_dev_v3.ok; pass-2 lane treats it as a Wave-1 blocker. No value read (trackb-select not run)."

## 9. Gate on the Phase-2 design (PREG2_FOLLOWUP_DRAFT + PHASE2_IMPLEMENTATION_PLAN; 2026-10-03; read-only) — **GO-WITH-EDITS: O1 = T0a; O9 = YES under conditions; TEST2 = optional replication, never the Q1 confirmation**

Read: both documents in full; DL 400–407. No TEST value exists (DL 406); no value was read here beyond the already-disclosed VALIDATION rows the draft cites.

### 9.1 O1 / O9 — may a second registered study score on the RETRO_TEST manifests built under ALL_METHODS_LOCK 04a3cd19?
**Yes (O9), as T0a.** The manifests are data whose build was COUNTS_ONLY (register entries 5–6); no model has been scored; the single-look principle is
per claim and per sitting, and ALL_METHODS_LOCK's "no change" clause governs Phase-1 claims, not the data. What T0a must guarantee: no human sees any
TEST value before every arm of both studies is locked and scored. T0b breaks that (PRE's and FA_WARM_FROZEN's TEST values would be known while Phase-2
selection, edge rules, futility and bug fixes are still being decided) and is REFUSED as a confirmation route; TEST2-only is REFUSED as the primary route
(availability REPORTED only; different estimand — users active in December; post-Black-Friday / Cyber-Monday shift; wider CI). TEST2 is welcome as a
**pre-registered replication** (owner's O2; Fable approves the manifest spec), scored in the same sitting if ready, else later under its own register.
Owner consequence, stated plainly: the Phase-1 TEST result waits for Phase 2 (INFERENCE ≈ 10-17..10-21: code 3–4 days + runs 8–11 days); he said he prefers that.

**Conditions of the one sitting (binding):**
S1. ALL_METHODS_LOCK and its H plan (109 reevals, 14 device, 39 panels, 6 baselines, onnx) execute exactly as locked; Phase 2 adds its own plan under
    `P2_METHODS_LOCK.json` (arms → checkpoint shas, rows shas, contrasts P2_CONTRASTS_v1 sha, code-tree digest, selection-file shas, bootstrap seed 20261004).
S2. `P2_METHODS_LOCK` + `P2_SCRIPT_FREEZE` (p2_report + p2_block_index shas) are registered and on the server **before the first TEST reeval of ANY arm**,
    Phase-1 included; register entries: LOCKED_EVALUATION for the Phase-1 lock and a second LOCKED_EVALUATION binding the P2 lock sha (tool unchanged).
S3. Order inside the sitting: all Phase-1 scoring → all Phase-2 scoring → Phase-1 `analysis` → P2 analysis; **nothing under `$R/TEST`, `out_test_*` or
    `p2_out_test_*` is fetched or opened until both RETENTION_RESULT files exist** (Ops writes the DL row first, fetches second).
S4. A Phase-2 arm without a sealed checkpoint at lock time is declared not run in the P2 lock (06D §8 wording). **No second sitting on RETRO_TEST, ever**;
    a Phase-2 arm missing from the sitting is scored on TEST2 only, or not at all.
S5. PRE's and FA_WARM_FROZEN's TEST rows are produced once (Phase-1 plan) and hashed into both blocks; no arm is scored twice on the same role.
S6. Labels: Phase-1 K1…K21 CONFIRMATORY (PREG1), unchanged. Phase-2 Q1…Q8 = **CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP)** with disclosure D-a verbatim
    (hypotheses generated after the Phase-1 VALIDATION read; arms selected on CALIB; locked before any TEST value). Drop the "_T0" suffix. TEST2:
    REPLICATED_FRESH / NOT_REPLICATED / TEST2_NOT_AVAILABLE; Phase-1 arms on TEST2 = FRESH_PERIOD_AUDIT (DESCRIPTIVE). FB (U_NEW_VAL) stays
    HOLDOUT_USERS_SAME_PERIOD, never a Q1 confirmation (as drafted). Phase-2 MAIN VALIDATION numbers = POST_VALIDATION_EXPLORATORY (as drafted).
S7. GPU precedence: Phase-2 lanes never share a GPU with H steps; Phase 2 (owner's top-priority axis) has priority over the ML-1M GPU move of §8 —
    §8's timing gate must include Phase-2 demand (≈ 190 GPU-h on 2 A40 ≈ 4–6 days); if Phase 2 takes both A40s, ML-1M stays on CPU (§8 option A). One DL row before BM-03 effort is spent.

### 9.2 O4 — grids, ties, edges, futility, cap: **acceptable with four edits**
- **WA grid** (6 server points × client LR {0.003, 0.01, 0.03}; WA-ext FedAdam {1e-5, 1e-4} at c 0.1; incumbent reused only under P2-T3 bit-identity;
  η_s = 0 anchor; **mark 0 never a PB candidate**): ACCEPTED as written. WB / WC: ACCEPTED.
- **E-O4a Tie band for CALIB selection → 0.001** (INFERENCE ≈ 3 paired-bootstrap SE at 10,070 CALIB users; Phase-1's 0.0005 is below CALIB resolution, so
  ties would be decided by noise). The tie-preference order (bytes → trained params → smaller server step → smaller client LR → label) is ACCEPTED.
  The **TEST equivalence margin stays ±0.0005** (practical relevance, §9.4) — two different roles, both written.
- **E-O4b Futility threshold → 0.001** (= the CALIB tie band), both conditions as drafted; it skips WB / WC only; the primary path is unaffected; recorded.
- **Edge rule** (one extension per axis, ≤ 3 runs per stage, re-applied once, EDGE_UNRESOLVED disclosed beside every Q): ACCEPTED.
- **E-O4c Cap 250 GPU-h ACCEPTED; drop order re-ordered by tier:** tertiary-only arms first (DP1, SECAGG-MAIN, DPFTRL s2027) → exploration cells
  (WC3/WC4 → WB2 → WC1/WC2 → WB3) → C2 last (secondary Q8). Everything dropped is declared not run in the P2 lock.
- **E-O4d (O7) DP arms keep T = 384 and the registered z** (comparability with K12 / K12w; no new accountant). W*'s server / client recipe; local passes 2
  / batch 16 for BOTH FA1024 and DP arms (matched pair for Q6); disclosed that a small-step recipe over 384 rounds may move little — that is the registered
  mechanism's answer, not a defect to tune away.

### 9.3 Soundness check — no fatal flaw; required disclosures / fixes
- Leakage: CALIB ⊂ MAIN and CALIB_VALIDATION ⊂ the read window — the Phase-1 precedent, disclosed (D-b). Selection tool refuses MAIN / TEST manifests (test). OK.
- Multiplicity: Q1 unadjusted primary; Holm over Q2–Q7 (+Q8); tertiary with DESCRIPTIVE bands; Q3 = Q1 − (C1_FROZEN − PRE) dependency listed. OK.
- **E-S1 Power (must be in the registration, D-h):** MDE ≈ 0.00047 vs a plausible ceiling ≈ 0.001 → the study is powered for roughly the full ceiling, not
  half of it. Register the §9.4 grid as the reading (SUCCESS / EQUIVALENT by TOST ±0.0005 on the 90 % CI / FAILURE / INCONCLUSIVE) and write D-h:
  "Phase 2 can confirm a gain ≥ ≈ 0.0005 MRR@20 or equivalence within ±0.0005; a true gain between them is likely to read INCONCLUSIVE." No promise of a positive result.
- Confounds: C1_FULL / C1_FROZEN vs W* (Q3 same trained subset) and C2 (matched size / recipe) make a null interpretable; the interpretation grid is ACCEPTED.
  **E-S2:** C1's CALIB screen measures headroom at 32,768 users, MAIN at 131,072 — the futility rule may under-read headroom; disclose it beside FUTILITY_SKIP_WB_WC.
- **E-S3** D1-ext observer: the test must prove θ trajectory, ledger and checkpoint digests bit-identical with the observer on and off (as planned) and that the
  DP path logs post-noise quantities only (06L_E E1). Privacy audits E1–E5 and devices (§8) are DESCRIPTIVE / tertiary and ACCEPTED as written; E1's PRE-null
  validity check is required before any ε_LB is reported. UNEW_HOLDOUT TRAIN-window rows as non-members do not touch any TEST row — OK.
- **E-S4 Scope split:** Part A (C0 / C1 / WA–WC / W* ×2 / FT_P2_WSTAR ×2 / DP chain / Q1–Q11; P2-00..07, 12, 13) is the critical path to the sitting.
  Part B (SecAgg, DP-FTRL, LDP, audits, devices; P2-08..10, 14) runs in parallel and is declared not run where unfinished — it never delays the sitting.
- O3 (C2): recommend YES (14 GPU-h; removes the M8 confound) — owner. O5 (second θ_pre seed): recommend NO (D-c discloses it). O6: NO (scope). O8: owner's devices; thresholds are DESCRIPTIVE design targets, accepted.

### 9.4 Process
- Model assignment of the plan is right: **Opus worker + Opus review** for P2-02 (digest-sensitive), P2-04 (LoRA), P2-06 (selection = decides the arm), P2-09,
  P2-10, P2-11, P2-12; **Opus worker + Sonnet review** for P2-01, P2-03, P2-05, P2-13; **Sonnet** for P2-07, P2-14 and the lock tool of P2-00.
- Fable touches exactly five points: (1) **a delta re-review of the edited `contracts/PREG2.md` before registration — REQUIRED** (text only, ≤ 30 min);
  (2) P2_METHODS_LOCK before the sitting; (3) P2_SCRIPT_FREEZE; (4) the DP-FTRL accountant concurrence; (5) the TEST2 manifest spec if O2 = yes.
  Everything else is Ops + the listed reviewers. Every run wave gets a DL row; the thesis reference is updated after registration and each selection.

### 9.5 DL text
"Fable gate on the Phase-2 design (PREG2_FOLLOWUP_DRAFT + PHASE2_IMPLEMENTATION_PLAN; reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003 §9): GO-WITH-EDITS.
O1 = T0a (one TEST sitting for Phase 1 + Phase 2); O9 = YES: a second registered study may score on the RETRO_TEST manifests built under ALL_METHODS_LOCK
04a3cd19 (data built COUNTS_ONLY, no model scored) under conditions S1–S7 (Phase-1 plan unchanged; P2_METHODS_LOCK + P2_SCRIPT_FREEZE registered before the
first TEST reeval of any arm; LOCKED_EVALUATION entries for both locks; all scoring then both analyses, nothing fetched before both RETENTION_RESULT files
exist; unfinished P2 arms declared not run; no second RETRO_TEST sitting ever; labels CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) with D-a). T0b REFUSED as a
confirmation route; TEST2 = optional pre-registered replication (owner O2; Fable approves the spec), never the Q1 confirmation. O4: grids accepted; CALIB tie
band and futility threshold 0.001; TEST equivalence margin ±0.0005; cap 250 GPU-h; drop order by tier (tertiary → exploration → C2 last). O7: T = 384 and the
registered z kept. Required additions: power disclosure D-h and the §9.4 reading grid; futility headroom caveat; D1-ext bit-identity tests; Part A / Part B
scope split. Phase 2 has GPU priority over the ML-1M move (§8); the Phase-1 TEST result waits (INFERENCE ≈ 10-17..10-21). Fable: delta re-review of PREG2.md
before registration, then P2_METHODS_LOCK, P2_SCRIPT_FREEZE, DP-FTRL accountant, TEST2 spec. No value read."

## 10. Delta re-review of contracts/PREG2.md (3958e365…) + P2_CONTRASTS_v1.json (self 18fa8b67…) + PHASE2_IMPLEMENTATION_PLAN (T0a) — **GO to register after five textual edits (no further Fable pass; Ops verifies by diff)**

Read-only. MEASURED: the PREG2.md changelog maps every §9 item to a section and the diff against the unchanged draft confirms each: T0b removed /
refused, TEST2 = replication only, S1–S7 verbatim, labels (CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP), FRESH_PERIOD_AUDIT, REPLICATED_FRESH…), CALIB tie
band 0.001 with the TEST margin ±0.0005 kept apart, futility 0.001 + D-j, drop order by tier, T = 384 / registered z / passes 2 + D-i, D-h and the §9.4
reading grid, D1-ext bit-identity + E1 PRE-null-first, Part A / Part B, O5 = NO, O6 = NO, S7 GPU precedence. The plan is consistent (T0a only; P2-08a Part A,
P2-08b Part B; 0.001 in P2-06 and its tests; drop order; S7). No value read.

**Consequential edits confirmed:** (i) NO_MEASURABLE_CALIB_GAIN at 0.001 — yes, it must equal the CALIB tie band. (ii) Q9–Q14 = TERTIARY_P2 — yes; Q14 is
reported only in the TEST2 family (TEST2_NOT_AVAILABLE otherwise), never in the RETRO_TEST sitting's band.

**Edits before the O_EXCL write (text only):**
- **E1 (§6.3 S5).** Name every Phase-1 arm whose TEST rows Phase 2 reuses: PRE, FA_WARM_FROZEN, FT_PRE (Q5) and C_FULL (Q9), "produced once by the Phase-1
  plan, hash-bound into both blocks".
- **E2 (§6.3, new S8 — sealing of Phase-2 MAIN values).** "No Phase-2 MAIN VALIDATION value (RESULT_ROWS, metrics.jsonl, best_metric, REPORT lines) is
  opened by a human before both sitting analyses are written; the PRACTICAL_BEST checkpoint is the run's own mechanical rule; the selection tools read
  CALIB rows only (test). POST_VALIDATION_EXPLORATORY is the label under which they are reported after the sitting." Reason: otherwise tiers, contrasts or
  the Q8 decision could be adjusted with knowledge of the Phase-2 MAIN deltas — the same discipline Phase 1 kept until §9.
- **E3 (§12 deadlines).** O3 (C2 / Q8) is decided by DL row **before the first Phase-2 MAIN run**; O2: TEST2 joins the sitting only if its manifests, gate
  and register exist before P2_METHODS_LOCK, else it is a later replication; O8 before P2-14 starts.
- **E4 (§13 TO-VERIFY, frozen-tool constraint).** `tools/test_exposure_register.py` (7d59e828) is in ANA_SCRIPT_FREEZE_v3 `scripts` and may NOT change
  (freeze_check would refuse H). Ops verifies that its purpose enum admits the two LOCKED_EVALUATION entries with different lock shas; if it does not, the
  Phase-2 entry uses the existing look purpose with the P2 lock sha in `all_methods_lock_sha256` and the P2 lock named in `observations_description` —
  never a tool edit.
- **E5 (P2_CONTRASTS_v1.json — register NOW, in the same DL row as PREG2.md, not at P2_METHODS_LOCK).** Reason: E2 — the registry must be fixed before any
  Phase-2 run exists. Fix first: (a) `self_sha256` does **not** verify under the utf8-compact `sort_keys` rule that retention_report `_self_ok` and G use
  (MEASURED: recomputation ≠ 18fa8b67) — recompute it exactly as company/gates/*.json do; (b) `draft` → false, `status` → REGISTERED wording, `authority[0]`
  → "contracts/PREG2.md sha <registered>"; (c) write the Q8 rule in-file: "Q8 is in the Holm family iff O3 = YES (DL row before the first Phase-2 MAIN run);
  if NO, P2_CONTRASTS_v2 removes Q8 and nothing else, its sha registered before P2_METHODS_LOCK". Content otherwise ACCEPTED: hierarchy (Q1 primary; Holm
  Q2–Q7 (+Q8); tertiary Q9, Q11, Q12a/b, Q13, Q14; Q10 derived), rule map (Q6/Q7/Q11/Q12 CONTROLLED_BUDGET), seeds, populations (Q14 TEST2_UNEW_HOLDOUT),
  reading_rules Q1 (SUCCESS / EQUIVALENT ±0.0005 / FAILURE / INCONCLUSIVE) and Q13 (TOST [99, 101]), dp_disclosure, algebraic dependencies, empty 06P keys.

**Not required:** another Fable pass. Fable next at P2_METHODS_LOCK, P2_SCRIPT_FREEZE, the DP-FTRL accountant, and the TEST2 spec (if O2 = yes).

**DL text.** "Fable delta re-review of contracts/PREG2.md (candidate 3958e365) + company/gates/P2_CONTRASTS_v1.json + PHASE2_IMPLEMENTATION_PLAN (T0a):
GO to register after five textual edits (§10 E1–E5; no further Fable pass): S5 names PRE / FA_WARM_FROZEN / FT_PRE / C_FULL as the reused Phase-1 rows; new
S8 seals every Phase-2 MAIN VALIDATION value until both sitting analyses are written (selection reads CALIB only); O3 decided before the first Phase-2 MAIN
run, O2 in the sitting only if TEST2 is ready before P2_METHODS_LOCK, O8 before P2-14; test_exposure_register.py (frozen, 7d59e828) is never edited — Ops
verifies its enum admits the two LOCKED_EVALUATION entries; P2_CONTRASTS_v1 registered NOW with PREG2 (same row) after self_sha256 is recomputed under the
utf8-compact rule (the current value does not verify), draft false, Q8 conditional rule in-file (P2_CONTRASTS_v2 removes Q8 if O3 = NO). Confirmed:
NO_MEASURABLE_CALIB_GAIN at 0.001; Q9–Q14 TERTIARY_P2 (Q14 in the TEST2 family only). Then: PREG2.md O_EXCL read-only + sha; P2_CONTRASTS_v1 read-only + self
sha; both in this row. No value read."

## 11. B16 concurrence (ML-1M → NVIDIA_A40-FP32-DET; read-only) — **CONCUR WITH EDITS (E1–E5), then register before the first production GPU run**

MEASURED (DL 409 / 412; smoke only, SMOKE_NOT_EVIDENCE, no metric read): GPU two-run determinism PASSED (test + C_FULL L200 B24 ac7aeadb ×2, FA L200
57295801 ×2); CPU path byte-identical vs the real code_bench4; C_FULL L200 B24 ≈ 34 min vs ≈ 5.4 days CPU; FA L200 ≈ 1.5 h; projected ML-1M final ≈ 10-05/06
≤ 10-08 → the §8 timing gate is met. Wave 2 stays CPU (0.95 s / round). No ML-1M value read; trackb-select not run. The §8 conditions (code_bench5 beside
bench4, device class in every record, mixed device refused, hard switch, no two complete sets, fallback only by DL row for a mechanical cause, GPU precedence,
pass-2 blocker) are all in the draft. Phase 2 is 3–4 days from its first run, so the GPUs are otherwise idle: the move costs nothing on the critical path.

**Edits:**
- **E1 (B16.4 — the TEST look device).** Replace "scored on CPU by the unchanged `final`" with the uniform rule: **one device class per dataset for every
  computation of that dataset, training and TEST look alike.** For ML-1M, `final` (scoring AND the FT arms' per-user fine-tune at TEST) runs with
  `--device cuda` under NVIDIA_A40-FP32-DET (BM-05: the BM-03 device plumbing applied to `final`; tests: CPU path byte-identical, GPU two-run identical
  TEST digests on a smoke; Sonnet review). The ML-1M LOCK pins code_bench5 including `final`. S3 ×3 keep `final` on CPU (code_bench4, unchanged).
  Reason: FT arms selected on GPU-trained runs but fine-tuned at TEST on CPU would be a device mix inside the dataset; mixed is REFUSED. If BM-05 is not
  reviewed when the graph completes, the LOCK waits for it (hours), never the mixed route.
- **E2 (B16.3 — relaunch guard).** The CPU Wave-1 driver re-runs failed jobs (`--resume`, DL 360), so killing ML-1M processes alone creates `.fail` markers
  it would relaunch. Order: (1) mark every ML-1M job in the CPU driver's state dir as SUPERSEDED_DEVICE in a form the driver treats as never-run-again
  (DONE-equivalent skip or an explicit dataset exclusion), (2) stop the ML-1M processes (pids logged), (3) run the driver in plan / dry mode and MEASURE that
  its plan holds no ml_1m job, (4) seal `runs_bench/ml_1m` read-only with the marker. S3 `.ok` states untouched; Wave 2 untouched.
- **E3 (B16.5).** Add: the GPU driver never shares a GPU with an H step (S7) and yields between jobs to any Phase-2 lane or H step through a yield file it
  polls; `laneD3_INT8_pass2.sh` `W1_BUSY_RE` gains `run_ml1m_gpu.sh` (new lane sha in a DL row) so K16 pass 2 waits for it.
- **E4 (B16.2).** Label the determinism evidence "smoke-scale; the production guard is the runner's device-class refusal set (BM-03 NCs); full-length runs are
  not re-run for determinism"; the smoke roots stay SMOKE_NOT_EVIDENCE and are never indexed by the LOCK.
- **E5 (B16.6).** Disclosure also in the Track-B table and the cross-dataset table ("device class differs by dataset; ratios are within-dataset").

**BM-04 driver review:** Sonnet is enough, with this checklist in the review: DAG = the registered ML-1M graph (Track A → decide-warm → seed3; Track B full
grid per B15 → trackb-select → winners' s2027; LOCK gate last); write-once state only under `runs_bench_gpu`; no job-name collision with `runs_bench`; plan /
dry mode on the real state before launch (Ops reads it); resume = identical hashes (runner refuses otherwise); one yield test (stop between jobs, resume);
exactly one ML-1M driver anywhere (the CPU branch dead per E2); explicit GPU slots; logs value-free (names, rc, times only).

**DL text.** "Fable B16 CONCURRENCE (§8 condition; reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003 §11): timing gate met (MEASURED smoke: GPU determinism PASS,
C_FULL L200 B24 ≈ 34 min vs ≈ 5.4 days CPU, ML-1M final ≈ 10-05/06); CONCUR WITH EDITS: E1 one device class per dataset for training AND the TEST look —
`final` gains --device cuda for ML-1M (BM-05, Sonnet review; LOCK pins code_bench5 incl. final; S3 unchanged on CPU; mixed REFUSED); E2 relaunch guard before
killing CPU ML-1M jobs (state marked SUPERSEDED_DEVICE, driver plan verified free of ml_1m, dirs sealed); E3 GPU driver yields to Phase-2 / H via a yield file,
pass-2 lane pattern updated; E4 evidence labelled smoke-scale, runner refusals are the production guard; E5 disclosure in the Track-B and cross-dataset tables.
BM-04 driver: Sonnet review with the §11 checklist + Ops plan-mode read before launch. Register B16 (O_EXCL, read-only, sha) before the first production GPU
run; CPU ML-1M dirs never opened. No value read."

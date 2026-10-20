# 11 — File map

*Root = `.local_ai/ppsi_v4/` ("V4/"). "Authoritative" = the binding record; "archive" = kept for provenance, do not use for
new work. Sizes MEASURED 2026-10-03.*

## 1. Top level of V4/

| Path | What it is | Status |
|---|---|---|
| `HANDOVER/` | these 14 documents + `reference/` (byte copies of key primary files) | guide (originals win) |
| `OWNER_SCOPE_V4.md` | the V4 scope receipt (package `PPSI_SINGLE_BACKBONE_V4_2026_09_26`), gate order | authoritative scope |
| `company/` | management records (below) | authoritative |
| `contracts/` | amendments, locks, contracts, selection files (below) | authoritative |
| `code/` | the source of every deployed tree (below) | authoritative source |
| `tools/` | analysis and ops tools (below) | frozen where stated |
| `results/VALIDATION_v3/` | output of the single VALIDATION read: REPORT.md, CLAIM_REGISTER.json, HEADLINE_TEMPLATE.md, METRIC_TABLE.csv, OUTPUT_MANIFEST.json (RETENTION_RESULT.json and curves/ are server-only) | authoritative copy (DL 400) |
| `deploy_archive/` | tars + SYNC_MANIFESTs of trees r3, s2, r4–r9, r7b, bench5, bench5b, p2r1; ENV_LOCK_A40 | archive / provenance |
| `data_v2/`, `data_v3/`, `data_fs/` | local receipts and copies of the DATA-06 / DATA-07/08 / FS builds | archive |
| `census_v5_100/`, `featstat_v5_100/`, `g0/` | population census, feature statistics, G0 resource audit (`census_v5_100/_superseded_run1/` is QUARANTINED, DL 33) | archive |
| `runs_g1/`, `runs_fs/`, `runs_p2/`, `runs_p2_copies/`, `runs_p3/`, `runs_ds/`, `runs_ds3*_deploy/` | small local copies of server receipts / scripts of earlier phases (contest P2 = equal-effort tuning, not Phase 2) | archive |
| `logs/`, `review/`, `state/` | empty | — |
| `_pytest_tmp/` (≈ 300 MB), `code/_pytest_tmp/`, `code/.train03r1_tmp/`, `__pycache__/` | test scratch | **exclude from any ZIP** |

## 2. company/

| Path | Content |
|---|---|
| `DECISION_LOG.md` | **the** chronological record (≈ 435 rows) |
| `RESUME_HERE.md` | resume blocks, newest on top (§2t = 10-03); older copies `RESUME_HERE_prev_*.md` |
| `COMPANY_CHARTER.md`, `MILESTONES.md`, `RESOURCE_BOARD.md`, `RISK_REGISTER.md`, `TASK_BOARD.json` | operating model (early; TASK_BOARD stale after 09-27) |
| `gates/` | `ALL_METHODS_LOCK.json` (`04a3cd19…`), `ANA_SCRIPT_FREEZE{,_v2,_v3}.json` (v3 binding), `ANA_CONTRASTS{,_v2,_v3,_v4}.json` (v4 binding), `P2_CONTRASTS_v{1,2,3}.json` (v3 binding), `REGISTER_ENTRY_5_v3.json`, `REGISTER_ENTRY_6_v3.json`, `test_build_v3/` (TEST_EXT registry + DATA-09 receipt), G1 / G1B / G2 files, `_drafts_*`, `_withdrawn_*` (never used) |
| `registers/` | `TEST_EXPOSURE_REGISTER.json` (6 entries), `FL_IMPROVEMENT_LADDER.md` |
| `tasks/` | work-packet instructions (scope, owned files, tests) |
| `handoffs/` | what each worker did, with shas and test counts (ANA-*, BM-*, DATA-*, DEV-*, FL-*, INT-*, P2-*, OPS-*, …) |
| `reviews/` | independent reviews `REV-*.md`; Fable rulings / gates / audits `FABLE_*`, `*_FABLE_*`; position memos and syntheses; `MULTI_EXPERT_REVIEW_20260929/` (14 parts); realism / ladder / practice audits |
| `plans/` | `PHASE2_IMPLEMENTATION_PLAN.md`, `MASTER_PLAN_V5.md` (the 09-28 plan; §9/§11 commands partly stale, see endgame README), `BENCHMARK_PLAN_V1.md`, `REST_OF_PROJECT_PLAN.md`, others (historical) |
| `plans/ops_scripts/` | every server script: `endgame/` (B, B2, P, D, E, F, G, H + lib + pack/deploy r7–r9 + `metric_table.py`), `lanes_r6w/` (warm line, DP chains, N33, D3, INT8 lanes), `bench/` (Wave 1 / 2 drivers, ML-1M GPU driver, B13 stopper, LOCK_FINAL_RUNBOOK), `p2/laneP2_RUN1.sh`, plus pack / deploy / status / watch scripts |
| `handover/PPSI_V4_TEAM_HANDOVER.md` | the 09-30 narrative (state DL 268) |
| `presentations/` | 30 Sep supervisor deck (pptx / pdf variants), speaker script, Arabic guide, data notes, build scripts |
| `thesis_draft/` | outline, chapter 3, chapter 4 drafts (no results) |
| `research/` | BM-00 protocol and comparisons; literature register |
| `evidence/reeval_records/` | real run records used as test fixtures |
| `decisions/` | ADR-001 … 005 (early architecture decisions) |
| `owner/` | the owner's POP 5–100 amendment text |
| `prompts/` | prompts used for the Fable deep sessions and reviews |
| `receipts/phase1/` | T3 A40 receipt etc. |

## 3. contracts/

`PREG1_AMENDMENT_01 … 06P.md`, `PREG1_AMENDMENT_07.md` + `_ADDENDUM_B12 … B16.md` (binding; `_DRAFT` / `_drafts_*` = not
binding), `PREG2.md` (binding; `PREG2_FOLLOWUP_DRAFT.md` and `_drafts_PREG2_candidate_3958e365.md` = drafts),
`ARCHITECTURE_LOCK.json`, `DATA_PROTOCOL_LOCK_V2.json` (v1 historical), `PREG1_LOCK_VALUES.json`, `EVALUATION_CONTRACT_V4.md`,
`CLIENT_CONTRACT_V1.md`, `DATA_CONTRACT_CLEAN_V2.md`, `FEATURE_SCHEMA_CLEAN_V2.json`, `MANIFEST_SPEC_V4.json` (+ addendum);
sub-folders `fl/` (LR_SELECTION, LR_SELECTION_EXT, FL_TUNING_S2 / S3 / SELECTION), `dp/` (S_RECORD), `device/` (SELECTION_FT_C,
SELECTION_LO), `eval_device/` (A40 pin), `manifest_registry/`, `model_variants/`, `hp_registry/`, `derived_schemas/`,
`strata/` (empty locally).

**Server-only registered files (not in the ZIP):** `contracts/fl/FL_WARM_SELECTION_S7.json` / `_S8.json` (copies:
`../thesis_reference/sources/local/S579_*`, `S580_*`), `contracts/dp/S_RECORD_WARM_FROZEN.json`,
`contracts/device/SELECTION_FT_FA.json`, `contracts/strata/STRATA_CUTPOINTS_MAIN.json`,
`contracts/manifest_registry/REGISTRY_V2_TEST_EXT.json` (local copy in `company/gates/test_build_v3/`).

## 4. code/ and tools/

| Package | Role |
|---|---|
| `v4eval/` | the frozen evaluator (full ranking, metrics, bootstrap, device pin) — **never edit** |
| `v4models/` | SASRec / GRU models and variants |
| `v4train/` | central trainer, schedules, selection, recipes |
| `fl/` | federated runtime: server (FedAvg / FedAvgM / FedAdam), client solvers, DP + accountants, quantisation, personalisation, comm ledger; Phase-2 `update_log.py`, `sparse_rows.py`, `peft.py` |
| `v4int/` | launchers and integration: `launch_central.py`, `launch_fl.py`, `launch_device.py`, `launch_trunc.py`, `lr_screen.py`, `reeval.py`, `pretrain_small.py`, `c_full.py`, `retro_store.py`, `dump_rows_retro.py`; Phase 2 `central_ft.py`, `c_full_small.py`, `p2_select.py`, `p2_probe.py` |
| `v4data*/` | data builders: `v4data09` (gated RETRO_TEST manifests), `v4data10` (pretraining band), `v4data08` (C_FULL), `v4data06`… |
| `v4dev/` | ONNX export / quantise / bench, rows dump, browser pages (`browser/`, Phase-2 `p2_bench.*`, `serve_coi.py`) |
| `v4base/` | baselines (POP, LAST, SKNN, RULE, KNN_EMB) and strata |
| `v4trunc/`, `v4hist/`, `v4fs*/`, `v4sup/` | truncation study (not run), history windows, feature search, support |
| `v4bench/`, `v4bench_w2/` | benchmark runner (code_bench*), Wave-2 converters and sampled metrics |
| `tests_*` | one suite per package (run with `pytest -p no:cacheprovider`, temp dir on F:, NCs via `--runxfail -k test_nc_`) |
| `_archive_*` | pre-edit copies kept for byte-identity checks |

`tools/`: `retention_report.py` (`9a3b2c2a`, frozen), `ana_block_index.py` (`6e7043bb`, frozen), `test_exposure_register.py`
(`7d59e828`, frozen), `g2_run.py` (contest G2), `code_sync.py`, `write_data_protocol_lock*.py`, Phase-2 `p2_t3_check.py`,
`p2_noop_check.py`, `p2_edge_server.sh`, `p2_packages_sha.py`; `_archive_r7/`, `_archive_r8/` (previous tool versions).

## 5. Outside V4/

- `../thesis_reference/` — the living thesis reference (STATE.md, CHANGELOG.md, CLAUDE_EXECUTION_REFERENCE_EN.md, registers,
  sources S001–S602, tools `add_source_batch.py`, `verify_reference.py`, `verify_ref11.py`).
- `../sasrec_v1/tools/lobot.py`, `../gru_v1/tools/lobot.py` — the server channel (token stored outside the project).
- `../gru_v1/`, `../sasrec_v1/`, `../feature_opt_v1/` — earlier campaigns (read-only history; source of TEST prior exposure).
- `F:\…\eda\rees46_nextitem_v1` — canonical local copy of the frozen dataset; `eda/data/raw/` — raw REES46 CSVs.

## 6. What is NOT in the ZIP (server only)

The frozen dataset and raw data on the server; every model checkpoint (C_BIG, C_SMALL, C_FULL, PRE, all FL / device runs);
all run directories (`runs_p3`, `runs_ds`, `runs_ds3`, `runs_pretrain`, `runs_p2`, `runs_bench*`), their RESULT_ROWS / rank
files / logs; the VALIDATION block and RETENTION_RESULT.json; the deployed code trees; benchmark raw / processed data
(`bench_raw`, `bench_proc*`); the server-only registered files of §3. Access them through the server (file 07).

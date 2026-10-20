# G0 evidence packet for Fable (planning / reconciliation). Facts and questions only.

You are Fable 5.1, the Research Director. You are read-only, and your ruling is binding within the owner's scope.
- **Scope:** `F:\Queens University\Microsoft Project\eda\.local_ai\ppsi_v4\OWNER_SCOPE_V4.md`.
- **Plan package:** `F:\Queens University\Microsoft Project\eda\PPSI_SINGLE_BACKBONE_V4_2026_09_26\PPSI_SINGLE_BACKBONE_V4_2026_09_26\`.
  Read 02–09 as needed.
- **Your independence:** form your own view. The lead's position exists in a separate memo, and you will see it only
  after you answer.
- **Evidence roots:** `...\.local_ai\ppsi_v4\g0\` and `...\.local_ai\ppsi_v4\company\handoffs\`.

## A. Owner scope in one paragraph
1. GRU and SASRec are compared centrally on one clean protocol, and ONE family wins on VALIDATION: the mean client-macro
   MRR@20 over seeds 2026/2027.
2. Only the winner goes on to C / FedAvg / FedProx / joint PFL / CPF (a capacity-matched central-personalized control)
   / Local-only, plus cold-start and on-device feasibility.
3. FedBN, Purchase Intent and loser downstream runs are out.
4. Data, split and OOV changes are allowed before the new lock, with limits: v1 stays immutable, no score-driven
   filtering, and inspected periods stay retrospective.
5. There is no deadline, but no waste.
6. The thesis question is how much quality is retained from centralized to FL, PFL and LO under a common data,
   architecture, capacity and evaluation contract.

## B. Live resources (PROJECT_MEASURED 13:31Z; `g0/SERVER_CENSUS.json`)
- **GPUs:** 2× A16, 15.4 GiB each, idle.
- **Memory:** cgroup 64 GiB.
- **Disk:** 12.8 GiB free on a shared 49 GiB volume. The target is ≥ 10 GiB free; the floor is 8 GiB.
- **Software:** torch 2.13, flwr 1.33, onnxruntime 1.24.1 (CPU), onnx 1.20.
- **Raw CSVs:** only on the local PC; not on the server.
- **v1 dataset:** frozen with hash 860e5a1c…, byte-identical locally.

## C. Data facts (PROJECT_MEASURED, exact; `g0/data_census/*`, handoff `DATA-01.md`, `g0/BARRIER_COUNT.json`)
- **Old v1 splits:**
  - TRAIN [10-01, 11-16): 52.76M decisions; 40,180,608 loss examples.
  - VS [11-16, 11-20) and VC [11-20, 11-23) → proposed V4 VALIDATION [11-16, 11-23): 14,357,645 decisions; 11,288,056
    eval rows; 96,871 target-OOV; 1,578,912 users.
  - TEST [11-23, 11-30): 5,862,968 eval rows, previously inspected.
  - LABEL_GRACE [11-30, 12-01): 0 decisions.
- **Label time:** 0 TRAIN loss rows with a target observed at or after 11-16; 0 VALIDATION rows past the end; 133 TEST
  targets matured in grace.
- **Targets are within-session and within-partition.** Only 170 VS decisions are censored purely by the old Nov 20
  VS/VC barrier. The internal barrier can be kept (minimal-diff) or re-derived at a cost of 170 targets.
- **Catalogue** (support = TRAIN representative views; reproduced with 0 mismatches):

  | Support | K | TRAIN positive coverage | VALIDATION label coverage |
  |---|---|---|---|
  | ≥ 5 | 158,486 | 99.859% | 99.142% |
  | ≥ 2 | 179,315 | 99.978% | 99.464% |
  | ≥ 1 | 191,890 | 100% | 99.556% |

- **Clients** (one persistent user = one client):
  - 2,694,998 warm users with ≥ 1 eligible TRAIN example. Per user: median 6, p90 35, p99 138, max 11,628. 473,402
    users have exactly 1 example.
  - VALIDATION: 1,023,523 users with queries (642,228 warm, 381,295 cold).
  - TEST: 746,662 users (420,693 warm, 325,969 cold).
- **Nested salted-hash cohorts** of warm users (salt PPSI_V4_POPULATION_20260926):

  | N | TRAIN examples | Users with ≥ 1 VALIDATION query | VALIDATION rows |
  |---|---|---|---|
  | 16,384 | 252,870 | 3,903 | 49,911 |
  | 65,536 | 990,362 | 15,487 | 200,985 |
  | 262,144 | 3,906,929 | 62,368 | 808,610 |
  | ALL | 40,180,608 | 642,228 | 8,430,391 |

- **Open semantics:** `AMBIGUOUS_BOUNDARY` (same-timestamp multi-item barrier) and `BOUNDARY_UNRESOLVED` have no prose
  definition in the contract. They are rare: 1,773 + 170 in VS.

## D. Costs (`g0/BENCH_*.json` PROJECT_MEASURED, synthetic shapes, strict FP32; `g0/COST_FORECAST.json` INFERENCE)
- **Parameters:**
  - GRU (executed ContextGRU F4, item 128, hidden 256, tied head): 20.96M.
  - SASRec (executed v3 A1, d256, 2 blocks, 4 heads, untied head plus recentering): 83.09M.
- **One local step** (forward, full-K CE, backward, clip, AdamW) at b = 1 / 16 / 64 / 256:
  - GRU: 18.5 / 21.4 / 28.1 / 56.2 ms.
  - SASRec: 60.4 / 61.0 / 76.8 / 142.8 ms.

  The floor comes from the full-K head plus AdamW over every parameter.
- **Central throughput at B = 256:** GRU about 4,556 decisions/s; SASRec about 1,793 (strict FP32).
- **State copy:** 1.1 / 4.1 ms. **Virtual bytes per client visit (down + up):** 168 / 665 MB.
- **Forecast** (6 EFE, 2 passes, 3 visits per client, local batch 64; excludes data loading, evaluation and
  calibration):

  | Population | Downstream core, 2 seeds (FA + FP + PF + LO + C/CPF), GRU / SASRec | Central 6 EFE per seed, GRU / SASRec |
  |---|---|---|
  | ALL | 908 / 2,701 GPU-h | 14.7 / 37.4 h |
  | 262,144 | 88 / 263 GPU-h | 1.4 / 3.6 h |
  | 65,536 | 22 / 66 GPU-h | 0.4 / 0.9 h |

## E. Prior results that constrain the plan (PROJECT_MEASURED; living reference S217, S236)
- **SASRec campaign** (v1 protocol, full TRAIN, TF32 training / FP32 evaluation, retrospective TEST): A1 e5 scored
  0.170475 against GRU F0 at 0.168695, a difference of +0.00178 with CI [+0.00154, +0.00202]. The ID-only SASRec was
  below GRU (−0.0022, descriptive). The VS→TEST margin shrank from +0.0126 to +0.0018.
- **Feature study:** KEEP_CURRENT.
  - D3 channels gave +0.00026, below the gate.
  - Lean parity failed its rank-identity gate: 0.998683 against 0.999.
  - M1 is confirmed on all of TRAIN: a time-gap normalizer was fitted with mean 1.86 over all rows, against 11.82 over
    valid rows only.
  - The full-population duplicate and dead-flag checks were 0 in 6 relations (8 columns) plus 6 dead flags.

## F. Feature locality map (DATA-02; code-cited; `g0/feature_map/FEATURE_LOCALITY_MAP.{json,md}`, `OPEN_ITEMS.md`)
- **Coverage:** the 59 rich input channels are 6 tokens, 19 numeric, 15 flags, 12 user-context and 6 masks, plus the
  output catalogue.
- **By class:** LOCAL_EVENT 27, LOCAL_HISTORY 26, SHARED_CATALOGUE 3, SHARED_TRAIN_STATISTIC 3, UNSUPPORTED_FUTURE 0.
  - SHARED_CATALOGUE = the category, brand and main-category tokens. They are recovered per product, pooled across all
    users (`metadata.py:73-156`, `pipeline3.py:55`).
  - SHARED_TRAIN_STATISTIC = the item-token vocabulary and support gate, `log1p_item_observed_age_s` (anchored to a
    TRAIN-pooled `first_seen_train`), and the output catalogue.
- **Pooled information:** 36 of 59 channels depend on other users' pooled interactions. For example, every z-scored
  numeric channel inherits a normalizer pooled over TRAIN. This is the "bootstrap information" every method would
  share.
- **M1 root cause:** `log1p_time_since_prev_user_event_s` and `_prev_session_last_event_s` are zero-filled when
  unavailable (`userctx.py:224-235`). They are missing both from the valid-fit predicate (`pipeline6.py:399-407`) and
  from the load-time re-zero map (`loaders.py:60-63`).
- **A look-alike that is not M1:** the category-run counts encode a genuine zero as z(0) ≠ 0.
- **Open items:**
  - a duplicated usable-rule implementation (`metadata.py` vs `pipeline4.py`);
  - `category_code_missing` vs `main_category_missing` diverge on empty roots;
  - `userctx.py` is not byte-verified against the frozen build.

## G. ONNX export feasibility (EDGE-01; `g0/export_smoke/*_REPORT.json`; random-init weights; server-CPU PROXY only)
- **Exporter mode:** the default dynamo exporter is unusable because `onnxscript` is not installed (no installs are
  allowed). Only the legacy exporter was tried.
- **SASRec: WORKS.** Opset 18; dynamic batch; sequence length baked to L = 50 (the model uses `lengths.max().item()`,
  and right padding keeps per-row masking).
  - **Parity with torch FP32:** max |Δlogit| 1–2.1e-6; top-20 and target rank identical in 8/8 scenarios (lengths
    1/8/50, batch 1/16, missing features, mixed padding).
  - **Server-CPU ORT latency:** 5.7 ms (length 1, batch 1); 36.6 ms p50 and 37.9 ms p95 (length 50, batch 16).
  - **Sizes:** full model 317 MB; query-only 161.6 MB.
- **GRU: BLOCKED natively.** `pack_padded_sequence(enforce_sorted=False)` crashes the legacy exporter's peephole pass
  (`vector::_M_range_check`) at opsets 12–18, and opset 9 fails on `Sort`.
- **Pending (EDGE-02):** an export-only adapter. It runs the unchanged GRU on the padded tensor and gathers the state at
  `lengths-1`. This should be exactly equivalent, because `DATA_CONTRACT.md:116` guarantees right padding (INFERENCE
  until the parity test is run).
- **The lead's note:** device convenience must not decide the family (package 05 §D). This is recorded as feasibility
  evidence only.

## H. Decisions requested (answer each with evidence and exact lock values)
1. **Population:**
   - FULL, or a locked cohort? If a cohort, which N?
   - Does the central family contest run on the same population as the downstream methods, or on ALL?
   - How are cold users evaluated?
2. **Catalogue:** the threshold (1, 2 or 5), and whether support is fitted on the full TRAIN or on the cohort TRAIN.
3. **VALIDATION:** the merged window, minimal-diff (keep the Nov 20 barrier) or re-derived. Any subperiod reporting.
4. **Feature set and transforms:** clean Rich = v1 rich with the proven redundancies and dead flags removed, plus the M1
   valid-only refit, trained from scratch. Keep, amend, or specify.
5. **Precision:** strict FP32 everywhere, or a preregistered TF32 branch?
6. **Central recipes, per family:** passes, LR plan, and whether bounded LR calibration is warranted on the chosen
   population. Also the stopping rule.
7. **FL, PF and LO settings:**
   - clients per round, local passes and local batch;
   - the local solver and optimizer-state lifecycle;
   - the FedProx μ grid;
   - PF private dimension and regularization;
   - LO budget;
   - endpoint and cap EFE.
8. **Evaluation:** primary and secondary metrics; the bootstrap unit; per-seed reporting.
9. **Device target:** a declared proxy, or waiting for owner hardware.
10. **Blockers and optional work to skip.**

Return format: ROUTING, VERDICT, numbered DECISIONS (with evidence path and lines), LOCK_VALUES, BLOCKERS,
OPTIONAL_WORK_SKIPPED, OPEN_DISAGREEMENT_RISKS. Keep literature-backed facts separate from project decisions.

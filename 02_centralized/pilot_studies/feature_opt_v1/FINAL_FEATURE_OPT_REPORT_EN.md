# Feature study final report — `SASREC_FEATURE_OPT_V1` (English companion)

> A separate add-on study to the SASRec campaign. **All numbers are on VALIDATION_SELECT only** (10k-user cohort, 112,365 decisions),
> a split already used for the campaign's selection; this is adaptive development, not a fresh holdout. VC and TEST were never read.

## 1. Summary
- **Decision: `KEEP_CURRENT`**: the pinned incumbent stays unchanged.
- Pin: A1_RICH256 seed 2027 epoch 5 (VS 0.186513). σ̂ = 0.000869 (max two-seed gap; a range, not an SD) → gate-2(b) threshold max(0.002, 2σ̂) = 0.002000.
- **C1_D3** (5 causal repeat channels, zero-init) versus CONTROL after the same warm-start adaptation (5M decisions): Δ +0.000262, CI [+0.000040, +0.000475], Δ micro R@20 +0.000329 → gate 2(b): **FAIL**.
- **Probes (reliance of the incumbent, not retraining):** no removal candidate passes gate 2(a) (Δ ≥ +0.001, CI > 0, both slices positive). Every side-feature group is used: blanking it hurts.
- **Hygiene (lean view):** max|Δlogit| 2.10e-05, rank identity 0.998683 (148 changes), metric Δ -7.5e-08 → status **PARITY_FAILED** (failed: rank_identical_ge_0.999); no equivalence claim, and the selected schema is the current F4.
- **(MEASURED)** The warm-start adaptation itself lowered the metric: CONTROL 0.185454 versus the pinned incumbent 0.187117 (-0.001663); C1_D3 0.185716 (-0.001401). Both arms are below the incumbent, so nothing is adopted.
- **Protection T0→T2:** dataset tree equal = True; dataset metadata changed = []; changed existing files = ['.local_ai/sasrec_v1/tools/event_probe.py'] (the lead's read-only monitor); additions = the campaign's lock files and report tools (expected).
- **No causal and no SOTA claim.** Probes measure one checkpoint's reliance; the screen is exploratory warm-start adaptation with one seed.

## 2. Measured audit (full population)
- Exact duplicates and dead flags on the full population (TRAIN 52,760,314 rows, VS 10,930,090): non-zero violations = {'TRAIN': {}, 'VALIDATION_SELECT': {}}; max local_hour rotation residual = 6.5e-08.
- **M1 defect (confirmed on all TRAIN):** two time-gap normalizers were fitted on all rows including the masked zeros:
| Feature | mask | fitted mean / std (all TRAIN rows) | valid-only mean / std (TRAIN) | VS valid mean |
|---|---|---|---|---|
| log1p_time_since_prev_session_last_event_s | prev_session_available | 1.8615 / 4.3549 | 11.8169 / 1.6580 | 11.4827 |
| log1p_time_since_prev_user_event_s | prev_user_event_available | 4.9503 / 3.3790 | 5.3789 / 3.1782 | 5.3559 |

## 3. Probes
| Probe | type | Δ macro MRR@20 | 95% CI | Δ micro R@20 | weekday Δ | weekend Δ |
|---|---|---|---|---|---|---|
| P1_ALL_SIDE_OFF | OOD anchor | -0.046669 | [-0.049070, -0.044077] | -0.164348 | -0.041016 | -0.049463 |
| P2_MAIN_CATEGORY_UNAVAILABLE | coherent absence | -0.006262 | [-0.007527, -0.004957] | -0.010822 | -0.005850 | -0.006314 |
| P3_BRAND_UNAVAILABLE | coherent absence | -0.005687 | [-0.006903, -0.004337] | -0.013848 | -0.006789 | -0.005328 |
| P4_CURRENT_PRICE_UNAVAILABLE | coherent absence | -0.009761 | [-0.011340, -0.008255] | -0.012789 | -0.007057 | -0.011124 |
| P5_PREV_GAP_UNAVAILABLE | coherent absence | -0.003619 | [-0.004456, -0.002867] | -0.011249 | -0.005136 | -0.003085 |
| P6_ITEM_AGE_UNAVAILABLE | coherent absence | -0.024654 | [-0.026842, -0.022451] | -0.052677 | -0.017636 | -0.028089 |
| P7_COLD_USER | coherent absence | -0.004068 | [-0.005173, -0.002887] | -0.013287 | -0.002999 | -0.004702 |
| P8_CALENDAR_FROZEN | single-point counterfactual | +0.000660 | [-0.000192, +0.001502] | +0.002608 | -0.000122 | +0.000979 |

## 4. Screen (exploratory warm-start adaptation)
| Arm | cohort macro MRR@20 | cohort micro R@20 | Δ vs CONTROL | 95% CI | gate 2(b) |
|---|---|---|---|---|---|
| CONTROL | 0.185454 | 0.464273 | — | — | control |
| C1_D3 | 0.185716 | 0.464602 | +0.000262 | [+0.000040, +0.000475] | FAIL {"delta_ge_threshold": false, "ci_lower_gt_0": true, "recall_guard": true} |

## 5. Deliverables
- `SELECTED_FEATURES.json`: decision `KEEP_CURRENT`, model in use "pinned incumbent (unchanged)", selected schema "current full F4 schema (lean parity failed; no equivalence claim)".
- `TRANSFORM_STATE.json`: status `PARITY_FAILED_NOT_ADOPTED`, fit split TRAIN.

## 6. Stages
| Stage | state | attempt | end (UTC) | disk free GiB |
|---|---|---|---|---|
| T1_protect | DONE | 1 | 2026-09-26T07:24:28Z | 13.44 |
| T1_diff | DONE | 1 | 2026-09-26T07:24:28Z | 13.44 |
| stream_fit | DONE | 1 | 2026-09-26T07:24:58Z | 13.44 |
| full_profile | DONE | 1 | 2026-09-26T07:34:29Z | 13.44 |
| pin_baseline | DONE | 1 | 2026-09-26T07:34:30Z | 13.44 |
| probes | DONE | 1 | 2026-09-26T07:58:43Z | 13.43 |
| screen_control_c1 | DONE | 1 | 2026-09-26T08:27:57Z | 12.81 |
| finalize | DONE | 1 | 2026-09-26T08:27:59Z | 12.81 |
| T2_protect | DONE | 1 | 2026-09-26T08:28:00Z | 12.81 |
| T2_diff | DONE | 1 | 2026-09-26T08:28:00Z | 12.81 |

## 7. Incidents
- FO-01 (counts/names), Sonnet B defects, Review C findings, the P4 dtype bug, the memory-guard kill: all fixed before running.
- FO-02: the reused GRU `train.py` puts `gru_v1/code` first on sys.path; probes failed twice on the isolation guard → fix in `fo_env.py` (+3 tests) → Fable APPROVE_RESUME; DONE stages valid (14/14 byte-identical).

## 8. Limits
- VS reused; no fresh holdout; no VC/TEST. Probes are reliance, not removal-with-retraining. The screen is a one-seed warm start on 5M decisions.

# POPULATION_5_100 — scientific synthesis (Operations Director) — BINDING = Fable R2 + R4

## Inputs and record
**Opus:**
- `POPULATION_5_100_OPUS_POSITION.md`, written first and withheld.

**Fable produced four outputs** because messages queued to it were processed in order. All are preserved:

| output | file | what it did | status |
|---|---|---|---|
| R1 | `POPULATION_5_100_FABLE_POSITION.md` | APPROVE_WITH_LOCKS | superseded |
| R2 | `POPULATION_5_100_FABLE_POSITION_FINAL.md` | APPROVE_WITH_AMENDMENTS; incorporates the owner's speed directive (DL39) | **binding** |
| R3 | `POPULATION_5_100_FABLE_POSITION_R3_STEPB.md` | a re-ruling plus the STEP B answer | not adopted |
| R4 | `POPULATION_5_100_FABLE_R4_NO_CHANGE.md` | after re-reading, confirms R2 with a budget delta | **binding delta** |

**Other inputs:**
- DECISION_LOG 39–43.
- Owner directives: DL31, 34–37, 39.
- Evidence: CENSUS-02, FEATSTAT-01, the cost model, and the INT-01 and SYS-04 rates.
- No V4 model outcome exists.

## Locked values
| topic | locked value (R2 + R4) | Opus position |
|---|---|---|
| main pool | 5 ≤ n ≤ 100 under catalogue A, TRAIN-only: 1,446,986 users, 28,623,386 examples, `e2a082cd…fbc70` | agree |
| key | `selection_key_v2`, salt `PPSI_V4_POP_5_100_SEL_20260926`, over the C1.2 digest; ascending rank r | agree |
| MAIN / CALIB | r < 131,072 (`563e7663…28b3`) / r < 32,768 (`9be38de1…1f02`), nested | agree |
| U_NEW_VAL / U_NEW_TEST | r ∈ [1,348,682, 1,381,450) / [1,381,450, 1,414,218), 32,768 each | differs; Fable adopted |
| FS | r ≥ 1,414,218 (the top 32,768), per the owner's DL39 | differs; Fable adopted |
| tie-break universe | r < 1,348,682 | differs; Fable adopted |
| panels | TRUE_COLD = the 381,295 n = 0 VAL users (A0 / A1 reported separately); COLD_H 16,384; SPARSE_PANEL 16,384 (B with ≥ 1 E2E VAL row); HEAVY_PANEL 16,384 (D, the same rule); WARM_TRUNCATION 16,384 (MAIN); M04S retired | differs; Fable adopted |
| catalogue | A: K = 158,486, support ≥ 5, order unchanged | agree |
| shared statistics | **full TRAIN as fitted (Clean V2) for all six arms** [PROXY]. Classified as pooled from excluded clients or deployment-available bootstrap. Strict-unseen is BOUNDED (≈ 0.01 z) and disclosed. A cohort-refit sensitivity is OPTIONAL after ARCHITECTURE_LOCK, and its infrastructure (TRANSFORM_STATE_FS, the provider override) is being built for that purpose only | **DISAGREEMENT:** Opus proposed a cohort refit (option C). Recorded for the owner's ratification |
| history | USER_HISTORY_L50 (owner); SESSION_L50 as an ablation only | agree |
| FL work | B 16, 2 passes, 3 visits, n_consumed weighting, 64 per round, full sweeps, 1,024 rounds per EFE, 6,144 rounds; μ / λ / local-LR grids recalibrated on CALIB at 2 EFE; no client is ever dropped | agree |
| unseen protocol | C / FA / FP: shared θ, no adaptation. PF / CPF: p_u = 0 (Study A) and reconstructed from support (Study B). LO standard. The FEW_SHOT recipe (passes {1, 2, 4}) is chosen once on U_NEW_VAL FIXED_20. U_NEW_TEST is evaluated once after ALL_METHODS_LOCK. h ∈ {0, 1, 5, 10, 20, 50} × (h_ctx, h_sup). Strata 1+ / 5+ / 10+ are descriptive. Backtests are out of scope | narrower than Opus; Fable adopted |
| unchanged | T1 / T2 / T3; STRICT training and EVAL_ROWS_ONLY (confirmed); C3; bootstrap 1,000 at seed 20260926 | agree |
| EFE at MAIN | 2,587,776 examples; 10,108.5 updates per EFE; 60,651 updates to 6.0 | agree |

## D3 — the feature search (R2 + R4) [PROXY]
1. **L3 (53 → 30) is adopted statistically:** exact dependencies, dead or near-constant channels, and near-duplicates.
2. **Prerequisite:** DATA-05's causality and identical-builder proofs. A SESSION_L50 screen is refused.
3. **Training:** FS = the top 32,768. Train L0 and L3 per family at **seeds 2026 and 2027** (R4 delta): 6 EFE,
   USER_HISTORY_L50, catalogue A, full-TRAIN statistics, the executed v1 peak LR (1e-3), no calibration. The statistic is
   the CONTROLLED_BUDGET 6.0 endpoint client-macro MRR@20.
4. **Grouped permutation importance** over the 14 groups G1…G12c on both L3 seeds:
   - 3 permutation seeds each, 168 evaluations in total;
   - item_tokens is never permuted; a value and its flag or mask move jointly; the whole [50] column is swapped.
5. **Noise rule:** a group is within noise iff its paired-user-bootstrap 95% CI (W 1,000, seed 20260926) includes 0. A
   group is dropped only if it is within noise for BOTH families.
6. **S\*** = L3 minus the dropped groups, at both seeds.
   - **ACCEPT** iff, for the WORSE family, the seed-mean relative drop versus L0 has point ≤ 1.0% AND 95% UB ≤ 2.0%.
   - Fallback: restore the most important dropped group and retrain once, at both seeds. If that fails, lock L3 if L3
     passes, else L0.
   - At most 8 runs per family. About 5 GPU-h expected, or 6.4 GPU-h if the fallback fires, plus about 1.1 GPU-h of
     permutation evaluations.
7. **Discipline:**
   - only relative drops, CIs and importances are published before ARCHITECTURE_LOCK;
   - FS users never enter any other set;
   - the search stops at the first accepted schema;
   - screen rows are labelled FEATURE_SEARCH_SCREEN and are refused by selection.

## Expected compute and limitations
Per Fable R2 D10:
- **Compute:** about 100 GPU-h if GRU wins, about 250 GPU-h if SASRec wins. The tie-break, if triggered, takes about
  35 GPU-h for stage 1 and about 70 GPU-h in total.
- **Limitations:**
  - the scope is 5–100 users;
  - the panels are secondary;
  - FL is simulated, so no privacy claim is made;
  - the participation gap is measured within the same period;
  - VALIDATION and TEST were inspected before;
  - S\* is not claimed optimal.

## Open for the owner's ratification ([PROXY] items)
1. D3, the fast search rule.
2. D5: full-TRAIN statistics (Fable) against a cohort refit (Opus).
3. D7, the unseen-user protocol and sizes.
4. The earlier question: EQUIVALENT → "GRU on cost".

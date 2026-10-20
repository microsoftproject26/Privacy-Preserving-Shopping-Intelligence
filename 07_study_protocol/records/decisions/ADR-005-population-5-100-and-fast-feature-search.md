# ADR-005 — Population 5–100, USER_HISTORY_L50, the partition, shared statistics and the fast feature search

- **Status:** ACCEPTED 2026-09-26 (UTC). The owner directives (DECISION_LOG 31, 34–37, 39) plus Fable's FINAL ruling
  as owner-proxy. The [PROXY] items await owner ratification.

## Decision
The locked values are in `reviews/POPULATION_5_100_SCIENTIFIC_SYNTHESIS.md`, which follows
`reviews/POPULATION_5_100_FABLE_POSITION_FINAL.md`:
- **Main pool:** 5 ≤ n ≤ 100 (TRAIN-only, catalogue A): 1,446,986 users and 28,623,386 examples.
- **Key:** selection_key_v2 (salt `PPSI_V4_POP_5_100_SEL_20260926`).
- **Partition:**
  - MAIN r < 131,072 and CALIB r < 32,768 (nested);
  - U_NEW_VAL r ∈ [1,348,682, 1,381,450) and U_NEW_TEST r ∈ [1,381,450, 1,414,218);
  - FS r ≥ 1,414,218;
  - the tie-break universe r < 1,348,682.
- **Panels:** TRUE_COLD (381,295; A0 and A1 separately), plus COLD_H, SPARSE_PANEL, HEAVY_PANEL and WARM_TRUNCATION of
  16,384 users each.
- **Catalogue:** A, with K = 158,486.
- **Shared statistics:** full TRAIN as fitted, for all six arms [PROXY].
- **History:** USER_HISTORY_L50 (owner), with SESSION_L50 as an ablation.
- **FL local work:** B 16, 2 passes, 3 visits, 64 clients per round, n_consumed weighting.
- **Unseen-user protocol** [PROXY]: Study A (no adaptation) and Study B (a reconstructed personal state), with the
  FEW_SHOT recipe chosen once on U_NEW_VAL.
- **Feature search** [PROXY]: the fast, importance-guided D3 procedure, with acceptance for the worse family at point ≤ 1%
  and UB95 ≤ 2%.

## Options considered
- N_MAIN ∈ {32,768; 65,536; 131,072; 262,144; ALL}.
- Catalogue A or B.
- Shared statistics fitted on full TRAIN, on the pool, or on the training cohort.
- The feature search as a rung ladder or importance-guided.
- The sizes and adaptation arms for unseen users.

## Evidence
- CENSUS-02, FEATSTAT-01, `plans/POP5100_COST_MODEL.json`, and the INT-01 and SYS-04 rates.
- Literature: Yuan et al. 2022 (participation gap); Liang et al. 2018 (strong generalization); Motley 2022; FedRecon
  2021; Ji et al. 2023 (global timeline); Gusak et al. 2025 (Time to Split).

## Disagreement and resolution
- **Shared statistics (D5):** Opus proposed a cohort refit (option C). Fable ruled full TRAIN (option A), for
  single-factor attribution, comparability, no simulated federated-statistics round, and speed. The ruling follows
  Fable, and the disagreement is recorded for the owner.
- **Band layout, unseen handling and panels:** Fable's versions are adopted.

## Consequences
- A new data version is built from the locked partition: MAIN, CALIB, U_NEW, TB and panel manifests, with
  USER_HISTORY_L50 windows and the Clean V2 transform state.
- M01–M06 are HISTORICAL.
- The feature-search runs start after DATA-05's causality proofs.
- S\* is locked by an addendum to this ADR.
- The contest runs under S\*, with the LR re-calibrated on CALIB.

## Thesis relevance
- Methodology: the population definition, the evaluation design with the participation gap, and feature minimization.
- Limitations: the 5–100 scope, and full-TRAIN shared statistics.

## Addendum 1: consolidation (DECISION_LOG 42)
Fable issued three outputs (R1, R2 and R3) because queued messages were processed in order. They conflicted on the bands,
the FS size and D5.

**Consolidated:**
- **D1 per R2:** MAIN and CALIB at low ranks; U_NEW_VAL and U_NEW_TEST 32,768 each at [1,348,682, 1,414,218); FS the
  top 32,768; TB r < 1,348,682.
- **D5 per R3:** every z-normalizer is cohort-fitted (MAIN / FS / TB); the item-side bootstrap is full-TRAIN. This
  supersedes the 'full TRAIN as fitted' line in the Decision section above.
- **D3:** R2's procedure, with R3's compatible amendments (evaluations at 2/4/6 EFE, the LR sanity fallback, the
  ≤ 0.1% magnitude in the drop rule, digest-only retention, 2 jobs per GPU after a RAM check).
- **D8:** full strata CONTEXT_ONLY plus 16,384-user adaptive subsets.

**Rule applied:** the owner's speed directive governs sizes and procedure; the owner's §9 ('prefer the scientifically
cleaner option') governs D5.

## Addendum 2 (DECISION_LOG 43): supersedes Addendum 1 where they differ
Fable R4 confirmed the FINAL ruling R2 with no change, except a budget delta: L0, L3, S\* and the single fallback each
run at seeds 2026 and 2027, and importance is computed on both L3 seeds.

**BINDING = R2 + R4:**
- **D5:** full-TRAIN statistics as fitted, for all arms. The 'Shared statistics' line in the Decision section stands;
  Addendum 1's D5 is withdrawn.
- **D8:** 16,384-user panels. Addendum 1's D8 is withdrawn.
- **D3:** R2's procedure exactly, with R4's seeds.

Cohort-fitted normalizers exist only as the optional post-lock sensitivity analysis. The Opus disagreement on D5 is
listed for the owner's ratification.

## Addendum 3 (DECISION_LOG 53): feature search LOCKED — S\*
Written 2026-09-27T11:08Z by the Operations Director. Mechanical application of the D3 rule (R2 + R4); no judgement call.

**Locked schema S\* = `L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7`** (PROJECT_MEASURED):
- 25 information channels (widths: tokens 5, numeric 7, flags 2, user_context 6, user_mask 4); schema self_sha256
  `c53d47fe2555dd537e27e11d45a4e8b23739566e6b98179dcf521852aa918f87`; file sha256 `626ca93e…8b759`; derived from
  L3S (self `57cb1cd8…`); registered in `contracts/derived_schemas/REGISTRY.json` (self `06d1f136…`, file `c23454cb…`)
  with server θ0 for GRU and SASRec at seeds 2026 and 2027 (`THETA0_REGISTRY.json` file `e4d8eb86…`).
- Dropped groups (within noise for BOTH families, round A): G3 weekday_tokens; G4 utc_hour sin/cos; G7 item age + flag;
  G11 previous-session gap + mask; G12a previous view/session counts and recent distinct main categories.
- REQUIRED channels kept: item_tokens, category_tokens, session_offset_tokens. G13 SESSION_INTENT survived (not dropped).
- Acceptance (seed-mean relative drop versus L0S, FS cohort, CONTROLLED_BUDGET 6 EFE client-macro MRR@20; negative =
  S\* better): GRU −0.90% (95% UB −0.15%); SASRec −0.13% (UB +0.86%). Worse family SASRec passes (≤ 1.0%, UB ≤ 2.0%).
  No fallback; `restored_group = null`.
- Evidence: `runs_fs/FS_FINAL.json` (sha256 `799590a4…bf4c`, written 04:04:19Z), `runs_fs/stageB/DECISION_A.json`
  (`142a9915…`), `DECISION_B.json` (`d8257a4f…`), `runs_fs/autopilot.log`; local copies hash-identical to the server.
  Two failed attempts preserved under `runs_fs/_failed_attempt1/2` (DECISION_LOG 49, 52).

**Consequences:** every production arm (LR calibration, contest, downstream methods) uses S\*. Only relative drops are
published before ARCHITECTURE_LOCK; the FS rows remain FEATURE_SEARCH_SCREEN and are refused by architecture selection.
Session-scoped windows, L0S and L3S stay as named diagnostics. Fable confirms at the combined G1 gate.

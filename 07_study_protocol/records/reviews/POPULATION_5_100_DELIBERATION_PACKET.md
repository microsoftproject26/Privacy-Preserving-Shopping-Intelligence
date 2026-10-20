# POPULATION_5_100 deliberation packet for Fable (facts and questions only)

You are Fable 5.1, the Research Director. You are read-only, and your ruling is binding.
- **Owner-proxy:** the owner is offline and has named you owner-proxy for any blocking decision (DECISION_LOG 37).
  Owner-proxy rulings are provisional and will be listed for ratification.
- **Independence:**
  - Do NOT open `company/reviews/POPULATION_5_100_OPUS_POSITION.md` or `company/reviews/POPULATION_5_100_IMPACT_MATRIX_DRAFT.md`.
  - Do NOT open `census_v5_100/_superseded_run1/`. It is quarantined because it contains v1-model outcome levels on
    the new population (DECISION_LOG 33).
  - A synthesis follows your answer.
- **Root:** `F:\Queens University\Microsoft Project\eda\.local_ai\ppsi_v4\`.

## Binding owner inputs (do not reopen)
- **Owner amendment POP 5–100:** `company/owner/OWNER_AMENDMENT_POP_5_100_2026-09-26.md`, verbatim. Read all 15
  sections.
- **DECISION_LOG 34–37:**
  - 5 ≤ n ≤ 100 is reaffirmed;
  - **history policy USER_HISTORY_L50:** the input window is the last 50 representative events of the user's whole
    prior stream, across sessions and partition fences, strictly causal, identical for TRAIN and eval. It replaces the
    v1 session-scoped window. Ops verified from `code/snap_v1/reuse/vendor/rees46_eda/loaders.py` L263–L270 and
    `rees46_nextitem_v1/protocol.resolved.yaml` that v1 windows never leave the current session and reset at partition
    fences;
  - an evaluation on **unseen users** (never trained on) is adopted;
  - **feature choice is delegated to Ops + Fable.** The owner wants the fewest columns without much accuracy loss. Ops
    set a provisional rule in DECISION_LOG 36: the smallest schema whose MRR@20 drop is ≤ 1% relative on the WORSE of
    GRU and SASRec, CI-consistent, with identical columns for both. You may confirm or amend it.
- **Context the owner supplied:** an external analysis (ChatGPT). Its main proposals, summarized by Ops:
  - held-out user groups U_fit / U_new_val / U_new_test, with support/query per unseen user (citing Motley 2022 and
    FedRecon 2021);
  - strict unseen, meaning held-out users are excluded from fitting shared statistics;
  - query-count strata (1+, 5+, 10+) as descriptive only, never as a filter;
  - optional rolling-origin backtests on October origins, flagged as expensive;
  - separating h = 0 "no adaptation" from h = 0 "no history".

## Facts (PROJECT_MEASURED unless marked; files are under the root)
- **Strata census** (CENSUS-02; `census_v5_100/STRATA_CENSUS.json`, `CANDIDATE_N_PROJECTIONS.json`,
  `FL_LOCAL_WORK.json`, `CATALOGUE_OPTION_B.json`, `PRECISION_PRIOR.json`; handoff `company/handoffs/CENSUS-02.md`):
  - A0 (no TRAIN event): 600,554 users. A1 (TRAIN events but n = 0): 1,509,381.
  - **B (1–4):** 1,197,003 users / 2,503,629 examples.
  - **C (5–100):** 1,446,986 users / 28,623,386 examples (71.24% of supervision); n p50 13, p90 45. VALIDATION: 450,796
    users with ≥ 1 eval row and 5,823,491 eval rows (mean 12.9 per active user).
  - **D (> 100):** 51,009 users / 9,053,593 examples (22.53%).
  - **Candidate cohorts:** nested hash prefixes of C with the proposed selection_key_v2 salt
    `PPSI_V4_POP_5_100_SEL_20260926` over the pseudonym digest.

    | N | examples | VAL users | eval rows | rankable targets with an in-cohort positive |
    |---|---|---|---|---|
    | 32,768 | 646,067 | 10,070 | 127,530 | 90.8% |
    | 65,536 | 1,293,364 | 20,205 | 255,012 | 94.9% |
    | 131,072 | 2,587,776 | 40,764 | 519,714 | 97.6% |
    | 262,144 | 5,179,643 | 81,614 | 1,042,084 | 99.05% |
    | ALL C | 28,623,386 | 450,796 | 5,823,491 | 99.98% |

  - **Precision prior** (PRIOR_EVIDENCE_FROM_V1_MODELS, VALIDATION_CONFIRM, levels withheld): the paired GRU−SASRec SE
    of client-macro MRR@20 is about 0.00078, 0.00055 and 0.00039 at 65,536, 131,072 and 262,144. The per-user
    difference SD is 0.111 and the correlation 0.858.
  - **FL local work** (B = 16, 2 passes): a visit costs median 2 steps and at most 14 (mean 3.43). 60.6% of clients fit
    in one batch. Gini(n) is 0.44; the top 10% of clients hold 32.6% of examples. At 64 clients per round, one EFE takes
    N/128 rounds.
  - **Catalogue A vs B:**
    - B (support ≥ 5 counted within C) has K_B = 142,950 against 158,486, dropping 15,536 classes.
    - The rankable share of C's VALIDATION targets falls from 99.21% to 98.75%.
    - Circularity: 759 C users leave C and 386 D users enter.
    - The head is 8 MB (GRU) or 16 MB (SASRec) smaller.
- **Feature statistics** (FEATSTAT-01; `featstat_v5_100/FEATSTAT_SUMMARY.md` first, then the JSONs; handoff
  `company/handoffs/FEATSTAT-01.md`):
  - exact dependencies and dead channels on the pool, with every status transferring from n ≥ 1;
  - near-constant flags; clusters;
  - **normalizer provenance:** the 7 price/time/age normalizers barely move on a pool refit (mean |Δz| ≤ 0.021). The 10
    count-type ones move materially (clip bounds shrink, for example 140 → 61; mean |Δz| up to 0.28) because heavy
    users are excluded;
  - **item age** uses pooled first-seen times; pool-only first-seen changes 34% of values;
  - **a proposed nested ladder** L0 53 → L1 37 → L2 35 → L3 30 → L4 28 → L5 18 (no user-context block) → L6 16 → L7 13
    → L8 10, plus L9 ID-only as an anchor. It has a per-channel column: "carried by a cross-session 50-event window?".
    Parameter counts barely change, since the item tables are about 97% of each model.
- **Measured costs:**
  - central full-K training with the real provider in STRICT mode: GRU 20.9 and SASRec 10.0 updates/s, not data-bound
    (INT-01);
  - evaluation: 131 s (GRU) and 180 s (SASRec) per 808,610 rows (T3 receipt).
  - **INFERENCE** (`company/plans/POP5100_COST_MODEL.json`, the G0 visit model applied to the pool distribution): the
    downstream core over 2 seeds (C / FA / FP / PF / CPF / LO) costs about 36 / 101 GPU-h at 65,536, 72 / 202 at
    131,072 and 144 / 404 at 262,144 (GRU / SASRec). The contest is about 3.2 / 6.4 / 12.8 GPU-h. Two A16s.
- **PENDING:** DATA-05 (the USER_HISTORY_L50 builder, leakage proofs, the window-length census, and the GRU step cost
  with longer packed windows).
- **Existing locks that stay** (`contracts/PREG1_LOCK_VALUES.json`):
  - T1: selection trigger and staged tie-break;
  - T2: exposure point "end";
  - T3: frozen eval numerics;
  - STRICT training determinism (proposed; SYS-04 evidence);
  - eval_stream EVAL_ROWS_ONLY (proposed).

## Decisions requested (owner §15 synthesis list, plus the owner's new requirements)
- **D1.** The exact main-pool definition, and the deterministic partition of the pool into MAIN, CALIB, any
  feature-search cohort, the unseen-user groups and the tie-break universe. Include the selection salt.
- **D2.** Exact N_MAIN and N_CALIB, justified by the owner's §4 criteria (cost, coverage, precision,
  representativeness).
- **D3.** The minimal-sufficient feature-search procedure:
  - the candidate set or ladder;
  - which cohort and VALIDATION it uses (adaptive reuse against the architecture contest);
  - the margin and decision rule for the WORSE family;
  - seeds and budget;
  - the LR used during the search;
  - the stop rule.
- **D4.** Catalogue policy: A versus B, including the strict-unseen implication.
- **D5.** The normalization and shared-statistics population (full TRAIN / pool / training cohort), classified per
  owner §9, including item age and vocabularies.
- **D6.** The FL client-work policy: local batch, epochs versus steps, weighting, clients per round, participation,
  rounds, and retry accounting.
- **D7.** The unseen-user protocol:
  - group sizes;
  - support and query definitions under the global timeline;
  - how C / FA / FP / PF / CPF / LO handle a user who never participated;
  - what is chosen on U_new_val;
  - strict-unseen treatment of shared statistics;
  - the support-size (h) analysis;
  - whether rolling-origin backtests are in scope.
- **D8.** The non-main strata panels (TRUE COLD A0/A1, SPARSE, HEAVY) and the reconciled cold-start h design.
- **D9.** The affected protocol decisions, as an impact matrix per owner §14 (UNAFFECTED / RECOMPUTE_ONLY /
  RESEARCH_REVIEW_REQUIRED / SUPERSEDED) for at least every item in owner §14, plus the history window.
- **D10.** Expected compute, and the limitations or generalization scope for the thesis.

## Return format (target ≤ 1,500 words)
ROUTING, VERDICT, then a DECISION with LOCK_VALUES for D1–D10, then OPEN_DISAGREEMENT_RISKS. Read-only.

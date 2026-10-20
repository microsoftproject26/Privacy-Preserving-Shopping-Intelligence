# POPULATION_5_100 — Opus position (Operations Director). Independent memo, withheld from Fable until Fable answers.

## Status and authority
- **Written:** 2026-09-26.
- **Binding inputs:**
  - owner amendment POP 5–100 (`company/owner/OWNER_AMENDMENT_POP_5_100_2026-09-26.md`; DECISION_LOG 31);
  - owner decisions DECISION_LOG 34–37:
    - 5–100 reaffirmed;
    - USER_HISTORY_L50 approved;
    - unseen-user evaluation adopted;
    - feature choice delegated to Ops + Fable;
    - Fable is owner-proxy while the owner is away.
- **Evidence:**
  - CENSUS-02 (`census_v5_100/`, run of record);
  - FEATSTAT-01 (`featstat_v5_100/`);
  - INT-01 and SYS-04b/c measured step rates;
  - the cost model `company/plans/POP5100_COST_MODEL.json` (INFERENCE);
  - DATA-05 (USER_HISTORY_L50 windows) is still running.
- **Not read:** the quarantined `census_v5_100/_superseded_run1/`. TEST was not used.

## 1. Exact main-pool definition
- **Rule:** MAIN_POOL = users with 5 ≤ n ≤ 100, inclusive. n is the number of supervised TRAIN examples: split TRAIN,
  loss_mask true, an observed target with target_class ≥ 0 under catalogue A, and a target time before TRAIN_END.
- **Measured:** 1,446,986 users and 28,623,386 examples (71.24% of supervision); n has p50 13 and p90 45. The
  population sha256 is `e2a082cd…fbc70` (CENSUS-02).
- n is fixed under catalogue A, so the circularity of catalogue B never arises (§5).
- USER_HISTORY_L50 changes inputs only. It does not change n, the targets or the masks.

## 2. Deterministic partition of the pool
All groups are disjoint, with nested prefixes where marked.
- **Key:** selection_key_v2 = big-endian uint64 of `sha256(b"PPSI_V4_POP_5_100_SEL_20260926|" + digest32)[:8]`, over
  the CLIENT_CONTRACT C1.2 pseudonym digest. The keys are unique (CENSUS-02). Rank the pool by the key.

| group | key-rank range | N | role |
|---|---|---|---|
| **MAIN** (U_fit) | [0, 131,072) | **131,072** | C / FA / FP / PF / CPF / LO training and the family contest |
| **CALIB** | [0, 32,768), nested in MAIN | **32,768** | LR calibration; FL μ / λ calibration at G3 |
| **FEATSEARCH** | [131,072, 196,608) | 65,536 | the minimal-sufficient feature search (§4) |
| **U_NEW_VAL** | [196,608, 229,376) | 32,768 | unseen users: choosing adaptation settings (§8) |
| **U_NEW_TEST** | [229,376, 294,912) | 65,536 | unseen users: final report (§8) |
| **TIEBREAK universe** (only if T1 fires) | all ranks except U_NEW_VAL ∪ U_NEW_TEST | 1,348,682 | the full-universe tie-break, inside 5–100 (owner §7) |

The remaining ranks are reserve and are never used silently.

## 3. N_MAIN = 131,072 and N_CALIB = 32,768: why
From CENSUS-02 (PROJECT_MEASURED) and the cost model (INFERENCE):
- **The candidate MAIN sizes:**

| N | examples | VAL users | eval rows | rankable targets with an in-cohort positive | paired GRU−SASRec SE (v1 prior) | SASRec-winner downstream core (2 seeds) | GRU-winner |
|---|---|---|---|---|---|---|---|
| 65,536 | 1,293,364 | 20,205 | 255,012 | 94.9% | 0.00078 | 101 GPU-h | 36 |
| **131,072** | 2,587,776 | 40,764 | 519,714 | 97.6% | 0.00055 | **202 GPU-h** | 72 |
| 262,144 | 5,179,643 | 81,614 | 1,042,084 | 99.05% | 0.00039 | 404 GPU-h | 144 |

- **Why 131,072:** it is the largest size whose worst-case downstream (a SASRec winner) completes in about 4–5 days on
  the 2 A16s with FL calibration.
  - At 131,072, T1a's 0.0020 is about 3.6 SE, so the family decision is well powered. At 65,536 it is 2.6 SE.
  - 262,144 doubles the wall clock (about 8–9 days) for a 1.4× SE gain.
  - N must be frozen before the contest, so it cannot depend on the winner. It is sized for the worse case.
- **N_CALIB = N/4 = 32,768:** 10,070 VALIDATION users. It only has to separate the two registered LR peaks within a
  family (highly correlated arms), and it costs about 0.3 h (GRU) or 0.5 h (SASRec) per 6-EFE run.

## 4. Minimal-sufficient feature search (owner §6–§7; delegated)
- **Candidates:** the FEATSTAT-01 nested ladder under USER_HISTORY_L50, identical for both families:
  - L0 53 (Clean V2), L1 37 (exact redundancies and dead channels), L2 35, L3 30, L4 28, L5 18 (no user-context
    block), L6 16 (no pooled item age), L7 13, L8 10;
  - L9, ID only, is a reference anchor, never selectable.
- **Inputs to every run:**
  - the FEATSEARCH cohort;
  - catalogue A;
  - the normalization rule of §6, fitted on FEATSEARCH TRAIN;
  - the executed recipe with the default peak 1e-3 for every rung (LR calibration follows the schema lock, on CALIB);
  - 6 EFE, PRACTICAL_BEST client-macro MRR@20 on the FEATSEARCH VALIDATION users (about 20,200).
- **The rule (predeclared):**
  - rung S *passes* iff, for the WORSE family, the relative drop (MRR(S) − MRR(L0)) / MRR(L0) ≥ −1%, and the paired
    user-bootstrap 95% CI lower bound of MRR(S) − MRR(L0) ≥ −0.01 · MRR(L0);
  - the selected schema is the smallest S such that S and every tested rung above it pass (stop at the first failure
    going down);
  - untested rungs between the last pass and the first failure are then tested;
  - identical channels for both families (owner §7).
- **Stage A:** seed 2026 on L0, L1, L3, L5, L7 and L8, both families. That is 12 runs, about 9.6 GPU-h before any
  window-length effect on GRU.
- **Stage B:** seed 2027 on L0 and on the selected S, both families (4 runs). S is confirmed only if the rule holds on
  the 2-seed mean. Otherwise step up one rung and confirm again.
- **Secondary metrics:** NDCG@10 and Recall@20 are reported but have no role in the decision.
- **Code prerequisites** (needed before G1): the FEATSTAT notes list them.
  - derived schema files with self-hashes for each rung;
  - a provider channel projection;
  - a configurable SASRec side-token list;
  - a user-context branch that can be switched off (L5 and below);
  - θ0 and inventory re-registration per rung (server hashes).
- **Adaptive-reuse control:** the MAIN VALIDATION is untouched until the contest. The feature decision uses only
  FEATSEARCH.

## 5. Catalogue policy: keep A (global TRAIN support ≥ 5, K = 158,486)
- **Option B would:**
  - drop 15,536 classes;
  - affect only 0.12% of pool items and 0.14% of pool targets;
  - move 759 users out of C and 386 users into C (circularity);
  - save only 8 or 16 MB of head.
- **A keeps** comparability with v1 and does not confound the population change with a candidate-space change.
- **Strict-unseen caveat:** A's support counts include the unseen panels' TRAIN views.
  - The data build must measure how many classes would fall below support 5 without the U_NEW users, and disclose it.
  - A counts as interaction-derived catalogue information (a limitation), kept for comparability.
- No ADR is needed, because nothing changes.

## 6. Normalization population (owner §9)
- **Recommendation:** every fitted shared statistic is fitted on the TRAIN examples of the cohort that trains the
  model (MAIN for production runs, FEATSEARCH for feature-search runs). The same fitted block goes to all six methods.
- **Why:**
  - It is the cleanest provenance: no information comes from excluded clients (heavy, sparse, unseen, other cohorts).
  - In FL it is realizable as a one-shot federated-analytics round over the participants (sums, sums of squares,
    quantile sketches for the clip bounds). Centrally computed exact values are its stand-in, and this is disclosed.
- **Evidence (FEATSTAT-01):**
  - the 7 price/time/age normalizers barely move on a refit (mean |Δz| ≤ 0.021);
  - the 10 count-type ones move materially (for example the clip of prefix_representative_count goes 140 → 61; mean
    |Δz| up to 0.28), because heavy users are excluded. Full-TRAIN statistics are therefore NOT neutral for a 5–100
    benchmark.
- **The trade-off:** historical comparability with v1, which used full-TRAIN normalizers.
- **Other shared statistics:**
  - **Vocabularies:** unchanged. A pool-only vocabulary changes no pool token, and they are deployment-available
    attribute vocabularies.
  - **Metadata recovery:** the flags are near-zero on the pool. Kept as catalogue metadata.
  - **Item age** (pooled first-seen) is the one pooled-interaction channel. It is tested at L6. If it survives, its
    first-seen times come from the training cohort's TRAIN only, which changes 34% of values (mean |Δz| 0.018), or the
    item age is disclosed as catalogue listing-time metadata. **Fable to rule.**

## 7. FL client-work policy (owner §10)
Evidence: CENSUS-02 `FL_LOCAL_WORK`.
- **Local work:** keep epoch-based local work, 2 passes over the client's own examples per visit, with batch 16.
  - The step cost is nearly flat in b ≤ 64 (G0 bench), so B decides the number of steps.
  - At B = 16 the median visit is 2 steps and the maximum 14 (60.6% of clients fit one batch).
  - B = 8 would roughly double the FL cost. B = 32 would make most visits a single step, which is FedSGD-like.
  - Fixed local steps would over-sample 5-example clients (for example 4 × 16 = 64 exposures, about 13 passes) and
    break exact exposure accounting.
- **Weighting:** by consumed examples (2n per visit, proportional to n), for FedAvg and FedProx.
  - The 5–100 cap bounds the weight ratio at 20×: Gini 0.44, and the top 10% of clients hold 32.6% of examples.
- **Rounds and participation:** 64 clients per round, uniform participation without replacement in each cycle, and
  3 visits per client over 6 EFE.
  - That gives 1,024 rounds per EFE and 6,144 rounds per run at 131,072.
- **Unchanged (FL-02 pins):** max_attempts 2 with identical RNG, exposure point "end", and EFE accounting.

## 8. Unseen-user evaluation (owner-adopted; DECISION_LOG 34–36)
- **U_NEW_VAL and U_NEW_TEST are never trained on** by any run: central, FL, calibration, feature search or
  tie-break. They are also excluded from every normalizer fit (§6).
- **Support** = the user's own pre-TRAIN_END history, which is 5–100 supervised examples because the users come from
  the pool.
- **Queries** = decisions in the VALIDATION window (U_NEW_VAL: the choice of adaptation settings) and the TEST
  window (U_NEW_TEST: the final report, after ALL_METHODS_LOCK). The global timeline holds throughout.
- **Handling by method (predeclared):**

| method | handling of unseen users |
|---|---|
| C / FA / FP | (a) no adaptation: the history enters through USER_HISTORY_L50 only; (b) a named secondary local fine-tune on the support, with settings chosen on U_NEW_VAL |
| PF | p_u initialized at 0 and fitted on the support with the PF personal optimizer. Steps and LR are chosen on U_NEW_VAL |
| CPF | the matching central-personal mechanism on the support |
| LO | trains on the support, by the LO definition |

- **The participation gap** (Yuan et al., ICLR 2022) is reported per method as MAIN-VALIDATION versus U_NEW
  performance, together with Motley/FedRecon-style support/query framing.
- **Support-size analysis (secondary):** on U_NEW_TEST users with n ≥ 20 (a TRAIN-only selection), use h = 5, 10 and
  20 most-recent support examples with the SAME queries. This replaces the old M04W warm-truncation design.
- **Estimated cost:** LO on U_NEW_TEST takes about 4.0 h (GRU) or 11.5 h (SASRec) per seed. Evaluation without
  adaptation takes minutes.

## 9. Evaluation rules restated (owner §12–§13)
- **Unchanged:**
  - the global temporal split;
  - all eval decisions as targets (Successive);
  - user-macro metrics with micro alongside;
  - the paired user-level bootstrap;
  - OOV zero credit, with known-item conditional metrics;
  - full-catalogue exact ranking;
  - frozen T3 numerics;
  - TEST sealed.
- **Descriptive strata only (never a filter):** users with 1, 2–4, 5–9 and 10+ eval decisions; sessions and days per
  user.
- **Non-main strata:** evaluation panels of up to 65,536 hash-sampled users each, evaluation only:
  - TRUE COLD A0 (no TRAIN event) and A1 (TRAIN events, n = 0), reported separately. The cold-h panel from A0 keeps
    h ∈ {0, 1, 5, 10, 20, 50} VALIDATION-window context events;
  - SPARSE 1–4;
  - HEAVY > 100 (diagnostic).

  None of them enters the main comparison.

## 10. Architecture selection and tie-break
- **The contest:** MAIN, the final compact schema, USER_HISTORY_L50, the CALIB-calibrated LR, and seeds 2026 and 2027.
- **T1 rules and thresholds:** kept, since their SEs were re-checked in §3.
- **The tie-break universe:** 1,348,682 users. The cost is about 11.5 h (GRU) and 21.7 h (SASRec) per seed, so
  stage 1 is about 33 GPU-h and stage 2 about 66, both within the 65 / 125 caps.
- **Tie-break rows:** stay HISTORICAL_REFERENCE only.

## 11. Expected compute (INFERENCE; excludes DATA-05's GRU window-length effect)
| step | GPU-h |
|---|---|
| feature search, stages A + B | ≈ 13 |
| LR calibration on CALIB | ≈ 2 |
| family contest, 2 × 2 | ≈ 6.4 |
| downstream core (C / FA / FP / PF / CPF / LO × 2 seeds) | ≈ 72 (GRU) / ≈ 202 (SASRec) |
| FL calibration (μ × 3, λ × 3) on CALIB | ≈ 12 (GRU) / ≈ 36 (SASRec) |
| unseen users (LO, PF adaptation) | ≈ 12 (GRU) / ≈ 35 (SASRec) |
| **total** | **≈ 118 (GRU) / ≈ 295 (SASRec)**, i.e. about 2.5 or 6 days on 2 GPUs |

## 12. Limitations and generalization
- The main causal claims cover 5–100 users only: 53.7% of warm users and 71% of supervision.
- The heavy users (22.5% of supervision) and the sparse and cold users are covered only by secondary panels.
- VALIDATION and TEST were inspected before (retrospective).
- Catalogue A and the vocabularies are interaction-derived.
- The normalizers are central stand-ins for federated analytics.
- Single dataset; one-week windows.

## 13. Impact matrix
The full table is `POPULATION_5_100_IMPACT_MATRIX_DRAFT.md`, rows 1–39. Additions:

| # | Decision | Mark | Action |
|---|---|---|---|
| 40 | Input history window SESSION_L50 (v1 loader) | **SUPERSEDED** (owner, DECISION_LOG 36) | USER_HISTORY_L50 in the new data version (DATA-05); SESSION_L50 is a named diagnostic only |
| 41 | M04W warm-truncation panel | SUPERSEDED | replaced by the U_NEW_TEST support-size analysis (§8) |
| 42 | The normalizer fit population | RESEARCH_REVIEW_REQUIRED → proposed §6 | the training cohort's TRAIN; a new transform state |

## Questions where I most want Fable's independent view
1. The 1% relative margin with the stop-at-first-failure rule, and 1 + 1 seeds. Is that adequate given the 1-seed noise?
2. The normalization population (§6): the training cohort versus the full pool.
3. Item age: refit it versus classifying it as catalogue metadata.
4. Whether N_MAIN = 131,072 versus 262,144 is sound under the cost argument.
5. The C/FA/FP secondary fine-tune on unseen users: keep it, or drop it to limit scope?

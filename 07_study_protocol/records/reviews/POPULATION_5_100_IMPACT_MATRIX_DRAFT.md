# POPULATION_5_100 impact matrix: Opus DRAFT (Operations Director)

- **Purpose:** an input to `POPULATION_5_100_OPUS_POSITION.md`, required by owner amendment §14.
- **Independence:** this draft is withheld from Fable until Fable has answered.
- **Marks:**
  - UNAFFECTED
  - RECOMPUTE_ONLY: same rule, new numbers or hashes
  - RESEARCH_REVIEW_REQUIRED: the rule itself must be re-decided
  - SUPERSEDED: replaced by the owner

  "Review" below means the deliberation decides.

| # | Decision (current state, source) | Mark | Reason and required action |
|---|---|---|---|
| 1 | Client eligibility: n ≥ 1 eligible TRAIN example (G0 D1; DATA_PROTOCOL_LOCK population) | **SUPERSEDED** | Owner §1: 5 ≤ n ≤ 100, TRAIN-only. n is defined under catalogue A (see #7 on circularity) |
| 2 | Main cohort N = 262,144, salted-hash smallest keys (G0) | **SUPERSEDED** | Owner §2/§4: choose N from the exact census (CENSUS-02). Proposed new selection salt `PPSI_V4_POP_5_100_SEL_20260926` over the pseudonym digest, pending Fable |
| 3 | Calibration N = 65,536 nested prefix (G0) | **SUPERSEDED** | Owner §5: a nested deterministic prefix sized after N_MAIN |
| 4 | Feature schema Clean V2, 53 channels (schema 2.1.0; ADR-003) | RESEARCH_REVIEW_REQUIRED | Owner §6/§7: a minimal-sufficient search on the 5–100 pool, judged by the WORSE of the GRU and SASRec degradations, done BEFORE the architecture lock. Inputs are FEATSTAT-01; the procedure is decided in the deliberation |
| 5 | Normalizers: v1 z fitted on full TRAIN, and the M1 × 2 / F-1 refits on full TRAIN (TRANSFORM_STATE_CLEAN_V2) | RESEARCH_REVIEW_REQUIRED | Owner §9: full TRAIN versus the 5–100 pool, with a provenance classification. The same information block must go to all six methods. Input is the FEATSTAT-01 provenance table |
| 6 | Pooled bootstrap statistics (support counts, pooled popularity or recency channels, vocabularies) | RESEARCH_REVIEW_REQUIRED | As #5. Separate "deployment-available bootstrap" from "pooled from excluded clients" |
| 7 | Catalogue: global TRAIN support ≥ 5, K = 158,486 | RESEARCH_REVIEW_REQUIRED (default: keep) | Owner §8: compare A with B (support from the 5–100 universe). **Circularity:** n depends on the catalogue, so B would move users across strata. Input is CENSUS-02 `CATALOGUE_OPTION_B`. Any change needs its own ADR |
| 8 | OOV: zero credit end-to-end, plus known-item conditional ranking (EVALUATION_CONTRACT_V4) | UNAFFECTED (the rates are RECOMPUTE_ONLY) | Owner §12: re-reviewed, and no change is proposed |
| 9 | VALIDATION population = the cohort's warm VALIDATION users (M02 / M06 rules, eval_stream EVAL_ROWS_ONLY) | RECOMPUTE_ONLY | The definitions are unchanged, applied to the new cohort. The old M02 / M06 are HISTORICAL |
| 10 | Temporal splits: TRAIN, VALIDATION = VS ∪ VC, RETRO_TEST, GRACE | UNAFFECTED | Owner §12. TEST stays sealed |
| 11 | Architecture selection rule (T1: mean over 2 seeds of PRACTICAL_BEST client-macro MRR@20; the T1a–d trigger thresholds) | RESEARCH_REVIEW_REQUIRED (light) | The rule is kept. The thresholds 0.0020 / 0.0040 / 0.0002 were set for the old cohort's VALIDATION size, so re-check them against the new cohort's SE (CENSUS-02 precision prior). The contest now runs on the FINAL compact schema |
| 12 | Architecture tie-break over full TRAIN of all n ≥ 1 users (M_FULL_*; EVAL-04 addendum) | **SUPERSEDED** (the population) and RESEARCH_REVIEW_REQUIRED (the design) | Owner §7: it must stay inside the 5–100 universe. The natural analogue is the ALL-C pool. The EVAL-04 addendum structure is reusable; its numbers are not |
| 13 | EFE concept and the s(EFE) knots | UNAFFECTED (the cost is RECOMPUTE_ONLY) | Owner §11: keep EFE unless Fable finds a flaw. Recompute examples, updates and wall time per EFE for the new N and schema |
| 14 | LR calibration: peaks {1.0, 0.5} × 1e-3, calibration → calibration VALIDATION | RECOMPUTE_ONLY | A new calibration cohort. The multiplier set is kept unless the deliberation says otherwise |
| 15 | Stopping and retention: PRACTICAL_BEST in [0.5, 6], endpoint 6.0, continuation to 8.0 if STILL_IMPROVING, evaluations every 0.5 EFE | UNAFFECTED (the cadence cost is RECOMPUTE_ONLY) | — |
| 16 | LR exposure point "end" (PREG1 T2, pins P1–P7) | UNAFFECTED | — |
| 17 | Evaluation numerics (PREG1 T3: score_block = K, batch 1,024, strict flags, FP64) | UNAFFECTED (RECOMPUTE only if K changes) | Tied to #7 |
| 18 | Training determinism STRICT (amendment 01 item 4, SYS-04 evidence) | UNAFFECTED | — |
| 19 | Local batch 16 | RESEARCH_REVIEW_REQUIRED | Owner §10. Clients now hold 5–100 examples. Input is CENSUS-02 `FL_LOCAL_WORK` |
| 20 | Local work: 2 passes (epochs) | RESEARCH_REVIEW_REQUIRED | Owner §10: steps versus epochs |
| 21 | Client sampling: 64 clients per round, visits and participation | RESEARCH_REVIEW_REQUIRED | Owner §10: participation rate, clients per round, rounds per EFE |
| 22 | FedAvg weighting by consumed examples | RESEARCH_REVIEW_REQUIRED | Owner §10. The 5–100 cap compresses the weight concentration (Gini is in CENSUS-02) |
| 23 | FedProx μ grid and weighting | RESEARCH_REVIEW_REQUIRED | Owner §10; calibrated at G3 |
| 24 | PFL personal state (the p_u query offset, λ grid, separate clip) | RECOMPUTE_ONLY (the state size scales with N), with λ reviewed at G3 | — |
| 25 | CPF control (central personalized, distinct-user penalty) | RECOMPUTE_ONLY | — |
| 26 | Local-only replay: LO lr_peak · s(6(t+1)/T) | RECOMPUTE_ONLY | T per client is now 5–100 examples, so the step counts change. The pins stay |
| 27 | Cold-start definitions: M03 = no TRAIN events; M04 cold-h (candidate A); M04W warm truncation; M04S few-shot selection | RESEARCH_REVIEW_REQUIRED | Owner §13. True cold is n = 0, which includes A1 users (TRAIN events but n = 0) as well as A0. SPARSE 1–4 is a separate stratum that is never called cold. HEAVY > 100 is diagnostic. Reconcile h = {0, 1, 5, 10, 20, 50} with the strata. The truncation panels must draw from C |
| 28 | Communication accounting (bytes per round) | RECOMPUTE_ONLY | Depends on the schema widths and K |
| 29 | Device preprocessing (the on-device transform) | RECOMPUTE_ONLY after #4 and #5 | — |
| 30 | Model export inputs (the ONNX signature) | RECOMPUTE_ONLY after #4 | — |
| 31 | Statistics: user-level bootstrap, B, seed rule, W registrations | UNAFFECTED method (the W registrations are RECOMPUTE_ONLY) | — |
| 32 | Model topologies and widths (MODEL-01), θ0 hashes, inventories | RECOMPUTE_ONLY after #4 (and #7 if K changes) | The widths come from the schema. The server-only θ0 hash rule stays |
| 33 | DATA_PROTOCOL_LOCK v1.0.0 | **SUPERSEDED** in the population part; preserved | A new data version and lock v2 after the synthesis. The old lock file stays read-only as history |
| 34 | The v4data builder (hard-coded population selection, PRE_LOCK_IDS) | RECOMPUTE_ONLY (new code version) | The locked module cannot change, so a new versioned builder imports it unchanged or supersedes it with a new hash. It needs its own review |
| 35 | Real-data providers and launcher (INT-01) | UNAFFECTED (population-agnostic) | Any hard-coded manifest ID or count is removed (Ops instruction) |
| 36 | Owner question: EQUIVALENT → "GRU on cost" | UNAFFECTED (still pending) | — |
| 37 | Thesis claims and generalization scope | RESEARCH_REVIEW_REQUIRED | The main causal claims are restricted to 5–100 clients. Cold, sparse and heavy are secondary generalization strata, and the limitations must say so |
| 38 | NEW: reuse of VALIDATION for both feature selection and architecture selection | RESEARCH_REVIEW_REQUIRED (new) | Adaptive reuse. Options: run the feature search on the calibration cohort's VALIDATION and the contest on the main cohort, or disclose the reuse. The deliberation decides |
| 39 | NEW (owner question, DECISION_LOG 34): a user-disjoint UNSEEN_5_100 evaluation panel, plus the treatment of users with few VALIDATION/TEST events | RESEARCH_REVIEW_REQUIRED (new) | Ops recommendation: keep the global temporal split. Add a disjoint panel drawn from the same pool with the same hash rule (never trained on). Evaluate it in the same windows, with pre-cutoff history as input only. Do not filter on eval-window event counts. Report per-user eval-count strata as descriptive only. PF new-client handling (p_u = 0 versus local adaptation) and LO are predeclared. The owner's decision is pending |

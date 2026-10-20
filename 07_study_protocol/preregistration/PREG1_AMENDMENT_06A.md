# PREG1 amendment 06A — addendum to amendment 06 (registration only; Fable 5.1 full review, owner-mandated)

- **Status:** REGISTERED 2026-09-28 (~17:10Z) by Claude Fable 5.1 acting as reviewer + Operations Director, at the
  owner's explicit request ("let Fable review everything done and to be done, fix what is right, plan literally").
  Registered BEFORE any downstream value is read (C_SMALL, RULE, KNN_EMB, FL, device: none read; server down for the
  disk upgrade). Review: `company/reviews/FABLE_FULL_REVIEW_AND_PLAN_20260928.md`. Builds on amendments 03–06.
- **Unchanged:** every registered value of amendments 03–06 (arms, knobs, z, S rule, retention rule, strata, TEST seal).

1. **FT / LO recipe-selection panel (M04S substitute; supersedes INT-04's CALIBRATION substitute, DL 138/140):**
   the selection panel is the **UNSEEN_VAL** role of REGISTRY_V2 (support UNEW_VAL_SUPPORT_32768 → eval
   UNEW_VAL_VALIDATION_32768; 12,792 evaluable users; disjoint from MAIN, from UNSEEN_HOLDOUT and from every report
   panel). Reason: CALIB is the rank-0–32,768 prefix of MAIN, so selecting on CALIB_VALIDATION would let 25% of the
   report users take part in the selection; E11.9a registered a disjoint unseen-user panel (M04S). Grid unchanged
   (passes {1,2,4} × LR {0.1,0.3} × SMALL peak; tie → fewer passes, then smaller LR); selected ONCE per arm (FT_C,
   FT_FA; LO optional), recorded in SELECTION.json (O_EXCL) before any MAIN/panel application. Code change: INT-05
   (`SELECTION_ROLE = "UNSEEN_VAL"` in v4int/launch_device.py + tests). Fallback if UNSEEN_VAL is technically blocked:
   CALIBRATION, with a mandatory CALIB-disjoint diagnostic on MAIN_VALIDATION (as G2 condition 5).
2. **Cold start (proposal Step 4) — primary analysis is observational:** no E11.4 truncated-history builder exists, so
   COLD_H_16384 / WARM_TRUNCATION_16384 are OPTIONAL (DATA-07, last in the deferred tier). Registered primary analysis:
   per arm and seed, retention vs C_BIG (and vs C_SMALL) within bins of the client's TRAIN-period support count
   n_support ∈ {0 (TRUE_COLD), 5–10, 11–20, 21–50, 51–100} on MAIN_VALIDATION and on UNSEEN_HOLDOUT (Study A =
   context-only arms C_SMALL/FA/DP; Study B = LO/FT_C/FT_FA/PF). events-to-90%/95% = the smallest bin b with
   M(b) ≥ X · C_ref(b), C_ref = C_BIG PRACTICAL_BEST on the same bin, same seed; reported only if reached; no
   interpolation. Bins are fixed here, before any read. Disclosed: observational (history length is not manipulated).
3. **Communication / convergence curve:** quality vs rounds and vs cumulative bytes per client is taken from the 12
   in-run evaluation marks of every FL run (FA, FP, PF, FA_1024, DP) — no extra runs. Bytes = fl.comm.visit_bytes.
4. **Deferred tier order (re-ordered by value/cost):** DP ε = 1 (seed 2026; z = 1.3094 registered) → C_FULL seed 2026
   (amendment 06 §5) → CPF → KD_SMALL / FT_KD → DATA-07. Each starts only after the core (§5.1–5.8 of the review) is
   complete, with a DECISION_LOG row before its launch.
5. **A40 block:** before any downstream run on an A40: T3 evaluator receipt + a STRICT determinism receipt (two
   identical smoke runs, identical final-weight hashes) on the A40; C_BIG (best.pt, endpoint_6.0.pt) and C_SMALL
   (best, endpoint) re-evaluated on the A40 (evaluation only, no training) → all retention is computed inside one
   comparison block with eval_device_class = A40. A changed torch/CUDA version after the upgrade → stop and report.
6. **ONNX INT8 retention rows:** full MAIN_VALIDATION if the single-thread bench is ≤ 2 h (INFERENCE); otherwise the
   first 16,384 MAIN_VALIDATION users in population-selection-key order (registered fallback, deterministic).
7. **Retention pairing rule (clarification of amendment 06 §7):** numerator and denominator use the same checkpoint
   rule (PRACTICAL_BEST/PRACTICAL_BEST; CONTROLLED_BUDGET/CONTROLLED_BUDGET) and the same seed; then the 2-seed mean;
   paired user bootstrap 1,000, seed 20260926. Secondary NDCG@10 / HR@10 from the frozen evaluator's rows.
8. **Analysis script (ANA-01):** `tools/retention_report.py` produces every table/curve of the review §6; frozen by
   hash in `company/gates/ANA_SCRIPT_FREEZE.json` and run on VALIDATION (preflight) BEFORE ALL_METHODS_LOCK and the
   single TEST look.
9. **Disclosures registered for the thesis:** (a) the 6.0-endpoint reversal of the family order (G2 condition 5);
   (b) the FL local LR is the single SMALL peak (the {1×, 0.5×} grid of the realistic plan was not carried into
   amendment 06 or the code — untuned, like μ); (c) FedBN is inapplicable (LayerNorm backbone; FedLN not added by the
   owner); (d) purchase-AUC / comparable-offer tasks of the proposal are out of scope by agreement (next-item task);
   (e) battery is not measured; (f) per-client batching in launch_device reorders 6 / 1,251 rows by one rank position
   at ranks > 20 (no effect on @≤20 metrics); (g) VALIDATION values are max-of-marks; TEST is the only unbiased number.

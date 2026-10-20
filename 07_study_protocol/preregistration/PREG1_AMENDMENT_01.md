# PREG1 amendment 01 — clarifications only. PREG1_LOCK_VALUES.json is left unchanged because its hash is in DATA_PROTOCOL_LOCK.

- **Status:** PROPOSED by the Operations Director on 2026-09-26. It takes effect after Fable confirms it at G1.
- **Scope:** wording and registration only. No threshold, rule or value changes.

1. **The tie chain is superseded (T1).**
   - PKG 05 §B's exact-tie chain (|Δ| ≤ 1e-8 → mean NDCG@10 → measured latency → lexical order) can never be reached
     under the PREG1 T1 rules. Any |Δ̄| < 0.0020 fires trigger T1a first.
   - The family decision in that region is governed only by the staged tie-break, and then by the
     CONSISTENT_SMALL_ADVANTAGE / ARCHITECTURE_EQUIVALENT rules.
   - Source: the TRAIN-01 handoff; `code/v4train` selection tests.
2. **The tie-break evaluation manifest ID.**
   - The full warm VALIDATION manifest used by the full-TRAIN tie-break is registered as `M_FULL_VALIDATION_WARM`: the
     642,228 warm users and 8,430,391 rows.
   - The full-TRAIN training manifest is `M_FULL_TRAIN`: 2,694,998 clients and 40,180,608 examples.
   - Both are built with the locked v4data builder, and only if the T1 trigger fires. Each is hashed and registered in
     an addendum receipt before the launch.
   - MANIFEST_SPEC_V4 gains their field definitions (task EVAL-04).
3. **Endpoint numerics (T2).**
   - s(6.0) evaluates to 0.009999999999999995 in IEEE double. The "LR = 0.01 × peak at 6.0" pins are checked to
     1e-12, not as exact equality.
   - The 1e-12 endpoint tolerance, with a clamp to the knot, applies identically in `fl.schedule` and
     `v4train.schedule` (pin P3). The FL-02R1 conformance task implements it.
4. **Determinism mode during training (T3 context).**
   - The frozen T3 strict flags govern EVALUATION.
   - Training runs with deterministic algorithms in warn-only mode, as FL does, unless the G1 GPU check shows that
     strict mode works on the A16 for the training ops.
   - The mode used is recorded in every run config and in the comparison key.
5. **The evaluation stream (T3 context), a new comparison-key field from EVAL-03.**
   - Proposed freeze: `eval_stream = "EVAL_ROWS_ONLY"` for every method and every evaluation. It covers the manifest's
     rows with `eval_mask = has_observed_target`, which includes OOV targets (scored 0 end-to-end) and excludes
     censored rows (no observed target).
   - For M02 that is 808,610 rows. The other 191,579 manifest rows are censored and contribute nothing to any metric
     by definition.
   - Source: the EVAL-03 handoff.


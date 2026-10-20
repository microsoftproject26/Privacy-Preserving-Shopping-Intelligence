# SOTA_CLAIM_STATUS

> PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK

The locked headline `A1_RICH256_s2027` beats the existing locked GRU F0 baseline on the frozen, previously inspected REES46 TEST period (measured; paired user bootstrap +0.001780 [+0.001538, +0.002022], CI excludes 0).
Best among the configurations evaluated in this campaign (bounded search: A0/A1 × seeds × epochs 3–5).
Not shown to beat published SASRec-family variants: no same-protocol reproduction of external papers was run.
No fresh confirmatory holdout: VALIDATION_CONFIRM/TEST were inspected in the GRU project.
Systems S1–S4b are reported separately and are not a model-versus-model comparison with GRU.

## Machine-readable statuses
```json
{
  "training_status": "COMPLETE",
  "search_status": "COMPLETE_BOUNDED",
  "review_status": "PLAN_AND_CODE_APPROVED",
  "retrospective_evaluation_status": "COMPLETE",
  "better_than_gru_on_reused_test": true,
  "gru_bootstrap_key": "A1_RICH256_s2027_minus_GRU_F0_locked",
  "tf32_parity_gate_pass": true,
  "new_confirmatory_holdout_available": false,
  "universal_sota_proven": false,
  "label": "PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK",
  "lock_sha256": "004130c685676889030ef124c0482c152e6d2466807eac4c177e3ba5e8fa47d0",
  "systems_compared_with_gru_as_model_vs_model": false,
  "privacy_claim": "NONE",
  "causal_claim": "NONE"
}
```

# SOTA_CLAIM_STATUS

Not yet measured: the retrospective evaluation against GRU has not produced a paired bootstrap result. No SOTA claim is made.
Best among the configurations evaluated in this campaign (bounded search, see section 6).
Not shown to beat all published SASRec-family variants: no same-protocol reproduction of external SASRec papers was run in this campaign.
Fresh confirmatory holdout unavailable: this campaign's VALIDATION_CONFIRM/TEST were previously inspected in the GRU project (test_exposure=previously_seen_in_gru_project).

## Machine-readable statuses
```json
{
  "training_status": "COMPLETE",
  "search_status": "COMPLETE_BOUNDED",
  "review_status": "PLAN_AND_CODE_APPROVED",
  "retrospective_evaluation_status": "COMPLETE",
  "better_than_gru_on_reused_test": null,
  "gru_bootstrap_key": "A1_RICH256_s2026_minus_GRU_F0_locked",
  "tf32_parity_gate_pass": true,
  "new_confirmatory_holdout_available": false,
  "universal_sota_proven": false
}
```
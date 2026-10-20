# G1 residual code items (to be done ONCE, after the G0c population decision fixes the new manifest ids)

Recorded 2026-09-26T19:23Z by the Operations Director. Sources: the INT-01 and TRAIN-01R1 handoffs.

| # | Item | Owner | Why it waits for G0c |
|---|---|---|---|
| R1 | `v4train.provider.expected_manifests` and the trainer's own check hard-code M01 / M05 / M02 / M06. Replace them with a registered manifest map, keyed by data version and role (MAIN, CALIBRATION, TIEBREAK), read from the new data lock | Training (TRAIN-01R2) | The new population's manifest ids and data version do not exist yet |
| R2 | The frozen eval config carries no `eval_stream`; INT-01 reads the proposed value from PREG1_AMENDMENT_01 item 5. Once Fable confirms, `v4eval.frozen_config` must carry `eval_stream`. That changes the eval fingerprint, and possibly evaluator_code_sha256, so the T3 GPU receipt must be re-bound (or re-run if the numerics path changes) | Evaluation (EVAL-03R2) | Fable confirmation of item 5 (G1) |
| R3 | Selection has no built-in smoke concept; smoke rows are refused only through relabelling. Add an explicit `SMOKE_NOT_EVIDENCE` refusal in `v4train.selection` with a negative control | Training (TRAIN-01R2) | Bundled with R1, so v4train changes once |
| R4 | INT-01 chose the comparison-key values for provider rows (data version, population hash, feature hash, and so on). They must be confirmed against the new data lock v2 fields | Ops + Fable at G1 | Depends on the lock v2 field names |
| R5 | After R1–R3, the final complete OPS-07 server run of all suites (including tests_int and tests_train_gpu's probe) becomes the G1 evidence. `run1_final` (6/7) is superseded as evidence, but its record is kept | Ops | After R1–R3 |

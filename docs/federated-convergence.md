# T1 Convergence and Communication-to-Target (S2-PR-02 / #36)

`ppsi.federated.convergence` answers two questions about a **completed** federated run:
when did it settle, and how many measured bytes had been exchanged by then. It trains
nothing and measures nothing itself.

The module has two layers on purpose. `analyze_curve` and `sum_measured_communication`
are pure arithmetic with no notion of project identity. `analyze_run_curve` is the only
supported way to feed a real R1 value or real byte records in, because it refuses
anything whose identity it cannot verify against the frozen contracts.

## 1. The two rules are deliberately different

| | Self convergence | R1 target |
|---|---|---|
| Reference | 95% of this completed run's best finite VALIDATION score | 90% of an identity-matched R1 VALIDATION value |
| Observations required | three consecutive scheduled evaluations | one |
| Reported boundary | confirmation round, the third qualifying observation | first observed crossing |
| Also reported | streak start round | — |

Both comparisons are inclusive (`>=`). There is no epsilon and no rounding policy.

The self rule is **retrospective**. It compares against the best score of a run that has
already finished, which is unknowable while training. It must never be wired into an
online early-stopping controller.

The streak start and the confirmation round are both returned because they answer
different questions. Communication-to-self-convergence uses the **confirmation** round.

## 2. Metric scope

This v1 policy is T1-scoped and uses the already-frozen headline
`t1.next_distinct.mrr_at_20.macro` on C1 VALIDATION, direction `MAXIMIZE`, unit
`FRACTION`. The matching metric identity includes the slice and the macro averaging; the
overall slice, the micro average, a TEST metric and a runtime smoke diagnostic are all
rejected.

The function is reusable but the policy is not universal. A lower-is-better metric is
refused outright, so these fractions are never applied to LogLoss or Brier. Another
metric needs its own versioned policy.

## 3. Cadence, gaps and degenerate curves

Cadence is explicit and defaults to every round. A non-unit cadence is supported and is
recorded. Nothing is interpolated and no gap is bridged.

- A missing scheduled round breaks the streak and is reported in `cadence_gap_pairs`.
- A `null`, `NaN` or infinite observation breaks the streak and is listed in
  `missing_observation_rounds`.
- Duplicate or decreasing round numbers are rejected outright, as are values outside
  `[0, 1]` and non-real values.

Degenerate curves get explicit statuses instead of a fabricated answer:
`NO_VALID_OBSERVATIONS`, `NO_POSITIVE_REFERENCE` when the best score is zero,
`INSUFFICIENT_VALID_OBSERVATIONS`, `NOT_REACHED`, `PENDING_R1` when no R1 was supplied,
and `UNDEFINED_ZERO_R1`.

## 4. The identity guard

`analyze_run_curve` performs these checks before any R1 scalar is used:

1. The policy schema, metric id, task, cohort, split, direction and unit must match the
   curve's own declared identity, and the cadence must match the policy.
2. The curve's `comparison_compatibility_v1` tuple is validated with the existing helper.
3. The supplied R1 ExperimentResult is validated with the existing schema validator and
   must be `SUCCEEDED`, regime `R1`, and cover the task and cohort.
4. The R1 run's **resolved ExperimentConfig** must be supplied and must declare
   `evaluation_split = VALIDATION` explicitly. `ExperimentResult v1` carries no split
   field, so the split is read from the run's own contract. It is never inferred from a
   filename, a metric name, or a value the caller passes in.
5. That config must reproduce the result's compatibility tuple, binding the two together.
6. The complete tuples are compared with `compare_compatibility`. Any mismatch is fatal
   and no field, including `git_sha`, is dropped to make a comparison succeed.
7. Only then is the single matching metric selected. Zero matches or more than one match
   is an error rather than a guess.

Referenced files are verified against their recorded hashes using the owning artifact's
convention.

## 5. Communication boundary

The accumulated interval is training rounds 1 through the selected boundary, inclusive.
Round 1's download includes the initial model sent out for training. The server-only
evaluation at round 0 is outside the interval.

Input rows must carry `run_id`, `config_sha256`, `server_round`, `upload_bytes`,
`download_bytes` and `measurement_basis = MEASURED_APPLICATION_MODEL_PAYLOAD`. Records
belonging to another run or config, negative counts, duplicate rounds and estimated tensor
sizes are all rejected. Upload and download are kept separate throughout.

Missing measurements produce `PENDING_COMMUNICATION`; a missing intermediate round
produces `INCOMPLETE_COMMUNICATION` with null totals. **Absent bytes are never zero.** A
genuine zero is only legal inside a complete supplied measured record.

Measured retries, if a future approved run has any, must already be inside their own
round totals. They are neither double counted nor erased.

## 6. What is proven today, and what is not

`scripts/federated/validate_convergence.py` replays eight fixed JSON cases, six numeric
boundary cases and one hand-worked byte oracle, then writes
`docs/evidence/s2-pr-02/convergence_validation.v1.json`. The output is byte-stable across
repeated runs because it carries no timestamp and no working-tree state.

The byte oracle sums uploads of 10, 20 and 30 with downloads of 100, 100 and 100 for a
total of **360**. That is labelled `FIXTURE_PROOF`. It is hand-worked arithmetic and it is
**not** measured Flower traffic.

Two things stay explicitly pending and neither is a defect in this module:

- `real_r1_status = PENDING_R1`. No final R1 result exists yet, so no
  communication-to-target research result can be claimed.
- `actual_communication_status = PENDING_50_INSTRUMENTATION`. Real send and receive
  instrumentation belongs to #50 and is deliberately not duplicated here.

Runtime smoke accuracy from #35 is a diagnostic. It is never copied into the frozen MRR
field and never treated as a convergence curve.

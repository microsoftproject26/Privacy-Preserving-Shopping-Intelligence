# Federated MVP readiness (CP-FL scope)

This is a **readiness matrix for one scoped T1 pilot**, not an unrestricted go-ahead for the
multi-task federated study. Its verdict covers the exact regime that was executed and
nothing wider.

**Verdict: `GO_T1_MVP_ONLY`** on run `mvp-t1-003`, after a real-model canary and a completed
20-round FedAvg execution with independent verification.

## 1. What was required, and what was measured

| Readiness requirement | Evidence | Status |
|---|---|---|
| Real model trains through the shared local core on real frozen data | `canary.v1.json` | PASS |
| Central and Flower-local paths agree on one identical local update | max abs diff 0.0e+00 at atol 1e-6 | PASS |
| A repeated-batch fixture reduces its loss | 6.4225 to 1.8785 | PASS |
| Two loads of the common initialization are tensor-identical | `canary.v1.json` | PASS |
| Real batches pass the canonical batch validator | `canary.v1.json` | PASS |
| Every round receives the complete expected reply set | 20 rounds x 50 clients | PASS |
| Flower aggregation matches an independent float64 oracle | worst round used 5.1% of its derived allowance | PASS |
| Round 0 predictions identical across regimes | identical rank file digest | PASS |
| Exposure identical across regimes | one shared exposure digest | PASS |
| Payload bytes measured at the real message boundary | 1,000 uploads, 1,000 downloads | PASS |
| No client identity in public evidence | verification scan | PASS |
| Independent re-derivation of published metrics | 22 checks | PASS |

## 2. What this verdict does not authorise

- It does not authorise multi-task federated training. T2 and T3 were not trained here, and
  the T1 objective rejects any batch carrying a present T2 or T3 target.
- It does not authorise regimes R2B, R3, R4 or R5.
- It does not authorise a privacy claim. There is no differential privacy, no secure
  aggregation and no device isolation; clients are logical partitions in a simulation.
- It does not authorise a capacity claim beyond 50 concurrent logical clients on one host.
- It does not make this pilot the project's R1 denominator or its final R2A reference.

## 3. Numbers this readiness rests on

| Quantity | Value |
|---|---:|
| Pilot population | 1,000 clients |
| Scheduled participations | 1,000 |
| Distinct clients trained | 636 |
| Rounds completed | 20 |
| Contributing examples | 33,253 |
| Validation decisions scored | 438,185 |
| Centralized raw macro MRR@20 | 0.11268 |
| FedAvg raw macro MRR@20 | 0.09959 |
| Headline quality retention | 88.4% |
| Measured model payload | 19.04 GB |

## 4. Aggregation validity policy

Correctness of averaging is checked per tensor against a float64 oracle, under a bound
derived from float32 machine epsilon, the number of contributing clients, the aggregation
weights and the tensor magnitudes. It is `SCALE_AWARE_FLOAT32_FORWARD_ERROR_V1`.

The bound replaced a fixed 1e-6 absolute tolerance that a correct implementation could not
satisfy at 50 clients: the earlier `mvp-t1-001` attempt failed on a parameter no client had
trained, where the entire difference was accumulation noise. That attempt is retained at
`docs/evidence/mvp/mvp-t1-001/federated_attempt_failed.v1.json` and published no result.
Structural checks were not relaxed: a missing reply, duplicate client, non-positive weight,
key, shape or dtype mismatch, non-finite value or partial round still fails outright.

## 5. Open prerequisites before any wider federated go-ahead

- A final multi-task objective and head policy, frozen before execution.
- Fresh untrained common initializations for the full seed set under that final policy.
- A declared client-capacity envelope for the intended scale, measured rather than assumed.
- An explicit privacy-mechanism decision if any privacy claim is to be made.

# ADR-003 — Deployment and Measurement Readiness

- Status: Proposed, pending approval from all three members
- Decision date: 2026-09-12
- Decision owner: System / Deployment / Security lane
- Related issues: #44, #38, #39, #40, #41, #50, #27, #53

## Decision 1 — Quantization: ship FP32

**FP32 is the Phase 1 deployment artifact. INT8 is not accepted.**

CP-DEPLOY states three conditions for accepting INT8. Measured on the approved
deployment candidate `s2_ds_08_joint_lambda1.0_seed13`:

| Condition | Result | Verdict |
|---|---|---|
| At least 2x smaller | 9,535,393 → 2,715,815 bytes, 3.51x | Met |
| Not slower under the same benchmark | 1.88 ms → 2.36 ms median, one thread | **Not met** |
| No more than 1% relative loss on each headline metric | Not measurable here | Unanswered |

One condition fails outright, so the rule rejects INT8 without needing the third.

Evidence: `docs/evidence/s2-se-08/final_export.v1.json` and
`docs/evidence/s2-se-02/int8_benchmark.v1.json`.

### Why the earlier number pointed the other way

An INT8 benchmark on the prototype model showed 1.22 ms → 1.14 ms, a small gain.
That model had five history channels and five candidates. The deployment
candidate has a 20-step history and 100 candidates.

Dynamic quantization quantizes activations on every call. That per-call cost
scales with the shapes; the saving on weights does not. At prototype shapes the
saving wins, at deployment shapes the cost wins. A decision taken on the earlier
figure would have been wrong, and the two runs are kept side by side
deliberately rather than the first being replaced.

### What this does not claim

The quality condition is unanswered, not passed. It asks for relative loss on
MRR@20, PR-AUC and NDCG@5 computed on the frozen evaluation set, and that set
carries `user_id` and is not in this repository. What was bounded instead is how
far the outputs move under quantization: 0.0908 on `t1_logits`, 0.0765 on
`t2_logit`, 0.0000 on `t3_scores`. That is a bound, not a metric delta.

The `t3_scores` zero is not evidence of safety. That path is dominated by the
frozen retrieval rank, which quantization leaves untouched.

### Revisiting this

The 3.5x saving remains available. If model size becomes the binding constraint,
reopen this with two things the current evidence lacks: a static quantization
attempt with a proper calibration set, and the per-task metric delta on the
frozen evaluation set.

## Decision 2 — Deployment and measurement readiness: GO

The deployment path is reproducible, the exported model computes what the
trained model computes, and the measurements that back the Phase 1 claims are
taken rather than estimated.

### Export and parity

| Head | Max absolute difference |
|---|---:|
| `t1_logits` | 1.79e-07 |
| `t2_logit` | 1.49e-07 |
| `t3_scores` | 0.00e+00 |

Tolerance 1e-4. Checked across eight shape combinations covering batch size,
history length and candidate width alone and together, including a single row, a
single history step and a single candidate. Opset 17, `onnxruntime` 1.24.1,
`CPUExecutionProvider`.

Evidence: `docs/evidence/s2-se-08/final_export.v1.json`,
`docs/evidence/s2-se-01/onnx_parity.v1.json`, `docs/deployment-final-export.md`.

### The export refuses the wrong model

The architecture is read from `config/experiments/s2-ds-08/model_config.v1.json`,
the built parameter count is checked against the 2,379,263 that file declares,
and the weights are verified by SHA-256 against `deployment_candidate.v1.json`
before loading. Any mismatch stops the export.

This guard exists because the failure mode is silent: an untrained
initialisation, a search checkpoint, or a model built from library defaults all
export cleanly and produce numbers that mean nothing.

### Communication measurement

Upload and download are accumulated separately from the serialized arrays Flower
actually transmits, not estimated from parameter counts. On the validation run
the estimate was 544 bytes and the measured payload 800, so the estimate was 32%
low. The boundary is the model payload only; framing, headers, config and metric
records are outside it, and the artifact says so in a `measurement_boundary`
field.

A client whose reply never arrives is not recorded. It did not transmit zero
bytes; it did not transmit.

Evidence: `docs/evidence/s2-se-05/communication_bytes.v1.json`,
`docs/federated-communication-bytes.md`.

### Product representation

Hash collisions on the frozen catalogue, for the architecture record rather than
as a deployment blocker:

| Total buckets | Collision rate | Max bucket load |
|---:|---:|---:|
| 50,000 | 60.50% | 11 |
| 262,145 | 18.84% | 6 |
| 524,289 | 10.16% | 5 |
| 1,048,577 | 5.35% | 4 |

Evidence: `docs/evidence/s2-se-03/product_hash_collisions.v1.json`.

## Conditions on the GO

1. **Running the export against the trained weights is still outstanding.** The
   committed evidence was produced without `--weights`, so it measures the
   shipped architecture and not the shipped model. Size, latency and the export
   path carry over; quantization error does not, because it depends on the
   weight distribution. The evidence file records which of the two it is in a
   `status` field.
2. **The INT8 quality delta stays unmeasured** until someone runs it against the
   frozen evaluation set. This ADR rejects INT8 on speed, so nothing downstream
   depends on that number today.
3. **No privacy claim rests on this document.** The claims table in
   `docs/security/threat-model.md` governs. Flower and Ray here are a simulation
   on one shared host, which shows logical message separation and is not
   physical device isolation. Differential Privacy and Secure Aggregation are not
   implemented, and hashed or namespaced identifiers are not anonymous.

## Dependency note

This checkpoint's issue lists #39 and #41 as hard start dependencies. Both were
closed as not planned and folded into #53 and #38 respectively. Their evidence
exists and is cited above; the issue's dependency list should drop the two names
so it stops implying work that will never arrive under those numbers.

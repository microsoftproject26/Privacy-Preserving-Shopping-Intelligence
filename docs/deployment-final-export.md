# Final Multi-Task Export

The Phase 1 deployment candidate, exported and measured. This is the artifact Phase 1 closes on.

## What is exported

`s2_ds_08_joint_lambda1.0_seed13`, named by `config/experiments/s2-ds-08/deployment_candidate.v1.json`.

Nothing here assumes the architecture. It is read from
`config/experiments/s2-ds-08/model_config.v1.json`, the built parameter count is checked against
what that file declares, and the weights are verified by SHA-256 against the candidate contract
before they are loaded. Any mismatch stops the export rather than producing an artifact that looks
correct.

That guard exists because the failure it prevents is silent: an untrained initialisation, a search
checkpoint, or a model built from library defaults all export perfectly well and all produce numbers
that mean nothing.

## Reproduce

```text
uv run --locked python scripts/deployment/export_final_model.py --report docs/evidence/s2-se-08/final_export.v1.json
```

Add `--weights <path>` to export the trained checkpoint. The weights are not in this repository:
it is public, and the file is 9.53 MB of trained parameters distributed separately.

**Without `--weights` this measures the shipped architecture, not the shipped model.** The evidence
file records which of the two it was, in a `status` field, so the distinction cannot be lost by the
time someone quotes the numbers.

## The signature is 12 inputs, not 15

The batch carries all five history channels. This model consumes two, `category_id` and
`event_type_id`, so the exported graph has twelve inputs rather than fifteen.

The export follows the model rather than the batch. Requiring a caller to supply three tensors the
graph provably ignores would be a false contract, and it was caught the first time this ran against
the real configuration rather than the defaults.

| | |
|---|---|
| Inputs | `category_id`, `event_type_id`, `history_gap`, `lengths`, four `query_*`, `candidate_ids`, `candidate_category_id`, `candidate_price_band`, `candidate_rank` |
| Outputs | `t1_logits`, `t2_logit`, `t3_scores` |
| Dynamic axes | batch, history length, candidate width |
| Opset | 17 |

## Parity, all three heads

| Head | Max absolute difference |
|---|---:|
| `t1_logits` | 1.79e-07 |
| `t2_logit` | 1.49e-07 |
| `t3_scores` | 0.00e+00 |

Tolerance 1e-4, so the worst disagreement is roughly six hundred times inside it. Checked across
eight shape combinations covering each dynamic axis alone and together, including a single row, a
single history step, and a single candidate.

## Size, latency, and the INT8 decision

| | FP32 | INT8 |
|---|---:|---:|
| Serialized | 9,535,393 B | 2,715,815 B |
| Latency, median, one thread | 1.88 ms | 2.36 ms |

Against CP-DEPLOY's three conditions:

| Condition | Result |
|---|---|
| At least 2x smaller | Met, 3.51x |
| Not slower | **Not met.** INT8 is 26% slower |
| No more than 1% relative loss per headline metric | Not measured |

**Recommendation: FP32.**

This is worth reading carefully, because it reverses the earlier result. On the smaller prototype
measured in `docs/deployment-int8-benchmark.md`, INT8 was slightly faster. On the real deployment
candidate, with a 20-step history and 100 candidates, it is slower. Dynamic quantization pays a
per-call cost to quantize activations, and at this shape that cost exceeds what the smaller weights
save.

So a decision taken on the prototype's numbers would have been the wrong one. The three-megabyte
saving is real, and if size ever becomes the binding constraint it is available; today it costs
latency and buys nothing else that has been measured.

The third condition remains unanswered for the same reason as before: the per-task metric delta
needs the frozen evaluation set, which carries `user_id` and is not in this repository. Nothing in
this document should be read as a quality result.

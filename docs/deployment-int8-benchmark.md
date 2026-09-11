# INT8 Quantization Benchmark

Evidence for CP-DEPLOY's INT8 acceptance decision. This measures; it does not decide.

## The rule this was measured against

CP-DEPLOY states three conditions for accepting INT8:

1. at least 2x smaller,
2. no more than 1% relative loss on each approved headline metric,
3. not slower under the same benchmark.

Conditions 1 and 3 are answered below. Condition 2 is not, and the reason matters.

## Reproduce

```text
uv run --locked python scripts/deployment/benchmark_int8.py --report docs/evidence/s2-se-02/int8_benchmark.v1.json
```

Dynamic quantization with INT8 weights and no calibration set. Static quantization would need
calibration data drawn from real task examples, and those carry `user_id` and are deliberately not
in this repository, so dynamic is what makes this attempt reproducible here at all.

## Size

| Model | Serialized bytes |
|---|---:|
| FP32 | 14,678,873 |
| INT8 | 4,003,834 |

**3.67x smaller.** Condition 1 is met with room to spare.

## Latency

Same machine, same shapes, one thread, five warmups discarded, thirty timed repetitions.

| Model | Median | p95 |
|---|---:|---:|
| FP32 | 1.22 ms | 2.28 ms |
| INT8 | 1.14 ms | 2.90 ms |

**INT8 is 1.07x faster at the median.** Condition 3 is met on the stated rule, which names the
benchmark median.

The p95 goes the other way: INT8 is slower in the tail. On a model this small a single forward pass
is about a millisecond, so the tail is dominated by scheduling noise rather than by the model, and
neither the gain nor the loss should be treated as a strong result. What the numbers do establish is
that INT8 is not a latency regression here.

## Quality: what was measured and what was not

| Head | Max absolute output disagreement |
|---|---:|
| `t1_logits` | 0.0908 |
| `t2_logit` | 0.0765 |
| `t3_scores` | 0.0000 |

**This is not a metric delta and must not be read as one.** Condition 2 asks for relative loss on
each approved headline metric, which means MRR@20 for T1, PR-AUC for T2 and NDCG@5 for T3, computed
on the frozen evaluation set. That set carries `user_id` and is not in this repository, so this run
cannot produce it.

What the table gives is a bound on how far the outputs move. A shift of about 0.09 on an unnormalised
logit is not negligible, and whether it survives into the ranking metrics is exactly the open
question. The T3 zero is not evidence of safety either: the T3 path here is dominated by the frozen
retrieval rank, which quantization leaves untouched.

## Recommendation

Accept INT8 on size and speed, and treat quality as unanswered.

The honest reading is that INT8 clears two of three conditions convincingly and the third is
unmeasured rather than passed. The final export in S2-SE-08 should recompute the per-task metric
delta against the frozen evaluation set before INT8 becomes the shipped artifact, and keep FP32 if
the delta exceeds 1%. Nothing here justifies shipping INT8 on the strength of a size ratio alone.

## Caveat on the model measured

These numbers come from a model built from seed 13 rather than from a trained deployment candidate.
The architecture, shapes and operator mix are the ones that will ship, so size and latency carry
over. The output disagreement does not carry over with the same confidence, because quantization
error depends on the weight distribution and an untrained model's weights are not a trained model's.
Re-run this against the S2-DS-08 checkpoint when it exists.

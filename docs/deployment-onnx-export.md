# ONNX Export and Numerical Parity

One reusable export path for the Phase 1 model, and evidence that the exported graph computes what
PyTorch computes.

## Reproduce

```text
uv run --locked python scripts/deployment/export_onnx.py --output artifacts/onnx/session_gru.onnx --report docs/evidence/s2-se-01/onnx_parity.v1.json
```

Add `--checkpoint <path>` to export a trained encoder. Without it the model is built from the seed
alone, which exercises the export path but is not a deployment candidate.

The ONNX binary is not committed. It is 14 MB and the command above regenerates it exactly, so a
copy in git would be duplication rather than evidence. The parity report is committed.

## The frozen signature

| Input | Shape | Dtype |
|---|---|---|
| `category_id`, `product_bucket`, `event_type_id`, `brand_bucket`, `price_band` | `[batch, history_length]` | int64 |
| `history_gap` | `[batch, history_length, 1]` | float32 |
| `lengths` | `[batch]` | int64 |
| `query_category_id`, `query_product_bucket`, `query_brand_bucket`, `query_price_band` | `[batch]` | int64 |
| `candidate_ids` | `[batch, candidate_width]` | int64 |
| `candidate_category_id`, `candidate_price_band` | `[batch, candidate_width]` | int64 |
| `candidate_rank` | `[batch, candidate_width, 1]` | float32 |

| Output | Shape |
|---|---|
| `t1_logits` | `[batch, categories]` |
| `t2_logit` | `[batch]` |
| `t3_scores` | `[batch, candidate_width]` |

Three axes are dynamic: `batch`, `history_length`, `candidate_width`. Nothing else is, because
nothing else varies.

The masks and the four target tensors are not inputs. The forward pass never reads them, so
accepting them would put fields in the signature that cannot change the answer.

Opset 17, `CPUExecutionProvider`, `onnxruntime` 1.24.1, `torch` 2.13.0+cpu.

## Parity result

Ten shape combinations, each fed to both runtimes with identical inputs:

| Head | Max absolute difference |
|---|---:|
| `t1_logits` | 1.42e-07 |
| `t2_logit` | 1.79e-07 |
| `t3_scores` | 0.00e+00 |

Tolerance is 1e-4. The worst disagreement is roughly six hundred times smaller than that, and it
sits at float32 rounding, which is what agreement looks like rather than a margin that happens to
pass.

## The defect this found

The first export produced a graph that worked only at the batch size it was traced on. Any other
batch size failed inside ONNX Runtime:

```text
Attempting to broadcast an axis by a dimension other than 1. 2 by 4
```

The cause was in `encode_history`:

```python
rows = torch.arange(batch.batch_size, device=batch.lengths.device)
gathered = sequence[rows, last]
```

`Phase1Batch.batch_size` returns `int(self.lengths.shape[0])`. A Python integer becomes a constant
in the traced graph, so the exported model carried the traced batch size inside it.

It now uses `gather`, which expresses the same selection without materialising a row index:

```python
index = last.view(-1, 1, 1).expand(-1, 1, sequence.shape[-1])
gathered = sequence.gather(1, index).squeeze(1)
```

The two forms were compared directly over twenty combinations of batch size and history length, and
the maximum difference was exactly zero. The change is to how the row is selected, not to what the
model computes.

`test_a_batch_size_other_than_the_traced_one_still_agrees` is the regression guard. It is worth
keeping precisely because the failure was loud in ONNX Runtime and invisible in PyTorch.

## What the tests cover

- The graph validates and pins its opset, input names and output names.
- Outputs agree at the traced shape.
- Batch sizes 1, 2, 7 and 16 agree against a graph traced at 4.
- History lengths 1 and 12 and candidate widths 1 and 20 agree.
- A row with no history agrees and contributes zero.
- Padded positions are inert: rewriting them leaves every head unchanged.
- Repeated runs give identical numbers.
- A drifted input or output name is refused with a named error rather than a shape mismatch.

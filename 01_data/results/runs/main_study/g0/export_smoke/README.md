# G0 ONNX export feasibility smoke (Sonnet C)

Random-initialised weights only (no checkpoints), FP32, CPU only
(`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`), synthetic inputs only (no
dataset event/user rows -- only the frozen catalogue mapping used to learn
`K`/`product_idx_of_class`, same as every other model-building script in this
repo). Server: torch 2.13.0+cu130, onnx 1.20.0, onnxruntime 1.24.1 (CPU
provider), AMD EPYC.

Scripts: `code/g0_export/common.py`, `run_gru.py`, `run_sasrec.py`.
Raw JSON: `g0/export_smoke/GRU_REPORT.json`, `SASREC_REPORT.json`.
`.onnx` files are kept **only on the server** (`g0/export_smoke/*.onnx`) --
not copied locally, per the "large files stay on server" instruction.

## Environment-level blocker (affects both families)

The **default** exporter in this torch build (`torch.onnx.export(..., dynamo=True)`,
the only mode listed as default in `inspect.signature`) is completely unusable
here: it imports `onnxscript`, which is not installed, and no `pip install`
was permitted. Every export attempt below therefore falls back to the
**other** mode, `dynamo=False` (the legacy TorchScript-trace-based exporter).
Exact error, identical for every attempt:

```
ModuleNotFoundError: No module named 'onnxscript'
```

This is worth flagging to whoever owns the venv: if the dynamo/onnxscript
path is wanted later, `onnxscript` needs to be added to `~/venvs/ppsi-gpu`.

## Summary table

| Family  | Export (full logits) | Export (query-only) | Dynamic axes | Parity (max\|Δlogit\|) | p50 / p95 ms (b1, len50) | p50 / p95 ms (b16, len50) | ONNX size |
|---|---|---|---|---|---|---|---|
| **GRU** (ContextGRU, F4_rich) | **FAILED** (both exporter modes) | **FAILED** (both exporter modes) | n/a | n/a | n/a | n/a | n/a |
| **SASRec** (A1_RICH256) | OK (legacy exporter) | OK (legacy exporter) | batch: yes; seq: fixed L=50 (see below) | ~1.4e-6 (FP32 noise floor); top-20 and target-rank identical in all 8 scenarios | 6.96 / 7.06 ms | 36.6 / 37.9 ms | 317.0 MB (full) / 161.6 MB (query) |

All latencies are **SERVER-CPU PROXY numbers (AMD EPYC), NOT a phone/browser measurement.**

## GRU (ContextGRU, F4_rich_side_information) -- BLOCKED

`ContextGRU.encode()` calls
`nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)`
before the `nn.GRU`. Both exporter modes were tried, plus a follow-up sweep
(opsets 9/12/15/17/18, with/without `dynamic_axes`) -- all fail, so this is
not a dynamic-axes or opset artifact, it is inherent to exporting this
PackPadded(`enforce_sorted=False`)+GRU pattern in this torch build:

- **opset 9**: fails earlier, on the internal `sort` op that `enforce_sorted=False`
  needs (`aten::sort`/restore-order machinery):
  ```
  SymbolicValueError: Unsupported: ONNX export of operator Sort, input size not
  accessible. ... (node defined in torch/nn/utils/rnn.py(313): pack_padded_sequence)
  ```
- **opset 12/15/17/18** (i.e. every opset actually usable at `opset_version=18`
  and below that reaches the same code path): the PackPadded→GRU pattern *is*
  recognised (`prim::PackPadded` appears in the traced graph, confirmed by the
  "shape inference of prim::PackPadded type is missing" warning), but graph
  optimization then segfaults-as-exception inside PyTorch's own C++ ONNX
  peephole pass:
  ```
  RuntimeError: vector::_M_range_check: __n (which is 0) >= this->size() (which is 0)
    ... torch/onnx/_internal/torchscript_exporter/utils.py:693 _optimize_graph
    ... _C._jit_pass_onnx_peephole
  ```
  Reproduced identically with no `dynamic_axes` at all (fully static shapes) and
  with only the batch axis marked dynamic -- ruling out dynamic-axes as the
  cause. This looks like a genuine bug/incompatibility between
  `enforce_sorted=False`'s extra sort/scatter/restore ops and the ONNX peephole
  optimizer in torch 2.13.0+cu130.

**Consequence for the plan**: as currently written (frozen snapshot,
`code/snap_v1/reuse/model.py`, not modified by this task), ContextGRU cannot
be exported to ONNX in this environment via either `torch.onnx.export` mode.
Options for whoever owns the model code (not exercised here -- out of this
task's scope, which only owns `g0_export/`/`g0/export_smoke/`): (a) install
`onnxscript` and retry the dynamo path (untested, may hit its own dynamic-shape
issues with `.item()`-based lengths handling seen on SASRec's side); (b) change
`encode()` to call `pack_padded_sequence(..., enforce_sorted=True)` after
sorting by length outside the packed call (removes the extra sort/restore
ops that appear implicated); (c) replace the packed-GRU path with an
explicit masked loop/matmul formulation (SASRec-style), which is why the
"split encoder/head" and dynamic-axis probes below could not be run for GRU at all --
there is no `.onnx` file to test parity/latency against.

Full attempt records (both `dynamo=True` and `dynamo=False`, with warnings
and full traceback tails) are in `GRU_REPORT.json`.

## SASRec (SASRec-CE-Rich, A1_RICH256, d256/2 blocks/4 heads) -- WORKS

Exported successfully with the legacy exporter (`dynamo=False`), opset 18,
for both the full logits path and the query-encoder-only split. n_params =
83,094,368. Export warnings (`_diag`-verified, listed in full in
`SASREC_REPORT.json`) are the expected ones for a data-dependent trace:
"Converting a tensor to a Python boolean/number might cause the trace to be
incorrect" (from `_layout`'s validation asserts and `T = int(lengths.max().item())`)
and "Iterating over a tensor..." (the per-block Python `for` loop over
`self.blocks`, fully unrolled at trace time -- correct for a fixed
`cfg.blocks=2`, just not adaptive if the block count changed).

### Dynamic axes: batch yes, sequence length NO (fixed L=50, by construction)

`SASRecCE._layout` computes `T = int(lengths.max().item())` -- a Python-level
data-dependent conversion. Whatever `T` the trace's calibration batch happens
to have is baked into the graph as a **constant**, not a true dynamic axis
(this is unavoidable without changing the model code, which is out of this
task's scope). We calibrated the trace with a batch guaranteed to contain a
length-50 row, so **T bakes to the full L=50 window** -- the outcome the task
brief anticipated as "fixed L=50 with padding". Given that, the graph is
*not* naive fixed-shape: it still uses the real runtime `lengths` tensor
(via `valid = arange(T) < lengths`, a genuine graph-level comparison, plus
`torch.nonzero`/`index_select`/`index_copy`, which ARE natively dynamic ONNX
ops) to correctly mask per-row padding for **any** actual lengths in [1,50] at
inference -- confirmed empirically by parity holding exactly across length
1/8/50, batch 1/16, a "missing side-features" batch, and a batch with mixed
per-row lengths (see below). So: batch dimension is genuinely dynamic;
sequence width is fixed at the frozen L=50 window (which matches the whole
pipeline's contract -- every window is always materialised at width 50 with
`lengths` telling you how much of it is real, so a variable-width sequence
axis was never actually the right shape of dynamism to ask for here).

### A genuine, reportable side-effect: `attention_mask` and `position_ids` are dropped from the ONNX graph

Both exported graphs (full and query) **do not declare `attention_mask` or
`position_ids` as inputs** -- `onnxruntime`'s `InferenceSession.get_inputs()`
lists only `item_tokens, lengths, category_tokens, brand_tokens,
main_category_tokens, daypart_tokens, weekday_tokens, event_numeric_features,
event_quality_flags, user_context, user_context_masks`. Root cause (verified
by reading `_layout`): `attention_mask`/`position_ids` are used *only* inside
a consistency assert (`bad = ...; if bad.item(): raise ValueError(...)`) whose
result never reaches the traced output; the model's real computation
re-derives both the validity mask and the 0..T-1 positions from `lengths`
alone. The legacy exporter's dead-code elimination correctly prunes that
assert subgraph, and the now-unused input tensors disappear from the ONNX
graph entirely -- feeding them back by name raises
`onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: Invalid input
name: position_ids` (this is what our first parity run hit; fixed in
`common.py:filtered_feed`, which restricts the runtime feed dict to
`sess.get_inputs()`). **Practical implication**: at serving time, only
`item_tokens` + `lengths` (+ side channels) are actually required by this
exported graph -- `attention_mask`/`position_ids` can be omitted by callers
of the ONNX artifact (though the frozen store's contract still requires them
be internally consistent, since that's what an eager PyTorch call would
validate).

### Parity (PyTorch FP32 vs onnxruntime CPUExecutionProvider FP32)

All 8 scenarios (`len1_b1, len1_b16, len8_b1, len8_b16, len50_b1, len50_b16,
missing_b4_len8` [all side-channel tokens = MISSING(1), flags on, numeric
zeroed], `mixed_pad_b16` [random per-row lengths 1..50]):

- max |logit diff|: **1.0e-6 to 2.1e-6** (FP32 rounding-noise floor, not a
  correctness gap)
- top-20 index set identical: **100% of rows, every scenario**
- exact rank of a random target class identical: **100% of rows, every
  scenario**
- query-encoder-only split (`q` vector, then head reconstructed in numpy/torch
  from `model.output_embed`/`output_bias`) reproduces the same top-20/rank
  parity as the full path (e.g. `len8_b16`: max\|Δq\| = 2.1e-6) -- the
  split encoder/head option is numerically sound.

Full per-scenario numbers are in `SASREC_REPORT.json` (`parity_full`,
`parity_query_plus_reconstructed_head`).

### Latency (onnxruntime CPUExecutionProvider, intra_op_num_threads=4, 5 warm-up + 50 timed runs) -- SERVER-CPU PROXY (AMD EPYC), NOT a phone/browser measurement

| scenario | p50 ms | p95 ms |
|---|---|---|
| len1, batch1 | 5.74 | 5.79 |
| len1, batch16 | 12.89 | 13.63 |
| len8, batch1 | 5.68 | 5.71 |
| len8, batch16 | 16.34 | 16.82 |
| len50, batch1 | 6.96 | 7.06 |
| len50, batch16 | 36.62 | 37.91 |

First-load (session creation) time: 0.90 s (full graph), 0.38 s (query-only
graph). Process peak RSS: ~1.9 GB (this process also holds the PyTorch model
+ both onnxruntime sessions + K=158,486-wide logits tensors simultaneously;
not a serving-footprint number).

### File sizes (server-only; not copied locally)

- `sasrec_full.onnx`: 317.0 MB (item_embed [158489,256] + output_embed
  [158486,256] untied tables dominate; single file, no external-data
  companion was produced)
- `sasrec_query.onnx`: 161.6 MB (drops the untied output table -- exactly the
  point of the split option for a client/edge deployment)

## Files

- `code/g0_export/common.py` -- shared model builders, synthetic-batch
  generator (vocab/token conventions: PAD=0, MISSING=1, OOV=2, KNOWN>=3, from
  `snap_v1/reuse/vendor/rees46_eda/config.py` and `snap_v1/synthetic_store.py`),
  export/parity/latency helpers.
- `code/g0_export/run_gru.py`, `run_sasrec.py` -- one script per family;
  each writes its own `*_REPORT.json` here.
- `g0/export_smoke/GRU_REPORT.json`, `SASREC_REPORT.json` -- full strict-JSON
  records (export attempts incl. warnings/tracebacks, parity, latency, sizes).
- `g0/export_smoke/sasrec_full.onnx`, `sasrec_query.onnx` -- **server only**.

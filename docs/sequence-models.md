# Sequence models (`ppsi.seqrec`)

The deployment study compares the same next-item models trained in the cloud, with federated learning and on the
device. `ppsi.seqrec` holds those models: two families sized from one input layout, a deterministic builder, and the
adapters that plug them into the federated simulator (`ppsi.fedsim`).

## Families

| Family | Module | Topology | Head |
|---|---|---|---|
| GRU | `gru.ContextGRU` | per-channel embeddings + numeric/flag projection -> item-dim projection, LayerNorm, one-layer GRU (hidden 256), readout of the last real event, late-fused user context | tied: the item-embedding rows of the catalogue classes + output bias |
| SASRec | `sasrec.SASRecCE` | input LayerNorm over item + position embeddings and a gated side-feature residual, Pre-LN causal Transformer blocks, final LayerNorm, gated user-context residual | untied output table `[K, d]` + bias, recentered after every optimizer step |

Both score one decision per row (the next item after the window), never all positions, so no `[B, L, K]` tensor is
ever created. Class `j` is item token `j + 3` (tokens 0 / 1 / 2 are PAD / MISSING / OOV).

The SASRec recipe choices (input LayerNorm, no `sqrt(d)` input scale, an untied output table and per-step
recentering of its common mode) keep dense-softmax Adam training stable; the recentering is a per-query constant
logit shift, so it does not change the loss.

## Input layout

`features.InputWidths` declares the four float widths (numeric features, quality flags, user context, user-context
masks) and optionally their column names. Every model refuses a float block of another width, including offsetting
cases that a plain `Linear` would accept, and the adapters also check name arrays and the feature view.

## Sizes

`variants.py`: `GRU_D128` (default), `GRU_D256`, `SASREC_D256_B2` (default), `SASREC_D256_B3`, `SASREC_D384_B2` and
`SASREC_D64_B2`, the small on-device SASRec, parameter-matched to `GRU_D128` (about 20.7 M parameters at the REES46
catalogue, under 170 MB of FP32 traffic per federated visit).

## Building a model

```python
import torch
from ppsi.seqrec import InputWidths, build
from ppsi.seqrec.synthetic import synthetic_batch

widths = InputWidths(numeric_dim=15, flags_dim=15, user_context_dim=12, user_mask_dim=4)
bm = build("SASREC", widths, seed=2026, K=1000, variant="SASREC_D64_B2")
print(bm.init_sha256, bm.manifest.shared_numel)

batch = synthetic_batch(8, 1000, widths, torch.Generator().manual_seed(0))
bm.module.eval()
scores = bm.adapter.scores(batch)          # [8, 1000]
```

`build` is deterministic per seed and never touches the caller's RNG; the SASRec initial state is the recentered
init. `bm.adapter` is a `ppsi.fedsim` adapter, so the same object trains centrally, in a federated round or on one
device.

## Tests

```powershell
uv run python -m pytest tests/seqrec -q
```

Synthetic data only. Covered: causal attention (no gradient leaks into future events), padding / readout invariance,
the two SASRec implementations agreeing, the tied GRU head counted once, no `[B, L, K]` tensor in forward or backward,
width refusals, memorisation of random targets with the gradient and drift monitors, deterministic builds and both
families running through a federated round. Tests named `test_nc_*` are negative controls that must fail.

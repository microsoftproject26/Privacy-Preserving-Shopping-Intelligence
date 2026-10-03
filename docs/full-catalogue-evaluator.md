# Full-catalogue evaluator (`ppsi.evaluation.fullrank`)

The deployment study compares a cloud model, federated models and on-device models. Those numbers are only
comparable if every method is scored by the same code, on the same decisions, against the same candidates.
`ppsi.evaluation.fullrank` is that one evaluator. It depends only on numpy and torch.

## What it measures

* **Exact full-catalogue ranks.** Every item of the catalogue is a candidate; sampled negatives, candidate subsets
  and top-k shortcuts are refused. Rank rule: `rank = 1 + #{s_j > s_t} + #{j < t : s_j == s_t}` (the lower class id
  wins a tie; the target never benefits from one). The per-row tie diagnostic (`n_tied`, pessimistic ranks) is kept.
* **Two populations.** E2E = every evaluated decision, an out-of-vocabulary target scores 0; RANKABLE = the decisions
  whose target is in the catalogue. Coverage links them, and both identities
  (`micro E2E = coverage x micro RANKABLE`, `macro E2E = mean_u(cov_u x m_u)`) are re-checked on every result.
* **Metrics.** MRR, HR (= Recall, one target per decision) and NDCG at k in {1, 5, 10, 20}, micro and user-macro,
  accumulated in float64. An empty population or a zero denominator is `UNDEFINED`, never NaN or 0, and results
  serialise to strict JSON.
* **Paired user bootstrap.** One resample matrix per user set (1000 resamples, fixed seed, identified by its sha256)
  shared by every method, seed and contrast; ratios (retained quality) are bootstrapped directly; every registered
  contrast is reported whatever its sign.

## Module map

| Module | Role |
|---|---|
| `manifest.py` | validated evaluation manifest: no raw IDs or client index, ascending `decision_id`, mask / OOV / censoring consistency, ordered hashes, manifest-only strata |
| `ranking.py` | exact ranks from precomputed scores, or streamed from a query and the item table in score blocks |
| `config.py` | `EvalConfig`: the fixed scoring numerics (score block, chunk, batch, cache, deterministic flags) |
| `predict.py` | the prediction path for every model adapter (`query`, `head_weight`, `head_bias`) and for precomputed baseline scores; alignment of the artifact to the manifest |
| `metrics.py` | credits from ranks, micro / macro aggregation, coverage and the identity terms |
| `evaluate.py` | manifest + artifact -> full result (populations, OOV, censoring, ties, strata, identity checks) |
| `bootstrap.py`, `quality.py` | paired bootstrap, contrasts, retained / loss percentages |
| `rows.py` | result rows with the full comparison key; rows that differ in the key are historical references only |
| `report.py` | schema-checked lookups and the claim vocabulary (a missing bootstrap is `MISSING_EVIDENCE`) |

## Why the scoring numerics are fixed

A float32 GEMM is not bit-identical across block widths, so the score of an item can depend on how the item table is
split. The evaluator therefore scores each batch in ONE block of width `score_block >= K` (the model's logits bit for
bit), takes the target's score from the same block as its competitors, and refuses a nondeterministic kernel.
`EvalConfig` holds these numerics; its sha256 is recorded in every artifact, `evaluate()` refuses an artifact produced
under another config, and the score block, batch size, device class and stream form are part of every row's
comparison key.

## Example

```python
from ppsi.evaluation.fullrank.config import EvalConfig, apply_runtime_flags
from ppsi.evaluation.fullrank.evaluate import evaluate
from ppsi.evaluation.fullrank.manifest import build_eval_manifest
from ppsi.evaluation.fullrank.predict import predict_adapter

cfg = EvalConfig()
apply_runtime_flags(cfg)                       # strict FP32, deterministic algorithms
manifest = build_eval_manifest(header, columns)   # header must declare catalogue_K
artifact = predict_adapter(adapter, batches, manifest, model_sha256=model_digest, config=cfg)
result = evaluate(manifest, artifact, config=cfg)
print(result["E2E"]["macro"]["mrr@20"], result["coverage"])
```

`batches` are dicts with the model inputs plus `decision_id`, in manifest order, `cfg.eval_batch` rows each (the last
batch may be shorter); they may cover every manifest row or only the evaluated (non-censored) rows.

## Tests

```powershell
uv run python -m pytest tests/evaluation/fullrank -q
```

All fixtures are synthetic. Tests named `test_nc_*` are negative controls (strict expected failures): each runs a
property check against a deliberately wrong variant, such as an optimistic tie rule, float32 macro accumulation or an
unpaired bootstrap, and must fail.

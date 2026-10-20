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

## Reading a family of contrasts

`ppsi.evaluation.contrast_inference` turns the bootstrapped contrasts into the decisions of a preregistered
analysis. It lives outside the `fullrank` package, so the evaluator's code identity is unchanged.

| Function | What it gives |
|---|---|
| `seed_mean_resamples` | the per-resample seed mean of a delta or a retention ratio (seed s against seed s, same resample matrix) |
| `holm_step_down` | Holm over the registered family, on normal intervals from the bootstrap SE; adjusted p, the decision and a band that excludes the reference exactly when the contrast is rejected; Bonferroni band beside it |
| `equivalence` | two one-sided tests: equivalent iff the 90 % percentile interval lies inside the margin |
| `seed_stats` | seed spread, a seed-sensitivity flag and a user-plus-seed interval |
| `harm_shares` | the share of users harmed, helped and tied, with per-user delta quantiles |

How these were used in the deployment study:

* The primary pair was read unadjusted.
* The 18 secondary contrasts formed one Holm family of registered size: a contrast that could not be computed still
  counts.
* The phase-2 primary contrast was read with an equivalence margin of +-0.0005 MRR@20.

```python
from ppsi.evaluation import contrast_inference as ci
from ppsi.evaluation.fullrank import bootstrap as B

res = ci.seed_mean_resamples(plan, fl_by_seed, central_by_seed, "macro_e2e", kind="retained")
point = float(np.mean([B.ratio(plan, fl_by_seed[s], central_by_seed[s], "macro_e2e", "R")["point"]
                       for s in sorted(fl_by_seed)]))
holm = ci.holm_step_down([{"id": "R(FA / C)", "point": point, "resamples": res, "reference": 100.0}, ...],
                         family_size=18)
```

## Tests

```powershell
uv run python -m pytest tests/evaluation/fullrank tests/evaluation/test_contrast_inference.py -q
```

All fixtures are synthetic. Tests named `test_nc_*` are negative controls (strict expected failures): each runs a
property check against a deliberately wrong variant, such as an optimistic tie rule, float32 macro accumulation or an
unpaired bootstrap, and must fail.

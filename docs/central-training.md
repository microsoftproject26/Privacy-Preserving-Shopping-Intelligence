# Central training building blocks (`ppsi.central`)

The cloud arm of the deployment study trains the same models as the federated and on-device arms, and the
comparison is only fair if all of them use the same objective and the same notion of training time. `ppsi.central`
holds the pieces of central training that are shared with the federated simulator.

## Objective (`objective.py`)

* `ce_sum_count(logits, target_class, loss_mask)`: the full-catalogue cross-entropy of one `[B, K]` row per decision,
  returned as (sum over loss-eligible rows, their count); rows with `loss_mask = False` or `target_class < 0` get no
  gradient. Written without host syncs.
* `accumulate_step(scores_fn, batch, n_eff, micro, device)`: one effective batch split into microbatches, each
  backpropagating `sum_mb / n_eff`, so the accumulated gradient equals the full-batch gradient; a tail batch is
  normalised by its own count.

## Data order (`order.py`)

Pass `p` of seed `s` visits the decisions in `Generator(PCG64(SeedSequence([tag, s, p]))).permutation(N)`. The order
depends on (seed, pass) only, so both model families see identical data for a seed and a resume recomputes the
current pass exactly. Each permutation is hashed. The common `default_rng(seed + pass)` rule is avoided because it
makes seed 2026 pass 1 equal to seed 2027 pass 0 (a negative-control test shows it).

## Schedule and stopping rule (`schedule.py`)

Training time is counted in epochs of exposures (EFE = loss-contributing exposures / N), the same clock as
`ppsi.fedsim.schedule`, and the two agree bit for bit.

| EFE | Fraction of peak LR |
|---|---|
| 0 -> 0.02 | warm-up 0 -> 1 |
| 0.02 -> 2.0 | 1 |
| 2.0 -> 3.0 | 1 -> 0.1 |
| 3.0 -> 5.5 | 0.1 |
| 5.5 -> 6.0 | 0.1 -> 0.01 |
| 6.0 -> 8.0 | optional continuation for a still-improving run: 0.1, then 0.1 -> 0.01 |

The LR of an update uses the EFE at the END of the exposures it consumes, and `planned_lr_table` computes (and
hashes) the whole LR sequence before training. Evaluation every 0.5 EFE; no early stop before 6.0; a value is a new
best only if it beats the running best by more than 0.0002; a run is still improving at 6.0 if its best is at 5.5 or
6.0, and a continuation stops after two non-improving evaluations or at 8.0. An alternative warmup-stable-decay
schedule `S1` is available by name.

## Tests

```powershell
uv run python -m pytest tests/central -q
```

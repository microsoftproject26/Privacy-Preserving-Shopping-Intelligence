# Benchmark runner (`ppsi.benchrun`)

The deployment study's arms (cloud training, federated learning, on-device fine-tuning) are run on REES46. To compare
them with published sequential-recommendation results, `ppsi.benchrun` runs the same arms, with the same code, on
public benchmarks with leave-one-out splits (the `ppsi.benchmarks` layout:
`<dataset>/leave_one_out/{train.csv, holdout.csv}`).

## Protocol

* **Splits.** The inner validation row of each user is the reference trainer's (stable descending sort by time,
  rank 0, so timestamp ties pick the first row in file order); training decisions are the remaining events
  `t = 1 .. L-2`, one decision per event with a right-padded window of the last 50 events. The holdout is read only by
  the `test` command, after model selection.
* **Ranking.** Every catalogue item is a candidate except the user's seen items; ties go to the smaller item id; a
  seen or cold target is a miss that stays in the denominator. HR, NDCG at 10 / 20 and MRR@20, for all users, the
  pretraining band (15 % of users, seeded) and the others.
* **Model.** The ID-only SASRec of `ppsi.seqrec` at the benchmark's catalogue size, with the recentered init.

## Arms

| Arm | What runs |
|---|---|
| `C_FULL` | central training on every user's decisions (`ppsi.central` schedule, order and objective; 6 EFE, PB / CB checkpoints) |
| `PRE` | central pretraining on the band users only; its endpoint is the warm start of the warm federated arms |
| `FA`, `FA_WARM`, `FA_WARM_FROZEN` | federated training of the small model (`ppsi.fedsim`), cold, warm-started from PRE, or warm with frozen item tables |
| `FP*`, `PF*`, `FA_Q8*` | FedProx, personalised FedAvg, 8-bit uploads (each also `_WARM` / `_WARM_FROZEN`) |
| `S_CAL*`, `FA_1024*`, `DP8*` | the DP chain: clip-norm calibration on the unclipped trajectory, then Poisson-sampled training with clipping, without and with Gaussian noise |
| `FT_FA`, `FT_FA_WARM`, `FT_FA_WARM_FROZEN` | per-user on-device fine-tuning of a finished federated base, scored per user |

## Recipe

All hyperparameters come from a recipe: `recipe.DEFAULT_RECIPE` with **illustrative values (not tuned)**, and a
JSON file passed with `--recipe` is merged over it. Example:

```json
{
  "central": {"C_FULL": {"model_variant": "SASREC_D256_B2", "peak_lr": 0.001, "dropout": [0.2]}},
  "fl": {"server_opt": "fedadam", "server_lr": 0.02, "client_lr": 0.05, "local_passes": 1},
  "warm": {"FA_WARM": {"warm_lr_factor": 0.5, "warm_server_lr_factor": 0.5}},
  "ft": {"lr_frac": 0.1, "passes": 2},
  "dp": {"z": 1.0}
}
```

Federated rounds follow the exposure rule `ceil(endpoint_efe x N_clients / (group x (1 - dropout_p) x passes))`; set
the DP noise multiplier `z` from the accountant in `ppsi.fedsim` for your privacy target.

## Commands

```powershell
uv run python -m ppsi.benchrun.run describe --dataset s3_beauty --data-root data/bench
uv run python -m ppsi.benchrun.run run --dataset s3_beauty --data-root data/bench --runs-root runs --arm PRE --seed 2026
uv run python -m ppsi.benchrun.run run --dataset s3_beauty --data-root data/bench --runs-root runs --arm FA_WARM `
    --seed 2026 --pre-run runs/s3_beauty/PRE_s2026 --recipe recipe.json
uv run python -m ppsi.benchrun.run test --dataset s3_beauty --data-root data/bench --runs-root runs `
    --runs C_FULL_s2026 FA_WARM_s2026
```

Every run writes `ARM_SPEC.json`, `metrics.jsonl`, per-mark evaluation files, checkpoints and `RESULT.json`; a
finished run is never overwritten, `--resume` continues bitwise, and `--smoke` runs a short, clearly labelled
non-evidence version. `test` evaluates each finished run once (write-once files under `<run>/test/`).

## GPU device class

`--device cuda --gpu N` runs on one GPU with deterministic strict-FP32 numerics (`CUBLAS_WORKSPACE_CONFIG`, TF32 off,
deterministic algorithms with `warn_only` off). The data order, sampling, DP noise and model init are the same as on
the CPU, but floating-point reductions differ, so a GPU run is reproducible on its device class, not bit-equal to its
CPU twin. Every record carries the device class, and resumes, warm starts, S records and fine-tuning bases refuse a
different class.

## Tests

```powershell
uv run python -m pytest tests/benchrun -q
```

On a synthetic release: the data rules, the ranking against brute force, the recipe, central / federated / device
runs end to end (determinism, frozen tables, warm factors), the DP chain, the budget option, early stop, the wall
cap, Poisson inclusion, the device-class guards and the command line.

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

## The study's recipes

`config/benchrun/` holds the recipes used for the published results tables (`results/benchmarks/`):

| File | What it is |
|---|---|
| `registered_recipe.json` | the REES46 recipe transferred unchanged (the confirmatory runs, Track A) |
| `fixed_recipe_<dataset>.json` | the exploratory fixed recipe, one per dataset |
| `fixed_recipe_c_small.json` | the device-size central reference `C_SMALL`, run as the `C_FULL` arm with the d 64 model |

**The registered recipe.**

* Central models use the REES46 schedules: `C_FULL` is SASRec d 256 with 3 blocks, and `PRE` is SASRec d 64 with
  2 blocks.
* Federated training uses FedAdam (server step 0.01, `tau` 1e-3), client SGD with momentum (learning rate 0.1),
  one local pass of batch 16 and 6 epochs of exposure.
* Warm arms scale the client / server rates by 0.25 / 0.1 (`FA_WARM`) and 1 / 0.1 (`FA_WARM_FROZEN`).
* DP-FedAvg uses q = 2^-7, 384 rounds, delta 5e-6 and z 0.5642 (epsilon 8). On-device fine-tuning uses 0.3 x the
  small peak rate and 4 passes.

**The fixed recipe** keeps all of that except:

* FedAdam `tau = 1e-6` and a server step of 0.001 (see [`fedadam-tau.md`](fedadam-tau.md));
* 24 epochs of exposure for the federated arms;
* the warm start is PRE's best inner-validation checkpoint of a 96-epoch PRE run (`pre_start`);
* the warm client / server factors were chosen per dataset on inner validation.

Example for Beauty:

```powershell
$R = "config/benchrun/fixed_recipe_s3_beauty.json"
uv run python -m ppsi.benchrun.run run --dataset s3_beauty --data-root data/bench --runs-root runs_fixed `
    --arm PRE --seed 2026 --budget-efe 96 --dropout 0.5 --recipe $R
uv run python -m ppsi.benchrun.run run --dataset s3_beauty --data-root data/bench --runs-root runs_fixed `
    --arm FA --seed 2026 --recipe $R
uv run python -m ppsi.benchrun.run run --dataset s3_beauty --data-root data/bench --runs-root runs_fixed `
    --arm FA_WARM --seed 2026 --pre-run runs_fixed/s3_beauty/PRE__B96_L50_D0.5_s2026 --recipe $R
uv run python -m ppsi.benchrun.run run --dataset s3_beauty --data-root data/bench --runs-root runs_csmall `
    --arm C_FULL --seed 2026 --budget-efe 24 --recipe config/benchrun/fixed_recipe_c_small.json
```

PRE dropout by dataset:

* 0.5 on Beauty and Sports;
* 0.2 on Toys, MovieLens-1M, Video Games and Baby Products.

`pre_start` selects the PRE checkpoint a warm arm starts from. It is either `"endpoint"`, the default, or `"best"`.
With `match_cell` false, the PRE may be a longer-budget run with a different dropout; it must still use the same
maximum sequence length. At its default, `pre_start` does not enter the recipe hash.

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

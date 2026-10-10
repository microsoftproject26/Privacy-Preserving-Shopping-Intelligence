# FedAdam's tau and tiny federated updates

FedAdam (Reddi et al. 2021, "Adaptive Federated Optimization", Algorithm 2) is the server optimiser of the federated
arms. It turns the round's averaged client update `Delta` into a server step

```text
m     = beta1 m + (1 - beta1) Delta
v     = beta2 v + (1 - beta2) Delta^2
theta = theta + eta_s m / (sqrt(v) + tau)
```

`tau` is the adaptivity constant. In `ppsi.fedsim` it is `ServerOptConfig(..., tau=...)`; in the benchmark runner
it is `fl.server_tau` in the recipe. The default is `1e-3`, the value the project's federated recipes were
registered with, and it is kept so that earlier runs reproduce.

## What goes wrong when the updates are smaller than tau

Once the moments settle, `m ~ Delta` and `sqrt(v) ~ |Delta|`, so each coordinate moves by

```text
eta_s Delta / (|Delta| + tau)
```

* If `|Delta|` is well above `tau`, the step is about `eta_s` whatever the scale of `Delta`. This is the
  normalisation that makes Adam-style server optimisers attractive.
* If `|Delta|` is well below `tau`, the step is `eta_s Delta / tau`. That is plain server SGD with learning rate
  `eta_s / tau`: the normalisation is gone, and the step is proportional to an update that is already tiny.

With `tau = 1e-3` and updates of about `1e-7` the step is about `1e-4 x eta_s`, four orders of magnitude below what
the same `eta_s` would give with Adam-like normalisation. `tests/fedsim/test_fedsim_fedadam_tau.py` shows both regimes
on the optimiser itself.

## What we measured

The federated recipes here train a small SASRec with client SGD (momentum 0.9), one local pass of batch 16 and
FedAdam on the server. The updates are small, because each client holds a handful of events and the per-round mean
is taken over many clients.

* **REES46.** With `tau = 1e-3`, 99.3-100 % of the per-round update values were below `tau`. The median update
  was about `6e-8`, and the server step RMS was `5.6e-8`. The same run with `tau = 1e-6` had a step RMS of `1.2e-6`,
  about 20 times larger. Over 13,654 rounds, the warm-started federated model moved only about 0.06 % away from
  its server-pretrained starting point.
* **Public benchmarks.** About 100 % of the update values were below `tau`. Server steps ranged from about
  `1e-7` on the item table to `1.5e-5` on the Transformer blocks, and `tau = 1e-6` made them 70-100 times larger.
  For the cold FedAdam recipe (`eta_s = 0.01`, client learning rate 0.1), the step grew 93 times.

In other words, with `tau = 1e-3` the federated runs were barely training. Almost every cold federated learning curve on the
benchmarks was still at its best at the last checkpoint of its budget.

## Effect of the fix

* **Benchmarks.** We set `tau = 1e-6`, used a larger round budget and left the item tables trainable. The best
  federated arm then reached 76-91 % of the central model's validation NDCG@10 across the six datasets. With the
  original recipe, the federated arms had reached 13-81 % on the test split. The exploratory test results are in
  `results/benchmarks/`.
* **REES46 with a strong server pretrain.** Here a larger server step did not help. Starting from a model
  pretrained on 1.2 million users, larger steps lowered the calibration-split MRR@20 (0.1705 to 0.1663 at
  `eta_s = 3e-5`). The conclusion that federated training adds nothing over that pretrain is therefore not caused
  by the `tau` setting.

## Recommendation

* Use `tau = 1e-6` for this setting: small per-client datasets, client SGD and per-round mean updates of
  `1e-8`-`1e-6`. More generally, keep `tau` at or below the typical update size, and tune it together with `eta_s`,
  because with tiny updates the two only act through their ratio.
* Log the fraction of update values below `tau` for the first rounds. If it is close to 100 %, FedAdam is
  acting as server SGD with step `eta_s / tau`, and its learning curves will look flat.
* Keep the default `1e-3` only to reproduce the earlier runs.

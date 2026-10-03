# Federated simulator (`ppsi.fedsim`)

`ppsi.fedsim` simulates cross-device federated training of a next-item recommender in one Python process. It is the
training engine behind the federated and on-device arms of the deployment study (cloud vs. federated vs. on-device):
every browser profile is one simulated client, the server only ever receives model updates, and the bytes a real
deployment would move are counted even though no tensor leaves the process.

The package is model-agnostic. A model plugs in through a small adapter (`ppsi/fedsim/adapter.py`) that exposes the
query vector, the item head and a parameter-role manifest; `ppsi/fedsim/synthetic.py` contains a tiny model that the
tests use.

## What it covers

| Deployment | Methods in the simulator | Modules |
|---|---|---|
| Federated | FedAvg (`FA`), FedProx (`FP`), FedAvg with a personal on-device query vector (`PF`) | `client`, `aggregate`, `server`, `personal` |
| Federated, adaptive server | FedAdam and FedAvgM server optimisers | `server` |
| Federated with differential privacy | DP-FedAvg (flat L2 clipping, Gaussian noise on the sum, fixed denominator), RDP accounting for Poisson and fixed-size sampling | `dp`, `accountant`, `accountant_wor`, `participation` |
| Cheaper uploads | 8-bit stochastic quantisation (`FA_Q8`), row-sparse item uploads, frozen item tables | `quant`, `sparse_rows`, `runtime.freeze_item_tables` |
| On-device | local-only training from a shared start (`LO`), full-model fine-tuning of a cloud or federated model | `local_only`, `finetune` |

Supporting pieces: seeded client sampling, Poisson participation and client drop-out (`participation`), the learning
rate schedule over effective full-data epochs (`schedule`), the resumable run loop with atomic checkpoints (`runtime`,
`checkpoint`), per-round update-norm monitoring (`update_log`), virtual byte accounting (`comm`) and a process-pool
round driver (`pool`).

## Design rules

- **Deterministic.** Strict FP32, deterministic algorithms, and seeds derived with BLAKE2b from explicit parts
  (run seed, round, client key). A retried client visit, a resumed run and a run with a different number of workers
  produce the same bits.
- **Fixed reduction order.** Clients are aggregated in a fixed logical order, cut into contiguous shards that are
  summed in shard-index order, so the result never depends on which worker finishes first.
- **Nothing private leaves the client.** The personal vector and its optimizer state are never part of an upload;
  the aggregator refuses any upload whose keys are not exactly the shared parameters.
- **Planned before it runs.** The per-round learning rates are planned and hashed before the first round, and a run
  refuses to start if the plan does not end exactly at the endpoint.
- **Counted, not measured.** Bytes are theoretical payload sizes (FP32, tied tensors once); network time is not
  simulated.

## Quick start

```python
from ppsi.fedsim.client import LocalSolver
from ppsi.fedsim.numerics import enforce_strict_fp32, state_digest
from ppsi.fedsim.participation import ParticipationPlan
from ppsi.fedsim.runtime import FLRun, RunConfig
from ppsi.fedsim.server import Server
from ppsi.fedsim.synthetic import make_clients, make_tiny_adapter

enforce_strict_fp32()
clients = {c.key: c for c in make_clients(32, K=24, L=6, seed=0)}
n_decisions = sum(int((c.examples["target_class"] >= 0).sum()) for c in clients.values())

server_model = make_tiny_adapter(K=24, d=8, seed=1)
server = Server(server_model, init_sha256=state_digest(server_model.broadcast_state()))
plan = ParticipationPlan.build(list(clients), seed=2026, manifest_hash="demo", group_size=8)
cfg = RunConfig("demo-fedavg", "FA", seed=2026, lr_peak_local=0.05,
                solver=LocalSolver(lr=0.05, passes=2), exposure_point="end")

run = FLRun(cfg, plan, server, clients.__getitem__, n_decisions,
            workers=[make_tiny_adapter(K=24, d=8, seed=2)])
run.run()
print(run.summary()["rounds"], run.ledger.snapshot())   # rounds, (messages, bytes down, bytes up, retry bytes)
```

Run it from the repository root (`uv run python your_script.py`).

## Privacy accounting

The noise multiplier for a target epsilon comes from the RDP accountant of the Poisson-subsampled Gaussian:

```powershell
uv run python -m ppsi.fedsim.accountant --m 1024 --N 131072 --T 384 --delta 5e-6 --eps 8 1
```

`python -m ppsi.fedsim.accountant_wor` reports the same budget under fixed-size sampling without replacement
(Wang, Balle & Kasiviswanathan 2019) as a sensitivity analysis.

## Tests

```powershell
uv run python -m pytest tests/fedsim -q
```

The suite uses synthetic clients only and runs on a CPU in about a minute and a half. Tests named `test_nc_*` are
negative controls: each runs a property check against a deliberately wrong variant and is expected to fail (strict
`xfail`). The `test_fedsim_crosscheck_*` files re-derive the core rules (FedAvg weights, FedProx gradient, byte counts,
retries) from hand-computed toy values, independently of the main tests.

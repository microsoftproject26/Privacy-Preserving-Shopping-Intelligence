# Privacy-Preserving-Shopping-Intelligence-via-Federated-On-Device-Learning
_______________________________________________________________________________

## Phase 1 data pipeline

The reproducible REES46 October 2019 CSV-to-Parquet pipeline is documented in
[`docs/s1-ds-02-csv-to-parquet.md`](docs/s1-ds-02-csv-to-parquet.md).

Run its fast synthetic validation with:

```powershell
python scripts/data/convert_rees46.py self-test
```

Raw CSV, ZIP, and generated Parquet artifacts are intentionally excluded from
normal Git history. The repository contains their versioned manifests, hashes,
aggregate conversion report, and reproduction code.



## Phase 1 research protocol — status: `G1 = GO`

The data foundation every later model stands on is complete and frozen. It answers one question
before any modelling: **is there anything for a model to gain here?**

### The gate

| Criterion | Evidence |
|---|---|
| Cohort large enough to simulate a federation | **388,789** clients against a gate of 10,000 |
| Every task leaves room above a model-free rule | smallest room **0.144** |
| Every task can resolve differences far smaller than that room | weakest task **128 steps** |
| A sequence model beats the model-free rule on real data | macro gain **+0.0204** |
| That advantage survives on the thinnest clients | smallest bucket **+0.0184** |

Full verdict and the five conditions that travel with every number:
[`docs/evidence/s1-ds-09/g1_gate_v1.frozen.json`](docs/evidence/s1-ds-09/g1_gate_v1.frozen.json)

### Tasks

| Task | Notebook | Documentation |
|---|---|---|
| `S1-DS-03` canonical event | [notebook](notebooks/S1_DS_03_Canonical_Event_EN.ipynb) | [en](docs/s1-ds-03-canonical-event.md) · [ar](docs/ar/s1-ds-03-canonical-event.md) |
| `S1-DS-04` data audit | [notebook](notebooks/S1_DS_04_Data_Audit.ipynb) | [en](docs/s1-ds-04-data-audit.md) · [ar](docs/ar/s1-ds-04-data-audit.md) |
| `S1-DS-05/06` cohort + temporal split | [notebook](notebooks/S1_DS_05_06_Cohort_Temporal_Protocol.ipynb) | [en](docs/s1-ds-05-06-cohort-temporal-protocol.md) · [ar](docs/ar/s1-ds-05-06-cohort-temporal-protocol.md) |
| `S1-ALL-D1` + `S1-DS-07` protocol + task examples | [notebook](notebooks/S1_D1_DS_07_T3_Protocol_and_Task_Examples.ipynb) | [en](docs/s1-d1-ds-07-t3-protocol-and-task-examples.md) · [ar](docs/ar/s1-d1-ds-07-t3-protocol-and-task-examples.md) |
| `S1-DS-09` freeze + `G1` gate | [notebook](notebooks/S1_DS_09_Freeze_and_G1.ipynb) | [en](docs/s1-ds-09-freeze-and-g1.md) · [ar](docs/ar/s1-ds-09-freeze-and-g1.md) |
| `S2-SMOKE` first GRU on real data | [notebook](notebooks/S2_SMOKE_GRU_On_Real_Data.ipynb) | [en](docs/s2-smoke-gru-on-real-data.md) · [ar](docs/ar/s2-smoke-gru-on-real-data.md) |

### Decisions

`ADR-001` fixes how every comparison in this project is averaged and which slice `T1` is compared
on. All three decisions were written **before the numbers they affect existed** — see
[`docs/decisions/ADR-001-evaluation-protocol.md`](docs/decisions/ADR-001-evaluation-protocol.md).

### Getting the data

Task examples and cohort manifests carry `user_id`, so they are **not in this repository** — this
repo is public and the project is about privacy-preserving learning.

**Start here:** [`docs/data-access.md`](docs/data-access.md) — what to download for your task,
where to put it, and how to verify the checksums.

Committed here instead are the artifacts with no user-level data: the protocol, the vocabulary,
the item catalogue, the candidate lists, the price transform, and the gate verdict.

### What comes next

[`docs/ar/roadmap.md`](docs/ar/roadmap.md) — the remaining tasks per lane, the models to be tried
beyond the GRU, and how the final comparison table is meant to read.

## Training / Federated Development Environment

The canonical Phase 1 training environment is defined by:

- `.python-version`
- `pyproject.toml`
- `uv.lock`

First-time setup:

```powershell
uv python install 3.11.14
uv sync --locked --group dev
```

Validate:

```powershell
uv run --locked python scripts/env_smoke.py
```

Expected final line:

```text
SMOKE STATUS: PASS
```

Full setup, VS Code, CPU/GPU, and compatibility documentation:

[`docs/training-environment.md`](docs/training-environment.md)

## Local CI checks

Pull requests and pushes to `main` run one CI workflow. Run the same checks locally with:

```powershell
uv sync --locked --group dev
uv run --locked ruff check ppsi scripts tests
uv run --locked python -m pytest -q
uv run --locked python scripts/env_smoke.py
uv run --locked python -X utf8 scripts/experiments/validate_experiment_contracts.py
```

The workflow uses synthetic and committed fixtures only. It does not download REES46 or require a
GPU.

### Adding or changing a dependency

CI installs with `uv sync --locked`, which refuses to resolve anything `uv.lock` does not already
pin. Editing `pyproject.toml` without regenerating the lock therefore fails the very first CI step,
before lint or tests run, with:

```text
The lockfile at `uv.lock` needs to be updated, but `--locked` was provided.
```

After any change to `pyproject.toml`:

```powershell
uv lock
git add pyproject.toml uv.lock
```

Commit both together. A lock that arrives in a later commit leaves every commit in between
uninstallable.

### When CI fails

| First failing step | What it means | What to run |
|---|---|---|
| Install locked environment | `uv.lock` does not match `pyproject.toml` | `uv lock`, then commit `uv.lock` |
| Lint | Ruff findings; most are mechanical | `uv run --locked ruff check ppsi scripts tests --fix` |
| Test | A real test failure | `uv run --locked python -m pytest -q` |
| Environment smoke | The pinned interpreter or Torch is not usable | `uv python install`, then re-sync |
| Contract smoke | A frozen schema or identity no longer validates | Read the named contract before changing it |

The steps run in order and stop at the first failure, so a later step being untouched does not mean
it would pass.

Ruff enforces the pinned Python version, `3.11.14`. Syntax introduced in a later Python parses on a
newer local interpreter and fails here, so run the checks above before pushing rather than relying
on whichever Python is on your PATH.

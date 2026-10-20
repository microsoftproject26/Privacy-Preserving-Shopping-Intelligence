# Phase 1 Environment Readiness

Status: PARTIALLY READY
Captured: 2026-08-31
Issue: #3

The current machine can run the locked Python environment, small Polars work, and the existing
Flower/Ray smoke on CPU. Phase 1 is not fully ready until the other two machine profiles, shared
storage owner, and a GPU path are confirmed.

## Current decisions

- Ahmed's machine is suitable for development, tests, and small CPU simulations.
- Do not use drive C for project caches or data; its free space is critically low.
- Drive D can support development and small artifacts, but should not be declared the long-term
  raw-data and Parquet store with only about 30 GiB free.
- Native Windows Flower/Ray works, but WSL2 is the first fallback if Windows-specific Ray problems
  recur.
- Ahmed's machine has no NVIDIA/CUDA path. The team must name a teammate GPU or an approved
  Kaggle/cloud fallback.
- Keep artifact storage provisional until #23 makes the small final decision.

## Machine profiles

| Member / profile | OS and CPU | RAM | GPU / CUDA | Project Python | Workspace disk | WSL / cloud | Status |
|---|---|---:|---|---|---|---|---|
| Ahmed Sherif / ahmed-windows | Windows 11 Pro build 26200; Ryzen 7 PRO 4750U; 8 cores / 16 threads | 15.33 GiB; 1.84 GiB free at capture | AMD integrated; no NVIDIA; CUDA unavailable | 3.11.14 via uv lock | D: 30.16 GiB free; C: 3.62 GiB free | Ubuntu 24.04 on WSL2; HTTPS to GitHub/PyPI/Hugging Face works | CPU READY |
| Eid / pending | Pending teammate inventory | Pending | Pending | Pending | Pending | Pending | PENDING |
| Ahmed Abdelhamed / pending | Pending teammate inventory | Pending | Pending | Pending | Pending | Pending | PENDING |

System Python is 3.14.2 and is not the project runtime. Use the locked Python 3.11.14 environment.
Docker CLI is installed but the Docker daemon was not running during this audit.

## Runtime validation

| Probe | Command | Result | Time / peak process-tree memory |
|---|---|---|---|
| Locked environment and PyTorch CPU | .venv\Scripts\python.exe scripts\env_smoke.py | PASS; Python 3.11.14, Torch 2.13.0+cpu, CUDA false | 77.907 s / 273.34 MiB |
| Polars Parquet scan | Polars lazy scan of fixtures/smoke_sample.parquet | PASS; 23,109 rows, 12 columns | 2.500 s / 47.06 MiB |
| Flower/Ray synthetic smoke | .venv\Scripts\python.exe scripts\federated\fl_synthetic_smoke.py --config config\fl_synthetic_smoke.v1.json --output tmp\environment-audit\fl_synthetic_smoke_summary.v1.json | PASS; 2 clients, 3 rounds, 2 repetitions; deterministic | 299.323 s / 4,493.57 MiB |

The locked environment contains Flower 1.33.0, Ray 2.55.1, Polars 1.43.2, PyArrow 21.0.0,
scikit-learn 1.9.0, and LightGBM 4.7.0.

The Flower smoke reduced global loss from 1.9444400072 to 1.1246404648. Ray created more actors
than the two logical clients, so this successful smoke is not evidence that large-client simulation
will fit comfortably on this machine.

## Storage plan

| Data or artifact | Provisional location | Owner | Current decision |
|---|---|---|---|
| Raw REES46 CSV | Team-selected data drive outside Git | Pending | File is not present on Ahmed's machine |
| Processed Parquet | Ignored artifacts/full-run/ on the selected processing machine | Pending | Do not assign Ahmed's D: as long-term owner yet |
| Checkpoints and run results | Ignored local artifacts/ during development | Producing member | Shared location and retention are decided in #23 |
| Small schemas, configs, and aggregate evidence | Repository paths owned by their Issues | Git | Commit only reviewable, non-sensitive files |

The source manifest expects a raw file of about 5.67 GB. Conversion staging, processed Parquet,
checkpoints, caches, and repeated runs require substantially more headroom than the raw file alone.

## Fallback triggers

- Use WSL2 when native Windows Ray fails repeatedly or leaves stale processes.
- Move full Parquet processing to the teammate with the most RAM and at least 60 GiB practical free
  workspace after both profiles arrive.
- Use a teammate CUDA machine when available.
- If no teammate has a usable CUDA GPU, activate a team-approved Kaggle/cloud path before neural
  model training; record account owner, limits, and artifact download location.
- Reduce client count or use the stronger team machine if Flower/Ray approaches 70% of available
  RAM.

## Required teammate input

Each teammate should post the following in Issue #3:

1. OS/build, CPU cores/threads, total RAM, GPU/VRAM/driver, and CUDA availability.
2. Free space on the drive they would use for data and artifacts.
3. Python/uv status and the final line from uv run --locked python scripts/env_smoke.py.
4. Whether they can run the Flower/Ray smoke.
5. Whether they can provide a local GPU or approved Kaggle/cloud path.

After those two rows arrive, the team can name the Parquet machine, Flower/Ray machine, storage
owner, and GPU path and then close #3.

# S1-PR-01 — Reproducible Training Environment

## Purpose

This document defines the canonical Phase 1 Python environment for model
development, federated-learning simulation, baselines, testing, and CI.

The goal is to ensure that developers and CI resolve the same Python runtime
and dependency graph rather than relying on machine-specific or globally
installed packages.

## Canonical Runtime

| Component | Canonical value |
|---|---|
| Python | `3.11.14` |
| Environment manager | `uv` |
| Dependency specification | `pyproject.toml` |
| Exact dependency lock | `uv.lock` |
| Local virtual environment | `.venv/` |
| Canonical compute backend | CPU |
| GPU requirement | None |

The repository pins Python through `.python-version`.

The project must not depend on packages installed in Conda `base`, the system
Python installation, or other undocumented global environments.

---

## Dependency Contract

### Runtime / research dependencies

| Dependency | Version | Role |
|---|---:|---|
| PyTorch | `2.13.0+cpu` | Tensor operations and neural-model training |
| Flower | `1.33.0` | Federated-learning orchestration |
| Ray | `2.55.1` | Flower simulation execution |
| Polars | `1.43.2` | Large-scale tabular and Parquet processing |
| PyArrow | `21.0.0` | Arrow/Parquet interoperability |
| psutil | `7.2.2` | Process, CPU, and memory diagnostics |
| scikit-learn | `1.9.0` | Classical ML baselines, utilities, and metrics |
| LightGBM | `4.7.0` | Gradient-boosted tree baselines |

### Development dependencies

| Dependency | Version | Role |
|---|---:|---|
| pytest | `9.1.1` | Automated tests |
| Ruff | `0.16.3` | Linting and formatting |

Transitive dependencies are resolved exactly by `uv.lock`.
Developers must not edit `uv.lock` manually.

---

## First-Time Setup

### 1. Install uv

On Windows:

```powershell
winget install --id astral-sh.uv -e
```

Confirm:

```powershell
uv --version
```

### 2. Install the canonical Python runtime

```powershell
uv python install 3.11.14
```

### 3. Create/synchronize the project environment

From the repository root:

```powershell
uv sync --locked --group dev
```

`uv` creates `.venv/` automatically.

If the uv cache and repository are intentionally located on different Windows
drives, hard links may not be available. In that case this equivalent local
setup command may be used:

```powershell
uv sync --locked --group dev --link-mode=copy
```

The link mode changes only how cached files are materialized; it does not
change the locked dependency graph.

---

## Daily Developer Workflow

After pulling changes:

```powershell
git pull
uv sync --locked --group dev
```

Then validate the environment:

```powershell
uv run --locked python scripts/env_smoke.py
```

Developers should not run ad-hoc commands such as:

```text
pip install <package>
conda install <package>
```

inside the project environment.

A new direct dependency must be added deliberately through `pyproject.toml`
and the resulting `uv.lock` change must be included and validated in the
corresponding PR.

---

## VS Code

Select the repository virtual environment as the Python interpreter:

```text
Ctrl+Shift+P
→ Python: Select Interpreter
→ .venv\Scripts\python.exe
```

On Windows the interpreter path is:

```text
<repo>\.venv\Scripts\python.exe
```

After selection, VS Code's **Run Python File** button uses the project
environment.

The IDE configuration is developer convenience only. Reproducibility and CI
must use the canonical command-line entry points documented below.

---

## Canonical Setup Entry Point

```powershell
uv sync --locked --group dev
```

This command is the shared dependency setup contract for developers and CI.

---

## Canonical Smoke Entry Point

```powershell
uv run --locked python scripts/env_smoke.py
```

The smoke validates:

- exact supported Python runtime;
- presence of `uv.lock`;
- core package imports;
- core package versions;
- OS and architecture identity;
- Python executable identity;
- `uv.lock` SHA-256;
- PyTorch runtime identity;
- safe CUDA detection;
- physical NVIDIA hardware detection when available;
- CPU fallback;
- a deterministic tiny tensor operation.

A successful invocation ends with:

```text
SMOKE STATUS: PASS
```

A failed invariant returns a non-zero process exit code.

This same smoke command is the contract consumed by CI.

---

## CPU and GPU Policy

The canonical Phase 1 dependency environment intentionally uses the CPU build
of PyTorch.

This allows one dependency contract to execute on:

- development machines with NVIDIA GPUs;
- development machines without GPUs;
- CPU-only CI runners.

Therefore an NVIDIA GPU is **not** required for setup or smoke validation.

On a machine that physically contains an NVIDIA GPU, the smoke may report both:

```text
nvidia_hardware: <GPU name>
torch_cuda_available: False
selected_device: cpu
```

This is expected when using the canonical CPU PyTorch build.

GPU acceleration may later be introduced as a separately documented execution
profile if training workloads require it. It must not silently replace the
canonical CPU reproducibility baseline.

---

## Flower and Ray

Flower provides the federated-learning orchestration layer.

Ray provides the local distributed runtime used by Flower simulation.

The environment pins:

```text
Flower 1.33.0
Ray    2.55.1
```

Simulation-heavy work on native Windows may have platform-specific limitations.
If such problems are encountered in later federated tasks, WSL2/Linux may be
used for execution while preserving the same Python/dependency contract where
supported.

---

## Data-Pipeline Compatibility Boundary

The previously merged S1-DS-02 CSV-to-Parquet pipeline has its own minimal
runtime specification in `requirements.txt`.

Its recorded successful full-run environment used:

```text
Python  3.13.7
Polars  1.43.2
PyArrow 21.0.0
```

S1-PR-01 standardizes the **training/federated environment** on Python
`3.11.14` because it must satisfy the Flower/Ray runtime contract.

The S1-DS-02 self-test currently does not complete successfully under the
canonical Python 3.11.14 training environment.

This limitation is explicitly documented rather than silently changing the
already-merged data-conversion implementation inside S1-PR-01.

The frozen ProcessedRawParquet artifact produced by S1-DS-02 is unaffected.

If the team later requires one Python runtime to reproduce both raw conversion
and federated training, that compatibility work should be handled as an
explicit follow-up task/decision rather than an undocumented change to either
contract.

---

## Lint / Formatting Validation

Environment-owned Python code is checked with:

```powershell
uv run --locked ruff check scripts/env_smoke.py
uv run --locked ruff format --check scripts/env_smoke.py
```

---

## Clean-Environment Validation

To validate reproducibility from an empty local environment:

```powershell
Remove-Item -Recurse -Force .venv
uv sync --locked --group dev
uv run --locked python scripts/env_smoke.py
```

A successful clean reconstruction followed by `SMOKE STATUS: PASS` is the
primary local reproducibility oracle.

---

## CI Contract

CI must reuse the developer setup and smoke commands:

```powershell
uv sync --locked --group dev
uv run --locked python scripts/env_smoke.py
```

CI must not maintain a separately resolved dependency set.

This contract is handed to S1-SE-02 for initial CI integration.

---

## Dependency Changes

After this environment is consumed by downstream tasks:

- supported Python must not be changed casually;
- core dependencies must not be removed casually;
- `uv.lock` must not be edited manually;
- dependency changes must update both `pyproject.toml` and `uv.lock`;
- relevant smoke/tests must pass before merge;
- contract-changing runtime decisions require team review.

---

## Troubleshooting

### `uv` is not found

Confirm installation:

```powershell
winget list --id astral-sh.uv
```

Then restart the shell and verify:

```powershell
where.exe uv
uv --version
```

### `.venv` uses the wrong Python version

Remove it and recreate from the canonical project definition:

```powershell
Remove-Item -Recurse -Force .venv
uv sync --locked --group dev
```

Then:

```powershell
uv run --locked python --version
```

Expected:

```text
Python 3.11.14
```

### CUDA is unavailable

This is not an environment failure.

The canonical PyTorch distribution is CPU-only and the smoke must select CPU
safely.

### Windows cross-drive hard-link warning

If uv's cache is on `C:` and the repository is on another drive, Windows cannot
hard-link across filesystems.

Use:

```powershell
uv sync --locked --group dev --link-mode=copy
```

No dependency identity is changed.

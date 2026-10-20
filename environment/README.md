# environment - Python environments, path variables and data

## Python and uv

The repository pins Python 3.11.14 (`.python-version`) and manages the CPU environment with
`uv` from the committed `uv.lock`. Both resolve every shared package to the same version as the
GPU set below, so CPU-only work (tests, checks, `--from-results`) sees the same code.

```bash
uv sync --locked --group dev     # CPU environment from uv.lock (Python 3.11.14)
```

`requirements.txt` at the repository root is the same CPU set as a plain pip view of the lock.

## CPU lock vs GPU set

Two environments exist because the study trained on GPUs and is verified on CPUs:

| Environment | File | Use |
|---|---|---|
| CPU lock | `uv.lock` (root), `requirements.txt` | Everything that runs without a GPU: `tools/check.py`, prototype tests, `run_pipeline.py --from-results` |
| GPU set | `requirements-gpu.txt` | The NVIDIA A40 server set (CUDA 13.0, `torch 2.13.0+cu130`, Linux x86_64), installed with `uv pip install -r environment/requirements-gpu.txt` |

`ENV_LOCK_A40.txt` records the server venv's driver, CUDA and Torch build, and
`ENV_PKGS_A40.txt` its full package list; `requirements-gpu.txt` pins to those. Reference
copies of both A40 records are kept with the deployment manifests in
[`../07_study_protocol/deployment_manifests/`](../07_study_protocol/README.md).

## The `PPSI_*` path variables

All filesystem roots come from
[`ppsi_paths`](../00_common/code/ppsi_paths/__init__.py); each root can be overridden with an
environment variable, and with no variable set every root keeps its legacy server value, so the
code behaves exactly as before on the original server. A default expressed as
`<SERVER_ROOT>/...` follows the root that names it, so setting `PPSI_SERVER_HOME` alone moves
every server root.

| Variable | Root it controls |
|---|---|
| `PPSI_REPO_ROOT` | This repository (computed from the module location by default) |
| `PPSI_SERVER_HOME` | The JupyterHub server home |
| `PPSI_SERVER_BASE` | The owned physical copy of the delivery-review workspace |
| `PPSI_DELIVERY_REVIEW_ROOT` | The delivery-review root on the server |
| `PPSI_EXTENSIONS_ROOT` | The extensions tree of the closure phase on the server |
| `PPSI_USER_HOME` | The user home on the server |
| `PPSI_WORKSPACE_ROOT` | The project workspace on the server |
| `PPSI_PILOT_ROOT` | Pilot-study workspaces (the earlier GRU and SASRec campaigns) |
| `PPSI_PROJECT_ROOT` | The original flat research workspace of the main study |
| `PPSI_LEGACY_CODE_ROOT` | The old flat `code/` tree of that workspace |
| `PPSI_RUNS_ROOT` | Training-run and checkpoint trees |
| `PPSI_SHARED_DATA_ROOT` | Shared read-only data on the server |
| `PPSI_DATA_ROOT` | The frozen dataset (default `<shared data>/rees46_nextitem_v1`, read-only) |
| `PPSI_VENVS_ROOT` | Server virtual environments |
| `PPSI_RESULTS_ROOT` | Where stage results live (default: this repository; results under `<stage>/results/`) |

The module also maps legacy absolute server paths onto the configured roots (`legacy()`), and
locates files of the old flat research layout and of the prototype layout in the stage folders
(`locate()`, `prototype_path()`), which is how the hash-pinned research code finds its files
without edits.

## Data download and verification

The REES46 data is not redistributed. Download the source months from Kaggle, dataset
"eCommerce behavior data from multi category store" (REES46):
<https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store>.

The study uses October and November 2019 (the raw months hold 42,448,764 and 67,501,979 rows).
The raw-file identity record with the checksum is
[`../docs/prototype/data-source.md`](../docs/prototype/data-source.md); for the October file
`2019-Oct.csv` (5,668,612,855 bytes, 42,448,764 rows):

```text
FEDD938409B5F836EC89B39C861B13DAD99FC7CD9BEB1FDDD97A2D50488B5B80  2019-Oct.csv
```

Verify with `sha256sum -c` (or `certutil -hashfile 2019-Oct.csv SHA256` on Windows) before
building anything. All experiments then read the frozen derived dataset `rees46_nextitem_v1`
(409 files, 8.6 GiB, version hash
`860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd`), built by the
[`01_data`](../01_data/README.md) stage and expected at `PPSI_DATA_ROOT`. The dataset is
immutable: any change of preprocessing needs a new version, and the study's results are tied
to this one.

## Hardware

Main-study training and scoring ran on the Queen's University Lobot JupyterHub cluster,
2x NVIDIA A40, 128 CPU cores; the prototype backbone contest ran on 2x NVIDIA A16. No GPU is
needed for the CPU lock, the checks, or `--from-results`.

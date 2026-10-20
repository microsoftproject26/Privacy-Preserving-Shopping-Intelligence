# 00_common - shared code

## Purpose

Shared code used by every stage: the root of the `ppsi` package, the `ppsi_paths` module that
defines every filesystem root, the server launchers of the main study and the follow-up
analysis, and the shared test suites. Nothing here belongs to one experiment; the stage folders
import from it.

## Experiments

This stage runs no experiments of its own. It carries the shared measurement and leakage
conventions that the experiments rely on (measurement helpers in `ppsi` and the temporal-leakage
rules, both pinned by the prototype tests).

## Code map

| Package / file | What it does | Origin |
|---|---|---|
| `code/ppsi` | Root of the `ppsi` package: measurement helpers (`measurements.py`) and temporal-leakage rules (`temporal.py`) shared by all stages | Prototype stage |
| `code/ppsi_paths` | One place for every filesystem root the research code uses; each root overridable with a `PPSI_*` environment variable; maps legacy server paths and the prototype layout onto this repository | Restructure of the main study |
| `code/ppsi_env.py` | Environment bootstrap helpers shared by launchers and checks | Main study |
| `code/v4int` -> `code/study_launchers` (import name `v4int`, as used in the recorded runs) | Server launchers of the main study's research tree: central launches (`launch_central.py`), federated launches (`launch_fl.py`), device launches (`launch_device.py`), cloud references (`c_full.py`, `c_full_small.py`), pretraining (`pretrain_small.py`), re-evaluation (`reeval.py`), selection probes (`lr_screen.py`, `p2_probe.py`, `p2_select.py`) | Main study |
| `code/v4sup` -> `code/run_supervisor` (import name `v4sup`) | Companion launcher package of the follow-up analysis and closure re-runs (same style as `v4int`) | Follow-up analysis / closure phase |
| `code/experiment_extensions` -> `code/closure_experiments` (import name `experiment_extensions`) | Shared helpers for the per-stage `experiment_extensions` packages | Closure phase |
| `code/scripts` | Shared script entry points | Main study |
| `tests/prototype/` | Prototype tests of the shared code (`test_measurements.py`, `test_temporal_leakage.py`); run by `uv run python -m pytest` and `tools/check.py` | Prototype stage |
| `tests/research/` | Research test suites of the launchers (`tests_study_launchers`, formerly `tests_int`; `tests_run_supervisor`, formerly `tests_sup`); they need the server layout and the frozen dataset | Main study |

## How to run

This stage has no `run.py`; it is a library stage. The prototype tests run anywhere:

```bash
uv run python -m pytest 00_common/tests/prototype -q
```

The research suites under `tests/research/` expect the original server workspace (frozen dataset
and run trees) and are read-only verifications of the launchers; run them only with
`PPSI_*` roots pointed at a real server copy.

`ppsi_paths` is import-only, uses the standard library alone and does no filesystem work at
import time; `python -c "from ppsi_paths import describe; print(describe())"` prints every
configured root.

## results/ and figures/

None; this stage produces no results or figures. Curated outputs of the stages that use this
code live in their own `results/` folders.

## Known limitations

The research launchers (`study_launchers`, `run_supervisor`; import names `v4int`, `v4sup`) are
records of how the runs were started on the
original server; they hard-code nothing about paths (everything goes through `ppsi_paths`), but
they still expect the frozen dataset, the GPU environment and the run trees that exist only on
that server. Re-running a main-study training unit is done from the original flat code tree
recorded in [`../07_study_protocol/`](../07_study_protocol/README.md), not from this folder.

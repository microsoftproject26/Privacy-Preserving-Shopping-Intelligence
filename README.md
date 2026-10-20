# Privacy-Preserving Shopping Intelligence via Federated and On-Device Learning

This repository holds the code and the curated records of an MSc graduation project (Queen's
University, School of Computing; Digilians programme, 2026). The project measures how much
shopping-recommendation quality survives when a cloud-trained sequential recommender is replaced
by a federated or an on-device model. All main-study experiments rank the full catalogue of
158,486 products of the REES46 e-commerce log (October and November 2019) with SASRec
next-product models, score with client-macro MRR@20, and express quality as the retention of a
cloud reference; 131,072 users of the log simulate devices. The study was run under a
pre-registered, gated protocol with a single sealed test evaluation, and every number in this
repository carries the evidence label it was given in the final report.

## Research question

*"How much shopping recommendation quality can be retained when moving from cloud-based to
on-device or federated models?"*

## Headline answer

- **Confirmatory** (REES46 TEST, 26,254 users, 229,742 decisions; the single sealed test
  evaluation of 5-7 October 2026): the realistic edge system - a warm-start federated model with
  frozen item tables - keeps **88.49% [87.73, 89.18]** of the full-data cloud model; the
  server-pretrained model alone keeps 88.77% [88.03, 89.47]; the cold federated model keeps
  70.10% [69.19, 70.94].
- **Confirmatory**: federated training on top of the pretrained model changes MRR@20 by
  -0.000508 [-0.000933, -0.000037] (federation adds nothing over a strong server pretrain);
  on-device fine-tuning adds +0.002283 [0.001588, 0.003076].
- **Confirmatory (follow-up analysis)**: the tuned warm-start federated model is equivalent to
  the pretrained model within +/-0.0005 (Q1 -0.000081 [-0.000179, 0.000024], read as EQUIVALENT).
- **Exploratory**, after the test evaluation: the registered FedAdam stabilizer tau = 1e-3 was
  far larger than the observed update norms (~1e-7), so federated training barely moved (~0.06%
  in 13,654 rounds); with tau fixed, warm federated models reach 75-92% of the cloud model on
  six public benchmarks; the value of federation depends on how much data the server already has
  (federated training adds +0.00911 over a 100,000-user pretrain and nothing measurable over the
  1,217,610-user pretrain).

## Stage map

| Folder | Contents |
|---|---|
| [`00_common/`](00_common/README.md) | Shared code: root `ppsi` package, `ppsi_paths` filesystem roots, server launchers, shared tests |
| [`01_data/`](01_data/README.md) | REES46 data pipeline: census, frozen dataset builds, feature statistics, feature-search data |
| [`02_centralized/`](02_centralized/README.md) | Central training: pilot-study record, backbone contest, cloud references, server pretraining |
| [`03_federated_learning/`](03_federated_learning/README.md) | Federated simulator, federated recipe tuning stages, federated arms, tau-fixed re-runs |
| [`04_on_device/`](04_on_device/README.md) | ONNX/INT8 export, communication byte ledger, browser latency bench, on-device arms |
| [`05_privacy_and_security/`](05_privacy_and_security/README.md) | DP mechanisms, membership-inference and update-leakage audits, secure-aggregation simulation |
| [`06_evaluation_and_benchmarks/`](06_evaluation_and_benchmarks/README.md) | Full-catalogue evaluator, contrast statistics, six public benchmarks, frozen code records |
| [`07_study_protocol/`](07_study_protocol/README.md) | Pre-registration records, deployment manifests, handover, delivery verification (never edited) |
| [`environment/`](environment/README.md) | CPU lock and GPU requirements, `PPSI_*` path variables, data download and verification |
| [`docs/prototype/`](docs/prototype) | Documentation of the prototype stage (August-September 2026) |
| [`report/`](report) | Final report PDF and slides |
| `tools/`, `config/`, `fixtures/` | One-command check (`tools/check.py`), configuration, small test fixtures |

Each stage folder contains `code/`, `tests/prototype/`, `tests/research/`, `scripts/`,
`results/` (curated summaries; raw run trees live under `results/runs/main_study/` and
`results/runs/closure_phase/`) and `figures/`.

## Quick start

```bash
uv sync --locked --group dev                      # CPU environment from uv.lock (Python 3.11.14)
uv pip install -r environment/requirements-gpu.txt   # GPU set used on the NVIDIA A40 server (torch 2.13.0+cu130)
uv run python tools/check.py                      # one-command check: lock, package map, ruff, prototype tests
uv run python -m pytest -n auto                   # prototype tests on all cores
uv run python run_pipeline.py --from-results      # rebuild every stage's tables and figures from saved results (no GPU, no data)
uv run python run_pipeline.py --mode auto         # per training unit: skip if complete, resume from checkpoint if supported, else start
uv run python run_pipeline.py --force             # retrain from scratch (old outputs are moved aside first)
uv run python 03_federated_learning/run.py --from-results   # the same per stage (01..06 each have run.py)
```

All server paths come from `ppsi_paths` and can be overridden with environment variables
`PPSI_PROJECT_ROOT`, `PPSI_DATA_ROOT`, `PPSI_RUNS_ROOT`, `PPSI_RESULTS_ROOT`,
`PPSI_PILOT_ROOT` (see [`00_common/code/ppsi_paths/__init__.py`](00_common/code/ppsi_paths/__init__.py)
for the full list and defaults, and [`environment/README.md`](environment/README.md) for a
summary).

Research code of the main study is hash-pinned: it is kept byte-identical, so re-running an old
study is done from the original flat code tree recorded in
[`07_study_protocol/`](07_study_protocol/README.md); `--from-results` and the prototype tests
work from the stage layout.

## Data access

The REES46 data is not redistributed. Download it from Kaggle ("eCommerce behavior data from
multi category store", REES46) and verify the checksum in
[`docs/prototype/data-source.md`](docs/prototype/data-source.md): the raw October 2019 file
`2019-Oct.csv` is 5,668,612,855 bytes with 42,448,764 rows and SHA-256
`FEDD938409B5F836EC89B39C861B13DAD99FC7CD9BEB1FDDD97A2D50488B5B80`. All experiments read the
frozen derived dataset `rees46_nextitem_v1` (409 files, 8.6 GiB, version hash
`860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd`), which is built by the
[`01_data/`](01_data/README.md) stage and points `PPSI_DATA_ROOT` at its location.

## Checkpoints and run records

Checkpoints and full run records (73 GB) are published as GitHub Release `artifacts-v1.0` in
self-contained tar parts named `<stage>.partNN.tar` (each below 1.9 GB), plus `MANIFEST.tsv`
(path, bytes, sha256, part) and `SHA256SUMS`. Verify a download against `SHA256SUMS`, then
extract a part straight into a clone:

```bash
sha256sum -c SHA256SUMS
tar -xf 03_federated_learning.part01.tar -C <clone>
```

## Hardware

All main-study training and scoring ran on the Queen's University Lobot JupyterHub cluster,
2x NVIDIA A40, 128 CPU cores. Browser latency was measured on a laptop, a phone and a server
CPU.

## Repository conventions

- **Evidence labels.** Every reported number carries one of: *confirmatory* (main analysis;
  registered before the value existed and scored once on TEST in the single test evaluation),
  *confirmatory (follow-up)* (the same for the registered follow-up study, scored in the same
  test evaluation), *validation* (measured on the validation week, including the backbone
  contest and the single registered validation read; values at the best-of-12-marks rule are
  optimistic), *exploratory* (anything added after those reads, including every registered
  post-test look), *descriptive* (no claim: tuning reads, baselines, panels, external
  benchmarks). A label records when the governing rule was registered relative to when the
  value was read; results never move between labels.
- **Frozen records.** Everything under [`07_study_protocol/`](07_study_protocol/README.md) is a
  record and is never edited: the pre-registration and its amendments, locks and gates, the
  hash-chained TEST exposure register, the decision log extracts, the deployment manifests of
  the frozen server code trees, and the delivery-verification evidence.
- **Hash pins.** Research code of the main study is kept byte-identical; corrections go into a
  new tree, and each run records the tree it ran from.
- **No redrawing of results.** Curated summaries under `<stage>/results/` are derived from the
  saved result files of the sealed runs; if a summary and a primary file disagree, the primary
  file wins.

## Report

The [`report/`](report) folder holds the final report PDF and the defence slides of the project.

## How to cite

See [`CITATION.cff`](CITATION.cff) (title, group author, repository URL).

## License

MIT License; see [`LICENSE`](LICENSE).

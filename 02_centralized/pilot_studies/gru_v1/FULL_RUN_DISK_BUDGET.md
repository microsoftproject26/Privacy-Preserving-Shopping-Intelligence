# FULL_RUN_DISK_BUDGET.md — full-data GRU run

Measured 2026-09-22T22:21:56Z on `jupyter-group-digi2026-g8`, before any full-run artifact existed.

## Measured state

| Item | Bytes | GiB |
|---|---|---|
| Volume `/home/jovyan` size | 52,521,566,208 | 48.91 |
| Used | 26,572,636,160 | 24.75 |
| **Free** | **25,932,152,832** | **24.15** |
| Frozen dataset `shared_data/rees46_nextitem_v1` (read-only, never written) | 9,155,162,711 | 8.53 |
| `.local_ai/gru_v1` total | 8,553,684,386 | 7.97 |
| — `cache/` (pilot per-decision window cache: 1M TRAIN + 112,365 VS) | ≈6.4 GiB | 6.4 |
| — `experiments/` (pilot + resume-test checkpoints, 250 MB each) | ≈1.7 GiB | 1.7 |
| Shared GPU venv `~/venvs` (never touched) | ≈5.7 GiB | 5.7 |

Owner guidance: treat free space as ≤ 30 GiB, possibly ~26 GiB. Measured free is 24.15 GiB, so the budget below uses **24 GiB** as the starting point.

## Policy

- Target: keep **≥ 10 GiB free** at all times.
- Emergency floor: **8 GiB free**. Any writer whose projected post-write free space is below 8 GiB must not write. The trainer checks live `statvfs` before every checkpoint write, and the supervisor checks every 60 s. Either one stops the run cleanly with `latest.pt` intact.
- Teammates share the volume, so the guard uses **live** free space, not this budget.

## Full-data representation: zero new disk

The full TRAIN/VS input path is a **compact per-event store built in process RAM** from the frozen Parquet at process start:
- one row per event, stored once, in compact dtypes;
- a decision → (global end row, window length) index;
- windows gathered per batch.

Nothing is written to disk. The forbidden per-decision window cache is **not** scaled up. The pilot cache stores ~5.9 KB per decision, so 40.18M decisions would need ~237 GB.

| Representation | Where | Size |
|---|---|---|
| F4 compact event store, TRAIN+VALIDATION_SELECT rows (63.69M rows × ~156 B) | RAM (F4 process) | ≈ 9.9 GB RAM, 0 disk |
| F0 compact store (item tokens + decision index) | RAM (F0 process) | ≈ 0.6 GB RAM, 0 disk |
| Frozen dataset | read-only, sequential Parquet reads | 0 new bytes |

The pod memory limit is 64 GiB. At measurement, only ~1.0 GiB was anonymous memory; ~22 GiB was reclaimable page cache.

## Planned new disk writes

| Artifact | Count | Est. size |
|---|---|---|
| Checkpoints F0-h256 (`best`, `latest`, `previous`) | 3 × 251 MB | 0.75 GB |
| Checkpoints F4-h256 | 3 × 253 MB | 0.76 GB |
| Checkpoints F0-h512 (optional challenger) | 3 × ~270 MB | 0.81 GB |
| Transient `.tmp` during an atomic checkpoint write (per run) | 1 × ~270 MB | 0.27 GB × 3 runs |
| Metrics/logs (aggregated every 100 updates, epoch evals) | 3 runs | ≤ 0.1 GB |
| Baselines (POP counts, TRANS stats; no dense matrices, no prediction dumps) | — | ≤ 0.05 GB |
| Final evaluation outputs (aggregates, per-user MRR vectors for bootstrap) | — | ≤ 0.2 GB |
| Code, reports, agent reports | — | ≤ 0.05 GB |
| **Peak planned new bytes** | | **≤ 3.6 GB** |

Projected minimum free space: 24.15 − 3.6 ≈ **20.5 GiB**, well above both the 10 GiB target and the 8 GiB floor.

## Reclaimable reserve (not needed; recorded in case a teammate fills the disk)

These are regenerable payloads only, in priority order. Each would be inventoried and path-resolved before deletion:
1. `.local_ai/gru_v1/experiments/gru/test_resume_*/checkpoints/last.pt` (3 × 251 MB, resume-test scratch; results kept in their JSON).
2. `.local_ai/gru_v1/cache/train/*.npy` (≈5.4 GiB, regenerable by `window_cache.py`). It is kept for now because it is the byte-exact parity reference for the new store.

Never deleted: the frozen dataset, pilot `best.pt` files, any JSON/MD evidence, `artifacts/`.

## Forbidden (restated)

No full per-decision window cache. No copy of the frozen dataset. No separate F0/F4 backing caches on disk. No full prediction matrices. No unlimited checkpoints. No raw user IDs in logs.

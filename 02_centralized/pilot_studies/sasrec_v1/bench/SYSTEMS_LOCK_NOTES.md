# Systems lock notes (2026-09-24T00:10Z): Gate C condition 4

Real-data benchmark: concurrent pair, TF32 training, strict-FP32 eval, 1000 measured updates each; states discarded (`bench/SYSTEMS_LOCK.json`).

| arm | decisions/s | median step | data wait | process RSS | GPU peak alloc / reserved | store build | eval rows/s (strict FP32) |
|---|---|---|---|---|---|---|---|
| A0_ID256 (GPU0) | 3,459 | 73.9 ms | 0.06% | 6.75 GiB | 1.35 / 1.55 GiB | 65 s | 6,167 |
| A1_RICH256 (GPU1) | 3,370 | 75.8 ms | 0.08% | 16.92 GiB | 1.36 / 1.56 GiB | 129 s | 5,909 |

- **Context length (median step, A0/A1):** len-1 batches 47/49 ms; len-10 58/61 ms; len-50 108/112 ms.
- **RAM:** the pair RSS is 23.7 GiB of the 64 GiB cgroup, leaving ≈ 40 GiB of headroom (rule: ≥ 12 GiB). The cgroup memory.current peak of 48.7 GiB includes ≈ 22 GiB of reclaimable file page cache.
- **Microbatch:** 256, the registry default. 128×2 was not benchmarked on real data. The synthetic trend shows larger GEMMs more efficient on the A16 (B256 < B512 < B1024 in per-decision time), so 128×2 can only be slower; this is a systems choice frozen before training.
- **Concurrency:** both arms run concurrently on separate GPUs, as registered. Data wait is below 0.1%, so the GPU is the bottleneck.
- **Projection:**
  - A0 ≈ 21.0 h and A1 ≈ 21.6 h (6 epochs, 4 full-VS evals, cohort diagnostics);
  - training end ≈ 2026-09-24T21:53Z, report ≈ 23:53Z;
  - deadline 2026-09-25T12:20Z, so slack ≈ 12.4 h.
- **Disk:** 19.97 GiB free at launch.

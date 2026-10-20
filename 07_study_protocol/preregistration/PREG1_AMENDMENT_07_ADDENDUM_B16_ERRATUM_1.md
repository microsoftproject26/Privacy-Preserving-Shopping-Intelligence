# PREG1 amendment 07 — addendum §B16, ERRATUM 1 (Ops, 2026-10-06; before the ML-1M LOCK; no ML-1M TEST value exists)

- **Basis:** LOCK_FINAL_RUNBOOK revision (DL 470) found an internal inconsistency in the registered B16 text; DL 497 found a
  process-entry workaround that the ML-1M LOCK must disclose. Registered B16 (sha 85ba5e18…) is unchanged; this erratum is the
  reading that applies.
- **Status (MEASURED by Ops):** no ML-1M inner-validation or TEST value has been read; the ML-1M LOCK has not been written.

**E1. B16.4 code tree name.** B16.4 says "The ML-1M LOCK pins code_bench5 including `final`". B16.1 defines the ML-1M tree as
**code_bench5b** = code_bench5 + the BM-05 delta (`final --device`, accel fix), and B16.4's own first bullet requires
`final --device cuda`, which exists only in code_bench5b. Every ML-1M run was trained from code_bench5b
(run_ml1m_gpu.sh, CODE = code_bench5b). **Reading: the ML-1M LOCK pins code_bench5b including `final`.** code_bench5 stays
untouched and is never locked.

**E2. Process entry form of the ML-1M GPU jobs (disclosure; no code change).** The FT arms (`ft.ml_1m.2026 / 2027`, FT_FA_WARM)
start a multiprocessing "spawn" pool. Spawn re-imports the main module `v4bench.run`, whose import-time line sets
`CUDA_VISIBLE_DEVICES=""` (the CPU default), so every worker saw 0 GPUs and the jobs looped on BenchRefused (DL 497). From
2026-10-05T15:28Z the driver runs with `PY = company/plans/ops_scripts/bench/py_bm_spawnsafe.sh` (cde51e01…). It runs the
same module, with the same arguments, through `runpy.run_module("v4bench.run", run_name="__main__", alter_sys=False)`
under `-c`, so the workers do not re-import it. The code tree (code_bench5b) and every argument are unchanged; only the
process entry form differs. The same wrapper is used for `final --device cuda` on ML-1M if the FT arms are scored there.
The jobs that ran before 15:28Z (55 of 73) used the plain `-m v4bench.run` form; the identical-hash resume rule applies.

**E3. Driver placement.** For GPU sharing with the T0a sitting, the ML-1M driver was restarted with `--gpus "0"`
(DL 496 / 497) and later `--gpus "1"` (DL 507), with `--resume` (identical-hash rule). Device class NVIDIA_A40-FP32-DET is the same on both
GPUs.

Concurrence: to be confirmed at the ML-1M LOCK review (Fable), together with the LOCK itself.

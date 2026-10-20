# PREG1 amendment 07 — addendum §B16, ERRATUM 2 (Ops, 2026-10-06; after the ML-1M LOCK, before the ML-1M TEST look)

- **Basis:** Fable LOCK review 2026-10-06 (CONCUR_WITH_CONDITIONS): ERRATUM 1 E3 predates DL 519 and DL 522, and runbook §2.1
  wrongly says the driver builds the third-seed commands. Registered B16 (85ba5e18…) and ERRATUM 1 are unchanged; this adds to them.
- **Status (MEASURED by Ops):** no ML-1M inner-validation or TEST value has been read. The ML-1M LOCK exists
  (runs_bench_gpu/ml_1m/LOCK.json, sha ca7478e55c0c94f9ec3d032264a874398a509fced72d5068669cd638201a5153, DL 523).

**E4. Driver relaunch with a non-yielding YIELD_RE (DL 519).** At 21:02:57Z on 2026-10-05 the driver's default YIELD_RE matched
the sitting's `H_test.sh`; it stopped launching and exited at 23:07Z (64 / 73 ok, 0 fail). Ops relaunched
`run_ml1m_gpu.sh --resume --gpus "1" --slots-per-gpu 4` with the spawn-safe PY, `CUDA_VISIBLE_DEVICES=0,1` (the job's `--gpu 1`
selects the device) and a YIELD_RE that matches no process. So the B16.5 process-presence yield was replaced, for the last 9
training jobs, by watchdog v6b rule 1a, which SIGSTOPs ML-1M jobs on any GPU that runs a Phase-2 job. These are training jobs, not
a look. Same tree, same arguments, identical-hash resume rule.

**E5. Third seed launched by hand (DL 522).** SEED3_DECISION lists `C_FULL`. The DAG's `seed3` job (run_ml1m_gpu.sh:163) only
writes the decision; no 2028 job exists in the DAG. Runbook §2.1's sentence "the GPU driver builds the same commands with those
flags" is therefore wrong. Reading: the third-seed runs are launched by Ops, by hand, with the driver's exact command form.
C_FULL_s2028 ran from code_bench5b as `run --device cuda --gpu 1 --runs-root $V/runs_bench_gpu --dataset ml_1m --arm C_FULL
--seed 2028`, with the spawn-safe PY, CUBLAS_WORKSPACE_CONFIG=:4096:8 and OMP / MKL 16. It reached DONE_MAIN before the LOCK.

**E6. The wrapper is deposited.** company/plans/ops_scripts/bench/py_bm_spawnsafe.sh is now in the local tree. Its sha256 is
cde51e010a5882f346b4491d9a5da9a9e055df3853ee0c54f24b34fd5f4fd778, which equals the server file.

**E7. The ML-1M look (final), placement and form.**
- `final` runs only when no H step and no Phase-2 job can run on its GPU: after SITTING COMPLETE (the frozen sitting lane's step 8
  uses GPUs 0 and 1), and after the ML-1M driver has finished. The driver log ends "ML-1M GPU jobs finished; failed or skipped:
  0", so watchdog `ml1m_done` is true and no relaunch can happen.
- Form: the plain `-m v4bench.run` entry, because `--workers 1` runs FT in-process and no pool spawns. The command is
  `final --dataset ml_1m --runs-root $V/runs_bench_gpu --device cuda --gpu G --workers 1 --threads 16` from code_bench5b, with an
  absolute runs root.
- A running `final` is never killed.

Concurrence: Fable conditions of 2026-10-06; Ops records compliance in the DL.

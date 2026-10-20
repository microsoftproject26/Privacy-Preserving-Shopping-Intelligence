# BM-05 — `final` on the dataset's one device class (`--device {cpu,cuda} --gpu N`)

- **Authority:** company/reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md §11 E1: one device class per dataset for every
  computation, training and the TEST look alike. For ML-1M, `final` runs the scoring AND the FT arms' per-user fine-tune
  with `--device cuda` under NVIDIA_A40-FP32-DET. S3 ×3 keep `final` on CPU (code_bench4, unchanged).
- **Follows:** BM-03 (company/handoffs/BM-03.md, deployed as `$V/code_bench5`, deploy_archive/bench5/NEW_FILES.sha256).
- **Worker:** Opus, working locally only. There was no server access, no git and no Codex. No benchmark RESULT, LOCK, TEST
  file or metric value was opened. Every run used synthetic fixtures under `_pytest_tmp/`, which were deleted afterwards.
- **Status:** code and local CPU tests are done.
- **PENDING:**
  - the GPU two-run `final` test on the A40 (§5 b);
  - the Sonnet review;
  - the S3 decision to keep `final` on code_bench4 is unchanged and needs nothing from this pack.

## 1. Archive (made first) and delta pack

**Archive.** The pre-BM-05 files were copied to `code/_archive_bench5_pre_bm05/` (with SHA256SUMS.txt):
- `v4bench/{accel,final,run}.py`;
- `tests_bench/{test_bm03_device,bm03_digest_job}.py`.

Each copy equals the deployed code_bench5 file (`cmp` against `deploy_archive/bench5/bench5_files.tar`). accel.py was
restored from that tar because BM-05 also edits it (see §2).

**Delta pack.** These are the files that changed relative to the deployed code_bench5. Copy them over `$V/code_bench5`
(or into a new `code_bench5b`; see §6 Q1).

| file (code/) | deployed sha256 (NEW_FILES.sha256) | BM-05 sha256 | +/− |
|---|---|---|---|
| v4bench/accel.py | 73be7fb59720b239063f18c13191ed468e3ffa1ca305904c77a508c03d13b2a0 | 6fd1090a790517dedc2582722604e8db839cdcb133b3e61fbacf682230bc1c33 | 16/1 |
| v4bench/final.py | f00af2542250a32558875c17ce81901031ebac9ac5dff82b0fe35e1ef6ed7a5e | 5306d260e0861ab211ebdccd08b6069a239716632de1dcc9d36fdeaf8d015aa4 | 64/17 |
| v4bench/run.py | caeaa30d7ac2c97ff17a43a6379780970b8a693fae1682fb8c83f0e78068f634 | 0c8f3e6280a7692492847c06f5f8bcd7393fa5aabca9b9da492b53dba32c56fb | 9/6 |
| tests_bench/test_bm03_device.py | 70dc4693c9c0f19f2025b50a3f262cbc95420abd6be8e026bbeb9041c5994acd | 78376859a208990e09947d072eef28ea8e78ffb96e63fce232a58257bcfc4b4f | 3/2 |
| tests_bench/test_bm05_final.py | NEW | 21cab5c43254aee6cd97febf8cf65c7de6b30bc394dfb06e3534953a3e23031c | new |
| tests_bench/bm05_final_job.py | NEW | 599f61cbcf4cbf062e71e9d6fe50e6ff8b1e9838f341e120243c5f56ad8ce9bc | new |

The following are unchanged versus the deployed code_bench5: central, fl_engine, device, model, evaluate, selection, common,
data, metrics, recipe, `__init__`, `tests_bench/bm03_digest_job.py`, and every r6 module (`tree_digest` is unchanged).

## 2. What changed

- **final.py**
  - `final_eval(..., device=None)` resolves the device with `accel.torch_device`. This enforces and verifies the strict
    deterministic numerics on cuda.
  - It then calls the new **`check_lock_device`**, BEFORE the attempt is logged in TEST_ATTEMPTS.jsonl and before the gate or
    holdout is touched:
    - it refuses if the LOCK's `device_class` (a LOCK without the field, e.g. a bench4 LOCK, is CPU-FP32) differs from the
      look's class;
    - it refuses if any locked run's RESULT is of another class.
  - `_final_eval` passes the device to `score_run`:
    - plain arms: the model is built on the GPU and the checkpoint loaded into it; ranking stays on CPU, as in BM-03's
      `rank_chunk`;
    - FT arms: `device.ft_ranks(device="cuda")`, so the per-user fine-tune runs on the GPU in every spawned worker.
  - `accel.assert_numerics` runs before every preflight and TEST scoring call (an NC guard: TF32 / non-strict numerics are
    refused).
  - On cuda only, the TEST outputs carry the class:
    - TEST_RESULTS.json gets `device_class` and the `device` record;
    - TEST_LOOK.json, each `test/TEST_<rule>.json` and the attempt `start` row get `device_class`.
  - CPU outputs are unchanged byte for byte. The CPU class stays recorded in the LOCK (`device_class: CPU-FP32`), which
    TEST_LOOK pins through `lock_self_sha256`.
  - `verify_lock`, `open_test_gate` and the look order (preflight → holdout → commit_look → write) are unchanged.
- **run.py**
  - `--device {cpu,cuda} --gpu N` is now defined for `run` AND `final`. The BM-03 env setup (CUDA_VISIBLE_DEVICES,
    CUBLAS_WORKSPACE_CONFIG before torch is imported) applies to both.
  - `final` passes `device` only for cuda.
- **accel.py (hardening, found while testing BM-05)**
  - `is_cuda` is now true only for `"cuda[:i]"` or a cuda `torch.device`.
  - `torch_device` refuses unknown values instead of silently falling back to CPU.
  - Cause: in `final.score_run` the name `device` is also the imported FT module (`from . import device`). The BM-03
    `is_cuda` treated any non-"cpu" object as cuda, so my first draft passed the module as a device on the CPU path.
  - The CPU byte-identity test caught this (it crashed). The parameter is now named `dev`, and
    `test_is_cuda_is_strict_and_unknown_devices_refused` guards it.
  - BM-03 call sites only pass None, "cpu", "cuda" or `torch.device`, so they are unaffected. The BM-03 tests and its
    byte-identity guard still pass.
- **test_bm03_device.py**: one assertion changed. `final --device cuda` used to be an argparse error; it is now
  "refused: needs --gpu" (rc 2, env untouched).

## 3. CPU byte identity of `final` (MEASURED, local CPU)

`test_final_cpu_byte_identical_to_bench4` works as follows:
1. `bm05_final_job.py --stage prepare` trains PRE, FA, PF and FT_FA (all users) with this tree on CPU and writes a non-strict
   LOCK (`device_class: CPU-FP32`).
2. The tree is copied twice, byte for byte.
3. bench4's `final` (code/_archive_bench4, or `$BM_BENCH4_ROOT`, with no device argument) runs on copy A. This tree's `final`
   with `device="cpu"` runs on copy B.
4. Every file is compared: the 8 TEST ranks `.npy` byte for byte, plus `test/TEST_*.json`, TEST_RESULTS.json, TEST_LOOK.json
   and TEST_ATTEMPTS.jsonl, with only `utc` / timing and the runs root removed.

**Result: identical**, and no CPU TEST output carries `device_class`. The four arm types cover the plain, PF (personal
matrix) and FT (per-user fine-tune) scoring paths.

## 4. Tests (MEASURED)

**test_bm05_final.py** (7 tests):
- the CPU byte identity above;
- `is_cuda` strictness / unknown-device refusal;
- **NC GPU LOCK + cpu (or default) → refused**, with no attempt row and no TEST_LOOK;
- **NC CPU LOCK + cuda → refused**, also for a bench4-style LOCK without the field;
- **NC a locked run of another class → refused**;
- **CLI:** `--gpu` without cuda, cuda without `--gpu`, the default and `--device cpu` on a GPU LOCK → rc 2 with the
  device-class message, no look taken;
- **GPU two-run `final`** (skipped unless `BM_GPU_TEST`):
  - prepare on the GPU class, then `final --device cuda` on two copies;
  - every TEST digest must be identical;
  - TEST_RESULTS, TEST_LOOK, TEST_*.json and the attempt row must carry the class;
  - plus an NC: the GPU LOCK under cpu is refused.

**Results:**
- test_bm05_final.py + test_bm03_device.py: **19 passed / 2 skipped** (the two GPU tests).
- Full tests_bench: see the final line below. One file was ignored, `test_ml1m_gpu_script.py`. It is the BM-04 driver's
  test, new and still being edited by another worker during this run (modified 19:52). It failed 2–4 cases on its own driver
  guard ("a run_wave1.sh covering ml_1m is running", yield / lock tests). It does not import v4bench `final`, so the failure
  is not caused by BM-05; it is left to BM-04.

  Full suite (excluding test_ml1m_gpu_script.py): **109 passed / 24 skipped** in 530 s (BM-03 state was 103 / 23; +6 passed / +1 skipped = test_bm05_final.py; the skips are the two GPU tests and the server-only ones).

## 5. Server commands

Run these after the delta pack is applied.

    V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
    PY=/home/jovyan/venvs/ppsi-gpu/bin/python
    cd $V/code_bench5       # (or code_bench5b, §6 Q1)
    sha256sum v4bench/accel.py v4bench/final.py v4bench/run.py tests_bench/test_bm03_device.py \
              tests_bench/test_bm05_final.py tests_bench/bm05_final_job.py     # must equal §1

**(a) CPU suite + the `final` byte identity against the real code_bench4.** The GPU tests are skipped here.

    CUDA_VISIBLE_DEVICES= BM_BENCH4_ROOT=$V/code_bench4 nice -n 15 $PY -B -m pytest -q -p no:cacheprovider tests_bench \
      --basetemp $V/_pytest_tmp/bm05_cpu -rs

**(b) GPU two-run `final` test** (tiny synthetic fixture). It must not share a GPU with an H step (§11 E3). Expect 2 passed:

    BM_GPU_TEST=1 nice -n 15 $PY -B -m pytest -q -p no:cacheprovider tests_bench/test_bm05_final.py \
      tests_bench/test_bm03_device.py -k "gpu" --basetemp $V/_pytest_tmp/bm05_gpu -rs

**(c) ML-1M production TEST look.** This comes after the ML-1M GPU LOCK, written by code_bench5 with BM-05, in
`runs_bench_gpu`. The BM-04 driver stops before lock / final, so Ops runs it by hand, exactly once:

    nice -n 15 $PY -B -m v4bench.run final --dataset ml_1m --runs-root $V/runs_bench_gpu --device cuda --gpu 1 \
      --workers 1 --threads 16

- Without `--device cuda` it is REFUSED: "the LOCK's device class is NVIDIA_A40-FP32-DET, this look runs on CPU-FP32". No
  attempt is logged.
- `--workers W` spawns W FT processes on the one visible GPU.
- **S3 ×3:** use code_bench4 `final` as registered (or code_bench5 with the default cpu: same bytes, see §3).

## 6. Open questions

1. **Pack name.** A40 smoke evidence (DL 409 / 412) was measured on the deployed code_bench5. Overwriting it in place changes
   accel / final / run, though the `run` path is the same. The alternative is a fresh `code_bench5b` (= code_bench5 + this
   delta). The ML-1M LOCK must pin whichever tree runs both training and `final` (§11 E1: "the LOCK pins code_bench5 incl.
   final"). I recommend applying the delta BEFORE the first production GPU run, so one tree does both.
2. **BM-04 interaction.** `test_ml1m_gpu_script.py` asserts `--device cuda` only on `run` lines, and the driver stops before
   `final`. That is consistent with this pack, but the LOCK/FINAL runbook for ML-1M must now carry `--device cuda --gpu G`
   (§5 c).
3. **Retro check.** `final` on CPU of a bench4-written CPU LOCK is accepted (absent field = CPU-FP32). A bench4 LOCK can never
   be finalised with cuda. Confirm this is the intended S3 behaviour if any S3 look is ever run under code_bench5.

# REV-BM-03 — independent review of code_bench5 (opt-in deterministic GPU class)

Reviewer: Sonnet (read-only; no server, no benchmark values opened, no Codex).
Spec: FABLE_GATE_ENDGAME_SCRIPTS_20261003 section 8 condition 2. Handoff: company/handoffs/BM-03.md.
Diff: code/_archive_bench4/v4bench vs code/v4bench (all 9 changed files read in full as diffs; accel.py read whole).

## VERDICT: PASS_WITH_MINORS

No blocker found. The code does what the spec asks. Real GPU proof is still PENDING (the GPU two-run test and the
timing gate run on the A40 only); this review cannot replace them. Nothing here is a measured GPU result.

Local run (MEASURED, CPU, `pytest tests_bench/test_bm03_device.py --basetemp F:/tmp_pytest_rev`): 13 passed, 1 skipped
(the GPU test), 71 s. This includes `test_cpu_path_byte_identical_to_bench4` against code/_archive_bench4, so the
worker's 146/146 claim reproduces independently on a fresh release. I did not re-run the other 90 bench tests.

## 1. CPU path unchanged — OK

- Method is sound. `tests_bench/bm03_digest_job.py` runs old tree (call without `device`) and new tree (`device="cpu"`) in
  separate processes on the same synthetic release, hashes every `.pt`/`.npy` byte-for-byte and every JSON/JSONL after
  dropping only `utc` and wall-clock keys (digest_job.py:VOLATILE). `spec_sha256` and `init_sha256` are inside the hashed
  JSON, so hash stability is covered. The test also asserts no `device_class` in any CPU RESULT or spec.
- The one exclusion (FL `latest.pt` row bytes/digest in ckpt event logs) is limited to rows whose path is `latest.pt`
  (`_drop_latest`) and was shown to differ even old-vs-old. Acceptable.
- No default changed. `torch_device(None|"cpu")` returns `torch.device("cpu")` with no side effect (accel.py:57-62);
  `enforce_strict_fp32(warn_only=not cuda)` = `True` on CPU, same as bench4 (central.py:94, fl_engine.py:75, device.py:57);
  `build_model` moves only when the device is non-CPU (model.py:51); `rank_chunk` changes are guarded by `dev.type != "cpu"`
  (evaluate.py); the CPU spec has no new key (`if cuda:` at central.py:137, fl_engine.py:128); the CLI default is `cpu`.
- Spec hash for records without the field: stable (above). CPU class is implicit via `device_class_of` returning CPU-FP32
  when the field is absent (accel.py:31-38), so in-flight bench4 CPU runs can resume under bench5.
- Minor M1: no test resumes a directory actually written by bench4 under bench5 on CPU (hash equality implies it works,
  and the new `check_resume` read of ARM_SPEC is benign for CPU, but it is not exercised). Ops: if a CPU job is ever
  resumed under bench5, expect it to work; consider one real check before relying on it.
- Minor M2: the byte-identity job set is small (60 users, 30 items; PRE, C_FULL, FA, FA_WARM, FA_Q8, PF, FT_FA). Fine for
  a code-path guard; it does not cover the dropout-p or max_len variants.

## 2. CUDA determinism settings — OK (one residual risk that only the A40 can retire)

- Set before CUDA init: `prepare_cuda_env` (accel.py:45-54) sets `CUDA_VISIBLE_DEVICES` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  in `run.main` before `_threads`, and refuses if CUDA is already initialised. run.py:22 blanks `CUDA_VISIBLE_DEVICES` at
  import, and the override happens before any CUDA call; spawned FT workers inherit the env. For library callers
  `torch_device` refuses unless the env var is already correct (accel.py:63-64).
- Complete: TF32 off for matmul and cudnn, matmul precision "highest", `use_deterministic_algorithms(True, warn_only=False)`
  on cuda, cudnn deterministic on and benchmark off (fl/numerics.py:25-37). `assert_numerics` re-checks all of these at run
  start, at every central mark and every FL eval (central.py:205, fl_engine.py:170), and the FL client / FT step re-checks
  TF32. Nit: `torch_device` calls `enforce(dev)` before `device_class(dev)` so a non-A40 card is refused only after numerics
  are set; harmless.
- Strict-mode op audit of the code path actually used (SASRec "unpadded", ID-only): `index_select` and `index_copy`
  (backward = index_add), `nonzero`, `cumsum` on an int64 `lengths` tensor (the float-cumsum restriction does not apply),
  embedding forward/backward, explicit matmul + softmax + masked_fill attention (no SDPA / flash kernel, v4models/sasrec_v2.py:132-140),
  `F.dropout`, LayerNorm, GELU, `addmm`, 2-D `cross_entropy` (nll_loss, not nll_loss2d), AdamW, clip_grad_norm, elementwise
  server Adam, `torch.rand(generator=cuda)` for Q8. I found no op that is known to raise under strict mode. The same SASRec
  already ran STRICT on the A40 (tests_train_gpu/det_probe.py). Residual risk: `index_copy_` / `index_add_` CUDA coverage
  under torch 2.9 and the FL / FT / Q8 paths on GPU are only proven by the A40 test. The worker is right that a raise must
  be fixed by a reviewed change, never by `warn_only`.
- Minor M3: `lengths` is moved to the model device inside `_layout` (sasrec_v2.py:285), so the CPU `lengths` that
  `to_device` leaves alone is fine for this model.

## 3. RNG and device correctness — OK

- Data order (numpy PCG64), LR table, FL cohorts, drop-out, client batch order (CPU generator) and DP noise (CPU generator,
  then moved, fl/dp.py:160) are device-independent as documented.
- theta0: built, recentred and hashed on CPU, then `.to(device)` (model.py:48-52); `init_sha256` is device-independent
  (asserted in the GPU test). The adapter is rebuilt after the move; `SASRecAdapter(m)` after `.to()` is correct.
- Dropout: CUDA stream seeded from the same seed (central `torch.manual_seed` seeds all devices; FL/FT through
  `seeded_global_rng`, fl/numerics.py:74-88). Central resume restores `cuda_rng` (central.py:149-150). Documented
  as GPU-class-specific. Good.
- Eval: scores on device, ranking on CPU with exact counts (evaluate.py). Good.
- Minor M4: `device_record` stores `physical_gpu`; the RESULT/ARM_SPEC JSON therefore differs between two otherwise
  identical runs on different physical GPUs. Not a problem for the proposed test (same `--gpu` both times), but a
  digest comparison across different GPU ids would show a false difference in JSON (not in checkpoints).

## 4. Device-class record and refusal paths — OK, tests are real

Record: cuda only. `spec["device_class"]` (so spec_sha256 changes), ARM_SPEC `device`, every RESULT and STATUS, S record,
FT ARM_SPEC/RESULT/STATUS, plus LOCK and WINNER/decision always explicit.

| path | code | NC test |
|---|---|---|
| resume / restart of another class, central | central.py:140-141 | test_nc_resume_refuses_another_device_class (CPU leg on GPU ARM_SPEC) |
| same, FL | fl_engine.py:195-196 | same test |
| GPU leg on CPU ARM_SPEC, other torch/CUDA/cuDNN/card | accel.py:121-131 | test_nc_resume_cpu_run_on_the_gpu_class (monkeypatched device record) |
| theta_pre | fl_engine.py:103-105 | test_nc_theta_pre_s_record_and_ft_base_of_another_class |
| S record | fl_engine.py:115-116 | same |
| FT base | device.py:135-139 | same |
| trackb-select / decide-warm / seed3 | selection.py (same_device / one_class, before any value) | test_nc_trackb_select_refuses_mixed_devices, test_nc_decide_warm_and_seed3_refuse_mixed_devices |
| LOCK / make_lock | final.py:100, 133-142 | test_nc_lock_refuses_mixed_devices (runs, S record, pure GPU and pure CPU) |
| TF32 / numerics | accel.py:80-100 | test_nc_tf32_and_non_strict_numerics_refused |

- Minor M5 (test gap): the NC tests exercise the CPU leg refusing GPU artifacts; the converse (a GPU leg refusing a CPU
  theta_pre / S record / FT base) uses the same symmetric `!=` but is not run, because it needs a GPU or monkeypatching.
  Cheap to add by monkeypatching `accel.torch_device`; not blocking.
- Minor M6: `BM_ALLOW_ANY_GPU=1` makes any card pass as `CUDA_<name>-FP32-DET`. It can never equal the A40 class and the
  LOCK would refuse a mix, so it is safe, but Ops must not export it in the production environment.
- `final` / `verify_lock` / `open_test_gate` / `commit_look` untouched (confirmed in the diff); `verify_lock` still uses
  `tree_digest` (r6 files only), which is unchanged, so S3 LOCKs made by bench4 still verify. `code_sha256` in the LOCK
  covers v4bench/*.py, so a LOCK records which tree made it. Ops: run each dataset's `final` with the tree that made its
  runs (S3 with bench4, ML-1M with bench5) so that the recorded `code_sha256` stays meaningful.

## 5. Answers to the 6 open questions

1. TEST look device. Yes, state it in B16 as "trained NVIDIA_A40-FP32-DET, TEST scored CPU-FP32; FT arms fine-tune on
   CPU at TEST". Do not give `final` a `--device` (look code, needs Fable; and one CPU look is simpler to defend). The
   preflight only checks rank shape, not equality with validation, so no CPU-vs-GPU consistency check can false-fail.
   Disclose that validation ranks (GPU scores) and TEST ranks (CPU scores) come from different float reduction orders;
   ties are broken by ascending id in both.
2. Yes. The GPU set takes its own Track A, decide-warm and seed3 decisions; the CPU records are SUPERSEDED_DEVICE. Say this
   in B16. The new code enforces it (a GPU run cannot consume a CPU-class decision).
3. Strict coverage: see section 2. I found no op that should raise; the A40 test (b) is the proof. If it raises, fix by
   reviewed change.
4. Keep 16 threads for the GPU class. Changing it is unmeasured; the timing gate will show host contention.
5. Agree. `RUN_SUMMARY` is written by the Ops driver (not v4bench), so Ops must add `device_class` from RESULT/STATUS.
   metrics.jsonl without a header is fine; the class is recoverable from the run's ARM_SPEC/RESULT.
6. Yes, intended and slightly strict in the right direction. Resume on another A40 passes (card name, torch, CUDA,
   cuDNN compared; `physical_gpu` is not). A driver-only change is not detected; acceptable, but mention it if a
   driver upgrade happens mid-run.

## 6. Server commands — correct and safe, with two notes

- (a) CPU suite with `CUDA_VISIBLE_DEVICES=` and `BM_BENCH4_ROOT=$V/code_bench4`: fine; `BM_BENCH4_ROOT` must point at a
  tree that contains both `v4bench` and the r6 libs (a full code_bench4), as the handoff says.
- (b) Note: it hard-codes `BM_GPU_TEST=1` while (c)-(e) say "use the free id". Use the same free GPU id in all of them.
- (c) Label and file names match the code: `C_FULL__B24_L200_D0.1_s2026/ckpt/endpoint_6.0.pt` (central.py:254, run.py:200-203)
  and `FA__B6_L200_D0.1_s2026/ckpt/smoke_theta.pt` (fl_engine.py:267). Separate roots `runs_bm03_det_{A,B}` and
  SMOKE labelling keep it away from the CPU run dirs. `ml_1m` is in `DATASETS`.
- (d) Timing runs go to `runs_bm03_timing`, also separate. It re-uses that root if repeated (smoke dirs are write-once);
  use a new root on any re-run.
- (e) Fresh root `runs_bench_gpu`, manifest first, is right. It needs a `DEVICE_ARGS` driver passthrough, and
  `--gpu N` must be the id that `ok/H_dev_v3.ok` has released.
- No command writes into a CPU run dir. A mistaken `--runs-root` pointing at the CPU root would still be refused (device
  class and write-once), as the handoff says.
- Ops pre-step stays important: diff `_archive_bench4` against `$V/code_bench4` before packing (the local tree is only
  assumed equal to the server's).

## Findings list (file:line)

- M1 tests_bench/test_bm03_device.py:57 — no resume-from-bench4-dir check on CPU.
- M2 tests_bench/bm03_digest_job.py — small job set; guard not exhaustive.
- M3 v4bench/accel.py:63-71 — `enforce` runs before the A40 name check; harmless ordering.
- M4 v4bench/accel.py:132-140 — `physical_gpu` in the record makes JSON differ across GPU ids.
- M5 tests_bench/test_bm03_device.py:231 — reverse-direction NCs (GPU leg vs CPU artifacts) are not run.
- M6 v4bench/accel.py:101-106 — `BM_ALLOW_ANY_GPU` must stay unset in production.
- R1 (pending, not a defect) — GPU determinism and timing evidence exist only after the A40 runs (b)-(d).

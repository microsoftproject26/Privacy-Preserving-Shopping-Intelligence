# Sonnet B (trainer) — Wave-1 report

Owner: Sonnet B. Files created (local mirror, uploaded to `$W/code/`):

- `code/train_full.py` — the full-data trainer.
- `code/supervise_full.sh` — process supervisor (restart policy).
- `code/launch_full.sh` — tmux launcher (does not itself start a production run).
- `code/test_train_full.py` — pytest suite (20 test functions, 27 collected items after parametrisation).
- `code/fakes_for_tests.py` — pilot-cache-backed `FakeStore`/`FakeDecisionSet`/`fake_evaluate_model`, used only by the test suite.
- This report, also uploaded to `$W/agent_reports/sonnetB_trainer.md`.

`model.py`, `objective.py`, `train.py`, `window_cache.py`, `evaluator.py`, `synthetic_memorisation.py` were imported, never edited.

Local paths (Windows mirror): `F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1\code\{train_full.py,supervise_full.sh,launch_full.sh,test_train_full.py,fakes_for_tests.py}`, report at `F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1\agent_reports\sonnetB_trainer.md`.

**Commands used** (server, via `tools/lobot.py cmd`):
```
CUDA_VISIBLE_DEVICES=0 ~/venvs/ppsi-gpu/bin/python -m pytest test_train_full.py -v -s
```
run from `$W/code`. Exit code of the pytest process itself was 0 (all tests passed or xfailed as expected); see §3 for the per-test breakdown and §5 for `train_full.py`'s own exit-code contract (0/3/4/5/6/other).

## Bugs found and fixed during testing (both in `train_full.py` itself; found before any test ran against the real `full_store.py`/`eval_full.py`)
1. **`check_gpu_busy` didn't exclude its own PID.** Once any earlier call in the same process had touched CUDA, every subsequent startup check saw that process's own already-open CUDA context reported by `nvidia-smi --query-compute-apps` and treated it as "foreign", spuriously returning exit 6 (GPU_BUSY). Fixed by excluding `os.getpid()` from the foreign-PID list. No effect on production (`supervise_full.sh` always launches a fresh subprocess, so self-exclusion is a no-op there).
2. **`write_checkpoint_atomic` left a stray `.tmp` file on a validation failure.** Harmless to correctness (readers only ever look at `latest.pt`/`previous.pt`/`best.pt`) but a real disk-hygiene gap under repeated failures. Fixed: the `.tmp` is removed before the exception propagates.

## 1. What `train_full.py` implements

- CLI: `--arm {F0_item_only_h256,F4_rich_side_information_h256} --run-dir DIR [--resume] [--max-epochs 6] [--adamw-impl foreach|fused] [--ckpt-every-updates 10000] [--diag-every-updates 20000] [--benchmark-updates N] [--test-tiny]`, plus two test-only flags gated behind `--test-tiny` (`--stop-after-updates`, `--break-resume-perm-seed`, see §4).
- Store access strictly through `CompactStore.build/.decisions/.batch` and `eval_full.evaluate_model` signatures (`FULL_INTERFACES.md` §1–3); both are **lazily imported** only on the non-test-tiny path, so `train_full.py` never hard-fails if `full_store.py`/`eval_full.py` aren't present yet.
- Launch-gate asserts (production only, skipped/relaxed under `--test-tiny`): exact population counts (40,180,608 / 8,826,488 / 790,459), `loss_mask.all()`, strictly-ascending `decision_id`, TRAIN/VS disjointness (gate 4, `_assert_disjoint_sorted`, searchsorted-based), exactly one visible CUDA device, `product_idx_of_class == arange(K)+3`.
- Data order: `epoch_permutation(epoch, n_train, seed_base=2026)` = `np.random.default_rng(2026+epoch).permutation(n_train)`, applied via `iter_positions()`; permutation sha256 logged once per epoch (`epoch_start` records); deterministic across resume (recomputed from the epoch index alone, no persisted RNG needed for this part).
- Update step: `zero_grad(set_to_none=True)` → forward → `objective.exact_ce_loss` → `loss_sum.backward()` → divide every grad by `n` once → `clip_grad_norm_(...,1.0)` → **fail-closed finite check on loss+grad_norm BEFORE `optimizer.step()`** (isolated in `_check_finite`, monkeypatchable) → `update += 1` → set LR → `optimizer.step()`. Non-finite ⇒ no step, no checkpoint, `status.json` state `STOPPED_NONFINITE`, exit 3.
- AdamW: `torch.optim.AdamW(build_param_groups(model, wd=1e-5), lr, betas=(0.9,0.999), eps=1e-8, foreach=True|fused=True)`; `--adamw-impl` stored in `config`; resume refuses a mismatch (`ConfigError` → exit 5).
- LR: 100-update linear warmup then 1e-3; the pure function `selection_step` decides the once-only 1e-3→1e-4 step (recorded via `SelectionState.lr_decayed`/`lr_decay_epoch`); `compute_lr(update, lr_decayed)` applies it from the next update onward automatically (always epoch ≥ 1, safely after warmup).
- Selection/stopping: `selection_step(state, epoch, metric, max_epochs=...)` — a pure function, unit-tested on 5 scripted scenarios (plain improvement, sub-threshold improvement that still updates best, one-time LR step, patience-2 stop, max-epochs stop, exact-tie keeps the earlier epoch).
- Epoch-end crash-safety ordering exactly as specified: finish last update → `save latest (eval_pending=True)` → diagnostic-cohort eval (logged only, RNG saved/restored) → full-VS eval → `selection_step` → `save best` if raw-improved → `save latest (eval_pending=False)` → stop or continue. On resume, an `eval_pending=True` checkpoint re-runs the epoch-end eval (idempotent: deterministic eval, no training happened in between) before anything else.
- Checkpoints: `best.pt` / `latest.pt` / `previous.pt` only, `.tmp` → `torch.load` validate (required-key set + `update` match) → `os.replace` rotate (`latest→previous`, `tmp→latest`); a failed validation removes its own `.tmp` and never touches the existing files (proven by test #2). Disk guard (`os.statvfs($W)`, free − expected ≥ 8 GiB) runs before every write; a failure raises `DiskGuardStop` → `STOPPED_DISK`, exit 4, with no checkpoint written.
- Logging: `metrics.jsonl` — 100-update aggregates (`update_agg`), `epoch_start`/`epoch_end_accounting`/`epoch_end`/`eval`/`diag_eval` records; **plus a per-update `update_debug` record emitted only when `--test-tiny`** (production never does this — needed because the spec's 100-update cadence can't resolve losses at tiny test update-counts). `status.json` heartbeat ≤60s, atomic. `summary.json` at the end (final status, best epoch/metric, best checkpoint path+sha256, resume_count, selection history).
- Input pipeline: one-ahead double-buffered prefetch via `ThreadPoolExecutor(max_workers=1)` (`store.batch` + `pin_memory()` in the background while the GPU runs the current step); order/content are a pure function of `iter_positions`, unaffected by thread timing.
- Exit codes: 0 COMPLETE, 3 NONFINITE, 4 DISK, 5 CONFIG/ASSERT, 6 GPU_BUSY (checked via `nvidia-smi --query-compute-apps` on the physical GPU named by `CUDA_VISIBLE_DEVICES`, excluding the trainer's own PID), anything else = an uncaught-exception crash (handled by the supervisor's restart policy).

## 2. `supervise_full.sh` / `launch_full.sh`

- `supervise_full.sh <arm> <physical_gpu> <run_dir> [extra_args]`: refuses to start under 8 GiB free; adds `--resume` iff `checkpoints/latest.pt` exists; exit 0 → `SUPERVISOR_DONE`; 3/4/5 → `SUPERVISOR_STOPPED`, no restart; 6 → 300s wait, retry uncounted; anything else → up to 5 restarts, 60s backoff, all logged to `<run_dir>/logs/supervisor.log`. The trainer child now runs backgrounded-then-`wait`ed (its PID recorded at `<run_dir>/logs/train_full.pid`) purely so the test suite can target it with SIGKILL; production behaviour (one blocking child at a time) is unchanged. The optional 4th arg is test-only passthrough (e.g. `--test-tiny ...`); `launch_full.sh` never supplies it.
- `launch_full.sh <arm> <gpu> <run_dir>`: starts `supervise_full.sh` detached in tmux session `gru-full-{F0,F4}`, refuses if that session exists, prints `<session> <pid>`. Does not itself start any production run.

## 3. Test results (server, GPU 0, `~/venvs/ppsi-gpu/bin/python -m pytest code/test_train_full.py -v -s`)

Final full-suite run (single invocation, all 27 items): **26 passed, 1 xfailed, 0 failed**, wall time 335.9s (5:36). Server scratch `$W/tests_full/B` confirmed **empty** (`du -sh` → 4.0K = bare directory) after the run; `$W` free disk 25 GiB throughout (well above the 8/10 GiB gates).

```
test_supervisor_sigkill_mid_epoch_resumes_and_finishes            PASSED
test_resume_determinism[foreach]                                  PASSED
test_resume_determinism[fused]                                    PASSED
test_resume_broken_perm_seed_xfails                                XFAIL (expected)
test_test_tiny_refuses_run_dir_outside_tests_full                 PASSED
test_resume_refuses_adamw_impl_mismatch                            PASSED
test_checkpoint_rotation_survives_injected_failure                 PASSED
test_nonfinite_stop_exit_code_and_no_checkpoint                    PASSED
test_disk_guard_stop_exit_code_and_checkpoints_intact               PASSED
test_selection_step_improvement_and_ties                            PASSED
test_selection_step_sub_threshold_improvement_still_updates_best    PASSED
test_selection_step_lr_step_once_and_patience_stop                  PASSED
test_selection_step_max_epochs_stop_without_patience                PASSED
test_selection_step_tie_keeps_earlier_epoch                         PASSED
test_vs_leakage_assert_fires_on_injected_overlap                    PASSED
test_updates_per_epoch_and_last_batch_accounting                    PASSED
test_train_batches_always_have_loss_mask_all_true                   PASSED
test_synthetic_memorisation_under_launch_config[F0_item_only-foreach]              PASSED
test_synthetic_memorisation_under_launch_config[F0_item_only-fused]                PASSED
test_synthetic_memorisation_under_launch_config[F4_rich_side_information-foreach]  PASSED
test_synthetic_memorisation_under_launch_config[F4_rich_side_information-fused]    PASSED
test_assert_single_visible_gpu_passes_in_this_env                   PASSED
test_assert_single_visible_gpu_fails_on_multi_device                PASSED
test_benchmark_smoke[F0_item_only_h256-foreach]                     PASSED
test_benchmark_smoke[F0_item_only_h256-fused]                       PASSED
test_benchmark_smoke[F4_rich_side_information_h256-foreach]         PASSED
test_benchmark_smoke[F4_rich_side_information_h256-fused]           PASSED
```

**Benchmark pre-check** (FakeStore, 300 updates, GPU 0, A16; Opus will run the real benchmark against `full_store.py` later):

| arm | adamw_impl | median dec/s | data-wait fraction | median step (s) | peak GPU mem alloc / reserved | host RSS |
|---|---|---|---|---|---|---|
| F0_item_only_h256 | foreach | 4,244 | 0.0008 | 0.0603 | 951.6 MB / 3,129.0 MB | 1.91 GB |
| F0_item_only_h256 | fused | 4,780 | 0.0009 | 0.0536 | 951.6 MB / 3,129.0 MB | 1.91 GB |
| F4_rich_side_information_h256 | foreach | 4,090 | 0.0014 | 0.0626 | 965.1 MB / 3,131.0 MB | 1.92 GB |
| F4_rich_side_information_h256 | fused | 4,600 | 0.0014 | 0.0557 | 965.1 MB / 3,131.0 MB | 1.93 GB |

`fused` is consistently ~9-13% faster than `foreach` on both arms in this pre-check; data-wait fraction is negligible (<0.15%) with the FakeStore's memmapped reads, so the double-buffered prefetch thread is not the bottleneck here — real `CompactStore` I/O characteristics may differ.

### One test-ordering note (not a trainer defect)

`test_supervisor_sigkill_mid_epoch_resumes_and_finishes` spawns `train_full.py` as a **separate OS subprocess** and that subprocess's startup `check_gpu_busy()` correctly refuses to start if `nvidia-smi --query-compute-apps` shows any *other* PID already holding a CUDA context on the target GPU. When this test ran **after** other in-process tests that had already built real models on GPU 0 within the same long-lived pytest process, it correctly saw the pytest process itself as a "foreign" GPU user and entered its 300s GPU_BUSY retry loop, which our test's much shorter polling deadlines then failed on. This is the GPU_BUSY protection working exactly as specified, not a bug — the fix was to reorder the test file so this subprocess-based test runs **first**, before any other test touches CUDA in-process. It now passes reliably in the full-suite run above. Documented here since it's a real interaction Opus/CI should be aware of if these tests are ever split across files or reordered by a test runner: **run the supervisor test before, or in full isolation from, any in-process CUDA test.**

## 4. Interface assumptions / deviations (read before integrating)

- **`fakes_for_tests.py` is test-only** and is never imported by `train_full.py`'s production code path. It backs `FakeDecisionSet.end_row/user_bucket/prior_view_count/prefix_position` with harmless placeholders (cache row index / 0 / 0 / length) because the pilot window cache does not store the real values for those fields; nothing in `train_full.py` or the fake evaluator reads them for anything beyond structural completeness. Real `full_store.py` should populate them properly.
- **Diagnostic cohort in test-tiny mode** = the fake VS set itself (both are tiny already), not a genuine sub-cohort — acceptable since the diagnostic eval is logged-only and untested for selection correctness.
- **Per-update `update_debug` log records** are emitted only under `--test-tiny` (see §1); this is an additive, test-only relaxation of the logging cadence, not a change to production behaviour.
- **Two test-only CLI flags**, both refused unless `--test-tiny` is also given (else `ConfigError`/exit 5):
  - `--stop-after-updates N` — simulates a clean external kill immediately after update N finishes all of its own bookkeeping (checkpoint/eval if due exactly at N). Used to interrupt resume-determinism tests at an exact update count instead of only at epoch boundaries.
  - `--break-resume-perm-seed` — reconstructs epoch permutations from the wrong seed base on resume, for the deliberately-broken `xfail(strict=True)` test proving the resume-determinism test has power to catch a real bug.
  - A third env-gated hook, `TRAIN_FULL_TEST_UPDATE_SLEEP` (only read when `--test-tiny`), adds a fixed per-update sleep **after** that update's own checkpoint/eval work — used solely by the supervisor SIGKILL test so an external process can reliably observe a just-written checkpoint before killing, instead of racing a ~16-update FakeStore epoch to its natural completion.
- **Checkpoint size estimate** (`estimate_checkpoint_bytes`) is a heuristic (≈3× model parameter bytes + 50 MiB overhead, ×1.3 margin) used only for the disk guard, not for planning; observed real checkpoints for this architecture are the ~250 MB the resource table already assumes.
- **No deviation from `FULL_RUN_PROTOCOL.md`** sections 4–6, 10 (gates 4/6/7), 11, 12 was made; where the protocol was silent (e.g. exact checkpoint-size estimate, prefetch mechanics, per-update vs. per-100-update test logging), the choice is recorded here rather than in the protocol.

## 5. Unresolved issues / follow-ups for Opus

- `full_store.py` (Sonnet A) and `eval_full.py` (Sonnet C) did not exist on the server as of this work; `train_full.py`'s production path (`build_store`/`get_evaluate_fn`, non-test-tiny) has **not** been exercised against the real modules — only against `fakes_for_tests.py`. Once both land, a smoke run of `train_full.py --arm F0_item_only_h256 --run-dir <scratch> --benchmark-updates 50` against the real `CompactStore` (still no `--test-tiny`, so real data, but `--benchmark-updates` writes nothing) is the cheapest first integration check.
- `run_launch_asserts` computes VS-leakage disjointness via `np.searchsorted` assuming `ds_train.decision_id` is sorted ascending (per the `DecisionSet` contract) — this is O((N+M) log N); at production scale (40.18M + 8.83M) this should take a few seconds, not benchmarked here since it was only exercised against the tiny FakeStore.
- The benchmark numbers reported in §3 are on the **FakeStore** (pilot cache, real model, real optimizer) purely as a pre-check per the task brief ("Opus runs the real benchmark later"); they are not representative of production I/O (`full_store.py`'s real `CompactStore.batch` may have different throughput characteristics than `window_cache.CacheReader`).

# Fable FO-02 correction-only review (verbatim; claude-fable-5-1; returned ≈2026-09-26T07:50Z)

Brief: review/fo02/FO02_CORRECTION_BRIEF.md. Agent: sasrec-fable-reviewer (read-only), agentId a92532a638cc454b4.

ROUTING: claude-fable-5-1

**Q1. Fix correct and sufficient?** Yes, for the server as configured.
- Root cause confirmed [evidence]: `code\snap_sasrec\reuse\train.py:35-49` hard-codes `gru_v1/code` + `/vendor` at `sys.path[0]`; `reuse\eval_full.py:43` imports `train` at module level; `code\fo_eval.py:25` and `snap_sasrec\sasrec_data.py:19` lazily import `full_store`. `server_probes.log` shows exactly that breach twice, caught before any write (`probe_runner.py:116` precedes `:117-134`).
- Pre-import order is sound [evidence]: `fo_env.py:39-40` loads `rees46_eda.config/loaders`, `index_utils`, `window_cache`, `full_store`, `evaluator`, `model`, `objective` while only snapshot dirs are on `sys.path` (`paths.py:39-41`); `eval_full` then pulls `train` from `snap_sasrec/reuse` (first hit for the name), and `train`'s own imports (`:55-58`) are already cached. The scrub at `:52` removes both GRU entries; `assert_isolated` `:81-83` re-checks.
- `train` is the only reuse module hard-coding a foreign dir [evidence]: `lock_and_eval.py:51-55` and `baselines_full.py:62-66` use `Path(__file__)`-relative dirs, so a later import resolves in `snap_sasrec/reuse` and inserts only snapshot dirs; both plus `verify_frozen` are now in `ISOLATED_MODULES` (`fo_env.py:57-58`). `SERVER_ISOLATION_CHECK.json` confirms 20/20 in-snapshot, `foreign_sys_path: []` after importing them.
- `rees46_eda.*` cannot escape [inference from Python import semantics]: the package object is loaded from the snapshot vendor first, so submodule lookup uses its `__path__`, not `sys.path`; the package has only `config`/`loaders` (Glob).
- Every stage module imports `fo_env` before any snapshot module [evidence]: `feature_view.py:33`, `fo_eval.py:15`, `fo_models.py:21`, `probe_runner.py:26`, `screen.py:31`, `finalize_fo.py:22`, `stream_fit.py:27`, `pin_baseline.py:20`. No subprocess/multiprocessing/`num_workers` re-imports in study code; `fo_models.py:29` uses `torch.load(weights_only=True)` (no module resolution).
- Residual, non-blocking [evidence]: `finalize_fo.py` never calls `assert_isolated` (only `probe_runner.py:116`, `screen.py:161`). Pre-existing approved design; with the fix its `eval_full` is guaranteed snapshot-loaded anyway.
- `_is_foreign` compares literal strings; a differently spelled alias of `gru_v1/code` would pass [inference], but no code inserts one.

**Q2. DONE outputs valid?** Yes; no re-run.
- `stream_fit` imports only `fo_env`, `feature_view` -> `rees46_eda.loaders`, `sasrec_model` (torch-only imports, `train` lazy at `:379-380`); `pin_baseline` imports `fo_env` only; neither imports `train`, so `gru_v1/code` was never on `sys.path` in those processes [evidence]. `full_profile` imports no snapshot/GRU module (`profile_train_sample.py:45-64`); `T1_protect`/`T1_diff` hash files and call `ps`/`tmux` only [evidence].
- Independently, `BYTE_IDENTITY_SNAPSHOT_VS_GRU.txt` shows 14/14 sha256 equal and the hashes match `SNAPSHOT_MANIFEST.json` [evidence], so even a foreign load would have executed identical bytes.
- Probe attempts wrote nothing (`audit/` mkdir and `COHORT_DS.npz` come after the failing guard) [evidence], so resume is clean.

**Q3. CUDA_VISIBLE_DEVICES side effect?** Acceptable. `train.py:33` is `setdefault`; the queue sets it for GPU stages (`fo_queue.py:111`) [evidence]. `finalize` already transitively imported `train` under the approved code (`finalize_fo.py:24`) [evidence]; `pin_baseline`/`stream_fit` are DONE and do not use torch/cuda. The check script sets `""` first, which `setdefault` respects [evidence].

**Q4. Diff scope / tests** [evidence]: `fo_env.py` changes are confined to lines 29-58 and 81-83 versus `fo_env.py.approved_ReviewC`; `test_fo_env.py` adds exactly the three FO-02 tests (lines 105-136), the rest is identical to the approved copy. The new tests are meaningful locally (the inert `/home/...` string still gets inserted and scrubbed) and include negative controls. `SERVER_PYTEST.txt`: 128 passed. I did not execute anything or recompute the new sha256s (lead's claim). Resume via `fo_queue.stage()` skips DONE and gives `probes` two fresh attempts (`fo_queue.py:120-123`) [evidence]; `fo_queue.py` itself is unchanged.

Optional, not required for resume: add `fo_env.assert_isolated()` to `finalize_fo.main` for symmetry.

VERDICT: APPROVE_RESUME

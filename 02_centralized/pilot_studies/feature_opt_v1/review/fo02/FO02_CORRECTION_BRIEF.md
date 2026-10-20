# FO-02 correction-only review brief (lead: Opus, 2026-09-26T07:47Z)

You are Fable 5.1, the owner's delegated decision authority for `SASREC_FEATURE_OPT_V1`. The owner said: "have Fable review
everything and do what is best without coming back to me". This is a **correction-only** review of one defect fix, in the
spirit of Review C (`review/FABLE_CODE_REVIEW.md`, pass file `review/FEATURE_CODE_REVIEW_PASS.json`). Read-only.

## What happened (MEASURED)
- The campaign finished (`sasrec_v1/state/FINAL_STAGE_DONE` 07:24:17Z). The queue `fo_queue.py` started automatically.
  T1_protect, T1_diff, stream_fit, full_profile and pin_baseline were DONE
  (`review/fo02/server_fo_queue.log`, `server_fo_status.json`).
- `probes` failed twice with the same signature and the queue stopped with state FAILED, fail-closed, GPUs idle
  (`review/fo02/server_probes.log`):
  `RuntimeError: isolation breach: module full_store loaded from .../gru_v1/code/full_store.py, not from the snapshot`.

## Root cause (MEASURED by reading the code and by a local reproduction)
- The snapshot's reused GRU `snap_sasrec/reuse/train.py` (lines 35–49) hard-codes
  `WORKSPACE = .../gru_v1` and inserts `gru_v1/code` and `gru_v1/code/vendor` at `sys.path[0]` when it is imported.
- `eval_full.py` imports `train` at module level. The study imports `eval_full` (through `fo_eval.py`). From then
  on, every lazy import resolves in the GRU tree first. That covers `from full_store import CompactStore` inside
  `fo_eval`/`sasrec_data`, and also `train`'s own `import model`, `objective`, `evaluator` and `window_cache`.
- The local tests never saw it: on Windows the directory `/home/jovyan/.../gru_v1/code` does not exist, so the
  inserted string is inert. With the approved `fo_env.py` the string is on `sys.path` locally too; see the
  reproduction below.
- The guard `fo_env.assert_isolated()` worked as designed and stopped the stage.

## Impact on the stages already DONE (MEASURED)
- `review/fo02/BYTE_IDENTITY_SNAPSHOT_VS_GRU.txt`: every reuse module and every vendored `rees46_eda` `.py` in the
  snapshot is **byte-identical** to the file in `gru_v1/code`, with 14 of 14 sha256 equal. `diff -rq -x __pycache__`
  is clean.
- stream_fit and pin_baseline imported `fo_env`. full_profile, T1_protect and T1_diff do not.
- Even if any of these modules was loaded from the GRU tree, the executed bytes were identical. So the lead's view is
  that the DONE outputs (`audit/M1_FIT.json`, `audit/DUPLICATES_FULL.json`, `audit/TRAIN_PROFILE_FULL.json`,
  `config/BASELINE_PIN.json`, `config/NOISE_REFERENCE.json`) are unaffected.
- `gru_v1/code` is the frozen GRU tree, which is read-only for this study; nothing was written there.

## The fix (study code only; the snapshot is NOT edited; its manifest still verifies, 24 files)
- `code/fo_env.py` (approved sha `49a32aaf…`, preserved as `review/fo02/fo_env.py.approved_ReviewC`; new sha `5cdaa082…`):
  1. After `import paths`, pre-import every reuse module from the snapshot, dependencies first:
     `rees46_eda.config, rees46_eda.loaders, index_utils, window_cache, full_store, evaluator, model, objective, eval_full, train`.
     `eval_full` must come before `train`. A first attempt with `train` before `eval_full` still breached on the
     server, because `eval_full` then resolved in the GRU tree. That is recorded below.
  2. Scrub `sys.path` of `FOREIGN_CODE_DIRS` (`gru_v1/code` and `sasrec_v1/code`, with any subdirectory), so later
     lazy imports stay in the snapshot.
  3. `assert_isolated()` also checks `index_utils`, `window_cache`, `model`, `objective`, `baselines_full`,
     `lock_and_eval` and `verify_frozen`. It now raises if a foreign code directory re-appears on `sys.path`.
- `code/tests/test_fo_env.py` (approved sha `e2345dcc…`, preserved; new sha `373d20a6…`) gains 3 tests:
  - the pre-imports come from the snapshot;
  - `sys.path` is scrubbed and a late `full_store` import stays in the snapshot;
  - negative controls: a foreign entry re-inserted in both separator spellings raises, and a sibling
    `code_evil` is not treated as foreign.
- Side effect to judge: importing `train` at `fo_env` import time runs its `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")`.
  - For GPU stages the queue sets `CUDA_VISIBLE_DEVICES` first, so there is no change.
  - For CPU stages (finalize, T2) the value becomes "0" when it was unset. finalize already imported `eval_full`,
    and through it `train`, in the approved code, so behaviour is unchanged.

## Verification (MEASURED)
- Local: 128/128 tests pass (125 approved plus 3 new).
- Local reproduction, in a scratchpad copy:
  - with the approved `fo_env`, after `import fo_eval` the list `sys.path` contains
    `\home\jovyan\...\gru_v1\code` and `...\code\vendor`;
  - with the fix it is empty.
- Server: `review/fo02/SERVER_PYTEST.txt` shows 128 passed.
- Server: `review/fo02/SERVER_ISOLATION_CHECK.json` shows the following:
  - it imports every stage module (probe_runner, screen, finalize_fo, pin_baseline, stream_fit) and forces the lazy
    `from full_store import CompactStore`;
  - 20 isolated modules, all under `snap_sasrec`;
  - `foreign_sys_path: []`;
  - snapshot manifest verified (24 files).
- The check script is `review/fo02/server_isolation_check.py`; it runs CPU only.

## Resume plan (lead)
- Restart `fo_queue.py` in tmux `fo-queue`. DONE stages are skipped; `probes` gets 2 fresh attempts; nothing is
  deleted.
- The pass file stays. A new `review/fo02/FO02_REVIEW_PASS.json` records the new hashes if you approve.

## Questions (answer each)
1. Is the fix correct and sufficient for isolation on the server? Is there any other module or lazy import that can
   still escape (for example `baselines_full` / `lock_and_eval` if imported later, or `rees46_eda.*` submodules)?
2. Are the DONE stage outputs valid given the byte-identity evidence, or must any stage be re-run? A re-run would
   need its write-once outputs moved aside, and evidence is never deleted.
3. Is the CUDA_VISIBLE_DEVICES side effect acceptable?
4. Verdict: APPROVE_RESUME / FIX_REQUIRED (with exact fixes) / STOP.

# REV-BM-05 — independent read-only review of BM-05 (`final` on the dataset's one device class)

VERDICT: PASS_WITH_MINORS

Scope: diff of code/_archive_bench5_pre_bm05 vs code/v4bench/{accel,final,run}.py, test_bm03_device.py, new test_bm05_final.py /
bm05_final_job.py. No server, no benchmark values, no Codex. CPU pytest (`tests_bench`, minus test_ml1m_gpu_script.py): all
passed or skipped, no failures (GPU tests skipped as expected).

## (1) CPU `final` byte-identical to bench4 — OK
- final_eval with device None/"cpu": `torch_device` returns cpu with no side effect (accel.py:75-78); `devs`, `dkw` are empty,
  so score_run / build_model / ft_ranks are called with the same arguments as before (final.py:213-221, 262-282). The new
  `assert_numerics(None)` is a no-op. Start row, commit_look, TEST_*.json and TEST_RESULTS gain no keys on CPU. run.py passes
  `device` only for cuda (run.py:127). Defaults are unchanged (`--device cpu`).
- test_final_cpu_byte_identical_to_bench4 is a real comparison: bench4's final vs this tree on byte-identical copies
  (8 rank .npy plus all TEST json / attempts, only utc stripped), over plain, PF and FT arms. Method sound.

## (2) Refusal ordering — OK
- final_eval order: torch_device -> check_lock_device -> append_jsonl("start") -> _final_eval (final.py:262-270). The refusal
  raises before the attempt row, before open_test_gate / token, preflight, holdout read and commit_look.
- check_lock_device (final.py:232-247) reads only LOCK.json and each locked RESULT.json. No TEST/holdout data is read, and no
  register / exposure record is written anywhere in final.py (none exists there). "Before logging the attempt" therefore
  means no TEST_ATTEMPTS.jsonl, no TEST_LOOK, no data read. Covered for LOCK-vs-device (both directions, incl. a bench4 LOCK
  with no field) and run-vs-device by test_bm05_final.py NCs; CLI path checked (rc 2, no look/attempt).
- Minor M1: `torch_device("cuda")` runs before the check, so a mismatched cuda look initialises CUDA first. Harmless, but
  the check could precede it. Minor M2: on the CLI, verify_lock and D.load (train data only) run before the refusal;
  they touch no TEST data. Minor M3: a missing LOCK makes check_lock_device return, so the "start" row IS logged before
  open_test_gate refuses (pre-existing behaviour; run.py's verify_lock catches it first on the CLI, direct API callers only).

## (3) cuda path — OK in code, GPU run still PENDING
- FT arms: score_run passes `device="cuda"` to ft_ranks (final.py:215-217); each worker (or the main process when workers=1)
  calls accel.torch_device -> enforce + CUBLAS/visibility checks (device.py:56-57); the adapter is built on the GPU after the
  CPU init (BM-03 path); rank chunks rank on CPU. Plain/PF arms build the model on dev and load the checkpoint into it.
  `assert_numerics` runs before each preflight / TEST scoring call (final.py:317, 328).
- Class recorded: TEST_RESULTS (`device_class` + `device` record), TEST_LOOK, each TEST_<rule>.json, and the start row. The
  CPU class is pinned through the LOCK's device_class (lock_self_sha256).
- Minor M4: the cuda tests that run here use `_as_cuda` monkeypatching, but only to exercise the refusal paths (nothing hides
  the scoring path); the scoring path is covered solely by the skipped GPU two-run test, which asserts digest equality and
  class fields but not that FT actually ran on cuda. Also preflight uses workers=1 but the TEST look uses W workers:
  bit-equality of GPU ranks across worker counts is untested (the two-run test uses the same W). The A40 run (handoff 5b)
  must be executed before the ML-1M look; it is the gate for item 3.

## (4) accel.py fix — correct; the delta is not functionally needed by `run`, but should be deployed first
- Old is_cuda treated any non-"cpu" object as GPU (a module named `device` in final.score_run). The new one is true only
  for "cuda[:i]" or a type=="cuda" torch.device (accel.py:41-47); torch_device now refuses unknown values (accel.py:50-56,
  75) instead of silently going to CPU.
- CPU-path behaviour unchanged: all other call sites (central.py:93, device.py:56/136, fl_engine.py:74) pass None, "cpu",
  "cuda" or torch.device only, and test_bm03_device byte-identity guards pass. Already-run bench5 smoke runs
  (runs_bm03_*) used those same inputs, so their results/classification are unaffected (cuda -> still cuda, cpu -> still cpu).
  A "cpu:0" string, previously mis-detected as GPU, now correctly is CPU; no past run used it.
- Deployment: production `run` jobs gain nothing functionally. But `code_sha256` (bench_code_sha256 over all v4bench/*.py,
  common.py:154) is recorded in the LOCK and the E1 spec says the LOCK pins one tree for training AND `final`. So deploy the delta
  (in place or as code_bench5b) BEFORE the first production ML-1M GPU run; otherwise training would be by a tree whose
  final.py cannot look on cuda (old final has no --device, and the old tree's `final` would be CPU-only, then refused by the
  class check upstream). verify_lock checks only the r6 tree digest, not the bench code sha, so a late swap would not be
  caught mechanically. Unchanged: S3 x3 keep code_bench4 `final`.

## (5) Tests — real, with caveats in M4
Byte-identity runs two real subprocess trees; NCs call real final_eval / R.main; the strictness regression
(test_is_cuda_is_strict...) reproduces the module-as-device bug. No hidden monkeypatch on the CPU identity or CLI tests.

## Conditions before the ML-1M look
Run the A40 two-run test (handoff 5b) and require PASS; deploy the delta before the first GPU `run`; hash-check the
six files against handoff section 1.

# REV-OPS-B13-P2 (Sonnet, read-only)

Verdict: PASS-WITH-FIXES (no BLOCKER, no MAJOR)

Stopper: matches exact dataset, arm, seed 2026, max-len, dropout and budget; excludes --smoke, seed 2027, Track-A (not a grid cell) and non-ml_1m. The real process chain (python -> subshell -> wrapper -> driver 506600) matches the kill logic: only the subshell and wrapper are SIGKILLed, never the driver. The clock is the only trigger. Marker status INCOMPLETE_B13_BACKSTOP is consistent with common.py:165 (refuse_overwrite), final.py:40 and central.py:156. trackb-select runs on DONE cells only. .ok states are written, so the driver does not relaunch.

Latency-only: no export, accuracy or RSS. Shas and protocol are asserted before and after timing. The write is write-once. bench.py differs from r7 only by the r7b/dev05 and P2 changes. Local run: 13 passed, 1 skipped, 6 xfailed; --runxfail -k test_nc_ gives 6/6 AssertionError (non-vacuous).

Lane: waits for Wave 1 only. SIGSTOP targets only run_wave2.sh and amazon2023_ pids. EXIT trap SIGCONTs on all exit paths.

MINOR:
- laneD3_INT8_pass2.sh:56 and :65. LOAD0 is sampled before the 20 s settle sleep, and PASS2_HOST_LOAD.json is written only after the bench. A failed bench still leaves that file, which trips the write-once check at line 46 on any rerun. Write the file before the bench, or delete it on failure.
- laneD3_INT8_pass2.sh:17-18. W2_DRV_RE matches any process whose args contain run_wave2.sh. This is acceptable, but the pids logged should be reviewed.

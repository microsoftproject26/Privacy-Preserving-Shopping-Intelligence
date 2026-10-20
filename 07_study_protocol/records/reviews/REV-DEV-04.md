# REV-DEV-04 (Claude Sonnet, read-only)
Verdict: PASS-WITH-FIXES (no BLOCKER, no MAJOR).
- View built from the checkpoint's own meta + RUN_START + registry (checkpointio resolve_feature_view / load_view_schema); every disagreement refuses. OK.
- ONNX signature includes the extension keys (scoring.py extension_input_keys). OK.
- dump_rows binds rows to the view (batch_view, check_rows_view); stale rows without batch_view are refused for a non-identity projection. OK.
- v4eval: byte-identical to r6 tar (diff --strip-trailing-cr, no differences). Retention/rank path in bench.py unchanged (diff touches only view checks, flags, timing).
- NCs: 19 of 19 collected failed under --runxfail (2 more are onnx skips locally). Not vacuous.
- MINOR bench.py:check_bench_rows: with an identity (base) projection, rows without meta are accepted on key/shape checks alone.
- MINOR check_rows_view (dump_rows.py): the manifest and history binding is skipped when the rows meta lacks the field.
- MINOR test_dump_rows.py:360: one NC fails with ValueError instead of AssertionError. This is pre-existing (file unchanged from r6).
- Note: v4int/reeval.py and launch_fl.py differ from the r6 tar, outside the DEV-04 scope. Confirm they are intended.

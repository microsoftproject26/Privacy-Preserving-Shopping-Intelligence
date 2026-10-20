# AMENDMENT FO-02 — import isolation fix (2026-09-26, lead Opus; approved by Fable, APPROVE_RESUME)

- **Incident:** the probes stage failed twice at 07:35–07:36Z with `isolation breach: module full_store loaded from gru_v1/code`.
  The queue stopped fail-closed; no probe output was written.
- **Cause:** the reused GRU `train.py` (snapshot, unchanged) inserts `gru_v1/code` at `sys.path[0]`; `eval_full` imports `train`,
  so later lazy imports resolved in the GRU tree. This was server-only; locally the path does not exist.
- **Fix:** `code/fo_env.py` pre-imports every reuse module from the snapshot (`eval_full` before `train`) and scrubs the foreign
  code directories from `sys.path`. `assert_isolated()` covers more modules and checks `sys.path`. There are 3 new tests
  (128/128 local and server).
- **Validity of earlier stages:** unaffected. stream_fit and pin_baseline never imported `train`. The 14/14 reused files are
  byte-identical to the GRU tree.
- **Records:**
  - `review/fo02/` (brief, evidence, the verbatim review, `FO02_REVIEW_PASS.json`);
  - the approved Review C files are preserved as `*.approved_ReviewC`;
  - `review/FEATURE_CODE_REVIEW_PASS.json` is unchanged.

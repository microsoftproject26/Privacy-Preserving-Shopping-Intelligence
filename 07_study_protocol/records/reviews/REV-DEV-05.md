# REV-DEV-05 — independent review of DEV-05 (Sonnet, read-only; transcribed by Ops from the reviewer's hand-back, 2026-10-02)

**Verdict: PASS.** No BLOCKER, MAJOR or MINOR findings.

- **Shas and diff:** the r7b shas match the DEV-04.md DEV-05 table. dump_rows_retro, bench and dump_rows were diffed against
  deploy_archive/r7/code_ds3r7.tar.
- **TEST rows:** the row selection is unchanged. `batch_view` is recorded before the filter without altering the batches. The
  byte-identity test was read in the handoff, not re-run.
- **View and binding:** the dump uses the shared `batch_view_of` and `view_binding_record`. `--ckpt` and `--family` are required
  together. The history policy and the MAIN manifest role are checked before any TEST file is opened.
- **Bench:** it refuses a binding with another checkpoint's sha256, and RETRO_TEST rows that are unbound or bound to another split.
- **v4eval:** all 15 .py files are identical to r7 (the evaluator sha was not re-hashed).
- **NCs:** 8 / 8 touched `test_nc_` fail under `--runxfail`. AssertionError was confirmed for 4 / 8, incl.
  test_nc_changing_batch_view_would_dump, so they are non-vacuous.
- **endgame/H_test.sh:128** passes `--ckpt "${P[1]}" --family SASREC`, consistent with the bench line.

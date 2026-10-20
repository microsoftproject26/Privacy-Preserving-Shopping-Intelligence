# PREG1 amendment 07 — addendum §B15: B13 rescinded, ML-1M Track B runs to completion (DRAFT 2026-10-03; owner decision)

- **Basis:** owner decision in chat on 2026-10-03, after Ops showed the schedule estimate. The owner wants the full ML-1M Track-B
  grid rather than the clock backstop. Fable review: company/reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md §7.
- **Status:** registered before 2026-10-06T00:00Z, the B13 trigger time, so the backstop has never fired. No benchmark holdout or
  TEST value has been read, and no ML-1M inner-validation value has been read either. Amendment 07 B1–B12 and B14 are unchanged.

**B15.1 B13 is rescinded.** Every ML-1M Track-B seed-2026 grid cell (C_FULL, PRE, FA and FA_WARM\*, as listed by run_wave1.sh) runs
to completion:
- trackb-select runs over the full registered grid (R5 / O1 rule, B10);
- the per-arm winners' seed-2027 runs complete as registered;
- the ML-1M LOCK and its single TEST look follow only after every cell has a RESULT.json.

The clock is no longer a stopping rule for ML-1M.

**B15.2 Mechanics.** Ops stops the armed b13_stopper process. Its .pid, .log and .nohup files are kept as evidence, and the script
stays read-only and is never run again for ML-1M. No run directory is touched. A cell that fails mechanically is resumed under the
runner's identical-hash resume rule. It is marked INCOMPLETE (B10 Q4) only by a new DL decision that records the cause, never on
a value.

**B15.3 Schedule (INFERENCE, not a result).** The estimate uses step rates measured 2026-10-03 ~15:15Z under the current CPU load:
- C_FULL L200 at budget 24 runs ≈ 0.17 EFE/h, about 6 days per cell;
- the seed-2026 grid should complete ≈ 2026-10-09/10;
- ML-1M final ≈ 2026-10-11 to 10-16, depending on which cell wins and its seed-2027 run.

The other three datasets, Wave 2 and REES46 are unaffected.

**B15.4 Disclosure.** The ML-1M LOCK records that B13 was registered (DL 368) and rescinded before its trigger (this addendum).

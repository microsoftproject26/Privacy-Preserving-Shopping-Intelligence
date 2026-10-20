# PREG1 amendment 07 — addendum §B15: B13 rescinded, ML-1M Track B runs to completion (REGISTERED 2026-10-03, DL 401)

- **Basis:** owner decision in chat on 2026-10-03, after Ops showed the schedule estimate. The owner wants the full ML-1M Track-B
  grid rather than the clock backstop. Fable review: company/reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md §7 (GO-WITH-EDITS,
  edits E1–E6 applied).
- **Status (MEASURED by Ops at registration, 2026-10-03T15:26Z):**
  - trackb-select has not run on ML-1M Track B: there is no runs_bench/ml_1m/trackb directory and no tbsel.*.ml_1m state file.
  - No Track-B RESULT.json or inner-validation value has been opened. The Ops progress checks read only directory listings and
    metrics.jsonl training-progress fields (efe, step_s, and the update-200 training loss), never an evaluation value.
  - The decision is schedule-only.
  - It is registered before 2026-10-06T00:00Z, the B13 trigger time, so the backstop never fired.
  - Amendment 07 B1–B12 and B14 are unchanged.

**B15.1 B13 is rescinded.** Every ML-1M Track-B seed-2026 grid cell (C_FULL, PRE, FA and FA_WARM\*, as listed by run_wave1.sh) runs
to completion:
- trackb-select runs over the full registered grid (R5 / O1 rule, B10);
- the per-arm winners' seed-2027 runs complete as registered;
- the ML-1M LOCK and its single TEST look follow only after every cell has a RESULT.json.

The clock is no longer a stopping rule for ML-1M.

**B15.2 Mechanics.**
- **Disarm:** b13_stopper was disarmed at 2026-10-03T15:26:18Z, before 2026-10-06T00:00Z, by stopping pid 859883 with SIGTERM.
- **Evidence kept:** runs_ds3/queues/b13_stopper.pid, b13_stopper.log, b13_stopper.nohup.
- **The script:** sha 3e4aa02b… stays read-only in company/plans/ops_scripts/bench/ and on the server, and is never run for ML-1M.
- **Run directories:** none is touched.
- **Failures:** a cell that fails mechanically is resumed under the runner's identical-hash resume rule. A cell that cannot be
  resumed with identical hashes becomes INCOMPLETE by a DL row that records the cause. It is listed in the ML-1M LOCK disclosure
  and in the Track-B table, and the per-arm winner is then chosen among the completed cells (B10 Q4). No cell is re-run with
  changed hashes, and nothing is ever decided on a value.

**B15.3 Schedule (INFERENCE, not a result).** The estimate uses step rates measured 2026-10-03 ~15:15Z under the current CPU load:
- C_FULL L200 at budget 24 runs ≈ 0.17 EFE/h, about 6 days per cell;
- the seed-2026 grid should complete ≈ 2026-10-09/10;
- ML-1M final ≈ 2026-10-11 to 10-16, depending on which cell wins and its seed-2027 run.

The other three datasets, Wave 2 and REES46 are unaffected.

**B15.4 Disclosure.** The ML-1M LOCK records this addendum's file sha and the sentence "B13 rescinded before its trigger on schedule
grounds; no value read."

**B15.5 Consequence for REES46.** K16 INT8 pass 2 (latency only, DL 368) waits for every Wave-1 driver to exit, so it now follows
the ML-1M grid (≈ 10-09/10, INFERENCE):
- the REES46 TEST look (H) is not gated on pass 2;
- the TEST bench's in-run latency stays loaded-host under the lock's `onnx_latency_rule`, and the thesis device-latency number is
  pass 2;
- H's CPU steps (base, onnx) run beside the 6-slot Wave-1 load as registered.

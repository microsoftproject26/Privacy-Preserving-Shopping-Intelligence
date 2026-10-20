# Gate P decisions (Opus lead), 2026-09-24

Source: `review/FABLE_GATE_P_RESPONSE.md` (verbatim). Verdict: **APPROVE_WITH_FIXES**, routing `claude-fable-5-1`, Read-only calls.
Every MUST_FIX is ACCEPTED. The table says how each one is implemented, and these rulings are binding for code, tests and the locked registry.

| # | Fable requirement | Decision | Implementation |
|---|---|---|---|
| 1 | D1 keep window-start positions; PAD contributes 0; assert position_ids-1 == cumsum(mask)-1 | ACCEPT | The model derives positions from the validity mask. It raises unless the mask is a contiguous prefix of `lengths`; the unpadded path carries no PAD rows at all. The optional forward kwarg `position_ids` is asserted to be `(position_ids-1) == arange` on valid cells and 0 on PAD whenever it is supplied: the trainer supplies it every update, and so do the tests. The reused audited evaluator does not pass it, and there the prefix check is the guard. Recency-anchored positions are future work. |
| 2 | Enforce TF32-off + deterministic + canary | ACCEPT | `precision.py` sets `matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `set_float32_matmul_precision("highest")`, `use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic=True`, `benchmark=False`, and asserts them at startup and before every checkpoint. The values go in config.json and each checkpoint. A canary test must detect TF32 when it is deliberately enabled. |
| 3 | Pair liveness status file + heartbeat + timeout ≥ 2× partner epoch time; never continue solo; pure joint rule tested incl. GRU trajectories | ACCEPT | `pair-dir/<run>.status.json` holds {RUNNING, EPOCH_DONE(e, record_sha256), STOPPED, FAILED} and a heartbeat. The waiting timeout is max(2 × the partner's measured mean epoch seconds, 6 h). When it expires, or the partner reports STOPPED/FAILED, the waiting arm writes a resumable checkpoint and exits 7. It never trains on alone. `pair_should_stop` is pure and unit-tested, including on the GRU F0/F4 series. |
| 4 | Resume stores u/epoch/cursor/RNG/optimizer; best/patience recomputed from persisted records; coverage bitset == N; u == 156,956 after epoch 1 | ACCEPT | Epoch-end records persist in `epoch_records.jsonl` and inside the checkpoint. On resume, StopState is rebuilt by replaying them through `epoch_end` and asserted equal to the stored state. A per-epoch coverage bitset (N bits) sits in the checkpoint and must be all-ones at epoch end. `u == epoch × U` is asserted at every epoch end. |
| 5 | Same-tensor self-term; eval dropout 0; int32 rank vector per epoch-end; fingerprint before comparisons | ACCEPT | The reused `rank_targets` gathers the target score from the same logits. Ranks are saved per epoch-end as compressed int32 `.npz` (≤ ~2 GB total, in the disk budget). Before any GRU/TRANS/POP comparison, the population fingerprint (decision_id sha, eval_mask, target_class, eval_user_key sha) must equal the archived one. |
| 6 | Logits exactly K; sum/n across accumulation; hand-formula test; init loss ≈ ln K | ACCEPT | `sasrec_objective.ce_sum` + `ce_sum_reference`. The ln K check is part of the smoke and test suites. |
| 7 | Diagnostic cohort is a fixed VS subset on a separate path; selection reads only epoch-end records; store refuses other splits | ACCEPT | The diagnostic uses the archived fixed 112,365-decision VS cohort, logged as `diag_eval` and never read by stopping or selection. The adapter refuses any split other than TRAIN/VALIDATION_SELECT outside `lock_retro.py`. |
| 8 | One rotation function, pre-write free-space check (refuse < 10 GiB, abort < 8); measure RSS of two rich processes before parallel rich replicas | ACCEPT | `ckpt_io.save_atomic` is the only writer. If free space after the write would fall below 10 GiB, it first reclaims allowlisted files (the `latest/previous.pt` of COMPLETED sasrec runs only). If it would still fall below 8 GiB, it refuses and the trainer exits 4, leaving a resumable checkpoint. The queue runs two rich replicas concurrently only if the measured RSS of the two leaves ≥ 12 GiB headroom; otherwise they run sequentially. |
| 9 | Registry fields + dated hash before the first real-data update | ACCEPT | `config/run_registry.locked.json` (see below) plus a sha256 receipt. |

Rulings adopted as-is:
- D2: explicit FP32 causal attention over unpadded ops. SDPA only if it passes the equivalence tests.
- D4: queue GPU0 A0→C0, GPU1 A1→C1→R0.
- D5: epoch-end full-VS eval, @1/2/5/10/20.
- D6: full-catalogue smoothing formula.
- D7: rich fusion with a mandatory zero-gate equivalence test.
- **D8: KEEP C0_WIDE384**, no replacement.
- D9: strict FP32.
- The schedule, stopping and selection rules stay UNCHANGED.
- Clarification: the first post-floor miss is judged at the epoch-4 eval against the running best, including pre-floor epochs.

Rejected: none.

The 8 must-fail tests from Fable are mandatory and are in `code/tests/` (owner Sonnet-tests, plus Sonnet-data for data parity). Each has a negative control that must really fail.

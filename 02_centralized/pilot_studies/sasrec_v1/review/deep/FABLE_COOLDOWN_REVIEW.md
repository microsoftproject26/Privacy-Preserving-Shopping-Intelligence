# Fable continue_cooldown.py review, verbatim (all passes)

- models: ['claude-fable-5-1']

ROUTING: claude-fable-5-1

VERDICT: FIX_REQUIRED

Correct as written [evidence]:
- lr swap `continue_cooldown.py:135-148`: parent `train_sasrec.py:705` resolves `sched.lr_at` on the module at call time, so the swap takes effect; restored in `finally`; only the fetch thread runs concurrently and never reads `lr_at`; `lr_phase` (`sasrec_schedule.py:75`) labels 4U<u as "floor", so `train_sasrec.py:721-724` hold. Formula/bounds match Q1 (`FABLE_HIGH_RECALL_RESPONSE.md:14`), lr(5U)=1e-6.
- Init `:93-132`: source `latest.pt` at end of epoch 4 has update=4U, epoch_idx0=3, offset=n, eval_pending=False, best incl. epoch 4 (`train_sasrec.py:763-775`); the asserts at `:101-103` match; perm=default_rng(seed+4); best_ckpt -> source `best_weights.pt`.
- Stop: `effective_epochs`=5 -> `fixed_epoch_should_stop(5,5)` True; `run():829-833` does not pre-finalize (4<5). Resume: parent replay over the persisted records equals `obj["best"]`; `u == 4U+ceil(offset/256)` holds. Attestation confirms crash/resume bitwise identity.

Defects:
1. `continue_cooldown.py:13` claims "the source run is never modified", but `save_latest`/`save_best` (`train_sasrec.py:438-449`) pass `runs_root=$S/runs`; `ckpt_io.py:176-178,139-147` deletes `latest.pt`/`previous.pt` of every COMPLETE run with `SUPERVISOR_DONE` (written on exit 0, `supervise_run.sh:125`) whenever free-after-write < 10 GiB. The A0/A1 source runs qualify, so a cooldown checkpoint can erase the source optimizer-state checkpoint the other arm (or a restart) needs [evidence]. Fix: in `setup()` raise `ConfigError` unless `ckpt_io.free_bytes(run_dir) >= TARGET_FREE_BYTES + 3*size(source latest.pt)` (or override `save_latest`/`save_best` with a runs_root outside `$S/runs`) [inference].
2. `continue_cooldown.py:159-169` drops the parent's guard `train_sasrec.py:1114-1117`; `--stop-after-updates` (os._exit(99)) is accepted on a production run [evidence]. Re-add the check.
3. `:97-105` verifies only registry sha, code map, counters and `registry_entry`; parent resume (`train_sasrec.py:563-571`) also refuses `dataset_version_hash`, `schedule`, `microbatch`, `n_train`, `channels`, `precision_training` drift. A flipped `bench/TF32_PARITY.json` would silently run epoch 5 at a different precision [inference]. Mirror those checks.
4. Minor: `eta()` (`train_sasrec.py:670`) uses `planned_epochs`=4 -> status.json reports 0 remaining hours; missing source `latest.pt` raises an uncaught FileNotFoundError (exit 1, not 5); `GATE_C_COOLDOWN_PASS.json` sha is not recorded in `config["cooldown"]`.

Operational note [inference]: with `--diag-every-updates 40000`, cohort diags fire at u=640,000/680,000/720,000/760,000 (4U=627,824), not at 4U+40k/80k; map the Q1 go/no-go gate to those.

After fixing 1-3, re-run `attest_cooldown.py` and issue `GATE_C_COOLDOWN_PASS.json` for the new sha; the lr/init/resume/stop logic needs no further change.

---

ROUTING: claude-fable-5-1

VERDICT: APPROVE_FOR_REAL_TRAINING

Approved sha256 of `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\code\continue_cooldown.py`: `2f33832fff1b9d57a79786a73d42e5de5398e6ae906fec1425af96f47453ae18` (the value recorded in `...\code\tests\evidence\COOLDOWN_ATTESTED_EVIDENCE.json:4`; I cannot hash files, so the lead must confirm the local file matches before writing `GATE_C_COOLDOWN_PASS.json`).

Fix verification [evidence]:
1. `:44,106-121,197-203`: every cooldown write passes `runs_root=NO_RECLAIM_ROOT`; `ckpt_io.reclaim` returns `[]` for a nonexistent root (`ckpt_io.py:129-130`), so no run's checkpoints can be deleted on the cooldown's behalf; `:79-86` disk precheck on fresh starts. Overrides keep the parent's precision assert, required keys and expected_update.
2. `:220-223` restores the parent's test-only guard; `guard_negative` rc 5.
3. `:146-152` compares all eight config keys plus the source dataset hash.
4. `:97-103` eta, `:81-82` missing-source ConfigError, `:90` gate-file sha recorded.
The unchanged lr swap, init, resume and stop logic remain as approved; 17/17 attested checks pass, source tree hash unchanged, resume bitwise-identical.

Residual notes, non-blocking [inference]: `finalize` (`:198-203`) runs outside the `run()` try/except, so a `DiskFloor` there exits as a crash, not code 4; the supervisor's resume path (`train_sasrec.py:829-833`) re-finalizes, so no data is lost. The `.hold` rename of source `SUPERVISOR_DONE` markers is an operational safeguard outside this file and is not part of this approval.

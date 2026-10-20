# Gate C brief (Opus → Fable 5.1): actual-code review BEFORE any real-data training

Task and output format: `F:\Queens University\Microsoft Project\eda\SASREC_PERFORMANCE_EXECUTION_V1\11_FABLE_CODE_REVIEW.md`.

Accepted plan:
- `review/GATE_P_DECISIONS.md` (Gate P fixes 1–9);
- `review/FABLE_GATE_P2_RESPONSE.md` (your deadline plan);
- the owner's choice "TF32 + 6 epochs", frozen in `config/run_registry.locked.json` **v2**.

v2 summary:
- A0_ID256 on GPU0 and A1_RICH256 on GPU1, seed 2026 only. No C0/C1/R0, no replicas.
- Schedule S′: warmup 2000; peak to U; cosine to 0.1× over epoch 2; floor from epoch 3.
- Fixed 6 epochs, i.e. 4 full low-LR epochs. No early stopping, no barrier.
- Full-VS eval only at epochs 3–6; best = raw max over evaluated epochs.
- A one-time, systems-only deadline cap to 5 or 4, applied to both arms identically.
- TF32 training; every evaluation in strict FP32.
- TF32 parity gate before launch (`code/tf32_parity_gate.py`). If it fails, the fallback is strict FP32 with 5 epochs.

All paths are under `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\`. The local mirror is byte-identical to the server: `lobot sync --dry-run` reports 0 differing files.

## Code to review (critical paths; hashes in `review/GATE_C_CODE_HASHES.json`)
| File | Owner | Critical content |
|---|---|---|
| `code/sasrec_model.py` | Opus | `_layout` (right-padding/position check), `_encode` unpadded + padded reference, `CausalBlock._attend/forward_tokens`, tied head `output_weight`/`forward`, rich residual gates, generator-scoped init |
| `code/sasrec_objective.py` | Opus | `ce_sum` (sum/count, smoothing), `ce_sum_reference` |
| `code/sasrec_schedule.py` | Opus | `lr_at`, `lr_phase`, `floor_start_epoch`, `eval_epochs`, `epoch_cap`, `fixed_epoch_should_stop`, `select_configuration` (StopState/individual/pair are the retained v1 functions, not used by the trainer) |
| `code/precision.py` | Opus | `configure_strict_fp32`, `training_precision`, `strict_eval`, asserts, `tf32_canary` |
| `code/train_sasrec.py` | Opus | Gate C firewall `gate_c_check`; `precision_plan`; exposure `epoch_permutation`/`Coverage`; `launch_asserts`; `build_optimizer`; `train_step` (microbatch sum/count, clip, u increment, lr); `epoch_end` (coverage, u==e·U, floor-phase counts, strict eval, best, cap); `init_state` resume (replay from persisted records); `run_benchmark` |
| `code/sasrec_data.py` | Sonnet-data | `predictor_kwargs` allowlist/forbidden keys; `check_right_padded`; `repack_right`; `build_training_store` / `build_eval_store` split restrictions |
| `code/ckpt_io.py` | Sonnet-runtime | `save_atomic` (partial→fsync→reload/validate→rotate), `check_disk_and_reclaim`, `reclaim` allowlist, `load_resume` |
| `code/lock_retro.py` | Opus | lock (O_EXCL, pair_check, integrity), eval (claim/resume, strict FP32, archived-rank fingerprint gate `verify_archived`, bootstraps) |
| `code/tf32_parity_gate.py` | Opus | pass rule |
| `code/queue_runner.py`, `code/supervise_run.sh` | Sonnet-runtime | queue, one-time `epoch_cap`, exit-code policy, resume bounds, pair comparability |
| `code/reuse/*` | GRU v1 (audited earlier, byte-identical, read-only) | `eval_full.rank_targets/aggregate/evaluate_model`, `full_store.CompactStore` |

## Independent test evidence (server `$S/tests/evidence/`, mirrored under `code/tests/evidence/` where present)
- `A_tests.json`: data adapter, 27/27 pass. Includes a real full-population check (40,180,608, ascending, unique, all loss_mask), the catalogue bijection from file, a real crop at >50, a real missing-brand/zero-price survival check, and a real shared-client-2 user check.
- `A_parity.json`: CompactStore vs the frozen loader on 4,210 FRESH decisions (seed 777, 8 buckets, edge cases): 0 mismatches, max float diff 0.0.
- `B_tests.json`: independent adversarial tests (attention, head/objective, schedule S′, evaluator/precision, rich, trainer exposure/resume), each with a mutant negative control. Numbers: see the file; it is being updated for v2 when you read this.
- `C_tests.json`: runtime, 17/17 pass (atomic checkpoint failure injection, reclaim allowlist, simulated quota termination, disk floor, bounded retries, epoch-cap orchestration, pair comparability).
- `OPUS_integration.json`:
  - tiny end-to-end A0/A1 runs, TF32 training / strict eval; the canary shows 2.9e-4 during training vs 5.7e-7 in strict phases;
  - a crash at update 37 then resume gives bitwise-identical best weights and metrics;
  - A0/A1 permutation digests are equal;
  - `lock_retro` mechanics: a fresh-process strict-FP32 reload reproduces the trainer's best metric exactly;
  - the bootstrap runs.
- `bench/SYSTEMS_EXPERIMENTS_PRE_GATE_C.md`: unpadded == padded reference (logits max |Δ| 0.0; grads ≤ 1.2e-6 rel). SDPA and bucketing were rejected.

## Owner directive (2026-09-23 ~23:05Z; owner now offline)
- **Delegation.** You act as the owner's delegate for in-scope decisions. The prohibited list is unchanged: no data/task/eval changes, no TEST tuning, no push, no teammate processes.
- **Hardware ladder.** Results are in `bench/SYSTEMS_EXPERIMENTS_PRE_GATE_C.md`. The directive's thread policy is implemented in `paths.py` and `train_sasrec.configure_threads`. The ETA from real throughput is in `Trainer.eta()`.
- **TF32.** It is decided by YOU from the evidence of the pre-registered gate `review/TF32_PARITY_PREREG.md` (written before any comparison). `tf32_parity_gate.py` writes only `bench/TF32_PARITY_EVIDENCE.json`; `bench/TF32_PARITY.json` (read by the trainer) is written from your ruling, in a separate short call after this review.

## Known limitations (disclosed, not hidden)
- The reused evaluator does not pass `position_ids`, so there the layout check relies on the mask-prefix test.
- The diagnostic cohort is the archived GRU 112,365-decision VS cohort (monitoring only).
- Dropout trajectories differ between A0 and A1; only the common-weight init and the data order are matched.
- The archived-rank fingerprint gate can only execute after the lock (VC/TEST are unavailable before it). It fails closed.

## What I need
Your verdict per 11_FABLE_CODE_REVIEW.md. Read the actual code; don't rely on this summary. Begin with `ROUTING: <model id>`. If something critical is missing, ask for it precisely.

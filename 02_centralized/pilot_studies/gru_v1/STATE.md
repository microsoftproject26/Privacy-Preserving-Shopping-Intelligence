# STATE.md — ContextGRU-v1

**Phase:** COMPLETE (2026-09-23). Final report `FINAL_FULL_GRU_REPORT_AR.md`, results `FINAL_FULL_GRU_RESULTS.json`.

## Milestones
- [x] FABLE REVIEW COMPLETE: APPROVE_WITH_FIXES (`FABLE_FULL_RUN_REVIEW.md`)
- [x] Scientific protocol frozen (`FULL_RUN_PROTOCOL.md` §1–12). No challenger arm (h512 skipped).
- [x] FULL DATA PATH VERIFIED: gates i/ii byte-exact (1,112,365 + 4,206 decisions), iii within 1.5e-8, store≡Parquet reader on full TRAIN fit + all VS rows (`tests_full/opus/integration.json`, `gates_A_rerun.json`)
- [x] TWO-GPU BENCHMARK COMPLETE: fused AdamW (+12.4%), CONCURRENT (ratios 1.002 / 0.997), protocol §13.1
- [x] FULL TRAINING STARTED 2026-09-23T00:51:20Z: F0 on GPU0 (tmux `gru-full-F0`), F4 on GPU1 (`gru-full-F4`); run dirs `experiments/full/<arm>/`
- [x] F0 COMPLETE 2026-09-23T17:11:30Z: 6 epochs (max), best epoch 5 = 0.17387 full-VS macro MRR@20, 0 restarts
- [x] F4 COMPLETE 2026-09-23T17:50:35Z: 6 epochs (max), best epoch 6 = 0.16600, 0 restarts (LR step fired only at the final epoch-end, so no epoch ran at 1e-4)
- [x] MODEL LOCKED 2026-09-23T17:51:17Z: selected F0_item_only_h256 (0.173874 > 0.165996), `locks/MODEL_LOCK.json`; integrity 409/409 re-verified just before
- [x] CONFIRMATION COMPLETE 2026-09-23T18:07:58Z (once): F0 0.1690 / F4 0.1564 / TRANS 0.1487 / POP 0.0202 macro MRR@20
- [x] TEST COMPLETE 2026-09-23T18:43:44Z (once): F0 0.1687 / F4 0.1565 / TRANS 0.1476 / POP 0.0211 macro MRR@20. Primary GRU−TRANS +0.0211 CI [0.0208, 0.0213] -> GRU BEATS TRANS. Secondary F4−F0 −0.0122 CI [−0.0124, −0.0119] (confounded by LR-step timing)
- [x] FINAL REPORT COMPLETE: `FINAL_FULL_GRU_REPORT_AR.md` + `FINAL_FULL_GRU_RESULTS.json`

## Current
- Runs: `experiments/full/F0_item_only_h256`, `experiments/full/F4_rich_side_information_h256` (status.json, metrics.jsonl, logs/supervisor.log, checkpoints/{latest,previous,best}.pt).
- Launched via `code/launch_full.sh <arm> <gpu> <run_dir> "--adamw-impl fused"`. The supervisor auto-resumes (≤5 restarts). To resume manually after a pod restart, re-run the same launch_full.sh command: it resumes from latest/previous.
- ~4,700 / ~4,500 dec/s → ~2.4–2.5 h per epoch + ~18 min full-VS eval. Max 6 epochs.
- Epoch 1 (full VS, macro MRR@20 end-to-end): F0 0.15437, F4 0.15479 (TRANS-full 0.16164). Accounting exact (156,956 updates, 40,180,608 exposures), same permutation sha both arms, n=8,826,488 / 790,459 users, closed n=8,776,943 (49,545 OOV kept at zero credit). Eval ≈ 17.5–18 min.
- Next after both stop: `code/lock_and_eval.py lock` → `confirm` → `test` → final report.
- Code: Wave 1 (Sonnet A/B/C) + Opus integration fixes; see `agent_reports/` and `tests_full/opus/`.
- Disk budget: `FULL_RUN_DISK_BUDGET.md` (24.15 GiB free at start; ≤ 3.6 GB planned new writes; floor 8 GiB).

## Owner standing instruction (2026-09-23)
Never stop on a blocker. Transient server/connection failures: retry with backoff. A real decision point: one concise Fable call, then act on its decision as the owner's and record it here. Hard limits still apply: teammates' resources, frozen data, no push, shared env, sealed TEST.

## Reference numbers (full VS, fit on all 40,180,608 TRAIN loss decisions)
- POP-full macro MRR@20 0.021579; TRANS-full 0.161638 (`tests_full/C/baselines_full_vs.json`)
- Full-VS GRU eval cost ≈ 18 min (8.3k rows/s, 2.5 GiB GPU)

## History
- Pilot result stays on record: NO_GO (F0 0.100795 < 0.9 × TRANS 0.13308), pilot stopped. See `SESSION_REPORT.md`.

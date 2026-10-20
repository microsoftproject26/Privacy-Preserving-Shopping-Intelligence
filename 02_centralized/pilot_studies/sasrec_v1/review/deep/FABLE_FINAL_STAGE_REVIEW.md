# Fable final-stage (lock_retro_v3b, ensembles, final_stage.sh) review, verbatim (all passes)

- models: ['claude-fable-5-1']

ROUTING: claude-fable-5-1

VERDICT: FIX_REQUIRED

Defects (blocking first):

1. `tools\campaign_v3b.py:197` + `code\lock_retro_v3b.py:51-62` — S4a will silently lack epoch 4. The keeper loops only over `A1_RICH256_s2027`; nothing writes `runs/A1_RICH256_s2026/checkpoints/epoch4_weights.pt`, and `release_completed` (:169-170) never fires without it. The weights exist in the source `latest.pt` (`model_state_dict`, end-of-epoch-4 verified by `continue_cooldown.py:143-145`), but `ensembles.load_state_dict_file` only unwraps `"state_dict"`. Amendment 2 says S4a = A1_s2026 {3,4,5}; with e4 absent (and e5 absent if cool5 fails) S4a degenerates to a single checkpoint = the headline re-evaluated on TEST. Fix: add the s2026 runs to the keeper list (orchestrator is idempotent; restart) or an explicit epoch-4 fallback in `epoch_weight_file`, and refuse an S4 entry with <2 epochs. [evidence]

2. `code\lock_retro_v3b.py:114-117` — `pair_check` compares the summary returned by `run_summary`, i.e. A1's cool5 (5 epochs) vs A0's base run (4 epochs) whenever A1 cool5 COMPLETES NO-GO and A0 cool5 is skipped. Then `comparable=False` and the pre-registered A1−A0 bootstrap is dropped silently (no `bootstraps_skipped` record, unlike `lock_retro.py:363-364`), although both bests come from the matched 4-epoch trajectories. Fix: compare the base summaries when both `best_epoch <= 4`; always record the skip reason. [evidence]

3. `code\lock_retro_v3b.py:41-48,68-73` — `run_summary` raises `SystemExit` if any of the three singles has no COMPLETE summary (e.g. A1_s2027 STOPPED_HARD_STOP or crashed). `select` then exits 2 (`final_stage.sh:11`) and nothing is locked/evaluated at 05:00Z unattended. `select_configuration` already supports ineligible rows; drop the dependent S1/S3/S4b with disclosure instead. [evidence]

Minor:
4. `:82-89` — no check that the summary-recorded `best_weights.sha256` equals the file hash (base `lock_retro.py:139-140` did). [evidence]
5. `:193-199` — no `DONE_PATH` short-circuit (base `:284-286`); manifest guards a re-look, so low risk. [evidence]
6. `:218` — A0 evaluated on the F4 store; it receives item kwargs only (`gru_v1/code/train.py:253-267`, `sasrec_model.py:147`), but F0/F4 item-token equality is not asserted (fingerprint covers population only). [inference]
7. `:142-151,305` — `--workers` is ignored (partition hard-wired to 2). Cosmetic. [evidence]

Verified OK:
- Selection: rows = {A0_s2026, A1_s2026, A1_s2027}, cool5 summary preferred iff COMPLETE (`:41-48`); `best_metric` is the raw max over evaluated epochs incl. 5 (`continue_cooldown.py:164`, `train_sasrec.py:737`); tie rule = existing (`sasrec_schedule.py:212-215`). [evidence]
- Lock before any VC/TEST row: O_EXCL write (`:137`); `build_eval_store` refuses without the lock (`sasrec_data.py:259-270`); entries fix S1–S3 members and S4 files with sha256 (`:100-113`); eval verifies hashes (`:164-166,174-176`). [evidence]
- Routing: `ensembles.py:34-48` exposes the union; `build_named_kwargs` therefore emits exactly `_RICH_KEYS`; A0 member gets 3 kwargs and would raise otherwise (`sasrec_model.py:366-369`); test `test_ensembles.py:32-45`. [evidence]
- Second look: claim keyed by lock sha, race-safe (`:181-190`); campaign never relaunches (`:259,276`); deterministic disjoint shards (`test_ensembles.py:63-72`). [evidence]
- Finalize: fingerprint equality vs F0 rebuild (`:261`), archived sha + exact-match gate (`:269-272`), bootstraps head−GRU, head−TRANS, A1−A0 if comparable (`:277-287`). [evidence]
- Strict FP32: `configure_strict_fp32` + `strict_eval` (`:200,222`; `precision.py:22-34,104-119`). [evidence]
- S4 epoch 5 source: `continue_cooldown.py:197-203`; epochs missing disclosed in the lock (`:110`). [evidence]

Missing context: server state of `runs/A1_RICH256_s2026/checkpoints/` (whether epoch 3/4 weight files were extracted manually) — defect 1 stands unless the lead confirms they exist.

---

ROUTING: claude-fable-5-1

VERDICT: FIX_REQUIRED

Remaining defect (blocking):

- `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\code\lock_retro_v3b.py:281-283` — `model` is never assigned in the revised `cmd_eval`. The `build_model(lock["entries"][n], lock, device)` call was lost in the edit; line 283 (`evaluate_model(model, ...)`) raises `NameError` on the first task of every worker, both `final_stage.sh` attempts fail, and `finalize` exits with "missing evaluations". No VC/TEST row is scored (the stores are built but nothing is evaluated), so this is a total-failure bug, not a leak. Fix: insert `model = build_model(lock["entries"][n], lock, device)` after the F0/F4 consistency check (:280) and before `t0 = time.time()` (:281). [evidence]

Verified fixed:
- Blocker 1: `MIN_S4_EPOCHS` guard and `dropped_entries` (:72, :137-139, :173). [evidence]
- Blocker 2: `pair_check` basis rules (:145-162) and `bootstraps_skipped` (:344-345). [evidence]
- Blocker 3: `run_summary` -> `(None, None)`, `unavailable_runs` (:80-89), dependent-system drops (:124-134), one-seed variation (:354-360). [evidence]
- Minor 4: summary sha vs file (:100-101). Minor 5: DONE short-circuits (:242-244, :308-310). Minor 6: item-only singles on F0 (:262), 20k-row F0/F4 equality before a mixed ensemble (:274-280). Minor 7: `--workers` `choices=[2]` (:374). [evidence]
- Epoch-4 weight files for the s2026 runs: accepted on your server listing [inference; not verifiable from the local mirror].

No other changes needed; the unit tests do not exercise `cmd_eval`, which is why 4/4 still pass.

---

ROUTING: claude-fable-5-1

VERDICT: APPROVE

Verified in the current `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\code\lock_retro_v3b.py`: `model = build_model(...)` restored at :280 after the F0/F4 check; `stores.clear()` at :297; device from `SASREC_EVAL_DEVICE` at :248 (default cuda:0); finalize records unavailable bootstrap pairs at :347-349 instead of raising; DONE short-circuits at :241-243 and :308-310. [evidence]

`code\tests\test_v3b_mechanics.py` runs the real `cmd_eval` for both shards and `cmd_finalize` on synthetic CPU stores with `build_eval_store`, lock/claim/done paths and `OUT` monkeypatched (:40-49); it checks shard completeness, rank equality against a direct `evaluate_model`, the S4 mean, worker-rerun idempotence, finalize outputs and the second-look refusal (:71-101). No VC/TEST access. [evidence]

No remaining defects. Non-blocking note: on real data a missing archived rank file would fail at :332-334 (loud), not fall into the new skip path — acceptable. [inference]

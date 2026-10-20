# Fable Gate C response (verbatim)

- agent type: sasrec-fable-reviewer (Read/Grep/Glob only)
- transcript model ids: ['claude-fable-5-1']

## First review

ROUTING: claude-fable-5-1

**VERDICT: FIX_REQUIRED** (evidence attribution, not a code defect; one correction-only review suffices)

**REVIEWED_CODE_HASHES**
Reference `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\review\GATE_C_CODE_HASHES.json` (23:18:37Z). I reviewed the local mirror in full for: train_sasrec.py (all 1129 lines), sasrec_model.py, sasrec_objective.py, sasrec_schedule.py, precision.py, sasrec_data.py, ckpt_io.py, lock_retro.py, tf32_parity_gate.py, queue_runner.py, supervise_run.sh, paths.py; reused originals eval_full.py, full_store.py (`decisions`/`batch`), train.py (`build_named_kwargs`, `VOCAB_SIZES`, `load_product_idx_of_class`). [inference] I cannot hash files myself; I rely on the brief's "0 differing files" statement for mirror identity.

**COVERAGE**
Reviewed [evidence]: repack/positions (`sasrec_model.py:207-238,272-275`; store layout `full_store.py:611-633`); QKV/causal mask incl. diagonal (`:85,:91-99,:168`); PAD never reaches real queries (unpadded `index_copy` + `o.index_select`, `:105-111`); last-real-token readout (`:284-285`); tied head as a view of one Parameter, logits exactly K (`:323-324,:343`; `train_sasrec.py:691`); one loss per frozen decision, sum/count across microbatches incl. the 128 tail (`train_sasrec.py:679-697`; `sasrec_objective.py:34-49`); lr set after `u+=1`, phase counters, `u==e·U`, floor-count assert (`:700-704,:713-719`); best = raw max, tie earlier (`:211-217`); coverage bitset and resume replay from persisted records incl. perm digest and cursor/u consistency (`:186-208,:544-606`); Gate C firewall and parity fallback (`:139-173`); exact rank with same-tensor self-term, OOV rank −1, end_to_end vs closed, macro by `eval_user_key` (`eval_full.py:89-103,:161-172`); strict-FP32 eval wrapper and per-phase asserts (`precision.py:81-119`; `train_sasrec.py:335,:728,:775`); F0/F4 allowlist + forbidden keys, split restrictions and lock-gated VC/TEST store (`sasrec_data.py:35-52,:250-273`); atomic checkpoint partial→fsync→reload→rotate and reclaim allowlist (`ckpt_io.py:108-118,:193-245`); lock O_EXCL, pair_check, integrity, claim/resume, archived fingerprint gate fails closed (`lock_retro.py:78-84,:145-171,:214-234,:269-279`); parity gate rule identical to the pre-registration incl. cohort band, same batches via `default_rng(12345)` (`tf32_parity_gate.py:93-102`; `train_sasrec.py:958`); queue one-time cap, exit-code policy, bounded retries (`queue_runner.py:334-362`; `supervise_run.sh:121-164`).
Not reviewed: synthetic_store.py (test-only but inside the firewall set), render_report.py, resources.py, bench_campaign.py, `evaluator._credits_from_ranks/_population_block` (byte-identical to the earlier audit), test source files (only their evidence JSON). Rich-feature causality: [inference] name lists (`config.py:400-429`) are prefix/"current-context-event" features with no target field; I rely on the GRU-project audit plus A_parity 0 mismatches, not a fresh derivation.

**BLOCKERS**
1. Evidence does not attest the code under review. `code/tests/evidence/B_tests.json:571,578` records `paths.py a87b7891…`, `train_sasrec.py e4da9847…`; `OPUS_integration.json:185,190,195` records the same plus `tf32_parity_gate.py da5f64a6…`. `GATE_C_CODE_HASHES.json:9,18,19` states `beaae271…`, `b4f6be21…`, `fae5d9c2…`. So the 7 trainer exposure/resume tests and the crash-at-37 bitwise-resume, permutation-digest and mechanics integration ran on an earlier trainer. `FULL_SUITE.log` ("112 passed") carries no hashes or timestamp; `C_tests.json` carries no hashes at all. [evidence] Why it matters: exactly the "unrecorded API/backend change" class; resume/exposure correctness of `fae5d9c2` is asserted, not evidenced. Minimum fix (no code change): re-run `tests/test_trainer_exposure_resume.py`, the tiny A0/A1 + `--stop-after-updates 37` resume + `lock_retro.py eval --mechanics-split` integration, and the full suite against the exact GATE_C hashes; regenerate B_tests.json, OPUS_integration.json, C_tests.json with `production_files_sha256` equal to GATE_C_CODE_HASHES.json (C_tests should also list ckpt_io.py, queue_runner.py, supervise_run.sh, resources.py). Then one correction-only call with the three files.

**NONBLOCKING** (≤5)
1. `queue_runner.py:456-472` — `_run_cmd(timeout=3600)` for lock (re-hashes the whole frozen dataset) and retro eval (2 models × VC+TEST at ≈5.6k rows/s plus F0/F4 store builds). [inference] Likely to exceed 1 h → rc 255 → REPORT with partial results; eval is claim/manifest-resumable, but the owner is offline. Raise to ≥6 h (file is outside the firewall set).
2. `train_sasrec.py:642-657` + `queue_runner.py:334-362` — cap race: if `epoch_cap.json` (=4) lands between the two arms' epoch-4 ends, one arm continues to 5 → pair INCOMPLETE (fails closed, never miscompared). [inference] Window ≈ one 60 s poll after both reach epoch 2; acceptable, log it.
3. `train_sasrec.py:66-67` — `REVIEWED_CODE` excludes lock_retro.py, tf32_parity_gate.py, queue_runner.py, supervise_run.sh; the lead should pin them in GATE_C_PASS.json and re-check at lock time (lock does record `code_sha256_map()`, `lock_retro.py:169`).
4. `precision.py:33` — `use_deterministic_algorithms(True, warn_only=True)` without `CUBLAS_WORKSPACE_CONFIG`; bitwise resume is evidenced empirically, not guaranteed. Setting it in `supervise_run.sh:110` avoids re-hashing firewalled files.
5. `sasrec_schedule.py:28` — legacy `FLOOR_START_EPOCH=4` is still the default for the retained v1 `epoch_end`; unused by the v2 trainer (`best_update`), keep documented.

**INDEPENDENT_TEST_EVIDENCE**
[evidence] A: 27/27 incl. real 40,180,608 population, catalogue bijection, >50 crop, shared client 2; A_parity 4,210 fresh decisions, 0 mismatches, max Δ 0.0. B: 68/68 with mutant negative controls listed (causal-mask disable, untie head, mean-of-means aggregation, warmup off-by-one, forced TF32, fingerprint mutations); S′ boundaries and fixed-epoch stop tested. C: 17/17 incl. failure injection, disk floor, bounded retries, cap orchestration. Integration: canary 2.9e-4 (training) vs 5.7e-7 (strict); crash/resume bitwise identical; fresh-process strict eval reproduces best exactly. All B/integration items for the trainer are attributed to the superseded hash (Blocker 1).

**NUMERICAL_AND_RESOURCE_NOTES**
[evidence] Unpadded == padded reference (logits Δ 0.0; grads ≤1.2e-6 rel); init loss ≈ ln K; canary threshold 2e-5 sits two decades from both regimes. TF32 remains parity-gated; `precision_plan` falls back to strict FP32/5 epochs on any non-`pass:true`. [inference] Checkpoint reload every 10k updates (~0.5 GB with AdamW states) is cheap; two resident CompactStores (F4 arm largest) must be confirmed against the 12 GiB headroom in the real-data benchmark before launch; `check_gpu_free` + `CUDA_VISIBLE_DEVICES` give per-arm isolation; no Git, secrets, or GRU-workspace writes found.

## Correction-only review

ROUTING: claude-fable-5-1

**VERDICT: APPROVE_FOR_REAL_TRAINING**

**Approved hash reference**
`training_code_sha256` in `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\review\GATE_C_CODE_HASHES.json` as of **2026-09-23T23:44:30Z** (train_sasrec.py `fae5d9c2…`, sasrec_model.py `d110b73a…`, sasrec_objective.py `5e03bf50…`, sasrec_schedule.py `5db6cede…`, precision.py `45212e6b…`, paths.py `beaae271…`, sasrec_data.py `e0872eaf…`, ckpt_io.py `51a9303d…`, synthetic_store.py `b8bf3a18…`, plus the 14 `reuse/*` hashes), registry `74414b38…`, prereg `3612bb87…`.

**Basis** [evidence]
- `code/tests/evidence/GATE_C_ATTESTED_EVIDENCE.json`: one run 23:26:53Z–23:33:44Z, `hashes_start == hashes_end` for every production, test, registry and prereg file, and every firewall-set hash equals the code I reviewed in the first pass and the regenerated manifest (`attested_hashes_equal_current: true`). Suite 112 passed / 0 failed / 0 skipped with a per-case junit table, now including `test_runtime.py` (17) on the revised queue_runner. Integration on the same hashes: A0/A1 COMPLETE; crash rc 99 at update 37 → `resumed from latest.pt u=37 epoch_idx0=2 offset=1280` → COMPLETE; best weights bitwise identical, metrics identical; perm digests equal; phases 4/12 · 16 cosine · 16 floor · 16 floor; canary 2.9e-4 vs 5.7e-7; mechanics fresh-process metric == trainer best; bootstrap present.
- Changed hunks verified and outside the firewall set: `queue_runner.py:464-479` (6 h lock, 8 h retro, one resumable rerun on non-zero rc); `supervise_run.sh:110-111` and `tf32_parity_gate.py:40-41` (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, thread env); `lock_retro.py:163-167` (records `runtime_code_drift_since_gate_c` from `pinned_runtime_sha256`, empty-safe if absent).
- [inference] The attestation script `tests/attest_gate_c.py` is self-reported by the implementing side; I accept it because the per-case table, start/end hashes and the CLI tails are internally consistent, and the firewall re-checks hashes at every launch.

**Remaining conditions** (binding)
1. `review/GATE_C_PASS.json` must carry `verdict: APPROVE_FOR_REAL_TRAINING` and a `code_sha256` map equal to the `training_code_sha256` block above including all `reuse/*` entries (the firewall at `train_sasrec.py:139-151` compares both); `pinned_runtime_sha256` = lock_retro.py `5672f6b1…`, queue_runner.py `9084c509…`, tf32_parity_gate.py `316e761f…`, supervise_run.sh `cf47c098…`, render_report.py `948eb14a…`, bench_campaign.py `5346af2c…`, resources.py `f181b19c…`.
2. This approval covers exactly these bytes. Any edit to a firewalled file re-opens Gate C; the firewall enforces it and no exception is granted. Runtime-file drift must appear in the lock's `runtime_code_drift_since_gate_c` and be disclosed in the report.
3. TF32 is still gated: run `tf32_parity_gate.py` (`bench/TF32_PARITY_EVIDENCE.json`), then my separate ruling writes `bench/TF32_PARITY.json`. No production `--precision-mode registry` launch before that file exists; a non-pass applies strict FP32 / 5 epochs automatically (`precision_plan`).
4. Before launching both arms, record the real-data concurrent benchmark's RSS for the F4 arm against the registry's 12 GiB headroom and the microbatch choice (256 vs 128×2) in `bench/`; systems-only, frozen before the first real update.
5. Accepted as documented: cap race (fails closed to pair INCOMPLETE), `FLOOR_START_EPOCH=4` legacy default, `A1_RICH256_s2026`/`A0_ID256_s2026` literals in `lock_retro.py:358`.
6. Claims remain narrowed per Gate P2: "trained TF32 / evaluated strict FP32" with the parity threshold stated, single seed with user-bootstrap CI only, fixed-budget training, VC/TEST labelled previously inspected, no attribution of SASRec-vs-GRU differences.

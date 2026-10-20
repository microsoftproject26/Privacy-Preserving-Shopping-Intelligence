# SASREC_INTERFACES.md — the single shared contract (owner: Opus lead)

This is the only shared contract between workers. Code against it. If you need a change, say so in your report instead of deviating silently. Opus integrates.

- Server workspace: `S=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1`, code in `$S/code`.
- Local mirror: `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\` (edit here, upload with `tools/lobot.py put <local> @W/<rel>`, where `@W` = `$S`).
- Python: `~/venvs/ppsi-gpu/bin/python` (torch 2.13.0+cu130, numpy 2.4.6, pyarrow 21.0.0, pytest 9.1.1). Never create a venv, never `uv run`, never pip install.
- The authoritative science is in `F:\Queens University\Microsoft Project\eda\SASREC_PERFORMANCE_EXECUTION_V1\` (03 architecture, 04 protocol, 05 tests, 07 systems) plus `review/GATE_P_DECISIONS.md` (Fable Gate P rulings, binding).

## 0. Reused audited modules (READ-ONLY, byte-identical copy of GRU v1)
`$S/code/reuse/`: `full_store.py` (CompactStore, DecisionSet), `eval_full.py` (rank_targets, aggregate, evaluate_model, per_user_metric, paired_bootstrap), `evaluator.py`, `train.py` (helpers: set_determinism, rng_state_snapshot, restore_rng_state, build_param_groups, load_product_idx_of_class, VOCAB_SIZES, build_named_kwargs, gather_*), `baselines_full.py`, `verify_frozen.py`, `lock_and_eval.py` (reference only), `vendor/rees46_eda/`.
Import them via `import paths` (see §1), then e.g. `from full_store import CompactStore`. Never edit, never copy-modify them. `REUSE_MANIFEST.json` holds their hashes.

Store batch facts (the reused `CompactStore.batch(ds, positions)`, CPU tensors):
- The layout is right-padded. Real events sit at columns `0..len-1` in chronological order, PAD=0 after.
- `position_ids` = 1..len on real events and 0 on PAD. `last_valid_index = len-1`.
- ID keys (channels="F0"): `item_tokens attention_mask position_ids lengths last_valid_index target_class loss_mask eval_mask decision_id client_id`.
- Rich keys (channels="F4") add:
  - `category_tokens brand_tokens main_category_tokens daypart_tokens weekday_tokens` (int64);
  - `event_numeric_features` float32 [B,50,19] and `event_quality_flags` int8 [B,50,15];
  - `user_context` float32 [B,12] and `user_context_masks` float32 [B,6].
- The training store uses `splits=("TRAIN","VALIDATION_SELECT")` only.

## 1. `paths.py` (Opus)
Constants `S, CODE, REUSE, DATASET_ROOT, VERSION_HASH, K=158486, ITEM_VOCAB=158489, L=50, N_TRAIN=40180608, N_VS=8826488, N_VS_USERS=790459, U_PROD=156956`. Importing it puts `REUSE` and `REUSE/vendor` on `sys.path` and sets thread env defaults (3).

## 2. `sasrec_model.py` (Opus)
```python
@dataclass(frozen=True)
class SASRecConfig:
    run_id: str; group: str          # "item_only" | "rich_frozen_channels"
    d_model: int; blocks: int; heads: int; ff_multiplier: int = 4
    dropout: float = 0.2             # input, residual, FFN-inner AND attention-probability dropout
    max_len: int = 50; ln_eps: float = 1e-5
    @classmethod
    def from_registry(cls, entry: dict) -> "SASRecConfig"

class SASRecCE(nn.Module):
    def __init__(self, cfg, *, vocab_sizes: dict, product_idx_of_class, seed: int, impl: str = "unpadded")
    # attributes read by reuse/train.build_named_kwargs (so the audited evaluator works unchanged):
    _categorical_channels: tuple[str, ...]   # ("item_tokens",) or item + 5 side channels
    numeric_dim: int (0|19); flags_dim: int (0|15); use_user_context: bool
    item_embed: nn.Embedding                 # [ITEM_VOCAB, d], padding_idx=0; the ONLY item Parameter
    output_bias: nn.Parameter                # [K]
    impl: str                                # "unpadded" (production) | "padded_reference" (tests)
    side_enabled: bool                       # rich only; False disables every side/user residual (tests)
    def forward(self, *, item_tokens, lengths, attention_mask, category_tokens=None, brand_tokens=None,
                main_category_tokens=None, daypart_tokens=None, weekday_tokens=None,
                event_numeric_features=None, event_quality_flags=None,
                user_context=None, user_context_masks=None) -> Tensor   # [B, K] logits
    def query(self, **same_kwargs) -> Tensor                           # [B, d], post final LN
    def hidden_states(self, **same_kwargs) -> Tensor                   # [B, T, d] after the last block, PAD rows 0
def build_sasrec(cfg, *, seed, device, impl="unpadded") -> SASRecCE     # loads the catalogue map + VOCAB_SIZES
def parameter_inventory(model) -> dict
```
Semantics (03 §2–5 + Gate P): logits `= query @ item_embed.weight[3:3+K].T + output_bias`. Causal attention reads only real positions ≤ own position. The query is taken at `lengths-1`. Targets, decision/client/eval ids and split names are never forward args.

## 3. `sasrec_objective.py` (Opus)
`ce_sum(logits, target_class, loss_mask, *, label_smoothing=0.0) -> (loss_sum, n_contrib)` selects `loss_mask & target>=0` and returns the SUM (divided once by the caller). Smoothing uses the full-K formula `(1-ε)·CE + ε·mean_j(-log p_j)`.

## 4. `sasrec_schedule.py` (Opus, pure functions)
```python
def lr_at(u: int, peak: float, U: int, warmup: int = 2000, floor_mult: float = 0.1) -> float   # u 1-based
def lr_phase(u: int, U: int, warmup: int = 2000) -> str   # "warmup"|"peak"|"cosine"|"floor"
@dataclass
class StopState: best_metric, best_epoch, post_floor_misses, history
def epoch_end(state, epoch: int, metric: float, *, floor_start_epoch=4, min_delta=2e-4) -> (StopState, dict)
def individual_should_stop(state, epoch, *, min_epochs=5, max_epochs=8, patience=2) -> (bool, str|None)
def pair_should_stop(state_a, state_b, epoch, *, min_epochs=5, max_epochs=8, patience=2) -> (bool, str|None)
def select_configuration(summaries: list[dict], registry_order: list[str]) -> dict
```
`epoch` is 1-based = the epoch that just completed. Best = raw max; an exact tie keeps the earlier epoch.

## 5. `train_sasrec.py` (Opus)
CLI: `python train_sasrec.py --run-id <registry id> --seed <int> --run-dir DIR [--resume] [--pair-dir DIR --pair-partner <run_id>] [--max-epochs 8] [--microbatch 256|128] [--ckpt-every-updates 10000] [--diag-every-updates 40000] [--benchmark-updates N] [--test-tiny]`
- One visible GPU via `CUDA_VISIBLE_DEVICES`.
- Exit codes:
  - 0 COMPLETE, 3 NONFINITE, 4 DISK, 5 CONFIG/ASSERT, 6 GPU_BUSY, 7 PAIR_PARTNER_STOPPED;
  - anything else is a crash, and the supervisor may resume it.
- Run dir:
  - `config.json` (resolved) and `status.json` (heartbeat ≤60 s: state, epoch, update, pid, lr, dps, disk);
  - `metrics.jsonl` and `summary.json` (final_status, best_epoch, best_metric, epochs_completed, exposures, low_lr_updates_by_epoch, stop_reason, history, checkpoint hashes, code hashes);
  - `checkpoints/{latest.pt, previous.pt, best_weights.pt}` and `logs/train.log`.
- Checkpoint I/O goes through `ckpt_io.py`.
- Pair barrier files: `<pair-dir>/<run_id>.epoch<e>.json`.

`summary.json` schema (strict JSON; readers must tolerate absent optional fields):
```
run_id, seed, group, config{...registry entry + implementation fields}, registry_sha256,
final_status: COMPLETE | STOPPED_NONFINITE | STOPPED_DISK | STOPPED_PAIR | RUNNING,
stop_reason: max_epochs | patience | pair_patience | null,
epochs_completed, updates, exposures, best_epoch, best_metric, micro_recall20_at_best, n_params,
epoch_records: [ {epoch, full_vs: {macro_mrr20, micro_mrr20, macro_recall20, micro_recall20, macro_ndcg20,
                   micro_ndcg20, at_k: {"1": {...}, "2": ..., "5": ..., "10": ..., "20": ...}, n, users},
                  lr_first, lr_last, updates_by_phase{warmup,peak,cosine,floor}, low_lr_updates,
                  train_loss_mean, epoch_seconds, eval_seconds, decisions_per_s_median,
                  peak_gpu_mem_allocated_bytes, peak_gpu_mem_reserved_bytes, rss_bytes_max,
                  coverage_ok, perm_sha256, stop_record{...} } ],
checkpoints: {best_weights: {path, sha256, epoch}, latest: {path, sha256, update}},
ranks: {"<epoch>": {path, sha256}}, code_sha256{file: sha}, precision{...},
start_utc, end_utc, resume_count, physical_gpu, wall_seconds
```

## 6. `sasrec_data.py` (Sonnet-data)
- `PREDICTOR_ALLOWLIST_ID` and `PREDICTOR_ALLOWLIST_RICH` hold the forward kwarg names.
- `FORBIDDEN_PREDICTOR_KEYS` = `target_class loss_mask eval_mask decision_id client_id eval_user_key user_bucket split prefix_position prior_view_count model_segment_id last_valid_index position_ids`. (The model derives positions itself, and tests assert them against `position_ids-1`.)
- `predictor_kwargs(batch, model) -> dict`: exactly the allowlisted keys the model declares. It raises on any forbidden key.
- `check_right_padded(batch) -> None`: raises unless each row's valid mask is a contiguous prefix of length `lengths[b]`, `position_ids == arange+1` on valid cells and 0 on PAD, `last_valid_index == lengths-1`, and every per-event side tensor is zero on PAD.
- `repack_right(batch) -> dict`: the generic left/scattered→right repack. It moves valid events to a contiguous prefix in order, applies the SAME permutation to every per-event tensor, and rebuilds positions/lengths/last index. Used by tests and for defensive checks; the store already emits the right layout.
- `build_training_store(channels) -> CompactStore` (TRAIN+VS only) and `build_eval_store(channels, split)` (retrospective only, VC or TEST).
- Parity script `tests/parity_store_vs_loader.py`. It checks ≥4096 fresh decisions over ≥8 buckets (seed ≠ GRU's), comparing the reused CompactStore with the frozen `DecisionWindowLoader.collate`, plus edge cases (len 1, 2, 49, 50, prefix>50, OOV item, missing brand/category, zero price, target_class=-1 with and without an observed target, shared client_idx=2 users). Output goes to `$S/tests/evidence/`.

## 7. Runtime modules (Sonnet-runtime)
- `ckpt_io.py`:
  - `save_atomic(obj, ckpt_dir, name, *, expected_update, reserve_bytes)`: write `.partial`, then flush+fsync, reload and validate keys/update, then rotate latest→previous and os.replace. Check disk before the write (free − estimate ≥ 8 GiB floor).
  - `load_resume(ckpt_dir) -> (obj, name)`: latest, else previous.
  - `save_weights(model_state, path)` and `sha256_file`.
- `resources.py`: cgroup (memory.max/current, cpu.max), disk free, per-GPU util/mem via nvidia-smi, own process RSS. `sample() -> dict`, `sampler_loop(out_jsonl, interval)`.
- `supervise_run.sh <run_id> <seed> <gpu> <run_dir> [extra trainer args]`:
  - resumes on crash up to 3 times, and passes `--resume` if latest/previous exists;
  - exit 0 → `SUPERVISOR_DONE`; exits 3/4/5/7 → `SUPERVISOR_STOPPED` with no retry;
  - exit 6 (GPU busy) waits 10 min and retries, at most 12 times.
- `queue_runner.py`: the detached state machine over `config/run_registry.locked.json` (04 phases, GPU plan in GATE_P_DECISIONS). State is atomic JSON in `$S/state/queue_state.json`, and a restart resumes from it. It launches supervisors in tmux sessions `sasrec-<run_id>-s<seed>`, calls `sasrec_schedule.select_configuration`, then replicas, then `lock_retro.py`, then `render_report.py`. There is no LLM between stages.
- `render_report.py`: `FINAL_SASREC_REPORT_AR.md`, `FINAL_SASREC_RESULTS.json` (strict JSON, no NaN/Inf) and `SOTA_CLAIM_STATUS.md` from result files (08 sections).

## 8. `lock_retro.py` (Opus)
It writes the new lock `$S/locks/SASREC_LOCK.json` (O_EXCL) with `test_exposure="previously_seen_in_GRU_project"`. It then runs one resumable retrospective VC+TEST evaluation of: the anchors' bests, the selected configuration's 3 seeds, and the GRU/TRANS/POP ranks reused after a fingerprint check. Paired user bootstraps. It never touches `gru_v1/locks`.
Output `$S/retro_eval/RETRO_RESULTS.json` (strict JSON):
```
{label: "PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK", lock_sha256,
 splits: {"VALIDATION_CONFIRM": {...}, "TEST": {n, users, oov, fingerprint}},
 models: {"<name>": {"VALIDATION_CONFIRM": HEADLINE, "TEST": HEADLINE, "source": "evaluated"|"archived_ranks"}},
 bootstraps: {"TEST": {"<a>_minus_<b>": {diff, ci95:[lo,hi], n_users, n_resamples, seed}}},
 seed_variation: {config, seeds:[...], test_macro_mrr20:{seed: v}, mean, std}}
HEADLINE = {end_to_end:{n, users, micro:{recall@k, mrr@k, ndcg@k for k in 1,2,5,10,20}, macro:{same}},
            closed:{same}, cohorts:{warm/unseen: {n, macro_mrr@20, micro_recall@20}}, slices:{...}}
```

## 9. Gate P deltas (binding; see `review/GATE_P_DECISIONS.md`)
- **Model positions.** `SASRecCE.forward/query/hidden_states` take an optional `position_ids` (store convention). When it is given, the model raises unless it equals 1..len on valid cells and 0 on PAD. The trainer always passes it; the reused evaluator does not.
- **`precision.py` (Opus).**
  - `configure_strict_fp32(seed)` seeds everything and turns TF32 off (matmul+cudnn), sets matmul precision "highest", and turns on cudnn deterministic, benchmark off and `use_deterministic_algorithms(True, warn_only=True)`.
  - `assert_strict_fp32()` raises if any of those is violated. `precision_record()` returns them.
  - `tf32_canary(device)` returns the relative error of an FP32 GEMM against FP64; the threshold is 2e-5.
- **`ckpt_io.save_atomic` disk policy.**
  - Estimate = the size of the file to be written. If `free − estimate < 10 GiB`, first call `reclaim(allowlist)`. It deletes ONLY `latest.pt`/`previous.pt` of runs whose run dir has `SUPERVISOR_DONE` and `summary.json` final_status COMPLETE, strictly under `$S/runs/`, and logs each deletion.
  - If `free − estimate < 8 GiB` after that, raise `DiskFloor` without writing; the trainer exits 4.
  - `best_weights.pt` and rank files are never reclaimed.
- **Pair status files** (`<pair-dir>/<run_id>.status.json`): `{state: RUNNING|EPOCH_DONE|STOPPED|FAILED, epoch, record_sha256, heartbeat_utc, mean_epoch_seconds}`, written by the trainer. The supervisor writes FAILED/STOPPED when it gives up.
- **Rank vectors.** The trainer saves `ranks/epoch<e>.npz` (int32, compressed) for full VS at each epoch end.
- **Fable's 8 must-fail tests.** They are mandatory and each needs a negative control that really fails. Owners:
  - #1 PAD/layout equivalence, #2 prefix consistency/causality, #3 tying, #6 objective, #7 rich: Sonnet-tests;
  - #4 schedule/stopping/selection: Sonnet-tests;
  - #5 exposure/coverage/resume: Sonnet-tests (trainer-level, after `train_sasrec.py` lands);
  - #8 evaluator/precision canary: Sonnet-tests;
  - data parity (05 A + E31): Sonnet-data;
  - runtime (05 E33–E37): Sonnet-runtime.

## 10. `synthetic_store.py` (Sonnet-data): real-data-free fixtures
Before Gate C nothing may update parameters on real data. Trainer and runtime tests therefore use a synthetic store with the SAME interface as `CompactStore`:
```python
class SyntheticStore:
    @classmethod
    def build(cls, *, channels: str, n_train: int = 4096, n_vs: int = 1024, K: int = 997,
              seed: int = 0, task: str = "mapping") -> "SyntheticStore"
    K: int; item_vocab: int                 # K + 3
    vocab_sizes: dict                       # item_tokens=K+3 plus small side vocabularies (>= 5 rows each)
    product_idx_of_class: np.ndarray        # arange(K) + 3
    version_hash: str = "SYNTHETIC"
    splits: tuple = ("TRAIN", "VALIDATION_SELECT")
    def decisions(self, split: str, kind: str)   # reuse full_store.DecisionSet, every field filled
    def batch(self, ds, positions) -> dict  # identical keys/dtypes/layout to CompactStore.batch(channels)
```
- `task="mapping"`: the target is a fixed bijective function of the LAST context item, so a correct model memorises it to >95% top-1 with dropout off. `task="random"`: random targets.
- Window lengths are drawn over 1..50 and include 1 and 50.
- VS includes users with several decisions, several distinct users sharing client_idx=2, OOV targets (target_class=-1 with eval_mask True), and rows with eval_mask False.
- The trainer's `--test-tiny` uses it with warmup scaled down (a CLI override allowed only with `--test-tiny`).

## Common rules
- Threads 3 per process (OMP/MKL/torch/pyarrow).
- No raw user_id on disk or in logs.
- VC/TEST are never loaded outside `lock_retro.py`.
- Never write in `/home/jovyan/shared_data/`. Never touch `~/ahmed`, `~/.vscode-tunnel-ahmed*`, tmux `tunnel-*`, `gru_v1/`, or any process you did not start.
- Write files with LF line endings (Windows `write_text` default is CRLF: use `newline="\n"` or bytes). Run `bash -n` on every uploaded .sh.
- Use `@W/...` remote paths with lobot.py (Git Bash mangles `/home/...`), or `MSYS_NO_PATHCONV=1`.

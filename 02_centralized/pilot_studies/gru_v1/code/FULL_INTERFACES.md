# FULL_INTERFACES.md — contract between the three Wave-1 modules (owner: Opus)

This is the only shared contract. Each Sonnet agent codes against it. If an agent needs to change it, it must say so in its report; it must not silently deviate. Opus integrates.

Server workspace: `W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1`
Code dir: `$W/code` (already on `sys.path` via each entrypoint, with `$W/code/vendor` for `rees46_eda`).
Local mirror: `F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1\code` (edit here, then upload with `tools/lobot.py put`).
Python: `~/venvs/ppsi-gpu/bin/python` (torch 2.13.0+cu130, numpy 2.4.6, pyarrow 21.0.0, scipy available). Never create a venv. Never use `uv run`.

## Frozen, read-only pilot modules (import them, never edit)
`model.py`, `objective.py`, `evaluator.py`, `window_cache.py` (`CacheReader`), `index_utils.py`, `train.py` (reuse `build_named_kwargs`, `build_param_groups`, `set_determinism`, `VOCAB_SIZES`, `load_product_idx_of_class`, the git/env/hardware helpers), `baselines.py`, `build_manifests.py`, `vendor/rees46_eda/*`.

## 1. `full_store.py` (owner: Sonnet A)

```python
EXPECTED_VERSION_HASH = "860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd"
L = 50

@dataclass
class DecisionSet:
    split: str                 # "TRAIN" | "VALIDATION_SELECT" | "VALIDATION_CONFIRM" | "TEST"
    kind: str                  # "loss" (loss_mask rows) | "eval" (eval_mask rows)
    decision_id: np.ndarray    # int64, strictly ascending = canonical order
    end_row: np.ndarray        # int64, global row of the decision's own event in the store
    length: np.ndarray         # int16/int32, window length = min(L, position_in_segment), 1..50
    target_class: np.ndarray   # int64, -1 = no target / OOV
    loss_mask: np.ndarray      # bool
    eval_mask: np.ndarray      # bool
    client_idx: np.ndarray     # int64 (2 = OOV / unseen client)
    user_bucket: np.ndarray    # int16
    eval_user_key: np.ndarray | None   # int64 for kind="eval": dense position of the row's user_id in the
                                       # sorted distinct user_id list over THIS split's eval_mask rows
                                       # (identical to build_manifests.build_vs_cohort). None for kind="loss".
    prior_view_count: np.ndarray       # int32, reporting-only activity slice: rint(expm1(raw log1p_prev_view_count))
    prefix_position: np.ndarray        # int32, raw position_in_segment (may exceed 50), reporting only
    def __len__(self) -> int
    def take(self, positions: np.ndarray) -> "DecisionSet"   # subset, keeps given order
    def sha256(self) -> str                                    # over decision_id bytes, order-sensitive

class CompactStore:
    @classmethod
    def build(cls, dataset_root: Path, *, splits: tuple[str, ...], channels: str,   # "F0" | "F4"
              buckets: list[int] | None = None, context_length: int = 50,
              n_threads: int = 3, log=print) -> "CompactStore": ...
    context_length: int
    channels: str              # "F0" stores item tokens only; "F4" stores every channel
    splits: tuple[str, ...]    # rows of these splits are loaded; decisions only for these splits
    n_rows: int
    version_hash: str          # read from VERSION.json and asserted == EXPECTED_VERSION_HASH
    def decisions(self, split: str, kind: str) -> DecisionSet
    def batch(self, ds: DecisionSet, positions: np.ndarray) -> dict[str, torch.Tensor]   # CPU tensors
    def decision_item_tokens(self, ds: DecisionSet) -> np.ndarray
        # int64 [len(ds)]: item token of each decision's OWN event
        #   = batch(...)["item_tokens"][i, lengths[i]-1]
        #   = the "last context item" that TRANS conditions on
```

Semantics of `batch(ds, positions)` for decisions `ds.take(positions)`:
- Output keys, dtypes and values are **identical** to `window_cache.CacheReader.read()` output for the same decisions. That is itself identical to `DecisionWindowLoader.collate()` after the int32→int64 casts:
  - `item_tokens category_tokens brand_tokens main_category_tokens daypart_tokens weekday_tokens position_ids lengths last_valid_index target_class client_id decision_id` → int64
  - `attention_mask loss_mask eval_mask` → bool
  - `event_numeric_features` float32 `[B,L,19]`, `event_quality_flags` int8 `[B,L,15]`, `user_context` float32 `[B,12]`, `user_context_masks` float32 `[B,6]`
  - Right padding, zeros beyond `lengths`.
- For `channels="F0"` only these keys are present: `item_tokens attention_mask position_ids lengths last_valid_index target_class loss_mask eval_mask decision_id client_id`.
- `client_id` = the index `client_idx` (asserted never 0).
- **Rows of splits not in `splits` are never loaded.** A training store uses `splits=("TRAIN","VALIDATION_SELECT")`. Only the locked final evaluator builds a store with VC/TEST.

## 2. `eval_full.py` (owner: Sonnet C)

```python
def rank_targets(scores: torch.Tensor, target: torch.Tensor, class_chunk: int = 32768) -> torch.Tensor
    # scores [B,K] float32 OR int64 on any device; target [B] int64 >= 0; returns [B] int64 ranks,
    # exactly evaluator._ranks_chunked semantics (1 + #greater + #equal-with-lower-class-id).
def aggregate(ranks: np.ndarray, ds: DecisionSet, k: int = 20) -> dict
    # ranks int64 per ds row (-1 where no rankable target); returns the SAME structure as evaluator.evaluate()
    # (k, end_to_end, closed, user_count, cohorts.warm/unseen) computed via evaluator's own helpers,
    # plus "slices": {"prior_views": {...}, "prefix_len": {...}} (diagnostic).
def evaluate_model(model, store, ds, device, *, eval_batch: int = 1024, k: int = 20,
                   return_ranks: bool = False) -> dict
    # model.eval() + no_grad; CPU/CUDA RNG state saved and restored; restores model.training;
    # iterates ds in canonical order in chunks of eval_batch via store.batch; forward via
    # train.build_named_kwargs; ranks on the GPU; returns aggregate(...) (+ "ranks" if requested).
def per_user_metric(ranks: np.ndarray, ds: DecisionSet, metric: str = "mrr", k: int = 20,
                    denominator: str = "end_to_end") -> tuple[np.ndarray, np.ndarray]  # (user_keys, per-user mean)
def paired_bootstrap(user_keys_a, vals_a, user_keys_b, vals_b, n: int = 1000, seed: int = 2026) -> dict
    # same user set required; returns {"diff": a-b, "ci95": [lo, hi], "n_users": ...}
```

## 3. `train_full.py` (owner: Sonnet B)

CLI: `python train_full.py --arm {F0_item_only_h256|F4_rich_side_information_h256} --run-dir DIR [--resume] [--max-epochs 6] [--adamw-impl foreach|fused] [--benchmark-updates N] [--test-tiny ...]`
- The GPU is chosen by the environment: `CUDA_VISIBLE_DEVICES` is set by the supervisor, and the trainer asserts exactly 1 visible device.
- It consumes `CompactStore` / `DecisionSet` / `eval_full.evaluate_model` only through the signatures above.
- Behaviour is fully specified by `FULL_RUN_PROTOCOL.md` §4–§6, §11, §12.

## Wave-1 resource partition (to avoid collisions)
| Agent | GPU | Server scratch dir (own it, clean it) | Log prefix |
|---|---|---|---|
| A | none (`CUDA_VISIBLE_DEVICES=""`) | `$W/tests_full/A/` | `$W/logs/A_*` |
| B | GPU 0 only (`CUDA_VISIBLE_DEVICES=0`) | `$W/tests_full/B/` (test checkpoints; keep ≤ 3 GB, delete own files when done) | `$W/logs/B_*` |
| C | GPU 1 only (`CUDA_VISIBLE_DEVICES=1`) | `$W/tests_full/C/` | `$W/logs/C_*` |

## Common rules
- Threads: set `OMP_NUM_THREADS=3`, `MKL_NUM_THREADS=3`, `torch.set_num_threads(3)`, `pyarrow.set_cpu_count(3)`, `pyarrow.set_io_thread_count(3)` in every process.
- No raw `user_id` ever written to disk or logs. `eval_user_key` integers are fine.
- VALIDATION_CONFIRM / TEST decisions are never loaded in Wave 1 (not even in tests).
- Never write inside `/home/jovyan/shared_data/`. Never touch `~/ahmed`, `~/.vscode-tunnel-ahmed`, tmux sessions `tunnel-*`, or any process you did not start.

# T-K / T-L result report

Scope: **implementation only**. The pilot (T-13, F0-then-F4 over 2 passes)
was **not** launched. No acceptance threshold was changed. All work ran on
Lobot, GPU 0 only (`CUDA_VISIBLE_DEVICES=0`), after confirming both A16s
were idle (`nvidia-smi` showed 0 MiB used / no processes, checked
immediately before every GPU run, see commands below).

Files delivered (all in the ALLOWED FILES list):

- `code/train.py` -- T-K training engine (checkpoint/resume).
- `code/baselines.py` -- T-L POP/TRANS baselines.
- `code/test_resume.py` -- T13 resume-determinism tests.
- `artifacts/BASELINES.json` -- T-L's data output (mirrors the existing
  `BENCHMARK.json`/`PILOT_MANIFEST.json` convention: emitted on Lobot,
  mirrored to the local `artifacts/` directory).

---

## 1. T-K training engine -- design summary

- **Optimizer**: `AdamW(lr=1e-3, betas=(0.9,0.999), eps=1e-8)`, two explicit
  param groups built from `p.ndim >= 2` (every Linear/Embedding/GRU weight
  matrix -> `weight_decay=1e-5`; every bias AND LayerNorm param, all
  `ndim==1` -- gets `weight_decay=0`). Same rule `benchmark.py:make_optimizer`
  already used; not reinvented.
- **Grad clip**: global `clip_grad_norm_(model.parameters(), 1.0)`, applied
  AFTER the loss-accumulator division (see below), BEFORE `optimizer.step()`.
- **LR schedule**: `lr_at_update(u) = 1e-3 * min(1, u/100)` -- 100-update
  linear warmup then constant, keyed off the GLOBAL update counter (not
  reset per pass).
- **Determinism**: `set_determinism(2026)` seeds python/numpy/torch/cuda,
  sets `cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `allow_tf32=False` (both matmul and cudnn), and additionally calls
  `torch.use_deterministic_algorithms(True, warn_only=True)`. This last
  call is not in the task brief's literal list but was necessary in
  practice: CUDA embedding-backward is only deterministic under duplicate
  indices (very likely inside a 50-token window) with this flag; without it
  T13 could plausibly fail the 1e-4 tolerance. `warn_only=True` means an op
  with no deterministic kernel warns instead of hard-failing. No warnings
  were observed in any run (checked by grepping both full logs for
  "warn"/"Warning": 0 matches).
- **Loss reduction**: `objective.exact_ce_loss` -> `(loss_sum, n)`, wrapped
  in one `objective.LossAccumulator` per update. Gradients are accumulated
  as a raw, UNSCALED sum (`loss_sum.backward()`), then every parameter's
  `.grad` is divided ONCE by the accumulated `n` before clipping -- this is
  mathematically identical to `(loss_sum/n).backward()` but is written to
  generalise correctly to `accumulation > 1` (sum/sum, divide once, never a
  per-microbatch mean). `accumulation != 1` raises `NotImplementedError`
  since BENCHMARK.json only measured/authorises accumulation=1.
- **`assert loss_mask.all()`** on every TRAIN microbatch, before the model
  ever sees it.
- **Data order**: `pass_permutation(pass_index)` = one permutation of
  `arange(1_000_000)` from `np.random.default_rng(2026 + pass_index)` --
  a pure function of `pass_index` alone (F0 and F4 arms, run independently,
  see identical order by construction; a correct resume reconstructs the
  in-progress pass's order from nothing but the checkpointed `pass_index`).
- **Exposure**: `consumed_decisions` counts every row DRAWN from the
  manifest (not just loss-contributing rows), incremented by the exact
  microbatch size every update; a full run reaches `passes * 1_000_000`
  exactly.
- **Checkpoint contents** (`save_checkpoint`): model `state_dict`, optimizer
  `state_dict`, RNG snapshot (`random.getstate()`, `np.random.get_state()`,
  `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`),
  `pass_index`, `offset_in_pass`, `update_number`, `consumed_decisions`,
  `manifest_sha256`, `version_hash`, `config`, `best_metric`. Written
  atomically (temp file + `Path.replace`). `last.pt` is overwritten every
  `checkpoint_every` updates and at pass end; `best.pt` is written only
  when a pass-end VS evaluation improves the selection metric.
- **Resume**: loads `checkpoints/last.pt`, asserts
  `manifest_sha256`/`version_hash` match the current environment (refuses
  to resume against a different dataset/manifest), restores model,
  optimizer, and all RNG state, then reconstructs the current pass's
  permutation from the checkpointed `pass_index` and continues from
  `offset_in_pass`.
- **Run directory** (`experiments/gru/<run_id>/`): `config.yaml`,
  `environment.json`, `git_state.json`, `hardware.json`, `metrics.jsonl`
  (one JSON object per update: `update, pass_index, loss, lr, grad_norm,
  batch_size, consumed_decisions, seconds` -- no raw user identifiers ever
  logged), `summary.json`, `checkpoints/`, `logs/train.log`. Verified
  present below (section 4).
- **Eval hook**: after each pass, `run_vs_evaluation` scores the VS cohort
  (`cache/vs`, 112,365 rows) through `evaluator.evaluate`, with
  `eval_user_key` joined positionally from `validation_select_cohort.parquet`
  (cache row order == manifest row order, asserted). Selection metric =
  **Client-Macro MRR@20**, resolved as `end_to_end.macro["mrr@20"]` --
  see "unresolved concerns" below for why `end_to_end` (not `closed`) was
  chosen. This code path is implemented but NOT exercised by T13 (a
  320-update run never reaches a pass boundary at ~3907 updates/pass); see
  concerns.

## 2. T-L baselines -- design summary

Both fitted **only** on `cache/train` (the 1,000,000-row TRAIN pilot
manifest); VS cohort (`cache/vs`) is read only for evaluation.

- **POP**: `pop_count = bincount(target_class[target_class>=0], minlength=K)`.
  Scored by broadcasting the same `[K]` vector to every row (a zero-copy
  expanded view, never materialised as `[n,K]`).
- **TRANS**: last context item TOKEN = `item_tokens[i, last_valid_index[i]]`
  (not the target class). Rows with `target_class<0`, or last-item ==
  PAD(0) or OOV(2), are excluded from fitting; a `scipy.sparse.csr_matrix`
  of shape `(158489, K)` holds `trans_count[src_token, target_class]`. At
  eval time: `score = trans_dense * (max_pop+1) + pop_count`, computed in
  int64 then cast to float32 for `evaluator.evaluate` (which hard-codes
  float32). The "fallback when the last item was unseen or is OOV(2)" rule
  falls out for free: an excluded/absent source token's `trans_matrix` row
  is all-zero, so the composite collapses to `pop_count` alone.
- **Exactness check**: `max_composite_score = 6,362,279` (measured:
  `max_pop=4,931`, `max_trans=1,289`), which is `<= 2^24 = 16,777,216`, so
  every composite score is exactly representable in float32 -- the
  "ordering is exact" requirement holds for real, not just by assumption.
  Recorded in `BASELINES.json.composite_score_float32_exact = true`.
- Both use the identical `evaluator.evaluate(..., k=20)` call and therefore
  the identical tie rule as the GRU.

## 3. T13 resume-determinism -- actual numbers

Test: `code/test_resume.py`, group `F0_item_only` (cheapest arm; T13 tests
checkpoint/resume mechanics, not model quality). Run A: uninterrupted,
320 updates. Run B: 300 updates, checkpoint, resume, continue to 320.
Losses at updates 301-310 (from `metrics.jsonl`, run confirmed twice,
identical both times -- values below are from the second, exit-code-verified
run):

| update | full run loss | resumed run loss | relative diff |
|---|---|---|---|
| 301 | 10.786445617675781 | 10.786445617675781 | 0.0 |
| 302 | 10.743846893310547 | 10.743846893310547 | 0.0 |
| 303 | 10.983940124511719 | 10.983940124511719 | 0.0 |
| 304 | 11.182132720947266 | 11.182132720947266 | 0.0 |
| 305 | 11.026067733764648 | 11.026067733764648 | 0.0 |
| 306 | 11.06169319152832  | 11.06169319152832  | 0.0 |
| 307 | 10.912307739257812 | 10.912307739257812 | 0.0 |
| 308 | 10.59906005859375  | 10.59906005859375  | 0.0 |
| 309 | 10.922122955322266 | 10.922122955322266 | 0.0 |
| 310 | 10.795821189880371 | 10.795821189880371 | 0.0 |

**Max relative difference: 0.0** (bit-identical, well under the required
1e-4). `experiments/gru/t13_comparison.json` holds this table; reproduced
identically across two independent full test-session runs.

**Broken variant** (`_break_data_order_restore=True`: resume reconstructs
the pass permutation with an unseeded RNG instead of `seed 2026 +
pass_index`): `test_t13_resume_broken_data_order_xfails` is marked
`xfail(strict=True)` and **XFAILed** on both runs (`1 passed, 1 xfailed`),
i.e. its internal assertion `max_rel <= 1e-4` genuinely failed as required
(strict xfail would itself fail the suite if the assertion had
unexpectedly passed). Measured divergence: max relative diff
**0.0330** (run 1) / **0.0370** (run 2) -- three orders of magnitude over
the 1e-4 tolerance, confirming the test has real power to catch this class
of bug rather than passing vacuously. Sample (run 1,
`experiments/gru/t13_broken_comparison.json`): update 301 full=10.786446,
broken-resumed=11.068045 (rel 0.0261); update 307 full=10.912308,
broken-resumed=10.559317 (rel 0.0323, the run's worst).

## 4. Commands run, with exit codes

All via the Lobot Jupyter-kernel bridge (`rx`), from
`/home/jovyan/venvs/ppsi-gpu/bin/python`, `CUDA_VISIBLE_DEVICES=0`.

| # | Command | Exit code |
|---|---|---|
| 1 | `nvidia-smi` (pre-flight, both GPUs) | 0 -- both idle, 0 MiB used, no processes |
| 2 | sync `train.py`, `baselines.py`, `test_resume.py` to Lobot `code/` (`sync_kernel.py`) | "synced 3/3 files" |
| 3 | `python code/baselines.py` (1st attempt) | crashed: `TypeError: 'generator' object is not callable` in `run_baseline` (bug: pre-invoked the `scores_fn` factory before handing it to `evaluate`, which itself calls `scores_fn()`) -- **fixed**, re-synced |
| 4 | `python code/baselines.py` (2nd attempt, unverified exit code) | ran to completion, wrote `BASELINES.json` |
| 5 | `python code/baselines.py` (3rd, exit-code-verified: `...; echo EXITCODE=$?`) | **`EXITCODE=0`**, identical numbers to run 4 (deterministic fit/eval) |
| 6 | `nvidia-smi -i 0` (pre-flight before GPU test) | 0 -- GPU 0 idle |
| 7 | `python -m pytest code/test_resume.py -v -s` (1st run) | `1 passed, 1 xfailed in 76.94s` |
| 8 | `python -m pytest code/test_resume.py -v -s` (2nd, exit-code-verified) | **`EXITCODE=0`**, `1 passed, 1 xfailed in 75.82s`, identical loss numbers to run 7 |

## 5. Run-directory listing (all required files present)

`experiments/gru/test_resume_full/` (uninterrupted 320-update run):

```
config.yaml            497 B
environment.json       292 B
git_state.json         164 B
hardware.json          137 B
metrics.jsonl        59,908 B  (320 lines, one JSON record per update)
summary.json            177 B
checkpoints/last.pt  250,790,611 B
logs/train.log          184 B
```

`experiments/gru/test_resume_interrupted/` (300 then resume to 320):
same file set, `checkpoints/last.pt` (250,790,675 B, saved twice: once at
update 300 before the interruption, once at update 320), `logs/train.log`
417 B (two "fresh"/"resumed" lines). `test_resume_broken_data_order/` is
identical in structure (428 B log).

`summary.json` for every run: `{"group": "F0_item_only", "update_number":
320, "pass_index": 0, "offset_in_pass": 81920, "consumed_decisions": 81920,
"best_metric": null, "target_update": 320}` -- `best_metric: null` /
no `checkpoints/best.pt` is EXPECTED here: `run_eval=False` was passed
(and even with it True, 320 updates never reaches a pass boundary at
~3,907 updates/pass), so the pass-end eval hook never fires in this test.

`environment.json` (representative): `torch==2.13.0+cu130`, CUDA 13.0,
cuDNN 92000, Python 3.11.14, `NVIDIA A16`, `cuda_visible_devices: "0"`.
`git_state.json`: `head=bfc90a0ad5dae255f8cb49ca49431aea6dbb1cac`,
`dirty=false` (`.local_ai/` is git-excluded per `DECISIONS.md` D1, so these
new files never show as repo-dirty). `hardware.json`: 256 CPUs, both A16s
listed via `nvidia-smi --query-gpu`.

## 6. Baseline metrics (real numbers, from `artifacts/BASELINES.json`)

Fit: `n_fit_rows=1,000,000`, `max_pop=4,931`, `max_trans=1,289`,
`max_composite_score=6,362,279` (float32-exact), `n_trans_pairs_fit=998,897`,
`n_trans_nonzero_src_tgt=663,944`. Eval: VS cohort `n_vs_rows=112,365`,
`k=20`, `user_count=10,000` for both baselines on both denominators.

**End-to-end denominator:**

| metric | POP micro | POP macro | TRANS micro | TRANS macro |
|---|---|---|---|---|
| Recall@20 | 0.048903 | 0.086198 | 0.285302 | 0.348277 |
| MRR@20 | 0.011450 | 0.021886 | 0.106394 | 0.133082 |
| NDCG@20 | 0.019507 | 0.035816 | 0.146380 | 0.181336 |

**Closed-catalogue denominator** (`n=111,866`, `user_count=9,987`):

| metric | POP micro | POP macro | TRANS micro | TRANS macro |
|---|---|---|---|---|
| Recall@20 | 0.049121 | 0.086431 | 0.286575 | 0.349116 |
| MRR@20 | 0.011501 | 0.021938 | 0.106868 | 0.133416 |
| NDCG@20 | 0.019594 | 0.035907 | 0.147032 | 0.181782 |

`BASELINES.json` additionally carries the warm/unseen cohort cuts for both
baselines and both denominators (not reproduced here for brevity). TRANS
clearly dominates POP on every metric, as expected for a first-order model
on next-item prediction.

## 7. Unresolved concerns

1. **"Client-Macro MRR@20" (T-K's selection metric) is not defined
   verbatim anywhere read** (task brief, FABLE_PLAN.md). I resolved it as
   `evaluate(...)["end_to_end"]["macro"]["mrr@20"]` -- macro (grouped by
   `eval_user_key`, i.e. per evaluated "client") MRR@20 on the end-to-end
   denominator -- matching FABLE_PLAN.md sec. F's explicit "Baseline
   comparison (Micro+Macro MRR@20, **end-to-end denominator**)" convention
   used for GO-1/GO-2. It could instead mean the `closed` denominator; the
   pilot task should confirm before this metric gates any best-checkpoint
   selection or go/no-go decision.
2. **The eval hook and `best.pt` checkpointing are implemented but never
   exercised by T13** (a 320-update test run is far short of one pass, and
   `run_eval=False` was passed regardless, to keep the test fast and
   GPU-cheap). They are structurally exercised by the SAME code path as the
   training loop (same `run_training` function), but their first real test
   will be the pilot itself -- worth a quick smoke check (e.g. a ~1-pass
   dry run) before trusting `best.pt` selection in the actual F0/F4 runs.
3. **`accumulation > 1` is implemented (sum-then-divide-once gradient
   accumulation) but explicitly rejected at runtime** (`NotImplementedError`)
   since BENCHMARK.json only measured/authorises microbatch 256,
   accumulation 1. If E1's mitigation branch (microbatch 128, accumulation
   2) is ever needed, this path exists but is untested.
4. **The deliberately-broken T13 variant only breaks the FIRST pass
   reconstructed after resume** (a targeted, single-point-of-failure bug
   injection at the resume call site). If a future run crosses a pass
   boundary after such a resume, subsequent passes revert to the correctly
   seeded permutation. This matches the literal "data-order RNG not
   restored on resume" framing but is narrower than "every subsequent pass
   is also unseeded" -- noted for anyone reusing `_break_data_order_restore`
   for a longer-running scenario.
5. **`torch.use_deterministic_algorithms(True, warn_only=True)`** was added
   beyond the task brief's literal determinism list (seed, cudnn.deterministic,
   allow_tf32=False) because CUDA embedding-backward needed it in practice
   for T13's 1e-4 tolerance to hold with duplicate indices inside a window.
   No warnings were observed in either run's log, but this should be
   re-verified if the pilot uses a different PyTorch/CUDA/cuDNN version than
   this environment's `torch==2.13.0+cu130` / CUDA 13.0 / cuDNN 92000.
6. **First-update loss (13.31) is noticeably above `ln(K) ≈ 11.97`**
   (untrained-CE reference from FABLE_PLAN.md sec. F). This is plausible
   given random weight init (the tied output head is not exactly uniform at
   init) and is outside T-K/T-L's scope (the learning-signal-floor check is
   a pilot acceptance criterion, not a unit-test requirement here) -- flagged
   for awareness before the pilot runs, not treated as a bug.

# T-M result — GRU pilot execution and go/no-go evidence

Scope: **execution and measurement only**. This report does not make the
GO/NO-GO call — it applies the frozen criteria (DECISIONS.md addendum,
task brief) mechanically to the measured numbers. No threshold was
re-derived, tuned, or reinterpreted after seeing results.

Both pilot arms (`F0_item_only`, `F4_rich_side_information`) ran to
completion on Lobot, GPU 0 only (`CUDA_VISIBLE_DEVICES=0`), using
`code/train.py`'s `run_training` (T-K engine, unmodified, imported) via
the new orchestrator `code/run_pilot.py`. `code/pilot_report.py` computed
every criterion from the recorded outputs and wrote
`artifacts/PILOT_RESULTS.json`.

## 0. Files delivered (ALLOWED FILES only)

- `code/run_pilot.py` — orchestrates both arms: pre-flight `nvidia-smi`
  check, calls `train.run_training` unmodified, peak-VRAM/RSS sampling,
  P3 double-eval, per-user end-to-end MRR@20 reconstruction (reuses
  `evaluator._ranks_chunked` / `_credits_from_ranks` unmodified).
- `code/pilot_report.py` — reads both arms' `metrics.jsonl` /
  `summary.json` / `resource_usage.json` / `p3_double_eval.json` /
  `per_user_mrr_end_to_end_best.json`, applies the frozen thresholds
  mechanically, writes `artifacts/PILOT_RESULTS.json`.
- `artifacts/PILOT_RESULTS.json` — full per-arm/per-pass/both-denominator
  metrics, cohort cuts, mechanical PASS/FAIL per criterion with numbers,
  GO-1/GO-2, bootstrap CI.
- `artifacts/PILOT_RUN_META.json` — raw run summaries + resource usage
  (mirrored from Lobot).
- Outputs on Lobot: `experiments/gru/T-M_F0_item_only/`,
  `experiments/gru/T-M_F4_rich_side_information/` (each with
  `config.yaml`, `environment.json`, `git_state.json`, `hardware.json`,
  `metrics.jsonl`, `summary.json`, `checkpoints/{last,best}.pt`,
  `logs/train.log`, `p3_double_eval.json`,
  `per_user_mrr_end_to_end_best.json`, `resource_usage.json`).

## 1. Pre-flight and environment

- `nvidia-smi` immediately before the pilot launch: GPU 0 **0 MiB used,
  0 compute processes** ("No running processes found"). `run_pilot.py`
  additionally re-checks this itself at process start and aborts if any
  compute process is present on GPU 0 (none was).
- `environment.json` (both runs, identical): `torch==2.13.0+cu130`, CUDA
  13.0, cuDNN 92000, Python 3.11.14, `NVIDIA A16`,
  `cuda_visible_devices: "0"`.
- `git_state.json` (both runs): `head=bfc90a0ad5dae255f8cb49ca49431aea6dbb1cac`,
  `dirty=false`.
- `hardware.json`: 256 CPUs, both A16s listed (15356 MiB each).
- `config.yaml` (both runs) matches the frozen config exactly: microbatch
  256, accumulation 1, lr 1e-3, betas (0.9,0.999), eps 1e-8, weight_decay
  1e-5, warmup 100, grad_clip_norm 1.0, seed 2026, train_rows 1,000,000,
  `manifest_sha256=adbdc827cb48376f28ee4c781adabd3b7cebf1d0c1033f4a9658f77d063bcdba`
  (matches `PILOT_MANIFEST.json`), `version_hash=860e5a1c…` (matches).

## 2. Bug found and fixed during execution

The first pilot launch crashed immediately after pre-flight, inside
`run_arm`, at `torch.cuda.reset_peak_memory_stats(device)` →
`RuntimeError: Invalid device argument`. Root cause (reproduced in
isolation): the CUDA context had not yet been initialized — calling
`reset_peak_memory_stats` as the very first CUDA op on some torch/driver
combinations fails this way; `torch.cuda.init()` beforehand fixes it.
Fix applied to `run_pilot.py` (one line, `torch.cuda.init()` right after
device selection, before any memory-stat call); re-synced; re-ran from a
clean `nvidia-smi`-verified-idle GPU. No run directories existed yet from
the crashed attempt (it failed before `init_run_dir`), so nothing needed
cleanup beyond re-verifying idle state. This is the only code defect
found during T-M; it is purely measurement-harness code (not `train.py`
or any existing-work file).

A pilot dry run (1 pass, F0 arm only, in a throwaway
`experiments/gru/T-M_smoke_test/` directory, deleted before the real
runs) was executed first to validate the full pipeline — training,
checkpointing, pass-end eval, P3 double-eval, per-user reconstruction —
end to end before committing to the real 2-pass/2-arm run. It passed:
P3 identical, per-user-reconstructed macro MRR@20 matched the official
evaluator value to full float precision, exit code 0.

## 3. Commands run, with exit codes

| # | Command | Result |
|---|---|---|
| 1 | `nvidia-smi` (pre-flight, GPU 0) | 0 MiB used, no processes |
| 2 | `sync_kernel.py code/ → Lobot code/` | "synced 22/22 files" |
| 3 | `nohup .../python /tmp/smoke_test.py` (1-pass F0 dry run) | exit 0, `SMOKE TEST OK`, P3 identical=True, per-user consistency_ok=True |
| 4 | removed `experiments/gru/T-M_smoke_test/` | cleaned |
| 5 | `nohup .../python code/run_pilot.py` (1st real attempt) | **crashed**: `RuntimeError: Invalid device argument` in `torch.cuda.reset_peak_memory_stats` |
| 6 | fixed `run_pilot.py` (+`torch.cuda.init()`), re-synced 22/22 files | ok |
| 7 | re-verified no stray run dirs; `nvidia-smi` GPU 0 idle | confirmed |
| 8 | `nohup .../python code/run_pilot.py` (2nd attempt) | **exit 0**, both arms complete, `PILOT_RUN_META.json` written |
| 9 | `.../python code/pilot_report.py` | **exit 0**, `PILOT_RESULTS.json` written |

## 4. Per-arm, per-pass metrics (end-to-end denominator, k=20)

| Arm | Pass | micro R@20 | micro MRR@20 | micro NDCG@20 | macro R@20 | macro MRR@20 | macro NDCG@20 |
|---|---|---|---|---|---|---|---|
| F0 | 1 | 0.184613 | 0.064283 | 0.090914 | 0.242391 | 0.083424 | 0.118674 |
| F0 | 2 | 0.242380 | 0.081747 | 0.117270 | 0.296207 | **0.100795** | 0.144173 |
| F4 | 1 | 0.206256 | 0.064829 | 0.095922 | 0.270588 | 0.087367 | 0.127815 |
| F4 | 2 | 0.261202 | 0.079591 | 0.119508 | 0.316218 | **0.101160** | 0.148727 |

(closed-catalogue denominator, warm/unseen cohort cuts, both denominators
for both cuts: full numbers in `artifacts/PILOT_RESULTS.json` — e.g. F0
pass 2 closed macro MRR@20 = 0.101071, unseen macro MRR@20 = 0.110379,
warm macro MRR@20 = 0.097542; F4 pass 2 closed macro MRR@20 = 0.101399,
unseen 0.110080, warm 0.098132. Not reproduced in full here for brevity.)

Both arms improved from pass 1 to pass 2 on every metric shown — no
collapse, no plateau at the untrained level.

## 5. Baselines (measured, T-L, unchanged)

| | micro R@20 | micro MRR@20 | macro R@20 | macro MRR@20 |
|---|---|---|---|---|
| POP | 0.04890 | 0.01145 | 0.08620 | 0.02189 |
| TRANS | 0.28530 | 0.10639 | 0.34828 | **0.13308** |

## 6. Mechanical PASS/FAIL — engineering criteria

All numbers below are read directly from `artifacts/PILOT_RESULTS.json`.

| Criterion | F0 | F4 | Numbers |
|---|---|---|---|
| **P1** both arms finish 2 passes, exactly 2,000,000 decisions, zero non-finite losses | **PASS** | **PASS** | F0: `pass_index=1, update_number=7814=target_update, consumed_decisions=2000000, n_nonfinite_losses=0/7814`. F4: identical structure, `consumed_decisions=2000000, n_nonfinite_losses=0/7814`. |
| **P3** two evaluations of the same (best) checkpoint bit-identical | **PASS** | **PASS** | F0: eval_a=eval_b=`0.10079484432935715` (macro MRR@20, full float repr, exact equality). F4: eval_a=eval_b=`0.10115985572338104`. |
| **P5** `max_memory_reserved` ≤ 8 GiB; host RSS ≤ 24 GB | **PASS** | **PASS** | F0 peak reserved = `2.00391 GiB`; F4 peak reserved = `2.01563 GiB` (both ≪ 8). Peak host RSS (process-wide, monotonic, both arms share one process) = `9729.11 MB` ≈ `9.73 GB` (≪ 24 GB / 24,000 MB). |
| Learning-signal floor: EMA(200) at end of pass 1 ≤ EMA(first 200) − 1.0 nat | **PASS** | **PASS** | F0: EMA(first 200)=`12.15389`, EMA(end pass1)=`8.87722`, Δ=`3.27667` (≥1.0). F4: EMA(first 200)=`12.24808`, EMA(end pass1)=`8.56135`, Δ=`3.68674` (≥1.0). EMA defined as a running exponential moving average, α=2/201, seeded at the first pass-1 loss, evaluated at update 200 and at the last pass-1 update (3907) — see `pilot_report.py:ema_series`. |
| Overfitting check: pass-2 micro MRR@20 ≥ pass-1 − 0.002 | **PASS** | **PASS** | F0: pass1=`0.064283`, pass2=`0.081747` ≥ required min `0.062283`. F4: pass1=`0.064829`, pass2=`0.079591` ≥ required min `0.062829`. (Pass 2 improved on pass 1 in both arms — no degradation of any kind, let alone beyond tolerance.) |

## 7. Mechanical PASS/FAIL — baseline gates (GRU-F0, best checkpoint = pass 2)

**GO-1** (must hold for BOTH micro and macro) — **PASS**:

| Sub-check | Value | Threshold | Result |
|---|---|---|---|
| micro Recall@20(F0) | 0.242380 | ≥ 0.07335 | PASS |
| macro Recall@20(F0) | 0.296207 | ≥ 0.12930 | PASS |
| micro MRR@20(F0) | 0.081747 | > 0.01145 | PASS |
| macro MRR@20(F0) | 0.100795 | > 0.02189 | PASS |

**GO-2** (macro MRR@20(F0) vs TRANS = 0.13308) — **NO_GO**:

- macro MRR@20(F0, best checkpoint) = **0.100795**
- `≥ 0.13308` → APPROVED (not met)
- `0.11977 ≤ x < 0.13308` → APPROVED_WITH_ONE_CHANGE (not met — 0.100795 < 0.11977)
- `< 0.11977` → **NO_GO** (met)

Mechanical verdict: **GO-2 = NO_GO**. This is the raw number produced by
the frozen rule; I do not reinterpret it further — the go/no-go call
itself is Opus's, not mine.

## 8. F4 vs F0 paired bootstrap (informational — never gates the GRU decision)

Paired bootstrap over the 10,000 cohort users, 1,000 resamples, seed 2026,
on macro MRR@20(end-to-end) difference, both arms' **best** checkpoints
(F0 pass 2, F4 pass 2):

- point estimate (mean per-user F4−F0 difference): **+0.000365**
- 95% CI: **[−0.001005, +0.001698]**

The interval straddles zero — F4's richer side information produced no
detectable macro-MRR@20 lift over F0 at this exposure budget (2M
decisions/arm). Consistent with the aggregate numbers (F4 pass 2 macro
MRR@20 = 0.101160 vs F0 = 0.100795 — a 0.000365 gap, the same number the
bootstrap point estimate reproduces exactly, as it should since the
bootstrap mean over all 10,000 users converges to the same aggregate
difference).

## 9. Training wall-clock, throughput, peak VRAM, peak host RSS

| Arm | Train wall-clock | Post-arm analysis wall-clock | Total wall-clock | Throughput (train) | Peak VRAM reserved | Peak host RSS |
|---|---|---|---|---|---|---|
| F0 | 613.89 s | 143.92 s | 757.81 s | 3257.9 decisions/s | 2.00391 GiB | 9729.11 MB (measured after F0's post-arm analysis) |
| F4 | 630.09 s | 138.83 s | 768.91 s | 3174.2 decisions/s | 2.01563 GiB | 9729.11 MB (unchanged — process-wide peak already reached during F0) |

"Post-arm analysis" = P3 double-eval (2 full-catalogue forward passes over
the 112,365-row VS cohort) + per-user end-to-end MRR@20 reconstruction (1
more full-catalogue forward pass), all on the arm's best checkpoint.
Total pilot wall-clock (both arms, sequential on GPU 0, single process):
**757.81 + 768.91 = 1526.72 s ≈ 25.4 minutes.**

Peak host RSS is process-wide (both arms ran in one Python process,
sequentially) and is monotonic non-decreasing across the process
lifetime by construction (`ru_maxrss`); it plateaued at 9729.11 MB by the
end of F0's post-arm analysis and did not grow further for F4 — this is
the correct, most conservative reading of "peak host RSS during the
pilot," well under the 24 GB bound.

## 10. Correctness cross-checks performed

- Per-user MRR@20 reconstruction (used for the bootstrap) is built by
  reusing (not modifying) `evaluator._ranks_chunked` /
  `_credits_from_ranks` and mirroring `_population_block`'s per-user
  grouping. Its aggregate (mean of the 10,000 per-user means) was
  cross-checked against the official `evaluator.evaluate()` macro MRR@20
  for the same checkpoint: F0 reconstructed=`0.10079484` vs
  official=`0.10079484` (exact match); F4 reconstructed=`0.10115986` vs
  official=`0.10115986` (exact match, matching the `P3_double_eval` A/B
  values too).
- `manifest_sha256` / `version_hash` in every run's `config.yaml` match
  `artifacts/PILOT_MANIFEST.json` exactly — both arms trained against the
  identical, frozen 1,000,000-row TRAIN manifest and 112,365-row VS
  cohort.
- Data order: `pass_permutation(pass_index)` (train.py, unmodified) is a
  pure function of `pass_index` and the global seed 2026 — both arms
  therefore saw identical per-pass row order by construction, not by
  post-hoc comparison.

## 11. Unresolved concerns / notes for the reader

1. **GO-2's middle band ("APPROVED_WITH_ONE_CHANGE") explicitly allows
   extending to 3 passes or changing LR as the single permitted lever.**
   F0's pass-2 macro MRR@20 (0.100795) sits below even that middle band's
   floor (0.11977), so the frozen rule resolves cleanly to NO_GO without
   needing to weigh which of those two levers would apply — that
   ambiguity never arises here.
2. Both arms were still improving substantially from pass 1 to pass 2
   (F0: +0.0174 macro MRR@20, F4: +0.0138) with no sign of plateauing.
   Whether that trend would close the gap to TRANS (0.13308) with more
   passes/exposures is exactly the kind of question the frozen GO-2 rule
   defers to Opus's judgment, not something I extrapolate here.
3. F4 vs F0's near-zero bootstrap gap (CI spanning zero) at 2M exposures
   does not by itself mean richer side information is useless — it may
   need more exposure to pay off, or the gain may genuinely be small at
   this feature-group choice. This bootstrap is explicitly informational
   per the frozen criteria and was not used to decide anything.
4. Peak host RSS (9729 MB) is noticeably higher than the ~2.7–3 GB seen
   in earlier tasks' manifest-building steps; this is expected since the
   pilot process loads the full VS cache into memory multiple times
   (official pass-end eval + P3's two evals + per-user reconstruction, x2
   arms) rather than once, and holds model/optimizer/checkpoint state
   concurrently. It is still under half the 24 GB bound, so this is noted
   for awareness, not flagged as a problem.
5. No threshold, denominator choice, or checkpoint-selection rule was
   changed to produce this result. GO-2 is NO_GO as measured; this report
   states that plainly per the task brief's instruction that "a failing
   pilot that is honestly reported is a correct outcome."

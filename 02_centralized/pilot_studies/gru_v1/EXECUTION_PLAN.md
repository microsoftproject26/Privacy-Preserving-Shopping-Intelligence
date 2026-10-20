# EXECUTION_PLAN.md — ContextGRU-v1 pilot

Authoritative plan = FABLE_PLAN.md as amended by DECISIONS.md. This file is the task queue only.

## Environment / workflow (read once, then follow)

- **Code is written LOCALLY** at `F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1\code\`
  (real editor tools, fast iteration). Local env: torch 2.9.1+cpu, numpy 2.4.1, pyarrow 22.0.0, pytest 9.1.1.
- **Code is RUN ON LOBOT** at `/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1/code/`.
  Lobot env `~/venvs/ppsi-gpu/bin/python`: torch 2.13.0+cu130 (CUDA, 2×A16), numpy 2.4.6, pyarrow 21.0.0, yaml 6.0.3.
- **Helpers** (in the session scratchpad):
  - `sync_kernel.py <local_dir> <abs_remote_dir>` — push code to Lobot (use `MSYS_NO_PATHCONV=1`; the Contents API cannot write dot-paths, this uses the kernel channel).
  - `rx "<python code>"` — execute python on Lobot; wrap shell with `subprocess.run(..., shell=True, capture_output=True, text=True)`.
- **Never** modify `/home/jovyan/shared_data/rees46_nextitem_v1` (read-only), the shared venv, `~/ahmed`, or anything under `ppsi/`.
- Tests that need neither the frozen data nor a GPU run locally first (fast); everything else runs on Lobot.
- Large output goes to `.local_ai/gru_v1/logs/<name>.log`; report ≤30 lines + exit code + the log path.

## Task queue (Opus-ordered; Fable's numbering in brackets)

| # | Task | Needs server data? | Depends on |
|---|---|---|---|
| T-A [5] | `ContextGRU-v1` model module, spec-driven from FEATURE_GROUPS.json; class-map validation w/ gather fallback; parameter-inventory printout | no | — |
| T-B [7] | Exact-CE objective (mask + accumulation-correct denominator); sampled-CE behind a default-off flag | no | — |
| T-C [8] | Evaluator: chunked exact rank, tie rule, both denominators, macro by `eval_user_key`, cohort cuts | no | — |
| T-D [6,10] | Model/objective/evaluator tests T1,T4,T5,T6,T7,T8,T9,T12 incl. strict-xfail broken variants | no | T-A,T-B,T-C |
| T-E [1] | Vendor loader (4 files byte-identical) + `VENDOR_MANIFEST.json`; assert `layout=="frozen"` on the server copy | yes | — |
| T-F [2] | Cross-platform batch digest T2 (Windows vs Lobot) — version skew is real (torch 2.9 vs 2.13, pyarrow 22 vs 21) | yes | T-E |
| T-G [3] | Projected-index utils; 1M TRAIN manifest (`loss_mask` only, seed 2026); 10k-user VS cohort w/ dense `eval_user_key`; `PILOT_MANIFEST.json` | yes | T-E |
| T-H [4] | Bucket-major window cache builder (bounded RSS, evict per bucket, memmap, digest header) + T3 parity | yes | T-G |
| T-I [10] | Synthetic memorisation T12 (F0 and F4 wiring) on GPU | no (GPU yes) | T-A,T-B |
| T-J [11] | Benchmark script → JSON; mechanically evaluate **E1/E2** and print the verdict | yes | T-H,T-A,T-B |
| T-K [12] | Training loop: AdamW groups, warmup, clip, accumulation, checkpoint/resume, metrics.jsonl + T13 | yes | T-J |
| T-L [9] | POP and TRANS baselines on the same manifest, through the same evaluator | yes | T-G,T-C |
| T-M [13] | Pilot runs: F0 then F4, same seed/order, eval after each pass | yes | T-K,T-L |
| T-N [14] | Go/no-go report generator applying §F criteria literally + paired bootstrap | yes | T-M |
| T-O [15] | Repair `post_freeze_smoke.py:101` with T11 (EDA workspace, outside frozen data) | yes | T-E |
| T-P [16] | Targeted reproducibility checks (read-only) — before the full run, not before the pilot | yes | T-E |

Gate order: **T-D and T-I must pass before T-M.** **T-J's E1/E2 verdict must be recorded before T-M.**
Full training is authorised automatically only if every P1–P5 + GO-1 + GO-2 criterion in FABLE_PLAN.md §F passes.

## Sonnet task template

```
TASK: <id + one line>
PURPOSE:
ALLOWED FILES:            (write only inside .local_ai/gru_v1/code/)
READ-ONLY REFERENCES:
MUST IMPLEMENT:
MUST NOT CHANGE:          frozen dataset, shared venv, ppsi/, teammate files, acceptance thresholds
TESTS TO RUN:             (command + expected)
OUTPUT REPORT PATH:       .local_ai/gru_v1/task_<id>_result.md
```

Result file must contain: files changed · commands run with exit codes · test counts pass/fail ·
performance numbers if applicable · unresolved concerns. No "tests passed" without the numbers.

# Gate P2: owner budget change (Opus → Fable 5.1), 2026-09-24

The owner rejected the 5–7 day plan. **New hard requirement: the whole campaign must finish within 38 hours from now**, including the remaining engineering, Gate C, benchmark, training, the retrospective evaluation and the report. The owner asked you to propose the improvements that make this possible, with the best achievable quality inside it. No real-data training has happened yet and no validation score exists, so re-scoping now is pre-registration, not outcome-driven.

Context you already reviewed: `review/GATE_P_BRIEF.md` and your `review/FABLE_GATE_P_RESPONSE.md`. The package files in `F:\Queens University\Microsoft Project\eda\SASREC_PERFORMANCE_EXECUTION_V1\` (03, 04, 07, `config/run_registry.json`) stay authoritative except where the owner's deadline forces change. Read nothing else.

## Fixed time costs (estimates)
- Before launch: ≈ 3–3.5 h. Workers finish, integration tests, Gate C code review + fixes, a short real-data benchmark.
- After training: ≈ 1.5–2 h. Lock, retrospective VC+TEST eval (≈ 0.3–0.6 h per model per GPU; baselines reuse archived ranks), report.
- **Training window ≈ 32 h wall on 2×A16, about 64 GPU-hours, at most 2 concurrent jobs.** RAM and disk are not binding.

## Measured production-model speed (Lobot A16, synthetic tokens, real length mix, fused AdamW, exact [B,K] CE)
Hours per full epoch (40.18M decisions) at each batch size, plus one full-VS eval (8.83M rows, exact ranking):

| model | precision | B=256 | B=512 | B=1024 | eval |
|---|---|---|---|---|---|
| d128 | FP32 strict | 2.75 | – | – | 0.30 |
| d128 | TF32 matmul | 2.29 | – | – | 0.28 |
| d256 | FP32 strict | 4.80 | 4.29 | 4.23 | 0.44 |
| d256 | TF32 matmul | 3.66 | 3.15 | 2.94 | 0.35 |
| d384 | FP32 strict | 7.67 | – | – | 0.62 |
| d384 | TF32 matmul | 5.46 | – | – | 0.45 |

- The rich arm costs about 5–10% more than d256 ID; the real-data benchmark will confirm this.
- The GEMMs, dominated by the [B,K] head, are ≈ 45% of the step. The rest is small-kernel overhead that no loss-preserving change removes: bucketed attention was slower, SDPA mem-efficient saved 7% but is non-deterministic, and torch.compile is untested.
- The current registry needs ≥ 5 epochs per run × 7 runs, which projects to ≈ 5–7 days.

## Levers I see (you may add, reject or combine)
1. **Compressed update-based schedule.** For example: warmup 2000; peak to U; cosine to 0.1× across epoch 2; floor epochs 3–4; fixed 4 epochs. This keeps 2 full low-LR epochs (the GRU lesson) but halves the exposure.
2. **Full-VS eval only where stopping or selection can use it.** For example, at floor epochs only; best = raw max over the evaluated epochs.
3. **Fewer runs.** For example, the A0‖A1 pair only; or the pair plus one challenger; no or fewer seed replicas.
4. **TF32 matmul for training** (+15–40%), with the canonical eval kept strict FP32. This is a numerical-policy change that needs owner approval and a registered parity check, e.g. a paired FP32 vs TF32 loss trajectory over N updates.
5. **Larger effective batch 512/1024** with a registered LR rule. It gains little in FP32 alone, more combined with TF32, but it is a new optimisation treatment.
6. **Smaller width.** d128 is cheaper, but likely weaker.

## What I need (≤ 1400 output tokens, binding plan except where the owner must approve)
- ROUTING line.
- VERDICT on feasibility within 38 h.
- A **RECOMMENDED plan**, with a concrete table: runs × GPU × epochs × schedule × batch × precision × eval epochs × expected hours, a critical-path total ≤ 38 h, and ≥ 2 h of slack.
- One **ALTERNATIVE**: a strict-FP32-only plan with no owner numerical approval, and its expected quality cost.
- Every change from the registered protocol, each tagged [evidence]/[inference], plus which items need explicit owner approval.
- Revised stopping/selection/replication rules, and the claims that must be narrowed as a consequence.
- The minimum tests to add for any new element (e.g. the TF32 parity gate).

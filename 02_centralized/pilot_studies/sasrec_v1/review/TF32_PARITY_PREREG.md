# TF32 parity gate: pre-registered design and tolerances (written 2026-09-23T23:15Z, BEFORE any comparison exists)

Owner directive section E and Fable Gate P2 test 1. Three isolated real-data benchmark states of A0_ID256 (d256), each discarded afterwards:

| state | training precision | init/dropout seed | GPU |
|---|---|---|---|
| FP32_A | strict FP32 | 2026 | 0 |
| TF32_A | TF32 matmul | 2026 | 1 (concurrently with FP32_A) |
| FP32_B | strict FP32 | 2027 | 0 (after FP32_A) |

The three states share:
- the same real TRAIN batches, in the same order (benchmark rng 12345);
- the same targets and the same optimizer (fused AdamW, betas 0.9/0.98, wd 1e-5, clip 1.0);
- the real schedule from u=1 (warmup 2000, then peak 5e-4);
- 3000 updates, logging loss and grad norm at every update.

At the end of each state, in strict FP32:
- the 112,365-decision diagnostic VS cohort macro MRR@20 and micro Recall@20;
- a deterministic parameter sample (≤ 20,000 elements per tensor, rng 777) at init and at the end.

## Pass criteria (all required)
1. **Loss band.** Over updates 2001–3000, |mean loss(TF32_A) − mean loss(FP32_A)| ≤ |mean loss(FP32_A) − mean loss(FP32_B)| + 0.005 × mean loss(FP32_A).
2. **Finite.** No NaN/Inf loss or grad norm in any state.
3. **Grad norms.** The median grad norm over updates 2001–3000 of TF32_A is within ±5% of FP32_A's.
4. **Cohort metric.** |cohort macro MRR@20(TF32_A) − (FP32_A)| ≤ |(FP32_A) − (FP32_B)| + 0.003 absolute.

## Descriptive only (reviewed, not a pass criterion)
- **Relative parameter-trajectory difference:** ‖(θ_TF32A − θ0) − (θ_FP32A − θ0)‖ / ‖θ_FP32A − θ0‖ on the sample, per tensor group. This value is expected to be non-zero, because floating-point differences compound over 3000 steps.
- Micro Recall@20 on the cohort.

## Decision
- `code/tf32_parity_gate.py` computes everything into `bench/TF32_PARITY_EVIDENCE.json`.
- Fable 5.1, read-only and acting as the owner's delegate, reviews that evidence and rules **TF32 ACCEPTABLE** or **TF32 NOT ACCEPTABLE**.
- Its ruling is written to `bench/TF32_PARITY.json` (`pass` = ACCEPTABLE). That file is the only one the trainer reads.
- NOT ACCEPTABLE, a failed rule, or any error applies the registered fallback automatically: strict FP32, 5 epochs (3 low-LR epochs).

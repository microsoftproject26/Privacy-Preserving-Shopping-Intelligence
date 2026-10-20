# AUTO_DECISION 2026-09-24T00:05Z: TF32 training

- decision authority: Fable 5.1 as owner delegate (owner directive sections E, I, J)
- transcript model ids: ['claude-fable-5-1']
- evidence: bench/TF32_PARITY_EVIDENCE.json (sha256 54b02f27…), prereg review/TF32_PARITY_PREREG.md (sha256 3612bb87…)
- **DECISION: TF32 ACCEPTABLE** → TF32 training, strict-FP32 evaluation, 6 epochs (4 low-LR)
- recorded concern: the aggregate parameter-trajectory figure (0.27%) is dominated by output_bias; report the per-tensor values (0.5–2.5% for weights, up to ~4.9% for qkv biases) as the honest descriptive figure

## Fable ruling (verbatim)

ROUTING: claude-fable-5-1

RULING: TF32 ACCEPTABLE

Criteria relied on (all four pre-registered pass criteria in `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\review\TF32_PARITY_PREREG.md`, checked against `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\bench\TF32_PARITY_EVIDENCE.json`; arithmetic re-verified by hand):
1. Loss band [evidence]: |8.005154 − 8.005170| = 1.6e-5 vs tolerance 0.005924 + 0.005×8.00517 = 0.04595. Pass by ~3 orders of magnitude.
2. Finite [evidence]: `all_finite: true` for FP32_A, TF32_A, FP32_B.
3. Grad norms [evidence]: median ratio 1.000124 (1.87420 / 1.87397), well inside ±5%.
4. Cohort metric [evidence]: |0.090172 − 0.090261| = 8.8e-5 vs band 2.21e-4 + 0.003 = 0.00322. Pass. Micro Recall@20 (descriptive) also within seed spread (0.26083 vs 0.26072 / 0.25950).

Concerns:
- Parameter trajectory [evidence/inference]: per-tensor relative differences are 0.5–2.5% for weights, up to 4.7–4.9% for `blocks.*.qkv.bias`. The reported `overall` = 0.27% is smaller than all per-tensor values except `output_bias` (0.014%), so the aggregate is norm-dominated by `output_bias` and is not a representative summary; the per-tensor values are the honest descriptive figure. These are compounding FP differences with no effect on loss, grad norm, or the strict-FP32 cohort metric, consistent with the prereg's stated expectation. Not a pass criterion; noted, not disqualifying.
- Extrapolation [inference]: 3000 updates is a small fraction of a 6-epoch run; divergence beyond the window is unmeasured. Mitigated by strict-FP32 evaluation of every reported number.
- Seed band from one pair (FP32_A/FP32_B) [inference]: a narrow reference, but TF32 deltas sit far below it.
- FP32_A vs TF32_A ran on different GPUs [evidence]: a hardware confound folded into the TF32 delta; conservative direction.
- Not independently verified [evidence]: raw per-update logs (server-only), the prereg sha256 3612bb87…, and `code/tf32_parity_gate.py` itself (Gate C approved). Ruling rests on the JSON summaries as instructed.

Condition: claim wording must read "trained TF32 / evaluated strict FP32", stating the loss tolerance (seed band + 0.5%) and cohort tolerance (seed band + 0.003).

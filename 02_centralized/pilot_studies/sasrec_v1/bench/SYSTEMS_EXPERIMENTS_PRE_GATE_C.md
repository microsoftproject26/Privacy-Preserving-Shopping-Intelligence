# Pre-Gate-C systems experiments (synthetic tokens only; no real-data updates)

All runs on Lobot A16 GPU1, torch 2.13.0+cu130, strict FP32 (TF32 off), d256 / 2 blocks / 4 heads, batch 256, real-like length mix, fused AdamW, exact [256,K] CE. Scripts: `$S/tests/opus_smoke_model.py`, `opus_prof_model.py`, `opus_attn_exp.py`.

| Variant | ms/step | Deterministic | Decision |
|---|---|---|---|
| production `unpadded` (single right-padded buffer for the causal product, explicit FP32 math attention) | 110.6 (deterministic algorithms on); 105.7 with them off | yes (repeat runs bitwise equal) | **KEEP** |
| + length-bucketed attention (caps 4/8/16/32/50) | 113–115 | yes | REJECT: more tiny kernels outweigh the smaller attention products |
| SDPA mem-efficient backend (is_causal) | 104.9 | PyTorch warns its backward is non-deterministic by default | REJECT: violates the strict determinism/resume policy for a 7% gain |
| SDPA math backend | 114.8 | yes | REJECT: no gain |

Profile of the kept variant (per step):
- GEMMs ≈ 46 ms, of which the [256,K] head forward+backward is ≈ 24 ms;
- fused AdamW 7.3 ms;
- small elementwise/fill kernels ≈ 20 ms;
- softmax/log-softmax ≈ 6 ms;
- embedding backward ≈ 5 ms.

The GPU is busy the whole step (CUDA time ≈ wall time), so it is not launch-bound on the CPU side. The A16 (62.5 W per GPU) sustains roughly 1.3 TFLOPS on these FP32 GEMMs.

Unpadded vs padded reference, eval mode:
- logits: max |Δ| 0.0 (ID), 0.0 (rich), later 2.8e-7 under bucketing;
- training-mode gradients with dropout 0: rel ≤ 1.2e-6;
- init loss ≈ ln K.

torch.compile was not attempted. Dynamic token counts and dropout RNG semantics under inductor would add review risk for an uncertain gain, so it is recorded as future work.

## Owner directive (D-ladder) experiments, 2026-09-23 ~23:10Z (synthetic tokens, GPU1)
Production model, d256, TF32 training (`bench/compile_evalbatch_synthetic.json`, `bench/speed_levers_synthetic.json`):

| Lever | Result | Decision |
|---|---|---|
| D1 two-GPU independent arms | planned (A0 GPU0, A1 GPU1); no DDP | KEEP |
| D2 threads | torch intra-op 3, inter-op 1; OMP/MKL 3, OPENBLAS/NUMEXPR 1 (`paths.py`, `train_sasrec.configure_threads`) | KEEP (data wait was <0.3% in the GRU store path; re-checked in the real benchmark) |
| D3 compact RAM store | reused CompactStore, resident for the process lifetime | KEEP |
| D4 prefetch | 1-thread prefetch of the NEXT batch + pinned memory + non_blocking H2D, identical batches | KEEP |
| D6 fused AdamW | on (resume/tiny tests pass with it) | KEEP |
| D7 SDPA mem-efficient | 7% faster but PyTorch reports a non-deterministic backward | REJECT (determinism/resume contract) |
| D8 torch.compile (FFN blocks, dynamic) | 82.3 vs 83.9 ms/step (-2%), +10 s compile, review risk | REJECT (not material) |
| G exact eval batch | 1024: 5,665 / 2048: 5,617 / 4096: 5,601 rows/s (strict FP32) | KEEP 1024 |
| TF32 training | d256 step 84 ms vs 110 ms strict (-24%) | parity-gated (review/TF32_PARITY_PREREG.md) |
| F larger batch | B512/B1024 per-decision gains 12-25%, but it is an optimisation treatment | NOT USED (Fable P2 rejected it; owner F) |

# PREG1 amendment 04 — the realistic downstream plan (registration only)

- **Status:** REGISTERED by the Operations Director on 2026-09-27, BEFORE any P2/P3 value has been read. Authority: the
  owner's research question and decisions (DECISION_LOG 83, 86: FL at SMALL; no acceptability threshold; add the cheap
  items and DP privacy levels) and Fable's final plan `company/reviews/DOWNSTREAM_FINAL_PLAN_FABLE.md` (verbatim; it is
  the binding detail of this amendment). Builds on PREG1_AMENDMENT_03 (BIG/SMALL; ≤ 170 MB device cap).
- **Supersedes** Amendment 03 §4's "acceptable" verdicts and its threshold trigger. Unchanged: P2, P3, G2, the staggered
  seal, TEST at ALL_METHODS_LOCK.
- **Research question answered:** "How much shopping recommendation quality can be retained when moving from
  cloud-based to on-device or federated models?"

## 1. Answer format: RETENTION (no pass/fail threshold)
R = 100 × arm / C_BIG on client-macro MRR@20, seed mean over 2026/2027, paired user bootstrap CI (1,000 resamples, seed
20260926); VALIDATION interim, MAIN TEST at ALL_METHODS_LOCK. Personalisation arms also report the paired gain versus
their base. Curves: retention vs bytes per client, vs ε, vs model size, vs stratum. Sanity trigger (not a verdict):
C_SMALL 6.0 value < 0.5× C_BIG → Fable.

## 2. Arms
| Group | Arms |
|---|---|
| Server | C_BIG (the P3 winner runs), C_SMALL, KD_SMALL (teacher C_BIG, top-50 ranking distillation on MAIN TRAIN, 2 seeds, unconditional) |
| Federated | FA, FP (64 clients per round), FA_1024 (1,024 per round, clipping on, no noise), DP-FA ε = 8 and ε = 1 |
| Personalisation | PF, CPF (personal component only), FT_C, FT_FA, FT_KD (whole SMALL model fine-tuned on the client's own data) |
| Device only | LO |
| Baselines | popularity, last-item, session-kNN (BASE-01), BASELINE_KNN_EMB (cached SMALL item table; BIG as reference), BASELINE_RULE (category/brand affinity) |

## 3. Realism knobs, DP mechanism, personalisation budget, support/causality, ONNX protocol, strata
Exactly as in Fable's final plan §2–§3 and the realistic plan (company/reviews/DOWNSTREAM_REALISTIC_FABLE_PLAN.md §2):
64 clients per round with 10% seeded drop-out (6,827 rounds); B16 × 2 passes; FedProx μ 0.01 untuned; FT budget passes
{1, 2, 4} × LR {0.1, 0.3}× SMALL peak, selected once on M04S FIXED_20; DP-FedAvg with flat L2 clipping S (median client
delta norm over FA_1024 seed-2026 rounds 1–20), Gaussian noise z·S on the sum, fixed denominator m = 1,024, T = 384
rounds, δ = 5e-6, RDP accountant, z from (m, T, δ, ε) registered before the first DP run; strata from TRAIN-period events
with cut points fixed on MAIN TRAIN before any read (category-diversity terciles, brand-loyalty terciles, persona 3×3,
activity quantiles); ONNX: C_SMALL seed 2026, opset 17, onnxruntime CPU single-thread batch 1, latency median/p95 over
1,000 queries, peak RSS, FP32 and INT8 size, INT8 retention via T3, parity vs PyTorch.

## 4. Metrics
Primary client-macro MRR@20. Secondary NDCG@10 and HR@10: **already computed by the frozen evaluator**
(`code/v4eval/metrics.py` FAMILIES mrr/hr/ndcg; PROMINENT macro ndcg@10, hr@10), so no evaluator change and no separate
script is needed (this corrects Fable §2's assumption). Also micro metrics, bytes (per visit, per client, fleet),
rounds, consumed EFE, local updates, model size, device-time proxy, support labels per client, harmed-client share.

## 5. Cost and fallback (INFERENCE)
≈ 140–165 GPU-h ≈ 3.5–4.5 days of wall clock on 2 A16s after G2. Fallback order if the owner caps wall clock: FT_KD →
DP ε = 1 second seed → FT panels HEAVY/SPARSE to 8,192 → DP ε = 1.

## 6. After G2 (not registered values yet)
z values (before the first DP run), S, the SMALL recipe, absolute LRs, SYS-04c/SYS-05 benches; code FL-03 (drop-out,
full-model FT), FL-04 (cohort size, clipping, noise, accountant), TRAIN-05 (KD), BASE-02 (kNN-emb, rule, strata),
DEV-01 (ONNX) — all under Opus gates.

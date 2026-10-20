# AUTO_DECISION (2026-09-24T16:50Z): SASRec training pathology. Owner offline; Fable decides as the owner's delegate.

## Situation (measured; evidence paths on the server under $S, key numbers below)
Both v2 anchors have been running since 00:18Z and are in epoch 5 of 6. Full-VS strict-FP32 macro MRR@20 at the evaluated epochs:

| arm | epoch 3 | epoch 4 |
|---|---|---|
| A0_ID256 | 0.09133 | 0.09009 |
| A1_RICH256 | 0.11791 | 0.11681 |

References: GRU F0 epoch 1 reached 0.154 (0.174 best), and full-TRAIN TRANS is 0.162.

The training loss (A0, 100-update means) is pathological:

| update | mean loss | lr |
|---|---|---|
| 3k | 7.84 | |
| 12k | 7.55 (minimum) | |
| 30k | 7.82 | |
| 157k | 8.22 | 5e-4 |
| 280k | 8.77 | |
| 320k | 9.02 | 5e-5 |
| 720k | 8.93 | 5e-5 |

- The mean grad norm rises 1.8 → 6.6 while the LR falls 10×.
- A1 behaves the same way, but milder: loss 7.44 at 12k, 7.2–7.3 afterwards, grad norm up to 4.6.
- The 112k diagnostic cohort peaked early (A0 0.105 at 80k updates) and has declined since.

Root-cause evidence (`bench/diag_item_embeddings.json`; read-only CPU analysis of A0's previous.pt at update ≈ 720k):
- Item-embedding row norms are ≈ 161.7 **uniformly across every TRAIN-target-frequency bucket**: 0, 1–4, 5–49 → 161.7; 50–499 → 162.6; 500+ → 164.5.
- In the healthy GRU F0 the norms are 11–13.
- The Adam v row-means are ≈ 3e-17 for rare items and 7e-8 for frequent ones.

Inference (mechanism):
- Exact full-catalogue CE gives every item row a small softmax-denominator gradient each step. Adam's per-coordinate normalisation turns those into ~lr-sized steps in a common direction (−mean query) for all non-target rows, so the zero-sum property of the raw gradient is lost.
- The resulting common offset c leaves the logits unchanged (softmax shift invariance, so the loss does not resist it).
- Through TIED weights and the √d = 16 input scaling (z = 16·E + P), however, c dominates the residual stream at the input. The item identity washes out, the loss rises and validation falls.
- TF32 plausibly worsens this: the small item-specific part rides on a ≈ 2600-norm vector at the input GEMMs. The 3000-update parity gate could not see drift that develops later.
- The GRU (β2 = 0.999, no √d input scaling, an input Linear+LN, no final LN) stayed at norms ≈ 12.

## Options
**A. Continue** both runs to epoch 6 (end ≈ 22:00Z) and report an honest failure.
- Correct, but a known-bad trajectory, and it wastes ≈ 10 GPU-hours.
- The best checkpoints (epoch 3) are already saved.

**B. Stop both runs now** (keep all artifacts as the failed-v2 evidence), run a quick systems/training-dynamics A/B, then relaunch a fixed recipe, labelled v3.
- The A/B: ≈ 25 min, 2 variants per GPU in 2 rounds, 20k real-data updates each, discarded benchmark states. It is judged on TRAINING dynamics only: loss still decreasing after 12k, grad norm not climbing, common-mode embedding norm growth, plus the cohort as descriptive.
- The relaunch uses a compressed schedule that fits the deadline: 4 epochs (peak epoch 1, cosine epoch 2, floor epochs 3–4 = 2 genuine low-LR epochs), full-VS eval at epochs 3–4, ≈ 14 h.
- Timeline: relaunch ≈ 17:45Z → end ≈ 08:00Z → report ≈ 10:00Z. Deadline 12:20Z, slack ≈ 2.3 h.
- Candidate fixes:
  - F1: AdamW betas (0.9, 0.999) (GRU-proven);
  - F2: F1 + no √d input scaling (z = E + P);
  - F3: F1 + strict FP32 training;
  - F4: F1 + larger Adam eps for the item table only.

**C. Stop only A0** and keep A1: this breaks the matched pair. Not recommended.

## Time impact
- A: done at ≈ 22:00Z, poor result.
- B: ≈ 10:00Z report, with 2.3 h slack. Risk: if the fix fails as well, report both failures honestly.

## Scientific impact
- B is a numerical/optimisation-failure fix decided on TRAINING signals, not validation tuning. It is disclosed as a v3 recipe change after a failed v2 run; the v2 artifacts stay as evidence.
- The owner's "6 epochs" becomes 4 (still ≥ 2 genuine low-LR epochs), to protect the deadline.
- TEST is untouched either way.

## Decision needed
1. A, B or C.
2. If B: which fix variants to A/B (≤ 4), the objective training-signal selection rule for choosing ONE variant (fixed now, before the A/B runs), the relaunch schedule and precision, and whether TF32 stays allowed.

≤ 600 tokens. Begin with `ROUTING:`.

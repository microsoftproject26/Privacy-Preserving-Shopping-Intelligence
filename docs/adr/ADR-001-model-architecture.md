# ADR-001 — Phase-1 Model Architecture

- Status: Accepted for the MVP
- Decision date: 2026-09-12
- Decision owner: Data & Model lane
- Related issues: #42, #47

## Decision

Freeze `shared_session_encoder_v1` as the Phase-1 architecture. It is a one-layer GRU with
hidden width 128, dropout 0.3, right-padded histories of length 20, and the frozen
`phase1_batch_spec_v1`. The history channels are ordered as `category_id`, then
`event_type_id`, with the elapsed-time feature enabled. T1 and T2 share the encoder.

T3 remains part of the output interface, but no T3 loss enters the selected shared-encoder
checkpoint. The MVP recommendation path uses the frozen model-free retrieval order for T3.
This is an explicit evidence-based deviation from the original three-learned-head objective,
not an assertion that a learned T3 head succeeded.

## Evidence

The S2-DS-05 comparison changed only the sequence core under the same data, evaluator, budget,
and seed:

| Core | T1 slice macro MRR@20 | Parameters | Training time |
|---|---:|---:|---:|
| GRU | 0.3477 | 2,379,263 | 27.5 min |
| LSTM | 0.3471 | 2,412,287 | 34.8 min |
| Transformer | 0.3440 | 2,420,863 | 46.0 min |
| TCN | 0.3398 | 2,329,727 | 28.9 min |

GRU and LSTM differ by 0.0006, inside the observed 0.0007 T1 seed spread, while the LSTM took
27% longer. This supports GRU as the smaller, faster tied choice; it does not claim that GRU
is universally better under separately tuned budgets.

For joint training, the selected `lambda_t2=1.0` checkpoint scores T1 0.3467 and T2 0.1337.
A separately fine-tuned T2 checkpoint reaches 0.1424 but damages T1 to 0.1791, below the T1
model-free baseline of 0.3143. The joint checkpoint is therefore the deployment candidate.

For T3, the frozen retrieval order scores macro NDCG@5 0.2707. The best corrected query-aware
reranker peaks below it (approximately 0.2596 across the three measured seeds), so the learned
T3 term is rejected for this MVP.

## Frozen Contract

The machine-readable record is
`config/experiments/s2-ds-08/model_config.v1.json`. The experiment validator registers
`shared_session_encoder_v1` and rejects field, channel-order, batch-spec, dtype, and basic
parameter-contract drift. CommonInitialization v1 records for seeds 13, 42, and 2026 bind to
the exact ModelConfig artifact hash.

## Consequences

- R1 and R2 must instantiate this exact contract and use the same seed-specific initialization.
- Sequential full-backbone T2 fine-tuning is not a valid multi-task deployment path.
- The MVP recommendation service combines learned T1/T2 outputs with model-free T3 retrieval.
- A future learned T3 head requires a new pre-registered experiment and a new ADR revision; it
  must beat the frozen retrieval baseline before entering the shared objective.

## Approval Note

The task owner requested expedited MVP closure on 2026-09-12. The architecture and the T3
deviation are therefore recorded as the MVP decision in the repository. This note does not
claim that three separate GitHub reviewer approvals were recorded; teams that retain that
governance requirement should add those signatures without changing the frozen evidence.

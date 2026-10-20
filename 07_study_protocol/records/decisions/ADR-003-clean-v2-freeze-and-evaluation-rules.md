# ADR-003 — Clean V2 freeze and evaluation rules (G0(b))

## Decision
Freeze Clean V2 and write DATA_PROTOCOL_LOCK after the lock-text fixes. The decisions are:
- **Features:** Clean V2 = 53 input channels (numeric 15, flags 15, user context 12, masks 4).
  - 6 exact-redundancy drops;
  - 0 spec-dead drops (six all-zero-in-TRAIN flags are kept and disclosed);
  - an M1 valid-only refit with re-zeroing;
  - F-1: an extended re-zero gate plus a valid-only refit of `current_minus_recent_mean_log_price`.
- **Stopping (binding):**
  - every run completes 6.0 EFE;
  - PRACTICAL_BEST is the strictly-best checkpoint (Δ > 0.0002) in [0.5, 6.0];
  - CONTROLLED_BUDGET is the 6.0 endpoint;
  - a 6→8 continuation is allowed only if the run is STILL_IMPROVING, is labelled PRACTICAL_BEST_EXTENDED, and is
    secondary.
- **Censoring classes:** LEGACY_BARRIER 170; AMBIGUOUS_NEXT_BARRIER 4,489; RIGHT_EDGE 152; NO_NEXT 3,064,778.
- **Cold start:**
  - M04 = the 16,384 smallest-key cold VALIDATION users;
  - WARM_TRUNCATION = the 16,384 smallest-key warm-cohort VALIDATION users;
  - M04S (ranks 16,384–32,767 of M03) is used to select the few-shot recipe once;
  - h = 0 is the query event only;
  - FIXED_20 is added.
- **FL mechanics:**
  - a recentered θ0 is the per-seed init for every method (SASRec);
  - p has its own gradient clip of 1.0;
  - the CPF penalty counts each distinct user once per step;
  - n_shards = 2; max_attempts = 2.
- **Central optimizer (executed values):**
  - AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 1e-5 on ndim ≥ 2 parameters, clip 1.0;
  - peak LR 1e-3 for both families; B = 256;
  - dropout: GRU 0.20 / 0.10, SASRec 0.2;
  - label smoothing 0.

## Question
Are the Wave A contracts a correct and complete implementation of the G0 decisions, and how should the open
confirmation items be resolved?

## Options considered
- F-1: adopt or reject.
- Stopping: early stop after the 3.0-EFE reference, or complete 6.0.
- The cold-h panel: candidate A or candidate B.
- The 140 ambiguous rows: legacy-barrier or ambiguous class.

## Project evidence
S328–S331 (QA-02), S319–S323 (QA-01), S324–S325 (RES-02), FL-01 receipts, and the contracts S300–S304 (Wave A
versions).

## Opus and Fable positions
`reviews/G0B_PROTOCOL_OPUS_POSITION.md` and `reviews/G0B_PROTOCOL_FABLE_POSITION.md`.

## Agreement and disagreement
They agreed on everything except P11. On P11 Opus preferred B and Fable preferred A; A was adopted for its validity
(no selection on evaluation-window activity). See `reviews/G0B_PROTOCOL_SCIENTIFIC_SYNTHESIS.md`.

## Consequences
- The absolute cold-start support at h = 50 may be small; it is reported with its n.
- No early stopping means the full budget is spent even on runs that have already converged.
- The recentered θ0 changes the init gauge but not the predictions.

## Validation required
- The lock-text fixes (EVAL-01R2, DATA-03R1).
- The DATA-04 build of the sidecars and manifests, with per-manifest receipts.
- Writing DATA_PROTOCOL_LOCK.json.
- Before G1: the GPU determinism receipt, OI-05, FL-02, and the owner's conditional pre-approval about the disk.

## Thesis relevance
This defines the Methodology sections on data preparation, evaluation, cold-start design and the training budget.

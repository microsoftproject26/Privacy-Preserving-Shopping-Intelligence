# Deep review + research request (owner-requested), 2026-09-24T17:00Z

The owner asks Fable 5.1 (max effort) for four things:
1. A deep review of the code and of everything done so far.
2. The true root cause of the poor SASRec training, found by reading the code and the evidence.
3. A fix designed after deep web research, backed by papers and reputable open-source projects.
4. If the choice of input columns is part of the cause, choose the appropriate columns.

You are the owner's delegate: your decisions are binding for in-scope choices. You cannot change the frozen data, targets, the evaluation protocol, OOV handling or the catalogue, and you cannot use TEST.

All paths are relative to `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\` unless absolute.

## 1. What was done (short)
- **Task.** One exact full-catalogue CE decision per frozen TRAIN decision (40,180,608); K = 158,486; context ≤ 50 events. Selection uses full VALIDATION_SELECT, 8,826,488 rows, macro-by-user MRR@20; strict-FP32 exact ranking with OOV targets kept at zero credit.
- **References on this exact protocol** (full VS, macro MRR@20):

| model | full VS macro MRR@20 |
|---|---|
| GRU F0 (item-only) | 0.1544 at epoch 1; best 0.1739 |
| GRU F4 (rich) | 0.1548 at epoch 1; best 0.1660 |
| TRANS (item→item transitions) | 0.1616 |
| POP | 0.0216 |

- **Gates.** Gate P approved with fixes; the owner imposed a 38 h deadline, then Gate P2 cut the plan to 2 runs. Gate C approved the code (`review/FABLE_GATE_C_RESPONSE.md`). The TF32 parity gate passed at 3000 updates (`review/AUTO_DECISION_TF32.md`).
- **Runs.** v2 has been running since 00:18Z, now in epoch 5 of 6:
  - A0 `A0_ID256`: item only, d256, 2 blocks, 4 heads, FF 1024, dropout 0.2 everywhere;
  - A1 `A1_RICH256`: the same plus rich side channels.
  - Shared settings: AdamW lr 5e-4, betas (0.9, 0.98), eps 1e-8, wd 1e-5 on matrices/embeddings, clip 1.0, batch 256; schedule S′ (warmup 2000, peak epoch 1, cosine epoch 2, floor 5e-5 epochs 3–6); TF32 training, strict-FP32 eval.
- **Code.**
  - `code/sasrec_model.py`: input z = √d·E[item] + P[pos]; rich = + sigmoid(a_event)·LN(side_proj(...)) + dropout; Pre-LN blocks; final LN; tied head `logits = LN(h_last) @ E[3:3+K]^T + b`.
  - `code/train_sasrec.py`: trainer and optimizer groups in `build_optimizer`.
  - `code/sasrec_objective.py`, `code/sasrec_schedule.py`, `code/precision.py`.
  - The healthy GRU for comparison: `F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1\code\model.py` (ContextGRU: concat embeddings → Linear → LN → dropout → GRU(256) → readout Linear(128) → tied head + bias; AdamW lr 1e-3, betas (0.9, 0.999), wd 1e-5; see `..\gru_v1\code\train_full.py`).
- **Package specification.** The original contract for this SASRec is in `F:\Queens University\Microsoft Project\eda\SASREC_PERFORMANCE_EXECUTION_V1\03_ARCHITECTURE_CONTRACT.md`. It is not sacred if it is the cause, but every deviation must be justified.

## 2. The problem (evidence)
- Validation metrics and per-update trajectories: `review/deep/trajectories.json` (update_agg every 2k updates, the diagnostic cohort, epoch records).
- Full VS macro MRR@20:

| arm | epoch 3 | epoch 4 |
|---|---|---|
| A0 | 0.0913 | 0.0901 |
| A1 | 0.1179 | 0.1168 |

- **A0 train loss:** a minimum of 7.55 at ≈ 12k updates, then 8.22 at 157k (peak LR), ≈ 9.0 at 320k, and ≈ 8.9 through 720k at floor LR 5e-5.
- **A0 mean grad norm:** 1.8 → 6.6 and still rising.
- **A0 cohort MRR:** peaked at 0.105 (80k), declining since.
- **A1:** similar, but milder.
- **Embedding norms** (`review/deep/diag_item_embeddings.json`; A0 previous.pt at ≈ 720k updates):
  - item-embedding row norms are ≈ 161.7 uniformly across TRAIN-target-frequency buckets: 0 / 1–4 / 5–49 → 161.7, 50–499 → 162.6, 500+ → 164.5;
  - the GRU's rows are ≈ 11–13;
  - the Adam v row-mean is ≈ 3e-17 for rare items and 7e-8 for frequent ones;
  - output_bias medians range from −17 to −40.
- **My hypothesis (please verify or refute it from the code and the literature):**
  1. Adam's per-coordinate normalisation of the dense softmax-denominator gradients drives all item rows along a common direction.
  2. Softmax shift invariance leaves the loss indifferent to that common offset.
  3. Tying plus the √d = 16 input scaling (and Pre-LN, with no input LN) lets the common offset dominate the residual stream, washing out item identity.
  4. TF32 may amplify this at large norms.
  5. Alternatively or additionally, 0.2 attention/residual dropout, lr, the warmup length, or the final LN + tied head (fixed-norm query against large-norm keys) may matter.

## 3. The columns question
- **A0** uses item ids only.
- **A1** adds, per event:
  - category/brand/main-category tokens (vocab 663 / 3956 / 16);
  - daypart/weekday (8 / 10);
  - 19 numeric features: log1p_price, utc/local hour/weekday sin-cos, is_weekend, log1p gaps/elapsed/item age, prefix counts, …;
  - 15 quality flags;
  - at readout, 12 user-context values + 6 masks.
- The exact lists and semantics are in `F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1\code\vendor\rees46_eda\config.py` (EVENT_NUMERIC_FEATURES, SESSION_PREFIX_FEATURES, EVENT_QUALITY_FLAGS, USER_CONTEXT_FEATURES, USER_CONTEXT_MASKS, FEATURE_GROUPS).
- The frozen groups F0–F4 are defined there. You may choose among them, or pick a subset of their frozen columns; never create new features.
- A1 > A0 here, while in the GRU F4 ≈ F0 at epoch 1. Is this because the side embeddings are NOT tied to the softmax, so they don't drift? Decide whether the columns are part of the cause, and which columns the next run should use.

## 4. Constraints
- **Time.** The owner deadline is 2026-09-25T12:20Z; about 19 h remain at 17:00Z today.
  - One d256 epoch ≈ 3.2–3.3 h (TF32; ≈ 4.8 h strict FP32) plus ≈ 0.42 h per full-VS eval.
  - Both GPUs are busy with the v2 runs until we stop them.
  - Short real-data A/B states (e.g. 20k updates ≈ 25 min) are possible once they are stopped.
  - If the fix needs more time than the deadline allows, say so explicitly with the trade-off; the owner is online now.
- **Precision.** TF32 training passed parity only for the early trajectory. You may keep or drop it.
- **Scope.** Stay within SASRec-family causal attention with exact full-catalogue CE, one loss per decision, context 50, full catalogue, exact evaluation.
  - Architectural/optimiser fixes (normalisation, scaling, tying details, optimiser settings, eps, LR, dropout) are in scope if justified.
  - Sampled softmax is out of scope.

## 5. What to return (strict format; ≤ 2500 tokens)
- `ROUTING:` line.
- `ROOT_CAUSE`: ranked causes with [evidence] / [literature: URL] / [inference] tags, with the code lines that implement them.
- `CODE_REVIEW_FINDINGS`: any other defects you find (file:line, severity).
- `LITERATURE_AND_PROJECTS`: what papers and implementations do for tied embeddings + full-softmax CE + Adam in SASRec/transformer recommenders: scaling, input LayerNorm, optimiser, eps, weight decay, embedding norm control, lr/warmup, dropout. Give URLs; distinguish fact from inference.
- `FIX`: an exact, minimal change list (model / optimiser / schedule / precision), each with its justification and citation.
- `COLUMNS_DECISION`: are the columns a cause (yes/no/partly, with evidence), and which columns (frozen group or subset) each run uses.
- `VERIFICATION_PLAN`: a quick real-data A/B (≤ 4 variants, ≤ 25 min each, 2 GPUs) with a PRE-REGISTERED selection rule on TRAINING signals, plus the go/no-go threshold.
- `RELAUNCH_PLAN`: runs × GPU × epochs × schedule × precision × eval epochs × hours, and its fit against the deadline.
- `CURRENT_RUNS`: stop now or let them finish, and what to keep as evidence.
- `CLAIMS_IMPACT`.

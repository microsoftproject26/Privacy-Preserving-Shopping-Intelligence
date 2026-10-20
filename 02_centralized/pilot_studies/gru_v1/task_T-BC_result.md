# Task T-B / T-C result — training objective + evaluator

## Files created

- `code\objective.py` — `exact_ce_loss`, `LossAccumulator`, `sampled_ce_loss` (default-OFF fallback).
- `code\evaluator.py` — `evaluate(scores_fn, rows, k=20, chunk_size=...)`.
- `code\test_objective.py` — T7 masking/denominator, T7 accumulation equivalence, zero-contributing-rows autograd test, sampled-CE default-OFF gate + toy test, 2 strict-xfail broken variants.
- `code\test_evaluator.py` — T9 OOV-merge macro fixture (+ OOV extension), T8 tiny exhaustive reference, 6 strict-xfail broken variants.
- No `code\__init__.py` was created (per instructions).

## Exact pytest command + exit code

Run from `.local_ai\gru_v1` with the specified local python:

```
F:\Queens University\Microsoft Project\eda\next_item_eda\.venv\Scripts\python.exe -m pytest code\test_objective.py code\test_evaluator.py -v
```

**Exit code: 0**

```
============================= test session starts =============================
platform win32 -- Python 3.13.11, pytest-9.1.1, pluggy-1.6.0
collected 21 items

code/test_objective.py::test_t7_masking_and_denominator PASSED
code/test_objective.py::test_t7_accumulation_equivalence_uneven_microbatches PASSED
code/test_objective.py::test_zero_contributing_rows_is_finite_and_autograd_connected PASSED
code/test_objective.py::test_broken_divide_by_batch_size XFAIL
code/test_objective.py::test_broken_average_of_microbatch_means XFAIL
code/test_objective.py::test_sampled_ce_disabled_by_default_raises PASSED
code/test_objective.py::test_sampled_ce_toy_logq_correction_and_accidental_hit_masking PASSED
code/test_evaluator.py::test_t9_macro_and_micro_mrr PASSED
code/test_evaluator.py::test_t9_broken_group_by_client_id XFAIL
code/test_evaluator.py::test_t9_oov_extension_end_to_end_vs_closed PASSED
code/test_evaluator.py::test_t9_broken_drop_oov_rows XFAIL
code/test_evaluator.py::test_t8_bruteforce_ranks_match_hand_derivation PASSED
code/test_evaluator.py::test_t8_chunked_ranks_match_bruteforce_for_all_chunk_sizes PASSED
code/test_evaluator.py::test_t8_micro_macro_recall_mrr_ndcg[1] PASSED
code/test_evaluator.py::test_t8_micro_macro_recall_mrr_ndcg[4] PASSED
code/test_evaluator.py::test_t8_micro_macro_recall_mrr_ndcg[11] PASSED
code/test_evaluator.py::test_t8_chunk_sizes_agree_exactly PASSED
code/test_evaluator.py::test_t8_broken_group_by_client_id XFAIL
code/test_evaluator.py::test_t8_broken_drop_oov_rows XFAIL
code/test_evaluator.py::test_t8_broken_target_always_wins_ties XFAIL
code/test_evaluator.py::test_t8_broken_filter_seen_items XFAIL

======================== 13 passed, 8 xfailed in 1.55s =========================
```

## Counts

- **passed: 13**
- **failed: 0**
- **xfailed: 8** (all strict; none xpassed)

Breakdown of the 8 strict-xfail broken variants: 2 in `test_objective.py` (divide-by-B, average-of-microbatch-means), 6 in `test_evaluator.py` (T9 group-by-`client_id`, T9 drop-OOV-rows, T8 group-by-`client_id`, T8 drop-OOV-rows, T8 "target wins ties" rank rule, T8 filter-seen-items). Every one genuinely raised `AssertionError` against the correct hand-derived expected values (verified individually before wiring the tolerance fix below) — none passed by accident (`strict=True` would have turned an accidental pass into a hard failure, and none did).

## T9 fixture — computed values

Computed directly from `evaluate()` (not just asserted in the test; printed via a standalone script against the same fixture builders the tests use):

| Quantity | Value |
|---|---|
| Macro MRR@20 (A unseen×2 rank1, B unseen rank25, C warm rank2) | **0.5** |
| Micro MRR@20 | **0.625** |
| `user_count` (macro, grouped by `eval_user_key`) | **3** |
| OOV extension — end-to-end macro MRR@20 (C gets an extra OOV row) | **0.41666666...** (0.4167, matches spec) |
| OOV extension — closed-catalogue macro MRR@20 (unchanged) | **0.5** |
| OOV extension — end-to-end n / closed n | **5 / 4** |

Broken `client_id`-grouping variant (documented inline in the xfail test): macro MRR@20 = 0.58333... with `user_count == 2`, confirmed to differ from the correct 0.5/3 and therefore fail as required.

## Chunk-size agreement (T8)

`evaluate()` was run with `chunk_size ∈ {1, 4, 11}` over the K=11, 7-decision T8 fixture; all three produced **byte-for-byte identical** `n` / micro / macro dictionaries (`test_t8_chunk_sizes_agree_exactly`, direct dict equality — not just approximate). Per-row ranks from the internal chunked accumulator (`_ranks_chunked`) were also compared directly against an independent brute-force stable-sort-by-`(-score, class_id)` reference for the same three chunk sizes and matched exactly (`test_t8_chunked_ranks_match_bruteforce_for_all_chunk_sizes`).

T8 hand-derived ranks (k=3): R1 (tie-with-lower-id) = 3 (hit, target loses the tie as required), R2 (tie-with-higher-id) = 5 (miss, target wins the tie as required), R3 = OOV (zero credit, end-to-end only), R4 = excluded (`eval_mask=False`), R5 (revisit, item also in context) = 1 (hit — confirms no candidate filtering), R6 = 2 (hit), R7 = 10 (miss). End-to-end n=6, closed n=5 (R4 always excluded; R3 dropped only from the closed population). Micro/macro Recall@3, MRR@3, NDCG@3 for both denominators were hand-computed from these ranks and hard-coded in `EXPECTED` in `test_evaluator.py`; the implementation matches them to float32 precision (`abs=1e-6`; T9's power-of-2 fractions matched to `abs=1e-9`).

## Notes on implementation choices (not semantic reinterpretation)

- `LossAccumulator.mean()` raises `ZeroDivisionError` if called with zero accumulated count — callers must check `.n` first for the zero-contributing-rows case, matching the brief's explicit statement that `exact_ce_loss` itself (not the accumulator) is responsible for returning the autograd-connected zero.
- `evaluate()`'s `scores_fn` contract: a zero-argument callable whose call returns an iterable of `[B,K]` score batches covering `rows` in order (batch size is the caller's choice). This is the API surface implied by "`scores_fn` yields `[B,K]` scores" — the brief did not pin down the exact call signature further, and this shape was exercised with multiple different batch sizes across the tests (2, 3, 4) to confirm batching boundaries don't affect results.
- Return-dict shape (`end_to_end` / `closed`, each with `n`/`micro`/`macro`, plus top-level `user_count` mirroring `end_to_end.macro.user_count`, plus `cohorts.warm`/`cohorts.unseen` with the same nested structure) was likewise not pinned down beyond "Return micro/macro metrics for both denominators, `user_count`, and the cohort cuts (warm/unseen)" — this is one reasonable concretization, not a semantic decision about the metrics themselves.

## Unresolved concerns

None blocking. One item worth flagging for whoever wires this into the real training/eval loop: `evaluate()`'s `scores_fn` and `rows` shapes above are my concretization of an API the brief left partly open (see notes above) — if the loader/model integration code (T-A's `model.py` consumer, not yet written) expects a different calling convention, a thin adapter will be needed; the metric semantics themselves (rank formula, both denominators, OOV zero-credit-but-counted, macro grouping strictly by `eval_user_key`, warm/unseen cohort cut on `client_id`) are implemented exactly as specified and are the part this task was scoped to get right.

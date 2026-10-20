# Gate P brief (Opus lead → Fable 5.1), 2026-09-24

Your task and output format: `F:\Queens University\Microsoft Project\eda\SASREC_PERFORMANCE_EXECUTION_V1\10_FABLE_PLAN_REVIEW.md`.
Read in the same folder: `02_HANDOFF_AND_LESSONS.md`, `03_ARCHITECTURE_CONTRACT.md` §2–7, `04_EXPERIMENT_PROTOCOL.md`, `config/run_registry.json`. Nothing else is needed. No code exists yet. Do not read historical TEST tables.
Begin your answer with `ROUTING: <exact model id from your system prompt>`.

## 1. Live preflight (measured today on Lobot, not historical)
- Server `jupyter-group-digi2026-g8`. Both A16s idle (3 MiB used, no compute apps), ~14.6 GiB usable each. Pod cgroup memory.max = 64 GiB. cpu.max = `max` (no visible CPU quota; affinity 256). Policy stays at 3 threads per training process.
- Free disk 21.871 GiB. torch 2.13.0+cu130, numpy 2.4.6, pyarrow 21.0.0, all unchanged.
- Frozen dataset re-hashed 409/409 OK, version hash as in 02.
- GRU v1 locks/checkpoints intact and untouched. The new workspace `.local_ai/sasrec_v1` is Git-excluded.
- The audited GRU store/evaluator/baseline modules are reused as a byte-identical, read-only, hashed copy (14 files).

## 2. Batch contract actually produced by the reused CompactStore (read-only facts)
- Right-padded: real events occupy columns `0..len-1` in chronological order; PAD=0 after. `position_ids` = 1..len on real events, 0 on PAD; `last_valid_index = len-1`; `attention_mask` bool; `lengths` 1..50.
- ID arm keys: `item_tokens attention_mask position_ids lengths last_valid_index target_class loss_mask eval_mask decision_id client_id`.
- Rich arm adds:
  - categorical tokens: category/brand/main_category/daypart/weekday; vocab rows 663/3956/16/8/10;
  - `event_numeric_features [B,L,19]` (already normalised by frozen normalisers; sin/cos unscaled; missing-flagged values zeroed);
  - `event_quality_flags [B,L,15]` int8;
  - `user_context [B,12]` and `user_context_masks [B,6]` at the decision's own event.
- Item vocab 158,489 rows (PAD 0, MISSING 1, OOV 2); K=158,486 with `product_idx_of_class == arange(K)+3`, asserted from the catalogue file.
- Forward allowlist: the reused evaluator calls the model only with the item/mask/length keys plus the arm's declared side channels. target_class, decision_id, client_id, eval_user_key and split names are never passed.
- TRAIN window length (read-only statistic over all 40,180,608 decisions):
  - mean 8.31, median 5;
  - 18.1% have length 1 and 1.8% have length 50;
  - **99.0% of random 256-batches contain at least one length-50 window**, so cropping trailing PAD columns saves nothing.

## 3. Sizing micro-benchmark (SYNTHETIC tokens, throwaway module, real length histogram)
Pre-LN 2 blocks, FF 4d, dropout 0.2, exact [256,K] CE, fused AdamW, strict FP32. Median ms/step; decisions/s in brackets.

| d (heads) | head+CE+AdamW only | padded L50 | crop-to-batch-max | unpadded token-wise ops | eval rows/s (unpadded, B=1024, exact rank) |
|---|---|---|---|---|---|
| 128 (4) | 35.1 (7293) | 72.0 (3554) | 72.4 | 56.5 (4528) | 8983 |
| 256 (4) | 53.6 (4780) | 140.6 (1821) | 141.0 | 88.6 (2888) | 6769 |
| 384 (6) | 74.1 (3454) | 245.7 (1042) | 246.2 | 134.0 (1910) | 5087 |

"Unpadded" means LN/QKV/FFN/residual/dropout run on the valid tokens only (N≈2,128 per batch). q/k/v are scattered into the right-padded [B,T] layout only for the causal attention product, then gathered back. The math on real positions is identical. The FP32 [B,K] head sets the floor.

Projected wall time with the unpadded layout: epoch + full-VS eval ≈ 2.7 h (d128), 4.2 h (d256), 4.5 h (rich d256), 6.3 h (d384).

| Phase | Duration |
|---|---|
| A0‖A1 | 27–36 h |
| GPU0 C0 ‖ GPU1 C1→R0 | 41–56 h |
| Replicas | 16–50 h, depending on the selected configuration |
| Retrospective eval | ~3 h |
| **Total** | **≈ 3.6–6.0 days**, at 6–8 epochs per run |

Production code may be 10–20% faster (valid-only embedding lookup) or slower (deterministic kernels). It will be re-benchmarked after Gate C.

Disk: a full checkpoint is ≈ 0.51 GB (d256), 0.78 GB (d384) or 0.25 GB (d128). The plan keeps latest+previous per active run, best as weights only, and drops `previous` at completion. Peak new disk ≈ 9 GB, leaving ≥ 12.8 GiB free.
RAM: rich store ≈ 15.4 GiB and ID store ≈ 2 GiB; the concurrent pair needs ≈ 26 GiB of 64.

## 4. Proposed implementation decisions (accept, amend or reject each)
- **D1 Positions.** Contract 03 §2: real-event positions 0..len-1 counted from the oldest retained event, derived from the mask (`cumsum-1`) and asserted equal to `position_ids-1`. Note: for the 1.8% of windows cropped at 50, position 0 is "50 events ago", not session start. Keep as-is or not?
- **D2 Attention.** Production uses an explicit FP32 math causal attention (lower-triangular incl. diagonal, right padding, no fully-masked rows), deterministic, over the unpadded token-wise ops. SDPA would be used only if it is faster AND passes the 05 B-tests, which is unlikely to matter at T≤50.
- **D3 Pair barrier (A0/A1).** Epochs 1–4 run free. From epoch 5 on, after each epoch-end eval, an arm writes its record and waits (GPU idle) for the partner's same-epoch record. Both then apply one deterministic joint rule (04 Phase 2), so neither arm can get an extra epoch. If the partner is permanently stopped, the waiting arm exits cleanly with a resumable checkpoint.
- **D4 Queue.** GPU0 A0→C0, GPU1 A1→C1→R0, then both GPUs run the two replica seeds. Selection is a pure function of the five summaries. Lock, retrospective eval and report are automatic. There is no LLM between stages.
- **D5 Eval.** Full VS at each epoch end; @1/2/5/10/20 micro+macro from the same rank vector. The 112k-decision diagnostic cohort is logged every 40k updates (monitoring only, never selection).
- **D6 C1 smoothing.** `F.cross_entropy(label_smoothing=0.05)` over all K classes = (1−ε)·CE + ε·mean_j(−log p_j). Is this acceptable as the full-catalogue formula?
- **D7 Rich fusion.** Exactly 03 §5. Common weights are initialised identically to A0 from the same seed; rich-only weights come from a separate generator.
- **D8 Challenger budget.** C0_WIDE384 is the most expensive run (≈ 38–50 h), and at d384 the head dominates its cost. You may keep it, or replace it with ONE alternative (e.g. d256 with 3 blocks), before preregistration.
- **D9 Precision.** Strict FP32 everywhere, no AMP study in this campaign.

## 5. What I need from you
Your verdict and structured output per 10_FABLE_PLAN_REVIEW.md. Also give an explicit keep/replace decision for D8, an answer on D1, and any change to the schedule/stopping/min-low-LR rules before preregistration.

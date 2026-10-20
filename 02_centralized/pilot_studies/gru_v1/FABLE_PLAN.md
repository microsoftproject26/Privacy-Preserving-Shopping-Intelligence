# FABLE 5.1 — Pre-implementation review of ContextGRU-v1

VERDICT: **FIX_REQUIRED** — architecture is compatible with the frozen contract; five non-architecture items must be fixed first. No second review round needed if they are addressed.

Labels: LIT = literature-backed · PSI = project-specific inference (file:line) · EMP = must benchmark.

## Mandatory corrections (severity order)

1. **[HIGH] No evaluation-user identity in the batch.** `client_id` is `client_idx` with unseen users collapsed to OOV=2 (`loaders.py:320-321`; `pipeline7.py:90` `coalesce(cl.client_idx,2)`). 1,139,657 VS clients, ~45% unseen → macro-MRR grouped on `client_id` merges ~500k users into one group. Fix = dense `eval_user_key` (§E).
2. **[HIGH] Loader memory model incompatible with a TRAIN-spread 1M manifest.** `BucketStore` reads every column incl. 9 string cols as object arrays (`loaders.py:94-102`, no projection at `:191`), builds a per-row Python dict (`:112`), and `_buckets` is never evicted (`:174,:188-192`). 75,083,291 rows / 64 buckets ≈ 1.17M rows/bucket; a spread manifest touches all 64. Plus a 4,204,379-entry str→int dict per loader instance (`:183-185`), duplicated per worker. Fix = bucket-major pre-materialised window cache (§G).
3. **[HIGH] `target_class == -1` semantics ambiguous.** `DATA_CONTRACT.md:131-132` says exclude from the denominator, but `pipeline5.py:394-404` sets −1 for BOTH no-target (`eval_mask=false`) and OOV-target (`eval_mask=true`). Evaluator must use denominator = `eval_mask`, and `eval_mask & target_class<0` = OOV = zero credit. Wrong handling silently drops 49,545 VS rows and inflates every metric.
4. **[MEDIUM] Vacuous readiness check.** `next_item_eda/scripts/post_freeze_smoke.py:101`: `bool(np.any(b["client_id"]==OOV)) or True`. Replace with the deterministic cold-user fixture (T11).
5. **[MEDIUM] Pilot TRAIN population must filter `loss_mask`.** `index("TRAIN")` returns all 52,760,314 rows; only 40,180,608 have `loss_mask=true`. The loader does not filter (`loaders.py:194-231`).
6. **[MEDIUM] Validate class→token map, then exploit it.** `pipeline5.py:304-305` defines `product_idx = output_class + 3`; assert at startup, gather fallback otherwise.
7. **[MEDIUM] Loader vendoring / cross-platform parity.** `rees46_eda` absent on Lobot; `config.py:15` imports yaml at module import; shared venv immutable. Needs a hash-pinned vendor plus a cross-platform digest.
8. **[LOW] Stale prose.** `DATA_CONTRACT.md:137-152` claims the frozen copy needs folder renaming; `loaders.py:121-149` opens the frozen layout directly. Follow the loader.

## Confirmed facts (file:line)

- Right padding, valid `[0,length)`, `last_valid_index = lengths-1`: `loaders.py:273-282`; asserted for all splits and L∈{20,50} at `pipeline7.py:216-232`.
- One index row per decision; scalar `target_class/loss_mask/eval_mask`: `loaders.py:255-257,316-318`; each eligible TRAIN decision emitted exactly once `pipeline7.py:305-310`.
- Window = decision's own event + ≤49 earlier events of the same model segment; target strictly later, never in the window: `loaders.py:267-271`, `DATA_CONTRACT.md:121`.
- `loss_mask ⇒ target_class ≥ 0` (`pipeline7.py:245-246`); `eval_mask ∧ target_class=-1 ⇔ target_oov`.
- Index columns incl. raw `user_id` STRING and `target_oov`: `pipeline7.py:89-96`.
- Vocabs: catalogue 158,486; clients 4,204,379; brand 3,953; category_id 660; main_category 13; daypart 5; weekday 7. All model indices start at 3 (KNOWN_START).
- Widths: 19 numeric (14 event + 5 session-prefix), 15 flags, 12 user-ctx, 6 masks. F4 concat = 240.
- **F4 − F3 = exactly one channel, `log1p_price`.** F2/F3/F4 all carry all 15 quality flags including the three price flags; F3 and F4 both carry all 12 user-context values including `recent_mean_log_price`.
- Local clock = UTC+3 fixed, no DST (`config.py:229-235`).
- Split sizes: TRAIN 52,760,314 / 40,180,608 loss-contributing / 4,204,379 clients; VS 10,930,090 / 8,826,488 evaluable / 1,139,657 clients / 49,545 OOV targets; TEST 5,862,968 / 198,319 OOV.
- Cold rebuild compared 5 of 74 feature columns, did not compare `normalizers.json`.
- Training repo: `ppsi/models/session_gru.py:29-36` consumes `Phase1Batch` — pattern reusable, code not.
- `feature_group_view`/`gru_inputs` pass through only `OPTIONAL_CHANNELS` (`loaders.py:424-443`) → any new key such as `eval_user_key` is structurally dropped before model inputs.

## B. ARCHITECTURE

No alternative proposed; no defect justifies one.

Embeddings: item 128 (rows = max(product_idx)+1 = 158,489, padding_idx=0), category 32 (663), brand 32 (3,956), main 8 (16), daypart 4 (8), weekday 4 (10); numeric(19)⊕flags(15)=34 → Linear(34→32)+GELU.
Concat 240 (F4) / 128 (F0) → Linear(→128) → LayerNorm → Dropout(0.20).

Backbone: `nn.GRU(128,256,num_layers=1,batch_first=True,dropout=0)`, `pack_padded_sequence(enforce_sorted=False)`, readout `h_n[-1]` → [B,256], zero initial state, no state carry.

User context: concat(12 values, 6 masks) → Linear(18→64)+GELU → late fusion at readout only.

Readout: F0 Linear(256→128); F3/F4 Linear(320→128); Dropout(0.10) → q[B,128].

Tied head: assert `product_idx_of_class == arange(K)+3`; then `W_out = E_item.weight[3:3+K]` (view, one Parameter, no copy/detach); else gather. `logits = q @ W_out.T + b`, shape **[B,158486]**, never [B,L,K].

Parameter inventory: F4 **20,964,062 (83.9 MB FP32)**, F0 20,791,190. Implementation must print and match.

## C. TRAINING OBJECTIVE — exact full-catalogue CE is PRIMARY

[B=256,K=158486] FP32 logits = 154.8 MiB; ~3× for log-softmax + grad ≈ 465 MiB peak head activations. Head compute ≈122 MFLOP/decision fwd+bwd → a small fraction of one A16. Not a VRAM/FLOP problem.

Loss: rows with `loss_mask & target_class>=0`; `cross_entropy(..., reduction="sum")/n_sel`; accumulation sums numerators and denominators separately. No label smoothing / class weights.

AdamW lr 1e-3, betas (0.9,0.999), eps 1e-8, wd 1e-5 on matrices only, clip 1.0 after unscale, 100-update linear warmup then constant, effective batch 256, seed 2026.

**Pre-registered decision rule:**

- **E1 (VRAM)** microbatch 256 / F4 / FP32, `max_memory_reserved` after 50 steps: ≤8 GiB → keep exact CE at 256. 8–13 GiB → microbatch 128 + accumulation 2, still exact. >13 GiB at 128 → implementation bug (likely a [B,L,K] head or unfreed eval buffer); fix, do not change objective. Expected ≈1.5–2.5 GiB.
- **E2 (throughput)** GPU-only cached-batch step, 20 warmup + median of 50, synchronized, microbatch 256 / F4: ≥2,000 valid decisions/s → exact CE primary (40.18M-decision epoch ≤5.6 h GPU). 800–2,000 → keep exact for pilot, flag for full run. <800 AND head+loss ≥60% of step time → run the sampled recipe as a separately named configuration; never switch mid-run.

**Sampled-CE fallback (complete spec, only if E2 fails):** proposal `q(j) ∝ representative_views(j)^0.5` (TRAIN-only support); ONE shared set of N=2048 negatives per microbatch via `multinomial(replacement=True)`; duplicates kept; no in-batch mixing in v1; candidates `[positive, n_1..n_N]`; subtract `log(N·q(j))` from every column including the positive; accidental hits (negative == row target) → −inf before softmax; CE with target index 0; same mask/denominator. **Evaluation is always exact full-catalogue.** Never compare loss values across objectives. The frozen `negative_sampler` (`loaders.py:503-527`) is uniform/per-decision and is NOT the training proposal.

## D. BATCH/PADDING CONTRACT — assertions the implementation must test

Confirmed: right padding, `pack_padded_sequence(enforce_sorted=False)` applies directly, no compaction adapter needed. Training only the last position neither drops a contributing decision nor re-counts a label (prefix positions belong to other decisions with their own rows; supervising them would count each label up to 50×). Only adapter work: `event_quality_flags` int8→float32; `lengths` to CPU for packing.

Assert on every real batch:

1. `attention_mask == (arange(L)[None,:] < lengths[:,None])`
2. `last_valid_index == lengths-1`; `1 ≤ lengths ≤ L`
3. `item_tokens[~attention_mask]==0`; `item_tokens[attention_mask]!=1`
4. all categorical tokens < their table rows (an item table sized K instead of K+3 is the classic failure)
5. `target_class ∈ {-1} ∪ [0,K)`; `loss_mask ⇒ target_class≥0`; TRAIN has no eval_mask; future splits no loss_mask
6. pilot training batches: `loss_mask.all()`
7. numerics and user_context finite
8. no code path indexes `item_tokens` with `target_class`
9. Cold users carry z-scored constants (e.g. `(0-2.71272)/1.4865` for `log1p_prev_view_count`) alongside `has_history_mask=0`; only the two price user-features are re-zeroed after normalisation (`loaders.py:60-63,309-311`). Record, do NOT "fix" in the trainer.

## E. EVALUATION CONTRACT

Selection metric: **Client-Macro MRR@20 on VALIDATION_SELECT only.** Also report Micro MRR@20 / Recall@20 / NDCG@20 and Macro Recall@20, plus closed-catalogue variants with explicit denominators.

Rank: `1 + #{j: s_j > s_t} + #{j < t: s_j == s_t}` (ascending class id wins ties; the target never benefits). Computed exactly in class-chunks accumulating both counts; top-k never materialised.

Candidates: full frozen catalogue K=158,486; seen/revisited items NOT filtered.

Denominator (end-to-end) = `eval_mask`; among them `target_class==-1` = OOV = credit 0 but counted. `eval_mask=false` rows excluded entirely. Closed-catalogue denominator = `eval_mask & target_class>=0`.

**Macro grouping fix:** the cohort builder reads the projected index (`decision_id,user_id,client_idx,user_bucket,target_class,eval_mask,target_oov`), takes `sorted(set(user_id))` over eval rows, samples 10,000 with `default_rng(2026)`, assigns `eval_user_key = dense position in that sorted list`. Persist `decision_id,user_bucket,eval_user_key,client_is_oov,target_oov` — **never `user_id`**. Attach `eval_user_key` to the batch beside the loader output; it is NOT in `OPTIONAL_CHANNELS` so `feature_group_view`/`gru_inputs` drop it and the model cannot consume it. `client_id` is used only for the warm/unseen cohort cut, never as a grouping key. The history-available cut uses `user_context_masks[:,0]`.

**Negative fixture (T9):** users A(unseen, ranks 1,1), B(unseen, rank 25), C(warm, rank 2). Correct macro MRR@20 over distinct users = mean(1.0, 0.0, 0.5) = **0.5**, user_count 3. Grouping by `client_id` gives 0.5833 / count 2 → must fail. Micro = 0.625. Extension with an OOV-target row for C: end-to-end macro = 0.4167, closed-catalogue macro stays 0.5; dropping −1 rows returns 0.5 → must fail.

**Tiny exhaustive reference (T8):** K=11, 7 decisions, k=3; includes tie-with-lower-id, tie-with-higher-id, OOV target under eval_mask, an eval_mask=false row, a revisit, users A/B/C. Chunk sizes {1,4,11} must match brute force. Broken variants that must fail: group by client_id; drop −1 under eval_mask; target-wins-ties; filter seen items.

## F. PILOT PLAN

TRAIN manifest: rows with `loss_mask=true` (40,180,608), sorted by `decision_id`, `default_rng(2026).choice(n, 1_000_000, replace=False)`. Persist `decision_id,user_bucket` + sha256 in `PILOT_MANIFEST.json` with `version_hash 860e5a1c…`.

VS cohort: 10,000 users, all their eval_mask decisions; report actual counts (≈77k expected, EMP).

Arms: GRU-F0 (`F0_item_only`) vs GRU-F4 (`F4_rich_side_information`), same loss/optimiser/seed/order. Accurate label: "item-only vs all side information", NOT a price ablation.

Baselines on the same 1M manifest, same evaluator/tie rule: POP (`count(target_class=j)`) and TRANS (first-order transition with popularity tie-break/fallback, int64 composite score).

Exposure: exactly 2 passes × 1,000,000 = 2,000,000 positives per arm = 7,813 updates at effective batch 256.

**Pre-registered acceptance criteria:**

Engineering PASS (all required):
- P1 both arms finish 2 passes; zero non-finite losses; consumed loss-contributing decisions == 2,000,000 exactly.
- P2 resume: interrupt at update 300, resume, losses at updates 301–310 match the uninterrupted run to rel 1e-4 (FP32 deterministic).
- P3 two evaluations of the same checkpoint → bit-identical metrics.
- P4 all §H tests pass and every deliberately broken variant fails.
- P5 `max_memory_reserved` ≤8 GiB at microbatch 256 and host RSS ≤24 GB.

Throughput (reported; gates the full-run plan): E2 as in §C; end-to-end decisions/s ≥ 0.5 × GPU-only decisions/s.

Learning signal floor: training-loss EMA(200) at end of pass 1 ≤ EMA(first 200) − 1.0 nat (untrained CE ≈ ln K = 11.97).

Baseline comparison (Micro+Macro MRR@20, end-to-end denominator):
- **GO-1 (mandatory)**: `Recall@20(F0) ≥ max(1.5×Recall@20(POP), Recall@20(POP)+0.02)` AND `MRR@20(F0) > MRR@20(POP)`.
- **GO-2**: `MacroMRR@20(F0) ≥ MacroMRR@20(TRANS)` → full run approved. `0.9×TRANS ≤ F0 < TRANS` → approved with exactly ONE permitted change (LR 1e-3→5e-4 or 2e-3 from the pass-1 slope, OR extension to 3 passes — not both). `< 0.9×TRANS` → NO-GO, engineering review.
- Overfitting check: pass-2 Micro MRR@20 ≥ pass-1 − 0.002, else use the best checkpoint.

Side-information gain (never gates the GRU): paired bootstrap over the 10,000 cohort users (1,000 resamples, seed 2026) of `MacroMRR@20(F4) − MacroMRR@20(F0)`; 95% CI lower bound >0 = gain detected; contains 0 = none detectable at pilot exposure; upper bound <0 = harmful. Report overall, warm-only, unseen-only.

Full run warranted iff P1–P5 and GO-1 and GO-2 (either approving branch). The full run restarts from scratch on all 40,180,608 contributing decisions; a pilot checkpoint is never reported as a full-data model.

## G. RESOURCE PLAN

Persistent GPU: weights 84 MB + grads 84 MB + AdamW moments 168 MB ≈ 336 MB + context. The tied item table gets a DENSE gradient (81 MB) every step — do not use sparse embeddings.

Activations: head ≈465 MiB @ microbatch 256; GRU/embeddings/user branch <60 MiB @ B=256,L=50. Expected peak reserved ≈1.5–2.5 GiB. Start microbatch 128 for FP32 correctness, then benchmark {128,256,512}; E1 decides. Eval scoring: 512 rows × K × 4 B = 324 MB per chunk.

AMP: Phase 1 strict FP32 (`allow_tf32=False`, `cudnn.deterministic=True`). Phase 2 TF32 matmul + measure (preferred first speed-up, no GradScaler). Phase 3 only if the head dominates: `autocast(bfloat16)` around the head matmul only, `logits.float()` before CE, GRU kept FP32 (cuDNN RNN bf16 support is version-dependent); FP16+GradScaler only if bf16 is unsupported. AMP is justified by the benchmark, never a default.

**Loader fix:** pre-materialised bucket-major window cache — group the manifest by `user_bucket`, process one bucket at a time (or ≤4 processes on disjoint buckets), evict via `loader._buckets.pop(b)`, write memmapped arrays mirroring the batch schema (tokens int32, cast back to int64 on load): ≈5.9 KB/decision at L=50 → ≈5.9 GB for 1M + cohort; F0 needs only `item_tokens` (200 B/decision). Record a blake2b digest + manifest sha256 in the header. Training/eval iterate the memmap with a trivial DataLoader; no forked worker ever builds a `DecisionWindowLoader`. Index reads use pyarrow column projection instead of `loader.index()`. The direct loader is retained as the parity reference (T3).

**Benchmark (EMP, JSON out):** (1) GPU-only cached-batch steps for B∈{128,256,512} × {F0,F4} × {FP32,TF32}, 20 warmup + 50 timed, CUDA events around embed+GRU / head+loss / backward / optimizer, allocated+reserved, decisions/s and tokens/s; (2) direct-loader collate throughput + RSS on 4,096 decisions from 4 buckets, cold and warm; (3) cache-build throughput per bucket; (4) 500-step end-to-end from cache. No epoch estimate before these.

**Loader location:** vendor `__init__.py, config.py, loaders.py, torch_adapters.py` byte-identical with per-file sha256 + source `code_hash 02fbb071…` in `VENDOR_MANIFEST.json`. Risks: (a) `import yaml` at `config.py:15` — check in `~/venvs/ppsi-gpu`; if absent the single permitted edit is a lazy import, recorded as a diff; (b) pyarrow/NumPy skew Windows vs Lobot — caught by T2; (c) the frozen layout opens natively (`loaders.py:125-127`), no renaming, dataset stays read-only.

## H. BLOCKING TESTS (each with the variant that MUST fail)

- **T1** padding invariance vs an independent unpadded per-row forward, FP32/TF32-off, max|Δ|≤1e-5, incl. length 1 and 50. Broken: `output[:,L-1]` trailing-PAD readout; left-padded input.
- **T2** cross-platform batch digest: 256 fixed TRAIN decision_ids from buckets 0,1; blake2b over every batch array in a fixed key order; Windows == Lobot. Broken: change one normaliser constant.
- **T3** cache parity: 512 random manifest rows, cache == direct `collate` exactly (after int32→int64). Broken: off-by-one row in the writer.
- **T4** class→token map + head candidates: assert `product_idx_of_class == arange(K)+3`; a permuted synthetic catalogue routes to gather and matches brute-force dot products; `W_out.shape==(K,128)`; no row of W_out is `E_item[0:3]`. Broken: table sized K instead of K+3 → out-of-range must raise.
- **T5** tied-head gradient + inventory: `E_item.weight.grad[3+t]` nonzero, `grad[0]` zero, exactly one item Parameter, printed inventory == §B numbers. Broken: `.detach()` / copied output matrix.
- **T6** feature-group exclusion: F0 output bit-identical when all non-item channels are permuted; the F0 state dict has no category/brand/user params. Broken: F0 built from an F4 spec.
- **T7** loss masking + denominator: 3 of 8 rows masked → loss is the mean over 5; two-microbatch accumulation == single batch. Broken: divide by B; include masked rows.
- **T8** evaluator exhaustive reference (§E).
- **T9** OOV-merge macro fixture (§E).
- **T10** shape/range guards (§D 1–7) on real TRAIN and VS batches; pilot batches assert `loss_mask.all()`.
- **T11** deterministic cold-user fixture replacing `post_freeze_smoke.py:101`: 8 smallest VS decision_ids with `client_idx==2` and 8 with `>=3`; assert both non-empty and exact; plus one unseen user with ≥2 in-partition decisions asserting `has_history_mask==0` then `==1` later. Broken: the original `or True` passes vacuously.
- **T12** synthetic memorisation (Stage B): 256 synthetic decisions, K_syn=1000, target a deterministic function of the last token, dropout/wd off, ≤300 full-batch updates → final loss ≤0.05×initial and top-1 ≥0.95, for F0 and F4 wiring. Negative control: constant input → top-1 ≤ majority rate + 0.05.
- **T13** resume determinism (P2).

## I. DEFERRED

Sampled CE with in-batch mixing; BPR-max; multiple seeds; LR schedules beyond constant-after-warmup; L=20 arm; F1/F2/F3 arms (accurate labels: F1→F2 = time+gaps+age+all flags; F2→F3 = session-prefix + user context; F3→F4 = per-event `log1p_price` only); a trainer-side selector dropping the linearly redundant `local_hour_sin/cos` (an exact rotation of the UTC pair under fixed +3h; the weekday pairs are NOT redundant because the shift crosses midnight); learned user-ID embedding; SASRec; DDP; hidden-state carry-over; widening the cold-rebuild digest to all 74 columns.

Targeted reproducibility checks before the full run (read-only): recompute clip/mean/std for the 5 z-scored event numerics and 9 z-scored user-context features vs `mappings/normalizers.json` within 1e-4 relative; invariants on a 100k TRAIN sample (`prefix_item_count == position_in_segment`, `prefix_representative_count == position_in_segment-1`, `log1p_prefix_elapsed_s == log1p_segment_elapsed_s`, `is_weekend == (local_weekday_idx ∈ {8,9})`, local hour pair == UTC pair rotated +3h, daypart consistency, `price_missing ⇒ log1p_price == 0`). A narrow digest, not a blanket correctness claim.

## J. EXECUTION TASKS (Fable's numbering)

1 vendor loader + yaml check + frozen-layout assert · 2 cross-platform digest (T2) · 3 projected index utils + manifests (1M TRAIN, 10k-user cohort, PILOT_MANIFEST.json) · 4 window cache builder + T3 · 5 ContextGRU-v1 module + class-map validation + inventory · 6 model tests T1,T4,T5,T6,T10 with strict-xfail broken variants · 7 exact-CE objective + T7, sampled-CE behind a disabled flag with its own toy test · 8 evaluator + T8,T9 · 9 POP/TRANS baselines · 10 synthetic memorisation T12 · 11 benchmark script + mechanical E1/E2 verdict · 12 training loop + checkpoint/resume + T13 · 13 pilot runs F0 then F4 · 14 go/no-go report generator + paired bootstrap · 15 repair post_freeze_smoke.py:101 (T11) · 16 targeted reproducibility checks.

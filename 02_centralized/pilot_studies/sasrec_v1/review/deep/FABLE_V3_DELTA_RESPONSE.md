# Fable v3 delta confirmation (verbatim)

- agent: sasrec-fable-reviewer (Read/Grep/Glob)
- transcript model ids: ['claude-fable-5-1']
- tools: ['Grep', 'Read']

## Correction-only review

ROUTING: claude-fable-5-1

VERDICT: APPROVE_FOR_REAL_TRAINING (conditional on attested overall_pass=true)

Checks (all [evidence] unless marked):

- (1) Untied head: `output_embed` [K,d] created after `final_norm`/`input_norm` and drawn from the common generator `g` before the rich `g2` scope, so A0/A1 common init is identical — `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\code\sasrec_model.py:158-161, 203-208`. `output_weight()` returns it (:334); arange gate intact (:139-140); `forward` uses `addmm(output_bias, q, W^T)` (:372).
- (2) Input LN, no √d: both paths apply `input_norm` before dropout (:291 unpadded, :309 padded with `*vm` after LN, so PAD rows are re-zeroed; LN is per-token, hence the two impls stay equivalent). `scale` removed; `math` still needed for `sqrt(dh)` (:102).
- (3) BETAS (0.9, 0.999) — `code\train_sasrec.py:58`; recorded in run config (:524) and registry (`training.betas`, `optimizer`).
- (4) `recenter_output_()` runs once per update, immediately after `opt.step()`, after the non-finite guard, accumulation=1 — `train_sasrec.py:708-709`. Loss-neutral by construction (:341-342).
- (5) `stat_rows` fixed 8192-row sample, rng 4242 (:509-510, `np` imported :48); `emb` logged in `update_agg` every 100 updates (:877-886).
- Optimizer groups: `ndim>=2` rule → `output_embed` in the 1e-5 decay group, `input_norm` params in no-decay; duplicate-Parameter guard intact (:297-303).
- Version-3 check (:386); FALLBACK_EPOCHS_FP32=4 (:61).
- Registry v3: peak_lr 1e-3 both arms, planned 4 / min 4 / eval 3..4, `go_no_go_20k`, `v2_failure`, `deviations_from_03`, implementation strings — `config\run_registry.locked.json:37-42, 53, 65, 148-151, 250-297`.
- Tests: `forward_kwargs(batch)` sed applied (`tests\test_head_objective.py:49, 65`; conftest signature :156); `retie_head_weight` / `recenter_by_row_` present with `types`/`torch` imported (`tests\mutants\model_mutants.py:70, 86`); `test_rich.py:15-16` prefixes updated; `test_attention.py:36-38` reference builder uses `F.layer_norm` with `input_norm` params.
- Scope: the three scripts contain only the listed find/replace pairs; no other code references the tied head (grep of `code\` excl. tests).

Defects (none blocks training; correct at LOCK/report, do not silently re-hash the attested registry):

1. `config\run_registry.locked.json:26` — `"tie_item_embeddings": true` is stale and contradicts `implementation.tying` (:148). Not read by any code [evidence: grep], but a self-contradiction in a locked record.
2. `config\run_registry.locked.json:246` — `fallback_if_gate_fails` says planned_epochs 5 vs `FALLBACK_EPOCHS_FP32 = 4`. Moot (parity passed) [inference].
3. `code\SASREC_INTERFACES.md:56` — documents the v2 tied-head semantics; doc drift only.
4. `code\tests\test_attention.py:9` — `import math` may now be unused; harmless [inference].

## Errata (per Fable: record at lock/report, do not re-hash the attested registry)
1. registry training.tie_item_embeddings=true is stale; v3 is UNTIED (implementation.tying).
2. registry precision.fallback_if_gate_fails mentions 5 epochs; code FALLBACK_EPOCHS_FP32=4 (moot, parity passed).
3. code/SASREC_INTERFACES.md sec 2 still documents the v2 tied head.

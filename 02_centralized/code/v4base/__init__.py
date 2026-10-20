"""v4base — the non-neural next-item BASELINEs registered in `contracts/PREG1_AMENDMENT_02.md` §4 and
`contracts/PREG1_AMENDMENT_04.md` §2 (Fable's amended BEST_BACKBONE ruling / the realistic downstream plan):
POP (global popularity of the fitting cohort's TRAIN targets), LAST (last-item repeat, scored by recency in the
causal history), SKNN (session item-kNN, Ludewig & Jannach 2018 style, vanilla/idf-free cosine, most-recent
sampling; grid k in {50,100,200,500} x sample_size in {500,1000,5000}), KNN_EMB (BASE-02: cosine of the mean
embedding of the permitted history's last 10 items against a cached item-embedding table read from a GRU/SASRec
checkpoint) and RULE (BASE-02: category/brand affinity x MAIN-TRAIN popularity). `strata` (BASE-02) computes and
freezes TRAIN-period user strata (category diversity, brand loyalty, persona, activity quantiles), used by Ops/Fable
for reporting, never fed into a T3 evaluation.

Task packets: `company/tasks/BASE-01.md`, `company/tasks/BASE-02.md`. Design and every deviation are written up in
`company/handoffs/BASE-01.md` / `company/handoffs/BASE-02.md`.

Scope discipline (hard rule): this package and `tests_base` are the ONLY new code; `v4eval`, `v4int`, `v4train`,
`v4models`, `v4hist`, `v4data` are read-only dependencies here, imported through their existing public functions
exactly as `v4train` / `v4int` already do (never monkeypatched, never edited).

  sessions   shared Clean V2 batch readers (item classes, causal-history mask, current-session mask)
  pop        POP: fit() counts TRAIN target_class; PopScorer serves a broadcast [B, K] row (no [B, K] copy)
  last       LAST: LastItemScorer, stateless, scores the causal history by recency (finite sentinel; see scoring.py)
  sknn       SKNN: SessionPool.fit() over TRAIN, SessionKNNScorer(k, sample_size); score_session is the core algorithm
  grid       the registered SKNN grid (k x sample_size) and CALIB_VALIDATION macro-MRR@20 selection
  knn_emb    KNN_EMB (BASE-02): load_item_table() reads a GRU/SASRec checkpoint's own state dict; KnnEmbScorer
             (empty-history fallback = the SAME MAIN-TRAIN popularity counts POP already builds); the last
             `history_window` VALID items (skips OOV gaps), never a fixed column-recency cutoff
  rule       RULE (BASE-02): fit_category_brand_map() (one user-de-duplicated MAIN-TRAIN scan), RuleTables/RuleScorer
  strata     strata (BASE-02): scan_user_train_stats() (user-de-duplicated) + build_registry()/write_registry()
             (O_EXCL, self-hashed, keyed by client_key_hex) + assign_stratum_from_values()/assign_stratum_for_user()
             (NO_TRAIN_STRATUM sentinel for a user absent from the registry); `python -B -m v4base.strata --out <json>`
  identity   BASE-02: real per-decision user identity from a provider's OWN verified manifest tables (client_ord /
             client_key_hex for TRAIN, eval_user_key / client_key_hex for eval) — never the batch, never a raw
             dense rank used as a cross-manifest join key
  guard      the Amendment 02 §4 resource rule: os.nice(10), torch.set_num_threads(<= 4), a hard 16 GiB RSS guard
  scoring    the ONE frozen T3 path every baseline goes through (v4eval.predict.predict_scores + evaluate()),
             and BASELINE-labelled result rows (v4eval.rows.rows_from_result, run_label="BASELINE")
  run        the CLI (`python -B -m v4base.run --baseline {POP,LAST,SKNN,KNN_EMB,RULE} --phase {CALIBRATION,MAIN}
             --out DIR`)
"""
from __future__ import annotations

__all__: list = []

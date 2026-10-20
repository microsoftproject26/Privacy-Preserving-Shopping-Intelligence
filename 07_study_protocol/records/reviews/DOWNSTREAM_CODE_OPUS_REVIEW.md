# FL-0304 + BASE-02 + DEV-01 — independent Opus review (verbatim; read-only agent; received 2026-09-28 ~05:1xZ)

A. FL-0304 — ACCEPT_WITH_FIXES. DP-FedAvg mechanics correct (flat L2 clip over shared keys; z·S noise on the sum /
fixed m; dropped clients zero, m unchanged; CPU generator seeded (seed,"dp_noise",r); in-process ≡ pool; default digests
unchanged). Accountant correct (fractional series matches TF-Privacy _compute_log_a_frac; integer/composition/both
conversions correct; bisection returns the safe side). Independent mpmath check: ε(0.5642)=7.99825 @α2.9;
ε(0.56415)=7.99998; ε(1.3094)=0.99985 @α16; ε(1.3093)=1.00007 → z=0.5642 / 1.3094 are the correct 4-dp round-ups.
1 MED finetune.py:137-146 support_cutoff=None skips causality → make it required; for cold/U_NEW clients with several
  queries the cutoff must be the earliest decision_ts (or fine-tune per query).
2 LOW-MED runtime.py:236-242 uniform sampling + group_size==m enforced only if z>0 → enforce for every DP config incl.
  FA_1024 and the calibration pass (FA_1024 → DP differs only by noise).
3 LOW runtime.py:354-356, dp.py:192-196 per-client pre-clip norms logged/checkpointed in z>0 runs = non-private side
  output → calibration only, or mark non-releasable.
4 LOW finetune.py:129 FTRunner overwrites its adapter → refuse a server/live-worker adapter.
B. BASE-02 — ACCEPT_WITH_FIXES. Scoring via the frozen T3 path; fit reads MAIN TRAIN only.
1 HIGH strata.py:85-211 key space: client_ord / eval_user_key are per-manifest dense (M05/M06 re-densified);
  MANIFEST_SPEC: cross-manifest joins only by client_key_hex → translate client_ord → client_key_hex via the TRAIN
  clients table, key the registry by client_key_hex; eval_user_key → client_key_hex (manifest users table) → registry;
  users with no MAIN-TRAIN rows (cold/U_NEW) get an explicit NO_TRAIN stratum, never bucketed.
2 MED strata.py:108-120, rule.py:236-245 overlapping windows count each event up to ~50× (biases top_brand_share;
  diversity unaffected) → dedupe per user/event.
3 MED run.py:156-158 SMALL and BIG rows share label "KNN_EMB" → BASELINE_KNN_EMB_SMALL / _BIG + checkpoint sha256.
4 LOW knn_emb.py:130 last-10 = last 10 columns, not last 10 valid items.
5 LOW disclose per-item category share inflating tiny brands; Ops to confirm the quartile choice.
C. DEV-01 — ACCEPT_WITH_FIXES. Pack-free GRU query mathematically identical (1 layer, uni, zero h0, output at lengths−1
≡ packed h_n, same submodules, eval) — kernel-level float differences only; full-length SASRec trace valid for shorter
inputs (causal mask keeps real positions on real keys; lengths live input).
1 MED bench.py:198 parity vs the same exported reimplementation → also vs the production path (packed GRU, SASRec unpadded).
2 MED bench.py:152,194 ru_maxrss is whole-process peak → measure each session in a fresh subprocess.
3 MED bench.py:185 parity on synthetic tokens → with --rows compute FP32 and INT8 parity on 1,000 real rows.
4 LOW scoring.py:227 clamp silently scores length-0 rows → validate lengths ≥ 1 / right-padding in the rows loader;
  record onnxruntime version + optimisation level.
FL open questions: (1) accept normalised s(6·E/E_total), report raw EFE; (2) dropout_p=0 for FA_1024 and DP (T = 3N/m
and 6.0 EFE hold only without drop-out; no privacy credit taken); (3) unclipped calibration pass is the only
non-circular reading — run it with the exact FA_1024 config (seed 2026), disclose rounds 2–20 follow the unclipped
trajectory; (4) keep sweep sampling for FA/FP/PF (disclose partial 4th visit); FA_1024/DP uniform (disclose the
FA→FA_1024 sampling change); (5) register the standard conversion; report the improved one for information only.

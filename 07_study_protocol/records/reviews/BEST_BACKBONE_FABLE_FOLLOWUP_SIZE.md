# BEST_BACKBONE — Fable 5.1 FOLLOW-UP ruling on the owner directive (model size in the search; 6 EFE; no legacy anchoring)
Received 2026-09-27; saved verbatim by the Operations Director. Owner directive: DECISION_LOG 60. Supersedes the P2 parts
of `BEST_BACKBONE_FABLE_POSITION.md`; the rest of the amended ruling (P3, P4, P5, baselines, claims) stands.

---
Fable FOLLOW-UP RULING on the owner directive (model size in the search; 6 EFE; no legacy anchoring) — APPROVE_WITH_AMENDMENTS. The rest of the amended ruling (P3, P4, P5, baselines, claims) stands.

Verified: CLIENT_CONTRACT_V1 has NO device-memory or export limit (CP5 "expected dense payload bytes" is PENDING); SYS-02: visit bytes GRU 168 MB / SASRec 665 MB, cost driven by the per-visit full-K head + AdamW over all params; THETA0_REGISTRY keyed schema→build→family→seed (GRU 20,965,706; SASRec 83,097,548 numel); build.py takes only `family` (SASREC_ENTRY fixed, GRU dims hard-coded in gru_v2.py L234–252, item embedding 128 in _EMBED_DIM).

(a) Design — AMEND. A 12-point Sobol/LHS over mixed factors gives each size level a random 2–6 points at random LRs, so the size effect (the owner's question) is confounded with the LR draw. Rule: full factorial on the discrete factors, Latin hypercube on log-LR only (Bergstra & Bengio 2012, JMLR, for the continuous factor):
• GRU: d {128, 256} × schedule {S0, S1} × dropout {0.1, 0.2, 0.3} = 12 cells, one trial each.
• SASRec: size {(256,2), (256,3), (384,2)} × schedule {S0, S1} = 6 cells × 2 trials; dropout rotated so each level appears 4× and every (size, dropout) pair ≥ 1×.
• LR: one 12-stratum LHS on log10 LR ∈ [3e-4, 3e-3]; permutation from the registered seed; identical procedure and seed for both families.
• Every trial: seed 2026, full 6 EFE on CALIB_TRAIN, STRICT, S*, B 256; evals at 2/4/6 (endpoint read at 6.0); checkpoints saved at 5.0/5.5/6.0 for CKPT_AVG.
• Stage 2: the best trial of EACH size level re-run at seed 2027 (GRU 2, SASRec 3), so the size decision is 2-seed. Count 12 + per-size re-runs for both; disclosed.
Cost (INFERENCE; d256/d384 STRICT rates unmeasured, assumed 12 / 6.5 upd/s): GRU ≈ 4.5, SASRec ≈ 8.5 GPU-h; ≈ 13 GPU-h, ≈ 9 h wall. Precondition: SYS-04c synthetic STRICT bench of the three new variants (2 reps, ≈ 0.2 GPU-h) before P2.

(b) Levels — as proposed, with exact definitions:
• GRU_D256: item_tokens embedding 256, input_proj→256, LayerNorm(256), nn.GRU(256, 256) (hidden unchanged — the scoring dim is the bottleneck), readout_proj→256, tied head. ≈ 41.5M params, ≈ 166 MB, visit ≈ 333 MB (I).
• SASREC_D256_B3: one more CausalBlock (≈ +0.8M). SASREC_D384_B2: 4 heads (head dim 96), ff ×4, untied, C7 recentering; ≈ 126M params, ≈ 503 MB, visit ≈ 1.0 GB, AdamW client state ≈ 2.0 GB (I).
• Admissibility (proxy; no cap is registered): visit bytes ≤ 2× the registered SASRec 665 MB and per-client AdamW state ≤ 4 GiB → all five variants admissible. The winner's payload and a SYS-02b FL forecast are registered before ARCHITECTURE_LOCK; CP5 is filed then.
• Dropout level p: GRU input p, readout p/2; SASRec single p. Schedules S0/S1 as ruled. Fixed: AdamW (0.9, 0.999), wd 1e-5, clip 1.0, B 256, label smoothing 0, 6 EFE.

(c) Selection — ACCEPT with the stage-2 change: tuned = higher 2-seed mean at 6.0 among the per-size re-runs; tie |Δ| ≤ 0.0005 → smaller model, then lower LR. Exclusions: non-finite, or a 6.0 value < 0.5× the family's best. No KEEP_REGISTERED and no adoption margin (owner). CKPT_AVG stays polish only; carried iff it beats the endpoint by > 0.0005 on the 2-seed mean.

(d) TRAIN-02 scope: a variant registry `contracts/model_variants/MODEL_VARIANT_REGISTRY.json` (GRU_D128, GRU_D256, SASREC_D256_B2, SASREC_D256_B3, SASREC_D384_B2; the first and third are the current builds); v4models: `d` argument for GRU, SASRecConfig from the variant entry, build(family, variant, …); FL adapters variant-agnostic (topology derived from the module); θ0 per variant × seeds 2026/2027 built on the server (CPU build, C7 recentering for SASRec) into a NEW `THETA0_VARIANTS.json` — `derived_schemas/REGISTRY.json` and `THETA0_REGISTRY.json` stay byte-identical; inventories per variant; RunConfig and the comparison key gain `model_variant` and `theta0_init_sha256` (R4 addition authorized here); HP registry, TUNING role, knots/dropout as registered values (as ruled). Tests: per-variant adapter equivalence, θ0 determinism (two builds), resume-exact, one short STRICT bitwise run per variant. Opus gate review + full server suite suffices — no Fable call — iff the diff stays within v4models / v4train / fl adapters / tests, with v4eval, v4data, DATA-06 and LOCK_V2 untouched (T3 stays BIND). PREG1_AMENDMENT_02 registers the variants, the LHS seed/procedure, the levels and the rules before the first P2 launch.

(e) Triggers (added to the amended list): 1. SYS-04c: a variant below 0.5× the assumed rate or above 8 GiB peak on the A16. 2. A size level with no valid 6.0 run. 3. Tuned variant ≠ the registered build → SYS-02b; if the downstream core exceeds 1.5× the current forecast, Ops brings reduction options (methods, rounds, seeds) to me — never a smaller backbone. 4. Any byte change to derived_schemas/REGISTRY.json or THETA0_REGISTRY.json, or a diff outside (d). 5. Existing triggers stand (MAIN_VALIDATION read before P3; lock edits; T1/HOLD; > 125% budget; non-bitwise STRICT; baseline routing through T3).

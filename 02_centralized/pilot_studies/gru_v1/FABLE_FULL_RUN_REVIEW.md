# FABLE_FULL_RUN_REVIEW.md

One bounded Fable 5.1 review call (planning/review only, zero tool uses), made 2026-09-22 ~22:30 UTC before any full-run code, benchmark or validation result existed. Input: `02_FABLE_ONE_CALL_BRIEF.md` plus measured facts (24.15 GiB free, pilot cache 6.4 GiB, pilot step-time breakdown) and Opus's concrete proposal (in-RAM compact per-event store, full-VS selection, concurrent F0/F4). Response recorded verbatim below. Opus's disposition of each point is in `FULL_RUN_PROTOCOL.md` §0.

---

VERDICT:
APPROVE_WITH_FIXES

EPOCH_POLICY:
Keep min 2 / max 6 / patience 2 / +0.001, with clarifications: (a) best checkpoint = highest raw full-VS macro MRR@20; the +0.001 threshold governs stopping only; (b) patience = consecutive epoch-end evals with delta-vs-best < +0.001; (c) min_epochs=2 is redundant (patience 2 cannot stop before epoch 3) but harmless; (d) express any budget cap in epochs, not wall-clock, so contention cannot bias arms; (e) add a cheap 10k-cohort diagnostic every ~5M decisions, logged only, never used for selection or stopping. Expect stop at epoch 4-5: ~3 h/epoch plus ~10-20 min full-VS eval at B_eval=1024 (logits 1024x158k fp32 = 0.65 GB, fine).

LR_POLICY:
One simple pre-declared plateau step, identical rule for all arms: 100-update linear warmup -> 1e-3 constant; at the first epoch-end eval with delta-vs-best < +0.001, set lr=1e-4 for all param groups from the next epoch (continue from latest weights, no restore, never raised again); that eval still counts toward patience. Record the decay epoch per arm. Rationale: the pilot-validated 1e-3 is untouched in the steep regime; the step only recovers the final polish that a constant-LR Adam noise floor forgoes, and unlike cosine it needs no knowledge of total epochs. Fused AdamW is acceptable (same update rule, ULP-level differences) but must be fixed before launch, never switched mid-run, identical across arms.

H512_CHALLENGER:
SKIP
Reason: The GRU is ~0.3M of 20.8M params; 20.3M are item embeddings. With mean context 4.9 and a tied 128-d scoring space, capacity in the recurrent state is not the bottleneck; expected effect is below the +0.001 threshold and it delays the lock by ~1 day. If any challenger is ever run, pre-register F0-d256 (embedding/scoring dim) instead, exploratory only.

DATA_PATH:
Semantics are preserved iff the three gates pass, with these additions: (1) targeted edge-case parity in gate (ii): positions 1/49/50/51, first/last event of a session, sessions crossing the TRAIN/VS boundary, contexts containing loss_mask=0 events (same-product repeats), users whose first VS decision context starts in TRAIN; (2) the 12 user-context values + 6 masks must be stored per event (as-of-decision), not per user; (3) startup asserts: exactly 40,180,608 permuted decision_ids, all TRAIN; 8,826,488 VS decisions / 790,459 eval users; hash and log each epoch permutation; (4) state gate (iii) tolerance explicitly (e.g., |delta macro MRR@20| <= 1e-6 across devices, since GPU-vs-pilot kernels can flip rare fp32 ties). Full-VS selection is correct and necessary: the 10k cohort has SE ~0.0025 on macro MRR@20, larger than the +0.001 rule it would have to drive; full VS has SE ~3e-4. Keep the cohort as a continuity diagnostic only.

TWO_GPU:
Scientifically fair. Numerics depend only on permutation, seed and kernels, not on timing; no time-based triggers exist. Conditions: pin via CUDA_VISIBLE_DEVICES; cap torch/numpy threads (~3 per process) to avoid oversubscribing 8 cores; RAM budget = 2 x ~10 GB stores + eval buffers + Parquet decode transients, log peak RSS against 64 GiB. Contention affects wall-clock only; serialise only if the benchmark shows a slowdown that breaks the budget. Optional: build the compact store once as a memory-mapped file (~10 GB of 24 GB free) shared by both trainers and the evaluator; this also makes gates (i)-(iii) reproducible from the same bytes.

BLOCKERS:
- None requires redesign; all must be done before launch:
- Pre-register the full-run decision rule now: primary = best GRU arm vs full-scale TRANS on TEST macro MRR@20 (paired user bootstrap, CI must exclude 0, state minimum effect); secondary = F4 vs F0; the pilot NO_GO remains a separate recorded result. Without this the run is exploratory.
- VS-leakage assertion: unit test that no VS decision_id ever enters a training batch and optimizer step count x 256 matches the TRAIN decision count per epoch.
- Re-run the correctness suite (synthetic memorisation, bit-identical resume, no non-finite) under fused AdamW + per-100-update logging before launch.
- Extend parity gate (ii) with the edge cases above; write down gate (iii) tolerance.

TOP_5_RATIONALE:
1. The pilot was embedding-starved: 2M exposures over 158k items is ~13 per item on average, far fewer in the tail, while count-based TRANS saturates fast at small N. A +20.8% pass1->pass2 gain with no overfitting says the curve is steep; 40x data is a legitimate new hypothesis, not a re-litigation of the NO_GO.
2. Power: 790k VS users give SE ~3e-4, so +0.001 is a real threshold and the F4-F0 CI shrinks from about +/-0.0014 to roughly +/-0.0002, making that comparison decisive either way.
3. Budget fits: ~3 h/epoch/arm, <=6 epochs + evals ~22 h per arm concurrently; fused AdamW and aggregated logging recover ~30% of the 3,258 -> 4,401 dec/s gap.
4. h512 targets the wrong bottleneck (recurrent capacity vs embedding/scoring dimension and data), costs a day, and cannot change the lock decision.
5. Determinism (fixed per-epoch permutation, fixed seed, no timing-dependent logic) makes concurrent training fair; the only threats are pre-registration gaps and VS leakage, both cheap to close.

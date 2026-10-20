# FEATURE_EXPERIMENT_PROTOCOL — SASREC_FEATURE_OPT_V1 (frozen 2026-09-25T11:05Z, before any study score)

The binding decision is `review/AUTO_DECISION_SCOPE.md` (Fable `claude-fable-5-1`, owner-delegated):
STAGED_WITH_EVIDENCE_GATES. The machine-readable registry is `config/candidate_registry.json`
(sha256 recorded in `review/FEATURE_CODE_REVIEW_PASS.json`). Every selection uses VALIDATION_SELECT
only, and that split has been used extensively before. This is therefore adaptive development, not a
fresh holdout. VC/TEST are never read.

## 0. Isolation and protection
- **Code isolation.** The study imports only a byte-identical, hashed snapshot of the campaign code
  (`code/snap_sasrec`, 24 files, identical on the server and locally). `fo_env.assert_isolated()`
  proves the load paths.
- **Protection snapshots.**
  - T0 (study start), T1 (campaign done) and T2 (study done) hash 185 campaign code, config and lock
    files, plus the dataset metadata and tree.
  - `protect_diff.py` reports any change.
- **Scheduling.** GPU and heavy work start only after the campaign's `FINAL_STAGE_DONE`, with no
  campaign process alive and no compute process of anyone on the GPU used. Free disk must be at least
  8 GiB plus the stage's peak. Nothing is ever deleted to make room.

## 1. Measured audit (done; a bounded sample of 3/64 buckets)
M1–M5 are in `audit/MEASURED_FINDINGS.md`. Population-exact numbers come from `stream_fit.py` and
the full profile (all 64 buckets) after the campaign.

## 2. Baseline and noise
- **Baseline:** `config/BASELINE_RULE.md` (VS-only, pinned at FINAL_STAGE_DONE), and
  `config/BASELINE_PIN.json`.
- **σ̂:** max |A1 s2026 − A1 s2027| over matched epochs (0.002 if unavailable), written to
  `config/NOISE_REFERENCE.json` before any candidate training.

## 3. Probes (reliance diagnostics, not retraining evidence)
- **Set:** baseline plus P1–P8 on the 10k-user VS cohort (112,365 decisions).
- **Ranking and scoring:** exhaustive K; strict FP32; the audited evaluator.
- **Statistics:** paired user bootstrap (B=1000, seed 20260926); weekday/weekend slices.
- **Encoding check:** before any score, every injected "unavailable" encoding is compared with real
  rows in that state. Any mismatch aborts.
- **Probe semantics:** see `feature_view.apply_probe`.
- **Labels:** P1 is an OOD anchor, P8 is a single-point counterfactual, and the others are coherent
  absence.
- **Gate 2(a), removal candidate C2:** Δ ≥ +0.001, CI lower bound > 0, and both slices positive.
  - P1 is excluded.
  - A negative Δ never creates a candidate.
  - At most one candidate.

## 4. Hygiene deliverable (no training)
- **Lean view.** Drop the 9 exact duplicate channels and the 6 dead flags. The event branch goes
  from 34 to 22 inputs, the user branch from 18 to 16.
- **M1 repair.** Valid-only TRAIN refit, keeping the recorded clip bounds so the fold stays exact,
  and re-zero by mask.
- **Folded weights.** `fold_lean_state_dict` makes the result an exact re-parametrisation of the
  incumbent.
- **Checks:** parity on the cohort (max |Δlogit|, rank identity, metric Δ), and an explicit parameter
  mapping.
- **Status:** this is hygiene, never presented as an accuracy gain.

## 5. Screen (EXPLORATORY warm-start adaptation, ≤ 6 GPU-h)
- **Arms:** CONTROL, C1_D3 and, only if gate 2(a) names one, C2.
- **Common setup for every arm:**
  - the pinned weights, loaded strictly;
  - a fresh AdamW;
  - the same 5,000,000 unique TRAIN loss decisions, in the same order (manifest sha `15cbb263…`);
  - 19,532 updates (the last one of 64);
  - LR: linear warmup to 1e-4 over 200 updates, then cosine to 1e-5;
  - the campaign recipe and precision policy (TF32 training, FP32 eval), seed 2026.
- **C1_D3:** five causal repeat channels. They use known tokens only, no fit, no cache and no raw
  ids, and are zero-initialised, so C1 equals CONTROL at initialisation.
- **Gate 2(b), finalist:**
  - Δ(candidate − control) ≥ max(0.002, 2σ̂);
  - bootstrap CI lower bound > 0;
  - micro R@20 no more than 0.001 below control;
  - at most one finalist.

## 6. Pair (registered; implemented only if a finalist exists)
Per registry `pair`. CONTROL_REUSE is preferred when it is valid. Otherwise the pair runs
sequentially, from scratch, with the frozen 5-epoch schedule. Acceptance on full VS uses the same
thresholds, and the candidate must also be at least as good as the pinned incumbent. A correction-only
Fable review of the pair code is required before that training.

## 7. Outputs
- `FEATURE_INVENTORY.csv`
- `FEATURE_DEPENDENCIES.json`
- `TRANSFORM_LINEAGE.json`
- `TRAIN_PROFILE*`
- `GROUP_PROBE_RESULTS.json`
- `LEAN_PARITY.json`
- `FEATURE_SELECTION_RESULTS.csv`
- `SELECTED_FEATURES.json`
- `TRANSFORM_STATE.json`
- `FINAL_FEATURE_OPT_RESULTS.json`
- `FINAL_FEATURE_OPT_REPORT_AR.md`
- `review/FABLE_FEATURE_PLAN.md`, which is `review/AUTO_DECISION_SCOPE.md`
- `review/FABLE_CODE_REVIEW.md`

The compute cost is recorded separately from the campaign's ETA.

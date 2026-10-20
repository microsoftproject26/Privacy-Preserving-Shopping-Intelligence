# PREG1 amendment 06L (part A) — S2/S3 grid completion, DP sensitivity trigger, PF/FP interpretations (registration only)

- **Status:** REGISTERED 2026-09-29 (~18:00Z) by the Operations Director (Claude Opus 5.5, session eda-7b) from the multi-expert
  review `company/reviews/MULTI_EXPERT_REVIEW_20260929/00_SYNTHESIS.md` §4 Gate A, BEFORE any S2 value is read (the S2 screens run
  since 14:44Z; none has finished; the S2 selection file does not exist) and before any S3, S4, PF, FP, DP or MAIN FL value exists.
  Builds on 06F, 06G (+ v0 = τ² clarification), 06H, 06I §3, 06J (incl. §4(ii) ≥ 0.002 incumbent rule and the raw-value reading).

## A1. FedAvgM grid completion (adds to 06J §1)
Reason (review [6-F1]): FedAvgM with β = 0.9 multiplies the effective server step by ≈ 1/(1−β) = 10, so FedAvgM at m = 8 is an
effective step ≈ 10× past the FedAvg optimum; the registered S2 set tests FedAvgM only there. Added to S2: **FedAvgM (β 0.9, η 1)
at m ∈ {1, 2}** (FA, CALIB, seed 2026, full 6.0 EFE, same protocol). Edge rule: if FedAvgM at m = 1 is the best FedAvgM cell, one
run at m = 0.5 is added. The S2 selection (06J §1 rule incl. §4(ii)) is computed over the S1b winner (incumbent), the five
launched S2 runs, these two runs and any registered edge runs; it is written only after all of them finish.

## A2. S3 combination cells (amends 06J §2; re-instates the combination run 06J withdrew)
Whatever S2 selects, S3 also screens **client SGD-momentum 0.9 × FedAdam (η_s 0.01, τ 1e-3, v0 = τ²) × FL_CONST shape × client
lr ∈ {0.1, 1.0}** (2 CALIB runs), in addition to the 06J §2 cells built on the S2 winner. The S3 winner is the best over all S3 cells
and the S2 winner (incumbent) under 06J's rule; ties → incumbent, then AdamW before SGD, then smaller SGD lr, then FedAvg-family
before FedAdam.

## A3. DP server-optimiser sensitivity (amends 06H)
The 06H DP_OPT_SENS trigger fires for ANY non-FedAvg selected server optimiser (FedAvgM included, not only FedAdam). Disclosed:
with momentum β the DP noise injected per round is carried forward with weight ≈ 1/(1−β); the privacy accounting is unchanged
(post-processing) but the utility cost of noise can grow — reported with the note.

## A4. Clipping and the LR schedule (minimum registered now)
FA_1024 / DP runs report the per-round fraction of clipped client deltas (logged when z = 0; for z > 0 only the aggregate count, no
per-client norms). The "clip follows the LR" variant S_r = S·f(r) (data-independent, accountant unchanged) is NICE unless S3 keeps
the central LR shape, in which case it becomes SHOULD-high and needs its own registration before any DP launch.

## A5. PF interpretation (K5)
PF's personal learning rate stays as registered (SMALL peak × round factor; not scaled by the selected m*). Registered
interpretation before any PF read: with a small personal component and a much larger shared step, **PF ≈ FA by construction**
is a possible outcome; K5 is reported with this caveat, and the PF − FA paired Δ is DESCRIPTIVE.

## A6. Sparse-aware aggregation — NOT adopted now (disclosed as a limitation; possible NICE rung after S4).

## A7. FedProx interpretation (K4)
The FedProx proximal term acts on top of a per-visit Adam(W) or SGD optimiser with few steps; μ = 0.01 is untuned. Registered
before any FP read: K4 measures "FedProx μ 0.01 under the selected client optimiser"; no general claim about FedProx is made.

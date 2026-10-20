# PREG1 amendment 06L (part C) — analysis, framing and arm-set decisions from the multi-expert review (registration only)

- **Status:** REGISTERED 2026-09-29 by the Operations Director (Claude Opus 5.5, session eda-7b) on the owner's instruction "review the
  notes, do what is right after analysing them, without contradicting any of our instructions". Source:
  `company/reviews/MULTI_EXPERT_REVIEW_20260929/00_SYNTHESIS.md` §4 Gates C–E. Registered BEFORE the ANA freeze and before any
  downstream MAIN / panel / device / ONNX / C_FULL / PRE value is read (the only reads so far are registered CALIB screen values).
- **Owner decisions applied:** C_SMALL_FULL (review [5-R4]) is NOT adopted (owner, 2026-09-29). A December-2019 holdout (D6) needs a
  new data version and owner approval → not adopted. Anything added after the first VALIDATION read is POST_VALIDATION_EXPLORATORY.

## C1. Privacy matrix (replaces the one-column ladder of 06B §3)
Every arm is described by four columns: data seen at TRAINING by the server; data seen at INFERENCE by the server; formal guarantee;
against whom. Placements corrected: FT_C on MAIN = "L1 base (server-trained on raw events) + local fine-tune"; FT_PRE = L5 (base
trained only on opted-in users); PRE-alone = L1 on opted-in data; RULE / POP = "device + published aggregate"; LAST added to L6;
a column "secure aggregation: not simulated" for every FL arm. Non-DP FL updates reveal the user's viewed items to the server
(input-table rows and output rows move); S8 removes the input-table leak only partly (brand/category deltas may still leak).

## C2. Threat model and DP statement (text, binding for the thesis)
Adversary = an honest-but-curious browser vendor (the server); the retailer's own logs are out of scope. DP guarantee = central DP
with a trusted aggregator, client-level add/remove adjacency, (ε, δ = 5e-6), conditional on the public bootstrap block (catalogue,
transform statistics, θ0 / θ_pre from opted-in users); the FL hyper-parameter tuning on CALIB is not accounted.

## C3. Anchor every answer on LAST (the no-training device heuristic)
Baseline runs also write per-row rank files. Reported for every arm: paired Δ(arm − LAST) with CI; headroom
R* = (arm − LAST)/(C_ref − LAST); flag BELOW_DEVICE_HEURISTIC when the CI of Δ lies below 0. New confirmatory contrast **K11b =
FA − LAST** (secondary). The device-only envelope {LO, LAST, RULE} is reported as one row group.

## C4. Contrast registry v3
`company/gates/ANA_CONTRASTS_v3.json` = K1–K21 + K11b + (C7) + (C8), with the algebraic dependencies listed (K3 = K2/K1;
K17–K19 = the matching K × K15) and without the word "co-headline" for K17. MASTER_PLAN §9's contrast-file path points to v3.

## C5. Frozen arm set before the preflight
The arm set, seeds, recipes and checkpoint rules for the VALIDATION preflight are exactly those registered up to this amendment
plus the ladder's selected FL configuration; nothing else is confirmatory.

## C6. Wording of the as-registered FL row
"Matched-exposure retention of the registered FL recipe" (not "lower bound"); the central-LR-shape waste is "≈ 50 % of the exposure
budget at ≤ 0.1× peak", with the peak marks cited.

## C7. Deployment branch promoted to secondary confirmatory (owner's realism priority)
R(FA_WARM / C_FULL) and R(PRE / C_FULL) (2 FA_WARM seeds vs C_FULL s2026; PRE single seed), K20, K21, **K9c = FT_PRE − PRE on MAIN**
(new arm FT_PRE: θ_pre + the registered FT_C procedure with SELECTION_FT_C, seeds 2026/2027), **K9b = FT_C − C_SMALL on
UNSEEN_HOLDOUT**. The Edge answer and the L5 row lead with these.

## C8. S8 and bytes
Paired contrast S8 − S7 reported with the bytes ratio; one bytes axis for every privacy level (raw events → model download → FL
traffic → S8). S8 disclosure corrected: 265,162 uploaded parameters (to be re-measured), brand/category deltas may still leak,
off-support rows move ≈ 1e-9, not exactly 0. S8 runs with 2 seeds unconditionally (supersedes 06K §3's conditional second seed).

## C9. K16 quantiser cell
K16 = dynamic INT8 quantisation (QInt8 weights / QUInt8 activations as exported), per-tensor, reduce_range False, onnxruntime 1.24.1,
single thread; one DESCRIPTIVE cell per_channel + reduce_range True; peak RSS measured in a process without torch loaded.

## C10. Cold start
The minimal DATA-07 set (T41 on both panels; Study A for C_BIG / C_SMALL / FA on COLD_H; Study B for FT_C on COLD_H) is SHOULD-high,
and MUST for any "events to parity" sentence. Added: matched-h parity M(h)/C_BIG(h) and a persistence rule (parity must hold at the
next registered h too, with the CI lower bound ≥ X) beside the registered first-hit rule.

## C11. Deployed-size family comparison (descriptive, evaluation only)
GRU_D128 (the contest's 02B_GRU best@6.0, 20.96M parameters, already trained) vs C_SMALL (20.71M), PRACTICAL_BEST and
CONTROLLED_BUDGET, on the A40 block; never a downstream family (ARCHITECTURE_LOCK unchanged).

## C12. Headline template and claim register
A number-free abstract/headline template and a CLAIM register (allowed / forbidden sentence per contrast) are frozen together with
the ANA script.

## C13–C15. Statistics additions (DESCRIPTIVE unless stated)
Per-seed ratios, seed spread and a SEED_SENSITIVE flag beside every K; a Holm band beside the unadjusted CIs; a CALIB-disjoint
diagnostic (MAIN \ CALIB) for every CALIB-tuned arm; harmed-client share for every K; a K14 equivalence margin of ±1 % relative
registered now (before any FA_Q8 read); repeat vs novel target stratum.

## C16–C17. Arm additions (only these)
Adopted: an evaluation-only "no-FT" reference of C_SMALL and FA on UNSEEN_VAL (descriptive); DP8_FROZEN (S8 + DP ε = 8, seed 2026,
its z = 0 twin = S8) at SHOULD-high. Not adopted: FT_PU (parameter-efficient p_u fit), per-row Q8 / bit sweeps, DP8_FULL at the
1.35M scale (stays NICE; no launch path written), KD, the other NICE items of the review.

## D / E. Before the TEST look and in the write-up (recorded now; executed later)
D1 restore the registered TEST contest singles (02B_GRU, 02B_SASREC on A40) in the single TEST look; D2 disclose the TEST
partition fence + counts receipt; D4 Black-Friday sub-period (descriptive). E1 Queen's "Statement of Originality and Disclosure of
Use of AI" (supervisor approval) is required front matter; the public GitHub documents are corrected at the next daily push
(viewed-item leak, previously inspected TEST, "full pool" = ≈ 93 % of the 5–100 pool); E4–E9 as listed in the review.

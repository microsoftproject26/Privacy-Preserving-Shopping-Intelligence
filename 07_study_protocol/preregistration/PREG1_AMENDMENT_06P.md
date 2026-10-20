# PREG1 amendment 06P — contrasts for the 06N pretrained-line hierarchy (registration only)

- **Status:** REGISTERED 2026-10-01 by the Operations Director with Fable concurrence. The owner delegated this decision to Ops
  and Fable jointly (DL 316–317). Source: reviews/FABLE_ENDGAME_REVIEW_20261001.md §3 D1 + §5.
- **No MAIN or TEST value has been read.** This is registered before ANA-06's freeze. It amends 06N N1 and ANA_CONTRASTS_v3 (as
  ANA_CONTRASTS_v4).
- **Tier change from the review draft:** the P4 warm-method contrasts are SECONDARY, per the literal 06N N1 text (the review's
  tertiary split was withdrawn).
- **The v4 contrast file:** company/gates/ANA_CONTRASTS_v4.json carries P1–P6 verbatim. It is built by ANA-06; its sha256 is
  registered by a later DL row before the ANA freeze.

**P1. Resolution of FA_WARM\*.** FA_WARM\* = FA_WARM_FROZEN (S8), by 06N N2 on the registered CALIB selections (DL 292). In K20, K20b
and K21 the arm `FA_WARM` is read as `FA_WARM_FROZEN`: K20 = R(FA_WARM_FROZEN / C_BIG), K20b = R(FA_WARM_FROZEN / C_FULL) (2 seeds
vs C_FULL s2026), K21 = FA_WARM_FROZEN − PRE (paired Δ, 2 seeds, PRACTICAL_BEST). **Primary = the pair {K20b, K21}**, read unadjusted
and reported together; a claim about the deployment branch requires both. K2 = R(FA / C_BIG) becomes SECONDARY ("random-start
ablation"). K20c = R(PRE / C_FULL) unchanged (secondary, seed 2026).
**P2. The S7 variant.** FA_WARM (full-model warm start) keeps 2 MAIN seeds. Its rows are reported as the registered descriptive
comparison S8 − S7 (06L_C C8, with the bytes ratio) and as DESCRIPTIVE R rows (R(FA_WARM / C_FULL), FA_WARM − PRE) beside the
primary; no confirmatory contrast is defined on FA_WARM.
**P3. K_INIT = FA_WARM_FROZEN − FA** (paired Δ, seeds 2026/2027, PRACTICAL_BEST, CONTROLLED_BUDGET beside; "value of the pretrained
start"), SECONDARY, in the Holm family.
**P4. Warm-line method contrasts** (all on MAIN, the warm reference = C_FULL per DL 273): K4w = R(FP_WARM_FROZEN / C_FULL), K5w =
R(PF_WARM_FROZEN / C_FULL) (2 seeds, PRACTICAL_BEST, SECONDARY; PF_WARM_FROZEN − FA_WARM_FROZEN DESCRIPTIVE with the 06L A5 caveat);
K12w = R(DP8_FROZEN / FA_1024_WARM_FROZEN) (2 seeds, CONTROLLED_BUDGET / CONTROLLED_BUDGET, SECONDARY; PRACTICAL_BEST ratio
DESCRIPTIVE with the 06D §3 disclosure); K13w = R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN) (seed 2026, CONTROLLED_BUDGET, SECONDARY);
K14w = R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN) (2 seeds, TOST ±1 % relative as 06L_C C15, SECONDARY); R(FA_1024_WARM_FROZEN /
FA_WARM_FROZEN) DESCRIPTIVE (clipping + Poisson-cohort cost). The cold K4, K5, K12, K13, K14 keep their text and tiers on the N33
re-runs.
**P5. FT_FA_WARM_FROZEN − FA_WARM_FROZEN** (paired Δ, 2 seeds, PRACTICAL_BEST, base = the same-seed best.pt, harmed-client share
beside) is SECONDARY (06N N1 supersedes 06M_B B4's DESCRIPTIVE); FT_FA_WARM − FA_WARM stays DESCRIPTIVE.
**P6. Multiplicity (corrected, Fable §5 concurrence).** The primary pair {K20b, K21} is read unadjusted, outside Holm. Holm family
(secondary, 19): K1, K2, K3, K8, K9b, K9c, K11b, K12, K17, K20, K20c, K_INIT, P5 (FT_FA_WARM_FROZEN − FA_WARM_FROZEN), K4w, K5w,
K12w, K13w, K14w. Tertiary (unadjusted, Holm band beside): K4, K5, K6, K7, K9, K10, K11, K13, K14, K15, K16, K18, K19. The 06L_C
C13 convention is unchanged (unadjusted 95 % paired bootstrap CIs, Holm band beside; K14 / K14w TOST ±1 % on the unadjusted 90 %
CI). DESCRIPTIVE, outside every family: S8 − S7; the S7 R rows; PF_WARM_FROZEN − FA_WARM_FROZEN; R(FA_1024_WARM_FROZEN /
FA_WARM_FROZEN); the PRACTICAL_BEST ratios of K12 / K13 / K12w / K13w. registered_seeds = the v3 map + K13w [2026]; every other new
id [2026, 2027]; no seed cuts.
**P7.** `ANA_CONTRASTS_v4.json` carries P1–P6 verbatim, the algebraic dependencies (K20b_s = K20_s × C_BIG_s / C_FULL; K20b − K20c
= 100·K21 / C_FULL), `post_validation_rule` unchanged; its sha256 is registered by the DL row and copied into ANA_SCRIPT_FREEZE and
ALL_METHODS_LOCK. The `--claims-only` output is regenerated under v4 and frozen with it.

---
**ERRATUM 1 (2026-10-01, DL 324; before any freeze; no value read).** In P6, "Holm family (secondary, 19)" is a counting error.
The named list is binding and has **18** members: K1, K2, K3, K8, K9b, K9c, K11b, K12, K17, K20, K20c, K_INIT, P5, K4w, K5w,
K12w, K13w, K14w. Fable's original list (FABLE_ENDGAME_REVIEW §5) also enumerates these 18. Nothing else changes.

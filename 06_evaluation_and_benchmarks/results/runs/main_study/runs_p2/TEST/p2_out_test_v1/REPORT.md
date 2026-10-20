# Phase-2 analysis (PREG2) — TEST

script tools/p2_report.py 934cc8a0e93f3edd44e5c950f89996149a1f2b26dd7737f9c78b383b7dff215e; retention_report 9a3b2c2a918e…; bootstrap seed 20261004 (1000 resamples); MAIN users 26254; W 0654b73fd097d99d71591aeaa609b367d41305ba92431d3e199792fe7386413d

**CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP)** claims (Q1-Q8) are printed with D-a: "Phase-2 hypotheses were generated after the Phase-1 VALIDATION read; arms were selected on CALIB; everything was locked before any TEST value existed."

## Contrasts (registered rule; E2E client-macro MRR@20)

| id | contrast | tier | label | rule | seeds | seeds status | point | 95% CI | vs ref | Holm | harmed share | equivalence (90% CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | P2_WSTAR − PRE | primary | CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) | PRACTICAL_BEST | 2026,2027 | COMPLETE | -0.000081 | [-0.000179, 0.000024] | includes | NOT_IN_HOLM_FAMILY | 0.0715 | EQUIVALENT [-0.000164, 0.000007] |
| Q2 | P2_C1_FT_CENTRAL_FULL − PRE | secondary | CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) | PRACTICAL_BEST | 2026,2027 | COMPLETE | -0.000212 | [-0.000628, 0.000208] | includes | not rejected | 0.2876 | - |
| Q3 | P2_WSTAR − P2_C1_FT_CENTRAL_FROZEN | secondary | CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) | PRACTICAL_BEST | 2026,2027 | COMPLETE | 0.000166 | [0.000027, 0.000300] | above | not rejected | 0.1246 | - |
| Q4 | P2_WSTAR − FA_WARM_FROZEN | secondary | CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) | PRACTICAL_BEST | 2026,2027 | COMPLETE | 0.000427 | [0.000007, 0.000893] | above | not rejected | 0.3275 | - |
| Q5 | FT_P2_WSTAR − FT_PRE | secondary | CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) | PRACTICAL_BEST | 2026,2027 | COMPLETE | -0.000006 | [-0.000093, 0.000075] | includes | not rejected | 0.0975 | - |
| Q6 | R(P2_WSTAR_DP8 / P2_WSTAR_FA1024) | secondary | CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) | CONTROLLED_BUDGET | 2026,2027 | COMPLETE | 100.00 | [100.00, 100.00] | includes | not rejected | 0.0011 | - |
| Q7 | P2_WSTAR_DP8 − PRE | secondary | CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) | CONTROLLED_BUDGET | 2026,2027 | COMPLETE | -0.000037 | [-0.000097, 0.000020] | includes | not rejected | 0.0289 | - |
| Q8 | P2_C2_CFULL_SMALL − PRE | secondary | CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) | PRACTICAL_BEST | 2026 | COMPLETE | 0.000541 | [-0.000457, 0.001591] | includes | not rejected | 0.3594 | - |
| Q9 | R(P2_WSTAR / C_FULL) | tertiary | TERTIARY_P2 (unadjusted; DESCRIPTIVE Holm band) | PRACTICAL_BEST | 2026,2027 | COMPLETE | 88.72 | [87.95, 89.43] | below | COMPUTED | 0.4683 | - |
| Q11 | R(P2_WSTAR_DP1 / P2_WSTAR_FA1024) | tertiary | TERTIARY_P2 (unadjusted; DESCRIPTIVE Holm band) | CONTROLLED_BUDGET | 2026 | COMPLETE | 99.99 | [99.98, 100.00] | includes | COMPUTED | 0.0022 | - |
| Q12a | R(P2_WSTAR_DPFTRL8 / P2_WSTAR) | tertiary | TERTIARY_P2 (unadjusted; DESCRIPTIVE Holm band) | CONTROLLED_BUDGET | 2026 | NOT_RUN | DECLARED_NOT_RUN | - | DECLARED_NOT_RUN | NOT_COMPUTED | - | - |
| Q12b | P2_WSTAR_DPFTRL8 − PRE | tertiary | TERTIARY_P2 (unadjusted; DESCRIPTIVE Holm band) | CONTROLLED_BUDGET | 2026 | NOT_RUN | DECLARED_NOT_RUN | - | DECLARED_NOT_RUN | NOT_COMPUTED | - | - |
| Q13 | R(P2_WSTAR_SECAGG / P2_WSTAR) | tertiary | TERTIARY_P2 (unadjusted; DESCRIPTIVE Holm band) | PRACTICAL_BEST | 2026 | NOT_RUN | DECLARED_NOT_RUN | - | DECLARED_NOT_RUN | NOT_COMPUTED | - | - |
| Q14 | P2_WSTAR − PRE | tertiary | TEST2_NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |

Beside every Q (PREG2 §5.6-§5.8): W* fedadam|eta=3e-06|c=0.1; CALIB_GAIN 0.000013; flags ['EDGE_UNRESOLVED', 'NO_MEASURABLE_CALIB_GAIN', 'FUTILITY_SKIP_WB_WC']; D-j: Futility uses CALIB headroom (32,768 users); MAIN has 131,072 users, so FUTILITY_SKIP_WB_WC may under-read the headroom available on MAIN.

## Q1 — the registered reading (PREG2 §9.4)

**EQUIVALENT** (conditions true: ['EQUIVALENT']; overlap False); per-seed points [-7.509241015846158e-05, -8.658585555840204e-05]

PREG2 §9.4 (registered): SUCCESS iff the Q1 95% CI lower bound > 0 and both per-seed points > 0; EQUIVALENT iff the Q1 90% paired user-bootstrap percentile CI lies inside ±0.0005 MRR@20 (TOST, alpha 0.05; the practical-relevance margin, distinct from the CALIB tie band 0.001); FAILURE iff the 95% CI upper bound < 0; INCONCLUSIVE otherwise. Precedence when several hold: the listed order (SUCCESS, EQUIVALENT, FAILURE); every condition that holds is recorded in conditions_true.

D-h: Phase 2 can confirm a gain ≥ ≈ 0.0005 MRR@20 or equivalence within ±0.0005; a true gain between them is likely to read INCONCLUSIVE.

## Interpretation grid (Q2 x Q1)

cell NO_REGISTERED_CELL: none of the four registered rows holds (e.g. Q1 INCONCLUSIVE): no grid reading is claimed

| state | holds | definition |
|---|---|---|
| Q1_EQUIVALENT | True | EQUIVALENT holds for Q1 (90% CI inside ±0.0005) |
| Q1_NONPOSITIVE_OR_EQUIVALENT | True | the §9.4 reading of Q1 is FAILURE or EQUIVALENT |
| Q1_POSITIVE | False | the §9.4 reading of Q1 is SUCCESS |
| Q2_EQUIVALENT | False | Q2's 90% paired user-bootstrap CI lies inside ±0.0005 (the Q1 margin applied to the grid's 'EQUIVALENT to 0'; DESCRIPTIVE TOST) |
| Q2_POSITIVE | False | Q2 (secondary) is Holm-rejected with a positive point (PRACTICAL_BEST) |

DEPLOYMENT-GRADE: **False** (Q1 SUCCESS False, Q5 Holm-positive False, Q7 Holm-positive False) — PREG2 §9.4: DEPLOYMENT-GRADE = SUCCESS + Q5 > 0 (Holm-rejected) + Q7 > 0 (Holm-rejected); not a test

## Q10 headroom capture (derived, DESCRIPTIVE)

UNDEFINED: point UNDEF(Q2_POINT_NOT_POSITIVE) CI UNDEF(Q2_POINT_NOT_POSITIVE); per seed [(2026, 'UNDEF(Q2_POINT_NOT_POSITIVE)'), (2027, 'UNDEF(Q2_POINT_NOT_POSITIVE)')]

## Multiplicity

Holm step-down family ['Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8'] (size 7, alpha 0.05; rejected 0); tertiary DESCRIPTIVE band ['Q9', 'Q11', 'Q12a', 'Q12b', 'Q13']; Q1 outside Holm; TEST2 family ['Q14']: TEST2_NOT_AVAILABLE. A family member without data (declared not run / absent) is NOT_COMPUTED and the family size is unchanged.

## TEST2

TEST2_NOT_AVAILABLE: owner decision O2 = NO (P2_CONTRASTS_v3 Q14 'only if TEST2 exists'; PREG2 §6.2 / §12): no December-2019 data version and no TEST2 register exist, so the Q1 replication family and Q14 are not scored

## Declared not run (P2_METHODS_LOCK)

- P2_WSTAR_DPFTRL8 seed None: declared not run: Part B reduced to fit the schedule and budget (owner decision, DL 467); Part B never delays the sitting (PREG2 §2.1, §9.3 E-S4)
- P2_WSTAR_SECAGG seed None: declared not run: Part B reduced to fit the schedule and budget (owner decision, DL 467); Part B never delays the sitting (PREG2 §2.1, §9.3 E-S4)

## Descriptive sections (E1-E5, devices, D1-ext)

- D1_EXT (D1-ext update-norm summaries of the P2 FL runs (§3.3)): NOT_RUN [DESCRIPTIVE (tertiary)]
- DEVICES (real-device measurements, thresholds DESCRIPTIVE design targets (§8)): REPORTED [DESCRIPTIVE (tertiary)]
- E1_MIA (E1 user-level membership inference on final models (PREG2 §7.1)): REPORTED [DESCRIPTIVE (tertiary)]
- E2_CANARY (E2 canary exposure and one-run audit on FL training (§7.2)): NOT_RUN [DESCRIPTIVE (tertiary)]
- E3_UPDATE_LEAKAGE (E3 leakage from individual FL updates (§7.3)): NOT_RUN [DESCRIPTIVE (tertiary)]
- E4_SECAGG (E4 secure-aggregation simulation: bytes, rounds, unmasking time, dropout success (§7.4)): REPORTED [DESCRIPTIVE (tertiary)]
- E5_DPFTRL_LDP (E5 DP-FTRL accountant / LDP feasibility (§7.5)): NOT_RUN [DESCRIPTIVE (tertiary)]

## Values (DESCRIPTIVE)

| arm | seed | rule | MRR@20 | NDCG@10 | HR@10 |
|---|---|---|---|---|---|
| PRE | 2026 | PRACTICAL_BEST | 0.160129 | 0.195295 | 0.334120 |
| PRE | 2026 | CONTROLLED_BUDGET | 0.160129 | 0.195295 | 0.334120 |
| FA_WARM_FROZEN | 2026 | PRACTICAL_BEST | 0.159797 | 0.194925 | 0.333793 |
| FA_WARM_FROZEN | 2027 | PRACTICAL_BEST | 0.159444 | 0.194896 | 0.334597 |
| FA_WARM_FROZEN | 2026 | CONTROLLED_BUDGET | 0.159272 | 0.194595 | 0.334106 |
| FA_WARM_FROZEN | 2027 | CONTROLLED_BUDGET | 0.159333 | 0.194558 | 0.333799 |
| FT_PRE | 2026 | PRACTICAL_BEST | 0.162496 | 0.198071 | 0.338050 |
| FT_PRE | 2027 | PRACTICAL_BEST | 0.162328 | 0.198049 | 0.338446 |
| C_FULL | 2026 | PRACTICAL_BEST | 0.180388 | 0.217198 | 0.359396 |
| C_FULL | 2026 | CONTROLLED_BUDGET | 0.180388 | 0.217198 | 0.359396 |
| P2_WSTAR | 2026 | PRACTICAL_BEST | 0.160054 | 0.195231 | 0.334084 |
| P2_WSTAR | 2027 | PRACTICAL_BEST | 0.160042 | 0.195245 | 0.334151 |
| P2_WSTAR | 2026 | CONTROLLED_BUDGET | 0.159782 | 0.194911 | 0.333636 |
| P2_WSTAR | 2027 | CONTROLLED_BUDGET | 0.159707 | 0.194856 | 0.333625 |
| P2_WSTAR_FA1024 | 2026 | PRACTICAL_BEST | 0.160131 | 0.195293 | 0.334111 |
| P2_WSTAR_FA1024 | 2027 | PRACTICAL_BEST | 0.160131 | 0.195294 | 0.334112 |
| P2_WSTAR_FA1024 | 2026 | CONTROLLED_BUDGET | 0.160093 | 0.195266 | 0.334097 |
| P2_WSTAR_FA1024 | 2027 | CONTROLLED_BUDGET | 0.160087 | 0.195260 | 0.334090 |
| P2_WSTAR_DP8 | 2026 | PRACTICAL_BEST | 0.160130 | 0.195292 | 0.334111 |
| P2_WSTAR_DP8 | 2027 | PRACTICAL_BEST | 0.160131 | 0.195294 | 0.334111 |
| P2_WSTAR_DP8 | 2026 | CONTROLLED_BUDGET | 0.160095 | 0.195268 | 0.334099 |
| P2_WSTAR_DP8 | 2027 | CONTROLLED_BUDGET | 0.160089 | 0.195263 | 0.334097 |
| P2_WSTAR_DP1 | 2026 | PRACTICAL_BEST | 0.160130 | 0.195292 | 0.334111 |
| P2_WSTAR_DP1 | 2026 | CONTROLLED_BUDGET | 0.160084 | 0.195261 | 0.334105 |
| P2_C1_FT_CENTRAL_FULL | 2026 | PRACTICAL_BEST | 0.159899 | 0.195127 | 0.333736 |
| P2_C1_FT_CENTRAL_FULL | 2027 | PRACTICAL_BEST | 0.159935 | 0.195175 | 0.333796 |
| P2_C1_FT_CENTRAL_FULL | 2026 | CONTROLLED_BUDGET | 0.159815 | 0.195079 | 0.333791 |
| P2_C1_FT_CENTRAL_FULL | 2027 | CONTROLLED_BUDGET | 0.159672 | 0.194973 | 0.333858 |
| P2_C1_FT_CENTRAL_FROZEN | 2026 | PRACTICAL_BEST | 0.159888 | 0.195085 | 0.333960 |
| P2_C1_FT_CENTRAL_FROZEN | 2027 | PRACTICAL_BEST | 0.159877 | 0.195067 | 0.333959 |
| P2_C1_FT_CENTRAL_FROZEN | 2026 | CONTROLLED_BUDGET | 0.159736 | 0.194858 | 0.333605 |
| P2_C1_FT_CENTRAL_FROZEN | 2027 | CONTROLLED_BUDGET | 0.159701 | 0.194879 | 0.333740 |
| P2_C2_CFULL_SMALL | 2026 | PRACTICAL_BEST | 0.160670 | 0.195803 | 0.334310 |
| P2_C2_CFULL_SMALL | 2026 | CONTROLLED_BUDGET | 0.160670 | 0.195803 | 0.334310 |
| FT_P2_WSTAR | 2026 | PRACTICAL_BEST | 0.162445 | 0.198022 | 0.338006 |
| FT_P2_WSTAR | 2027 | PRACTICAL_BEST | 0.162366 | 0.198098 | 0.338495 |

## Labels (PREG2 §6.4)

| claim | label |
|---|---|
| Phase-1 K1…K21 (all registered K) | CONFIRMATORY (PREG1), unchanged |
| Phase-2 Q1…Q8 on RETRO_TEST (the sitting) | CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP), always printed with disclosure D-a verbatim |
| Phase-2 Q9…Q14 on RETRO_TEST | TERTIARY_P2 (unadjusted; DESCRIPTIVE Holm band) |
| Phase-2 arms on MAIN VALIDATION | POST_VALIDATION_EXPLORATORY |
| Q1 on TEST2 | REPLICATED_FRESH (same sign, 95 % CI excludes 0) / NOT_REPLICATED / TEST2_NOT_AVAILABLE |
| Phase-1 locked arms on TEST2 | FRESH_PERIOD_AUDIT (DESCRIPTIVE) |
| Fallback FB (§6.5) | HOLDOUT_USERS_SAME_PERIOD, never a Q1 confirmation |
| audits E1–E5, devices §8, D1-ext | DESCRIPTIVE (tertiary) |

## Disclosures (PREG2 §11, verbatim)

- D-a. Phase-2 hypotheses were generated after the Phase-1 VALIDATION read; arms were selected on CALIB; everything was locked before any TEST value existed.
- D-b. CALIB selection data are a subset of the MAIN users and of the VALIDATION window that was read.
- D-c. θ_pre is one seed (no second seed was run). The PRE recipe was not re-tuned at its scale (Phase-1 note 3).
- D-d. C_FULL differs from PRE in size and recipe. C2 (if run) is the matched reference.
- D-e. Central DP trusts the aggregator. SecAgg is simulated (costs counted, cryptography not executed). DP-FTRL / LDP are the untrusted-server variants.
- D-f. Device numbers come from the listed devices only. On-device training cost is a PC proxy unless stated otherwise.
- D-g. RETRO_TEST was inspected at period level by earlier v1 campaigns (Phase-1 E12(d)).
- D-h. Phase 2 can confirm a gain ≥ ≈ 0.0005 MRR@20 or equivalence within ±0.0005; a true gain between them is likely to read INCONCLUSIVE.
- D-i. The DP arms keep T = 384 and the registered z. A small-step recipe over 384 rounds may move θ little; that is the registered mechanism's answer, not a defect to tune away.
- D-j. Futility uses CALIB headroom (32,768 users); MAIN has 131,072 users, so FUTILITY_SKIP_WB_WC may under-read the headroom available on MAIN.
- D-k. The Phase-1 TEST result was held for the Phase-2 runs (owner decision, DL 406). No Phase-1 arm, contrast or rule changed.

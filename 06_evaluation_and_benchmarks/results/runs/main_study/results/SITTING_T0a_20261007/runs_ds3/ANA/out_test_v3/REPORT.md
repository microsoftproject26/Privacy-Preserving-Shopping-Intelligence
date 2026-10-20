# PPSI V4 retention report — split TEST

- RETENTION_RESULT.json self_sha256: `516a1e8e5462ef258916c6b88089771705568eb51d70d85c4fc18441f91808d1`
- script sha256: `9a3b2c2a918ef2baacc5299426941f1eb94c366f5bb6aa7c29b4bfdbca0afe58`; contrasts sha256: `04a3cd196149e2acf2ab5bb004f58c997199aa52e32f3c11c9ae647255beed23` (content `df7884b81fa7cb79589e033b41457eccf560d0c35feb0cdca693107913fe4264`)
- block index sha256: `009355eb8b6f8eed6a9e70248b66c4d012f85aa2f18b2fbee061723c2e4df0a8`
- comparison key: `{"candidate_order_hash": "ab606d5e3728c993d31048f6d613838658d3a5ce8a0d0bd023611ce0ff027476", "data_protocol_hash": "bc699a97dd2fab054e5a5c00f9661dd32b0677941d3f23f2e4b81751b3b7b330", "eval_device_class": "NVIDIA_A40-FP32-DET", "eval_score_block": "158486", "eval_stream": "E2E_ROWS", "evaluator_code_sha256": "61b64e61c1a2e4f5a6fecd42f0ad3b98de26562d13a07758573cc9496b8d962e", "feature_contract_hash": "e97600f508497398da47f48b029d49a6fffee316e6a0c1ea20e046fe81fda5aa", "label_policy_hash": "75acea25716829cba74a7703f17208f405b455c72af2ec19f413585636f7e59a", "objective_id": "EXACT_FULL_K_CE_SUM_OVER_LOSS_ELIGIBLE_DIVIDE_ONCE_LS0", "precision_id": "FP32_STRICT:matmul_tf32=off,cudnn_tf32=off,float32_matmul_precision=highest,deterministic_algorithms=STRICT", "split": "RETRO_TEST"}`; device pin `db744ca3f0da27ac7b65c41b4b32f2c68ec9d0bd66bbc0dc33e874a226834e6b`
- bootstrap: paired user, 1000 resamples, seed 20260926; ratios bootstrapped directly; seed mean on the same W.
- contrasts: format PPSI_V4_ANA_CONTRASTS_4.
- multiplicity (06D §4; 06I §1; 06L-C C4 / C13): 33 pre-registered contrasts (v3's 26 with K20 / K20b / K21 on FA_WARM_FROZEN, + K_INIT, P5, K4w, K5w, K12w, K13w, K14w); primary = the pair K20b + K21, read unadjusted, outside Holm (06P P1 / P6); Holm step-down over the secondary family of 18 (K1, K2, K3, K8, K9b, K9c, K11b, K12, K17, K20, K20c, K_INIT, P5, K4w, K5w, K12w, K13w, K14w; 06P P6); tertiary unadjusted with a DESCRIPTIVE Holm band beside (step-down within the tertiary set); unadjusted 95% paired user-bootstrap percentile CIs (1,000 resamples, seed 20260926) for every contrast; K14 / K14w TOST on the unadjusted 90% CI; algebraic dependencies are not independent evidence; every other number is DESCRIPTIVE (06P).
- contrasts registry: company/gates/ANA_CONTRASTS_v4.json (06P).
- Anything added after the first VALIDATION read is POST_VALIDATION_EXPLORATORY (06L-C status line; §C5): never confirmatory, never in the loaded registry.
- **Only section 1 (K1-K21, K9b, K9c, K11b, K20b, K20c, K_INIT, P5, K4w, K5w, K12w, K13w, K14w) is CONFIRMATORY (each K under its registered rule and seeds, bold); every other number is DESCRIPTIVE.**

- sanity trigger (amendment 04 §1 / 06E §6, not a verdict): R(C_SMALL CB / C_BIG PB) = 93.09 -> fired = False; R(C_SMALL PB / C_BIG PB) = 94.69

## 1. Confirmatory contrasts (06B §2; rule map 06D §3; hierarchy 06D §4; seeds 06E §2) — CONFIRMATORY

| id | contrast | question | hierarchy | rule | label | seeds | seeds status | point | 95% CI | CI vs 100/0 | users | decisions | harmed share | seed spread (SEED_SENSITIVE) | Holm band | claim key |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **K1** | **R(C_SMALL / C_BIG)** | **size cost** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **94.69** | **[94.03, 95.31]** | **below** | **26254** | **229742** | **0.4331** | **0.79 (SEED_SENSITIVE)** | **[93.70, 95.69]** | **BELOW** |
| K1 | R(C_SMALL / C_BIG) | size cost | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 99.77 | [98.90, 100.57] | includes | 26254 | 229742 | 0.3802 | 0.37 (no) | - | - |
| **K2** | **R(FA / C_BIG)** | **federated retention (headline)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **85.11** | **[84.30, 85.90]** | **below** | **26254** | **229742** | **0.4929** | **0.14 (no)** | **[83.92, 86.30]** | **BELOW** |
| K2 | R(FA / C_BIG) | federated retention (headline) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 91.21 | [90.23, 92.15] | below | 26254 | 229742 | 0.4353 | 0.42 (no) | - | - |
| **K3** | **R(FA / C_SMALL)** | **federation cost at matched size** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **89.88** | **[89.15, 90.62]** | **below** | **26254** | **229742** | **0.4717** | **0.60 (no)** | **[88.76, 91.00]** | **BELOW** |
| K3 | R(FA / C_SMALL) | federation cost at matched size | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 91.43 | [90.65, 92.20] | below | 26254 | 229742 | 0.4494 | 0.76 (no) | - | - |
| **K4** | **R(FP / C_BIG)** | **FedProx retention** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **85.11** | **[84.30, 85.90]** | **below** | **26254** | **229742** | **0.4927** | **0.16 (no)** | **[83.96, 86.27]** | **BELOW** |
| K4 | R(FP / C_BIG) | FedProx retention | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 91.21 | [90.23, 92.15] | below | 26254 | 229742 | 0.4354 | 0.40 (no) | - | - |
| **K5** | **R(PF / C_BIG)** | **personalised-FL retention** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **85.12** | **[84.32, 85.90]** | **below** | **26254** | **229742** | **0.4928** | **0.12 (no)** | **[83.98, 86.26]** | **BELOW** |
| K5 | R(PF / C_BIG) | personalised-FL retention | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 91.22 | [90.25, 92.15] | below | 26254 | 229742 | 0.4354 | 0.44 (no) | - | - |
| **K6** | **R(FT_FA / C_BIG)** | **deployed on-device system (FL + local fine-tune)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **87.56** | **[86.75, 88.33]** | **below** | **26254** | **229742** | **0.4803** | **0.46 (no)** | **[86.42, 88.70]** | **BELOW** |
| K6 | R(FT_FA / C_BIG) | deployed on-device system (FL + local fine-tune) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K7** | **R(FT_C / C_BIG)** | **cloud-trained + local fine-tune** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **96.35** | **[95.70, 97.02]** | **below** | **26254** | **229742** | **0.4204** | **1.35 (SEED_SENSITIVE)** | **[95.40, 97.31]** | **BELOW** |
| K7 | R(FT_C / C_BIG) | cloud-trained + local fine-tune | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K8** | **R(LO / C_BIG)** | **device-only retention** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **2.10** | **[1.88, 2.34]** | **below** | **26254** | **229742** | **0.7768** | **0.21 (no)** | **[1.75, 2.45]** | **BELOW** |
| K8 | R(LO / C_BIG) | device-only retention | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K9** | **FT_C − C_SMALL** | **local fine-tuning gain (paired Δ)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.002470** | **[0.001955, 0.002993]** | **above** | **26254** | **229742** | **0.3348** | **0.000823 (SEED_SENSITIVE)** | **[0.001710, 0.003230]** | **ABOVE** |
| K9 | FT_C − C_SMALL | local fine-tuning gain (paired Δ) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K9b** | **FT_C − C_SMALL** | **local fine-tuning gain (paired Δ) for base-unseen users (UNSEEN_HOLDOUT)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **0.002732** | **[0.001512, 0.004171]** | **above** | **6650** | **59094** | **0.2716** | - | **[0.000991, 0.004472]** | **ABOVE** |
| K9b | FT_C − C_SMALL | local fine-tuning gain (paired Δ) for base-unseen users (UNSEEN_HOLDOUT) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K9c** | **FT_PRE − PRE** | **local fine-tuning gain (paired Δ) over θ_pre on MAIN: the realistic L5 answer (FT_PRE = θ_pre + the registered FT_C procedure with SELECTION_FT_C)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.002283** | **[0.001588, 0.003076]** | **above** | **26254** | **229742** | **0.3100** | **0.000168 (no)** | **[0.001261, 0.003304]** | **ABOVE** |
| K9c | FT_PRE − PRE | local fine-tuning gain (paired Δ) over θ_pre on MAIN: the realistic L5 answer (FT_PRE = θ_pre + the registered FT_C procedure with SELECTION_FT_C) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K10** | **FT_FA − FA** | **local fine-tuning gain after FL (paired Δ)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.003637** | **[0.003105, 0.004221]** | **above** | **26254** | **229742** | **0.3230** | **0.000472 (no)** | **[0.002875, 0.004399]** | **ABOVE** |
| K10 | FT_FA − FA | local fine-tuning gain after FL (paired Δ) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K11** | **FA − LO** | **collaboration value (paired Δ)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.123378** | **[0.121064, 0.125358]** | **above** | **26254** | **229742** | **0.0093** | **0.000233 (no)** | **[0.120480, 0.126276]** | **ABOVE** |
| K11 | FA − LO | collaboration value (paired Δ) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K11b** | **FA − LAST** | **collaboration value over the best own-history heuristic (paired Δ; LAST = the no-training device heuristic)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.050933** | **[0.048639, 0.052886]** | **above** | **26254** | **229742** | **0.2618** | **0.000086 (no)** | **[0.047694, 0.054172]** | **ABOVE** |
| K11b | FA − LAST | collaboration value over the best own-history heuristic (paired Δ; LAST = the no-training device heuristic) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.050933 | [0.048639, 0.052886] | above | 26254 | 229742 | 0.2618 | 0.000086 (no) | - | - |
| K12 | R(DP8 / FA_1024) | privacy cost at ε = 8 | secondary | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 98.52 | [97.70, 99.38] | below | 26254 | 229742 | 0.1495 | 4.40 (SEED_SENSITIVE) | - | - |
| **K12** | **R(DP8 / FA_1024)** | **privacy cost at ε = 8** | **secondary** | **CONTROLLED_BUDGET** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **98.52** | **[97.70, 99.38]** | **below** | **26254** | **229742** | **0.1495** | **4.40 (SEED_SENSITIVE)** | **[97.38, 99.65]** | **BELOW** |
| K13 | R(DP1 / FA_1024) | privacy cost at ε = 1 (seed 2026) | tertiary | PRACTICAL_BEST | DESCRIPTIVE | [2026] | COMPLETE | 94.38 | [92.53, 96.00] | below | 26254 | 229742 | 0.1522 | - | - | - |
| **K13** | **R(DP1 / FA_1024)** | **privacy cost at ε = 1 (seed 2026)** | **tertiary** | **CONTROLLED_BUDGET** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **94.38** | **[92.53, 96.00]** | **below** | **26254** | **229742** | **0.1522** | - | **[92.29, 96.47]** | **BELOW** |
| **K14** | **R(FA_Q8 / FA)** | **compression cost (8-bit uploads)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **100.11** | **[99.85, 100.38]** | **includes** | **26254** | **229742** | **0.3396** | **0.17 (no)** | **[99.80, 100.42]** | **INCLUDES** |
| K14 | R(FA_Q8 / FA) | compression cost (8-bit uploads) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 100.11 | [99.85, 100.38] | includes | 26254 | 229742 | 0.3396 | 0.17 (no) | - | - |
| **K15** | **R(C_BIG / C_FULL)** | **cloud data advantage (seed 2026; if C_FULL runs)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **82.30** | **[81.54, 83.10]** | **below** | **26254** | **229742** | **0.5177** | - | **[81.26, 83.34]** | **BELOW** |
| K15 | R(C_BIG / C_FULL) | cloud data advantage (seed 2026; if C_FULL runs) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026] | COMPLETE | 77.04 | [76.28, 77.83] | below | 26254 | 229742 | 0.5515 | - | - | - |
| K15 | R(C_BIG / C_FULL) | cloud data advantage (seed 2026; if C_FULL runs) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 77.02 | [76.37, 77.71] | below | 26254 | 229742 | 0.5998 | 0.02 (no) | - | - |
| K15 | R(C_BIG / C_FULL) | cloud data advantage (seed 2026; if C_FULL runs) | tertiary | PRACTICAL_BEST | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 82.54 | [81.92, 83.20] | below | 26254 | 229742 | 0.5599 | 0.49 (no) | - | - |
| **K16** | **R(INT8 / FP32) for C_SMALL** | **on-device quantisation cost (seed 2026)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **99.98** | **[99.75, 100.23]** | **includes** | **16384** | **142316** | **0.2158** | - | **[99.70, 100.26]** | **INCLUDES** |
| K16 | R(INT8 / FP32) for C_SMALL | on-device quantisation cost (seed 2026) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K17** | **R(FA / C_FULL)** | **federated retention vs the full-data cloud model (seed 2026)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **70.10** | **[69.19, 70.94]** | **below** | **26254** | **229742** | **0.5720** | - | **[68.82, 71.39]** | **BELOW** |
| K17 | R(FA / C_FULL) | federated retention vs the full-data cloud model (seed 2026) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026] | COMPLETE | 70.10 | [69.19, 70.94] | below | 26254 | 229742 | 0.5720 | - | - | - |
| K17 | R(FA / C_FULL) | federated retention vs the full-data cloud model (seed 2026) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 70.25 | [69.51, 70.98] | below | 26254 | 229742 | 0.6133 | 0.30 (no) | - | - |
| K17 | R(FA / C_FULL) | federated retention vs the full-data cloud model (seed 2026) | secondary | PRACTICAL_BEST | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 70.25 | [69.51, 70.98] | below | 26254 | 229742 | 0.6133 | 0.30 (no) | - | - |
| **K18** | **R(FT_C / C_FULL)** | **cloud-trained + local fine-tune vs the full-data cloud model (seed 2026)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **79.85** | **[79.09, 80.74]** | **below** | **26254** | **229742** | **0.5292** | - | **[78.76, 80.95]** | **BELOW** |
| K18 | R(FT_C / C_FULL) | cloud-trained + local fine-tune vs the full-data cloud model (seed 2026) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| K18 | R(FT_C / C_FULL) | cloud-trained + local fine-tune vs the full-data cloud model (seed 2026) | tertiary | PRACTICAL_BEST | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 79.53 | [78.86, 80.28] | below | 26254 | 229742 | 0.5687 | 0.64 (no) | - | - |
| **K19** | **R(LO / C_FULL)** | **device-only vs the full-data cloud model (seed 2026)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **1.64** | **[1.45, 1.88]** | **below** | **26254** | **229742** | **0.7861** | - | **[1.37, 1.91]** | **BELOW** |
| K19 | R(LO / C_FULL) | device-only vs the full-data cloud model (seed 2026) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| K19 | R(LO / C_FULL) | device-only vs the full-data cloud model (seed 2026) | tertiary | PRACTICAL_BEST | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 1.74 | [1.55, 1.93] | below | 26254 | 229742 | 0.8131 | 0.18 (no) | - | - |
| **K20** | **R(FA_WARM_FROZEN / C_BIG)** | **deployment-branch federated retention vs the C_BIG cloud model (warm start from theta_pre, S8 frozen item tables = FA_WARM*; seeds 2026, 2027)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **107.39** | **[106.50, 108.32]** | **above** | **26254** | **229742** | **0.3416** | **0.49 (no)** | **[106.05, 108.74]** | **ABOVE** |
| K20 | R(FA_WARM_FROZEN / C_BIG) | deployment-branch federated retention vs the C_BIG cloud model (warm start from theta_pre, S8 frozen item tables = FA_WARM*; seeds 2026, 2027) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 114.86 | [113.69, 116.01] | above | 26254 | 229742 | 0.3043 | 0.49 (no) | - | - |
| **K20b** | **R(FA_WARM_FROZEN / C_FULL)** | **deployment-branch federated retention vs its matched cloud reference C_FULL (2 FA_WARM_FROZEN seeds vs C_FULL s2026) — PRIMARY with K21 (06N N1; 06P P1)** | **primary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **88.49** | **[87.73, 89.18]** | **below** | **26254** | **229742** | **0.4737** | **0.20 (no)** | - | **BELOW** |
| K20b | R(FA_WARM_FROZEN / C_FULL) | deployment-branch federated retention vs its matched cloud reference C_FULL (2 FA_WARM_FROZEN seeds vs C_FULL s2026) — PRIMARY with K21 (06N N1; 06P P1) | primary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 88.31 | [87.56, 89.00] | below | 26254 | 229742 | 0.4727 | 0.03 (no) | - | - |
| **K20c** | **R(PRE / C_FULL)** | **the pretrained model alone vs its matched cloud reference C_FULL (PRE single seed)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **88.77** | **[88.03, 89.47]** | **below** | **26254** | **229742** | **0.4677** | - | **[87.69, 89.85]** | **BELOW** |
| K20c | R(PRE / C_FULL) | the pretrained model alone vs its matched cloud reference C_FULL (PRE single seed) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026] | COMPLETE | 88.77 | [88.03, 89.47] | below | 26254 | 229742 | 0.4677 | - | - | - |
| **K21** | **FA_WARM_FROZEN − PRE** | **value of federated training on the private users over the pretrained model alone (paired Δ) — PRIMARY with K20b (06N N1; 06P P1)** | **primary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **-0.000508** | **[-0.000933, -0.000037]** | **below** | **26254** | **229742** | **0.3247** | **0.000353 (no)** | - | **BELOW** |
| K21 | FA_WARM_FROZEN − PRE | value of federated training on the private users over the pretrained model alone (paired Δ) — PRIMARY with K20b (06N N1; 06P P1) | primary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | -0.000827 | [-0.001331, -0.000341] | below | 26254 | 229742 | 0.3244 | 0.000061 (no) | - | - |
| **K_INIT** | **FA_WARM_FROZEN − FA** | **value of the pretrained start (paired Δ, seeds 2026 / 2027; 06N N1; 06P P3)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.033119** | **[0.031774, 0.034516]** | **above** | **26254** | **229742** | **0.2590** | **0.000439 (no)** | **[0.031139, 0.035099]** | **ABOVE** |
| K_INIT | FA_WARM_FROZEN − FA | value of the pretrained start (paired Δ, seeds 2026 / 2027; 06N N1; 06P P3) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.032800 | [0.031478, 0.034221] | above | 26254 | 229742 | 0.2614 | 0.000025 (no) | - | - |
| **P5** | **FT_FA_WARM_FROZEN − FA_WARM_FROZEN** | **local fine-tuning gain after the pretrained-line FL (paired Δ, base = the same-seed best.pt, harmed-client share beside; 06P P5)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.002045** | **[0.001324, 0.002807]** | **above** | **26254** | **229742** | **0.3320** | **0.000427 (no)** | **[0.001023, 0.003066]** | **ABOVE** |
| P5 | FT_FA_WARM_FROZEN − FA_WARM_FROZEN | local fine-tuning gain after the pretrained-line FL (paired Δ, base = the same-seed best.pt, harmed-client share beside; 06P P5) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K4w** | **R(FP_WARM_FROZEN / C_FULL)** | **FedProx retention on the pretrained line vs C_FULL (06P P4)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **88.49** | **[87.73, 89.18]** | **below** | **26254** | **229742** | **0.4737** | **0.20 (no)** | **[87.45, 89.52]** | **BELOW** |
| K4w | R(FP_WARM_FROZEN / C_FULL) | FedProx retention on the pretrained line vs C_FULL (06P P4) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 88.32 | [87.56, 89.00] | below | 26254 | 229742 | 0.4728 | 0.02 (no) | - | - |
| **K5w** | **R(PF_WARM_FROZEN / C_FULL)** | **personalised-FL retention on the pretrained line vs C_FULL (06P P4)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **88.35** | **[87.60, 89.07]** | **below** | **26254** | **229742** | **0.4751** | **0.10 (no)** | **[87.33, 89.38]** | **BELOW** |
| K5w | R(PF_WARM_FROZEN / C_FULL) | personalised-FL retention on the pretrained line vs C_FULL (06P P4) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 88.35 | [87.61, 89.04] | below | 26254 | 229742 | 0.4726 | 0.00 (no) | - | - |
| K12w | R(DP8_FROZEN / FA_1024_WARM_FROZEN) | privacy cost at ε = 8 on the pretrained line (06P P4) | secondary | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 99.99 | [99.96, 100.02] | includes | 26254 | 229742 | 0.0524 | 0.05 (SEED_SENSITIVE) | - | - |
| **K12w** | **R(DP8_FROZEN / FA_1024_WARM_FROZEN)** | **privacy cost at ε = 8 on the pretrained line (06P P4)** | **secondary** | **CONTROLLED_BUDGET** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **100.00** | **[99.95, 100.07]** | **includes** | **26254** | **229742** | **0.1395** | **0.02 (no)** | **[99.93, 100.08]** | **INCLUDES** |
| K13w | R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN) | privacy cost at ε = 1 on the pretrained line (seed 2026; 06P P4) | secondary | PRACTICAL_BEST | DESCRIPTIVE | [2026] | COMPLETE | 100.06 | [99.99, 100.13] | includes | 26254 | 229742 | 0.0604 | - | - | - |
| **K13w** | **R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN)** | **privacy cost at ε = 1 on the pretrained line (seed 2026; 06P P4)** | **secondary** | **CONTROLLED_BUDGET** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **100.03** | **[99.92, 100.15]** | **includes** | **26254** | **229742** | **0.1532** | - | **[99.89, 100.17]** | **INCLUDES** |
| **K14w** | **R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN)** | **compression cost (8-bit uploads) on the pretrained line (06P P4)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **99.85** | **[99.72, 99.98]** | **below** | **26254** | **229742** | **0.2845** | **0.34 (SEED_SENSITIVE)** | **[99.69, 100.01]** | **BELOW** |
| K14w | R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN) | compression cost (8-bit uploads) on the pretrained line (06P P4) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 100.02 | [100.00, 100.04] | includes | 26254 | 229742 | 0.0379 | 0.04 (SEED_SENSITIVE) | - | - |

- K12 PRACTICAL_BEST (DESCRIPTIVE): best-of-12 checkpoint selection on the users' own validation data is not covered by the DP accountant (06D §3).
- K13 PRACTICAL_BEST (DESCRIPTIVE): best-of-12 checkpoint selection on the users' own validation data is not covered by the DP accountant (06D §3).
- K12w PRACTICAL_BEST (DESCRIPTIVE): best-of-12 checkpoint selection on the users' own validation data is not covered by the DP accountant (06D §3).
- K13w PRACTICAL_BEST (DESCRIPTIVE): best-of-12 checkpoint selection on the users' own validation data is not covered by the DP accountant (06D §3).

### 1b. Deployment branch (06K; DL 206; 06L-C C7: the Edge answer and the L5 row lead with these) — K20, K20b, K20c, K21, K9c

| id | contrast | question | hierarchy | rule | label | seeds | seeds status | point | 95% CI | CI vs 100/0 | users | decisions | harmed share | seed spread (SEED_SENSITIVE) | Holm band | claim key |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **K20** | **R(FA_WARM_FROZEN / C_BIG)** | **deployment-branch federated retention vs the C_BIG cloud model (warm start from theta_pre, S8 frozen item tables = FA_WARM*; seeds 2026, 2027)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **107.39** | **[106.50, 108.32]** | **above** | **26254** | **229742** | **0.3416** | **0.49 (no)** | **[106.05, 108.74]** | **ABOVE** |
| K20 | R(FA_WARM_FROZEN / C_BIG) | deployment-branch federated retention vs the C_BIG cloud model (warm start from theta_pre, S8 frozen item tables = FA_WARM*; seeds 2026, 2027) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 114.86 | [113.69, 116.01] | above | 26254 | 229742 | 0.3043 | 0.49 (no) | - | - |
| **K20b** | **R(FA_WARM_FROZEN / C_FULL)** | **deployment-branch federated retention vs its matched cloud reference C_FULL (2 FA_WARM_FROZEN seeds vs C_FULL s2026) — PRIMARY with K21 (06N N1; 06P P1)** | **primary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **88.49** | **[87.73, 89.18]** | **below** | **26254** | **229742** | **0.4737** | **0.20 (no)** | - | **BELOW** |
| K20b | R(FA_WARM_FROZEN / C_FULL) | deployment-branch federated retention vs its matched cloud reference C_FULL (2 FA_WARM_FROZEN seeds vs C_FULL s2026) — PRIMARY with K21 (06N N1; 06P P1) | primary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 88.31 | [87.56, 89.00] | below | 26254 | 229742 | 0.4727 | 0.03 (no) | - | - |
| **K20c** | **R(PRE / C_FULL)** | **the pretrained model alone vs its matched cloud reference C_FULL (PRE single seed)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **88.77** | **[88.03, 89.47]** | **below** | **26254** | **229742** | **0.4677** | - | **[87.69, 89.85]** | **BELOW** |
| K20c | R(PRE / C_FULL) | the pretrained model alone vs its matched cloud reference C_FULL (PRE single seed) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026] | COMPLETE | 88.77 | [88.03, 89.47] | below | 26254 | 229742 | 0.4677 | - | - | - |
| **K21** | **FA_WARM_FROZEN − PRE** | **value of federated training on the private users over the pretrained model alone (paired Δ) — PRIMARY with K20b (06N N1; 06P P1)** | **primary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **-0.000508** | **[-0.000933, -0.000037]** | **below** | **26254** | **229742** | **0.3247** | **0.000353 (no)** | - | **BELOW** |
| K21 | FA_WARM_FROZEN − PRE | value of federated training on the private users over the pretrained model alone (paired Δ) — PRIMARY with K20b (06N N1; 06P P1) | primary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | -0.000827 | [-0.001331, -0.000341] | below | 26254 | 229742 | 0.3244 | 0.000061 (no) | - | - |
| **K9c** | **FT_PRE − PRE** | **local fine-tuning gain (paired Δ) over θ_pre on MAIN: the realistic L5 answer (FT_PRE = θ_pre + the registered FT_C procedure with SELECTION_FT_C)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.002283** | **[0.001588, 0.003076]** | **above** | **26254** | **229742** | **0.3100** | **0.000168 (no)** | **[0.001261, 0.003304]** | **ABOVE** |
| K9c | FT_PRE − PRE | local fine-tuning gain (paired Δ) over θ_pre on MAIN: the realistic L5 answer (FT_PRE = θ_pre + the registered FT_C procedure with SELECTION_FT_C) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |


- (1) PRE is trained on 1,217,610 users (9x MAIN) while C_BIG saw 131,072: K20 = R(FA_WARM / C_BIG) can exceed 100 and mixes a data advantage into the FL retention. The matched cloud reference of the deployment branch is C_FULL (TIEBREAK = PRE users + MAIN users): R(FA_WARM / C_FULL) and R(PRE / C_FULL) are printed beside K20 as DESCRIPTIVE, and the thesis text for the deployment branch leads with them. K21 = FA_WARM - PRE stays the value of federated training on the private users.
- (2) theta_pre is ONE seed (2026) shared by FA_WARM s2026 and s2027: their seed spread measures FL variance only.
- (3) PRE applies the SMALL recipe (tuned at MAIN scale, 2.6M decisions) to 24.1M decisions at 6.0 EFE (about 9x more updates) without re-tuning: this can only understate PRE. PRE marks on UNSEEN_VAL are DESCRIPTIVE and never select.

### 1c. Per-K statistics (06L-C C13-C15) — DESCRIPTIVE beside each K

Holm band: Holm step-down on normal intervals from the bootstrap SE (06L-C C13); after the first non-rejection every non-rejected K's band is at alpha / (m - r), r = the number of rejections (REV-DELTA-03 M3: band and decision agree); Bonferroni band beside it; family = 18 registered contrasts, 18 computed.

| id | rule | per-seed points | seed spread | SEED_SENSITIVE | user + seed interval | Holm p | Holm reject | Holm band | Bonferroni band | harmed share | harmed share, seed-vs-seed null | K14 equivalence (±1 % rel.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K1 | PRACTICAL_BEST | {"2026": 95.08581373071921, "2027": 94.29684760953091} | 0.79 | True | [93.68, 95.70] | 0.0000 | True | [93.70, 95.69] | [93.70, 95.69] | 0.4331 | 0.3615 | - |
| K2 | PRACTICAL_BEST | {"2026": 85.18063394818586, "2027": 85.0392439189217} | 0.14 | False | [84.31, 85.91] | 0.0000 | True | [83.92, 86.30] | [83.91, 86.31] | 0.4929 | 0.3569 | - |
| K3 | PRACTICAL_BEST | {"2026": 89.58290475319001, "2027": 90.18248867773018} | 0.60 | False | [88.94, 90.83] | 0.0000 | True | [88.76, 91.00] | [88.75, 91.02] | 0.4717 | 0.3569 | - |
| K4 | PRACTICAL_BEST | {"2026": 85.19173899690013, "2027": 85.03304077313481} | 0.16 | False | [84.31, 85.91] | 0.0000 | - | [83.96, 86.27] | [83.96, 86.27] | 0.4927 | 0.3571 | - |
| K5 | PRACTICAL_BEST | {"2026": 85.18207931151248, "2027": 85.05774532115228} | 0.12 | False | [84.33, 85.91] | 0.0000 | - | [83.98, 86.26] | [83.97, 86.27] | 0.4928 | 0.3569 | - |
| K6 | PRACTICAL_BEST | {"2026": 87.78952548975798, "2027": 87.32482807574421} | 0.46 | False | [86.65, 88.47] | 0.0000 | - | [86.42, 88.70] | [86.40, 88.72] | 0.4803 | 0.3582 | - |
| K7 | PRACTICAL_BEST | {"2026": 97.02691073810492, "2027": 95.68009482500187} | 1.35 | True | [94.87, 97.83] | 0.0000 | - | [95.40, 97.31] | [95.37, 97.34] | 0.4204 | 0.3644 | - |
| K8 | PRACTICAL_BEST | {"2026": 1.996749098737878, "2027": 2.2066384055333885} | 0.21 | False | [1.79, 2.41] | 0.0000 | True | [1.75, 2.45] | [1.75, 2.46] | 0.7768 | 0.0219 | - |
| K9 | PRACTICAL_BEST | {"2026": 0.0028817516187788217, "2027": 0.002058383633994715} | 0.000823 | True | [0.001501, 0.003439] | 0.0000 | - | [0.001710, 0.003230] | [0.001678, 0.003262] | 0.3348 | 0.3644 | - |
| K9b | PRACTICAL_BEST | {"2026": 0.00273151849214906} | - | - | - | 0.0003 | True | [0.000991, 0.004472] | [0.000710, 0.004753] | 0.2716 | - | - |
| K9c | PRACTICAL_BEST | {"2026": 0.0023668266323711162, "2027": 0.0021988124928734587} | 0.000168 | False | [0.001520, 0.003045] | 0.0000 | True | [0.001261, 0.003304] | [0.001147, 0.003419] | 0.3100 | 0.1838 | - |
| K10 | PRACTICAL_BEST | {"2026": 0.003873159040757801, "2027": 0.0034011339187255962} | 0.000472 | False | [0.002921, 0.004353] | 0.0000 | - | [0.002875, 0.004399] | [0.002831, 0.004443] | 0.3230 | 0.3582 | - |
| K11 | PRACTICAL_BEST | {"2026": 0.12349475266259911, "2027": 0.12326161053709148} | 0.000233 | False | [0.121255, 0.125502] | 0.0000 | - | [0.120480, 0.126276] | [0.120265, 0.126492] | 0.0093 | 0.3569 | - |
| K11b | PRACTICAL_BEST | {"2026": 0.05088981955354592, "2027": 0.05097596100442513} | 0.000086 | False | [0.048752, 0.053113] | 0.0000 | True | [0.047694, 0.054172] | [0.047608, 0.054258] | 0.2618 | 0.3569 | - |
| K12 | CONTROLLED_BUDGET | {"2026": 100.71764905473636, "2027": 96.31472607763433} | 4.40 | True | [94.11, 102.92] | 0.0044 | True | [97.38, 99.65] | [97.16, 99.88] | 0.1495 | 0.1636 | - |
| K13 | CONTROLLED_BUDGET | {"2026": 94.37912496070706} | - | - | - | 0.0000 | - | [92.29, 96.47] | [91.85, 96.91] | 0.1522 | - | - |
| K14 | PRACTICAL_BEST | {"2026": 100.19256693116817, "2027": 100.02230405315628} | 0.17 | False | [99.79, 100.43] | 0.8771 | - | [99.80, 100.42] | [99.71, 100.51] | 0.3396 | 0.3607 | True (90% CI [99.89, 100.35]) |
| K15 | PRACTICAL_BEST | {"2026": 82.3001700716403} | - | - | - | 0.0000 | - | [81.26, 83.34] | [81.16, 83.44] | 0.5177 | 0.3676 | - |
| K16 | PRACTICAL_BEST | {"2026": 99.98162984720607} | - | - | - | 0.8843 | - | [99.70, 100.26] | [99.62, 100.35] | 0.2158 | - | - |
| K17 | PRACTICAL_BEST | {"2026": 70.10380660745832} | - | - | - | 0.0000 | True | [68.82, 71.39] | [68.77, 71.43] | 0.5720 | 0.3569 | - |
| K18 | PRACTICAL_BEST | {"2026": 79.85331255271896} | - | - | - | 0.0000 | - | [78.76, 80.95] | [78.62, 81.09] | 0.5292 | 0.3644 | - |
| K19 | PRACTICAL_BEST | {"2026": 1.6433279041652182} | - | - | - | 0.0000 | - | [1.37, 1.91] | [1.33, 1.96] | 0.7861 | 0.0219 | - |
| K20 | PRACTICAL_BEST | {"2026": 107.63666352178012, "2027": 107.14768396098766} | 0.49 | False | [106.36, 108.43] | 0.0000 | True | [106.05, 108.74] | [105.99, 108.80] | 0.3416 | 0.2821 | - |
| K20b | PRACTICAL_BEST | {"2026": 88.58515713786424, "2027": 88.38951084861765} | 0.20 | False | [87.73, 89.25] | - | - | - | - | 0.4737 | 0.2821 | - |
| K20c | PRACTICAL_BEST | {"2026": 88.76901843240111} | - | - | - | 0.0000 | True | [87.69, 89.85] | [87.63, 89.90] | 0.4677 | - | - |
| K21 | PRACTICAL_BEST | {"2026": -0.0003316644219135967, "2027": -0.0006845876055883438} | 0.000353 | False | [-0.001072, 0.000056] | - | - | - | - | 0.3247 | 0.2821 | - |
| K_INIT | PRACTICAL_BEST | {"2026": 0.033338209954898945, "2027": 0.03289914532034499} | 0.000439 | False | [0.031671, 0.034567] | 0.0000 | True | [0.031139, 0.035099] | [0.031009, 0.035229] | 0.2590 | 0.2821 | - |
| P5 | PRACTICAL_BEST | {"2026": 0.001831197746063995, "2027": 0.0022580160850617648} | 0.000427 | False | [0.001178, 0.002911] | 0.0000 | True | [0.001023, 0.003066] | [0.000886, 0.003203] | 0.3320 | 0.2792 | - |
| K4w | PRACTICAL_BEST | {"2026": 88.58449761134726, "2027": 88.38818842394932} | 0.20 | False | [87.73, 89.25] | 0.0000 | True | [87.45, 89.52] | [87.37, 89.61] | 0.4737 | 0.2820 | - |
| K5w | PRACTICAL_BEST | {"2026": 88.30366310284647, "2027": 88.40371462609649} | 0.10 | False | [87.61, 89.10] | 0.0000 | True | [87.33, 89.38] | [87.23, 89.48] | 0.4751 | 0.2670 | - |
| K12w | CONTROLLED_BUDGET | {"2026": 100.01388612285058, "2027": 99.99283223660885} | 0.02 | False | [99.94, 100.07] | 1.0000 | False | [99.93, 100.08] | [99.91, 100.10] | 0.1395 | 0.1448 | - |
| K13w | CONTROLLED_BUDGET | {"2026": 100.03367809596655} | - | - | - | 1.0000 | False | [99.89, 100.17] | [99.86, 100.21] | 0.1532 | - | - |
| K14w | PRACTICAL_BEST | {"2026": 99.67976324277362, "2027": 100.01799904932854} | 0.34 | True | [99.49, 100.21] | 0.0734 | False | [99.69, 100.01] | [99.65, 100.05] | 0.2845 | 0.2670 | True (90% CI [99.75, 99.96]) |

### 1e. Registered DESCRIPTIVE contrasts (06L-C C8: S8 − S7 with the bytes ratio)

| id | contrast | question | hierarchy | rule | label | seeds | seeds status | point | 95% CI | CI vs 100/0 | users | decisions | harmed share | seed spread (SEED_SENSITIVE) | Holm band | claim key | bytes ratio S8 / S7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S8-S7 | FA_WARM_FROZEN − FA_WARM | cost of not sending the item tables (paired Δ, reported with the bytes ratio; DESCRIPTIVE) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.001193 | [0.000627, 0.001802] | above | 26254 | 229742 | 0.3478 | 0.000912 (SEED_SENSITIVE) | - | ABOVE | - |
| S8-S7 | FA_WARM_FROZEN − FA_WARM | cost of not sending the item tables (paired Δ, reported with the bytes ratio; DESCRIPTIVE) | - | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.001802 | [0.001073, 0.002623] | above | 26254 | 229742 | 0.3435 | 0.000470 (no) | - | - | - |
| D_S7_C_FULL | R(FA_WARM / C_FULL) | the S7 full-model warm start vs C_FULL (DESCRIPTIVE beside the primary; 06P P2) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 87.98 | [87.31, 88.62] | below | 26254 | 229742 | 0.5122 | 0.63 (no) | - | BELOW | - |
| D_S7_C_FULL | R(FA_WARM / C_FULL) | the S7 full-model warm start vs C_FULL (DESCRIPTIVE beside the primary; 06P P2) | - | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 87.47 | [86.74, 88.11] | below | 26254 | 229742 | 0.5174 | 0.09 (no) | - | - | - |
| D_S7_PRE | FA_WARM − PRE | the S7 full-model warm start over theta_pre (DESCRIPTIVE beside the primary; 06P P2) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | -0.001701 | [-0.002283, -0.001048] | below | 26254 | 229742 | 0.3623 | 0.000559 (no) | - | BELOW | - |
| D_S7_PRE | FA_WARM − PRE | the S7 full-model warm start over theta_pre (DESCRIPTIVE beside the primary; 06P P2) | - | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | -0.002629 | [-0.003443, -0.001825] | below | 26254 | 229742 | 0.3814 | 0.000409 (no) | - | - | - |
| D_PFw_FAw | PF_WARM_FROZEN − FA_WARM_FROZEN | personalisation on the pretrained line (DESCRIPTIVE, 06L A5 caveat; 06P P4) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | -0.000241 | [-0.000448, -0.000036] | below | 26254 | 229742 | 0.2862 | 0.000533 (SEED_SENSITIVE) | - | BELOW | - |
| D_PFw_FAw | PF_WARM_FROZEN − FA_WARM_FROZEN | personalisation on the pretrained line (DESCRIPTIVE, 06L A5 caveat; 06P P4) | - | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.000067 | [0.000012, 0.000121] | above | 26254 | 229742 | 0.0781 | 0.000058 (SEED_SENSITIVE) | - | - | - |
| D_FA1024w_FAw | R(FA_1024_WARM_FROZEN / FA_WARM_FROZEN) | clipping + Poisson-cohort cost on the pretrained line (DESCRIPTIVE; 06P P4) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 100.13 | [99.86, 100.39] | includes | 26254 | 229742 | 0.3284 | 0.26 (SEED_SENSITIVE) | - | INCLUDES | - |
| D_FA1024w_FAw | R(FA_1024_WARM_FROZEN / FA_WARM_FROZEN) | clipping + Poisson-cohort cost on the pretrained line (DESCRIPTIVE; 06P P4) | - | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 100.09 | [99.81, 100.39] | includes | 26254 | 229742 | 0.3282 | 0.06 (no) | - | - | - |
| D_FTw_S7 | FT_FA_WARM − FA_WARM | local fine-tuning gain after the S7 warm FL (DESCRIPTIVE; 06P P5) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.002617 | [0.001921, 0.003335] | above | 26254 | 229742 | 0.3360 | 0.000198 (no) | - | ABOVE | - |
| D_FTw_S7 | FT_FA_WARM − FA_WARM | local fine-tuning gain after the S7 warm FL (DESCRIPTIVE; 06P P5) | - | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - | - |

## 2. Retention table (MAIN) — DESCRIPTIVE

### PRACTICAL_BEST

| arm | seed | MRR@20 | NDCG@10 | HR@10 | R% vs C_BIG [CI] | R% vs C_SMALL [CI] | bytes/visit | bytes/client | bytes/client (dropped visits charged, 06I 7b) | bytes/client (dropped visits charged down + up; DESCRIPTIVE) | bytes fleet | ε | params | model bytes | rounds | consumed EFE | still improving at 6.0 (PB mark >= 5.5; DESCRIPTIVE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_BIG | 2026 | 0.148460 | 0.180823 | 0.307510 | 100.00 [100.00, 100.00] | 105.17 [104.31, 106.07] | - | - | - | - | - | - | - | - | - | - | - |
| C_BIG | 2027 | 0.148808 | 0.180803 | 0.306315 | 100.00 [100.00, 100.00] | 106.05 [105.10, 107.05] | - | - | - | - | - | - | - | - | - | - | - |
| **C_BIG** | mean(2) |  |  |  | 100.00 [100.00, 100.00] | 105.61 [104.92, 106.35] |  |  |  |  |  |  |  |  |  |  |  |
| C_FULL | 2026 | 0.180388 | 0.217198 | 0.359396 | 121.51 [120.34, 122.64] | 127.79 [126.37, 129.09] | - | - | - | - | - | - | - | - | - | - | - |
| C_FULL | 2027 | 0.179743 | 0.216537 | 0.358480 | 120.79 [119.52, 122.00] | 128.09 [126.70, 129.54] | - | - | - | - | - | - | - | - | - | - | - |
| **C_FULL** | mean(2) |  |  |  | 121.15 [120.19, 122.07] | 127.94 [126.72, 129.05] |  |  |  |  |  |  |  |  |  |  |  |
| C_SMALL | 2026 | 0.141164 | 0.172880 | 0.298664 | 95.09 [94.27, 95.86] | 100.00 [100.00, 100.00] | - | - | - | - | - | - | - | - | - | - | - |
| C_SMALL | 2027 | 0.140321 | 0.171813 | 0.296948 | 94.30 [93.42, 95.15] | 100.00 [100.00, 100.00] | - | - | - | - | - | - | - | - | - | - | - |
| **C_SMALL** | mean(2) |  |  |  | 94.69 [94.03, 95.31] | 100.00 [100.00, 100.00] |  |  |  |  |  |  |  |  |  |  |  |
| FA | 2026 | 0.126459 | 0.156389 | 0.275321 | 85.18 [84.20, 86.10] | 89.58 [88.61, 90.48] | - | - | - | - | - | - | - | - | - | - | - |
| FA | 2027 | 0.126545 | 0.156173 | 0.274340 | 85.04 [84.06, 86.04] | 90.18 [89.28, 91.12] | - | - | - | - | - | - | - | - | - | - | - |
| **FA** | mean(2) |  |  |  | 85.11 [84.30, 85.90] | 89.88 [89.15, 90.62] |  |  |  |  |  |  |  |  |  |  |  |
| FP | 2026 | 0.126476 | 0.156385 | 0.275262 | 85.19 [84.21, 86.11] | 89.59 [88.63, 90.50] | - | - | - | - | - | - | - | - | - | - | - |
| FP | 2027 | 0.126536 | 0.156175 | 0.274367 | 85.03 [84.05, 86.04] | 90.18 [89.27, 91.11] | - | - | - | - | - | - | - | - | - | - | - |
| **FP** | mean(2) |  |  |  | 85.11 [84.30, 85.90] | 89.89 [89.16, 90.63] |  |  |  |  |  |  |  |  |  |  |  |
| PF | 2026 | 0.126461 | 0.156335 | 0.275120 | 85.18 [84.19, 86.12] | 89.58 [88.63, 90.49] | - | - | - | - | - | - | - | - | - | - | - |
| PF | 2027 | 0.126573 | 0.156225 | 0.274452 | 85.06 [84.08, 86.07] | 90.20 [89.29, 91.14] | - | - | - | - | - | - | - | - | - | - | - |
| **PF** | mean(2) |  |  |  | 85.12 [84.32, 85.90] | 89.89 [89.17, 90.64] |  |  |  |  |  |  |  |  |  |  |  |
| FA_1024 | 2026 | 0.034199 | 0.042724 | 0.078680 | 23.04 [22.27, 23.85] | 24.23 [23.41, 25.06] | - | - | - | - | - | inf | - | - | - | - | - |
| FA_1024 | 2027 | 0.034197 | 0.041880 | 0.075145 | 22.98 [22.18, 23.81] | 24.37 [23.53, 25.27] | - | - | - | - | - | inf | - | - | - | - | - |
| **FA_1024** | mean(2) |  |  |  | 23.01 [22.24, 23.76] | 24.30 [23.49, 25.09] |  |  |  |  |  |  |  |  |  |  |  |
| DP8 | 2026 | 0.034444 | 0.043055 | 0.079115 | 23.20 [22.42, 24.02] | 24.40 [23.57, 25.25] | - | - | - | - | - | 8.0000 | - | - | - | - | - |
| DP8 | 2027 | 0.032937 | 0.040897 | 0.074339 | 22.13 [21.35, 22.95] | 23.47 [22.65, 24.33] | - | - | - | - | - | 8.0000 | - | - | - | - | - |
| **DP8** | mean(2) |  |  |  | 22.67 [21.91, 23.47] | 23.94 [23.15, 24.77] |  |  |  |  |  |  |  |  |  |  |  |
| DP1 | 2026 | 0.032277 | 0.040755 | 0.076149 | 21.74 [20.96, 22.46] | 22.86 [22.05, 23.68] | - | - | - | - | - | 1.0000 | - | - | - | - | - |
| **DP1** | mean(1) |  |  |  | 21.74 [20.96, 22.46] | 22.86 [22.05, 23.68] |  |  |  |  |  |  |  |  |  |  |  |
| FA_Q8 | 2026 | 0.126703 | 0.156552 | 0.275452 | 85.34 [84.39, 86.33] | 89.76 [88.78, 90.73] | - | - | - | - | - | - | - | - | - | - | - |
| FA_Q8 | 2027 | 0.126573 | 0.156131 | 0.274086 | 85.06 [84.05, 86.05] | 90.20 [89.22, 91.17] | - | - | - | - | - | - | - | - | - | - | - |
| **FA_Q8** | mean(2) |  |  |  | 85.20 [84.36, 85.99] | 89.98 [89.23, 90.72] |  |  |  |  |  |  |  |  |  |  |  |
| FA_WARM | 2026 | 0.158148 | 0.193538 | 0.333211 | 106.53 [105.55, 107.51] | 112.03 [110.90, 113.02] | - | - | - | - | - | - | - | - | - | - | - |
| FA_WARM | 2027 | 0.158708 | 0.193972 | 0.333220 | 106.65 [105.64, 107.74] | 113.10 [111.95, 114.26] | - | - | - | - | - | - | - | - | - | - | - |
| **FA_WARM** | mean(2) |  |  |  | 106.59 [105.67, 107.43] | 112.57 [111.55, 113.47] |  |  |  |  |  |  |  |  |  |  |  |
| FA_WARM_FROZEN | 2026 | 0.159797 | 0.194925 | 0.333793 | 107.64 [106.66, 108.59] | 113.20 [112.15, 114.23] | - | - | - | - | - | - | - | - | - | - | - |
| FA_WARM_FROZEN | 2027 | 0.159444 | 0.194896 | 0.334597 | 107.15 [106.11, 108.26] | 113.63 [112.55, 114.82] | - | - | - | - | - | - | - | - | - | - | - |
| **FA_WARM_FROZEN** | mean(2) |  |  |  | 107.39 [106.50, 108.32] | 113.41 [112.47, 114.38] |  |  |  |  |  |  |  |  |  |  |  |
| DP8_FROZEN | 2026 | 0.159817 | 0.194998 | 0.333893 | 107.65 [106.61, 108.65] | 113.21 [112.15, 114.22] | - | - | - | - | - | 8.0000 | - | - | - | - | - |
| DP8_FROZEN | 2027 | 0.159801 | 0.194965 | 0.333794 | 107.39 [106.35, 108.46] | 113.88 [112.82, 115.06] | - | - | - | - | - | 8.0000 | - | - | - | - | - |
| **DP8_FROZEN** | mean(2) |  |  |  | 107.52 [106.61, 108.44] | 113.55 [112.59, 114.52] |  |  |  |  |  |  |  |  |  |  |  |
| FP_WARM_FROZEN | 2026 | 0.159796 | 0.194924 | 0.333793 | 107.64 [106.67, 108.59] | 113.20 [112.15, 114.24] | - | - | - | - | - | - | - | - | - | - | - |
| FP_WARM_FROZEN | 2027 | 0.159442 | 0.194892 | 0.334595 | 107.15 [106.10, 108.26] | 113.63 [112.55, 114.82] | - | - | - | - | - | - | - | - | - | - | - |
| **FP_WARM_FROZEN** | mean(2) |  |  |  | 107.39 [106.50, 108.32] | 113.41 [112.48, 114.37] |  |  |  |  |  |  |  |  |  |  |  |
| PF_WARM_FROZEN | 2026 | 0.159290 | 0.194563 | 0.333947 | 107.29 [106.28, 108.27] | 112.84 [111.81, 113.87] | - | - | - | - | - | - | - | - | - | - | - |
| PF_WARM_FROZEN | 2027 | 0.159470 | 0.194918 | 0.334611 | 107.16 [106.11, 108.29] | 113.65 [112.58, 114.84] | - | - | - | - | - | - | - | - | - | - | - |
| **PF_WARM_FROZEN** | mean(2) |  |  |  | 107.23 [106.33, 108.16] | 113.24 [112.30, 114.22] |  |  |  |  |  |  |  |  |  |  |  |
| FA_Q8_WARM_FROZEN | 2026 | 0.159286 | 0.194586 | 0.334036 | 107.29 [106.28, 108.26] | 112.84 [111.81, 113.85] | - | - | - | - | - | - | - | - | - | - | - |
| FA_Q8_WARM_FROZEN | 2027 | 0.159473 | 0.194917 | 0.334601 | 107.17 [106.10, 108.29] | 113.65 [112.57, 114.84] | - | - | - | - | - | - | - | - | - | - | - |
| **FA_Q8_WARM_FROZEN** | mean(2) |  |  |  | 107.23 [106.33, 108.15] | 113.24 [112.30, 114.21] |  |  |  |  |  |  |  |  |  |  |  |
| FA_1024_WARM_FROZEN | 2026 | 0.159787 | 0.194986 | 0.333909 | 107.63 [106.59, 108.62] | 113.19 [112.12, 114.21] | - | - | - | - | - | inf | - | - | - | - | - |
| FA_1024_WARM_FROZEN | 2027 | 0.159856 | 0.195003 | 0.333806 | 107.42 [106.37, 108.49] | 113.92 [112.85, 115.09] | - | - | - | - | - | inf | - | - | - | - | - |
| **FA_1024_WARM_FROZEN** | mean(2) |  |  |  | 107.53 [106.60, 108.45] | 113.56 [112.60, 114.52] |  |  |  |  |  |  |  |  |  |  |  |
| DP1_WARM_FROZEN | 2026 | 0.159880 | 0.195023 | 0.333810 | 107.69 [106.68, 108.67] | 113.26 [112.20, 114.27] | - | - | - | - | - | 1.0000 | - | - | - | - | - |
| **DP1_WARM_FROZEN** | mean(1) |  |  |  | 107.69 [106.68, 108.67] | 113.26 [112.20, 114.27] |  |  |  |  |  |  |  |  |  |  |  |
| PRE | 2026 | 0.160129 | 0.195295 | 0.334120 | 107.86 [106.85, 108.84] | 113.43 [112.39, 114.46] | - | - | - | - | - | - | - | - | - | - | - |
| **PRE** | mean(1) |  |  |  | 107.86 [106.85, 108.84] | 113.43 [112.39, 114.46] |  |  |  |  |  |  |  |  |  |  |  |
| FT_C | 2026 | 0.144046 | 0.175897 | 0.301918 | 97.03 [96.20, 97.89] | 102.04 [101.57, 102.54] | - | - | - | - | - | - | - | - | - | - | - |
| FT_C | 2027 | 0.142380 | 0.174304 | 0.300490 | 95.68 [94.77, 96.52] | 101.47 [101.02, 101.92] | - | - | - | - | - | - | - | - | - | - | - |
| **FT_C** | mean(2) |  |  |  | 96.35 [95.70, 97.02] | 101.75 [101.38, 102.14] |  |  |  |  |  |  |  |  |  |  |  |
| FT_PRE | 2026 | 0.162496 | 0.198071 | 0.338050 | 109.45 [108.40, 110.50] | 115.11 [113.93, 116.23] | - | - | - | - | - | - | - | - | - | - | - |
| FT_PRE | 2027 | 0.162328 | 0.198049 | 0.338446 | 109.09 [108.01, 110.24] | 115.68 [114.49, 116.91] | - | - | - | - | - | - | - | - | - | - | - |
| **FT_PRE** | mean(2) |  |  |  | 109.27 [108.27, 110.25] | 115.40 [114.33, 116.44] |  |  |  |  |  |  |  |  |  |  |  |
| FT_FA | 2026 | 0.130332 | 0.160117 | 0.278760 | 87.79 [86.79, 88.69] | 92.33 [91.34, 93.25] | - | - | - | - | - | - | - | - | - | - | - |
| FT_FA | 2027 | 0.129946 | 0.159850 | 0.278834 | 87.32 [86.30, 88.28] | 92.61 [91.60, 93.64] | - | - | - | - | - | - | - | - | - | - | - |
| **FT_FA** | mean(2) |  |  |  | 87.56 [86.75, 88.33] | 92.47 [91.70, 93.23] |  |  |  |  |  |  |  |  |  |  |  |
| FT_FA_WARM | 2026 | 0.160865 | 0.196737 | 0.337783 | 108.36 [107.29, 109.39] | 113.96 [112.82, 115.03] | - | - | - | - | - | - | - | - | - | - | - |
| FT_FA_WARM | 2027 | 0.161226 | 0.196984 | 0.337698 | 108.35 [107.27, 109.40] | 114.90 [113.73, 116.11] | - | - | - | - | - | - | - | - | - | - | - |
| **FT_FA_WARM** | mean(2) |  |  |  | 108.35 [107.40, 109.24] | 114.43 [113.43, 115.42] |  |  |  |  |  |  |  |  |  |  |  |
| FT_FA_WARM_FROZEN | 2026 | 0.161629 | 0.197395 | 0.338079 | 108.87 [107.80, 109.85] | 114.50 [113.37, 115.62] | - | - | - | - | - | - | - | - | - | - | - |
| FT_FA_WARM_FROZEN | 2027 | 0.161702 | 0.197490 | 0.338191 | 108.67 [107.54, 109.78] | 115.24 [114.02, 116.47] | - | - | - | - | - | - | - | - | - | - | - |
| **FT_FA_WARM_FROZEN** | mean(2) |  |  |  | 108.77 [107.80, 109.67] | 114.87 [113.86, 115.92] |  |  |  |  |  |  |  |  |  |  |  |
| LO | 2026 | 0.002964 | 0.003424 | 0.005194 | 2.00 [1.76, 2.27] | 2.10 [1.85, 2.39] | - | - | - | - | - | - | - | - | - | - | - |
| LO | 2027 | 0.003284 | 0.003819 | 0.005970 | 2.21 [1.96, 2.49] | 2.34 [2.07, 2.65] | - | - | - | - | - | - | - | - | - | - | - |
| **LO** | mean(2) |  |  |  | 2.10 [1.88, 2.34] | 2.22 [1.99, 2.48] |  |  |  |  |  |  |  |  |  |  |  |
| C_SMALL_FP32 | 2026 | 0.140405 | 0.172255 | 0.298570 | 95.21 [94.01, 96.34] | 100.00 [100.00, 100.00] | - | - | - | - | - | - | - | - | - | - | - |
| **C_SMALL_FP32** | mean(1) |  |  |  | 95.21 [94.01, 96.34] DEVICE_MIXED | 100.00 [100.00, 100.00] |  |  |  |  |  |  |  |  |  |  |  |
| C_SMALL_INT8 | 2026 | 0.140379 | 0.172078 | 0.298041 | 95.20 [94.02, 96.35] | 99.98 [99.75, 100.23] | - | - | - | - | - | - | - | - | - | - | - |
| **C_SMALL_INT8** | mean(1) |  |  |  | 95.20 [94.02, 96.35] DEVICE_MIXED | 99.98 [99.75, 100.23] |  |  |  |  |  |  |  |  |  |  |  |
| KNN_EMB_SMALL | None | 0.067317 | 0.081258 | 0.137640 | 45.34 [44.30, 46.44]; 45.24 [44.22, 46.35] | 47.69 [46.52, 48.85]; 47.97 [46.77, 49.15] | - | - | - | - | - | - | - | - | - | - | - |
| **KNN_EMB_SMALL** | mean(2) |  |  |  | 45.29 [44.28, 46.37] | 47.83 [46.66, 48.99] |  |  |  |  |  |  |  |  |  |  |  |
| KNN_EMB_BIG | None | 0.078724 | 0.094152 | 0.154174 | 53.03 [51.92, 54.22]; 52.90 [51.80, 54.06] | 55.77 [54.56, 57.01]; 56.10 [54.85, 57.46] | - | - | - | - | - | - | - | - | - | - | - |
| **KNN_EMB_BIG** | mean(2) |  |  |  | 52.96 [51.89, 54.13] | 55.93 [54.74, 57.21] |  |  |  |  |  |  |  |  |  |  |  |
| RULE | None | 0.034620 | 0.043728 | 0.084285 | 23.32 [22.59, 24.06]; 23.27 [22.53, 24.01] | 24.52 [23.72, 25.34]; 24.67 [23.88, 25.45] | - | - | - | - | - | - | - | - | - | - | - |
| **RULE** | mean(2) |  |  |  | 23.29 [22.57, 24.02] | 24.60 [23.80, 25.39] |  |  |  |  |  |  |  |  |  |  |  |
| SKNN | None | 0.110911 | 0.146952 | 0.283162 | 74.71 [73.81, 75.60]; 74.53 [73.62, 75.48] | 78.57 [77.61, 79.53]; 79.04 [77.99, 80.06] | - | - | - | - | - | - | - | - | - | - | - |
| **SKNN** | mean(2) |  |  |  | 74.62 [73.80, 75.52] | 78.80 [77.89, 79.77] |  |  |  |  |  |  |  |  |  |  |  |
| LAST | None | 0.075569 | 0.105851 | 0.207364 | 50.90 [50.00, 51.93]; 50.78 [49.91, 51.82] | 53.53 [52.52, 54.63]; 53.85 [52.87, 55.00] | - | - | - | - | - | - | - | - | - | - | - |
| **LAST** | mean(2) |  |  |  | 50.84 [49.97, 51.83] | 53.69 [52.74, 54.80] |  |  |  |  |  |  |  |  |  |  |  |
| POP | None | 0.018796 | 0.024216 | 0.048081 | 12.66 [12.06, 13.27]; 12.63 [12.04, 13.24] | 13.32 [12.69, 13.98]; 13.40 [12.74, 14.05] | - | - | - | - | - | - | - | - | - | - | - |
| **POP** | mean(2) |  |  |  | 12.65 [12.05, 13.26] | 13.36 [12.72, 14.01] |  |  |  |  |  |  |  |  |  |  |  |

| arm | R% NDCG@10 vs C_BIG | R% HR@10 vs C_BIG |
|---|---|---|
| C_BIG | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| C_FULL | 119.94 [119.12, 120.75] | 116.95 [116.20, 117.70] |
| C_SMALL | 95.32 [94.76, 95.85] | 97.03 [96.47, 97.56] |
| FA | 86.43 [85.75, 87.09] | 89.55 [88.91, 90.15] |
| FP | 86.43 [85.74, 87.08] | 89.54 [88.90, 90.14] |
| PF | 86.43 [85.74, 87.09] | 89.53 [88.89, 90.12] |
| FA_1024 | 23.40 [22.67, 24.10] | 25.06 [24.34, 25.79] |
| DP8 | 23.22 [22.50, 23.92] | 25.00 [24.27, 25.71] |
| DP1 | 22.54 [21.80, 23.25] | 24.76 [24.02, 25.51] |
| FA_Q8 | 86.47 [85.78, 87.13] | 89.53 [88.88, 90.13] |
| FA_WARM | 107.16 [106.37, 107.90] | 108.57 [107.83, 109.27] |
| FA_WARM_FROZEN | 107.80 [107.06, 108.56] | 108.89 [108.16, 109.58] |
| DP8_FROZEN | 107.84 [107.07, 108.56] | 108.78 [108.00, 109.51] |
| FP_WARM_FROZEN | 107.80 [107.06, 108.56] | 108.89 [108.15, 109.58] |
| PF_WARM_FROZEN | 107.70 [106.96, 108.46] | 108.92 [108.19, 109.60] |
| FA_Q8_WARM_FROZEN | 107.71 [106.97, 108.47] | 108.93 [108.21, 109.62] |
| FA_1024_WARM_FROZEN | 107.84 [107.08, 108.57] | 108.78 [108.00, 109.51] |
| DP1_WARM_FROZEN | 107.85 [107.00, 108.68] | 108.55 [107.75, 109.40] |
| PRE | 108.00 [107.16, 108.81] | 108.65 [107.84, 109.49] |
| FT_C | 96.84 [96.30, 97.36] | 98.14 [97.60, 98.66] |
| FT_PRE | 109.54 [108.75, 110.31] | 110.21 [109.46, 110.93] |
| FT_FA | 88.48 [87.83, 89.14] | 90.84 [90.21, 91.42] |
| FT_FA_WARM | 108.88 [108.08, 109.59] | 110.04 [109.30, 110.74] |
| FT_FA_WARM_FROZEN | 109.20 [108.40, 109.98] | 110.17 [109.44, 110.90] |
| LO | 2.00 [1.80, 2.23] | 1.82 [1.65, 2.01] |
| C_SMALL_FP32 | 95.72 [94.76, 96.67] | 97.18 [96.32, 98.09] |
| C_SMALL_INT8 | 95.62 [94.67, 96.58] | 97.01 [96.13, 97.95] |
| KNN_EMB_SMALL | 44.94 [44.03, 45.91] | 44.85 [44.02, 45.73] |
| KNN_EMB_BIG | 52.07 [51.12, 53.12] | 50.23 [49.38, 51.24] |
| RULE | 24.18 [23.46, 24.87] | 27.46 [26.71, 28.20] |
| SKNN | 81.27 [80.49, 82.02] | 92.26 [91.42, 92.99] |
| LAST | 58.54 [57.56, 59.59] | 67.56 [66.46, 68.71] |
| POP | 13.39 [12.82, 13.97] | 15.67 [15.07, 16.25] |

### CONTROLLED_BUDGET

| arm | seed | MRR@20 | NDCG@10 | HR@10 | R% vs C_BIG [CI] | R% vs C_SMALL [CI] | bytes/visit | bytes/client | bytes/client (dropped visits charged, 06I 7b) | bytes/client (dropped visits charged down + up; DESCRIPTIVE) | bytes fleet | ε | params | model bytes | rounds | consumed EFE | still improving at 6.0 (PB mark >= 5.5; DESCRIPTIVE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_BIG | 2026 | 0.138963 | 0.169829 | 0.290656 | 100.00 [100.00, 100.00] | 100.05 [99.06, 101.08] | - | - | - | - | - | - | - | - | - | - | - |
| C_BIG | 2027 | 0.138422 | 0.169380 | 0.289767 | 100.00 [100.00, 100.00] | 100.42 [99.34, 101.53] | - | - | - | - | - | - | - | - | - | - | - |
| **C_BIG** | mean(2) |  |  |  | 100.00 [100.00, 100.00] | 100.24 [99.44, 101.12] |  |  |  |  |  |  |  |  |  |  |  |
| C_FULL | 2026 | 0.180388 | 0.217198 | 0.359396 | 129.81 [128.49, 131.09] | 129.87 [128.43, 131.23] | - | - | - | - | - | - | - | - | - | - | - |
| C_FULL | 2027 | 0.179743 | 0.216537 | 0.358480 | 129.85 [128.42, 131.28] | 130.40 [128.88, 131.94] | - | - | - | - | - | - | - | - | - | - | - |
| **C_FULL** | mean(2) |  |  |  | 129.83 [128.68, 130.94] | 130.14 [128.83, 131.43] |  |  |  |  |  |  |  |  |  |  |  |
| C_SMALL | 2026 | 0.138894 | 0.169979 | 0.292584 | 99.95 [98.93, 100.95] | 100.00 [100.00, 100.00] | - | - | - | - | - | - | - | - | - | - | - |
| C_SMALL | 2027 | 0.137840 | 0.169319 | 0.292773 | 99.58 [98.49, 100.67] | 100.00 [100.00, 100.00] | - | - | - | - | - | - | - | - | - | - | - |
| **C_SMALL** | mean(2) |  |  |  | 99.77 [98.90, 100.57] | 100.00 [100.00, 100.00] |  |  |  |  |  |  |  |  |  |  |  |
| FA | 2026 | 0.126459 | 0.156389 | 0.275321 | 91.00 [89.85, 92.04] | 91.05 [90.02, 92.02] | - | - | - | - | - | - | - | - | - | - | - |
| FA | 2027 | 0.126545 | 0.156173 | 0.274340 | 91.42 [90.24, 92.58] | 91.81 [90.80, 92.81] | - | - | - | - | - | - | - | - | - | - | - |
| **FA** | mean(2) |  |  |  | 91.21 [90.23, 92.15] | 91.43 [90.65, 92.20] |  |  |  |  |  |  |  |  |  |  |  |
| FP | 2026 | 0.126476 | 0.156385 | 0.275262 | 91.01 [89.85, 92.04] | 91.06 [90.03, 92.03] | - | - | - | - | - | - | - | - | - | - | - |
| FP | 2027 | 0.126536 | 0.156175 | 0.274367 | 91.41 [90.23, 92.57] | 91.80 [90.79, 92.80] | - | - | - | - | - | - | - | - | - | - | - |
| **FP** | mean(2) |  |  |  | 91.21 [90.23, 92.15] | 91.43 [90.65, 92.20] |  |  |  |  |  |  |  |  |  |  |  |
| PF | 2026 | 0.126461 | 0.156335 | 0.275120 | 91.00 [89.84, 92.07] | 91.05 [90.04, 92.02] | - | - | - | - | - | - | - | - | - | - | - |
| PF | 2027 | 0.126573 | 0.156225 | 0.274452 | 91.44 [90.26, 92.60] | 91.83 [90.82, 92.83] | - | - | - | - | - | - | - | - | - | - | - |
| **PF** | mean(2) |  |  |  | 91.22 [90.25, 92.15] | 91.44 [90.66, 92.22] |  |  |  |  |  |  |  |  |  |  |  |
| FA_1024 | 2026 | 0.034199 | 0.042724 | 0.078680 | 24.61 [23.77, 25.49] | 24.62 [23.78, 25.46] | - | - | - | - | - | inf | - | - | - | - | - |
| FA_1024 | 2027 | 0.034197 | 0.041880 | 0.075145 | 24.70 [23.84, 25.61] | 24.81 [23.97, 25.71] | - | - | - | - | - | inf | - | - | - | - | - |
| **FA_1024** | mean(2) |  |  |  | 24.66 [23.82, 25.50] | 24.72 [23.89, 25.56] |  |  |  |  |  |  |  |  |  |  |  |
| DP8 | 2026 | 0.034444 | 0.043055 | 0.079115 | 24.79 [23.93, 25.65] | 24.80 [23.93, 25.67] | - | - | - | - | - | 8.0000 | - | - | - | - | - |
| DP8 | 2027 | 0.032937 | 0.040897 | 0.074339 | 23.79 [22.95, 24.68] | 23.89 [23.04, 24.78] | - | - | - | - | - | 8.0000 | - | - | - | - | - |
| **DP8** | mean(2) |  |  |  | 24.29 [23.47, 25.13] | 24.35 [23.54, 25.17] |  |  |  |  |  |  |  |  |  |  |  |
| DP1 | 2026 | 0.032277 | 0.040755 | 0.076149 | 23.23 [22.39, 24.00] | 23.24 [22.39, 24.05] | - | - | - | - | - | 1.0000 | - | - | - | - | - |
| **DP1** | mean(1) |  |  |  | 23.23 [22.39, 24.00] | 23.24 [22.39, 24.05] |  |  |  |  |  |  |  |  |  |  |  |
| FA_Q8 | 2026 | 0.126703 | 0.156552 | 0.275452 | 91.18 [90.02, 92.34] | 91.22 [90.14, 92.18] | - | - | - | - | - | - | - | - | - | - | - |
| FA_Q8 | 2027 | 0.126573 | 0.156131 | 0.274086 | 91.44 [90.21, 92.62] | 91.83 [90.76, 92.85] | - | - | - | - | - | - | - | - | - | - | - |
| **FA_Q8** | mean(2) |  |  |  | 91.31 [90.35, 92.26] | 91.52 [90.73, 92.33] |  |  |  |  |  |  |  |  |  |  |  |
| FA_WARM | 2026 | 0.157705 | 0.192742 | 0.331615 | 113.49 [112.25, 114.69] | 113.54 [112.41, 114.63] | - | - | - | - | - | - | - | - | - | - | - |
| FA_WARM | 2027 | 0.157295 | 0.192300 | 0.331132 | 113.63 [112.33, 114.86] | 114.11 [112.96, 115.30] | - | - | - | - | - | - | - | - | - | - | - |
| **FA_WARM** | mean(2) |  |  |  | 113.56 [112.43, 114.63] | 113.83 [112.82, 114.82] |  |  |  |  |  |  |  |  |  |  |  |
| FA_WARM_FROZEN | 2026 | 0.159272 | 0.194595 | 0.334106 | 114.61 [113.38, 115.88] | 114.67 [113.58, 115.75] | - | - | - | - | - | - | - | - | - | - | - |
| FA_WARM_FROZEN | 2027 | 0.159333 | 0.194558 | 0.333799 | 115.11 [113.77, 116.44] | 115.59 [114.37, 116.84] | - | - | - | - | - | - | - | - | - | - | - |
| **FA_WARM_FROZEN** | mean(2) |  |  |  | 114.86 [113.69, 116.01] | 115.13 [114.07, 116.22] |  |  |  |  |  |  |  |  |  |  |  |
| DP8_FROZEN | 2026 | 0.159493 | 0.194701 | 0.333604 | 114.77 [113.51, 116.04] | 114.83 [113.73, 115.94] | - | - | - | - | - | 8.0000 | - | - | - | - | - |
| DP8_FROZEN | 2027 | 0.159417 | 0.194657 | 0.333648 | 115.17 [113.86, 116.46] | 115.65 [114.44, 116.86] | - | - | - | - | - | 8.0000 | - | - | - | - | - |
| **DP8_FROZEN** | mean(2) |  |  |  | 114.97 [113.83, 116.14] | 115.24 [114.15, 116.33] |  |  |  |  |  |  |  |  |  |  |  |
| FP_WARM_FROZEN | 2026 | 0.159291 | 0.194611 | 0.334109 | 114.63 [113.40, 115.88] | 114.69 [113.59, 115.78] | - | - | - | - | - | - | - | - | - | - | - |
| FP_WARM_FROZEN | 2027 | 0.159330 | 0.194555 | 0.333791 | 115.10 [113.77, 116.44] | 115.59 [114.37, 116.84] | - | - | - | - | - | - | - | - | - | - | - |
| **FP_WARM_FROZEN** | mean(2) |  |  |  | 114.87 [113.70, 116.01] | 115.14 [114.07, 116.22] |  |  |  |  |  |  |  |  |  |  |  |
| PF_WARM_FROZEN | 2026 | 0.159368 | 0.194710 | 0.334245 | 114.68 [113.45, 115.93] | 114.74 [113.65, 115.82] | - | - | - | - | - | - | - | - | - | - | - |
| PF_WARM_FROZEN | 2027 | 0.159371 | 0.194580 | 0.333763 | 115.13 [113.81, 116.47] | 115.62 [114.41, 116.85] | - | - | - | - | - | - | - | - | - | - | - |
| **PF_WARM_FROZEN** | mean(2) |  |  |  | 114.91 [113.74, 116.06] | 115.18 [114.14, 116.27] |  |  |  |  |  |  |  |  |  |  |  |
| FA_Q8_WARM_FROZEN | 2026 | 0.159330 | 0.194640 | 0.334116 | 114.66 [113.43, 115.90] | 114.71 [113.61, 115.81] | - | - | - | - | - | - | - | - | - | - | - |
| FA_Q8_WARM_FROZEN | 2027 | 0.159335 | 0.194571 | 0.333830 | 115.11 [113.77, 116.43] | 115.59 [114.37, 116.84] | - | - | - | - | - | - | - | - | - | - | - |
| **FA_Q8_WARM_FROZEN** | mean(2) |  |  |  | 114.88 [113.72, 116.02] | 115.15 [114.09, 116.23] |  |  |  |  |  |  |  |  |  |  |  |
| FA_1024_WARM_FROZEN | 2026 | 0.159471 | 0.194642 | 0.333490 | 114.76 [113.50, 116.04] | 114.81 [113.70, 115.91] | - | - | - | - | - | inf | - | - | - | - | - |
| FA_1024_WARM_FROZEN | 2027 | 0.159429 | 0.194642 | 0.333589 | 115.18 [113.90, 116.46] | 115.66 [114.45, 116.86] | - | - | - | - | - | inf | - | - | - | - | - |
| **FA_1024_WARM_FROZEN** | mean(2) |  |  |  | 114.97 [113.83, 116.10] | 115.24 [114.16, 116.32] |  |  |  |  |  |  |  |  |  |  |  |
| DP1_WARM_FROZEN | 2026 | 0.159525 | 0.194765 | 0.333776 | 114.80 [113.55, 116.05] | 114.85 [113.77, 115.94] | - | - | - | - | - | 1.0000 | - | - | - | - | - |
| **DP1_WARM_FROZEN** | mean(1) |  |  |  | 114.80 [113.55, 116.05] | 114.85 [113.77, 115.94] |  |  |  |  |  |  |  |  |  |  |  |
| PRE | 2026 | 0.160129 | 0.195295 | 0.334120 | 115.23 [113.98, 116.47] | 115.29 [114.18, 116.40] | - | - | - | - | - | - | - | - | - | - | - |
| **PRE** | mean(1) |  |  |  | 115.23 [113.98, 116.47] | 115.29 [114.18, 116.40] |  |  |  |  |  |  |  |  |  |  |  |
| KNN_EMB_SMALL | None | 0.067317 | 0.081258 | 0.137640 | 48.44 [47.29, 49.61]; 48.63 [47.52, 49.87] | 48.47 [47.26, 49.66]; 48.84 [47.63, 50.13] | - | - | - | - | - | - | - | - | - | - | - |
| **KNN_EMB_SMALL** | mean(2) |  |  |  | 48.54 [47.42, 49.73] | 48.65 [47.47, 49.85] |  |  |  |  |  |  |  |  |  |  |  |
| KNN_EMB_BIG | None | 0.078724 | 0.094152 | 0.154174 | 56.65 [55.46, 57.98]; 56.87 [55.69, 58.19] | 56.68 [55.43, 57.98]; 57.11 [55.82, 58.52] | - | - | - | - | - | - | - | - | - | - | - |
| **KNN_EMB_BIG** | mean(2) |  |  |  | 56.76 [55.62, 58.02] | 56.90 [55.67, 58.19] |  |  |  |  |  |  |  |  |  |  |  |
| RULE | None | 0.034620 | 0.043728 | 0.084285 | 24.91 [24.13, 25.72]; 25.01 [24.22, 25.83] | 24.93 [24.13, 25.72]; 25.12 [24.32, 25.94] | - | - | - | - | - | - | - | - | - | - | - |
| **RULE** | mean(2) |  |  |  | 24.96 [24.19, 25.76] | 25.02 [24.23, 25.81] |  |  |  |  |  |  |  |  |  |  |  |
| SKNN | None | 0.110911 | 0.146952 | 0.283162 | 79.81 [78.76, 80.84]; 80.13 [79.07, 81.25] | 79.85 [78.83, 80.90]; 80.46 [79.36, 81.54] | - | - | - | - | - | - | - | - | - | - | - |
| **SKNN** | mean(2) |  |  |  | 79.97 [79.06, 80.97] | 80.16 [79.19, 81.18] |  |  |  |  |  |  |  |  |  |  |  |
| LAST | None | 0.075569 | 0.105851 | 0.207364 | 54.38 [53.42, 55.45]; 54.59 [53.63, 55.70] | 54.41 [53.40, 55.60]; 54.82 [53.75, 55.96] | - | - | - | - | - | - | - | - | - | - | - |
| **LAST** | mean(2) |  |  |  | 54.49 [53.56, 55.55] | 54.62 [53.61, 55.75] |  |  |  |  |  |  |  |  |  |  |  |
| POP | None | 0.018796 | 0.024216 | 0.048081 | 13.53 [12.88, 14.17]; 13.58 [12.93, 14.25] | 13.53 [12.88, 14.20]; 13.64 [12.98, 14.29] | - | - | - | - | - | - | - | - | - | - | - |
| **POP** | mean(2) |  |  |  | 13.55 [12.91, 14.21] | 13.58 [12.93, 14.25] |  |  |  |  |  |  |  |  |  |  |  |

| arm | R% NDCG@10 vs C_BIG | R% HR@10 vs C_BIG |
|---|---|---|
| C_BIG | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| C_FULL | 127.87 [126.92, 128.81] | 123.68 [122.89, 124.57] |
| C_SMALL | 100.03 [99.39, 100.67] | 100.85 [100.24, 101.48] |
| FA | 92.14 [91.35, 92.93] | 94.70 [93.99, 95.43] |
| FP | 92.14 [91.35, 92.92] | 94.69 [93.98, 95.43] |
| PF | 92.14 [91.35, 92.95] | 94.68 [93.95, 95.43] |
| FA_1024 | 24.94 [24.18, 25.75] | 26.50 [25.74, 27.27] |
| DP8 | 24.75 [23.99, 25.54] | 26.44 [25.69, 27.22] |
| DP1 | 24.00 [23.20, 24.76] | 26.20 [25.42, 27.00] |
| FA_Q8 | 92.18 [91.40, 92.96] | 94.68 [93.97, 95.42] |
| FA_WARM | 113.51 [112.62, 114.39] | 114.18 [113.38, 115.02] |
| FA_WARM_FROZEN | 114.72 [113.78, 115.62] | 115.07 [114.26, 115.91] |
| DP8_FROZEN | 114.78 [113.84, 115.71] | 114.96 [114.12, 115.80] |
| FP_WARM_FROZEN | 114.73 [113.79, 115.63] | 115.07 [114.26, 115.91] |
| PF_WARM_FROZEN | 114.76 [113.83, 115.67] | 115.09 [114.26, 115.93] |
| FA_Q8_WARM_FROZEN | 114.74 [113.79, 115.64] | 115.08 [114.25, 115.92] |
| FA_1024_WARM_FROZEN | 114.76 [113.83, 115.68] | 114.93 [114.07, 115.78] |
| DP1_WARM_FROZEN | 114.68 [113.66, 115.67] | 114.84 [113.87, 115.79] |
| PRE | 114.99 [113.99, 115.95] | 114.95 [113.97, 115.90] |
| KNN_EMB_SMALL | 47.91 [46.94, 48.97] | 47.43 [46.56, 48.41] |
| KNN_EMB_BIG | 55.51 [54.48, 56.65] | 53.12 [52.16, 54.18] |
| RULE | 25.78 [25.01, 26.53] | 29.04 [28.27, 29.82] |
| SKNN | 86.64 [85.75, 87.51] | 97.57 [96.62, 98.45] |
| LAST | 62.41 [61.39, 63.53] | 71.45 [70.31, 72.67] |
| POP | 14.28 [13.68, 14.89] | 16.57 [15.95, 17.18] |

## 3. Personalisation gains (paired) — DESCRIPTIVE

| contrast | rule | seeds | delta MRR@20 | 95% CI | CI vs 0 | harmed-client share |
|---|---|---|---|---|---|---|
| FT_C - C_SMALL | PRACTICAL_BEST | [2026, 2027] | 0.002470 | [0.001955, 0.002993] | above | 0.3348 |
| FT_C - C_SMALL | CONTROLLED_BUDGET | NOT_AVAILABLE | - | - | - | - |
| FT_FA - FA | PRACTICAL_BEST | [2026, 2027] | 0.003637 | [0.003105, 0.004221] | above | 0.3230 |
| FT_FA - FA | CONTROLLED_BUDGET | NOT_AVAILABLE | - | - | - | - |
| PF - FA | PRACTICAL_BEST | [2026, 2027] | 0.000015 | [-0.000040, 0.000071] | includes | 0.0737 |
| PF - FA | CONTROLLED_BUDGET | [2026, 2027] | 0.000015 | [-0.000040, 0.000071] | includes | 0.0737 |

## 4. Privacy-level ladder (06B §3) — DESCRIPTIVE

| level | arm | R% vs C_BIG (PB, seed mean) | bytes/client | leaves device (training) | leaves device (inference) | guarantee |
|---|---|---|---|---|---|---|
| L0 cloud-train / cloud-infer | C_BIG | 100.00 [100.00, 100.00] | - | raw events | query context | none |
| L0 cloud-train / cloud-infer | C_FULL | 121.15 [120.19, 122.07] | - | raw events | query context | none |
| L0 cloud-train / cloud-infer | C_SMALL | 94.69 [94.03, 95.31] | - | raw events | query context | none |
| L1 cloud-train / device-infer | C_SMALL_FP32 | 95.21 [94.01, 96.34] | - | raw events (model trained centrally once) | nothing | none |
| L1 cloud-train / device-infer | C_SMALL_INT8 | 95.20 [94.02, 96.35] | - | raw events (model trained centrally once) | nothing | none |
| L1 cloud-train / device-infer | KNN_EMB_SMALL | 45.29 [44.28, 46.37] | - | raw events (model trained centrally once) | nothing | none |
| L1 cloud-train / device-infer | KNN_EMB_BIG | 52.96 [51.89, 54.13] | - | raw events (model trained centrally once) | nothing | none |
| L2 federated | FA | 85.11 [84.30, 85.90] | - | model deltas (dense FP32; 8-bit for FA_Q8) | nothing | none (server sees clipped/unclipped deltas) |
| L2 federated | FP | 85.11 [84.30, 85.90] | - | model deltas (dense FP32; 8-bit for FA_Q8) | nothing | none (server sees clipped/unclipped deltas) |
| L2 federated | FA_1024 | 23.01 [22.24, 23.76] | - | model deltas (dense FP32; 8-bit for FA_Q8) | nothing | none (server sees clipped/unclipped deltas) |
| L2 federated | FA_Q8 | 85.20 [84.36, 85.99] | - | model deltas (dense FP32; 8-bit for FA_Q8) | nothing | none (server sees clipped/unclipped deltas) |
| L2 federated | PF | 85.12 [84.32, 85.90] | - | model deltas (dense FP32; 8-bit for FA_Q8) | nothing | none (server sees clipped/unclipped deltas) |
| L3 federated + DP | DP8 | 22.67 [21.91, 23.47] | - | clipped deltas + Gaussian noise on the sum | nothing | user-level (ε = 8.0, δ = 5e-6)-DP of the released model |
| L3 federated + DP | DP1 | 21.74 [20.96, 22.46] | - | clipped deltas + Gaussian noise on the sum | nothing | user-level (ε = 1.0, δ = 5e-6)-DP of the released model |
| L4 federated + local personalisation | PF | 85.12 [84.32, 85.90] | - | as L2; personal parameters never leave | nothing | none beyond L2 |
| L4 federated + local personalisation | FT_FA | 87.56 [86.75, 88.33] | - | as L2; personal parameters never leave | nothing | none beyond L2 |
| L5 cloud model + local personalisation | FT_C | 96.35 [95.70, 97.02] | - | nothing after the download | nothing | none (download only) |
| L6 device-only | LO | 2.10 [1.88, 2.34] | - | nothing | nothing | data never leaves (shared catalogue/θ0 only) |
| L6 device-only | RULE | 23.29 [22.57, 24.02] | - | nothing | nothing | data never leaves (shared catalogue/θ0 only) |

Disclosed (CLIENT_CONTRACT C5.6): every arm shares the pooled bootstrap block (catalogue, transform statistics, θ0).

## 5. Curves — DESCRIPTIVE (PRACTICAL_BEST; full series in curves/*.csv)

### 5a. R vs ε

| arm | ε (registered) | z | rule | R% vs C_BIG | R% vs FA_1024 | ε fields (DP_WOR receipt) |
|---|---|---|---|---|---|---|
| FA_1024 | inf | - | PRACTICAL_BEST | 23.01 [22.24, 23.76] | - - | "-" |
| FA_1024 | inf | - | CONTROLLED_BUDGET | 24.66 [23.82, 25.50] | - - | "-" |
| DP8 | 8.0000 | - | PRACTICAL_BEST | 22.67 [21.91, 23.47] | 98.52 [97.70, 99.38] | "-" |
| DP8 | 8.0000 | - | CONTROLLED_BUDGET | 24.29 [23.47, 25.13] | 98.52 [97.70, 99.38] | "-" |
| DP1 | 1.0000 | - | PRACTICAL_BEST | 21.74 [20.96, 22.46] | 94.38 [92.53, 96.00] | "-" |
| DP1 | 1.0000 | - | CONTROLLED_BUDGET | 23.23 [22.39, 24.00] | 94.38 [92.53, 96.00] | "-" |

### 5b. R vs model size

| arm | params | model bytes | R% PB | R% CB | note |
|---|---|---|---|---|---|
| C_FULL | - | - | 121.15 [120.19, 122.07] | 129.83 [128.68, 130.94] |  |
| C_BIG | - | - | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |  |
| C_SMALL | - | - | 94.69 [94.03, 95.31] | 99.77 [98.90, 100.57] |  |
| C_SMALL_FP32 | - | - | 95.21 [94.01, 96.34] | - - | INT8 / ONNX FP32 on its users_subset when given (06A §6) |
| C_SMALL_INT8 | - | - | 95.20 [94.02, 96.35] | - - | INT8 / ONNX FP32 on its users_subset when given (06A §6) |

### 5c. R vs cumulative bytes per client (FL marks)

| arm | seed | mark | round | cum bytes/client (prorated; registered series) | cum bytes/client (dropped visits charged their download; second series) | cum bytes/client (dropped visits charged down + up; DESCRIPTIVE) | MRR@20 | R% | 95% CI |
|---|---|---|---|---|---|---|---|---|---|

06I §7(b) (variant registered in DECISION_LOG row 218): bytes per client are reported with the registered definition (06A §3: visits x (down + up), dropped clients not charged) and, beside it, with the dropped visits charged their download (a dropped client receives the model and never uploads): bytes_per_client_charged_dropped = visits x (down + up) + dropped visits x down per client (client mean). The variant that charges a dropped visit down + up (bytes_per_client_charged_dropped_up_down) is DESCRIPTIVE only. The curves carry the registered charged variant as a second series (the up + down variant as a DESCRIPTIVE third series).

### 5d. R per stratum (MAIN)

| family | group | arm | users | R% vs C_BIG | 95% CI |
|---|---|---|---|---|---|
| category_diversity | T0 | C_BIG | 1284 | 100.00 | [100.00, 100.00] |
| category_diversity | T0 | C_FULL | 1284 | 127.05 | [121.25, 133.58] |
| category_diversity | T0 | C_SMALL | 1284 | 91.12 | [87.71, 94.69] |
| category_diversity | T0 | FA | 1284 | 75.68 | [71.20, 80.06] |
| category_diversity | T0 | FP | 1284 | 75.68 | [71.20, 80.07] |
| category_diversity | T0 | PF | 1284 | 75.69 | [71.19, 80.02] |
| category_diversity | T0 | FA_1024 | 1284 | 15.96 | [13.07, 18.92] |
| category_diversity | T0 | DP8 | 1284 | 15.81 | [12.99, 18.70] |
| category_diversity | T0 | DP1 | 1284 | 15.83 | [13.26, 18.76] |
| category_diversity | T0 | FA_Q8 | 1284 | 76.19 | [71.70, 80.50] |
| category_diversity | T0 | FA_WARM | 1284 | 108.30 | [103.22, 113.70] |
| category_diversity | T0 | FA_WARM_FROZEN | 1284 | 110.04 | [104.48, 115.61] |
| category_diversity | T0 | DP8_FROZEN | 1284 | 110.13 | [104.38, 115.77] |
| category_diversity | T0 | FP_WARM_FROZEN | 1284 | 110.04 | [104.48, 115.61] |
| category_diversity | T0 | PF_WARM_FROZEN | 1284 | 109.77 | [104.35, 115.32] |
| category_diversity | T0 | FA_Q8_WARM_FROZEN | 1284 | 109.73 | [104.35, 115.29] |
| category_diversity | T0 | FA_1024_WARM_FROZEN | 1284 | 110.33 | [104.57, 115.87] |
| category_diversity | T0 | DP1_WARM_FROZEN | 1284 | 109.59 | [103.81, 116.33] |
| category_diversity | T0 | PRE | 1284 | 109.74 | [104.03, 116.36] |
| category_diversity | T0 | FT_C | 1284 | 94.73 | [91.22, 98.46] |
| category_diversity | T0 | FT_PRE | 1284 | 113.75 | [108.21, 119.64] |
| category_diversity | T0 | FT_FA | 1284 | 77.65 | [73.10, 82.08] |
| category_diversity | T0 | FT_FA_WARM | 1284 | 112.01 | [106.59, 117.71] |
| category_diversity | T0 | FT_FA_WARM_FROZEN | 1284 | 114.04 | [108.30, 119.95] |
| category_diversity | T0 | LO | 1284 | 1.30 | [0.55, 2.23] |
| category_diversity | T0 | KNN_EMB_SMALL | 1284 | 49.26 | [43.92, 55.05] |
| category_diversity | T0 | KNN_EMB_BIG | 1284 | 64.02 | [57.86, 71.84] |
| category_diversity | T0 | RULE | 1284 | 22.74 | [19.14, 26.83] |
| category_diversity | T0 | SKNN | 1284 | 83.39 | [78.61, 88.41] |
| category_diversity | T0 | LAST | 1284 | 61.23 | [56.41, 66.83] |
| category_diversity | T0 | POP | 1284 | 7.96 | [5.69, 10.72] |
| category_diversity | T1 | C_BIG | 8632 | 100.00 | [100.00, 100.00] |
| category_diversity | T1 | C_FULL | 8632 | 119.72 | [117.96, 121.53] |
| category_diversity | T1 | C_SMALL | 8632 | 94.36 | [93.17, 95.54] |
| category_diversity | T1 | FA | 8632 | 85.88 | [84.55, 87.19] |
| category_diversity | T1 | FP | 8632 | 85.88 | [84.54, 87.18] |
| category_diversity | T1 | PF | 8632 | 85.90 | [84.57, 87.21] |
| category_diversity | T1 | FA_1024 | 8632 | 26.18 | [24.71, 27.65] |
| category_diversity | T1 | DP8 | 8632 | 25.64 | [24.30, 27.08] |
| category_diversity | T1 | DP1 | 8632 | 24.14 | [22.83, 25.55] |
| category_diversity | T1 | FA_Q8 | 8632 | 85.98 | [84.63, 87.35] |
| category_diversity | T1 | FA_WARM | 8632 | 104.39 | [102.93, 105.90] |
| category_diversity | T1 | FA_WARM_FROZEN | 8632 | 105.30 | [103.79, 106.86] |
| category_diversity | T1 | DP8_FROZEN | 8632 | 105.56 | [104.00, 107.11] |
| category_diversity | T1 | FP_WARM_FROZEN | 8632 | 105.30 | [103.79, 106.87] |
| category_diversity | T1 | PF_WARM_FROZEN | 8632 | 105.26 | [103.72, 106.84] |
| category_diversity | T1 | FA_Q8_WARM_FROZEN | 8632 | 105.28 | [103.76, 106.87] |
| category_diversity | T1 | FA_1024_WARM_FROZEN | 8632 | 105.58 | [104.02, 107.13] |
| category_diversity | T1 | DP1_WARM_FROZEN | 8632 | 105.72 | [104.11, 107.46] |
| category_diversity | T1 | PRE | 8632 | 105.80 | [104.06, 107.57] |
| category_diversity | T1 | FT_C | 8632 | 95.84 | [94.75, 96.99] |
| category_diversity | T1 | FT_PRE | 8632 | 106.99 | [105.30, 108.67] |
| category_diversity | T1 | FT_FA | 8632 | 88.33 | [86.96, 89.74] |
| category_diversity | T1 | FT_FA_WARM | 8632 | 106.03 | [104.49, 107.63] |
| category_diversity | T1 | FT_FA_WARM_FROZEN | 8632 | 106.25 | [104.70, 107.89] |
| category_diversity | T1 | LO | 8632 | 1.82 | [1.48, 2.19] |
| category_diversity | T1 | KNN_EMB_SMALL | 8632 | 47.28 | [45.41, 49.17] |
| category_diversity | T1 | KNN_EMB_BIG | 8632 | 53.69 | [51.74, 55.62] |
| category_diversity | T1 | RULE | 8632 | 24.67 | [23.39, 25.97] |
| category_diversity | T1 | SKNN | 8632 | 72.69 | [71.13, 74.27] |
| category_diversity | T1 | LAST | 8632 | 49.68 | [48.18, 51.22] |
| category_diversity | T1 | POP | 8632 | 13.87 | [12.77, 14.88] |
| category_diversity | T2 | C_BIG | 16338 | 100.00 | [100.00, 100.00] |
| category_diversity | T2 | C_FULL | 16338 | 121.61 | [120.38, 122.98] |
| category_diversity | T2 | C_SMALL | 16338 | 95.14 | [94.24, 96.02] |
| category_diversity | T2 | FA | 16338 | 85.29 | [84.25, 86.28] |
| category_diversity | T2 | FP | 16338 | 85.29 | [84.25, 86.30] |
| category_diversity | T2 | PF | 16338 | 85.30 | [84.27, 86.30] |
| category_diversity | T2 | FA_1024 | 16338 | 21.58 | [20.59, 22.47] |
| category_diversity | T2 | DP8 | 16338 | 21.34 | [20.41, 22.23] |
| category_diversity | T2 | DP1 | 16338 | 20.70 | [19.71, 21.63] |
| category_diversity | T2 | FA_Q8 | 16338 | 85.35 | [84.31, 86.34] |
| category_diversity | T2 | FA_WARM | 16338 | 107.80 | [106.56, 109.00] |
| category_diversity | T2 | FA_WARM_FROZEN | 16338 | 108.47 | [107.28, 109.72] |
| category_diversity | T2 | DP8_FROZEN | 16338 | 108.52 | [107.33, 109.75] |
| category_diversity | T2 | FP_WARM_FROZEN | 16338 | 108.47 | [107.28, 109.72] |
| category_diversity | T2 | PF_WARM_FROZEN | 16338 | 108.24 | [107.05, 109.45] |
| category_diversity | T2 | FA_Q8_WARM_FROZEN | 16338 | 108.23 | [107.06, 109.45] |
| category_diversity | T2 | FA_1024_WARM_FROZEN | 16338 | 108.51 | [107.32, 109.72] |
| category_diversity | T2 | DP1_WARM_FROZEN | 16338 | 108.75 | [107.45, 110.11] |
| category_diversity | T2 | PRE | 16338 | 108.97 | [107.65, 110.30] |
| category_diversity | T2 | FT_C | 16338 | 96.77 | [95.92, 97.63] |
| category_diversity | T2 | FT_PRE | 16338 | 110.34 | [109.16, 111.65] |
| category_diversity | T2 | FT_FA | 16338 | 87.77 | [86.73, 88.81] |
| category_diversity | T2 | FT_FA_WARM | 16338 | 109.50 | [108.26, 110.84] |
| category_diversity | T2 | FT_FA_WARM_FROZEN | 16338 | 109.92 | [108.71, 111.21] |
| category_diversity | T2 | LO | 16338 | 2.33 | [2.04, 2.64] |
| category_diversity | T2 | KNN_EMB_SMALL | 16338 | 43.82 | [42.55, 45.16] |
| category_diversity | T2 | KNN_EMB_BIG | 16338 | 51.77 | [50.42, 53.25] |
| category_diversity | T2 | RULE | 16338 | 22.50 | [21.55, 23.53] |
| category_diversity | T2 | SKNN | 16338 | 75.19 | [74.16, 76.32] |
| category_diversity | T2 | LAST | 16338 | 50.83 | [49.74, 52.01] |
| category_diversity | T2 | POP | 16338 | 12.23 | [11.50, 13.04] |
| brand_loyalty | T0 | C_BIG | 10616 | 100.00 | [100.00, 100.00] |
| brand_loyalty | T0 | C_FULL | 10616 | 123.96 | [122.12, 125.69] |
| brand_loyalty | T0 | C_SMALL | 10616 | 95.65 | [94.54, 96.71] |
| brand_loyalty | T0 | FA | 10616 | 85.43 | [84.17, 86.74] |
| brand_loyalty | T0 | FP | 10616 | 85.44 | [84.15, 86.74] |
| brand_loyalty | T0 | PF | 10616 | 85.43 | [84.16, 86.75] |
| brand_loyalty | T0 | FA_1024 | 10616 | 18.56 | [17.36, 19.77] |
| brand_loyalty | T0 | DP8 | 10616 | 18.42 | [17.27, 19.61] |
| brand_loyalty | T0 | DP1 | 10616 | 18.29 | [17.15, 19.52] |
| brand_loyalty | T0 | FA_Q8 | 10616 | 85.52 | [84.27, 86.80] |
| brand_loyalty | T0 | FA_WARM | 10616 | 108.89 | [107.34, 110.44] |
| brand_loyalty | T0 | FA_WARM_FROZEN | 10616 | 109.75 | [108.15, 111.34] |
| brand_loyalty | T0 | DP8_FROZEN | 10616 | 109.95 | [108.22, 111.68] |
| brand_loyalty | T0 | FP_WARM_FROZEN | 10616 | 109.75 | [108.15, 111.34] |
| brand_loyalty | T0 | PF_WARM_FROZEN | 10616 | 109.56 | [107.98, 111.16] |
| brand_loyalty | T0 | FA_Q8_WARM_FROZEN | 10616 | 109.62 | [108.01, 111.21] |
| brand_loyalty | T0 | FA_1024_WARM_FROZEN | 10616 | 109.98 | [108.24, 111.70] |
| brand_loyalty | T0 | DP1_WARM_FROZEN | 10616 | 110.37 | [108.52, 112.20] |
| brand_loyalty | T0 | PRE | 10616 | 110.52 | [108.71, 112.37] |
| brand_loyalty | T0 | FT_C | 10616 | 97.44 | [96.34, 98.54] |
| brand_loyalty | T0 | FT_PRE | 10616 | 111.82 | [110.06, 113.51] |
| brand_loyalty | T0 | FT_FA | 10616 | 88.02 | [86.70, 89.41] |
| brand_loyalty | T0 | FT_FA_WARM | 10616 | 111.08 | [109.54, 112.74] |
| brand_loyalty | T0 | FT_FA_WARM_FROZEN | 10616 | 111.54 | [109.97, 113.23] |
| brand_loyalty | T0 | LO | 10616 | 1.75 | [1.43, 2.11] |
| brand_loyalty | T0 | KNN_EMB_SMALL | 10616 | 41.07 | [39.51, 42.77] |
| brand_loyalty | T0 | KNN_EMB_BIG | 10616 | 50.76 | [48.94, 52.66] |
| brand_loyalty | T0 | RULE | 10616 | 19.48 | [18.39, 20.59] |
| brand_loyalty | T0 | SKNN | 10616 | 78.07 | [76.54, 79.56] |
| brand_loyalty | T0 | LAST | 10616 | 51.36 | [49.83, 52.93] |
| brand_loyalty | T0 | POP | 10616 | 9.88 | [9.07, 10.77] |
| brand_loyalty | T1 | C_BIG | 8811 | 100.00 | [100.00, 100.00] |
| brand_loyalty | T1 | C_FULL | 8811 | 120.84 | [119.05, 122.71] |
| brand_loyalty | T1 | C_SMALL | 8811 | 94.55 | [93.38, 95.79] |
| brand_loyalty | T1 | FA | 8811 | 85.10 | [83.71, 86.57] |
| brand_loyalty | T1 | FP | 8811 | 85.10 | [83.71, 86.58] |
| brand_loyalty | T1 | PF | 8811 | 85.15 | [83.72, 86.62] |
| brand_loyalty | T1 | FA_1024 | 8811 | 22.86 | [21.58, 24.14] |
| brand_loyalty | T1 | DP8 | 8811 | 22.58 | [21.35, 23.87] |
| brand_loyalty | T1 | DP1 | 8811 | 21.59 | [20.33, 22.91] |
| brand_loyalty | T1 | FA_Q8 | 8811 | 85.21 | [83.83, 86.76] |
| brand_loyalty | T1 | FA_WARM | 8811 | 107.04 | [105.48, 108.78] |
| brand_loyalty | T1 | FA_WARM_FROZEN | 8811 | 108.05 | [106.53, 109.83] |
| brand_loyalty | T1 | DP8_FROZEN | 8811 | 107.88 | [106.31, 109.73] |
| brand_loyalty | T1 | FP_WARM_FROZEN | 8811 | 108.05 | [106.52, 109.82] |
| brand_loyalty | T1 | PF_WARM_FROZEN | 8811 | 107.91 | [106.37, 109.65] |
| brand_loyalty | T1 | FA_Q8_WARM_FROZEN | 8811 | 107.88 | [106.34, 109.59] |
| brand_loyalty | T1 | FA_1024_WARM_FROZEN | 8811 | 107.89 | [106.29, 109.71] |
| brand_loyalty | T1 | DP1_WARM_FROZEN | 8811 | 107.72 | [106.00, 109.78] |
| brand_loyalty | T1 | PRE | 8811 | 107.98 | [106.30, 110.01] |
| brand_loyalty | T1 | FT_C | 8811 | 95.65 | [94.53, 96.89] |
| brand_loyalty | T1 | FT_PRE | 8811 | 109.09 | [107.47, 110.86] |
| brand_loyalty | T1 | FT_FA | 8811 | 86.66 | [85.32, 88.17] |
| brand_loyalty | T1 | FT_FA_WARM | 8811 | 108.25 | [106.79, 109.99] |
| brand_loyalty | T1 | FT_FA_WARM_FROZEN | 8811 | 108.69 | [107.19, 110.32] |
| brand_loyalty | T1 | LO | 8811 | 1.93 | [1.57, 2.34] |
| brand_loyalty | T1 | KNN_EMB_SMALL | 8811 | 43.58 | [41.82, 45.50] |
| brand_loyalty | T1 | KNN_EMB_BIG | 8811 | 50.75 | [48.92, 52.68] |
| brand_loyalty | T1 | RULE | 8811 | 23.24 | [21.94, 24.59] |
| brand_loyalty | T1 | SKNN | 8811 | 74.44 | [72.76, 76.16] |
| brand_loyalty | T1 | LAST | 8811 | 50.11 | [48.58, 51.89] |
| brand_loyalty | T1 | POP | 8811 | 12.42 | [11.39, 13.38] |
| brand_loyalty | T2 | C_BIG | 6827 | 100.00 | [100.00, 100.00] |
| brand_loyalty | T2 | C_FULL | 6827 | 118.19 | [116.45, 119.92] |
| brand_loyalty | T2 | C_SMALL | 6827 | 93.72 | [92.47, 94.97] |
| brand_loyalty | T2 | FA | 6827 | 84.75 | [83.24, 86.11] |
| brand_loyalty | T2 | FP | 6827 | 84.75 | [83.25, 86.11] |
| brand_loyalty | T2 | PF | 6827 | 84.74 | [83.26, 86.11] |
| brand_loyalty | T2 | FA_1024 | 6827 | 28.38 | [26.84, 30.02] |
| brand_loyalty | T2 | DP8 | 6827 | 27.74 | [26.18, 29.33] |
| brand_loyalty | T2 | DP1 | 6827 | 25.94 | [24.49, 27.47] |
| brand_loyalty | T2 | FA_Q8 | 6827 | 84.82 | [83.37, 86.21] |
| brand_loyalty | T2 | FA_WARM | 6827 | 103.40 | [101.91, 104.85] |
| brand_loyalty | T2 | FA_WARM_FROZEN | 6827 | 103.91 | [102.37, 105.41] |
| brand_loyalty | T2 | DP8_FROZEN | 6827 | 104.28 | [102.66, 105.73] |
| brand_loyalty | T2 | FP_WARM_FROZEN | 6827 | 103.92 | [102.37, 105.41] |
| brand_loyalty | T2 | PF_WARM_FROZEN | 6827 | 103.76 | [102.20, 105.23] |
| brand_loyalty | T2 | FA_Q8_WARM_FROZEN | 6827 | 103.72 | [102.20, 105.21] |
| brand_loyalty | T2 | FA_1024_WARM_FROZEN | 6827 | 104.27 | [102.68, 105.73] |
| brand_loyalty | T2 | DP1_WARM_FROZEN | 6827 | 104.53 | [102.66, 106.20] |
| brand_loyalty | T2 | PRE | 6827 | 104.62 | [102.74, 106.32] |
| brand_loyalty | T2 | FT_C | 6827 | 95.84 | [94.68, 97.06] |
| brand_loyalty | T2 | FT_PRE | 6827 | 106.48 | [104.90, 108.03] |
| brand_loyalty | T2 | FT_FA | 6827 | 87.99 | [86.51, 89.39] |
| brand_loyalty | T2 | FT_FA_WARM | 6827 | 105.26 | [103.74, 106.79] |
| brand_loyalty | T2 | FT_FA_WARM_FROZEN | 6827 | 105.61 | [103.94, 107.11] |
| brand_loyalty | T2 | LO | 6827 | 2.70 | [2.20, 3.23] |
| brand_loyalty | T2 | KNN_EMB_SMALL | 6827 | 52.09 | [50.04, 54.08] |
| brand_loyalty | T2 | KNN_EMB_BIG | 6827 | 57.96 | [55.88, 60.00] |
| brand_loyalty | T2 | RULE | 6827 | 27.82 | [26.32, 29.43] |
| brand_loyalty | T2 | SKNN | 6827 | 70.77 | [69.21, 72.34] |
| brand_loyalty | T2 | LAST | 6827 | 51.03 | [49.40, 52.59] |
| brand_loyalty | T2 | POP | 6827 | 16.14 | [14.95, 17.51] |
| persona | d0_l0 | C_BIG | 295 | 100.00 | [100.00, 100.00] |
| persona | d0_l0 | C_FULL | 295 | 127.60 | [117.69, 139.72] |
| persona | d0_l0 | C_SMALL | 295 | 88.38 | [81.65, 95.24] |
| persona | d0_l0 | FA | 295 | 77.93 | [68.22, 88.56] |
| persona | d0_l0 | FP | 295 | 77.92 | [68.23, 88.54] |
| persona | d0_l0 | PF | 295 | 77.92 | [68.24, 88.56] |
| persona | d0_l0 | FA_1024 | 295 | 16.63 | [11.41, 23.31] |
| persona | d0_l0 | DP8 | 295 | 16.27 | [11.20, 22.86] |
| persona | d0_l0 | DP1 | 295 | 16.64 | [11.42, 22.52] |
| persona | d0_l0 | FA_Q8 | 295 | 77.50 | [67.89, 87.52] |
| persona | d0_l0 | FA_WARM | 295 | 103.76 | [92.25, 116.55] |
| persona | d0_l0 | FA_WARM_FROZEN | 295 | 103.59 | [92.49, 115.46] |
| persona | d0_l0 | DP8_FROZEN | 295 | 103.22 | [91.82, 115.66] |
| persona | d0_l0 | FP_WARM_FROZEN | 295 | 103.59 | [92.49, 115.46] |
| persona | d0_l0 | PF_WARM_FROZEN | 295 | 103.01 | [91.70, 115.05] |
| persona | d0_l0 | FA_Q8_WARM_FROZEN | 295 | 102.99 | [91.73, 115.03] |
| persona | d0_l0 | FA_1024_WARM_FROZEN | 295 | 103.26 | [91.59, 115.43] |
| persona | d0_l0 | DP1_WARM_FROZEN | 295 | 106.41 | [94.28, 120.72] |
| persona | d0_l0 | PRE | 295 | 108.18 | [95.47, 122.83] |
| persona | d0_l0 | FT_C | 295 | 92.74 | [84.67, 100.69] |
| persona | d0_l0 | FT_PRE | 295 | 110.24 | [97.50, 123.98] |
| persona | d0_l0 | FT_FA | 295 | 79.81 | [69.65, 89.92] |
| persona | d0_l0 | FT_FA_WARM | 295 | 108.73 | [96.53, 122.30] |
| persona | d0_l0 | FT_FA_WARM_FROZEN | 295 | 108.87 | [97.03, 122.33] |
| persona | d0_l0 | LO | 295 | 1.76 | [0.29, 3.95] |
| persona | d0_l0 | KNN_EMB_SMALL | 295 | 46.06 | [36.51, 57.03] |
| persona | d0_l0 | KNN_EMB_BIG | 295 | 61.00 | [48.73, 73.36] |
| persona | d0_l0 | RULE | 295 | 18.45 | [13.28, 24.66] |
| persona | d0_l0 | SKNN | 295 | 77.33 | [68.91, 87.15] |
| persona | d0_l0 | LAST | 295 | 54.54 | [45.75, 64.65] |
| persona | d0_l0 | POP | 295 | 8.26 | [4.96, 12.54] |
| persona | d0_l1 | C_BIG | 440 | 100.00 | [100.00, 100.00] |
| persona | d0_l1 | C_FULL | 440 | 126.25 | [116.62, 136.89] |
| persona | d0_l1 | C_SMALL | 440 | 95.38 | [88.54, 102.41] |
| persona | d0_l1 | FA | 440 | 75.86 | [68.31, 84.32] |
| persona | d0_l1 | FP | 440 | 75.87 | [68.33, 84.34] |
| persona | d0_l1 | PF | 440 | 75.88 | [68.30, 84.34] |
| persona | d0_l1 | FA_1024 | 440 | 15.32 | [10.52, 20.33] |
| persona | d0_l1 | DP8 | 440 | 15.53 | [10.92, 20.53] |
| persona | d0_l1 | DP1 | 440 | 15.21 | [11.09, 19.43] |
| persona | d0_l1 | FA_Q8 | 440 | 76.07 | [68.74, 83.98] |
| persona | d0_l1 | FA_WARM | 440 | 109.40 | [101.82, 118.56] |
| persona | d0_l1 | FA_WARM_FROZEN | 440 | 109.85 | [102.23, 119.12] |
| persona | d0_l1 | DP8_FROZEN | 440 | 110.47 | [102.92, 120.95] |
| persona | d0_l1 | FP_WARM_FROZEN | 440 | 109.85 | [102.23, 119.13] |
| persona | d0_l1 | PF_WARM_FROZEN | 440 | 109.59 | [102.16, 118.92] |
| persona | d0_l1 | FA_Q8_WARM_FROZEN | 440 | 109.51 | [102.19, 118.95] |
| persona | d0_l1 | FA_1024_WARM_FROZEN | 440 | 111.03 | [103.12, 121.51] |
| persona | d0_l1 | DP1_WARM_FROZEN | 440 | 109.00 | [100.28, 120.64] |
| persona | d0_l1 | PRE | 440 | 108.65 | [99.87, 120.36] |
| persona | d0_l1 | FT_C | 440 | 99.01 | [92.63, 105.70] |
| persona | d0_l1 | FT_PRE | 440 | 111.60 | [104.22, 120.85] |
| persona | d0_l1 | FT_FA | 440 | 76.38 | [68.77, 84.38] |
| persona | d0_l1 | FT_FA_WARM | 440 | 111.88 | [104.22, 120.80] |
| persona | d0_l1 | FT_FA_WARM_FROZEN | 440 | 112.36 | [105.12, 120.81] |
| persona | d0_l1 | LO | 440 | 1.06 | [0.32, 1.95] |
| persona | d0_l1 | KNN_EMB_SMALL | 440 | 46.51 | [38.64, 55.48] |
| persona | d0_l1 | KNN_EMB_BIG | 440 | 63.95 | [53.90, 76.25] |
| persona | d0_l1 | RULE | 440 | 21.78 | [16.83, 27.33] |
| persona | d0_l1 | SKNN | 440 | 85.31 | [77.41, 94.68] |
| persona | d0_l1 | LAST | 440 | 64.25 | [55.91, 74.08] |
| persona | d0_l1 | POP | 440 | 8.43 | [4.49, 13.60] |
| persona | d0_l2 | C_BIG | 549 | 100.00 | [100.00, 100.00] |
| persona | d0_l2 | C_FULL | 549 | 127.63 | [118.34, 138.63] |
| persona | d0_l2 | C_SMALL | 549 | 89.40 | [83.85, 95.42] |
| persona | d0_l2 | FA | 549 | 74.32 | [68.41, 81.68] |
| persona | d0_l2 | FP | 549 | 74.31 | [68.41, 81.67] |
| persona | d0_l2 | PF | 549 | 74.34 | [68.40, 81.67] |
| persona | d0_l2 | FA_1024 | 549 | 16.07 | [11.33, 21.54] |
| persona | d0_l2 | DP8 | 549 | 15.76 | [11.32, 20.67] |
| persona | d0_l2 | DP1 | 549 | 15.86 | [11.25, 20.80] |
| persona | d0_l2 | FA_Q8 | 549 | 75.59 | [69.33, 83.22] |
| persona | d0_l2 | FA_WARM | 549 | 110.26 | [102.08, 119.39] |
| persona | d0_l2 | FA_WARM_FROZEN | 549 | 114.25 | [105.62, 123.87] |
| persona | d0_l2 | DP8_FROZEN | 549 | 114.14 | [105.30, 123.76] |
| persona | d0_l2 | FP_WARM_FROZEN | 549 | 114.24 | [105.62, 123.86] |
| persona | d0_l2 | PF_WARM_FROZEN | 549 | 114.12 | [105.38, 124.02] |
| persona | d0_l2 | FA_Q8_WARM_FROZEN | 549 | 114.12 | [105.37, 124.11] |
| persona | d0_l2 | FA_1024_WARM_FROZEN | 549 | 114.17 | [105.32, 123.84] |
| persona | d0_l2 | DP1_WARM_FROZEN | 549 | 111.88 | [102.45, 122.99] |
| persona | d0_l2 | PRE | 549 | 111.49 | [102.12, 122.34] |
| persona | d0_l2 | FT_C | 549 | 92.54 | [86.83, 98.10] |
| persona | d0_l2 | FT_PRE | 549 | 117.70 | [109.05, 127.17] |
| persona | d0_l2 | FT_FA | 549 | 77.51 | [71.53, 84.62] |
| persona | d0_l2 | FT_FA_WARM | 549 | 114.19 | [105.45, 123.62] |
| persona | d0_l2 | FT_FA_WARM_FROZEN | 549 | 118.66 | [109.76, 128.89] |
| persona | d0_l2 | LO | 549 | 1.19 | [0.11, 3.16] |
| persona | d0_l2 | KNN_EMB_SMALL | 549 | 53.44 | [44.25, 64.19] |
| persona | d0_l2 | KNN_EMB_BIG | 549 | 65.97 | [55.07, 78.56] |
| persona | d0_l2 | RULE | 549 | 26.10 | [19.80, 33.38] |
| persona | d0_l2 | SKNN | 549 | 85.63 | [77.90, 95.10] |
| persona | d0_l2 | LAST | 549 | 62.95 | [55.25, 71.67] |
| persona | d0_l2 | POP | 549 | 7.41 | [3.93, 12.00] |
| persona | d1_l0 | C_BIG | 1899 | 100.00 | [100.00, 100.00] |
| persona | d1_l0 | C_FULL | 1899 | 123.13 | [119.11, 127.51] |
| persona | d1_l0 | C_SMALL | 1899 | 95.57 | [92.84, 98.28] |
| persona | d1_l0 | FA | 1899 | 86.80 | [83.59, 90.30] |
| persona | d1_l0 | FP | 1899 | 86.79 | [83.58, 90.29] |
| persona | d1_l0 | PF | 1899 | 86.77 | [83.55, 90.31] |
| persona | d1_l0 | FA_1024 | 1899 | 20.37 | [17.53, 23.35] |
| persona | d1_l0 | DP8 | 1899 | 20.29 | [17.56, 23.34] |
| persona | d1_l0 | DP1 | 1899 | 20.07 | [17.16, 23.36] |
| persona | d1_l0 | FA_Q8 | 1899 | 86.46 | [83.14, 89.91] |
| persona | d1_l0 | FA_WARM | 1899 | 105.37 | [101.69, 109.03] |
| persona | d1_l0 | FA_WARM_FROZEN | 1899 | 106.49 | [102.84, 110.30] |
| persona | d1_l0 | DP8_FROZEN | 1899 | 106.52 | [102.76, 110.64] |
| persona | d1_l0 | FP_WARM_FROZEN | 1899 | 106.49 | [102.83, 110.30] |
| persona | d1_l0 | PF_WARM_FROZEN | 1899 | 106.48 | [102.89, 110.43] |
| persona | d1_l0 | FA_Q8_WARM_FROZEN | 1899 | 106.60 | [102.96, 110.71] |
| persona | d1_l0 | FA_1024_WARM_FROZEN | 1899 | 106.55 | [102.77, 110.66] |
| persona | d1_l0 | DP1_WARM_FROZEN | 1899 | 106.19 | [101.77, 110.47] |
| persona | d1_l0 | PRE | 1899 | 106.22 | [101.83, 110.62] |
| persona | d1_l0 | FT_C | 1899 | 97.79 | [95.16, 100.59] |
| persona | d1_l0 | FT_PRE | 1899 | 108.75 | [104.94, 112.87] |
| persona | d1_l0 | FT_FA | 1899 | 89.20 | [85.94, 92.75] |
| persona | d1_l0 | FT_FA_WARM | 1899 | 108.37 | [104.81, 112.14] |
| persona | d1_l0 | FT_FA_WARM_FROZEN | 1899 | 108.55 | [104.85, 112.49] |
| persona | d1_l0 | LO | 1899 | 1.68 | [0.97, 2.54] |
| persona | d1_l0 | KNN_EMB_SMALL | 1899 | 39.73 | [36.05, 43.68] |
| persona | d1_l0 | KNN_EMB_BIG | 1899 | 47.92 | [43.84, 52.06] |
| persona | d1_l0 | RULE | 1899 | 20.15 | [17.72, 22.58] |
| persona | d1_l0 | SKNN | 1899 | 81.19 | [77.30, 85.23] |
| persona | d1_l0 | LAST | 1899 | 50.28 | [46.53, 54.13] |
| persona | d1_l0 | POP | 1899 | 10.47 | [8.60, 12.57] |
| persona | d1_l1 | C_BIG | 3021 | 100.00 | [100.00, 100.00] |
| persona | d1_l1 | C_FULL | 3021 | 123.00 | [120.18, 126.34] |
| persona | d1_l1 | C_SMALL | 3021 | 94.05 | [92.05, 95.88] |
| persona | d1_l1 | FA | 3021 | 85.73 | [83.39, 88.06] |
| persona | d1_l1 | FP | 3021 | 85.73 | [83.38, 88.06] |
| persona | d1_l1 | PF | 3021 | 85.83 | [83.51, 88.14] |
| persona | d1_l1 | FA_1024 | 3021 | 23.93 | [21.77, 26.37] |
| persona | d1_l1 | DP8 | 3021 | 23.64 | [21.44, 25.98] |
| persona | d1_l1 | DP1 | 3021 | 22.79 | [20.63, 25.28] |
| persona | d1_l1 | FA_Q8 | 3021 | 86.64 | [84.18, 89.04] |
| persona | d1_l1 | FA_WARM | 3021 | 107.13 | [104.50, 110.00] |
| persona | d1_l1 | FA_WARM_FROZEN | 3021 | 108.89 | [106.06, 111.68] |
| persona | d1_l1 | DP8_FROZEN | 3021 | 109.01 | [106.18, 111.85] |
| persona | d1_l1 | FP_WARM_FROZEN | 3021 | 108.89 | [106.08, 111.69] |
| persona | d1_l1 | PF_WARM_FROZEN | 3021 | 108.87 | [106.05, 111.65] |
| persona | d1_l1 | FA_Q8_WARM_FROZEN | 3021 | 108.93 | [106.13, 111.66] |
| persona | d1_l1 | FA_1024_WARM_FROZEN | 3021 | 109.05 | [106.24, 111.92] |
| persona | d1_l1 | DP1_WARM_FROZEN | 3021 | 109.19 | [106.00, 112.32] |
| persona | d1_l1 | PRE | 3021 | 109.25 | [106.01, 112.47] |
| persona | d1_l1 | FT_C | 3021 | 94.94 | [93.00, 96.82] |
| persona | d1_l1 | FT_PRE | 3021 | 109.56 | [106.64, 112.56] |
| persona | d1_l1 | FT_FA | 3021 | 87.16 | [84.83, 89.40] |
| persona | d1_l1 | FT_FA_WARM | 3021 | 108.61 | [105.90, 111.54] |
| persona | d1_l1 | FT_FA_WARM_FROZEN | 3021 | 109.15 | [106.38, 112.16] |
| persona | d1_l1 | LO | 3021 | 1.28 | [0.86, 1.83] |
| persona | d1_l1 | KNN_EMB_SMALL | 3021 | 42.53 | [39.63, 45.42] |
| persona | d1_l1 | KNN_EMB_BIG | 3021 | 50.22 | [47.15, 53.60] |
| persona | d1_l1 | RULE | 3021 | 21.44 | [19.44, 23.57] |
| persona | d1_l1 | SKNN | 3021 | 74.15 | [71.62, 76.76] |
| persona | d1_l1 | LAST | 3021 | 49.02 | [46.30, 51.82] |
| persona | d1_l1 | POP | 3021 | 11.80 | [10.32, 13.37] |
| persona | d1_l2 | C_BIG | 3712 | 100.00 | [100.00, 100.00] |
| persona | d1_l2 | C_FULL | 3712 | 116.37 | [114.13, 118.59] |
| persona | d1_l2 | C_SMALL | 3712 | 94.11 | [92.39, 95.80] |
| persona | d1_l2 | FA | 3712 | 85.65 | [83.78, 87.48] |
| persona | d1_l2 | FP | 3712 | 85.65 | [83.79, 87.48] |
| persona | d1_l2 | PF | 3712 | 85.62 | [83.73, 87.47] |
| persona | d1_l2 | FA_1024 | 3712 | 29.75 | [27.51, 31.89] |
| persona | d1_l2 | DP8 | 3712 | 28.89 | [26.81, 31.00] |
| persona | d1_l2 | DP1 | 3712 | 26.51 | [24.58, 28.60] |
| persona | d1_l2 | FA_Q8 | 3712 | 85.38 | [83.54, 87.29] |
| persona | d1_l2 | FA_WARM | 3712 | 102.27 | [100.24, 104.20] |
| persona | d1_l2 | FA_WARM_FROZEN | 3712 | 102.58 | [100.52, 104.60] |
| persona | d1_l2 | DP8_FROZEN | 3712 | 103.00 | [100.99, 104.93] |
| persona | d1_l2 | FP_WARM_FROZEN | 3712 | 102.58 | [100.51, 104.60] |
| persona | d1_l2 | PF_WARM_FROZEN | 3712 | 102.51 | [100.44, 104.52] |
| persona | d1_l2 | FA_Q8_WARM_FROZEN | 3712 | 102.47 | [100.39, 104.50] |
| persona | d1_l2 | FA_1024_WARM_FROZEN | 3712 | 103.02 | [100.99, 104.95] |
| persona | d1_l2 | DP1_WARM_FROZEN | 3712 | 103.33 | [101.02, 105.52] |
| persona | d1_l2 | PRE | 3712 | 103.44 | [101.15, 105.74] |
| persona | d1_l2 | FT_C | 3712 | 95.71 | [93.96, 97.36] |
| persona | d1_l2 | FT_PRE | 3712 | 104.70 | [102.66, 106.86] |
| persona | d1_l2 | FT_FA | 3712 | 88.75 | [86.73, 90.66] |
| persona | d1_l2 | FT_FA_WARM | 3712 | 103.52 | [101.47, 105.40] |
| persona | d1_l2 | FT_FA_WARM_FROZEN | 3712 | 103.56 | [101.50, 105.63] |
| persona | d1_l2 | LO | 3712 | 2.21 | [1.66, 2.82] |
| persona | d1_l2 | KNN_EMB_SMALL | 3712 | 53.08 | [50.39, 55.73] |
| persona | d1_l2 | KNN_EMB_BIG | 3712 | 58.02 | [55.36, 60.81] |
| persona | d1_l2 | RULE | 3712 | 28.40 | [26.35, 30.44] |
| persona | d1_l2 | SKNN | 3712 | 68.64 | [66.62, 70.80] |
| persona | d1_l2 | LAST | 3712 | 49.88 | [47.68, 51.98] |
| persona | d1_l2 | POP | 3712 | 16.43 | [14.79, 18.06] |
| persona | d2_l0 | C_BIG | 8422 | 100.00 | [100.00, 100.00] |
| persona | d2_l0 | C_FULL | 8422 | 124.03 | [122.06, 126.01] |
| persona | d2_l0 | C_SMALL | 8422 | 95.93 | [94.71, 97.12] |
| persona | d2_l0 | FA | 8422 | 85.39 | [84.12, 86.71] |
| persona | d2_l0 | FP | 8422 | 85.40 | [84.13, 86.71] |
| persona | d2_l0 | PF | 8422 | 85.39 | [84.13, 86.70] |
| persona | d2_l0 | FA_1024 | 8422 | 18.21 | [16.85, 19.59] |
| persona | d2_l0 | DP8 | 8422 | 18.06 | [16.76, 19.43] |
| persona | d2_l0 | DP1 | 8422 | 17.93 | [16.51, 19.24] |
| persona | d2_l0 | FA_Q8 | 8422 | 85.59 | [84.24, 86.95] |
| persona | d2_l0 | FA_WARM | 8422 | 109.90 | [108.17, 111.72] |
| persona | d2_l0 | FA_WARM_FROZEN | 8422 | 110.74 | [108.91, 112.66] |
| persona | d2_l0 | DP8_FROZEN | 8422 | 111.00 | [109.06, 112.91] |
| persona | d2_l0 | FP_WARM_FROZEN | 8422 | 110.74 | [108.91, 112.66] |
| persona | d2_l0 | PF_WARM_FROZEN | 8422 | 110.53 | [108.74, 112.41] |
| persona | d2_l0 | FA_Q8_WARM_FROZEN | 8422 | 110.57 | [108.78, 112.44] |
| persona | d2_l0 | FA_1024_WARM_FROZEN | 8422 | 111.02 | [109.07, 112.94] |
| persona | d2_l0 | DP1_WARM_FROZEN | 8422 | 111.49 | [109.45, 113.51] |
| persona | d2_l0 | PRE | 8422 | 111.62 | [109.64, 113.65] |
| persona | d2_l0 | FT_C | 8422 | 97.52 | [96.25, 98.79] |
| persona | d2_l0 | FT_PRE | 8422 | 112.59 | [110.68, 114.43] |
| persona | d2_l0 | FT_FA | 8422 | 88.05 | [86.66, 89.46] |
| persona | d2_l0 | FT_FA_WARM | 8422 | 111.80 | [110.01, 113.61] |
| persona | d2_l0 | FT_FA_WARM_FROZEN | 8422 | 112.34 | [110.48, 114.11] |
| persona | d2_l0 | LO | 8422 | 1.76 | [1.42, 2.13] |
| persona | d2_l0 | KNN_EMB_SMALL | 8422 | 41.21 | [39.49, 42.95] |
| persona | d2_l0 | KNN_EMB_BIG | 8422 | 51.05 | [49.01, 53.15] |
| persona | d2_l0 | RULE | 8422 | 19.36 | [18.10, 20.58] |
| persona | d2_l0 | SKNN | 8422 | 77.38 | [75.75, 79.05] |
| persona | d2_l0 | LAST | 8422 | 51.50 | [49.81, 53.13] |
| persona | d2_l0 | POP | 8422 | 9.80 | [8.84, 10.71] |
| persona | d2_l1 | C_BIG | 5350 | 100.00 | [100.00, 100.00] |
| persona | d2_l1 | C_FULL | 5350 | 119.27 | [117.31, 121.50] |
| persona | d2_l1 | C_SMALL | 5350 | 94.79 | [93.35, 96.28] |
| persona | d2_l1 | FA | 5350 | 85.36 | [83.73, 87.01] |
| persona | d2_l1 | FP | 5350 | 85.36 | [83.73, 87.01] |
| persona | d2_l1 | PF | 5350 | 85.38 | [83.77, 87.09] |
| persona | d2_l1 | FA_1024 | 5350 | 22.76 | [21.10, 24.42] |
| persona | d2_l1 | DP8 | 5350 | 22.45 | [20.83, 24.08] |
| persona | d2_l1 | DP1 | 5350 | 21.35 | [19.84, 22.95] |
| persona | d2_l1 | FA_Q8 | 5350 | 85.01 | [83.35, 86.70] |
| persona | d2_l1 | FA_WARM | 5350 | 106.84 | [104.99, 108.78] |
| persona | d2_l1 | FA_WARM_FROZEN | 5350 | 107.47 | [105.60, 109.58] |
| persona | d2_l1 | DP8_FROZEN | 5350 | 107.07 | [105.18, 109.16] |
| persona | d2_l1 | FP_WARM_FROZEN | 5350 | 107.46 | [105.59, 109.57] |
| persona | d2_l1 | PF_WARM_FROZEN | 5350 | 107.27 | [105.36, 109.38] |
| persona | d2_l1 | FA_Q8_WARM_FROZEN | 5350 | 107.19 | [105.33, 109.32] |
| persona | d2_l1 | FA_1024_WARM_FROZEN | 5350 | 107.03 | [105.15, 109.09] |
| persona | d2_l1 | DP1_WARM_FROZEN | 5350 | 106.82 | [104.62, 109.10] |
| persona | d2_l1 | PRE | 5350 | 107.22 | [105.04, 109.53] |
| persona | d2_l1 | FT_C | 5350 | 95.84 | [94.40, 97.34] |
| persona | d2_l1 | FT_PRE | 5350 | 108.66 | [106.72, 110.81] |
| persona | d2_l1 | FT_FA | 5350 | 87.06 | [85.47, 88.68] |
| persona | d2_l1 | FT_FA_WARM | 5350 | 107.81 | [105.89, 109.87] |
| persona | d2_l1 | FT_FA_WARM_FROZEN | 5350 | 108.19 | [106.30, 110.34] |
| persona | d2_l1 | LO | 5350 | 2.35 | [1.83, 2.98] |
| persona | d2_l1 | KNN_EMB_SMALL | 5350 | 43.99 | [41.80, 46.21] |
| persona | d2_l1 | KNN_EMB_BIG | 5350 | 50.17 | [47.93, 52.53] |
| persona | d2_l1 | RULE | 5350 | 24.35 | [22.57, 26.08] |
| persona | d2_l1 | SKNN | 5350 | 73.89 | [71.87, 75.85] |
| persona | d2_l1 | LAST | 5350 | 49.80 | [47.91, 51.84] |
| persona | d2_l1 | POP | 5350 | 13.03 | [11.85, 14.32] |
| persona | d2_l2 | C_BIG | 2566 | 100.00 | [100.00, 100.00] |
| persona | d2_l2 | C_FULL | 2566 | 119.73 | [117.01, 122.36] |
| persona | d2_l2 | C_SMALL | 2566 | 93.75 | [91.76, 95.68] |
| persona | d2_l2 | FA | 2566 | 84.90 | [82.53, 87.18] |
| persona | d2_l2 | FP | 2566 | 84.89 | [82.53, 87.18] |
| persona | d2_l2 | PF | 2566 | 84.91 | [82.50, 87.20] |
| persona | d2_l2 | FA_1024 | 2566 | 28.07 | [25.43, 30.77] |
| persona | d2_l2 | DP8 | 2566 | 27.73 | [25.02, 30.36] |
| persona | d2_l2 | DP1 | 2566 | 26.60 | [24.06, 29.14] |
| persona | d2_l2 | FA_Q8 | 2566 | 85.35 | [82.92, 87.56] |
| persona | d2_l2 | FA_WARM | 2566 | 104.18 | [101.64, 106.76] |
| persona | d2_l2 | FA_WARM_FROZEN | 2566 | 104.50 | [101.99, 107.02] |
| persona | d2_l2 | DP8_FROZEN | 2566 | 104.83 | [102.18, 107.48] |
| persona | d2_l2 | FP_WARM_FROZEN | 2566 | 104.51 | [102.00, 107.03] |
| persona | d2_l2 | PF_WARM_FROZEN | 2566 | 104.18 | [101.64, 106.69] |
| persona | d2_l2 | FA_Q8_WARM_FROZEN | 2566 | 104.17 | [101.59, 106.65] |
| persona | d2_l2 | FA_1024_WARM_FROZEN | 2566 | 104.79 | [102.15, 107.40] |
| persona | d2_l2 | DP1_WARM_FROZEN | 2566 | 105.32 | [102.51, 108.08] |
| persona | d2_l2 | PRE | 2566 | 105.47 | [102.74, 108.27] |
| persona | d2_l2 | FT_C | 2566 | 96.59 | [94.52, 98.63] |
| persona | d2_l2 | FT_PRE | 2566 | 107.67 | [105.09, 110.35] |
| persona | d2_l2 | FT_FA | 2566 | 88.37 | [86.01, 90.63] |
| persona | d2_l2 | FT_FA_WARM | 2566 | 106.73 | [104.05, 109.26] |
| persona | d2_l2 | FT_FA_WARM_FROZEN | 2566 | 106.93 | [104.26, 109.52] |
| persona | d2_l2 | LO | 2566 | 3.74 | [2.82, 4.79] |
| persona | d2_l2 | KNN_EMB_SMALL | 2566 | 50.25 | [46.98, 53.35] |
| persona | d2_l2 | KNN_EMB_BIG | 2566 | 56.63 | [53.38, 60.05] |
| persona | d2_l2 | RULE | 2566 | 27.14 | [24.50, 29.82] |
| persona | d2_l2 | SKNN | 2566 | 71.95 | [69.54, 74.39] |
| persona | d2_l2 | LAST | 2566 | 51.06 | [48.58, 53.64] |
| persona | d2_l2 | POP | 2566 | 17.01 | [14.97, 19.29] |
| activity | Q0 | C_BIG | 2977 | 100.00 | [100.00, 100.00] |
| activity | Q0 | C_FULL | 2977 | 119.65 | [116.47, 122.74] |
| activity | Q0 | C_SMALL | 2977 | 95.07 | [93.03, 97.18] |
| activity | Q0 | FA | 2977 | 87.36 | [84.94, 89.54] |
| activity | Q0 | FP | 2977 | 87.37 | [84.94, 89.54] |
| activity | Q0 | PF | 2977 | 87.33 | [84.90, 89.54] |
| activity | Q0 | FA_1024 | 2977 | 27.88 | [25.17, 30.59] |
| activity | Q0 | DP8 | 2977 | 27.38 | [24.83, 30.01] |
| activity | Q0 | DP1 | 2977 | 24.72 | [22.41, 27.13] |
| activity | Q0 | FA_Q8 | 2977 | 87.34 | [84.96, 89.70] |
| activity | Q0 | FA_WARM | 2977 | 107.12 | [104.50, 109.88] |
| activity | Q0 | FA_WARM_FROZEN | 2977 | 107.71 | [105.13, 110.57] |
| activity | Q0 | DP8_FROZEN | 2977 | 108.48 | [105.79, 111.47] |
| activity | Q0 | FP_WARM_FROZEN | 2977 | 107.70 | [105.13, 110.57] |
| activity | Q0 | PF_WARM_FROZEN | 2977 | 107.42 | [104.79, 110.41] |
| activity | Q0 | FA_Q8_WARM_FROZEN | 2977 | 107.42 | [104.79, 110.40] |
| activity | Q0 | FA_1024_WARM_FROZEN | 2977 | 108.51 | [105.80, 111.50] |
| activity | Q0 | DP1_WARM_FROZEN | 2977 | 107.87 | [104.85, 111.17] |
| activity | Q0 | PRE | 2977 | 107.81 | [104.80, 111.14] |
| activity | Q0 | FT_C | 2977 | 95.97 | [93.92, 98.07] |
| activity | Q0 | FT_PRE | 2977 | 109.49 | [106.70, 112.60] |
| activity | Q0 | FT_FA | 2977 | 88.32 | [85.86, 90.56] |
| activity | Q0 | FT_FA_WARM | 2977 | 107.96 | [105.33, 110.71] |
| activity | Q0 | FT_FA_WARM_FROZEN | 2977 | 108.49 | [105.70, 111.40] |
| activity | Q0 | LO | 2977 | 0.02 | [0.01, 0.05] |
| activity | Q0 | KNN_EMB_SMALL | 2977 | 47.16 | [43.87, 50.63] |
| activity | Q0 | KNN_EMB_BIG | 2977 | 53.80 | [50.24, 57.36] |
| activity | Q0 | RULE | 2977 | 28.86 | [26.31, 31.43] |
| activity | Q0 | SKNN | 2977 | 74.32 | [71.83, 76.78] |
| activity | Q0 | LAST | 2977 | 48.33 | [45.58, 51.03] |
| activity | Q0 | POP | 2977 | 15.50 | [13.46, 17.80] |
| activity | Q1 | C_BIG | 6235 | 100.00 | [100.00, 100.00] |
| activity | Q1 | C_FULL | 6235 | 120.74 | [118.71, 122.90] |
| activity | Q1 | C_SMALL | 6235 | 95.24 | [93.78, 96.60] |
| activity | Q1 | FA | 6235 | 85.84 | [84.21, 87.50] |
| activity | Q1 | FP | 6235 | 85.83 | [84.22, 87.50] |
| activity | Q1 | PF | 6235 | 85.90 | [84.30, 87.54] |
| activity | Q1 | FA_1024 | 6235 | 24.79 | [23.13, 26.53] |
| activity | Q1 | DP8 | 6235 | 24.54 | [22.95, 26.24] |
| activity | Q1 | DP1 | 6235 | 23.82 | [22.19, 25.45] |
| activity | Q1 | FA_Q8 | 6235 | 85.81 | [84.16, 87.45] |
| activity | Q1 | FA_WARM | 6235 | 106.97 | [105.08, 108.84] |
| activity | Q1 | FA_WARM_FROZEN | 6235 | 107.69 | [105.74, 109.62] |
| activity | Q1 | DP8_FROZEN | 6235 | 107.86 | [105.78, 109.82] |
| activity | Q1 | FP_WARM_FROZEN | 6235 | 107.69 | [105.73, 109.62] |
| activity | Q1 | PF_WARM_FROZEN | 6235 | 107.52 | [105.51, 109.50] |
| activity | Q1 | FA_Q8_WARM_FROZEN | 6235 | 107.49 | [105.46, 109.47] |
| activity | Q1 | FA_1024_WARM_FROZEN | 6235 | 107.84 | [105.75, 109.78] |
| activity | Q1 | DP1_WARM_FROZEN | 6235 | 108.18 | [105.94, 110.39] |
| activity | Q1 | PRE | 6235 | 108.51 | [106.35, 110.80] |
| activity | Q1 | FT_C | 6235 | 96.36 | [94.92, 97.82] |
| activity | Q1 | FT_PRE | 6235 | 108.98 | [106.93, 110.89] |
| activity | Q1 | FT_FA | 6235 | 87.10 | [85.49, 88.69] |
| activity | Q1 | FT_FA_WARM | 6235 | 107.93 | [106.03, 109.74] |
| activity | Q1 | FT_FA_WARM_FROZEN | 6235 | 108.24 | [106.20, 110.11] |
| activity | Q1 | LO | 6235 | 0.05 | [0.01, 0.09] |
| activity | Q1 | KNN_EMB_SMALL | 6235 | 45.35 | [43.34, 47.32] |
| activity | Q1 | KNN_EMB_BIG | 6235 | 51.88 | [49.78, 54.06] |
| activity | Q1 | RULE | 6235 | 24.25 | [22.75, 25.93] |
| activity | Q1 | SKNN | 6235 | 75.12 | [73.44, 77.03] |
| activity | Q1 | LAST | 6235 | 48.71 | [46.84, 50.53] |
| activity | Q1 | POP | 6235 | 13.19 | [12.01, 14.47] |
| activity | Q2 | C_BIG | 6851 | 100.00 | [100.00, 100.00] |
| activity | Q2 | C_FULL | 6851 | 121.17 | [119.35, 123.08] |
| activity | Q2 | C_SMALL | 6851 | 94.27 | [92.95, 95.56] |
| activity | Q2 | FA | 6851 | 84.24 | [82.86, 85.73] |
| activity | Q2 | FP | 6851 | 84.26 | [82.86, 85.77] |
| activity | Q2 | PF | 6851 | 84.25 | [82.86, 85.75] |
| activity | Q2 | FA_1024 | 6851 | 22.97 | [21.55, 24.46] |
| activity | Q2 | DP8 | 6851 | 22.81 | [21.35, 24.34] |
| activity | Q2 | DP1 | 6851 | 22.25 | [20.82, 23.84] |
| activity | Q2 | FA_Q8 | 6851 | 84.45 | [83.00, 85.98] |
| activity | Q2 | FA_WARM | 6851 | 105.92 | [104.22, 107.66] |
| activity | Q2 | FA_WARM_FROZEN | 6851 | 106.85 | [105.07, 108.63] |
| activity | Q2 | DP8_FROZEN | 6851 | 106.89 | [105.10, 108.72] |
| activity | Q2 | FP_WARM_FROZEN | 6851 | 106.85 | [105.06, 108.63] |
| activity | Q2 | PF_WARM_FROZEN | 6851 | 106.77 | [105.02, 108.57] |
| activity | Q2 | FA_Q8_WARM_FROZEN | 6851 | 106.79 | [105.03, 108.59] |
| activity | Q2 | FA_1024_WARM_FROZEN | 6851 | 106.91 | [105.13, 108.74] |
| activity | Q2 | DP1_WARM_FROZEN | 6851 | 106.98 | [104.97, 109.10] |
| activity | Q2 | PRE | 6851 | 107.13 | [105.12, 109.29] |
| activity | Q2 | FT_C | 6851 | 95.23 | [93.94, 96.53] |
| activity | Q2 | FT_PRE | 6851 | 108.73 | [106.78, 110.51] |
| activity | Q2 | FT_FA | 6851 | 86.15 | [84.65, 87.56] |
| activity | Q2 | FT_FA_WARM | 6851 | 107.50 | [105.66, 109.38] |
| activity | Q2 | FT_FA_WARM_FROZEN | 6851 | 108.27 | [106.37, 110.11] |
| activity | Q2 | LO | 6851 | 0.53 | [0.34, 0.76] |
| activity | Q2 | KNN_EMB_SMALL | 6851 | 45.13 | [43.10, 47.20] |
| activity | Q2 | KNN_EMB_BIG | 6851 | 52.70 | [50.57, 54.84] |
| activity | Q2 | RULE | 6851 | 23.02 | [21.54, 24.48] |
| activity | Q2 | SKNN | 6851 | 74.70 | [72.95, 76.49] |
| activity | Q2 | LAST | 6851 | 51.76 | [49.99, 53.62] |
| activity | Q2 | POP | 6851 | 12.99 | [11.85, 14.29] |
| activity | Q3 | C_BIG | 10191 | 100.00 | [100.00, 100.00] |
| activity | Q3 | C_FULL | 10191 | 121.90 | [120.29, 123.54] |
| activity | Q3 | C_SMALL | 10191 | 94.50 | [93.47, 95.60] |
| activity | Q3 | FA | 10191 | 84.50 | [83.27, 85.69] |
| activity | Q3 | FP | 10191 | 84.50 | [83.26, 85.68] |
| activity | Q3 | PF | 10191 | 84.49 | [83.26, 85.67] |
| activity | Q3 | FA_1024 | 10191 | 20.25 | [19.04, 21.40] |
| activity | Q3 | DP8 | 10191 | 19.77 | [18.63, 20.89] |
| activity | Q3 | DP1 | 10191 | 19.00 | [17.87, 20.14] |
| activity | Q3 | FA_Q8 | 10191 | 84.63 | [83.35, 85.82] |
| activity | Q3 | FA_WARM | 10191 | 106.64 | [105.15, 108.16] |
| activity | Q3 | FA_WARM_FROZEN | 10191 | 107.48 | [105.88, 109.09] |
| activity | Q3 | DP8_FROZEN | 10191 | 107.43 | [105.85, 109.08] |
| activity | Q3 | FP_WARM_FROZEN | 10191 | 107.48 | [105.89, 109.10] |
| activity | Q3 | PF_WARM_FROZEN | 10191 | 107.30 | [105.70, 108.92] |
| activity | Q3 | FA_Q8_WARM_FROZEN | 10191 | 107.31 | [105.70, 108.90] |
| activity | Q3 | FA_1024_WARM_FROZEN | 10191 | 107.44 | [105.84, 109.08] |
| activity | Q3 | DP1_WARM_FROZEN | 10191 | 107.81 | [106.09, 109.58] |
| activity | Q3 | PRE | 10191 | 107.96 | [106.25, 109.78] |
| activity | Q3 | FT_C | 10191 | 97.27 | [96.25, 98.39] |
| activity | Q3 | FT_PRE | 10191 | 109.78 | [108.18, 111.48] |
| activity | Q3 | FT_FA | 10191 | 88.62 | [87.39, 89.96] |
| activity | Q3 | FT_FA_WARM | 10191 | 109.37 | [107.76, 111.08] |
| activity | Q3 | FT_FA_WARM_FROZEN | 10191 | 109.57 | [107.98, 111.25] |
| activity | Q3 | LO | 10191 | 5.28 | [4.72, 5.92] |
| activity | Q3 | KNN_EMB_SMALL | 10191 | 44.75 | [43.08, 46.32] |
| activity | Q3 | KNN_EMB_BIG | 10191 | 53.61 | [51.91, 55.42] |
| activity | Q3 | RULE | 10191 | 21.02 | [19.93, 22.17] |
| activity | Q3 | SKNN | 10191 | 74.33 | [72.93, 75.82] |
| activity | Q3 | LAST | 10191 | 52.45 | [51.08, 53.91] |
| activity | Q3 | POP | 10191 | 11.10 | [10.22, 12.01] |

### 5e. R per support-count bin (06A §2, observational)

User counts per bin: `{"MAIN": {"11-20": 6836, "21-50": 8136, "5-10": 7424, "51-100": 3858}, "TRUE_COLD": {"0": 325969}, "UNSEEN_HOLDOUT": {"11-20": 1819, "21-50": 2036, "5-10": 1810, "51-100": 985}}`

| role | bin | arm | study | users | R% vs C_BIG | 95% CI |
|---|---|---|---|---|---|---|
| MAIN | 11-20 | C_BIG | OTHER | 6836 | 100.00 | [100.00, 100.00] |
| MAIN | 11-20 | C_FULL | OTHER | 6836 | 122.41 | [120.49, 124.51] |
| MAIN | 11-20 | C_SMALL | A | 6836 | 94.43 | [93.15, 95.70] |
| MAIN | 11-20 | FA | A | 6836 | 85.08 | [83.60, 86.56] |
| MAIN | 11-20 | FP | OTHER | 6836 | 85.09 | [83.63, 86.59] |
| MAIN | 11-20 | PF | B | 6836 | 85.07 | [83.58, 86.55] |
| MAIN | 11-20 | FA_1024 | A | 6836 | 23.16 | [21.74, 24.60] |
| MAIN | 11-20 | DP8 | A | 6836 | 23.21 | [21.75, 24.65] |
| MAIN | 11-20 | DP1 | A | 6836 | 22.56 | [21.15, 24.15] |
| MAIN | 11-20 | FA_Q8 | OTHER | 6836 | 85.37 | [83.81, 86.86] |
| MAIN | 11-20 | FA_WARM | OTHER | 6836 | 107.38 | [105.58, 109.12] |
| MAIN | 11-20 | FA_WARM_FROZEN | OTHER | 6836 | 108.33 | [106.51, 110.08] |
| MAIN | 11-20 | DP8_FROZEN | OTHER | 6836 | 108.77 | [106.85, 110.60] |
| MAIN | 11-20 | FP_WARM_FROZEN | OTHER | 6836 | 108.32 | [106.49, 110.08] |
| MAIN | 11-20 | PF_WARM_FROZEN | OTHER | 6836 | 108.17 | [106.30, 109.90] |
| MAIN | 11-20 | FA_Q8_WARM_FROZEN | OTHER | 6836 | 108.17 | [106.30, 109.86] |
| MAIN | 11-20 | FA_1024_WARM_FROZEN | OTHER | 6836 | 108.78 | [106.88, 110.59] |
| MAIN | 11-20 | DP1_WARM_FROZEN | OTHER | 6836 | 109.18 | [107.10, 111.24] |
| MAIN | 11-20 | PRE | OTHER | 6836 | 109.39 | [107.30, 111.53] |
| MAIN | 11-20 | FT_C | B | 6836 | 95.51 | [94.27, 96.82] |
| MAIN | 11-20 | FT_PRE | OTHER | 6836 | 110.69 | [108.84, 112.55] |
| MAIN | 11-20 | FT_FA | B | 6836 | 86.90 | [85.35, 88.38] |
| MAIN | 11-20 | FT_FA_WARM | OTHER | 6836 | 108.88 | [107.03, 110.66] |
| MAIN | 11-20 | FT_FA_WARM_FROZEN | OTHER | 6836 | 109.66 | [107.87, 111.38] |
| MAIN | 11-20 | LO | B | 6836 | 0.28 | [0.16, 0.44] |
| MAIN | 11-20 | KNN_EMB_SMALL | OTHER | 6836 | 44.38 | [42.38, 46.28] |
| MAIN | 11-20 | KNN_EMB_BIG | OTHER | 6836 | 51.93 | [49.84, 54.11] |
| MAIN | 11-20 | RULE | OTHER | 6836 | 23.36 | [21.86, 24.91] |
| MAIN | 11-20 | SKNN | OTHER | 6836 | 74.77 | [73.02, 76.65] |
| MAIN | 11-20 | LAST | OTHER | 6836 | 51.27 | [49.54, 53.12] |
| MAIN | 11-20 | POP | OTHER | 6836 | 13.17 | [12.05, 14.44] |
| MAIN | 21-50 | C_BIG | OTHER | 8136 | 100.00 | [100.00, 100.00] |
| MAIN | 21-50 | C_FULL | OTHER | 8136 | 121.33 | [119.69, 123.05] |
| MAIN | 21-50 | C_SMALL | A | 8136 | 94.75 | [93.60, 95.90] |
| MAIN | 21-50 | FA | A | 8136 | 84.91 | [83.60, 86.32] |
| MAIN | 21-50 | FP | OTHER | 8136 | 84.90 | [83.59, 86.31] |
| MAIN | 21-50 | PF | B | 8136 | 84.92 | [83.60, 86.32] |
| MAIN | 21-50 | FA_1024 | A | 8136 | 22.24 | [20.73, 23.62] |
| MAIN | 21-50 | DP8 | A | 8136 | 21.61 | [20.14, 22.89] |
| MAIN | 21-50 | DP1 | A | 8136 | 20.76 | [19.30, 22.03] |
| MAIN | 21-50 | FA_Q8 | OTHER | 8136 | 85.00 | [83.67, 86.40] |
| MAIN | 21-50 | FA_WARM | OTHER | 8136 | 106.75 | [105.20, 108.35] |
| MAIN | 21-50 | FA_WARM_FROZEN | OTHER | 8136 | 107.74 | [106.13, 109.37] |
| MAIN | 21-50 | DP8_FROZEN | OTHER | 8136 | 107.68 | [105.99, 109.36] |
| MAIN | 21-50 | FP_WARM_FROZEN | OTHER | 8136 | 107.74 | [106.13, 109.37] |
| MAIN | 21-50 | PF_WARM_FROZEN | OTHER | 8136 | 107.49 | [105.87, 109.09] |
| MAIN | 21-50 | FA_Q8_WARM_FROZEN | OTHER | 8136 | 107.49 | [105.87, 109.11] |
| MAIN | 21-50 | FA_1024_WARM_FROZEN | OTHER | 8136 | 107.69 | [105.98, 109.38] |
| MAIN | 21-50 | DP1_WARM_FROZEN | OTHER | 8136 | 108.37 | [106.40, 110.29] |
| MAIN | 21-50 | PRE | OTHER | 8136 | 108.48 | [106.48, 110.36] |
| MAIN | 21-50 | FT_C | B | 8136 | 96.69 | [95.56, 97.90] |
| MAIN | 21-50 | FT_PRE | OTHER | 8136 | 109.40 | [107.65, 111.22] |
| MAIN | 21-50 | FT_FA | B | 8136 | 87.81 | [86.54, 89.13] |
| MAIN | 21-50 | FT_FA_WARM | OTHER | 8136 | 108.70 | [107.03, 110.43] |
| MAIN | 21-50 | FT_FA_WARM_FROZEN | OTHER | 8136 | 109.48 | [107.70, 111.32] |
| MAIN | 21-50 | LO | B | 8136 | 1.77 | [1.41, 2.14] |
| MAIN | 21-50 | KNN_EMB_SMALL | OTHER | 8136 | 44.44 | [42.66, 46.25] |
| MAIN | 21-50 | KNN_EMB_BIG | OTHER | 8136 | 53.18 | [51.26, 55.16] |
| MAIN | 21-50 | RULE | OTHER | 8136 | 21.08 | [19.80, 22.34] |
| MAIN | 21-50 | SKNN | OTHER | 8136 | 74.09 | [72.47, 75.60] |
| MAIN | 21-50 | LAST | OTHER | 8136 | 51.43 | [49.87, 53.08] |
| MAIN | 21-50 | POP | OTHER | 8136 | 12.01 | [10.95, 13.09] |
| MAIN | 5-10 | C_BIG | OTHER | 7424 | 100.00 | [100.00, 100.00] |
| MAIN | 5-10 | C_FULL | OTHER | 7424 | 119.55 | [117.71, 121.46] |
| MAIN | 5-10 | C_SMALL | A | 7424 | 94.87 | [93.57, 96.18] |
| MAIN | 5-10 | FA | A | 7424 | 85.95 | [84.32, 87.45] |
| MAIN | 5-10 | FP | OTHER | 7424 | 85.95 | [84.32, 87.46] |
| MAIN | 5-10 | PF | B | 7424 | 85.99 | [84.34, 87.54] |
| MAIN | 5-10 | FA_1024 | A | 7424 | 26.32 | [24.88, 27.89] |
| MAIN | 5-10 | DP8 | A | 7424 | 25.87 | [24.46, 27.37] |
| MAIN | 5-10 | DP1 | A | 7424 | 24.49 | [23.05, 25.98] |
| MAIN | 5-10 | FA_Q8 | OTHER | 7424 | 85.81 | [84.18, 87.39] |
| MAIN | 5-10 | FA_WARM | OTHER | 7424 | 106.08 | [104.50, 107.83] |
| MAIN | 5-10 | FA_WARM_FROZEN | OTHER | 7424 | 106.76 | [105.11, 108.53] |
| MAIN | 5-10 | DP8_FROZEN | OTHER | 7424 | 107.00 | [105.32, 108.81] |
| MAIN | 5-10 | FP_WARM_FROZEN | OTHER | 7424 | 106.76 | [105.11, 108.53] |
| MAIN | 5-10 | PF_WARM_FROZEN | OTHER | 7424 | 106.59 | [104.94, 108.32] |
| MAIN | 5-10 | FA_Q8_WARM_FROZEN | OTHER | 7424 | 106.58 | [104.91, 108.33] |
| MAIN | 5-10 | FA_1024_WARM_FROZEN | OTHER | 7424 | 107.00 | [105.33, 108.83] |
| MAIN | 5-10 | DP1_WARM_FROZEN | OTHER | 7424 | 106.99 | [105.12, 108.95] |
| MAIN | 5-10 | PRE | OTHER | 7424 | 107.14 | [105.30, 109.21] |
| MAIN | 5-10 | FT_C | B | 7424 | 95.91 | [94.54, 97.27] |
| MAIN | 5-10 | FT_PRE | OTHER | 7424 | 108.00 | [106.30, 109.83] |
| MAIN | 5-10 | FT_FA | B | 7424 | 86.98 | [85.35, 88.52] |
| MAIN | 5-10 | FT_FA_WARM | OTHER | 7424 | 107.24 | [105.66, 108.92] |
| MAIN | 5-10 | FT_FA_WARM_FROZEN | OTHER | 7424 | 107.41 | [105.61, 109.09] |
| MAIN | 5-10 | LO | B | 7424 | 0.03 | [0.01, 0.05] |
| MAIN | 5-10 | KNN_EMB_SMALL | OTHER | 7424 | 46.43 | [44.27, 48.37] |
| MAIN | 5-10 | KNN_EMB_BIG | OTHER | 7424 | 52.81 | [50.62, 54.77] |
| MAIN | 5-10 | RULE | OTHER | 7424 | 26.54 | [25.09, 27.97] |
| MAIN | 5-10 | SKNN | OTHER | 7424 | 74.74 | [73.03, 76.41] |
| MAIN | 5-10 | LAST | OTHER | 7424 | 48.30 | [46.62, 49.99] |
| MAIN | 5-10 | POP | OTHER | 7424 | 14.31 | [13.15, 15.61] |
| MAIN | 51-100 | C_BIG | OTHER | 3858 | 100.00 | [100.00, 100.00] |
| MAIN | 51-100 | C_FULL | OTHER | 3858 | 121.84 | [119.14, 124.64] |
| MAIN | 51-100 | C_SMALL | A | 3858 | 94.67 | [92.87, 96.44] |
| MAIN | 51-100 | FA | A | 3858 | 83.76 | [81.66, 85.89] |
| MAIN | 51-100 | FP | OTHER | 3858 | 83.77 | [81.67, 85.89] |
| MAIN | 51-100 | PF | B | 3858 | 83.73 | [81.62, 85.84] |
| MAIN | 51-100 | FA_1024 | A | 3858 | 17.05 | [15.43, 18.65] |
| MAIN | 51-100 | DP8 | A | 3858 | 16.87 | [15.36, 18.33] |
| MAIN | 51-100 | DP1 | A | 3858 | 16.28 | [14.68, 17.73] |
| MAIN | 51-100 | FA_Q8 | OTHER | 3858 | 83.99 | [81.81, 86.13] |
| MAIN | 51-100 | FA_WARM | OTHER | 3858 | 105.82 | [103.30, 108.58] |
| MAIN | 51-100 | FA_WARM_FROZEN | OTHER | 3858 | 106.21 | [103.73, 109.10] |
| MAIN | 51-100 | DP8_FROZEN | OTHER | 3858 | 105.87 | [103.34, 108.59] |
| MAIN | 51-100 | FP_WARM_FROZEN | OTHER | 3858 | 106.21 | [103.74, 109.10] |
| MAIN | 51-100 | PF_WARM_FROZEN | OTHER | 3858 | 106.25 | [103.78, 109.09] |
| MAIN | 51-100 | FA_Q8_WARM_FROZEN | OTHER | 3858 | 106.28 | [103.76, 109.04] |
| MAIN | 51-100 | FA_1024_WARM_FROZEN | OTHER | 3858 | 105.91 | [103.36, 108.64] |
| MAIN | 51-100 | DP1_WARM_FROZEN | OTHER | 3858 | 104.88 | [102.35, 107.82] |
| MAIN | 51-100 | PRE | OTHER | 3858 | 105.10 | [102.60, 108.04] |
| MAIN | 51-100 | FT_C | B | 3858 | 98.25 | [96.53, 99.99] |
| MAIN | 51-100 | FT_PRE | OTHER | 3858 | 109.05 | [106.34, 111.85] |
| MAIN | 51-100 | FT_FA | B | 3858 | 89.56 | [87.29, 91.70] |
| MAIN | 51-100 | FT_FA_WARM | OTHER | 3858 | 109.02 | [106.27, 111.82] |
| MAIN | 51-100 | FT_FA_WARM_FROZEN | OTHER | 3858 | 108.45 | [105.76, 111.22] |
| MAIN | 51-100 | LO | B | 3858 | 11.04 | [9.70, 12.52] |
| MAIN | 51-100 | KNN_EMB_SMALL | OTHER | 3858 | 46.47 | [43.82, 49.26] |
| MAIN | 51-100 | KNN_EMB_BIG | OTHER | 3858 | 54.86 | [51.96, 57.96] |
| MAIN | 51-100 | RULE | OTHER | 3858 | 20.94 | [19.17, 22.77] |
| MAIN | 51-100 | SKNN | OTHER | 3858 | 75.28 | [72.90, 77.82] |
| MAIN | 51-100 | LAST | OTHER | 3858 | 54.34 | [52.12, 57.04] |
| MAIN | 51-100 | POP | OTHER | 3858 | 9.36 | [8.25, 10.55] |
| TRUE_COLD | 0 | C_BIG | OTHER | 325969 | 100.00 | [100.00, 100.00] |
| TRUE_COLD | 0 | C_SMALL | A | 325969 | 95.12 | [94.89, 95.38] |
| TRUE_COLD | 0 | FA | A | 325969 | 85.71 | [85.41, 85.99] |
| TRUE_COLD | 0 | FA_WARM_FROZEN | OTHER | 325969 | 104.83 | [104.54, 105.10] |
| UNSEEN_HOLDOUT | 11-20 | C_BIG | OTHER | 1819 | 100.00 | [100.00, 100.00] |
| UNSEEN_HOLDOUT | 11-20 | C_FULL | OTHER | 1819 | 122.56 | [118.63, 126.80] |
| UNSEEN_HOLDOUT | 11-20 | C_SMALL | A | 1819 | 97.85 | [94.80, 101.16] |
| UNSEEN_HOLDOUT | 11-20 | FA | A | 1819 | 92.60 | [88.69, 96.53] |
| UNSEEN_HOLDOUT | 11-20 | FA_WARM_FROZEN | OTHER | 1819 | 107.64 | [104.43, 111.12] |
| UNSEEN_HOLDOUT | 11-20 | FT_C | B | 1819 | 99.00 | [95.77, 102.23] |
| UNSEEN_HOLDOUT | 11-20 | LO | B | 1819 | 0.19 | [0.03, 0.45] |
| UNSEEN_HOLDOUT | 21-50 | C_BIG | OTHER | 2036 | 100.00 | [100.00, 100.00] |
| UNSEEN_HOLDOUT | 21-50 | C_FULL | OTHER | 2036 | 123.17 | [119.23, 127.77] |
| UNSEEN_HOLDOUT | 21-50 | C_SMALL | A | 2036 | 95.27 | [92.08, 98.10] |
| UNSEEN_HOLDOUT | 21-50 | FA | A | 2036 | 86.22 | [82.69, 89.55] |
| UNSEEN_HOLDOUT | 21-50 | FA_WARM_FROZEN | OTHER | 2036 | 108.03 | [104.47, 111.91] |
| UNSEEN_HOLDOUT | 21-50 | FT_C | B | 2036 | 98.88 | [95.41, 102.15] |
| UNSEEN_HOLDOUT | 21-50 | LO | B | 2036 | 2.18 | [1.25, 3.38] |
| UNSEEN_HOLDOUT | 5-10 | C_BIG | OTHER | 1810 | 100.00 | [100.00, 100.00] |
| UNSEEN_HOLDOUT | 5-10 | C_FULL | OTHER | 1810 | 119.55 | [115.36, 123.80] |
| UNSEEN_HOLDOUT | 5-10 | C_SMALL | A | 1810 | 94.70 | [91.27, 97.86] |
| UNSEEN_HOLDOUT | 5-10 | FA | A | 1810 | 86.48 | [82.41, 90.23] |
| UNSEEN_HOLDOUT | 5-10 | FA_WARM_FROZEN | OTHER | 1810 | 107.55 | [103.80, 111.71] |
| UNSEEN_HOLDOUT | 5-10 | FT_C | B | 1810 | 94.61 | [91.20, 97.84] |
| UNSEEN_HOLDOUT | 5-10 | LO | B | 1810 | 0.09 | [0.00, 0.22] |
| UNSEEN_HOLDOUT | 51-100 | C_BIG | OTHER | 985 | 100.00 | [100.00, 100.00] |
| UNSEEN_HOLDOUT | 51-100 | C_FULL | OTHER | 985 | 120.50 | [113.92, 127.89] |
| UNSEEN_HOLDOUT | 51-100 | C_SMALL | A | 985 | 94.54 | [89.91, 99.58] |
| UNSEEN_HOLDOUT | 51-100 | FA | A | 985 | 83.91 | [79.05, 88.69] |
| UNSEEN_HOLDOUT | 51-100 | FA_WARM_FROZEN | OTHER | 985 | 108.80 | [102.49, 115.54] |
| UNSEEN_HOLDOUT | 51-100 | FT_C | B | 985 | 98.29 | [92.92, 103.95] |
| UNSEEN_HOLDOUT | 51-100 | LO | B | 985 | 11.20 | [7.93, 14.96] |

Events-to-90% / 95% (smallest bin with M >= X C_ref; bin 0 from TRUE_COLD; no interpolation):

| panel | arm | seed | events-to-90% | events-to-95% |
|---|---|---|---|---|
| MAIN | C_BIG | 2026 | 0 | 0 |
| MAIN | C_BIG | 2027 | 5-10 | 5-10 |
| MAIN | C_FULL | 2026 | 5-10 | 5-10 |
| MAIN | C_FULL | 2027 | 5-10 | 5-10 |
| MAIN | C_SMALL | 2026 | 0 | 0 |
| MAIN | C_SMALL | 2027 | 5-10 | NOT_REACHED |
| MAIN | FA | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | FA | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | FP | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | FP | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | PF | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | PF | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | FA_1024 | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | FA_1024 | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | DP8 | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | DP8 | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | DP1 | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | FA_Q8 | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | FA_Q8 | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | FA_WARM | 2026 | 5-10 | 5-10 |
| MAIN | FA_WARM | 2027 | 5-10 | 5-10 |
| MAIN | FA_WARM_FROZEN | 2026 | 0 | 0 |
| MAIN | FA_WARM_FROZEN | 2027 | 5-10 | 5-10 |
| MAIN | DP8_FROZEN | 2026 | 5-10 | 5-10 |
| MAIN | DP8_FROZEN | 2027 | 5-10 | 5-10 |
| MAIN | FP_WARM_FROZEN | 2026 | 5-10 | 5-10 |
| MAIN | FP_WARM_FROZEN | 2027 | 5-10 | 5-10 |
| MAIN | PF_WARM_FROZEN | 2026 | 5-10 | 5-10 |
| MAIN | PF_WARM_FROZEN | 2027 | 5-10 | 5-10 |
| MAIN | FA_Q8_WARM_FROZEN | 2026 | 5-10 | 5-10 |
| MAIN | FA_Q8_WARM_FROZEN | 2027 | 5-10 | 5-10 |
| MAIN | FA_1024_WARM_FROZEN | 2026 | 5-10 | 5-10 |
| MAIN | FA_1024_WARM_FROZEN | 2027 | 5-10 | 5-10 |
| MAIN | DP1_WARM_FROZEN | 2026 | 5-10 | 5-10 |
| MAIN | PRE | 2026 | 5-10 | 5-10 |
| MAIN | FT_C | 2026 | 5-10 | 5-10 |
| MAIN | FT_C | 2027 | 5-10 | 5-10 |
| MAIN | FT_PRE | 2026 | 5-10 | 5-10 |
| MAIN | FT_PRE | 2027 | 5-10 | 5-10 |
| MAIN | FT_FA | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | FT_FA | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | FT_FA_WARM | 2026 | 5-10 | 5-10 |
| MAIN | FT_FA_WARM | 2027 | 5-10 | 5-10 |
| MAIN | FT_FA_WARM_FROZEN | 2026 | 5-10 | 5-10 |
| MAIN | FT_FA_WARM_FROZEN | 2027 | 5-10 | 5-10 |
| MAIN | LO | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | LO | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | KNN_EMB_SMALL | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | KNN_EMB_SMALL | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | KNN_EMB_BIG | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | KNN_EMB_BIG | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | RULE | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | RULE | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | SKNN | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | SKNN | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | LAST | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | LAST | 2027 | NOT_REACHED | NOT_REACHED |
| MAIN | POP | 2026 | NOT_REACHED | NOT_REACHED |
| MAIN | POP | 2027 | NOT_REACHED | NOT_REACHED |
| UNSEEN_HOLDOUT | C_BIG | 2026 | 0 | 0 |
| UNSEEN_HOLDOUT | C_FULL | 2026 | 5-10 | 5-10 |
| UNSEEN_HOLDOUT | C_SMALL | 2026 | 0 | 0 |
| UNSEEN_HOLDOUT | FA | 2026 | 11-20 | NOT_REACHED |
| UNSEEN_HOLDOUT | FA_WARM_FROZEN | 2026 | 0 | 0 |
| UNSEEN_HOLDOUT | FT_C | 2026 | 5-10 | 11-20 |
| UNSEEN_HOLDOUT | LO | 2026 | NOT_REACHED | NOT_REACHED |

### 5f. Controlled cold start (DATA-07 per-h rows; contract §11.7; C_ref = C_BIG FULL history)

| arm | panel | panel type | seed | h | M | C_ref | R% | M (FULL history) |
|---|---|---|---|---|---|---|---|---|

| arm | panel | panel type | seed | events-to-90% (h) | events-to-95% (h) | M (FULL history, E11.10) |
|---|---|---|---|---|---|---|

06L-C §C10 beside the registered first-hit rule: matched-h parity P(h) = M(h) / C_BIG(h) and the persistence rule. 06L-C §C10 persistence rule (beside the registered first-hit rule): the smallest registered h of the series' panel type such that at h AND at the next registered h the point ratio is >= X and the 95% paired user-bootstrap percentile CI lower bound of the ratio (1,000 resamples, seed 20260926, one W per user set, the ratio per resample) is >= 100 X; the largest registered h is never a persistent hit (it has no next h). Statuses: NO_C_REF (a reference row is missing before a hit), NO_NEXT_H (the point holds at h but the next registered h has no row), NO_CI (the points hold at h and at the next h but a CI is not computable: no per-user evidence for a cell, or UNDEFINED), NOT_REACHED (every h decided and none persists). Applied to R(h) = M(h) / C_ref (C_BIG FULL history) and to the matched-h parity P(h) = M(h) / C_BIG(h). DESCRIPTIVE; an 'events to parity' sentence needs a persistent hit.

| arm | panel | panel type | seed | h | R% | R 95% CI | C_BIG(h) | P% (matched h) | P 95% CI |
|---|---|---|---|---|---|---|---|---|---|

| arm | panel | panel type | seed | R: persistent 90% | R: persistent 95% | P: first-hit 90% | P: first-hit 95% | P: persistent 90% | P: persistent 95% |
|---|---|---|---|---|---|---|---|---|---|

## 6. Other panels — DESCRIPTIVE

| panel | rule | arm | R% vs C_BIG (seed mean) |
|---|---|---|---|
| HEAVY_PANEL | PRACTICAL_BEST | C_BIG | 100.00 [100.00, 100.00] |
| HEAVY_PANEL | PRACTICAL_BEST | C_FULL | 131.80 [129.76, 134.09] |
| HEAVY_PANEL | PRACTICAL_BEST | C_SMALL | 95.56 [94.28, 96.93] |
| HEAVY_PANEL | PRACTICAL_BEST | FA | 84.88 [83.41, 86.46] |
| HEAVY_PANEL | PRACTICAL_BEST | FA_WARM_FROZEN | 115.01 [113.22, 116.83] |
| HEAVY_PANEL | CONTROLLED_BUDGET | C_FULL | - - |
| HEAVY_PANEL | CONTROLLED_BUDGET | DP8 | - - |
| HEAVY_PANEL | CONTROLLED_BUDGET | DP8_FROZEN | - - |
| HEAVY_PANEL | CONTROLLED_BUDGET | PRE | - - |

| panel | rule | arm | R% vs C_BIG (seed mean) |
|---|---|---|---|
| SPARSE_PANEL | PRACTICAL_BEST | C_BIG | 100.00 [100.00, 100.00] |
| SPARSE_PANEL | PRACTICAL_BEST | C_FULL | 119.75 [116.62, 122.94] |
| SPARSE_PANEL | PRACTICAL_BEST | C_SMALL | 93.32 [91.04, 95.58] |
| SPARSE_PANEL | PRACTICAL_BEST | FA | 85.87 [83.40, 88.37] |
| SPARSE_PANEL | PRACTICAL_BEST | FA_WARM_FROZEN | 105.40 [102.89, 108.08] |
| SPARSE_PANEL | CONTROLLED_BUDGET | C_FULL | - - |
| SPARSE_PANEL | CONTROLLED_BUDGET | DP8 | - - |
| SPARSE_PANEL | CONTROLLED_BUDGET | DP8_FROZEN | - - |
| SPARSE_PANEL | CONTROLLED_BUDGET | PRE | - - |

| panel | rule | arm | R% vs C_BIG (seed mean) |
|---|---|---|---|
| TRUE_COLD | PRACTICAL_BEST | C_BIG | 100.00 [100.00, 100.00] |
| TRUE_COLD | PRACTICAL_BEST | C_SMALL | 95.12 [94.89, 95.38] |
| TRUE_COLD | PRACTICAL_BEST | FA | 85.71 [85.41, 85.99] |
| TRUE_COLD | PRACTICAL_BEST | FA_WARM_FROZEN | 104.83 [104.54, 105.10] |
| TRUE_COLD | CONTROLLED_BUDGET | DP8 | - - |
| TRUE_COLD | CONTROLLED_BUDGET | DP8_FROZEN | - - |

| panel | rule | arm | R% vs C_BIG (seed mean) |
|---|---|---|---|
| UNSEEN_HOLDOUT | PRACTICAL_BEST | C_BIG | 100.00 [100.00, 100.00] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | C_FULL | 121.59 [119.08, 123.94] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | C_SMALL | 95.76 [93.86, 97.53] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | FA | 87.86 [85.91, 89.83] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | FA_WARM_FROZEN | 107.87 [105.75, 109.76] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | FT_C | 97.60 [95.75, 99.55] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | LO | 2.18 [1.65, 2.83] |
| UNSEEN_HOLDOUT | CONTROLLED_BUDGET | C_FULL | - - |
| UNSEEN_HOLDOUT | CONTROLLED_BUDGET | DP8 | - - |
| UNSEEN_HOLDOUT | CONTROLLED_BUDGET | DP8_FROZEN | - - |
| UNSEEN_HOLDOUT | CONTROLLED_BUDGET | PRE | - - |

## 7. VALIDATION -> TEST shift indicator (06B §7b) — DESCRIPTIVE

```
{
 "by_rule": {
  "CONTROLLED_BUDGET": {
   "2026": {
    "delta": -0.013907281081701167,
    "test": 0.1389626760661601,
    "test_over_validation_pct": 90.902541388005,
    "validation": 0.15286995714786128
   },
   "2027": {
    "delta": -0.01415727793470925,
    "test": 0.13842166921409152,
    "test_over_validation_pct": 90.72134249235411,
    "validation": 0.15257894714880077
   }
  },
  "PRACTICAL_BEST": {
   "2026": {
    "delta": -0.012736181175988487,
    "test": 0.14845994856588937,
    "test_over_validation_pct": 92.0989534944897,
    "validation": 0.16119612974187786
   },
   "2027": {
    "delta": -0.012412702145668864,
    "test": 0.14880808079515087,
    "test_over_validation_pct": 92.30080519443622,
    "validation": 0.16122078294081973
   }
  }
 },
 "label": "DESCRIPTIVE (shift indicator, 06B \u00a77b)"
}
```

## 8. LAST anchor and the device-only envelope (06L-C C3) — DESCRIPTIVE (K11b is in section 1)

- anchor: `LAST|None|RULE_FREE|MAIN`; BELOW_DEVICE_HEURISTIC = the 95% CI of Δ(arm - LAST) lies below 0 (06L-C §C3).

| arm | rule | Δ(arm − LAST) [CI] | BELOW_DEVICE_HEURISTIC | R* vs C_BIG % [CI] | R* vs C_FULL % [CI] | harmed share vs LAST |
|---|---|---|---|---|---|---|
| C_BIG | PRACTICAL_BEST | 0.073065 [0.070889, 0.075079] | False | NOT_APPLICABLE | - | 0.2051 |
| C_BIG | CONTROLLED_BUDGET | 0.063123 [0.060906, 0.065023] | False | NOT_APPLICABLE | - | 0.2221 |
| C_FULL | PRACTICAL_BEST | 0.104497 [0.102164, 0.106648] | False | 143.02 [140.71, 145.43] | - | 0.1436 |
| C_FULL | CONTROLLED_BUDGET | 0.104497 [0.102164, 0.106648] | False | 165.55 [162.24, 169.17] | - | 0.1436 |
| C_SMALL | PRACTICAL_BEST | 0.065174 [0.062865, 0.067236] | False | 89.20 [87.79, 90.50] | - | 0.2290 |
| C_SMALL | CONTROLLED_BUDGET | 0.062798 [0.060388, 0.064883] | False | 99.48 [97.59, 101.26] | - | 0.2325 |
| FA | PRACTICAL_BEST | 0.050933 [0.048639, 0.052886] | False | 69.71 [67.84, 71.49] | - | 0.2618 |
| FA | CONTROLLED_BUDGET | 0.050933 [0.048639, 0.052886] | False | 80.69 [78.44, 82.75] | - | 0.2618 |
| FP | PRACTICAL_BEST | 0.050937 [0.048640, 0.052900] | False | 69.71 [67.85, 71.50] | - | 0.2618 |
| FP | CONTROLLED_BUDGET | 0.050937 [0.048640, 0.052900] | False | 80.70 [78.46, 82.75] | - | 0.2618 |
| PF | PRACTICAL_BEST | 0.050948 [0.048681, 0.052934] | False | 69.73 [67.88, 71.49] | - | 0.2617 |
| PF | CONTROLLED_BUDGET | 0.050948 [0.048681, 0.052934] | False | 80.71 [78.48, 82.80] | - | 0.2617 |
| FA_1024 | PRACTICAL_BEST | -0.041371 [-0.043070, -0.039629] | True | -56.62 [-60.30, -53.42] | - | 0.4925 |
| FA_1024 | CONTROLLED_BUDGET | -0.041371 [-0.043070, -0.039629] | True | -65.54 [-69.97, -61.62] | - | 0.4925 |
| DP8 | PRACTICAL_BEST | -0.041879 [-0.043609, -0.040201] | True | -57.32 [-60.86, -54.06] | - | 0.4934 |
| DP8 | CONTROLLED_BUDGET | -0.041879 [-0.043609, -0.040201] | True | -66.35 [-70.67, -62.42] | - | 0.4934 |
| DP1 | PRACTICAL_BEST | -0.043293 [-0.044941, -0.041616] | True | -59.39 [-62.85, -56.12] | - | 0.4950 |
| DP1 | CONTROLLED_BUDGET | -0.043293 [-0.044941, -0.041616] | True | -68.29 [-72.76, -64.42] | - | 0.4950 |
| FA_Q8 | PRACTICAL_BEST | 0.051069 [0.048787, 0.053065] | False | 69.90 [68.04, 71.70] | - | 0.2620 |
| FA_Q8 | CONTROLLED_BUDGET | 0.051069 [0.048787, 0.053065] | False | 80.90 [78.69, 83.02] | - | 0.2620 |
| FA_WARM | PRACTICAL_BEST | 0.082859 [0.080495, 0.085009] | False | 113.40 [111.50, 115.24] | 79.05 [77.77, 80.24] | 0.1902 |
| FA_WARM | CONTROLLED_BUDGET | 0.081931 [0.079537, 0.084085] | False | 129.80 [127.16, 132.41] | 78.16 [76.80, 79.36] | 0.1938 |
| FA_WARM_FROZEN | PRACTICAL_BEST | 0.084052 [0.081641, 0.086233] | False | 115.04 [113.18, 117.03] | 80.19 [78.88, 81.40] | 0.1896 |
| FA_WARM_FROZEN | CONTROLLED_BUDGET | 0.083733 [0.081348, 0.085949] | False | 132.65 [129.97, 135.42] | 79.88 [78.58, 81.07] | 0.1898 |
| DP8_FROZEN | PRACTICAL_BEST | 0.084240 [0.081800, 0.086420] | False | 115.30 [113.34, 117.27] | 80.37 [79.06, 81.60] | 0.1896 |
| DP8_FROZEN | CONTROLLED_BUDGET | 0.083886 [0.081493, 0.086066] | False | 132.90 [130.23, 135.68] | 80.03 [78.75, 81.31] | 0.1891 |
| FP_WARM_FROZEN | PRACTICAL_BEST | 0.084050 [0.081644, 0.086227] | False | 115.04 [113.18, 117.03] | 80.19 [78.88, 81.40] | 0.1896 |
| FP_WARM_FROZEN | CONTROLLED_BUDGET | 0.083741 [0.081356, 0.085958] | False | 132.67 [129.99, 135.43] | 79.89 [78.60, 81.09] | 0.1898 |
| PF_WARM_FROZEN | PRACTICAL_BEST | 0.083810 [0.081374, 0.086011] | False | 114.71 [112.82, 116.74] | 79.96 [78.65, 81.21] | 0.1904 |
| PF_WARM_FROZEN | CONTROLLED_BUDGET | 0.083800 [0.081412, 0.086025] | False | 132.76 [130.06, 135.53] | 79.95 [78.66, 81.14] | 0.1898 |
| FA_Q8_WARM_FROZEN | PRACTICAL_BEST | 0.083810 [0.081376, 0.085993] | False | 114.71 [112.83, 116.73] | 79.96 [78.66, 81.22] | 0.1904 |
| FA_Q8_WARM_FROZEN | CONTROLLED_BUDGET | 0.083763 [0.081373, 0.085981] | False | 132.70 [130.00, 135.46] | 79.91 [78.61, 81.12] | 0.1898 |
| FA_1024_WARM_FROZEN | PRACTICAL_BEST | 0.084252 [0.081788, 0.086430] | False | 115.31 [113.37, 117.30] | 80.38 [79.08, 81.61] | 0.1898 |
| FA_1024_WARM_FROZEN | CONTROLLED_BUDGET | 0.083881 [0.081474, 0.086082] | False | 132.89 [130.16, 135.72] | 80.02 [78.73, 81.26] | 0.1893 |
| DP1_WARM_FROZEN | PRACTICAL_BEST | 0.084311 [0.081841, 0.086507] | False | 115.67 [113.53, 117.82] | 80.43 [79.13, 81.65] | 0.1892 |
| DP1_WARM_FROZEN | CONTROLLED_BUDGET | 0.083955 [0.081594, 0.086157] | False | 132.44 [129.32, 135.49] | 80.10 [78.82, 81.35] | 0.1884 |
| PRE | PRACTICAL_BEST | 0.084560 [0.082171, 0.086712] | False | 116.01 [113.91, 118.08] | 80.67 [79.40, 81.92] | 0.1889 |
| PRE | CONTROLLED_BUDGET | 0.084560 [0.082171, 0.086712] | False | 133.39 [130.27, 136.45] | 80.67 [79.40, 81.92] | 0.1889 |
| FT_C | PRACTICAL_BEST | 0.067644 [0.065341, 0.069785] | False | 92.58 [91.30, 93.93] | - | 0.2215 |
| FT_PRE | PRACTICAL_BEST | 0.086843 [0.084467, 0.088990] | False | 118.86 [116.79, 121.09] | 82.85 [81.49, 84.16] | 0.1804 |
| FT_FA | PRACTICAL_BEST | 0.054570 [0.052302, 0.056568] | False | 74.69 [72.92, 76.34] | - | 0.2527 |
| FT_FA_WARM | PRACTICAL_BEST | 0.085476 [0.083120, 0.087594] | False | 116.99 [114.94, 118.99] | 81.55 [80.25, 82.77] | 0.1829 |
| FT_FA_WARM_FROZEN | PRACTICAL_BEST | 0.086096 [0.083793, 0.088266] | False | 117.84 [115.83, 119.82] | 82.14 [80.80, 83.42] | 0.1823 |
| LO | PRACTICAL_BEST | -0.072445 [-0.073673, -0.071211] | True | -99.15 [-103.07, -95.80] | - | 0.5690 |
| C_SMALL_FP32 | PRACTICAL_BEST | 0.064924 [0.061929, 0.067740] | False | 90.19 [87.83, 92.47] | - | 0.2376 |
| C_SMALL_INT8 | PRACTICAL_BEST | 0.064898 [0.061946, 0.067717] | False | 90.16 [87.85, 92.47] | - | 0.2371 |
| KNN_EMB_SMALL | RULE_FREE | -0.008252 [-0.009594, -0.006856] | True | -11.29 [-13.28, -9.26] | - | 0.3623 |
| KNN_EMB_BIG | RULE_FREE | 0.003154 [0.001744, 0.004500] | False | 4.32 [2.39, 6.20] | - | 0.3199 |
| RULE | RULE_FREE | -0.040949 [-0.042456, -0.039446] | True | -56.05 [-59.25, -53.17] | - | 0.4770 |
| SKNN | RULE_FREE | 0.035341 [0.033764, 0.036814] | False | 48.37 [46.73, 49.99] | - | 0.2217 |
| POP | RULE_FREE | -0.056773 [-0.058247, -0.055303] | True | -77.70 [-81.54, -74.47] | - | 0.5286 |

### device-only envelope {LO, LAST, RULE} (06L-C §C3)

| arm | MRR@20 (seed mean) | R% vs C_BIG [CI] | Δ vs LAST [CI] |
|---|---|---|---|
| LO | 0.003124 | 2.10 [1.88, 2.34] | -0.072445 [-0.073673, -0.071211] |
| LAST | 0.075569 | 50.84 [49.97, 51.83] | ANCHOR |
| RULE | 0.034620 | 23.29 [22.57, 24.02] | -0.040949 [-0.042456, -0.039446] |

- best point (DESCRIPTIVE, never a selection): {"arm": "LAST", "value": 0.0755693057940293}; LO is an untuned from-θ0 control, not the device-only ceiling (review 10 [9-R1 (d)]).

## 9. Privacy matrix (06L-C C1) and one bytes axis (06L-C C8) — DESCRIPTIVE

| level | arm | R% vs C_BIG (PB, seed mean) | bytes/client | server sees (training) | server sees (inference) | formal guarantee | against whom | secure aggregation |
|---|---|---|---|---|---|---|---|---|
| L0 cloud-train / cloud-infer | C_BIG | 100.00 [100.00, 100.00] | - | raw events | query context | none | - | n/a (no federated training) |
| L0 cloud-train / cloud-infer | C_FULL | 121.15 [120.19, 122.07] | - | raw events | query context | none | - | n/a (no federated training) |
| L0 cloud-train / cloud-infer | C_SMALL | 94.69 [94.03, 95.31] | - | raw events | query context | none | - | n/a (no federated training) |
| L0 cloud-train / cloud-infer | SKNN | 74.62 [73.80, 75.52] | - | raw events | query context | none | - | n/a (no federated training) |
| L1 cloud-train / device-infer | C_SMALL_FP32 | 95.21 [94.01, 96.34] | - | raw events (model trained centrally once) | nothing | none | - | n/a (no federated training) |
| L1 cloud-train / device-infer | C_SMALL_INT8 | 95.20 [94.02, 96.35] | - | raw events (model trained centrally once) | nothing | none | - | n/a (no federated training) |
| L1 cloud-train / device-infer | KNN_EMB_SMALL | 45.29 [44.28, 46.37] | - | raw events (model trained centrally once) | nothing | none | - | n/a (no federated training) |
| L1 cloud-train / device-infer | KNN_EMB_BIG | 52.96 [51.89, 54.13] | - | raw events (model trained centrally once) | nothing | none | - | n/a (no federated training) |
| L1 on opted-in data (PRE-alone) | PRE | 107.86 [106.85, 108.84] | - | raw events of the opted-in users only (PRETRAIN_TRAIN_5_100_NONMAIN); none of the MAIN users | nothing | none (the MAIN users' data is never used) | - | n/a (no federated training) |
| L1 base (server-trained on raw events) + local fine-tune | FT_C | 96.35 [95.70, 97.02] | - | raw events of the same users (the base) and nothing after the download | nothing | none | - | n/a (no federated training) |
| L2 federated | FA | 85.11 [84.30, 85.90] | - | model deltas; non-DP deltas reveal the user's viewed items (input-table rows and output rows move; 06I §7(a), 06L-C §C1) | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L2 federated | FP | 85.11 [84.30, 85.90] | - | model deltas; non-DP deltas reveal the user's viewed items (input-table rows and output rows move; 06I §7(a), 06L-C §C1) | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L2 federated | FA_1024 | 23.01 [22.24, 23.76] | - | model deltas; non-DP deltas reveal the user's viewed items (input-table rows and output rows move; 06I §7(a), 06L-C §C1) | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L2 federated | FA_Q8 | 85.20 [84.36, 85.99] | - | model deltas; non-DP deltas reveal the user's viewed items (input-table rows and output rows move; 06I §7(a), 06L-C §C1) | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L2 + consenting-cohort pretraining | FA_WARM | 106.59 [105.67, 107.43] | - | model deltas; non-DP deltas reveal the user's viewed items (input-table rows and output rows move; 06I §7(a), 06L-C §C1); θ_pre from opted-in users | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L2 + frozen item tables (S8) | FA_WARM_FROZEN | 107.39 [106.50, 108.32] | - | encoder deltas only (265,162 uploaded parameters, to be re-measured); the input-table item-set leak is removed only partly: brand/category deltas may still leak, off-support rows move ≈ 1e-9, not exactly 0 (06L-C §C8) | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L3 federated + DP | DP8 | 22.67 [21.91, 23.47] | - | clipped deltas + Gaussian noise on the sum | nothing | central DP with a trusted aggregator, client-level add/remove adjacency, (ε, δ = 5e-6), conditional on the public bootstrap block (catalogue, transform statistics, θ0 / θ_pre from opted-in users); the FL hyper-parameter tuning on CALIB is not accounted (06L-C §C2) | anyone who sees the released model; not the trusted aggregator itself | secure aggregation: not simulated |
| L3 federated + DP | DP1 | 21.74 [20.96, 22.46] | - | clipped deltas + Gaussian noise on the sum | nothing | central DP with a trusted aggregator, client-level add/remove adjacency, (ε, δ = 5e-6), conditional on the public bootstrap block (catalogue, transform statistics, θ0 / θ_pre from opted-in users); the FL hyper-parameter tuning on CALIB is not accounted (06L-C §C2) | anyone who sees the released model; not the trusted aggregator itself | secure aggregation: not simulated |
| L3 federated + DP | DP8_FROZEN | 107.52 [106.61, 108.44] | - | clipped deltas + Gaussian noise on the sum | nothing | central DP with a trusted aggregator, client-level add/remove adjacency, (ε, δ = 5e-6), conditional on the public bootstrap block (catalogue, transform statistics, θ0 / θ_pre from opted-in users); the FL hyper-parameter tuning on CALIB is not accounted (06L-C §C2) | anyone who sees the released model; not the trusted aggregator itself | secure aggregation: not simulated |
| L4 federated + local personalisation | PF | 85.12 [84.32, 85.90] | - | as L2; personal parameters never leave | nothing | none beyond L2 | the honest-but-curious browser vendor sees every shared delta | secure aggregation: not simulated |
| L4 federated + local personalisation | FT_FA | 87.56 [86.75, 88.33] | - | as L2; personal parameters never leave | nothing | none beyond L2 | the honest-but-curious browser vendor sees every shared delta | secure aggregation: not simulated |
| L4 federated + local personalisation | FT_FA_WARM | 108.35 [107.40, 109.24] | - | as L2; personal parameters never leave | nothing | none beyond L2 | the honest-but-curious browser vendor sees every shared delta | secure aggregation: not simulated |
| L4 federated + local personalisation | FT_FA_WARM_FROZEN | 108.77 [107.80, 109.67] | - | as L2; personal parameters never leave | nothing | none beyond L2 | the honest-but-curious browser vendor sees every shared delta | secure aggregation: not simulated |
| L5 (base trained only on opted-in users) + local fine-tune | FT_PRE | 109.27 [108.27, 110.25] | - | nothing of the MAIN users (θ_pre from opted-in users; download only) | nothing | none needed for the MAIN users (their data never leaves the device) | - | n/a (no federated training) |
| L6 device-only | LO | 2.10 [1.88, 2.34] | 0 | nothing | nothing | data never leaves (shared catalogue / θ0 only) | - | n/a (no federated training) |
| L6 device-only | LAST | 50.84 [49.97, 51.83] | 0 | nothing | nothing | data never leaves (shared catalogue / θ0 only) | - | n/a (no federated training) |
| device + published aggregate | RULE | 23.29 [22.57, 24.02] | - | nothing of the user (a published MAIN-TRAIN popularity / category aggregate is downloaded) | nothing | none for the users in the aggregate | - | n/a (no federated training) |
| device + published aggregate | POP | 12.65 [12.05, 13.26] | - | nothing of the user (a published MAIN-TRAIN popularity / category aggregate is downloaded) | nothing | none for the users in the aggregate | - | n/a (no federated training) |

### Bytes axis (raw events → model download → FL traffic → S8)

| step | level | arm | bytes/client | bytes/client incl. final download | status | rule |
|---|---|---|---|---|---|---|
| 0 | device-only | LO | 0 | - | COMPUTED | no upload; only the shared bootstrap block (catalogue / θ0) |
| 0 | device + published aggregate | RULE | - | - | AGGREGATE_SIZE_NOT_MEASURED | the published MAIN-TRAIN popularity / category aggregate is downloaded (size not measured) |
| 0 | device-only | LAST | 0 | - | COMPUTED | no upload; only the shared bootstrap block (catalogue / θ0) |
| 1 | raw events to the server (L0 / L1 cloud training) | C_BIG / C_SMALL / C_FULL | - | - | NOT_REGISTERED (06L-C §C8: the per-event size convention needs a DL row; records.EVENT_SIZE) | bytes_per_event (records.EVENT_SIZE) x mean TRAIN decisions per MAIN client (TRAIN manifest) |
| 2 | model download FP32 (L1 device inference; L5 / FT base; PRE-alone) | C_SMALL | - | - | NO_MODEL_BYTES | the model file bytes (entry meta model_bytes), downloaded once |
| 2 | model download INT8 (ONNX dynamic quantisation) | C_SMALL_INT8 | - | - | NO_MODEL_BYTES | the model file bytes (entry meta model_bytes), downloaded once |
| 3 | FL training traffic (+ the final download) | FA | - | - | NO_BYTES_RECORD | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | FP | - | - | NO_BYTES_RECORD | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | PF | - | - | NO_BYTES_RECORD | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | FA_1024 | - | - | NO_BYTES_RECORD | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | DP8 | - | - | NO_BYTES_RECORD | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | DP1 | - | - | NO_BYTES_RECORD | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | FA_Q8 | - | - | NO_BYTES_RECORD | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | FA_WARM | - | - | NO_BYTES_RECORD | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 4 | S8: frozen tables once + encoder rounds | FA_WARM_FROZEN | - | - | NO_BYTES_RECORD | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | DP8_FROZEN | - | - | NO_BYTES_RECORD | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | FP_WARM_FROZEN | - | - | NO_BYTES_RECORD | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | PF_WARM_FROZEN | - | - | NO_BYTES_RECORD | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | FA_Q8_WARM_FROZEN | - | - | NO_BYTES_RECORD | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | FA_1024_WARM_FROZEN | - | - | NO_BYTES_RECORD | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | DP1_WARM_FROZEN | - | - | NO_BYTES_RECORD | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |

## 10. Labelled rows (06L-C C11, C16-C17) — DESCRIPTIVE

### 10a. DP8_FROZEN vs its z = 0 twin S8

- R(DP8_FROZEN / FA_WARM_FROZEN) PRACTICAL_BEST: 100.12 [99.86, 100.38] (ε = 8.000000, z = -)
- R(DP8_FROZEN / FA_WARM_FROZEN) CONTROLLED_BUDGET: 100.10 [99.81, 100.40] (ε = 8.000000, z = -)

### 10b. No-FT references on UNSEEN_VAL (the selection panel; never a headline)

- NOT_AVAILABLE: no UNSEEN_VAL role in the block

- 

### 10c. Deployed-size family comparison GRU_D128 vs C_SMALL (06L-C C11)

- NOT_AVAILABLE: no GRU_D128 entry

- GRU_D128 (the contest's 02B_GRU best@6.0, 20.96M parameters, already trained) vs C_SMALL (20.71M), PRACTICAL_BEST and CONTROLLED_BUDGET, on the A40 block; never a downstream family (ARCHITECTURE_LOCK unchanged).

## 11. CALIB-disjoint diagnostic (MAIN \ CALIB; 06L-C C14) and repeat / novel target strata (C15) — DESCRIPTIVE

- 30,259 CALIB-disjoint users, of whom 21,968 have ≥ 1 eval row (manifest users: MAIN 36,037, CALIB 5,778; E2E users: MAIN 26,254, CALIB 10,070); counts {"e2e": {"calib_users": 10070, "disjoint_users": 21968, "main_users": 26254}, "manifest": {"calib_users": 5778, "disjoint_users": 30259, "main_users": 36037}}; CALIB_VALIDATION users -> MAIN by client_key_hex; report only, never selection (06L-C §C14).

| arm | R% vs C_BIG on MAIN \ CALIB [CI] | R% vs C_BIG on MAIN [CI] |
|---|---|---|
| C_SMALL | 94.75 [93.96, 95.47] | 94.69 [94.03, 95.31] |
| FA | 85.02 [84.22, 85.85] | 85.11 [84.30, 85.90] |
| FP | 85.02 [84.22, 85.85] | 85.11 [84.30, 85.90] |
| PF | 85.02 [84.21, 85.86] | 85.12 [84.32, 85.90] |
| FA_1024 | 23.03 [22.19, 23.82] | 23.01 [22.24, 23.76] |
| DP8 | 22.70 [21.86, 23.51] | 22.67 [21.91, 23.47] |
| DP1 | 21.90 [21.04, 22.65] | 21.74 [20.96, 22.46] |
| FA_Q8 | 85.19 [84.35, 86.03] | 85.20 [84.36, 85.99] |
| FA_WARM | 106.51 [105.47, 107.61] | 106.59 [105.67, 107.43] |
| FA_WARM_FROZEN | 107.30 [106.27, 108.40] | 107.39 [106.50, 108.32] |
| DP8_FROZEN | 107.41 [106.33, 108.51] | 107.52 [106.61, 108.44] |
| FP_WARM_FROZEN | 107.30 [106.27, 108.41] | 107.39 [106.50, 108.32] |
| PF_WARM_FROZEN | 107.16 [106.14, 108.29] | 107.23 [106.33, 108.16] |
| FA_Q8_WARM_FROZEN | 107.17 [106.13, 108.29] | 107.23 [106.33, 108.15] |
| FA_1024_WARM_FROZEN | 107.42 [106.35, 108.53] | 107.53 [106.60, 108.45] |
| DP1_WARM_FROZEN | 107.70 [106.45, 108.89] | 107.69 [106.68, 108.67] |
| FT_FA | 87.43 [86.60, 88.25] | 87.56 [86.75, 88.33] |
| FT_FA_WARM | 108.26 [107.19, 109.35] | 108.35 [107.40, 109.24] |
| FT_FA_WARM_FROZEN | 108.85 [107.72, 109.91] | 108.77 [107.80, 109.67] |

| id | contrast | rule | on MAIN \ CALIB [CI] |
|---|---|---|---|
| K1 | R(C_SMALL / C_BIG) | PRACTICAL_BEST | 94.75 [93.96, 95.47] |
| K2 | R(FA / C_BIG) | PRACTICAL_BEST | 85.02 [84.22, 85.85] |
| K3 | R(FA / C_SMALL) | PRACTICAL_BEST | 89.73 [88.87, 90.51] |
| K4 | R(FP / C_BIG) | PRACTICAL_BEST | 85.02 [84.22, 85.85] |
| K5 | R(PF / C_BIG) | PRACTICAL_BEST | 85.02 [84.21, 85.86] |
| K6 | R(FT_FA / C_BIG) | PRACTICAL_BEST | 87.43 [86.60, 88.25] |
| K9 | FT_C − C_SMALL | PRACTICAL_BEST | 0.002345 [0.001803, 0.002897] |
| K10 | FT_FA − FA | PRACTICAL_BEST | 0.003565 [0.002980, 0.004188] |
| K11 | FA − LO | PRACTICAL_BEST | 0.123022 [0.120669, 0.125273] |
| K11b | FA − LAST | PRACTICAL_BEST | 0.051741 [0.049452, 0.054015] |
| K12 | R(DP8 / FA_1024) | CONTROLLED_BUDGET | 98.56 [97.63, 99.50] |
| K13 | R(DP1 / FA_1024) | CONTROLLED_BUDGET | 94.65 [92.94, 96.44] |
| K14 | R(FA_Q8 / FA) | PRACTICAL_BEST | 100.20 [99.87, 100.50] |
| K17 | R(FA / C_FULL) | PRACTICAL_BEST | 70.44 [69.55, 71.39] |
| K20 | R(FA_WARM_FROZEN / C_BIG) | PRACTICAL_BEST | 107.30 [106.27, 108.40] |
| K20b | R(FA_WARM_FROZEN / C_FULL) | PRACTICAL_BEST | 88.78 [87.95, 89.56] |
| K21 | FA_WARM_FROZEN − PRE | PRACTICAL_BEST | -0.000448 [-0.000941, 0.000052] |
| K_INIT | FA_WARM_FROZEN − FA | PRACTICAL_BEST | 0.033012 [0.031558, 0.034511] |
| P5 | FT_FA_WARM_FROZEN − FA_WARM_FROZEN | PRACTICAL_BEST | 0.002300 [0.001412, 0.003123] |
| K4w | R(FP_WARM_FROZEN / C_FULL) | PRACTICAL_BEST | 88.78 [87.95, 89.56] |
| K5w | R(PF_WARM_FROZEN / C_FULL) | PRACTICAL_BEST | 88.67 [87.85, 89.47] |
| K12w | R(DP8_FROZEN / FA_1024_WARM_FROZEN) | CONTROLLED_BUDGET | 100.00 [99.94, 100.07] |
| K13w | R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN) | CONTROLLED_BUDGET | 100.04 [99.91, 100.16] |
| K14w | R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN) | PRACTICAL_BEST | 99.88 [99.73, 100.03] |

- REPEAT = RANKABLE and LAST tie count 0 (target in the L50 window); NOVEL = the other E2E rows; counts {"E2E_rows": 229742, "NOVEL_rows": 166968, "REPEAT_rows": 62774, "repeat_share_of_E2E_rows": 0.2732369353448651}.

| stratum | arm | users | R% vs C_BIG | 95% CI |
|---|---|---|---|---|
| REPEAT | C_BIG | 15819 | 100.00 | [100.00, 100.00] |
| REPEAT | C_FULL | 15819 | 126.49 | [125.30, 127.81] |
| REPEAT | C_SMALL | 15819 | 87.48 | [86.74, 88.27] |
| REPEAT | FA | 15819 | 74.86 | [73.99, 75.75] |
| REPEAT | FP | 15819 | 74.86 | [73.98, 75.76] |
| REPEAT | PF | 15819 | 74.90 | [74.01, 75.80] |
| REPEAT | FA_1024 | 15819 | 21.82 | [20.97, 22.70] |
| REPEAT | DP8 | 15819 | 21.48 | [20.68, 22.32] |
| REPEAT | DP1 | 15819 | 20.79 | [19.97, 21.65] |
| REPEAT | FA_Q8 | 15819 | 74.83 | [74.00, 75.70] |
| REPEAT | FA_WARM | 15819 | 103.62 | [102.55, 104.75] |
| REPEAT | FA_WARM_FROZEN | 15819 | 105.60 | [104.51, 106.74] |
| REPEAT | DP8_FROZEN | 15819 | 105.87 | [104.71, 107.04] |
| REPEAT | FP_WARM_FROZEN | 15819 | 105.59 | [104.51, 106.74] |
| REPEAT | PF_WARM_FROZEN | 15819 | 105.16 | [104.04, 106.27] |
| REPEAT | FA_Q8_WARM_FROZEN | 15819 | 105.14 | [104.03, 106.23] |
| REPEAT | FA_1024_WARM_FROZEN | 15819 | 105.87 | [104.70, 107.04] |
| REPEAT | DP1_WARM_FROZEN | 15819 | 105.62 | [104.38, 106.87] |
| REPEAT | PRE | 15819 | 105.90 | [104.62, 107.14] |
| REPEAT | FT_C | 15819 | 92.61 | [91.82, 93.49] |
| REPEAT | FT_PRE | 15819 | 113.20 | [112.05, 114.52] |
| REPEAT | FT_FA | 15819 | 81.12 | [80.20, 82.04] |
| REPEAT | FT_FA_WARM | 15819 | 111.11 | [109.98, 112.38] |
| REPEAT | FT_FA_WARM_FROZEN | 15819 | 112.70 | [111.53, 114.01] |
| REPEAT | LO | 15819 | 3.01 | [2.72, 3.33] |
| REPEAT | KNN_EMB_SMALL | 15819 | 67.20 | [65.72, 68.76] |
| REPEAT | KNN_EMB_BIG | 15819 | 84.54 | [82.92, 86.27] |
| REPEAT | RULE | 15819 | 30.84 | [29.87, 31.89] |
| REPEAT | SKNN | 15819 | 101.01 | [99.42, 102.59] |
| REPEAT | LAST | 15819 | 100.58 | [99.08, 102.21] |
| REPEAT | POP | 15819 | 12.07 | [11.40, 12.78] |
| NOVEL | C_BIG | 24303 | 100.00 | [100.00, 100.00] |
| NOVEL | C_FULL | 24303 | 117.15 | [115.81, 118.50] |
| NOVEL | C_SMALL | 24303 | 104.18 | [103.15, 105.22] |
| NOVEL | FA | 24303 | 97.98 | [96.81, 99.05] |
| NOVEL | FP | 24303 | 97.98 | [96.81, 99.04] |
| NOVEL | PF | 24303 | 97.98 | [96.79, 99.06] |
| NOVEL | FA_1024 | 24303 | 22.91 | [21.96, 23.94] |
| NOVEL | DP8 | 24303 | 22.53 | [21.58, 23.50] |
| NOVEL | DP1 | 24303 | 21.68 | [20.67, 22.69] |
| NOVEL | FA_Q8 | 24303 | 98.11 | [96.92, 99.21] |
| NOVEL | FA_WARM | 24303 | 113.98 | [112.59, 115.43] |
| NOVEL | FA_WARM_FROZEN | 24303 | 113.86 | [112.38, 115.33] |
| NOVEL | DP8_FROZEN | 24303 | 113.70 | [112.22, 115.23] |
| NOVEL | FP_WARM_FROZEN | 24303 | 113.86 | [112.38, 115.32] |
| NOVEL | PF_WARM_FROZEN | 24303 | 114.01 | [112.55, 115.48] |
| NOVEL | FA_Q8_WARM_FROZEN | 24303 | 114.01 | [112.55, 115.48] |
| NOVEL | FA_1024_WARM_FROZEN | 24303 | 113.71 | [112.24, 115.23] |
| NOVEL | DP1_WARM_FROZEN | 24303 | 113.83 | [112.09, 115.48] |
| NOVEL | PRE | 24303 | 113.78 | [112.10, 115.40] |
| NOVEL | FT_C | 24303 | 100.40 | [99.37, 101.42] |
| NOVEL | FT_PRE | 24303 | 107.24 | [105.76, 108.63] |
| NOVEL | FT_FA | 24303 | 94.23 | [93.09, 95.33] |
| NOVEL | FT_FA_WARM | 24303 | 107.53 | [106.11, 108.89] |
| NOVEL | FT_FA_WARM_FROZEN | 24303 | 107.00 | [105.54, 108.39] |
| NOVEL | LO | 24303 | 0.35 | [0.28, 0.44] |
| NOVEL | KNN_EMB_SMALL | 24303 | 15.91 | [15.28, 16.56] |
| NOVEL | KNN_EMB_BIG | 24303 | 13.99 | [13.48, 14.57] |
| NOVEL | RULE | 24303 | 13.85 | [13.15, 14.57] |
| NOVEL | SKNN | 24303 | 57.47 | [56.63, 58.23] |
| NOVEL | LAST | 24303 | 0.00 | [0.00, 0.00] |
| NOVEL | POP | 24303 | 12.47 | [11.74, 13.24] |

## 12. Claim register and headline template (06L-C C12) — DESCRIPTIVE (text, frozen with the script)

- CLAIM_REGISTER.json (PPSI_V4_ANA_CLAIM_REGISTER_1; 39 contrasts) and HEADLINE_TEMPLATE.md are written beside this report; the template key of each K is the `claim key` column of section 1.

- A exceeds B only when the paired 95% CI of the contrast lies above its reference (0 for a paired Δ, 100 for a retention R); an interval that includes the reference reads 'no detectable difference' (the claim vocabulary of EVALUATION_CONTRACT_V4 §7), never 'no difference' or 'equivalent': an absence of evidence is not evidence of absence, and only a registered equivalence test (K14's margin, 06L-C C15) can support 'equivalent'.
- Only the loaded registry's contrasts (report section 1, CONFIRMATORY rows under their registered rule and seeds) carry confirmatory wording; every other number is DESCRIPTIVE and is never phrased as a finding.
- Every retention sentence names the recipe and the reference: 'matched-exposure retention of the registered FL recipe' (06L-C C6), relative to a cloud model trained on 5-100-event users; never 'FL retains X %' without the recipe (06I §6(c)); never 'a lower bound' (06L-C C6).
- VALIDATION values are max-of-marks; the single TEST look is the only unbiased number (06A §9(g)).
- A contrast flagged SEED_SENSITIVE carries the flag in the same sentence; every CI covers user sampling only, conditional on the trained models.
- An arm flagged BELOW_DEVICE_HEURISTIC (CI of Δ(arm − LAST) below 0) carries the flag in the same sentence as its retention (06L-C C3).
- Non-DP federated arms are never called private: the uploaded deltas reveal the viewed items and secure aggregation is not simulated (06I §7(a); 06L-C C1); a DP arm states the 06L-C C2 guarantee in full.
- K17-K19 and K20b are compositions with K15 (see algebraic_dependencies); they are never presented as independent evidence or as a second headline (06L-C C4).
- The backbone was selected under rule T1; never 'SASRec beats GRU' (06B §7(c)); GRU_D128 vs C_SMALL is DESCRIPTIVE and never a family choice (06L-C C11).
- Anything added after the first VALIDATION read is POST_VALIDATION_EXPLORATORY and never confirmatory (06L-C C5).
- An 'events to parity' (or events-to-X) sentence is written only from a persistent hit of the 06L-C C10 rule (the parity holds at the next registered h too, with the CI lower bound >= X), naming the rule and the reference (C_BIG FULL history or matched-h C_BIG(h)); a first-hit h alone, NO_CI, NO_NEXT_H or NO_C_REF supports no such sentence; the curves are DESCRIPTIVE.

## 12b. ANA-05 additions (C_FULL s2027; 06M; 06M_B B2 / B4 / B5; S8 comm) — DESCRIPTIVE

- Descriptive source seeds: [["C_FULL", 2027, "CONTROLLED_BUDGET", "HEAVY_PANEL"], ["C_FULL", 2027, "PRACTICAL_BEST", "HEAVY_PANEL"], ["C_FULL", 2027, "CONTROLLED_BUDGET", "MAIN"], ["C_FULL", 2027, "PRACTICAL_BEST", "MAIN"], ["C_FULL", 2027, "CONTROLLED_BUDGET", "SPARSE_PANEL"], ["C_FULL", 2027, "PRACTICAL_BEST", "SPARSE_PANEL"], ["C_FULL", 2027, "CONTROLLED_BUDGET", "UNSEEN_HOLDOUT"], ["C_FULL", 2027, "PRACTICAL_BEST", "UNSEEN_HOLDOUT"]]; 06D §8 NICE seed; 06E §2 / DL 176: a run of the second seed of a single-seed contrast is DESCRIPTIVE (K15, K17-K19, K20b, K20c stay seed 2026 confirmatory); it appears only in the DESCRIPTIVE retention table and in a contrast's DESCRIPTIVE `extra_seeds` row.

- S7 (FA_WARM): BOUND_TO_VALIDATION_RUNS + FILE_VERIFIED; warm selection sha256 d67a0a851665495b092abbba4456f26a3423d6ef21665f5a0e2b65e2d78dc000; record {"file": "records/FL_WARM_SELECTION_S7.json", "file_sha256": "d67a0a851665495b092abbba4456f26a3423d6ef21665f5a0e2b65e2d78dc000", "self_sha256": "c7bd76d61e0bdf8494556eace553fa9a27255edfac1d3750181bc1abe9239c18", "winner_factor": 0.25, "winner_label": "f=0.25|sf=0.1", "winner_server_lr_factor": 0.1}.
- S8 (FA_WARM_FROZEN): BOUND_TO_VALIDATION_RUNS + FILE_VERIFIED; warm selection sha256 8b79d3052799dfa0ba208dce5db138da88aaa7024ba6511a374c59ce8c55a98a; record {"file": "records/FL_WARM_SELECTION_S8.json", "file_sha256": "8b79d3052799dfa0ba208dce5db138da88aaa7024ba6511a374c59ce8c55a98a", "self_sha256": "53bc7a5a1312310e3071bcd14d8fb449d6ed6225206e5483d0235440803eed3d", "winner_factor": 1.0, "winner_label": "f=1|sf=0.1", "winner_server_lr_factor": 0.1}.

### FT_FA_WARM − FA_WARM (06M_B B4; the full production stack)

| contrast | status | seeds | Δ MRR@20 | 95% CI | CI vs 0 | harmed share | skipped |
|---|---|---|---|---|---|---|---|
| FT_FA_WARM − FA_WARM | COMPUTED | [2026, 2027] | 0.002617 | [0.001921, 0.003335] | above | 0.3360 | - |
| FT_FA_WARM_FROZEN − FA_WARM_FROZEN | COMPUTED | [2026, 2027] | 0.002045 | [0.001324, 0.002807] | above | 0.3320 | - |

### Quality during rollout (06M_B B5)

- 06M_B B5 (DESCRIPTIVE): per FL run (PRACTICAL_BEST entry's 12 marks) the macro-E2E MRR@20 per mark; AUC = trapezoid over the marks 0.5 .. 6.0 EFE (MRR x EFE), auc_mean = AUC / 5.5 (with its paired user-bootstrap CI, per resample on the same W); first mark reaching a reference (PRE-alone = theta_pre, seed 2026, PRACTICAL_BEST label; LAST = the seed-free rank file): POINT rule = the first mark whose value >= the reference's value, CI rule = the first mark whose paired delta 95% CI lies above 0; NOT_REACHED / NO_REFERENCE otherwise; seed mean of the per-seed AUC

| arm | seed | AUC (MRR x EFE) | AUC / 5.5 | 95% CI | first mark ≥ PRE (point) | first mark > PRE (CI) | first mark ≥ LAST (point) | first mark > LAST (CI) |
|---|---|---|---|---|---|---|---|---|

### S8 communication from the run's own record (06L-C C8)

- NOT_AVAILABLE

- WARM_PREVIEW: WARM_PREVIEW / S7-S8 CALIB screen run (06M M2; 06M_B B1-B2): DESCRIPTIVE CALIB values, never a block entry, a selection or a contrast (the registered S7 / S8 screens decide only through FL_WARM_SELECTION_S7|S8.json)

## 12c. ANA-06: the pretrained line (PREG1_AMENDMENT_06N) — DESCRIPTIVE until registered

- FA_WARM* = FA_WARM_FROZEN (S8; DETERMINED_FROM_FILES); CALIB winners {"S7": 0.17058229165688377, "S8": 0.17070942466659556}; 06N N2: FA_WARM* = the variant (S7 = FA_WARM full model, S8 = FA_WARM_FROZEN frozen tables) whose registered FL_WARM_SELECTION_S7|S8.json has the higher CALIB value of its selected cell (values[winner.label]); |S8 - S7| <= 0.0005 -> S8. Read from the two files only (CALIB values), never from a MAIN value; on TEST the VALIDATION_RESULT's choice (and the files, when given, must agree).

- IN_LOADED_FILE: None

- K20b (primary): see confirmatory K20b.
- K21 (primary): see confirmatory K21.
- K20 (secondary): see confirmatory K20.
- K_INIT (secondary): see confirmatory K_INIT.
- P5 (secondary): see confirmatory P5.
- K4w (secondary): see confirmatory K4w.
- K5w (secondary): see confirmatory K5w.
- K12w (secondary): see confirmatory K12w.
- K13w (secondary): see confirmatory K13w.
- K14w (secondary): see confirmatory K14w.
- K2 (secondary): see confirmatory K2.

Needs registration:
- 06P P6 'Tertiary (unadjusted, Holm band beside)': implemented as a DESCRIPTIVE step-down band within the 13 tertiary contrasts (no decision); confirm
- 06P names the FT_FA_WARM_FROZEN - FA_WARM_FROZEN contrast 'P5' (used as its id)

## 13. Appendix — records (LR screen = CALIB, not MAIN), inputs, checks

```
{
 "DP_WOR": {
  "by_z": {
   "0.5642": {
    "eps_poisson_on_integer_orders_2_256_info": 7.999847800151169,
    "eps_rdp_floor_prop11_c1_info": 7.999847800151169,
    "eps_rdp_floor_prop11_c2_info": 1132.4998892194433,
    "eps_registered_poisson_add_remove": 7.998245751758989,
    "eps_registered_reference_from_packet": 7.99825,
    "eps_wor_replace_one_c1_packet_literal": 9.959133390202531,
    "eps_wor_replace_one_c2_mechanism_faithful": 1388.1448872449914
   },
   "1.3094": {
    "eps_poisson_on_integer_orders_2_256_info": 0.9998462503769788,
    "eps_rdp_floor_prop11_c1_info": 0.9998462503769788,
    "eps_rdp_floor_prop11_c2_info": 5.329469016140623,
    "eps_registered_poisson_add_remove": 0.9998462503769788,
    "eps_registered_reference_from_packet": 0.99985,
    "eps_wor_replace_one_c1_packet_literal": 1.5024015270523532,
    "eps_wor_replace_one_c2_mechanism_faithful": 6.68772582621574
   }
  },
  "self_sha256_ok": "NO_SELF_SHA256"
 },
 "SELECTION": [
  {
   "arm": "FT_C",
   "cell": "FT_C_f0.3_P4",
   "file": "records/SELECTION_FT_C.json",
   "format": "PPSI_V4_DEVICE_SELECTION_1",
   "role": null,
   "rule": "max; ties -> canonical order",
   "seed": 2026,
   "selection_role": "UNSEEN_VAL",
   "self_sha256": "43e4fbee19246913ef8e93281b14fd30c08bcf7025a42edd193cacad4cf36e0c",
   "values_primary": {
    "FT_C_f0.1_P1": 0.15361845782138328,
    "FT_C_f0.1_P2": 0.15403840614558736,
    "FT_C_f0.1_P4": 0.1549313099463226,
    "FT_C_f0.3_P1": 0.15445898208569125,
    "FT_C_f0.3_P2": 0.1557278073279543,
    "FT_C_f0.3_P4": 0.1577851409217914
   }
  },
  {
   "arm": "LO",
   "cell": "LO_P4",
   "file": "records/SELECTION_LO.json",
   "format": "PPSI_V4_DEVICE_SELECTION_1",
   "role": null,
   "rule": "max; ties -> canonical order",
   "seed": 2026,
   "selection_role": "UNSEEN_VAL",
   "self_sha256": "06f498d0988cc2e533950ae819fa121b09617861d55a62a308287610447c46e9",
   "values_primary": {
    "LO_P1": 3.517503232757823e-05,
    "LO_P2": 0.00047777668482142246,
    "LO_P4": 0.004636922314985888
   }
  },
  {
   "arm": "FT_FA",
   "cell": "FT_FA_f0.3_P4",
   "file": "records/SELECTION_FT_FA.json",
   "format": "PPSI_V4_DEVICE_SELECTION_1",
   "role": null,
   "rule": "max; ties -> canonical order",
   "seed": 2026,
   "selection_role": "UNSEEN_VAL",
   "self_sha256": "bec202ae057068d6174ebb7d8ddbff95fc8c0be44b681c45882284c4dd6c4fec",
   "values_primary": {
    "FT_FA_f0.1_P1": 0.13957911561700007,
    "FT_FA_f0.1_P2": 0.14051031423314145,
    "FT_FA_f0.1_P4": 0.1418404026169788,
    "FT_FA_f0.3_P1": 0.14126247468411587,
    "FT_FA_f0.3_P2": 0.14293584391976072,
    "FT_FA_f0.3_P4": 0.14521750090254493
   }
  }
 ],
 "STRATA": {
  "cuts": {
   "activity_quantile_cuts": [
    7.0,
    13.0,
    25.0
   ],
   "brand_loyalty_tercile_cuts": [
    0.3333333333333333,
    0.5384615384615384
   ],
   "category_diversity_tercile_cuts": [
    1.0,
    2.0
   ]
  },
  "n_users": 131072,
  "registry_self_sha256": "31a5d45f30fce5f0901752be48201fcb15bd21e05ab4be169c6937253bd0789f",
  "source_manifest_id": "MAIN_TRAIN_5_100_131072"
 },
 "S_RECORD": {
  "S": 0.28436270792896057,
  "disclosure": "non-private; follows the unclipped trajectory (amendment 06 §2 Q3)",
  "format": "PPSI_V4_DP_S_RECORD_1",
  "label": "S_CALIBRATION",
  "n_norms": 20625,
  "rule": "median client-delta L2 norm over FA_1024 seed-2026 rounds 1-20",
  "seed": 2026,
  "self_sha256": "ce90d0cea289ed148c4d3c1a2f6eb8c14075dc43fe0b99487a7947bc03e7a31e"
 },
 "S_RECORD_WARM_FROZEN": {
  "S": 0.2754088599106647,
  "arm": "S_CAL_WARM_FROZEN",
  "file": "records/S_RECORD_WARM_FROZEN.json",
  "file_sha256": "2e47ba398432dd148229277326030af30e7222306b234c05ef58f4c8063ca213",
  "format": "PPSI_V4_DP_S_RECORD_WARM_1",
  "init_from": {
   "checkpoint_bytes": 82853197,
   "checkpoint_sha256": "cf08588c11f019cc760f129ba7ba6b23806297660e0170f9938436a5cb88cf78",
   "family": "SASREC",
   "fl_init_kind": "endpoint_6.0",
   "kind": "PRETRAIN_FOR_FL",
   "model_variant": "SASREC_D64_B2",
   "run_id": "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026",
   "run_label": "PRETRAIN_FOR_FL",
   "seed": 2026,
   "state_digest": "ee33345be4f20b61a53cd80fa11e1af702031bfcd1425af490825cd0019dc54a",
   "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN"
  },
  "label": "S_CALIBRATION",
  "n_norms": 20625,
  "rule": "median client-delta L2 norm over FA_1024 seed-2026 rounds 1-20",
  "seed": 2026,
  "self_sha256": "5201ba8f7e32b356a1a7340a7673142898f41d18d090a0521935d29861ddf90f",
  "stage": "S8",
  "warm_variant": "FA_WARM_FROZEN"
 },
 "TRUNC_ROWS": {
  "files": [],
  "n_rows": 0
 },
 "VALIDATION_RESULT": {
  "self_sha256": "a5eba858a53111e3179bab46aeb6940dc739ceeb6caffebe27203bb5fcef823a"
 },
 "WARM_SELECTION_S7": {
  "arm": "FA_WARM",
  "file": "records/FL_WARM_SELECTION_S7.json",
  "file_sha256": "d67a0a851665495b092abbba4456f26a3423d6ef21665f5a0e2b65e2d78dc000",
  "format": "PPSI_V4_FL_WARM_SELECTION_2",
  "init_from": {
   "checkpoint_bytes": 82853197,
   "checkpoint_sha256": "cf08588c11f019cc760f129ba7ba6b23806297660e0170f9938436a5cb88cf78",
   "family": "SASREC",
   "fl_init_kind": "endpoint_6.0",
   "kind": "PRETRAIN_FOR_FL",
   "model_variant": "SASREC_D64_B2",
   "run_id": "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026",
   "run_label": "PRETRAIN_FOR_FL",
   "seed": 2026,
   "state_digest": "ee33345be4f20b61a53cd80fa11e1af702031bfcd1425af490825cd0019dc54a",
   "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN"
  },
  "label": "CALIB selection record (values not carried here)",
  "no_measurable_gain": false,
  "self_sha256": "c7bd76d61e0bdf8494556eace553fa9a27255edfac1d3750181bc1abe9239c18",
  "stage": "S7",
  "tuning_selection_sha256": "9bca186e5025f5caa9e762f3c425fb3151923b79e60f01e6ecb063aac9c715ec",
  "winner_factor": 0.25,
  "winner_label": "f=0.25|sf=0.1",
  "winner_server_lr_factor": 0.1
 },
 "WARM_SELECTION_S8": {
  "arm": "FA_WARM_FROZEN",
  "file": "records/FL_WARM_SELECTION_S8.json",
  "file_sha256": "8b79d3052799dfa0ba208dce5db138da88aaa7024ba6511a374c59ce8c55a98a",
  "format": "PPSI_V4_FL_WARM_SELECTION_2",
  "init_from": {
   "checkpoint_bytes": 82853197,
   "checkpoint_sha256": "cf08588c11f019cc760f129ba7ba6b23806297660e0170f9938436a5cb88cf78",
   "family": "SASREC",
   "fl_init_kind": "endpoint_6.0",
   "kind": "PRETRAIN_FOR_FL",
   "model_variant": "SASREC_D64_B2",
   "run_id": "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026",
   "run_label": "PRETRAIN_FOR_FL",
   "seed": 2026,
   "state_digest": "ee33345be4f20b61a53cd80fa11e1af702031bfcd1425af490825cd0019dc54a",
   "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN"
  },
  "label": "CALIB selection record (values not carried here)",
  "no_measurable_gain": false,
  "self_sha256": "53bc7a5a1312310e3071bcd14d8fb449d6ed6225206e5483d0235440803eed3d",
  "stage": "S8",
  "tuning_selection_sha256": "9bca186e5025f5caa9e762f3c425fb3151923b79e60f01e6ecb063aac9c715ec",
  "winner_factor": 1.0,
  "winner_label": "f=1|sf=0.1",
  "winner_server_lr_factor": 0.1
 }
}
```

Library files sha256:

```
{
 "tools/g2_run.py": "3893a49a71be49f5baf338b93195e2e885eecb9326705902913bd4d5e1abb44d",
 "v4base/identity.py": "f7d645f3dc4571cd4951c0f55f83e0a18b8883d547a029c57144d427cd543ed1",
 "v4base/strata.py": "78025da935bb0c1fd2fba9af77727855a2e909b48410507351e90e250981a5f2",
 "v4eval/bootstrap.py": "693d921ee019fa1e2c667ac355789e885f1b17ad8228f0ece1a3e374997595d4",
 "v4eval/metrics.py": "79aeb5617279d64f18a5bd00b8524581207d3a605f4a98d1ceb383a274a33ff5",
 "v4eval/quality.py": "b117c9804bfbc827e936453e79f4ae4d567545cfde1790b46152e02084229276",
 "v4eval/values.py": "a55fcf0f32b749d1d606a65103cbea8121f128b6ecb367c8af3e317b88ed7c95",
 "v4train/bootstrap_ci.py": "47bae8c9eb72fa3757d9090afa6e87877b29fbd7dd1538351909bc4bbc4bf567"
}
```

## 14. Registered disclosures (verbatim)

Amendment 06A §9:

9. **Disclosures registered for the thesis:** (a) the 6.0-endpoint reversal of the family order (G2 condition 5); (b) the FL local LR is the single SMALL peak (the {1×, 0.5×} grid of the realistic plan was not carried into amendment 06 or the code — untuned, like μ); (c) FedBN is inapplicable (LayerNorm backbone; FedLN not added by the owner); (d) purchase-AUC / comparable-offer tasks of the proposal are out of scope by agreement (next-item task); (e) battery is not measured; (f) per-client batching in launch_device reorders 6 / 1,251 rows by one rank position at ranks > 20 (no effect on @≤20 metrics); (g) VALIDATION values are max-of-marks; TEST is the only unbiased number.

Amendment 06B §7:

(a) The review's proposal-bibliography corrections ([5] title; [6] IEEE Intelligent Systems 36(5) 2021; [8] Luo et al. CIKM 2022 vs Zhang et al. IJCAI 2023 PFedRec; [12] unverified) go into the thesis bibliography. (b) RETRO_TEST contains Black Friday 29 Nov 2019 — the VALIDATION→TEST change of C_BIG is reported as a shift indicator. (c) The contest selects the backbone under rule T1; never phrased as "SASRec beats GRU". (d) Downstream arms trained on A40; the contest on A16. (e) UNSEEN_VAL is a selection panel, never a headline panel; applying the FT recipe to TRUE_COLD/SPARSE is an extrapolation.

Amendment 06B §1:

FL local LR screened on CALIB (3 points); μ and server LR (1.0) untuned.

Amendment 06D §3:

K12 = R(DP8 / FA_1024) and K13 = R(DP1 / FA_1024) are CONFIRMATORY under the pairing CONTROLLED_BUDGET / CONTROLLED_BUDGET (the round-384 endpoint of both arms: a data-independent release). Their PRACTICAL_BEST ratios are reported as DESCRIPTIVE with the disclosure "best-of-12 selection on the users' own validation data is not covered by the accountant". Every other contrast keeps PRACTICAL_BEST primary with CONTROLLED_BUDGET reported (06B §2).

Amendment 06D §4:

Primary headline: **K2**. Secondary: K1, K3, K8, K12. Tertiary: the rest.

Notes:

- 06A §9(b) is superseded for the FL arms by amendment 06B §1 (the local-LR screen); both are printed as registered.
- Support bins (06A §2) are observational: history length is not manipulated.
- Every number outside the confirmatory section (the loaded registry: K1-K19 / K1-K21) is DESCRIPTIVE.
- 06E §9 (e) The seed-mean retention is the mean of per-seed ratios (06A §7 reading). (f) The contrast-text check folds whitespace and dash glyphs. (g) FL cumulative bytes at a mark are the fleet mean prorated by rounds.
- Device arms (LO, FT_C, FT_FA) and the ONNX rows exist under PRACTICAL_BEST only: K6-K11, K16, K18 and K19 have no CONTROLLED_BUDGET row (06E §5; REV-ANA-01 A.1 DISCLOSE).
- Only retained% is reported; loss_b = 100 - retained_b per resample, so its CI is the mirror interval (E7.5).

Amendment 06K / DL 206:

- (1) PRE is trained on 1,217,610 users (9x MAIN) while C_BIG saw 131,072: K20 = R(FA_WARM / C_BIG) can exceed 100 and mixes a data advantage into the FL retention. The matched cloud reference of the deployment branch is C_FULL (TIEBREAK = PRE users + MAIN users): R(FA_WARM / C_FULL) and R(PRE / C_FULL) are printed beside K20 as DESCRIPTIVE, and the thesis text for the deployment branch leads with them. K21 = FA_WARM - PRE stays the value of federated training on the private users.
- (2) theta_pre is ONE seed (2026) shared by FA_WARM s2026 and s2027: their seed spread measures FL variance only.
- (3) PRE applies the SMALL recipe (tuned at MAIN scale, 2.6M decisions) to 24.1M decisions at 6.0 EFE (about 9x more updates) without re-tuning: this can only understate PRE. PRE marks on UNSEEN_VAL are DESCRIPTIVE and never select.

Amendment 06I §7(b):

06I §7(b) (variant registered in DECISION_LOG row 218): bytes per client are reported with the registered definition (06A §3: visits x (down + up), dropped clients not charged) and, beside it, with the dropped visits charged their download (a dropped client receives the model and never uploads): bytes_per_client_charged_dropped = visits x (down + up) + dropped visits x down per client (client mean). The variant that charges a dropped visit down + up (bytes_per_client_charged_dropped_up_down) is DESCRIPTIVE only. The curves carry the registered charged variant as a second series (the up + down variant as a DESCRIPTIVE third series).

Amendment 06L-C (verbatim):

- C1: Every arm is described by four columns: data seen at TRAINING by the server; data seen at INFERENCE by the server; formal guarantee; against whom. Placements corrected: FT_C on MAIN = "L1 base (server-trained on raw events) + local fine-tune"; FT_PRE = L5 (base trained only on opted-in users); PRE-alone = L1 on opted-in data; RULE / POP = "device + published aggregate"; LAST added to L6; a column "secure aggregation: not simulated" for every FL arm. Non-DP FL updates reveal the user's viewed items to the server (input-table rows and output rows move); S8 removes the input-table leak only partly (brand/category deltas may still leak).
- C2: Adversary = an honest-but-curious browser vendor (the server); the retailer's own logs are out of scope. DP guarantee = central DP with a trusted aggregator, client-level add/remove adjacency, (ε, δ = 5e-6), conditional on the public bootstrap block (catalogue, transform statistics, θ0 / θ_pre from opted-in users); the FL hyper-parameter tuning on CALIB is not accounted.
- C3: Baseline runs also write per-row rank files. Reported for every arm: paired Δ(arm − LAST) with CI; headroom R* = (arm − LAST)/(C_ref − LAST); flag BELOW_DEVICE_HEURISTIC when the CI of Δ lies below 0. New confirmatory contrast **K11b = FA − LAST** (secondary). The device-only envelope {LO, LAST, RULE} is reported as one row group.
- C4: `company/gates/ANA_CONTRASTS_v3.json` = K1–K21 + K11b + (C7) + (C8), with the algebraic dependencies listed (K3 = K2/K1; K17–K19 = the matching K × K15) and without the word "co-headline" for K17.
- C5: The arm set, seeds, recipes and checkpoint rules for the VALIDATION preflight are exactly those registered up to this amendment plus the ladder's selected FL configuration; nothing else is confirmatory.
- C6: "Matched-exposure retention of the registered FL recipe" (not "lower bound"); the central-LR-shape waste is "≈ 50 % of the exposure budget at ≤ 0.1× peak", with the peak marks cited.
- C7: R(FA_WARM / C_FULL) and R(PRE / C_FULL) (2 FA_WARM seeds vs C_FULL s2026; PRE single seed), K20, K21, **K9c = FT_PRE − PRE on MAIN** (new arm FT_PRE: θ_pre + the registered FT_C procedure with SELECTION_FT_C, seeds 2026/2027), **K9b = FT_C − C_SMALL on UNSEEN_HOLDOUT**. The Edge answer and the L5 row lead with these.
- C8: Paired contrast S8 − S7 reported with the bytes ratio; one bytes axis for every privacy level (raw events → model download → FL traffic → S8). S8 disclosure corrected: 265,162 uploaded parameters (to be re-measured), brand/category deltas may still leak, off-support rows move ≈ 1e-9, not exactly 0. S8 runs with 2 seeds unconditionally (supersedes 06K §3's conditional second seed).
- C11: GRU_D128 (the contest's 02B_GRU best@6.0, 20.96M parameters, already trained) vs C_SMALL (20.71M), PRACTICAL_BEST and CONTROLLED_BUDGET, on the A40 block; never a downstream family (ARCHITECTURE_LOCK unchanged).
- C12: A number-free abstract/headline template and a CLAIM register (allowed / forbidden sentence per contrast) are frozen together with the ANA script.
- C13-C15: Per-seed ratios, seed spread and a SEED_SENSITIVE flag beside every K; a Holm band beside the unadjusted CIs; a CALIB-disjoint diagnostic (MAIN \ CALIB) for every CALIB-tuned arm; harmed-client share for every K; a K14 equivalence margin of ±1 % relative registered now (before any FA_Q8 read); repeat vs novel target stratum.
- C16-C17: Adopted: an evaluation-only "no-FT" reference of C_SMALL and FA on UNSEEN_VAL (descriptive); DP8_FROZEN (S8 + DP ε = 8, seed 2026, its z = 0 twin = S8) at SHOULD-high.

How the 06L-C items are computed (ANA-04):

- 06L-C C3: LAST enters as a seed-free BASELINE_CPU rank file (v4base.run ranks_int32.npy, CPU-scored exact ranks on the same manifest rows; the key differs only in eval_device_class / precision_id / objective_id placeholders); Δ(arm − LAST) pairs every seed of the arm with the one LAST file on the same W; R* per resample = (arm_b − LAST_b) / (C_BIG_b − LAST_b), seed mean of the per-seed R*; a resample with C_BIG_b <= LAST_b makes the R* CI UNDEFINED.
- 06L-C C13: the seed spread is max - min of the per-seed points; SEED_SENSITIVE = spread > half the width of the user-bootstrap CI of the seed mean; the user + seed interval = point ± 1.96 x sqrt(var_boot + s²_seed / n_seeds) (DESCRIPTIVE; the percentile CIs stay conditional on the trained models).
- 06L-C C13: the Holm band uses normal intervals from the bootstrap SE (sd of the seed-mean resamples): z = (point - ref) / SE, two-sided p, Holm step-down over the loaded registry's family (the registered count, so a missing contrast never shrinks the family), band = point ± z_(1 - α_j / 2) x SE at the contrast's Holm level α_j for a rejected contrast and at α / (m - r) (r = the number of rejections) for every contrast from the first non-rejection on, so a band excludes the reference exactly when Holm rejects (REV-DELTA-03 M3); the Bonferroni band (α / m) is printed beside it. It corrects 06D §4's "not resolvable with 1,000 resamples" (true for 99.7% percentile endpoints only).
- 06L-C C14: MAIN \ CALIB = the MAIN E2E users whose client_key_hex is not a CALIB_VALIDATION user (the CALIB manifest's users table); on VALIDATION the counts must be the registered 51,211 / 12,715 / 38,496 when the production loader supplies them. Report only, never selection.
- 06L-C C15: the harmed-client share of a ratio contrast is the share of users whose per-user macro difference (numerator - denominator, seed mean) is < 0; the seed-vs-seed null is the same share for s2027 - s2026 of the same arm on the same users (a noise floor).
- 06L-C C15: REPEAT rows = RANKABLE rows whose target is in the user's L50 window, read from LAST's per-row tie count (LAST scores the window's distinct items by a unique recency, every other class ties at the sentinel, so the target is in the window iff its tie count is 0); NOVEL = every other E2E row (OOV targets included).
- 06L-C C8: the raw-event level of the bytes axis needs a registered per-event size (records.EVENT_SIZE, a DL row); without it that level reads NOT_REGISTERED. TRAIN decisions per client come from the MAIN TRAIN-layout manifest (a design quantity, not a result). FL levels add the final model download to the training traffic.
- 06L-C C16: the no-FT UNSEEN_VAL references are C_SMALL / FA re-evaluated on the selection panel; the FT value beside them is the SELECTION record's value of the selected cell (a max over cells: optimistic); UNSEEN_VAL users are base-unseen, MAIN users base-seen (review 10 [9-R3]).
- 06L-C C17 (followed literally): DP8_FROZEN's "z = 0 twin = S8" differs from DP8_FROZEN in cohort and round count (the DP plan's 1,024 clients per round / 384 rounds vs S8's FA plan), so R(DP8_FROZEN / S8) mixes the DP noise with that change of cohort and rounds; it is not a pure privacy cost.
- GATE-B B3 (INFERENCE): the S8 bytes curve charges the fleet-mean one-time frozen-table download in full from the first mark and prorates only the training traffic, so the early marks overstate S8's cumulative bytes (conservative against S8).
- 06L-C C8: the raw-event level of the bytes axis uses the client's MAIN TRAIN decisions as its event count (a design quantity from the TRAIN-layout manifest, not the raw event log), times the registered per-event size.

How the ANA-05 items are computed:

- ANA-05 C_FULL s2027 (06D §8 NICE seed; 06E §2 / DL 176): admitted as a DESCRIPTIVE source seed; its v4int.reeval rows carry comparison_status / source_seed_status DESCRIPTIVE; K15, K17-K19, K20b and K20c stay seed 2026 (the s2027 pairs appear only in the retention table and in a contrast's DESCRIPTIVE extra_seeds row).
- ANA-05 06M_B B2: an FA_WARM / FA_WARM_FROZEN MAIN run must carry launch_fl's LAUNCH warm_selection (the 2-D FL_WARM_SELECTION_S7|S8.json) whose winner factors equal its arm_spec factors; the registered file itself, when given in records, must equal that binding. Rows-only / REEVAL entries (no run dir) carry no LAUNCH binding and are listed as such.
- ANA-05 06M_B B4: FT_FA_WARM(_FROZEN) = theta_pre -> FA_WARM (S8) MAIN -> the registered FT_C procedure; its base (start_base) must be the block's base-arm run of the same seed; the paired delta FT_FA_WARM - FA_WARM (+ _FROZEN) is reported with the harmed-client share (06L-C C13-C15), PRACTICAL_BEST (a best.pt base) only; DESCRIPTIVE.
- ANA-05 06M_B B5: 'quality during rollout' for FA, FA_WARM and S8 from each run's 12 marks: trapezoid AUC of the macro MRR@20 over 0.5 .. 6.0 EFE and the first mark reaching PRE-alone / LAST (a point rule and a paired-CI rule); DESCRIPTIVE, VALIDATION marks only (the marks are the same user set as the PRACTICAL_BEST selection).
- ANA-05 06L-C C8: S8's bytes per visit, upload / download bytes, uploaded and frozen parameter counts come from the run's own fl.comm record (RUN_SUMMARY bytes.per_visit); the registered '265,162 uploaded parameters (to be re-measured)' is printed beside the measured count.
- ANA-05 06M M2: WARM_PREVIEW runs (and the S7 / S8 CALIB screens) are CALIB runs: they are refused as block entries (index and report) and never enter any contrast.

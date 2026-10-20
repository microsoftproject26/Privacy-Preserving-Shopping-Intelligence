# PPSI V4 retention report — split VALIDATION

- RETENTION_RESULT.json self_sha256: `a5eba858a53111e3179bab46aeb6940dc739ceeb6caffebe27203bb5fcef823a`
- script sha256: `9a3b2c2a918ef2baacc5299426941f1eb94c366f5bb6aa7c29b4bfdbca0afe58`; contrasts sha256: `f1769f0b8a0f11dc3861abdff02d9668449c3a47aa114462335462b653354515` (content `df7884b81fa7cb79589e033b41457eccf560d0c35feb0cdca693107913fe4264`)
- block index sha256: `bd6e5dff6ee2dcd0b646605648b56b3a7132257f65b55ffd005f4f7f03961d27`
- comparison key: `{"candidate_order_hash": "ab606d5e3728c993d31048f6d613838658d3a5ce8a0d0bd023611ce0ff027476", "data_protocol_hash": "bc699a97dd2fab054e5a5c00f9661dd32b0677941d3f23f2e4b81751b3b7b330", "eval_device_class": "NVIDIA_A40-FP32-DET", "eval_score_block": "158486", "eval_stream": "E2E_ROWS", "evaluator_code_sha256": "61b64e61c1a2e4f5a6fecd42f0ad3b98de26562d13a07758573cc9496b8d962e", "feature_contract_hash": "e97600f508497398da47f48b029d49a6fffee316e6a0c1ea20e046fe81fda5aa", "label_policy_hash": "75acea25716829cba74a7703f17208f405b455c72af2ec19f413585636f7e59a", "objective_id": "EXACT_FULL_K_CE_SUM_OVER_LOSS_ELIGIBLE_DIVIDE_ONCE_LS0", "precision_id": "FP32_STRICT:matmul_tf32=off,cudnn_tf32=off,float32_matmul_precision=highest,deterministic_algorithms=STRICT", "split": "VALIDATION"}`; device pin `db744ca3f0da27ac7b65c41b4b32f2c68ec9d0bd66bbc0dc33e874a226834e6b`
- bootstrap: paired user, 1000 resamples, seed 20260926; ratios bootstrapped directly; seed mean on the same W.
- contrasts: format PPSI_V4_ANA_CONTRASTS_4.
- multiplicity (06D §4; 06I §1; 06L-C C4 / C13): 33 pre-registered contrasts (v3's 26 with K20 / K20b / K21 on FA_WARM_FROZEN, + K_INIT, P5, K4w, K5w, K12w, K13w, K14w); primary = the pair K20b + K21, read unadjusted, outside Holm (06P P1 / P6); Holm step-down over the secondary family of 18 (K1, K2, K3, K8, K9b, K9c, K11b, K12, K17, K20, K20c, K_INIT, P5, K4w, K5w, K12w, K13w, K14w; 06P P6); tertiary unadjusted with a DESCRIPTIVE Holm band beside (step-down within the tertiary set); unadjusted 95% paired user-bootstrap percentile CIs (1,000 resamples, seed 20260926) for every contrast; K14 / K14w TOST on the unadjusted 90% CI; algebraic dependencies are not independent evidence; every other number is DESCRIPTIVE (06P).
- contrasts registry: company/gates/ANA_CONTRASTS_v4.json (06P).
- Anything added after the first VALIDATION read is POST_VALIDATION_EXPLORATORY (06L-C status line; §C5): never confirmatory, never in the loaded registry.
- **Only section 1 (K1-K21, K9b, K9c, K11b, K20b, K20c, K_INIT, P5, K4w, K5w, K12w, K13w, K14w) is CONFIRMATORY (each K under its registered rule and seeds, bold); every other number is DESCRIPTIVE.**

- sanity trigger (amendment 04 §1 / 06E §6, not a verdict): R(C_SMALL CB / C_BIG PB) = 94.74 -> fired = False; R(C_SMALL PB / C_BIG PB) = 95.80

## 1. Confirmatory contrasts (06B §2; rule map 06D §3; hierarchy 06D §4; seeds 06E §2) — CONFIRMATORY

| id | contrast | question | hierarchy | rule | label | seeds | seeds status | point | 95% CI | CI vs 100/0 | users | decisions | harmed share | seed spread (SEED_SENSITIVE) | Holm band | claim key |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **K1** | **R(C_SMALL / C_BIG)** | **size cost** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **95.80** | **[95.33, 96.26]** | **below** | **40764** | **519714** | **0.4638** | **0.21 (no)** | **[95.09, 96.50]** | **BELOW** |
| K1 | R(C_SMALL / C_BIG) | size cost | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 100.01 | [99.48, 100.58] | includes | 40764 | 519714 | 0.4090 | 0.01 (no) | - | - |
| **K2** | **R(FA / C_BIG)** | **federated retention (headline)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **86.07** | **[85.56, 86.62]** | **below** | **40764** | **519714** | **0.5444** | **0.36 (no)** | **[85.24, 86.91]** | **BELOW** |
| K2 | R(FA / C_BIG) | federated retention (headline) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 90.86 | [90.27, 91.52] | below | 40764 | 519714 | 0.4831 | 0.19 (no) | - | - |
| **K3** | **R(FA / C_SMALL)** | **federation cost at matched size** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **89.85** | **[89.36, 90.34]** | **below** | **40764** | **519714** | **0.5257** | **0.57 (SEED_SENSITIVE)** | **[89.12, 90.59]** | **BELOW** |
| K3 | R(FA / C_SMALL) | federation cost at matched size | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 90.85 | [90.30, 91.36] | below | 40764 | 519714 | 0.5072 | 0.20 (no) | - | - |
| **K4** | **R(FP / C_BIG)** | **FedProx retention** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **86.07** | **[85.56, 86.62]** | **below** | **40764** | **519714** | **0.5445** | **0.35 (no)** | **[85.27, 86.88]** | **BELOW** |
| K4 | R(FP / C_BIG) | FedProx retention | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 90.86 | [90.27, 91.52] | below | 40764 | 519714 | 0.4830 | 0.19 (no) | - | - |
| **K5** | **R(PF / C_BIG)** | **personalised-FL retention** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **86.09** | **[85.57, 86.63]** | **below** | **40764** | **519714** | **0.5449** | **0.35 (no)** | **[85.29, 86.89]** | **BELOW** |
| K5 | R(PF / C_BIG) | personalised-FL retention | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 90.88 | [90.30, 91.53] | below | 40764 | 519714 | 0.4834 | 0.18 (no) | - | - |
| **K6** | **R(FT_FA / C_BIG)** | **deployed on-device system (FL + local fine-tune)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **90.11** | **[89.58, 90.67]** | **below** | **40764** | **519714** | **0.5183** | **0.26 (no)** | **[89.32, 90.90]** | **BELOW** |
| K6 | R(FT_FA / C_BIG) | deployed on-device system (FL + local fine-tune) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K7** | **R(FT_C / C_BIG)** | **cloud-trained + local fine-tune** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **98.20** | **[97.72, 98.68]** | **below** | **40764** | **519714** | **0.4422** | **0.41 (no)** | **[97.60, 98.81]** | **BELOW** |
| K7 | R(FT_C / C_BIG) | cloud-trained + local fine-tune | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K8** | **R(LO / C_BIG)** | **device-only retention** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **3.10** | **[2.91, 3.30]** | **below** | **40764** | **519714** | **0.8390** | **0.35 (SEED_SENSITIVE)** | **[2.80, 3.41]** | **BELOW** |
| K8 | R(LO / C_BIG) | device-only retention | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K9** | **FT_C − C_SMALL** | **local fine-tuning gain (paired Δ)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.003879** | **[0.003436, 0.004340]** | **above** | **40764** | **519714** | **0.3602** | **0.000326 (no)** | **[0.003230, 0.004529]** | **ABOVE** |
| K9 | FT_C − C_SMALL | local fine-tuning gain (paired Δ) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K9b** | **FT_C − C_SMALL** | **local fine-tuning gain (paired Δ) for base-unseen users (UNSEEN_HOLDOUT)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **0.004865** | **[0.003873, 0.005919]** | **above** | **10224** | **132225** | **0.3082** | - | **[0.003310, 0.006420]** | **ABOVE** |
| K9b | FT_C − C_SMALL | local fine-tuning gain (paired Δ) for base-unseen users (UNSEEN_HOLDOUT) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K9c** | **FT_PRE − PRE** | **local fine-tuning gain (paired Δ) over θ_pre on MAIN: the realistic L5 answer (FT_PRE = θ_pre + the registered FT_C procedure with SELECTION_FT_C)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.003976** | **[0.003329, 0.004640]** | **above** | **40764** | **519714** | **0.3454** | **0.000241 (no)** | **[0.003028, 0.004924]** | **ABOVE** |
| K9c | FT_PRE − PRE | local fine-tuning gain (paired Δ) over θ_pre on MAIN: the realistic L5 answer (FT_PRE = θ_pre + the registered FT_C procedure with SELECTION_FT_C) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K10** | **FT_FA − FA** | **local fine-tuning gain after FL (paired Δ)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.006508** | **[0.006044, 0.006971]** | **above** | **40764** | **519714** | **0.3420** | **0.000153 (no)** | **[0.005841, 0.007174]** | **ABOVE** |
| K10 | FT_FA − FA | local fine-tuning gain after FL (paired Δ) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K11** | **FA − LO** | **collaboration value (paired Δ)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.133756** | **[0.132176, 0.135451]** | **above** | **40764** | **519714** | **0.0131** | **0.001129 (no)** | **[0.131467, 0.136045]** | **ABOVE** |
| K11 | FA − LO | collaboration value (paired Δ) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K11b** | **FA − LAST** | **collaboration value over the best own-history heuristic (paired Δ; LAST = the no-training device heuristic)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.067647** | **[0.066146, 0.069190]** | **above** | **40764** | **519714** | **0.2222** | **0.000558 (no)** | **[0.065327, 0.069967]** | **ABOVE** |
| K11b | FA − LAST | collaboration value over the best own-history heuristic (paired Δ; LAST = the no-training device heuristic) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.067647 | [0.066146, 0.069190] | above | 40764 | 519714 | 0.2222 | 0.000558 (no) | - | - |
| K12 | R(DP8 / FA_1024) | privacy cost at ε = 8 | secondary | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 98.71 | [98.09, 99.40] | below | 40764 | 519714 | 0.1820 | 3.98 (SEED_SENSITIVE) | - | - |
| **K12** | **R(DP8 / FA_1024)** | **privacy cost at ε = 8** | **secondary** | **CONTROLLED_BUDGET** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **98.71** | **[98.09, 99.40]** | **below** | **40764** | **519714** | **0.1820** | **3.98 (SEED_SENSITIVE)** | **[97.88, 99.53]** | **BELOW** |
| K13 | R(DP1 / FA_1024) | privacy cost at ε = 1 (seed 2026) | tertiary | PRACTICAL_BEST | DESCRIPTIVE | [2026] | COMPLETE | 96.27 | [95.06, 97.42] | below | 40764 | 519714 | 0.1848 | - | - | - |
| **K13** | **R(DP1 / FA_1024)** | **privacy cost at ε = 1 (seed 2026)** | **tertiary** | **CONTROLLED_BUDGET** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **96.27** | **[95.06, 97.42]** | **below** | **40764** | **519714** | **0.1848** | - | **[94.83, 97.71]** | **BELOW** |
| **K14** | **R(FA_Q8 / FA)** | **compression cost (8-bit uploads)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **99.82** | **[99.62, 100.00]** | **includes** | **40764** | **519714** | **0.3870** | **0.20 (SEED_SENSITIVE)** | **[99.60, 100.04]** | **INCLUDES** |
| K14 | R(FA_Q8 / FA) | compression cost (8-bit uploads) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 99.82 | [99.62, 100.00] | includes | 40764 | 519714 | 0.3870 | 0.20 (SEED_SENSITIVE) | - | - |
| **K15** | **R(C_BIG / C_FULL)** | **cloud data advantage (seed 2026; if C_FULL runs)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **83.01** | **[82.41, 83.58]** | **below** | **40764** | **519714** | **0.5790** | - | **[82.22, 83.79]** | **BELOW** |
| K15 | R(C_BIG / C_FULL) | cloud data advantage (seed 2026; if C_FULL runs) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026] | COMPLETE | 78.72 | [78.10, 79.26] | below | 40764 | 519714 | 0.6113 | - | - | - |
| K15 | R(C_BIG / C_FULL) | cloud data advantage (seed 2026; if C_FULL runs) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 78.54 | [78.04, 78.99] | below | 40764 | 519714 | 0.6508 | 0.36 (no) | - | - |
| K15 | R(C_BIG / C_FULL) | cloud data advantage (seed 2026; if C_FULL runs) | tertiary | PRACTICAL_BEST | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 82.90 | [82.43, 83.35] | below | 40764 | 519714 | 0.6176 | 0.21 (no) | - | - |
| **K16** | **R(INT8 / FP32) for C_SMALL** | **on-device quantisation cost (seed 2026)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **99.84** | **[99.62, 100.07]** | **includes** | **16384** | **206445** | **0.2633** | - | **[99.58, 100.10]** | **INCLUDES** |
| K16 | R(INT8 / FP32) for C_SMALL | on-device quantisation cost (seed 2026) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K17** | **R(FA / C_FULL)** | **federated retention vs the full-data cloud model (seed 2026)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **71.60** | **[71.02, 72.21]** | **below** | **40764** | **519714** | **0.6360** | - | **[70.71, 72.49]** | **BELOW** |
| K17 | R(FA / C_FULL) | federated retention vs the full-data cloud model (seed 2026) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026] | COMPLETE | 71.60 | [71.02, 72.21] | below | 40764 | 519714 | 0.6360 | - | - | - |
| K17 | R(FA / C_FULL) | federated retention vs the full-data cloud model (seed 2026) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 71.36 | [70.82, 71.87] | below | 40764 | 519714 | 0.6721 | 0.48 (no) | - | - |
| K17 | R(FA / C_FULL) | federated retention vs the full-data cloud model (seed 2026) | secondary | PRACTICAL_BEST | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 71.36 | [70.82, 71.87] | below | 40764 | 519714 | 0.6721 | 0.48 (no) | - | - |
| **K18** | **R(FT_C / C_FULL)** | **cloud-trained + local fine-tune vs the full-data cloud model (seed 2026)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **81.34** | **[80.79, 81.91]** | **below** | **40764** | **519714** | **0.5845** | - | **[80.59, 82.10]** | **BELOW** |
| K18 | R(FT_C / C_FULL) | cloud-trained + local fine-tune vs the full-data cloud model (seed 2026) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| K18 | R(FT_C / C_FULL) | cloud-trained + local fine-tune vs the full-data cloud model (seed 2026) | tertiary | PRACTICAL_BEST | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 81.41 | [80.95, 81.89] | below | 40764 | 519714 | 0.6179 | 0.14 (no) | - | - |
| **K19** | **R(LO / C_FULL)** | **device-only vs the full-data cloud model (seed 2026)** | **tertiary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **2.43** | **[2.25, 2.61]** | **below** | **40764** | **519714** | **0.8464** | - | **[2.19, 2.67]** | **BELOW** |
| K19 | R(LO / C_FULL) | device-only vs the full-data cloud model (seed 2026) | tertiary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| K19 | R(LO / C_FULL) | device-only vs the full-data cloud model (seed 2026) | tertiary | PRACTICAL_BEST | DESCRIPTIVE (all seeds) | [2026, 2027] | - | 2.57 | [2.41, 2.74] | below | 40764 | 519714 | 0.8654 | 0.29 (SEED_SENSITIVE) | - | - |
| **K20** | **R(FA_WARM_FROZEN / C_BIG)** | **deployment-branch federated retention vs the C_BIG cloud model (warm start from theta_pre, S8 frozen item tables = FA_WARM*; seeds 2026, 2027)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **106.98** | **[106.28, 107.67]** | **above** | **40764** | **519714** | **0.3628** | **0.02 (no)** | **[106.01, 107.95]** | **ABOVE** |
| K20 | R(FA_WARM_FROZEN / C_BIG) | deployment-branch federated retention vs the C_BIG cloud model (warm start from theta_pre, S8 frozen item tables = FA_WARM*; seeds 2026, 2027) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 112.92 | [112.16, 113.73] | above | 40764 | 519714 | 0.3230 | 0.16 (no) | - | - |
| **K20b** | **R(FA_WARM_FROZEN / C_FULL)** | **deployment-branch federated retention vs its matched cloud reference C_FULL (2 FA_WARM_FROZEN seeds vs C_FULL s2026) — PRIMARY with K21 (06N N1; 06P P1)** | **primary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **88.81** | **[88.27, 89.36]** | **below** | **40764** | **519714** | **0.5306** | **0.00 (no)** | - | **BELOW** |
| K20b | R(FA_WARM_FROZEN / C_FULL) | deployment-branch federated retention vs its matched cloud reference C_FULL (2 FA_WARM_FROZEN seeds vs C_FULL s2026) — PRIMARY with K21 (06N N1; 06P P1) | primary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 88.81 | [88.27, 89.35] | below | 40764 | 519714 | 0.5295 | 0.04 (no) | - | - |
| **K20c** | **R(PRE / C_FULL)** | **the pretrained model alone vs its matched cloud reference C_FULL (PRE single seed)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **89.03** | **[88.49, 89.57]** | **below** | **40764** | **519714** | **0.5252** | - | **[88.27, 89.80]** | **BELOW** |
| K20c | R(PRE / C_FULL) | the pretrained model alone vs its matched cloud reference C_FULL (PRE single seed) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026] | COMPLETE | 89.03 | [88.49, 89.57] | below | 40764 | 519714 | 0.5252 | - | - | - |
| **K21** | **FA_WARM_FROZEN − PRE** | **value of federated training on the private users over the pretrained model alone (paired Δ) — PRIMARY with K20b (06N N1; 06P P1)** | **primary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **-0.000433** | **[-0.000758, -0.000101]** | **below** | **40764** | **519714** | **0.3624** | **0.000003 (no)** | - | **BELOW** |
| K21 | FA_WARM_FROZEN − PRE | value of federated training on the private users over the pretrained model alone (paired Δ) — PRIMARY with K20b (06N N1; 06P P1) | primary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | -0.000440 | [-0.000801, -0.000087] | below | 40764 | 519714 | 0.3588 | 0.000086 (no) | - | - |
| **K_INIT** | **FA_WARM_FROZEN − FA** | **value of the pretrained start (paired Δ, seeds 2026 / 2027; 06N N1; 06P P3)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.033704** | **[0.032688, 0.034760]** | **above** | **40764** | **519714** | **0.2642** | **0.000554 (no)** | **[0.032207, 0.035201]** | **ABOVE** |
| K_INIT | FA_WARM_FROZEN − FA | value of the pretrained start (paired Δ, seeds 2026 / 2027; 06N N1; 06P P3) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.033697 | [0.032691, 0.034751] | above | 40764 | 519714 | 0.2647 | 0.000472 (no) | - | - |
| **P5** | **FT_FA_WARM_FROZEN − FA_WARM_FROZEN** | **local fine-tuning gain after the pretrained-line FL (paired Δ, base = the same-seed best.pt, harmed-client share beside; 06P P5)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.004033** | **[0.003429, 0.004672]** | **above** | **40764** | **519714** | **0.3658** | **0.000258 (no)** | **[0.003151, 0.004915]** | **ABOVE** |
| P5 | FT_FA_WARM_FROZEN − FA_WARM_FROZEN | local fine-tuning gain after the pretrained-line FL (paired Δ, base = the same-seed best.pt, harmed-client share beside; 06P P5) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |
| **K4w** | **R(FP_WARM_FROZEN / C_FULL)** | **FedProx retention on the pretrained line vs C_FULL (06P P4)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **88.81** | **[88.28, 89.37]** | **below** | **40764** | **519714** | **0.5306** | **0.00 (no)** | **[88.10, 89.53]** | **BELOW** |
| K4w | R(FP_WARM_FROZEN / C_FULL) | FedProx retention on the pretrained line vs C_FULL (06P P4) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 88.81 | [88.27, 89.35] | below | 40764 | 519714 | 0.5295 | 0.05 (no) | - | - |
| **K5w** | **R(PF_WARM_FROZEN / C_FULL)** | **personalised-FL retention on the pretrained line vs C_FULL (06P P4)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **88.80** | **[88.25, 89.34]** | **below** | **40764** | **519714** | **0.5294** | **0.06 (no)** | **[88.09, 89.50]** | **BELOW** |
| K5w | R(PF_WARM_FROZEN / C_FULL) | personalised-FL retention on the pretrained line vs C_FULL (06P P4) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 88.81 | [88.29, 89.35] | below | 40764 | 519714 | 0.5294 | 0.00 (no) | - | - |
| K12w | R(DP8_FROZEN / FA_1024_WARM_FROZEN) | privacy cost at ε = 8 on the pretrained line (06P P4) | secondary | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 100.00 | [99.98, 100.01] | includes | 40764 | 519714 | 0.0730 | 0.00 (no) | - | - |
| **K12w** | **R(DP8_FROZEN / FA_1024_WARM_FROZEN)** | **privacy cost at ε = 8 on the pretrained line (06P P4)** | **secondary** | **CONTROLLED_BUDGET** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **99.96** | **[99.93, 100.00]** | **includes** | **40764** | **519714** | **0.1787** | **0.02 (no)** | **[99.92, 100.01]** | **INCLUDES** |
| K13w | R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN) | privacy cost at ε = 1 on the pretrained line (seed 2026; 06P P4) | secondary | PRACTICAL_BEST | DESCRIPTIVE | [2026] | COMPLETE | 99.99 | [99.96, 100.03] | includes | 40764 | 519714 | 0.0858 | - | - | - |
| **K13w** | **R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN)** | **privacy cost at ε = 1 on the pretrained line (seed 2026; 06P P4)** | **secondary** | **CONTROLLED_BUDGET** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **99.88** | **[99.80, 99.96]** | **below** | **40764** | **519714** | **0.1974** | - | **[99.79, 99.98]** | **BELOW** |
| **K14w** | **R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN)** | **compression cost (8-bit uploads) on the pretrained line (06P P4)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **99.99** | **[99.90, 100.09]** | **includes** | **40764** | **519714** | **0.3129** | **0.07 (no)** | **[99.88, 100.10]** | **INCLUDES** |
| K14w | R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN) | compression cost (8-bit uploads) on the pretrained line (06P P4) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 99.99 | [99.98, 100.01] | includes | 40764 | 519714 | 0.0537 | 0.02 (no) | - | - |

- K12 PRACTICAL_BEST (DESCRIPTIVE): best-of-12 checkpoint selection on the users' own validation data is not covered by the DP accountant (06D §3).
- K13 PRACTICAL_BEST (DESCRIPTIVE): best-of-12 checkpoint selection on the users' own validation data is not covered by the DP accountant (06D §3).
- K12w PRACTICAL_BEST (DESCRIPTIVE): best-of-12 checkpoint selection on the users' own validation data is not covered by the DP accountant (06D §3).
- K13w PRACTICAL_BEST (DESCRIPTIVE): best-of-12 checkpoint selection on the users' own validation data is not covered by the DP accountant (06D §3).

### 1b. Deployment branch (06K; DL 206; 06L-C C7: the Edge answer and the L5 row lead with these) — K20, K20b, K20c, K21, K9c

| id | contrast | question | hierarchy | rule | label | seeds | seeds status | point | 95% CI | CI vs 100/0 | users | decisions | harmed share | seed spread (SEED_SENSITIVE) | Holm band | claim key |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **K20** | **R(FA_WARM_FROZEN / C_BIG)** | **deployment-branch federated retention vs the C_BIG cloud model (warm start from theta_pre, S8 frozen item tables = FA_WARM*; seeds 2026, 2027)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **106.98** | **[106.28, 107.67]** | **above** | **40764** | **519714** | **0.3628** | **0.02 (no)** | **[106.01, 107.95]** | **ABOVE** |
| K20 | R(FA_WARM_FROZEN / C_BIG) | deployment-branch federated retention vs the C_BIG cloud model (warm start from theta_pre, S8 frozen item tables = FA_WARM*; seeds 2026, 2027) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 112.92 | [112.16, 113.73] | above | 40764 | 519714 | 0.3230 | 0.16 (no) | - | - |
| **K20b** | **R(FA_WARM_FROZEN / C_FULL)** | **deployment-branch federated retention vs its matched cloud reference C_FULL (2 FA_WARM_FROZEN seeds vs C_FULL s2026) — PRIMARY with K21 (06N N1; 06P P1)** | **primary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **88.81** | **[88.27, 89.36]** | **below** | **40764** | **519714** | **0.5306** | **0.00 (no)** | - | **BELOW** |
| K20b | R(FA_WARM_FROZEN / C_FULL) | deployment-branch federated retention vs its matched cloud reference C_FULL (2 FA_WARM_FROZEN seeds vs C_FULL s2026) — PRIMARY with K21 (06N N1; 06P P1) | primary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 88.81 | [88.27, 89.35] | below | 40764 | 519714 | 0.5295 | 0.04 (no) | - | - |
| **K20c** | **R(PRE / C_FULL)** | **the pretrained model alone vs its matched cloud reference C_FULL (PRE single seed)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026]** | **COMPLETE** | **89.03** | **[88.49, 89.57]** | **below** | **40764** | **519714** | **0.5252** | - | **[88.27, 89.80]** | **BELOW** |
| K20c | R(PRE / C_FULL) | the pretrained model alone vs its matched cloud reference C_FULL (PRE single seed) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026] | COMPLETE | 89.03 | [88.49, 89.57] | below | 40764 | 519714 | 0.5252 | - | - | - |
| **K21** | **FA_WARM_FROZEN − PRE** | **value of federated training on the private users over the pretrained model alone (paired Δ) — PRIMARY with K20b (06N N1; 06P P1)** | **primary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **-0.000433** | **[-0.000758, -0.000101]** | **below** | **40764** | **519714** | **0.3624** | **0.000003 (no)** | - | **BELOW** |
| K21 | FA_WARM_FROZEN − PRE | value of federated training on the private users over the pretrained model alone (paired Δ) — PRIMARY with K20b (06N N1; 06P P1) | primary | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | -0.000440 | [-0.000801, -0.000087] | below | 40764 | 519714 | 0.3588 | 0.000086 (no) | - | - |
| **K9c** | **FT_PRE − PRE** | **local fine-tuning gain (paired Δ) over θ_pre on MAIN: the realistic L5 answer (FT_PRE = θ_pre + the registered FT_C procedure with SELECTION_FT_C)** | **secondary** | **PRACTICAL_BEST** | **CONFIRMATORY** | **[2026, 2027]** | **COMPLETE** | **0.003976** | **[0.003329, 0.004640]** | **above** | **40764** | **519714** | **0.3454** | **0.000241 (no)** | **[0.003028, 0.004924]** | **ABOVE** |
| K9c | FT_PRE − PRE | local fine-tuning gain (paired Δ) over θ_pre on MAIN: the realistic L5 answer (FT_PRE = θ_pre + the registered FT_C procedure with SELECTION_FT_C) | secondary | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - |


- (1) PRE is trained on 1,217,610 users (9x MAIN) while C_BIG saw 131,072: K20 = R(FA_WARM / C_BIG) can exceed 100 and mixes a data advantage into the FL retention. The matched cloud reference of the deployment branch is C_FULL (TIEBREAK = PRE users + MAIN users): R(FA_WARM / C_FULL) and R(PRE / C_FULL) are printed beside K20 as DESCRIPTIVE, and the thesis text for the deployment branch leads with them. K21 = FA_WARM - PRE stays the value of federated training on the private users.
- (2) theta_pre is ONE seed (2026) shared by FA_WARM s2026 and s2027: their seed spread measures FL variance only.
- (3) PRE applies the SMALL recipe (tuned at MAIN scale, 2.6M decisions) to 24.1M decisions at 6.0 EFE (about 9x more updates) without re-tuning: this can only understate PRE. PRE marks on UNSEEN_VAL are DESCRIPTIVE and never select.

### 1c. Per-K statistics (06L-C C13-C15) — DESCRIPTIVE beside each K

Holm band: Holm step-down on normal intervals from the bootstrap SE (06L-C C13); after the first non-rejection every non-rejected K's band is at alpha / (m - r), r = the number of rejections (REV-DELTA-03 M3: band and decision agree); Bonferroni band beside it; family = 18 registered contrasts, 18 computed.

| id | rule | per-seed points | seed spread | SEED_SENSITIVE | user + seed interval | Holm p | Holm reject | Holm band | Bonferroni band | harmed share | harmed share, seed-vs-seed null | K14 equivalence (±1 % rel.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K1 | PRACTICAL_BEST | {"2026": 95.69055970605243, "2027": 95.90149997017855} | 0.21 | False | [95.29, 96.30] | 0.0000 | True | [95.09, 96.50] | [95.09, 96.50] | 0.4638 | 0.3960 | - |
| K2 | PRACTICAL_BEST | {"2026": 86.25359479525116, "2027": 85.89458846466356} | 0.36 | False | [85.42, 86.73] | 0.0000 | True | [85.24, 86.91] | [85.24, 86.91] | 0.5444 | 0.3940 | - |
| K3 | PRACTICAL_BEST | {"2026": 90.13803980268246, "2027": 89.5654275390617} | 0.57 | True | [89.11, 90.59] | 0.0000 | True | [89.12, 90.59] | [89.11, 90.60] | 0.5257 | 0.3940 | - |
| K4 | PRACTICAL_BEST | {"2026": 86.25075296092923, "2027": 85.89678792176436} | 0.35 | False | [85.43, 86.72] | 0.0000 | - | [85.27, 86.88] | [85.27, 86.88] | 0.5445 | 0.3941 | - |
| K5 | PRACTICAL_BEST | {"2026": 86.26768109172573, "2027": 85.9178834040468} | 0.35 | False | [85.45, 86.74] | 0.0000 | - | [85.29, 86.89] | [85.29, 86.90] | 0.5449 | 0.3941 | - |
| K6 | PRACTICAL_BEST | {"2026": 90.2432038236359, "2027": 89.9786429788056} | 0.26 | False | [89.51, 90.72] | 0.0000 | - | [89.32, 90.90] | [89.31, 90.92] | 0.5183 | 0.3923 | - |
| K7 | PRACTICAL_BEST | {"2026": 97.9958867156848, "2027": 98.40879648524701} | 0.41 | False | [97.58, 98.82] | 0.0000 | - | [97.60, 98.81] | [97.50, 98.90] | 0.4422 | 0.3958 | - |
| K8 | PRACTICAL_BEST | {"2026": 2.926053102068987, "2027": 3.280131447471269} | 0.35 | True | [2.70, 3.50] | 0.0000 | True | [2.80, 3.41] | [2.80, 3.41] | 0.8390 | 0.0389 | - |
| K9 | PRACTICAL_BEST | {"2026": 0.0037160979174215847, "2027": 0.004042283072241276} | 0.000326 | False | [0.003324, 0.004434] | 0.0000 | - | [0.003230, 0.004529] | [0.003210, 0.004548] | 0.3602 | 0.3958 | - |
| K9b | PRACTICAL_BEST | {"2026": 0.004865228086567336} | - | - | - | 0.0000 | True | [0.003310, 0.006420] | [0.003269, 0.006461] | 0.3082 | - | - |
| K9c | PRACTICAL_BEST | {"2026": 0.004096865711617936, "2027": 0.003855460337779948} | 0.000241 | False | [0.003291, 0.004661] | 0.0000 | True | [0.003028, 0.004924] | [0.002995, 0.004957] | 0.3454 | 0.2338 | - |
| K10 | PRACTICAL_BEST | {"2026": 0.006431095345588722, "2027": 0.006584344663429692} | 0.000153 | False | [0.006013, 0.007002] | 0.0000 | - | [0.005841, 0.007174] | [0.005813, 0.007203] | 0.3420 | 0.3923 | - |
| K11 | PRACTICAL_BEST | {"2026": 0.1343207722184593, "2027": 0.1331916744254244} | 0.001129 | False | [0.131777, 0.135735] | 0.0000 | - | [0.131467, 0.136045] | [0.131336, 0.136176] | 0.0131 | 0.3940 | - |
| K11b | PRACTICAL_BEST | {"2026": 0.06792568470201922, "2027": 0.06736815615535817} | 0.000558 | False | [0.065968, 0.069326] | 0.0000 | True | [0.065327, 0.069967] | [0.065225, 0.070069] | 0.2222 | 0.3940 | - |
| K12 | CONTROLLED_BUDGET | {"2026": 100.6968351575337, "2027": 96.71626477543117} | 3.98 | True | [94.75, 102.66] | 0.0003 | True | [97.88, 99.53] | [97.72, 99.69] | 0.1820 | 0.1922 | - |
| K13 | CONTROLLED_BUDGET | {"2026": 96.26911805208312} | - | - | - | 0.0000 | - | [94.83, 97.71] | [94.53, 98.01] | 0.1848 | - | - |
| K14 | PRACTICAL_BEST | {"2026": 99.71848512331422, "2027": 99.92077566818509} | 0.20 | True | [99.54, 100.09] | 0.1265 | - | [99.60, 100.04] | [99.54, 100.10] | 0.3870 | 0.3964 | True (90% CI [99.66, 99.97]) |
| K15 | PRACTICAL_BEST | {"2026": 83.00856241002954} | - | - | - | 0.0000 | - | [82.22, 83.79] | [82.16, 83.85] | 0.5790 | 0.4019 | - |
| K16 | PRACTICAL_BEST | {"2026": 99.83946474016732} | - | - | - | 0.1695 | - | [99.58, 100.10] | [99.50, 100.18] | 0.2633 | - | - |
| K17 | PRACTICAL_BEST | {"2026": 71.59786906651006} | - | - | - | 0.0000 | True | [70.71, 72.49] | [70.66, 72.54] | 0.6360 | 0.3940 | - |
| K18 | PRACTICAL_BEST | {"2026": 81.34497678365108} | - | - | - | 0.0000 | - | [80.59, 82.10] | [80.52, 82.17] | 0.5845 | 0.3958 | - |
| K19 | PRACTICAL_BEST | {"2026": 2.428874615381541} | - | - | - | 0.0000 | - | [2.19, 2.67] | [2.16, 2.70] | 0.8464 | 0.0389 | - |
| K20 | PRACTICAL_BEST | {"2026": 106.9905148230744, "2027": 106.97205322551028} | 0.02 | False | [106.30, 107.66] | 0.0000 | True | [106.01, 107.95] | [105.94, 108.02] | 0.3628 | 0.3139 | - |
| K20b | PRACTICAL_BEST | {"2026": 88.81128826972363, "2027": 88.8095439421101} | 0.00 | False | [88.28, 89.34] | - | - | - | - | 0.5306 | 0.3139 | - |
| K20c | PRACTICAL_BEST | {"2026": 89.03355666466766} | - | - | - | 0.0000 | True | [88.27, 89.80] | [88.21, 89.86] | 0.5252 | - | - |
| K21 | PRACTICAL_BEST | {"2026": -0.0004316278223436276, "2027": -0.0004350151700116267} | 0.000003 | False | [-0.000769, -0.000097] | - | - | - | - | 0.3624 | 0.3139 | - |
| K_INIT | PRACTICAL_BEST | {"2026": 0.0334271125125194, "2027": 0.03398125371151245} | 0.000554 | False | [0.032502, 0.034907] | 0.0000 | True | [0.032207, 0.035201] | [0.032067, 0.035342] | 0.2642 | 0.3139 | - |
| P5 | PRACTICAL_BEST | {"2026": 0.003903966935541181, "2027": 0.004162294880736328} | 0.000258 | False | [0.003342, 0.004724] | 0.0000 | True | [0.003151, 0.004915] | [0.003052, 0.005014] | 0.3658 | 0.3199 | - |
| K4w | PRACTICAL_BEST | {"2026": 88.81616707733572, "2027": 88.81325758386465} | 0.00 | False | [88.28, 89.35] | 0.0000 | True | [88.10, 89.53] | [88.00, 89.63] | 0.5306 | 0.3141 | - |
| K5w | PRACTICAL_BEST | {"2026": 88.76735024190268, "2027": 88.82278186872756} | 0.06 | False | [88.25, 89.34] | 0.0000 | True | [88.09, 89.50] | [87.97, 89.62] | 0.5294 | 0.3082 | - |
| K12w | CONTROLLED_BUDGET | {"2026": 99.97336595616359, "2027": 99.95384185289241} | 0.02 | False | [99.92, 100.01] | 0.1051 | False | [99.92, 100.01] | [99.91, 100.02] | 0.1787 | 0.1828 | - |
| K13w | CONTROLLED_BUDGET | {"2026": 99.88211839716949} | - | - | - | 0.0085 | True | [99.79, 99.98] | [99.76, 100.00] | 0.1974 | - | - |
| K14w | PRACTICAL_BEST | {"2026": 99.95398538382194, "2027": 100.02247080591754} | 0.07 | False | [99.87, 100.10] | 0.8057 | False | [99.88, 100.10] | [99.85, 100.13] | 0.3129 | 0.3095 | True (90% CI [99.91, 100.07]) |

### 1e. Registered DESCRIPTIVE contrasts (06L-C C8: S8 − S7 with the bytes ratio)

| id | contrast | question | hierarchy | rule | label | seeds | seeds status | point | 95% CI | CI vs 100/0 | users | decisions | harmed share | seed spread (SEED_SENSITIVE) | Holm band | claim key | bytes ratio S8 / S7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S8-S7 | FA_WARM_FROZEN − FA_WARM | cost of not sending the item tables (paired Δ, reported with the bytes ratio; DESCRIPTIVE) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.000854 | [0.000415, 0.001301] | above | 40764 | 519714 | 0.3844 | 0.000123 (no) | - | ABOVE | 0.0950 |
| S8-S7 | FA_WARM_FROZEN − FA_WARM | cost of not sending the item tables (paired Δ, reported with the bytes ratio; DESCRIPTIVE) | - | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.001826 | [0.001262, 0.002408] | above | 40764 | 519714 | 0.3799 | 0.000087 (no) | - | - | - |
| D_S7_C_FULL | R(FA_WARM / C_FULL) | the S7 full-model warm start vs C_FULL (DESCRIPTIVE beside the primary; 06P P2) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 88.25 | [87.77, 88.71] | below | 40764 | 519714 | 0.5699 | 0.18 (no) | - | BELOW | - |
| D_S7_C_FULL | R(FA_WARM / C_FULL) | the S7 full-model warm start vs C_FULL (DESCRIPTIVE beside the primary; 06P P2) | - | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 87.75 | [87.26, 88.24] | below | 40764 | 519714 | 0.5688 | 0.33 (no) | - | - | - |
| D_S7_PRE | FA_WARM − PRE | the S7 full-model warm start over theta_pre (DESCRIPTIVE beside the primary; 06P P2) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | -0.001287 | [-0.001726, -0.000833] | below | 40764 | 519714 | 0.3996 | 0.000120 (no) | - | BELOW | - |
| D_S7_PRE | FA_WARM − PRE | the S7 full-model warm start over theta_pre (DESCRIPTIVE beside the primary; 06P P2) | - | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | -0.002266 | [-0.002842, -0.001630] | below | 40764 | 519714 | 0.4142 | 0.000173 (no) | - | - | - |
| D_PFw_FAw | PF_WARM_FROZEN − FA_WARM_FROZEN | personalisation on the pretrained line (DESCRIPTIVE, 06L A5 caveat; 06P P4) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | -0.000030 | [-0.000191, 0.000145] | includes | 40764 | 519714 | 0.3130 | 0.000111 (no) | - | INCLUDES | 1.0000 |
| D_PFw_FAw | PF_WARM_FROZEN − FA_WARM_FROZEN | personalisation on the pretrained line (DESCRIPTIVE, 06L A5 caveat; 06P P4) | - | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.000011 | [-0.000032, 0.000052] | includes | 40764 | 519714 | 0.1017 | 0.000091 (SEED_SENSITIVE) | - | - | - |
| D_FA1024w_FAw | R(FA_1024_WARM_FROZEN / FA_WARM_FROZEN) | clipping + Poisson-cohort cost on the pretrained line (DESCRIPTIVE; 06P P4) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 100.15 | [99.96, 100.35] | includes | 40764 | 519714 | 0.3721 | 0.02 (no) | - | INCLUDES | 0.8900 |
| D_FA1024w_FAw | R(FA_1024_WARM_FROZEN / FA_WARM_FROZEN) | clipping + Poisson-cohort cost on the pretrained line (DESCRIPTIVE; 06P P4) | - | CONTROLLED_BUDGET | DESCRIPTIVE | [2026, 2027] | COMPLETE | 99.76 | [99.54, 99.99] | below | 40764 | 519714 | 0.3795 | 0.01 (no) | - | - | - |
| D_FTw_S7 | FT_FA_WARM − FA_WARM | local fine-tuning gain after the S7 warm FL (DESCRIPTIVE; 06P P5) | - | PRACTICAL_BEST | DESCRIPTIVE | [2026, 2027] | COMPLETE | 0.004187 | [0.003571, 0.004821] | above | 40764 | 519714 | 0.3703 | 0.000186 (no) | - | ABOVE | - |
| D_FTw_S7 | FT_FA_WARM − FA_WARM | local fine-tuning gain after the S7 warm FL (DESCRIPTIVE; 06P P5) | - | CONTROLLED_BUDGET | DESCRIPTIVE | NOT_AVAILABLE | NOT_AVAILABLE | - | - | - | - | - | - | - | - | - | - |

## 2. Retention table (MAIN) — DESCRIPTIVE

### PRACTICAL_BEST

| arm | seed | MRR@20 | NDCG@10 | HR@10 | R% vs C_BIG [CI] | R% vs C_SMALL [CI] | bytes/visit | bytes/client | bytes/client (dropped visits charged, 06I 7b) | bytes/client (dropped visits charged down + up; DESCRIPTIVE) | bytes fleet | ε | params | model bytes | rounds | consumed EFE | still improving at 6.0 (PB mark >= 5.5; DESCRIPTIVE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_BIG | 2026 | 0.161196 | 0.196655 | 0.334324 | 100.00 [100.00, 100.00] | 104.50 [103.84, 105.15] | - | - | - | - | - | - | - | - | - | - | - |
| C_BIG | 2027 | 0.161221 | 0.196701 | 0.334372 | 100.00 [100.00, 100.00] | 104.27 [103.64, 104.90] | - | - | - | - | - | - | - | - | - | - | - |
| **C_BIG** | mean(2) |  |  |  | 100.00 [100.00, 100.00] | 104.39 [103.88, 104.89] |  |  |  |  |  |  |  |  |  |  |  |
| C_FULL | 2026 | 0.194192 | 0.235151 | 0.391644 | 120.47 [119.65, 121.35] | 125.89 [124.97, 126.89] | - | - | - | - | - | - | - | - | - | - | - |
| C_FULL | 2027 | 0.194716 | 0.235384 | 0.390927 | 120.78 [120.05, 121.60] | 125.94 [125.05, 126.88] | - | - | - | - | - | - | - | - | - | - | - |
| **C_FULL** | mean(2) |  |  |  | 120.62 [119.97, 121.32] | 125.92 [125.14, 126.74] |  |  |  |  |  |  |  |  |  |  |  |
| C_SMALL | 2026 | 0.154249 | 0.188848 | 0.324768 | 95.69 [95.10, 96.30] | 100.00 [100.00, 100.00] | - | - | - | - | - | - | - | - | - | - | - |
| C_SMALL | 2027 | 0.154613 | 0.189034 | 0.324274 | 95.90 [95.33, 96.49] | 100.00 [100.00, 100.00] | - | - | - | - | - | - | - | - | - | - | - |
| **C_SMALL** | mean(2) |  |  |  | 95.80 [95.33, 96.26] | 100.00 [100.00, 100.00] |  |  |  |  |  |  |  |  |  |  |  |
| FA | 2026 | 0.139037 | 0.171410 | 0.299358 | 86.25 [85.59, 86.99] | 90.14 [89.49, 90.80] | 165669672 | 994562798 | 1049540601 | 1104518409 | 130359335118936 | - | 20708708 | - | 13654 | 6.006 | True |
| FA | 2027 | 0.138480 | 0.170913 | 0.299084 | 85.89 [85.27, 86.55] | 89.57 [88.91, 90.19] | 165669672 | 993795575 | 1049156989 | 1104518409 | 130258773628032 | - | 20708708 | - | 13654 | 6.000 | True |
| **FA** | mean(2) |  |  |  | 86.07 [85.56, 86.62] | 89.85 [89.36, 90.34] |  |  |  |  |  |  |  |  |  |  | any=True |
| FP | 2026 | 0.139033 | 0.171404 | 0.299354 | 86.25 [85.58, 86.98] | 90.14 [89.49, 90.79] | 165669672 | 994562798 | 1049540601 | 1104518409 | 130359335118936 | - | 20708708 | - | 13654 | 6.006 | True |
| FP | 2027 | 0.138483 | 0.170894 | 0.299011 | 85.90 [85.27, 86.56] | 89.57 [88.91, 90.19] | 165669672 | 993795575 | 1049156989 | 1104518409 | 130258773628032 | - | 20708708 | - | 13654 | 6.000 | True |
| **FP** | mean(2) |  |  |  | 86.07 [85.56, 86.62] | 89.85 [89.36, 90.33] |  |  |  |  |  |  |  |  |  |  | any=True |
| PF | 2026 | 0.139060 | 0.171429 | 0.299378 | 86.27 [85.59, 86.99] | 90.15 [89.51, 90.81] | 165669672 | 994562798 | 1049540601 | 1104518409 | 130359335118936 | - | 20708708 | - | 13654 | 6.006 | True |
| PF | 2027 | 0.138517 | 0.170970 | 0.299179 | 85.92 [85.28, 86.58] | 89.59 [88.93, 90.20] | 165669672 | 993795575 | 1049156989 | 1104518409 | 130258773628032 | - | 20708708 | - | 13654 | 6.000 | True |
| **PF** | mean(2) |  |  |  | 86.09 [85.57, 86.63] | 89.87 [89.37, 90.35] |  |  |  |  |  |  |  |  |  |  | any=True |
| FA_1024 | 2026 | 0.034929 | 0.044257 | 0.082935 | 21.67 [21.09, 22.23] | 22.64 [22.07, 23.24] | 165669672 | 496891468 | 496891468 | 496891468 | 65128558465656 | inf | 20708708 | - | 384 | 6.006 | True |
| FA_1024 | 2027 | 0.034986 | 0.043340 | 0.078805 | 21.70 [21.07, 22.30] | 22.63 [21.99, 23.24] | 165669672 | 497704194 | 497704194 | 497704194 | 65235084064752 | inf | 20708708 | - | 384 | 6.006 | True |
| **FA_1024** | mean(2) |  |  |  | 21.68 [21.09, 22.24] | 22.64 [22.04, 23.21] |  |  |  |  |  |  |  |  |  |  | any=True |
| DP8 | 2026 | 0.035172 | 0.044328 | 0.082357 | 21.82 [21.25, 22.37] | 22.80 [22.22, 23.40] | 165669672 | 496891468 | 496891468 | 496891468 | 65128558465656 | 8.0000 | 20708708 | - | 384 | 6.006 | True |
| DP8 | 2027 | 0.033837 | 0.042320 | 0.077600 | 20.99 [20.39, 21.56] | 21.89 [21.27, 22.47] | 165669672 | 497704194 | 497704194 | 497704194 | 65235084064752 | 8.0000 | 20708708 | - | 384 | 6.006 | True |
| **DP8** | mean(2) |  |  |  | 21.40 [20.83, 21.96] | 22.34 [21.76, 22.91] |  |  |  |  |  |  |  |  |  |  | any=True |
| DP1 | 2026 | 0.033625 | 0.042577 | 0.079857 | 20.86 [20.28, 21.40] | 21.80 [21.22, 22.36] | 165669672 | 496891468 | 496891468 | 496891468 | 65128558465656 | 1.0000 | 20708708 | - | 384 | 6.006 | True |
| **DP1** | mean(1) |  |  |  | 20.86 [20.28, 21.40] | 21.80 [21.22, 22.36] |  |  |  |  |  |  |  |  |  |  | any=True |
| FA_Q8 | 2026 | 0.138646 | 0.170955 | 0.298784 | 86.01 [85.35, 86.69] | 89.88 [89.23, 90.53] | 103543748 | 621602968 | 676580770 | 690325359 | 81474744182524 | - | 20708708 | - | 13654 | 6.006 | True |
| FA_Q8 | 2027 | 0.138370 | 0.170552 | 0.298104 | 85.83 [85.18, 86.49] | 89.49 [88.85, 90.10] | 103543748 | 621123452 | 676484866 | 690325359 | 81411893127488 | - | 20708708 | - | 13654 | 6.000 | True |
| **FA_Q8** | mean(2) |  |  |  | 85.92 [85.38, 86.48] | 89.69 [89.20, 90.16] |  |  |  |  |  |  |  |  |  |  | any=True |
| FA_WARM | 2026 | 0.171550 | 0.209848 | 0.360119 | 106.42 [105.65, 107.21] | 111.22 [110.50, 111.98] | 165669672 | 994562798 | 1049540601 | 1104518409 | 130359335118936 | - | 20708708 | - | 13654 | 6.006 | False |
| FA_WARM | 2027 | 0.171669 | 0.209820 | 0.359602 | 106.48 [105.73, 107.20] | 111.03 [110.28, 111.75] | 165669672 | 993795575 | 1049156989 | 1104518409 | 130258773628032 | - | 20708708 | - | 13654 | 6.000 | False |
| **FA_WARM** | mean(2) |  |  |  | 106.45 [105.78, 107.12] | 111.12 [110.50, 111.75] |  |  |  |  |  |  |  |  |  |  | any=False |
| FA_WARM_FROZEN | 2026 | 0.172465 | 0.210774 | 0.360867 | 106.99 [106.21, 107.79] | 111.81 [111.08, 112.54] | 2110584 | 94449988 | 95150387 | 95850791 | 1660740457992 | - | 263822 | - | 13654 | 6.006 | False |
| FA_WARM_FROZEN | 2027 | 0.172461 | 0.210805 | 0.361148 | 106.97 [106.24, 107.74] | 111.54 [110.81, 112.24] | 2110584 | 94439590 | 95144876 | 95850167 | 1659459333504 | - | 263822 | - | 13654 | 6.000 | False |
| **FA_WARM_FROZEN** | mean(2) |  |  |  | 106.98 [106.28, 107.67] | 111.68 [111.02, 112.33] |  |  |  |  |  |  |  |  |  |  | any=False |
| DP8_FROZEN | 2026 | 0.172739 | 0.211013 | 0.360941 | 107.16 [106.39, 107.94] | 111.99 [111.25, 112.71] | 2110584 | 84010589 | 84010589 | 84010589 | 829719113832 | 8.0000 | 263822 | - | 384 | 6.006 | False |
| DP8_FROZEN | 2027 | 0.172702 | 0.210935 | 0.360764 | 107.12 [106.42, 107.88] | 111.70 [110.97, 112.40] | 2110584 | 84095190 | 84095190 | 84095190 | 831076219344 | 8.0000 | 263822 | - | 384 | 6.006 | False |
| **DP8_FROZEN** | mean(2) |  |  |  | 107.14 [106.49, 107.82] | 111.84 [111.16, 112.49] |  |  |  |  |  |  |  |  |  |  | any=False |
| FP_WARM_FROZEN | 2026 | 0.172474 | 0.210779 | 0.360860 | 107.00 [106.22, 107.79] | 111.81 [111.08, 112.55] | 2110584 | 94449988 | 95150387 | 95850791 | 1660740457992 | - | 263822 | - | 13654 | 6.006 | False |
| FP_WARM_FROZEN | 2027 | 0.172468 | 0.210809 | 0.361142 | 106.98 [106.24, 107.74] | 111.55 [110.81, 112.25] | 2110584 | 94439590 | 95144876 | 95850167 | 1659459333504 | - | 263822 | - | 13654 | 6.000 | False |
| **FP_WARM_FROZEN** | mean(2) |  |  |  | 106.99 [106.29, 107.67] | 111.68 [111.03, 112.33] |  |  |  |  |  |  |  |  |  |  | any=False |
| PF_WARM_FROZEN | 2026 | 0.172379 | 0.210710 | 0.361054 | 106.94 [106.16, 107.74] | 111.75 [111.03, 112.50] | 2110584 | 94449988 | 95150387 | 95850791 | 1660740457992 | - | 263822 | - | 13654 | 6.006 | False |
| PF_WARM_FROZEN | 2027 | 0.172487 | 0.210853 | 0.361262 | 106.99 [106.26, 107.76] | 111.56 [110.83, 112.27] | 2110584 | 94439590 | 95144876 | 95850167 | 1659459333504 | - | 263822 | - | 13654 | 6.000 | False |
| **PF_WARM_FROZEN** | mean(2) |  |  |  | 106.96 [106.29, 107.65] | 111.66 [111.00, 112.32] |  |  |  |  |  |  |  |  |  |  | any=False |
| FA_Q8_WARM_FROZEN | 2026 | 0.172385 | 0.210703 | 0.361008 | 106.94 [106.16, 107.75] | 111.76 [111.03, 112.49] | 1319306 | 89699718 | 90400117 | 90575347 | 1038113077078 | - | 263822 | - | 13654 | 6.006 | False |
| FA_Q8_WARM_FROZEN | 2027 | 0.172500 | 0.210840 | 0.361173 | 107.00 [106.27, 107.76] | 111.57 [110.83, 112.27] | 1319306 | 89692985 | 90398270 | 90574723 | 1037312258336 | - | 263822 | - | 13654 | 6.000 | False |
| **FA_Q8_WARM_FROZEN** | mean(2) |  |  |  | 106.97 [106.29, 107.66] | 111.66 [111.02, 112.33] |  |  |  |  |  |  |  |  |  |  | any=False |
| FA_1024_WARM_FROZEN | 2026 | 0.172743 | 0.211027 | 0.360969 | 107.16 [106.39, 107.95] | 111.99 [111.25, 112.71] | 2110584 | 84010589 | 84010589 | 84010589 | 829719113832 | inf | 263822 | - | 384 | 6.006 | False |
| FA_1024_WARM_FROZEN | 2027 | 0.172713 | 0.210968 | 0.360835 | 107.13 [106.41, 107.89] | 111.71 [110.98, 112.42] | 2110584 | 84095190 | 84095190 | 84095190 | 831076219344 | inf | 263822 | - | 384 | 6.006 | False |
| **FA_1024_WARM_FROZEN** | mean(2) |  |  |  | 107.15 [106.49, 107.82] | 111.85 [111.18, 112.49] |  |  |  |  |  |  |  |  |  |  | any=False |
| DP1_WARM_FROZEN | 2026 | 0.172734 | 0.211005 | 0.360931 | 107.16 [106.39, 107.93] | 111.98 [111.25, 112.71] | 2110584 | 84010589 | 84010589 | 84010589 | 829719113832 | 1.0000 | 263822 | - | 384 | 6.006 | False |
| **DP1_WARM_FROZEN** | mean(1) |  |  |  | 107.16 [106.39, 107.93] | 111.98 [111.25, 112.71] |  |  |  |  |  |  |  |  |  |  | any=False |
| PRE | 2026 | 0.172896 | 0.211115 | 0.360895 | 107.26 [106.49, 108.06] | 112.09 [111.36, 112.81] | - | - | - | - | - | - | - | - | - | - | - |
| **PRE** | mean(1) |  |  |  | 107.26 [106.49, 108.06] | 112.09 [111.36, 112.81] |  |  |  |  |  |  |  |  |  |  |  |
| FT_C | 2026 | 0.157966 | 0.192913 | 0.329722 | 98.00 [97.41, 98.60] | 102.41 [102.05, 102.77] | - | - | - | - | - | - | - | - | - | - | - |
| FT_C | 2027 | 0.158655 | 0.193339 | 0.329254 | 98.41 [97.80, 99.02] | 102.61 [102.26, 102.96] | - | - | - | - | - | - | - | - | - | - | - |
| **FT_C** | mean(2) |  |  |  | 98.20 [97.72, 98.68] | 102.51 [102.21, 102.83] |  |  |  |  |  |  |  |  |  |  |  |
| FT_PRE | 2026 | 0.176993 | 0.215714 | 0.366910 | 109.80 [109.09, 110.61] | 114.74 [113.94, 115.53] | - | - | - | - | - | - | - | - | - | - | - |
| FT_PRE | 2027 | 0.176752 | 0.215414 | 0.366503 | 109.63 [108.90, 110.43] | 114.32 [113.55, 115.10] | - | - | - | - | - | - | - | - | - | - | - |
| **FT_PRE** | mean(2) |  |  |  | 109.72 [109.10, 110.40] | 114.53 [113.81, 115.25] |  |  |  |  |  |  |  |  |  |  |  |
| FT_FA | 2026 | 0.145469 | 0.178047 | 0.306413 | 90.24 [89.54, 90.95] | 94.31 [93.64, 94.99] | - | - | - | - | - | - | - | - | - | - | - |
| FT_FA | 2027 | 0.145064 | 0.177522 | 0.305738 | 89.98 [89.28, 90.69] | 93.82 [93.16, 94.50] | - | - | - | - | - | - | - | - | - | - | - |
| **FT_FA** | mean(2) |  |  |  | 90.11 [89.58, 90.67] | 94.07 [93.56, 94.58] |  |  |  |  |  |  |  |  |  |  |  |
| FT_FA_WARM | 2026 | 0.175829 | 0.214632 | 0.366308 | 109.08 [108.33, 109.88] | 113.99 [113.20, 114.80] | - | - | - | - | - | - | - | - | - | - | - |
| FT_FA_WARM | 2027 | 0.175763 | 0.214498 | 0.365918 | 109.02 [108.27, 109.80] | 113.68 [112.89, 114.51] | - | - | - | - | - | - | - | - | - | - | - |
| **FT_FA_WARM** | mean(2) |  |  |  | 109.05 [108.39, 109.75] | 113.83 [113.14, 114.56] |  |  |  |  |  |  |  |  |  |  |  |
| FT_FA_WARM_FROZEN | 2026 | 0.176369 | 0.215395 | 0.367572 | 109.41 [108.70, 110.24] | 114.34 [113.54, 115.17] | - | - | - | - | - | - | - | - | - | - | - |
| FT_FA_WARM_FROZEN | 2027 | 0.176623 | 0.215548 | 0.367557 | 109.55 [108.84, 110.35] | 114.24 [113.44, 115.06] | - | - | - | - | - | - | - | - | - | - | - |
| **FT_FA_WARM_FROZEN** | mean(2) |  |  |  | 109.48 [108.83, 110.21] | 114.29 [113.54, 115.03] |  |  |  |  |  |  |  |  |  |  |  |
| LO | 2026 | 0.004717 | 0.005488 | 0.008392 | 2.93 [2.72, 3.15] | 3.06 [2.84, 3.29] | - | - | - | - | - | - | - | - | - | - | - |
| LO | 2027 | 0.005288 | 0.006147 | 0.009420 | 3.28 [3.05, 3.51] | 3.42 [3.18, 3.66] | - | - | - | - | - | - | - | - | - | - | - |
| **LO** | mean(2) |  |  |  | 3.10 [2.91, 3.30] | 3.24 [3.03, 3.45] |  |  |  |  |  |  |  |  |  |  |  |
| C_SMALL_FP32 | 2026 | 0.152698 | 0.187015 | 0.321586 | 95.69 [94.74, 96.69] | 100.00 [100.00, 100.00] | - | - | - | - | - | - | - | - | - | - | - |
| **C_SMALL_FP32** | mean(1) |  |  |  | 95.69 [94.74, 96.69] DEVICE_MIXED | 100.00 [100.00, 100.00] |  |  |  |  |  |  |  |  |  |  |  |
| C_SMALL_INT8 | 2026 | 0.152452 | 0.186706 | 0.321090 | 95.54 [94.55, 96.53] | 99.84 [99.62, 100.07] | - | - | - | - | - | - | - | - | - | - | - |
| **C_SMALL_INT8** | mean(1) |  |  |  | 95.54 [94.55, 96.53] DEVICE_MIXED | 99.84 [99.62, 100.07] |  |  |  |  |  |  |  |  |  |  |  |
| KNN_EMB_SMALL | None | 0.072936 | 0.088720 | 0.151832 | 45.25 [44.57, 45.99]; 45.24 [44.52, 45.98] | 47.28 [46.52, 48.09]; 47.17 [46.42, 47.97] | - | - | - | - | - | - | - | - | - | - | - |
| **KNN_EMB_SMALL** | mean(2) |  |  |  | 45.24 [44.57, 45.95] | 47.23 [46.47, 47.99] |  |  |  |  |  |  |  |  |  |  |  |
| KNN_EMB_BIG | None | 0.081965 | 0.099176 | 0.165709 | 50.85 [50.12, 51.63]; 50.84 [50.08, 51.60] | 53.14 [52.32, 53.99]; 53.01 [52.22, 53.85] | - | - | - | - | - | - | - | - | - | - | - |
| **KNN_EMB_BIG** | mean(2) |  |  |  | 50.84 [50.12, 51.58] | 53.08 [52.26, 53.89] |  |  |  |  |  |  |  |  |  |  |  |
| RULE | None | 0.037691 | 0.047984 | 0.093164 | 23.38 [22.87, 23.92]; 23.38 [22.90, 23.91] | 24.43 [23.89, 24.98]; 24.38 [23.84, 24.93] | - | - | - | - | - | - | - | - | - | - | - |
| **RULE** | mean(2) |  |  |  | 23.38 [22.90, 23.91] | 24.41 [23.88, 24.96] |  |  |  |  |  |  |  |  |  |  |  |
| SKNN | None | 0.111502 | 0.148717 | 0.290382 | 69.17 [68.59, 69.81]; 69.16 [68.55, 69.76] | 72.29 [71.61, 72.93]; 72.12 [71.47, 72.77] | - | - | - | - | - | - | - | - | - | - | - |
| **SKNN** | mean(2) |  |  |  | 69.17 [68.60, 69.75] | 72.20 [71.58, 72.81] |  |  |  |  |  |  |  |  |  |  |  |
| LAST | None | 0.071112 | 0.101637 | 0.205635 | 44.12 [43.56, 44.68]; 44.11 [43.55, 44.68] | 46.10 [45.46, 46.76]; 45.99 [45.34, 46.61] | - | - | - | - | - | - | - | - | - | - | - |
| **LAST** | mean(2) |  |  |  | 44.11 [43.57, 44.69] | 46.05 [45.41, 46.66] |  |  |  |  |  |  |  |  |  |  |  |
| POP | None | 0.019430 | 0.024905 | 0.049145 | 12.05 [11.66, 12.50]; 12.05 [11.67, 12.49] | 12.60 [12.18, 13.07]; 12.57 [12.16, 13.05] | - | - | - | - | - | - | - | - | - | - | - |
| **POP** | mean(2) |  |  |  | 12.05 [11.67, 12.50] | 12.58 [12.17, 13.06] |  |  |  |  |  |  |  |  |  |  |  |

| arm | R% NDCG@10 vs C_BIG | R% HR@10 vs C_BIG |
|---|---|---|
| C_BIG | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| C_FULL | 119.62 [119.04, 120.18] | 117.03 [116.52, 117.52] |
| C_SMALL | 96.07 [95.71, 96.46] | 97.06 [96.71, 97.45] |
| FA | 87.03 [86.58, 87.51] | 89.49 [89.07, 89.93] |
| FP | 87.02 [86.58, 87.51] | 89.48 [89.05, 89.93] |
| PF | 87.05 [86.60, 87.53] | 89.51 [89.09, 89.95] |
| FA_1024 | 22.27 [21.71, 22.77] | 24.19 [23.68, 24.67] |
| DP8 | 22.03 [21.49, 22.55] | 23.92 [23.41, 24.45] |
| DP1 | 21.65 [21.12, 22.15] | 23.89 [23.34, 24.38] |
| FA_Q8 | 86.82 [86.36, 87.30] | 89.26 [88.83, 89.71] |
| FA_WARM | 106.69 [106.13, 107.25] | 107.63 [107.12, 108.13] |
| FA_WARM_FROZEN | 107.17 [106.63, 107.78] | 107.97 [107.46, 108.51] |
| DP8_FROZEN | 107.27 [106.75, 107.85] | 107.93 [107.41, 108.47] |
| FP_WARM_FROZEN | 107.18 [106.63, 107.78] | 107.97 [107.46, 108.51] |
| PF_WARM_FROZEN | 107.17 [106.61, 107.77] | 108.02 [107.51, 108.56] |
| FA_Q8_WARM_FROZEN | 107.17 [106.61, 107.76] | 108.00 [107.49, 108.55] |
| FA_1024_WARM_FROZEN | 107.28 [106.75, 107.86] | 107.94 [107.42, 108.49] |
| DP1_WARM_FROZEN | 107.30 [106.70, 107.94] | 107.96 [107.37, 108.58] |
| PRE | 107.35 [106.76, 108.02] | 107.95 [107.37, 108.56] |
| FT_C | 98.19 [97.82, 98.58] | 98.55 [98.19, 98.92] |
| FT_PRE | 109.60 [109.07, 110.19] | 109.68 [109.14, 110.21] |
| FT_FA | 90.39 [89.93, 90.89] | 91.54 [91.14, 92.02] |
| FT_FA_WARM | 109.09 [108.56, 109.67] | 109.50 [109.00, 110.06] |
| FT_FA_WARM_FROZEN | 109.56 [109.00, 110.14] | 109.93 [109.42, 110.49] |
| LO | 2.96 [2.78, 3.14] | 2.66 [2.52, 2.82] |
| C_SMALL_FP32 | 96.02 [95.27, 96.81] | 96.89 [96.12, 97.60] |
| C_SMALL_INT8 | 95.87 [95.11, 96.66] | 96.74 [95.94, 97.49] |
| KNN_EMB_SMALL | 45.11 [44.47, 45.76] | 45.41 [44.83, 45.99] |
| KNN_EMB_BIG | 50.43 [49.79, 51.08] | 49.56 [48.97, 50.13] |
| RULE | 24.40 [23.91, 24.91] | 27.86 [27.34, 28.40] |
| SKNN | 75.61 [75.11, 76.20] | 86.85 [86.29, 87.42] |
| LAST | 51.68 [51.05, 52.32] | 61.50 [60.75, 62.20] |
| POP | 12.66 [12.26, 13.09] | 14.70 [14.26, 15.16] |

### CONTROLLED_BUDGET

| arm | seed | MRR@20 | NDCG@10 | HR@10 | R% vs C_BIG [CI] | R% vs C_SMALL [CI] | bytes/visit | bytes/client | bytes/client (dropped visits charged, 06I 7b) | bytes/client (dropped visits charged down + up; DESCRIPTIVE) | bytes fleet | ε | params | model bytes | rounds | consumed EFE | still improving at 6.0 (PB mark >= 5.5; DESCRIPTIVE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_BIG | 2026 | 0.152870 | 0.186671 | 0.317830 | 100.00 [100.00, 100.00] | 100.00 [99.26, 100.76] | - | - | - | - | - | - | - | - | - | - | - |
| C_BIG | 2027 | 0.152579 | 0.186614 | 0.318397 | 100.00 [100.00, 100.00] | 99.99 [99.24, 100.69] | - | - | - | - | - | - | - | - | - | - | - |
| **C_BIG** | mean(2) |  |  |  | 100.00 [100.00, 100.00] | 99.99 [99.43, 100.52] |  |  |  |  |  |  |  |  |  |  |  |
| C_FULL | 2026 | 0.194192 | 0.235151 | 0.391644 | 127.03 [126.16, 128.03] | 127.03 [126.13, 127.98] | - | - | - | - | - | - | - | - | - | - | - |
| C_FULL | 2027 | 0.194716 | 0.235384 | 0.390927 | 127.62 [126.72, 128.56] | 127.60 [126.74, 128.59] | - | - | - | - | - | - | - | - | - | - | - |
| **C_FULL** | mean(2) |  |  |  | 127.32 [126.60, 128.14] | 127.32 [126.54, 128.09] |  |  |  |  |  |  |  |  |  |  |  |
| C_SMALL | 2026 | 0.152870 | 0.187038 | 0.320703 | 100.00 [99.24, 100.74] | 100.00 [100.00, 100.00] | - | - | - | - | - | - | - | - | - | - | - |
| C_SMALL | 2027 | 0.152597 | 0.186592 | 0.319688 | 100.01 [99.32, 100.76] | 100.00 [100.00, 100.00] | - | - | - | - | - | - | - | - | - | - | - |
| **C_SMALL** | mean(2) |  |  |  | 100.01 [99.48, 100.58] | 100.00 [100.00, 100.00] |  |  |  |  |  |  |  |  |  |  |  |
| FA | 2026 | 0.139037 | 0.171410 | 0.299358 | 90.95 [90.18, 91.77] | 90.95 [90.25, 91.64] | - | - | - | - | - | - | - | - | - | - | True |
| FA | 2027 | 0.138480 | 0.170913 | 0.299084 | 90.76 [89.98, 91.50] | 90.75 [90.09, 91.42] | - | - | - | - | - | - | - | - | - | - | True |
| **FA** | mean(2) |  |  |  | 90.86 [90.27, 91.52] | 90.85 [90.30, 91.36] |  |  |  |  |  |  |  |  |  |  | any=True |
| FP | 2026 | 0.139033 | 0.171404 | 0.299354 | 90.95 [90.18, 91.77] | 90.95 [90.24, 91.64] | - | - | - | - | - | - | - | - | - | - | True |
| FP | 2027 | 0.138483 | 0.170894 | 0.299011 | 90.76 [89.98, 91.50] | 90.75 [90.10, 91.42] | - | - | - | - | - | - | - | - | - | - | True |
| **FP** | mean(2) |  |  |  | 90.86 [90.27, 91.52] | 90.85 [90.30, 91.37] |  |  |  |  |  |  |  |  |  |  | any=True |
| PF | 2026 | 0.139060 | 0.171429 | 0.299378 | 90.97 [90.19, 91.79] | 90.97 [90.27, 91.65] | - | - | - | - | - | - | - | - | - | - | True |
| PF | 2027 | 0.138517 | 0.170970 | 0.299179 | 90.78 [90.02, 91.56] | 90.77 [90.12, 91.45] | - | - | - | - | - | - | - | - | - | - | True |
| **PF** | mean(2) |  |  |  | 90.88 [90.30, 91.53] | 90.87 [90.33, 91.37] |  |  |  |  |  |  |  |  |  |  | any=True |
| FA_1024 | 2026 | 0.034929 | 0.044257 | 0.082935 | 22.85 [22.25, 23.43] | 22.85 [22.26, 23.45] | - | - | - | - | - | inf | - | - | - | - | True |
| FA_1024 | 2027 | 0.034986 | 0.043340 | 0.078805 | 22.93 [22.26, 23.58] | 22.93 [22.26, 23.55] | - | - | - | - | - | inf | - | - | - | - | True |
| **FA_1024** | mean(2) |  |  |  | 22.89 [22.27, 23.48] | 22.89 [22.26, 23.49] |  |  |  |  |  |  |  |  |  |  | any=True |
| DP8 | 2026 | 0.035172 | 0.044328 | 0.082357 | 23.01 [22.40, 23.60] | 23.01 [22.40, 23.62] | - | - | - | - | - | 8.0000 | - | - | - | - | True |
| DP8 | 2027 | 0.033837 | 0.042320 | 0.077600 | 22.18 [21.53, 22.79] | 22.17 [21.57, 22.78] | - | - | - | - | - | 8.0000 | - | - | - | - | True |
| **DP8** | mean(2) |  |  |  | 22.59 [21.99, 23.17] | 22.59 [22.00, 23.17] |  |  |  |  |  |  |  |  |  |  | any=True |
| DP1 | 2026 | 0.033625 | 0.042577 | 0.079857 | 22.00 [21.36, 22.56] | 22.00 [21.39, 22.57] | - | - | - | - | - | 1.0000 | - | - | - | - | True |
| **DP1** | mean(1) |  |  |  | 22.00 [21.36, 22.56] | 22.00 [21.39, 22.57] |  |  |  |  |  |  |  |  |  |  | any=True |
| FA_Q8 | 2026 | 0.138646 | 0.170955 | 0.298784 | 90.70 [89.93, 91.54] | 90.70 [90.05, 91.38] | - | - | - | - | - | - | - | - | - | - | True |
| FA_Q8 | 2027 | 0.138370 | 0.170552 | 0.298104 | 90.69 [89.89, 91.45] | 90.68 [90.01, 91.32] | - | - | - | - | - | - | - | - | - | - | True |
| **FA_Q8** | mean(2) |  |  |  | 90.69 [90.10, 91.35] | 90.69 [90.13, 91.20] |  |  |  |  |  |  |  |  |  |  | any=True |
| FA_WARM | 2026 | 0.170716 | 0.209024 | 0.359289 | 111.67 [110.81, 112.55] | 111.67 [110.96, 112.48] | - | - | - | - | - | - | - | - | - | - | False |
| FA_WARM | 2027 | 0.170544 | 0.208917 | 0.359407 | 111.77 [110.95, 112.63] | 111.76 [111.00, 112.51] | - | - | - | - | - | - | - | - | - | - | False |
| **FA_WARM** | mean(2) |  |  |  | 111.72 [111.01, 112.49] | 111.72 [111.04, 112.38] |  |  |  |  |  |  |  |  |  |  | any=False |
| FA_WARM_FROZEN | 2026 | 0.172499 | 0.210817 | 0.361182 | 112.84 [111.98, 113.73] | 112.84 [112.10, 113.59] | - | - | - | - | - | - | - | - | - | - | False |
| FA_WARM_FROZEN | 2027 | 0.172413 | 0.210730 | 0.361013 | 113.00 [112.19, 113.92] | 112.99 [112.21, 113.76] | - | - | - | - | - | - | - | - | - | - | False |
| **FA_WARM_FROZEN** | mean(2) |  |  |  | 112.92 [112.16, 113.73] | 112.91 [112.24, 113.59] |  |  |  |  |  |  |  |  |  |  | any=False |
| DP8_FROZEN | 2026 | 0.172028 | 0.210315 | 0.360354 | 112.53 [111.67, 113.41] | 112.53 [111.77, 113.28] | - | - | - | - | - | 8.0000 | - | - | - | - | False |
| DP8_FROZEN | 2027 | 0.171934 | 0.210135 | 0.359956 | 112.69 [111.86, 113.57] | 112.67 [111.92, 113.41] | - | - | - | - | - | 8.0000 | - | - | - | - | False |
| **DP8_FROZEN** | mean(2) |  |  |  | 112.61 [111.82, 113.39] | 112.60 [111.92, 113.29] |  |  |  |  |  |  |  |  |  |  | any=False |
| FP_WARM_FROZEN | 2026 | 0.172499 | 0.210817 | 0.361184 | 112.84 [111.98, 113.73] | 112.84 [112.10, 113.59] | - | - | - | - | - | - | - | - | - | - | False |
| FP_WARM_FROZEN | 2027 | 0.172409 | 0.210727 | 0.361016 | 113.00 [112.19, 113.92] | 112.98 [112.21, 113.75] | - | - | - | - | - | - | - | - | - | - | False |
| **FP_WARM_FROZEN** | mean(2) |  |  |  | 112.92 [112.16, 113.73] | 112.91 [112.24, 113.59] |  |  |  |  |  |  |  |  |  |  | any=False |
| PF_WARM_FROZEN | 2026 | 0.172464 | 0.210773 | 0.361122 | 112.82 [111.96, 113.70] | 112.82 [112.08, 113.58] | - | - | - | - | - | - | - | - | - | - | False |
| PF_WARM_FROZEN | 2027 | 0.172470 | 0.210754 | 0.360960 | 113.04 [112.23, 113.96] | 113.02 [112.25, 113.80] | - | - | - | - | - | - | - | - | - | - | False |
| **PF_WARM_FROZEN** | mean(2) |  |  |  | 112.93 [112.15, 113.74] | 112.92 [112.24, 113.60] |  |  |  |  |  |  |  |  |  |  | any=False |
| FA_Q8_WARM_FROZEN | 2026 | 0.172472 | 0.210801 | 0.361197 | 112.82 [111.97, 113.72] | 112.82 [112.09, 113.58] | - | - | - | - | - | - | - | - | - | - | False |
| FA_Q8_WARM_FROZEN | 2027 | 0.172416 | 0.210752 | 0.361087 | 113.00 [112.19, 113.92] | 112.99 [112.22, 113.75] | - | - | - | - | - | - | - | - | - | - | False |
| **FA_Q8_WARM_FROZEN** | mean(2) |  |  |  | 112.91 [112.15, 113.72] | 112.91 [112.23, 113.58] |  |  |  |  |  |  |  |  |  |  | any=False |
| FA_1024_WARM_FROZEN | 2026 | 0.172074 | 0.210331 | 0.360281 | 112.56 [111.67, 113.43] | 112.56 [111.77, 113.32] | - | - | - | - | - | inf | - | - | - | - | False |
| FA_1024_WARM_FROZEN | 2027 | 0.172014 | 0.210226 | 0.360041 | 112.74 [111.91, 113.63] | 112.72 [111.97, 113.45] | - | - | - | - | - | inf | - | - | - | - | False |
| **FA_1024_WARM_FROZEN** | mean(2) |  |  |  | 112.65 [111.87, 113.45] | 112.64 [111.96, 113.34] |  |  |  |  |  |  |  |  |  |  | any=False |
| DP1_WARM_FROZEN | 2026 | 0.171871 | 0.210162 | 0.360177 | 112.43 [111.57, 113.29] | 112.43 [111.67, 113.20] | - | - | - | - | - | 1.0000 | - | - | - | - | False |
| **DP1_WARM_FROZEN** | mean(1) |  |  |  | 112.43 [111.57, 113.29] | 112.43 [111.67, 113.20] |  |  |  |  |  |  |  |  |  |  | any=False |
| PRE | 2026 | 0.172896 | 0.211115 | 0.360895 | 113.10 [112.24, 114.00] | 113.10 [112.34, 113.87] | - | - | - | - | - | - | - | - | - | - | - |
| **PRE** | mean(1) |  |  |  | 113.10 [112.24, 114.00] | 113.10 [112.34, 113.87] |  |  |  |  |  |  |  |  |  |  |  |
| KNN_EMB_SMALL | None | 0.072936 | 0.088720 | 0.151832 | 47.71 [47.00, 48.50]; 47.80 [47.08, 48.55] | 47.71 [46.95, 48.51]; 47.80 [47.03, 48.62] | - | - | - | - | - | - | - | - | - | - | - |
| **KNN_EMB_SMALL** | mean(2) |  |  |  | 47.76 [47.05, 48.50] | 47.75 [47.00, 48.55] |  |  |  |  |  |  |  |  |  |  |  |
| KNN_EMB_BIG | None | 0.081965 | 0.099176 | 0.165709 | 53.62 [52.84, 54.44]; 53.72 [52.92, 54.51] | 53.62 [52.78, 54.47]; 53.71 [52.89, 54.58] | - | - | - | - | - | - | - | - | - | - | - |
| **KNN_EMB_BIG** | mean(2) |  |  |  | 53.67 [52.89, 54.46] | 53.67 [52.85, 54.49] |  |  |  |  |  |  |  |  |  |  |  |
| RULE | None | 0.037691 | 0.047984 | 0.093164 | 24.66 [24.13, 25.24]; 24.70 [24.18, 25.28] | 24.66 [24.11, 25.24]; 24.70 [24.16, 25.27] | - | - | - | - | - | - | - | - | - | - | - |
| **RULE** | mean(2) |  |  |  | 24.68 [24.16, 25.25] | 24.68 [24.15, 25.24] |  |  |  |  |  |  |  |  |  |  |  |
| SKNN | None | 0.111502 | 0.148717 | 0.290382 | 72.94 [72.25, 73.64]; 73.08 [72.39, 73.76] | 72.94 [72.30, 73.61]; 73.07 [72.44, 73.77] | - | - | - | - | - | - | - | - | - | - | - |
| **SKNN** | mean(2) |  |  |  | 73.01 [72.38, 73.66] | 73.00 [72.40, 73.64] |  |  |  |  |  |  |  |  |  |  |  |
| LAST | None | 0.071112 | 0.101637 | 0.205635 | 46.52 [45.95, 47.16]; 46.61 [46.00, 47.23] | 46.52 [45.90, 47.19]; 46.60 [45.96, 47.24] | - | - | - | - | - | - | - | - | - | - | - |
| **LAST** | mean(2) |  |  |  | 46.56 [45.98, 47.16] | 46.56 [45.95, 47.21] |  |  |  |  |  |  |  |  |  |  |  |
| POP | None | 0.019430 | 0.024905 | 0.049145 | 12.71 [12.30, 13.18]; 12.73 [12.33, 13.20] | 12.71 [12.30, 13.20]; 12.73 [12.31, 13.22] | - | - | - | - | - | - | - | - | - | - | - |
| **POP** | mean(2) |  |  |  | 12.72 [12.31, 13.19] | 12.72 [12.30, 13.20] |  |  |  |  |  |  |  |  |  |  |  |

| arm | R% NDCG@10 vs C_BIG | R% HR@10 vs C_BIG |
|---|---|---|
| C_BIG | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| C_FULL | 126.05 [125.43, 126.71] | 123.00 [122.41, 123.59] |
| C_SMALL | 100.09 [99.64, 100.55] | 100.65 [100.24, 101.09] |
| FA | 91.71 [91.18, 92.25] | 94.06 [93.61, 94.57] |
| FP | 91.70 [91.17, 92.25] | 94.05 [93.60, 94.55] |
| PF | 91.73 [91.19, 92.28] | 94.08 [93.63, 94.58] |
| FA_1024 | 23.47 [22.88, 24.00] | 25.42 [24.90, 25.93] |
| DP8 | 23.21 [22.66, 23.75] | 25.14 [24.60, 25.67] |
| DP1 | 22.81 [22.23, 23.34] | 25.13 [24.54, 25.65] |
| FA_Q8 | 91.49 [90.96, 92.03] | 93.82 [93.38, 94.32] |
| FA_WARM | 111.96 [111.35, 112.61] | 112.96 [112.36, 113.55] |
| FA_WARM_FROZEN | 112.93 [112.28, 113.62] | 113.51 [112.92, 114.14] |
| DP8_FROZEN | 112.64 [111.97, 113.29] | 113.22 [112.61, 113.86] |
| FP_WARM_FROZEN | 112.93 [112.28, 113.62] | 113.51 [112.92, 114.14] |
| PF_WARM_FROZEN | 112.92 [112.27, 113.61] | 113.49 [112.89, 114.12] |
| FA_Q8_WARM_FROZEN | 112.93 [112.29, 113.62] | 113.53 [112.93, 114.16] |
| FA_1024_WARM_FROZEN | 112.66 [112.01, 113.33] | 113.22 [112.60, 113.87] |
| DP1_WARM_FROZEN | 112.58 [111.89, 113.33] | 113.32 [112.65, 114.02] |
| PRE | 113.09 [112.40, 113.86] | 113.55 [112.87, 114.29] |
| KNN_EMB_SMALL | 47.53 [46.89, 48.24] | 47.73 [47.11, 48.33] |
| KNN_EMB_BIG | 53.14 [52.46, 53.86] | 52.09 [51.46, 52.72] |
| RULE | 25.71 [25.20, 26.25] | 29.29 [28.73, 29.87] |
| SKNN | 79.68 [79.09, 80.29] | 91.28 [90.64, 91.91] |
| LAST | 54.46 [53.81, 55.13] | 64.64 [63.88, 65.39] |
| POP | 13.34 [12.93, 13.80] | 15.45 [14.99, 15.93] |

## 3. Personalisation gains (paired) — DESCRIPTIVE

| contrast | rule | seeds | delta MRR@20 | 95% CI | CI vs 0 | harmed-client share |
|---|---|---|---|---|---|---|
| FT_C - C_SMALL | PRACTICAL_BEST | [2026, 2027] | 0.003879 | [0.003436, 0.004340] | above | 0.3602 |
| FT_C - C_SMALL | CONTROLLED_BUDGET | NOT_AVAILABLE | - | - | - | - |
| FT_FA - FA | PRACTICAL_BEST | [2026, 2027] | 0.006508 | [0.006044, 0.006971] | above | 0.3420 |
| FT_FA - FA | CONTROLLED_BUDGET | NOT_AVAILABLE | - | - | - | - |
| PF - FA | PRACTICAL_BEST | [2026, 2027] | 0.000030 | [-0.000012, 0.000074] | includes | 0.0989 |
| PF - FA | CONTROLLED_BUDGET | [2026, 2027] | 0.000030 | [-0.000012, 0.000074] | includes | 0.0989 |

## 4. Privacy-level ladder (06B §3) — DESCRIPTIVE

| level | arm | R% vs C_BIG (PB, seed mean) | bytes/client | leaves device (training) | leaves device (inference) | guarantee |
|---|---|---|---|---|---|---|
| L0 cloud-train / cloud-infer | C_BIG | 100.00 [100.00, 100.00] | - | raw events | query context | none |
| L0 cloud-train / cloud-infer | C_FULL | 120.62 [119.97, 121.32] | - | raw events | query context | none |
| L0 cloud-train / cloud-infer | C_SMALL | 95.80 [95.33, 96.26] | - | raw events | query context | none |
| L1 cloud-train / device-infer | C_SMALL_FP32 | 95.69 [94.74, 96.69] | - | raw events (model trained centrally once) | nothing | none |
| L1 cloud-train / device-infer | C_SMALL_INT8 | 95.54 [94.55, 96.53] | - | raw events (model trained centrally once) | nothing | none |
| L1 cloud-train / device-infer | KNN_EMB_SMALL | 45.24 [44.57, 45.95] | - | raw events (model trained centrally once) | nothing | none |
| L1 cloud-train / device-infer | KNN_EMB_BIG | 50.84 [50.12, 51.58] | - | raw events (model trained centrally once) | nothing | none |
| L2 federated | FA | 86.07 [85.56, 86.62] | 994562798 | model deltas (dense FP32; 8-bit for FA_Q8) | nothing | none (server sees clipped/unclipped deltas) |
| L2 federated | FP | 86.07 [85.56, 86.62] | 994562798 | model deltas (dense FP32; 8-bit for FA_Q8) | nothing | none (server sees clipped/unclipped deltas) |
| L2 federated | FA_1024 | 21.68 [21.09, 22.24] | 496891468 | model deltas (dense FP32; 8-bit for FA_Q8) | nothing | none (server sees clipped/unclipped deltas) |
| L2 federated | FA_Q8 | 85.92 [85.38, 86.48] | 621602968 | model deltas (dense FP32; 8-bit for FA_Q8) | nothing | none (server sees clipped/unclipped deltas) |
| L2 federated | PF | 86.09 [85.57, 86.63] | 994562798 | model deltas (dense FP32; 8-bit for FA_Q8) | nothing | none (server sees clipped/unclipped deltas) |
| L3 federated + DP | DP8 | 21.40 [20.83, 21.96] | 496891468 | clipped deltas + Gaussian noise on the sum | nothing | user-level (ε = 8.0, δ = 5e-6)-DP of the released model |
| L3 federated + DP | DP1 | 20.86 [20.28, 21.40] | 496891468 | clipped deltas + Gaussian noise on the sum | nothing | user-level (ε = 1.0, δ = 5e-6)-DP of the released model |
| L4 federated + local personalisation | PF | 86.09 [85.57, 86.63] | 994562798 | as L2; personal parameters never leave | nothing | none beyond L2 |
| L4 federated + local personalisation | FT_FA | 90.11 [89.58, 90.67] | - | as L2; personal parameters never leave | nothing | none beyond L2 |
| L5 cloud model + local personalisation | FT_C | 98.20 [97.72, 98.68] | - | nothing after the download | nothing | none (download only) |
| L6 device-only | LO | 3.10 [2.91, 3.30] | - | nothing | nothing | data never leaves (shared catalogue/θ0 only) |
| L6 device-only | RULE | 23.38 [22.90, 23.91] | - | nothing | nothing | data never leaves (shared catalogue/θ0 only) |

Disclosed (CLIENT_CONTRACT C5.6): every arm shares the pooled bootstrap block (catalogue, transform statistics, θ0).

## 5. Curves — DESCRIPTIVE (PRACTICAL_BEST; full series in curves/*.csv)

### 5a. R vs ε

| arm | ε (registered) | z | rule | R% vs C_BIG | R% vs FA_1024 | ε fields (DP_WOR receipt) |
|---|---|---|---|---|---|---|
| FA_1024 | inf | 0.0000 | PRACTICAL_BEST | 21.68 [21.09, 22.24] | - - | "-" |
| FA_1024 | inf | 0.0000 | CONTROLLED_BUDGET | 22.89 [22.27, 23.48] | - - | "-" |
| DP8 | 8.0000 | 0.5642 | PRACTICAL_BEST | 21.40 [20.83, 21.96] | 98.71 [98.09, 99.40] | {"eps_poisson_on_integer_orders_2_256_info": 7.999847800151169, "eps_rdp_floor_prop11_c1_info": 7.999847800151169, "eps_rdp_floor_prop11_c2_info": 1132.4998892194433, "eps_registered_poisson_add_remove": 7.998245751758989, "eps_registered_reference_from_packet": 7.99825, "eps_wor_replace_one_c1_packet_literal": 9.959133390202531, "eps_wor_replace_one_c2_mechanism_faithful": 1388.1448872449914} |
| DP8 | 8.0000 | 0.5642 | CONTROLLED_BUDGET | 22.59 [21.99, 23.17] | 98.71 [98.09, 99.40] | {"eps_poisson_on_integer_orders_2_256_info": 7.999847800151169, "eps_rdp_floor_prop11_c1_info": 7.999847800151169, "eps_rdp_floor_prop11_c2_info": 1132.4998892194433, "eps_registered_poisson_add_remove": 7.998245751758989, "eps_registered_reference_from_packet": 7.99825, "eps_wor_replace_one_c1_packet_literal": 9.959133390202531, "eps_wor_replace_one_c2_mechanism_faithful": 1388.1448872449914} |
| DP1 | 1.0000 | 1.3094 | PRACTICAL_BEST | 20.86 [20.28, 21.40] | 96.27 [95.06, 97.42] | {"eps_poisson_on_integer_orders_2_256_info": 0.9998462503769788, "eps_rdp_floor_prop11_c1_info": 0.9998462503769788, "eps_rdp_floor_prop11_c2_info": 5.329469016140623, "eps_registered_poisson_add_remove": 0.9998462503769788, "eps_registered_reference_from_packet": 0.99985, "eps_wor_replace_one_c1_packet_literal": 1.5024015270523532, "eps_wor_replace_one_c2_mechanism_faithful": 6.68772582621574} |
| DP1 | 1.0000 | 1.3094 | CONTROLLED_BUDGET | 22.00 [21.36, 22.56] | 96.27 [95.06, 97.42] | {"eps_poisson_on_integer_orders_2_256_info": 0.9998462503769788, "eps_rdp_floor_prop11_c1_info": 0.9998462503769788, "eps_rdp_floor_prop11_c2_info": 5.329469016140623, "eps_registered_poisson_add_remove": 0.9998462503769788, "eps_registered_reference_from_packet": 0.99985, "eps_wor_replace_one_c1_packet_literal": 1.5024015270523532, "eps_wor_replace_one_c2_mechanism_faithful": 6.68772582621574} |

### 5b. R vs model size

| arm | params | model bytes | R% PB | R% CB | note |
|---|---|---|---|---|---|
| C_FULL | - | - | 120.62 [119.97, 121.32] | 127.32 [126.60, 128.14] |  |
| C_BIG | - | - | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |  |
| C_SMALL | - | - | 95.80 [95.33, 96.26] | 100.01 [99.48, 100.58] |  |
| C_SMALL_FP32 | - | - | 95.69 [94.74, 96.69] | - - | INT8 / ONNX FP32 on its users_subset when given (06A §6) |
| C_SMALL_INT8 | - | - | 95.54 [94.55, 96.53] | - - | INT8 / ONNX FP32 on its users_subset when given (06A §6) |

### 5c. R vs cumulative bytes per client (FL marks)

| arm | seed | mark | round | cum bytes/client (prorated; registered series) | cum bytes/client (dropped visits charged their download; second series) | cum bytes/client (dropped visits charged down + up; DESCRIPTIVE) | MRR@20 | R% | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| FA | 2026 | 0.5 | 1138 | 82892373 | 87474528 | 92056683 | 0.070506 | 43.74 | [43.05, 44.44] |
| FA | 2026 | 1.0 | 2277 | 165857587 | 175025923 | 184194259 | 0.089710 | 55.65 | [54.91, 56.32] |
| FA | 2026 | 1.5 | 3415 | 248749960 | 262500451 | 276250942 | 0.098139 | 60.88 | [60.14, 61.57] |
| FA | 2026 | 2.0 | 4552 | 331569493 | 349898112 | 368226732 | 0.106620 | 66.14 | [65.35, 66.90] |
| FA | 2026 | 2.5 | 5686 | 414170505 | 437065172 | 459959841 | 0.107559 | 66.73 | [65.98, 67.44] |
| FA | 2026 | 3.0 | 6831 | 497572761 | 525077768 | 552582778 | 0.113898 | 70.66 | [69.94, 71.40] |
| FA | 2026 | 3.5 | 7967 | 580319453 | 612398562 | 644477674 | 0.115023 | 71.36 | [70.64, 72.06] |
| FA | 2026 | 4.0 | 9105 | 663211827 | 699873090 | 736534357 | 0.118646 | 73.60 | [72.93, 74.31] |
| FA | 2026 | 4.5 | 10242 | 746031359 | 787270751 | 828510147 | 0.119302 | 74.01 | [73.32, 74.73] |
| FA | 2026 | 5.0 | 11380 | 828923733 | 874745279 | 920566830 | 0.124137 | 77.01 | [76.30, 77.74] |
| FA | 2026 | 5.5 | 12518 | 911816106 | 962219807 | 1012623513 | 0.125921 | 78.12 | [77.42, 78.84] |
| FA | 2026 | 6.0 | 13654 | 994562798 | 1049540601 | 1104518409 | 0.139037 | 86.25 | [85.59, 86.99] |
| FA | 2027 | 0.5 | 1134 | 82537292 | 87135200 | 91733109 | 0.069615 | 43.18 | [42.47, 43.89] |
| FA | 2027 | 1.0 | 2278 | 165802426 | 175038789 | 184275153 | 0.088379 | 54.82 | [54.11, 55.57] |
| FA | 2027 | 1.5 | 3413 | 248412502 | 262250828 | 276089156 | 0.096231 | 59.69 | [58.93, 60.43] |
| FA | 2027 | 2.0 | 4544 | 330731441 | 349155512 | 367579585 | 0.103317 | 64.08 | [63.36, 64.79] |
| FA | 2027 | 2.5 | 5682 | 413559869 | 436598067 | 459636268 | 0.109794 | 68.10 | [67.33, 68.85] |
| FA | 2027 | 3.0 | 6820 | 496388298 | 524040623 | 551692951 | 0.114427 | 70.98 | [70.26, 71.69] |
| FA | 2027 | 3.5 | 7961 | 579435079 | 611713695 | 643992314 | 0.115821 | 71.84 | [71.12, 72.55] |
| FA | 2027 | 4.0 | 9109 | 662991350 | 699924639 | 736857931 | 0.117742 | 73.03 | [72.31, 73.81] |
| FA | 2027 | 4.5 | 10240 | 745310289 | 786829323 | 828348360 | 0.119419 | 74.07 | [73.33, 74.85] |
| FA | 2027 | 5.0 | 11375 | 827920365 | 874041362 | 920162363 | 0.121953 | 75.64 | [74.93, 76.36] |
| FA | 2027 | 5.5 | 12521 | 911331068 | 962098628 | 1012866193 | 0.124260 | 77.07 | [76.38, 77.80] |
| FA | 2027 | 6.0 | 13654 | 993795575 | 1049156989 | 1104518409 | 0.138480 | 85.89 | [85.27, 86.55] |
| FP | 2026 | 0.5 | 1138 | 82892373 | 87474528 | 92056683 | 0.070493 | 43.73 | [43.05, 44.43] |
| FP | 2026 | 1.0 | 2277 | 165857587 | 175025923 | 184194259 | 0.089706 | 55.65 | [54.90, 56.31] |
| FP | 2026 | 1.5 | 3415 | 248749960 | 262500451 | 276250942 | 0.098120 | 60.87 | [60.12, 61.55] |
| FP | 2026 | 2.0 | 4552 | 331569493 | 349898112 | 368226732 | 0.106603 | 66.13 | [65.33, 66.89] |
| FP | 2026 | 2.5 | 5686 | 414170505 | 437065172 | 459959841 | 0.107561 | 66.73 | [65.98, 67.43] |
| FP | 2026 | 3.0 | 6831 | 497572761 | 525077768 | 552582778 | 0.113892 | 70.65 | [69.93, 71.40] |
| FP | 2026 | 3.5 | 7967 | 580319453 | 612398562 | 644477674 | 0.115019 | 71.35 | [70.63, 72.06] |
| FP | 2026 | 4.0 | 9105 | 663211827 | 699873090 | 736534357 | 0.118651 | 73.61 | [72.94, 74.31] |
| FP | 2026 | 4.5 | 10242 | 746031359 | 787270751 | 828510147 | 0.119292 | 74.00 | [73.32, 74.73] |
| FP | 2026 | 5.0 | 11380 | 828923733 | 874745279 | 920566830 | 0.124138 | 77.01 | [76.29, 77.74] |
| FP | 2026 | 5.5 | 12518 | 911816106 | 962219807 | 1012623513 | 0.125928 | 78.12 | [77.43, 78.85] |
| FP | 2026 | 6.0 | 13654 | 994562798 | 1049540601 | 1104518409 | 0.139033 | 86.25 | [85.58, 86.98] |
| FP | 2027 | 0.5 | 1134 | 82537292 | 87135200 | 91733109 | 0.069612 | 43.18 | [42.47, 43.89] |
| FP | 2027 | 1.0 | 2278 | 165802426 | 175038789 | 184275153 | 0.088364 | 54.81 | [54.10, 55.55] |
| FP | 2027 | 1.5 | 3413 | 248412502 | 262250828 | 276089156 | 0.096235 | 59.69 | [58.93, 60.43] |
| FP | 2027 | 2.0 | 4544 | 330731441 | 349155512 | 367579585 | 0.103319 | 64.09 | [63.36, 64.79] |
| FP | 2027 | 2.5 | 5682 | 413559869 | 436598067 | 459636268 | 0.109787 | 68.10 | [67.33, 68.84] |
| FP | 2027 | 3.0 | 6820 | 496388298 | 524040623 | 551692951 | 0.114433 | 70.98 | [70.26, 71.70] |
| FP | 2027 | 3.5 | 7961 | 579435079 | 611713695 | 643992314 | 0.115811 | 71.83 | [71.11, 72.55] |
| FP | 2027 | 4.0 | 9109 | 662991350 | 699924639 | 736857931 | 0.117744 | 73.03 | [72.31, 73.81] |
| FP | 2027 | 4.5 | 10240 | 745310289 | 786829323 | 828348360 | 0.119421 | 74.07 | [73.33, 74.85] |
| FP | 2027 | 5.0 | 11375 | 827920365 | 874041362 | 920162363 | 0.121928 | 75.63 | [74.91, 76.35] |
| FP | 2027 | 5.5 | 12521 | 911331068 | 962098628 | 1012866193 | 0.124259 | 77.07 | [76.38, 77.80] |
| FP | 2027 | 6.0 | 13654 | 993795575 | 1049156989 | 1104518409 | 0.138483 | 85.90 | [85.27, 86.56] |
| PF | 2026 | 0.5 | 1138 | 82892373 | 87474528 | 92056683 | 0.070509 | 43.74 | [43.06, 44.43] |
| PF | 2026 | 1.0 | 2277 | 165857587 | 175025923 | 184194259 | 0.089704 | 55.65 | [54.91, 56.31] |
| PF | 2026 | 1.5 | 3415 | 248749960 | 262500451 | 276250942 | 0.098128 | 60.87 | [60.13, 61.57] |
| PF | 2026 | 2.0 | 4552 | 331569493 | 349898112 | 368226732 | 0.106640 | 66.16 | [65.37, 66.92] |
| PF | 2026 | 2.5 | 5686 | 414170505 | 437065172 | 459959841 | 0.107593 | 66.75 | [66.00, 67.45] |
| PF | 2026 | 3.0 | 6831 | 497572761 | 525077768 | 552582778 | 0.113938 | 70.68 | [69.96, 71.41] |
| PF | 2026 | 3.5 | 7967 | 580319453 | 612398562 | 644477674 | 0.115062 | 71.38 | [70.66, 72.08] |
| PF | 2026 | 4.0 | 9105 | 663211827 | 699873090 | 736534357 | 0.118701 | 73.64 | [72.95, 74.34] |
| PF | 2026 | 4.5 | 10242 | 746031359 | 787270751 | 828510147 | 0.119314 | 74.02 | [73.33, 74.75] |
| PF | 2026 | 5.0 | 11380 | 828923733 | 874745279 | 920566830 | 0.124169 | 77.03 | [76.30, 77.74] |
| PF | 2026 | 5.5 | 12518 | 911816106 | 962219807 | 1012623513 | 0.125991 | 78.16 | [77.48, 78.89] |
| PF | 2026 | 6.0 | 13654 | 994562798 | 1049540601 | 1104518409 | 0.139060 | 86.27 | [85.59, 86.99] |
| PF | 2027 | 0.5 | 1134 | 82537292 | 87135200 | 91733109 | 0.069626 | 43.19 | [42.48, 43.90] |
| PF | 2027 | 1.0 | 2278 | 165802426 | 175038789 | 184275153 | 0.088377 | 54.82 | [54.10, 55.58] |
| PF | 2027 | 1.5 | 3413 | 248412502 | 262250828 | 276089156 | 0.096234 | 59.69 | [58.93, 60.43] |
| PF | 2027 | 2.0 | 4544 | 330731441 | 349155512 | 367579585 | 0.103349 | 64.10 | [63.38, 64.82] |
| PF | 2027 | 2.5 | 5682 | 413559869 | 436598067 | 459636268 | 0.109821 | 68.12 | [67.35, 68.85] |
| PF | 2027 | 3.0 | 6820 | 496388298 | 524040623 | 551692951 | 0.114470 | 71.00 | [70.30, 71.72] |
| PF | 2027 | 3.5 | 7961 | 579435079 | 611713695 | 643992314 | 0.115811 | 71.83 | [71.12, 72.55] |
| PF | 2027 | 4.0 | 9109 | 662991350 | 699924639 | 736857931 | 0.117822 | 73.08 | [72.34, 73.85] |
| PF | 2027 | 4.5 | 10240 | 745310289 | 786829323 | 828348360 | 0.119479 | 74.11 | [73.36, 74.89] |
| PF | 2027 | 5.0 | 11375 | 827920365 | 874041362 | 920162363 | 0.121990 | 75.67 | [74.96, 76.39] |
| PF | 2027 | 5.5 | 12521 | 911331068 | 962098628 | 1012866193 | 0.124327 | 77.12 | [76.44, 77.84] |
| PF | 2027 | 6.0 | 13654 | 993795575 | 1049156989 | 1104518409 | 0.138517 | 85.92 | [85.28, 86.58] |
| FA_1024 | 2026 | 0.5 | 32 | 41407622 | 41407622 | 41407622 | 0.018287 | 11.34 | [10.96, 11.76] |
| FA_1024 | 2026 | 1.0 | 64 | 82815245 | 82815245 | 82815245 | 0.015925 | 9.88 | [9.54, 10.22] |
| FA_1024 | 2026 | 1.5 | 96 | 124222867 | 124222867 | 124222867 | 0.022205 | 13.77 | [13.32, 14.26] |
| FA_1024 | 2026 | 2.0 | 128 | 165630489 | 165630489 | 165630489 | 0.022295 | 13.83 | [13.37, 14.32] |
| FA_1024 | 2026 | 2.5 | 160 | 207038112 | 207038112 | 207038112 | 0.024320 | 15.09 | [14.58, 15.60] |
| FA_1024 | 2026 | 3.0 | 192 | 248445734 | 248445734 | 248445734 | 0.027065 | 16.79 | [16.25, 17.33] |
| FA_1024 | 2026 | 3.5 | 224 | 289853356 | 289853356 | 289853356 | 0.029173 | 18.10 | [17.52, 18.64] |
| FA_1024 | 2026 | 4.0 | 256 | 331260979 | 331260979 | 331260979 | 0.030058 | 18.65 | [18.10, 19.18] |
| FA_1024 | 2026 | 4.5 | 288 | 372668601 | 372668601 | 372668601 | 0.031281 | 19.41 | [18.85, 19.94] |
| FA_1024 | 2026 | 5.0 | 320 | 414076223 | 414076223 | 414076223 | 0.032349 | 20.07 | [19.49, 20.62] |
| FA_1024 | 2026 | 5.5 | 352 | 455483845 | 455483845 | 455483845 | 0.033670 | 20.89 | [20.31, 21.45] |
| FA_1024 | 2026 | 6.0 | 384 | 496891468 | 496891468 | 496891468 | 0.034929 | 21.67 | [21.09, 22.23] |
| FA_1024 | 2027 | 0.5 | 32 | 41475349 | 41475349 | 41475349 | 0.019572 | 12.14 | [11.75, 12.56] |
| FA_1024 | 2027 | 1.0 | 64 | 82950699 | 82950699 | 82950699 | 0.014273 | 8.85 | [8.56, 9.17] |
| FA_1024 | 2027 | 1.5 | 96 | 124426048 | 124426048 | 124426048 | 0.021529 | 13.35 | [12.92, 13.82] |
| FA_1024 | 2027 | 2.0 | 128 | 165901398 | 165901398 | 165901398 | 0.021884 | 13.57 | [13.12, 14.02] |
| FA_1024 | 2027 | 2.5 | 160 | 207376747 | 207376747 | 207376747 | 0.023762 | 14.74 | [14.24, 15.21] |
| FA_1024 | 2027 | 3.0 | 192 | 248852097 | 248852097 | 248852097 | 0.025465 | 15.80 | [15.30, 16.29] |
| FA_1024 | 2027 | 3.5 | 224 | 290327446 | 290327446 | 290327446 | 0.028072 | 17.41 | [16.91, 17.94] |
| FA_1024 | 2027 | 4.0 | 256 | 331802796 | 331802796 | 331802796 | 0.029000 | 17.99 | [17.45, 18.50] |
| FA_1024 | 2027 | 4.5 | 288 | 373278145 | 373278145 | 373278145 | 0.031070 | 19.27 | [18.74, 19.82] |
| FA_1024 | 2027 | 5.0 | 320 | 414753495 | 414753495 | 414753495 | 0.032199 | 19.97 | [19.41, 20.54] |
| FA_1024 | 2027 | 5.5 | 352 | 456228844 | 456228844 | 456228844 | 0.033854 | 21.00 | [20.40, 21.59] |
| FA_1024 | 2027 | 6.0 | 384 | 497704194 | 497704194 | 497704194 | 0.034986 | 21.70 | [21.07, 22.30] |
| DP8 | 2026 | 0.5 | 32 | 41407622 | 41407622 | 41407622 | 0.018264 | 11.33 | [10.95, 11.75] |
| DP8 | 2026 | 1.0 | 64 | 82815245 | 82815245 | 82815245 | 0.015814 | 9.81 | [9.48, 10.16] |
| DP8 | 2026 | 1.5 | 96 | 124222867 | 124222867 | 124222867 | 0.022278 | 13.82 | [13.36, 14.31] |
| DP8 | 2026 | 2.0 | 128 | 165630489 | 165630489 | 165630489 | 0.022039 | 13.67 | [13.21, 14.16] |
| DP8 | 2026 | 2.5 | 160 | 207038112 | 207038112 | 207038112 | 0.023992 | 14.88 | [14.37, 15.38] |
| DP8 | 2026 | 3.0 | 192 | 248445734 | 248445734 | 248445734 | 0.026513 | 16.45 | [15.91, 16.98] |
| DP8 | 2026 | 3.5 | 224 | 289853356 | 289853356 | 289853356 | 0.028806 | 17.87 | [17.32, 18.42] |
| DP8 | 2026 | 4.0 | 256 | 331260979 | 331260979 | 331260979 | 0.029709 | 18.43 | [17.91, 18.98] |
| DP8 | 2026 | 4.5 | 288 | 372668601 | 372668601 | 372668601 | 0.031312 | 19.42 | [18.87, 19.96] |
| DP8 | 2026 | 5.0 | 320 | 414076223 | 414076223 | 414076223 | 0.032190 | 19.97 | [19.38, 20.53] |
| DP8 | 2026 | 5.5 | 352 | 455483845 | 455483845 | 455483845 | 0.033351 | 20.69 | [20.10, 21.24] |
| DP8 | 2026 | 6.0 | 384 | 496891468 | 496891468 | 496891468 | 0.035172 | 21.82 | [21.25, 22.37] |
| DP8 | 2027 | 0.5 | 32 | 41475349 | 41475349 | 41475349 | 0.019687 | 12.21 | [11.81, 12.64] |
| DP8 | 2027 | 1.0 | 64 | 82950699 | 82950699 | 82950699 | 0.014399 | 8.93 | [8.63, 9.24] |
| DP8 | 2027 | 1.5 | 96 | 124426048 | 124426048 | 124426048 | 0.021535 | 13.36 | [12.89, 13.82] |
| DP8 | 2027 | 2.0 | 128 | 165901398 | 165901398 | 165901398 | 0.021849 | 13.55 | [13.09, 14.02] |
| DP8 | 2027 | 2.5 | 160 | 207376747 | 207376747 | 207376747 | 0.023631 | 14.66 | [14.17, 15.13] |
| DP8 | 2027 | 3.0 | 192 | 248852097 | 248852097 | 248852097 | 0.025211 | 15.64 | [15.12, 16.14] |
| DP8 | 2027 | 3.5 | 224 | 290327446 | 290327446 | 290327446 | 0.027531 | 17.08 | [16.55, 17.59] |
| DP8 | 2027 | 4.0 | 256 | 331802796 | 331802796 | 331802796 | 0.028602 | 17.74 | [17.20, 18.25] |
| DP8 | 2027 | 4.5 | 288 | 373278145 | 373278145 | 373278145 | 0.030083 | 18.66 | [18.13, 19.19] |
| DP8 | 2027 | 5.0 | 320 | 414753495 | 414753495 | 414753495 | 0.031101 | 19.29 | [18.74, 19.82] |
| DP8 | 2027 | 5.5 | 352 | 456228844 | 456228844 | 456228844 | 0.032648 | 20.25 | [19.69, 20.81] |
| DP8 | 2027 | 6.0 | 384 | 497704194 | 497704194 | 497704194 | 0.033837 | 20.99 | [20.39, 21.56] |
| DP1 | 2026 | 0.5 | 32 | 41407622 | 41407622 | 41407622 | 0.018200 | 11.29 | [10.91, 11.70] |
| DP1 | 2026 | 1.0 | 64 | 82815245 | 82815245 | 82815245 | 0.016093 | 9.98 | [9.64, 10.36] |
| DP1 | 2026 | 1.5 | 96 | 124222867 | 124222867 | 124222867 | 0.021893 | 13.58 | [13.09, 14.07] |
| DP1 | 2026 | 2.0 | 128 | 165630489 | 165630489 | 165630489 | 0.021549 | 13.37 | [12.92, 13.83] |
| DP1 | 2026 | 2.5 | 160 | 207038112 | 207038112 | 207038112 | 0.022958 | 14.24 | [13.77, 14.71] |
| DP1 | 2026 | 3.0 | 192 | 248445734 | 248445734 | 248445734 | 0.025731 | 15.96 | [15.44, 16.47] |
| DP1 | 2026 | 3.5 | 224 | 289853356 | 289853356 | 289853356 | 0.027815 | 17.26 | [16.72, 17.79] |
| DP1 | 2026 | 4.0 | 256 | 331260979 | 331260979 | 331260979 | 0.028490 | 17.67 | [17.16, 18.21] |
| DP1 | 2026 | 4.5 | 288 | 372668601 | 372668601 | 372668601 | 0.029992 | 18.61 | [18.07, 19.15] |
| DP1 | 2026 | 5.0 | 320 | 414076223 | 414076223 | 414076223 | 0.031554 | 19.57 | [19.01, 20.10] |
| DP1 | 2026 | 5.5 | 352 | 455483845 | 455483845 | 455483845 | 0.032324 | 20.05 | [19.49, 20.60] |
| DP1 | 2026 | 6.0 | 384 | 496891468 | 496891468 | 496891468 | 0.033625 | 20.86 | [20.28, 21.40] |
| FA_Q8 | 2026 | 0.5 | 1138 | 51807835 | 56389989 | 57535540 | 0.070381 | 43.66 | [42.98, 44.33] |
| FA_Q8 | 2026 | 1.0 | 2277 | 103661195 | 112829531 | 115121638 | 0.089690 | 55.64 | [54.88, 56.32] |
| FA_Q8 | 2026 | 1.5 | 3415 | 155469030 | 169219520 | 172657177 | 0.098438 | 61.07 | [60.29, 61.80] |
| FA_Q8 | 2026 | 2.0 | 4552 | 207231339 | 225559958 | 230142159 | 0.106606 | 66.13 | [65.32, 66.87] |
| FA_Q8 | 2026 | 2.5 | 5686 | 258857073 | 281751740 | 287475464 | 0.107724 | 66.83 | [66.04, 67.58] |
| FA_Q8 | 2026 | 3.0 | 6831 | 310983585 | 338488592 | 345364913 | 0.113843 | 70.62 | [69.92, 71.34] |
| FA_Q8 | 2026 | 3.5 | 7967 | 362700369 | 394779478 | 402799336 | 0.115147 | 71.43 | [70.74, 72.14] |
| FA_Q8 | 2026 | 4.0 | 9105 | 414508204 | 451169468 | 460334876 | 0.118837 | 73.72 | [73.06, 74.44] |
| FA_Q8 | 2026 | 4.5 | 10242 | 466270514 | 507509905 | 517819857 | 0.119371 | 74.05 | [73.36, 74.79] |
| FA_Q8 | 2026 | 5.0 | 11380 | 518078349 | 563899895 | 575355397 | 0.123734 | 76.76 | [76.04, 77.50] |
| FA_Q8 | 2026 | 5.5 | 12518 | 569886184 | 620289884 | 632890936 | 0.125616 | 77.93 | [77.24, 78.66] |
| FA_Q8 | 2026 | 6.0 | 13654 | 621602968 | 676580770 | 690325359 | 0.138646 | 86.01 | [85.35, 86.69] |
| FA_Q8 | 2027 | 0.5 | 1134 | 51585909 | 56183817 | 57333306 | 0.069520 | 43.12 | [42.43, 43.83] |
| FA_Q8 | 2027 | 1.0 | 2278 | 103626719 | 112863082 | 115172196 | 0.088255 | 54.74 | [54.02, 55.49] |
| FA_Q8 | 2027 | 1.5 | 3413 | 155258118 | 169096444 | 172556061 | 0.096556 | 59.89 | [59.06, 60.65] |
| FA_Q8 | 2027 | 2.0 | 4544 | 206707556 | 225131627 | 229737691 | 0.103206 | 64.02 | [63.30, 64.72] |
| FA_Q8 | 2027 | 2.5 | 5682 | 258475425 | 281513623 | 287273231 | 0.109986 | 68.22 | [67.46, 68.99] |
| FA_Q8 | 2027 | 3.0 | 6820 | 310243295 | 337895620 | 344808770 | 0.114348 | 70.93 | [70.19, 71.67] |
| FA_Q8 | 2027 | 3.5 | 7961 | 362147635 | 394426250 | 402495985 | 0.115898 | 71.89 | [71.18, 72.61] |
| FA_Q8 | 2027 | 4.0 | 9109 | 414370406 | 451303695 | 460537110 | 0.117416 | 72.83 | [72.12, 73.60] |
| FA_Q8 | 2027 | 4.5 | 10240 | 465819844 | 507338877 | 517718740 | 0.119505 | 74.12 | [73.36, 74.88] |
| FA_Q8 | 2027 | 5.0 | 11375 | 517451243 | 563572239 | 575102604 | 0.121780 | 75.54 | [74.79, 76.26] |
| FA_Q8 | 2027 | 5.5 | 12521 | 569583034 | 620350594 | 633042612 | 0.124227 | 77.05 | [76.35, 77.73] |
| FA_Q8 | 2027 | 6.0 | 13654 | 621123452 | 676484866 | 690325359 | 0.138370 | 85.83 | [85.18, 86.49] |
| FA_WARM | 2026 | 0.5 | 1138 | 82892373 | 87474528 | 92056683 | 0.171307 | 106.27 | [105.53, 107.05] |
| FA_WARM | 2026 | 1.0 | 2277 | 165857587 | 175025923 | 184194259 | 0.170863 | 106.00 | [105.23, 106.78] |
| FA_WARM | 2026 | 1.5 | 3415 | 248749960 | 262500451 | 276250942 | 0.171550 | 106.42 | [105.65, 107.21] |
| FA_WARM | 2026 | 2.0 | 4552 | 331569493 | 349898112 | 368226732 | 0.170237 | 105.61 | [104.86, 106.41] |
| FA_WARM | 2026 | 2.5 | 5686 | 414170505 | 437065172 | 459959841 | 0.170816 | 105.97 | [105.21, 106.77] |
| FA_WARM | 2026 | 3.0 | 6831 | 497572761 | 525077768 | 552582778 | 0.170201 | 105.59 | [104.78, 106.34] |
| FA_WARM | 2026 | 3.5 | 7967 | 580319453 | 612398562 | 644477674 | 0.170202 | 105.59 | [104.84, 106.38] |
| FA_WARM | 2026 | 4.0 | 9105 | 663211827 | 699873090 | 736534357 | 0.169706 | 105.28 | [104.57, 106.02] |
| FA_WARM | 2026 | 4.5 | 10242 | 746031359 | 787270751 | 828510147 | 0.170048 | 105.49 | [104.75, 106.27] |
| FA_WARM | 2026 | 5.0 | 11380 | 828923733 | 874745279 | 920566830 | 0.170148 | 105.55 | [104.83, 106.35] |
| FA_WARM | 2026 | 5.5 | 12518 | 911816106 | 962219807 | 1012623513 | 0.169926 | 105.42 | [104.69, 106.19] |
| FA_WARM | 2026 | 6.0 | 13654 | 994562798 | 1049540601 | 1104518409 | 0.170716 | 105.91 | [105.18, 106.69] |
| FA_WARM | 2027 | 0.5 | 1134 | 82537292 | 87135200 | 91733109 | 0.171669 | 106.48 | [105.73, 107.20] |
| FA_WARM | 2027 | 1.0 | 2278 | 165802426 | 175038789 | 184275153 | 0.171481 | 106.36 | [105.63, 107.12] |
| FA_WARM | 2027 | 1.5 | 3413 | 248412502 | 262250828 | 276089156 | 0.170676 | 105.86 | [105.13, 106.61] |
| FA_WARM | 2027 | 2.0 | 4544 | 330731441 | 349155512 | 367579585 | 0.170821 | 105.95 | [105.21, 106.70] |
| FA_WARM | 2027 | 2.5 | 5682 | 413559869 | 436598067 | 459636268 | 0.170582 | 105.81 | [105.13, 106.50] |
| FA_WARM | 2027 | 3.0 | 6820 | 496388298 | 524040623 | 551692951 | 0.170726 | 105.90 | [105.16, 106.63] |
| FA_WARM | 2027 | 3.5 | 7961 | 579435079 | 611713695 | 643992314 | 0.170511 | 105.76 | [105.00, 106.54] |
| FA_WARM | 2027 | 4.0 | 9109 | 662991350 | 699924639 | 736857931 | 0.170292 | 105.63 | [104.91, 106.39] |
| FA_WARM | 2027 | 4.5 | 10240 | 745310289 | 786829323 | 828348360 | 0.170070 | 105.49 | [104.74, 106.24] |
| FA_WARM | 2027 | 5.0 | 11375 | 827920365 | 874041362 | 920162363 | 0.169657 | 105.23 | [104.53, 105.97] |
| FA_WARM | 2027 | 5.5 | 12521 | 911331068 | 962098628 | 1012866193 | 0.169926 | 105.40 | [104.68, 106.12] |
| FA_WARM | 2027 | 6.0 | 13654 | 993795575 | 1049156989 | 1104518409 | 0.170544 | 105.78 | [105.08, 106.55] |
| FA_WARM_FROZEN | 2026 | 0.5 | 1138 | 82835569 | 82893944 | 82952320 | 0.171976 | 106.69 | [105.91, 107.51] |
| FA_WARM_FROZEN | 2026 | 1.0 | 2277 | 83892522 | 84009324 | 84126126 | 0.171840 | 106.60 | [105.82, 107.41] |
| FA_WARM_FROZEN | 2026 | 1.5 | 3415 | 84948547 | 85123724 | 85298902 | 0.172183 | 106.82 | [106.06, 107.58] |
| FA_WARM_FROZEN | 2026 | 2.0 | 4552 | 86003644 | 86237144 | 86470647 | 0.172129 | 106.78 | [106.01, 107.60] |
| FA_WARM_FROZEN | 2026 | 2.5 | 5686 | 87055957 | 87347628 | 87639300 | 0.172367 | 106.93 | [106.16, 107.74] |
| FA_WARM_FROZEN | 2026 | 3.0 | 6831 | 88118478 | 88468883 | 88819290 | 0.172248 | 106.86 | [106.10, 107.65] |
| FA_WARM_FROZEN | 2026 | 3.5 | 7967 | 89172647 | 89581324 | 89990004 | 0.171824 | 106.59 | [105.81, 107.39] |
| FA_WARM_FROZEN | 2026 | 4.0 | 9105 | 90228672 | 90695724 | 91162780 | 0.172465 | 106.99 | [106.21, 107.79] |
| FA_WARM_FROZEN | 2026 | 4.5 | 10242 | 91283769 | 91809145 | 92334525 | 0.172000 | 106.70 | [105.94, 107.53] |
| FA_WARM_FROZEN | 2026 | 5.0 | 11380 | 92339794 | 92923545 | 93507301 | 0.172256 | 106.86 | [106.08, 107.64] |
| FA_WARM_FROZEN | 2026 | 5.5 | 12518 | 93395819 | 94037945 | 94680076 | 0.172071 | 106.75 | [105.98, 107.54] |
| FA_WARM_FROZEN | 2026 | 6.0 | 13654 | 94449988 | 95150387 | 95850791 | 0.172499 | 107.01 | [106.24, 107.85] |
| FA_WARM_FROZEN | 2027 | 0.5 | 1134 | 82830421 | 82888997 | 82947574 | 0.171982 | 106.67 | [105.97, 107.44] |
| FA_WARM_FROZEN | 2027 | 1.0 | 2278 | 83891195 | 84008863 | 84126533 | 0.171841 | 106.59 | [105.84, 107.36] |
| FA_WARM_FROZEN | 2027 | 1.5 | 3413 | 84943624 | 85119920 | 85296217 | 0.171822 | 106.58 | [105.84, 107.32] |
| FA_WARM_FROZEN | 2027 | 2.0 | 4544 | 85992344 | 86227060 | 86461778 | 0.172228 | 106.83 | [106.07, 107.59] |
| FA_WARM_FROZEN | 2027 | 2.5 | 5682 | 87047554 | 87341053 | 87634554 | 0.172215 | 106.82 | [106.08, 107.59] |
| FA_WARM_FROZEN | 2027 | 3.0 | 6820 | 88102764 | 88455046 | 88807330 | 0.171876 | 106.61 | [105.90, 107.39] |
| FA_WARM_FROZEN | 2027 | 3.5 | 7961 | 89160756 | 89571975 | 89983197 | 0.172039 | 106.71 | [105.98, 107.48] |
| FA_WARM_FROZEN | 2027 | 4.0 | 9109 | 90225239 | 90695757 | 91166278 | 0.172461 | 106.97 | [106.24, 107.74] |
| FA_WARM_FROZEN | 2027 | 4.5 | 10240 | 91273959 | 91802897 | 92331840 | 0.172577 | 107.04 | [106.30, 107.81] |
| FA_WARM_FROZEN | 2027 | 5.0 | 11375 | 92326388 | 92913954 | 93501524 | 0.172401 | 106.94 | [106.21, 107.68] |
| FA_WARM_FROZEN | 2027 | 5.5 | 12521 | 93389016 | 94035778 | 94682544 | 0.172213 | 106.82 | [106.06, 107.60] |
| FA_WARM_FROZEN | 2027 | 6.0 | 13654 | 94439590 | 95144876 | 95850167 | 0.172413 | 106.94 | [106.22, 107.72] |
| DP8_FROZEN | 2026 | 0.5 | 32 | 78207856 | 78207856 | 78207856 | 0.172739 | 107.16 | [106.39, 107.94] |
| DP8_FROZEN | 2026 | 1.0 | 64 | 78735377 | 78735377 | 78735377 | 0.172552 | 107.04 | [106.27, 107.84] |
| DP8_FROZEN | 2026 | 1.5 | 96 | 79262898 | 79262898 | 79262898 | 0.172386 | 106.94 | [106.17, 107.73] |
| DP8_FROZEN | 2026 | 2.0 | 128 | 79790419 | 79790419 | 79790419 | 0.172292 | 106.88 | [106.12, 107.68] |
| DP8_FROZEN | 2026 | 2.5 | 160 | 80317940 | 80317940 | 80317940 | 0.172234 | 106.85 | [106.07, 107.63] |
| DP8_FROZEN | 2026 | 3.0 | 192 | 80845462 | 80845462 | 80845462 | 0.172194 | 106.82 | [106.04, 107.63] |
| DP8_FROZEN | 2026 | 3.5 | 224 | 81372983 | 81372983 | 81372983 | 0.172100 | 106.76 | [105.98, 107.55] |
| DP8_FROZEN | 2026 | 4.0 | 256 | 81900504 | 81900504 | 81900504 | 0.172028 | 106.72 | [105.95, 107.53] |
| DP8_FROZEN | 2026 | 4.5 | 288 | 82428025 | 82428025 | 82428025 | 0.172042 | 106.73 | [105.95, 107.52] |
| DP8_FROZEN | 2026 | 5.0 | 320 | 82955546 | 82955546 | 82955546 | 0.172095 | 106.76 | [105.95, 107.54] |
| DP8_FROZEN | 2026 | 5.5 | 352 | 83483068 | 83483068 | 83483068 | 0.172048 | 106.73 | [105.96, 107.52] |
| DP8_FROZEN | 2026 | 6.0 | 384 | 84010589 | 84010589 | 84010589 | 0.172028 | 106.72 | [105.95, 107.51] |
| DP8_FROZEN | 2027 | 0.5 | 32 | 78282966 | 78282966 | 78282966 | 0.172702 | 107.12 | [106.42, 107.88] |
| DP8_FROZEN | 2027 | 1.0 | 64 | 78811350 | 78811350 | 78811350 | 0.172520 | 107.01 | [106.30, 107.76] |
| DP8_FROZEN | 2027 | 1.5 | 96 | 79339734 | 79339734 | 79339734 | 0.172454 | 106.97 | [106.24, 107.72] |
| DP8_FROZEN | 2027 | 2.0 | 128 | 79868118 | 79868118 | 79868118 | 0.172313 | 106.88 | [106.16, 107.63] |
| DP8_FROZEN | 2027 | 2.5 | 160 | 80396502 | 80396502 | 80396502 | 0.172209 | 106.82 | [106.08, 107.58] |
| DP8_FROZEN | 2027 | 3.0 | 192 | 80924886 | 80924886 | 80924886 | 0.172126 | 106.76 | [106.05, 107.50] |
| DP8_FROZEN | 2027 | 3.5 | 224 | 81453270 | 81453270 | 81453270 | 0.172104 | 106.75 | [106.04, 107.49] |
| DP8_FROZEN | 2027 | 4.0 | 256 | 81981654 | 81981654 | 81981654 | 0.172010 | 106.69 | [105.99, 107.43] |
| DP8_FROZEN | 2027 | 4.5 | 288 | 82510038 | 82510038 | 82510038 | 0.172041 | 106.71 | [106.00, 107.44] |
| DP8_FROZEN | 2027 | 5.0 | 320 | 83038422 | 83038422 | 83038422 | 0.172018 | 106.70 | [105.97, 107.41] |
| DP8_FROZEN | 2027 | 5.5 | 352 | 83566806 | 83566806 | 83566806 | 0.171935 | 106.65 | [105.92, 107.38] |
| DP8_FROZEN | 2027 | 6.0 | 384 | 84095190 | 84095190 | 84095190 | 0.171934 | 106.65 | [105.92, 107.38] |
| FP_WARM_FROZEN | 2026 | 0.5 | 1138 | 82835569 | 82893944 | 82952320 | 0.171978 | 106.69 | [105.91, 107.51] |
| FP_WARM_FROZEN | 2026 | 1.0 | 2277 | 83892522 | 84009324 | 84126126 | 0.171848 | 106.61 | [105.83, 107.41] |
| FP_WARM_FROZEN | 2026 | 1.5 | 3415 | 84948547 | 85123724 | 85298902 | 0.172185 | 106.82 | [106.06, 107.58] |
| FP_WARM_FROZEN | 2026 | 2.0 | 4552 | 86003644 | 86237144 | 86470647 | 0.172128 | 106.78 | [106.01, 107.59] |
| FP_WARM_FROZEN | 2026 | 2.5 | 5686 | 87055957 | 87347628 | 87639300 | 0.172365 | 106.93 | [106.15, 107.74] |
| FP_WARM_FROZEN | 2026 | 3.0 | 6831 | 88118478 | 88468883 | 88819290 | 0.172259 | 106.86 | [106.10, 107.65] |
| FP_WARM_FROZEN | 2026 | 3.5 | 7967 | 89172647 | 89581324 | 89990004 | 0.171827 | 106.59 | [105.81, 107.40] |
| FP_WARM_FROZEN | 2026 | 4.0 | 9105 | 90228672 | 90695724 | 91162780 | 0.172474 | 107.00 | [106.22, 107.79] |
| FP_WARM_FROZEN | 2026 | 4.5 | 10242 | 91283769 | 91809145 | 92334525 | 0.172006 | 106.71 | [105.94, 107.53] |
| FP_WARM_FROZEN | 2026 | 5.0 | 11380 | 92339794 | 92923545 | 93507301 | 0.172254 | 106.86 | [106.08, 107.64] |
| FP_WARM_FROZEN | 2026 | 5.5 | 12518 | 93395819 | 94037945 | 94680076 | 0.172084 | 106.75 | [105.98, 107.55] |
| FP_WARM_FROZEN | 2026 | 6.0 | 13654 | 94449988 | 95150387 | 95850791 | 0.172499 | 107.01 | [106.24, 107.85] |
| FP_WARM_FROZEN | 2027 | 0.5 | 1134 | 82830421 | 82888997 | 82947574 | 0.171983 | 106.68 | [105.97, 107.44] |
| FP_WARM_FROZEN | 2027 | 1.0 | 2278 | 83891195 | 84008863 | 84126533 | 0.171846 | 106.59 | [105.85, 107.36] |
| FP_WARM_FROZEN | 2027 | 1.5 | 3413 | 84943624 | 85119920 | 85296217 | 0.171826 | 106.58 | [105.84, 107.32] |
| FP_WARM_FROZEN | 2027 | 2.0 | 4544 | 85992344 | 86227060 | 86461778 | 0.172232 | 106.83 | [106.07, 107.60] |
| FP_WARM_FROZEN | 2027 | 2.5 | 5682 | 87047554 | 87341053 | 87634554 | 0.172212 | 106.82 | [106.08, 107.59] |
| FP_WARM_FROZEN | 2027 | 3.0 | 6820 | 88102764 | 88455046 | 88807330 | 0.171878 | 106.61 | [105.90, 107.39] |
| FP_WARM_FROZEN | 2027 | 3.5 | 7961 | 89160756 | 89571975 | 89983197 | 0.172044 | 106.71 | [105.99, 107.48] |
| FP_WARM_FROZEN | 2027 | 4.0 | 9109 | 90225239 | 90695757 | 91166278 | 0.172468 | 106.98 | [106.24, 107.74] |
| FP_WARM_FROZEN | 2027 | 4.5 | 10240 | 91273959 | 91802897 | 92331840 | 0.172572 | 107.04 | [106.30, 107.80] |
| FP_WARM_FROZEN | 2027 | 5.0 | 11375 | 92326388 | 92913954 | 93501524 | 0.172399 | 106.93 | [106.21, 107.68] |
| FP_WARM_FROZEN | 2027 | 5.5 | 12521 | 93389016 | 94035778 | 94682544 | 0.172200 | 106.81 | [106.06, 107.57] |
| FP_WARM_FROZEN | 2027 | 6.0 | 13654 | 94439590 | 95144876 | 95850167 | 0.172409 | 106.94 | [106.21, 107.72] |
| PF_WARM_FROZEN | 2026 | 0.5 | 1138 | 82835569 | 82893944 | 82952320 | 0.171985 | 106.69 | [105.91, 107.52] |
| PF_WARM_FROZEN | 2026 | 1.0 | 2277 | 83892522 | 84009324 | 84126126 | 0.171870 | 106.62 | [105.83, 107.43] |
| PF_WARM_FROZEN | 2026 | 1.5 | 3415 | 84948547 | 85123724 | 85298902 | 0.172161 | 106.80 | [106.05, 107.56] |
| PF_WARM_FROZEN | 2026 | 2.0 | 4552 | 86003644 | 86237144 | 86470647 | 0.172159 | 106.80 | [106.03, 107.63] |
| PF_WARM_FROZEN | 2026 | 2.5 | 5686 | 87055957 | 87347628 | 87639300 | 0.172379 | 106.94 | [106.16, 107.74] |
| PF_WARM_FROZEN | 2026 | 3.0 | 6831 | 88118478 | 88468883 | 88819290 | 0.172298 | 106.89 | [106.14, 107.69] |
| PF_WARM_FROZEN | 2026 | 3.5 | 7967 | 89172647 | 89581324 | 89990004 | 0.171857 | 106.61 | [105.82, 107.41] |
| PF_WARM_FROZEN | 2026 | 4.0 | 9105 | 90228672 | 90695724 | 91162780 | 0.172470 | 106.99 | [106.21, 107.79] |
| PF_WARM_FROZEN | 2026 | 4.5 | 10242 | 91283769 | 91809145 | 92334525 | 0.172073 | 106.75 | [105.98, 107.56] |
| PF_WARM_FROZEN | 2026 | 5.0 | 11380 | 92339794 | 92923545 | 93507301 | 0.172278 | 106.88 | [106.09, 107.68] |
| PF_WARM_FROZEN | 2026 | 5.5 | 12518 | 93395819 | 94037945 | 94680076 | 0.172074 | 106.75 | [105.99, 107.54] |
| PF_WARM_FROZEN | 2026 | 6.0 | 13654 | 94449988 | 95150387 | 95850791 | 0.172464 | 106.99 | [106.22, 107.81] |
| PF_WARM_FROZEN | 2027 | 0.5 | 1134 | 82830421 | 82888997 | 82947574 | 0.171987 | 106.68 | [105.97, 107.45] |
| PF_WARM_FROZEN | 2027 | 1.0 | 2278 | 83891195 | 84008863 | 84126533 | 0.171838 | 106.59 | [105.84, 107.35] |
| PF_WARM_FROZEN | 2027 | 1.5 | 3413 | 84943624 | 85119920 | 85296217 | 0.171823 | 106.58 | [105.84, 107.33] |
| PF_WARM_FROZEN | 2027 | 2.0 | 4544 | 85992344 | 86227060 | 86461778 | 0.172234 | 106.83 | [106.06, 107.60] |
| PF_WARM_FROZEN | 2027 | 2.5 | 5682 | 87047554 | 87341053 | 87634554 | 0.172216 | 106.82 | [106.09, 107.59] |
| PF_WARM_FROZEN | 2027 | 3.0 | 6820 | 88102764 | 88455046 | 88807330 | 0.171905 | 106.63 | [105.92, 107.41] |
| PF_WARM_FROZEN | 2027 | 3.5 | 7961 | 89160756 | 89571975 | 89983197 | 0.172075 | 106.73 | [106.01, 107.50] |
| PF_WARM_FROZEN | 2027 | 4.0 | 9109 | 90225239 | 90695757 | 91166278 | 0.172487 | 106.99 | [106.26, 107.76] |
| PF_WARM_FROZEN | 2027 | 4.5 | 10240 | 91273959 | 91802897 | 92331840 | 0.172586 | 107.05 | [106.31, 107.81] |
| PF_WARM_FROZEN | 2027 | 5.0 | 11375 | 92326388 | 92913954 | 93501524 | 0.172412 | 106.94 | [106.21, 107.68] |
| PF_WARM_FROZEN | 2027 | 5.5 | 12521 | 93389016 | 94035778 | 94682544 | 0.172268 | 106.85 | [106.11, 107.61] |
| PF_WARM_FROZEN | 2027 | 6.0 | 13654 | 94439590 | 95144876 | 95850167 | 0.172470 | 106.98 | [106.25, 107.75] |
| FA_Q8_WARM_FROZEN | 2026 | 0.5 | 1138 | 82439655 | 82498030 | 82512635 | 0.171999 | 106.70 | [105.92, 107.53] |
| FA_Q8_WARM_FROZEN | 2026 | 1.0 | 2277 | 83100346 | 83217148 | 83246370 | 0.171841 | 106.60 | [105.83, 107.42] |
| FA_Q8_WARM_FROZEN | 2026 | 1.5 | 3415 | 83760458 | 83935634 | 83979461 | 0.172185 | 106.82 | [106.07, 107.58] |
| FA_Q8_WARM_FROZEN | 2026 | 2.0 | 4552 | 84419989 | 84653489 | 84711908 | 0.172136 | 106.79 | [106.03, 107.60] |
| FA_Q8_WARM_FROZEN | 2026 | 2.5 | 5686 | 85077780 | 85369450 | 85442422 | 0.172385 | 106.94 | [106.16, 107.75] |
| FA_Q8_WARM_FROZEN | 2026 | 3.0 | 6831 | 85741951 | 86092356 | 86180022 | 0.172233 | 106.85 | [106.10, 107.63] |
| FA_Q8_WARM_FROZEN | 2026 | 3.5 | 7967 | 86400902 | 86809580 | 86911825 | 0.171818 | 106.59 | [105.80, 107.40] |
| FA_Q8_WARM_FROZEN | 2026 | 4.0 | 9105 | 87061014 | 87528066 | 87644916 | 0.172450 | 106.98 | [106.19, 107.78] |
| FA_Q8_WARM_FROZEN | 2026 | 4.5 | 10242 | 87720545 | 88245921 | 88377362 | 0.171983 | 106.69 | [105.93, 107.51] |
| FA_Q8_WARM_FROZEN | 2026 | 5.0 | 11380 | 88380656 | 88964407 | 89110453 | 0.172239 | 106.85 | [106.08, 107.63] |
| FA_Q8_WARM_FROZEN | 2026 | 5.5 | 12518 | 89040767 | 89682893 | 89843544 | 0.172048 | 106.73 | [105.95, 107.52] |
| FA_Q8_WARM_FROZEN | 2026 | 6.0 | 13654 | 89699718 | 90400117 | 90575347 | 0.172472 | 107.00 | [106.22, 107.83] |
| FA_Q8_WARM_FROZEN | 2027 | 0.5 | 1134 | 82436204 | 82494779 | 82509434 | 0.171996 | 106.68 | [105.97, 107.45] |
| FA_Q8_WARM_FROZEN | 2027 | 1.0 | 2278 | 83099283 | 83216951 | 83246390 | 0.171796 | 106.56 | [105.82, 107.32] |
| FA_Q8_WARM_FROZEN | 2027 | 1.5 | 3413 | 83757146 | 83933442 | 83977549 | 0.171823 | 106.58 | [105.84, 107.31] |
| FA_Q8_WARM_FROZEN | 2027 | 2.0 | 4544 | 84412691 | 84647408 | 84706130 | 0.172201 | 106.81 | [106.04, 107.58] |
| FA_Q8_WARM_FROZEN | 2027 | 2.5 | 5682 | 85072293 | 85365792 | 85439221 | 0.172197 | 106.81 | [106.07, 107.59] |
| FA_Q8_WARM_FROZEN | 2027 | 3.0 | 6820 | 85731895 | 86084176 | 86172312 | 0.171863 | 106.60 | [105.90, 107.38] |
| FA_Q8_WARM_FROZEN | 2027 | 3.5 | 7961 | 86393236 | 86804455 | 86907336 | 0.172056 | 106.72 | [105.99, 107.50] |
| FA_Q8_WARM_FROZEN | 2027 | 4.0 | 9109 | 87058634 | 87529152 | 87646868 | 0.172500 | 107.00 | [106.27, 107.76] |
| FA_Q8_WARM_FROZEN | 2027 | 4.5 | 10240 | 87714179 | 88243117 | 88375450 | 0.172559 | 107.03 | [106.28, 107.79] |
| FA_Q8_WARM_FROZEN | 2027 | 5.0 | 11375 | 88372042 | 88959608 | 89106608 | 0.172370 | 106.92 | [106.18, 107.65] |
| FA_Q8_WARM_FROZEN | 2027 | 5.5 | 12521 | 89036281 | 89683042 | 89844853 | 0.172221 | 106.82 | [106.07, 107.58] |
| FA_Q8_WARM_FROZEN | 2027 | 6.0 | 13654 | 89692985 | 90398270 | 90574723 | 0.172416 | 106.94 | [106.21, 107.71] |
| FA_1024_WARM_FROZEN | 2026 | 0.5 | 32 | 78207856 | 78207856 | 78207856 | 0.172743 | 107.16 | [106.39, 107.95] |
| FA_1024_WARM_FROZEN | 2026 | 1.0 | 64 | 78735377 | 78735377 | 78735377 | 0.172502 | 107.01 | [106.25, 107.82] |
| FA_1024_WARM_FROZEN | 2026 | 1.5 | 96 | 79262898 | 79262898 | 79262898 | 0.172424 | 106.97 | [106.20, 107.76] |
| FA_1024_WARM_FROZEN | 2026 | 2.0 | 128 | 79790419 | 79790419 | 79790419 | 0.172325 | 106.90 | [106.13, 107.71] |
| FA_1024_WARM_FROZEN | 2026 | 2.5 | 160 | 80317940 | 80317940 | 80317940 | 0.172321 | 106.90 | [106.13, 107.71] |
| FA_1024_WARM_FROZEN | 2026 | 3.0 | 192 | 80845462 | 80845462 | 80845462 | 0.172252 | 106.86 | [106.07, 107.66] |
| FA_1024_WARM_FROZEN | 2026 | 3.5 | 224 | 81372983 | 81372983 | 81372983 | 0.172237 | 106.85 | [106.06, 107.63] |
| FA_1024_WARM_FROZEN | 2026 | 4.0 | 256 | 81900504 | 81900504 | 81900504 | 0.172041 | 106.73 | [105.94, 107.52] |
| FA_1024_WARM_FROZEN | 2026 | 4.5 | 288 | 82428025 | 82428025 | 82428025 | 0.172108 | 106.77 | [105.98, 107.57] |
| FA_1024_WARM_FROZEN | 2026 | 5.0 | 320 | 82955546 | 82955546 | 82955546 | 0.172100 | 106.76 | [105.98, 107.55] |
| FA_1024_WARM_FROZEN | 2026 | 5.5 | 352 | 83483068 | 83483068 | 83483068 | 0.172107 | 106.77 | [105.99, 107.57] |
| FA_1024_WARM_FROZEN | 2026 | 6.0 | 384 | 84010589 | 84010589 | 84010589 | 0.172074 | 106.75 | [105.98, 107.56] |
| FA_1024_WARM_FROZEN | 2027 | 0.5 | 32 | 78282966 | 78282966 | 78282966 | 0.172713 | 107.13 | [106.41, 107.89] |
| FA_1024_WARM_FROZEN | 2027 | 1.0 | 64 | 78811350 | 78811350 | 78811350 | 0.172530 | 107.01 | [106.30, 107.76] |
| FA_1024_WARM_FROZEN | 2027 | 1.5 | 96 | 79339734 | 79339734 | 79339734 | 0.172424 | 106.95 | [106.23, 107.71] |
| FA_1024_WARM_FROZEN | 2027 | 2.0 | 128 | 79868118 | 79868118 | 79868118 | 0.172379 | 106.92 | [106.19, 107.69] |
| FA_1024_WARM_FROZEN | 2027 | 2.5 | 160 | 80396502 | 80396502 | 80396502 | 0.172228 | 106.83 | [106.10, 107.58] |
| FA_1024_WARM_FROZEN | 2027 | 3.0 | 192 | 80924886 | 80924886 | 80924886 | 0.172187 | 106.80 | [106.07, 107.54] |
| FA_1024_WARM_FROZEN | 2027 | 3.5 | 224 | 81453270 | 81453270 | 81453270 | 0.172211 | 106.82 | [106.11, 107.55] |
| FA_1024_WARM_FROZEN | 2027 | 4.0 | 256 | 81981654 | 81981654 | 81981654 | 0.172069 | 106.73 | [106.03, 107.48] |
| FA_1024_WARM_FROZEN | 2027 | 4.5 | 288 | 82510038 | 82510038 | 82510038 | 0.172012 | 106.69 | [105.98, 107.43] |
| FA_1024_WARM_FROZEN | 2027 | 5.0 | 320 | 83038422 | 83038422 | 83038422 | 0.172007 | 106.69 | [105.97, 107.43] |
| FA_1024_WARM_FROZEN | 2027 | 5.5 | 352 | 83566806 | 83566806 | 83566806 | 0.171967 | 106.67 | [105.94, 107.40] |
| FA_1024_WARM_FROZEN | 2027 | 6.0 | 384 | 84095190 | 84095190 | 84095190 | 0.172014 | 106.69 | [105.98, 107.45] |
| DP1_WARM_FROZEN | 2026 | 0.5 | 32 | 78207856 | 78207856 | 78207856 | 0.172734 | 107.16 | [106.39, 107.93] |
| DP1_WARM_FROZEN | 2026 | 1.0 | 64 | 78735377 | 78735377 | 78735377 | 0.172532 | 107.03 | [106.26, 107.83] |
| DP1_WARM_FROZEN | 2026 | 1.5 | 96 | 79262898 | 79262898 | 79262898 | 0.172385 | 106.94 | [106.17, 107.71] |
| DP1_WARM_FROZEN | 2026 | 2.0 | 128 | 79790419 | 79790419 | 79790419 | 0.172233 | 106.85 | [106.08, 107.65] |
| DP1_WARM_FROZEN | 2026 | 2.5 | 160 | 80317940 | 80317940 | 80317940 | 0.172183 | 106.82 | [106.05, 107.61] |
| DP1_WARM_FROZEN | 2026 | 3.0 | 192 | 80845462 | 80845462 | 80845462 | 0.172117 | 106.77 | [105.99, 107.57] |
| DP1_WARM_FROZEN | 2026 | 3.5 | 224 | 81372983 | 81372983 | 81372983 | 0.172106 | 106.77 | [105.97, 107.56] |
| DP1_WARM_FROZEN | 2026 | 4.0 | 256 | 81900504 | 81900504 | 81900504 | 0.172027 | 106.72 | [105.95, 107.51] |
| DP1_WARM_FROZEN | 2026 | 4.5 | 288 | 82428025 | 82428025 | 82428025 | 0.172009 | 106.71 | [105.96, 107.50] |
| DP1_WARM_FROZEN | 2026 | 5.0 | 320 | 82955546 | 82955546 | 82955546 | 0.171949 | 106.67 | [105.90, 107.46] |
| DP1_WARM_FROZEN | 2026 | 5.5 | 352 | 83483068 | 83483068 | 83483068 | 0.171875 | 106.63 | [105.85, 107.43] |
| DP1_WARM_FROZEN | 2026 | 6.0 | 384 | 84010589 | 84010589 | 84010589 | 0.171871 | 106.62 | [105.86, 107.42] |

06I §7(b) (variant registered in DECISION_LOG row 218): bytes per client are reported with the registered definition (06A §3: visits x (down + up), dropped clients not charged) and, beside it, with the dropped visits charged their download (a dropped client receives the model and never uploads): bytes_per_client_charged_dropped = visits x (down + up) + dropped visits x down per client (client mean). The variant that charges a dropped visit down + up (bytes_per_client_charged_dropped_up_down) is DESCRIPTIVE only. The curves carry the registered charged variant as a second series (the up + down variant as a DESCRIPTIVE third series).

### 5d. R per stratum (MAIN)

| family | group | arm | users | R% vs C_BIG | 95% CI |
|---|---|---|---|---|---|
| category_diversity | T0 | C_BIG | 1982 | 100.00 | [100.00, 100.00] |
| category_diversity | T0 | C_FULL | 1982 | 128.02 | [124.09, 132.69] |
| category_diversity | T0 | C_SMALL | 1982 | 93.94 | [91.55, 96.77] |
| category_diversity | T0 | FA | 1982 | 79.81 | [76.73, 83.12] |
| category_diversity | T0 | FP | 1982 | 79.80 | [76.70, 83.11] |
| category_diversity | T0 | PF | 1982 | 79.82 | [76.72, 83.09] |
| category_diversity | T0 | FA_1024 | 1982 | 13.50 | [11.03, 16.15] |
| category_diversity | T0 | DP8 | 1982 | 13.43 | [11.16, 15.92] |
| category_diversity | T0 | DP1 | 1982 | 13.94 | [11.73, 16.42] |
| category_diversity | T0 | FA_Q8 | 1982 | 80.06 | [77.16, 83.30] |
| category_diversity | T0 | FA_WARM | 1982 | 111.04 | [106.81, 115.84] |
| category_diversity | T0 | FA_WARM_FROZEN | 1982 | 111.62 | [107.00, 116.61] |
| category_diversity | T0 | DP8_FROZEN | 1982 | 111.86 | [107.37, 116.88] |
| category_diversity | T0 | FP_WARM_FROZEN | 1982 | 111.62 | [107.00, 116.61] |
| category_diversity | T0 | PF_WARM_FROZEN | 1982 | 111.80 | [107.22, 116.68] |
| category_diversity | T0 | FA_Q8_WARM_FROZEN | 1982 | 111.84 | [107.27, 116.77] |
| category_diversity | T0 | FA_1024_WARM_FROZEN | 1982 | 111.87 | [107.38, 116.86] |
| category_diversity | T0 | DP1_WARM_FROZEN | 1982 | 112.09 | [107.46, 117.69] |
| category_diversity | T0 | PRE | 1982 | 111.85 | [107.16, 117.49] |
| category_diversity | T0 | FT_C | 1982 | 96.34 | [93.85, 98.97] |
| category_diversity | T0 | FT_PRE | 1982 | 114.41 | [109.95, 119.32] |
| category_diversity | T0 | FT_FA | 1982 | 84.12 | [80.98, 87.37] |
| category_diversity | T0 | FT_FA_WARM | 1982 | 114.32 | [109.95, 119.34] |
| category_diversity | T0 | FT_FA_WARM_FROZEN | 1982 | 114.68 | [110.23, 119.47] |
| category_diversity | T0 | LO | 1982 | 2.16 | [1.40, 2.98] |
| category_diversity | T0 | KNN_EMB_SMALL | 1982 | 48.65 | [44.54, 52.85] |
| category_diversity | T0 | KNN_EMB_BIG | 1982 | 60.60 | [56.35, 65.19] |
| category_diversity | T0 | RULE | 1982 | 23.98 | [21.05, 27.11] |
| category_diversity | T0 | SKNN | 1982 | 79.12 | [75.22, 83.13] |
| category_diversity | T0 | LAST | 1982 | 53.18 | [49.99, 56.44] |
| category_diversity | T0 | POP | 1982 | 5.75 | [4.29, 7.29] |
| category_diversity | T1 | C_BIG | 13446 | 100.00 | [100.00, 100.00] |
| category_diversity | T1 | C_FULL | 13446 | 118.82 | [117.75, 119.94] |
| category_diversity | T1 | C_SMALL | 13446 | 95.34 | [94.49, 96.18] |
| category_diversity | T1 | FA | 13446 | 85.89 | [84.90, 86.84] |
| category_diversity | T1 | FP | 13446 | 85.89 | [84.89, 86.83] |
| category_diversity | T1 | PF | 13446 | 85.87 | [84.87, 86.82] |
| category_diversity | T1 | FA_1024 | 13446 | 25.28 | [24.30, 26.30] |
| category_diversity | T1 | DP8 | 13446 | 24.88 | [23.90, 25.88] |
| category_diversity | T1 | DP1 | 13446 | 23.95 | [23.02, 24.96] |
| category_diversity | T1 | FA_Q8 | 13446 | 85.73 | [84.74, 86.63] |
| category_diversity | T1 | FA_WARM | 13446 | 104.81 | [103.71, 105.91] |
| category_diversity | T1 | FA_WARM_FROZEN | 13446 | 105.39 | [104.30, 106.45] |
| category_diversity | T1 | DP8_FROZEN | 13446 | 105.80 | [104.72, 106.88] |
| category_diversity | T1 | FP_WARM_FROZEN | 13446 | 105.40 | [104.31, 106.45] |
| category_diversity | T1 | PF_WARM_FROZEN | 13446 | 105.35 | [104.24, 106.46] |
| category_diversity | T1 | FA_Q8_WARM_FROZEN | 13446 | 105.35 | [104.24, 106.47] |
| category_diversity | T1 | FA_1024_WARM_FROZEN | 13446 | 105.81 | [104.73, 106.90] |
| category_diversity | T1 | DP1_WARM_FROZEN | 13446 | 106.28 | [105.05, 107.54] |
| category_diversity | T1 | PRE | 13446 | 106.35 | [105.14, 107.59] |
| category_diversity | T1 | FT_C | 13446 | 97.57 | [96.72, 98.37] |
| category_diversity | T1 | FT_PRE | 13446 | 108.00 | [106.84, 109.11] |
| category_diversity | T1 | FT_FA | 13446 | 89.53 | [88.62, 90.49] |
| category_diversity | T1 | FT_FA_WARM | 13446 | 107.17 | [106.04, 108.24] |
| category_diversity | T1 | FT_FA_WARM_FROZEN | 13446 | 107.67 | [106.55, 108.78] |
| category_diversity | T1 | LO | 13446 | 2.77 | [2.45, 3.14] |
| category_diversity | T1 | KNN_EMB_SMALL | 13446 | 48.24 | [47.05, 49.52] |
| category_diversity | T1 | KNN_EMB_BIG | 13446 | 53.44 | [52.16, 54.70] |
| category_diversity | T1 | RULE | 13446 | 26.22 | [25.23, 27.14] |
| category_diversity | T1 | SKNN | 13446 | 67.47 | [66.49, 68.50] |
| category_diversity | T1 | LAST | 13446 | 43.79 | [42.87, 44.68] |
| category_diversity | T1 | POP | 13446 | 14.14 | [13.42, 14.88] |
| category_diversity | T2 | C_BIG | 25336 | 100.00 | [100.00, 100.00] |
| category_diversity | T2 | C_FULL | 25336 | 121.20 | [120.33, 122.01] |
| category_diversity | T2 | C_SMALL | 25336 | 96.21 | [95.54, 96.78] |
| category_diversity | T2 | FA | 25336 | 86.63 | [85.84, 87.31] |
| category_diversity | T2 | FP | 25336 | 86.63 | [85.83, 87.31] |
| category_diversity | T2 | PF | 25336 | 86.68 | [85.90, 87.34] |
| category_diversity | T2 | FA_1024 | 25336 | 20.07 | [19.42, 20.70] |
| category_diversity | T2 | DP8 | 25336 | 19.85 | [19.18, 20.49] |
| category_diversity | T2 | DP1 | 25336 | 19.48 | [18.82, 20.14] |
| category_diversity | T2 | FA_Q8 | 25336 | 86.45 | [85.68, 87.16] |
| category_diversity | T2 | FA_WARM | 25336 | 107.13 | [106.32, 107.94] |
| category_diversity | T2 | FA_WARM_FROZEN | 25336 | 107.63 | [106.81, 108.47] |
| category_diversity | T2 | DP8_FROZEN | 25336 | 107.62 | [106.78, 108.47] |
| category_diversity | T2 | FP_WARM_FROZEN | 25336 | 107.63 | [106.81, 108.48] |
| category_diversity | T2 | PF_WARM_FROZEN | 25336 | 107.61 | [106.79, 108.46] |
| category_diversity | T2 | FA_Q8_WARM_FROZEN | 25336 | 107.61 | [106.80, 108.47] |
| category_diversity | T2 | FA_1024_WARM_FROZEN | 25336 | 107.63 | [106.78, 108.47] |
| category_diversity | T2 | DP1_WARM_FROZEN | 25336 | 107.34 | [106.39, 108.27] |
| category_diversity | T2 | PRE | 25336 | 107.48 | [106.54, 108.41] |
| category_diversity | T2 | FT_C | 25336 | 98.72 | [98.08, 99.34] |
| category_diversity | T2 | FT_PRE | 25336 | 110.43 | [109.54, 111.33] |
| category_diversity | T2 | FT_FA | 25336 | 90.89 | [90.13, 91.56] |
| category_diversity | T2 | FT_FA_WARM | 25336 | 109.82 | [109.02, 110.73] |
| category_diversity | T2 | FT_FA_WARM_FROZEN | 25336 | 110.22 | [109.37, 111.13] |
| category_diversity | T2 | LO | 25336 | 3.37 | [3.11, 3.64] |
| category_diversity | T2 | KNN_EMB_SMALL | 25336 | 43.17 | [42.28, 44.05] |
| category_diversity | T2 | KNN_EMB_BIG | 25336 | 48.57 | [47.63, 49.42] |
| category_diversity | T2 | RULE | 25336 | 21.61 | [21.00, 22.25] |
| category_diversity | T2 | SKNN | 25336 | 69.49 | [68.72, 70.25] |
| category_diversity | T2 | LAST | 25336 | 43.66 | [42.99, 44.40] |
| category_diversity | T2 | POP | 25336 | 11.22 | [10.70, 11.73] |
| brand_loyalty | T0 | C_BIG | 16052 | 100.00 | [100.00, 100.00] |
| brand_loyalty | T0 | C_FULL | 16052 | 124.23 | [123.04, 125.49] |
| brand_loyalty | T0 | C_SMALL | 16052 | 96.88 | [96.08, 97.73] |
| brand_loyalty | T0 | FA | 16052 | 86.50 | [85.52, 87.48] |
| brand_loyalty | T0 | FP | 16052 | 86.49 | [85.52, 87.48] |
| brand_loyalty | T0 | PF | 16052 | 86.52 | [85.53, 87.49] |
| brand_loyalty | T0 | FA_1024 | 16052 | 16.30 | [15.56, 17.04] |
| brand_loyalty | T0 | DP8 | 16052 | 16.35 | [15.58, 17.09] |
| brand_loyalty | T0 | DP1 | 16052 | 16.34 | [15.55, 17.08] |
| brand_loyalty | T0 | FA_Q8 | 16052 | 86.15 | [85.16, 87.11] |
| brand_loyalty | T0 | FA_WARM | 16052 | 109.81 | [108.62, 111.00] |
| brand_loyalty | T0 | FA_WARM_FROZEN | 16052 | 110.62 | [109.38, 111.82] |
| brand_loyalty | T0 | DP8_FROZEN | 16052 | 110.40 | [109.18, 111.64] |
| brand_loyalty | T0 | FP_WARM_FROZEN | 16052 | 110.63 | [109.39, 111.84] |
| brand_loyalty | T0 | PF_WARM_FROZEN | 16052 | 110.55 | [109.30, 111.75] |
| brand_loyalty | T0 | FA_Q8_WARM_FROZEN | 16052 | 110.55 | [109.29, 111.73] |
| brand_loyalty | T0 | FA_1024_WARM_FROZEN | 16052 | 110.40 | [109.15, 111.64] |
| brand_loyalty | T0 | DP1_WARM_FROZEN | 16052 | 110.39 | [109.07, 111.71] |
| brand_loyalty | T0 | PRE | 16052 | 110.51 | [109.19, 111.85] |
| brand_loyalty | T0 | FT_C | 16052 | 99.60 | [98.74, 100.51] |
| brand_loyalty | T0 | FT_PRE | 16052 | 113.37 | [112.14, 114.62] |
| brand_loyalty | T0 | FT_FA | 16052 | 90.62 | [89.68, 91.59] |
| brand_loyalty | T0 | FT_FA_WARM | 16052 | 112.86 | [111.65, 114.11] |
| brand_loyalty | T0 | FT_FA_WARM_FROZEN | 16052 | 113.47 | [112.21, 114.73] |
| brand_loyalty | T0 | LO | 16052 | 2.52 | [2.25, 2.83] |
| brand_loyalty | T0 | KNN_EMB_SMALL | 16052 | 39.77 | [38.73, 40.86] |
| brand_loyalty | T0 | KNN_EMB_BIG | 16052 | 46.19 | [45.07, 47.37] |
| brand_loyalty | T0 | RULE | 16052 | 19.56 | [18.75, 20.39] |
| brand_loyalty | T0 | SKNN | 16052 | 71.68 | [70.62, 72.73] |
| brand_loyalty | T0 | LAST | 16052 | 43.32 | [42.39, 44.19] |
| brand_loyalty | T0 | POP | 16052 | 9.11 | [8.54, 9.68] |
| brand_loyalty | T1 | C_BIG | 13721 | 100.00 | [100.00, 100.00] |
| brand_loyalty | T1 | C_FULL | 13721 | 119.97 | [118.78, 121.14] |
| brand_loyalty | T1 | C_SMALL | 13721 | 94.98 | [94.14, 95.80] |
| brand_loyalty | T1 | FA | 13721 | 85.95 | [84.96, 86.87] |
| brand_loyalty | T1 | FP | 13721 | 85.96 | [84.97, 86.89] |
| brand_loyalty | T1 | PF | 13721 | 85.99 | [85.00, 86.93] |
| brand_loyalty | T1 | FA_1024 | 13721 | 21.16 | [20.27, 22.12] |
| brand_loyalty | T1 | DP8 | 13721 | 20.79 | [19.93, 21.75] |
| brand_loyalty | T1 | DP1 | 13721 | 20.67 | [19.77, 21.64] |
| brand_loyalty | T1 | FA_Q8 | 13721 | 85.69 | [84.66, 86.65] |
| brand_loyalty | T1 | FA_WARM | 13721 | 105.49 | [104.34, 106.55] |
| brand_loyalty | T1 | FA_WARM_FROZEN | 13721 | 105.96 | [104.78, 107.14] |
| brand_loyalty | T1 | DP8_FROZEN | 13721 | 106.27 | [105.08, 107.46] |
| brand_loyalty | T1 | FP_WARM_FROZEN | 13721 | 105.96 | [104.78, 107.14] |
| brand_loyalty | T1 | PF_WARM_FROZEN | 13721 | 105.98 | [104.81, 107.17] |
| brand_loyalty | T1 | FA_Q8_WARM_FROZEN | 13721 | 105.97 | [104.81, 107.16] |
| brand_loyalty | T1 | FA_1024_WARM_FROZEN | 13721 | 106.27 | [105.09, 107.46] |
| brand_loyalty | T1 | DP1_WARM_FROZEN | 13721 | 106.44 | [105.12, 107.70] |
| brand_loyalty | T1 | PRE | 13721 | 106.51 | [105.19, 107.77] |
| brand_loyalty | T1 | FT_C | 13721 | 97.34 | [96.50, 98.17] |
| brand_loyalty | T1 | FT_PRE | 13721 | 108.69 | [107.54, 109.81] |
| brand_loyalty | T1 | FT_FA | 13721 | 89.41 | [88.38, 90.36] |
| brand_loyalty | T1 | FT_FA_WARM | 13721 | 107.88 | [106.76, 109.03] |
| brand_loyalty | T1 | FT_FA_WARM_FROZEN | 13721 | 108.35 | [107.18, 109.51] |
| brand_loyalty | T1 | LO | 13721 | 2.90 | [2.58, 3.23] |
| brand_loyalty | T1 | KNN_EMB_SMALL | 13721 | 43.68 | [42.49, 44.95] |
| brand_loyalty | T1 | KNN_EMB_BIG | 13721 | 49.67 | [48.39, 51.01] |
| brand_loyalty | T1 | RULE | 13721 | 23.49 | [22.48, 24.42] |
| brand_loyalty | T1 | SKNN | 13721 | 69.20 | [68.17, 70.28] |
| brand_loyalty | T1 | LAST | 13721 | 44.04 | [43.00, 44.99] |
| brand_loyalty | T1 | POP | 13721 | 11.72 | [11.03, 12.42] |
| brand_loyalty | T2 | C_BIG | 10991 | 100.00 | [100.00, 100.00] |
| brand_loyalty | T2 | C_FULL | 10991 | 117.52 | [116.44, 118.73] |
| brand_loyalty | T2 | C_SMALL | 10991 | 95.51 | [94.60, 96.38] |
| brand_loyalty | T2 | FA | 10991 | 85.76 | [84.73, 86.77] |
| brand_loyalty | T2 | FP | 10991 | 85.75 | [84.73, 86.77] |
| brand_loyalty | T2 | PF | 10991 | 85.75 | [84.72, 86.77] |
| brand_loyalty | T2 | FA_1024 | 10991 | 27.87 | [26.76, 28.97] |
| brand_loyalty | T2 | DP8 | 10991 | 27.34 | [26.25, 28.41] |
| brand_loyalty | T2 | DP1 | 10991 | 25.79 | [24.74, 26.82] |
| brand_loyalty | T2 | FA_Q8 | 10991 | 85.92 | [84.91, 86.94] |
| brand_loyalty | T2 | FA_WARM | 10991 | 103.94 | [102.85, 105.05] |
| brand_loyalty | T2 | FA_WARM_FROZEN | 10991 | 104.24 | [103.12, 105.42] |
| brand_loyalty | T2 | DP8_FROZEN | 10991 | 104.63 | [103.52, 105.81] |
| brand_loyalty | T2 | FP_WARM_FROZEN | 10991 | 104.24 | [103.12, 105.43] |
| brand_loyalty | T2 | PF_WARM_FROZEN | 10991 | 104.23 | [103.12, 105.42] |
| brand_loyalty | T2 | FA_Q8_WARM_FROZEN | 10991 | 104.26 | [103.15, 105.45] |
| brand_loyalty | T2 | FA_1024_WARM_FROZEN | 10991 | 104.65 | [103.53, 105.83] |
| brand_loyalty | T2 | DP1_WARM_FROZEN | 10991 | 104.52 | [103.28, 105.86] |
| brand_loyalty | T2 | PRE | 10991 | 104.63 | [103.40, 105.92] |
| brand_loyalty | T2 | FT_C | 10991 | 97.64 | [96.72, 98.49] |
| brand_loyalty | T2 | FT_PRE | 10991 | 106.95 | [105.82, 108.15] |
| brand_loyalty | T2 | FT_FA | 10991 | 90.30 | [89.28, 91.30] |
| brand_loyalty | T2 | FT_FA_WARM | 10991 | 106.27 | [105.13, 107.39] |
| brand_loyalty | T2 | FT_FA_WARM_FROZEN | 10991 | 106.49 | [105.31, 107.67] |
| brand_loyalty | T2 | LO | 10991 | 3.92 | [3.50, 4.40] |
| brand_loyalty | T2 | KNN_EMB_SMALL | 10991 | 52.61 | [51.23, 53.97] |
| brand_loyalty | T2 | KNN_EMB_BIG | 10991 | 56.95 | [55.53, 58.33] |
| brand_loyalty | T2 | RULE | 10991 | 27.28 | [26.30, 28.31] |
| brand_loyalty | T2 | SKNN | 10991 | 66.49 | [65.45, 67.53] |
| brand_loyalty | T2 | LAST | 10991 | 45.02 | [44.12, 46.03] |
| brand_loyalty | T2 | POP | 10991 | 15.48 | [14.60, 16.33] |
| persona | d0_l0 | C_BIG | 445 | 100.00 | [100.00, 100.00] |
| persona | d0_l0 | C_FULL | 445 | 131.99 | [122.47, 143.73] |
| persona | d0_l0 | C_SMALL | 445 | 95.23 | [89.56, 101.59] |
| persona | d0_l0 | FA | 445 | 80.42 | [73.25, 88.23] |
| persona | d0_l0 | FP | 445 | 80.40 | [73.22, 88.22] |
| persona | d0_l0 | PF | 445 | 80.48 | [73.28, 88.26] |
| persona | d0_l0 | FA_1024 | 445 | 10.66 | [7.24, 14.59] |
| persona | d0_l0 | DP8 | 445 | 10.70 | [7.41, 14.40] |
| persona | d0_l0 | DP1 | 445 | 12.50 | [8.69, 16.95] |
| persona | d0_l0 | FA_Q8 | 445 | 80.56 | [73.22, 88.26] |
| persona | d0_l0 | FA_WARM | 445 | 116.00 | [107.42, 126.73] |
| persona | d0_l0 | FA_WARM_FROZEN | 445 | 119.40 | [109.90, 130.82] |
| persona | d0_l0 | DP8_FROZEN | 445 | 121.65 | [112.04, 133.03] |
| persona | d0_l0 | FP_WARM_FROZEN | 445 | 119.40 | [109.90, 130.82] |
| persona | d0_l0 | PF_WARM_FROZEN | 445 | 119.50 | [110.22, 130.76] |
| persona | d0_l0 | FA_Q8_WARM_FROZEN | 445 | 119.50 | [110.18, 130.77] |
| persona | d0_l0 | FA_1024_WARM_FROZEN | 445 | 121.80 | [112.04, 133.17] |
| persona | d0_l0 | DP1_WARM_FROZEN | 445 | 121.48 | [110.81, 133.95] |
| persona | d0_l0 | PRE | 445 | 119.24 | [109.28, 131.14] |
| persona | d0_l0 | FT_C | 445 | 97.03 | [91.93, 103.05] |
| persona | d0_l0 | FT_PRE | 445 | 119.92 | [110.89, 131.55] |
| persona | d0_l0 | FT_FA | 445 | 85.19 | [78.03, 92.98] |
| persona | d0_l0 | FT_FA_WARM | 445 | 117.73 | [108.96, 128.92] |
| persona | d0_l0 | FT_FA_WARM_FROZEN | 445 | 120.99 | [111.85, 132.76] |
| persona | d0_l0 | LO | 445 | 1.89 | [0.72, 3.37] |
| persona | d0_l0 | KNN_EMB_SMALL | 445 | 42.29 | [36.80, 48.53] |
| persona | d0_l0 | KNN_EMB_BIG | 445 | 57.42 | [49.77, 65.82] |
| persona | d0_l0 | RULE | 445 | 20.23 | [15.51, 25.76] |
| persona | d0_l0 | SKNN | 445 | 78.41 | [70.87, 86.28] |
| persona | d0_l0 | LAST | 445 | 50.95 | [44.35, 57.92] |
| persona | d0_l0 | POP | 445 | 5.41 | [3.20, 8.45] |
| persona | d0_l1 | C_BIG | 697 | 100.00 | [100.00, 100.00] |
| persona | d0_l1 | C_FULL | 697 | 123.38 | [117.18, 130.17] |
| persona | d0_l1 | C_SMALL | 697 | 91.74 | [87.75, 95.62] |
| persona | d0_l1 | FA | 697 | 80.39 | [74.99, 85.78] |
| persona | d0_l1 | FP | 697 | 80.38 | [74.99, 85.78] |
| persona | d0_l1 | PF | 697 | 80.38 | [75.00, 85.67] |
| persona | d0_l1 | FA_1024 | 697 | 17.45 | [13.51, 22.03] |
| persona | d0_l1 | DP8 | 697 | 17.08 | [13.49, 21.22] |
| persona | d0_l1 | DP1 | 697 | 17.61 | [14.09, 21.56] |
| persona | d0_l1 | FA_Q8 | 697 | 80.16 | [74.78, 85.53] |
| persona | d0_l1 | FA_WARM | 697 | 110.21 | [104.05, 117.00] |
| persona | d0_l1 | FA_WARM_FROZEN | 697 | 107.66 | [101.67, 114.50] |
| persona | d0_l1 | DP8_FROZEN | 697 | 107.80 | [101.39, 115.17] |
| persona | d0_l1 | FP_WARM_FROZEN | 697 | 107.66 | [101.67, 114.49] |
| persona | d0_l1 | PF_WARM_FROZEN | 697 | 108.60 | [102.40, 115.51] |
| persona | d0_l1 | FA_Q8_WARM_FROZEN | 697 | 108.59 | [102.43, 115.51] |
| persona | d0_l1 | FA_1024_WARM_FROZEN | 697 | 107.74 | [101.33, 115.05] |
| persona | d0_l1 | DP1_WARM_FROZEN | 697 | 110.35 | [103.20, 118.56] |
| persona | d0_l1 | PRE | 697 | 110.41 | [103.10, 118.67] |
| persona | d0_l1 | FT_C | 697 | 95.82 | [92.05, 99.68] |
| persona | d0_l1 | FT_PRE | 697 | 112.13 | [105.60, 119.47] |
| persona | d0_l1 | FT_FA | 697 | 84.17 | [79.20, 89.69] |
| persona | d0_l1 | FT_FA_WARM | 697 | 113.76 | [107.31, 121.27] |
| persona | d0_l1 | FT_FA_WARM_FROZEN | 697 | 112.08 | [105.63, 119.23] |
| persona | d0_l1 | LO | 697 | 1.81 | [0.86, 3.06] |
| persona | d0_l1 | KNN_EMB_SMALL | 697 | 49.63 | [42.45, 57.31] |
| persona | d0_l1 | KNN_EMB_BIG | 697 | 61.61 | [53.92, 69.52] |
| persona | d0_l1 | RULE | 697 | 23.87 | [19.24, 29.09] |
| persona | d0_l1 | SKNN | 697 | 75.87 | [70.02, 82.14] |
| persona | d0_l1 | LAST | 697 | 50.30 | [45.35, 55.66] |
| persona | d0_l1 | POP | 697 | 7.05 | [4.30, 10.60] |
| persona | d0_l2 | C_BIG | 840 | 100.00 | [100.00, 100.00] |
| persona | d0_l2 | C_FULL | 840 | 130.25 | [123.45, 137.29] |
| persona | d0_l2 | C_SMALL | 840 | 95.35 | [91.42, 99.31] |
| persona | d0_l2 | FA | 840 | 79.03 | [74.24, 84.23] |
| persona | d0_l2 | FP | 840 | 79.01 | [74.21, 84.22] |
| persona | d0_l2 | PF | 840 | 79.01 | [74.23, 84.19] |
| persona | d0_l2 | FA_1024 | 840 | 11.41 | [7.85, 15.49] |
| persona | d0_l2 | DP8 | 840 | 11.57 | [8.13, 15.59] |
| persona | d0_l2 | DP1 | 840 | 11.50 | [8.04, 15.58] |
| persona | d0_l2 | FA_Q8 | 840 | 79.76 | [74.69, 84.76] |
| persona | d0_l2 | FA_WARM | 840 | 109.30 | [103.03, 115.61] |
| persona | d0_l2 | FA_WARM_FROZEN | 840 | 111.19 | [104.81, 117.85] |
| persona | d0_l2 | DP8_FROZEN | 840 | 110.47 | [104.02, 116.73] |
| persona | d0_l2 | FP_WARM_FROZEN | 840 | 111.19 | [104.81, 117.85] |
| persona | d0_l2 | PF_WARM_FROZEN | 840 | 110.75 | [104.50, 117.42] |
| persona | d0_l2 | FA_Q8_WARM_FROZEN | 840 | 110.86 | [104.57, 117.49] |
| persona | d0_l2 | FA_1024_WARM_FROZEN | 840 | 110.49 | [104.05, 116.67] |
| persona | d0_l2 | DP1_WARM_FROZEN | 840 | 108.73 | [101.97, 115.99] |
| persona | d0_l2 | PRE | 840 | 109.27 | [102.58, 116.40] |
| persona | d0_l2 | FT_C | 840 | 96.52 | [92.80, 100.26] |
| persona | d0_l2 | FT_PRE | 840 | 113.66 | [107.25, 120.16] |
| persona | d0_l2 | FT_FA | 840 | 83.56 | [79.10, 88.67] |
| persona | d0_l2 | FT_FA_WARM | 840 | 113.08 | [106.94, 119.69] |
| persona | d0_l2 | FT_FA_WARM_FROZEN | 840 | 113.80 | [107.54, 120.10] |
| persona | d0_l2 | LO | 840 | 2.63 | [1.36, 4.12] |
| persona | d0_l2 | KNN_EMB_SMALL | 840 | 51.14 | [44.61, 58.78] |
| persona | d0_l2 | KNN_EMB_BIG | 840 | 61.40 | [54.48, 69.12] |
| persona | d0_l2 | RULE | 840 | 26.05 | [21.51, 31.07] |
| persona | d0_l2 | SKNN | 840 | 82.49 | [76.72, 88.97] |
| persona | d0_l2 | LAST | 840 | 56.99 | [51.72, 62.49] |
| persona | d0_l2 | POP | 840 | 4.75 | [3.01, 6.87] |
| persona | d1_l0 | C_BIG | 2904 | 100.00 | [100.00, 100.00] |
| persona | d1_l0 | C_FULL | 2904 | 123.25 | [120.19, 126.30] |
| persona | d1_l0 | C_SMALL | 2904 | 97.29 | [95.36, 99.08] |
| persona | d1_l0 | FA | 2904 | 85.63 | [83.34, 87.89] |
| persona | d1_l0 | FP | 2904 | 85.64 | [83.33, 87.89] |
| persona | d1_l0 | PF | 2904 | 85.64 | [83.33, 87.89] |
| persona | d1_l0 | FA_1024 | 2904 | 16.89 | [14.88, 18.99] |
| persona | d1_l0 | DP8 | 2904 | 16.96 | [14.96, 19.15] |
| persona | d1_l0 | DP1 | 2904 | 17.14 | [15.15, 19.28] |
| persona | d1_l0 | FA_Q8 | 2904 | 85.22 | [82.72, 87.46] |
| persona | d1_l0 | FA_WARM | 2904 | 110.94 | [107.93, 113.98] |
| persona | d1_l0 | FA_WARM_FROZEN | 2904 | 111.52 | [108.52, 114.53] |
| persona | d1_l0 | DP8_FROZEN | 2904 | 111.32 | [108.20, 114.45] |
| persona | d1_l0 | FP_WARM_FROZEN | 2904 | 111.54 | [108.54, 114.57] |
| persona | d1_l0 | PF_WARM_FROZEN | 2904 | 111.57 | [108.60, 114.58] |
| persona | d1_l0 | FA_Q8_WARM_FROZEN | 2904 | 111.56 | [108.55, 114.62] |
| persona | d1_l0 | FA_1024_WARM_FROZEN | 2904 | 111.31 | [108.20, 114.44] |
| persona | d1_l0 | DP1_WARM_FROZEN | 2904 | 112.84 | [109.31, 116.41] |
| persona | d1_l0 | PRE | 2904 | 113.05 | [109.55, 116.67] |
| persona | d1_l0 | FT_C | 2904 | 100.12 | [97.98, 101.90] |
| persona | d1_l0 | FT_PRE | 2904 | 115.22 | [112.21, 118.20] |
| persona | d1_l0 | FT_FA | 2904 | 89.65 | [87.32, 91.84] |
| persona | d1_l0 | FT_FA_WARM | 2904 | 113.91 | [111.01, 117.02] |
| persona | d1_l0 | FT_FA_WARM_FROZEN | 2904 | 114.95 | [111.92, 118.11] |
| persona | d1_l0 | LO | 2904 | 2.72 | [2.00, 3.59] |
| persona | d1_l0 | KNN_EMB_SMALL | 2904 | 40.98 | [38.23, 43.66] |
| persona | d1_l0 | KNN_EMB_BIG | 2904 | 47.64 | [44.66, 50.43] |
| persona | d1_l0 | RULE | 2904 | 21.56 | [19.58, 24.00] |
| persona | d1_l0 | SKNN | 2904 | 71.08 | [68.72, 73.56] |
| persona | d1_l0 | LAST | 2904 | 43.19 | [41.00, 45.45] |
| persona | d1_l0 | POP | 2904 | 8.91 | [7.50, 10.39] |
| persona | d1_l1 | C_BIG | 4660 | 100.00 | [100.00, 100.00] |
| persona | d1_l1 | C_FULL | 4660 | 120.52 | [118.41, 122.61] |
| persona | d1_l1 | C_SMALL | 4660 | 94.19 | [92.75, 95.63] |
| persona | d1_l1 | FA | 4660 | 84.94 | [83.13, 86.67] |
| persona | d1_l1 | FP | 4660 | 84.95 | [83.14, 86.67] |
| persona | d1_l1 | PF | 4660 | 84.91 | [83.11, 86.63] |
| persona | d1_l1 | FA_1024 | 4660 | 22.16 | [20.51, 23.79] |
| persona | d1_l1 | DP8 | 4660 | 21.74 | [20.12, 23.33] |
| persona | d1_l1 | DP1 | 4660 | 21.65 | [19.93, 23.35] |
| persona | d1_l1 | FA_Q8 | 4660 | 84.68 | [82.92, 86.40] |
| persona | d1_l1 | FA_WARM | 4660 | 103.62 | [101.66, 105.47] |
| persona | d1_l1 | FA_WARM_FROZEN | 4660 | 104.20 | [102.18, 106.22] |
| persona | d1_l1 | DP8_FROZEN | 4660 | 105.02 | [103.09, 107.03] |
| persona | d1_l1 | FP_WARM_FROZEN | 4660 | 104.20 | [102.17, 106.22] |
| persona | d1_l1 | PF_WARM_FROZEN | 4660 | 104.17 | [102.14, 106.13] |
| persona | d1_l1 | FA_Q8_WARM_FROZEN | 4660 | 104.17 | [102.15, 106.13] |
| persona | d1_l1 | FA_1024_WARM_FROZEN | 4660 | 105.04 | [103.10, 107.06] |
| persona | d1_l1 | DP1_WARM_FROZEN | 4660 | 105.57 | [103.47, 107.82] |
| persona | d1_l1 | PRE | 4660 | 105.67 | [103.49, 107.90] |
| persona | d1_l1 | FT_C | 4660 | 96.47 | [95.03, 97.92] |
| persona | d1_l1 | FT_PRE | 4660 | 107.53 | [105.63, 109.48] |
| persona | d1_l1 | FT_FA | 4660 | 87.87 | [86.17, 89.43] |
| persona | d1_l1 | FT_FA_WARM | 4660 | 106.53 | [104.67, 108.52] |
| persona | d1_l1 | FT_FA_WARM_FROZEN | 4660 | 106.97 | [104.98, 108.93] |
| persona | d1_l1 | LO | 4660 | 2.34 | [1.84, 2.90] |
| persona | d1_l1 | KNN_EMB_SMALL | 4660 | 43.82 | [41.70, 45.98] |
| persona | d1_l1 | KNN_EMB_BIG | 4660 | 49.60 | [47.28, 51.83] |
| persona | d1_l1 | RULE | 4660 | 24.79 | [23.27, 26.52] |
| persona | d1_l1 | SKNN | 4660 | 68.82 | [66.99, 70.56] |
| persona | d1_l1 | LAST | 4660 | 43.84 | [42.07, 45.58] |
| persona | d1_l1 | POP | 4660 | 12.31 | [11.05, 13.57] |
| persona | d1_l2 | C_BIG | 5882 | 100.00 | [100.00, 100.00] |
| persona | d1_l2 | C_FULL | 5882 | 116.20 | [114.79, 117.80] |
| persona | d1_l2 | C_SMALL | 5882 | 95.40 | [94.29, 96.51] |
| persona | d1_l2 | FA | 5882 | 86.60 | [85.28, 87.92] |
| persona | d1_l2 | FP | 5882 | 86.59 | [85.27, 87.90] |
| persona | d1_l2 | PF | 5882 | 86.57 | [85.26, 87.91] |
| persona | d1_l2 | FA_1024 | 5882 | 30.21 | [28.79, 31.78] |
| persona | d1_l2 | DP8 | 5882 | 29.65 | [28.21, 31.19] |
| persona | d1_l2 | DP1 | 5882 | 27.75 | [26.31, 29.25] |
| persona | d1_l2 | FA_Q8 | 5882 | 86.57 | [85.22, 88.00] |
| persona | d1_l2 | FA_WARM | 5882 | 103.43 | [102.00, 104.88] |
| persona | d1_l2 | FA_WARM_FROZEN | 5882 | 104.01 | [102.56, 105.54] |
| persona | d1_l2 | DP8_FROZEN | 5882 | 104.38 | [102.94, 105.99] |
| persona | d1_l2 | FP_WARM_FROZEN | 5882 | 104.02 | [102.57, 105.55] |
| persona | d1_l2 | PF_WARM_FROZEN | 5882 | 103.93 | [102.45, 105.44] |
| persona | d1_l2 | FA_Q8_WARM_FROZEN | 5882 | 103.94 | [102.47, 105.47] |
| persona | d1_l2 | FA_1024_WARM_FROZEN | 5882 | 104.38 | [102.93, 105.98] |
| persona | d1_l2 | DP1_WARM_FROZEN | 5882 | 104.47 | [102.93, 106.22] |
| persona | d1_l2 | PRE | 5882 | 104.48 | [102.92, 106.22] |
| persona | d1_l2 | FT_C | 5882 | 97.38 | [96.30, 98.58] |
| persona | d1_l2 | FT_PRE | 5882 | 105.78 | [104.30, 107.36] |
| persona | d1_l2 | FT_FA | 5882 | 90.54 | [89.24, 91.93] |
| persona | d1_l2 | FT_FA_WARM | 5882 | 105.22 | [103.81, 106.77] |
| persona | d1_l2 | FT_FA_WARM_FROZEN | 5882 | 105.59 | [104.09, 107.18] |
| persona | d1_l2 | LO | 5882 | 3.06 | [2.56, 3.63] |
| persona | d1_l2 | KNN_EMB_SMALL | 5882 | 53.59 | [51.86, 55.44] |
| persona | d1_l2 | KNN_EMB_BIG | 5882 | 57.91 | [56.14, 59.79] |
| persona | d1_l2 | RULE | 5882 | 28.75 | [27.42, 30.20] |
| persona | d1_l2 | SKNN | 5882 | 65.36 | [63.99, 66.82] |
| persona | d1_l2 | LAST | 5882 | 43.97 | [42.81, 45.26] |
| persona | d1_l2 | POP | 5882 | 17.14 | [16.00, 18.45] |
| persona | d2_l0 | C_BIG | 12703 | 100.00 | [100.00, 100.00] |
| persona | d2_l0 | C_FULL | 12703 | 124.20 | [122.85, 125.56] |
| persona | d2_l0 | C_SMALL | 12703 | 96.85 | [95.90, 97.76] |
| persona | d2_l0 | FA | 12703 | 86.92 | [85.91, 87.98] |
| persona | d2_l0 | FP | 12703 | 86.91 | [85.90, 87.98] |
| persona | d2_l0 | PF | 12703 | 86.94 | [85.90, 88.01] |
| persona | d2_l0 | FA_1024 | 12703 | 16.36 | [15.52, 17.24] |
| persona | d2_l0 | DP8 | 12703 | 16.40 | [15.54, 17.23] |
| persona | d2_l0 | DP1 | 12703 | 16.28 | [15.34, 17.15] |
| persona | d2_l0 | FA_Q8 | 12703 | 86.57 | [85.47, 87.63] |
| persona | d2_l0 | FA_WARM | 12703 | 109.33 | [108.16, 110.60] |
| persona | d2_l0 | FA_WARM_FROZEN | 12703 | 110.11 | [108.80, 111.44] |
| persona | d2_l0 | DP8_FROZEN | 12703 | 109.81 | [108.48, 111.12] |
| persona | d2_l0 | FP_WARM_FROZEN | 12703 | 110.11 | [108.81, 111.44] |
| persona | d2_l0 | PF_WARM_FROZEN | 12703 | 110.00 | [108.71, 111.31] |
| persona | d2_l0 | FA_Q8_WARM_FROZEN | 12703 | 110.01 | [108.70, 111.32] |
| persona | d2_l0 | FA_1024_WARM_FROZEN | 12703 | 109.79 | [108.48, 111.12] |
| persona | d2_l0 | DP1_WARM_FROZEN | 12703 | 109.43 | [107.88, 110.86] |
| persona | d2_l0 | PRE | 12703 | 109.61 | [108.07, 111.06] |
| persona | d2_l0 | FT_C | 12703 | 99.56 | [98.63, 100.48] |
| persona | d2_l0 | FT_PRE | 12703 | 112.71 | [111.41, 114.08] |
| persona | d2_l0 | FT_FA | 12703 | 91.05 | [89.98, 92.11] |
| persona | d2_l0 | FT_FA_WARM | 12703 | 112.45 | [111.15, 113.73] |
| persona | d2_l0 | FT_FA_WARM_FROZEN | 12703 | 112.86 | [111.49, 114.21] |
| persona | d2_l0 | LO | 12703 | 2.50 | [2.21, 2.82] |
| persona | d2_l0 | KNN_EMB_SMALL | 12703 | 39.39 | [38.15, 40.64] |
| persona | d2_l0 | KNN_EMB_BIG | 12703 | 45.46 | [44.14, 46.86] |
| persona | d2_l0 | RULE | 12703 | 19.05 | [18.24, 19.95] |
| persona | d2_l0 | SKNN | 12703 | 71.60 | [70.39, 72.73] |
| persona | d2_l0 | LAST | 12703 | 43.09 | [42.03, 44.16] |
| persona | d2_l0 | POP | 12703 | 9.29 | [8.56, 9.97] |
| persona | d2_l1 | C_BIG | 8364 | 100.00 | [100.00, 100.00] |
| persona | d2_l1 | C_FULL | 8364 | 119.41 | [117.98, 120.72] |
| persona | d2_l1 | C_SMALL | 8364 | 95.69 | [94.72, 96.66] |
| persona | d2_l1 | FA | 8364 | 86.95 | [85.72, 88.20] |
| persona | d2_l1 | FP | 8364 | 86.96 | [85.73, 88.21] |
| persona | d2_l1 | PF | 8364 | 87.04 | [85.79, 88.26] |
| persona | d2_l1 | FA_1024 | 8364 | 20.88 | [19.75, 22.00] |
| persona | d2_l1 | DP8 | 8364 | 20.53 | [19.43, 21.68] |
| persona | d2_l1 | DP1 | 8364 | 20.35 | [19.23, 21.52] |
| persona | d2_l1 | FA_Q8 | 8364 | 86.69 | [85.41, 87.94] |
| persona | d2_l1 | FA_WARM | 8364 | 106.19 | [104.89, 107.56] |
| persona | d2_l1 | FA_WARM_FROZEN | 8364 | 106.83 | [105.36, 108.26] |
| persona | d2_l1 | DP8_FROZEN | 8364 | 106.86 | [105.43, 108.27] |
| persona | d2_l1 | FP_WARM_FROZEN | 8364 | 106.83 | [105.36, 108.26] |
| persona | d2_l1 | PF_WARM_FROZEN | 8364 | 106.81 | [105.34, 108.23] |
| persona | d2_l1 | FA_Q8_WARM_FROZEN | 8364 | 106.80 | [105.33, 108.21] |
| persona | d2_l1 | FA_1024_WARM_FROZEN | 8364 | 106.87 | [105.45, 108.29] |
| persona | d2_l1 | DP1_WARM_FROZEN | 8364 | 106.63 | [104.90, 108.21] |
| persona | d2_l1 | PRE | 8364 | 106.69 | [104.97, 108.25] |
| persona | d2_l1 | FT_C | 8364 | 97.96 | [97.01, 99.00] |
| persona | d2_l1 | FT_PRE | 8364 | 109.09 | [107.61, 110.54] |
| persona | d2_l1 | FT_FA | 8364 | 90.70 | [89.44, 91.98] |
| persona | d2_l1 | FT_FA_WARM | 8364 | 108.20 | [106.81, 109.65] |
| persona | d2_l1 | FT_FA_WARM_FROZEN | 8364 | 108.86 | [107.43, 110.43] |
| persona | d2_l1 | LO | 8364 | 3.30 | [2.90, 3.73] |
| persona | d2_l1 | KNN_EMB_SMALL | 8364 | 43.15 | [41.70, 44.64] |
| persona | d2_l1 | KNN_EMB_BIG | 8364 | 48.80 | [47.25, 50.47] |
| persona | d2_l1 | RULE | 8364 | 22.72 | [21.66, 23.88] |
| persona | d2_l1 | SKNN | 8364 | 68.91 | [67.69, 70.04] |
| persona | d2_l1 | LAST | 8364 | 43.67 | [42.52, 44.81] |
| persona | d2_l1 | POP | 8364 | 11.75 | [10.92, 12.62] |
| persona | d2_l2 | C_BIG | 4269 | 100.00 | [100.00, 100.00] |
| persona | d2_l2 | C_FULL | 4269 | 117.69 | [115.80, 119.64] |
| persona | d2_l2 | C_SMALL | 4269 | 95.69 | [94.28, 97.07] |
| persona | d2_l2 | FA | 4269 | 85.48 | [83.94, 87.08] |
| persona | d2_l2 | FP | 4269 | 85.47 | [83.93, 87.08] |
| persona | d2_l2 | PF | 4269 | 85.50 | [83.99, 87.10] |
| persona | d2_l2 | FA_1024 | 4269 | 26.74 | [24.82, 28.57] |
| persona | d2_l2 | DP8 | 4269 | 26.14 | [24.25, 27.88] |
| persona | d2_l2 | DP1 | 4269 | 24.94 | [23.10, 26.79] |
| persona | d2_l2 | FA_Q8 | 4269 | 85.82 | [84.18, 87.41] |
| persona | d2_l2 | FA_WARM | 4269 | 103.95 | [102.14, 105.75] |
| persona | d2_l2 | FA_WARM_FROZEN | 4269 | 103.60 | [101.79, 105.42] |
| persona | d2_l2 | DP8_FROZEN | 4269 | 104.18 | [102.33, 106.02] |
| persona | d2_l2 | FP_WARM_FROZEN | 4269 | 103.60 | [101.79, 105.42] |
| persona | d2_l2 | PF_WARM_FROZEN | 4269 | 103.76 | [102.02, 105.59] |
| persona | d2_l2 | FA_Q8_WARM_FROZEN | 4269 | 103.80 | [102.08, 105.59] |
| persona | d2_l2 | FA_1024_WARM_FROZEN | 4269 | 104.23 | [102.37, 106.07] |
| persona | d2_l2 | DP1_WARM_FROZEN | 4269 | 104.00 | [101.95, 105.99] |
| persona | d2_l2 | PRE | 4269 | 104.20 | [102.19, 106.25] |
| persona | d2_l2 | FT_C | 4269 | 98.19 | [96.81, 99.47] |
| persona | d2_l2 | FT_PRE | 4269 | 107.75 | [106.02, 109.59] |
| persona | d2_l2 | FT_FA | 4269 | 90.89 | [89.33, 92.41] |
| persona | d2_l2 | FT_FA_WARM | 4269 | 106.87 | [105.07, 108.74] |
| persona | d2_l2 | FT_FA_WARM_FROZEN | 4269 | 106.78 | [105.02, 108.67] |
| persona | d2_l2 | LO | 4269 | 5.38 | [4.63, 6.24] |
| persona | d2_l2 | KNN_EMB_SMALL | 4269 | 51.38 | [49.34, 53.40] |
| persona | d2_l2 | KNN_EMB_BIG | 4269 | 54.90 | [52.76, 56.89] |
| persona | d2_l2 | RULE | 4269 | 25.27 | [23.82, 26.92] |
| persona | d2_l2 | SKNN | 4269 | 65.92 | [64.29, 67.65] |
| persona | d2_l2 | LAST | 4269 | 44.89 | [43.41, 46.50] |
| persona | d2_l2 | POP | 4269 | 14.54 | [13.29, 15.74] |
| activity | Q0 | C_BIG | 4682 | 100.00 | [100.00, 100.00] |
| activity | Q0 | C_FULL | 4682 | 120.48 | [118.39, 122.51] |
| activity | Q0 | C_SMALL | 4682 | 96.58 | [95.05, 98.09] |
| activity | Q0 | FA | 4682 | 86.68 | [84.95, 88.53] |
| activity | Q0 | FP | 4682 | 86.66 | [84.92, 88.51] |
| activity | Q0 | PF | 4682 | 86.64 | [84.89, 88.50] |
| activity | Q0 | FA_1024 | 4682 | 25.87 | [24.00, 27.75] |
| activity | Q0 | DP8 | 4682 | 25.43 | [23.65, 27.25] |
| activity | Q0 | DP1 | 4682 | 25.10 | [23.31, 27.02] |
| activity | Q0 | FA_Q8 | 4682 | 87.37 | [85.61, 89.31] |
| activity | Q0 | FA_WARM | 4682 | 108.23 | [106.25, 110.60] |
| activity | Q0 | FA_WARM_FROZEN | 4682 | 108.30 | [106.34, 110.57] |
| activity | Q0 | DP8_FROZEN | 4682 | 108.71 | [106.81, 110.92] |
| activity | Q0 | FP_WARM_FROZEN | 4682 | 108.31 | [106.35, 110.59] |
| activity | Q0 | PF_WARM_FROZEN | 4682 | 108.38 | [106.43, 110.66] |
| activity | Q0 | FA_Q8_WARM_FROZEN | 4682 | 108.41 | [106.44, 110.68] |
| activity | Q0 | FA_1024_WARM_FROZEN | 4682 | 108.75 | [106.88, 110.97] |
| activity | Q0 | DP1_WARM_FROZEN | 4682 | 109.36 | [107.03, 111.75] |
| activity | Q0 | PRE | 4682 | 109.20 | [106.87, 111.57] |
| activity | Q0 | FT_C | 4682 | 97.72 | [96.21, 99.27] |
| activity | Q0 | FT_PRE | 4682 | 109.36 | [107.29, 111.74] |
| activity | Q0 | FT_FA | 4682 | 88.44 | [86.78, 90.20] |
| activity | Q0 | FT_FA_WARM | 4682 | 108.92 | [106.86, 111.15] |
| activity | Q0 | FT_FA_WARM_FROZEN | 4682 | 108.59 | [106.47, 110.87] |
| activity | Q0 | LO | 4682 | 0.03 | [0.01, 0.05] |
| activity | Q0 | KNN_EMB_SMALL | 4682 | 45.24 | [43.12, 47.56] |
| activity | Q0 | KNN_EMB_BIG | 4682 | 49.67 | [47.23, 51.97] |
| activity | Q0 | RULE | 4682 | 28.34 | [26.63, 30.21] |
| activity | Q0 | SKNN | 4682 | 69.90 | [68.08, 71.87] |
| activity | Q0 | LAST | 4682 | 41.84 | [40.23, 43.51] |
| activity | Q0 | POP | 4682 | 14.40 | [13.18, 15.80] |
| activity | Q1 | C_BIG | 9735 | 100.00 | [100.00, 100.00] |
| activity | Q1 | C_FULL | 9735 | 119.84 | [118.41, 121.46] |
| activity | Q1 | C_SMALL | 9735 | 95.82 | [94.72, 96.81] |
| activity | Q1 | FA | 9735 | 87.77 | [86.48, 89.04] |
| activity | Q1 | FP | 9735 | 87.77 | [86.48, 89.06] |
| activity | Q1 | PF | 9735 | 87.79 | [86.48, 89.07] |
| activity | Q1 | FA_1024 | 9735 | 24.68 | [23.49, 25.91] |
| activity | Q1 | DP8 | 9735 | 24.24 | [23.04, 25.41] |
| activity | Q1 | DP1 | 9735 | 23.41 | [22.22, 24.60] |
| activity | Q1 | FA_Q8 | 9735 | 87.51 | [86.23, 88.78] |
| activity | Q1 | FA_WARM | 9735 | 106.88 | [105.46, 108.29] |
| activity | Q1 | FA_WARM_FROZEN | 9735 | 108.03 | [106.51, 109.50] |
| activity | Q1 | DP8_FROZEN | 9735 | 108.10 | [106.64, 109.59] |
| activity | Q1 | FP_WARM_FROZEN | 9735 | 108.03 | [106.52, 109.51] |
| activity | Q1 | PF_WARM_FROZEN | 9735 | 108.00 | [106.55, 109.48] |
| activity | Q1 | FA_Q8_WARM_FROZEN | 9735 | 108.04 | [106.61, 109.54] |
| activity | Q1 | FA_1024_WARM_FROZEN | 9735 | 108.08 | [106.61, 109.54] |
| activity | Q1 | DP1_WARM_FROZEN | 9735 | 107.84 | [106.28, 109.52] |
| activity | Q1 | PRE | 9735 | 107.91 | [106.35, 109.55] |
| activity | Q1 | FT_C | 9735 | 96.94 | [95.90, 97.97] |
| activity | Q1 | FT_PRE | 9735 | 109.23 | [107.74, 110.72] |
| activity | Q1 | FT_FA | 9735 | 89.62 | [88.39, 90.95] |
| activity | Q1 | FT_FA_WARM | 9735 | 108.36 | [106.86, 109.88] |
| activity | Q1 | FT_FA_WARM_FROZEN | 9735 | 109.45 | [107.88, 111.07] |
| activity | Q1 | LO | 9735 | 0.05 | [0.02, 0.09] |
| activity | Q1 | KNN_EMB_SMALL | 9735 | 45.56 | [44.08, 47.14] |
| activity | Q1 | KNN_EMB_BIG | 9735 | 49.90 | [48.34, 51.47] |
| activity | Q1 | RULE | 9735 | 25.73 | [24.62, 26.98] |
| activity | Q1 | SKNN | 9735 | 69.48 | [68.21, 70.75] |
| activity | Q1 | LAST | 9735 | 41.93 | [40.82, 43.14] |
| activity | Q1 | POP | 9735 | 13.48 | [12.51, 14.41] |
| activity | Q2 | C_BIG | 10747 | 100.00 | [100.00, 100.00] |
| activity | Q2 | C_FULL | 10747 | 119.94 | [118.60, 121.30] |
| activity | Q2 | C_SMALL | 10747 | 96.15 | [95.27, 97.07] |
| activity | Q2 | FA | 10747 | 85.71 | [84.72, 86.83] |
| activity | Q2 | FP | 10747 | 85.71 | [84.72, 86.83] |
| activity | Q2 | PF | 10747 | 85.77 | [84.78, 86.92] |
| activity | Q2 | FA_1024 | 10747 | 22.07 | [20.96, 23.15] |
| activity | Q2 | DP8 | 10747 | 21.94 | [20.82, 22.99] |
| activity | Q2 | DP1 | 10747 | 21.15 | [20.07, 22.19] |
| activity | Q2 | FA_Q8 | 10747 | 85.56 | [84.54, 86.67] |
| activity | Q2 | FA_WARM | 10747 | 105.96 | [104.72, 107.23] |
| activity | Q2 | FA_WARM_FROZEN | 10747 | 105.99 | [104.74, 107.40] |
| activity | Q2 | DP8_FROZEN | 10747 | 106.42 | [105.11, 107.81] |
| activity | Q2 | FP_WARM_FROZEN | 10747 | 106.00 | [104.75, 107.41] |
| activity | Q2 | PF_WARM_FROZEN | 10747 | 105.90 | [104.60, 107.26] |
| activity | Q2 | FA_Q8_WARM_FROZEN | 10747 | 105.89 | [104.59, 107.25] |
| activity | Q2 | FA_1024_WARM_FROZEN | 10747 | 106.43 | [105.12, 107.83] |
| activity | Q2 | DP1_WARM_FROZEN | 10747 | 106.48 | [105.04, 107.95] |
| activity | Q2 | PRE | 10747 | 106.66 | [105.22, 108.15] |
| activity | Q2 | FT_C | 10747 | 98.19 | [97.32, 99.06] |
| activity | Q2 | FT_PRE | 10747 | 109.38 | [108.04, 110.75] |
| activity | Q2 | FT_FA | 10747 | 89.39 | [88.38, 90.49] |
| activity | Q2 | FT_FA_WARM | 10747 | 108.68 | [107.38, 110.05] |
| activity | Q2 | FT_FA_WARM_FROZEN | 10747 | 108.84 | [107.52, 110.23] |
| activity | Q2 | LO | 10747 | 0.69 | [0.54, 0.86] |
| activity | Q2 | KNN_EMB_SMALL | 10747 | 44.72 | [43.38, 46.13] |
| activity | Q2 | KNN_EMB_BIG | 10747 | 50.47 | [49.09, 51.94] |
| activity | Q2 | RULE | 10747 | 22.87 | [21.78, 23.92] |
| activity | Q2 | SKNN | 10747 | 69.45 | [68.31, 70.62] |
| activity | Q2 | LAST | 10747 | 44.38 | [43.29, 45.47] |
| activity | Q2 | POP | 10747 | 12.12 | [11.29, 12.96] |
| activity | Q3 | C_BIG | 15600 | 100.00 | [100.00, 100.00] |
| activity | Q3 | C_FULL | 15600 | 121.71 | [120.62, 122.93] |
| activity | Q3 | C_SMALL | 15600 | 95.27 | [94.55, 95.99] |
| activity | Q3 | FA | 15600 | 84.99 | [84.15, 85.85] |
| activity | Q3 | FP | 15600 | 84.99 | [84.15, 85.85] |
| activity | Q3 | PF | 15600 | 85.00 | [84.16, 85.87] |
| activity | Q3 | FA_1024 | 15600 | 18.00 | [17.21, 18.73] |
| activity | Q3 | DP8 | 15600 | 17.77 | [17.01, 18.51] |
| activity | Q3 | DP1 | 15600 | 17.54 | [16.75, 18.32] |
| activity | Q3 | FA_Q8 | 15600 | 84.63 | [83.77, 85.49] |
| activity | Q3 | FA_WARM | 15600 | 105.95 | [105.02, 106.96] |
| activity | Q3 | FA_WARM_FROZEN | 15600 | 106.57 | [105.61, 107.69] |
| activity | Q3 | DP8_FROZEN | 15600 | 106.51 | [105.51, 107.61] |
| activity | Q3 | FP_WARM_FROZEN | 15600 | 106.58 | [105.61, 107.69] |
| activity | Q3 | PF_WARM_FROZEN | 15600 | 106.58 | [105.61, 107.66] |
| activity | Q3 | FA_Q8_WARM_FROZEN | 15600 | 106.56 | [105.57, 107.65] |
| activity | Q3 | FA_1024_WARM_FROZEN | 15600 | 106.52 | [105.52, 107.61] |
| activity | Q3 | DP1_WARM_FROZEN | 15600 | 106.48 | [105.37, 107.64] |
| activity | Q3 | PRE | 15600 | 106.63 | [105.54, 107.79] |
| activity | Q3 | FT_C | 15600 | 99.24 | [98.52, 100.01] |
| activity | Q3 | FT_PRE | 15600 | 110.41 | [109.33, 111.60] |
| activity | Q3 | FT_FA | 15600 | 91.52 | [90.66, 92.36] |
| activity | Q3 | FT_FA_WARM | 15600 | 109.83 | [108.80, 110.90] |
| activity | Q3 | FT_FA_WARM_FROZEN | 15600 | 110.27 | [109.19, 111.42] |
| activity | Q3 | LO | 15600 | 7.95 | [7.45, 8.52] |
| activity | Q3 | KNN_EMB_SMALL | 15600 | 45.41 | [44.31, 46.45] |
| activity | Q3 | KNN_EMB_BIG | 15600 | 52.14 | [50.93, 53.28] |
| activity | Q3 | RULE | 15600 | 20.54 | [19.73, 21.34] |
| activity | Q3 | SKNN | 15600 | 68.50 | [67.64, 69.44] |
| activity | Q3 | LAST | 15600 | 46.14 | [45.28, 47.02] |
| activity | Q3 | POP | 15600 | 10.27 | [9.62, 10.85] |

### 5e. R per support-count bin (06A §2, observational)

User counts per bin: `{"MAIN": {"11-20": 10685, "21-50": 12617, "5-10": 11648, "51-100": 5814}, "TRUE_COLD": {"0": 381295}, "UNSEEN_HOLDOUT": {"11-20": 2743, "21-50": 3149, "5-10": 2878, "51-100": 1454}}`

| role | bin | arm | study | users | R% vs C_BIG | 95% CI |
|---|---|---|---|---|---|---|
| MAIN | 11-20 | C_BIG | OTHER | 10685 | 100.00 | [100.00, 100.00] |
| MAIN | 11-20 | C_FULL | OTHER | 10685 | 120.12 | [118.77, 121.38] |
| MAIN | 11-20 | C_SMALL | A | 10685 | 95.80 | [94.79, 96.76] |
| MAIN | 11-20 | FA | A | 10685 | 86.03 | [84.86, 87.17] |
| MAIN | 11-20 | FP | OTHER | 10685 | 86.02 | [84.86, 87.16] |
| MAIN | 11-20 | PF | B | 10685 | 86.07 | [84.91, 87.22] |
| MAIN | 11-20 | FA_1024 | A | 10685 | 22.89 | [21.81, 23.93] |
| MAIN | 11-20 | DP8 | A | 10685 | 22.67 | [21.62, 23.64] |
| MAIN | 11-20 | DP1 | A | 10685 | 21.89 | [20.82, 22.95] |
| MAIN | 11-20 | FA_Q8 | OTHER | 10685 | 85.61 | [84.47, 86.75] |
| MAIN | 11-20 | FA_WARM | OTHER | 10685 | 106.04 | [104.81, 107.33] |
| MAIN | 11-20 | FA_WARM_FROZEN | OTHER | 10685 | 106.68 | [105.44, 108.06] |
| MAIN | 11-20 | DP8_FROZEN | OTHER | 10685 | 106.92 | [105.68, 108.24] |
| MAIN | 11-20 | FP_WARM_FROZEN | OTHER | 10685 | 106.69 | [105.45, 108.05] |
| MAIN | 11-20 | PF_WARM_FROZEN | OTHER | 10685 | 106.55 | [105.27, 107.93] |
| MAIN | 11-20 | FA_Q8_WARM_FROZEN | OTHER | 10685 | 106.56 | [105.28, 107.94] |
| MAIN | 11-20 | FA_1024_WARM_FROZEN | OTHER | 10685 | 106.91 | [105.68, 108.23] |
| MAIN | 11-20 | DP1_WARM_FROZEN | OTHER | 10685 | 107.08 | [105.68, 108.68] |
| MAIN | 11-20 | PRE | OTHER | 10685 | 107.25 | [105.90, 108.83] |
| MAIN | 11-20 | FT_C | B | 10685 | 97.73 | [96.77, 98.70] |
| MAIN | 11-20 | FT_PRE | OTHER | 10685 | 109.38 | [108.00, 110.71] |
| MAIN | 11-20 | FT_FA | B | 10685 | 89.28 | [88.19, 90.42] |
| MAIN | 11-20 | FT_FA_WARM | OTHER | 10685 | 108.67 | [107.35, 109.95] |
| MAIN | 11-20 | FT_FA_WARM_FROZEN | OTHER | 10685 | 109.13 | [107.74, 110.48] |
| MAIN | 11-20 | LO | B | 10685 | 0.38 | [0.28, 0.48] |
| MAIN | 11-20 | KNN_EMB_SMALL | OTHER | 10685 | 44.60 | [43.33, 45.92] |
| MAIN | 11-20 | KNN_EMB_BIG | OTHER | 10685 | 49.73 | [48.36, 51.14] |
| MAIN | 11-20 | RULE | OTHER | 10685 | 23.48 | [22.46, 24.54] |
| MAIN | 11-20 | SKNN | OTHER | 10685 | 69.46 | [68.39, 70.60] |
| MAIN | 11-20 | LAST | OTHER | 10685 | 43.55 | [42.47, 44.60] |
| MAIN | 11-20 | POP | OTHER | 10685 | 12.05 | [11.27, 12.82] |
| MAIN | 21-50 | C_BIG | OTHER | 12617 | 100.00 | [100.00, 100.00] |
| MAIN | 21-50 | C_FULL | OTHER | 12617 | 121.05 | [119.75, 122.32] |
| MAIN | 21-50 | C_SMALL | A | 12617 | 95.79 | [94.99, 96.65] |
| MAIN | 21-50 | FA | A | 12617 | 85.31 | [84.26, 86.29] |
| MAIN | 21-50 | FP | OTHER | 12617 | 85.31 | [84.26, 86.30] |
| MAIN | 21-50 | PF | B | 12617 | 85.33 | [84.27, 86.31] |
| MAIN | 21-50 | FA_1024 | A | 12617 | 19.52 | [18.61, 20.41] |
| MAIN | 21-50 | DP8 | A | 12617 | 19.41 | [18.52, 20.33] |
| MAIN | 21-50 | DP1 | A | 12617 | 19.00 | [18.10, 19.88] |
| MAIN | 21-50 | FA_Q8 | OTHER | 12617 | 85.13 | [84.04, 86.14] |
| MAIN | 21-50 | FA_WARM | OTHER | 12617 | 106.80 | [105.70, 107.90] |
| MAIN | 21-50 | FA_WARM_FROZEN | OTHER | 12617 | 107.30 | [106.13, 108.47] |
| MAIN | 21-50 | DP8_FROZEN | OTHER | 12617 | 107.38 | [106.19, 108.54] |
| MAIN | 21-50 | FP_WARM_FROZEN | OTHER | 12617 | 107.31 | [106.13, 108.48] |
| MAIN | 21-50 | PF_WARM_FROZEN | OTHER | 12617 | 107.30 | [106.09, 108.44] |
| MAIN | 21-50 | FA_Q8_WARM_FROZEN | OTHER | 12617 | 107.28 | [106.07, 108.43] |
| MAIN | 21-50 | FA_1024_WARM_FROZEN | OTHER | 12617 | 107.38 | [106.20, 108.54] |
| MAIN | 21-50 | DP1_WARM_FROZEN | OTHER | 12617 | 107.48 | [106.10, 108.74] |
| MAIN | 21-50 | PRE | OTHER | 12617 | 107.69 | [106.36, 109.03] |
| MAIN | 21-50 | FT_C | B | 12617 | 98.73 | [97.88, 99.57] |
| MAIN | 21-50 | FT_PRE | OTHER | 12617 | 110.60 | [109.38, 111.82] |
| MAIN | 21-50 | FT_FA | B | 12617 | 90.46 | [89.44, 91.45] |
| MAIN | 21-50 | FT_FA_WARM | OTHER | 12617 | 109.97 | [108.82, 111.10] |
| MAIN | 21-50 | FT_FA_WARM_FROZEN | OTHER | 12617 | 110.20 | [109.01, 111.42] |
| MAIN | 21-50 | LO | B | 12617 | 3.11 | [2.75, 3.49] |
| MAIN | 21-50 | KNN_EMB_SMALL | OTHER | 12617 | 45.52 | [44.31, 46.82] |
| MAIN | 21-50 | KNN_EMB_BIG | OTHER | 12617 | 52.18 | [50.89, 53.53] |
| MAIN | 21-50 | RULE | OTHER | 12617 | 21.20 | [20.28, 22.15] |
| MAIN | 21-50 | SKNN | OTHER | 12617 | 68.66 | [67.56, 69.72] |
| MAIN | 21-50 | LAST | OTHER | 12617 | 45.39 | [44.46, 46.42] |
| MAIN | 21-50 | POP | OTHER | 12617 | 11.36 | [10.66, 12.11] |
| MAIN | 5-10 | C_BIG | OTHER | 11648 | 100.00 | [100.00, 100.00] |
| MAIN | 5-10 | C_FULL | OTHER | 11648 | 120.03 | [118.70, 121.38] |
| MAIN | 5-10 | C_SMALL | A | 11648 | 96.33 | [95.43, 97.31] |
| MAIN | 5-10 | FA | A | 11648 | 87.51 | [86.40, 88.57] |
| MAIN | 5-10 | FP | OTHER | 11648 | 87.51 | [86.39, 88.57] |
| MAIN | 5-10 | PF | B | 11648 | 87.51 | [86.41, 88.60] |
| MAIN | 5-10 | FA_1024 | A | 11648 | 25.24 | [24.08, 26.36] |
| MAIN | 5-10 | DP8 | A | 11648 | 24.69 | [23.55, 25.81] |
| MAIN | 5-10 | DP1 | A | 11648 | 24.06 | [22.93, 25.12] |
| MAIN | 5-10 | FA_Q8 | OTHER | 11648 | 87.74 | [86.63, 88.87] |
| MAIN | 5-10 | FA_WARM | OTHER | 11648 | 107.34 | [106.12, 108.61] |
| MAIN | 5-10 | FA_WARM_FROZEN | OTHER | 11648 | 107.92 | [106.65, 109.23] |
| MAIN | 5-10 | DP8_FROZEN | OTHER | 11648 | 108.15 | [106.87, 109.44] |
| MAIN | 5-10 | FP_WARM_FROZEN | OTHER | 11648 | 107.93 | [106.66, 109.24] |
| MAIN | 5-10 | PF_WARM_FROZEN | OTHER | 11648 | 107.94 | [106.69, 109.24] |
| MAIN | 5-10 | FA_Q8_WARM_FROZEN | OTHER | 11648 | 107.99 | [106.73, 109.32] |
| MAIN | 5-10 | FA_1024_WARM_FROZEN | OTHER | 11648 | 108.16 | [106.88, 109.46] |
| MAIN | 5-10 | DP1_WARM_FROZEN | OTHER | 11648 | 108.08 | [106.57, 109.50] |
| MAIN | 5-10 | PRE | OTHER | 11648 | 108.02 | [106.54, 109.41] |
| MAIN | 5-10 | FT_C | B | 11648 | 97.38 | [96.47, 98.31] |
| MAIN | 5-10 | FT_PRE | OTHER | 11648 | 109.23 | [107.93, 110.63] |
| MAIN | 5-10 | FT_FA | B | 11648 | 89.36 | [88.27, 90.44] |
| MAIN | 5-10 | FT_FA_WARM | OTHER | 11648 | 108.43 | [107.18, 109.72] |
| MAIN | 5-10 | FT_FA_WARM_FROZEN | OTHER | 11648 | 109.06 | [107.78, 110.40] |
| MAIN | 5-10 | LO | B | 11648 | 0.05 | [0.02, 0.08] |
| MAIN | 5-10 | KNN_EMB_SMALL | OTHER | 11648 | 45.47 | [44.10, 46.83] |
| MAIN | 5-10 | KNN_EMB_BIG | OTHER | 11648 | 49.82 | [48.40, 51.27] |
| MAIN | 5-10 | RULE | OTHER | 11648 | 27.18 | [26.19, 28.15] |
| MAIN | 5-10 | SKNN | OTHER | 11648 | 69.88 | [68.72, 71.06] |
| MAIN | 5-10 | LAST | OTHER | 11648 | 41.82 | [40.73, 42.88] |
| MAIN | 5-10 | POP | OTHER | 11648 | 14.11 | [13.22, 14.96] |
| MAIN | 51-100 | C_BIG | OTHER | 5814 | 100.00 | [100.00, 100.00] |
| MAIN | 51-100 | C_FULL | OTHER | 5814 | 122.00 | [120.16, 123.67] |
| MAIN | 51-100 | C_SMALL | A | 5814 | 94.59 | [93.36, 95.78] |
| MAIN | 51-100 | FA | A | 5814 | 84.68 | [83.34, 86.19] |
| MAIN | 51-100 | FP | OTHER | 5814 | 84.68 | [83.34, 86.19] |
| MAIN | 51-100 | PF | B | 5814 | 84.69 | [83.34, 86.22] |
| MAIN | 51-100 | FA_1024 | A | 5814 | 16.18 | [14.98, 17.45] |
| MAIN | 51-100 | DP8 | A | 5814 | 15.95 | [14.75, 17.16] |
| MAIN | 51-100 | DP1 | A | 5814 | 15.82 | [14.61, 17.08] |
| MAIN | 51-100 | FA_Q8 | OTHER | 5814 | 84.25 | [82.86, 85.73] |
| MAIN | 51-100 | FA_WARM | OTHER | 5814 | 104.50 | [102.88, 106.11] |
| MAIN | 51-100 | FA_WARM_FROZEN | OTHER | 5814 | 104.76 | [103.07, 106.40] |
| MAIN | 51-100 | DP8_FROZEN | OTHER | 5814 | 104.79 | [103.08, 106.54] |
| MAIN | 51-100 | FP_WARM_FROZEN | OTHER | 5814 | 104.76 | [103.07, 106.40] |
| MAIN | 51-100 | PF_WARM_FROZEN | OTHER | 5814 | 104.83 | [103.13, 106.53] |
| MAIN | 51-100 | FA_Q8_WARM_FROZEN | OTHER | 5814 | 104.81 | [103.12, 106.48] |
| MAIN | 51-100 | FA_1024_WARM_FROZEN | OTHER | 5814 | 104.80 | [103.08, 106.54] |
| MAIN | 51-100 | DP1_WARM_FROZEN | OTHER | 5814 | 104.52 | [102.66, 106.39] |
| MAIN | 51-100 | PRE | OTHER | 5814 | 104.59 | [102.75, 106.46] |
| MAIN | 51-100 | FT_C | B | 5814 | 99.82 | [98.57, 100.96] |
| MAIN | 51-100 | FT_PRE | OTHER | 5814 | 109.49 | [107.80, 111.30] |
| MAIN | 51-100 | FT_FA | B | 5814 | 92.72 | [91.28, 94.17] |
| MAIN | 51-100 | FT_FA_WARM | OTHER | 5814 | 109.13 | [107.41, 110.87] |
| MAIN | 51-100 | FT_FA_WARM_FROZEN | OTHER | 5814 | 109.53 | [107.83, 111.28] |
| MAIN | 51-100 | LO | B | 5814 | 15.53 | [14.43, 16.78] |
| MAIN | 51-100 | KNN_EMB_SMALL | OTHER | 5814 | 45.43 | [43.70, 47.17] |
| MAIN | 51-100 | KNN_EMB_BIG | OTHER | 5814 | 52.39 | [50.55, 54.24] |
| MAIN | 51-100 | RULE | OTHER | 5814 | 19.63 | [18.46, 20.86] |
| MAIN | 51-100 | SKNN | OTHER | 5814 | 68.12 | [66.62, 69.58] |
| MAIN | 51-100 | LAST | OTHER | 5814 | 47.49 | [46.07, 48.85] |
| MAIN | 51-100 | POP | OTHER | 5814 | 9.01 | [8.08, 9.97] |
| TRUE_COLD | 0 | C_BIG | OTHER | 381295 | 100.00 | [100.00, 100.00] |
| TRUE_COLD | 0 | C_SMALL | A | 381295 | 96.12 | [95.94, 96.33] |
| TRUE_COLD | 0 | FA | A | 381295 | 86.30 | [86.08, 86.54] |
| TRUE_COLD | 0 | FA_WARM_FROZEN | OTHER | 381295 | 104.63 | [104.39, 104.89] |
| UNSEEN_HOLDOUT | 11-20 | C_BIG | OTHER | 2743 | 100.00 | [100.00, 100.00] |
| UNSEEN_HOLDOUT | 11-20 | C_FULL | OTHER | 2743 | 125.07 | [121.83, 128.55] |
| UNSEEN_HOLDOUT | 11-20 | C_SMALL | A | 2743 | 99.28 | [96.90, 101.67] |
| UNSEEN_HOLDOUT | 11-20 | FA | A | 2743 | 89.14 | [86.51, 91.79] |
| UNSEEN_HOLDOUT | 11-20 | FA_WARM_FROZEN | OTHER | 2743 | 109.59 | [107.06, 112.17] |
| UNSEEN_HOLDOUT | 11-20 | FT_C | B | 2743 | 101.54 | [99.11, 104.21] |
| UNSEEN_HOLDOUT | 11-20 | LO | B | 2743 | 0.28 | [0.13, 0.48] |
| UNSEEN_HOLDOUT | 21-50 | C_BIG | OTHER | 3149 | 100.00 | [100.00, 100.00] |
| UNSEEN_HOLDOUT | 21-50 | C_FULL | OTHER | 3149 | 127.05 | [123.98, 130.17] |
| UNSEEN_HOLDOUT | 21-50 | C_SMALL | A | 3149 | 95.59 | [93.44, 97.75] |
| UNSEEN_HOLDOUT | 21-50 | FA | A | 3149 | 88.38 | [85.93, 90.79] |
| UNSEEN_HOLDOUT | 21-50 | FA_WARM_FROZEN | OTHER | 3149 | 111.44 | [108.84, 114.19] |
| UNSEEN_HOLDOUT | 21-50 | FT_C | B | 3149 | 99.86 | [97.57, 102.04] |
| UNSEEN_HOLDOUT | 21-50 | LO | B | 3149 | 3.17 | [2.34, 4.11] |
| UNSEEN_HOLDOUT | 5-10 | C_BIG | OTHER | 2878 | 100.00 | [100.00, 100.00] |
| UNSEEN_HOLDOUT | 5-10 | C_FULL | OTHER | 2878 | 120.60 | [117.74, 123.85] |
| UNSEEN_HOLDOUT | 5-10 | C_SMALL | A | 2878 | 94.00 | [91.53, 96.43] |
| UNSEEN_HOLDOUT | 5-10 | FA | A | 2878 | 85.70 | [83.16, 88.26] |
| UNSEEN_HOLDOUT | 5-10 | FA_WARM_FROZEN | OTHER | 2878 | 105.51 | [102.88, 108.25] |
| UNSEEN_HOLDOUT | 5-10 | FT_C | B | 2878 | 95.73 | [93.34, 98.01] |
| UNSEEN_HOLDOUT | 5-10 | LO | B | 2878 | 0.12 | [0.02, 0.30] |
| UNSEEN_HOLDOUT | 51-100 | C_BIG | OTHER | 1454 | 100.00 | [100.00, 100.00] |
| UNSEEN_HOLDOUT | 51-100 | C_FULL | OTHER | 1454 | 133.61 | [128.92, 139.07] |
| UNSEEN_HOLDOUT | 51-100 | C_SMALL | A | 1454 | 96.91 | [93.65, 100.01] |
| UNSEEN_HOLDOUT | 51-100 | FA | A | 1454 | 87.13 | [83.83, 90.55] |
| UNSEEN_HOLDOUT | 51-100 | FA_WARM_FROZEN | OTHER | 1454 | 112.32 | [108.45, 116.18] |
| UNSEEN_HOLDOUT | 51-100 | FT_C | B | 1454 | 102.11 | [98.56, 105.55] |
| UNSEEN_HOLDOUT | 51-100 | LO | B | 1454 | 14.30 | [11.99, 16.82] |

Events-to-90% / 95% (smallest bin with M >= X C_ref; bin 0 from TRUE_COLD; no interpolation):

| panel | arm | seed | events-to-90% | events-to-95% |
|---|---|---|---|---|
| MAIN | C_BIG | 2026 | 0 | 0 |
| MAIN | C_BIG | 2027 | 5-10 | 5-10 |
| MAIN | C_FULL | 2026 | 5-10 | 5-10 |
| MAIN | C_FULL | 2027 | 5-10 | 5-10 |
| MAIN | C_SMALL | 2026 | 0 | 0 |
| MAIN | C_SMALL | 2027 | 5-10 | 5-10 |
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
| MAIN | FT_FA | 2026 | 21-50 | NOT_REACHED |
| MAIN | FT_FA | 2027 | 21-50 | NOT_REACHED |
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
| UNSEEN_HOLDOUT | FA | 2026 | NOT_REACHED | NOT_REACHED |
| UNSEEN_HOLDOUT | FA_WARM_FROZEN | 2026 | 0 | 0 |
| UNSEEN_HOLDOUT | FT_C | 2026 | 5-10 | 5-10 |
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
| HEAVY_PANEL | PRACTICAL_BEST | C_FULL | 132.54 [131.16, 133.88] |
| HEAVY_PANEL | PRACTICAL_BEST | C_SMALL | 95.41 [94.50, 96.31] |
| HEAVY_PANEL | PRACTICAL_BEST | FA | 85.38 [84.37, 86.41] |
| HEAVY_PANEL | PRACTICAL_BEST | FA_WARM_FROZEN | 113.40 [112.28, 114.45] |
| HEAVY_PANEL | CONTROLLED_BUDGET | C_FULL | - - |
| HEAVY_PANEL | CONTROLLED_BUDGET | DP8 | - - |
| HEAVY_PANEL | CONTROLLED_BUDGET | DP8_FROZEN | - - |
| HEAVY_PANEL | CONTROLLED_BUDGET | PRE | - - |

| panel | rule | arm | R% vs C_BIG (seed mean) |
|---|---|---|---|
| SPARSE_PANEL | PRACTICAL_BEST | C_BIG | 100.00 [100.00, 100.00] |
| SPARSE_PANEL | PRACTICAL_BEST | C_FULL | 117.56 [116.23, 118.96] |
| SPARSE_PANEL | PRACTICAL_BEST | C_SMALL | 96.82 [95.80, 97.84] |
| SPARSE_PANEL | PRACTICAL_BEST | FA | 86.89 [85.67, 88.08] |
| SPARSE_PANEL | PRACTICAL_BEST | FA_WARM_FROZEN | 104.69 [103.54, 105.87] |
| SPARSE_PANEL | CONTROLLED_BUDGET | C_FULL | - - |
| SPARSE_PANEL | CONTROLLED_BUDGET | DP8 | - - |
| SPARSE_PANEL | CONTROLLED_BUDGET | DP8_FROZEN | - - |
| SPARSE_PANEL | CONTROLLED_BUDGET | PRE | - - |

| panel | rule | arm | R% vs C_BIG (seed mean) |
|---|---|---|---|
| TRUE_COLD | PRACTICAL_BEST | C_BIG | 100.00 [100.00, 100.00] |
| TRUE_COLD | PRACTICAL_BEST | C_SMALL | 96.12 [95.94, 96.33] |
| TRUE_COLD | PRACTICAL_BEST | FA | 86.30 [86.08, 86.54] |
| TRUE_COLD | PRACTICAL_BEST | FA_WARM_FROZEN | 104.63 [104.39, 104.89] |
| TRUE_COLD | CONTROLLED_BUDGET | DP8 | - - |
| TRUE_COLD | CONTROLLED_BUDGET | DP8_FROZEN | - - |

| panel | rule | arm | R% vs C_BIG (seed mean) |
|---|---|---|---|
| UNSEEN_HOLDOUT | PRACTICAL_BEST | C_BIG | 100.00 [100.00, 100.00] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | C_FULL | 125.40 [123.69, 127.16] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | C_SMALL | 96.28 [94.94, 97.43] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | FA | 87.63 [86.32, 89.01] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | FA_WARM_FROZEN | 109.28 [107.87, 110.71] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | FT_C | 99.36 [98.06, 100.71] |
| UNSEEN_HOLDOUT | PRACTICAL_BEST | LO | 2.85 [2.46, 3.27] |
| UNSEEN_HOLDOUT | CONTROLLED_BUDGET | C_FULL | - - |
| UNSEEN_HOLDOUT | CONTROLLED_BUDGET | DP8 | - - |
| UNSEEN_HOLDOUT | CONTROLLED_BUDGET | DP8_FROZEN | - - |
| UNSEEN_HOLDOUT | CONTROLLED_BUDGET | PRE | - - |

## 7. VALIDATION -> TEST shift indicator (06B §7b) — DESCRIPTIVE

```
{
 "status": "NOT_APPLICABLE"
}
```

## 8. LAST anchor and the device-only envelope (06L-C C3) — DESCRIPTIVE (K11b is in section 1)

- anchor: `LAST|None|RULE_FREE|MAIN`; BELOW_DEVICE_HEURISTIC = the 95% CI of Δ(arm - LAST) lies below 0 (06L-C §C3).

| arm | rule | Δ(arm − LAST) [CI] | BELOW_DEVICE_HEURISTIC | R* vs C_BIG % [CI] | R* vs C_FULL % [CI] | harmed share vs LAST |
|---|---|---|---|---|---|---|
| C_BIG | PRACTICAL_BEST | 0.090097 [0.088502, 0.091595] | False | NOT_APPLICABLE | - | 0.1615 |
| C_BIG | CONTROLLED_BUDGET | 0.081613 [0.080095, 0.083042] | False | NOT_APPLICABLE | - | 0.1769 |
| C_FULL | PRACTICAL_BEST | 0.123343 [0.121723, 0.124863] | False | 136.90 [135.61, 138.27] | - | 0.1033 |
| C_FULL | CONTROLLED_BUDGET | 0.123343 [0.121723, 0.124863] | False | 151.13 [149.52, 152.95] | - | 0.1033 |
| C_SMALL | PRACTICAL_BEST | 0.083320 [0.081759, 0.084941] | False | 92.48 [91.65, 93.31] | - | 0.1867 |
| C_SMALL | CONTROLLED_BUDGET | 0.081622 [0.080045, 0.083168] | False | 100.01 [99.03, 101.08] | - | 0.1884 |
| FA | PRACTICAL_BEST | 0.067647 [0.066146, 0.069190] | False | 75.08 [74.06, 76.10] | - | 0.2222 |
| FA | CONTROLLED_BUDGET | 0.067647 [0.066146, 0.069190] | False | 82.89 [81.73, 84.10] | - | 0.2222 |
| FP | PRACTICAL_BEST | 0.067646 [0.066148, 0.069193] | False | 75.08 [74.06, 76.10] | - | 0.2221 |
| FP | CONTROLLED_BUDGET | 0.067646 [0.066148, 0.069193] | False | 82.89 [81.72, 84.10] | - | 0.2221 |
| PF | PRACTICAL_BEST | 0.067677 [0.066158, 0.069238] | False | 75.12 [74.09, 76.13] | - | 0.2223 |
| PF | CONTROLLED_BUDGET | 0.067677 [0.066158, 0.069238] | False | 82.92 [81.78, 84.14] | - | 0.2223 |
| FA_1024 | PRACTICAL_BEST | -0.036154 [-0.037425, -0.034991] | True | -40.13 [-41.85, -38.48] | - | 0.5350 |
| FA_1024 | CONTROLLED_BUDGET | -0.036154 [-0.037425, -0.034991] | True | -44.30 [-46.24, -42.46] | - | 0.5350 |
| DP8 | PRACTICAL_BEST | -0.036607 [-0.037847, -0.035461] | True | -40.63 [-42.39, -38.97] | - | 0.5363 |
| DP8 | CONTROLLED_BUDGET | -0.036607 [-0.037847, -0.035461] | True | -44.86 [-46.81, -43.01] | - | 0.5363 |
| DP1 | PRACTICAL_BEST | -0.037486 [-0.038685, -0.036320] | True | -41.61 [-43.36, -39.95] | - | 0.5375 |
| DP1 | CONTROLLED_BUDGET | -0.037486 [-0.038685, -0.036320] | True | -45.85 [-47.85, -43.98] | - | 0.5375 |
| FA_Q8 | PRACTICAL_BEST | 0.067396 [0.065868, 0.068965] | False | 74.80 [73.75, 75.88] | - | 0.2213 |
| FA_Q8 | CONTROLLED_BUDGET | 0.067396 [0.065868, 0.068965] | False | 82.58 [81.40, 83.81] | - | 0.2213 |
| FA_WARM | PRACTICAL_BEST | 0.100498 [0.098863, 0.102035] | False | 111.54 [110.39, 112.77] | 81.65 [80.81, 82.47] | 0.1480 |
| FA_WARM | CONTROLLED_BUDGET | 0.099518 [0.097854, 0.101121] | False | 121.94 [120.53, 123.46] | 80.86 [79.98, 81.70] | 0.1497 |
| FA_WARM_FROZEN | PRACTICAL_BEST | 0.101351 [0.099671, 0.103013] | False | 112.49 [111.27, 113.75] | 82.35 [81.49, 83.20] | 0.1468 |
| FA_WARM_FROZEN | CONTROLLED_BUDGET | 0.101344 [0.099594, 0.103039] | False | 124.18 [122.71, 125.77] | 82.34 [81.48, 83.22] | 0.1472 |
| DP8_FROZEN | PRACTICAL_BEST | 0.101609 [0.099946, 0.103297] | False | 112.78 [111.58, 114.04] | 82.55 [81.66, 83.42] | 0.1467 |
| DP8_FROZEN | CONTROLLED_BUDGET | 0.100869 [0.099221, 0.102567] | False | 123.60 [122.10, 125.15] | 81.95 [81.08, 82.83] | 0.1478 |
| FP_WARM_FROZEN | PRACTICAL_BEST | 0.101359 [0.099677, 0.103017] | False | 112.50 [111.28, 113.75] | 82.35 [81.49, 83.21] | 0.1468 |
| FP_WARM_FROZEN | CONTROLLED_BUDGET | 0.101342 [0.099592, 0.103036] | False | 124.17 [122.70, 125.77] | 82.34 [81.48, 83.22] | 0.1472 |
| PF_WARM_FROZEN | PRACTICAL_BEST | 0.101321 [0.099594, 0.102950] | False | 112.46 [111.28, 113.71] | 82.32 [81.48, 83.21] | 0.1473 |
| PF_WARM_FROZEN | CONTROLLED_BUDGET | 0.101355 [0.099626, 0.103042] | False | 124.19 [122.71, 125.78] | 82.35 [81.50, 83.22] | 0.1472 |
| FA_Q8_WARM_FROZEN | PRACTICAL_BEST | 0.101331 [0.099626, 0.102963] | False | 112.47 [111.27, 113.74] | 82.33 [81.48, 83.22] | 0.1472 |
| FA_Q8_WARM_FROZEN | CONTROLLED_BUDGET | 0.101333 [0.099602, 0.103024] | False | 124.16 [122.68, 125.75] | 82.33 [81.47, 83.21] | 0.1471 |
| FA_1024_WARM_FROZEN | PRACTICAL_BEST | 0.101617 [0.099957, 0.103286] | False | 112.79 [111.57, 114.05] | 82.56 [81.68, 83.43] | 0.1468 |
| FA_1024_WARM_FROZEN | CONTROLLED_BUDGET | 0.100932 [0.099289, 0.102655] | False | 123.67 [122.18, 125.24] | 82.00 [81.12, 82.91] | 0.1474 |
| DP1_WARM_FROZEN | PRACTICAL_BEST | 0.101622 [0.099976, 0.103338] | False | 112.81 [111.42, 114.27] | 82.57 [81.67, 83.44] | 0.1466 |
| DP1_WARM_FROZEN | CONTROLLED_BUDGET | 0.100759 [0.099113, 0.102457] | False | 123.24 [121.55, 125.01] | 81.86 [80.99, 82.73] | 0.1475 |
| PRE | PRACTICAL_BEST | 0.101784 [0.100164, 0.103467] | False | 112.99 [111.57, 114.44] | 82.70 [81.81, 83.55] | 0.1460 |
| PRE | CONTROLLED_BUDGET | 0.101784 [0.100164, 0.103467] | False | 124.49 [122.80, 126.20] | 82.70 [81.81, 83.55] | 0.1460 |
| FT_C | PRACTICAL_BEST | 0.087199 [0.085609, 0.088787] | False | 96.78 [95.92, 97.64] | - | 0.1726 |
| FT_PRE | PRACTICAL_BEST | 0.105761 [0.104159, 0.107395] | False | 117.39 [116.21, 118.69] | 85.93 [85.13, 86.75] | 0.1314 |
| FT_FA | PRACTICAL_BEST | 0.074155 [0.072590, 0.075698] | False | 82.31 [81.35, 83.32] | - | 0.2037 |
| FT_FA_WARM | PRACTICAL_BEST | 0.104684 [0.103075, 0.106274] | False | 116.19 [115.00, 117.48] | 85.05 [84.23, 85.87] | 0.1335 |
| FT_FA_WARM_FROZEN | PRACTICAL_BEST | 0.105384 [0.103806, 0.107049] | False | 116.97 [115.79, 118.37] | 85.62 [84.80, 86.44] | 0.1318 |
| LO | PRACTICAL_BEST | -0.066109 [-0.067019, -0.065131] | True | -73.38 [-75.16, -71.75] | - | 0.6208 |
| C_SMALL_FP32 | PRACTICAL_BEST | 0.081444 [0.078744, 0.084078] | False | 92.22 [90.50, 94.02] | - | 0.1883 |
| C_SMALL_INT8 | PRACTICAL_BEST | 0.081199 [0.078464, 0.083859] | False | 91.94 [90.14, 93.74] | - | 0.1880 |
| KNN_EMB_SMALL | RULE_FREE | 0.001825 [0.000769, 0.002816] | False | 2.03 [0.86, 3.12] | - | 0.3615 |
| KNN_EMB_BIG | RULE_FREE | 0.010854 [0.009768, 0.011883] | False | 12.05 [10.90, 13.15] | - | 0.3153 |
| RULE | RULE_FREE | -0.033421 [-0.034484, -0.032249] | True | -37.09 [-38.60, -35.59] | - | 0.5081 |
| SKNN | RULE_FREE | 0.040390 [0.039272, 0.041428] | False | 44.83 [43.87, 45.78] | - | 0.2136 |
| POP | RULE_FREE | -0.051682 [-0.052763, -0.050565] | True | -57.36 [-59.14, -55.62] | - | 0.5771 |

### device-only envelope {LO, LAST, RULE} (06L-C §C3)

| arm | MRR@20 (seed mean) | R% vs C_BIG [CI] | Δ vs LAST [CI] |
|---|---|---|---|
| LO | 0.005002 | 3.10 [2.91, 3.30] | -0.066109 [-0.067019, -0.065131] |
| LAST | 0.071112 | 44.11 [43.57, 44.69] | ANCHOR |
| RULE | 0.037691 | 23.38 [22.90, 23.91] | -0.033421 [-0.034484, -0.032249] |

- best point (DESCRIPTIVE, never a selection): {"arm": "LAST", "value": 0.07111177187116745}; LO is an untuned from-θ0 control, not the device-only ceiling (review 10 [9-R1 (d)]).

## 9. Privacy matrix (06L-C C1) and one bytes axis (06L-C C8) — DESCRIPTIVE

| level | arm | R% vs C_BIG (PB, seed mean) | bytes/client | server sees (training) | server sees (inference) | formal guarantee | against whom | secure aggregation |
|---|---|---|---|---|---|---|---|---|
| L0 cloud-train / cloud-infer | C_BIG | 100.00 [100.00, 100.00] | - | raw events | query context | none | - | n/a (no federated training) |
| L0 cloud-train / cloud-infer | C_FULL | 120.62 [119.97, 121.32] | - | raw events | query context | none | - | n/a (no federated training) |
| L0 cloud-train / cloud-infer | C_SMALL | 95.80 [95.33, 96.26] | - | raw events | query context | none | - | n/a (no federated training) |
| L0 cloud-train / cloud-infer | SKNN | 69.17 [68.60, 69.75] | - | raw events | query context | none | - | n/a (no federated training) |
| L1 cloud-train / device-infer | C_SMALL_FP32 | 95.69 [94.74, 96.69] | - | raw events (model trained centrally once) | nothing | none | - | n/a (no federated training) |
| L1 cloud-train / device-infer | C_SMALL_INT8 | 95.54 [94.55, 96.53] | - | raw events (model trained centrally once) | nothing | none | - | n/a (no federated training) |
| L1 cloud-train / device-infer | KNN_EMB_SMALL | 45.24 [44.57, 45.95] | - | raw events (model trained centrally once) | nothing | none | - | n/a (no federated training) |
| L1 cloud-train / device-infer | KNN_EMB_BIG | 50.84 [50.12, 51.58] | - | raw events (model trained centrally once) | nothing | none | - | n/a (no federated training) |
| L1 on opted-in data (PRE-alone) | PRE | 107.26 [106.49, 108.06] | - | raw events of the opted-in users only (PRETRAIN_TRAIN_5_100_NONMAIN); none of the MAIN users | nothing | none (the MAIN users' data is never used) | - | n/a (no federated training) |
| L1 base (server-trained on raw events) + local fine-tune | FT_C | 98.20 [97.72, 98.68] | - | raw events of the same users (the base) and nothing after the download | nothing | none | - | n/a (no federated training) |
| L2 federated | FA | 86.07 [85.56, 86.62] | 994562798 | model deltas; non-DP deltas reveal the user's viewed items (input-table rows and output rows move; 06I §7(a), 06L-C §C1) | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L2 federated | FP | 86.07 [85.56, 86.62] | 994562798 | model deltas; non-DP deltas reveal the user's viewed items (input-table rows and output rows move; 06I §7(a), 06L-C §C1) | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L2 federated | FA_1024 | 21.68 [21.09, 22.24] | 496891468 | model deltas; non-DP deltas reveal the user's viewed items (input-table rows and output rows move; 06I §7(a), 06L-C §C1) | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L2 federated | FA_Q8 | 85.92 [85.38, 86.48] | 621602968 | model deltas; non-DP deltas reveal the user's viewed items (input-table rows and output rows move; 06I §7(a), 06L-C §C1) | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L2 + consenting-cohort pretraining | FA_WARM | 106.45 [105.78, 107.12] | 994562798 | model deltas; non-DP deltas reveal the user's viewed items (input-table rows and output rows move; 06I §7(a), 06L-C §C1); θ_pre from opted-in users | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L2 + frozen item tables (S8) | FA_WARM_FROZEN | 106.98 [106.28, 107.67] | 94449988 | encoder deltas only (265,162 uploaded parameters, to be re-measured); the input-table item-set leak is removed only partly: brand/category deltas may still leak, off-support rows move ≈ 1e-9, not exactly 0 (06L-C §C8) | nothing | none | the honest-but-curious browser vendor sees every delta (06L-C §C2) | secure aggregation: not simulated |
| L3 federated + DP | DP8 | 21.40 [20.83, 21.96] | 496891468 | clipped deltas + Gaussian noise on the sum | nothing | central DP with a trusted aggregator, client-level add/remove adjacency, (ε, δ = 5e-6), conditional on the public bootstrap block (catalogue, transform statistics, θ0 / θ_pre from opted-in users); the FL hyper-parameter tuning on CALIB is not accounted (06L-C §C2) | anyone who sees the released model; not the trusted aggregator itself | secure aggregation: not simulated |
| L3 federated + DP | DP1 | 20.86 [20.28, 21.40] | 496891468 | clipped deltas + Gaussian noise on the sum | nothing | central DP with a trusted aggregator, client-level add/remove adjacency, (ε, δ = 5e-6), conditional on the public bootstrap block (catalogue, transform statistics, θ0 / θ_pre from opted-in users); the FL hyper-parameter tuning on CALIB is not accounted (06L-C §C2) | anyone who sees the released model; not the trusted aggregator itself | secure aggregation: not simulated |
| L3 federated + DP | DP8_FROZEN | 107.14 [106.49, 107.82] | 84010589 | clipped deltas + Gaussian noise on the sum | nothing | central DP with a trusted aggregator, client-level add/remove adjacency, (ε, δ = 5e-6), conditional on the public bootstrap block (catalogue, transform statistics, θ0 / θ_pre from opted-in users); the FL hyper-parameter tuning on CALIB is not accounted (06L-C §C2) | anyone who sees the released model; not the trusted aggregator itself | secure aggregation: not simulated |
| L4 federated + local personalisation | PF | 86.09 [85.57, 86.63] | 994562798 | as L2; personal parameters never leave | nothing | none beyond L2 | the honest-but-curious browser vendor sees every shared delta | secure aggregation: not simulated |
| L4 federated + local personalisation | FT_FA | 90.11 [89.58, 90.67] | - | as L2; personal parameters never leave | nothing | none beyond L2 | the honest-but-curious browser vendor sees every shared delta | secure aggregation: not simulated |
| L4 federated + local personalisation | FT_FA_WARM | 109.05 [108.39, 109.75] | - | as L2; personal parameters never leave | nothing | none beyond L2 | the honest-but-curious browser vendor sees every shared delta | secure aggregation: not simulated |
| L4 federated + local personalisation | FT_FA_WARM_FROZEN | 109.48 [108.83, 110.21] | - | as L2; personal parameters never leave | nothing | none beyond L2 | the honest-but-curious browser vendor sees every shared delta | secure aggregation: not simulated |
| L5 (base trained only on opted-in users) + local fine-tune | FT_PRE | 109.72 [109.10, 110.40] | - | nothing of the MAIN users (θ_pre from opted-in users; download only) | nothing | none needed for the MAIN users (their data never leaves the device) | - | n/a (no federated training) |
| L6 device-only | LO | 3.10 [2.91, 3.30] | 0 | nothing | nothing | data never leaves (shared catalogue / θ0 only) | - | n/a (no federated training) |
| L6 device-only | LAST | 44.11 [43.57, 44.69] | 0 | nothing | nothing | data never leaves (shared catalogue / θ0 only) | - | n/a (no federated training) |
| device + published aggregate | RULE | 23.38 [22.90, 23.91] | - | nothing of the user (a published MAIN-TRAIN popularity / category aggregate is downloaded) | nothing | none for the users in the aggregate | - | n/a (no federated training) |
| device + published aggregate | POP | 12.05 [11.67, 12.50] | - | nothing of the user (a published MAIN-TRAIN popularity / category aggregate is downloaded) | nothing | none for the users in the aggregate | - | n/a (no federated training) |

### Bytes axis (raw events → model download → FL traffic → S8)

| step | level | arm | bytes/client | bytes/client incl. final download | status | rule |
|---|---|---|---|---|---|---|
| 0 | device-only | LO | 0 | - | COMPUTED | no upload; only the shared bootstrap block (catalogue / θ0) |
| 0 | device + published aggregate | RULE | - | - | AGGREGATE_SIZE_NOT_MEASURED | the published MAIN-TRAIN popularity / category aggregate is downloaded (size not measured) |
| 0 | device-only | LAST | 0 | - | COMPUTED | no upload; only the shared bootstrap block (catalogue / θ0) |
| 1 | raw events to the server (L0 / L1 cloud training) | C_BIG / C_SMALL / C_FULL | - | - | NOT_REGISTERED (06L-C §C8: the per-event size convention needs a DL row; records.EVENT_SIZE) | bytes_per_event (records.EVENT_SIZE) x mean TRAIN decisions per MAIN client (TRAIN manifest) |
| 2 | model download FP32 (L1 device inference; L5 / FT base; PRE-alone) | C_SMALL | - | - | NO_MODEL_BYTES | the model file bytes (entry meta model_bytes), downloaded once |
| 2 | model download INT8 (ONNX dynamic quantisation) | C_SMALL_INT8 | - | - | NO_MODEL_BYTES | the model file bytes (entry meta model_bytes), downloaded once |
| 3 | FL training traffic (+ the final download) | FA | 994562798 | 1077397630 | COMPUTED | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | FP | 994562798 | 1077397630 | COMPUTED | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | PF | 994562798 | 1077397630 | COMPUTED | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | FA_1024 | 496891468 | 579726300 | COMPUTED | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | DP8 | 496891468 | 579726300 | COMPUTED | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | DP1 | 496891468 | 579726300 | COMPUTED | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | FA_Q8 | 621602968 | 704437800 | COMPUTED | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 3 | FL training traffic (+ the final download) | FA_WARM | 994562798 | 1077397630 | COMPUTED | visits x (down + up) per client (06A §3), registered mean; + one final model download |
| 4 | S8: frozen tables once + encoder rounds | FA_WARM_FROZEN | 94449988 | 95505276 | COMPUTED | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | DP8_FROZEN | 84010589 | 85065877 | COMPUTED | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | FP_WARM_FROZEN | 94449988 | 95505276 | COMPUTED | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | PF_WARM_FROZEN | 94449988 | 95505276 | COMPUTED | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | FA_Q8_WARM_FROZEN | 89699718 | 90755006 | COMPUTED | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | FA_1024_WARM_FROZEN | 84010589 | 85065877 | COMPUTED | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |
| 4 | S8: frozen tables once + encoder rounds | DP1_WARM_FROZEN | 84010589 | 85065877 | COMPUTED | PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3); + one final model download |

## 10. Labelled rows (06L-C C11, C16-C17) — DESCRIPTIVE

### 10a. DP8_FROZEN vs its z = 0 twin S8

- R(DP8_FROZEN / FA_WARM_FROZEN) PRACTICAL_BEST: 100.15 [99.96, 100.35] (ε = 8.000000, z = 0.5642)
- R(DP8_FROZEN / FA_WARM_FROZEN) CONTROLLED_BUDGET: 99.72 [99.50, 99.95] (ε = 8.000000, z = 0.5642)

### 10b. No-FT references on UNSEEN_VAL (the selection panel; never a headline)

- NOT_AVAILABLE: no UNSEEN_VAL role in the block

- 

### 10c. Deployed-size family comparison GRU_D128 vs C_SMALL (06L-C C11)

- NOT_AVAILABLE: no GRU_D128 entry

- GRU_D128 (the contest's 02B_GRU best@6.0, 20.96M parameters, already trained) vs C_SMALL (20.71M), PRACTICAL_BEST and CONTROLLED_BUDGET, on the A40 block; never a downstream family (ARCHITECTURE_LOCK unchanged).

## 11. CALIB-disjoint diagnostic (MAIN \ CALIB; 06L-C C14) and repeat / novel target strata (C15) — DESCRIPTIVE

- 38,496 CALIB-disjoint users, of whom 30,694 have ≥ 1 eval row (manifest users: MAIN 51,211, CALIB 12,715; E2E users: MAIN 40,764, CALIB 10,070); counts {"e2e": {"calib_users": 10070, "disjoint_users": 30694, "main_users": 40764}, "manifest": {"calib_users": 12715, "disjoint_users": 38496, "main_users": 51211}}; CALIB_VALIDATION users -> MAIN by client_key_hex; report only, never selection (06L-C §C14).

| arm | R% vs C_BIG on MAIN \ CALIB [CI] | R% vs C_BIG on MAIN [CI] |
|---|---|---|
| C_SMALL | 95.71 [95.15, 96.27] | 95.80 [95.33, 96.26] |
| FA | 86.05 [85.33, 86.73] | 86.07 [85.56, 86.62] |
| FP | 86.05 [85.33, 86.73] | 86.07 [85.56, 86.62] |
| PF | 86.07 [85.36, 86.75] | 86.09 [85.57, 86.63] |
| FA_1024 | 21.90 [21.30, 22.57] | 21.68 [21.09, 22.24] |
| DP8 | 21.59 [21.02, 22.24] | 21.40 [20.83, 21.96] |
| DP1 | 21.00 [20.44, 21.65] | 20.86 [20.28, 21.40] |
| FA_Q8 | 85.90 [85.18, 86.60] | 85.92 [85.38, 86.48] |
| FA_WARM | 105.99 [105.17, 106.75] | 106.45 [105.78, 107.12] |
| FA_WARM_FROZEN | 106.50 [105.65, 107.30] | 106.98 [106.28, 107.67] |
| DP8_FROZEN | 106.68 [105.80, 107.50] | 107.14 [106.49, 107.82] |
| FP_WARM_FROZEN | 106.51 [105.65, 107.31] | 106.99 [106.29, 107.67] |
| PF_WARM_FROZEN | 106.48 [105.62, 107.29] | 106.96 [106.29, 107.65] |
| FA_Q8_WARM_FROZEN | 106.48 [105.63, 107.30] | 106.97 [106.29, 107.66] |
| FA_1024_WARM_FROZEN | 106.69 [105.81, 107.48] | 107.15 [106.49, 107.82] |
| DP1_WARM_FROZEN | 106.73 [105.73, 107.63] | 107.16 [106.39, 107.93] |
| FT_FA | 90.12 [89.46, 90.84] | 90.11 [89.58, 90.67] |
| FT_FA_WARM | 108.72 [107.94, 109.48] | 109.05 [108.39, 109.75] |
| FT_FA_WARM_FROZEN | 109.20 [108.41, 110.01] | 109.48 [108.83, 110.21] |

| id | contrast | rule | on MAIN \ CALIB [CI] |
|---|---|---|---|
| K1 | R(C_SMALL / C_BIG) | PRACTICAL_BEST | 95.71 [95.15, 96.27] |
| K2 | R(FA / C_BIG) | PRACTICAL_BEST | 86.05 [85.33, 86.73] |
| K3 | R(FA / C_SMALL) | PRACTICAL_BEST | 89.90 [89.27, 90.51] |
| K4 | R(FP / C_BIG) | PRACTICAL_BEST | 86.05 [85.33, 86.73] |
| K5 | R(PF / C_BIG) | PRACTICAL_BEST | 86.07 [85.36, 86.75] |
| K6 | R(FT_FA / C_BIG) | PRACTICAL_BEST | 90.12 [89.46, 90.84] |
| K9 | FT_C − C_SMALL | PRACTICAL_BEST | 0.003946 [0.003416, 0.004489] |
| K10 | FT_FA − FA | PRACTICAL_BEST | 0.006616 [0.006089, 0.007190] |
| K11 | FA − LO | PRACTICAL_BEST | 0.134676 [0.132777, 0.136467] |
| K11b | FA − LAST | PRACTICAL_BEST | 0.068275 [0.066458, 0.070074] |
| K12 | R(DP8 / FA_1024) | CONTROLLED_BUDGET | 98.58 [97.87, 99.34] |
| K13 | R(DP1 / FA_1024) | CONTROLLED_BUDGET | 95.83 [94.50, 97.26] |
| K14 | R(FA_Q8 / FA) | PRACTICAL_BEST | 99.83 [99.62, 100.06] |
| K17 | R(FA / C_FULL) | PRACTICAL_BEST | 71.87 [71.17, 72.56] |
| K20 | R(FA_WARM_FROZEN / C_BIG) | PRACTICAL_BEST | 106.50 [105.65, 107.30] |
| K20b | R(FA_WARM_FROZEN / C_FULL) | PRACTICAL_BEST | 88.70 [88.15, 89.30] |
| K21 | FA_WARM_FROZEN − PRE | PRACTICAL_BEST | -0.000465 [-0.000836, -0.000117] |
| K_INIT | FA_WARM_FROZEN − FA | PRACTICAL_BEST | 0.033219 [0.031985, 0.034454] |
| P5 | FT_FA_WARM_FROZEN − FA_WARM_FROZEN | PRACTICAL_BEST | 0.004388 [0.003603, 0.005065] |
| K4w | R(FP_WARM_FROZEN / C_FULL) | PRACTICAL_BEST | 88.70 [88.15, 89.30] |
| K5w | R(PF_WARM_FROZEN / C_FULL) | PRACTICAL_BEST | 88.68 [88.15, 89.30] |
| K12w | R(DP8_FROZEN / FA_1024_WARM_FROZEN) | CONTROLLED_BUDGET | 99.97 [99.94, 100.01] |
| K13w | R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN) | CONTROLLED_BUDGET | 99.90 [99.81, 99.98] |
| K14w | R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN) | PRACTICAL_BEST | 99.98 [99.88, 100.08] |

- REPEAT = RANKABLE and LAST tie count 0 (target in the L50 window); NOVEL = the other E2E rows; counts {"E2E_rows": 519714, "NOVEL_rows": 373482, "REPEAT_rows": 146232, "repeat_share_of_E2E_rows": 0.2813701381913899}.

| stratum | arm | users | R% vs C_BIG | 95% CI |
|---|---|---|---|---|
| REPEAT | C_BIG | 26695 | 100.00 | [100.00, 100.00] |
| REPEAT | C_FULL | 26695 | 123.63 | [122.75, 124.54] |
| REPEAT | C_SMALL | 26695 | 89.26 | [88.74, 89.82] |
| REPEAT | FA | 26695 | 75.99 | [75.36, 76.62] |
| REPEAT | FP | 26695 | 75.99 | [75.37, 76.62] |
| REPEAT | PF | 26695 | 75.99 | [75.36, 76.62] |
| REPEAT | FA_1024 | 26695 | 20.28 | [19.69, 20.84] |
| REPEAT | DP8 | 26695 | 20.14 | [19.56, 20.69] |
| REPEAT | DP1 | 26695 | 19.72 | [19.14, 20.29] |
| REPEAT | FA_Q8 | 26695 | 75.93 | [75.28, 76.55] |
| REPEAT | FA_WARM | 26695 | 102.90 | [102.14, 103.67] |
| REPEAT | FA_WARM_FROZEN | 26695 | 104.10 | [103.33, 104.91] |
| REPEAT | DP8_FROZEN | 26695 | 104.46 | [103.68, 105.27] |
| REPEAT | FP_WARM_FROZEN | 26695 | 104.11 | [103.33, 104.92] |
| REPEAT | PF_WARM_FROZEN | 26695 | 103.82 | [103.05, 104.61] |
| REPEAT | FA_Q8_WARM_FROZEN | 26695 | 103.81 | [103.03, 104.61] |
| REPEAT | FA_1024_WARM_FROZEN | 26695 | 104.47 | [103.69, 105.27] |
| REPEAT | DP1_WARM_FROZEN | 26695 | 104.20 | [103.27, 105.06] |
| REPEAT | PRE | 26695 | 104.56 | [103.64, 105.40] |
| REPEAT | FT_C | 26695 | 97.06 | [96.49, 97.66] |
| REPEAT | FT_PRE | 26695 | 116.17 | [115.31, 117.06] |
| REPEAT | FT_FA | 26695 | 85.91 | [85.24, 86.56] |
| REPEAT | FT_FA_WARM | 26695 | 114.68 | [113.82, 115.58] |
| REPEAT | FT_FA_WARM_FROZEN | 26695 | 116.02 | [115.11, 116.90] |
| REPEAT | LO | 26695 | 4.51 | [4.26, 4.75] |
| REPEAT | KNN_EMB_SMALL | 26695 | 68.37 | [67.29, 69.40] |
| REPEAT | KNN_EMB_BIG | 26695 | 82.39 | [81.18, 83.52] |
| REPEAT | RULE | 26695 | 31.11 | [30.41, 31.80] |
| REPEAT | SKNN | 26695 | 94.35 | [93.24, 95.52] |
| REPEAT | LAST | 26695 | 88.14 | [87.17, 89.09] |
| REPEAT | POP | 26695 | 11.28 | [10.80, 11.73] |
| NOVEL | C_BIG | 38479 | 100.00 | [100.00, 100.00] |
| NOVEL | C_FULL | 38479 | 120.06 | [119.02, 121.14] |
| NOVEL | C_SMALL | 38479 | 104.63 | [103.94, 105.32] |
| NOVEL | FA | 38479 | 98.60 | [97.78, 99.47] |
| NOVEL | FP | 38479 | 98.60 | [97.78, 99.47] |
| NOVEL | PF | 38479 | 98.61 | [97.79, 99.49] |
| NOVEL | FA_1024 | 38479 | 22.26 | [21.52, 23.04] |
| NOVEL | DP8 | 38479 | 21.87 | [21.15, 22.63] |
| NOVEL | DP1 | 38479 | 20.98 | [20.23, 21.81] |
| NOVEL | FA_Q8 | 38479 | 98.43 | [97.64, 99.29] |
| NOVEL | FA_WARM | 38479 | 114.98 | [114.04, 115.96] |
| NOVEL | FA_WARM_FROZEN | 38479 | 114.82 | [113.86, 115.86] |
| NOVEL | DP8_FROZEN | 38479 | 114.58 | [113.57, 115.59] |
| NOVEL | FP_WARM_FROZEN | 38479 | 114.83 | [113.86, 115.86] |
| NOVEL | PF_WARM_FROZEN | 38479 | 115.05 | [114.06, 116.08] |
| NOVEL | FA_Q8_WARM_FROZEN | 38479 | 115.07 | [114.09, 116.10] |
| NOVEL | FA_1024_WARM_FROZEN | 38479 | 114.59 | [113.58, 115.60] |
| NOVEL | DP1_WARM_FROZEN | 38479 | 114.83 | [113.70, 116.03] |
| NOVEL | PRE | 38479 | 114.66 | [113.51, 115.88] |
| NOVEL | FT_C | 38479 | 98.89 | [98.21, 99.60] |
| NOVEL | FT_PRE | 38479 | 104.05 | [103.06, 105.00] |
| NOVEL | FT_FA | 38479 | 93.05 | [92.27, 93.83] |
| NOVEL | FT_FA_WARM | 38479 | 104.18 | [103.21, 105.10] |
| NOVEL | FT_FA_WARM_FROZEN | 38479 | 103.83 | [102.90, 104.81] |
| NOVEL | LO | 38479 | 0.20 | [0.16, 0.26] |
| NOVEL | KNN_EMB_SMALL | 38479 | 16.50 | [16.04, 16.96] |
| NOVEL | KNN_EMB_BIG | 38479 | 14.09 | [13.71, 14.46] |
| NOVEL | RULE | 38479 | 14.96 | [14.43, 15.54] |
| NOVEL | SKNN | 38479 | 52.94 | [52.36, 53.52] |
| NOVEL | LAST | 38479 | 0.00 | [0.00, 0.00] |
| NOVEL | POP | 38479 | 12.45 | [11.89, 13.02] |

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

- S7 (FA_WARM): BOUND_AND_FILE_VERIFIED; warm selection sha256 d67a0a851665495b092abbba4456f26a3423d6ef21665f5a0e2b65e2d78dc000; record {"file": "records/FL_WARM_SELECTION_S7.json", "file_sha256": "d67a0a851665495b092abbba4456f26a3423d6ef21665f5a0e2b65e2d78dc000", "self_sha256": "c7bd76d61e0bdf8494556eace553fa9a27255edfac1d3750181bc1abe9239c18", "winner_factor": 0.25, "winner_label": "f=0.25|sf=0.1", "winner_server_lr_factor": 0.1}.
- S8 (FA_WARM_FROZEN): BOUND_AND_FILE_VERIFIED; warm selection sha256 8b79d3052799dfa0ba208dce5db138da88aaa7024ba6511a374c59ce8c55a98a; record {"file": "records/FL_WARM_SELECTION_S8.json", "file_sha256": "8b79d3052799dfa0ba208dce5db138da88aaa7024ba6511a374c59ce8c55a98a", "self_sha256": "53bc7a5a1312310e3071bcd14d8fb449d6ed6225206e5483d0235440803eed3d", "winner_factor": 1.0, "winner_label": "f=1|sf=0.1", "winner_server_lr_factor": 0.1}.

### FT_FA_WARM − FA_WARM (06M_B B4; the full production stack)

| contrast | status | seeds | Δ MRR@20 | 95% CI | CI vs 0 | harmed share | skipped |
|---|---|---|---|---|---|---|---|
| FT_FA_WARM − FA_WARM | COMPUTED | [2026, 2027] | 0.004187 | [0.003571, 0.004821] | above | 0.3703 | - |
| FT_FA_WARM_FROZEN − FA_WARM_FROZEN | COMPUTED | [2026, 2027] | 0.004033 | [0.003429, 0.004672] | above | 0.3658 | - |

### Quality during rollout (06M_B B5)

- 06M_B B5 (DESCRIPTIVE): per FL run (PRACTICAL_BEST entry's 12 marks) the macro-E2E MRR@20 per mark; AUC = trapezoid over the marks 0.5 .. 6.0 EFE (MRR x EFE), auc_mean = AUC / 5.5 (with its paired user-bootstrap CI, per resample on the same W); first mark reaching a reference (PRE-alone = theta_pre, seed 2026, PRACTICAL_BEST label; LAST = the seed-free rank file): POINT rule = the first mark whose value >= the reference's value, CI rule = the first mark whose paired delta 95% CI lies above 0; NOT_REACHED / NO_REFERENCE otherwise; seed mean of the per-seed AUC

| arm | seed | AUC (MRR x EFE) | AUC / 5.5 | 95% CI | first mark ≥ PRE (point) | first mark > PRE (CI) | first mark ≥ LAST (point) | first mark > LAST (CI) |
|---|---|---|---|---|---|---|---|---|
| FA | 2026 | 0.611863 | 0.111248 | [0.109882, 0.112668] | NOT_REACHED | NOT_REACHED | 1.000000 | 1.000000 |
| FA | 2027 | 0.607696 | 0.110490 | [0.109110, 0.111839] | NOT_REACHED | NOT_REACHED | 1.000000 | 1.000000 |
| FA_WARM | 2026 | 0.937354 | 0.170428 | [0.168635, 0.172305] | NOT_REACHED | NOT_REACHED | 0.500000 | 0.500000 |
| FA_WARM | 2027 | 0.937924 | 0.170532 | [0.168738, 0.172379] | NOT_REACHED | NOT_REACHED | 0.500000 | 0.500000 |
| FA_WARM_FROZEN | 2026 | 0.946810 | 0.172147 | [0.170408, 0.173978] | NOT_REACHED | NOT_REACHED | 0.500000 | 0.500000 |
| FA_WARM_FROZEN | 2027 | 0.946935 | 0.172170 | [0.170447, 0.173988] | NOT_REACHED | NOT_REACHED | 0.500000 | 0.500000 |

### S8 communication from the run's own record (06L-C C8)

| arm | seed | bytes/visit | up | down | uploaded params (measured) | frozen params | frozen bytes once | per-visit ratio vs FA_WARM |
|---|---|---|---|---|---|---|---|---|
| FA_WARM_FROZEN | 2026 | 2110584 | 1055296 | 1055288 | 263822 | 20444886 | 81779544 | 0.0127 |
| FA_WARM_FROZEN | 2027 | 2110584 | 1055296 | 1055288 | 263822 | 20444886 | 81779544 | 0.0127 |
| DP8_FROZEN | 2026 | 2110584 | 1055296 | 1055288 | 263822 | 20444886 | 81779544 | 0.0127 |
| DP8_FROZEN | 2027 | 2110584 | 1055296 | 1055288 | 263822 | 20444886 | 81779544 | 0.0127 |
| FP_WARM_FROZEN | 2026 | 2110584 | 1055296 | 1055288 | 263822 | 20444886 | 81779544 | 0.0127 |
| FP_WARM_FROZEN | 2027 | 2110584 | 1055296 | 1055288 | 263822 | 20444886 | 81779544 | 0.0127 |
| PF_WARM_FROZEN | 2026 | 2110584 | 1055296 | 1055288 | 263822 | 20444886 | 81779544 | 0.0127 |
| PF_WARM_FROZEN | 2027 | 2110584 | 1055296 | 1055288 | 263822 | 20444886 | 81779544 | 0.0127 |
| FA_Q8_WARM_FROZEN | 2026 | 1319306 | 264018 | 1055288 | 263822 | 20444886 | 81779544 | 0.0080 |
| FA_Q8_WARM_FROZEN | 2027 | 1319306 | 264018 | 1055288 | 263822 | 20444886 | 81779544 | 0.0080 |
| FA_1024_WARM_FROZEN | 2026 | 2110584 | 1055296 | 1055288 | 263822 | 20444886 | 81779544 | 0.0127 |
| FA_1024_WARM_FROZEN | 2027 | 2110584 | 1055296 | 1055288 | 263822 | 20444886 | 81779544 | 0.0127 |
| DP1_WARM_FROZEN | 2026 | 2110584 | 1055296 | 1055288 | 263822 | 20444886 | 81779544 | 0.0127 |

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

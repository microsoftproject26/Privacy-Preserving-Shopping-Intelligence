# Headline / abstract template (number-free; 06L-C C12; 06P hierarchy)

Placeholders `{Kx}`, `{Kx_CI}`, `{Kx_CLAIM}` are filled from the frozen RETENTION_RESULT.json of the single TEST look (`{Kx_CLAIM}` = the claim-register template key stamped on that K); the wording follows CLAIM_REGISTER.json.

1. Task. Next different viewed item within the session (views only), given the user's prior events across sessions (USER_HISTORY_L50); every retention is relative to a cloud model trained on 5-100-event users.
2. Cloud. The C_BIG cloud model reaches {K15} % [{K15_CI}] of the full-data cloud model C_FULL (K15, one seed).
3. Federated from a random start (secondary: the random-start ablation, K2). The matched-exposure retention of the registered FL recipe <{FL_RECIPE}> is {K2} % [{K2_CI}] of C_BIG ({K2_CLAIM}); the pretrained start adds {K_INIT} [{K_INIT_CI}] (K_INIT); against the no-training device heuristic LAST, FA − LAST = {K11b} [{K11b_CI}] ({K11b_CLAIM}); headroom R* = {FA_R_STAR} %.
4. Federated + DP. At ε = 8 (central DP, trusted aggregator, client-level, δ = 5e-6, conditional on the public bootstrap block; tuning not accounted) the model keeps {K12} % [{K12_CI}] of its non-private twin (K12, CONTROLLED_BUDGET).
5. Cloud model shipped + fine-tuned on device. FT_C on MAIN is an L1 base + local fine-tune: K7 = {K7} % [{K7_CI}]; the L5 answers are K9c = FT_PRE − PRE = {K9c} [{K9c_CI}] and, for base-unseen users, K9b = {K9b} [{K9b_CI}].
6. Device-only. The device-only envelope (LO, LAST, RULE) retains {ENVELOPE_R} % of C_BIG; LO alone (an untuned from-θ0 control) {K8} % [{K8_CI}].
7. Deployment branch (PRIMARY, the pair K20b + K21; the Edge answer). Federation from the pretrained model with the item tables frozen (FA_WARM* = S8) retains {K20b} % [{K20b_CI}] of its matched cloud reference C_FULL (K20b, {K20b_CLAIM}), the pretrained model alone {K20c} % [{K20c_CI}] (K20c); federated training on the private users adds {K21} [{K21_CI}] over θ_pre (K21, {K21_CLAIM}); freezing the item tables costs {S8_S7} [{S8_S7_CI}] at {S8_BYTES_RATIO} of the bytes (S8 − S7, DESCRIPTIVE).
8. Privacy. The adversary is an honest-but-curious browser vendor; non-DP federated updates reveal the viewed items and secure aggregation is not simulated; only the DP arms carry a formal guarantee (06L-C C1, C2).

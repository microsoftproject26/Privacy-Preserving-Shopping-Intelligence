# 04 — Results of the single VALIDATION read (Phase 1, REES46)

> **These are VALIDATION results, not TEST results.** They come from the one registered §9 read of 2026-10-03
> (F, 15:04–15:14Z server; DL 400). PRACTICAL_BEST values are the best of 12 marks on the same users' validation rows, so
> they are optimistic; the single TEST look is the only unbiased number (06A §9(g)) and has **not** happened (ON HOLD, DL 406).
> Everything computed on these data after DL 400 is POST_VALIDATION_EXPLORATORY.

**Sources (all MEASURED).** Copies of the report, claim register, headline template and the per-role metric tables are in
`HANDOVER/reference/VALIDATION_v3_*` — read the full report there; this file is the guided summary.
- `results/VALIDATION_v3/REPORT.md` (2,353 lines; sha `99445b2b…`) — the frozen report: contrasts (§1), retention tables
  (§2), personalisation (§3), privacy ladder (§4), curves (§5), panels (§6), LAST anchor (§8), privacy matrix and bytes
  axis (§9), labelled rows (§10), CALIB-disjoint and repeat / novel strata (§11), claim rules (§12), disclosures (§14).
- `results/VALIDATION_v3/CLAIM_REGISTER.json` (allowed / forbidden sentence per contrast, 39 contrasts) and
  `HEADLINE_TEMPLATE.md` (number-free abstract, filled only from TEST).
- `results/VALIDATION_v3/METRIC_TABLE.csv` — the registered DESCRIPTIVE metric table (DL 385): 6,192 cells, HR / NDCG /
  MRR @ 1, 5, 10, 20, micro and macro, E2E and RANKABLE, per arm, seed, rule and role.
- `results/VALIDATION_v3/OUTPUT_MANIFEST.json` — file shas; RETENTION_RESULT.json `0d765880…` (server only:
  `runs_ds3/ANA/out_val_v3/`).
- Server: script `retention_report.py` `9a3b2c2a…`, contrasts v4 `f1769f0b…`, block index `bd6e5dff…`, device pin
  `db744ca3…` (A40 class `NVIDIA_A40-FP32-DET`), evaluator `61b64e61…`.

Population: MAIN_VALIDATION, **40,764 users with ≥ 1 evaluation row, 519,714 decisions** (51,211 manifest users).
Bootstrap: paired user, 1,000 resamples, seed 20260926. Sanity trigger (amendment 04 §1): R(C_SMALL CB / C_BIG PB) = 94.74,
not fired.

## 1. Headline (registered hierarchy, 06P)

| Contrast | Meaning | Point | 95 % CI | Claim key |
|---|---|---|---|---|
| **K20b** R(FA_WARM_FROZEN / C_FULL) | the deployable warm federated system vs its matched cloud model | **88.81 %** | [88.27, 89.36] | BELOW 100 |
| **K21** FA_WARM_FROZEN − PRE | what federated training adds over the shipped pretrained model | **−0.000433** | [−0.000758, −0.000101] | BELOW 0 |
| K20c R(PRE / C_FULL) | the pretrained model alone vs C_FULL (one seed) | 89.03 % | [88.49, 89.57] | BELOW |
| K20 R(FA_WARM_FROZEN / C_BIG) | warm FL vs the same-users cloud model (mixes a 9× data advantage) | 106.98 % | [106.28, 107.67] | ABOVE |
| K_INIT FA_WARM_FROZEN − FA | value of the pretrained start | +0.033704 | [0.032688, 0.034760] | ABOVE |
| K2 R(FA / C_BIG) | cold federated retention (the ablation) | 86.07 % | [85.56, 86.62] | BELOW |
| K9c FT_PRE − PRE | local fine-tuning gain over θ_pre | +0.003976 | [0.003329, 0.004640] | ABOVE |

Reading (INFERENCE, consistent with the claim rules): **the deployable pretrained federated system keeps about 89 % of the
full-data cloud model, but on this recipe the federated rounds add nothing over the pretrained model alone (slightly
negative, CI below 0). Pretraining is what matters (K_INIT ≈ +0.034, much larger than any federated effect), and per-device
fine-tuning adds a small, reliable gain (K9c, P5 ≈ +0.004).** The deployment-branch claim needs both K20b and K21 (06P P1).

## 2. All registered contrasts (CONFIRMATORY rows; REPORT §1 and §1c)

Point, unadjusted 95 % paired-bootstrap CI; Holm = step-down over the 18 secondary contrasts (primary pair outside Holm;
tertiary rows carry a DESCRIPTIVE band only). "SS" = SEED_SENSITIVE (seed spread > half the CI width).

| Id | Contrast | Tier | Rule / seeds | Point | 95 % CI | Holm reject | Notes |
|---|---|---|---|---|---|---|---|
| K20b | R(FA_WARM_FROZEN / C_FULL) | primary | PB / 2 vs C_FULL s2026 | 88.81 | [88.27, 89.36] | outside Holm | per seed 88.81 / 88.81 |
| K21 | FA_WARM_FROZEN − PRE | primary | PB / 2 | −0.000433 | [−0.000758, −0.000101] | outside Holm | harmed share 0.36 |
| K1 | R(C_SMALL / C_BIG) | secondary | PB / 2 | 95.80 | [95.33, 96.26] | yes | CB 100.01 [99.48, 100.58] |
| K2 | R(FA / C_BIG) | secondary | PB / 2 | 86.07 | [85.56, 86.62] | yes | CB 90.86 |
| K3 | R(FA / C_SMALL) | secondary | PB / 2 | 89.85 | [89.36, 90.34] | yes | SS (0.57) |
| K8 | R(LO / C_BIG) | secondary | PB / 2 | 3.10 | [2.91, 3.30] | yes | SS (0.35); LO is an untuned control |
| K9b | FT_C − C_SMALL, UNSEEN_HOLDOUT | secondary | PB / 2026 | +0.004865 | [0.003873, 0.005919] | yes | 10,224 users |
| K9c | FT_PRE − PRE | secondary | PB / 2 | +0.003976 | [0.003329, 0.004640] | yes | |
| K11b | FA − LAST | secondary | PB / 2 | +0.067647 | [0.066146, 0.069190] | yes | cold FL beats LAST |
| K12 | R(DP8 / FA_1024) | secondary | **CB** / 2 | 98.71 | [98.09, 99.40] | yes (p 0.0003) | **SS (3.98): per seed 100.70 / 96.72** |
| K17 | R(FA / C_FULL) | secondary | PB / 2026 | 71.60 | [71.02, 72.21] | yes | = K2 × K15 |
| K20 | R(FA_WARM_FROZEN / C_BIG) | secondary | PB / 2 | 106.98 | [106.28, 107.67] | yes | CB 112.92 |
| K20c | R(PRE / C_FULL) | secondary | PB / 2026 | 89.03 | [88.49, 89.57] | yes | |
| K_INIT | FA_WARM_FROZEN − FA | secondary | PB / 2 | +0.033704 | [0.032688, 0.034760] | yes | |
| P5 | FT_FA_WARM_FROZEN − FA_WARM_FROZEN | secondary | PB / 2 | +0.004033 | [0.003429, 0.004672] | yes | harmed share 0.37 |
| K4w | R(FP_WARM_FROZEN / C_FULL) | secondary | PB / 2 | 88.81 | [88.28, 89.37] | yes | ≈ K20b |
| K5w | R(PF_WARM_FROZEN / C_FULL) | secondary | PB / 2 | 88.80 | [88.25, 89.34] | yes | ≈ K20b |
| K12w | R(DP8_FROZEN / FA_1024_WARM_FROZEN) | secondary | **CB** / 2 | 99.96 | [99.93, 100.00] | **no** (p 0.105) | includes 100 |
| K13w | R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN) | secondary | **CB** / 2026 | 99.88 | [99.80, 99.96] | yes (p 0.0085) | |
| K14w | R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN) | secondary | PB / 2 | 99.99 | [99.90, 100.09] | **no** (p 0.81) | TOST EQUIVALENT (90 % CI [99.91, 100.07]) |
| K4 | R(FP / C_BIG) | tertiary | PB / 2 | 86.07 | [85.56, 86.62] | band only | FP ≈ FA |
| K5 | R(PF / C_BIG) | tertiary | PB / 2 | 86.09 | [85.57, 86.63] | band only | PF ≈ FA |
| K6 | R(FT_FA / C_BIG) | tertiary | PB / 2 | 90.11 | [89.58, 90.67] | band only | |
| K7 | R(FT_C / C_BIG) | tertiary | PB / 2 | 98.20 | [97.72, 98.68] | band only | |
| K9 | FT_C − C_SMALL | tertiary | PB / 2 | +0.003879 | [0.003436, 0.004340] | band only | |
| K10 | FT_FA − FA | tertiary | PB / 2 | +0.006508 | [0.006044, 0.006971] | band only | |
| K11 | FA − LO | tertiary | PB / 2 | +0.133756 | [0.132176, 0.135451] | band only | |
| K13 | R(DP1 / FA_1024) | tertiary | **CB** / 2026 | 96.27 | [95.06, 97.42] | band only | |
| K14 | R(FA_Q8 / FA) | tertiary | PB / 2 | 99.82 | [99.62, 100.00] | band only (p 0.13) | SS (0.20); TOST EQUIVALENT (90 % CI [99.66, 99.97]) |
| K15 | R(C_BIG / C_FULL) | tertiary | PB / 2026 | 83.01 | [82.41, 83.58] | band only | |
| K16 | R(INT8 / FP32), C_SMALL ONNX-CPU | tertiary | PB / 2026 | 99.84 | [99.62, 100.07] | band only | 16,384 users / 206,445 decisions; includes 100 ("no detectable difference"; no TOST registered) |
| K18 | R(FT_C / C_FULL) | tertiary | PB / 2026 | 81.34 | [80.79, 81.91] | band only | |
| K19 | R(LO / C_FULL) | tertiary | PB / 2026 | 2.43 | [2.25, 2.61] | band only | |

PB ratios of K12 / K13 / K12w / K13w are DESCRIPTIVE (best-of-12 selection on the users' validation data is outside the DP
accountant): K12 98.71, K13 96.27, K12w 100.00, K13w 99.99.

**Registered DESCRIPTIVE contrasts (REPORT §1e):** S8 − S7 = FA_WARM_FROZEN − FA_WARM = **+0.000854** [0.000415, 0.001301]
at a bytes ratio of 0.095 (freezing the tables did not cost quality here; it slightly helped); R(FA_WARM / C_FULL) 88.25
[87.77, 88.71]; FA_WARM − PRE = −0.001287 [−0.001726, −0.000833]; PF_WARM_FROZEN − FA_WARM_FROZEN −0.000030 (CI includes 0);
R(FA_1024_WARM_FROZEN / FA_WARM_FROZEN) 100.15 (includes 100); FT_FA_WARM − FA_WARM +0.004187 [0.003571, 0.004821].

## 3. Every arm on MAIN (PRACTICAL_BEST; REPORT §2)

MRR@20 per seed (2026 / 2027) and seed-mean retention with 95 % CI. Baselines are seed-free (one rank file; the two CIs
in the report are against each C_BIG seed).

| Arm | MRR@20 s2026 / s2027 | R % vs C_BIG | R % vs C_SMALL | R % NDCG@10 | R % HR@10 | Bytes per client (training) | Still improving at 6.0? |
|---|---|---|---|---|---|---|---|
| C_BIG | 0.161196 / 0.161221 | 100 | 104.39 [103.88, 104.89] | 100 | 100 | — | — |
| C_FULL | 0.194192 / 0.194716 | 120.62 [119.97, 121.32] | 125.92 | 119.62 | 117.03 | — | — |
| C_SMALL | 0.154249 / 0.154613 | 95.80 [95.33, 96.26] | 100 | 96.07 | 97.06 | — | — |
| PRE (s2026) | 0.172896 | 107.26 [106.49, 108.06] | 112.09 | 107.35 | 107.95 | — | — |
| FA | 0.139037 / 0.138480 | 86.07 [85.56, 86.62] | 89.85 | 87.03 | 89.49 | 994,562,798 | yes |
| FP | 0.139033 / 0.138483 | 86.07 [85.56, 86.62] | 89.85 | 87.02 | 89.48 | 994,562,798 | yes |
| PF | 0.139060 / 0.138517 | 86.09 [85.57, 86.63] | 89.87 | 87.05 | 89.51 | 994,562,798 | yes |
| FA_Q8 | 0.138646 / 0.138370 | 85.92 [85.38, 86.48] | 89.69 | 86.82 | 89.26 | 621,602,968 | yes |
| FA_1024 | 0.034929 / 0.034986 | 21.68 [21.09, 22.24] | 22.64 | 22.27 | 24.19 | 496,891,468 | yes |
| DP8 | 0.035172 / 0.033837 | 21.40 [20.83, 21.96] | 22.34 | 22.03 | 23.92 | 496,891,468 | yes |
| DP1 (s2026) | 0.033625 | 20.86 [20.28, 21.40] | 21.80 | 21.65 | 23.89 | 496,891,468 | yes |
| FA_WARM (S7) | 0.171550 / 0.171669 | 106.45 [105.78, 107.12] | 111.12 | 106.69 | 107.63 | 994,562,798 | no |
| **FA_WARM_FROZEN (S8)** | 0.172465 / 0.172461 | 106.98 [106.28, 107.67] | 111.68 | 107.17 | 107.97 | 94,449,988 | no |
| FP_WARM_FROZEN | 0.172474 / 0.172468 | 106.99 | 111.68 | 107.18 | 107.97 | 94,449,988 | no |
| PF_WARM_FROZEN | 0.172379 / 0.172487 | 106.96 | 111.66 | 107.17 | 108.02 | 94,449,988 | no |
| FA_Q8_WARM_FROZEN | 0.172385 / 0.172500 | 106.97 | 111.66 | 107.17 | 108.00 | 89,699,718 | no |
| FA_1024_WARM_FROZEN | 0.172743 / 0.172713 | 107.15 | 111.85 | 107.28 | 107.94 | 84,010,589 | no |
| DP8_FROZEN | 0.172739 / 0.172702 | 107.14 [106.49, 107.82] | 111.84 | 107.27 | 107.93 | 84,010,589 | no |
| DP1_WARM_FROZEN (s2026) | 0.172734 | 107.16 | 111.98 | 107.30 | 107.96 | 84,010,589 | no |
| FT_C | 0.157966 / 0.158655 | 98.20 [97.72, 98.68] | 102.51 | 98.19 | 98.55 | — | — |
| **FT_PRE** | 0.176993 / 0.176752 | **109.72** [109.10, 110.40] | 114.53 | 109.60 | 109.68 | — (download only) | — |
| FT_FA | 0.145469 / 0.145064 | 90.11 [89.58, 90.67] | 94.07 | 90.39 | 91.54 | — | — |
| FT_FA_WARM | 0.175829 / 0.175763 | 109.05 | 113.83 | 109.09 | 109.50 | — | — |
| FT_FA_WARM_FROZEN | 0.176369 / 0.176623 | 109.48 [108.83, 110.21] | 114.29 | 109.56 | 109.93 | — | — |
| LO | 0.004717 / 0.005288 | 3.10 [2.91, 3.30] | 3.24 | 2.96 | 2.66 | 0 | — |
| C_SMALL_FP32 (ONNX, 16,384 users) | 0.152698 | 95.69 (DEVICE_MIXED) | 100 | 96.02 | 96.89 | — | — |
| C_SMALL_INT8 (ONNX, 16,384 users) | 0.152452 | 95.54 (DEVICE_MIXED) | 99.84 | 95.87 | 96.74 | — | — |
| SKNN | 0.111502 | 69.17 | 72.20 | 75.61 | 86.85 | — | — |
| KNN_EMB_BIG | 0.081965 | 50.84 | 53.08 | 50.43 | 49.56 | — | — |
| KNN_EMB_SMALL | 0.072936 | 45.24 | 47.23 | 45.11 | 45.41 | — | — |
| LAST | 0.071112 | 44.11 | 46.05 | 51.68 | 61.50 | 0 | — |
| RULE | 0.037691 | 23.38 | 24.41 | 24.40 | 27.86 | — | — |
| POP | 0.019430 | 12.05 | 12.58 | 12.66 | 14.70 | — | — |

Notes: bytes per client follow 06A §3 (visits × (download + upload), dropped clients not charged); the variants that charge
dropped visits are in REPORT §2. S8 bytes include the one-time 81.78 MB frozen-table download. The ONNX rows are on a
16,384-user subset in a separate CPU block, hence "DEVICE_MIXED" against A40 C_BIG; K16 compares them only to each other.
CONTROLLED_BUDGET table: REPORT §2 "CONTROLLED_BUDGET" (C_BIG CB 0.152870 / 0.152579; C_SMALL CB = C_BIG CB within 0.01 %;
warm arms CB ≈ PB; cold FL CB = PB because their best mark is the 6.0 endpoint).

## 4. All metrics (DESCRIPTIVE; macro, E2E denominator, MAIN, PRACTICAL_BEST or RULE_FREE; METRIC_TABLE.csv)

Values copied from the registered metric table and rounded to 4 decimals for display (the exact values, micro averages and
the RANKABLE denominator are in the CSV). Seed means and CIs on these metrics are **not** registered: if you compute them,
label them POST_VALIDATION_EXPLORATORY (DL 385). HR@k = Recall@k (one target per decision); HR@1 = NDCG@1 = MRR@1.

| Arm | Seed | HR@1 | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR@5 | MRR@10 | MRR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C_BIG | 2026 | 0.0883 | 0.2425 | 0.3343 | 0.4325 | 0.1670 | 0.1967 | 0.2215 | 0.1422 | 0.1544 | 0.1612 |
| C_BIG | 2027 | 0.0876 | 0.2429 | 0.3344 | 0.4325 | 0.1671 | 0.1967 | 0.2215 | 0.1422 | 0.1544 | 0.1612 |
| C_FULL | 2026 | 0.1095 | 0.2903 | 0.3916 | 0.4957 | 0.2024 | 0.2352 | 0.2615 | 0.1734 | 0.1870 | 0.1942 |
| C_FULL | 2027 | 0.1102 | 0.2908 | 0.3909 | 0.4952 | 0.2030 | 0.2354 | 0.2618 | 0.1741 | 0.1875 | 0.1947 |
| C_SMALL | 2026 | 0.0821 | 0.2329 | 0.3248 | 0.4269 | 0.1592 | 0.1888 | 0.2147 | 0.1350 | 0.1472 | 0.1542 |
| C_SMALL | 2027 | 0.0829 | 0.2329 | 0.3243 | 0.4261 | 0.1595 | 0.1890 | 0.2148 | 0.1354 | 0.1476 | 0.1546 |
| PRE | 2026 | 0.0939 | 0.2582 | 0.3609 | 0.4713 | 0.1780 | 0.2111 | 0.2390 | 0.1516 | 0.1652 | 0.1729 |
| FA | 2026 | 0.0718 | 0.2112 | 0.2994 | 0.3976 | 0.1430 | 0.1714 | 0.1962 | 0.1205 | 0.1322 | 0.1390 |
| FA | 2027 | 0.0715 | 0.2117 | 0.2991 | 0.3976 | 0.1427 | 0.1709 | 0.1958 | 0.1201 | 0.1317 | 0.1385 |
| FP | 2026 | 0.0718 | 0.2112 | 0.2994 | 0.3976 | 0.1430 | 0.1714 | 0.1962 | 0.1205 | 0.1322 | 0.1390 |
| FP | 2027 | 0.0715 | 0.2117 | 0.2990 | 0.3976 | 0.1428 | 0.1709 | 0.1958 | 0.1201 | 0.1317 | 0.1385 |
| PF | 2026 | 0.0718 | 0.2114 | 0.2994 | 0.3977 | 0.1430 | 0.1714 | 0.1962 | 0.1206 | 0.1323 | 0.1391 |
| PF | 2027 | 0.0715 | 0.2118 | 0.2992 | 0.3976 | 0.1428 | 0.1710 | 0.1958 | 0.1202 | 0.1317 | 0.1385 |
| FA_Q8 | 2026 | 0.0714 | 0.2109 | 0.2988 | 0.3972 | 0.1426 | 0.1710 | 0.1958 | 0.1202 | 0.1318 | 0.1386 |
| FA_Q8 | 2027 | 0.0714 | 0.2109 | 0.2981 | 0.3974 | 0.1424 | 0.1706 | 0.1956 | 0.1199 | 0.1315 | 0.1384 |
| FA_1024 | 2026 | 0.0156 | 0.0544 | 0.0829 | 0.1180 | 0.0351 | 0.0443 | 0.0531 | 0.0288 | 0.0325 | 0.0349 |
| FA_1024 | 2027 | 0.0175 | 0.0514 | 0.0788 | 0.1135 | 0.0345 | 0.0433 | 0.0521 | 0.0290 | 0.0326 | 0.0350 |
| DP8 | 2026 | 0.0162 | 0.0542 | 0.0824 | 0.1170 | 0.0353 | 0.0443 | 0.0531 | 0.0291 | 0.0328 | 0.0352 |
| DP8 | 2027 | 0.0159 | 0.0514 | 0.0776 | 0.1104 | 0.0339 | 0.0423 | 0.0506 | 0.0281 | 0.0316 | 0.0338 |
| DP1 | 2026 | 0.0149 | 0.0522 | 0.0799 | 0.1147 | 0.0337 | 0.0426 | 0.0513 | 0.0276 | 0.0312 | 0.0336 |
| FA_WARM | 2026 | 0.0926 | 0.2565 | 0.3601 | 0.4715 | 0.1764 | 0.2098 | 0.2380 | 0.1501 | 0.1638 | 0.1715 |
| FA_WARM | 2027 | 0.0930 | 0.2558 | 0.3596 | 0.4705 | 0.1763 | 0.2098 | 0.2379 | 0.1502 | 0.1640 | 0.1717 |
| FA_WARM_FROZEN | 2026 | 0.0936 | 0.2568 | 0.3609 | 0.4709 | 0.1772 | 0.2108 | 0.2386 | 0.1510 | 0.1648 | 0.1725 |
| FA_WARM_FROZEN | 2027 | 0.0933 | 0.2576 | 0.3611 | 0.4721 | 0.1774 | 0.2108 | 0.2389 | 0.1510 | 0.1648 | 0.1725 |
| FP_WARM_FROZEN | 2026 | 0.0936 | 0.2568 | 0.3609 | 0.4709 | 0.1772 | 0.2108 | 0.2386 | 0.1510 | 0.1648 | 0.1725 |
| FP_WARM_FROZEN | 2027 | 0.0933 | 0.2576 | 0.3611 | 0.4721 | 0.1774 | 0.2108 | 0.2389 | 0.1510 | 0.1648 | 0.1725 |
| PF_WARM_FROZEN | 2026 | 0.0934 | 0.2570 | 0.3611 | 0.4722 | 0.1771 | 0.2107 | 0.2388 | 0.1509 | 0.1647 | 0.1724 |
| PF_WARM_FROZEN | 2027 | 0.0933 | 0.2577 | 0.3613 | 0.4722 | 0.1774 | 0.2109 | 0.2389 | 0.1510 | 0.1648 | 0.1725 |
| FA_Q8_WARM_FROZEN | 2026 | 0.0934 | 0.2571 | 0.3610 | 0.4721 | 0.1772 | 0.2107 | 0.2388 | 0.1509 | 0.1647 | 0.1724 |
| FA_Q8_WARM_FROZEN | 2027 | 0.0934 | 0.2576 | 0.3612 | 0.4721 | 0.1774 | 0.2108 | 0.2389 | 0.1511 | 0.1648 | 0.1725 |
| FA_1024_WARM_FROZEN | 2026 | 0.0937 | 0.2580 | 0.3610 | 0.4711 | 0.1778 | 0.2110 | 0.2389 | 0.1515 | 0.1651 | 0.1727 |
| FA_1024_WARM_FROZEN | 2027 | 0.0936 | 0.2580 | 0.3608 | 0.4710 | 0.1778 | 0.2110 | 0.2388 | 0.1514 | 0.1651 | 0.1727 |
| DP8_FROZEN | 2026 | 0.0937 | 0.2580 | 0.3609 | 0.4711 | 0.1778 | 0.2110 | 0.2389 | 0.1515 | 0.1651 | 0.1727 |
| DP8_FROZEN | 2027 | 0.0936 | 0.2580 | 0.3608 | 0.4711 | 0.1778 | 0.2109 | 0.2388 | 0.1514 | 0.1650 | 0.1727 |
| DP1_WARM_FROZEN | 2026 | 0.0936 | 0.2579 | 0.3609 | 0.4712 | 0.1778 | 0.2110 | 0.2389 | 0.1514 | 0.1651 | 0.1727 |
| FT_C | 2026 | 0.0850 | 0.2375 | 0.3297 | 0.4313 | 0.1631 | 0.1929 | 0.2186 | 0.1387 | 0.1509 | 0.1580 |
| FT_C | 2027 | 0.0863 | 0.2376 | 0.3293 | 0.4304 | 0.1638 | 0.1933 | 0.2189 | 0.1395 | 0.1516 | 0.1587 |
| FT_PRE | 2026 | 0.0975 | 0.2632 | 0.3669 | 0.4765 | 0.1823 | 0.2157 | 0.2434 | 0.1556 | 0.1694 | 0.1770 |
| FT_PRE | 2027 | 0.0971 | 0.2634 | 0.3665 | 0.4765 | 0.1821 | 0.2154 | 0.2432 | 0.1554 | 0.1691 | 0.1768 |
| FT_FA | 2026 | 0.0773 | 0.2189 | 0.3064 | 0.4039 | 0.1498 | 0.1780 | 0.2027 | 0.1271 | 0.1387 | 0.1455 |
| FT_FA | 2027 | 0.0770 | 0.2192 | 0.3057 | 0.4047 | 0.1496 | 0.1775 | 0.2025 | 0.1267 | 0.1382 | 0.1451 |
| FT_FA_WARM | 2026 | 0.0961 | 0.2625 | 0.3663 | 0.4766 | 0.1811 | 0.2146 | 0.2425 | 0.1544 | 0.1682 | 0.1758 |
| FT_FA_WARM | 2027 | 0.0961 | 0.2616 | 0.3659 | 0.4760 | 0.1808 | 0.2145 | 0.2423 | 0.1543 | 0.1681 | 0.1758 |
| FT_FA_WARM_FROZEN | 2026 | 0.0966 | 0.2629 | 0.3676 | 0.4767 | 0.1816 | 0.2154 | 0.2430 | 0.1549 | 0.1688 | 0.1764 |
| FT_FA_WARM_FROZEN | 2027 | 0.0967 | 0.2633 | 0.3676 | 0.4777 | 0.1819 | 0.2155 | 0.2434 | 0.1552 | 0.1690 | 0.1766 |
| LO | 2026 | 0.0030 | 0.0066 | 0.0084 | 0.0103 | 0.0049 | 0.0055 | 0.0060 | 0.0043 | 0.0046 | 0.0047 |
| LO | 2027 | 0.0034 | 0.0075 | 0.0094 | 0.0117 | 0.0055 | 0.0061 | 0.0067 | 0.0049 | 0.0051 | 0.0053 |
| LAST | — | 0.0003 | 0.1572 | 0.2056 | 0.2418 | 0.0859 | 0.1016 | 0.1109 | 0.0620 | 0.0685 | 0.0711 |
| SKNN | — | 0.0355 | 0.1992 | 0.2904 | 0.3823 | 0.1192 | 0.1487 | 0.1720 | 0.0929 | 0.1051 | 0.1115 |
| KNN_EMB_BIG | — | 0.0443 | 0.1234 | 0.1657 | 0.2143 | 0.0855 | 0.0992 | 0.1114 | 0.0730 | 0.0786 | 0.0820 |
| KNN_EMB_SMALL | — | 0.0393 | 0.1084 | 0.1518 | 0.2033 | 0.0747 | 0.0887 | 0.1017 | 0.0637 | 0.0694 | 0.0729 |
| RULE | — | 0.0154 | 0.0579 | 0.0932 | 0.1421 | 0.0367 | 0.0480 | 0.0603 | 0.0297 | 0.0343 | 0.0377 |
| POP | — | 0.0081 | 0.0295 | 0.0491 | 0.0750 | 0.0187 | 0.0249 | 0.0314 | 0.0151 | 0.0176 | 0.0194 |

(LAST's HR@1 ≈ 0 is expected: the target is the next *different* item, and LAST's top-1 is the last viewed item.)

## 5. Panels (DESCRIPTIVE; seed 2026; REPORT §6 and METRIC_TABLE)

Retention vs C_BIG on the same panel (PRACTICAL_BEST, REPORT §6):

| Panel (users / decisions) | C_FULL | C_SMALL | FA | FA_WARM_FROZEN | Other |
|---|---|---|---|---|---|
| UNSEEN_HOLDOUT — never-trained users of the pool (10,224 / 132,225) | 125.40 | 96.28 | 87.63 | 109.28 | FT_C 99.36; LO 2.85 |
| TRUE_COLD — users with 0 TRAIN examples (381,295 / 2,857,665) | not admitted | 96.12 | 86.30 | 104.63 | — |
| SPARSE — 1–4 TRAIN examples (16,384 / 129,696) | 117.56 | 96.82 | 86.89 | 104.69 | — |
| HEAVY — > 100 TRAIN examples (16,384 / 617,944) | 132.54 | 95.41 | 85.38 | 113.40 | — |

Absolute macro MRR@20 / NDCG@10 / HR@10 on the panels (METRIC_TABLE; PB unless "CB"):

| Panel | C_BIG | C_FULL (CB = PB) | C_SMALL | PRE (CB) | FA | DP8 (CB) | FA_WARM_FROZEN | DP8_FROZEN (CB) |
|---|---|---|---|---|---|---|---|---|
| UNSEEN_HOLDOUT | 0.1579 / 0.1928 / 0.3285 | 0.1980 / 0.2381 / 0.3915 | 0.1520 / 0.1857 / 0.3186 | 0.1736 / 0.2116 / 0.3603 | 0.1383 / 0.1710 / 0.2992 | 0.0353 / 0.0443 / 0.0816 | 0.1725 / 0.2107 / 0.3601 | 0.1728 / 0.2108 / 0.3593 |
| TRUE_COLD | 0.1551 / 0.1892 / 0.3226 | — | 0.1491 / 0.1826 / 0.3156 | — | 0.1338 / 0.1649 / 0.2888 | 0.0442 / 0.0558 / 0.1028 | 0.1622 / 0.1987 / 0.3428 | 0.1639 / 0.2006 / 0.3456 |
| SPARSE | 0.1675 / 0.2035 / 0.3433 | 0.1970 / 0.2377 / 0.3938 | 0.1622 / 0.1972 / 0.3343 | 0.1764 / 0.2154 / 0.3681 | 0.1456 / 0.1787 / 0.3096 | 0.0456 / 0.0572 / 0.1043 | 0.1754 / 0.2143 / 0.3668 | 0.1764 / 0.2150 / 0.3669 |
| HEAVY | 0.1219 / 0.1503 / 0.2629 | 0.1616 / 0.1969 / 0.3336 | 0.1163 / 0.1441 / 0.2554 | 0.1380 / 0.1701 / 0.2986 | 0.1041 / 0.1294 / 0.2318 | 0.0168 / 0.0213 / 0.0404 | 0.1383 / 0.1707 / 0.3005 | 0.1372 / 0.1693 / 0.2979 |

UNSEEN_HOLDOUT also has FT_C 0.1569 / 0.1905 / 0.3233 and LO 0.0045 / 0.0053 / 0.0083. C_FULL s2027 panel rows exist as
DESCRIPTIVE (B2, DL 365). Device arms on SPARSE / HEAVY / TRUE_COLD were declared not run (DL 317 D3).

## 6. LAST anchor, device-only envelope, repeat vs novel targets (REPORT §8, §11)

- Δ(arm − LAST): C_BIG +0.0901; FA +0.0676 (K11b); FA_WARM_FROZEN +0.1014; PRE +0.1018; FT_PRE +0.1058. Headroom R* vs
  C_BIG: FA 75.08 %, FA_WARM_FROZEN 112.49 %, FT_PRE 117.39 %.
- **BELOW_DEVICE_HEURISTIC = True** (CI of Δ vs LAST below 0) for **FA_1024, DP8, DP1** (cold DP chain), LO, RULE and POP.
- Device-only envelope {LO, LAST, RULE}: best point = LAST (44.11 % of C_BIG). LO is an untuned from-θ0 control, not the
  device-only ceiling.
- REPEAT targets (target already in the user's 50-event window; 28.1 % of E2E rows) vs NOVEL targets: FA keeps 75.99 % of
  C_BIG on REPEAT but 98.60 % on NOVEL; LAST keeps 88.14 % on REPEAT and 0 on NOVEL; FT_PRE 116.17 % / 104.05 %. The cold
  FL loss is concentrated on repeat-item targets (DESCRIPTIVE).
- CALIB-disjoint diagnostic (MAIN \ CALIB, 30,694 E2E users): every CALIB-tuned arm keeps the same retention within about
  0.5 points (e.g. K2 86.05 vs 86.07; K20b 88.70 vs 88.81; K21 −0.000465), so CALIB tuning did not leak into the MAIN read.

## 7. Privacy, compression and quantisation findings

- **Cold DP chain.** The Poisson configuration that DP needs (about 1,024 clients per round, 384 rounds, clipping) is itself
  weak on a random start: FA_1024 (no noise) reaches only 21.68 % of C_BIG, below LAST. DP8 / DP1 keep 98.71 % / 96.27 % of
  that twin at CONTROLLED_BUDGET (K12 is SEED_SENSITIVE: 100.70 vs 96.72). So the measured DP cost is small *relative to a
  weak twin*; the large loss comes from the DP-compatible training schedule (INFERENCE). The registered note (06D §3; 06I
  §6(b)) applies: with d = 20.7 M parameters, m = 1,024 and z = 0.5642 the per-round noise norm is ≈ 2.5 · S vs a signal ≤ S,
  a property of the simulated scale.
- **Warm DP chain.** DP8_FROZEN and DP1_WARM_FROZEN keep 99.96 % / 99.88 % of their z = 0 twin (K12w includes 100; K13w just
  below), and all of them sit at PRE's level. **Caveat (D1, §9):** the warm DP runs moved θ by only 0.013–0.077 % of θ_pre,
  so "DP costs nothing here" mostly reflects that warm FL barely changes the pretrained model; it is not evidence that DP is
  free in general.
- **8-bit uploads** cost nothing measurable: K14 99.82 % and K14w 99.99 %, both EQUIVALENT under the registered ±1 % TOST.
  FA_Q8 cuts training bytes per client from 994.6 MB to 621.6 MB (cold) and S8's from 94.4 MB to 89.7 MB.
- **Frozen item tables (S8)** cut per-visit traffic from 165.7 MB to 2.11 MB (ratio 0.0127; 263,822 uploaded parameters
  MEASURED vs the registered 265,162 "to be re-measured") and did not cost quality (S8 − S7 = +0.000854).
- **On-device INT8 (K16).** Dynamic INT8 ONNX keeps 99.84 % [99.62, 100.07] of FP32 ONNX for C_SMALL on the 16,384-user
  subset (ONNX-CPU block; onnxruntime 1.24.1, opset 17). The device **latency** number is K16 pass 2 (PENDING; pass-1
  latency was measured on a loaded host and is never the device number; DL 368).
- Privacy ladder (REPORT §4, §9): L0 cloud C_BIG 100 / C_FULL 120.6; L1 shipped C_SMALL ≈ 95.7, PRE alone 107.3; L2 FL 86.1
  (cold) or 107.0 (warm, S8); L3 DP 21.4 (cold) or 107.1 (warm, DP8_FROZEN); L4 FT_FA 90.1 / FT_FA_WARM_FROZEN 109.5; L5 FT_PRE
  109.7; L6 LO 3.1 / LAST 44.1 (all % of C_BIG).

## 8. Personalisation, PF, FedProx

- Local fine-tuning helps every base: FT_C − C_SMALL +0.0039 (K9); FT_FA − FA +0.0065 (K10); FT_PRE − PRE +0.0040 (K9c);
  FT_FA_WARM_FROZEN − FA_WARM_FROZEN +0.0040 (P5); for never-seen users +0.0049 (K9b). Harmed-client shares are 31–37 %
  (a seed-vs-seed noise floor of about 0.23–0.40 is printed beside them, REPORT §1c), so the gain is an average.
- **PF ≈ FA** (PF − FA +0.00003, CI includes 0) and **FP ≈ FA** — both registered possible outcomes (06L A5 / A7): with
  FedProx μ = 0.01 and a personal vector in a model whose state is dominated by item tables, neither changes much.

## 9. The D1 diagnostic (MEASURED, exploratory; checkpoints only; DL 407)

After the read, the owner asked whether the warm FL had been done properly. Ops measured the relative parameter distance
‖θ − θ_pre‖ / ‖θ_pre‖ over all 50 tensors of the warm checkpoints:

| Run | best.pt | endpoint 6.0 |
|---|---|---|
| FA_WARM_FROZEN (S8), both seeds | 0.53 % | 0.67 % |
| FA_WARM (S7) | 0.09 % | 0.26 % |
| DP8_FROZEN | 0.013 % | 0.077 % |
| FA_1024_WARM_FROZEN | 0.004 % | 0.030 % |

The change concentrates in small gate / bias / LayerNorm tensors (up to 20–45 % relative); the large attention and
feed-forward matrices barely move; S8's item tables are unchanged by design. With FedAdam at server LR 0.001 the per-step
size is about the LR whatever the gradient, so small tensors move and big ones do not, and the best checkpoints were
selected near the start. **Interpretation (INFERENCE, DL 407; PREG2 §1):** the warm FL was close to a no-op around θ_pre;
K21 ≈ 0 is a property of the registered recipe (a cold-start recipe applied to fine-tuning, with a screen that only moved
the client LR upward and stopped at the smallest server LR), **not proof that federated learning cannot add value**.
Caveat (PREG2 §3.2): the "all 50 tensors" norm is dominated by the frozen item tables; Phase 2 defines the no-op test on the
trained subset only.

## 10. Honest interpretation — what can and cannot be said (INFERENCE, to be confirmed or not by TEST)

Can be said (on VALIDATION, under the registered recipe and checkpoint rule):
1. A same-users cloud model (C_BIG) loses 4.2 % when shrunk to the device size (K1 95.80 %).
2. Cold federated training of the device model keeps 86.1 % of C_BIG (K2) and about 90 % of the same-size cloud model (K3),
   and clearly beats every device-only heuristic (K11b +0.068 over LAST).
3. Pretraining on separate opted-in users is the single biggest lever: θ_pre alone is better than C_BIG (107.3 %) because
   it saw 9× more users; starting FL from it adds +0.034 MRR@20 over cold FL (K_INIT).
4. The deployable warm federated system keeps 88.8 % of the full-data cloud model (K20b), essentially the same as θ_pre
   alone (89.0 %, K20c); federated rounds on the private users did not add quality (K21 slightly negative).
5. The best deployable system on VALIDATION is a download-only one: θ_pre + on-device fine-tuning (FT_PRE, 109.7 % of C_BIG;
   +0.004 over PRE), which needs no federated traffic at all (L5).
6. 8-bit uploads and frozen item tables cut traffic massively at no measurable cost; INT8 inference shows no detectable
   accuracy loss on CPU.
7. DP at ε = 8 / ε = 1 costs little relative to its non-private twin, but the DP-compatible schedule on a random start is
   weak (≈ 21 % of C_BIG, below LAST).

Must not be said:
- "FL retains X %" without naming the recipe and reference (06I §6(c)); "a lower bound" (06L_C C6).
- "Federated learning cannot add value over pretraining" — D1 shows the recipe barely trained (that is Phase 2's question).
- "SASRec beats GRU" — the backbone was selected under rule T1 (and the order reverses at the 6.0 endpoint; DL 138).
- "Private" for any non-DP arm; "DP is free" (see §7 caveat).
- Any TEST statement, or any "events to parity" sentence (no persistent hit exists; TRUNC not run).

Not available in this read: the VALIDATION → TEST shift indicator (NOT_APPLICABLE until TEST), no-FT references on
UNSEEN_VAL (not in the block), GRU_D128 vs C_SMALL (declared not run), the controlled cold-start curves (TRUNC not run),
K16 device latency (pass 2 pending).

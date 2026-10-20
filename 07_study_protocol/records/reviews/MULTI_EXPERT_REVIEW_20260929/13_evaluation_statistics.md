# 13 — Evaluation and statistics (part 12 of the multi-expert review, 2026-09-29)

Reviewer: evaluation-methodology / statistics expert (Claude Opus 5.5). READ-ONLY. I read contracts, gates, code and
reviews. I did not read any RETRO_TEST row, any downstream RESULT_ROWS value, any rank file or any runs_* evidence file,
and I ran no project script or test. The only numbers I quote are those allowed by the lead: G2 and DL 132 (MAIN
baselines and contest runs), the CALIB ladder screen values (FL_IMPROVEMENT_LADDER / DL 218), and structural counts
from DATA_PROTOCOL_LOCK_V2. Tags: MEASURED (path:line), REPORTED (a project document's own number), INFERENCE, OPINION.
Paths are relative to `.local_ai/ppsi_v4/`.

---

## 1. What the project does (MEASURED)

### 1.1 Metric and ranking
- **Ranking is exact, over the full catalogue.** All K = 158,486 classes are candidates. Nothing is filtered, and
  there are no sampled negatives, no ANN and no top-k (`contracts/EVALUATION_CONTRACT_V4.md:498-526`). The rank is
  `1 + #{s_j > s_t} + #{j < t : s_j == s_t}` (L486-496). Ranks above 20 are kept exactly (L528-534).
- **Ties and non-finite scores.** A pessimistic-rank tie diagnostic is reported (L536-541; PREG1_LOCK_VALUES T3
  `per_row_tie_diagnostic`, `contracts/PREG1_LOCK_VALUES.json:64`). Non-finite scores are refused (G1, L512-515).
- **Credits and aggregation.** Credits HR/MRR/NDCG@k are computed from int64 ranks in FP64. OOV targets get rank −1,
  which gives 0 credit, and they stay in the denominator (`code/v4eval/metrics.py:31-40`; contract L360-365, L443-453).
  Client-macro means the mean over users of each user's mean. It is grouped by eval_user_key (`metrics.py:48-63`;
  contract L455-461).
- **Metric roles.** The primary metric is macro MRR@20 on E2E. Macro NDCG@10 and HR@10 are prominent. The full grid is
  micro/macro × {MRR, HR, NDCG} × k ∈ {1, 5, 10, 20} × {E2E, RANKABLE} (contract L463-471; amendment 04
  `contracts/PREG1_AMENDMENT_04.md:37-41`). The macro/micro OOV identities are asserted with a counterexample fixture
  (contract L397-439).
- **Frozen numerics (T3).** Score block 158,486, class chunk 32,768, eval batch 1,024, strict FP32, FP64 accumulation.
  The frozen device class is A16 (`PREG1_LOCK_VALUES.json:53-67`).

### 1.2 Retention and bootstrap
- **Retention formula.** R = 100·M/C, with no clamping. A zero denominator is UNDEFINED. Every row carries a
  matched_central_row_id (contract L610-629). The answer format is RETENTION with no pass/fail threshold (amendment
  04 §1, `PREG1_AMENDMENT_04.md:9-15`; it supersedes amendment 03 §4, whose threshold was "PENDING the owner" and
  was never registered, `PREG1_AMENDMENT_03.md:34-37`, `PREG1_AMENDMENT_04.md:7-8`).
- **Bootstrap design.** It is a paired user bootstrap: 1,000 resamples, seed 20260926, and one shared resample matrix W
  per evaluated user set, registered by sha256 (contract L545-567).
  - W is `default_rng(20260926).integers(0, n, n)` per row (`code/v4eval/bootstrap.py:33-40`).
  - The statistics are exact weighted sums with the bincount multiplicities (L132-159).
  - The CI is the 95% percentile interval from `np.percentile` with linear interpolation (L162-164).
  - The user set is the E2E users of the view (`code/v4train/bootstrap_ci.py:42-43`).
- **Ratios are bootstrapped directly per resample.** The code never divides CI endpoints (contract L583-588;
  `tools/retention_report.py:1496-1546`).
- **Seed mean.** The point is the mean of the per-seed ratios. The CI is the percentile of the per-resample mean of
  the per-seed ratios on the same W (`retention_report.py:1537-1545`; disclosed 06E §9e,
  `contracts/PREG1_AMENDMENT_06E.md:60-61`).
- **Pairing rule (06A §7).** Numerator and denominator use the same checkpoint rule and the same seed, then the 2-seed
  mean (`contracts/PREG1_AMENDMENT_06A.md:36-38`). This is enforced by `check_pairs`, which also enforces the same
  role and user subset (`retention_report.py:1474-1493`).
- **Harmed-client share.** It is computed only for **delta** contrasts: the share of users whose per-user macro delta
  (seed mean) is below 0, plus the improved and tied shares (`retention_report.py:1549-1583`). Ratio contrasts print
  "-" in the harmed-share column (`retention_report.py:2658`). Amendment 04 lists the harmed-client share among the
  reported metrics (`PREG1_AMENDMENT_04.md:41`).

### 1.3 Contrasts, hierarchy and multiplicity
- **Contrasts.** K1–K16 come from 06B §2 (`contracts/PREG1_AMENDMENT_06B.md:25-48`), K17–K19 from 06I §1
  (`contracts/PREG1_AMENDMENT_06I.md:6-15`), and K20/K21 from 06K §2 (`contracts/PREG1_AMENDMENT_06K.md:18-26`).
- **Pre-lock registry.** The pre-lock registry file is `company/gates/ANA_CONTRASTS_v2.json` with **19** contrasts
  (L8-104). Its hierarchy is: primary K2; secondary K1, K3, K8, K12, K17; tertiary the rest (L116-142). The rule map is
  K12/K13 → CONTROLLED_BUDGET (L105-108). K13 and K15–K19 are registered as single-seed (L144-163).
- **No v3 file yet.** No `ANA_CONTRASTS_v3` file (K1–K21) exists in `company/gates/` (MEASURED by `ls`). The script
  labels K20/K21 DESCRIPTIVE when the loaded file is v2 (`retention_report.py:1900-1930`).
- **Multiplicity policy (06D §4).** "All sixteen are reported with unadjusted 95% paired user-bootstrap CIs … a
  Holm/Bonferroni-adjusted 99.7% percentile is not resolvable with 1,000 resamples"
  (`contracts/PREG1_AMENDMENT_06D.md:74-78`). The v2 file repeats this for 19 contrasts (`ANA_CONTRASTS_v2.json:143`).

### 1.4 Checkpoint rules, device class and TEST
- **Checkpoint rules.** PRACTICAL_BEST is the strictly-best of the 12 marks in [0.5, 6.0]. CONTROLLED_BUDGET is the
  6.0 endpoint (contract L838-861). PB is primary. K12/K13 are confirmatory under CB/CB, because best-of-12 selection
  on the users' own validation data sits outside the DP accountant (06D §3, L53-60). VALIDATION values are
  max-of-marks, and TEST is the only unbiased number (06A §9g, L47).
- **Device class.** The A16 is the frozen class. A second frozen class, NVIDIA_A40-FP32-DET, is registered by a pin
  file. Rows of the two classes never mix inside a retention block, and C_BIG and C_SMALL are re-evaluated on the A40
  (06D §1, L13-30; `retention_report.py:128-135, 1129-1195`). The A16 and A40 ranks differ on 12,635 / 14,063 M02 rows
  for random-init models (06E §9h, L63-64).
- **TEST.** There is one look after ALL_METHODS_LOCK. The register is append-only and hash-chained (contract
  L655-688). TEST manifests are built only after the lock (06D §2, L32-50). Register entry 3 records that the v1
  campaigns observed legacy-TEST metrics, so V4 TEST is "retrospective" (`company/registers/TEST_EXPOSURE_REGISTER.json`
  entry 3; `contracts/DATA_PROTOCOL_LOCK_V2.json:80`).
- **Black Friday and the shift indicator.** RETRO_TEST (11-23 → 11-30) contains Black Friday, 29 Nov 2019 (06B §7b,
  L93). The only shift quantity is the VALIDATION→TEST change of the C_BIG level, which is unpaired
  (`retention_report.py:2389-2407`). TEST has no sub-period field (`contracts/MANIFEST_SPEC_V4.json:396`:
  "subperiod = NONE").

### 1.5 Baselines and sizes
- **Baselines.** POP, LAST, SKNN, RULE and KNN_EMB enter the analysis as `historical_reference` + RULE_FREE CPU rows,
  with no rank files (`tools/ana_block_index.py:33-36`). They are therefore point-only, with CI "NO_RANKS"
  (`retention_report.py:1426-1432`).
- **MAIN baseline values** (REPORTED, DL 132): SKNN 0.11138, LAST 0.07111, POP 0.01943 macro MRR@20 on
  MAIN_VALIDATION. The C_BIG candidates are SASRec D256_B3 PB 0.16120 / 0.16122 (@3.0) and CB 0.15287 / 0.15258
  (`company/DECISION_LOG.md:136`).
- **Sizes** (MEASURED): MAIN_VALIDATION = 51,211 users, 644,694 decisions, 519,714 eval rows. CALIB_VALIDATION =
  12,715 users (`contracts/DATA_PROTOCOL_LOCK_V2.json`, the manifest block after L240).
- **Read status** (REPORTED): no MAIN downstream value has been read. Only the CALIB screens have been read, and the ANA
  freeze file does not exist yet (`company/DECISION_LOG.md` rows 214, 218-225; `ls company/gates`). The rules below
  can therefore still be registered **before** their governed values are read.

### 1.6 Code check (retention_report.py, ana_block_index.py) — correctness of ratio, pairing and join

I found no bug. The mechanics are correct.

- **Pairing is exact.** One W per view; `_check_pairing` requires identical sorted user keys (`bootstrap.py:111-114`).
- **Subset views work.** They mask `eval_mask` rather than cutting ranks, so a full-population denominator is
  evaluated on exactly the numerator's subset (`retention_report.py:979-983`).
- **Rank files are bound to their rows.** They reproduce the macro values within 1e-9 (L986-999).
- **Ratios are handled correctly.** Zero-denominator resamples make the CI UNDEFINED and no resample is dropped
  (L1521-1528).
- **Seed pairing covers the edge cases.** The shared-seed PRE and the RULE_FREE cases are handled (L1466-1493,
  L1605-1621). A seed that appears twice is refused (L1492-1493).
- **The block-index builder copies evidence and never parses it.** Only LAUNCH, RUN_SUMMARY, STATUS, REEVAL_RECORD,
  BENCH_RESULT and rows.meta are parsed (`ana_block_index.py:9-13, 68-70`).

---

## 2. State of the art (literature + real systems) — verified citations only

| Topic | Source (verified by web search) | What it says that matters here |
|---|---|---|
| Sampled vs full ranking | W. Krichene, S. Rendle, "On Sampled Metrics for Item Recommendation", KDD 2020, doi:10.1145/3394486.3403226 | Sampled metrics are inconsistent with exact metrics and can flip A-vs-B orderings. Full-catalogue ranking is the fix. |
| Same | S. Rendle, "Evaluation Metrics for Item Recommendation under Sampling", arXiv:1912.02263 (2019) | As the sample size shrinks, sampled metrics collapse toward AUC. |
| Sequential-rec sampling | A. Dallmann, D. Zoller, A. Hotho, "A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models", RecSys 2021, doi:10.1145/3460231.3475943 | Uniform and popularity sampling rank sequential models (SASRec etc.) differently from full evaluation. |
| Baselines | M. Ferrari Dacrema, P. Cremonesi, D. Jannach, "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches", RecSys 2019, doi:10.1145/3298689.3347058 | Neural methods must be compared against well-tuned simple baselines. Six of seven reproducible methods were often beaten. |
| Effect sizes / CIs in IR | T. Sakai, "Statistical Reform in Information Retrieval?", SIGIR Forum 48(1):3-12, 2014 | Report effect sizes and CIs, not only p-values. |
| Significance in ML/NLP | R. Dror, G. Baumer, S. Shlomov, R. Reichart, "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing", ACL 2018, aclanthology.org/P18-1128 | Choose and report the test for the experimental setup. Multiple comparisons and repeated datasets must be handled. |
| Variance sources | X. Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks", MLSys 2021 (proceedings.mlsys.org; arXiv:2103.03098) | Data sampling, initialisation and hyper-parameter choice all change results markedly. Randomise several sources, because a single-source CI understates the variance. |
| Bootstrap | B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993 | B ≈ 1,000 is adequate for percentile CIs. Far fewer resamples suffice for standard errors. Ratio statistics are bootstrapped directly (plug-in). |
| Multiplicity with dependent statistics | J. P. Romano, M. Wolf, "Stepwise Multiple Testing as Formalized Data Snooping", Econometrica 73(4):1237-1282, 2005, doi:10.1111/j.1468-0262.2005.00615.x | A bootstrap max-statistic step-down controls the familywise error. It uses the joint dependence of the statistics, which a shared resample matrix provides for free. |
| FL evaluation practice | J. Wang, Z. Charles, Z. Xu, et al., "A Field Guide to Federated Optimization", arXiv:2107.06917 (2021) | Report distributions over clients, several seeds and tuned baselines, and evaluate federated simulations as a deployment would. |
| Per-client fairness in FL | T. Li, M. Sanjabi, A. Beirami, V. Smith, "Fair Resource Allocation in Federated Learning", ICLR 2020 (openreview ByexElSYDr) | FL quality is reported as a distribution over clients (variance, worst percentiles), not only a mean. |
| Fresh data availability | REES46 datasets page (rees46.com/en/datasets): "Multi-category Behavior events Oct 2019 – Apr 2020" | A never-inspected post-November window exists for the same store. |

**Real systems (INFERENCE / OPINION, no citation claimed).** Production recommenders validate offline with exact
top-k over the full candidate set on a time-forward holdout. Then they run online A/B tests with per-user
randomisation, user-clustered variance, CUPED-style variance reduction and pre-registered guardrail metrics, including
"share of users harmed". Federated keyboard or recommendation deployments use federated evaluation, where clients
compute metrics on held-out local data and only aggregates return. That is what the PRACTICAL_BEST marks on
MAIN_VALIDATION emulate. The project's offline design is close to best industrial offline practice. Only the online
step, which is impossible here, is missing.

---

## 3. Comparison with the state of the art

### 3.1 Strengths (well above a typical RecSys or MSc evaluation)

- **S1 — ranking.** It uses the full catalogue with exact integer ranks, a deterministic adverse tie rule, a
  pessimistic-rank sensitivity and a refusal of non-finite scores. This is exactly what Krichene & Rendle 2020 and
  Dallmann et al. 2021 ask for, and at K = 158,486 it is expensive and rarely done.
- **S2 — denominator.** The E2E denominator gives OOV targets zero credit, and the macro identity is asserted with a
  counterexample fixture. The primary is client-macro. That is the right unit for an FL / per-device question: each
  device counts once, which matches "per-client" FL reporting.
- **S3 — bootstrap.** The paired bootstrap uses one shared W per user set, registered by hash. Ratios are bootstrapped
  per resample, the seed mean is taken on the same W, and zero denominators are UNDEFINED rather than dropped. This is
  textbook plug-in bootstrap and is implemented correctly (§1.6).
- **S4 — pre-registration discipline.** Contrasts, rule maps, seeds and hierarchy were registered before any
  governed read. The registry file is refused if its text differs. There is a hash-chained TEST register, an O_EXCL
  lock and one TEST look with a registered procedure for the device arms. The single TEST look means
  best-of-marks selection on VALIDATION does not bias the reported TEST number (06A §9g).
- **S5 — device blocking.** A16 and A40 rows are never mixed. The central references are re-evaluated on the A40, and
  the A16 values are kept as a device-numerics sensitivity. The ONNX-CPU sub-block is separate for K16. This is more
  careful than the literature norm (OPINION): device numerics are real (12,635 of the M02 rows differ for a
  random-init model) but almost certainly small for trained models (INFERENCE).
- **S6 — honest DP rule.** CONTROLLED_BUDGET is confirmatory for K12/K13 because the accountant does not cover
  best-of-12 selection (06D §3).
- **S7 — estimation, not a threshold.** Dropping the unregistered acceptability threshold (amendment 04) in favour of
  retention with CIs is the Sakai / "statistical reform" position. Keep it.

### 3.2 Weaknesses, errors and missing pieces

**W1 — the CIs capture user sampling only, not training randomness (MAJOR).**
- The contract says so (L606: "conditional on the trained models"). But the claim vocabulary "A exceeds B iff CI > 0"
  (L600-603) is applied to 2-seed means, and to one seed for K13 and K15–K19, as if the CI were total uncertainty.
- *Scale* (INFERENCE from G2):
  - The G2 seed-mean Δ CI [−0.00518, −0.00326] on MAIN implies a paired SE of the Δ of about 0.00049.
  - The central seed-to-seed gaps are 0.00002 (B3 PB), 0.00053 (B2 PB) and 0.00029 (B3 CB), from DL 132.
  - So the seed noise of a central model is of the **same order as the paired user-bootstrap SE of a contrast**. For
    that kind of contrast, ignoring it could understate the SD by up to about √2.
  - FL seeds also randomise client sampling and drop-out order. Seed variance there is plausibly larger (Bouthillier
    et al. 2021).
- With two seeds, the between-seed variance has one degree of freedom. It cannot be estimated well, but it can be shown
  and bounded.

**W2 — the multiplicity handling is only a stated hierarchy (MEDIUM).**
- **The reason given is wrong.** 06D §4 says Holm/Bonferroni is "not resolvable with 1,000 resamples". That holds only
  for *percentile* endpoints at 99.7%. It is false for (a) normal intervals from the bootstrap SE, where B = 1,000
  estimates the SE to about ±2.2% (INFERENCE, since SE(ŝ)/s ≈ 1/√(2B)), and (b) a simultaneous max-|t| band à la
  Romano & Wolf 2005, whose critical value is the 95th percentile of the per-resample maximum. That is resolvable at
  B = 1,000 and uses the shared W's exact joint dependence.
- **The family size overstates independence.** Several contrasts are algebraic functions of others on the same W,
  exactly per seed and per resample (INFERENCE, from the definitions in `ANA_CONTRASTS_v2.json:8-104`):
  - K3_s = 100·K2_s / K1_s.
  - K11_s = (K2_s − K8_s)·C_BIG_s / 100.
  - K17 = K2(s2026)·K15 / 100, K18 = K7(s2026)·K15 / 100 and K19 = K8(s2026)·K15 / 100.
  - So of 21 "confirmatory" contrasts only about 15 are free quantities. **K17, the "co-headline", is not independent
    evidence.** It is K2's seed-2026 value times the cloud-data factor K15.
- **Consequence.** The thesis must not count K17 as a replication of K2. With a max-t band, the redundancy costs
  nothing.

**W3 — there is no absolute anchor with uncertainty, and a possibly sub-trivial FL arm (MAJOR for interpretation).**
- The MAIN baselines are LAST 0.07111 (a device-only, zero-training heuristic), SKNN 0.11138 and POP 0.01943, against
  C_BIG PB ≈ 0.1612 (DL 132). So LAST alone "retains" about 44% and SKNN about 69% (INFERENCE, arithmetic on REPORTED
  values).
- The registered CALIB FL screen is at 0.023–0.029 (S0 to S1b; `company/registers/FL_IMPROVEMENT_LADDER.md`, rows
  S0–S1b), against a central model at about 0.11–0.12 on CALIB (REPORTED context line). **That is below the LAST level
  measured on MAIN** (INFERENCE: CALIB ⊂ MAIN and LAST is not learned, so its CALIB value should be similar, though it
  is not measured).
- If MAIN FA ends near the CALIB level, then "FL retains ≈ 20–25%" is technically correct but uninformative, even
  misleading. A reader must see that a device-only repeat-last-item heuristic keeps about 44% with no training and no
  privacy cost (Ferrari Dacrema et al. 2019's lesson).
- The baselines enter only as point-only, no-rank, historical rows (`ana_block_index.py:33-36`). So no paired CI of
  FA − LAST or LO − LAST exists, and no baseline-normalised retention is registered.

**W4 — the harmed-client share is missing for the ratio contrasts, and there is no null reference (MEDIUM).**
- Amendment 04 §4 promises a harmed-client share. The code gives it only for deltas (K9, K10, K11, K21), not for the
  headline K2, nor for K1, K3, K8 or K17 (`retention_report.py:2658`).
- With about 10 eval rows per user (519,714 / 51,211), per-user MRR is very noisy. Even two seeds of the *same* arm
  will "harm" a large share of users (INFERENCE).
- A harmed share without a same-arm seed-vs-seed null (for example, C_BIG s2026 vs s2027 on the same W) cannot be
  interpreted.
- No per-client distribution is registered (quantiles of per-user Δ, worst decile), although that is the FL norm (Li et
  al. 2020; Wang et al. 2021).

**W5 — the registry is inconsistent about K20/K21, and the plan has a stale command (MUST fix, trivial).**
- 06K §2 registers K20/K21 as secondary K-contrasts (`PREG1_AMENDMENT_06K.md:22-24`). But the only pre-lock file is
  v2 (K1–K19), DL 214 (4) says the ANA freeze uses v2, and under v2 the script silently turns K20/K21 into DESCRIPTIVE
  (`retention_report.py:1900-1930`).
- MASTER_PLAN_V5 §9 still passes `company/gates/ANA_CONTRASTS.json`, the v1 file with K1–K16
  (`company/plans/MASTER_PLAN_V5.md:170-171`). The frozen script refuses it except as a "DRY CHECK ONLY" with
  `--allow-v1-contrasts`.

**W6 — there is a window between the VALIDATION read and ALL_METHODS_LOCK (MAJOR, process).**
- The order is: ANA preflight (first read), then ALL_METHODS_LOCK, then TEST (06B §6.5, `PREG1_AMENDMENT_06B.md:86-87`;
  MASTER_PLAN §9-§10). "No recipe, arm, selection or script changes" applies only **after** the lock
  (`MASTER_PLAN_V5.md:185-186`).
- Nothing registered forbids adding a ladder stage, an arm or a seed after seeing MAIN VALIDATION retention.
- The FL ladder is an open-ended, owner-driven search: "further stages … to register"
  (`FL_IMPROVEMENT_LADDER.md`, row "next"). The temptation is real. TEST would stay a single look, but the *arm set*
  could become outcome-dependent.

**W7 — FL tuning on CALIB ⊂ MAIN makes the VALIDATION K2 optimistic (MINOR, VALIDATION only).**
- The ladder selects many configurations on CALIB_VALIDATION: 12,715 users, which is 25% of the MAIN_VALIDATION users.
  VALIDATION K2 therefore includes the users on whom the FA recipe was selected.
- The central recipe (P2) was also CALIB-tuned, so the bias partly cancels. But the FL search is much larger (≥ 11 CALIB
  FL runs so far vs a 4-LR mini-ladder for C_SMALL). The TEST look removes the period overlap but not the design
  asymmetry.
- The G2 contest had a CALIB-disjoint diagnostic (condition 5). The retention report has none.

**W8 — winner's curse in the VALIDATION denominator (DISCLOSE).**
- C_BIG is the best family × candidate × mark on MAIN_VALIDATION (T1 on P3; PB @3.0). The VALIDATION denominator is a
  maximum, so VALIDATION R is biased downward.
- At TEST the checkpoint is fixed and the bias disappears. It is small, because the family gap was decisive (INFERENCE).

**W9 — TEST provenance and shift (MEDIUM, external validity).**
- The v1 campaigns observed metrics in the TEST window (register entry 3). The central families and the SASRec recipe
  have a lineage through those campaigns. The FL, device and DP numerators have never seen TEST.
- So adaptivity mostly touches the denominator side (OPINION: acceptable, and disclosed).
- Black Friday falls inside TEST. The only shift quantity is the unpaired C_BIG level change. Nothing registered checks
  whether *retention* moves under the shift, for example whether FL vs central degrade differently on 29-30 Nov. The
  user bootstrap cannot capture time-cluster shocks (a one-week window is one draw; contract L605-608).

**W10 — sensitivity to the checkpoint rule (DISCLOSE; already partly disclosed).**
- C_BIG PB (@3.0) is 0.1612 and CB (6.0) is 0.1529 (DL 132), a gap of about 5%.
- K2 CB/CB could therefore look about 5 R-points better for FL than PB/PB purely because the central model over-fits
  after 3.0 EFE (INFERENCE). Both are reported, which is good. The thesis must lead with PB/PB and explain the gap.

**W11 — E8.5 text vs K15/K17–K19 (MINOR, wording).**
- The contract says "A cohort method is never divided by a full-data central score" (`EVALUATION_CONTRACT_V4.md:627-628`),
  yet K15 and K17–K19 divide by C_FULL.
- The code evaluates C_FULL on the same MAIN view and W, which is the population-matching intent, so this is not a
  statistical error. But the text contradicts it and no amendment reconciles them.

**W12 — bootstrap interval type (MINOR).**
- The percentile interval with B = 1,000 is adequate for 95%. The Monte-Carlo SE of an endpoint is about 0.085 × the
  bootstrap SD (INFERENCE: √(p(1−p)/B) / φ(z_p) at p = 0.025).
- With n ≈ 51k users and smooth ratio-of-means statistics, percentile ≈ BCa ≈ normal. BCa is not needed.
- No bias, skewness or Monte-Carlo-error diagnostic is reported. That matters only when an endpoint sits within about
  0.1 SE of 100 or 0.

### 3.3 Unrealistic choices (OPINION)
- **One week of TEST containing Black Friday is a single time draw.** Retention on one shopping-peak week does not
  measure a typical week. A fresh December window exists for the same store (§2), but using it would need a new data
  version.
- **The retention is conditional on a very asymmetric tuning effort.** There is a large FL ladder against light central
  tuning. That is legitimate for "strongest realistic FL", but R is then "best-effort FL vs lightly-tuned central", and
  the thesis must say so.

---

## 4. Recommendations

Read status for every row: no governed downstream value has been read and the ANA freeze has not happened (DL
218-225). A new amendment (for example 06L) registered **before the ANA freeze and before the §9 preflight** is
therefore legitimate for every item. Nothing here touches G2 / T1d, which is read and final.

| ID | Item | Class | Action | Cost (INFERENCE) | Affects | Pre-registration |
|---|---|---|---|---|---|---|
| E1 | **Anchor retention on trivial baselines.** Dump rank files for POP / LAST / SKNN / RULE / KNN_EMB (CPU, deterministic re-run of v4base, or add a rank dump). Report paired Δ with CI for FA − LAST, LO − LAST, FT_C − LAST and C_SMALL − SKNN. Report a baseline-normalised retention R* = (arm − LAST)/(C_BIG − LAST) with CI, DESCRIPTIVE. Print a flag `BELOW_DEVICE_HEURISTIC` for any arm whose CI lies below LAST. | MUST | ADD | 2–4 h CPU + a small ANA change (+ tests, one review) | Finished baselines (CPU re-run only, no GPU); ANA script before freeze | Yes. Register now; governed values unread |
| E2 | **Fix the K20/K21 registry.** Write `ANA_CONTRASTS_v3.json` (K1–K21, 06K hierarchy/seeds) before the ANA freeze, or amend 06K to demote K20/K21 to DESCRIPTIVE. Update MASTER_PLAN §9 (L170-171) to the file actually frozen. | MUST | CHANGE | Minutes | ANA freeze, lock | Yes (a registration act). Unread |
| E3 | **Close the preflight-to-lock window.** Register that the arm list, the seeds, the FL/central recipes (the ladder's final selection file) and the checkpoint rules are fixed **before** the VALIDATION preflight is read. Any arm, seed or stage added after the read is labelled `POST_VALIDATION_EXPLORATORY`, is never confirmatory and never enters K1–K21. | MUST | ADD | Minutes | Process only | Yes. Unread |
| E4 | **Seed variance.** For every K, print per-seed R with CI, the seed spread \|R_2026 − R_2027\|, and a DESCRIPTIVE "user + seed" interval (bootstrap variance of the seed mean + s²_seed/n_seeds). Add the flag `SEED_SENSITIVE` when the seed spread exceeds the user-CI half-width. Restrict the claim vocabulary to contrasts not flagged, or state it. Register K15/K17–K19 with seeds {2026, 2027} if C_FULL s2027 completes (it is running, DL 216). A 3rd FA seed (the numerator of K2) is worth it if one MAIN FL slot is free. | SHOULD (high) | ADD | ANA change 1–2 h; a 3rd FA seed ≈ one MAIN FL run | ANA; optionally one new run | Yes. Unread |
| E5 | **Multiplicity done properly and cheaply.** Beside the unadjusted 95% CIs, add a DESCRIPTIVE simultaneous 95% band for the secondary family {K1, K3, K8, K12, K17} (and optionally all K). Use the max-\|t\| over the shared W with bootstrap-SE studentisation (Romano & Wolf 2005), or Holm on normal intervals from the bootstrap SE. Correct the 06D §4 sentence ("not resolvable"). List the algebraic dependencies (K3 = K2/K1, K11, K17–K19 = K×K15) and state that K17 is not independent of K2. Formally adopt fixed-sequence gatekeeping: K2 first at α = 0.05, then the secondary family. | SHOULD | ADD | 1–2 h code + tests | ANA before freeze | Yes. 06D §4's governed values are unread |
| E6 | **Harmed-client share for every K (ratios included), with a null.** Add the per-user Δ (arm − reference) share < 0 / > 0 / = 0, the quantiles P10 / P25 / P50 / P75 / P90 of the per-user Δ, and the worst-decile mean. Add a same-arm seed-vs-seed null (C_BIG s2026 vs s2027; FA s2026 vs s2027) on the same users. Put the paired absolute Δ (MRR points) with CI beside every R. | SHOULD | ADD | 1–2 h | ANA before freeze | Yes. Unread (amendment 04 §4 already promises the share) |
| E7 | **TEST shift sensitivity.** Pre-register the TEST sub-periods TEST_A [11-23, 11-29) and TEST_BF [11-29, 11-30) (Black Friday + Saturday) by decision_ts. Report K1, K2, K8 and C_BIG per sub-period (DESCRIPTIVE; each sub-period with its own W) and the per-sub-period change of R, not only of the C_BIG level. | SHOULD | ADD | A row-filter view in ANA (small) + DATA-09 already carries decision_ts | TEST analysis only | Yes, before ALL_METHODS_LOCK. Unread |
| E8 | **CALIB-disjoint diagnostic.** Report K2, K3 and K8 on MAIN_VALIDATION users outside CALIB (users_subset = MAIN \ CALIB), as the G2 condition-5 diagnostic did. DESCRIPTIVE, VALIDATION and TEST. | SHOULD | ADD | Trivial (users_subset exists) | ANA | Yes. Unread |
| E9 | **Bootstrap diagnostics.** For K2 (and every K whose endpoint is within 0.2 SE of 100 or 0), print the bootstrap bias, the skewness, the Monte-Carlo SE of each endpoint, and a BCa interval as a check. | NICE | ADD | < 1 h | ANA | Optional (descriptive) |
| E10 | **Fresh prospective holdout.** A never-inspected, post-Black-Friday window from the same store (REES46 publishes Oct 2019–Apr 2020): for example, decisions in 2019-12-02 → 12-09 for the MAIN users, history carried causally. Evaluate the locked checkpoints only (C_BIG, C_SMALL, FA, LO/FT_C re-run, baselines), for K1, K2, K8. Label PROSPECTIVE_HOLDOUT. It is the only way to escape the TEST reuse (register entry 3) and the one-shopping-week issue. | NICE (high value) | ADD | Days: a new data version, the pipeline on the Dec file, catalogue drift → OOV grows | New data; no finished run changes | Yes, and the owner must approve a new data version. Unread by construction |
| E11 | **E8.5 wording.** A one-line clarification: E8.5 forbids *population* mismatch. C_FULL evaluated on the MAIN view with the same W is allowed (K15, K17–K19). | SHOULD | CHANGE (text) | Minutes | Contract text | Yes (clarification) |
| E12 | **Disclosures.** (a) The CIs are conditional on the trained models, on the one TEST week and on the users (time clusters are not captured). (b) Percentile intervals and mean-of-ratios. (c) VALIDATION max-of-marks, plus the winner's curse in the C_BIG VALIDATION denominator. (d) TEST was previously inspected by the v1 central campaigns; the numerators never saw it. (e) R depends on the metric (MRR@20 vs HR@10). (f) Tuning asymmetry, FL ladder vs central mini-ladder. (g) PB/PB vs CB/CB sensitivity (C_BIG 0.1612 vs 0.1529, DL 132). (h) Baselines are CPU rows outside the A40 key. (i) K17 = K2(s2026) × K15. | DISCLOSE-ONLY | — | Write-up | Thesis ch. 8–9 | No |
| E13 | **Keep** full-K exact ranking, the tie rule and pessimistic sensitivity, E2E macro primary with OOV = 0, the shared-W paired bootstrap, direct ratio bootstrap, UNDEFINED zero denominators, the device-class blocking, the hash-chained TEST register and single look, the K12/K13 CB rule, and no acceptability threshold (estimation framing). | — | KEEP | 0 | — | — |

---

## 5. Score: 8 / 10

The evaluation core is excellent and in places exceeds published RecSys practice. It ranks exactly over the full
158k-item catalogue with an adverse tie rule and a pessimistic sensitivity. The denominator is honest (E2E, OOV = 0)
and the macro is the correct client-macro. The paired bootstrap is correctly implemented, with one shared,
hash-registered resample matrix, direct per-resample ratios and no dropped resamples. Device numerics are blocked.
Pre-registration is hash-enforced, and the single TEST look has an executable, gated path. I checked the ratio, pairing
and join code in `retention_report.py`, `ana_block_index.py` and `v4eval/bootstrap.py` and found no bug.

The points lost are about inference and interpretation, not arithmetic:
- The CIs ignore training randomness, and G2 shows central seed noise is of the same order as a contrast's paired SE.
- The multiplicity treatment rests on a wrong argument ("not resolvable") and counts algebraically redundant contrasts
  (K3, K11, K17–K19) as separate findings.
- The headline retention has no uncertainty-bearing anchor to the trivial device-only baseline (LAST ≈ 44% of C_BIG),
  although the CALIB screens suggest the FL arm may sit below it.
- The harmed-client share promised by amendment 04 is missing for the ratio contrasts and has no null.
- The registry and plan are inconsistent (K20/K21; MASTER_PLAN §9 still names the v1 file).
- The arm set is not frozen before the first VALIDATION read.

All of these can be fixed cheaply and legitimately now, because no governed downstream value has been read.

---

## 6. UNVERIFIED items / open questions
- **Seed variance and SE scale.** The scale of FL seed variance is unknown until two MAIN FA seeds exist. My "same
  order as the paired SE" statement uses central G2 numbers only (INFERENCE). The per-user SD of macro MRR@20 in V4 is
  unmeasured by me. I used the G2 CI width to back out a paired SE of about 0.00049 (INFERENCE).
- **LAST on CALIB.** LAST on CALIB_VALIDATION is not measured, as far as I found. The "FL screen below LAST" statement is
  an inference across CALIB and MAIN.
- **December 2019 file.** Whether the REES46 December 2019 file shares product ids and the catalogue construction with
  the frozen Oct–Nov data is unverified. The dataset page confirms only that Oct 2019–Apr 2020 exists.
- **Real-systems paragraph (§2).** The production-practice paragraph (A/B testing, CUPED, harmed-user guardrails) is
  OPINION with no citation verified in this review. **UNVERIFIED.**
- **Efron & Tibshirani 1993.** The "B = 1,000 suffices for CIs" guidance is attributed through secondary sources in my
  search. The book itself was not opened (bibliographic entry verified; the specific page is **UNVERIFIED**).
- **Open question to the chief scientist.** Should K2's confirmatory claim wording be conditioned on "seed-robust" (E4)
  before the freeze? My recommendation is yes.

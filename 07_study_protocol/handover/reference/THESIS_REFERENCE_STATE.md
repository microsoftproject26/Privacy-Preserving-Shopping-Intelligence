# STATE — living thesis reference (REF-1.1)

- **Last update:** 2026-10-01, by Claude Sonnet 5.5 (single writer); see the newest "Update" block at the end (REF-1.1-claude-v4-warm-bench).
- **What this file is not:** a live server heartbeat. Every status below is as of the last
  captured snapshot:
  - server file copies taken 2026-09-25T12:15Z;
  - live status snapshot taken 2026-09-25T12:31:47Z (source S175).
- **Previous state:** `backups/STATE.md.REF-1.0`.

## Read first, every session
1. This file and the latest `CHANGELOG.md` entry.
2. `CLAUDE_EXECUTION_REFERENCE_EN.md`, the execution reference (primary, English), for the relevant section.
   The Arabic edition `CLAUDE_EXECUTION_REFERENCE_AR.md` is retained unchanged.
3. `amendments/RECONCILIATION_REF-1.1_EN.md` (Arabic original: `RECONCILIATION_REF-1.1.md`), before quoting
   any number that REF-1.0 also states.

## Verified (MEASURED, with sources)
- **Frozen dataset** `rees46_nextitem_v1`: hash `860e5a1c…8dbd`, 409/409 files [S69] [S66]. Split
  counts [S172].
- **GRU v1:**
  - full runs complete;
  - lock at 17:51:17Z, then one VC/TEST look;
  - TEST macro MRR@20: F0 0.168695, TRANS 0.147604;
  - bootstrap GRU−TRANS +0.021091, CI [0.020835, 0.021345];
  - F4−F0 −0.012191, confounded by LR-step timing.

  [S120] [S88] [S02]
- **SASRec v2:** FAILED and stopped. VS A0 0.0913/0.0901, A1 0.1179/0.1168 [S141] [S142] [S140].
- **SASRec v3, seed 2026:** COMPLETE. Full-VS macro MRR@20: A0 e3 0.173762, A1 e3 0.181421 (best)
  [S135] [S136].
- **Feature study:**
  - bounded-sample audit M1–M5 [S44];
  - Fable scope decision STAGED_WITH_EVIDENCE_GATES [S53];
  - code Review C APPROVE, with 125/125 tests [S56] [S57] [S59].

## New since the 12:31Z snapshot (MEASURED, 13:50Z copies)
- **A1_RICH256_s2026_cool5 COMPLETE** at 13:41:01Z. Full-VS epoch-5 macro MRR@20 **0.186298**, micro R@20 0.470434
  [S183]. That is +0.004877 versus e3 and +0.012424 versus GRU F0 (VS only, one seed).
- `cooldowns_go = true`. `A0_ID256_s2026_cool5` was launched at 13:41:52Z on GPU1 [S184] [S185].

- **A0_ID256_s2026_cool5 COMPLETE** at 17:40:47Z. Full-VS epoch-5 macro MRR@20 **0.178585**, micro R@20 0.454042
  [S186]. That is +0.004823 versus e3 and +0.004711 versus GRU F0 (VS only, one seed).

- **VS-descriptive systems on idle GPU1** (Fable GO_WITH_CHANGES; 21:22–23:04Z):
  - S2 (A0 e5 + A1 e5) macro MRR@20 0.186293, i.e. −0.000004 versus A1 e5, CI crossing 0;
  - S4a (average of A1 e3/e4/e5) 0.184492, i.e. −0.001806, CI below 0;
  - the member-rank mechanics check is 100% identical;
  - GPU0 was unaffected (0 aborts).

  [S198] [S199] [S200] [S202]. Official system numbers come only from the final-stage VC/TEST.

- **A1_RICH256_s2027 COMPLETE** at 00:48Z. Full-VS e3 0.181504 (best), e4 0.180630. The two-seed gaps are
  e3 0.000083 and e4 0.000869 [S203]. The s2027 cooldown was launched at 00:48:59Z on GPU0 [S205].

- **A1_RICH256_s2027_cool5 COMPLETE** at 05:04:31Z. Full-VS e5 macro MRR@20 **0.186513**, micro R@20 0.470746 [S206].
  Two-seed gaps: e3 0.000083, e4 0.000869, e5 0.000216 (a range, not an SD).
- **SASRec LOCK** written at 05:05:18Z (sha `004130c6…a47d0`), headline **A1_RICH256_s2027 epoch 5** by the VS-only
  rule; 8 entries, none dropped; RETRO_CLAIM at 05:05:21Z; the retrospective VC/TEST evaluation started at 05:05:19Z
  [S207] [S208] [S209].

- **Retrospective VC/TEST COMPLETE** at 07:24:16Z (PREVIOUSLY INSPECTED). TEST macro MRR@20: headline 0.170475 versus
  GRU F0 0.168695, difference +0.001780 with CI [+0.001538, +0.002022]; versus TRANS +0.022871. The fingerprint gate
  passed 8/8. The ID-only A0 is below GRU F0 on TEST (−0.002211, descriptive). The best system is S3 at 0.173224
  [S217] [S212].
- **SASRec final deliverables** (lead) are at the sasrec_v1 root [S217]–[S220]. The automatic v2-era render is
  preserved in `final/auto_render/` (I039, D048).

## PENDING at the last snapshot (an ETA is not completion)
- ~~A1_RICH256_s2026_cool5 RUNNING at u=753,564 (12:31Z)~~ → COMPLETE at 13:41Z (above) [S175] [S183].
- ~~A1_RICH256_s2027 RUNNING at u=135,303 (12:31Z)~~ → COMPLETE at 00:48Z (above). ~~A1_RICH256_s2027_cool5 RUNNING~~ → COMPLETE at 05:04Z [S206].
- ~~Retrospective VC/TEST evaluation RUNNING (started 05:05:19Z)~~ → COMPLETE at 07:24Z [S212].
- **Feature study:** T1_protect, T1_diff, stream_fit, full_profile and pin_baseline DONE by 07:34:30Z. `probes`
  FAILED twice on an import-isolation breach (I040). It was fixed, then Fable ruled APPROVE_RESUME (D049). The queue
  resumed at 07:52:51Z [S226] [S232].
- **Feature study DONE 08:28Z: KEEP_CURRENT** (D050).
  - No removal candidate; every side-feature probe hurts.
  - C1_D3 − CONTROL = +0.000262, below the 0.002 threshold.
  - Both warm-start arms are below the pin.
  - Lean parity failed its rank-identity gate (0.998683 < 0.999).

  Deliverables: [S236]–[S239], [S249].
- **Conditional cooldowns:** A0 COMPLETE at 17:40Z [S186]. The s2027 cooldown waits for its base run, which is
  RUNNING at u≈358k (17:41Z) [S187].
- **S1–S4 systems;** SASRec lock plus one retrospective VC/TEST; SASRec final report. None of these
  exist [S175].
- **Feature study queue:** QUEUED_RESOURCE, waiting for the campaign's FINAL_STAGE_DONE. There are
  no probe, screen or final-decision results [S175].
- **Purchase_Intent / federated / on-device:** PLAN only [S10] [S11].

## Next documentation actions (when the artifacts exist)
1. Capture `runs/*/summary.json` (cool5, s2027) → full-VS epoch 5 and seed-2027 results.
2. Capture `locks/SASREC_LOCK.json`, `retro_eval/RETRO_RESULTS.json` and `FINAL_SASREC_*` → the
   retrospective results, labelled "previously inspected".
3. Capture `feature_opt_v1`: `audit/M1_FIT.json`, `GROUP_PROBE_RESULTS.json`, `LEAN_PARITY.json`,
   `FINAL_FEATURE_OPT_RESULTS.json`, `state/FINAL_DECISION.json`.
4. Resolve the MISSING_EVIDENCE items (reference §13):
   - the population behind `06_price_status.csv`;
   - the origin of S19;
   - the identity of `userctx.py`;
   - `EVALUATION_EXPOSURE_REGISTER`;
   - the queue-stop log;
   - the human review before training.
5. Optionally regenerate HTML for the execution reference (the REF-1.0 HTML covers the master only).
6. Keep the English edition primary; update the Arabic edition only if the owner asks for it.

## Safety (unchanged)
- Do not alter running jobs, raw or frozen data, model weights, team resources, credentials or
  GitHub.
- This documentation work started no training.
- Local and untracked. The parent folder is not a git repository (verified 2026-09-25).

## PPSI V4 (current programme, from 2026-09-26) — see reference §14
- **Adoption:** V4 adopted (D051) under the owner's operating model (D052).
- **Workspace:** `.local_ai/ppsi_v4/` (company/, contracts/, g0/, code/).
- **G0 planning:** Fable APPROVE_WITH_FIXES (D053). Cohort 262,144, K = 158,486, strict FP32, a common 6-EFE
  schedule.
- **Wave A contracts:** written, with receipts closed (E38).
- **FL-01:** RUNNING.
- **G0: ACCEPTED** (D059). DATA_PROTOCOL_LOCK was written from the final SERVER receipt (self-hash
  `da51d271…`).
- **Pre-G1 locks:** T1–T3 (D058).
- **Wave B code:** tested on CPU (E40).
- **Wave B GPU receipts and integration:** measured (E41). STRICT training determinism is proposed (D060).
- **OWNER AMENDMENT POP 5–100 (D061):** the population lock is SUPERSEDED. MAIN is 5–100 eligible TRAIN examples
  (1,446,986 users / 28,623,386 examples, exact preview).
- **Next:** CENSUS-02 and FEATSTAT-01 (running), then the POPULATION_5_100 deliberation, data lock v2, the G1
  residuals (D062) and Fable G1. **No V4 production training yet.**

## Update REF-1.1-claude-v4-p2main — 2026-09-27T22:14Z (Claude Opus 5.5)
- Data lock v2 + S* locked; **G1 CLOSED** 2026-09-27T16:09Z (S401).
- **GRU P2 DONE** (CALIB tuning, HP_SEARCH): D128 0.124844 vs D256 0.115286 (two-seed means; not comparable with v1 TEST) [S411–S412].
- G1b part 1 approved with conditions [S416–S417]; code_p3 deployed [S414]; dry runs validated [S419–S421].
- **OWNER DIRECTIVE: all training on MAIN** → Fable: 8 MAIN runs (2 candidates per family × 2 seeds), one sealed G2 on MAIN_VALIDATION, TEST sealed [S422–S423].
- **PENDING:** amendment 05, GRU MAIN runs, SASRec P2 (running), SASRec MAIN runs, G2. Structured registers not yet updated for 2026-09-27. See EN reference §14.11.

## Update REF-1.1-claude-v4-p3run — 2026-09-28T01:38Z
- Amendment 05 registered [S424]. GRU D128 both seeds done (sealed; no value read); GRU D256 (C2) running since 01:31Z [S430–S437].
- code_p3b deployed (TRAIN-05/05R1, Opus ACCEPT) [S425–S429]; MAIN baselines on CPU. Codex OFF (owner). See EN §14.12.

## Update REF-1.1-claude-v4-g2prep — 2026-09-28T04:29Z
- SASRec candidates + G1b part 2 [S438–S442]; G2 runner frozen [S443–S447]; 8 MAIN runs scheduled, 4 done. See EN §14.13.

## Update REF-1.1-claude-v4-gru-done — 2026-09-28T05:48Z
- All GRU contest runs done; downstream code verified (z 0.5642 / 1.3094) [S449–S457]. See EN §14.14.

## Update REF-1.1-claude-v4-g2 — 2026-09-28T11:34Z
- **G2 DONE: winner SASRec (DECISIVE_STABLE; T1 delta GRU−SASRec −0.00427, CI [−0.00518, −0.00326])**; ARCHITECTURE_LOCK written [S458–S466]. See EN §14.15.

## Update REF-1.1-claude-v4-fable-review — 2026-09-28T17:00Z (Claude Fable 5.1)
- C_SMALL (SASREC_D64_B2) both seeds DONE_MAIN at 6.0 (values not read); server SHUT DOWN 16:10Z for the disk upgrade (50→350 GB) and the A40 move.
- **Fable 5.1 full review (owner-mandated):** nothing done needs a re-run or a scale change; proposal Steps 1–4 covered or disclosed [S467].
- **Amendment 06A registered** before any downstream read: FT/LO selection panel = UNSEEN_VAL (CALIB ⊂ MAIN); observational cold-start bins;
  FL curves from the 12 in-run marks; deferred tier DP ε=1 → C_FULL → CPF → KD → DATA-07; A40 block; ANA-01 frozen script; disclosures [S468].
- **PENDING:** server return; INT-05/DEV-02/ANA-01 code; A40 receipts; all downstream arms; ALL_METHODS_LOCK; ONE TEST look. Structured registers still PENDING.

## Update REF-1.1-claude-v4-deep-06d — 2026-09-28T22:25Z (Claude Fable 5.1)
- Amendments 06B, 06C registered by Ops (DL 148, 150); the downstream code wave done locally (INT-05, DEV-02/03, FL-05/06/07, ANA-01,
  DATA-07/08) [S470–S471, S481–S490]. Server back on 2× A40; this project gets ONE card (owner).
- **Fable deep expert session:** two blockers (A40 device-class refusal; no executable TEST look) → **amendment 06D** [S472] +
  **MASTER_PLAN_V5** [S473] + packets INT-06 / DATA-09 / ANA-02 / INT-07 [S475–S480]; reviews REV-FL-01 / REV-ANA-01 dispatched.
  Local suites all green. Nothing launched on the server. See EN §14.16–§14.17.
- **PENDING:** review verdicts; INT-06 / DATA-09 / ANA-02 / INT-07; ANA freeze; code_ds3 pack; owner's GPU index + go; §2–§12 of the
  plan; structured registers (still PENDING since 2026-09-27).

## Update REF-1.1-claude-v4-review-ladder — 2026-09-30 (Claude Sonnet 5.5, catch-up for 2026-09-28T22:25Z to 2026-09-29T21:32Z)
Sources **S509–S552** captured (44 requested files; 8 were byte-identical to earlier sources and mapped instead, see
`reports/SOURCE_BATCH_S509-S552.json`). Ids S528, S529, S534–S539 are therefore unused (never reuse). Evidence label per line.
- **MEASURED (server receipts, as recorded in the decision log [S551] and RESUME_HERE [S552]):** server back on 2 x A40; this
  project owns GPU 0 (owner). T3 A40 receipt complete (GRU + SASREC reproducible; A40 vs A16 rank differences 12,635 / 14,063 rows,
  so a separate device class is necessary); STRICT r2 PASS 2026-09-29T00:37:17Z; A40 pin written 00:37:27Z (amendment 06D [S472]).
- **MEASURED (registered CALIB screen reads only), FL improvement ladder [S545]:** S0 0.023473 (as registered, local LR x1) ->
  S1a x2 0.025562 -> S1b x8 0.028796 (x4 0.027570, x16 0.022909; interior winner) [S547, S548]. S2 (FedAdam / FedAvgM server optimiser)
  was RUNNING and UNREAD at the last record (5 of 7 runs done at 21:26Z; the 7 include the FedAvgM m 1 and m 2 runs added by 06L [S543]).
- **REPORTED:** DP sampling corrected to Poisson (amendment 06C [S471]; z(eps 8) 0.5642, z(eps 1) 1.3094, z(eps 4) 0.7244,
  re-derived independently by REV-FL-01 [S491]).
- **REPORTED (review, OPINION-tagged inside the source):** realism audit [S525]; multi-expert review, 14 experts, overall 7/10
  (mean 6.96), 8-8.5 if the MUST list is done before the first read [S509–S524]; its Gates A–E registered as 06L [S543] and 06L-C
  [S544] with ANA_CONTRASTS_v3 [S546]. Independent code reviews REV-DELTA-01/02/02b/03 [S530–S533]; REV-FL-01 [S491], REV-ANA-01 [S492].
- **Selections registered on UNSEEN_VAL (cell only; no MAIN or TEST value):** FT_C -> **FT_C_f0.3_P4** [S549]; LO -> **LO_P4** [S550].
- **RUNNING at the last record (an ETA is not a result):** C_FULL s2026 and s2027 (about 3.5 of 6.0 EFE at 21:26Z), PRE pretraining on
  1,217,610 non-MAIN users (about 2.5 EFE), LADDER_S1a on MAIN (descriptive, unread), LO MAIN s2026 / s2027 then FT_C, S2 screens.
- **PENDING:** S2 selection, S3, S4, S5-S8 ladder stages; FA_WARM / FA_WARM_FROZEN; ANA script freeze; ALL_METHODS_LOCK; the single
  TEST look. The TEST read path is built (DATA-09, INT-07) and **sealed**. **No MAIN or TEST result has been read; none is stated here.**
- Structured registers (DECISION / EXPERIMENT / INCIDENT / CLAIM) still PENDING. See EN reference §14.18–§14.20.

## Update REF-1.1-claude-v4-ladder-s3 — 2026-09-30T11:25Z (Claude Sonnet 5.5)
Sources **S553–S566** (DL rows 236–257, amendments 06L-D / 06L-E / 06M / 06M-B, FL_TUNING_S2 / S3, ladder register, REV-FL-14, REV-DEV-FTPRE, real-world practice audit, FL-14 / DEV-FTPRE / FIX-DEV-FTPRE handoffs). EN reference §14.21.
- **MEASURED (CALIB screen reads only):** FL ladder S0 0.023473 -> S1b x8 0.028796 -> S2 FedAdam eta_s 0.01 tau 1e-3 x1 0.034247 -> **S3 FedAdam + FL_CONST + client SGD-m lr 0.1 0.094936** (+177 % over S2, x4.04 over S0) [S558] [S559] [S560]. S2 edges: eta_s 0.003 0.027238, eta_s 0.001 0.021876, tau 1e-5 0.024374, FedAvgM x0.5 0.024236. S3b not triggered (D = 0).
- **INFERENCE:** the S2 post-peak decline came from the fresh client AdamW; the S3 model is still rising at 6.0 EFE (under-trained), so S5 (longer budget) is expected to help.
- **RUNNING (an ETA is not a result):** S4 (passes 1 / 4, batch 8; launched 11:06Z); warm-start preview WP1–WP7 (descriptive, unread); C_FULL / PRE panel; FT_PRE and LO MAIN runs. **PENDING:** S4 selection, S5–S8, ANA freeze, ALL_METHODS_LOCK, the single TEST look, structured registers.
- **Code:** code_ds3r4 deployed (06L-D selector, 06L-E clipping rule, FT_PRE); REV-FL-14 and REV-DEV-FTPRE PASS-WITH-FIXES, no blocker.
- **Erratum (DL 257):** C_BIG / C_FULL = SASREC_D256_B3 (3 blocks), not 2.
- **MAIN and TEST values remain sealed; none is stated in this reference.**

## Update REF-1.1-claude-v4-warm-bench — 2026-10-01 (Claude Sonnet 5.5, catch-up for 2026-09-30T11:35Z to 2026-10-01T20:15Z; DL 258–307)
Sources **S567–S582** captured (16 files, all new; none byte-identical to an earlier source; `reports/SOURCE_BATCH_S567-S582.json`). S579 / S580 (the S7 / S8 selection files) were fetched read-only from the server because the local `contracts/fl/` mirror lacks them; their sha256 equal DL 292. EN reference §14.22. Evidence label per line.
- **MEASURED (CALIB screen reads / registered receipts only):** S4 final: passes 1 0.107562 (x4.58 over S0) -> `FL_TUNING_SELECTION` sha 9bca186e [S567]. **S7 / S8 selection (DL 292) [S579] [S580]:** S7 winner f0.25 / sf0.1 0.170582 (incumbent 0.135997); S8 winner f1 / sf0.1 0.170709 (incumbent 0.168461); **FA_WARM* = S8 FA_WARM_FROZEN** (tie band, +0.000127).
  code_ds3r5 (DL 264), **code_ds3r6** (DL 291; 680 files, r5 == r6 on 7 tiny-run arms) deployed. S_RECORD 0.284363 (DL 279) and S_RECORD_WARM_FROZEN (sha 2e47ba39, DL 300) registered before any DP run.
- **MEASURED, DESCRIPTIVE and NON-MAIN (06O, DL 293) [S577]:** central models on never-trained new users (UNEW_VAL_VALIDATION_32768, n = 10,237), macro MRR@20: C_FULL s2026 0.195204, s2027 0.194014, PRE theta_pre 0.173081. Never a contrast / selection / claim.
- **PLAN / REPORTED (owner decisions):** **06N [S576]: the pretrained (warm) line is primary** (K20b = R(FA_WARM* / C_FULL), K21 = FA_WARM* - PRE); cold line = K2 ablation + K_INIT; pretraining option 1 (DL 273); larger server model = future work (DL 276).
  External benchmarks: **owner decision DL 302 — methods = the REES46 recipe only**; Wave-1 splits verified byte-identical with the published eSASRec release; MC / SeqRules / PCTM reproduced 12 / 12 (DL 303); Taobao and Amazon-2023 All_Beauty dropped (DL 297), Taobao / Amazon 2023 / MBHT deferred (DL 299); PREG1_AMENDMENT_07 is a **DRAFT, not registered** [S578] [S581].
- **RUNNING at the last record (DL 307, 2026-10-01T20:07Z; an ETA is not a result):** FA_WARM x2, FA_WARM_FROZEN x2 (about 10,400–11,200 of 13,654 rounds), warm FP / PF / FA_Q8 x6, the warm DP chain (FA_1024_WARM_FROZEN x2, DP8_FROZEN x2, DP1_WARM_FROZEN x1); 15 FL jobs, 30 lanes. Cold FA s2026 / s2027 DONE (sealed, unread). FT x4 lanes and the cold N3.3 re-runs (DL 306) are WAITING.
- **INFERENCE:** the frozen-table warm preview (CALIB about 0.167) sits at PRE's level, so the FL gain over PRE may be small (K21 decides at the read); a fresh client AdamW destroys the pretrained tables (DL 267).
- **Disclosures:** DL 289 vs 290 overlapping workers (retention_report e63ec147 -> 6e6ac09a); the m1 check read `LAUNCH_1.json` instead of `LAUNCH_0.json` (DL 295); stale RUNNING status in stopped cold dirs (DL 303).
- **PENDING:** warm / DP MAIN completions; FT x4; cold N3.3; ANA-06 before any warm-line indexing; C_FULL s2027 reeval; ANA freeze; ALL_METHODS_LOCK; the section 9 read; the single TEST look; BM-02 (deferred to the 10-02 quota reset), the 07 registration and every benchmark run; structured registers (DECISION / EXPERIMENT / INCIDENT / CLAIM / RESULTS).
- **No MAIN or TEST value has been read; none is stated in this reference.** See EN §14.22.

## Update REF-1.1-claude-v4-endgame-prep — 2026-10-02 (Claude Sonnet 5.5, catch-up; DL 311–353, to 2026-10-02T10:22Z)
Sources **S583–S601** (19 files, all new; `reports/SOURCE_BATCH_S583-S601.json`; DL 308–310 captured afterwards as S602). EN reference §14.23. Evidence label per line.
- **REPORTED (registered):** **06P** [S584] FA_WARM* = FA_WARM_FROZEN; primary pair {K20b, K21}; **ERRATUM 1: Holm family = 18, not 19**; sha 75f8dbad…; `ANA_CONTRASTS_v4.json` f1769f0b… [S586]; D2 (N33 runs registered, FP_s2026 / FP_s2027 / FA_1024_s2026 STOPPED_REALLOCATED), D3 run-or-declare, D4 command fixes. **Amendment 07 REGISTERED** (sha 29beb5e3…, B1–B11) [S585].
- **MEASURED:** code_ds3r7 deployed (tar c9e20797…, 684 files); **tiny-run digests r6 == r7 for all 7 arms**; one known Windows-pin test failure (PRE_FL10_TRAJ). FL-16b, DEV-04, ANA-06 reviewed (PASS / PASS-WITH-FIXES, no blocker). REV-BM-02 first FAIL (1 BLOCKER, 7 MAJOR), fixed over Revisions 2–3 (delta PASS-WITH-FIXES). Fable audits: ops 0 BLOCKER / 0 MAJOR / 6 MINOR; benchmarks 0 BLOCKER / 3 MAJOR (fixed). SELECTION_FT_FA registered (a645edaf…).
- **Benchmarks:** Wave 1 LAUNCHED 23:18Z on CPU, 112 of 234 jobs done, 0 failed at 10:22Z (no lock / final / TEST look). Wave 2 = Amazon 2023 last_out x 3 only; mbht_tmall and mbht_taobao DROPPED.
- **Autonomous night (MEASURED receipts):** warm line (4 MAIN runs) and warm DP chain complete; FT_FA_WARM x2, FT_FA_WARM_FROZEN s2027, FT_C / LO UNSEEN_HOLDOUT s2026 done; FT_FA MAIN, FT_FA_WARM_FROZEN s2026, cold N33 FP / PF / FA_Q8 and N33 DP chains running; INT8 r7 queued (waits for Wave 1). All sealed, unread.
- **PENDING:** remaining runs, INT8, panel reevals on r7, ANA freeze, `--check-only`, section 9 read, ALL_METHODS_LOCK, the single TEST look, Wave 1 finish, Wave-2 addendum, structured registers.
- **No MAIN, VALIDATION or TEST value has been read; none is stated in this reference.** See EN §14.23.

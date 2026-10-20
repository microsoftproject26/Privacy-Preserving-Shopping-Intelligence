# 00 — SYNTHESIS of the multi-expert review (2026-09-29, written 17:55Z)

Lead reviewer: Claude Opus 5.5. Fourteen same-type experts (general-purpose, Opus), one per part, files 02–15 in this folder;
project map in 01. READ-ONLY review: no server contact, no git, no tests, no RETRO_TEST row, no downstream RESULT_ROWS value.
The only result values used are those the project itself already read and quoted: the CALIB FL screen values (DL 181, 218;
`contracts/fl/LR_SELECTION_EXT.json`), the G2 contest values and the MAIN POP/LAST/SKNN baselines (DL 129–132).
Tags: MEASURED (path:line) · REPORTED (a project document's number) · INFERENCE · OPINION. Paths are relative to
`.local_ai/ppsi_v4/` unless stated. State checked at 17:31Z: DECISION_LOG ends at row 225; S2 screens RUNNING and UNREAD.

---------------------------------------------------------------------------------------------------------------------
## 1. Executive verdict

1. **Rigour: well above MSc level, and above most published FL-recsys work.** The experts found no label leakage, no
   evaluation-path bug, no accountant error and no aggregation bug (parts 2, 6, 7, 12). Supporting evidence:
   full-catalogue exact ranking, a hash-sealed single TEST look, pre-registered rules, and a DP accountant reproduced to 1e-11.
2. **Realistic Edge answer: not yet convincing.** As registered, the headline federated arm keeps only ≈ 25 % of the matched
   central model on CALIB (0.0288 vs ≈ 0.11–0.12; REPORTED, DL 218). The zero-training "repeat last item" heuristic (LAST)
   gets ≈ 44 % of C_BIG on MAIN (0.0711 vs 0.1612; REPORTED, DL 132). So unless the tuning ladder roughly doubles FL, the
   thesis would report "federated learning is worse than doing nothing clever on the device". INFERENCE: CALIB and MAIN
   differ in size and round count.
3. **The gap is the recipe, not a bug (part 6).** Contributing factors: fresh client AdamW every visit on a 158k-row softmax,
   server LR 1 without momentum, the central LR shape mapped onto rounds, and sparse item-row dilution. The runs peak early
   (best marks at 1.5–3.5 of 6.0 EFE, MEASURED `contracts/fl/LR_SELECTION_EXT.json`), so "a lower bound limited by budget"
   is not supported.
4. **The privacy story is stated more strongly than it holds (parts 1, 7, 8, 9).**
   - Uploaded deltas reveal the items a user viewed; the input and output tables are untied (`code/v4models/sasrec_v2.py:196,202`).
   - There is no secure aggregation.
   - FT_C on MAIN is a cloud model trained on the same users, not L5.
   - RULE uses pooled popularity, so it is not L6.
   - There is no threat model.
5. **The device-only side rests on weak arms and weak evidence (parts 9, 11).** LO is an untuned 20.7 M-parameter model
   started from random init, LAST is missing from the privacy ladder, and the device evidence is a server-CPU proxy with a
   contaminated memory number and no energy or on-device training cost.
6. **Four execution blockers would break registered MUST steps later (all confirmed below).**
   - `reeval.py` refuses C_FULL and PRE.
   - `dump_rows` crashes on real data.
   - S8 is not admitted by the analysis script.
   - The production code tree exists only in an OS temp folder.
7. **Almost everything is still fixable legitimately.** No governed downstream value has been read, the ANA script is not
   frozen, and TEST is sealed. The only hard deadline is **before the S2 selection is written (INFERENCE ≈ 21–23Z today)**.
8. **Versus deployed systems (verified by the experts):**
   - Gboard's FL model is 1.4 M parameters / 1.4 MB and beat the server model (Hard et al. 2018).
   - This project ships ≈ 165 MB per FL visit (≈ 100× larger).
   - Production DP runs at ≈ 0.23 S noise per round; MAIN-scale DP here is 2.5 S. The planned DP8_FULL reaches ≈ 0.32 S.
   - The project already has the right production-shaped answer: pretraining on consenting users (PRE), warm-started FL
     (FA_WARM), frozen item tables (S8), and DP at full scale (DP8_FULL). It is registered only as descriptive or low-tier and
     should become the Edge headline.
9. **Overall score (mean of 14 parts): 7.0 / 10** (OPINION). The process is 8–9 and the realistic-deployment science is 5–6.
   The experts estimate 8–8.5 if the MUST list below is done before the first VALIDATION read.
10. **Bottom line (OPINION):** keep every finished run; fix framing, analysis and four code paths now; promote the
    deployment branch; anchor every retention figure to LAST. No finished run needs to be redone.

---------------------------------------------------------------------------------------------------------------------
## 2. Scorecard

| # | Part (file) | Score | One-line reason |
|---|---|---|---|
| 1 | Question & Edge framing (02) | 6.5 | Algorithmically excellent (8–9). As an Edge answer ≈ 5: FT_C / RULE are placed at the wrong privacy level, there is no threat model, no device-only floor, and K17 is an algebraic composition |
| 2 | Data (03) | 8 | No temporal leakage; exemplary censoring, out-of-vocabulary handling and population design. At TEST some user-context features silently omit the VALIDATION week; the "cloud" is a 5–100-user cloud |
| 3 | Features & history (04) | 7 | S* is in substance device-computable (21 of 24 inputs local). The same undisclosed partition fence; SASRec positions count from the oldest event; catalogue threshold counts views, not users |
| 4 | Backbone contest (05) | 7 | Procedure ≈ 9. The verdict flips with the checkpoint rule (SASRec over-fits); the families were never size-matched; the registered TEST check of the contest has dropped out of the plan |
| 5 | Central references (06) | 7 | Matched denominators are sound. `reeval.py` refuses C_FULL/PRE; vs-C_FULL ratios mix scale, size and federation; the 170 MB cap mixes inference size with FL payload |
| 6 | Federated training (07) | 6 | Clean, bug-free engine, but the registered recipe is pre-2019. FedAvgM is only screened ≈ 10× past the optimum; the SGD × FedAdam cell can be skipped; no sparse-aware aggregation; PF is inert |
| 7 | Differential privacy (08) | 7 | Accountant and mechanism correct (reproduced). Fixed clip under a decaying LR; FedAvgM noise amplification not covered; the guarantee is over-stated |
| 8 | Communication (09) | 6 | Byte ledger exact. In practice only 3 byte points; K14 is weakly informative; S8 cannot enter the analysis (and would be ≈ 13× under-counted); S8 still leaks brand/category |
| 9 | On-device arms (10) | 6 | Causality and selection hygiene ≈ 9. LO is close to a strawman, LAST is missing, FT_C is at the wrong level, and there is no parameter-efficient adaptation |
| 10 | Cold start & strata (11) | 6 | The T41 negative control is stronger than the literature's. But the primary Step 4 answer is confounded and observational; the controlled study is low-tier and cannot produce its key cells |
| 11 | Device measurement (12) | 5 | Careful export and parity checks. `dump_rows` crash still unfixed; peak memory includes the torch import; INT8 setting risky on the server CPU; browser protocol thin; no energy or training cost |
| 12 | Evaluation & statistics (13) | 8 | Exact full-K ranking and a correct paired bootstrap (no bug found). Intervals ignore seed variance; the multiplicity argument is wrong; no uncertainty-bearing baseline anchor; K20/K21 have no registry file |
| 13 | Engineering & reproducibility (14) | 7 | Numerics ≈ 9. Production tree only in a temp folder; three trees never archived; no environment lock; θ_pre provenance still open |
| 14 | Thesis write-up (15) | 6 | Strong raw material. The chapter map predates 06E–06K; the Queen's AI statement is unplanned; threats list about half complete; bibliography errors confirmed |

---------------------------------------------------------------------------------------------------------------------
## 3. Cross-check: what the lead verified, and how disagreements were resolved

### 3.1 Top findings re-verified in the files (all CONFIRMED unless stated)
| Finding | Evidence (MEASURED) | Experts |
|---|---|---|
| `reeval.py` refuses any central checkpoint whose run is not the registry's MAIN pair. C_FULL trains on FULL_TRAIN_5_100_TIEBREAK (`code/v4int/c_full.py:50`) and `base_reg` is the LOCK_V2 base registry (`reeval.py:525`), so C_FULL (and PRE) cannot be re-evaluated on MAIN or MAIN_TEST | `code/v4int/reeval.py:557-559` | 5 |
| TEST user-context replays TRAIN then TEST only, so "observations between fences are intentionally omitted" (the VALIDATION week at TEST) | `next_item_eda/src/rees46_eda/pipeline6.py:1-13` vs "identical for TRAIN/VALIDATION/TEST" `contracts/DATA_PROTOCOL_LOCK_V2.json:90` | 2, 3 (independently) |
| The registered TEST contest singles, C_BEST and SYSTEM are not in the TEST plan, and no 06* amendment withdraws them | `contracts/PREG1_AMENDMENT_02.md:58`, `PREG1_AMENDMENT_05.md:53`; absent from `company/plans/MASTER_PLAN_V5.md:187-196`; grep of 06* empty | 4 |
| S8 arm FA_WARM_FROZEN missing from the analysis script's arm list; unknown arms are refused | `tools/retention_report.py:125-127, 764-765` | 8 |
| events-to-X reports NOT_REACHED when the C reference row is missing (it should report NO_C_REF) | `tools/retention_report.py:2250-2259` | 10 |
| `dump_rows` concatenates every provider field with no filter; file unchanged since 08:45Z (sha 1bde41bd…) | `code/v4dev/dump_rows.py:139-148` | 11 (from DL 209) |
| PF personal LR is never multiplied by the selected local-LR multiplier (registered, 06B §1, but it makes PF inert) | `code/v4int/launch_fl.py:656` | 6, 9 |
| RULE uses MAIN-TRAIN popularity conditioned on category/brand, so it is not own-history only | `code/v4base/rule.py:9-12` vs `contracts/PREG1_AMENDMENT_06B.md:59` | 1, 3, 9 |
| SASRec input and output item tables are untied, so the input-table delta's support = the viewed items | `code/v4models/sasrec_v2.py:196, 202` | 1, 8 |
| FL screens peak early (best mark 1.5–3.5 of 6.0 EFE for ×2…×16) | `contracts/fl/LR_SELECTION_EXT.json` runs[*].practical_best_mark | 6 |
| code_ds3r3.tar exists only in another session's OS temp scratchpad (`C:/Users/Ahmed/AppData/Local/Temp/claude/.../bd85e2a3-…/scratchpad/code_ds3r3.tar`); no copy under `ppsi_v4/` | find on both roots | 13 |

### 3.2 Contradictions and duplicates resolved
| Topic | Positions | Ruling (who is right, why) |
|---|---|---|
| "Only downloads are charged" (the lead's own project map) vs part 8 | Part 8: completed visits are charged down + up; dropped visits add their download | **Part 8 is right** (`tools/retention_report.py:1023-1044`). Map corrected (01, §3 item 7) |
| Pooled channels: "34 of 53 channels use other users' data" (part 7) vs "21 of 24 S* inputs are local" (part 3) | Part 7 quotes `contracts/CLIENT_CONTRACT_V1.md:223`, which describes the full Clean V2 set of 53 channels | **Part 3 is right for S\***. The contract text is stale for the locked schema, so part 7's DP-3 wording must use part 3's S*-specific statement (F2) |
| "Equal-exposure retention is a LOWER BOUND" (06I §6(a); part 1 R16 KEEP) vs part 6 F4 (reword) | The best marks sit well before the 6.0 endpoint | **Part 6 is right.** The recorded peaks show saturation within the schedule, not budget exhaustion. Use "matched-exposure retention of recipe R". FA_X2 measures the budget effect; do not assert the bound |
| S8 upload size: "≈ 0.4 M params" (06K; used by parts 7 and 9) vs 265,162 (part 8, from the inventory arithmetic) | — | **Part 8's arithmetic reproduces 20,710,048 total exactly**, so use 265,162 params / 1.06 MB per direction. Part 7's noise figure for DP8_FROZEN improves slightly (√(d) smaller) |
| FA_FULL (non-DP FL on the full band): SHOULD (part 1 R6) vs NICE (part 5 R16; part 7 DP-5 "twin NICE") | Cost ≈ 110 process-h | **Conditional ruling:** it is SHOULD only if DP8_FULL is run (without it, DP8_FULL mixes DP, federation and population). Otherwise do not run either, and label DP8_FULL "not run" |
| K17–K19 status | Parts 1, 5, 12 independently: point estimates are exact compositions (K17 = K2·K15 = K3·K1·K15 at seed 2026) | **Unanimous. Keep them** (registered), but relabel as compositions and remove "co-headline" wording. They are not independent evidence |
| FL below LAST | Parts 1, 9, 12 (INFERENCE) | Agreed as a risk, not a result: the CALIB and MAIN populations differ, and MAIN FL has 4× the rounds. A paired, CI-bearing FA − LAST contrast on MAIN is the answer (M-C3) |
No disagreement remained that needed an extra adjudicating expert. Every conflict was settled by a file check.

---------------------------------------------------------------------------------------------------------------------
## 4. Ranked action list, in execution order (so nothing has to be redone)

Legend: tier MUST / SHOULD / NICE / DISCLOSE. Action KEEP / CHANGE / REDO / ADD. "Prereg" = needs a registration (amendment
or DL row) before its governed value is read. Status is OPEN for every item unless stated. Cost is INFERENCE. The source
expert IDs are in brackets (part-ID), e.g. [6-F1] = part 6 recommendation F1. **No item REDOes a finished run.**

### Gate A — before the S2 selection file is written (S2 running since 14:44Z; INFERENCE finish ≈ 21–23Z) — ONE amendment "06L-A"
| # | Tier | Item | Cost | Affects | Prereg |
|---|---|---|---|---|---|
| A1 | MUST | Add FedAvgM at m ∈ {1, 2} (β 0.9, η 1) + an edge rule to S2. The registered m = 8 is an effective step ≈ 10× past the FedAvg optimum [6-F1] | 2 CALIB runs ≈ 2.7 process-h each | S2, every MAIN FL arm | yes (S2 unread) |
| A2 | MUST | Register client SGD-m × FedAdam cells (η_s 0.01, lr {0.1, 1.0}, FL_CONST) in S3 **whatever S2 selects**. This re-instates the combination run that 06J withdrew [6-F2] | 2 CALIB runs | S3, final FA | yes |
| A3 | MUST | Extend the 06H DP_OPT_SENS trigger to any non-FedAvg S2 winner (FedAvgM included), with the 1/(1−β) noise-amplification note [7-DP-2] | registration; 2 runs only if triggered | FA_1024/DP8 s2026 | yes |
| A4 | MUST (if S3 keeps the S0 shape) / SHOULD-high | Clip follows the LR: S_r = S·f(r) (data-independent, accountant unchanged). Minimum: report FA_1024's per-round clipped fraction [7-DP-1] | ≈ 0.5 day code + review | S_CAL, FA_1024, DP (none run) | yes |
| A5 | SHOULD-high | PF personal LR: scale by m* or run a 2-point CALIB screen; otherwise register "PF ≈ FA by construction" as K5's interpretation [6-F6, 9-R5] | 0–2 CALIB runs | PF, K5 | yes (PF not launched) |
| A6 | SHOULD | Sparse-aware aggregation rung (FedSubAvg-style per-row counts on the input table; FA/FP only) [6-F3] | ≈ 0.5 day code + 1–2 CALIB runs | FA/FP | yes, before S3 selection |
| A7 | SHOULD | Register the FedProx-under-Adam interpretation of K4 before any FP read [6-F7] | text | K4 | yes |

### Gate B — execution blockers (code only, no rule change; do now, in parallel with Gate A)
| # | Tier | Item | Cost | Affects |
|---|---|---|---|---|
| B1 | MUST | `reeval.py`: accept the registered C_FULL (FULL_EXT) and PRE sources; review + server `--dry-run` [5-R1] | 2–4 h Opus + review | K15, K17–K19 at TEST; PRE-alone; K21 |
| B2 | MUST | `dump_rows`: keep numeric array fields only (the `dump_rows_retro.py:62` filter), add a test, name an owner (orphaned since DL 214) [11-R1] | 1–2 h | MUST-tier INT8 / K16 |
| B3 | MUST | Admit FA_WARM_FROZEN into `retention_report`; charge the 81.78 MB table download once, as a first-visit step [8-C1] | 2–3 h + review | S8 row and bytes curve |
| B4 | MUST | NO_C_REF status in `trunc_curves`; Study B events-to-X path (`launch_trunc.py:176`) [10-R4] | < 0.5 day | DATA-07 |
| B5 | MUST | Archive code_ds3r3 / ds3s2 tars into `ppsi_v4/` with manifests + DL shas; read-only hash manifests of the r2s / r2p / ds3pre server trees [13-R1] | 1 h | provenance of the ×8 LR selection, PRE, C_FULL manifest |
| B6 | MUST (before any FA_WARM launch) | Close θ_pre provenance: DL row with the server shas of r2p `pretrain_small.py` / `launch_central.py` vs a5550479 / a4ded7da [13-R3] | 10 min | PRE, FA_WARM, K20/K21 |
| B7 | MUST | Mirror every server-only registration file locally (pin, LR_SELECTION, strata cut points, registry extensions, …); environment lock (collect_env, pip freeze, driver, SM count) [13-R4, R5] | 45 min | reproducibility |
| B8 | SHOULD-high | Server suite run of code_ds3r3; A40 bitwise-resume receipt before pausing C_FULL; production-evaluator (61b64e61) A40 hash check; GPU-1 vs GPU-0 ranks-hash check [13-R8–R10, 5-R10] | ≈ 1.5 h total | A40 block integrity |

### Gate C — before the ANA freeze and the first VALIDATION read (one registration "06L-C" + ANA_CONTRASTS_v3)
| # | Tier | Item | Cost | Prereg |
|---|---|---|---|---|
| C1 | MUST | **Privacy ladder as a matrix** (training exposure / inference exposure / guarantee / against whom). Changes: FT_C on MAIN → "L1 base + local FT"; FT_PRE = true L5; PRE-alone; RULE/POP → "device + published aggregate"; add LAST to L6; SecAgg column "not simulated" [1-R1, 9-R2a, 3-F3, 9-R8] | ≈ 1 h text | text amendment |
| C2 | MUST | **Threat model**: honest-but-curious browser vendor; the retailer sees its own log (out of scope); what the server sees in each arm [1-R2]. **DP guarantee statement**: central, trusted aggregator, client-level add/remove, conditional on a public bootstrap block, whole FL tuning unaccounted [7-DP-3] | ≈ 2 h | text |
| C3 | MUST | **Anchor on LAST**: dump baseline rank files; paired Δ FA − LAST, LO − LAST, FT_C − LAST with CIs; headroom R* = (arm − LAST)/(C_ref − LAST); flag BELOW_DEVICE_HEURISTIC; K11b = FA − LAST; device-only envelope {LO, LAST, RULE} [12-E1, 1-R3, 9-R1] | 2–4 h CPU + ANA delta | yes (descriptive + K11b) |
| C4 | MUST | **ANA_CONTRASTS_v3** (K1–K21 + the new rows). Fix MASTER_PLAN §9's v1 file path. List the algebraic dependencies (K3 = K2/K1; K11; K17–K19 = K·K15); remove "co-headline" wording for K17 [12-E2, 12-E5, 5-R3, 1-R4] | minutes–1 h | yes |
| C5 | MUST | **Freeze the arm set, seeds, recipes and checkpoint rules before the preflight.** Anything added later is POST_VALIDATION_EXPLORATORY and never confirmatory [12-E3] | minutes | yes |
| C6 | MUST | Reword the "lower bound" claim to "matched-exposure retention of recipe R", correct the "≈ 2/3 of rounds" figure to "50 % of the budget", and cite the peak marks [6-F4] | text | text |
| C7 | MUST | **Promote the deployment branch to secondary confirmatory**: R(FA_WARM / C_FULL), R(PRE / C_FULL), K20/K21 with the seed/pairing rule; K9c = FT_PRE − PRE on MAIN; K9b = FT_C − C_SMALL on UNSEEN_HOLDOUT. Lead the L5 and Edge answers with these [1-R5, 5-R2, 9-R2] | FT_PRE ≈ one FT run per seed (4–10 h) + a small launcher label change | yes (all unread) |
| C8 | MUST | S8 / bytes: register the paired S8 − S7 contrast with the bytes ratio; put bytes for every privacy level on one axis (raw events KB → model download → FL traffic → S8); correct the S8 disclosure (265,162 params; brand/category deltas still leak; off-support rows ≈ 1e-9, not 0) [8-C2–C4] | 0 GPU; ≈ 4 h | yes |
| C9 | MUST | Register the K16 quantiser cell exactly (dynamic QInt8/QUInt8, per-tensor, reduce_range False, ORT 1.24.1) + the CPU flags, and add one descriptive per_channel + reduce_range cell. Peak RSS measured without torch [11-R3, R2] | ≈ 2–3 h | yes (K16 unread) |
| C10 | MUST | Cold start: raise the minimal DATA-07 set (T41 on both panels; Study A C_BIG/C_SMALL/FA on COLD_H; Study B FT_C on COLD_H) to SHOULD-high, and to MUST for any "events to parity" sentence. Add matched-h parity M(h)/C_BIG(h) and a persistence + CI parity rule beside the registered first-hit rule [10-R1–R3] | ≈ 5–12 GPU-h + 1–1.5 analyst-days | yes |
| C11 | MUST | Add a descriptive deployed-size family comparison: GRU_D128 (20.96 M, already trained) vs C_SMALL, PB and CB, A40 block [4-B3] | evaluation only | yes (C_SMALL unread) |
| C12 | MUST | Number-free headline and abstract template, plus the CLAIM register (allowed and forbidden sentences per contrast), frozen with the ANA script [14-W2, W10] | 3–4 h | recommended |
| C13 | SHOULD-high | Seed variance next to every K (per-seed R, spread, user + seed interval, SEED_SENSITIVE flag); Holm or Romano–Wolf band beside the unadjusted CIs, correcting 06D §4's "not resolvable" [12-E4, E5] | 2–4 h | yes |
| C14 | SHOULD-high | CALIB-disjoint diagnostic (MAIN \ CALIB) for every CALIB-tuned arm [2-D4, 6-F8, 12-E8] | trivial | yes |
| C15 | SHOULD | Harmed-client share for every K with a seed-vs-seed null [12-E6, 9-R6]. K14 equivalence margin before any FA_Q8 read [8-C6]. Repeat/novel-target stratum [2-D5]. Participation-gap difference-in-differences (FA gap − C_BIG gap) [10-R12]. Activity-gap contrast [10-R11]. De-confounded personas [10-R10] | ANA deltas ≈ 1–2 days | yes |
| C16 | SHOULD | Parameter-efficient on-device arm FT_PU (fit only p_u on a frozen base) [9-R4]. Descriptive C_SMALL/FA re-evaluation on UNSEEN_VAL (a "no-FT" reference) [9-R3] | 0.5–1 day code + runs | yes |
| C17 | SHOULD | Promote S8 (2 seeds, unconditional) as the production-shaped FL arm [1-R8]. DP8_FROZEN to SHOULD-high with its z = 0 twin [7-DP-4]. Keep DP8_FULL and write DP_Z_FULL + launch path [7-DP-5] (twin: see §3.2) | ≈ 10.7 process-h per run | yes |

### Gate D — before ALL_METHODS_LOCK / the single TEST look
| # | Tier | Item | Cost | Prereg |
|---|---|---|---|---|
| D1 | MUST | Restore the registered TEST contest singles (02B_GRU, 02B_SASREC on A40, VALIDATION + TEST, paired CI), or withdraw them in a dated amendment [4-B2] | ≈ 0.5 GPU-h | yes |
| D2 | MUST | Disclose the TEST partition fence (7 S* user-context channels, the view gap, prefix counts). Add a counts-only receipt inside the gated TEST build. Correct the lock wording [2-D1, 3-F1, F4] | 1–2 h CPU | counts entry in the TEST register |
| D3 | SHOULD | Descriptive TEST sensitivity with continuous-stream user-context for the headline arms [3-F5, 2-D2] | ≈ 1 day + ≈ 3 min eval per arm | yes |
| D4 | SHOULD | TEST sub-periods with Black Friday separated [12-E7, 2-D6]. TEST re-execution digest identity for LO/FT [9-R7]. ONNX model sha identity VALIDATION = TEST [11-R10] | small | yes |
| D5 | SHOULD-high | External timestamp: a hash manifest of contracts + gates + DL emailed to the supervisor (no GitHub mutation needed) [14-W5, 13-R16] | 1 h | — |
| D6 | NICE (high value) | Prospective December-2019 holdout (REES46 publishes Oct 2019–Apr 2020; verified by part 12). Owner approval of a new data version needed [12-E10] | days | yes |

### Gate E — write-up and packaging (no runs; start now, finish after TEST)
| # | Tier | Item |
|---|---|---|
| E1 | MUST | **Queen's "Statement of Originality and Disclosure of Use of AI"** (required front matter, SGSPA; REPORTED by part 14). Name the models and roles and the owner's review; confirm the supervisor's advance approval. The "no AI attribution in commits" rule must not become a thesis omission [14-W1, 13-R7] |
| E2 | MUST | Threats chapter in Wohlin's four categories, including CALIB ⊂ MAIN for FL tuning, the reversed tuning asymmetry (FL ≈ 25 configurations vs C_SMALL's 4), internal-only pre-registration and the AI process; contrast count = the frozen family size [14-W4] |
| E3 | MUST | One generated DISCLOSURES register (≥ 14 source files; reconcile WOR vs Poisson, "no DP" in CLIENT_CONTRACT, 16 vs 21). Appendix B as a generated timeline of what had been read before each amendment [14-W3, W6, W13] |
| E4 | MUST | Main body limited to 5 figures + 4 tables. Figures: retention by privacy level vs C_BIG and C_FULL with LAST marked; the FL ladder waterfall; quality vs bytes; the ε curve with the noise note; the cold-start curve. Tables: what changed from the proposal; the privacy matrix; K by tier; device measurements [14-W7] |
| E5 | MUST | Task wording: "next different viewed item within the session, views only, given ≤ 50 prior events across sessions". Every retention sentence says "relative to a cloud trained on 5–100 users" [2-D11, D3] |
| E6 | MUST | Backbone wording: "selected under T1", show both checkpoint rules, explain the reversal as SASRec over-fitting; never "SASRec beats GRU" [4-B4] |
| E7 | MUST | Bibliography fixes (verified): [5] conflates two papers; [6] = Lin et al., IEEE Intelligent Systems 36(5) 2021, and the news FedRec is Qi et al., Findings of EMNLP 2020; delete the "95 % / 60 %" claim; [8] = Luo et al. CIKM 2022, while PFedRec = Zhang et al. IJCAI 2023; [12] not found; add Reddi 2021 and Hsu 2019 [14-W8] |
| E8 | MUST | Reproducibility package: byte-identical trees, environment, data rebuild from ledger 860e5a1c…, run → tree table, tolerance statement. The public "clean port" is labelled a re-implementation unless proven equivalent [13-R2, R6, R7] |
| E9 | MUST | 170 MB cap: state its origin, and separate the inference download (82.8 MB FP32 / ≈ 21 MB INT8) from the FL payload (165.7 MB per visit) [5-R8]. Disclose C_FULL's untuned recipe and its upward bias on vs-C_FULL ratios [5-R5]. D5 normaliser proxy: owner ratification or disclose [3-F8] |
| E10 | SHOULD | Update the living thesis reference now (sources S509+, STATE header, CHANGELOG, §14.18) rather than after TEST. Fix the public doc (viewed-item leak, previously inspected TEST, "full pool" = ≈ 93 %) at the next **owner-approved** push [14-W10, W11, 1-R10] |
| DISCLOSE | — | Evaluation windows are non-stationary; the protocol is static (no continual adaptation) [2-D6, D7]; panels are returning users; TRUE_COLD A1 has TRAIN events [2-D8, D9]; SASRec position semantics [3-F10]; catalogue threshold ≥ 5 views is not k-anonymous [3-F6]; SMALL differs in depth (B3 → B2) [5-R6, 4-B10]; "GRU still improving" is untested [4-B5]; DP ε(δ) curve and idealised Poisson participation [7-DP-7, DP-9]; payload-only bytes [8-C11]; server latency is a parity proxy (browser ≈ 16.9× slower, TOSEM 2025) [11-R13]; CPU-bound FL loop ≈ 10 visits/s [13-R20]; provenance deviations (C_FULL from code_ds3r2, S1b from r2s, PRE from r2p, Codex overnight) [13-R21] |

### NICE (only after everything above)
KD_SMALL [5-R9, SHOULD if a slot opens]; C_SMALL_FULL [5-R4, SHOULD-low]; per-row Q8 and an 8/4/2-bit sweep [8-C8, C9; C8
becomes MUST if S3 selects client SGD]; LO_TUNED and RULE_PLUS [9-R11, R12]; FedAvgM warm cell [6-F13]; SASRec regularisation /
GRU 8.0 exploratory runs [4-B12]; empirical item-set recovery / membership audit [7-DP-11]; a second device class and a coarse
energy test [11-R8, R14]; a heavy-user bot audit [2-D12].

**Recommended order:** Gate A (tonight, before S2 is read) ∥ Gate B → Gate C (one registration + ANA_CONTRASTS_v3, then the
freeze) → the runs added in C7/C10/C11/C17 and A1/A2 → the VALIDATION preflight → Gate D → ALL_METHODS_LOCK → TEST → Gate E.
Registering first and running afterwards means nothing needs re-running.

---------------------------------------------------------------------------------------------------------------------
## 5. What is already strong (highlight in the thesis)
1. **Pre-registration discipline**: hash-sealed amendments written before each governed read; a hash-chained TEST exposure
   register; a single, gated TEST look (06D §2). Few FL or recommender papers do this.
2. **Evaluation core** (part 12, no bug found): exact full-catalogue ranking over 158,486 items with an adverse tie rule and a
   pessimistic check; OOV targets kept at zero credit; client-macro primary metric; one shared, hash-registered paired
   bootstrap matrix; per-resample ratio bootstrap.
3. **Data protocol** (part 2): global temporal split with no future-label use; targets built before catalogue filtering;
   exhaustive, fail-closed censoring; hash-rank populations proven disjoint; a user-disjoint UNSEEN panel for the
   participation gap.
4. **Numerics**: STRICT FP32 with read-back, a device-class separation justified by measured A16/A40 rank differences and by
   cuBLAS documentation, and a frozen evaluator hash in the comparison key.
5. **Correct DP accounting**: Poisson sampling with a fixed denominator; the accountant reproduced independently to 1e-11;
   confirmatory DP contrasts at the data-independent endpoint (06D §3); the noise-to-signal note registered before any read.
6. **Self-diagnosis**: the FL improvement ladder, registered stage by stage, with the register "when we did X, FL improved by
   Y" (S0 0.0235 → S1b 0.0288 on CALIB). Present it in full, including configurations that lost.
7. **Causality engineering** for the device arms (refuse, never filter) and selection on a disjoint panel (UNSEEN_VAL).
8. **The T41 golden-replay negative control** for truncated histories: stronger than anything in the cold-start literature
   (part 10).
9. **Device-computability of S\*** (21 of 24 inputs local; pooled content = catalogue, 3 vocabularies, 36 scalars; part 3).
   This answers "what can Edge compute locally" directly.
10. **The deployment branch idea** (PRE on consenting users → FA_WARM / FT_PRE / S8 / DP8_FULL): it mirrors how browsers
    actually ship personalisation. Make it the Edge answer (C7).

---------------------------------------------------------------------------------------------------------------------
## 6. Top 10 risks at the defence, with mitigations
| # | Examiner's challenge | Mitigation |
|---|---|---|
| 1 | "Your federated model is a strawman: ≈ 25 % retention, maybe below repeat-last-item." | A1/A2/A6 before S2 is read; the LAST anchor (C3); S8 / FA_WARM as the production-shaped arms (C7, C17); recipe-qualified claims (C6). Present the ladder waterfall |
| 2 | "Federated ≠ private: the server sees what users viewed." | Threat model + privacy matrix (C1, C2); SecAgg "not simulated"; DP arms as the only formal guarantee, stated precisely (C2); S8 leak correction (C8) |
| 3 | "Retention against which cloud? K17 is just K2 × K15." | C4 (compositions), C7 (data-matched deployment contrasts), E9 (C_FULL untuned, bias direction), the decomposition table [5-R3] |
| 4 | "You changed the protocol 20+ times; how do we know it was before the reads?" | Generated Appendix B timeline (E3); external hash timestamp to the supervisor (D5); C5 freeze of the arm set |
| 5 | "Who did this work?" (the AI-agent process; Queen's policy) | E1 statement with named roles and the owner's review; supervisor's advance approval; E8 package |
| 6 | "Your 'browser guidelines' come from a server CPU." | B2, C9, the upgraded browser protocol [11-R5], a one-visit training-cost proxy [11-R9], and coarse energy [11-R8]; otherwise downgrade the claim [11-R13] |
| 7 | "Events-to-parity is confounded; where is the controlled experiment?" | C10: the minimal DATA-07 set, matched-h parity and a persistence rule. Otherwise use "retention vs available history (observational)" wording only |
| 8 | "Your CIs ignore training noise and multiplicity; FL was tuned on users you report." | C13, C14; state the CI scope (E2) |
| 9 | "TEST was already seen, contains Black Friday, and its features skip a week." | D2 (fence disclosure + counts), D4 sub-periods, D3 sensitivity; optional December holdout (D6) |
| 10 | "SASRec only wins under one checkpoint rule, and the device model is a different size." | E6 wording; D1 (TEST singles); C11 (GRU_D128 vs C_SMALL at device size) |

---------------------------------------------------------------------------------------------------------------------
## 7. Arabic summary for the owner (ملخص بسيط)
1. المشروع قوي جدًا في الدقة والنظام: لا يوجد تسريب بيانات، ولا خطأ في التقييم، وحساب الخصوصية التفاضلية صحيح (أعدنا حسابه بشكل مستقل).
2. لكن الإجابة "الواقعية كأنها في Edge" غير مقنعة بعد: التعلم الموحد يحتفظ الآن بحوالي 25% فقط من جودة نموذج السحابة (على CALIB).
3. خوارزمية بسيطة بدون تدريب ("أعد آخر منتج شاهده المستخدم") تحصل على حوالي 44%، فقد يظهر التعلم الموحد أسوأ منها.
4. السبب ليس خطأ برمجيًا، بل إعدادات التدريب الموحد القديمة. أضيفوا الليلة قبل قراءة نتائج S2 تجربتين لـ FedAvgM وتجربتين لـ SGD مع FedAdam.
5. قصة الخصوصية مبالغ فيها: التحديثات المرسلة تكشف المنتجات التي شاهدها المستخدم، ولا يوجد تجميع آمن (secure aggregation).
6. FT_C وRULE في مستوى خصوصية خاطئ. يجب كتابة نموذج تهديد واضح.
7. هناك أربعة أعطال برمجية ستوقف خطوات مسجلة لاحقًا: أداة إعادة التقييم ترفض C_FULL وPRE، وأداة dump_rows تتعطل، وS8 غير مقبول في سكربت التحليل، والكود الإنتاجي محفوظ فقط في مجلد مؤقت.
8. معظم الإصلاحات رخيصة ومسموحة الآن، لأنه لم تُقرأ أي نتيجة محكومة ولم يُفتح TEST.
9. اجعلوا فرع النشر الواقعي (PRE ثم FA_WARM وFT_PRE وS8) هو الإجابة الرئيسية عن سؤال Edge، وقارنوا كل نتيجة مع خوارزمية "آخر منتج".
10. جامعة Queen's تشترط إقرارًا باستخدام الذكاء الاصطناعي في الرسالة، ويجب أخذ موافقة المشرف عليه.
11. التقييم العام 7 من 10، ويمكن أن يصل إلى 8 أو 8.5 إذا نُفذت قائمة MUST قبل أول قراءة للنتائج.

---------------------------------------------------------------------------------------------------------------------
## 8. Records
- Files: 01_PROJECT_MAP.md (lead; one correction applied, §3.2), 02–15 (experts), this file. Nothing else was written.
- Not done by this review (hard rules): no DECISION_LOG row, no RESUME_HERE edit, no thesis-reference capture, no amendment.
  The Ops session should capture this folder as a new source batch (next free id S509, per RESUME_HERE §2n), and turn Gates A and
  C into registrations. Gate A is time-critical.

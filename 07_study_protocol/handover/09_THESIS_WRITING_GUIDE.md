# 09 — Thesis writing guide

*Start from the existing drafts: `company/thesis_draft/outline.md` (copy: `HANDOVER/reference/THESIS_OUTLINE_DRAFT_20261002.md`),
`chapter3_methodology.md`, `chapter4_experimental_setup.md` (written before any result; `[RESULT PENDING]` slots). The
living reference `../thesis_reference/CLAUDE_EXECUTION_REFERENCE_EN.md` §14.x and `STATE.md` hold the sourced history up to
DL 353. Never state a TEST number before the sitting; never quote UNSEEN_VAL selection-panel values without the owner's
agreement.*

## 1. Proposed structure and the evidence for each part

| Chapter / section | Content | Evidence (files; DL rows) |
|---|---|---|
| 1 Introduction | motivation (browser-side shopping assistance, privacy); the question verbatim; no threshold; Edge framing (cloud / FL weights only / small local model); contributions as questions answered; departures from the proposal | proposal; DL 83, 86; 06N; 09-30 handover §2 (departures table) |
| 2 Background | FedAvg / FedProx / server optimisers (FedAdam, FedAvgM) and why FedBN does not apply (LayerNorm); federated recommendation; SASRec / GRU4Rec and full-catalogue evaluation (sampled metrics bias); FL initialisation / pretraining; DP-FedAvg, RDP accountant, Poisson subsampling; on-device ML / ONNX / INT8 | outline §2; `company/research/LITERATURE_REGISTER.json`; BM-00; 06C; 06G; 06N header citations |
| 3 Data and task | REES46, frozen v1, task, catalogue, USER_HISTORY_L50, S* features, splits, populations and disjointness, CALIB ⊂ MAIN | file 02; `DATA_PROTOCOL_LOCK_V2.json`; DL 31–55, 151, 207 |
| 4 Methods | backbone contest (P2 tuning, P3, G2, T1 rule and the CB reversal); models and payloads; FL recipe and the ladder S0 → S4; warm start S7 / S8; DP design; device arms and selections; baselines; LAST anchoring; privacy matrix and threat model | file 01; FL_IMPROVEMENT_LADDER; 06A–06P; DL 129–138, 181–271, 292 |
| 5 Pre-registration and evaluation protocol | amendments timeline, register-before-read, single VALIDATION read, freezes v1–v3 and why, ALL_METHODS_LOCK, TEST_EXPOSURE_REGISTER, T0a, statistics (bootstrap, Holm, TOST, harmed share, SEED_SENSITIVE), claim vocabulary | file 03; REPORT §12, §14; `ALL_METHODS_LOCK.json` |
| 6 Results (Phase 1) | primary pair; retention ladder across systems; pretrained start; FL variants and communication; privacy cost; personalisation; panels and cold users; device quantisation; the D1 diagnostic (exploratory) | file 04 (VALIDATION; replace by TEST when available, keep VALIDATION beside it); REPORT §1–§11; METRIC_TABLE |
| 7 Phase 2 (follow-up study) | motivation (K21, D1), design, controls, selection, results Q1–Q14 with the §9.4 reading grid | file 05; PREG2; P2_CONTRASTS_v3 |
| 8 External benchmarks | protocol, Track A / B, published comparisons, claim rule, Wave 2 | file 06; amendment 07 + B12–B16; LOCKs and TEST_LOOK files per dataset |
| 9 Discussion | what pretraining changes and assumes (consent band); data minimisation vs privacy; why the first FL recipe was weak (five mistakes); why warm FL was a near no-op; deployment guidelines (payload cap, frozen tables, 8-bit uploads, INT8, on-device fine-tuning); limitations | files 04 §10, 05, 13; 09-30 handover §9.1 |
| 10 Conclusion and future work | the answer with retention and intervals; future work: large teacher on opted-in users distilled into the small model (DL 276), DP at full-pool scale (DP8_FULL), real secure aggregation / distributed DP, December holdout, phones | DL 276; 06I §5; PREG2 §7 |
| Appendices | amendment list with shas; contrast files; arm list; code-tree digests; incident log; declared-not-run list; AI-use statement | file 03; `deploy_archive/`; 09-30 handover §13 |

**Figures / tables to plan (multi-expert review E4 suggested 5 figures + 4 tables):** (1) the retention ladder by system and
privacy level (REPORT §4 / §9); (2) quality vs cumulative bytes per client (REPORT §5c, `curves/bytes*.csv`); (3) the FL
ladder S0 → S4 on CALIB (FL_IMPROVEMENT_LADDER); (4) per-mark curves of cold FA vs warm S8 vs PRE (REPORT §5c, rollout §12b);
(5) retention by support bin / stratum and panels (REPORT §5d–§5e, §6); tables: the confirmatory contrasts (REPORT §1), arms
× metrics (METRIC_TABLE), the privacy matrix (REPORT §9), the benchmark Track-A table vs published rows.

## 2. Mandatory disclosures (write each one in the thesis)

1. **Threat model:** an honest-but-curious browser vendor; retailer logs out of scope (06L_C C2).
2. **DP statement:** central DP with a **trusted aggregator**, client-level add / remove adjacency, (ε, δ = 5e-6), conditional
   on the public bootstrap block (catalogue, transform statistics, θ0 / θ_pre); the FL tuning on CALIB, the clip norm S and the
   PB checkpoint choice are not accounted; DP noise drawn from a public deterministic non-cryptographic seed (simulation
   guarantee only, 06E §9(b)); the per-round noise-to-signal ratio is a property of the simulated scale (06D §3; 06I §6(b)).
3. **Non-DP federated updates reveal the viewed items**; secure aggregation is **not simulated**; S8 removes the input-table
   leak only partly (06I §7(a); 06L_C C1, C8). WB3 (Phase 2) reveals the set of touched item IDs (DL 418).
4. **θ_pre is a single seed (2026)** shared by both warm seeds: their spread measures FL variance only (06K note 2; PREG2 D-c).
5. **PRE's data advantage:** PRE saw 1,217,610 users (9× MAIN); K20 > 100 mixes a data advantage; the warm line is reported
   against C_FULL (06K note 1; DL 273). PRE reused the SMALL recipe at 9× data without re-tuning (note 3).
6. **C_FULL is a larger (D256, 3-block) model** than the device model and its recipe was not re-tuned at 1.35 M users; the C_FULL
   − PRE gap is confounded by size (PREG2 D-d; C2 is the matched reference in Phase 2).
7. **CALIB ⊂ MAIN** and CALIB_VALIDATION lies in the read window (06A §1; PREG2 D-b); the CALIB-disjoint diagnostic is reported.
8. **Backbone wording:** "SASRec was selected under the registered rule T1"; at the 6.0 endpoint the order reverses (GRU −
   SASRec +0.004213, DL 138); never "SASRec beats GRU" (06B §7(c)).
9. **Recipe-qualified wording:** "matched-exposure retention of the registered FL recipe"; never "FL retains X %" alone, never
   "a lower bound" (06I §6(c); 06L_C C6). K20b / K21 are "under the registered recipe".
10. **VALIDATION → TEST order:** VALIDATION values are max-of-marks; TEST is the only unbiased number (06A §9(g)); TEST was
    previously inspected at period level by the v1 campaigns (register entry 3); TEST contains Black Friday (06B §7(b)); the
    partition-fence detail (multi-expert review D2).
11. **Post-validation labels:** D1 and everything after DL 400 on VALIDATION are POST_VALIDATION_EXPLORATORY; Phase-2 claims
    carry disclosure D-a verbatim: *"Phase-2 hypotheses were generated after the Phase-1 VALIDATION read; arms were selected on
    CALIB; everything was locked before any TEST value existed."* Plus D-h (power), D-i (DP small-step), D-j (futility on CALIB),
    D-k (TEST held for Phase 2).
12. **Benchmarks:** B13 → B15 ("B13 rescinded before its trigger on schedule grounds; no value read"); B16 ("device class
    switched before any ML-1M value was read, on schedule grounds; the CPU runs are sealed SUPERSEDED_DEVICE and were never
    opened"; device class differs by dataset; ratios are within-dataset); B12 (per-seed θ_pre on benchmarks); Track-A 6-EFE
    ceiling vs published up-to-200-epoch training; published SAS+ / eSAS could not reach some holdout targets (R2).
13. **C2 decision history** YES → NO → YES within one day, before any Phase-2 run (DL 415, 424, 427).
14. **Freezes and refusals:** the analysis freeze needed v2 and v3 after two value-free tool refusals (DL 389–397), never read.
15. **Other registered disclosures** (REPORT §14 verbatim list): FL μ and server LR untuned in the early design; FedBN
    inapplicable; purchase-AUC / comparable offers out of scope; battery not measured; per-client batching reorders 6 / 1,251
    rows at ranks > 20; downstream on A40 vs contest on A16; UNSEEN_VAL is a selection panel; FT on TRUE_COLD / SPARSE is an
    extrapolation; the 170 MB cap origin; inference download vs FL payload are different quantities.
16. **K16:** fallback subset of 16,384 users; device latency = pass 2; pass-1 latency is loaded-host.
17. **Departures from the proposal** with reasons (task, datasets, metric, FedBN, threshold dropped, corrected bibliography:
    proposal refs [5], [6], [8], [12]; 06B §7(a)).

## 3. How to frame K21 and Phase 2 honestly

- State the registered result first and fully: K20b 88.81 % and K21 −0.000433 (VALIDATION; TEST when available). The
  deployment-branch claim needs both.
- Then the exploratory diagnosis, labelled as such: D1 shows the warm FL barely moved θ (0.5–0.7 %); the warm screen was a
  cold-start recipe with an upward-only client axis and an edge winner. Conclusion: *this recipe* did not add value; the
  study does not show that FL cannot.
- Phase 2 is a **pre-registered follow-up**, selected on CALIB, locked before any TEST value, scored in the same sitting
  (T0a). Report Q1 with its registered reading (SUCCESS / EQUIVALENT / FAILURE / INCONCLUSIVE), Q2 / C2 as the headroom
  controls, and the interpretation grid. Do not promise or imply a gain: the plausible ceiling (~0.001) is near the MDE
  (~0.0005), so INCONCLUSIVE is a likely, honest outcome (D-h).
- Whatever Phase 2 finds, the practical message on VALIDATION is already clear: pretraining on consenting users plus
  on-device fine-tuning (FT_PRE) is the strongest deployable option and needs no federated traffic; federated rounds must be
  justified by a measured gain over that.

## 4. Writing rules

- Every number with its source (file + DL row) and its label (MEASURED / INFERENCE …); VALIDATION vs TEST always explicit.
- Use the claim vocabulary (REPORT §12): "exceeds" only if the CI excludes the reference; "no detectable difference"
  otherwise; "equivalent" only for TOST-tested contrasts (K14, K14w, Q1, Q13).
- Fill the number-free `HEADLINE_TEMPLATE.md` only from the TEST RETENTION_RESULT.
- Keep Track A / Track B and benchmark / REES46 numbers separate; benchmark ratios are DESCRIPTIVE.
- Citation checks still open (outline gaps): the eSASRec author list (Tikhonovich 2025 per the Fable audit, DL 329), the
  proposal's FedRec "95 % / 60 %" claim (not re-verified), the corrected proposal references.

## 5. AI-use and originality statement (Queen's requirement)

Queen's asks for a "Statement of Originality and Disclosure of Use of AI" (multi-expert review E1; supervisor pre-approval was
an open question on the 30 Sep deck). The records show that planning, code, operations, reviews and drafts were produced by AI
agents (Claude models; some early code by OpenAI Codex) under the owner's direction, with the owner taking every binding
decision (file 00 §5; file 03 §8). Write the statement accordingly and have the supervisor approve it.

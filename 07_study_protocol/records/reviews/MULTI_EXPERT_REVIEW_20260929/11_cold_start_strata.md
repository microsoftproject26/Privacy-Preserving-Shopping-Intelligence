# 11 — Cold start and user strata (expert review, part 10)

Reviewer: Claude Opus 5.5 acting as a cold-start / user-heterogeneity expert. READ-ONLY. Written 2026-09-29.
Tags: **MEASURED** (read from a file, path:line), **REPORTED** (a project document's own claim), **INFERENCE**, **OPINION**.
Paths are relative to `.local_ai/ppsi_v4/` unless absolute. I read no RETRO_TEST row, no RESULT_ROWS value, no ranks or metrics
file, and no runs_* content. I did not run any test or project script. No DATA-07 run exists yet (DL 178: "none exists yet";
no later DL row launches T41 or Study A/B, DL 180–225), so **no cold-start governed value has been read** and everything
below can still be registered before the read. `ANA_SCRIPT_FREEZE.json` is not present in `company/gates/` (MEASURED, directory
listing), so the analysis script is not frozen either.

---

## 1. What the project does (MEASURED)

### 1.1 The proposal's asks
- Step 4 (proposal PDF, pdftotext lines 116–119): "Study how many local browsing events a new user needs before on-device
  recommendations match centralized quality. Analyze: (a) Time to quality parity, (b) Few-shot adaptation from initial
  interactions, (c) Impact of category diversity on learning speed." Step 3 (L113–114) repeats "how many local events before
  on-device matches centralized?". Contributions (L159–162): "Persona-specific analysis showing which user behavioral patterns are
  most/least affected by privacy constraints" and "Cold-start characterization … broken down by user type". Expected outcome
  (L193–194): "pre-trained models + few-shot local adaptation enable useful recommendations from day one".

### 1.2 User populations (strata by TRAIN support)
- Owner amendment POP 5–100 (DL 31): TRUE_COLD (0) / SPARSE (1–4) / MAIN (5–100) / HEAVY (>100) eligible TRAIN examples; only
  MAIN is the matched benchmark.
- Manifests (MEASURED, `contracts/DATA_PROTOCOL_LOCK_V2.json` L258–L410): MAIN_VALIDATION 51,211 users; UNEW_HOLDOUT_VALIDATION
  12,862 users (support = own TRAIN examples, never in any training set); SPARSE_PANEL 16,384 users (37,432 support decisions,
  ≈ 2.3/user); HEAVY_PANEL 16,384 (3,036,857 support decisions, ≈ 185/user); TRUE_COLD_VALIDATION 381,295 users (eval-only);
  COLD_H_16384 (16,384 users, 50,529 panel queries, 151,542 support rows, parent TRUE_COLD); WARM_TRUNCATION_16384 (16,384 MAIN
  users, 177,709 queries, 1,251,955 support rows). Definitions: `company/tasks/DATA-06.md` L19–L23 (TRUE_COLD = n = 0 users with
  ≥ 1 E2E VALIDATION row, split A0 299,035 no TRAIN event / A1 82,260 TRAIN events but no label; COLD_H = the 16,384 smallest
  selection_key_v2 of TRUE_COLD; SPARSE/HEAVY = smallest-key stratum B/D users with ≥ 1 E2E row; WARM_TRUNCATION = smallest-key
  MAIN users with ≥ 1 E2E row). Disjointness: `company/handoffs/DATA-06.md` L84–L88.
- Note: REALISM_AUDIT D2 says "TRUE_COLD = 381,295 users with no TRAIN events" (`company/reviews/REALISM_AUDIT_20260929.md`
  L118). That is inaccurate: 82,260 of them have TRAIN events but zero eligible labels (DATA-06 handoff L57; CLIENT_CONTRACT C3.2).

### 1.3 Primary cold-start analysis: observational support bins (06A §2)
- `contracts/PREG1_AMENDMENT_06A.md` item 2: "retention vs C_BIG (and vs C_SMALL) within bins of the client's TRAIN-period
  support count n_support ∈ {0 (TRUE_COLD), 5–10, 11–20, 21–50, 51–100} on MAIN_VALIDATION and on UNSEEN_HOLDOUT … events-to-
  90%/95% = the smallest bin b with M(b) ≥ X · C_ref(b), C_ref = C_BIG PRACTICAL_BEST on the same bin, same seed; reported only if
  reached; no interpolation … Disclosed: observational (history length is not manipulated)."
- Code: `tools/retention_report.py:174` `SUPPORT_BINS = ((0,0,"0"), (5,10,…), …, (51,100,"51-100"))`; `:178`
  `BIN_PANELS = ("MAIN", "UNSEEN_HOLDOUT")`; `:2140–2144` `support_bin` maps 1–4 and > 100 to `OUT_OF_REGISTERED_BINS` (dropped from
  the curve, `:2162`); `:2168–2193` `events_to_x_bins`: bin 0 always comes from the TRUE_COLD role (`:2182`); the first bin in
  order whose point M ≥ X·C_ref is returned, else "NOT_REACHED" (`:2189–2192`). Values are per seed, point estimates.
  Per-bin retention rows do carry a paired-bootstrap CI (`_subgroup_rows`, `:2087–2108`).

### 1.4 Controlled truncation study DATA-07 (contract §11; 06B §6.4)
- Contract rules (MEASURED, `contracts/EVALUATION_CONTRACT_V4.md`): h ∈ {0,1,5,10,20,50} counts observed events, not labels
  (E11.1, L692–L694); query event always visible, h = strictly earlier events, H_h = the h most recent prior events (L716–L731);
  FIXED_50 / FIXED_20 = the same query q_u for every h; ALL_AVAILABLE(h) = earliest query with ≥ h prior events, queries differ by
  h, reported separately (L723–L731); support labels strictly before decision_ts(q), query target never support (E11.3, L734–L742);
  LOCAL_HISTORY recomputed from the window, SHARED frozen, frozen normalisers (E11.4–E11.5, L744–L793); Study A = context-only with
  frozen parameters (E11.8, L795–L802); Study B = adaptation on own support with a frozen recipe (E11.9, L804–L824); events-to-X
  (E11.10, L826–L836): "C_ref … the central C method … CONTEXT_ONLY; the same queries; the full available prior history …
  events-to-X% is the smallest registered h with M(h) ≥ X · C_ref … No interpolation, no extrapolation."
- Scope in force: 06B §6.4 "Study A on WARM_TRUNCATION_16384 and Study B on COLD_H_16384"; 06D §8 tier **SHOULD-medium**: "DATA-07
  T41 on both panels + Study A (evaluation-only) on WARM_TRUNCATION; Study B (FT_C, LO) on COLD_H seed 2026"
  (`contracts/PREG1_AMENDMENT_06D.md` L101); 06I §8 keeps it SHOULD-medium (L75). MASTER_PLAN_V5 §8.3 (L162) runs Study A with the
  default panel (WARM_TRUNCATION) only; §13 cut order cuts Study B then Study A/T41 early (L229–L231).
- Code (MEASURED, `code/v4int/launch_trunc.py`): `ARMS_B = ("FT_C", "LO")` (:66); Study A may run on either panel, default
  WARM_TRUNCATION (:68–69); Study B cells are H_h only, no FULL cell (:766–767); `events_to_x` first-hit rule (:115–128);
  `events_to_x_table` takes C_ref = C_BIG Study A FULL row of the same panel and seed (:139–182) and reads M's own FULL row with
  `val(pm, p, hq, True)[0]` (:176).
- T41 (REPORTED, `company/handoffs/DATA-07R1.md` "T41 golden replay"): full bucket 0 — 173,921/173,921 VALIDATION query decisions
  bitwise equal except 7 admitted brand-OOV cells (inside 48 flagged rows); 622,168/622,168 TRAIN loss decisions, 0 admitted;
  negative controls (future-event leak, no carry-in, merged VS/VC stream, future-event receipt) fail as intended. Production T41
  on the two panels also checks deletion invariance on 512 sampled (q, h) with h < available (`launch_trunc.py:78`, `:513–520`)
  and a full-history user-context leak control (`:521`). No server T41 receipt exists yet (DL 178).
- Study B recipe: the arm's UNSEEN_VAL `SELECTION.json` (`launch_trunc.py:185–206`), i.e. selected on 5–100 warm unseen users
  (06A §1; 06E §1). The contract's registered cold recipe panel M04S (E11.9a, L812–L817: "M03 users at population-key ranks
  16,384–32,767 … selected once, on M04S FIXED_20") was not rebuilt under POP 5–100 (REGISTRY_V2 has no M04S entry, MEASURED).
- TEST: "M09 (TEST cold-h) is NOT registered" (06D L41) — DATA-07 is VALIDATION-only; the bins/panels do get TEST rows
  (UNSEEN_HOLDOUT_TEST, TRUE_COLD_TEST, SPARSE/HEAVY_TEST, 06D L38–L48).

### 1.5 User strata / personas (amendment 04 §3; DL 135)
- `contracts/PREG1_AMENDMENT_04.md` §3: "strata from TRAIN-period events with cut points fixed on MAIN TRAIN before any read
  (category-diversity terciles, brand-loyalty terciles, persona 3×3, activity quantiles)". DL 135: cut points registered on the
  server (file 0406e24c…, self 31a5d45f…). The file is **not in the local mirror** (`contracts/strata/` is empty, MEASURED), so I
  could not verify cut values or group sizes.
- Definitions (MEASURED, `code/v4base/strata.py`): diversity = number of distinct known main categories and loyalty = top-brand
  share, both over the user's **single longest TRAIN window** (≤ L = 50 events) (`:87–101`, docstring L22–L27); activity =
  quartiles of n_train decisions (`:58–60`, judgment 1); ties at a cut fall into the lower bin (docstring judgment 4). Strata are
  computed for MAIN only (`tools/retention_report.py:2117–2137`); non-MAIN users get NO_TRAIN. Output: R vs C_BIG per group with CI,
  label DESCRIPTIVE (`:2101–2106`). All strata/bin outputs are DESCRIPTIVE; confirmatory = K1–K21 only (06B §2).
- Harmed-client share is reported for paired deltas (`retention_report.py:1550–1581`).

### 1.6 Unseen users (participation gap)
- MASTER_PLAN_V5 §12.2 row 7.6 (L215) promises a "UNSEEN_HOLDOUT participation gap". The report builds per-panel retention tables
  (`retention_report.py:2429–2431`) and bins on UNSEEN_HOLDOUT, but there is **no code that computes a gap** (MEASURED: no
  "gap" computation in retention_report.py; grep).

---

## 2. State of the art (literature and real systems; verified citations only)

**Cold-start evaluation and methods.**
- Schein, Popescul, Ungar, Pennock, "Methods and metrics for cold-start recommendations", SIGIR 2002, doi:10.1145/564376.564421
  — the classic framing of cold start as its own evaluation regime with its own metrics.
- Lam, Vu, Le, Duong, "Addressing cold-start problem in recommendation systems", ICUIMC 2008, doi:10.1145/1352793.1352837.
- Volkovs, Yu, Poutanen, "DropoutNet: Addressing Cold Start in Recommender Systems", NeurIPS 2017
  (https://proceedings.neurips.cc/paper/2017/hash/dbd22ba3bd0df8f385bdac3e9f8be207-Abstract.html) — **separate warm, user-cold
  and item-cold test splits**, with one model evaluated in all three.
- Lee, Im, Jang, Cho, Chung, "MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation", KDD 2019,
  doi:10.1145/3292500.3330859; Dong et al., "MAMO: Memory-Augmented Meta-Optimization for Cold-start Recommendation", KDD 2020,
  doi:10.1145/3394486.3403113 — few-shot user adaptation from a small support set with a query set; the support/query episode is
  the same structure as the project's E11.3/Study B.
- Wang, Ding, Caverlee, "Sequential Recommendation for Cold-start Users with Meta Transitional Learning" (MetaTL), SIGIR 2021,
  doi:10.1145/3404835.3463089 — cold-start **sequential** recommendation as a few-shot problem over a user's first transitions.
- Hidasi, Karatzoglou, Baltrunas, Tikk, "Session-based Recommendations with Recurrent Neural Networks", ICLR 2016, arXiv:1511.06939
  — session models for anonymous or new users; this is why a context-only sequence model (Study A) is a real cold-start baseline.
- Ji, Sun, Zhang, Li, "A Critical Study on Data Leakage in Recommender System Offline Evaluation", ACM TOIS 41(3), 2023
  (arXiv:2010.11060) — use a global timeline. The project's chronological split and strictly-causal support rule meet this.

**Federated / on-device cold start and personalisation in real systems.**
- Yuan, Morningstar, Ning, Singhal, "What Do We Mean by Generalization in Federated Learning?", ICLR 2022
  (https://openreview.net/pdf?id=VimqQq-i_Q) — separate the **participation gap** (clients never trained on) from the
  out-of-sample gap; UNSEEN_HOLDOUT is this design.
- Wang, Mathews, Kiddon, Eichner, Beaufays, Ramage, "Federated Evaluation of On-device Personalization", arXiv:1910.10252, 2019
  (Google, Gboard) — personalisation judged per user in production, with the distribution of per-user gains ("a significant fraction
  of users benefit"). The project's harmed-client share is the same idea.
- Paulik et al. (Apple), "Federated Evaluation and Tuning for On-Device Personalization: System Design & Applications",
  arXiv:2102.08503, 2021 — personalisation hyper-parameters are tuned federatedly on the deployed population (news, ASR).
  Analogue: selecting the FT recipe on a disjoint population (UNSEEN_VAL). Apple tunes on the population it deploys to.
- Zhang, Long, Zhou, Zhang, Yan, Yang, "When Federated Recommendation Meets Cold-Start Problem: Separating Item Attributes and User
  Interactions" (IFedRec), WWW 2024 (arXiv:2305.12650) — item cold start in FedRec. Wahab, Rjoub, Bentahar, Cohen, "Federated
  against the cold: A trust-based federated learning approach to counter the cold start problem in recommendation systems",
  Information Sciences 601:189–206, 2022. Neither measures "events to parity" against a central model. **OPINION:** I found no
  FedRec paper that does, which makes the project's Step 4 design a real contribution if it is executed.

**User-group heterogeneity and fairness.**
- Ekstrand et al., "All The Cool Kids, How Do They Fit In?: Popularity and Demographic Biases in Recommender Evaluation and
  Effectiveness", FAT* 2018, PMLR 81:172–186 (https://proceedings.mlr.press/v81/ekstrand18b.html) — report effectiveness per user
  group, not only in aggregate.
- Li, Chen, Fu, Ge, Zhang, "User-oriented Fairness in Recommendation", WWW 2021, doi:10.1145/3442381.3449866 — groups users by
  **activity** (advantaged vs disadvantaged) and shows systematic quality gaps. Rahmani, Naghiaei, Dehghan, Aliannejadi,
  "Experiments on Generalizability of User-Oriented Fairness in Recommender Systems", SIGIR 2022, doi:10.1145/3477495.3531718 —
  the conclusions depend on how users are grouped and on the base model.

**Practice that follows from this literature (OPINION, supported by the works above).** (i) Keep warm, user-cold and item-cold
evaluations separate (DropoutNet; Schein). (ii) Measure cold-start curves with a support/query protocol on the same users
(MeLU/MAMO/MetaTL). (iii) Keep the participation gap separate from within-client generalisation (Yuan et al.). (iv) Report per-user
gain distributions, not only means (Wang et al. 2019). (v) Group users by activity, and treat activity as the main confounder of
any other grouping (Li et al.; Rahmani et al.).

---

## 3. Comparison with the state of the art

### 3.1 Strengths
- **S1 Population coverage is better than typical FedRec work.** TRUE_COLD (381,295), SPARSE (1–4), MAIN (5–100), HEAVY (> 100)
  and a user-disjoint UNSEEN_HOLDOUT are all built, frozen and hashed (DATA_PROTOCOL_LOCK_V2 L258–L410). This covers the
  DropoutNet-style warm vs user-cold split and the Yuan et al. participation gap.
- **S2 The truncation design is causally sound where it matters.** FIXED_20/FIXED_50 use the same user, the same query and the
  same target, and vary only the permitted history (contract L723–L727). That identifies the effect of available history for those
  queries. Support is strictly causal (E11.3).
- **S3 T41 is stronger than anything I know in the cold-start literature.** Bitwise golden replay at h = FULL against the
  production feature path, deletion invariance (the h-window output does not change when pre-window events are physically
  deleted), and a leak negative control. The 7 admitted brand-OOV cells (≈ 0.004 %) are disclosed and justified (the brand strings
  are not in the frozen data). The deletion-invariance gate is what proves correctness at intermediate h, not only at FULL.
- **S4 Pre-registration.** Bins, h grid, panels, strata cut points and the events-to-X rule were all fixed before any read (06A,
  DL 135, contract §11). Nothing has been read, so improvements are still legal (see §4).
- **S5 Labels are kept distinct.** WARM_TRUNCATION is never reported as cold start (B-F3 fix; `launch_trunc.py` trunc_labels).
- **S6 Per-user outcome distributions.** The harmed-client share on the paired deltas matches the Gboard federated-evaluation
  practice.

### 3.2 Weaknesses and errors

**W1 — The registered primary answer to Step 4 is observational and confounded (INFERENCE, agreeing with Fable C.9,
`FABLE_THESIS_RESEARCH_REVIEW_20260928.md` L186–L188).**
- Users in the 51–100 bin differ in kind from users in the 5–10 bin: more engaged, more sessions, more VALIDATION queries each,
  and longer in-VALIDATION context.
- The within-bin ratio M(b)/C_BIG(b) is a valid paired comparison on the same users. The **trend across bins** is not the causal
  effect of history.
- **The bin variable is not the inference-time history.**
  - n_support counts eligible TRAIN *labels*, not events.
  - A VALIDATION query's context also contains that week's events, capped at L50.
  - So a 5–10-bin user can be scored with a 50-event context. For context-only arms (C_SMALL, FA, DP) the bin mostly indexes
    user type.
  - Above ≈ 50 events the context is identical (L50 cap), so the 51–100 and HEAVY differences are pure user type.
- The regime that *is* the cold-start problem is missing from the curve.
  - 1–4 labels map to `OUT_OF_REGISTERED_BINS` (`retention_report.py:2144`), and BIN_PANELS excludes SPARSE_PANEL (`:178`).
  - The curve jumps from TRUE_COLD ("0") to 5–10.
  - The SPARSE panel is evaluated (MASTER_PLAN §6.6–6.7), but it is not placed in the curve.
- "Bin 0" (TRUE_COLD) is a different population. Those users have no TRAIN labels but have in-week context. For Study B arms at
  bin 0 there is no support at all, so FT_C = C_SMALL and LO = θ0. Few-shot adaptation *within* the week is not measured anywhere
  in the bins.

**W2 — "events-to-X" as defined will probably be degenerate or noisy (INFERENCE).**
- **Bins, matched C_ref.** For context-only arms, M(b)/C_BIG(b) has no reason to rise with b: two fixed global models get the
  same context.
  - If R is ≥ 90 % at bin 0, the first-hit rule (`retention_report.py:2189–2192`) returns "0" ("no events needed").
  - If R is below 90 % everywhere, it returns NOT_REACHED. With FA at the currently REPORTED CALIB level (DL 218: FA 0.0288 vs
    the ≈ 0.11–0.12 central values quoted in the project map), NOT_REACHED is likely for every FL arm.
  - Neither answer is a "time to parity".
- **The first-hit rule does not require parity to persist.** One noisy bin above X ends the search.
- **No uncertainty is used.** Both rules use per-seed point values (`launch_trunc.py:124`; `retention_report.py:2191`). No CI
  enters the parity decision.
  - COLD_H has 50,529 queries across all its sub-panels (DATA_PROTOCOL_LOCK_V2 L392), so the FIXED_20/FIXED_50 cells on COLD_H
    are probably small (eligibility is PENDING; E11.2 reports it at launch).
  - Rough sizing, INFERENCE with an assumed per-query MRR s.d. ≈ 0.3 and n ≈ 2,000: SE ≈ 0.007 on a value ≈ 0.15, so ±9 % at 95 %
    on R(h).
  - At that noise level, "90 % vs 95 %" parity cannot be resolved.
- **DATA-07 C_ref is C_BIG at FULL history (E11.10), not C_BIG at the same h.** That answers "when does a new on-device user reach
  the quality a fully-profiled cloud user gets", which is a legitimate question. It is not the fair *paradigm* comparison.
  - For a genuinely new Edge user, the cloud also has only those h events.
  - The Edge-relevant parity is M(h) vs C_BIG(h) on the same queries.
  - Study A produces C_BIG(h) rows, but neither `events_to_x_table` nor `trunc_curves` (`retention_report.py:2250–2262`) computes
    the matched-h ratio.

**W3 — The scope in force cannot deliver the few-shot parity result (MEASURED from code plus the plan text).**
- **No C_ref on COLD_H.** 06D §8 and MASTER_PLAN §8.3 run Study A only on WARM_TRUNCATION. Study B runs on COLD_H. Both the E2X
  tool and `trunc_curves` need a C_BIG Study A FULL row **on the same panel** (`launch_trunc.py:156–158`, "one seed and one
  panel"; `retention_report.py:2250–2253`). So Study B (FT_C/LO on COLD_H) has no C_ref.
- **Wrong status when C_ref is missing.** `trunc_curves` then sets R = NO_C_REF but reports `events_to_90 = "NOT_REACHED"`
  (`:2253`, `:2259–2261`). That reads as a substantive negative finding. It should read NO_C_REF.
- **The E2X command cannot process Study B rows.** Study B rows have no FULL cell (`launch_trunc.py:766–767`), so
  `events_to_x_table` hits `val(pm, p, hq, True)[0]` on an empty list (`:176`) and raises IndexError, not LaunchRefused.
  - INFERENCE from the code path; no test covers E2X on Study B rows. `test_studies.py` E2X tests use Study A rows only.
  - The documented usage lists only `SA_<arm>` rows (`launch_trunc.py:13–14`), so this is a gap, not a crash in the plan as
    written. It still means the few-shot parity table is never produced.
- **The new-user context-only curve is missing.** Without Study A on COLD_H, the thesis has no causal curve for C_BIG, C_SMALL
  and FA on new users. WARM_TRUNCATION users are MAIN users the models were trained on: their TRAIN sequences shaped the item
  embeddings. That panel measures a returning user whose device kept only h events, not a new user (the contract labels it
  correctly).
- **Other gaps.**
  - Study B has only FT_C and LO (`ARMS_B`, `:66`). The realistic federated system with few-shot adaptation (FT_FA, the L4 ladder
    level) and PF are absent, although 06A §2 lists them as the Study B arms of the bins.
  - The Study B recipe is selected on warm 5–100 UNSEEN_VAL users (median ≈ 20 support examples) and applied to 0–50 support
    labels. This is the extrapolation that 06B §7(e) discloses for TRUE_COLD/SPARSE.
  - It departs from the contract's registered cold recipe panel M04S (E11.9a). With 4 passes at 0.3× peak on 1–5 labels,
    overfitting is plausible (INFERENCE). Study B may therefore *understate* few-shot adaptation.

**W4 — "Impact of category diversity on learning speed" (Step 4c) is not answered (MEASURED).**
- Diversity exists only as a MAIN stratum of overall retention (`retention_report.py:2117–2137`).
- It is never crossed with support bins or with h, so no learning-speed-by-diversity curve exists.
- COLD_H users are NO_TRAIN in the strata registry, so the truncation panels cannot be split by diversity at all.

**W5 — The persona grid is confounded with activity by construction (INFERENCE from `strata.py:87–101`).**
- **Diversity is not normalised for window length.** It is the count of distinct main categories in one window of ≤ 50 events,
  so it rises with the number of events.
- **Loyalty is mechanically high for short windows.** Top-brand share is 1.0 with one branded view.
- **Consequence.** The diversity and loyalty terciles, and the 3×3 persona cells, are partly activity quartiles in disguise. The
  "which behavioural patterns are most affected" claim (proposal L159) cannot be separated from "light vs heavy users" (the
  Li et al. 2021 axis).
- **Unbalanced cells are possible.** Diversity is a small integer and ties go to the lower bin, so terciles may be very unequal
  (cut file not local, group sizes unverified).
- **No heterogeneity estimate.** There is no between-group contrast (for example lowest vs highest activity quartile) with a CI.
  Hundreds of descriptive CIs invite cherry-picking. Fable C.11 already asks for them to be called descriptive.

**W6 — The participation gap is promised but not computed (MEASURED).**
- MAIN and UNSEEN_HOLDOUT are exchangeable random key bands of the same 5–100 pool, so the gap is identifiable. That is a good
  design.
- The report, however, only prints R(arm/C_BIG) per panel.
- R cancels the gap whenever the arm and C_BIG share training users. The relevant quantities are:
  - M_MAIN − M_UNSEEN per arm (unpaired, two populations);
  - the difference-in-differences (FA gap − C_BIG gap): does federation generalise worse to non-participants?
- Neither is implemented.

**W7 — The causal study is VALIDATION-only (MEASURED, 06D L41).**
- The cold-start truncation curve has no held-out confirmation. The bins get TEST rows, the truncation study does not.
- For WARM_TRUNCATION, the checkpoints were selected on MAIN_VALIDATION, which contains the same queries. This gives a mild
  optimistic bias on the FULL cell; within-model ratios across h mostly cancel it.

### 3.3 Unrealistic choices (for an Edge deployment)
- **U1 "Most recent h of a long history" is not "first h events of a new user".**
  - FIXED panels truncate backwards from a query that already has ≥ 20/50 prior events.
  - The retained h events belong to a user already deep in a shopping journey, so they are more informative than a new user's
    first h events.
  - Only ALL_AVAILABLE on COLD_H approximates onboarding. For A0 users (no TRAIN events), the earliest query with ≥ h prior events
    has H_h = the user's first h events (INFERENCE from the §11.2 rule). But its composition changes with h (survivorship: users
    who reach h+1 events).
  - A user-fixed onboarding curve would restrict ALL_AVAILABLE to the users present at every h. It needs no new run if per-cell
    ranks are saved (`launch_trunc.py:681` saves them).
- **U2 One-shot FT.** Few-shot adaptation happens once, on the support available at the query. A browser would adapt
  incrementally (proposal L86–L88: incremental/online learning). This is already disclosed as one-shot (06I §7c, C5). For Step 4b
  it means "learning speed" is measured as a set of one-shot snapshots at h, not as an online learning curve.

### 3.4 Missing pieces (summary)
1. A matched-h paradigm parity.
2. A parity rule that uses CIs and persistence.
3. C_BIG/C_SMALL/FA Study A on COLD_H.
4. E2X/trunc support for Study B.
5. FT_FA in Study B.
6. A cold-user recipe selection (or a disclosure).
7. A 1–4 bin from SPARSE and a > 100 bin from HEAVY.
8. Diversity × learning-speed.
9. Activity-adjusted personas.
10. A participation-gap estimate.
11. The correct NO_C_REF status.

---

## 4. Recommendations

Governed-value status for every item: **UNREAD** (no DATA-07, panel or bin value has been read; ANA not frozen). An amendment
written now is legitimate pre-registration.

| ID | Item | Class | Action | Cost (INFERENCE) | Affects | Pre-registration needed / status |
|---|---|---|---|---|---|---|
| R1 | **Run the minimal controlled cold-start set:** T41 on COLD_H and WARM_TRUNCATION; **Study A C_BIG, C_SMALL and the selected FA on COLD_H** (plus WARM_TRUNCATION as registered); Study B FT_C on COLD_H seed 2026 (LO as registered). Raise it from SHOULD-medium to at least SHOULD-high, and treat it as MUST for any "events to parity for a new user" sentence. Otherwise the thesis must use Fable's wording "retention as a function of available history (observational)" and drop the parity claim | MUST | CHANGE (tier + panel list) | REPORTED DATA-07 estimates: T41 20–40 CPU-min per panel; Study A ≈ 15–25 min per arm-seed (COLD_H is 3.5× smaller than WARM); Study B ≈ 3–10 A16-h per arm-seed. Total ≈ 5–12 GPU-h | nothing finished; new eval-only runs from locked checkpoints | Yes: a DL row / amendment adding COLD_H to Study A's panels (06B §6.4 names only WARM for A). UNREAD, so registrable |
| R2 | **Add matched-h parity:** R_match(h) = M(h)/C_BIG(h) on the same queries, beside E11.10's M(h)/C_BIG(FULL). Headline for the Edge question = matched-h; FULL-C_ref = "time to the fully-profiled cloud experience" | MUST | ADD (analysis) | ≈ 0.5 day Sonnet in `trunc_curves` + test; C_BIG(h) rows already produced by Study A | ANA script (unfrozen) | Yes (new descriptive definition); UNREAD |
| R3 | **Parity rule with uncertainty and persistence:** parity at h iff R(h') ≥ X for every registered h' ≥ h, judged on the point value AND reported with the paired-bootstrap CI of R(h) (the shared W already exists). A "CI-lower ≥ X" variant can be added as conservative. Apply to bins and to DATA-07 | MUST | CHANGE (definition) | ≈ 0.5–1 day (bootstrap per cell from saved ranks) | ANA script, launch_trunc E2X | Yes; the registered first-hit rule stays printed as registered, the new rule as an added descriptive row. UNREAD |
| R4 | **Fix `trunc_curves` status:** NO_C_REF, not NOT_REACHED, when the C_ref row is missing (`retention_report.py:2259–2261`). Make E2X accept Study B rows (no FULL cell → M_full = None) or refuse them with LaunchRefused (`launch_trunc.py:176`), with a test | MUST | CHANGE (code) | < 0.5 day Sonnet + Opus check | ANA script, launch_trunc (no run exists) | No (bug fix); needs the usual review; ANA not frozen |
| R5 | **Add 1–4 (SPARSE_PANEL) and > 100 (HEAVY_PANEL) bins** to the support-bin curve as DESCRIPTIVE, keeping the registered five bins unchanged. Label n_support as "supervised TRAIN examples", not events | SHOULD | ADD (analysis) | ≈ 2 h; panel rows already planned (MASTER_PLAN §6.6–6.7) | ANA script | Yes (additive descriptive); UNREAD |
| R6 | **Observational analogue of h:** bin VALIDATION queries by the number of prior events at the query (min(n, 50)), besides n_support. This is the variable context-only arms actually see | SHOULD | ADD | ≈ 1 day (needs per-row history length from the manifest or eval batch; INFERENCE that it is derivable) | ANA script | Yes; UNREAD |
| R7 | **Add FT_FA to Study B** (base = the selected FA checkpoint). It is the realistic L4 "federated + few-shot" system. PF is optional | SHOULD | ADD (code + run) | code ≈ 0.5 day (ARMS_B, base kind, tests); run ≈ 3–10 GPU-h | new run | Yes; UNREAD |
| R8 | **Recipe for Study B:** either rebuild the registered M04S (TRUE_COLD key ranks 16,384–32,767, disjoint from COLD_H) and select once on its FIXED_20/ALL_AVAILABLE, or keep UNSEEN_VAL and DISCLOSE the deviation from E11.9a and the likely understatement. Also report every non-selected grid cell descriptively (LO already does) | SHOULD (select) / DISCLOSE-ONLY (minimum) | ADD or DISCLOSE | rebuild ≈ 1 day DATA + ≈ 3–6 GPU-h selection | new manifest + run | Yes if rebuilt (before any COLD_H Study B read); UNREAD |
| R9 | **Step 4(c) diversity × learning speed:** cross support bin × diversity tercile on MAIN (both registered covariates), and on COLD_H FIXED_20 split users by a diversity covariate computed on their full available prior history (fixed across h, terciles fixed on the panel before read). Report R(h) curves per tercile | SHOULD | ADD (analysis) | ≈ 1 day | ANA script; COLD_H rows | Yes (new covariate + cut rule, before read); UNREAD |
| R10 | **De-confound personas:** report the persona and diversity tables within activity quartiles, and add a length-normalised diversity (for example distinct main categories among the last 10 known events, or entropy) as an extra DESCRIPTIVE family. Keep the registered families unchanged. Print group sizes | SHOULD | ADD | ≈ 0.5–1 day (CPU strata scan) | ANA script; new registry file O_EXCL | Yes; the registered cut file is unchanged and the new file is written before read |
| R11 | **One pre-declared heterogeneity contrast** (Li et al. 2021 axis): ΔR = R(arm/C_BIG, activity Q1) − R(arm/C_BIG, Q4) for K2 (FA) and K8 (LO), with a paired bootstrap CI. Label DESCRIPTIVE-with-CI; everything else stays descriptive | SHOULD | ADD | ≈ 0.5 day | ANA script | Yes; UNREAD |
| R12 | **Participation gap:** per arm M_MAIN − M_UNSEEN (unpaired two-population bootstrap), and the difference-in-differences (FA gap − C_BIG gap) as the federated generalisation cost to non-participants (Yuan et al. 2022) | SHOULD | ADD | ≈ 0.5 day | ANA script | Yes; UNREAD |
| R13 | **User-fixed onboarding curve on COLD_H:** ALL_AVAILABLE restricted to users present at every h (from saved per-cell ranks), as a descriptive "first h events" curve (U1) | NICE | ADD (analysis) | ≈ 0.5 day | ANA | Yes; UNREAD |
| R14 | **TEST confirmation of the truncation study** (register M09 cold-h RETRO_TEST) | NICE | ADD | DATA-09-style build + Study A eval ≈ 1 day | TEST look | Yes, before ALL_METHODS_LOCK; otherwise DISCLOSE (W7) |
| R15 | **Disclosures:** (a) the bins are observational and n_support counts labels; (b) FIXED panels keep the most recent h events of an experienced user (U1); (c) WARM_TRUNCATION = returning user with truncated device history; (d) FT is one-shot, not online; (e) Study B recipe from warm users; (f) VALIDATION-only truncation; (g) correct REALISM_AUDIT D2 (TRUE_COLD includes 82,260 users with TRAIN events but no labels); (h) no meta-learned initialisation (MeLU/MAML-style) was tested — future work | DISCLOSE-ONLY | KEEP + text | 0 | thesis text | No |
| R16 | Keep T41 (golden replay + deletion invariance + leak NC) exactly as built, and require the production receipts before any Study A/B read | MUST | KEEP | 0 | — | Already registered |
| R17 | Keep the registered five-bin rule and first-hit events-to-X printed as registered (never replaced; R3/R5 are added rows) | MUST | KEEP | 0 | — | Registered (06A §2) |

---

## 5. Score: **6 / 10**

**Justification (OPINION).**
- **The infrastructure is excellent.**
  - Frozen, disjoint cold / sparse / heavy / unseen panels.
  - A truncated-history builder with a bitwise golden replay, deletion invariance and leak controls. That verification is stronger
    than anything in the cold-start literature I know.
  - Causal support rules, and strata cut points registered before any read.
- **The analysis layer does not yet turn this into a convincing answer to Step 4.**
  - The registered *primary* answer is the confounded observational bin curve. Its units are labels, not events, and it skips the
    1–4 regime.
  - The events-to-X rule is a first-hit point estimate with no CI and no persistence requirement. It is likely to return "0" or
    NOT_REACHED.
  - The controlled study sits in a low tier and is scoped so that its most important cells cannot be produced: no C_BIG on COLD_H,
    no E2X path for Study B, and a NOT_REACHED/NO_C_REF mislabel.
  - Its parity reference is full-history cloud rather than matched-h. Few-shot uses a recipe tuned on warm users.
  - Step 4(c), category diversity × learning speed, is not addressed. The persona grid is confounded with activity by construction.
- **All of this is fixable before any read, at modest cost** (≈ 5–15 GPU-h plus 3–5 analyst-days, INFERENCE). With R1–R4 done I
  would score this part 8; adding R5–R12 would bring it to 9.

---

## 6. UNVERIFIED items and open questions

- **Q1** COLD_H and WARM_TRUNCATION sub-panel eligibility (FIXED_50 / FIXED_20 / ALL_AVAILABLE(h) users per h). No local receipt
  holds them (DATA06_BUILD_RECEIPT has only totals). The power statement in W2 is INFERENCE with assumed values. The first T41 or
  Study A launch writes ELIGIBILITY.json; read that (a structural count, not an outcome) before deciding R3's CI rule.
- **Q2** STRATA_CUTPOINTS_MAIN.json is registered on the server (DL 135) but absent locally. Cut values, tercile balance and group
  sizes are UNVERIFIED. REV-ANA-01 A-F16 also recorded the file as "not found" locally.
- **Q3** `events_to_x_table` crashing on Study B rows (IndexError at `launch_trunc.py:176`) is INFERENCE from reading the code. I
  did not execute it (lead rule: no project scripts).
- **Q4** Whether the manifest or eval batch exposes per-query prior-event counts for R6 is UNVERIFIED (the COLD_H/WARM panel
  queries carry `n_prior_events_available`, `launch_trunc.py:512`; the MAIN/UNSEEN eval manifests are not checked).
- **Q5** Whether plain fine-tuning from a pre-trained model is competitive with meta-learned initialisation for few-shot sequential
  cold start in this setting: I make no claim. The cited meta-learning papers report gains on their own benchmarks, not here.
- **Q6** The ≈ 0.11–0.12 central CALIB value used in W2 is REPORTED (project map §3.5), not read by me from a result file.

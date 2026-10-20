# Run allocation: cold-start vs pretrained-start MAIN runs — decision memo (2026-09-30 ~21:00Z)

- **Author:** Claude Fable 5.1, chief scientist (federated learning / on-device recommendation / MSc methodology). READ-ONLY session:
  this file is the only artefact written. No server access, no code, contract, register or DL edit, no Codex, no git.
- **Inputs read:** RESUME_HERE §2o; DECISION_LOG 245–281; registers/FL_IMPROVEMENT_LADDER.md; REAL_WORLD_FL_PRACTICE_AUDIT_20260930;
  amendments 06D, 06K, 06L_C, 06M_B; gates/ANA_CONTRASTS_v3.json; handover §1, §6, §9 (DP rows), §14; `code/v4int/launch_fl.py`
  (arm table and init gating). Web sources opened today are listed in §1.3.
- **Values used:** only CALIB screen values already in the ladder register and the preview table (DL 267, 271). **No MAIN,
  VALIDATION, TEST, PRE or S7/S8 value was read.** Every duration and date below is INFERENCE from the MEASURED rates in §0.
- **Tags:** MEASURED (read today from a primary source, with a location), REPORTED (a project document or a paper I did not re-open
  today), INFERENCE (my estimate or arithmetic), PLAN.

---------------------------------------------------------------------------------------------------------------------
## Bottom line (for the Arabic summary)

1. **The owner is right on the science and on the practice.** Deployed federated systems start from a model that already works
   (public-data pretraining, an existing model or heuristic weights); random initialisation in production appears only once, in
   Gboard 2018 at 1.5 M clients with ≈ 400 examples per client. The FL literature since 2022–23 (Nguyen et al. ICLR 2023, Chen et al.
   ICLR 2023, "Initialization Matters" 2025) says pretrained initialisation is the realistic setting and asks that methods be evaluated
   under **both** starts. So: the realistic headline is the pretrained start; the random start is a legitimate **ablation**, not the
   answer to the thesis question.
2. **13 cold runs are not "much better" than 2.** Two cold FA seeds answer K2 (the cold-start retention). The other 11 cold runs are
   tertiary contrasts (FedProx, PF, Q8) and the cold DP chain, whose result is already predicted by the registered noise note
   (per-round noise ≈ 2.5 × the signal at d = 20.7 M, 06D §3). They cost ≈ 280 process-hours and put the DP1 chain on the critical
   path (≈ Oct 3–4), pushing every date by 1–2 days.
3. **Recommendation = Option B, "warm-primary, cold as ablation":** keep cold FA ×2 (running) and the already-live cold FA_1024 → DP8
   seed 2026 as the cold DP anchor; **stop now** FP ×2 (→ FA_Q8), the PF chain, the second DP chain and DP1; give the machine to the
   warm line the moment S7/S8 select (≈ 01:30Z Oct 1): FA_WARM ×2, FA_WARM_FROZEN ×2, then FT_FA_WARM(_FROZEN) ×2 each and
   DP8_FROZEN seed 2026 (needs a small code task FL-16 first, see §4.3). Register amendment 06N (text in §4.2) **before** any MAIN
   value is read: primary headline becomes **K20b = R(FA_WARM / C_FULL)** with **K21 = FA_WARM − PRE** as its required companion;
   **K2 becomes a secondary, cold-start ablation row**. This holds or improves the planned dates (VALIDATION read ≈ Oct 2–3, TEST
   ≈ Oct 4, tables ≈ Oct 5–6) instead of slipping them.
4. The remaining cold method rows (FA_Q8, FP, PF, DP1, DP_OPT_SENS, FA_X2) become **NICE-if-idle, single seed**, registered now so they
   are not post hoc, run only if a slot is free before the ANA freeze; otherwise reported honestly as "not run".

---------------------------------------------------------------------------------------------------------------------
## 0. Facts and rates used (MEASURED unless marked)

| Item | Value | Source |
|---|---|---|
| CALIB cold ladder | S0 0.023473 → S4 0.107562 (FedAdam η_s 0.01 τ 1e-3 + FL_CONST + client SGD-m lr 0.1 + 1 pass, B16) | ladder register; DL 271 |
| Warm preview (S2 point, client AdamW) | FA_WARM_FROZEN f1 **0.1673**, flat 0.164–0.167 all marks; full-model warm collapsed to ≈ 0.03 | DL 267; register |
| S7/S8 confirmatory screens | 8 CALIB runs at the S4 point, 3,414 rounds each, started 18:37Z; 914/3,414 at ~20:20Z → finish ≈ 01:30Z Oct 1 | DL 281 (INFERENCE for the finish) |
| Machine | 2× A40, CPU-bound: load ≈ 15 on 16 guaranteed cores with ≈ 15 FL processes; fewer processes → each faster, ≈ proportional | DL 281 |
| MAIN FA-type run (FA, FP, PF, FA_Q8, FA_WARM, FA_WARM_FROZEN) | 13,654 rounds (P = 1) at ≈ 550 rounds/h under load 15 → ≈ **25 h** | DL 281 |
| MAIN Poisson DP-type run (FA_1024, DP8, DP1, DP8_FROZEN) | T = 384, ≈ 1,024 clients/round, 2 passes, ≈ 4 min/round → ≈ **26 h** | DL 281; handover §10 |
| Device FT run on MAIN (FT_C, FT_PRE) | ≈ 6.5–7 h each on the shared machine (FT_PRE s2026 12:51Z → 19:19Z) | DL 268, 274 |
| Unshared MAIN FL process | ≈ 10.7 h (REPORTED, P = 2 era; treated as the floor) | DL 169 via handover §14.5 |
| Running cold MAIN lanes | laneM26a FA_s2026 → PF_s2026; laneM27a FA_s2027 → PF_s2027; laneM26bn FP_s2026 → FA_Q8_s2026; laneM27bn FP_s2027 → FA_Q8_s2027; laneDPa FA_1024_s2026 (20:08Z) → DP8_s2026 → DP1_s2026; laneDPb (waits S8Tb) FA_1024_s2027 → DP8_s2027 | RESUME §2o; DL 271, 278, 280 |
| Registered hierarchy | primary K2 = R(FA / C_BIG); secondary incl. K20, K20b, K20c, K21, K9c, K12; tertiary K4, K5, K13, K14 … | ANA_CONTRASTS_v3; 06D §4; 06L_C C7 |
| Launcher gating | `launch_fl.py:178-183`: cold arms FA, FP, PF, FA_1024, DP, S_CAL, FA_Q8; warm arms only FA_WARM, FA_WARM_FROZEN. **`:664-667` refuses `--init-from` for every other arm** → any warm-line DP / PF / Q8 arm, including the registered DP8_FROZEN, needs a code change before it can launch | code (MEASURED today) |
| DP noise note (pre-registered) | d = 20,710,048, m = 1,024, z = 0.5642 → per-round noise norm ≈ 2.51·S vs signal ≤ S | 06D §3 |
| Frozen-encoder DP (INFERENCE, same formula) | d ≈ 265,162 → noise ≈ 0.5642·S·√265,162/1,024 ≈ **0.28·S** → ≈ 9× better noise-to-signal than cold DP8 | 06L_C C8 (param count "to be re-measured") |

**Accounting unit.** I count work in *process-hours at load 15* ("ph"): a MAIN FA-type run = 25 ph, a DP-type run = 26 ph, a device
FT run ≈ 7 ph, one CALIB screen ≈ 7 ph (5 ph remaining). The machine delivers ≈ 15 ph per wall-hour when ≥ 15 processes run; with
c < 15 processes each run takes ≈ 25 × c/15 h, not less than ≈ 11 h. GPU-hours are not the binding resource (CPU-bound), so I do not
express costs in GPU-hours; the two A40s are ≈ 98 % busy either way.

---------------------------------------------------------------------------------------------------------------------
## 1. What deployed systems, papers and theses actually do

### 1.1 Deployed systems (REPORTED from the practice audit, which opened the PDFs on 2026-09-30)
| System | Start | Compared cold vs warm? | Runs / seeds |
|---|---|---|---|
| Gboard NWP 2018 (Hard et al., arXiv 1811.03604) | random, 1.5 M clients, ≈ 400 examples/client, 3,000 rounds | no; FL model vs server-trained model, one run each, then live A/B | n = 1 per model |
| Gboard LMs with DP 2023 (Xu et al., 2305.18465; Zhang et al., 2306.14793) | **public C4 pretraining for every model**; "pretraining can reduce ∼1000 rounds"; separate in/out embeddings for warm start | pretrained only; the random start is not a production candidate | n = 1 per model + live metrics |
| Firefox URL bar (Hartmann et al., 1911.11807) | existing hand-tuned weights, "so that users do not interact with a bad model" | no | n = 1 |
| Apple News / keyboard (Paulik et al., 2102.08503) | an existing on-device algorithm; FL tunes a few parameters | no | n = 1, live A/B |
| Federated Reconstruction (Singhal et al., 2102.03448), DCCL, EdgeRec | a good global / cloud model; local parts on device | no | benchmark-style |
Consensus (audit §1.2): **inference on device, start from something good, federate only what needs private data; random init only at
Gboard-2018 scale.** Our setting (19.7 examples per client, 158k-row tables) is the opposite extreme. Production never reports a cold-start
headline; it reports the shipped (warm) system against the previous shipped system or a server-trained model.

### 1.2 FL methodology papers (opened today; MEASURED = quoted text)
- **Nguyen et al., ICLR 2023, "Where to Begin? On the Impact of Pre-Training and Initialization in FL"** (https://arxiv.org/abs/2206.15387):
  pretrained start gives "more accurate models (up to 40 %) than is possible when starting from random initialization" and "reduces the
  effect of both data and system heterogeneity"; they note prior evaluations "nearly always start federated training from a randomly
  initialized model" and recommend that future work evaluate methods **under both initialisations** (MEASURED, abstract/summary). A
  secondary search snippet reports the comparison repeated over 3 seeds (REPORTED, unverified).
- **Chen et al., ICLR 2023, "On the Importance and Applicability of Pre-Training for FL"** (https://arxiv.org/abs/2206.11488):
  "pre-training can not only improve FL, but also close its accuracy gap to the counterpart centralized learning", while it "seems to not
  alleviate local model drifting" (MEASURED, abstract).
- **"Initialization Matters: Unraveling the Impact of Pre-Training on FL", 2025** (https://arxiv.org/abs/2502.08024): "Starting from a
  pre-trained model can substantially reduce the adverse impact of data heterogeneity"; pretrained init "is becoming standard practice"
  (MEASURED, abstract).
- **Federated recommendation papers**: FCF (Ammad-ud-din et al. 2019, https://arxiv.org/abs/1901.09888) validates that "a collaborative
  filter can be federated without a loss of accuracy compared to a standard implementation" — the comparator is the centralised model,
  the start is the paper's own init, one run per setting (MEASURED, abstract only). Recent FedRec work reports pretrained-weight init beside
  default random init (seed 0) and finds the pretrained start better (search snippet, PMC12651314, REPORTED); WarmFed 2025
  (https://arxiv.org/html/2503.03110) frames "pre-trained start instead of random start" as the current direction (REPORTED).
- **Seeds in FL papers**: benchmark papers typically average 3–5 seeds on small models; production reports n = 1 plus live A/B. I could
  not open Reddi et al. 2021's appendix today (OpenReview blocked) → its seed count is UNVERIFIED and not used.

### 1.3 What this means for an MSc thesis with a Microsoft supervisor (INFERENCE)
- The **headline** should be the system a company would actually ship: pretrained on permitted data → federated on private users →
  (optionally) local fine-tune → on-device inference. That is exactly our deployment branch (06K, 06L_C C7, 06M_B B4), and 06L_C C7
  already says "the Edge answer and the L5 row lead with these". Making K20b primary is consistent with what is registered; it changes
  the label on the top of the hierarchy, not the experiment.
- The **cold start is the ablation** the literature asks for ("evaluate under both"): it isolates what federated optimisation alone can
  extract from 131k sparse clients. Two seeds with paired user-bootstrap CIs is enough for that ablation; the dominant uncertainty in
  recsys evaluation is across users, which the bootstrap covers, and per-seed spread is already reported (06L_C C13 SEED_SENSITIVE).
- Reporting **both** lines side by side, with the honest sentence "cold start retains X %; pretrained start retains Y %; federated
  training adds Δ over the pretrained model alone (K21)", is the strongest thesis narrative and matches Nguyen et al.'s recommendation.

---------------------------------------------------------------------------------------------------------------------
## 2. Is 13 cold runs much better than 2? What each extra cold run buys

The 13 cold MAIN runs = FA ×2, FP ×2, PF ×2, FA_Q8 ×2, FA_1024 ×2, DP8 ×2, DP1 ×1 (≈ 330 ph in total, ≈ 280 ph beyond FA ×2).

| Cold run(s) | Contrast (tier) | Scientific value on the COLD line (INFERENCE) | Cost | Same ph on the WARM line buys |
|---|---|---|---|---|
| FA s2026/s2027 | K2 (headline today), K3, K11, K11b, K17 | **Essential**: the cold-start ablation; already ≈ 2 h in; also the base for FT_FA (K6, K10) | 50 ph (running) | — keep |
| FP ×2 (FedProx μ 0.01, untuned) | K4 (tertiary) | Low: with FedAdam + client SGD + 1 local pass the prox term at μ 0.01 is a small perturbation of an already weak model; no production system uses FedProx | 50 ph (≈ 1 h sunk) | FA_WARM ×2 (the headline arm) |
| PF ×2 (personal vector) | K5 (tertiary) | Low on cold: personalisation on top of a weak global model is confounded by the base; the realistic personalisation rows are FT_FA_WARM(_FROZEN) (06M_B B4) | 50 ph | FA_WARM_FROZEN ×2 (phone-payload arm) |
| FA_Q8 ×2 | K14 equivalence (tertiary) | Medium: compression cost is a real deployment knob, and it is ≈ independent of the start; one seed answers the TOST test | 50 ph | FT_FA_WARM ×2 + FT_FA_WARM_FROZEN ×2 + DP8_FROZEN |
| FA_1024 ×2 → DP8 ×2 | K12 (secondary, CONTROLLED_BUDGET) | Medium-low: the registered note already predicts noise ≈ 2.5·S vs signal ≤ S → a near-total loss "as a property of the (m, T, d) design". One seed suffices to show it; the second seed adds a CI on a number expected ≈ 0 | 104 ph, 52 h chain | DP8_FROZEN (the realistic DP row, noise ≈ 0.28·S) + a second warm seed |
| DP1 | K13 (tertiary, single seed) | Low: noise ≈ 5.8·S; certainly destroyed; **it is the critical path** (3-run chain ≈ 78 h → Oct 3–4) | 26 ph, +26 h wall | 1 more FA-type warm run |

**Answer:** no. Runs 3–13 add tertiary rows and a predicted-negative DP result; they do not sharpen the cold headline, and they consume
the machine during the exact window in which the warm line — the realistic answer — should be running. The scientifically valuable cold
extras are FA_Q8 (one seed) and one cold DP anchor (FA_1024 → DP8, seed 2026, already live).

---------------------------------------------------------------------------------------------------------------------
## 3. Allocation options (all durations INFERENCE from §0 rates; t0 = Sep 30 20:30Z; S7/S8 select ≈ 01:30Z Oct 1)

Common to all options: S7/S8 screens finish (≈ 40 ph remaining); warm MAIN launches need `FL_WARM_SELECTION_S7|S8.json` (mechanical
select-warm; an auto-lane that waits for the files and launches saves ≈ 5 h versus waiting for the morning); device FT arms follow their
base runs (FT_C selection cell, smoke first); ANA freeze / tree r6 / C_FULL s2027 reeval proceed in parallel and are not on the machine's
critical path.

| Option | MAIN runs (beyond what all options share) | Work | Wall-clock / critical path | Dates (VALIDATION read → TEST → tables) | Scientific strength | Risk |
|---|---|---|---|---|---|---|
| **A — as-is** | cold 13 + DP_OPT_SENS 2 + FA_X2 1 + FT_FA 2; warm FA_WARM 2, FROZEN 2, FT_FA_WARM(_FROZEN) 4, DP8_FROZEN 1 | ≈ 650 ph | machine saturated ≈ 43 h; **critical path laneDPa FA_1024 → DP8 → DP1 = 78 h at load 15 → ≈ Oct 3 12:00Z–Oct 4 02:00Z**; PF/Q8 chains ≈ Oct 2–3 | **Oct 4 → Oct 5–6 → Oct 7–8** (slip 1–2 days) | complete matrix on both lines; K2 headline unchanged | headline stays the unrealistic start (owner's objection); warm line runs last and slowest; 3 cold DP rows predicted ≈ 0; every date slips |
| **B — warm-primary, cold as ablation (RECOMMENDED)** | cold: FA 2 (running), FA_1024 → DP8 s2026 (live), FT_FA 2; warm: FA_WARM 2, FROZEN 2, FT_FA_WARM 2, FT_FA_WARM_FROZEN 2, S_CAL_FROZEN + DP8_FROZEN s2026 (after FL-16). **Stopped:** FP 2, PF 2, FA_Q8 2, FA_1024/DP8 s2027, DP1, DP_OPT_SENS, FA_X2 → NICE-if-idle | ≈ 310 ph | ≈ 9–11 concurrent → FA-type ≈ 15–18 h. Cold FA done ≈ Oct 1 12:00Z; warm 4 launched ≈ 02:00Z Oct 1 → done ≈ Oct 1 18:00–22:00Z; FT_FA_WARM ×4 → Oct 2 ≈ 04:00Z; cold DP8 s2026 ≈ Oct 2 08:00Z; DP8_FROZEN (code FL-16 morning Oct 1 → S_CAL_FROZEN → run) ≈ Oct 2 12:00–18:00Z | **Oct 2 night / Oct 3 morning → Oct 4 → Oct 5–6** (holds or beats plan) | realistic headline (K20b + K21), phone-payload row (S8), realistic DP row, full production stack (FT_FA_WARM), cold ablation kept | K4/K5/K13/K14 "not run" unless idle; K12 single seed; DP8_FROZEN depends on a small code task + review (fallback: cold DP8 s2026 is the DP row) |
| **C — both full** | A + warm-line method set: PF_WARM 2, FA_WARM_Q8 2, FA_1024_WARM → DP8_WARM s2026 (all need FL-16 generalised) | ≈ 800 ph | saturated ≥ 53 h + chains; DP8_WARM chain from ≈ Oct 1 midday → Oct 3 evening; DP1 chain Oct 3–4 | **Oct 5 → Oct 6–7 → Oct 8–9** (slip 2–3 days) | most complete; both lines symmetric | dilutes the thesis into ~30 rows, half tertiary; largest schedule risk; code change under time pressure |
| **D — both reduced, single-seed methods** | B + cold FA_Q8 s2026 (no code) + PF_WARM s2026 + FA_1024_WARM → DP8_WARM s2026 (need FL-16) | ≈ 410 ph | ≈ 13 concurrent; DP8_WARM chain 2 × ≈ 22 h from ≈ Oct 1 12:00Z → Oct 3 ≈ 08:00Z | **Oct 3 afternoon → Oct 4–5 → Oct 6–7** (= plan) | B + compression row + "why frozen matters" DP row (full-model warm DP at d = 20.7 M) | DP8_WARM is the critical path; more code surface; K5 single seed |

Why B over D as the *binding* plan: D's additions are all "nice to know" and each one either needs code (PF_WARM, DP8_WARM) or a slot;
registering them as NICE-if-idle inside B gets the same rows when the machine has room, without letting them move the freeze.

---------------------------------------------------------------------------------------------------------------------
## 4. Recommendation

### 4.1 What is most accurate and best for the owner
**Option B.** Reasons, in order: (1) it makes the thesis answer the question a supervisor at Microsoft will ask — "what does the shipped
system retain?" — with the design the literature and production use (pretrained start), while keeping the cold start as the ablation the
literature asks for; (2) it moves ≈ 280 ph from predicted-low-value cold rows to the warm rows that decide the thesis (K20b, K21, S8−S7,
DP8_FROZEN, FT_FA_WARM); (3) it removes the 78-hour cold DP chain from the critical path, so the VALIDATION read stays ≈ Oct 2–3 instead of
≈ Oct 4; (4) nothing is thrown away scientifically: cold FA ×2 (running), cold DP8 s2026 (live chain), FT_FA and all central/device arms
stay; the dropped rows are reported as "not run" under the registered cut rule (06D §8), or run as NICE-if-idle.

Register the change **now**, before any MAIN value exists in anyone's hands: no MAIN FL value has been read (DL 281), so the change is a
pre-registration edit, not a post-hoc one. Disclose it verbatim in the thesis (date, reason, what was running).

### 4.2 Amendment text to register (proposed `contracts/PREG1_AMENDMENT_06N.md`; Ops registers with sha + DL row)

> # PREG1 amendment 06N — headline moved to the pretrained-start line; cold line kept as ablation; MAIN run allocation (registration only)
> - **Status:** REGISTERED 2026-10-01 <hh:mm>Z (server clock) by the Operations Director on the owner's decision of 2026-09-30 ("a device
>   never starts from an untrained model; the pretrained start is the correct line") and the chief-scientist memo
>   company/reviews/FABLE_RUN_ALLOCATION_20260930.md. Registered BEFORE any MAIN / VALIDATION / TEST / PRE value and before any S7/S8
>   selection value was read by a decision-maker (MEASURED: DL 281). Amends 06D §4/§8, 06K §4, 06L_C C7, ANA_CONTRASTS_v3 hierarchy;
>   nothing else changes. TEST stays sealed; one look.
> ## 1. Headline and hierarchy (replaces 06D §4 hierarchy and 06K §4's "the main FL headline (K2) stays cold-start")
> - **Primary: K20b = R(FA_WARM / C_FULL)** — retention of the realistic Edge federated system (θ_pre from 1,217,610 opted-in non-MAIN
>   users → FL on the 131,072 MAIN users) against its matched cloud reference C_FULL (s2026), 2 FA_WARM seeds, PRACTICAL_BEST.
> - **Required companion (secondary, reported in the same headline sentence): K21 = FA_WARM − PRE** (paired Δ). If K21's CI includes 0 or
>   lies below 0, the headline sentence states that the pretrained model alone retains K20c and that federated training on the private
>   users adds no measurable quality; that is a result, not a failure.
> - **Secondary:** K2 (now labelled "cold-start ablation: matched-exposure retention of the registered FL recipe from a random start"),
>   K20, K20c, K21, K9c, K1, K3, K8, K9b, K11b, K12 (seed 2026 only), K17, **S8−S7** (promoted from descriptive: FA_WARM_FROZEN −
>   FA_WARM, 2 seeds), **K22 = FT_FA_WARM − FA_WARM** and **K22f = FT_FA_WARM_FROZEN − FA_WARM_FROZEN** (paired Δ, 2 seeds, the
>   full production stack; from 06M_B B4), **K12f = R(DP8_FROZEN / FA_WARM_FROZEN) at CONTROLLED_BUDGET** (seed 2026; the round-384
>   endpoint of DP8_FROZEN against the endpoint_6.0 of its registered z = 0 twin S8, 06L_C C16; the sampling mismatch — Poisson 1,024 vs
>   sweep 64 — is disclosed; FA_1024_FROZEN s2026 is NICE-if-idle as the exact twin).
> - **Tertiary:** K6, K7, K9, K10, K11, K15, K16, K18, K19; K4, K5, K13, K14 only if their arm runs (§2), otherwise "not run".
> - Multiplicity statement: one primary contrast; the count of confirmatory contrasts actually evaluated is printed by the frozen analysis;
>   unadjusted 95 % paired user-bootstrap CIs with the Holm band beside them (06L_C C13) unchanged.
> ## 2. MAIN arm set (replaces the SHOULD-high list of 06D §8 / 06I §8 for the FL arms; central, device, baseline arms unchanged)
> - **MUST:** FA s2026/s2027 (cold ablation); FA_WARM s2026/s2027 and FA_WARM_FROZEN s2026/s2027 at the S7/S8 selections (06M_B B2,
>   DL 261 tie order); FT_FA ×2; FT_FA_WARM ×2 and FT_FA_WARM_FROZEN ×2 (SELECTION_FT_C cell, smoke first); FA_1024 s2026 → DP8 s2026
>   (cold DP anchor, K12 single seed); S_CAL_FROZEN (20 rounds on the S8 trajectory → S_RECORD_FROZEN, write-once) → DP8_FROZEN s2026.
> - **Stopped / withdrawn now, before any of their values exist:** FP s2026/s2027, PF s2026/s2027, FA_Q8 s2026/s2027, FA_1024 s2027,
>   DP8 s2027, DP1 s2026, DP_OPT_SENS, FA_X2. They become **NICE-if-idle, single seed 2026, in this priority order:** FA_Q8 (no code),
>   FA_X2 (no code), PF_WARM (needs FL-16), FA_1024_FROZEN (needs FL-16), DP8_WARM via FA_1024_WARM (needs FL-16), FP, DP1,
>   DP_OPT_SENS. Rule: a NICE-if-idle run is launched only when the FL process count is < 12 and its estimated finish precedes the
>   planned ANA freeze; a run that has not finished by the VALIDATION preflight is excluded from the block and reported "not finished".
>   Whatever is not run is reported "not run (allocation decision 06N)"; nothing is replaced by a value from another population.
> - The old-recipe runs in runs_ds3/FL/ and LADDER_S1a stay historical ladder evidence; the S7/S8 CALIB screens are unchanged.
> ## 3. Code (task FL-16, Opus, one review, then tree r6 together with the ANA-05 fixes)
> `launch_fl.py` accepts `--init-from` (+ `--freeze-item-tables`) for arms DP and S_CAL (and, for NICE-if-idle only, FA_1024, PF, FA_Q8):
> the warm init identity check (check_init_identity), the frozen-table bytes record and the DP noise dimension = uploaded parameter
> count (re-measured, 06L_C C8) apply; the S_CAL_FROZEN clip norm S_f = median unclipped encoder-delta norm, rounds 1–20, seed 2026 →
> contracts/dp/S_RECORD_FROZEN.json (write-once). Pinned cold digests unchanged (tiny-run r5 == r6 for all 7 arms). Until FL-16 is
> reviewed and deployed, no warm DP arm launches; if FL-16 is not deployed by 2026-10-02 12:00Z, DP8_FROZEN is reported "not run" and the
> cold DP8 s2026 row is the DP row.
> ## 4. Disclosures (thesis list)
> (a) The primary headline was moved from the cold start (K2) to the pretrained start (K20b) on 2026-10-01 before any MAIN value was read,
> on the owner's realism priority and the literature (Nguyen et al. 2023; Chen et al. 2023); K2 is reported as the ablation. (b) The cold
> arms FP, PF, FA_Q8, DP1, second-seed DP and DP_OPT_SENS were stopped before producing a value, for compute allocation, not for their
> results; their rows are "not run" or NICE-if-idle single-seed. (c) C_FULL (denominator of K20b) has one registered seed; θ_pre is one
> seed shared by both FA_WARM seeds. (d) The DP8_FROZEN twin mismatch (§1). (e) The FL tuning of both lines was selected on CALIB
> (⊂ MAIN), unchanged from 06L_C C2.

### 4.3 Lanes to stop NOW (by name; kill the process, mark LANE_STOP / LANE_WITHDRAWN, no value read)
| Lane | Action | Sunk cost |
|---|---|---|
| **laneM26bn** (FP_s2026 → FA_Q8_s2026) | stop FP_s2026 now; withdraw the chain | ≈ 1 h of FP |
| **laneM27bn** (FP_s2027 → FA_Q8_s2027) | stop FP_s2027 now; withdraw the chain | ≈ 1 h of FP |
| **laneDPb** (waits S8Tb → FA_1024_s2027 → DP8_s2027) | withdraw before it starts | 0 |
| **laneM26a / laneM27a** (FA → PF) | keep FA running; remove PF from the chain (replace the lane with FA-only or LANE_STOP after FA) | 0 |
| **laneDPa** (FA_1024_s2026 → DP8_s2026 → DP1_s2026) | keep FA_1024 → DP8; remove DP1 from the chain | 0 |
| laneS7Ta–d, laneS8Ta–d | **keep** (they select the warm line at ≈ 01:30Z) | — |
Then: an auto-lane that waits for `FL_WARM_SELECTION_S7.json` / `_S8.json`, dry-runs, and launches FA_WARM s2026/s2027 and
FA_WARM_FROZEN s2026/s2027 (`--fl-tuning … --init-from <PRE init record 5ed989f6> --warm-selection …`, forms in FL-15.md) so the warm line
starts at ≈ 02:00Z and not at the morning shift. Ops copies the selection files write-once and writes the DL row as usual.

### 4.4 Expected timeline under B (INFERENCE)
Oct 1 01:30Z S7/S8 select → 02:00Z warm MAIN ×4 launched → ≈ 12:00Z cold FA ×2 done → FT_FA smoke + ×2 → FL-16 reviewed + r6 deployed by
midday → S_CAL_FROZEN → DP8_FROZEN launched ≈ 14:00Z → Oct 1 18:00–22:00Z warm MAIN ×4 done → FT_FA_WARM(_FROZEN) ×4 → Oct 2 ≈ 04:00Z →
cold DP8 s2026 ≈ 08:00Z → DP8_FROZEN ≈ 12:00–18:00Z → ANA freeze (already prepared) → **VALIDATION read Oct 2 night / Oct 3 morning →
ALL_METHODS_LOCK → TEST look ≈ Oct 4 → tables ≈ Oct 5–6.** Slack ≈ 1 day against the deck's dates.

### 4.5 Risks and how B handles them
- **S7 full-model warm may still degrade θ_pre at the S4 point** (the preview collapsed under client AdamW; client SGD is expected to be
  gentler — INFERENCE). Then K21 ≤ 0 and the headline sentence says so; S8 (frozen tables, 0.167 flat in the preview) is the protected row
  and S8−S7 becomes the important secondary. No plan change is needed either way.
- **FL-16 slips** → DP8_FROZEN "not run"; the cold DP8 s2026 row remains (kept for exactly this reason). Hard stop 2026-10-02 12:00Z.
- **Machine contention** → fewer processes, not more: B's ≈ 9–11 concurrent runs is the right load for a 16-core guarantee.
- **Reviewer asks "why not FedProx / PF / ε = 1?"** → answer with the registered allocation decision, the pre-registered DP noise note
  and the production evidence; the cold FA_Q8 s2026 is the first NICE-if-idle run, so K14 will most likely exist.

---------------------------------------------------------------------------------------------------------------------
## 5. Sources opened today
- https://arxiv.org/abs/2206.15387 (Nguyen et al., ICLR 2023) · https://arxiv.org/abs/2206.11488 (Chen et al., ICLR 2023) ·
  https://arxiv.org/abs/2502.08024 (Initialization Matters, 2025) · https://arxiv.org/abs/1901.09888 (Ammad-ud-din et al., FCF) ·
  https://arxiv.org/abs/2405.04840 (federated adaptation of foundation-model recommenders; abstract only) ·
  https://arxiv.org/html/2503.03110 (WarmFed; search snippet) · https://pmc.ncbi.nlm.nih.gov/articles/PMC12651314/ (search snippet).
- Not opened / UNVERIFIED: Reddi et al. 2021 seed count (OpenReview blocked; arXiv PDF unreadable through the fetcher).
- REPORTED via company/reviews/REAL_WORLD_FL_PRACTICE_AUDIT_20260930.md (opened by that audit on 2026-09-30): Gboard 1811.03604,
  2305.18465, 2306.14793; Firefox 1911.11807; Apple 2102.08503; Federated Reconstruction 2102.03448.

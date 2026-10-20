# MAIN-scale design — Fable 5.1 ruling (verbatim; fresh agent, one message; received 2026-09-27 ~22:10Z; packet MAIN_TUNING_PACKET.md)

**VERDICT:** A, modified — no LR extension, exactly two candidates per family, one sealed G2 read.

**DESIGN**
1. P2 (02R1/02R2) stands unchanged as the screen; CALIB is the nested first quarter of MAIN; MAIN_VALIDATION never read in P2. SASRec P2 finishes as registered.
2. Candidates per family = best stage-2 config of each of the top-2 size levels by CALIB 2-seed mean (GRU: D128_S1_s4, D256_S1_s2; SASRec: 2 of 3, computed mechanically when its stage 2 ends, nobody reading earlier). No grid extension, no new configs.
3. Contest: 8 MAIN runs = 4 recipes × seeds 2026/2027, 6.0 EFE, marks 0.5, continuation per the staggered ruling. INFERENCE ≈35–45 GPU-h incl. continuations, ≈24 h wall.
4. G2, one script, one unsealing: (a) within family — higher 2-seed 6.0-endpoint mean on MAIN_VALIDATION; tie ≤0.0005 → smaller model, then lower LR; exclusions as 02R1 §3, both excluded → Fable; (b) between families — registered T1 on the selected recipes (ARCHITECTURE_EQUIVALENT → HOLD). CALIB-disjoint diagnostic printed for both steps. Non-selected candidates labelled CANDIDATE_NOT_SELECTED; never P4/P5/SYSTEM.
5. Seal unchanged; unseal only when all 8 complete, G1B_APPROVED holds part-1 (byte-identical), part-1R, part-2, all 8 RunConfig hashes match, G2 script hash frozen.
6. Code: prefer no trainer change — one 02b entry per (family, size), existing frozen driver run once per entry. If a code_p3 change is unavoidable: registry reader only, new frozen pack, reviewed, and its D128 dry-run hashes must equal d9469319…/4ed9a13a….

**REGISTER BEFORE ANY MAIN RUN (amendment 05)**
- Supersedes 02 §2 and the single-recipe 02b format; P2 rules untouched.
- Candidate rule (§2), candidate count fixed at 2 per family, no extension.
- Two-step G2 rule, tie/exclusion rules, label for non-selected rows.
- Unsealing preconditions (§5), code pack hash(es), dry-run equality test.
- C_BEST/SYSTEM/TEST apply to the selected recipe only; ledger publishes all candidates.
- Changes after the first MAIN launch refused (new amendment + Fable).

**ANSWERS**
- Q2: Part-1 findings stand (GRU P2 table, mechanical selection, disclosures). Its single-recipe 02B_GRU entry is superseded by part-1R (two recipes); D128_S1_s4 must be byte-identical across both; append-only.
- Q3: SASRec P2 continues, unchanged hash; it is the symmetric screen.
- Q4: Yes, once amendment 05 is registered and G1B condition 1 (cited sealed=True line, sealed/ path) is filed; D128 seeds sequential on GPU0 under pack f89b9ff4… with the pinned hashes; disk guard armed.

**DISCLOSURES**
- Within-size hyperparameters chosen on the nested 25% subset; size and final recipe chosen at full MAIN scale; every contest model trained on MAIN only.
- GRU LR edge (0.001687) not extended; 4× more updates on MAIN may shift the optimum.
- Family MAIN_VALIDATION values are max-of-2 (selection-biased, symmetric); TEST is the only unbiased number.
- 24.8% user overlap CALIB_VALIDATION/MAIN_VALIDATION; disjoint-subset ranking flips reported.
- SASRec's dropped size chosen on CALIB only.

**BASIS**
Selecting recipes on MAIN_VALIDATION is validation-only under EVALUATION_CONTRACT_V4 §10; TEST stays sealed. A folds C's confirmation into one sealed mechanical read, with 2 seeds per candidate, at half C's cost; B exceeds two days and adds nothing the owner asked for beyond what MAIN-scale size selection gives. Equal candidate counts and one procedure keep the contest fair; the LR extension is dropped because a symmetric version is unaffordable and the G1B evidence argues against a truncated gain.

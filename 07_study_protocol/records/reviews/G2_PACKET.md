# G2 packet — Ops, 2026-09-28T11:30Z. The single unsealing (amendment 05 §4–§5). Question: confirm the G2 outcome and the ARCHITECTURE_LOCK.

## Procedure (PROJECT_MEASURED)
All 8 MAIN runs final (GRU ×4 DONE_CONTINUATION at 7.0; SASRec ×4 DONE_MAIN at 6.0), each launch config == its dry run.
Frozen script tools/g2_run.py 3893a49a… (company/gates/G2_SCRIPT_FREEZE.json); preflight PASS 11:24Z (counts 51,211 /
12,715 / 38,496 via decision_id; gate PASS without sealed reads); full run 11:25Z, rc 0; G2_STARTED.json written before
the gate; G2_RESULT.json sha 3397a643…; ARCHITECTURE_LOCK.DRAFT.json sha 3a0db5b3… (runs_p3/g2/, local copies too).

## Step (a) within family — 2-seed mean, 6.0 endpoint, MAIN_VALIDATION macro MRR@20
- GRU: **02B_GRU (D128) 0.156937** vs 02B_GRU_C2 (D256) 0.153785 → SELECTED 02B_GRU (diff 0.00315 > 0.0005).
- SASRec: **02B_SASREC (D256_B3) 0.152724** vs 02B_SASREC_C2 (D256_B2) 0.150484 → SELECTED 02B_SASREC (diff 0.00224).
- No exclusions; CALIB-disjoint diagnostic selects the same recipes (no flip).

## Step (b) between families — registered T1 (PRACTICAL_BEST rows, seeds 2026/2027)
- GRU per seed 0.157141 / 0.156733 (evidence checkpoints best@6.0); SASRec 0.161196 / 0.161221 (best@3.0).
- delta_bar (GRU − SASREC) = **−0.004271**, paired user bootstrap CI95 [−0.005180, −0.003261] (1,000 resamples, 40,764
  users, 519,714 decisions); per-seed deltas −0.004056 / −0.004487; seed gaps GRU 0.000407, SASRec 0.000025.
- Triggers T1a–T1d all false → **DECISIVE_STABLE, winner SASREC** (losing family GRU forbidden downstream).
- Secondary NDCG@10 family means: GRU 0.19105, SASRec 0.19668.
- CALIB-disjoint diagnostic (38,496 users): delta_bar −0.004103, CI [−0.005255, −0.002908], DECISIVE_STABLE SASREC, no flip.

## Ops observations (for disclosure, not decisions)
1. The two steps use different registered row types: step (a) the 6.0 endpoint (amendment 05 §4a), step (b) PRACTICAL_BEST
   (best validated mark ≤ 6.0; T1). SASRec's PRACTICAL_BEST is at 3.0 EFE (early peak, consistent with its low-LR-edge
   P2 curves peaking at 4.0 then falling); on the 6.0 endpoint GRU D128 (0.1569) is above SASRec D256_B3 (0.1527).
   So the family verdict depends on using the best validated checkpoint — which is the registered T1 rule.
2. GRU runs fired the automatic continuation (STILL_IMPROVING at 6.0; stopped at 7.0); continuation rows are
   PRACTICAL_BEST_EXTENDED and were not used (registered).
3. Implication for downstream: C_BIG = the SASRec C1 runs at their PRACTICAL_BEST checkpoint (3.0); SMALL =
   SASREC_D64_B2 (amendment 03; code ready, TRAIN-04; registry + θ0 + mini-ladder next). The GRU C1 runs (D128) are the
   losing family's contest singles (P5 only).
4. Nothing reached TEST. No value was read before this unsealing (except the disclosed P2 incident, DL 105).

## Question
Confirm: (i) G2 outcome DECISIVE_STABLE SASREC with 02B_SASREC as the selected recipe; (ii) the ARCHITECTURE_LOCK may be
written from the draft; (iii) any disclosure/condition for the thesis or for the downstream start (SMALL = SASREC_D64_B2).

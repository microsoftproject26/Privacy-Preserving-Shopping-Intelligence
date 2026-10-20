# MAIN-scale tuning packet — Ops, 2026-09-27T22:05Z. GRU P3 launch is ON HOLD (nothing launched, nothing read).

## Owner directive (verbatim Arabic, 2026-09-27 ~22:03Z)
"شوفو الانسب بس انا عاوز الصح وعاوزكم متعكوش — وخلو التدريب كله كل الموديلات علي ال MAIN"
= "Choose what is most appropriate, but I want it done right, don't mess it up — and make ALL training, all models, on
MAIN." Trigger: the owner objected that P2 tuned on CALIBRATION, not MAIN.

## Facts (PROJECT_MEASURED from REGISTRY_V2 / contracts)
- CALIBRATION = 32,768 users, 646,067 TRAIN decisions; eval 12,715 users / 126,469 rankable rows. NESTED: the first
  32,768 of the 131,072 MAIN users. MAIN = 131,072 users, 2,587,776 TRAIN decisions; MAIN_VALIDATION 51,211 users /
  515,636 rankable rows (38,496 are CALIB-disjoint; already a registered diagnostic). v1 trained on 40.2M decisions.
- RETRO_TEST stays sealed until ALL_METHODS_LOCK (EVALUATION_CONTRACT_V4 §10: 'Architecture selection is
  validation-only'; one TEST look). So choosing a recipe on MAIN_VALIDATION does not touch TEST.
- GRU P2 (CALIB) done: D128_S1_s4 (lr 0.001687, top of the S1 sub-grid) 2-seed 0.124844; D256_S1_s2 (lr 0.000783)
  0.115286. The Ops concern (the owner's point): a 4× smaller tuning set can favour the smaller size and shift the
  LR optimum; the tuned GRU LR also sits at a grid edge. SASRec P2 (CALIB) is still running (stage 1 ~halfway; nobody
  has read its values), INFERENCE done ≈ 03:30–04:30Z.
- Code/gates in place: code_p3 frozen (G1b C3); 02B_GRU registered + G1B part 1 written; OPS-11 dry run validated
  (reproduced a P2 config hash exactly); GRU P3 MAIN hashes s2026 d9469319…, s2027 4ed9a13a…, sealed=True.
- Rates (MEASURED, P2): GRU_D128 ≈ 19.5 upd/s, SASREC_D256_B2 ≈ 8.6 upd/s. INFERENCE per MAIN run to 6.0 EFE (+
  continuation ≤ 8.0): GRU ≈ 2–2.5 h, SASRec ≈ 4.5–5.5 h; D256/B3 variants slower.

## Options (Ops recommends A)
A. **Screen on CALIB, decide on MAIN.** Keep the CALIB P2 as a cheap screen only. P3 on MAIN trains, per family, the
   best config of EACH size from P2 stage 2 (GRU: D128_S1_s4, D256_S1_s2; SASRec: its two) × 2 seeds = 8 MAIN runs;
   optional +1 LR step above the GRU edge. G2 selects mechanically on MAIN_VALIDATION (2-seed mean; size and family),
   CALIB-disjoint users reported as the diagnostic; TEST untouched until ALL_METHODS_LOCK. INFERENCE ≈ 30–35 GPU-h ≈
   15–18 h wall on 2 GPUs; GRU can start as soon as registered. Needs: amendment 05, a multi-recipe registry per family
   (small code change in code_p3 → new frozen tree, Codex + review), seal rules unchanged.
B. Full P2 grid re-run on MAIN (12 + 2 per family): INFERENCE ≈ 18 h GRU + ≈ 48 h SASRec; discards the CALIB work.
C. A MAIN confirmation stage (top-2 per size, 1 seed, on MAIN_VALIDATION) before the unchanged P3: ≈ 20 GPU-h extra,
   then P3 ≈ 14 GPU-h; two looks at MAIN_VALIDATION.

## Questions
1. Binding design (A / B / C / other) satisfying the owner's directive with no contamination of the contest or TEST.
2. What must be registered (amendment text points) before any MAIN run; does the G1b part-1 approval stand or is it
   replaced?
3. Does SASRec P2 on CALIB continue (as the screen) or stop?
4. May the GRU D128_S1_s4 MAIN runs (already registered and dry-run hashed) start now, before the rest is ready?

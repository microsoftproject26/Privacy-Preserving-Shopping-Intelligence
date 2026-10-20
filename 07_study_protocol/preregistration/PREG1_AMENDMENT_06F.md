# PREG1 amendment 06F — extension of the federated local-LR screen (registration only)

- **Status:** REGISTERED 2026-09-29T09:25Z by Claude Fable 5.1 (chief scientist, full authority per the owner 09:00Z), immediately
  after the registered CALIB screen read (DL 181) and BEFORE any MAIN FL value exists or is read. Supersedes the grid of
  amendment 06B §1; its rule, protocol and application are unchanged.
- **Registered read that triggers this amendment (MEASURED, the 06B §1 screen; the only permitted pre-preflight read):**
  CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST — m = 0.5: 0.020263; m = 1: 0.023473; m = 2: 0.025562 → the best is the
  TOP EDGE of the grid and the curve is monotone rising (+16 % then +9 % per doubling). For scale (INFERENCE from the CALIB ladder
  era, DL 143 / 92): a central SASREC_D64_B2 model on the same CALIB users reaches ≈ 0.11–0.12, so FA at 6.0 EFE currently retains
  ≈ 20–25 % of the matched central value. The thesis review (D-2) registered exactly this risk: an under-tuned local LR inflates the
  "federation cost" and the thesis would report an optimisation artefact as a finding.

## 1. Extension (adds to 06B §1)
- **What:** three further screen runs with the identical protocol (FA only, CALIBRATION role, seed 2026, 1,707 rounds = the full
  6.0-EFE schedule, 64/round sweep, 10 % drop-out, B16 × 2, 12 marks, PRACTICAL_BEST) at **m ∈ {4, 8, 16}** × the SMALL peak
  0.000711. If m = 16 is the best of the screened set, ONE more run at **m = 32** (the cap; no further extension whatever its value).
- **Rule (over the whole screened set {0.5, 1, 2, 4, 8, 16[, 32]}):** the highest CALIB_VALIDATION client-macro MRR@20 at
  PRACTICAL_BEST; every m within |Δ| ≤ 0.0005 of the best is tied and the tied m closest to 1 wins (|log2 m| ascending, then the
  smaller m). A run with a non-finite loss or metric is excluded (recorded). Written O_EXCL to `contracts/fl/LR_SELECTION_EXT.json`
  (the 06B file `LR_SELECTION.json` stays as the record of the first screen; the extended file supersedes it for every launch) + DL row.
- **Application (unchanged from 06B §1):** the selected m multiplies the local shared-model LR of FA, FP, PF, FA_1024, DP (all ε),
  S_CAL, FA_Q8; PF personal LR, μ, the FT grid and LO unchanged. **S_CAL and every MAIN FL run use the extended selection.**
- **Why this is legitimate under the project's pre-registration discipline:** (a) the only values read are the registered CALIB
  screen values; no MAIN, panel or TEST value of any FL arm exists; (b) the extension is one-directional in its effect on the
  research question: it can only make the federated arms stronger, i.e. reduce the measured federation cost, never inflate it;
  (c) it is a hyper-parameter grid extension on the calibration cohort, symmetric across every FL arm, not a change of any
  retention rule, arm definition, evaluation manifest or contrast; (d) the contest's "no LR extension" precedent (DL 92, 109)
  protected the fairness BETWEEN two families; here there is one family and one optimiser.
- **Disclosure:** the thesis reports the full 6–7-point CALIB LR curve (Figure: FA quality vs local-LR multiplier), states that the
  first registered grid ended at its edge and was extended once by registration before any MAIN FL run, and that server LR (1.0),
  μ, the local optimiser (fresh AdamW per visit) and the number of local passes remain untuned. If even the extended optimum
  retains far less than the central model, that is reported as a finding about FedAvg with dense full-catalogue softmax updates at
  this cohort scale, with the LR curve as evidence that it is not an LR artefact.

## 2. The MAIN FL runs started at m = 2 (09:12Z, DL 181)
Superseded before any value is read: FA s2026/s2027, FP s2026/s2027, FA_Q8 s2026/s2027 and S_CAL are STOPPED; their partial
directories are kept, never read, labelled `SUPERSEDED_LR_SCREEN_EXTENSION` (a DL row lists them). Every FL arm restarts from the
final selection. Nothing else changes (C_FULL, KNN_EMB, the device lanes, DP design).

## 3. Code (task FL-09, Opus): `launch_fl.py --lr-mult` accepts {0.5, 1, 2, 4, 8, 16, 32} (LR_MULT_TAG extended; the arm-spec
digest still drops the default m = 1 so every existing digest is unchanged); `lr_screen.py select` accepts ≥ 3 run directories
covering an arbitrary registered subset of the multipliers, applies the §1 rule, writes `LR_SELECTION_EXT.json` with every screened
value, and `launch_fl --lr-selection` accepts either file (the extended one is required for a MAIN run once it exists: the
launcher refuses `LR_SELECTION.json` when `LR_SELECTION_EXT.json` is present). Tests incl. NCs (a multiplier outside the set;
a selection file missing a screened run; the tie order). The screen runs may be launched from an interim tree (code_ds3r2 + FL-09)
because they are registered CALIB reads, not production MAIN runs; production MAIN FL runs only from the reviewed code_ds3r3.

## 4. Cost (INFERENCE, MEASURED rate ≈ 10.2 visits/s per process): 3 screen runs in parallel ≈ 2.7–3 h wall on GPU 0 (+ 2.7 h if
m = 32 is needed); the stopped MAIN runs lose ≈ 15 min each. Net effect on the schedule ≈ +3–6 h; the alternative was 13 FL runs
(≈ 140 process-hours) at a demonstrably sub-optimal LR.

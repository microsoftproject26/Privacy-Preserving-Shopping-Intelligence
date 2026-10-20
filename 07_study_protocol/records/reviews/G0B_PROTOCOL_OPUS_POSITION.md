# G0(b) protocol approval — Opus position. Independent memo, withheld from Fable until Fable answers.

- **F-1 (re-zero and valid-only refit of `current_minus_recent_mean_log_price`): ADOPT.**
  - PROJECT_MEASURED (S328): 117,315 of 52,760,314 TRAIN rows (0.22%) carry a value computed from an unknown current
    price. The valid-only refit gives mean 0.002066 and std 0.994534 on 48,434,230 valid rows.
  - Package 03 §6 ("write unknown=0 after normalization") covers it directly, so this applies the package rule rather
    than extending scope. It is disclosed as part of the Clean V2 package; its effect is not attributed separately.
- **P3 (stopping): ADOPT the reading in `G0_PLAN_OPUS_RESPONSE.md`.**
  - Every run completes 6.0 EFE.
  - PRACTICAL_BEST is the strictly-best 0.5-EFE checkpoint in [0, 6].
  - Patience and tolerance govern only an optional continuation from 6 to 8 EFE.
- **P11 (cold-h panel): candidate B, split into two panels.** QA-01's ≥ 50-event panel is needed because h can reach
  50 only when a user has 50 prior events.
  - COLD_START_PANEL = the 2,617 users in it who are cold.
  - WARM_TRUNCATION_PANEL = the 13,767 users in it who are warm.
  - Reported separately and never pooled.
  - Candidate A (a cold subset of M03) cannot serve h = 20 or 50 for most cold users, because they have few events.
- **P12 (the 140 AMBIGUOUS VS→VC rows): move them to CEN_AMBIGUOUS_NEXT_BARRIER.** They are ambiguous with or without
  the Nov 20 fence (code fact, S323), and no total changes.
- **P4 / P5 / P6: accept EVAL-01's proposals** (h = 0 = the query event only; the FIXED_20 addition; one query per user
  per sub-panel; C_ref = the matched central model on the same panel; the few-shot recipe is selected on a disjoint
  VALIDATION panel).
- **FL-01 choices:**
  1. Recenter θ0 on the server: ADOPT.
  2. A separate clip for p: ADOPT (it gives the exact FA equivalence).
  3. The CPF penalty counts each user once per step: ADOPT.
  4. `n_shards` = 2, one per GPU, frozen: ADOPT.

# G0 — Opus response to Fable's position (written after reading `G0_PLAN_FABLE_POSITION.md`)

## Where we independently agreed
1. A cohort of N = 262,144 warm users, with the central contest on the same cohort, not on ALL.
2. Catalogue support ≥ 5 on the full TRAIN (K = 158,486), disclosed as shared bootstrap information.
3. VALIDATION merged minimal-diff. Fable also counts the 140 AMBIGUOUS VS→VC rows as barrier-censored; I had
   counted only the 170 BOUNDARY_UNRESOLVED rows.
4. Clean Rich features, with the M1 valid-only refit and from-scratch training.
5. Strict FP32.
6. The FL defaults (64 clients per round, 2 passes, AdamW reset, μ grid and λ grid).
7. A declared device proxy.

## Where Fable's position differed from mine, and my response
| Topic | Mine | Fable | Response |
|---|---|---|---|
| Central schedule | Executed family recipes (GRU 6 passes; SASRec 5 including the cooldown) | One common EFE-shaped schedule (warmup to 0.02 EFE, peak to 2.0, decay to 0.1× by 3.0, low phase to 5.5, cooldown to 0.01× at 6.0) for both families, both to 6 EFE | **Adopt Fable's.** The executed schedules were timed in optimizer steps for about 157k steps per epoch; the cohort has about 15.3k steps per EFE, so their timings do not transfer (package 08 §3). A common schedule also makes the contest's C runs reusable as the downstream CONTROLLED_BUDGET rows (05 §E) with no extra runs. This is a design choice, not a correction; package 05 §C permits family-specific schedules. |
| LR calibration | Only if a recipe mismatch shows | Always, symmetric: {1.0, 0.5} × executed peak, 2 EFE, seed 2026 | **Adopt.** The 10× population change is itself the mismatch trigger, and the cost is small (about 3.4 GPU-h). Record it as adaptive development. |
| Local batch | 64 | 16 | **Adopt 16.** At 64, 90% of clients take exactly one step per pass. A proximal term is zero at the round start, so it cannot act in one step (06 §4). Cost: +41% FL time. The median client (6 examples) still takes one step per pass at 16; the FedProx effect is therefore concentrated on heavier clients, and this must be disclosed. |
| Cold users | One separate cold block | Split into COLD_NO_TRAIN_EVENTS and COLD_TRAIN_EVENTS_NO_LABEL, plus a 16,384-user cold-h panel | **Adopt.** It is a better population definition. |
| Blockers | — | 7 receipts before the lock | **Adopt all 7** as Wave A tasks. |

## One ambiguity to confirm (not a disagreement)
Fable's point 6 says "both families run to 6.0 EFE" and also "patience 2 evaluations after the 3.0-EFE reference".
Applied literally, early stopping could end a run before the 6.0 endpoint that the CONTROLLED_BUDGET row requires
(08 §2).

My reading, consistent with 08 §2–§3:
- every run completes the full schedule to 6.0 EFE, including the cooldown;
- PRACTICAL_BEST is the strictly-best 0.5-EFE checkpoint within [0, 6];
- tolerance and patience govern only an optional continuation from 6 to 8 EFE, which is allowed only under
  BUDGET_LIMITED_STILL_IMPROVING with a recorded budget.

To be confirmed at the G0 protocol-approval call.

## Remaining disagreement
None is material. My original positions on the schedule, calibration and local batch are superseded by better-founded
choices. They are kept on record in `G0_PLAN_OPUS_POSITION.md`.

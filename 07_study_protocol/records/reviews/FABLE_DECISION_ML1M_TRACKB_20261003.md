# Fable decision — ML-1M Track B device / schedule, split LOCK, K16 INT8 timing (2026-10-03, read-only)

**Scope read:** contracts/PREG1_AMENDMENT_07.md (B3, B6, B7 R5/R7/R9, B8 O1, B10 Q4, B11), PREG1_AMENDMENT_07_ADDENDUM_B12.md,
DECISION_LOG rows 341–367, handoffs/BM-02.md §5 (time projections) and Revisions 2–4. No benchmark holdout / TEST value and no
REES46 MAIN / VALIDATION / TEST value was read. No Codex, no git, no file other than this one written.

## Decision 1 — Option (1): ML-1M Track B stays on CPU as registered. Options (2) and (3) are REFUSED.
- B6 / R9 register the device class CPU-FP32 (16 fixed threads, deterministic). A mid-dataset device switch would need runner GPU
  support, a review cycle and a device-mix disclosure inside one selection grid (R5 ties at 0.0005 are at the float-noise scale),
  and the GPU is the REES46 endgame's resource from ≈ 10-03 18:00Z (panels, freeze, §9, TEST look). Option (3) discards ≈ 180
  finished jobs for no claim gain. The agent budget is ≈ 10 % until the weekly reset (≈ 10-08), so the engineering + review of
  option (2) cannot even be paid for before the CPU path finishes anyway. CPU time is the cheap resource here.
- Nothing downstream waits for ML-1M: R7 locks and looks per dataset. Amazon LOCK + `--final` proceed as soon as the 6 B12 s2027
  jobs finish and the runbook is reviewed; REES46 is unaffected (amendment 07 header).
- No second driver on ML-1M (DL 360 "no two drivers at once"): write-once refusals would mark overlapping jobs FAILED and skip
  their dependents. The 2 spare code_bench4 slots are not re-pointed at ML-1M.

## Decision 2 — Option (4) in the minimal form: a clock-only backstop, registered now, before any value is read.
No cell is dropped a priori (the 24-EFE × max_len-200 C_FULL cells are the ones that make the Track-B "best each system reaches"
claim honest against eSAS at len 200; early stop at 6 flat marks is already registered, B8 O1). The backstop bounds the tail only:

**Exact text — contracts/PREG1_AMENDMENT_07_ADDENDUM_B13.md (REGISTERED by Ops with a DL row, before any ML-1M TEST look):**
> **B13. ML-1M Track-B schedule backstop (clock-only).** Any ML-1M Track-B seed-2026 grid cell (C_FULL, PRE, FA, FA_WARM\*) that has
> no RUN_SUMMARY at **2026-10-06 00:00Z** is stopped by Ops and recorded INCOMPLETE under B10 Q4 (never lockable, never a winner);
> the per-arm Track-B winner is then chosen by the R5 / O1 rule among the completed cells only. The winner's seed-2027 run always
> runs to completion. The trigger is the clock, never a value; every backstopped cell is listed in the ML-1M LOCK disclosure and
> in the Track-B table (the omission can only disadvantage the central arm, the conservative direction for retention ratios).
> Nothing else changes; the other three datasets are unaffected.

INFERENCE on this basis: ML-1M grid done ≈ 10-05, winner s2027 tail ≤ ≈ 2 days, ML-1M LOCK ≈ 10-07, hard bound ≈ 10-08.

## Decision 3 — (a) NO split LOCK. ML-1M has one LOCK and one TEST look, after every Track-A and Track-B run (R7, B11).
- The runner refuses a second look; a split needs code + review + a "two looks on one dataset" disclosure, to gain ≈ 3 days on a
  secondary track. Not worth it.
- Allowed and ASKED: as soon as ML-1M Track A has its RUN_SUMMARYs (≈ 10-03 09:00Z), Ops runs the validation-only pre-lock steps
  `decide-warm` and `seed3` immediately (they are required before the LOCK anyway, R6 / R5), so any seed-2028 run overlaps Track B
  instead of extending the tail. `trackb-select` per arm runs as soon as that arm's grid is complete. No TEST value is touched.

## Decision 4 — (b) K16 INT8 does NOT wait for ML-1M Track B.
K16 is a REES46 device number on the owner's top-priority axis; the §9 read and the REES46 freeze must not inherit a 10-08 wait
from a secondary benchmark track. Two passes, registered now:

**Exact text (DL row + the register that holds DEV-04 / MASTER_PLAN 8.2; Ops places it, Fable concurs):**
> **K16 INT8 timing.** Pass 1 (export, accuracy on the sealed MAIN rows, RSS) runs as soon as the code_bench4 2-slot Amazon resume
> has exited (CPU load = the steady 6-slot Wave-1 driver + FL, ≈ 90 / 128, MEASURED and written beside the result); the MASTER_PLAN
> 8.2 fallback stays automated and disclosed if it triggers. Pass 2 runs after every Wave-1 driver has exited: **latency only**, on
> the identical pinned ONNX artifact and row set of pass 1 (sha asserted; no re-export, no re-evaluation). The reported device
> latency is pass 2; pass-1 latency is printed as "loaded host" and never used as the device number. Accuracy is pass 1 (sealed,
> opened only at §9). If the lane cannot be split into passes without code change, pass 1 runs whole under load (disclosed) and
> pass 2 re-measures latency only from its artifact.

## Integrity rules kept
No value read; selection on inner validation only; one LOCK and one TEST look per dataset; CPU-FP32 device class unchanged; every
change above is registered before any ML-1M look and is clock- or artifact-bound, never value-bound.

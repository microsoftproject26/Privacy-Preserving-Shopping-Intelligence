# PREG1 amendment 02R2 — supersedes the LR/dropout assignment of 02R1 §2. Everything else in 02R1 stands.

- **Status:** REGISTERED by the Operations Director on 2026-09-27, BEFORE any P2 launch (no 02R1 trial has started;
  none to void). Authority: the independent Opus gate review of TRAIN-02 (finding 5, uneven LR coverage per size level)
  and Fable's CONFIRM with four conditions (DECISION_LOG 70). PREG1_AMENDMENT_02R1.md is kept unchanged as the record.
- **Why:** 02R1's single 12-stratum draw in variant-major order gave different LR ranges to different size levels, so
  size was confounded with LR. Size is the owner's question, so every size level must see the SAME cells.

## 1. Stratified log-grid of LR values (registered values, not a procedure; no "LHS")
Stratum midpoints on log10 over [3e-4, 3e-3], full precision in the HP registry:
- GRU, 6 strata per size: ≈ 3.6e-4, 5.3e-4, 7.7e-4, 1.14e-3, 1.67e-3, 2.45e-3 (strata 0–5).
- SASRec, 4 strata per size: ≈ 4.0e-4, 7.1e-4, 1.26e-3, 2.24e-3 (strata 0–3).

## 2. Cells per size level (identical for every size level of a family)
- **GRU** (6 trials per size = 2 schedules × 3 dropouts): S0 → strata {0, 3, 5} with p {0.1, 0.2, 0.3} respectively;
  S1 → strata {1, 2, 4} with p {0.3, 0.2, 0.1} respectively. Mean stratum per schedule is balanced (8/3 vs 7/3); mean p is
  0.2 in both schedules, and the p-to-LR pairing is mirrored between schedules.
- **SASRec** (4 trials per size = 2 schedules × 2): S0 → strata {0, 3} with p {0.1, 0.3}; S1 → strata {1, 2} with p
  {0.2, 0.2}. Mean stratum per schedule 1.5 vs 1.5; mean p 0.2 in both schedules.
- GRU dropout p means input p and readout p/2; SASRec single p.

## 3. Resolution of an arithmetic conflict (Ops, recorded; reported at G1b)
Fable's condition 3 requires the (LR, schedule, dropout) cell set to be identical across the size levels of a family.
02R1's "each SASRec p level 4× per family" (12 trials, 3 sizes) would need 4/3 trials per p per size, which is
impossible with identical per-size sets. The identical-cells condition (size fairness) takes precedence. The SASRec
family counts are 0.1: 3, 0.2: 6, 0.3: 3, and every (size, p) pair appears at least once. The middle level is the
duplicated one, being the centre of the grid.

## 4. Tests required in the registry re-issue
- The (LR, schedule, dropout) cell multiset is byte-identical across size levels within each family.
- Every (size, p) pair appears at least once; the family p counts equal §3.
- The LR values equal §1 at full precision; the schedules equal 02R1 §2.

## 5. Disclosure
Within a size, each schedule sees only half of the LR strata, so the schedule and dropout effects remain partly
confounded with LR. This is acceptable because size is the owner's question and the per-size stage-2 re-runs are
unchanged. The HP registry is re-issued as a NEW file (`HP_REGISTRY_P2R2.json`); `HP_REGISTRY_P2.json` is kept and
never launched. Any trial launched under the old registry would be HP_SEARCH_VOID (none exists).
Totals are unchanged: 12 stage-1 trials per family, stage 2 as in 02R1, and the selection rules as in 02R1 §3.

## Erratum (2026-09-27, before any launch; Opus delta review finding 5; values computed, not transcribed)
The "≈" values in §1 are loose roundings. The exact stratum midpoints (4 significant digits) are: GRU 0.0003635, 0.0005335, 0.000783, 0.001149, 0.001687, 0.002476;
SASRec 0.0004001, 0.0007114, 0.001265, 0.00225. The HP registry (HP_REGISTRY_P2R2.json, self a3b48f20…) carries the full-precision values and is
authoritative.

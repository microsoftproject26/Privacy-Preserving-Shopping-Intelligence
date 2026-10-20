# POPULATION_5_100 — Fable R4 (verbatim summary of the final message)

ROUTING: claude-fable-5-1.

VERDICT: **NO CHANGE** to the FINAL ruling (`POPULATION_5_100_FABLE_POSITION_FINAL.md`, APPROVE_WITH_AMENDMENTS), except
one arithmetic delta. STEP B P1–P5 are answered by that ruling as adopted, and no further ruling is issued.

**DELTA (D3, amendment 5, budget line only):**
- "≤ 6 runs per family, ≈5 GPU-h total" becomes "≤ 8 runs per family (L0, L3, S*, and the single fallback retrain,
  each at seeds 2026/2027)".
- Expected cost: ≈5 GPU-h for 3 schemas, ≈6.4 GPU-h if the fallback fires (INFERENCE from the cost model at 32,768: 0.6
  GRU / 1.0 SASRec per schema at 2 seeds).
- Plus ≈1.1 GPU-h for the 168 permutation evaluations (14 groups × 3 permutation seeds × 2 families × 2 seeds; ≈21 s GRU
  and ≈28 s SASRec each, on ≈127.5k FS VAL rows).
- STEP B's cap of 10 schema evaluations is compatible, since at most 4 are visited.

**Cosmetic, no effect:** in D9, "10,108 updates/EFE" is 2,587,776 / 256 = 10,108.5. The 6.0-EFE total of 60,651 stands.

**Implementation-consistency note (not a ruling change):** the FINAL ruling supersedes the first ruling's FS band. FS is
the top 32,768 keys, with the U_NEW bands directly below it, and the shared statistics are full-TRAIN as fitted (FINAL D5,
disclosed). Confirm that the builder targets the FINAL D1 bands before any FS run.

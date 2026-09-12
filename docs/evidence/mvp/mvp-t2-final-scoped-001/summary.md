# T2 Final Scoped Matched Comparison Summary

- **Protocol**: `T2_FINAL_SCOPED_MATCHED_V1`
- **Status**: `T2_FINAL_SCOPED_SINGLE_SEED_READY`
- **Model Path**: `T2_FROM_SCRATCH_FINAL_SCOPED`
- **Full Corrected Validation**: 392,554 decisions, 11,297 positives
- **Training Population**: 200 eligible TRAIN clients
- **Schedule**: 10 rounds x 20 clients/round, batch 64, local epochs 1

## Results

| Regime | PR-AUC (AP) | Delta | Quality Retention |
|---|---|---|---|
| **R1 Centralized** | **0.0654**  | — | 100.00% |
| **R2A Flower FedAvg** | **0.0384**  | **-0.0270** | **58.71%** |

### Per-Seed Detail

| Seed | Round 0 AP | R1 AP (R10) | R2A AP (R10) | Delta | Retention |
|---|---|---|---|---|---|
| Seed 13 | 0.0371 | 0.0654 | 0.0384 | -0.0270 | 58.71% |

## Notes
- Quality Retention is the mean of per-seed retentions: **58.71%**.
- Companion ratio-of-means: **58.71%**.
- Evaluated on full corrected validation membership (392,554 decisions).
- Zero TEST rows accessed.

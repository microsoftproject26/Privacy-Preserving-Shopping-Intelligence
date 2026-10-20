Rebuilt from `privacy_dp_cost.csv` by `run_pipeline.py --from-results` (numbers copied, never recomputed).

| source | id | comparison | measure | seeds | value | retention_pct_or_ci_low | ci95_high |
|---|---|---|---|---|---|---|---|
| confirmatory test | K12 | R(DP8 / FA_1024) | retention_pct (same schedule, noise only) | 2026 2027 | 98.52 | 97.7 | 99.38 |
| confirmatory test | K13 | R(DP1 / FA_1024) | retention_pct (same schedule, noise only) | 2026 | 94.38 | 92.53 | 96 |
| confirmatory test | K12w | R(DP8_FROZEN / FA_1024_WARM_FROZEN) | retention_pct (same schedule, noise only) | 2026 2027 | 100.00 | 99.95 | 100.07 |
| confirmatory test | K13w | R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN) | retention_pct (same schedule, noise only) | 2026 | 100.03 | 99.92 | 100.15 |
| confirmatory test | Q6 | R(W_STAR_DP8 / W_STAR_FA_1024) | retention_pct (same schedule, noise only) | 2026 2027 | 100.00 | 100 | 100 |
| confirmatory test | Q11 | R(W_STAR_DP1 / W_STAR_FA_1024) | retention_pct (same schedule, noise only) | 2026 | 99.99 | 99.98 | 100 |
| exploratory validation | - | FA_COLD_TAUFIX_DP8 vs FA_COLD_TAUFIX | validation mrr@20, DP8 / non-private (DP run: 384 Poisson rounds with clipping and noise; best validation value per seed, 2026 and 2027: DP8 0.03804 and 0.03882, non-private 0.13061 and 0.13067; the value column shows the two-seed means) | 2026 2027 | 0.03843 / 0.13064 | 29.42 | None |
| exploratory validation | - | FA_WARM_PB100K_TAUFIX_DP8 vs FA_WARM_PB100K_TAUFIX | validation mrr@20, DP8 / non-private (DP run: 384 Poisson rounds with clipping and noise) | 2026 2027 | 0.1461 / 0.1533 | 95.29 | None |

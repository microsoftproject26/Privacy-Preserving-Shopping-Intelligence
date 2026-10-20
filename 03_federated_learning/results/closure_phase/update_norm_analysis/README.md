# P5 / T09 — existing D1-ext update-log summary

Status: analysis and figures complete. Label: **DESCRIPTIVE**. No training or TEST evaluation.

The analysis reads 29 existing Phase-2 MAIN/WA runs, covering 100,942 rounds. All recorded server steps have finite norms. Input hashes were checked before and after each run's analysis and did not change.

- `GROUP_SUMMARY.csv`: median, p99, range and mean per run, tensor group and metric; missing-value counts.
- `TOTAL_BY_ROUND.csv`: total aggregate norm/RMS, server step norm/RMS, drift, cosine and logged client count for every round. DP logs never include client counts.
- `UPDATE_NORM_SUMMARY.json`: input hashes, optimizer settings, run coverage, final drift, cosine summaries and interpretation limits.
- `WSTAR_TRAJECTORIES.{png,svg,pdf}`: both existing MAIN W* seeds.
- `WA_FEDADAM_TRAJECTORIES.{png,svg,pdf}`: existing FedAdam CALIB screen cells.

For W* seed 2026, median server-step RMS is `7.307290178722998e-08`; final relative drift on trained parameters is `0.0006002893924251696` (0.06003%). Seed 2027 ends at 0.05985%. The item tables are frozen and excluded from this trained-parameter denominator. These observations support the documented near-inert original trajectory.

`agg_delta_l2` is a group/vector L2 norm, not an individual update magnitude. `agg_delta_rms` equals L2 divided by the square root of the group's parameter count. These records cannot identify an exact fraction of coordinates below tau. The CSV instead reports the rigorous lower bound `max(0, 1 - L2^2/(numel*tau^2))` and the JSON separately reports the fraction of logged RMS values below the reference tau. Neither is mislabelled as the measured coordinate fraction. The 1e-3 reference is the original FedAdam tau; it is a descriptive reference for other optimizers too.

Re-run into a **new** output directory:

```bash
nice -n 19 /opt/conda/bin/python -B "/home/jovyan/EID PROJECT NEW/DELIVERY_REVIEW/code/summarize_update_norms.py" \
  --v "/home/jovyan/EID PROJECT NEW/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4" \
  --out "/home/jovyan/EID PROJECT NEW/DELIVERY_REVIEW/evidence/P5_UPDATE_NORMS_RERUN"
nice -n 19 /opt/conda/bin/python -B "/home/jovyan/EID PROJECT NEW/DELIVERY_REVIEW/code/plot_update_norms.py" \
  "/home/jovyan/EID PROJECT NEW/DELIVERY_REVIEW/evidence/P5_UPDATE_NORMS_RERUN"
```

The summarizer needs only Python's standard library. The plotting step uses the server's existing matplotlib installation; no shared environment was modified. Twelve synthetic checks passed for RMS/quantiles, the coordinate bound, missing values, no-op rounds, malformed numbers, duplicate/gapped rounds and inconsistent group totals.

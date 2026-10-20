# PREG1 amendment 06L (part D) — implementation reading of 06L A1–A4 (registration only)

- **Status:** REGISTERED 2026-09-29 ~22:00Z by the Operations Director, BEFORE any S2 value is read. At registration 5 of the 7 S2
  CALIB runs had finished (FedAvgM m 1 / 2 still running); no S2 RESULT_ROWS value has been opened; no S2 selection file exists.
- **Why:** the frozen selector (`code_ds3r3/v4int/lr_screen.py` b1e243f4) implements 06J only. `select-joint --stage S2` would refuse
  the two 06L A1 runs ("not registered"), and S3 has no place for the 06L A2 cells. The rules below only make 06L A1–A4 executable;
  they change no registered value, threshold or grid. Code: task FL-14; independent review REV-FL-14 before any use.

## D1. S2 run set (06J §1 + 06L A1)
- REQUIRED = the 06J §1 set (FedAdam η_s {0.01, 0.1} × m {1, m*}; FedAvgM β 0.9 η 1 at m*) **plus FedAvgM at m ∈ {1, 2}**. A cell
  that coincides with another (m* ∈ {1, 2}) is one run.
- A1 edge: if the raw-best FedAvgM cell among the required FedAvgM cells is m = 1, **FedAvgM m = 0.5** is REQUIRED before the S2
  file is written (EdgeRunsMissing otherwise). Conditions read raw values, as in 06J.
- Edge evaluation order: 06J η edge → A1 FedAvgM edge → 06J τ edge. The τ condition is evaluated once every edge whose condition
  holds is complete, on the S2 winner (06J §4(ii) applied) over incumbent + required + those edge runs.
- Selection rule unchanged: 06J §1 + §4(ii) (≥ 0.002 over the incumbent, 0.0005 band on raw values); ties FedAvg (incumbent) →
  FedAvgM → FedAdam, then smallest |log2 m|, then smaller η_s, then larger τ.

## D2. S3 run set (06J §2 + 06L A2)
- REQUIRED = the 06J §2 cells built on the S2 winner **plus** the two A2 cells: FedAdam (η_s 0.01, τ 1e-3, v0 = τ²), m 1, FL_CONST,
  client SGD-m 0.9 at client lr {0.1, 1.0}. An A2 cell identical to a 06J cell (S2 winner = FedAdam η_s 0.01 τ 1e-3) is one run.
- The 06J §2 SGD edge rule reads **only the 06J SGD cells** (those on the S2 winner's server optimiser); the A2 cells never trigger
  edge runs.
- Selection rule unchanged (06J §2 + §4(ii)); ties: incumbent → AdamW before SGD → smaller SGD lr → FedAvg family before FedAdam.

## D3. Downstream checks
`launch_fl.load_fl_tuning` re-checks the D1 / D2 sets from the stage files exactly as select-joint does; S4 (06J §3) is unchanged and
is built on whatever point wins S3 (an A2 cell included).

## D4. 06L A3 / A4 in code
- A3: the 06H DP_OPT_SENS trigger fires for any selected server optimiser other than plain FedAvg (FedAvgM included).
- A4: FA_1024 / DP runs record the per-round fraction of clipped client deltas when z = 0, and only the aggregate clipped count per
  round when z > 0 (no per-client norms written).

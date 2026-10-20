# PREG1 amendment 06H — Fable review of 06G (FedAdam): DP-arm server-optimiser sensitivity (registration only)

- **Status:** REGISTERED 2026-09-29T09:25Z by Claude Fable 5.1 (chief scientist) as the review note requested in DL 186, BEFORE any FedAdam
  run exists and before any MAIN FL value exists or is read. 06F and 06G stay in force unchanged; this adds one conditional
  sensitivity and one analytical disclosure.

## 1. Verdict on 06G (OPINION, with the algebra below): APPROVED
FedAdam as registered follows Reddi et al. 2021 Alg. 2 (server pseudo-gradient = the FedAvg aggregate delta; β1 0.9, β2 0.99,
τ 1e-3, no bias correction); the joint CALIB selection reads only registered screen values and is symmetric across the FL arms;
applying the server optimiser to the DP arms' noisy fixed-denominator mean is post-processing of the released quantity, so the
(ε, δ) accounting is unchanged; the tie order (FedAvg before FedAdam, then |log2 m|, then smaller η_s) is conservative.

## 2. The one risk that must be pre-registered (INFERENCE from the registered mechanism; no value read)
Under DP the server sees Δ_r = (Σ clip_S(δ_u) + z S ξ_r) / m per coordinate, whose noise standard deviation is z S / m
(≈ 5.5e-4 × S at z = 0.5642, m = 1,024; ≈ 1.3e-3 × S at ε = 1). FedAdam normalises every coordinate by √v_r + τ, and v_r
converges to the second moment of Δ_r. In the coordinates where the noise dominates the signal (most of the 20.7 M coordinates of a
sparse-update item table), √v_r ≈ z S / m, so the per-round step becomes ≈ η_s × (noise / its own std) = a random walk of size ≈ η_s
per coordinate per round, independent of S, whenever z S / m ≫ τ. FedAvg with server LR 1 instead moves those coordinates by the raw
noise ≈ z S / m per round. Whether FedAdam amplifies or damps the DP noise therefore depends on S / m versus τ and on η_s, and it can
make the measured "privacy cost" (K12, K13) a property of the server optimiser rather than of the privacy mechanism.

## 3. Registered conditional sensitivity (SHOULD-high; run only if the 06G joint selection picks FedAdam)
- **DP_OPT_SENS:** FA_1024 s2026 and DP8 s2026 re-run under FedAvg (server LR 1, the same selected local multiplier m*, the same
  S_RECORD produced under the selected configuration, everything else identical). Two extra runs (INFERENCE ≈ 2 × 10.7 process-hours).
  Reported as DESCRIPTIVE beside K12: R(DP8 / FA_1024) under FedAvg vs under FedAdam. If the FedAvg pairing shows a materially
  smaller privacy cost (the registered flag: the CONTROLLED_BUDGET ratio differs by more than 5 retention points), the thesis
  presents both and discusses the noise-normalisation mechanism of §2 as the cause; the confirmatory K12/K13 stay under the selected
  optimiser (06G §3), optimiser-matched between numerator and denominator.
- **S_CAL under the selected configuration** (06G §3 already requires it) — the S used by DP_OPT_SENS is that S_RECORD (no second
  calibration), disclosed.
- If the joint selection picks FedAvg, this amendment has no run and the §2 note is a paragraph in the DP appendix.

## 4. Disclosures added
(a) The FedAdam moments are checkpointed and resumed bitwise (FL-10 must prove it). (b) The DP guarantee covers the released sequence
θ_1..θ_T; the server moments are functions of it (post-processing). (c) The §2 mechanism is stated before any DP value is read.

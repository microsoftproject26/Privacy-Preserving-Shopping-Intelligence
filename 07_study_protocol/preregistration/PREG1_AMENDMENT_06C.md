# PREG1 amendment 06C — DP sampling corrected to Poisson (registration only)

- **Status:** REGISTERED 2026-09-28 (~20:45Z) by the Operations Director (Claude Opus 5.5) under the owner's standing
  instruction "do everything you see as right". Registered BEFORE any FA_1024 / S_CAL / DP run exists (none has been
  launched; no downstream value read). Trigger: FL-06 (`company/handoffs/FL-06.md`,
  `code/tests_fl/receipts/DP_WOR_SENSITIVITY_FL06.json`).
- **Finding (MEASURED by FL-06):** the registered accountant (`code/fl/accountant.py`) is the Poisson-subsampled Gaussian RDP
  bound under add/remove adjacency, but the registered sampler for FA_1024/DP is fixed-size uniform sampling without
  replacement (amendment 06 §2 Q4). For fixed-size sampling the natural adjacency is replace-one, whose sensitivity is 2S;
  under Wang–Balle–Kasiviswanathan 2019 Thm 9 the achieved ε at z = 0.5642 would be ≈ 1,388 (lower bound ≈ 1,133), not 8,
  and ≈ 6.7 instead of 1 at z = 1.3094. The registered ε values would therefore not describe the mechanism actually run.
- **Decision (replaces amendment 06 §2 Q4 for these arms only):** FA_1024, S_CAL and every DP arm use **Poisson sampling**:
  each of the N = 131,072 MAIN clients joins round r independently with probability q = 1,024 / 131,072 = 2⁻⁷, seeded by
  derive_seed(seed, "poisson", r); the **denominator stays fixed at m = 1,024** (the fixed-denominator estimator of McMahan,
  Ramage, Talwar & Zhang 2018, ICLR); T = 384 rounds; clipping S, noise z·S on the sum, δ = 5e-6 unchanged. The registered
  accountant, z(ε=8) = 0.5642, z(ε=1) = 1.3094 and z(ε=4) = 0.7244 are then exact for the mechanism (add/remove adjacency,
  user level). Empty or small rounds are valid (the noisy sum over the fixed denominator is still released).
- **Unchanged:** FA/FP/PF/FA_Q8 keep sweep sampling with 10% drop-out (no DP claim). FA_1024 changes sampling together with
  DP so that FA_1024 → DP isolates the noise (K12, K13). Disclosed: FA → FA_1024 changes cohort size and sampling scheme.
- **Code:** task FL-07 (after FL-05 lands, same files): `sampling="poisson"` in fl/participation.py; every config with a DP
  block (clip or noise) must use poisson with group_size == denominator (refuse otherwise); expected-exposure accounting for
  the LR schedule (planned n_r = q·N·E[visit rows]; actual n_consumed logged); tests incl. empirical inclusion rate ≈ q and a
  negative control that a WOR plan with z > 0 is refused. The WOR receipt stays as the documented reason for this change.
- **Thesis:** report ε under the registered Poisson/add-remove accountant (now matching the mechanism); cite FL-06's WOR numbers
  in the DP appendix as the reason fixed-size sampling was abandoned.

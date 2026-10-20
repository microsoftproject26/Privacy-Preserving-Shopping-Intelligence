# PREG1 amendment 06L (part E) — clipped-fraction reporting under DP (registration only)

- **Status:** REGISTERED 2026-09-29 ~22:55Z by the Operations Director, before any DP or FA_1024 MAIN run exists and before any S2
  value is read. Narrows 06L A4 / 06L_D D4; nothing else changes.
- **Why:** 06L_D D4 asked DP runs (z > 0) to record the aggregate clipped count per round. That count is a raw, data-dependent output
  released without noise, so it is not covered by the RDP accountant and would weaken the stated (ε, δ) guarantee (FL-14 handoff).
  Adaptive-clipping work that uses such counts noises them and charges the budget; we do not adopt adaptive clipping.

## E1. Rule
- z = 0 runs (FA_1024, S_CAL): record the per-round clipped fraction (`CLIP_LOG.jsonl`: round, n_clipped, n_deltas,
  clipped_fraction; no per-client norms). Implemented by FL-14.
- z > 0 runs (DP8, DP1, DP4 if run, DP8_FROZEN): **record nothing** about clipping. The thesis reports the clipped fraction of the
  matching z = 0 twin (FA_1024 at the same S and configuration) as the descriptive proxy, labelled as such.
- `fl/dp.py` is not changed for this.

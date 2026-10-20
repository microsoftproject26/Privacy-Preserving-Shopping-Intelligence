"""Building blocks of central (cloud) training, on the same clock as the federated simulator.

  objective  exact full-catalogue cross-entropy as (sum, count) and exact microbatch accumulation
  order      the deterministic per-seed data order (one permutation per seed and pass, hashed)
  schedule   the exposure-based LR schedule, the planned LR table and the stopping rule (no early stop, patience 2)
"""
from __future__ import annotations

__all__: list = []

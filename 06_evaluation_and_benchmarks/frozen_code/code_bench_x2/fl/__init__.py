"""fl — PPSI V4 FL / PFL / LO mechanics core (task FL-01, owner: Federated Systems Lead).

A compact, family-agnostic simulator core. It knows nothing about GRU or SASRec: a model family plugs in through the
adapter protocol in `fl.adapter` (query -> q [B, d], head_weight / head_bias, parameter-role manifest, optional
post_step). The two executed families are wired in `fl.families`, which imports `ppsi_env` first (mandatory isolation).

Modules
  numerics    strict FP32, determinism flags, stable seed derivation, state digests
  adapter     ModelAdapter protocol, parameter-role manifest (shared / private / buffer / alias), state load/extract
  client      local solver (fresh AdamW per visit, B = 16, tail batch, mean CE over full K, FedProx term, PF vector)
  aggregate   deterministic shard plan, streaming FP32 weighted accumulation, buffer rules, empty-round rejection
  personal    private-key store for p_u + its optimizer state, CPF loss with the same forward
  comm        virtual byte accounting (send / receive per visit; aliases once; private state 0)
  server      in-process round driver: broadcast, visits with identical-RNG retry, aggregation, recentering
  pool        process-pool round driver (2 concurrent workers), same shard reduction as the in-process driver
  local_only  LO: one client alone from theta0, persistent AdamW, procedural train -> eval hook -> discard
  synthetic   tiny synthetic family + synthetic clients (tests and micro-benchmark only)
  families    GRU / SASRec adapters over the snapshot models (imports ppsi_env first)
  lo_determinism   the 1,024-client LO determinism receipt (CLI: python -B -m fl.lo_determinism)
  bench_visit      per-visit CPU micro-benchmark (CLI: python -B -m fl.bench_visit)
  dp          FL-04 DP-FedAvg aggregation: flat L2 clipping of client deltas, noise on the sum, fixed denominator m
  accountant  FL-04 RDP accountant (Poisson-subsampled Gaussian), epsilon / z_for_epsilon, registration record
              (CLI: python -B -m fl.accountant --out <json>)
  finetune    FL-03 full-model client fine-tuning FT_C / FT_FA / FT_KD (budget grid passes x LR fraction)
  (FL-03 drop-out and uniform per-round sampling live in participation / runtime; all new options default OFF)

Nothing here reads dataset rows, touches a GPU, or writes outside the caller-supplied paths.
"""
from __future__ import annotations

__version__ = "FL-01.0"

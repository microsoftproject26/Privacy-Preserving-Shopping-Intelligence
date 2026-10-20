"""Federated learning simulator for the deployment study.

A compact, model-agnostic simulator of cross-device federated training in one process. It knows nothing about a
particular recommender: a model family plugs in through the adapter protocol in `ppsi.fedsim.adapter` (query ->
q [B, d], head_weight / head_bias, a parameter-role manifest and an optional post_step hook).

Modules
  numerics        strict FP32, determinism flags, stable seed derivation, state digests
  adapter         ModelAdapter protocol, parameter-role manifest (shared / private / buffer / alias), state load/extract
  client          local solver (fresh AdamW per visit, B = 16, tail batch, mean CE over the full catalogue, FedProx
                  term, personal vector)
  aggregate       deterministic shard plan, streaming FP32 weighted accumulation, buffer rules, empty-round rejection
  personal        private-key store for the personal vector p_u and its optimizer state; central personalised loss
  comm            virtual byte accounting (send / receive per visit; aliases once; private state 0 bytes)
  server          in-process round driver: broadcast, visits with identical-RNG retry, aggregation, server optimisers
  pool            process-pool round driver, same shard reduction as the in-process driver
  local_only      one client alone from theta0, persistent AdamW, train -> evaluation hook -> discard
  finetune        full-model on-device fine-tuning of a cloud or federated model
  dp              DP-FedAvg aggregation: flat L2 clipping of client deltas, noise on the sum, fixed denominator
  accountant      RDP accountant for the Poisson-subsampled Gaussian mechanism
  accountant_wor  RDP accountant for fixed-size sampling without replacement
  participation   seeded per-round client sampling, Poisson sampling and client drop-out
  schedule        learning-rate schedule over effective full-data epochs
  runtime         the resumable run loop (configuration, checkpoints, exposure counters, item-table freezing)
  checkpoint      atomic, hash-verified checkpoints with RNG state
  update_log      per-round update-norm monitoring
  quant           8-bit stochastic quantisation of client uploads
  sparse_rows     sparse upload of the item-table rows a client actually touched
  synthetic       a tiny synthetic family and synthetic clients (tests only)

Nothing here reads dataset rows, touches a GPU unless asked to, or writes outside the caller-supplied paths.
"""
from __future__ import annotations

__version__ = "1.0"

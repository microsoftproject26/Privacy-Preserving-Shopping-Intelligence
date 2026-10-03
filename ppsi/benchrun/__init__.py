"""Benchmark runner: the deployment-study arms (central, federated, on-device fine-tuning) on public sequential-
recommendation benchmarks with leave-one-out splits.

  data       train.csv / holdout.csv -> per-user sequences, decision rows, the inner validation split, the band
  metrics    full-catalogue ranking with seen-item filtering; HR / NDCG / MRR
  model      the ID-only SASRec (ppsi.seqrec) at the benchmark catalogue size
  recipe     the arms and their hyperparameters (a JSON recipe merged over illustrative defaults)
  central    C_FULL / PRE central training (ppsi.central schedule, order and objective)
  fl_engine  the federated arms (ppsi.fedsim), including the DP calibration chain
  device     per-user on-device fine-tuning of a federated base model
  holdout    the holdout (TEST) evaluation of finished runs
  accel      the opt-in deterministic GPU device class
  run        the command line
"""
from __future__ import annotations

__all__: list = []

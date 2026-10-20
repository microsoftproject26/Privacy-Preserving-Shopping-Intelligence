"""v4bench — CPU benchmark runner that applies the REES46 (PPSI V4) recipe, unchanged, to the eSASRec standard
leave-one-out splits (task BM-02; BENCHMARK_PLAN_V1 §14; to be registered as PREG1_AMENDMENT_07 before any run).

Nothing here adds a method. Every arm is a REES46 arm, built from the REES46 code:
  model      v4models.sasrec_v2.SASRecCE, ID-only (group item_only), the SASREC_D64_B2 / SASREC_D256_B3 entries of
             v4models.variants, C7 recentered theta0, the fl.families.SASRecAdapter;
  loss       v4train.objective (EXACT_FULL_K_CE) for central updates; fl.client.batch_ce (the same full-K CE) on clients;
  central    v4train.schedule (S0, planned LR table, marks, StopTracker, C3 continuation), v4train.order permutations,
             the C8 recipe constants of v4train.recipe (AdamW, B256, clip 1, wd 1e-5);
  federated  v4int.launch_fl.arm_spec (the REES46 arm specs, tuning point, warm factors, frozen tables, Poisson rule) +
             fl.runtime.FLRun / fl.server / fl.client / fl.participation / fl.quant / fl.dp / fl.accountant /
             fl.personal / fl.checkpoint;
  device     fl.finetune.FTRunner with the registered SELECTION_FT_C cell (v4int.launch_device semantics).
Modules: common (paths, pins, json), data (the adapter), metrics (full-catalogue ranking + metrics), model, recipe
(contracts + arm specs), evaluate, central, fl_engine, device, final (the TEST gate), run (the CLI).
"""
from __future__ import annotations

__version__ = "BM-02.0"

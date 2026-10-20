"""v4train — the PPSI V4 central trainer, calibration and family selection (task TRAIN-01; both families).

  schedule      s(EFE) on the locked knots (G0 D6; PREG1 T2) + continuation (G0B C3); exposure point "end" (a required
                run-config key); the planned per-update LR table (pin P2); the binding stopping rule (StopTracker)
  recipe        the frozen C8 central recipe and the hashed per-run configuration (COHORT or FULL_TRAIN population)
  evalconfig    the ONE frozen evaluation config (PREG1 T3) from v4eval.frozen_config.load()
  order         the deterministic per-seed data-order policy (pass permutations, sha256-recorded)
  provider      BatchProvider / EvalProvider protocols (DATA-04's v4data.store implements them) + synthetic providers
  registry      the manifest-role registry (roles -> manifest ids, verified by self-hash; provider bindings) and the
                derived-schema acceptance (TRAIN-01R2); registries/ holds the HISTORICAL population's registry file
  objective     full-K CE as (sum, count) without host syncs; exact microbatch accumulation (package 10 §2)
  checkpoint    atomic checkpoints (write, fsync, reload/validate, replace), strict JSON records, retention
  trainer       CentralTrainer: run / resume / continue_run; monitors; evaluation through v4eval every 0.5 EFE
  calibration   the {1.0, 0.5} x 1e-3 peak-LR calibration at 2.0 EFE (seed 2026, fresh inits, no weights kept)
  selection     the family selection with the PREG1 T1 trigger (T1a-T1d) and the staged full-TRAIN tie-break,
                the ARCHITECTURE_LOCK draft (never the lock), the owner question, the downstream / C_MATCHED guards
  bootstrap_ci  the v4eval paired user bootstrap CIs T1 needs (delta_bar for T1d; the stage-1 single-seed delta)
  cpf_stub      the CPF central-personalized mode: registered interface for G3, not implemented
  hp_registry   TRAIN-02: the P2 HP registry (per-family registered trials, LHS on log LR, O_EXCL, self-hashed)
  ckpt_avg      TRAIN-02: CKPT_AVG of a TUNING run (uniform average of the 5.0 / 5.5 / 6.0 snapshots, evaluated
                through the trainer's own frozen path, then deleted with sha256 logged)
  theta0_variants  TRAIN-02: theta0 per model variant x seed (THETA0_VARIANTS.json; run on the server by Ops)
  p2_driver     TRAIN-02: the P2 search driver (stage 1 / 2 through v4int/launch_central.py; 02R1 §3 selection)
  tuned_recipes TRAIN-02R1..R3 / TRAIN-03: the 02b tuned-recipes registry and the G1B_APPROVED format
  unseal        TRAIN-03: the G2 unseal gate of the sealed P3 contest runs (Fable staggered-P3 ruling, condition 2)
  p3_driver     TRAIN-03: the P3 contest queue of one family (--recipe-id; sequential / parallel; --code-root)

Imports `ppsi_env` first (isolation, T43). Run Python with -B. CPU-tested on synthetic data only; the real runs happen
after G1.
"""
from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

# FL-02 finding: snap_v1/reuse/train.py runs os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0") when ppsi_env
# pre-imports it, so after that import the variable is ALWAYS set. Capture what the launcher exported BEFORE the
# import (UNKNOWN when ppsi_env was already imported by the caller); the trainer refuses an unset launch value.
LAUNCH_CUDA_VISIBLE_DEVICES = ("UNKNOWN_PPSI_ENV_ALREADY_IMPORTED" if "ppsi_env" in _sys.modules
                               else _os.environ.get("CUDA_VISIBLE_DEVICES"))

_CODE = _Path(__file__).resolve().parents[1]
if str(_CODE) not in _sys.path:
    _sys.path.insert(0, str(_CODE))
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

__version__ = "TRAIN-03"

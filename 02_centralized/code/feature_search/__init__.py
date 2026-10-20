"""v4fs — the FAST FEATURE SEARCH (owner speed directive, DECISION_LOG 39; task SCHEMA-01).

  schemas      the FEATSTAT ladder rungs as derived schemas (contracts/derived_schemas: L0_CLEAN_V2, L3_NEAR_DUPLICATE,
               REGISTRY.json, THETA0_REGISTRY.json) and derive(base_id, drop_groups) ("L3 minus the given groups")
  importance   grouped permutation importance of a trained model on an EvalProvider under the frozen T3 evaluator
               (v4eval), with the paired user bootstrap; evaluation only
  run_fs       the FEATURE_SEARCH_SCREEN central run: v4int providers (history_policy USER_HISTORY_L50), the FS
               manifests (REGISTRY_FS.json), the v4train CentralTrainer, PRACTICAL_BEST retained only until the
               importance is computed; every row labelled FEATURE_SEARCH_SCREEN; --dry-run on synthetic providers
  theta0       the theta0 hash registration of a derived schema for the running torch build (server: seconds)

Nothing is imported at package import time: run_fs must export CUDA_VISIBLE_DEVICES / CUBLAS_WORKSPACE_CONFIG before
torch, v4train or ppsi_env are imported. Every other module imports v4models (ppsi_env first).
Labels: FEATURE_SEARCH_SCREEN rows are screening evidence only; v4train.selection refuses them (TRAIN-01R2 R3).
"""
__version__ = "SCHEMA-01.0"
RUN_LABEL = "FEATURE_SEARCH_SCREEN"

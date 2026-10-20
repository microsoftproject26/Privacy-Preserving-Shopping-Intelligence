"""v4int — real-data providers and the central launcher (task INT-01, the G1 integration gap).

  verify           fail-closed verification of a Clean V2 manifest before use: the manifest files' sha256 against
                   DATA_PROTOCOL_LOCK and MANIFEST_HASHES, the header (status, lock binding, content self-hash), the
                   TRANSFORM_STATE self-hash, the canonical order; every TEST manifest is refused
  cache            the manifest-scoped Clean V2 store: the snapshot CompactStore built bucket group by bucket group,
                   keeping only the event rows the manifest's decision windows touch (a materialized in-RAM cache);
                   batches go through the UNMODIFIED CleanV2Store.batch -> CompactStore.batch -> clean_v2_view path
  providers        StoreTrainProvider (M01 / M05; v4train.provider.BatchProvider) and StoreEvalProvider (M02 / M06;
                   v4train.provider.EvalProvider, manifest built through v4eval.manifest.build_eval_manifest)
  smoke            SMOKE_NOT_EVIDENCE labelling of smoke result rows (refused by v4train.selection)
  launch_central   the production entry point (python -B code/v4int/launch_central.py ...)
  retro_store      INT-07: the GATED RETRO_TEST store / history-window path (every entry point re-validates the
                   v4data09 TEST gate; the locked / default paths above are unchanged and still refuse TEST)
  dump_rows_retro  INT-07: v4dev.dump_rows --role MAIN_TEST as a gated wrapper (v4dev is not edited)

This package imports nothing at package import time: the launcher must export CUDA_VISIBLE_DEVICES before v4train /
ppsi_env / torch are imported (FL-02 finding), so `python -B -m v4int.launch_central` has to reach its own argument
parsing first. Every other module imports v4train (which imports ppsi_env first) before any snapshot module.

DATA-05 (DECISION_LOG 35-36): the providers, the cache and the launcher take a REQUIRED `history_policy`
(v4hist.policy: USER_HISTORY_L50 locked, SESSION_L50 the v1 diagnostic); see v4hist.windows.
"""
__version__ = "INT-01.1+DATA-05.1"

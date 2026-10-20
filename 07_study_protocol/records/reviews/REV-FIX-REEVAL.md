# REV-FIX-REEVAL (independent read-only review, 2026-09-30)

**Verdict: PASS** (0 BLOCKER, 0 MAJOR, 3 MINOR)

Reviewed: code/v4int/reeval.py (sha256 adfba078... confirmed), tests_int/test_gate_b_reeval.py, tests_int/test_fix_reeval_real_records.py, fixtures (all 6 byte-identical to company/evidence/reeval_records, cmp).

## Checks
1. **check_source_binding (reeval.py:401-442) is sound.**
   - config_sha256 is recomputed from RUN_START.config with the same expression as RunConfig.sha256 (v4train/recipe.py:438-440 versus reeval.py:393-398). It is compared to start.config_sha256, and check_central_meta (reeval.py:571-580) ties that to the checkpoint meta. So config.key is checkpoint-bound and cannot be edited alone.
   - config.key.manifest_registry_self_sha256 must equal the extension's self hash. A run whose comparison key names the base REGISTRY_V2 or the other source's extension refuses. Covered by refusal cases, both re-hashed and not re-hashed, and by test_cross_source_records_refuse.
   - The trainer view is rebuilt from the contracts extension, never from the record. It must pass check_trainer_view. Its self hash and file sha256 (recomputed with the shim's serialization) must equal key_base and RUN_START.manifest_registry. A field copied from the record without verification would fail here.
   - A `None` computed value is refused (`or got is None`, reeval.py:436). An exception in view derivation becomes ReevalRefused.
   - Residual, not a defect: config.key's extension hash is a launcher-declared field, and the only proof that the run actually trained on the extension data is the trainer-written view hash plus the config/checkpoint binding. The design cannot do better without the server data. Forging it needs a matching checkpoint meta, and the checkpoint sha is pinned.
2. **central_train_binding and the evaluated_test_role refusal are correct (reeval.py:314-326, 1090-1107).**
   - The fallback accepts RUN_START.manifest_registry.bindings.train only when its manifest_id equals train_provider.manifest_id.
   - The eval-only role now refuses with "none recorded" when no content_sha256 is available. It still refuses on a mismatch against the key train manifest.
   - This turns a fail-open path into a fail-closed one, so the TEST gates are stricter, not weaker. The `"train" in pair` guard keeps the MAIN-pair path unchanged.
   - Tested by test_central_train_binding_reads_the_real_record and test_eval_only_test_role_fails_closed_without_a_train_binding.
3. **No other check weakened.** The diff touched only the registry binding and the train binding. Checkpoint sha, the extension additions-only verify and pair, roles (TRUE_COLD refused), checkpoint kind/mark and the PRE init record are intact. The still-refuse test on the real shape passes (meta config_sha256, --expect-sha256, TRUE_COLD). `v4eval.evaluator_code_sha256()` computed now = 61b64e61c1a2e4f5a6fecd42f0ad3b98de26562d13a07758573cc9496b8d962e (unchanged).
4. **Tests.**
   - Run 1: `19 passed, 9 xfailed`.
   - `--runxfail -k test_nc_`: all 9 selected NCs fail. The 4 new NCs fail at the intended assertions: the old key_base rule refuses the real records, binding unchecked passes the refusals, config-key-only misses the view refusals, and the provider-only binding finds none. The 5 older GATE-B NCs also fail, and I checked the PRE-init-record one fails at its "refused is not None" assertion, the stated reason.
   - The real-record dry runs (C_FULL and PRE, MAIN PB/CB and the panels) assert the recorded binding values against the LAUNCH/SHIM records. The `rehash` helper is written independently of the code under test.

## Findings
1. **MINOR:** the view file hash relies on a hard-coded serialization, `json.dumps(view, indent=1, sort_keys=True)+"\n"` (reeval.py:418), duplicated from the shim's writer. It matches the real records, so it is verified today. Drift in the shim writer would produce a refusal (fail-closed), not a pass. Consider importing the writer's serializer.
2. **MINOR:** the test-only refusal table builds several cases through a convoluted `rehash(s) if ....update(...) is None else None` lambda (test_fix_reeval_real_records.py:~120). It works but is fragile to read.
3. **MINOR (PENDING, acknowledged by the handoff):** the real checkpoint meta and PRETRAIN_FOR_FL_INIT.json, and the eval-provider fingerprint reproduction, are still server-only. The dry runs use synthetic checkpoint and init records carrying the real hashes. The server dry run (handoff "Next" item 1) remains the closing evidence.

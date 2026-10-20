# TRAIN-05 + TRAIN-03R1 — independent Opus review (verbatim; fresh read-only agent; received 2026-09-27 ~23:14Z)

**VERDICT: ACCEPT_WITH_FIXES**
02B_GRU invariant holds: RunConfig recipe fields + `recipe_registry` path string come from launch_central `_recipe_values` (unchanged); resume compares only `recipe_sha256`; C1 fields are the old code's. No trainer change needed for any fix below.

FINDINGS
1. MAJOR tuned_recipes.py:193-198,207,214-215 — C2 is picked from the stage file's recorded `selection.configs`; `verify_stages` (86-102) only checks `status`/`tuned`/`carry` against the recomputed selection, so C2's validity/mean/n_params/carry are trusted on the file self-hash alone (violates the F1 rule; the load re-check reuses the same data so it cannot catch it). Fix: return the recomputed `sel` from `verify_stages`, require recorded `configs` == recomputed (canonical), derive C2 from `sel["configs"]`.
2. MAJOR unseal.py:200-217 — a family with neither C2 nor a `no_c2` record passes (e.g. `register-02b --family SASREC` without `--with-c2` → 6 runs unseal, SASREC C2 silently dropped). Fix: per family require `c2_id_of(f) in recipes` XOR `f in current["no_c2"]`; also require part-2's pin to contain `02B_SASREC_C2` when it exists.
3. MAJOR selection.py:295-297 — parameter counts are caller-supplied (`parameter_counts=`), so the tie-break can be spoofed/misread. Fix: read `shared_numel` from each candidate's RUN_START.json (unsealed, trainer-written) and require agreement across both seeds (or carry it in the unseal token record); refuse caller input.
4. MAJOR (ops) D128 vs code_p3b — code_p3 cannot read the new files (rejects G1B `_2`; its old re-verify compares `trial.hp_id` to `source_hp_id`, rejecting the C2 entry), and code_p3b cannot resume a code_p3-started run (`trainer_code_sha256` hashes every `v4train/*.py`). Fix: register C2 / write G1B 1R only AFTER both D128 seeds are final; record in RESUME_HERE.
5. MINOR tuned_recipes.py:89-92 — `FABLE_TRIGGER` stages are now registrable even without `--with-c2` and the error text still says "not registrable". Allow only on the with_c2/c2_only path (so `no_c2` is always recorded); fix the message.
6. MINOR tuned_recipes.py:333-337 — the new full-equality load check recomputes `run_evidence` with `stored=None` (absolute run_dir resolution; loses the R3 suffix tolerance). Build the expected entry with the stored evidence.
7. MINOR unseal.py:189 — pins may repeat; require strictly increasing versions and part-1 pin == v1 (the 63704173… entry).
8. MINOR selection.py:303-309 — the diagnostic `decide()` exception (NeedsFable/SelectionError) aborts G2; catch and report. Also filter primary rows by the MAIN_VALIDATION eval manifest so diagnostic rows passed by mistake are refused.
9. MINOR tests — no resume-across-part-1R test under `_2`; the F4 test replaces all of `L.run`, so the real launch dict is never scanned (factor out the launch-dict builder and scan its output).
10. MINOR flaky seal test (test_train03.py:82, dup test_train03_launcher.py:96): the `f"{v:.6f}"` substring can match inside timing floats (`step_wall_s`, `data_wait_s`) that vary per run — false positives only (cannot mask a leak) but invites rerun-until-green. Fix: whole-number match `(?<![\d.])form(?!\d)` or parse JSON and compare numeric leaves excluding the timing keys.
TRAIN-03R1 otherwise sound: the import fix works; the F2 resume test is genuinely `test_overrides=False`.

D128: the running D128 runs are unaffected (registry/G1B read only at start-up, continue in-process) — but once C2/1R are registered, neither tree can resume/relaunch them (finding 4).

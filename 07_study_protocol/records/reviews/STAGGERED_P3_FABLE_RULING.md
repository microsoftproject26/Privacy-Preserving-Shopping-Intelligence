# Staggered P3 start + S1 continuation — Fable 5.1 ruling (verbatim; received 2026-09-27; owner asked Ops to consult Fable)

Fable ruling on the staggered P3 start: APPROVE_WITH_CONDITIONS. S1 continuation: ALLOWED, same knots.

Basis: under STRICT, numerics depend on seed, permutation and kernels, not on timing or co-scheduling (as in my GRU full-run TWO_GPU finding); each family's recipe depends only on its own P2; so a staggered start is fair iff the seal holds.

Conditions:
1. Seal by construction, not by discipline: the trainer writes MAIN_VALIDATION mark rows and any metric-bearing log into a `sealed/` run subdirectory; STATUS/EVENTS and the console carry only mark index, timestamps, step rate, loss-finite flag and the boolean STILL_IMPROVING at 6.0 — Ops confirms from the code (one cited line) before launching GRU P3 that no metric value reaches an unsealed file.
2. Unsealing (the G2 script's first read) requires: all four contest runs complete; G1B_APPROVED.json holding BOTH 02b self-hashes (part-1 and part-2) with a test that the GRU entry is byte-identical across them; the four RunConfig hashes matching the pinned recipes.
3. G1b-part-1 packet contains GRU only; SASRec's P2 table comes in part-2. I pin each part's hash.
4. Continuation launches are automatic on the STILL_IMPROVING flag per the registered C3 rule (no Fable call; logged) — no human reads a value to decide it.
5. Disk projection for the overlap (GRU P3 checkpoints at 5.0/5.5/6.0 plus SASRec P2) stays above the 8 GiB guard; otherwise serialize.
6. GRU seeds sequential on GPU0 and SASRec seeds parallel is fine; the bitwise-STRICT trigger stands.

S1 continuation: allowed with the registered C3 knots ((7.5, 0.1), (8.0, 0.01)). Both S0 and S1 end at (6.0, 0.01), so the phase boundary and `s_of(6.0)` are identical; the continuation is a second constant-then-cooldown cycle, schedule-neutral, and disabling it for S1 would give the two families different rules. Disclose that under a cooldown schedule the 6.0 flag is expected to fire often; rows stay PRACTICAL_BEST_EXTENDED, never in T1 selection, only in C_BEST; CKPT_AVG stays 5.0/5.5/6.0 only; 8.0 remains the ceiling.

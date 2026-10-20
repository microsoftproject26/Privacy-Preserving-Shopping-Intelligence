# PREG1 amendment 07 — addendum §B12: DP S record per PRE seed (REGISTERED 2026-10-02, DL 357; Ops + Fable concurrence)

- **Trigger (MEASURED):** Wave-1 DP8_WARM_FROZEN s2027 and FA_1024 s2027 on S3-Beauty and S3-Sports REFUSED, by design. The single S
  record is bound (init_digest) to θ_pre of PRE s2026, while each seed's warm run starts from its own PRE seed. No value was read.
- **Rule.** S_CAL runs once PER PRE SEED: S_CAL s<S> from θ_pre s<S>, writing runs_bench/<dataset>/dp/S_RECORD_<VARIANT>_s<S>.json, each
  bound by init_digest. FA_1024 s<S> and DP8 s<S> bind their own seed's record.
  - Unchanged: the S rule (B7 R4: median client-delta L2 over rounds 1–20 of a z = 0 S_CAL run with the same configuration on the
    FL clients), z = 0.5642, q = 2⁻⁷ exactly, T = 384, δ = 5e-6, m = round(q·N).
  - The s2026 chain is unchanged; its record is renamed to the _s2026 form in the LOCK only.
  - Both S values are printed.
- **Disclosed:** on the benchmarks θ_pre is per seed (two PRE seeds), unlike REES46's single θ_pre, so the warm-arm seed spread
  includes the pretraining seed.
- Applies to every Wave-1 dataset. The refused s2027 jobs are re-run after the runner fix. Nothing else changes.

# Pre-G1 deliberations — scientific synthesis (Operations Director)

**Inputs:**
- `PREG1_OPUS_POSITION.md`, written first and withheld from Fable;
- `PREG1_FABLE_POSITION.md`, with Fable's T1–T3 ruling and its step-B ruling on the Opus response;
- the owner's directives (DECISION_LOG 18).

## T1 — CENTRAL_ARCHITECTURE_SELECTION_POPULATION
- **Agreed:**
  - the cohort contest is the selection instrument, and C for downstream is always the cohort central run;
  - a predeclared any-of trigger for a full-TRAIN tie-break: |Δ̄| < 0.0020; sign-inconsistent or |δ_s| ≤ 0.0002;
    |Δ̄| < 2 × the largest within-family seed gap; or the bootstrap CI containing 0;
  - the tie-break is a single-factor design: only the population changes; the recipe, the cohort-calibrated peak, 6.0
    EFE, "end", C3 and the frozen evaluator stay the same. It is evaluated on the full warm VALIDATION (642,228
    users);
  - tie-break rows are HISTORICAL_REFERENCE and never C_MATCHED.
- **Disagreed:** the number of tie-break seeds. Opus proposed 1, to save compute; Fable proposed 2, for replication
  (PKG 08 §5).
- **Why it existed:** a resource-versus-evidence trade-off. Both positions are defensible.
- **Resolution (Fable step B, accepting the Opus compromise):** a STAGED tie-break.
  - Stage 1 runs seed 2026 for both families (cap 65 GPU-h). It is decisive iff |Δ_full,2026| ≥ 0.0040 AND its paired
    bootstrap CI excludes 0.
  - Otherwise stage 2 runs seed 2027 (total cap 125 GPU-h), and the two-seed rule applies.
  - This serves both the owner's instruction to save compute and the replication requirement.
- **Remaining open item:** the EQUIVALENT outcome.
  - Fable's GRU-on-cost fallback amends PKG 05 §B and needs the owner's ratification.
  - Until then, EQUIVALENT means HOLD and ask the owner.
  - In an autonomous run with the owner unreachable, a proxy GRU decision permits only reversible calibration, labelled
    PROXY_DECISION_PENDING_RATIFICATION.
- **Disclosure:** the 0.000869 full-data seed gap is SASRec-only; GRU's full-data seed spread under the common schedule
  is unmeasured.

## T2 — LR exposure point
- **Agreed independently:** `exposure_point = "end"` for every unit of work (a central update, an FA/FP/PF round, an
  LO step; the PF personal LR follows).
- **Why it matters:** "start" wastes FA/FP/PF round 0 and every LO client's first step, which is 1 of 6 updates for a
  median client, and it never reaches the 0.01 cooldown floor for LO.
- **Fable's pins (1)–(7) are adopted as G1 blocking tests:**
  1. the key is explicit in every config, and a launcher refuses a missing key;
  2. a hashed planned round-LR table with min > 0 and the endpoint at 0.01 × peak;
  3. the FL and central s are equal bit-for-bit on the knots plus 10,000 grid points;
  4. FL round LRs equal central `lr_for_update` on a one-client fixture;
  5. LO: lr(0) > 0 and lr(T−1) = 0.01 × peak for T ∈ {1, 2, 6, 7, 300, 10⁴};
  6. T26/T27 re-run under "end";
  7. continuation only after a STILL_IMPROVING receipt.
- **Disclosed:** tiny LO clients skip the warmup segment.

## T3 — Frozen evaluation numerics
- **Agreed independently:**
  - a single scoring block (score_block = K = 158,486), bit-identical to the logits path;
  - eval_batch = 1,024 in manifest order;
  - fully deterministic GPU flags;
  - ONNX is device parity only, never a headline row.
- **Fable added:**
  - class_chunk 32,768 (recorded);
  - cache_bytes 1 GiB;
  - the comparison key gains eval_score_block, eval_batch and eval_device_class = "A16-FP32-DET";
  - a tie diagnostic per row;
  - a blocking G1 GPU receipt: pass-2 bitwise, single block equals logits bitwise, repeated M02 rank vectors
    identical, plus time and memory;
  - if memory contention appears, only eval_batch may change, never score_block.

## Consequences for the plan
- The G1 blockers now include: pins T2(1)–(7), the T3 GPU receipt, and the tie-break manifest spec (M_FULL_TRAIN plus
  the full warm VALIDATION manifest), which is built only if the trigger fires.
- **Disk:** a tie-break keeps strictly-best plus endpoint weights only, and these are deleted after ARCHITECTURE_LOCK.
  The live peak must fit the disk plan's guards (refuse below the 8 GiB floor, pause below 10 GiB).

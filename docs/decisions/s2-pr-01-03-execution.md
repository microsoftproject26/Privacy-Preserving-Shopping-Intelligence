# S2-PR-01/02/03 — bundled execution and measurement decisions

- Date: 2026-09-10
- Status: ACCEPTED_FOR_EXECUTION_UNDER_PROJECT_OWNER_DELEGATION
- Decision authority: Ahmed Abdelhamed's request to plan and execute #35/#36/#37 together, with necessary decisions documented in the repository.
- Planning author: ChatGPT; implementation executor: Claude Code.
- Scope: the local implementation bundle, technical runtime benchmark and initial T1 convergence-tool policy. Not final scientific FedAvg/R2A or CP-FL approval.
- Approval provenance: owner instructions in the project conversation. No external approval permalink was supplied. This document does not assert individual new approvals by Eid Abdelrihem or Ahmed Sherif.

## Existing authority preserved

Eid's accepted ADR-001 and G1 remain unchanged. T1 regime comparison uses macro MRR@20 on the category-change slice, with micro alongside. The vocabulary and already-dense T1 labels remain 588 classes / codes 0–587. No cohort is rebuilt, no second modulo-4 filter is applied, and no TEST outcome informs this work. The completed trainer, sampler, smoke and evaluator remain unchanged.

## D1 — one bundle, with a real local dependency barrier

Issue #37 originally requires #35 to be Done. For the requested single implementation session, a narrow scheduling exception replaces the inter-PR wait with #35's completed **local** technical gate: real scale trials, repeated chosen point, verified hashes and a PASS local summary. #37 must not run before that gate.

The later human PR must cross-reference this exception in #35/#37. Before merge, local readiness is not GitHub closure. No technical acceptance requirement or real-data evidence is waived. A failed #35 leaves #37 unexecuted; #36 is independent and may finish.

Alternative rejected: implementing/profiling all three concurrently against unfinished code. That makes measurements and failures hard to attribute.

## D2 — controlled real-data runtime workload

Benchmark virtual populations are nested sets of 1,000, 2,500 and 5,000 unique existing opaque client IDs. Eligible benchmark users have at least 32 real T1 TRAIN examples; every selected client uses exactly 32 examples in batches of 8. There are 20 participants per round and six rounds; the first is excluded only from steady-state timing summaries.

This selection changes the **benchmark population only**, not accepted C1 membership, final FL eligibility or scientific evaluation. Report excluded-from-benchmark counts. It intentionally does not characterize sparse clients or every non-IID production workload. No inference about TRAIN event-history length is made from TaskExample counts.

Rationale: fixed per-client work prevents runtime differences from merely reflecting different row counts. Rejected: silently repeating small users' rows, substituting synthetic users, or treating an arbitrary Python list as thousands of registered Flower clients.

## D3 — reuse the non-scientific #20 representation

Use the existing zero-history 588-output stub, seed 13, one local epoch, SGD LR 0.02/momentum 0, and fixed 256-row validation diagnostics. No architecture or learning-rate search. Real examples and user boundaries are retained; no claim about final sequential-model quality is made.

The source data for real sequential histories is not supplied by this bundle. The raw-event-derived distribution limitation from #19 remains visible and is not silently waived for CP-FL.

## D4 — population, participation and concurrency are separate

Register N actual virtual Flower SuperNodes, select 20 logical users per round, and use a separately bounded backend worker pool. The scale matrix fixes C=2 when hardware admission allows, otherwise C=1. Profiling considers C=1/2/4 under preregistered guards.

A stable ephemeral node map is one-to-one inside each simulation. Repeatability compares logical user selection, not incidental transport node IDs. Report actual participation coverage; dormant virtual users are not claimed to have trained.

## D5 — resource and profiling rules

The config predeclares conservative free-memory/disk, trial/round timeout, and process-tree RSS guards. They are operating safeguards, not measured capability claims. Child driver and descendants are measured together at 100 ms intervals. The maximum simultaneous summed RSS is reported as a **sampled RSS peak**; shared pages may be double-counted, so it is not unique physical RAM.

Round 1 is warm-up; rounds 2–6 supply time samples. p50 is the median; p95 uses nearest-rank `ceil(.95*n)-1`. Keep observations, failures and sample counts. Do not require byte-identical wall-time/RSS evidence.

Normal runtime is the twice-stable point with lowest warm-round p50; low-memory runtime is the twice-stable point with lowest conservative peak tree RSS. Ties prefer lower concurrency. These are recommendations for the measured workload/hardware only.

## D6 — T1 convergence-tool v1 semantics

- Self threshold: 95% of the **completed run's** best finite VALIDATION score.
- Self event: three consecutive scheduled qualifying observations, inclusive threshold. Record streak start and third-observation confirmation separately.
- This is retrospective analysis, not a directly deployable online early-stopping criterion.
- R1 target: first observed score reaching 90% of an identity-matched R1 VALIDATION headline value; one qualifying observation.
- Use the existing accepted T1 macro/category-change metric. No invented multi-task aggregate or ratios for LogLoss/Brier.
- Cadence is explicit. Missing scheduled observations, nulls or nonfinite observations break streaks; duplicate/decreasing rounds fail. No interpolation or post-hoc epsilon.
- Missing/zero R1, zero best, short/empty/never-crossed curves have explicit statuses.

The scalar numerical primitive does not authenticate a real R1. The project wrapper validates existing result schema, complete compatibility tuple, metric/split/seed identity and source hashes before supplying the scalar.

## D7 — bytes are consumed here, instrumented in #50

Current #36 readiness comment and #50 scope put actual send/receive instrumentation in #50. This bundle does not duplicate it. #36 consumes measured application model-payload upload/download totals with source/run identities, when available.

Sum training rounds 1 through the selected boundary inclusive. Initial model download for training is included in round 1; server-only initial validation is outside the communication interval. Self convergence uses the confirmation round; R1 target uses first crossing. Measured retry payloads remain included in their owning round.

Missing bytes are pending/incomplete, never zero. Tensor parameter sizes are not accepted as measured communication. Hand-worked byte fixtures are explicitly FIXTURE_PROOF, not actual Flower measurements.

## D8 — scientific and completion limits

#35/#37 measure this infrastructure workload; #36 proves deterministic curve/byte logic. This does not close #43, #50, #54, #55 or the research MVP. It does not provide formal privacy guarantees. Final scientific R2A participation, optimization and measurement decisions remain separately traceable.

## Evidence to fill from actual execution

The executor must link generated scale/convergence/profile evidence in the associated task reports and artifact registry. Do not write measured PASS values, machine capacities or runtimes into this decision document before executing them. Keep this decision identity fixed during runs to avoid evidence-hash cycles.

# Federated Runtime Scaling and Profiling (S2-PR-01 / #35 and S2-PR-03 / #37)

This is a **runtime** report. It says how a fixed federated workload behaves on one
machine as the declared client population grows. It is not a scientific result, not an
R2A finding, and not a capacity guarantee for another model, dataset or device.

Every number below was produced by `scripts/federated/fl_runtime_benchmark.py`, whose
exact executed source files are pinned per trial in `source_snapshot`, and is
reproduced in `docs/evidence/s2-pr-01/scale_summary.v1.json` and
`docs/evidence/s2-pr-03/profile_summary.v1.json`.

## 1. The three counts are not the same number

| Symbol | Meaning | Value used |
|---|---|---|
| `N` | declared population, real virtual Flower SuperNodes registered | 1,000 / 2,500 / 5,000 |
| `M` | participants per round, clients that receive a train message | 20 |
| `C` | backend concurrency, the CPU cap given to the Ray backend | 2 for the scale matrix |

`run_simulation` is called with `num_supernodes=N`, never with `M`. All `N` users exist
in the hashed population manifest; only `M` train in any round.

**Coverage is reported, not implied.** In the accepted `N=5,000` trial there were 120 fit
calls across six rounds, reaching 118 unique users out of 5,000 declared. No claim is made
that 5,000 users trained.

## 2. Workload and the explicit benchmark subset

The workload is the frozen non-scientific #20 stub: zero semantic history, 588 outputs,
embedding 4, hidden 8, SGD at learning rate 0.02, one local epoch, batch size 8, exactly
32 rows per client, six rounds, seed 13, CPU with one Torch thread per worker.

The benchmark pool contains only clients with at least 32 real T1 TRAIN TaskExamples, so
each selected client performs exactly four batches and population size is the only thing
that changes between points.

| Benchmark pool | Clients |
|---|---|
| With at least 32 T1 TRAIN examples (eligible) | 29,740 |
| Below 32, excluded from the benchmark only | 66,250 |

Those 66,250 clients are **not** removed from C1, from eligibility, or from any scientific
evaluation. They are outside this runtime benchmark and nothing else.

TaskExample counts before the benchmark filter, over 95,990 clients with T1 TRAIN rows.
These are **TaskExample counts, not raw-event history lengths**; the original #19 raw-event
distribution is still unavailable and is not inferred here.

| min | p50 | mean | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 1 | 19 | 32.4 | 71 | 104 | 213 | 1,216 |

## 3. Measured hardware

| Property | Value |
|---|---|
| Logical / physical CPUs | 20 / 14 |
| Total RAM | 15.73 GiB |
| Available RAM at the concurrency decision | 6.98 GiB |
| OS | Windows 11 (10.0.26200) |
| Device | CPU only; GPU memory is `NOT_APPLICABLE`, not a measured zero |

Concurrency was frozen at `C=2` before any run, because the machine met the declared rule
of at least 2 logical CPUs, 12 GiB total RAM and 6 GiB available RAM. The same `C` was used
for all three scale points.

## 4. Scale points

Warm-round statistics cover rounds 2 to 6. Round 1 stays in the raw records but is excluded
from every steady-state summary, because it absorbs first-use work: in the accepted trial
round 1 took 7.42 s against 0.68 s for round 6.

| N | Status | Warm p50 (s) | Warm p95 (s) | Sampled peak tree RSS (GiB) |
|---|---|---|---|---|
| 1,000 | STABLE | 0.525 | 0.540 | 1.775 |
| 2,500 | STABLE | 0.618 | 0.662 | 1.779 |
| 5,000 | STABLE | 0.644 | 0.688 | 1.784 |
| 5,000 (confirmation) | STABLE | 0.635 | 0.763 | 1.785 |

Every point registered exactly `N` nodes, selected exactly 20 clients in each of six rounds,
contributed exactly 640 examples per round, passed the independent FedAvg oracle within
`1e-6`, produced seven server evaluations for rounds 0 to 6, and changed the global model.
No point hit a resource guard, a timeout, a client retry or an incomplete memory
measurement, so no row is missing from the table above.

There were **no failed or skipped scale points** in this run. Had one failed a resource
guard, escalation to a larger population would have stopped and the remaining rows would
appear as `NOT_ATTEMPTED_SAFETY` rather than silently disappearing.

Memory grew very little between 1,000 and 5,000 declared nodes, about 10 MiB across a
fivefold population increase. Round time grew from 0.525 s to 0.644 s over the same range.
Neither figure should be extrapolated past 5,000.

## 5. Repeatability of the chosen point

`N=5,000` was repeated once in a fresh subprocess with the same population, logical samples,
data, initialization, workload and concurrency.

| Check | Result |
|---|---|
| Round selection digests identical | yes |
| Contributing examples per round identical | yes |
| Final parameter maximum absolute difference | 0.0 |
| Within `atol=1e-6`, `rtol=0` | yes |
| Final state byte-exact as well | yes |

Timing and RSS are **not** compared for equality. CPU scheduling changes measured time
without changing the experiment.

## 6. Stage timings

Measured with `time.perf_counter_ns()` inside the strategy and the client callback. Figures
below are from round 6 of the accepted `N=5,000`, `C=2` trial.

| Stage | Meaning | Seconds |
|---|---|---|
| `server_entry_to_node_binding_seconds` | ServerApp callback entry through node binding, not whole-simulation startup | 0.057 |
| `data_load_seconds` | client slice read, filter and batch build (p50) | 0.003 |
| `local_fit_seconds` | around `adapter.fit` on each client (p50) | 0.015 |
| `dispatch_to_replies_seconds` | configure completion to aggregate entry | 0.550 |
| `aggregation_seconds` | aggregation plus the required oracle checks | 0.065 |
| `server_evaluation_seconds` | the server diagnostic evaluation | 0.026 |
| `round_wall_seconds` | configure start to that round's evaluation end | 0.679 |

`dispatch_to_replies_seconds` covers transport, scheduling and client processing together.
It is not the sum of client fit times and it is not pure scheduler overhead.

Client call timestamps give an observed span of 0.315 s for the twenty clients in that
round. That is an observation about overlap on this one host. The configured cap `C` is
reported separately and no claim is made that both workers always ran at once.

## 7. Concurrency profile

| C | Status | Pooled warm p50 (s) | Max trial tree RSS (GiB) | Trials |
|---|---|---|---|---|
| 1 | TWICE_STABLE | 0.884 | 1.331 | two fresh trials |
| 2 | TWICE_STABLE | 0.639 | 1.785 | reused from the scale matrix |
| 4 | NOT_ATTEMPTED_SAFETY | not measured | not measured | none |

`C=4` was refused because the preflight requires 16 GiB total RAM and the machine has
15.73 GiB. The row stays visible rather than being dropped.

Two admissible points were each measured twice, so a comparative recommendation is
supported. The `C=2` trials were reused from the scale matrix because their workload,
inputs, source identities and config identity match exactly; identical expensive trials are
not repeated just to change a heading.

| Recommendation | Choice | Why |
|---|---|---|
| Normal | `C=2` | lowest pooled warm p50, 0.639 s against 0.884 s |
| Low memory | `C=1` | lowest maximum trial tree RSS, 1.331 GiB against 1.785 GiB |

These differ because the trade-off is real: the second worker buys about 28% lower round
time for roughly 465 MiB more sampled resident memory. The distinction was not forced.

## 8. Safety limits actually applied

These are conservative plan decisions, not statements about what this hardware can do.

| Guard | Threshold |
|---|---|
| Sampled tree RSS ceiling | 70% of total RAM |
| Available RAM floor during a trial | max(2 GiB, 10% of total) |
| Trial wall time | 600 s |
| Train round wait | 180 s |
| Node registration wait | 90 s |
| Free disk on repository and temp volumes | 3 GiB each |
| Whole-run real execution budget | 5,400 s |

The accepted trial was sampled 134 times at a 100 ms interval with a largest observed gap
of 0.141 s, no inaccessible process, and a lowest observed system availability of 3.93 GiB.
Each trial ran in its own supervised subprocess with single-threaded BLAS and Polars
settings applied to the child only. Cleanup terminates verified owned processes by
`(pid, create_time)`; it never kills by image name and never touches an unrelated Ray
cluster or the user's editor.

## 9. Attempt history before the accepted run

The measurements above come from one six-trial run. Two earlier attempts did not
produce accepted measurements, and both are recorded here because the summaries no
longer show them.

**Capacity refusal.** An attempt made while the machine had about 3.3 GiB of available
RAM was refused by the admission gate before any trial started. The gate requires at
least 4 GiB available for the single-worker fallback, so the run ended with
`BLOCKED_CAPACITY` and `local_gate=BLOCKED`, and profiling reported
`NOT_RUN_DEPENDENCY`. The threshold was not lowered. The attempt was retried only after
RAM was freed.

**A repetition rejected by an over-strict completeness rule.** In the first run after
RAM was freed, the scale matrix passed but the `C=1` repetition
`n5000-m20-c1-rep2` was recorded as `FAILED_INCOMPLETE_MEASUREMENT`, leaving profiling
`PARTIAL`. The cause was in the monitor, not the trial: process discovery counted a
`NoSuchProcess` raised while registering a process that had already exited in the same
bucket as `AccessDenied`. A normal exit is not an unreadable process, so a benign race
was being reported as an unverifiable memory profile.

**Correction and accepted rerun.** The monitor now separates the two conditions. Only a
genuine `AccessDenied` during discovery marks a measurement incomplete; a normal exit
does not. Because this is measurement code, the whole six-trial plan was rerun rather
than reusing the earlier numbers, and all six trials succeeded.

**Availability of the earlier attempts.** The artifacts of both earlier attempts are
**not** retained. Each rerun rewrote `scale_summary.v1.json`, `profile_summary.v1.json`
and the per-trial evidence directories in place, so the `BLOCKED_CAPACITY` summary and
the rejected `n5000-m20-c1-rep2` measurement no longer exist on disk. The private
`superseded-results` archive was never created, because the result-retirement path did
not fire during these reruns. This section is the only surviving record of those two
attempts, and it is written from the run output rather than from preserved artifacts.

## 10. What these numbers cannot support

- No R2A scientific claim, no model quality claim, no convergence claim. The diagnostic
  cross entropy moved from 6.443 to 6.333 across six rounds, which shows the pipeline
  trains and aggregates; it says nothing about model quality.
- No capacity statement for a different model, a larger representation, or another machine.
- The memory figure is a **sampled process-tree RSS** upper bound. Resident memory can
  double-count shared pages, so it is neither unique physical RAM nor an exact peak.
- A p95 over five or ten observations is descriptive, not a production tail bound.
- Configured concurrency is a cap, not a guarantee of simultaneous execution.
- Real sequential histories are still absent; the #19 raw-event distribution limitation
  remains open and is not waived here.

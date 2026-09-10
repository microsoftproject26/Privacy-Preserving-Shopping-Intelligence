# The S2-DS-01 pipeline, in the order it ran

These are the scripts that produced `0.3474`. They are here so the number can be
reproduced and, more usefully, so `S2-DS-05` and `S2-DS-06` reuse the same ladder,
evaluator and training loop rather than writing their own — two implementations of an
evaluation disagree by a small amount nobody can explain, and that has already cost this
project three wrong numbers.

## Order

| # | script | what it does | GPU |
|---|---|---|---|
| 1 | `prepare_data.py` | raw events + frozen T1 examples → cached history windows | no |
| 2 | `client_events.py` | TRAIN events per client — the `ADR-001` strata definition | no |
| 3 | `calibrate.py` | **the gate**: reproduce the published baseline, then S2-SMOKE's number | yes |
| 4 | `ladder.py` | rungs 1–5, one change per rung | yes |
| 5 | `sweep.py` | a coordinate sweep, and what selection over many configs costs | yes |
| 6 | `probe_lr.py`, `probe_schedule.py` | following the one axis that moved | yes |
| 7 | `determinism.py` | is the same seed reproducible, and what does forcing it cost | yes |
| 8 | `finalize.py` | the final model on three seeds, and every handoff artifact | yes |
| — | `prepare_heads.py` | T2 and T3 windows, for `S2-DS-06` and `S2-DS-07` | no |

`train.py` is the shared training loop and scorer; the others import it.

**Step 3 is a hard gate.** If the recomputed transition-table baseline does not match the
published `0.3085`, the pipeline is reading different rows and nothing above it means
anything. It is an `assert`, not a `print` — a `print` there is how a baseline collapse from
`0.3085` to `0.1866` once survived long enough to inflate a model's apparent gain to
`+0.1454`.

## Data

Private files are not in this repository. Point `PPSI_DATA_ROOT` at a directory holding:

```
raw/        processed_raw_parquet_v1.parquet
examples/   task_examples_t1_{train,}_v1  ·  vocabulary_v1.json  ·  item_catalog_v1.parquet
protocol/   data_protocol_v1.json · config.json · cohort_manifest · excluded_sessions
```

Without it the scripts fall back to the layout of the machine that ran them. See
`docs/data-access.md`.

**No path here points at TEST**, and every artifact records `test_rows_read = 0`.

## Why the loop is not `LocalTrainerCore`

The core packs and sha256-hashes the whole state dict on **every step**. This model carries
a 1.2M-parameter embedding table and trains 6,081 steps an epoch, so that cost would
dominate the run. The model still satisfies `Phase1Model` and
`tests/models/test_session_gru.py` proves a core step works, so the federated lane is
unaffected — the contract is honoured, the loop is ours.

## Two things that will bite a reader who changes them

**The window is right-padded**, real events at columns `0 … lengths-1` with the decision
event at `lengths-1`. Built the natural way they come out left-padded, the encoder reads
only padding for every window shorter than the limit — most of them, since the median
history is 5 events — and MRR@20 falls to `0.4509`, *below* a baseline that learns nothing.
Every shape and dtype check still passes.

**Off-diagonal suppression applies to the slice only.** The baseline is evaluated
off-diagonal, so a model that is not is being compared unfairly. It is worth about
`+0.13` MRR: get it wrong and the result does not look like a bug, it looks like a finding.

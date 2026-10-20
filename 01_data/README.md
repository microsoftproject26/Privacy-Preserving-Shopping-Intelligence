# 01_data - the REES46 data pipeline

## Purpose

Everything between the raw public log and the frozen dataset that every experiment reads:
conversion and canonical events, the census that fixed the study population, the history policy
that crosses session boundaries, the feature statistics, the locked input schema, and the
manifests of every population band. The stage produces one immutable dataset,
`rees46_nextitem_v1`, plus the frozen inputs each later stage consumes.

## Experiments, in order

1. **Prototype data audit and cohort (August-September 2026).** The prototype stage converted
   the raw October 2019 file (42,448,764 events) to Parquet without rejecting a row, defined
   half-open UTC windows and a TRAIN-only cohort rule (388,789 clients), and froze the
   privacy-motivated client-side transforms: blake2b-64 hashed identifiers, price bands, capped
   log time gap. Descriptive. A data-freeze gate stopped one stale eligibility rule before any
   TEST row was read.
2. **Pilot-study frozen dataset (September 2026).** The pilot studies rebuilt sessions with an
   inactivity gap of 3,600 s, kept views only, fixed the target before catalogue filtering
   (158,486 products with TRAIN support of at least 5), and froze TRAIN / validation-select /
   validation-confirm / TEST windows on October-November 2019. The frozen dataset passed 82
   final-gate checks; its cold rebuild was identical. Descriptive.
3. **Main-study census and population (26-27 September 2026).** A census of the TRAIN window
   counted 4,804,933 users with 40,180,608 eligible examples; the study lead fixed the
   population as the pool of users with 5 to 100 eligible TRAIN examples (1,446,986 users,
   28,623,386 examples, 71.24% of the supervision), cut into MAIN (131,072), CALIB (32,768),
   the PRE band (1,217,610), two unseen panels and a feature-search band. Descriptive.
4. **History policy.** The cross-session history policy `USER_HISTORY_L50` (last 50 events,
   strictly causal) was implemented and checked exhaustively: 0 causality violations, 87% of
   validation rows see TRAIN-period events. Descriptive.
5. **Feature search (September 2026).** A registered screen on the feature-search band
   accepted the reduced schema S* (25 information channels): the worse model family lost at
   most 1.0% of client-macro MRR@20 by rule; measured drops were -0.90% (GRU) and -0.13%
   (SASRec). Validation.

## Code map

| Package | What it does | Origin |
|---|---|---|
| `code/ppsi/data` | Prototype data pipeline: canonical events, windows, cohorts, example builders | Prototype stage |
| `code/ppsi/features` | Privacy-motivated feature encoding: blake2b-64 hashing, price bands, capped log time gap | Prototype stage |
| `code/census_v2` | Census of the TRAIN window (`census02.py`) that fixed the population strata | Main study |
| `code/featstat_v1` | Feature statistics fitted once on the full TRAIN period (one federated-statistics round) | Main study |
| `code/v4hist` | History policy `USER_HISTORY_L50`: policy, stream, windows, census | Main study |
| `code/v4data` (+ `frozen_inputs/`), `v4data06`, `v4data08`, `v4data09`, `v4data10` | Dataset builds, manifests and frozen inputs of the locked data versions | Main study |
| `code/v4fs_data` | Data builds of the feature-search band | Main study |
| `code/v4trunc` | Truncated-history panel builders (warm truncation panel) | Main study |
| `code/g0_export`, `g0_*.py` | Export adapters and gate-0 scripts (server census, forecast, federated cost probe) | Main study |
| `scripts/prototype/data` | Prototype data scripts | Prototype stage |
| `tests/prototype/features`, `tests/research/tests_data*` | Feature tests (CPU) and research suites of the builds | Prototype / main study |

## How to run

```bash
uv run python 01_data/run.py --from-results   # rebuild this stage's tables and figures from saved results
uv run python 01_data/run.py --mode auto      # skip complete builds, resume where supported, else start
uv run python 01_data/run.py --force          # rebuild from scratch (old outputs are moved aside first)
```

Building the dataset needs the raw REES46 months under `PPSI_DATA_ROOT`'s source location and,
for the frozen populations, the original server manifests; see
[`../environment/README.md`](../environment/README.md) for the download and checksum and
[`../docs/prototype/data-source.md`](../docs/prototype/data-source.md) for the raw-file record.
`--from-results` needs neither data nor GPU.

## results/ and figures/

`results/` holds the curated data summaries of this stage (census counts, manifest tables,
feature-search decision records) and, under `results/runs/main_study/` and
`results/runs/closure_phase/`, the saved run records that `--from-results` reads back.
`figures/` holds the rebuilt audit and census figures.

## Known limitations

One store over two months: the findings are specific to one catalogue and one traffic pattern.
The TEST week contains Black Friday (29 November 2019); the shift is reported, not corrected.
Users outside the 5-100 pool never train and serve only as evaluation panels. The earlier GRU
and SASRec pilot studies had inspected a legacy TEST window, so all main-study TEST results are
retrospective (exposure-register entry 3).

# Public benchmark datasets (`ppsi.benchmarks`)

The deployment study is run on REES46, but the same recipe is also checked on public sequential-recommendation
benchmarks so its numbers can be compared with published work. `ppsi.benchmarks` converts those datasets into one
layout and provides the sampled metrics some papers report.

## Layout

Every converted dataset is a directory in the ml-1m style used by public SASRec benchmark releases:

```text
<dataset>/
  leave_one_out/train.csv     user_id, item_id, rating, datetime, weight  (every interaction except the holdout)
  leave_one_out/holdout.csv   the same columns, exactly one row per user (the item to predict)
  statistics.csv              users, items, interactions, sparsity, date range, train / holdout counts
  STATS.json                  conversion record (source row counts, leakage counts)
  SHA256SUMS                  checksums of every file above
```

`check_layout.verify_layout(<dataset>)` re-reads a directory and checks the columns, dtypes, one holdout per user,
that every holdout user has history, the statistics row, and how many holdout items already appear in the user's
own history (reported, not hidden).

## Converters

| Dataset | Module | Notes |
|---|---|---|
| Amazon Reviews 2023, 5-core (`Video_Games`, `Baby_Products`, `Beauty_and_Personal_Care`) | `amazon2023` | uses the official `last_out_w_his` split: train + valid rows become the history, the test row the holdout; the global-time `timestamp_w_his` split is not one holdout per user and is only diagnosed |
| MBHT Taobao / Tmall (KDD 2022, RecBole `.inter` files) | `mbht` | a session is a pseudo-user; a session's sequence is its longest augmented row plus that row's target; behaviour types go to `item_type.csv`; datetimes are synthetic positions |

```powershell
uv run python -m ppsi.benchmarks.amazon2023 --raw-root <raw> --out-root <out>
uv run python -m ppsi.benchmarks.mbht --mbht-root <MBHT_dataset> --out-root <out>
```

The raw files are downloaded separately from their official sources; nothing is committed here.

## Sampled metrics

The study itself ranks the target against the full catalogue. `sampled_metrics` exists only to compare with papers
that rank against a sample of negatives; every output carries the label
`SECONDARY / for comparison with sampled-metric papers only`. Schemes: `uniform`, `popularity`, `fixed` (the MBHT
lists of 100 popular items), and `expected`, the exact expectation over uniform negatives computed from the full rank
(Krichene & Rendle 2020).

## Tests

```powershell
uv run python -m pytest tests/benchmarks -q
```

The converters are tested on small synthetic fixtures and the metrics against brute-force enumeration; no real data
is read.

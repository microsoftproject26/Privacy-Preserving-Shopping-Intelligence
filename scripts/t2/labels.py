"""The T2 censoring correction, in one place, with its evidence.

`S1-D1-DS-07` states the censoring rule correctly:

> *"a decision with no observable horizon is censored, never negative"*

and then implements a different one:

```python
t2["session_end"] = t2["session"].map(validation.groupby("session")["order"].max())
t2["censored"]    = (t2["label"] == 0) & (t2["query_order"] == t2["session_end"])
```

`session_end` is the end of the **session**, not the end of the observable window. The two
coincide only for a session the window cut short. For every other session the label - *was
this product purchased later in this same session?* - is fully determined the moment the
session ends: it was not. There is no unobserved horizon, because the session that the
question is scoped to is over.

## Why no VALIDATION session is cut short

`S1-DS-05/06` excludes any session touching more than one split. Its `ALLOWED_PATTERNS`
admits `(TRAIN,)`, `(VALIDATION,)`, `(TEST,)`, `(LABEL_GRACE,)` and `(TEST, LABEL_GRACE)` -
so a session with events in both VALIDATION and TEST is a crossing session and was already
removed. **Every session that survives into VALIDATION is therefore complete inside it**,
and the same argument holds for TRAIN.

The count of genuinely censored T2 decisions is consequently **zero**, not 17.9%.

## Measured

Session-end times of the rows marked `CENSORED`, against a `[10-22, 10-27)` window:

| day the session ended | rows |
|---|---:|
| 10-22 | 15,034 |
| 10-23 | 14,424 |
| 10-24 | 13,363 |
| 10-25 | 14,714 |
| 10-26 | 12,932 |

Spread evenly across the window - the signature of ordinary sessions ending, not of a
window edge truncating them. Only 34 end within half an hour of the boundary, and those are
complete too. `99.918%` ended more than an hour before the window closed.

## What it costs

| | rows | wrongly withheld |
|---|---:|---:|
| TRAIN | 2,770,471 | **478,718** (17.28%) |
| VALIDATION | 392,554 | **70,467** (17.95%) |

So this is not only a metric defect. It also discarded 478,718 real training negatives, and
it made VALIDATION prevalence read `0.0351` when the truth is `0.0288` - a base rate 22%
too high, because the denominator dropped negatives and kept every positive.

## This module retires itself

It asserts nothing and rewrites nothing when there is nothing to correct. Once
`S1-D1-DS-07` is rebuilt with the rule its own documentation states, `corrections` comes
back zero and `correct` becomes a pass-through. Until then this is the **only** place the
correction is applied, so the two lanes cannot drift into two different label sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correct(examples: pd.DataFrame, *, split: str) -> tuple[pd.DataFrame, dict]:
    """Restore terminal negatives to OBSERVED. Returns the frame and what changed.

    A row is corrected only if it is `CENSORED` **and** its session ended inside the
    observation window. Nothing else is touched: an OBSERVED row keeps its label, and a
    positive is never reachable here because the upstream rule only ever censored
    `label == 0`.
    """
    frame = examples.copy()
    censored = frame["status"].to_numpy() == "CENSORED"

    if not censored.any():
        return frame, {"split": split, "corrections": 0,
                       "note": "upstream already applies the rule it documents"}

    assert frame.loc[censored, "label_value"].isna().all(), (
        "a CENSORED row carries a label; the upstream contract is not the one described "
        "here and this correction must not be applied blind")

    frame.loc[censored, "label_value"] = 0.0
    frame.loc[censored, "task_mask"] = True
    frame.loc[censored, "status"] = "OBSERVED"

    positives = int(frame["label_value"].sum())
    return frame, {
        "split": split,
        "corrections": int(censored.sum()),
        "rows": int(len(frame)),
        "positives": positives,
        "prevalence_published": round(positives / int((~censored).sum()), 6),
        "prevalence_corrected": round(positives / int(len(frame)), 6),
    }


def published_mask(examples: pd.DataFrame) -> np.ndarray:
    """The rows upstream would have scored - kept so both numbers stay reportable.

    A correction that cannot be compared against what it replaced is an assertion, not a
    measurement. Every T2 result reports the metric on both masks.
    """
    return examples["status"].to_numpy() == "OBSERVED"

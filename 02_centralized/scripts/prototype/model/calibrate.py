"""Rung 0 - the gate. Reproduce what we already know before believing anything new.

Two things are checked, in order, and neither is optional:

1. **The baseline.** The transition table is rebuilt here and must reproduce the numbers
   `S1-D1-DS-07` published - 0.3085 micro and 0.3144 macro on the category-change slice.
   A recomputed baseline that disagrees with the published one means the two are not
   scoring the same thing, and every comparison after it is meaningless. This is an
   `assert`, not a `print`: a `print` here is exactly how a baseline collapse from 0.3085
   to 0.1866 survived long enough to inflate a model's apparent gain to +0.1454.

2. **The model.** A replica of the S2-SMOKE configuration must land near the 0.3348 that
   run measured. The tolerance is not guessed - the replica runs on all three project
   seeds and the gate is whether 0.3348 sits inside the observed spread.

There is a third question this run settles as a side effect. S2-SMOKE converted timestamps
assuming nanoseconds, and under this pandas they are microseconds, so its time-gap channel
was almost certainly dead. Running the replica both with and without that channel says
which one reproduces 0.3348, and therefore whether the smoke was really a four-channel
model or a three-channel one.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from train import CATEGORIES, OUTPUT, PROJECT, Split, score, train

from ppsi.models.evaluation import (
    micro_and_macro,
    rank_of_truth,
    reciprocal_rank,
    transition_table,
)
from ppsi.models.session_gru import SessionGRUConfig

SEEDS = (13, 42, 2026)
PUBLISHED = json.loads(
    (
        PROJECT
        / "S1-D1-DS-07_T3_Protocol_and_Task_Examples"
        / "output"
        / "task_headroom_v1.proposed.json"
    ).read_text(encoding="utf-8")
)


def published_slice_micro() -> float:
    for row in PUBLISHED["t1"]["category_change_slice"]["baselines"]:
        if row["baseline"].startswith("TRAIN transition table"):
            return float(row["mrr@20"])
    raise KeyError("the published slice baseline is missing")


def baseline(train_split: Split, validation: Split) -> dict:
    """The one-step transition table, rebuilt from TRAIN decisions and scored.

    Built over **decision rows only**. Given every event instead, the -1 that marks "no
    later different item" competes as a category, and because it is what every session
    ends on it takes rank 1 away from real answers.
    """
    current_train = train_split.data["query_category"]
    assert current_train.max() < CATEGORIES, (
        "a TRAIN decision has an out-of-vocabulary current category, which cannot happen: "
        "the vocabulary was built from these very rows"
    )
    table = transition_table(current_train, train_split.data["target"], CATEGORIES)

    current = validation.current_category
    truth = validation.data["target"]
    changed = current != truth
    clients = validation.data["client"]

    overall = np.zeros(len(validation), dtype="int64")
    sliced = np.zeros(len(validation), dtype="int64")
    # A current category unseen in TRAIN has no row in the table; an all-zero score vector
    # is the honest representation of "this rule knows nothing here".
    empty = np.zeros(CATEGORIES, dtype="float64")
    for start in range(0, len(validation), 8192):
        rows = np.arange(start, min(start + 8192, len(validation)))
        chunk = current[rows]
        scores = np.where(chunk[:, None] >= 0, table[np.clip(chunk, 0, None)], empty[None, :])
        overall[rows] = rank_of_truth(scores, truth[rows])
        sliced[rows] = rank_of_truth(scores, truth[rows], suppress=chunk)

    micro_all, macro_all = micro_and_macro(reciprocal_rank(overall), clients)
    micro_slice, macro_slice = micro_and_macro(reciprocal_rank(sliced), clients, changed)
    return {
        "overall_micro": round(micro_all, 4),
        "overall_macro": round(macro_all, 4),
        "slice_micro": round(micro_slice, 4),
        "slice_macro": round(macro_slice, 4),
    }


def main() -> None:
    started = time.time()
    print("loading windows...")
    train_split, validation = Split.load("train"), Split.load("validation")
    print(f"  TRAIN {len(train_split):,} decisions | VALIDATION {len(validation):,}")

    print("\n" + "=" * 74)
    print("1. THE BASELINE - recomputed, and asserted against what upstream published")
    print("=" * 74)
    measured = baseline(train_split, validation)
    expected_micro = published_slice_micro()
    print(pd.DataFrame([measured]).to_string(index=False))
    print(f"\n  published slice micro : {expected_micro}")
    print(f"  recomputed            : {measured['slice_micro']}")
    assert abs(measured["slice_micro"] - expected_micro) < 0.002, (
        f"the recomputed transition-table baseline is {measured['slice_micro']} against a "
        f"published {expected_micro}; the two notebooks are not scoring the same rows and "
        "no comparison below would mean anything"
    )
    print("  MATCH - the pipeline scores the same rows upstream did")

    print("\n" + "=" * 74)
    print("2. THE MODEL - a replica of S2-SMOKE, on all three seeds")
    print("=" * 74)
    sample = np.sort(np.random.default_rng(13).choice(len(train_split), 1_200_000, replace=False))

    variants = {
        "smoke replica (category + event + gap)": SessionGRUConfig(
            channels=("category_id", "event_type_id"), use_gap=True
        ),
        "same, without the time gap": SessionGRUConfig(
            channels=("category_id", "event_type_id"), use_gap=False
        ),
    }

    rows, curves = [], {}
    for label, config in variants.items():
        for seed in SEEDS:
            model, curve, best = train(
                train_split,
                validation,
                config=config,
                seed=seed,
                rows=sample,
                batch_size=512,
                learning_rate=0.002,
                max_epochs=4,
                patience=4,
                label=f"{label} | seed {seed}",
            )
            result = score(model, validation, legacy_clamp=True, popularity=np.zeros(CATEGORIES))
            rows.append(
                {
                    "variant": label,
                    "seed": seed,
                    "best_epoch": best,
                    "slice_macro": result["slice_macro"],
                    "slice_micro": result["slice_micro"],
                    "overall_micro": result["overall_micro"],
                }
            )
            curves[f"{label} | {seed}"] = curve
            print(f"    -> slice macro {result['slice_macro']}  micro {result['slice_micro']}\n")

    frame = pd.DataFrame(rows)
    print("=" * 74)
    print(frame.to_string(index=False))
    print()
    summary = frame.groupby("variant")["slice_macro"].agg(["min", "mean", "max"]).round(4)
    print(summary.to_string())

    target = 0.3348
    print(f"\n  S2-SMOKE measured : {target}")
    verdict = {}
    for variant, row in summary.iterrows():
        inside = row["min"] - 0.002 <= target <= row["max"] + 0.002
        verdict[variant] = bool(inside)
        print(
            f"  {'REPRODUCES' if inside else 'does not  '}  {variant}"
            f"   [{row['min']:.4f}, {row['max']:.4f}]"
        )

    (OUTPUT / "calibration.json").write_text(
        json.dumps(
            {
                "baseline_recomputed": measured,
                "baseline_published_slice_micro": expected_micro,
                "runs": rows,
                "curves": curves,
                "smoke_target": target,
                "reproduces": verdict,
                "seed_spread": summary.to_dict(),
                "test_rows_read": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

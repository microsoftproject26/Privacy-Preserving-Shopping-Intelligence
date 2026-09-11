"""Rungs 1-5 - one change per rung, so a gain can be attributed to a specific change.

An ablation that moves two things teaches nothing about either. Each rung below differs
from the one above it in exactly one way, and every rung is kept in the report including
the ones that cost accuracy. A channel that loses is a finding; deleting it is how a
feature set ends up justified by nothing.

Rung 2 carries a stated prediction. In S2-SMOKE the product channel scored **-0.0031**
while its validation loss bottomed at epoch 2 of 4 - it was memorising, on 1.2M decisions
with 1.4M parameters. Here it gets 3,113,814. The prediction is that it turns positive. If
it does not, that is reported as a falsified hypothesis, the way the modulo-to-blake2b
hash change was: that one predicted an improvement and measured -0.0103.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from train import CATEGORIES, OUTPUT, Split, score, train, train_events_per_client

from ppsi.models.evaluation import (
    by_client_history,
    coverage_of_top5,
    divergence_of_top1,
)
from ppsi.models.session_gru import SessionGRUConfig

SEED = 13
BASELINE_SLICE_MACRO = 0.3143

# Two different uncertainties, and the larger one governs.
#
# 0.0026 is the 95% CI half-width of the metric on this validation set - how precisely the
# split measures a fixed model. But comparing two configurations also has to survive the
# fact that training itself is not deterministic across seeds, and the calibration run
# measured that directly: the same configuration on seeds 13, 42 and 2026 spanned 0.0033.
#
# So a gain smaller than 0.0033 is not distinguishable from having trained again.
SAMPLING_HALF_WIDTH = 0.0026
SEED_SPREAD = 0.0033
HALF_WIDTH = SEED_SPREAD

RUNGS = [
    (
        "1. all TRAIN decisions",
        SessionGRUConfig(channels=("category_id", "event_type_id"), use_gap=True),
    ),
    (
        "2. + product identity",
        SessionGRUConfig(channels=("category_id", "event_type_id", "product_bucket"), use_gap=True),
    ),
    (
        "3. + brand",
        SessionGRUConfig(
            channels=("category_id", "event_type_id", "product_bucket", "brand_bucket"),
            use_gap=True,
        ),
    ),
    (
        "4. + price band",
        SessionGRUConfig(
            channels=(
                "category_id",
                "event_type_id",
                "product_bucket",
                "brand_bucket",
                "price_band",
            ),
            use_gap=True,
        ),
    ),
    (
        "5. two layers",
        SessionGRUConfig(
            channels=(
                "category_id",
                "event_type_id",
                "product_bucket",
                "brand_bucket",
                "price_band",
            ),
            use_gap=True,
            layers=2,
        ),
    ),
]


def main() -> None:
    started = time.time()
    train_split, validation = Split.load("train"), Split.load("validation")
    rows = np.arange(len(train_split))
    print(f"TRAIN {len(train_split):,} decisions | VALIDATION {len(validation):,}")
    print(f"baseline to beat: {BASELINE_SLICE_MACRO} slice macro, noise floor {HALF_WIDTH}\n")

    train_events = train_events_per_client()

    results, curves, diagnostics = [], {}, {}
    previous = None
    for label, config in RUNGS:
        model, curve, best = train(
            train_split,
            validation,
            config=config,
            seed=SEED,
            rows=rows,
            batch_size=CONFIG_BATCH,
            learning_rate=CONFIG_LR,
            max_epochs=12,
            patience=2,
            label=label,
        )
        measured = score(model, validation, legacy_clamp=False, popularity=np.zeros(CATEGORIES))

        gain = measured["slice_macro"] - BASELINE_SLICE_MACRO
        step = None if previous is None else round(measured["slice_macro"] - previous, 4)
        results.append(
            {
                "rung": label,
                "best_epoch": best,
                "slice_macro": measured["slice_macro"],
                "slice_micro": measured["slice_micro"],
                "overall_micro": measured["overall_micro"],
                "gain_over_baseline": round(gain, 4),
                "step_over_previous": step,
                "beats_noise": bool(gain > HALF_WIDTH),
            }
        )
        previous = measured["slice_macro"]
        curves[label] = curve

        # Beyond the headline: does it recommend, or does it name the popular few?
        strata = by_client_history(
            measured["_slice_values"], validation.data["client"], train_events, measured["_changed"]
        )
        diagnostics[label] = {
            "by_client_history": strata.round(4).to_dict(),
            "coverage": coverage_of_top5(measured["_top5"]),
            "per_client": divergence_of_top1(measured["_top1"], validation.data["client"]),
        }

        print(f"    slice macro {measured['slice_macro']}   gain {gain:+.4f}   step {step}\n")
        del model

    frame = pd.DataFrame(results)
    print("=" * 78)
    print(frame.to_string(index=False))

    (OUTPUT / "ladder.json").write_text(
        json.dumps(
            {
                "baseline_slice_macro": BASELINE_SLICE_MACRO,
                "noise_floor": HALF_WIDTH,
                "seed": SEED,
                "results": results,
                "curves": curves,
                "diagnostics": diagnostics,
                "test_rows_read": 0,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


CONFIG_BATCH = 512
CONFIG_LR = 0.002

if __name__ == "__main__":
    main()

"""Rung 6 - a coordinate sweep, and an honest account of what selection buys.

Not a full grid. One axis is varied at a time, the best value on that axis is kept, and the
next axis starts from there. A full grid over four axes would be dozens of runs for a model
whose ablation ladder has already shown where the signal is - and where it is not.

The uncomfortable part of any sweep is that **the winner is biased upward**. Measure a
dozen configurations against one validation split and the best of them owes part of its
margin to luck rather than signal, with nothing in the run to say which part. Three guards,
all reported:

* every configuration tried is written out, not only the winner
* the winner is re-run on all three seeds and the spread is published beside the mean
* the claim is the gain over baseline against the measured seed spread, never the absolute

If the spread is as wide as the gain, the honest sentence is "no measurable improvement",
and that is what gets written.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from ladder import RUNGS, SEED_SPREAD
from train import CATEGORIES, OUTPUT, Split, score, train

from ppsi.models.session_gru import SessionGRUConfig, build_model, parameter_count

SEED = 13
SEEDS = (13, 42, 2026)
BASELINE_SLICE_MACRO = 0.3143

AXES = [
    ("hidden", [128, 256]),
    ("category embedding", [64, 128]),
    ("dropout", [0.1, 0.2, 0.3]),
    ("learning rate", [0.002, 0.001]),
]


def apply_axis(config: SessionGRUConfig, axis: str, value, learning_rate: float):
    if axis == "hidden":
        return replace(config, hidden=value), learning_rate
    if axis == "category embedding":
        widths = dict(config.widths)
        widths["category_id"] = value
        return replace(config, widths=widths), learning_rate
    if axis == "dropout":
        return replace(config, dropout=value), learning_rate
    if axis == "learning rate":
        return config, value
    raise KeyError(axis)


def current_value(config: SessionGRUConfig, axis: str, learning_rate: float):
    return {
        "hidden": config.hidden,
        "category embedding": config.widths["category_id"],
        "dropout": config.dropout,
        "learning rate": learning_rate,
    }[axis]


def choose_starting_point(ladder: dict) -> tuple:
    """Which rung to tune - and it is not automatically the highest-scoring one.

    Every step in the ladder came out smaller than the seed spread, so taking the top
    number would be selection rather than measurement. When two configurations cannot be
    told apart, the tie breaks on a criterion the metric cannot fake: size. The project's
    premise is a model that runs on a phone.
    """
    results = ladder["results"]
    highest = max(results, key=lambda row: row["slice_macro"])
    simplest = results[0]
    difference = highest["slice_macro"] - simplest["slice_macro"]

    print(f"  highest  : {highest['rung']:24} {highest['slice_macro']}")
    print(f"  simplest : {simplest['rung']:24} {simplest['slice_macro']}")
    print(f"  difference {difference:+.4f}  against a seed spread of {SEED_SPREAD}")

    if abs(difference) < SEED_SPREAD:
        print(
            "  -> indistinguishable. Tuning the simpler model; both are confirmed on "
            "three seeds at the end.\n"
        )
        return simplest, highest
    print("  -> the difference exceeds the spread. Tuning the stronger model.\n")
    return highest, simplest


def confirm(
    label: str,
    config: SessionGRUConfig,
    learning_rate: float,
    train_split: Split,
    validation: Split,
    rows: np.ndarray,
) -> list:
    """Run one configuration on all three project seeds."""
    runs = []
    for seed in SEEDS:
        model, _, best_epoch = train(
            train_split,
            validation,
            config=config,
            seed=seed,
            rows=rows,
            batch_size=512,
            learning_rate=learning_rate,
            max_epochs=12,
            patience=2,
            label=f"{label} | seed {seed}",
        )
        measured = score(model, validation, legacy_clamp=False, popularity=np.zeros(CATEGORIES))
        runs.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "slice_macro": measured["slice_macro"],
                "slice_micro": measured["slice_micro"],
                "overall_micro": measured["overall_micro"],
            }
        )
        print(f"    slice macro {measured['slice_macro']}\n")
        del model
    return runs


def summarise(runs: list) -> tuple:
    values = pd.DataFrame(runs)["slice_macro"]
    return float(values.mean()), float(values.max() - values.min())


def main() -> None:
    started = time.time()
    ladder = json.loads((OUTPUT / "ladder.json").read_text(encoding="utf-8"))
    print("choosing which rung to tune")
    start_from, rival_row = choose_starting_point(ladder)

    by_label = dict(RUNGS)
    config = by_label[start_from["rung"]]
    rival = by_label[rival_row["rung"]]
    learning_rate = 0.002

    train_split, validation = Split.load("train"), Split.load("validation")
    rows = np.arange(len(train_split))

    tried, best_score = [], start_from["slice_macro"]
    for axis, values in AXES:
        for value in values:
            if value == current_value(config, axis, learning_rate) and tried:
                continue
            candidate, candidate_rate = apply_axis(config, axis, value, learning_rate)
            model, curve, best_epoch = train(
                train_split,
                validation,
                config=candidate,
                seed=SEED,
                rows=rows,
                batch_size=512,
                learning_rate=candidate_rate,
                max_epochs=12,
                patience=2,
                label=f"{axis} = {value}",
            )
            measured = score(model, validation, legacy_clamp=False, popularity=np.zeros(CATEGORIES))
            tried.append(
                {
                    "axis": axis,
                    "value": value,
                    "best_epoch": best_epoch,
                    "slice_macro": measured["slice_macro"],
                    "slice_micro": measured["slice_micro"],
                    "parameters": parameter_count(model),
                    "final_gap": curve[-1]["gap"],
                }
            )
            print(f"    slice macro {measured['slice_macro']}\n")
            if measured["slice_macro"] > best_score:
                best_score = measured["slice_macro"]
                config, learning_rate = candidate, candidate_rate
            del model
        print(f"  -> keeping {axis} = {current_value(config, axis, learning_rate)}\n")

    print("=" * 78)
    print(pd.DataFrame(tried).to_string(index=False))
    print(
        f"\nconfigurations tried: {len(tried)}"
        "   <- the winner's margin is partly selection, and this is how much of it"
    )

    print("\nconfirming on three seeds: the tuned model, and the rung it was chosen over")
    winner_runs = confirm("tuned", config, learning_rate, train_split, validation, rows)
    rival_runs = confirm(rival_row["rung"], rival, 0.002, train_split, validation, rows)

    winner_mean, winner_spread = summarise(winner_runs)
    rival_mean, rival_spread = summarise(rival_runs)
    gain = winner_mean - BASELINE_SLICE_MACRO

    print("=" * 78)
    print(pd.DataFrame(winner_runs).to_string(index=False))
    print(f"\n  tuned model        : mean {winner_mean:.4f}  spread {winner_spread:.4f}")
    print(f"  {rival_row['rung']:18} : mean {rival_mean:.4f}  spread {rival_spread:.4f}")
    print(f"  difference between them: {winner_mean - rival_mean:+.4f}")
    print(f"\n  baseline           : {BASELINE_SLICE_MACRO}")
    print(f"  gain over baseline : {gain:+.4f}   against a seed spread of {winner_spread:.4f}")

    survives = winner_spread < gain
    if survives:
        print(f"\n  The gain survives a seed change: {gain:.4f} against {winner_spread:.4f}.")
    else:
        print("\n  The spread is as wide as the gain: NO MEASURABLE IMPROVEMENT.")

    (OUTPUT / "sweep.json").write_text(
        json.dumps(
            {
                "started_from": start_from,
                "rival": {
                    "rung": rival_row["rung"],
                    "runs": rival_runs,
                    "mean": round(rival_mean, 4),
                    "spread": round(rival_spread, 4),
                },
                "axes": [axis for axis, _ in AXES],
                "configurations_tried": len(tried),
                "tried": tried,
                "selected": {
                    "channels": list(config.channels),
                    "use_gap": config.use_gap,
                    "hidden": config.hidden,
                    "layers": config.layers,
                    "dropout": config.dropout,
                    "widths": config.widths,
                    "learning_rate": learning_rate,
                    "parameters": parameter_count(build_model(13, config=config)),
                },
                "seed_confirmation": winner_runs,
                "mean_slice_macro": round(winner_mean, 4),
                "seed_spread": round(winner_spread, 4),
                "baseline_slice_macro": BASELINE_SLICE_MACRO,
                "gain_over_baseline": round(gain, 4),
                "survives_seed_change": bool(survives),
                "test_rows_read": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()

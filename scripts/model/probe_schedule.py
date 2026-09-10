"""Settle the learning-rate axis properly instead of walking a constant downward forever.

Every reduction kept paying: 0.3390 at 0.002, 0.3436 at 0.001, 0.3454 at 0.0005, 0.3470 at
0.0003. Each step was small enough to sit inside the noise on its own, but they all pointed
the same way and the best epoch moved from 3 to 14. That is a model that is still
undertrained, not a search that got lucky.

Continuing to halve a constant is the wrong instrument. A cosine schedule does both jobs at
once - large steps early so the model moves, small steps late so it settles - and typically
beats any constant. Two runs decide it:

* **cosine from 0.001**, the rate the sweep chose
* **constant 0.0001**, one more rung down, to check the constant trend really has flattened

If cosine wins, it is adopted and confirmed on three seeds. If the constant keeps climbing,
that is worth knowing too - it would mean the ceiling on epochs, not the rate, is the
binding constraint.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from ladder import SEED_SPREAD
from train import CATEGORIES, OUTPUT, Split, score, train

from ppsi.models.session_gru import SessionGRUConfig

SEED = 13
BASELINE = 0.3143
CANDIDATES = [
    ("cosine from 0.001", 0.001, "cosine", 30),
    ("constant 0.0001", 0.0001, None, 40),
]


def main() -> None:
    started = time.time()
    sweep = json.loads((OUTPUT / "sweep.json").read_text(encoding="utf-8"))
    probe = json.loads((OUTPUT / "probe_lr.json").read_text(encoding="utf-8"))
    chosen = sweep["selected"]
    config = SessionGRUConfig(
        channels=tuple(chosen["channels"]),
        use_gap=chosen["use_gap"],
        widths=dict(chosen["widths"]),
        hidden=chosen["hidden"],
        layers=chosen["layers"],
        dropout=chosen["dropout"],
    )

    best_constant = probe["best"]
    print(
        f"best constant so far: lr {best_constant['learning_rate']} "
        f"at {best_constant['slice_macro']}\n"
    )

    train_split, validation = Split.load("train"), Split.load("validation")
    rows = np.arange(len(train_split))

    tried = [
        {
            "setting": f"constant {best_constant['learning_rate']}",
            "slice_macro": best_constant["slice_macro"],
            "best_epoch": best_constant["best_epoch"],
            "source": "probe_lr",
        }
    ]

    for label, rate, schedule, ceiling in CANDIDATES:
        model, curve, best_epoch = train(
            train_split,
            validation,
            config=config,
            seed=SEED,
            rows=rows,
            batch_size=512,
            learning_rate=rate,
            max_epochs=ceiling,
            patience=3,
            label=label,
            schedule=schedule,
        )
        measured = score(model, validation, legacy_clamp=False, popularity=np.zeros(CATEGORIES))
        tried.append(
            {
                "setting": label,
                "slice_macro": measured["slice_macro"],
                "best_epoch": best_epoch,
                "epochs_run": len(curve),
                "hit_ceiling": len(curve) >= ceiling,
                "final_gap": curve[-1]["gap"],
                "source": "seed 13",
            }
        )
        print(
            f"    slice macro {measured['slice_macro']}   best epoch {best_epoch} of {len(curve)}\n"
        )
        del model

    frame = pd.DataFrame(tried)
    print("=" * 78)
    print(frame.to_string(index=False))

    winner = max(tried, key=lambda row: row["slice_macro"])
    print(f"\n  best: {winner['setting']} at {winner['slice_macro']}")
    for row in tried:
        if row.get("hit_ceiling"):
            print(
                f"  NOTE: {row['setting']} stopped at the epoch ceiling, not by "
                "converging - its number is a lower bound"
            )

    (OUTPUT / "probe_schedule.json").write_text(
        json.dumps(
            {
                "reason": "a constant rate kept improving; a schedule is the principled version",
                "tried": tried,
                "winner": winner,
                "threshold": SEED_SPREAD,
                "baseline": BASELINE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()

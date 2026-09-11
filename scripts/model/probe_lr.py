"""Follow the one axis that actually moved.

The sweep changed five things and four of them did nothing. Halving the learning rate did:
0.3390 at 0.002 against 0.3443 at 0.001, and the best epoch moved from 3 to 9. That is not
a hyperparameter preference, it is a diagnosis - the model was **undertrained**. It was
converging quickly to a worse optimum and early stopping was calling it done.

A coordinate sweep stops after one step on each axis. When one axis shows a trend that
strong, stopping there leaves a gain on the table for no reason other than the shape of the
search. So this walks it further and stops when it stops paying.
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
RATES = [0.0005, 0.0003]
BASELINE = 0.3143


def main() -> None:
    started = time.time()
    sweep = json.loads((OUTPUT / "sweep.json").read_text(encoding="utf-8"))
    chosen = sweep["selected"]
    config = SessionGRUConfig(
        channels=tuple(chosen["channels"]),
        use_gap=chosen["use_gap"],
        widths=dict(chosen["widths"]),
        hidden=chosen["hidden"],
        layers=chosen["layers"],
        dropout=chosen["dropout"],
    )

    print(
        f"sweep selected lr {chosen['learning_rate']} at {sweep['mean_slice_macro']} "
        f"(mean over three seeds)\n"
    )

    train_split, validation = Split.load("train"), Split.load("validation")
    rows = np.arange(len(train_split))

    tried = [
        {
            "learning_rate": float(chosen["learning_rate"]),
            "slice_macro": sweep["mean_slice_macro"],
            "best_epoch": None,
            "source": "sweep, mean of three seeds",
        }
    ]

    for rate in RATES:
        # A lower rate needs more room to converge, or the ceiling would be mistaken for
        # a plateau - the exact error that made 0.002 look like the right answer.
        model, curve, best_epoch = train(
            train_split,
            validation,
            config=config,
            seed=SEED,
            rows=rows,
            batch_size=512,
            learning_rate=rate,
            max_epochs=25,
            patience=3,
            label=f"learning rate = {rate}",
        )
        measured = score(model, validation, legacy_clamp=False, popularity=np.zeros(CATEGORIES))
        tried.append(
            {
                "learning_rate": rate,
                "slice_macro": measured["slice_macro"],
                "best_epoch": best_epoch,
                "epochs_run": len(curve),
                "final_gap": curve[-1]["gap"],
                "source": "seed 13",
            }
        )
        print(
            f"    slice macro {measured['slice_macro']}   best epoch {best_epoch} "
            f"of {len(curve)} run\n"
        )
        del model

    frame = pd.DataFrame(tried)
    print("=" * 74)
    print(frame.to_string(index=False))

    best = max(tried, key=lambda row: row["slice_macro"])
    improvement = best["slice_macro"] - sweep["mean_slice_macro"]
    print(f"\n  best here : lr {best['learning_rate']} at {best['slice_macro']}")
    print(f"  vs sweep  : {improvement:+.4f}   against a seed spread of {SEED_SPREAD}")
    if improvement > SEED_SPREAD:
        print("  -> worth adopting; it needs its own three-seed confirmation")
    else:
        print("  -> inside the noise; the sweep's choice stands")

    (OUTPUT / "probe_lr.json").write_text(
        json.dumps(
            {
                "reason": "the learning-rate axis showed the only trend larger than noise",
                "tried": tried,
                "best": best,
                "improvement_over_sweep": round(improvement, 4),
                "threshold": SEED_SPREAD,
                "adopt": bool(improvement > SEED_SPREAD),
                "baseline": BASELINE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()

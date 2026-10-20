"""Separate two things the word "seed spread" quietly conflates.

The ladder and the sweep trained the same configuration on the same seed and did not get
the same number - 0.3373 against 0.3379. That is not a bug in either; cuDNN's GRU kernels
are not deterministic by default, so a run is not reproducible even against itself.

Which means the 0.0033 measured across seeds 13, 42 and 2026 is really two effects added
together:

* **initialisation variance** - genuinely different starting weights per seed
* **kernel non-determinism** - the same seed, twice, landing somewhere else

Both belong in the answer to "would this gain survive rerunning", so the threshold built
from them is right. But `S2-PR-09` has to reproduce runs, and a claim of reproducibility
needs to say under what setting. This measures each part, and what determinism costs.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from ladder import RUNGS
from train import CATEGORIES, OUTPUT, Split, score, train

SEED = 13
REPEATS = 3


def run_once(config, train_split, validation, rows, label):
    started = time.time()
    model, curve, _ = train(
        train_split,
        validation,
        config=config,
        seed=SEED,
        rows=rows,
        batch_size=512,
        learning_rate=0.002,
        max_epochs=3,
        patience=3,
        label=label,
    )
    measured = score(model, validation, legacy_clamp=False, popularity=np.zeros(CATEGORIES))
    del model
    return {
        "slice_macro": measured["slice_macro"],
        "final_val_loss": curve[-1]["val_loss"],
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    config = dict(RUNGS)["1. all TRAIN decisions"]
    train_split, validation = Split.load("train"), Split.load("validation")
    # A subset: this is about run-to-run variation, not about the best achievable number,
    # and three epochs on a quarter of the data shows it just as clearly for a fraction of
    # the time.
    rows = np.sort(np.random.default_rng(13).choice(len(train_split), 800_000, replace=False))

    report = {}
    for mode in ("default", "deterministic"):
        if mode == "deterministic":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        print(f"\n=== {mode} kernels " + "=" * 50)
        runs = [
            run_once(config, train_split, validation, rows, f"{mode} | run {index + 1}")
            for index in range(REPEATS)
        ]
        values = [row["slice_macro"] for row in runs]
        spread = max(values) - min(values)
        report[mode] = {
            "runs": runs,
            "values": values,
            "spread": round(spread, 5),
            "mean_seconds": round(float(np.mean([r["seconds"] for r in runs])), 1),
        }
        print(f"  values {values}   spread {spread:.5f}")

    default_spread = report["default"]["spread"]
    deterministic_spread = report["deterministic"]["spread"]
    slowdown = report["deterministic"]["mean_seconds"] / report["default"]["mean_seconds"]

    print("\n" + "=" * 70)
    print(f"  same seed, default kernels      : spread {default_spread:.5f}")
    print(f"  same seed, deterministic kernels: spread {deterministic_spread:.5f}")
    print(f"  determinism costs               : {slowdown:.2f}x runtime")
    print("\n  measured across seeds (calibration): 0.00330")
    print(f"  of which same-seed non-determinism : {default_spread:.5f}")
    print(f"  leaving initialisation variance    : ~{0.0033 - default_spread:.5f}")

    report["interpretation"] = {
        "across_seed_spread": 0.0033,
        "same_seed_spread": default_spread,
        "initialisation_variance_estimate": round(0.0033 - default_spread, 5),
        "determinism_slowdown": round(slowdown, 2),
        "note": "the comparison threshold uses the across-seed figure, because that is "
        "what 'would this survive rerunning' actually means",
    }
    (OUTPUT / "determinism.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

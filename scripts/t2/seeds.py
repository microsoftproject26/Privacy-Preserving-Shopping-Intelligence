"""Does the T2 gain survive a seed change?

The ladder selected the fine-tuned rung on one seed. A number chosen on one seed carries
some margin that is simply the luckiest draw, and the only way to know how much is to run
the *selected* configuration again on seeds that took no part in selecting it.

This runs the winner and nothing else. Re-running the whole ladder would answer a different
question - "does the same rung win every time?" - which is worth knowing but is not what the
headline claims. The headline claims a gain, so the gain is what gets re-measured.

Seeds 13, 42 and 2026 are the three the repository already restricts itself to, and the same
three `S2-PR-09` will use, so R1 and its federated counterpart start from identical weights.

Reported: the mean, the spread, and the gain against the corrected baseline. If the spread is
wider than the gain, the honest statement is "no measurable improvement" - and that is what
would get written.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_t2 import BATCH, ENCODER, EPOCHS, LEARNING_RATE, find_checkpoint
from train_t2 import BEST_SIMPLE_CORRECTED, HALF_WIDTH, OUTPUT, SPEC, Split, evaluate, train
from ppsi.models.checkpoint import save as save_checkpoint

SEEDS = (13, 42, 2026)
RUNG = dict(freeze_encoder=False, pos_weight=None)


def main() -> None:
    started = time.time()
    checkpoint = find_checkpoint()
    train_split, validation = Split.load("train"), Split.load("validation")
    print(f"encoder   : {checkpoint.name}")
    print(f"decisions : {len(train_split):,} TRAIN, {len(validation):,} VALIDATION")
    print(f"baseline  : {BEST_SIMPLE_CORRECTED} (smoothed item+category, corrected labels)\n")

    rows, curves = [], {}
    for seed in SEEDS:
        model, curve, best_score, provenance = train(
            train_split, validation, config=ENCODER, seed=seed, checkpoint=checkpoint,
            learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH,
            label=f"fine-tuned | seed {seed}", **RUNG)
        measured = evaluate(model, validation)
        # `train` returns the best score, not the epoch it came from; the curve knows.
        best_epoch = max(curve, key=lambda row: row["pr_auc"])["epoch"]
        gain = measured["pr_auc"] - BEST_SIMPLE_CORRECTED
        assert abs(measured["pr_auc"] - best_score) < 1e-6, (
            f"re-scoring the restored best state gives {measured['pr_auc']} against the "
            f"{best_score} training reported; the wrong weights were reloaded")
        rows.append({"seed": seed, "pr_auc": measured["pr_auc"],
                     "roc_auc": measured["roc_auc"], "brier": measured["brier"],
                     "best_epoch": best_epoch, "gain": round(gain, 4)})
        curves[str(seed)] = curve
        save_checkpoint(OUTPUT / f"t2_finetuned_seed{seed}.pt", model=model, spec=SPEC,
                        seed=seed, rung="fine-tuned", labels="corrected")
        print(f"    seed {seed}:  PR-AUC {measured['pr_auc']:.4f}   "
              f"gain {gain:+.4f}   best epoch {best_epoch}\n")
        del model

    scores = np.array([r["pr_auc"] for r in rows])
    gains = np.array([r["gain"] for r in rows])
    spread = float(scores.max() - scores.min())

    print("=" * 74)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n  mean PR-AUC {scores.mean():.4f}   spread {spread:.4f}   "
          f"mean gain {gains.mean():+.4f}")

    # The claim the report is allowed to make.
    survives = bool(gains.min() > spread)
    print(f"  gain on every seed exceeds the spread: {survives}")

    (OUTPUT / "s2_ds_06_seeds.json").write_text(json.dumps({
        "task": "S2-DS-06", "configuration": "fine-tuned encoder, the selected rung",
        "labels": "corrected", "seeds": list(SEEDS),
        "baseline": BEST_SIMPLE_CORRECTED,
        "runs": rows,
        "mean_pr_auc": round(float(scores.mean()), 4),
        "seed_spread": round(spread, 4),
        "mean_gain": round(float(gains.mean()), 4),
        "min_gain": round(float(gains.min()), 4),
        "gain_survives_seed_change": survives,
        "old_half_width": HALF_WIDTH,
        "note": ("the seed spread measures run-to-run variation only. The interval that "
                 "answers whether the gain would survive different shoppers is the paired "
                 "client bootstrap in s2_ds_06_interval.json; the two are different "
                 "questions and both are reported."),
        "curves": curves,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()

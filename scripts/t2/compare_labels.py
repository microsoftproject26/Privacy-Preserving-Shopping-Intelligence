"""Did restoring the withheld negatives change the model, or only the measurement?

The censoring defect had two separable effects and they are easy to conflate:

1. **The metric.** 70,467 VALIDATION negatives were dropped from scoring, so PR-AUC was
   computed on a population with a 22% inflated base rate. This alone moves the published
   figure from `0.1797` to `0.1411`.
2. **The training data.** 478,718 TRAIN negatives were withheld from the loss, so the model
   never saw them.

Effect 1 is certain and arithmetic. Effect 2 is an empirical question, and the only way to
answer it is to score both models on the same corrected labels:

* `t2_2_trained_on_published_mask.pt` - trained without the restored negatives
* `t2_2.pt` - retrained with them

If the retrained model is no better, then the defect was **purely a measurement error**: the
model was always this good and the number was always wrong. That is a more useful thing to
be able to say than either half on its own, and it tells `S2-DS-08` that the extra negatives
are not worth budgeting for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interval import model_scores, smoothed_baseline
from train_t2 import OUTPUT, Split


def score(name: str, checkpoint: Path, split: Split, labels: np.ndarray,
          published: np.ndarray) -> dict:
    logits = model_scores(checkpoint, split)
    probability = 1.0 / (1.0 + np.exp(-logits))
    row = {
        "model": name, "checkpoint": checkpoint.name,
        "pr_auc_corrected": round(float(average_precision_score(labels, logits)), 4),
        "roc_auc_corrected": round(float(roc_auc_score(labels, logits)), 4),
        "brier_corrected": round(float(((probability - labels) ** 2).mean()), 5),
        "pr_auc_published_mask": round(
            float(average_precision_score(labels[published], logits[published])), 4),
        "mean_probability": round(float(probability.mean()), 4),
        # The restored rows are not ordinary negatives. If the model scores them high, they
        # are hard - and that is exactly why including them costs PR-AUC.
        "mean_probability_on_restored": round(float(probability[~published].mean()), 4),
        "p90_probability_on_restored": round(
            float(np.percentile(probability[~published], 90)), 4),
    }
    return row


def main() -> None:
    split = Split.load("validation")
    baseline, labels, _ = smoothed_baseline()
    published = split.published
    assert np.array_equal(labels.astype("float32"), split.label), (
        "the cache and the parquet disagree row by row; every score would be paired with "
        "the wrong decision")

    print(f"  scoring {len(labels):,} decisions, {int(labels.sum()):,} positives, "
          f"{int((~published).sum()):,} restored\n")

    rows = [
        score("trained on the published mask", OUTPUT / "t2_2_trained_on_published_mask.pt",
              split, labels, published),
        score("retrained on corrected labels", OUTPUT / "t2_2.pt",
              split, labels, published),
    ]

    base_pr = float(average_precision_score(labels, baseline))
    delta = rows[1]["pr_auc_corrected"] - rows[0]["pr_auc_corrected"]
    result = {
        "task": "S2-DS-06", "question": "did the restored TRAIN negatives change the model?",
        "restored_train_rows": 478718, "restored_validation_rows": 70467,
        "baseline_pr_auc_corrected": round(base_pr, 4),
        "models": rows,
        "difference_from_retraining": round(delta, 4),
        "verdict": ("the defect was a measurement error, not a training error"
                    if abs(delta) < 0.005 else
                    "retraining on the restored negatives changed the model materially"),
    }

    import pandas as pd
    print(pd.DataFrame(rows)[["model", "pr_auc_corrected", "pr_auc_published_mask",
                              "roc_auc_corrected", "mean_probability_on_restored"]]
          .to_string(index=False))
    print(f"\n  baseline (smoothed item+category) {base_pr:.4f}")
    print(f"  retraining moved PR-AUC by {delta:+.4f}")
    print(f"  -> {result['verdict']}")
    (OUTPUT / "s2_ds_06_label_ablation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

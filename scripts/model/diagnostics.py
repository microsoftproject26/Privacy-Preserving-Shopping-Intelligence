"""The three diagnostics `S2-PR-07` specifies that nobody has run.

Two of the five are already in `s2_ds_01_result.json`: category coverage (573 of 588 ever
reach a top-5, Gini `0.8233`) and per-client top-1 divergence (`2.342` distinct top-1s per
client). These are the rest, and one of them asks a question this project has never put to
the model at all.

## The question that matters most

A high MRR does not mean a good recommender, and the failure mode is specific: **a model
whose answer is a function of the current category alone is a transition table with 2.3
million parameters.** It would score well, because the transition table scores well. It
would show per-client variation, because clients browse different categories. And every
metric in this project would look exactly the same.

The project's whole premise depends on it being false. `R3` (personalized) and `R4`
(device-local) assume a client's own history changes what they are shown. The encoder is
session-local by construction - `build_windows` stops at the session boundary - so the
session is the *only* history it has. If the session contributes nothing beyond its last
event, there is nothing for personalization to act on, and the strongest claim in the
proposal quietly loses its basis.

So: **how often does the model give the same top-1 to two decisions that share a current
category but have different histories?** If that is near 1.0, the sequence model is
decoration.

## The other two

* **Popularity correlation** - if predicted rank tracks global popularity, we built an
  expensive popularity baseline.
* **Confidence reliability** - when the model is confident, is it right? If not, the product
  cannot threshold on it, which rules out "only show a recommendation when sure".
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))

from finalize import selected_config
from train import CATEGORIES, DEVICE, OUTPUT, SPEC, Split, to_batch
from ppsi.models.checkpoint import load_encoder
from ppsi.models.evaluation import popularity_correlation
from ppsi.models.session_gru import build_model

SEED = 13
BINS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01)


def collect(model, validation: Split) -> dict:
    """One pass: top-1, the winning probability, and the truth."""
    top1 = np.zeros(len(validation), dtype="int64")
    confidence = np.zeros(len(validation), dtype="float64")
    score_sum = np.zeros(CATEGORIES, dtype="float64")
    with torch.no_grad():
        for start in range(0, len(validation), 2048):
            rows = np.arange(start, min(start + 2048, len(validation)))
            logits = model(to_batch(validation, rows).to(DEVICE)).t1_logits.float()
            probability = torch.softmax(logits, dim=1)
            best, index = probability.max(dim=1)
            top1[rows] = index.cpu().numpy()
            confidence[rows] = best.cpu().numpy()
            score_sum += logits.sum(dim=0).cpu().numpy()
    return {"top1": top1, "confidence": confidence, "score_sum": score_sum}


def is_the_model_a_transition_table(top1: np.ndarray, current: np.ndarray,
                                    lengths: np.ndarray) -> dict:
    """Among decisions sharing a current category, how often is the top-1 the same?

    Reported as the **share of decisions whose top-1 equals the most common top-1 for their
    own current category**. At `1.0` the model is a lookup on the last event. Below that,
    the remainder is what the session history is contributing.

    Decisions with a history of length 1 are excluded from the comparison group, because
    there the current event *is* the whole session and agreement proves nothing.
    """
    frame = pd.DataFrame({"current": current, "top1": top1, "length": lengths})
    real = frame[frame["length"] > 1]

    modal = real.groupby("current")["top1"].agg(lambda column: column.mode().iat[0])
    agrees = real["top1"].to_numpy() == real["current"].map(modal).to_numpy()

    per_category = (real.assign(agrees=agrees).groupby("current")
                    .agg(decisions=("top1", "size"),
                         distinct_top1=("top1", "nunique"),
                         modal_share=("agrees", "mean")))
    busy = per_category[per_category["decisions"] >= 50]

    return {
        "decisions_compared": int(len(real)),
        "share_equal_to_their_category_modal_top1": round(float(agrees.mean()), 4),
        "distinct_top1_per_category_mean": round(float(per_category["distinct_top1"].mean()), 2),
        "categories_with_50plus_decisions": int(len(busy)),
        "modal_share_on_those_categories": round(float(busy["modal_share"].mean()), 4),
        "reading": ("1.0 would mean the top-1 is decided by the current category alone and "
                    "the sequence model is decoration; the shortfall below 1.0 is what the "
                    "session history changes"),
    }


def against_the_transition_table(top1: np.ndarray, current: np.ndarray, truth: np.ndarray,
                                 train_split: Split) -> dict:
    """The decisive version: does the model pick what the transition table picks?

    Agreeing with its own per-category modal answer only shows the model is *consistent*.
    The claim worth testing is sharper - that the model is an expensive lookup - and that
    means comparing it against the actual lookup.

    Also reported: accuracy@1 for both. If the model's top-1 matches the table's while the
    model still scores higher on MRR, its advantage lives in how it orders ranks 2 to 20 -
    a different and much more modest claim than "it predicts what you will do next".
    """
    from ppsi.models.evaluation import transition_table

    table = transition_table(train_split.data["query_category"],
                             train_split.data["target"], CATEGORIES)
    table_top1 = np.where(current >= 0, table[np.clip(current, 0, None)].argmax(axis=1), -1)

    known = current >= 0
    same = (top1 == table_top1) & known
    return {
        "decisions_with_a_table_row": int(known.sum()),
        "model_top1_equals_table_top1": round(float(same.sum() / known.sum()), 4),
        "model_accuracy_at_1": round(float((top1 == truth)[known].mean()), 4),
        "table_accuracy_at_1": round(float((table_top1 == truth)[known].mean()), 4),
        "reading": ("if the two top-1s mostly agree while the model still scores higher on "
                    "MRR, the model's advantage is in the ordering below rank 1, not in "
                    "what it puts first"),
    }


def reliability(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """When the model says it is sure, is it? Binned, with the count in each bin."""
    bins = []
    for low, high in zip(BINS[:-1], BINS[1:], strict=True):
        inside = (confidence >= low) & (confidence < high)
        count = int(inside.sum())
        bins.append({"from": low, "to": round(high, 2), "decisions": count,
                     "share_of_all": round(count / len(confidence), 4),
                     "mean_confidence": round(float(confidence[inside].mean()), 4) if count else None,
                     "accuracy": round(float(correct[inside].mean()), 4) if count else None})
    calibrated = [b for b in bins if b["decisions"] >= 1000]
    gap = (round(float(np.mean([abs(b["mean_confidence"] - b["accuracy"])
                                for b in calibrated])), 4) if calibrated else None)
    return {"bins": bins, "mean_absolute_gap_on_populated_bins": gap,
            "reading": ("a bin where confidence far exceeds accuracy means the product "
                        "cannot threshold on confidence to decide when to recommend")}


def main() -> None:
    started = time.time()
    config, _, _ = selected_config()
    validation = Split.load("validation")
    checkpoint = OUTPUT / f"s2_ds_01_gru_t1_seed{SEED}.pt"
    model = build_model(SEED, batch_spec=SPEC, config=config)
    load_encoder(model, checkpoint)
    model = model.to(DEVICE).eval()
    print(f"  {len(validation):,} VALIDATION decisions, {checkpoint.name}\n")

    gathered = collect(model, validation)
    truth = validation.data["target"]
    current = validation.current_category
    correct = (gathered["top1"] == truth).astype("float64")

    popularity = np.bincount(validation.data["target"], minlength=CATEGORIES).astype("float64")
    correlation = popularity_correlation(gathered["score_sum"][None, :], popularity)

    lookup = is_the_model_a_transition_table(gathered["top1"], current,
                                             validation.data["lengths"])
    bins = reliability(gathered["confidence"], correct)

    versus = against_the_transition_table(gathered["top1"], current, truth,
                                          Split.load("train"))

    print("  1. does the session do anything, or is this a transition table?")
    print(f"     top-1 equals its category's modal top-1 in "
          f"{lookup['share_equal_to_their_category_modal_top1'] * 100:.1f}% of decisions")
    print(f"     distinct top-1 per current category, mean "
          f"{lookup['distinct_top1_per_category_mean']}")
    print(f"     -> the session changes the answer "
          f"{(1 - lookup['share_equal_to_their_category_modal_top1']) * 100:.1f}% of the time")

    print("")
    print("  1b. the decisive version: model top-1 against the actual transition table")
    print(f"     the two agree on {versus['model_top1_equals_table_top1'] * 100:.1f}% "
          "of decisions")
    print("     accuracy@1  model {:.4f}   table {:.4f}".format(
        versus["model_accuracy_at_1"], versus["table_accuracy_at_1"]))

    print(f"\n  2. popularity correlation of the mean score vector: {correlation:+.4f}")
    print("     (high would mean we built an expensive popularity baseline)")

    print("\n  3. confidence reliability")
    print(pd.DataFrame(bins["bins"]).to_string(index=False))
    print(f"     mean |confidence - accuracy| on populated bins: "
          f"{bins['mean_absolute_gap_on_populated_bins']}")

    (OUTPUT / "diagnostics_full.json").write_text(json.dumps({
        "task": "S2-DS-01", "checkpoint": checkpoint.name,
        "decisions": int(len(validation)),
        "accuracy_at_1": round(float(correct.mean()), 4),
        "session_versus_lookup": lookup,
        "versus_the_transition_table": versus,
        "popularity_correlation": round(float(correlation), 4),
        "confidence_reliability": bins,
        "already_reported_elsewhere": ["category_coverage", "per_client_top1_divergence"],
    }, indent=2), encoding="utf-8")
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

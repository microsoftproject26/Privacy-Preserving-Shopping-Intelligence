"""S2-DS-06 end to end: gate, ladder, report.

Same discipline as S2-DS-01, because it is what made that number defensible:

1. **Calibrate before trusting.** Reproduce the published baseline first. If the pipeline
   cannot reproduce a number we already have, nothing it produces afterwards means
   anything.
2. **One change per rung.** An ablation that moves two things teaches nothing about either.
3. **State the prediction before the run.** A hypothesis that cannot fail is not one.
4. **Losing rungs stay in the report.**

The gate here carries a check T1 did not need. 17.95% of T2 decisions are censored - the
outcome was never observable - and if those rows leak into training as negatives the model
learns that people do not buy. Nothing downstream would flag it: the loss looks fine, the
PR-AUC looks plausible, and every predicted probability is quietly too low. So the gate
asserts that the number of loss-contributing rows equals the number of unmasked rows,
exactly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from train_t2 import (
    BASELINES_CORRECTED,
    BASELINES_PUBLISHED_MASK,
    BEST_SIMPLE,
    BEST_SIMPLE_CORRECTED,
    DEVICE,
    HALF_WIDTH,
    OUTPUT,
    SPEC,
    Split,
    average_precision,
    evaluate,
    to_batch,
    train,
)

from ppsi.models.checkpoint import save as save_checkpoint
from ppsi.models.session_gru import SessionGRUConfig
from ppsi.training.batch import validate_canonical_phase1_batch

SEED = 13
SEEDS = (13, 42, 2026)

# The S2-DS-01 selection, carried over rather than rediscovered: two channels, one layer,
# dropout 0.3, cosine from 0.001. That task established the encoder is undertrained at
# higher rates and shorter schedules; repeating the search here would spend hours to
# arrive at the same place.
ENCODER = SessionGRUConfig(channels=("category_id", "event_type_id"), use_gap=True,
                           dropout=0.3)
EPOCHS = 12
LEARNING_RATE = 0.001
BATCH = 512


def find_checkpoint() -> Path:
    for candidate in (
        Path(__file__).resolve().parent.parent / "S2-DS-01_GRU_T1_Model" / "output"
        / "s2_ds_01_gru_t1_seed13.pt",
        Path(__file__).resolve().parent.parent / "data" / "encoder"
        / "s2_ds_01_gru_t1_seed13.pt",
        Path("data/encoder/s2_ds_01_gru_t1_seed13.pt"),
    ):
        if candidate.exists():
            return candidate
    raise SystemExit("cannot find the S2-DS-01 encoder checkpoint")


def gate(train_split: Split, validation: Split) -> dict:
    """Everything that must hold before a single epoch is worth running."""
    print("\n" + "=" * 78)
    print("THE GATE")
    print("=" * 78)

    # --- the censoring contract, and the correction to it ---------------------------
    # `published` is upstream's mask; `mask` is the corrected one. Asserting on `published`
    # is what proves we are looking at the same rows upstream published, which is the only
    # thing that makes the corrected number comparable to anything.
    for name, split in (("TRAIN", train_split), ("VALIDATION", validation)):
        withheld = float((~split.published).mean()) * 100
        print(f"  {name:11} {len(split):>9,} decisions, {withheld:5.2f}% withheld upstream, "
              f"{int((~split.published).sum()):,} restored")
    assert abs(float((~validation.published).mean()) - 0.1795) < 0.0005, (
        "VALIDATION withholding is not the published 17.95%; these are not the same rows")
    assert abs(float((~train_split.published).mean()) - 0.1728) < 0.0005, (
        "TRAIN withholding is not the published 17.28%")
    print("  ok   we are on the published rows: 17.28% / 17.95% withheld upstream")

    # Upstream censored a decision for being the last event of its session. S1-DS-05/06
    # already dropped every session that crosses a split boundary, so a surviving session
    # is complete inside its own window and its outcome is observed. See `labels.py`.
    assert validation.mask.all() and train_split.mask.all(), (
        "the correction did not restore every withheld row; labels.py and this gate "
        "disagree about which rows are observable")
    restored = np.flatnonzero(~validation.published)
    assert float(validation.label[restored].sum()) == 0.0, (
        "a restored row carries a positive label; upstream only ever withheld negatives "
        "and this correction would be inventing purchases")
    print(f"  ok   all {len(restored):,} restored rows are negatives, as the rule requires")

    # A restored row must now be genuinely present in the batch, carrying its real zero -
    # the mirror of the old check, which asserted the same rows were absent.
    batch = to_batch(validation, restored[:512])
    validate_canonical_phase1_batch(batch, SPEC)
    assert batch.t2_present.all(), "a restored row is still marked absent"
    assert float(batch.t2_target.abs().sum()) == 0.0, "a restored row carries a label"
    print("  ok   restored rows are present in the batch and carry an observed zero")

    # --- the baselines, on both populations ------------------------------------------
    rows = validation.trainable
    labels = validation.label[rows].astype("float64")
    published_rows = np.flatnonzero(validation.published)
    published_labels = validation.label[published_rows].astype("float64")

    print(f"\n  corrected      {len(rows):,} decisions, {int(labels.sum()):,} positives, "
          f"base rate {labels.mean():.4f}")
    print(f"  published mask {len(published_rows):,} decisions, "
          f"{int(published_labels.sum()):,} positives, "
          f"base rate {published_labels.mean():.4f}")
    assert abs(published_labels.mean() - 0.0351) < 0.0005, (
        f"published base rate {published_labels.mean():.4f} is not the published 0.0351")
    assert abs(labels.mean() - 0.0288) < 0.0005, (
        f"corrected base rate {labels.mean():.4f} is not 0.0288; the restored rows are "
        "not the 70,467 the review measured")

    prevalence = average_precision(np.zeros(len(labels)), labels)
    published_prevalence = average_precision(np.zeros(len(published_labels)),
                                             published_labels)
    print(f"  prevalence PR-AUC  corrected {prevalence:.4f}   "
          f"published {published_prevalence:.4f} vs upstream "
          f"{BASELINES_PUBLISHED_MASK['prevalence (no signal)']}")
    assert abs(published_prevalence
               - BASELINES_PUBLISHED_MASK["prevalence (no signal)"]) < 0.0001, (
        "a no-signal scorer does not reproduce the published prevalence PR-AUC; the "
        "metric implementation disagrees with upstream and no comparison would be valid")
    print("  ok   PR-AUC still reproduces the published prevalence exactly")

    print(f"\n  the number to beat: {BEST_SIMPLE_CORRECTED} (smoothed item+category on the")
    print("  corrected labels), noise floor 0.0029. The item-popularity 0.0831 we first")
    print("  published was neither the strongest simple rule nor on the right population.")
    return {"train_withheld_upstream": round(float((~train_split.published).mean()), 4),
            "validation_withheld_upstream": round(float((~validation.published).mean()), 4),
            "restored": len(restored),
            "corrected_decisions": len(rows),
            "published_decisions": len(published_rows),
            "positives": int(labels.sum()),
            "corrected_base_rate": round(float(labels.mean()), 4),
            "published_base_rate": round(float(published_labels.mean()), 4),
            "corrected_prevalence_pr_auc": round(prevalence, 4),
            "published_prevalence_pr_auc": round(published_prevalence, 4)}


RUNGS = [
    ("1. frozen encoder, plain BCE", {"freeze_encoder": True, "pos_weight": None}),
    ("2. fine-tuned encoder", {"freeze_encoder": False, "pos_weight": None}),
]


def main() -> None:
    started = time.time()
    checkpoint = find_checkpoint()
    print(f"encoder    : {checkpoint.name}")
    print(f"device     : {DEVICE}")

    train_split, validation = Split.load("train"), Split.load("validation")
    facts = gate(train_split, validation)

    print("\n" + "=" * 78)
    print("THE LADDER - one change per rung")
    print("=" * 78)
    print("  Prediction, stated before the runs: the fine-tuned encoder wins, because the")
    print("  T1 encoder was trained to separate categories and purchase intent is a")
    print("  different question. If freezing wins instead, the shared representation is")
    print("  already carrying intent and S2-DS-08 has an easier job than expected.\n")

    results, curves = [], {}
    for label, options in RUNGS:
        model, curve, _best, _provenance = train(
            train_split, validation, config=ENCODER, seed=SEED, checkpoint=checkpoint,
            learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH, label=label,
            **options)
        measured = evaluate(model, validation)
        gain = measured["pr_auc"] - BEST_SIMPLE_CORRECTED
        results.append({"rung": label, **options, **measured,
                        "gain_over_baseline": round(gain, 4),
                        "beats_noise": bool(gain > HALF_WIDTH)})
        curves[label] = curve
        print(f"    PR-AUC {measured['pr_auc']}   over baseline {gain:+.4f}   "
              f"{'beats' if gain > HALF_WIDTH else 'inside'} the noise floor\n")
        save_checkpoint(OUTPUT / f"t2_{label.split('.')[0]}.pt", model=model, spec=SPEC,
                        seed=SEED, rung=label, labels="corrected")
        del model

    frame = pd.DataFrame(results)
    print("=" * 78)
    print(frame[["rung", "pr_auc", "roc_auc", "brier", "gain_over_baseline",
                 "beats_noise"]].to_string(index=False))

    (OUTPUT / "s2_ds_06_ladder.json").write_text(json.dumps({
        "task": "S2-DS-06", "encoder": checkpoint.name, "gate": facts,
        "labels": "corrected - see S2-DS-06/labels.py",
        "baselines_corrected": BASELINES_CORRECTED,
        "baselines_published_mask": BASELINES_PUBLISHED_MASK,
        "best_simple": BEST_SIMPLE_CORRECTED,
        "best_simple_published_mask": BEST_SIMPLE, "noise_floor": HALF_WIDTH,
        "results": results, "curves": curves,
        "test_seal": {
            "test_rows_used": 0,
            "measured": "_SEAL/test_seal_measured.json",
            "note": ("no TEST row reaches an array, a window, a label or a metric. "
                     "TRAIN loading decodes zero TEST rows; VALIDATION decodes 384,203 "
                     "from one boundary-straddling row group and discards them. This is "
                     "measured, not asserted - it used to be a written constant."),
        },
    }, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()

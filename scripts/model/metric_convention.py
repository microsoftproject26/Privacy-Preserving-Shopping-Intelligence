"""Does the T1 conclusion depend on which slice convention is frozen?

`S2-PR-07` shipped the frozen T1 harness and it computes ranks from raw scores. This lane
suppresses the current category first, on the argument that the transition-table baseline is
evaluated off-diagonal on the category-change slice and a model denied the same information
spends rank 1 on an answer the slice definition has already excluded.

Scoring our model through the frozen harness and comparing it to our published baseline
gives `0.2174` against `0.3143`, which reads as a GRU losing to a transition table. That is
a convention mismatch, and the tempting responses are both wrong: adding suppression to the
frozen harness settles it by changing someone else's contract, and dropping ours settles it
by discarding an argument nobody has refuted.

**The question can be answered instead of argued.** Score the baseline and the model under
*each* convention and compare like with like. If the gain survives both, the choice of
convention does not affect the conclusion and nobody has to win.

Nothing here changes any contract. It measures two, one of which we do not own.
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
from ppsi.models.evaluation import (micro_and_macro, rank_of_truth, reciprocal_rank,
                                    transition_table)
from ppsi.models.session_gru import build_model

PUBLISHED_BASELINE = 0.3143
PUBLISHED_MODEL = 0.3479          # seed 13; the three-seed mean is 0.3474


def baseline_scores(train_split: Split, validation: Split) -> np.ndarray:
    """The frozen transition table, as a dense score row per validation decision."""
    table = transition_table(train_split.data["query_category"],
                             train_split.data["target"], CATEGORIES)
    current = validation.current_category
    empty = np.zeros(CATEGORIES, dtype="float64")
    return np.where(current[:, None] >= 0, table[np.clip(current, 0, None)], empty[None, :])


def model_scores(validation: Split, checkpoint: Path) -> np.ndarray:
    config, _, _ = selected_config()
    model = build_model(13, batch_spec=SPEC, config=config)
    load_encoder(model, checkpoint)
    model = model.to(DEVICE).eval()
    out = np.zeros((len(validation), CATEGORIES), dtype="float32")
    with torch.no_grad():
        for start in range(0, len(validation), 2048):
            rows = np.arange(start, min(start + 2048, len(validation)))
            out[rows] = model(to_batch(validation, rows).to(DEVICE)).t1_logits.float().cpu()
    del model
    return out


def slice_macro(scores: np.ndarray, validation: Split, *, suppress: bool) -> float:
    current = validation.current_category
    truth = validation.data["target"]
    changed = current != truth
    ranks = np.zeros(len(validation), dtype="int64")
    for start in range(0, len(validation), 8192):
        rows = np.arange(start, min(start + 8192, len(validation)))
        ranks[rows] = rank_of_truth(scores[rows], truth[rows],
                                    suppress=current[rows] if suppress else None)
    _, macro = micro_and_macro(reciprocal_rank(ranks), validation.data["client"], changed)
    return float(macro)


def main() -> None:
    started = time.time()
    train_split, validation = Split.load("train"), Split.load("validation")
    checkpoint = OUTPUT / "s2_ds_01_gru_t1_seed13.pt"
    print(f"  {len(validation):,} VALIDATION decisions, checkpoint {checkpoint.name}\n")

    baseline = baseline_scores(train_split, validation)
    model = model_scores(validation, checkpoint)

    table = []
    for convention, suppress in (("suppressed (this lane)", True),
                                 ("raw (the frozen harness)", False)):
        base = slice_macro(baseline, validation, suppress=suppress)
        learned = slice_macro(model, validation, suppress=suppress)
        table.append({"convention": convention,
                      "baseline": round(base, 4), "model": round(learned, 4),
                      "gain": round(learned - base, 4),
                      "model_beats_baseline": bool(learned > base)})

    frame = pd.DataFrame(table)
    print(frame.to_string(index=False))

    # The published pair must reproduce, or this script is measuring something else.
    suppressed = table[0]
    assert abs(suppressed["baseline"] - PUBLISHED_BASELINE) < 0.002, (
        f"recomputed suppressed baseline {suppressed['baseline']} against a published "
        f"{PUBLISHED_BASELINE}; this is not scoring the rows the published numbers scored")
    assert abs(suppressed["model"] - PUBLISHED_MODEL) < 0.002, (
        f"recomputed suppressed model {suppressed['model']} against a published "
        f"{PUBLISHED_MODEL}")

    gains = [row["gain"] for row in table]
    robust = all(row["model_beats_baseline"] for row in table)
    print(f"\n  the model beats the baseline under every convention: {robust}")
    print(f"  gain ranges from {min(gains):+.4f} to {max(gains):+.4f}")
    if robust:
        print("\n  So the conclusion does not depend on the convention, and the two lanes")
        print("  do not need to agree on one before either can publish. What they must")
        print("  never do is quote a baseline from one and a model from the other.")

    (OUTPUT / "metric_convention.json").write_text(json.dumps({
        "task": "S2-DS-01",
        "question": "does the T1 conclusion depend on the slice convention?",
        "conventions": table,
        "conclusion_is_convention_independent": robust,
        "min_gain": round(min(gains), 4), "max_gain": round(max(gains), 4),
        "published_pair": {"baseline": PUBLISHED_BASELINE, "model": PUBLISHED_MODEL},
        "warning": ("mixing the two - a baseline measured under one convention against a "
                    "model measured under the other - gives 0.2174 against 0.3143 and reads "
                    "as the model losing. That is the only unsafe combination."),
    }, indent=2), encoding="utf-8")
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

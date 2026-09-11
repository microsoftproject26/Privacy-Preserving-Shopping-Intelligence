"""S2-DS-05: is the GRU the right core, or just the one we reached for first?

The board's wording is the point of the task: *"we want the final architecture to be
justified by quality-efficiency trade-offs rather than chosen by habit."* So this swaps the
sequence core and changes nothing else - same data, same split, same T1 evaluator, same
training budget - and reports quality beside cost.

| core | what it is |
|---|---|
| **GRU** | the mandatory reference, and the architecture every published number came from |
| **LSTM** | the obvious recurrent alternative: a second gate and a cell state |
| **TCN** | causal dilated convolutions - no recurrence, so the whole sequence is parallel |
| **transformer** | a small SASRec-shaped encoder with learned positions and causal masking |

## What is deliberately *not* equalised

**Parameter count.** Each core gets the same hidden width and the same layer count, and
whatever parameters that implies. Equalising capacity instead would mean tuning each
architecture separately, and then the comparison measures the tuning. The counts are
reported because they are part of the answer: a core that wins by being larger has not won
the question `S2-SE-02` is going to ask.

**Per-architecture hyperparameters.** One learning rate, one schedule, one epoch count, taken
from the GRU's own selected configuration. This is a time-boxed comparison, not four tuning
exercises, and the box is what the board asked for.

That biases the result *toward the GRU*, whose budget this is. The write-up says so rather
than presenting the ranking as if every core had been given its best shot.

## The gate

The GRU rung must reproduce `0.3479` - the published seed-13 number. If it does not, this
harness is not scoring what `S2-DS-01` scored and no comparison below it means anything.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
TASK = Path(__file__).resolve().parent
OUTPUT = TASK / "output"
OUTPUT.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJECT / "S2-DS-01_GRU_T1_Model"))

from finalize import selected_config
from train import CATEGORIES, SPEC, Split, score, train

from ppsi.models.checkpoint import save as save_checkpoint
from ppsi.models.session_gru import SEQUENCE_CORES, build_model, parameter_count

SEED = 13
EPOCHS = 30
BATCH = 512
GRU_REFERENCE = 0.3479        # S2-DS-01, seed 13, slice macro MRR@20
BASELINE = 0.3143             # the transition table, same slice, same suppression


def main() -> None:
    started = time.time()
    config, learning_rate, _ = selected_config()
    train_split, validation = Split.load("train"), Split.load("validation")
    rows = np.arange(len(train_split))
    print(f"  {len(train_split):,} TRAIN decisions, {len(validation):,} VALIDATION")
    print(f"  budget: {EPOCHS} epochs, batch {BATCH}, lr {learning_rate}, cosine schedule")
    print(f"  channels: {config.channels}, hidden {config.hidden}, layers {config.layers}")
    print(f"  reference: GRU {GRU_REFERENCE}, baseline {BASELINE}\n")

    results, curves = [], {}
    for core in SEQUENCE_CORES:
        # One field changes and every other stays whatever S2-DS-01 selected. Rebuilding
        # the config by hand would let a width or a channel drift between cores, and the
        # comparison would then be of four different models rather than four cores.
        variant = dataclasses.replace(config, core=core)
        model = build_model(SEED, batch_spec=SPEC, config=variant)
        parameters = parameter_count(model)
        del model

        core_started = time.time()
        model, curve, best_epoch = train(
            train_split, validation, config=variant, seed=SEED, rows=rows,
            batch_size=BATCH, learning_rate=learning_rate, max_epochs=EPOCHS,
            patience=EPOCHS, label=f"{core}  |  {parameters:,} parameters",
            schedule="cosine")
        minutes = (time.time() - core_started) / 60

        measured = score(model, validation, legacy_clamp=False,
                         popularity=np.zeros(CATEGORIES))
        row = {"core": core, "parameters": parameters,
               "slice_macro": measured["slice_macro"],
               "slice_micro": measured["slice_micro"],
               "overall_micro": measured.get("overall_micro"),
               "gain_over_baseline": round(measured["slice_macro"] - BASELINE, 4),
               "best_epoch": best_epoch,
               "train_minutes": round(minutes, 1),
               "minutes_per_point_of_gain": round(
                   minutes / max(measured["slice_macro"] - BASELINE, 1e-9) / 100, 1)}
        results.append(row)
        curves[core] = curve
        save_checkpoint(OUTPUT / f"t1_{core}_seed{SEED}.pt", model=model, spec=SPEC,
                        seed=SEED, core=core,
                        artifacts=PROJECT / "S1-D1-DS-07_T3_Protocol_and_Task_Examples/output")
        print(f"    {core}: slice macro {measured['slice_macro']:.4f}   "
              f"{parameters:,} parameters   {minutes:.0f} min\n")
        del model

        if core == "gru":
            # The gate. Everything below is measured against the GRU, so if the GRU itself
            # is not the published model, the ranking is a ranking of something else.
            drift = abs(measured["slice_macro"] - GRU_REFERENCE)
            if drift > 0.002:
                raise SystemExit(
                    f"the GRU rung scores {measured['slice_macro']:.4f} against the "
                    f"published {GRU_REFERENCE}. This harness is not scoring what "
                    "S2-DS-01 scored, so no comparison below it would mean anything.")
            print(f"  gate ok: the GRU reproduces {GRU_REFERENCE} (drift {drift:.4f})\n")

    frame = pd.DataFrame(results).sort_values("slice_macro", ascending=False)
    print("=" * 82)
    print(frame.to_string(index=False))

    best = frame.iloc[0]
    reference = next(r for r in results if r["core"] == "gru")
    margin = float(best["slice_macro"]) - reference["slice_macro"]
    decisive = bool(margin > 0.0007)     # T1's own three-seed spread

    print(f"\n  best: {best['core']} at {best['slice_macro']:.4f}")
    print(f"  over the GRU by {margin:+.4f}, against a three-seed spread of 0.0007")
    print(f"  -> {'a real difference' if decisive else 'INSIDE seed noise - not a ranking'}")

    (OUTPUT / "s2_ds_05_architectures.json").write_text(json.dumps({
        "task": "S2-DS-05", "question": "is the GRU the right core?",
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH,
        "learning_rate": learning_rate,
        "budget_note": ("one budget for every core, taken from the GRU's own selected "
                        "configuration. This biases the comparison toward the GRU and is "
                        "reported rather than corrected, because per-architecture tuning "
                        "would make the result a comparison of tuning effort."),
        "gru_reference": GRU_REFERENCE, "baseline": BASELINE,
        "seed_spread_on_t1": 0.0007,
        "results": results,
        "best_core": best["core"],
        "margin_over_gru": round(margin, 4),
        "margin_exceeds_seed_spread": decisive,
        "verdict": ("the ranking is inside single-seed noise and does not support changing "
                    "the architecture" if not decisive else
                    f"{best['core']} beats the GRU by more than the seed spread"),
        "curves": curves,
        "test_seal": {"test_rows_used": 0, "measured": "_SEAL/test_seal_measured.json"},
    }, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()

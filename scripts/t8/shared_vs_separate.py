"""S2-DS-ST1: is one shared encoder better than three separate models, or just smaller?

The proposal promises a shared backbone. Everything measured so far compares *ways of
sharing* - frozen, sequential, joint - and never asks the prior question: **would a model
built only for T2, owing nothing to T1, do better?**

If it would, sharing costs quality and the argument for it is purely about size. If it
would not, T1 pretraining is doing real work for T2 and sharing is free or better.

| arrangement | T1 | T2 | encoders on the device |
|---|---|---|---|
| shared, frozen encoder | 0.3479 | 0.1104 | 1 |
| shared, sequential fine-tune | 0.1791 | 0.1424 | 1 |
| shared, joint objective | 0.3467 | 0.1337 | 1 |
| **separate T2 model, this run** | *unaffected by construction* | **?** | **2** |

The separate model gets the same architecture, the same data, the same budget and the same
seed. Only its starting weights differ: random instead of the trained T1 encoder.

## Why the storage column matters as much as the score

Three separate models is three sets of encoder weights to ship, hold in memory and update on
a phone. The shared encoder is `2,379,263` parameters and the T2 head is `898,113` of them,
so separate models roughly double what the device carries for two tasks - before T3. A
separate model that wins by `0.002` has not made the case for that.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "S2-DS-01_GRU_T1_Model"))
sys.path.insert(0, str(PROJECT / "S2-DS-06_T2_Purchase_Head"))

import train_t2 as t2
from finalize import selected_config

from ppsi.models.checkpoint import save as save_checkpoint
from ppsi.models.session_gru import build_model, parameter_count

OUTPUT = Path(__file__).resolve().parent / "output"
SEED = 13
EPOCHS = 12          # the same budget S2-DS-06 gave the shared model
BATCH = 512
LEARNING_RATE = 0.001
T2_BASELINE = 0.0757

SHARED = {
    "frozen encoder + T2 head": {"t1": 0.3479, "t2": 0.1104, "encoders": 1},
    "sequential fine-tune": {"t1": 0.1791, "t2": 0.1424, "encoders": 1},
    "joint objective, lambda 1.0": {"t1": 0.3467, "t2": 0.1337, "encoders": 1},
}


def main() -> None:
    started = time.time()
    config, _, _ = selected_config()
    train_split, validation = t2.Split.load("train"), t2.Split.load("validation")
    print(f"  T2 {len(train_split):,} TRAIN, {len(validation):,} VALIDATION")
    print(f"  budget {EPOCHS} epochs, batch {BATCH}, lr {LEARNING_RATE}, seed {SEED}\n")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # No `load_encoder`. That single omission is the whole experiment: same architecture,
    # same data, same budget, random weights instead of the trained T1 encoder.
    model = build_model(SEED, batch_spec=t2.SPEC, config=config).to(t2.DEVICE)
    print(f"  separate T2 model: {parameter_count(model):,} parameters, randomly initialised")

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    loss_function = nn.BCEWithLogitsLoss(reduction="sum")
    generator = np.random.default_rng(SEED)
    rows = train_split.trainable

    best, curve = -1.0, []
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.time()
        model.train()
        order = rows.copy()
        generator.shuffle(order)
        total, seen = 0.0, 0
        for start in range(0, len(order), BATCH):
            chunk = np.sort(order[start:start + BATCH])
            batch = t2.to_batch(train_split, chunk).to(t2.DEVICE)
            present = batch.t2_present
            if not present.any():
                continue
            logit = model(batch).t2_logit
            loss = (loss_function(logit[present], batch.t2_target[present])
                    / max(int(present.sum()), 1))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.item() * len(chunk)
            seen += len(chunk)
        scheduler.step()

        measured = t2.evaluate(model, validation)
        row = {"epoch": epoch, "train_loss": round(total / max(seen, 1), 5),
               "pr_auc": measured["pr_auc"], "roc_auc": measured["roc_auc"],
               "brier": measured["brier"],
               "seconds": round(time.time() - epoch_started, 1)}
        curve.append(row)
        marker = ""
        if measured["pr_auc"] > best:
            best = measured["pr_auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(f"    epoch {epoch:>2}  loss {row['train_loss']:.4f}  "
              f"PR-AUC {row['pr_auc']:.4f}  ({row['seconds']:.0f}s){marker}")

    model.load_state_dict(best_state)
    save_checkpoint(OUTPUT / "t2_separate_model.pt", model=model, spec=t2.SPEC,
                    seed=SEED, arrangement="separate, no shared encoder",
                    artifacts=PROJECT / "S1-D1-DS-07_T3_Protocol_and_Task_Examples/output")

    joint = SHARED["joint objective, lambda 1.0"]["t2"]
    frozen = SHARED["frozen encoder + T2 head"]["t2"]
    table = [{"arrangement": name, **facts,
              "t2_gain_over_baseline": round(facts["t2"] - T2_BASELINE, 4)}
             for name, facts in SHARED.items()]
    table.append({"arrangement": "separate T2 model", "t1": None, "t2": best,
                  "encoders": 2, "t2_gain_over_baseline": round(best - T2_BASELINE, 4)})

    print("\n" + "=" * 78)
    print(pd.DataFrame(table).to_string(index=False))

    against_joint = best - joint
    against_frozen = best - frozen
    print(f"\n  separate vs the joint shared model : {against_joint:+.4f}")
    print(f"  separate vs the frozen shared model: {against_frozen:+.4f}")
    verdict = (
        "a separate T2 model beats the shared encoder, so sharing costs T2 quality"
        if against_joint > 0.003 else
        "the shared encoder matches or beats a separate T2 model, so sharing is not a "
        "compromise on T2 - it is free or better, and it halves what the device carries")
    print(f"  -> {verdict}")

    (OUTPUT / "s2_ds_st1_shared_vs_separate.json").write_text(json.dumps({
        "task": "S2-DS-ST1", "status": "EXECUTED",
        "question": "is one shared encoder better than separate models, or just smaller?",
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH,
        "learning_rate": LEARNING_RATE, "t2_baseline": T2_BASELINE,
        "arrangements": table,
        "separate_minus_joint": round(against_joint, 4),
        "separate_minus_frozen": round(against_frozen, 4),
        "storage": {"shared_encoder_parameters": 2379263,
                    "t2_head_parameters": 898113,
                    "note": ("two separate models roughly double what a device carries for "
                             "two tasks, before T3 is considered at all")},
        "verdict": verdict,
        "negative_transfer_note": "S2-DS-06/output/s2_ds_06_negative_transfer.json",
        "test_seal": {"test_rows_used": 0, "measured": "_SEAL/test_seal_measured.json"},
        "curve": curve,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()

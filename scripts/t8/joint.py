"""S2-DS-08: can one encoder serve T1 and T2 at once, or does sharing have to cost?

This task used to be a refinement - pick weights for three losses that are already known to
work. `S2-DS-06` changed that. Fine-tuning the encoder for T2 alone takes T1 from `0.3479`
to `0.1791` slice macro and `0.8604` to `0.4728` overall micro, **below T1's own model-free
baseline on both**. So training the heads one after another destroys the shared
representation, and a joint objective is not an optimisation on top of a working arrangement.
It is the only thing that could make the arrangement exist.

## What is being compared

| | T1 | T2 |
|---|---|---|
| **control** - frozen encoder, T2 head only | 0.3479, untouched by construction | 0.1104 |
| **sequential** - fine-tune the encoder for T2 | **0.1791** | 0.1424 |
| **joint** - this task | ? | ? |

The control is not a weak option. It already beats the T2 baseline by `+0.0347` while costing
T1 exactly nothing, so joint training has to earn its place against a real alternative rather
than against doing nothing.

## Selection is Pareto-constrained, and the margin was fixed before the run

Choosing the epoch with the best T2 is what produced the sequential disaster: nothing in a
T2 metric can see T1 collapsing. So every epoch is scored on **both** tasks, and the
selection rule is:

> **maximise T2 PR-AUC, subject to T1 slice macro staying within `0.003` of `0.3479`.**

`0.003` comes from the review that specified this task, and it is close to T1's own
three-seed spread of `0.0007` - tight enough that a real regression cannot hide inside it.
An epoch that breaks the constraint is recorded and refused, not quietly dropped.

## Starting point

Training starts from the **trained** `S2-DS-01` checkpoint, not from scratch. That matters:
if the model started cold, T1 would be below `0.3479` simply from having had fewer epochs,
and there would be no way to separate that from interference. Starting at `0.3479` makes
every point of T1 loss attributable to the joint objective.
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
T1_TASK = PROJECT / "S2-DS-01_GRU_T1_Model"
T2_TASK = PROJECT / "S2-DS-06_T2_Purchase_Head"
sys.path.insert(0, str(T1_TASK))
sys.path.insert(0, str(T2_TASK))

import train as t1
import train_t2 as t2
from finalize import selected_config

from ppsi.models.checkpoint import load_encoder
from ppsi.models.checkpoint import save as save_checkpoint
from ppsi.models.session_gru import build_model, parameter_count

OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)

SEED = 13
EPOCHS = 8
T1_BATCH = 512
T2_BATCH = 512
LEARNING_RATE = 0.0003        # a continuation, not a fresh fit: smaller than either task used

T1_REFERENCE = 0.3479         # seed 13, the checkpoint this starts from
T1_MARGIN = 0.003             # pre-registered, from the round-2 review
T2_BASELINE = 0.0757          # smoothed item+category, corrected labels
T2_CONTROL = 0.1104           # frozen encoder + T2 head - the option joint must beat

# One change per rung: how much the T2 loss is allowed to pull the shared encoder.
WEIGHTS = (0.1, 0.3, 1.0)


def evaluate_both(model, t1_validation, t2_validation) -> dict:
    """Both tasks, every epoch. Scoring T2 alone is what let the sequential run fail unseen."""
    from ppsi.models.evaluation import micro_and_macro, rank_of_truth, reciprocal_rank

    model.eval()
    current = t1_validation.current_category
    truth = t1_validation.data["target"]
    changed = current != truth
    ranks = np.zeros(len(t1_validation), dtype="int64")
    with torch.no_grad():
        for start in range(0, len(t1_validation), 2048):
            rows = np.arange(start, min(start + 2048, len(t1_validation)))
            logits = model(t1.to_batch(t1_validation, rows).to(t1.DEVICE))
            ranks[rows] = rank_of_truth(logits.t1_logits.float().cpu().numpy(),
                                        truth[rows], suppress=current[rows])
    _, t1_macro = micro_and_macro(reciprocal_rank(ranks), t1_validation.data["client"],
                                  changed)

    measured = t2.evaluate(model, t2_validation)
    return {"t1_slice_macro": round(float(t1_macro), 4),
            "t2_pr_auc": measured["pr_auc"], "t2_roc_auc": measured["roc_auc"],
            "t1_cost": round(float(t1_macro) - T1_REFERENCE, 4),
            "t2_gain": round(measured["pr_auc"] - T2_BASELINE, 4),
            "within_t1_margin": bool(T1_REFERENCE - float(t1_macro) <= T1_MARGIN)}


def main() -> None:
    started = time.time()
    config, _, _ = selected_config()
    checkpoint = T1_TASK / "output" / f"s2_ds_01_gru_t1_seed{SEED}.pt"

    t1_train, t1_validation = t1.Split.load("train"), t1.Split.load("validation")
    t2_train, t2_validation = t2.Split.load("train"), t2.Split.load("validation")
    print(f"  T1 {len(t1_train):,} train / {len(t1_validation):,} validation")
    print(f"  T2 {len(t2_train):,} train / {len(t2_validation):,} validation")
    print(f"  starting from {checkpoint.name}\n")

    print("  the two alternatives joint training has to beat:")
    print(f"    frozen encoder + T2 head : T1 {T1_REFERENCE}  T2 {T2_CONTROL}")
    print("    sequential fine-tuning   : T1 0.1791  T2 0.1424   <- below T1's own baseline")
    print(f"  selection: maximise T2 subject to T1 >= {T1_REFERENCE - T1_MARGIN:.4f}\n")

    t1_rows = np.arange(len(t1_train))
    t2_rows = t2_train.trainable
    t1_loss_function = nn.CrossEntropyLoss()
    t2_loss_function = nn.BCEWithLogitsLoss(reduction="sum")

    results, curves = [], {}
    for weight in WEIGHTS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = build_model(SEED, batch_spec=t1.SPEC, config=config)
        load_encoder(model, checkpoint, new_prefixes=("t2_head", "query_"))
        model = model.to(t1.DEVICE)
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        generator = np.random.default_rng(SEED)

        at_start = evaluate_both(model, t1_validation, t2_validation)
        print(f"  lambda_t2 = {weight}  |  {parameter_count(model):,} parameters")
        print(f"    epoch  0  T1 {at_start['t1_slice_macro']:.4f}  "
              f"T2 {at_start['t2_pr_auc']:.4f}   (the T2 head is untrained here)")
        curve = [{"epoch": 0, **at_start}]

        best, best_row, best_state = -1.0, None, None
        for epoch in range(1, EPOCHS + 1):
            epoch_started = time.time()
            model.train()
            order1 = t1_rows.copy(); generator.shuffle(order1)
            order2 = t2_rows.copy(); generator.shuffle(order2)
            steps = max(len(order1) // T1_BATCH, 1)
            total1 = total2 = 0.0

            for step in range(steps):
                chunk1 = np.sort(order1[step * T1_BATCH:(step + 1) * T1_BATCH])
                # T2 has fewer trainable rows than T1 has decisions, so it cycles. Both
                # tasks then see every epoch rather than T2 running out part-way through.
                start2 = (step * T2_BATCH) % max(len(order2) - T2_BATCH, 1)
                chunk2 = np.sort(order2[start2:start2 + T2_BATCH])
                if len(chunk1) == 0 or len(chunk2) == 0:
                    continue

                batch1 = t1.to_batch(t1_train, chunk1).to(t1.DEVICE)
                batch2 = t2.to_batch(t2_train, chunk2).to(t2.DEVICE)
                loss1 = t1_loss_function(model(batch1).t1_logits, batch1.t1_target)
                logit2 = model(batch2).t2_logit
                present = batch2.t2_present
                loss2 = (t2_loss_function(logit2[present], batch2.t2_target[present])
                         / max(int(present.sum()), 1))

                optimiser.zero_grad(set_to_none=True)
                (loss1 + weight * loss2).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                total1 += loss1.item(); total2 += loss2.item()

            measured = evaluate_both(model, t1_validation, t2_validation)
            row = {"epoch": epoch, "t1_loss": round(total1 / steps, 4),
                   "t2_loss": round(total2 / steps, 4), **measured,
                   "seconds": round(time.time() - epoch_started, 1)}
            curve.append(row)

            marker = ""
            if measured["within_t1_margin"] and measured["t2_pr_auc"] > best:
                best, best_row = measured["t2_pr_auc"], row
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                marker = "  <- best within margin"
            elif not measured["within_t1_margin"]:
                marker = f"  refused: T1 cost {measured['t1_cost']:+.4f}"
            print(f"    epoch {epoch:>2}  T1 {measured['t1_slice_macro']:.4f}  "
                  f"T2 {measured['t2_pr_auc']:.4f}  ({row['seconds']:.0f}s){marker}")

        if best_row is None:
            print(f"    NO epoch kept T1 within {T1_MARGIN}. Joint training at this weight "
                  "cannot hold both tasks.\n")
            results.append({"lambda_t2": weight, "any_epoch_within_margin": False,
                            "best_t2_within_margin": None,
                            "best_t1_seen": max(r["t1_slice_macro"] for r in curve[1:]),
                            "best_t2_seen": max(r["t2_pr_auc"] for r in curve[1:])})
        else:
            beats_control = bool(best_row["t2_pr_auc"] > T2_CONTROL)
            results.append({"lambda_t2": weight, "any_epoch_within_margin": True,
                            "epoch": best_row["epoch"],
                            "t1_slice_macro": best_row["t1_slice_macro"],
                            "t1_cost": best_row["t1_cost"],
                            "best_t2_within_margin": best_row["t2_pr_auc"],
                            "t2_gain_over_baseline": best_row["t2_gain"],
                            "beats_the_frozen_control": beats_control})
            model.load_state_dict(best_state)
            save_checkpoint(OUTPUT / f"joint_lambda{weight}.pt", model=model,
                            spec=t1.SPEC, seed=SEED, lambda_t2=weight,
                            artifacts=PROJECT / "S1-D1-DS-07_T3_Protocol_and_Task_Examples/output")
            print(f"    best within margin: T2 {best_row['t2_pr_auc']:.4f} at epoch "
                  f"{best_row['epoch']}, T1 cost {best_row['t1_cost']:+.4f}   "
                  f"{'beats' if beats_control else 'LOSES TO'} the frozen control\n")
        curves[str(weight)] = curve
        del model

    frame = pd.DataFrame(results)
    print("=" * 78)
    print(frame.to_string(index=False))

    winners = [r for r in results if r.get("beats_the_frozen_control")]
    verdict = ("a joint objective holds both tasks and beats the frozen control"
               if winners else
               "no joint weight beat the frozen encoder + T2 adapter within the T1 margin")
    print(f"\n  -> {verdict}")

    (OUTPUT / "s2_ds_08_joint.json").write_text(json.dumps({
        "task": "S2-DS-08", "question": "can one encoder serve T1 and T2 at once?",
        "seed": SEED, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "started_from": checkpoint.name,
        "alternatives": {
            "frozen_encoder_t2_head": {"t1": T1_REFERENCE, "t2": T2_CONTROL},
            "sequential_finetune": {"t1": 0.1791, "t2": 0.1424,
                                    "note": "below T1's own model-free baseline of 0.3143"},
        },
        "selection": {"rule": "maximise T2 PR-AUC subject to T1 within the margin",
                      "t1_reference": T1_REFERENCE, "t1_margin": T1_MARGIN,
                      "fixed_before_the_run": True, "source": "round-2 review"},
        "t2_baseline": T2_BASELINE,
        "results": results, "verdict": verdict, "curves": curves,
        "test_seal": {"test_rows_used": 0, "measured": "_SEAL/test_seal_measured.json"},
    }, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()

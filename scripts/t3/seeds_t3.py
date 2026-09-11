"""Is "learning loses to the frozen retrieval order" a seed accident?

The ladder ran one seed and both losses came in below the baseline, with paired client
bootstrap intervals entirely on the wrong side of zero. That interval answers *would this
hold for different shoppers?* It says nothing about *would this hold for a different random
initialisation?* - every resample scored the same trained weights.

A negative result deserves the same scrutiny as a positive one, and arguably more, because
the tempting failure here is to accept it and move on. If one seed in three had beaten the
baseline, the honest report would be "sometimes it helps and we do not know when" rather
than "learning does not help".

Only the better loss is re-run. Pointwise lost by 0.0521 and collapsed to 0.0333 on its
first epoch; spending forty minutes confirming that a clearly broken configuration is still
broken buys nothing. Listwise lost by 0.0202, which is the number worth testing.

Every seed re-asserts the anchor: the untrained model must reproduce the frozen retrieval
order exactly before training starts. That check is what makes each run's result a statement
about training rather than about initialisation.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_t3 as L
from prepared import (
    anchors_per_decision,
    build_candidate_tables,
    build_gain_matrix,
    build_ideal_gains,
    clients_of_windows,
    evaluable_decisions,
    per_client_means,
)
from train_t3 import (
    BEST_SIMPLE_MACRO,
    CEILING_MACRO,
    DEVICE,
    LOSSES,
    OUTPUT,
    SPEC,
    Candidates,
    Ranking,
    build_batch,
    candidate_table,
    load_encoder,
    resolve,
)

from ppsi.models.evaluation import paired_client_bootstrap
from ppsi.models.session_gru import build_model

SEEDS = (13, 42, 2026)
LOSS = "listwise"


def main() -> None:
    started = time.time()
    protocol = json.loads(resolve("t3_protocol_v1.proposed.json").read_text(encoding="utf-8"))
    items, mask, index = candidate_table(protocol)
    candidates = Candidates(items, mask, index)
    tables = build_candidate_tables(candidates)

    train_examples = pd.read_parquet(
        resolve("INTERNAL_DO_NOT_UPLOAD_task_examples_t3_train_v1.proposed.parquet"))
    validation_examples = pd.read_parquet(
        resolve("INTERNAL_DO_NOT_UPLOAD_task_examples_t3_v1.proposed.parquet"))
    train_split = Ranking.load("train", train_examples)
    validation = Ranking.load("validation", validation_examples)

    train_anchor = anchors_per_decision(train_split, candidates)
    validation_anchor = anchors_per_decision(validation, candidates)
    train_gains = build_gain_matrix(train_split, candidates, train_anchor)
    validation_gains = build_gain_matrix(validation, candidates, validation_anchor)
    # Only the evaluator needs a full-oracle denominator; the losses read gains
    # directly, so TRAIN never builds one.
    validation_ideal = build_ideal_gains(validation)
    owner = clients_of_windows(validation, validation_examples)
    scored = evaluable_decisions(validation)

    baseline = L.retrieval_order_ndcg(validation_gains, validation_ideal, validation_anchor,
                                      tables, scored, owner)
    baseline_per_client = per_client_means(baseline["_per_query"], owner[scored])
    print(f"\n  baseline  macro {baseline['macro']:.4f}  micro {baseline['micro']:.4f}  "
          f"over {baseline['clients']:,} clients")

    checkpoint = resolve("s2_ds_01_gru_t1_seed13.pt")
    loss_function = LOSSES[LOSS]
    train_decisions = np.arange(len(train_split.data["lengths"]))

    rows, curves = [], {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(seed, batch_spec=SPEC, config=L.ENCODER)
        load_encoder(model, checkpoint)
        model = model.to(DEVICE)
        optimiser = torch.optim.Adam(model.parameters(), lr=L.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=L.EPOCHS)
        generator = np.random.default_rng(seed)

        at_init = L.evaluate(model, validation, validation_gains, validation_ideal,
                             validation_anchor, tables, scored, owner)
        assert abs(at_init["macro_ndcg@5"] - baseline["macro"]) < 1e-4, (
            f"seed {seed}: untrained model scores {at_init['macro_ndcg@5']:.4f} against a "
            f"baseline of {baseline['macro']:.4f}; the anchor is broken and this seed's "
            "result would not be a statement about training")
        print(f"\n  seed {seed}  anchored at {at_init['macro_ndcg@5']:.4f}")

        # Epoch 0 is a candidate: if no epoch beats it, the retrieval order wins outright.
        best = at_init["macro_ndcg@5"]
        best_per_query, best_epoch = at_init["_per_query"], 0
        curve = [{"epoch": 0, "train_loss": None,
                  **{k: v for k, v in at_init.items() if not k.startswith("_")}}]

        for epoch in range(1, L.EPOCHS + 1):
            epoch_started = time.time()
            model.train()
            order = train_decisions.copy()
            generator.shuffle(order)
            total, seen = 0.0, 0
            for start in range(0, len(order), L.BATCH):
                chunk = np.sort(order[start:start + L.BATCH])
                batch = build_batch(train_split, chunk, train_gains[chunk],
                                    train_anchor, tables).to(DEVICE)
                output = model(batch)
                loss = loss_function(output.t3_scores, batch.t3_gains, batch.candidate_mask)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                total += loss.item() * len(chunk)
                seen += len(chunk)
            scheduler.step()

            measured = L.evaluate(model, validation, validation_gains, validation_ideal,
                                  validation_anchor, tables, scored, owner)
            marker = ""
            if measured["macro_ndcg@5"] > best:
                best = measured["macro_ndcg@5"]
                best_per_query, best_epoch = measured["_per_query"], epoch
                marker = "  <- best"
            curve.append({"epoch": epoch, "train_loss": round(total / max(seen, 1), 5),
                          **{k: v for k, v in measured.items() if not k.startswith("_")},
                          "seconds": round(time.time() - epoch_started, 1)})
            print(f"    epoch {epoch:>2}  loss {curve[-1]['train_loss']:.4f}  "
                  f"macro {measured['macro_ndcg@5']:.4f}  "
                  f"({curve[-1]['seconds']:.0f}s){marker}")

        interval = paired_client_bootstrap(
            per_client_means(best_per_query, owner[scored]).to_numpy(),
            baseline_per_client.to_numpy())
        rows.append({"seed": seed, "macro_ndcg@5": best, "best_epoch": best_epoch,
                     "beat_the_baseline": bool(best_epoch > 0),
                     "gain": round(best - BEST_SIMPLE_MACRO, 4),
                     "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
                     "above_zero": interval["above_zero"]})
        curves[str(seed)] = curve
        print(f"    seed {seed}: best {best:.4f} at epoch {best_epoch}, "
              f"gain {best - BEST_SIMPLE_MACRO:+.4f}")
        del model

    print("\n" + "=" * 74)
    print(pd.DataFrame(rows).to_string(index=False))
    gains = np.array([r["gain"] for r in rows])
    any_beat = any(r["beat_the_baseline"] for r in rows)
    print(f"\n  mean gain {gains.mean():+.4f}   spread {gains.max() - gains.min():.4f}")
    print(f"  any seed beat the retrieval order: {any_beat}")

    (OUTPUT / "s2_ds_07_seeds.json").write_text(json.dumps({
        "task": "S2-DS-07", "loss": LOSS, "seeds": list(SEEDS),
        "baseline_macro": BEST_SIMPLE_MACRO, "ceiling_macro": CEILING_MACRO,
        "runs": rows,
        "mean_gain": round(float(gains.mean()), 4),
        "seed_spread": round(float(gains.max() - gains.min()), 4),
        "any_seed_beat_the_baseline": any_beat,
        "verdict": ("the frozen retrieval order beats the learned reranker on every seed"
                    if not any_beat else
                    "at least one seed beat the retrieval order; the result is seed dependent"),
        "note": ("pointwise is not re-run: it lost by 0.0521 and collapsed to 0.0333 on its "
                 "first epoch, so confirming it is still broken buys nothing."),
        "curves": curves,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()

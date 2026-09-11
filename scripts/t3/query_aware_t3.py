"""The T3 experiment that was never actually run: a reranker that can see the query.

Everything published for T3 so far measured `session . candidate`. The query item was
computed by the encoder and then used only by the T2 head - confirmed by permuting every
query tensor across a batch and watching T3 move by exactly `0.0` while T2 moved by `0.81`.

That is not a hard comparison, it is an impossible one. The frozen retrieval order **is**
co-occurrence between the query item and each candidate. A model that cannot see the query
cannot express the baseline, so "the learned reranker lost" was a statement about a dot
product and not about T3.

This run is bounded and its acceptance bar was fixed before it started, taken from the
review that found the defect:

> **macro gain >= +0.010, with the paired client interval entirely above zero.**
> Otherwise the frozen retrieval order is the T3 deliverable and the learned reranker is
> dropped from the model comparison.

Four deliberate choices, each with a reason:

* **The encoder is frozen.** Fine-tuning it for T2 drove T1 from `0.3479` to `0.1791`,
  below its own model-free baseline. Until `S2-DS-08` shows a joint objective can hold both
  tasks, no head gets to move the backbone. This also isolates the question: whatever this
  run measures is the reranker, not a re-fitted encoder.
* **Only scoring queries train.** `listwise_loss` already drops rows whose gains are all
  zero - `per_row[total > 0]` - so 92.5% of the 1.74M TRAIN queries were forward passes
  that contributed nothing. Restricting to the queries that carry a positive changes no
  gradient and makes an epoch roughly sixteen times cheaper, which is what buys the seeds.
* **The rank prior is fixed, not learned.** `t3_rank_weight` stays at `-1`, so the frozen
  ordering cannot be un-learned and then partially re-learned; the cross head can only add.
* **Three matched seeds**, and every one of them re-asserts the anchor before training.
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

# Every TRAIN query item has a list now (`rebuild_candidates.py`), so the loss can see all
# 1.74M queries instead of the 6.14% a VALIDATION-derived anchor set left reachable. The
# frozen anchors inside the extended artifact are byte-identical, so the 0.2707 baseline and
# the 18,814 evaluable queries are exactly the ones the earlier runs used.
EXTENDED = True

SEEDS = (13, 42, 2026)
LOSS = "listwise"
EPOCHS = 30
BATCH = 1024
LEARNING_RATE = 0.001
ACCEPT_GAIN = 0.010          # fixed before the run, from the review
TRAINABLE = ("t3_", "candidate_")


def main() -> None:
    started = time.time()
    protocol = json.loads(resolve("t3_protocol_v1.proposed.json").read_text(encoding="utf-8"))
    items, mask, index = candidate_table(protocol, extended=EXTENDED)
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
    validation_ideal = build_ideal_gains(validation)
    owner = clients_of_windows(validation, validation_examples)
    scored = evaluable_decisions(validation)

    # The queries a listwise loss can actually learn from: a retrieved positive to rank
    # above the rest. The others were already contributing nothing.
    trainable_rows = np.flatnonzero(train_gains.max(axis=1) > 0)
    total_train = len(train_split.data["lengths"])
    print(f"\n  TRAIN queries            : {total_train:,}")
    print(f"  with a retrieved positive: {len(trainable_rows):,} "
          f"({len(trainable_rows) / total_train * 100:.2f}%)   <- the ones that train")

    baseline = L.retrieval_order_ndcg(validation_gains, validation_ideal, validation_anchor,
                                      tables, scored, owner)
    baseline_per_client = per_client_means(baseline["_per_query"], owner[scored])
    print(f"  baseline                 : macro {baseline['macro']:.4f}  "
          f"micro {baseline['micro']:.4f}  over {baseline['clients']:,} clients")
    print(f"  acceptance bar           : macro gain >= +{ACCEPT_GAIN}, CI above zero\n")

    checkpoint = resolve("s2_ds_01_gru_t1_seed13.pt")
    loss_function = LOSSES[LOSS]
    rows, curves = [], {}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(seed, batch_spec=SPEC, config=L.ENCODER)
        load_encoder(model, checkpoint)
        model = model.to(DEVICE)

        # Frozen encoder: only the reranker moves. Anything else would make this run a
        # measurement of a re-fitted backbone wearing a reranker's name.
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(TRAINABLE)
        # The prior is the baseline. Learning it would let the model drift off the anchor.
        model.t3_rank_weight.requires_grad = False
        training = [p for p in model.parameters() if p.requires_grad]
        print(f"  seed {seed}  |  {sum(p.numel() for p in training):,} trainable of "
              f"{sum(p.numel() for p in model.parameters()):,}")

        optimiser = torch.optim.Adam(training, lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
        generator = np.random.default_rng(seed)

        at_init = L.evaluate(model, validation, validation_gains, validation_ideal,
                             validation_anchor, tables, scored, owner)
        if abs(at_init["macro_ndcg@5"] - baseline["macro"]) >= 1e-4:
            raise SystemExit(
                f"seed {seed}: untrained model scores {at_init['macro_ndcg@5']:.4f} against "
                f"a baseline of {baseline['macro']:.4f}. The cross head is not zero at "
                "initialisation, so epoch 0 is not the retrieval order and nothing measured "
                "afterwards is a departure from it.")
        print(f"    anchored at {at_init['macro_ndcg@5']:.4f}")

        best = at_init["macro_ndcg@5"]
        best_per_query, best_epoch = at_init["_per_query"], 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        curve = [{"epoch": 0, "train_loss": None,
                  **{k: v for k, v in at_init.items() if not k.startswith("_")}}]

        for epoch in range(1, EPOCHS + 1):
            epoch_started = time.time()
            model.train()
            order = trainable_rows.copy()
            generator.shuffle(order)
            total, seen = 0.0, 0
            for start in range(0, len(order), BATCH):
                chunk = np.sort(order[start:start + BATCH])
                batch = build_batch(train_split, chunk, train_gains[chunk],
                                    train_anchor, tables).to(DEVICE)
                output = model(batch)
                loss = loss_function(output.t3_scores, batch.t3_gains, batch.candidate_mask)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(training, 1.0)
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
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                marker = "  <- best"
            curve.append({"epoch": epoch, "train_loss": round(total / max(seen, 1), 5),
                          **{k: v for k, v in measured.items() if not k.startswith("_")},
                          "seconds": round(time.time() - epoch_started, 1)})
            print(f"    epoch {epoch:>2}  loss {curve[-1]['train_loss']:.4f}  "
                  f"macro {measured['macro_ndcg@5']:.4f}  "
                  f"micro {measured['micro_ndcg@5']:.4f}  "
                  f"({curve[-1]['seconds']:.0f}s){marker}")

        interval = paired_client_bootstrap(
            per_client_means(best_per_query, owner[scored]).to_numpy(),
            baseline_per_client.to_numpy())
        gain = best - BEST_SIMPLE_MACRO
        accepted = bool(gain >= ACCEPT_GAIN and interval["above_zero"])
        rows.append({"seed": seed, "macro_ndcg@5": best, "best_epoch": best_epoch,
                     "beat_the_baseline": bool(best_epoch > 0),
                     "gain": round(gain, 4), "ci_low": interval["ci_low"],
                     "ci_high": interval["ci_high"], "above_zero": interval["above_zero"],
                     "meets_acceptance": accepted})
        curves[str(seed)] = curve
        model.load_state_dict(best_state)
        print(f"    best {best:.4f} at epoch {best_epoch}   gain {gain:+.4f}   "
              f"CI [{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]   "
              f"{'ACCEPTED' if accepted else 'below the bar'}\n")
        del model

    frame = pd.DataFrame(rows)
    print("=" * 78)
    print(frame.to_string(index=False))
    gains = np.array([r["gain"] for r in rows])
    all_accept = bool(frame["meets_acceptance"].all())
    print(f"\n  mean gain {gains.mean():+.4f}   spread {gains.max() - gains.min():.4f}")
    print(f"  every seed meets the bar: {all_accept}")
    # One line, not two: a line break inside an f-string replacement field is Python 3.12+
    # syntax and this repository pins 3.11. It parses on the machine it was written on and
    # fails in CI, which is the worst place to find out.
    outcome = ("the query-aware reranker is the T3 deliverable" if all_accept
               else "the frozen retrieval order remains the T3 deliverable")
    print(f"  -> {outcome}")

    (OUTPUT / "s2_ds_07_query_aware.json").write_text(json.dumps({
        "task": "S2-DS-07", "experiment": "query-aware cross-feature residual reranker",
        "loss": LOSS, "seeds": list(SEEDS), "epochs": EPOCHS,
        "encoder": "frozen", "rank_prior": "fixed at -1",
        "train_queries_total": int(total_train),
        "train_queries_used": len(trainable_rows),
        "baseline_macro": BEST_SIMPLE_MACRO, "ceiling_macro": CEILING_MACRO,
        "acceptance": {"min_gain": ACCEPT_GAIN, "interval_above_zero": True,
                       "fixed_before_the_run": True, "source": "round-2 review"},
        "runs": rows,
        "mean_gain": round(float(gains.mean()), 4),
        "seed_spread": round(float(gains.max() - gains.min()), 4),
        "meets_acceptance_on_every_seed": all_accept,
        "verdict": ("the query-aware reranker beats the frozen retrieval order"
                    if all_accept else
                    "the frozen retrieval order remains the T3 deliverable"),
        "curves": curves,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()

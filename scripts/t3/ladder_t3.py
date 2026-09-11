"""S2-DS-07: train the ranking head, and choose the loss by measuring it.

Nothing about the T3 loss is frozen upstream. `ContractSmokeObjective` uses squared error
and says outright it is not the final choice, so the two candidates run under identical
budgets and the loser stays in the report:

* **listwise** - softmax cross-entropy over the candidate set, weighted by gain. Pushes a
  graded candidate above the others *in its own list*, which is what NDCG rewards.
* **pointwise** - squared error against the gain. Pushes every score toward its own target
  in isolation, which is not the same thing and may not be the same ordering.

Prediction, stated before the runs: **listwise wins**, because NDCG scores an ordering and
pointwise never sees one. If pointwise wins instead, the gains are separable enough that
absolute calibration carries the ranking, and `S2-DS-08` can use a simpler loss.

Three properties of T3 drive the implementation, and each cost a run to learn:

* gains follow `INTENT` - purchase 2, cart 1, **view 0** - so 98.1% of rows score nothing.
  Training uses every row; evaluation counts only the 18,814 queries with a scoring
  positive, because that is what the published 0.2046 counts.
* the ceiling is **0.8201**, not 1.0: 18.4% of scoring positives were never retrieved, and
  end-to-end NDCG gives those zero rather than hiding them.
* **the candidates must carry real features, including the retrieval rank.** The first
  version filled category and price band with pad ids and gave no rank at all, so the model
  was asked to beat a popularity ordering while blind to popularity. It scored 0.0996
  against a 0.2046 baseline. `validate_canonical_phase1_batch` rejects a pad id in a valid
  position and now runs before training starts.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from prepared import (
    anchors_per_decision,
    build_candidate_tables,
    build_gain_matrix,
    build_ideal_gains,
    clients_of_windows,
    evaluable_decisions,
    macro_by_client,
    per_client_means,
)
from train_t3 import (
    BEST_SIMPLE,
    BEST_SIMPLE_MACRO,
    BEST_SIMPLE_MICRO,
    CEILING,
    CEILING_MACRO,
    CEILING_MICRO,
    TOLERANCE,
    DEVICE,
    HALF_WIDTH,
    LOSSES,
    OUTPUT,
    SPEC,
    Candidates,
    Ranking,
    build_batch,
    candidate_table,
    load_encoder,
    ndcg_at_k,
    resolve,
)
from ppsi.models.evaluation import paired_client_bootstrap
from ppsi.models.checkpoint import save as save_checkpoint
from ppsi.models.session_gru import SessionGRUConfig, build_model
from ppsi.training.batch import validate_canonical_phase1_batch

SEED = 13
ENCODER = SessionGRUConfig(channels=("category_id", "event_type_id"), use_gap=True,
                           dropout=0.3)
EPOCHS = 16
LEARNING_RATE = 0.001
BATCH = 1024


def evaluate(model, split: Ranking, gains: np.ndarray, ideal: np.ndarray,
             anchor_of: np.ndarray, tables, decisions: np.ndarray,
             owner: np.ndarray, *, batch_size: int = 2048) -> dict:
    """Macro first, because `ADR-001` says macro is the headline; micro printed beside it."""
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(decisions), batch_size):
            chunk = decisions[start:start + batch_size]
            batch = build_batch(split, chunk, gains[chunk], anchor_of, tables)
            scores = model(batch.to(DEVICE)).t3_scores.float().cpu().numpy()
            parts.append(ndcg_at_k(scores, gains[chunk], batch.candidate_mask.numpy(),
                                   ideal[chunk], k=5))
    ndcg = np.concatenate(parts)
    macro, clients = macro_by_client(ndcg, owner[decisions])
    return {"macro_ndcg@5": round(macro, 4), "micro_ndcg@5": round(float(ndcg.mean()), 4),
            "queries": int(len(ndcg)), "clients": clients, "_per_query": ndcg}


def retrieval_order_ndcg(gains: np.ndarray, ideal: np.ndarray, anchor_of: np.ndarray,
                         tables, decisions: np.ndarray, owner: np.ndarray) -> dict:
    """What the frozen retrieval order alone scores - the published 0.2046.

    The calibration this task was missing the first time round. Asserting the count of
    evaluable queries and the candidate recall does not prove the metric agrees with
    upstream; scoring the frozen ordering with our own NDCG does.
    """
    anchors = anchor_of[decisions]
    mask = np.where((anchors >= 0)[:, None], tables.mask[np.maximum(anchors, 0)], False)
    order = -np.tile(tables.rank, (len(decisions), 1))
    ndcg = ndcg_at_k(order, gains[decisions], mask, ideal[decisions], k=5)
    macro, clients = macro_by_client(ndcg, owner[decisions])
    return {"macro": macro, "micro": float(ndcg.mean()), "clients": clients,
            "_per_query": ndcg}


def main() -> None:
    started = time.time()
    protocol = json.loads(resolve("t3_protocol_v1.proposed.json").read_text(encoding="utf-8"))
    print(f"retrieval : {protocol['retrieval']['selected']}   device: {DEVICE}")

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
    train_ideal = build_ideal_gains(train_split)
    validation_ideal = build_ideal_gains(validation)
    validation_owner = clients_of_windows(validation, validation_examples)

    scored = evaluable_decisions(validation)
    print(f"\n  evaluable queries: {len(scored):,}   (published 18,814)")
    assert len(scored) == 18_814, (
        f"{len(scored):,} evaluable queries against 18,814; the gain rule or the "
        "eligibility is not the frozen one")

    # The gate that was missing: does our metric reproduce the published baseline?
    baseline = retrieval_order_ndcg(validation_gains, validation_ideal, validation_anchor,
                                    tables, scored, validation_owner)
    print(f"  frozen retrieval order:  macro {baseline['macro']:.4f}   "
          f"micro {baseline['micro']:.4f}   over {baseline['clients']:,} clients")
    assert abs(baseline["micro"] - BEST_SIMPLE_MICRO) < TOLERANCE, (
        f"our NDCG gives the frozen ordering {baseline['micro']:.4f} micro against a "
        f"published {BEST_SIMPLE_MICRO}; the metric disagrees with upstream")
    assert abs(baseline["macro"] - BEST_SIMPLE_MACRO) < TOLERANCE, (
        f"macro baseline {baseline['macro']:.4f} against {BEST_SIMPLE_MACRO}")
    assert baseline["clients"] == 4_591, f"{baseline['clients']:,} clients against 4,591"
    print("  ok  metric agrees with upstream on both averages")
    baseline_per_client = per_client_means(baseline["_per_query"], validation_owner[scored])

    # The check that would have caught candidates filled with pad ids.
    probe = build_batch(validation, scored[:256], validation_gains[scored[:256]],
                        validation_anchor, tables)
    validate_canonical_phase1_batch(probe, SPEC)
    distinct = len(probe.candidate_categorical_ids["candidate_category_id"].unique())
    assert distinct > 2, (
        f"candidates carry only {distinct} distinct categories; they are not being filled "
        "from the catalogue")
    print(f"  ok  canonical validator passes, {distinct} distinct candidate categories")

    train_decisions = np.arange(len(train_split.data["lengths"]))
    checkpoint = resolve("s2_ds_01_gru_t1_seed13.pt")
    print(f"  encoder: {checkpoint.name}\n")

    results, curves = [], {}
    for name, loss_function in LOSSES.items():
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = build_model(SEED, batch_spec=SPEC, config=ENCODER)
        provenance = load_encoder(model, checkpoint)
        model = model.to(DEVICE)
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
        generator = np.random.default_rng(SEED)

        print(f"  {name}  |  {len(train_decisions):,} decisions, batch {BATCH}")

        # Epoch 0 must BE the baseline, not merely be near it. The model is a residual
        # reranker anchored at the frozen retrieval order, so before a single step it
        # should reproduce that order exactly. If it does not, the rank prior is wired
        # wrong and every gain measured afterwards is a gain over the wrong thing.
        at_init = evaluate(model, validation, validation_gains, validation_ideal,
                           validation_anchor, tables, scored, validation_owner)
        print(f"    epoch  0  (untrained)          macro {at_init['macro_ndcg@5']:.4f}  "
              f"micro {at_init['micro_ndcg@5']:.4f}")
        assert abs(at_init["macro_ndcg@5"] - baseline["macro"]) < 1e-4, (
            f"untrained model scores {at_init['macro_ndcg@5']:.4f} macro against a "
            f"retrieval baseline of {baseline['macro']:.4f}; the residual reranker is not "
            "anchored at the frozen ordering and no gain over it would be attributable")
        curve = [{"epoch": 0, "train_loss": None,
                   **{k: v for k, v in at_init.items() if not k.startswith("_")},
                   "seconds": 0.0}]

        # Epoch 0 IS a candidate, and this is not a technicality. The model starts at the
        # frozen retrieval order, so "no epoch beat epoch 0" means training never improved
        # on doing nothing - and the honest deliverable is then the retrieval order itself,
        # not the least-bad trained epoch. Seeding `best` from -1.0 would have selected a
        # trained model that scores below the baseline and shipped it as the T3 result.
        best = at_init["macro_ndcg@5"]
        best_micro = at_init["micro_ndcg@5"]
        best_per_query = at_init["_per_query"]
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_epoch = 0
        for epoch in range(1, EPOCHS + 1):
            epoch_started = time.time()
            model.train()
            order = train_decisions.copy()
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
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                total += loss.item() * len(chunk)
                seen += len(chunk)
            scheduler.step()

            measured = evaluate(model, validation, validation_gains, validation_ideal,
                                validation_anchor, tables, scored, validation_owner)
            row = {"epoch": epoch, "train_loss": round(total / max(seen, 1), 5),
                   **{k: v for k, v in measured.items() if not k.startswith("_")},
                   "seconds": round(time.time() - epoch_started, 1)}
            curve.append(row)
            marker = ""
            # Selection on macro, the declared headline - not on whichever looks better.
            if measured["macro_ndcg@5"] > best:
                best = measured["macro_ndcg@5"]
                best_micro = measured["micro_ndcg@5"]
                best_per_query = measured["_per_query"]
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                marker = "  <- best"
            print(f"    epoch {epoch:>2}  loss {row['train_loss']:.4f}  "
                  f"macro {row['macro_ndcg@5']:.4f}  micro {row['micro_ndcg@5']:.4f}  "
                  f"({row['seconds']:.0f}s){marker}")

        # The interval that answers the question a reader actually asks. Seed spread says
        # how much a re-run moves; this says whether the gain would survive a different
        # sample of shoppers, which is the claim being made. Paired, because model and
        # baseline are scored on the same clients and a hard client drags both down at once.
        interval = paired_client_bootstrap(
            per_client_means(best_per_query, validation_owner[scored]).to_numpy(),
            baseline_per_client.to_numpy())

        gain = best - BEST_SIMPLE_MACRO
        if best_epoch == 0:
            print(f"    NO EPOCH BEAT THE RETRIEVAL ORDER. Best is epoch 0, the untrained "
                  f"anchor at {best:.4f}. The deliverable for this loss is the frozen "
                  "retrieval ordering; the trained model is a loss, and it is reported.")
        results.append({"loss": name, "gain_interval": interval,
                        "best_epoch": best_epoch,
                        "beat_the_baseline": bool(best_epoch > 0),
                        "macro_ndcg@5": best, "micro_ndcg@5": best_micro,
                        "gain_over_baseline": round(gain, 4),
                        "share_of_room": round(gain / (CEILING_MACRO - BEST_SIMPLE_MACRO), 4),
                        "beats_noise": bool(gain > HALF_WIDTH),
                        "encoder": provenance})
        curves[name] = curve
        model.load_state_dict(best_state)
        save_checkpoint(OUTPUT / f"t3_{name}.pt", model=model, spec=SPEC,
                        seed=SEED, loss=name)
        print(f"    best macro NDCG@5 {best:.4f}   over baseline {gain:+.4f}   "
              f"ceiling {CEILING_MACRO}\n")
        del model

    print("=" * 74)
    print(pd.DataFrame([{k: v for k, v in r.items() if k != "encoder"}
                        for r in results]).to_string(index=False))
    (OUTPUT / "s2_ds_07_ladder.json").write_text(json.dumps({
        "task": "S2-DS-07", "retrieval": protocol["retrieval"]["selected"],
        "gain_rule": protocol["gain_rule"]["gains"],
        "headline": "macro_ndcg@5",
        "best_simple_macro": BEST_SIMPLE_MACRO, "best_simple_micro": BEST_SIMPLE_MICRO,
        "measured_baseline_macro": round(baseline["macro"], 4),
        "measured_baseline_micro": round(baseline["micro"], 4),
        "ceiling_macro": CEILING_MACRO, "ceiling_micro": CEILING_MICRO,
        "noise_floor": HALF_WIDTH,
        "evaluable_queries": int(len(scored)), "clients": baseline["clients"],
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

"""Where does the T3 headroom actually go?

The frozen retrieval order scores `0.2707` macro against a ceiling of `0.8795`. That gap
looks like room for a reranker, and two learned rerankers have now failed to take any of it.
Before a third attempt, it is worth knowing how much of the gap a reranker could ever reach.

Four things can cost a query its score, and only one of them is a reranking problem:

| | reachable by a reranker? |
|---|---|
| **no candidate list** for this query item at all | no - retrieval was never asked |
| **retrieval miss**: the engaged item is not in the list | no - it cannot be ranked into view |
| **ranked outside the top 5** although present | **yes** - this is the reranker's job |
| **ranked inside the top 5** already | no - nothing left to win |

Nothing here is trained, and nothing here is tuned. It is arithmetic over the frozen lists
and the frozen gains, and it settles whether "the learned reranker lost" means *the model
was weak* or *the contest was small*.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_t3 import MAX_CANDIDATES, Candidates, Ranking, candidate_table, resolve, OUTPUT
from prepared import (anchors_per_decision, build_candidate_tables, build_gain_matrix,
                      build_ideal_gains, clients_of_windows, evaluable_decisions,
                      per_client_means)
import ladder_t3 as L


def main() -> None:
    protocol = json.loads(resolve("t3_protocol_v1.proposed.json").read_text(encoding="utf-8"))
    items, mask, index = candidate_table(protocol)
    candidates = Candidates(items, mask, index)
    tables = build_candidate_tables(candidates)
    examples = pd.read_parquet(
        resolve("INTERNAL_DO_NOT_UPLOAD_task_examples_t3_v1.proposed.parquet"))
    validation = Ranking.load("validation", examples)
    anchor = anchors_per_decision(validation, candidates)
    gains = build_gain_matrix(validation, candidates, anchor)
    ideal = build_ideal_gains(validation)
    owner = clients_of_windows(validation, examples)
    scored = evaluable_decisions(validation)

    # Where each evaluable query stands under the frozen ordering.
    anchors = anchor[scored]
    has_list = anchors >= 0
    query_gains = gains[scored]
    retrieved = query_gains.max(axis=1) > 0

    # The frozen order is the candidate rank, so "position under retrieval" is just the
    # column index of the best-gaining candidate.
    best_column = np.where(retrieved, query_gains.argmax(axis=1), MAX_CANDIDATES)
    in_top5 = retrieved & (best_column < 5)
    below_top5 = retrieved & (best_column >= 5)

    buckets = {
        "no candidate list": int((~has_list).sum()),
        "retrieval miss (list exists, item absent)": int((has_list & ~retrieved).sum()),
        "present but ranked outside the top 5": int(below_top5.sum()),
        "already inside the top 5": int(in_top5.sum()),
    }
    total = len(scored)
    print(f"\n  evaluable queries: {total:,}\n")
    for name, count in buckets.items():
        print(f"    {name:<44} {count:>7,}  {count / total * 100:5.2f}%")

    reachable = int(below_top5.sum())
    print(f"\n  A reranker can only ever act on the third row: "
          f"{reachable:,} queries, {reachable / total * 100:.2f}%.")

    # What a *perfect* reranker of the retrieved lists would score - the real ceiling for
    # this contest, as opposed to the oracle ceiling that includes unreachable items.
    baseline = L.retrieval_order_ndcg(gains, ideal, anchor, tables, scored, owner)
    perfect_order = -np.tile(np.arange(MAX_CANDIDATES, dtype="float32"), (len(scored), 1))
    perfect_scores = np.where(query_gains > 0, 1e6 - np.arange(MAX_CANDIDATES), perfect_order)
    candidate_mask = np.where(has_list[:, None], tables.mask[np.maximum(anchors, 0)], False)
    from train_t3 import ndcg_at_k
    perfect = ndcg_at_k(perfect_scores, query_gains, candidate_mask, ideal[scored], k=5)
    perfect_macro, _ = per_client_means(perfect, owner[scored]).mean(), None

    room = perfect_macro - baseline["macro"]
    print(f"\n  frozen retrieval order      macro {baseline['macro']:.4f}")
    print(f"  a PERFECT reranking of the same lists  macro {perfect_macro:.4f}")
    print(f"  the whole contest is worth  {room:+.4f} macro")
    print(f"  best learned attempt so far            macro 0.2599   "
          f"({(0.2599 - baseline['macro']):+.4f})")

    (OUTPUT / "s2_ds_07_miss_decomposition.json").write_text(json.dumps({
        "task": "S2-DS-07", "question": "how much of the T3 gap can a reranker reach?",
        "evaluable_queries": total,
        "buckets": buckets,
        "reachable_by_a_reranker": reachable,
        "reachable_share": round(reachable / total, 4),
        "retrieval_order_macro": round(baseline["macro"], 4),
        "perfect_reranking_of_the_same_lists_macro": round(float(perfect_macro), 4),
        "contest_worth_macro": round(float(room), 4),
        "oracle_ceiling_macro": 0.8795,
        "note": ("the oracle ceiling of 0.8795 includes items retrieval never returned. No "
                 "reranker can reach those, so the number to compare a reranker against is "
                 "the perfect-reranking figure, not the oracle ceiling."),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

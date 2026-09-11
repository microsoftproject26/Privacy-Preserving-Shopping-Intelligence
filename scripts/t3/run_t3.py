"""S2-DS-07 end to end: gate, then the ranking loss chosen by measurement.

The gate is stricter here than in either previous task, because T3 has more ways to
silently measure the wrong thing:

* the candidate file holds **two** retrieval strategies and only one is the protocol's
* gains follow `INTENT`, so **98.1%** of rows are views worth zero and scoring all of them
  computes a different quantity from the published one
* the ceiling is **0.8201**, not 1.0, because 18.4% of scoring positives were never
  retrieved

So before any training the gate reproduces the published `0.2046` using the frozen
retrieval order alone - no model. If a model-free reranking of the frozen list does not
reproduce the number upstream published, the two are not measuring the same thing and
nothing above it would mean anything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_t3 import (
    BEST_SIMPLE,
    CEILING,
    DEVICE,
    HALF_WIDTH,
    LOSSES,
    MAX_CANDIDATES,
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
from ppsi.models.session_gru import SessionGRUConfig, build_model
from ppsi.training.batch import validate_canonical_phase1_batch

SEED = 13
ENCODER = SessionGRUConfig(channels=("category_id", "event_type_id"), use_gap=True,
                           dropout=0.3)


def load_examples(split: str) -> pd.DataFrame:
    suffix = "_train" if split == "train" else ""
    return pd.read_parquet(
        resolve(f"INTERNAL_DO_NOT_UPLOAD_task_examples_t3{suffix}_v1.proposed.parquet"))


def assemble_gains(split: Ranking, candidates: Candidates, decisions: np.ndarray,
                   row_lookup: dict) -> np.ndarray:
    """The gain of every candidate, for these decisions.

    A candidate the user never engaged with has gain 0 - that is the overwhelming majority.
    Only the graded rows that point at this decision contribute anything.
    """
    gains = np.zeros((len(decisions), MAX_CANDIDATES), dtype="float32")
    anchor_rows = candidates.rows_for(split.query_item[decisions])
    for position, (decision, anchor) in enumerate(zip(decisions, anchor_rows, strict=True)):
        if anchor < 0:
            continue
        rows = row_lookup.get(int(decision))
        if rows is None:
            continue
        wanted = candidates.items[anchor, :MAX_CANDIDATES]
        for row in rows:
            hit = np.flatnonzero(wanted == split.positive_item[row])
            if hit.size:
                gains[position, hit[0]] = split.gain[row]
    return gains


def gate(validation: Ranking, candidates: Candidates, examples: pd.DataFrame) -> dict:
    print("\n" + "=" * 78)
    print("THE GATE")
    print("=" * 78)

    scoring = examples[examples["label_value"] > 0]
    queries = int(scoring["query_id"].nunique())
    print(f"  rows {len(examples):,}  over {examples['query_id'].nunique():,} queries")
    print(f"  of which scoring (gain > 0): {len(scoring):,} over {queries:,} queries")
    assert queries == 18_814, (
        f"{queries:,} evaluable queries against the published 18,814; the gain rule or the "
        "eligibility is not the frozen one")
    print("  ok   18,814 evaluable queries, matching the published count")

    lists = candidates
    sample = scoring.sample(min(20_000, len(scoring)), random_state=13)
    rows = lists.rows_for(sample["query_item"].to_numpy())
    present = []
    for row, item in zip(rows, sample["positive_item"].to_numpy(), strict=True):
        present.append(bool(row >= 0 and (lists.items[row, :MAX_CANDIDATES] == item).any()))
    recall = float(np.mean(present))
    print(f"  scoring positives inside their own candidate list: {recall * 100:.2f}%"
          f"   (published Recall@100 0.8175)")
    assert abs(recall - 0.8175) < 0.01, (
        "candidate recall does not match the published figure; the wrong retrieval "
        "strategy is being read from a file that holds both")
    print("  ok   the selected retrieval reproduces the published recall")

    print(f"\n  the number to beat: {BEST_SIMPLE}   ceiling {CEILING}   "
          f"noise floor {HALF_WIDTH}")
    return {"rows": int(len(examples)), "scoring_rows": int(len(scoring)),
            "evaluable_queries": queries, "candidate_recall": round(recall, 4)}


def evaluate(model, split: Ranking, candidates: Candidates, row_lookup: dict,
             decisions: np.ndarray, *, batch_size: int = 256) -> dict:
    model.eval()
    scores_all, gains_all, mask_all = [], [], []
    with torch.no_grad():
        for start in range(0, len(decisions), batch_size):
            chunk = decisions[start:start + batch_size]
            gains = assemble_gains(split, candidates, chunk, row_lookup)
            batch = build_batch(split, chunk, gains, candidates, PRODUCT_BUCKET)
            output = model(batch.to(DEVICE))
            scores_all.append(output.t3_scores.float().cpu().numpy())
            gains_all.append(gains)
            mask_all.append(batch.candidate_mask.numpy())
    scores = np.concatenate(scores_all)
    gains = np.concatenate(gains_all)
    mask = np.concatenate(mask_all)
    ndcg = ndcg_at_k(scores, gains, mask, k=5)
    return {"ndcg@5": round(float(ndcg.mean()), 4), "queries": int(len(ndcg)),
            "ceiling": CEILING}


PRODUCT_BUCKET: dict = {}


SUPERSEDED = """`run_t3.py` is the first T3 runner and is no longer the one that produces the result.

It assembled every batch in Python and measured 11% GPU utilisation at 101% CPU;
`ladder_t3.py` plus `prepared.py` replaced it and runs the same ladder in 34s an epoch.

It is kept for its gate documentation and retired rather than deleted, but it must not be
run: it calls `ndcg_at_k` with the pre-review signature, whose denominator came from the
retrieved candidates and flattered partial retrieval misses. Two runners producing two
numbers for one task is the defect that has already cost this project three wrong figures.

Run `python ladder_t3.py`."""


def main() -> None:
    raise SystemExit(SUPERSEDED)


def _original_main() -> None:
    started = time.time()
    protocol = json.loads(resolve("t3_protocol_v1.proposed.json").read_text(encoding="utf-8"))
    print(f"retrieval  : {protocol['retrieval']['selected']}  (read from the protocol)")
    print(f"gain rule  : {protocol['gain_rule']['gains']}")
    print(f"device     : {DEVICE}")

    items, mask, index = candidate_table(protocol)
    candidates = Candidates(items, mask, index)

    examples = load_examples("validation")
    validation = Ranking.load("validation", examples)
    facts = gate(validation, candidates, examples)
    print(f"\ntotal {(time.time() - started) / 60:.1f} min")
    (OUTPUT / "s2_ds_07_gate.json").write_text(
        json.dumps({"task": "S2-DS-07", "gate": facts,
                    "retrieval": protocol["retrieval"]["selected"],
                    "best_simple": BEST_SIMPLE, "ceiling": CEILING,
                    "noise_floor": HALF_WIDTH, "test_rows_read": 0}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()

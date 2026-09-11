"""A confidence interval for the T2 gain, by resampling shoppers rather than seeds.

`HALF_WIDTH = 0.0029` is a seed spread: how far three runs of the same configuration on the
same 42,312 clients happened to land apart. It is a real quantity and it answers a real
question - *would I get this number again?* - but it is not the question a reader asks of a
headline. They ask whether the gain would survive a different sample of shoppers, and three
seeds cannot speak to that, because all three saw exactly the same people.

PR-AUC is not an average of per-client values, so it cannot be bootstrapped the way a macro
NDCG can. It has to be **recomputed** inside each resample. So: draw clients with
replacement, take all of that client's decisions, score both the model and the baseline on
that resample, and keep the difference. The interval is over those differences.

Pairing is the part that matters. Two separate intervals, one per system, would be far wider
and would answer a different question - they include the variation in how hard the resampled
shoppers are, which affects both systems identically. Differencing inside each resample
removes it.

Clients, not rows, are the unit. A client contributes many correlated decisions, and
resampling rows would treat them as independent evidence and produce an interval far too
narrow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_t2 import BEST_SIMPLE_CORRECTED, DEVICE, OUTPUT, SPEC, Split, to_batch

from ppsi.models.checkpoint import load_encoder as shared_load_encoder
from ppsi.models.session_gru import SessionGRUConfig, build_model

RESAMPLES = 1000
SEED = 13
ALPHA = 100.0
EXAMPLES = Path(__file__).resolve().parent.parent / (
    "S1-D1-DS-07_T3_Protocol_and_Task_Examples/output")


def model_scores(checkpoint: Path, split: Split, *, batch_size: int = 4096) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = SessionGRUConfig(channels=("category_id", "event_type_id"), use_gap=True,
                              dropout=0.3)
    model = build_model(payload.get("seed", SEED), batch_spec=SPEC, config=config)
    # The pre-correction checkpoint predates the candidate-rank channel, so its T3 head is
    # the wrong shape and the two residual-reranker parameters are absent. Neither is read
    # when scoring T2. The shared loader drops exactly those and raises if an **encoder**
    # tensor ever fails to load, which is the only way this comparison could go quietly
    # wrong: a silently randomised encoder would still produce a plausible PR-AUC.
    provenance = shared_load_encoder(model, checkpoint)
    assert not [k for k in provenance["new_parameters"]
                if not k.startswith(("t3", "candidate_"))], (
        f"{checkpoint.name}: T2 parameters are missing, not just T3 ones")
    model = model.to(DEVICE).eval()

    rows = split.trainable
    out = np.zeros(len(rows), dtype="float64")
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            batch = to_batch(split, chunk).to(DEVICE)
            out[start:start + len(chunk)] = (
                model(batch).t2_logit.squeeze(-1).float().cpu().numpy())
    return out


def smoothed_baseline() -> tuple:
    """The strongest simple rule: an item's TRAIN purchase rate, backed off to its category.

    This is the baseline the review found and we had not tried. Fitted on TRAIN only, on the
    **corrected** labels, so it and the model are measured on the same population.
    """
    train = pd.read_parquet(
        EXAMPLES / "INTERNAL_DO_NOT_UPLOAD_task_examples_t2_train_v1.proposed.parquet")
    validation = pd.read_parquet(
        EXAMPLES / "INTERNAL_DO_NOT_UPLOAD_task_examples_t2_v1.proposed.parquet")
    train["y"] = train["label_value"].fillna(0.0)
    validation["y"] = validation["label_value"].fillna(0.0)

    item = train.groupby("item")["y"].agg(["sum", "size"])
    category = train.groupby("category")["y"].mean()
    prior = float(train["y"].mean())

    backoff = validation["category"].map(category).fillna(prior).to_numpy()
    hits = validation["item"].map(item["sum"]).fillna(0.0).to_numpy()
    seen = validation["item"].map(item["size"]).fillna(0.0).to_numpy()
    scores = (hits + ALPHA * backoff) / (seen + ALPHA)
    return scores, validation["y"].to_numpy(), validation["client"].to_numpy()


def main() -> None:
    validation = Split.load("validation")
    checkpoint = OUTPUT / "t2_2.pt"
    print(f"checkpoint : {checkpoint.name}")

    baseline, labels, clients = smoothed_baseline()
    # The baseline is built from the parquet and the model is scored from the .npy cache.
    # Equal row counts prove nothing about equal ORDER, and a silent misalignment here
    # would pair every model score with some other decision's label - producing a number
    # that looks like a result. Checked on the labels themselves, elementwise.
    assert len(baseline) == len(validation), (
        f"{len(baseline):,} baseline rows against {len(validation):,} cached rows; the "
        "baseline and the model are not scoring the same decisions")
    assert np.array_equal(labels.astype("float32"), validation.label), (
        "the cache and the parquet disagree row by row; the baseline scores would be "
        "paired with the wrong decisions")

    model = model_scores(checkpoint, validation)
    rows = validation.trainable
    assert len(rows) == len(validation), "the corrected split should score every row"

    observed_model = average_precision_score(labels, model)
    observed_base = average_precision_score(labels, baseline)
    print(f"  model    PR-AUC {observed_model:.6f}")
    print(f"  baseline PR-AUC {observed_base:.6f}   (declared {BEST_SIMPLE_CORRECTED})")
    print(f"  gain            {observed_model - observed_base:+.6f}")

    # Rows grouped by client once, so a resample is a concatenation of whole clients.
    order = np.argsort(clients, kind="stable")
    boundaries = np.flatnonzero(np.diff(clients[order])) + 1
    groups = np.split(order, boundaries)
    print(f"\n  {len(groups):,} clients, {len(labels):,} decisions, "
          f"{RESAMPLES:,} resamples")

    generator = np.random.default_rng(SEED)
    differences = np.zeros(RESAMPLES)
    for draw in range(RESAMPLES):
        picked = generator.integers(0, len(groups), size=len(groups))
        index = np.concatenate([groups[i] for i in picked])
        y = labels[index]
        if y.sum() == 0:
            differences[draw] = 0.0
            continue
        differences[draw] = (average_precision_score(y, model[index])
                             - average_precision_score(y, baseline[index]))
        if (draw + 1) % 200 == 0:
            print(f"    {draw + 1:>5,} / {RESAMPLES:,}")

    low, high = np.percentile(differences, [2.5, 97.5])
    result = {
        "task": "S2-DS-06", "metric": "PR-AUC", "labels": "corrected",
        "checkpoint": checkpoint.name,
        "model_pr_auc": round(float(observed_model), 6),
        "baseline_pr_auc": round(float(observed_base), 6),
        "baseline": f"smoothed item+category, alpha={ALPHA:.0f}",
        "gain": round(float(observed_model - observed_base), 6),
        "ci_low": round(float(low), 6), "ci_high": round(float(high), 6),
        "half_width": round(float((high - low) / 2), 6),
        "above_zero": bool(low > 0),
        "clients": len(groups), "decisions": len(labels),
        "resamples": RESAMPLES,
        "method": ("paired client-cluster bootstrap; PR-AUC recomputed for both systems "
                   "inside every resample, difference taken within the resample"),
        "supersedes": ("half_width 0.0029, which was the max-minus-min of three seeds on "
                       "one fixed set of clients and is not a confidence interval"),
    }
    print(f"\n  gain {result['gain']:+.4f}   95% CI "
          f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]   "
          f"{'above zero' if result['above_zero'] else 'INCLUDES ZERO'}")
    (OUTPUT / "s2_ds_06_interval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

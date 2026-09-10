"""Train the shared encoder's T1 head and score it against the frozen baselines.

The loop is ours rather than `LocalTrainerCore`'s, because the core packs and
sha256-hashes the entire state dict on every step and this model carries a 1.2M-parameter
embedding table; at 6,081 steps an epoch that cost would dominate the run. The model still
satisfies `Phase1Model`, and a test proves one core step works, so the federated lane is
unaffected.

Nothing here decides what a decision is. The decisions and their labels come from the
frozen task examples, and the baseline they are compared against comes from
`task_headroom_v1`. Both are asserted, not assumed.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# These scripts live in two places: beside the task folder on the machine that ran them,
# and in `scripts/model/` in the repository. Rather than hard-code either layout, walk up
# until the directory holding the `ppsi` package appears.
PROJECT = Path(__file__).resolve().parent.parent
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "ppsi" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:  # the task-folder layout, where the package sits in a sibling checkout
    sys.path.insert(0, str(PROJECT / "Repo_S2DS01"))

from ppsi.models.batch_spec import CATEGORY_OOV, phase1_batch_spec_v1
from ppsi.models.evaluation import (
    micro_and_macro,
    rank_of_truth,
    reciprocal_rank,
)
from ppsi.models.session_gru import SessionGRUConfig, build_model, parameter_count
from ppsi.training.batch import Phase1Batch

TASK = Path(__file__).resolve().parent
CACHE = TASK / "cache"
OUTPUT = TASK / "output"
OUTPUT.mkdir(exist_ok=True)

CONFIG = json.loads((TASK / "config.json").read_text(encoding="utf-8"))
SPEC = phase1_batch_spec_v1()
CATEGORIES = 588
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARRAYS = (
    "category",
    "product",
    "event",
    "brand",
    "price_band",
    "gap",
    "lengths",
    "target",
    "query_category",
    "query_product",
    "query_brand",
    "query_price_band",
    "client",
)


def train_events_per_client() -> pd.Series:
    """TRAIN events per client - the strata definition ADR-001 actually specifies.

    Counting decisions instead shifts every client's bucket: a client with twenty events
    may produce only a dozen decisions, and the 10-19 bucket then holds 3,272 clients
    where the protocol says 2,924. The counts come from `client_events.py`, which reads
    the same events the windows were built from.
    """
    index = np.load(CACHE / "train_events_per_client_index.npy")
    values = np.load(CACHE / "train_events_per_client_values.npy")
    return pd.Series(values, index=index)


@dataclass(slots=True)
class Split:
    """One split's windows, held in RAM.

    1.4 GB for both splits, which fits and makes shuffled indexing fast. Left as
    memory-maps, every shuffled batch would be a scatter of random reads across a file.
    """

    name: str
    data: dict

    @classmethod
    def load(cls, name: str) -> Split:
        return cls(name, {key: np.load(CACHE / f"{name}_{key}.npy") for key in ARRAYS})

    def __len__(self) -> int:
        return len(self.data["lengths"])

    @property
    def current_category(self) -> np.ndarray:
        """The decision's own category, with -1 restored.

        OOV is produced only from -1, so the mapping inverts exactly. The raw value is
        what the slice definition and the off-diagonal suppression both need: suppressing
        class 588 would index past the logits.
        """
        query = self.data["query_category"]
        return np.where(query == CATEGORY_OOV, -1, query)


def to_batch(split: Split, rows: np.ndarray) -> Phase1Batch:
    """One Phase1Batch, built for speed rather than for validation.

    `ppsi.data.batching.windows_to_batch` runs the canonical validator, which walks every
    tensor. That is worth paying once per run and not 6,081 times per epoch, so the
    notebook validates the first batch through that path and the loop uses this one.
    """
    take = lambda key, dtype: torch.from_numpy(
        np.ascontiguousarray(split.data[key][rows]).astype(dtype)
    )
    lengths = take("lengths", "int64")
    size, width = len(rows), split.data["category"].shape[1]
    return Phase1Batch(
        history_categorical_ids={
            "category_id": take("category", "int64"),
            "product_bucket": take("product", "int64"),
            "event_type_id": take("event", "int64"),
            "brand_bucket": take("brand", "int64"),
            "price_band": take("price_band", "int64"),
        },
        history_continuous_features=take("gap", "float32").unsqueeze(-1),
        lengths=lengths,
        history_mask=torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1),
        query_categorical_ids={
            "query_category_id": take("query_category", "int64"),
            "query_product_bucket": take("query_product", "int64"),
            "query_brand_bucket": take("query_brand", "int64"),
            "query_price_band": take("query_price_band", "int64"),
        },
        query_continuous_features=torch.zeros(size, 0, dtype=torch.float32),
        candidate_ids=torch.zeros(size, 1, dtype=torch.int64),
        candidate_categorical_ids={
            "candidate_category_id": torch.full((size, 1), 589, dtype=torch.int64),
            "candidate_price_band": torch.full((size, 1), 5, dtype=torch.int64),
        },
        candidate_continuous_features=torch.zeros(size, 1, 0, dtype=torch.float32),
        candidate_mask=torch.zeros(size, 1, dtype=torch.bool),
        t1_target=take("target", "int64"),
        t2_target=torch.zeros(size, 1, dtype=torch.float32),
        t3_gains=torch.zeros(size, 1, dtype=torch.float32),
        t1_present=torch.ones(size, dtype=torch.bool),
        t2_present=torch.zeros(size, dtype=torch.bool),
        t3_present=torch.zeros(size, dtype=torch.bool),
    )


def train(
    split: Split,
    validation: Split,
    *,
    config: SessionGRUConfig,
    seed: int,
    rows: np.ndarray,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    label: str,
    schedule: str | None = None,
) -> tuple:
    """Train to the lowest validation loss and return that checkpoint, not the last.

    Scoring the last epoch punishes whichever variant overfits first, which is always the
    larger one - so a comparison between variants would measure capacity rather than
    usefulness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(seed, batch_spec=SPEC, config=config).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Lowering a constant learning rate kept helping - 0.002 to 0.0003 bought +0.008 - and
    # that is the signature of a model that is still undertrained rather than of a lucky
    # constant. A schedule is the principled form of the same fix: high early so it moves,
    # low late so it settles, without hand-tuning a constant downward forever.
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max_epochs)
        if schedule == "cosine"
        else None
    )
    loss_function = nn.CrossEntropyLoss()
    generator = np.random.default_rng(seed)

    validation_rows = np.arange(len(validation))
    best_loss, best_epoch, best_state, curve = float("inf"), 0, None, []
    print(f"  {label}  |  {parameter_count(model):,} parameters  |  {len(rows):,} decisions")

    for epoch in range(1, max_epochs + 1):
        started = time.time()
        model.train()
        order = rows.copy()
        generator.shuffle(order)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = to_batch(split, np.sort(order[start : start + batch_size])).to(DEVICE)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_function(model(batch).t1_logits, batch.t1_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.item() * batch.batch_size
            seen += batch.batch_size

        model.eval()
        validation_total, validation_seen, correct = 0.0, 0, 0
        with torch.no_grad():
            for start in range(0, len(validation_rows), 2048):
                batch = to_batch(validation, validation_rows[start : start + 2048]).to(DEVICE)
                logits = model(batch).t1_logits
                validation_total += loss_function(logits, batch.t1_target).item() * batch.batch_size
                correct += int((logits.argmax(dim=1) == batch.t1_target).sum())
                validation_seen += batch.batch_size

        if scheduler is not None:
            scheduler.step()
        train_loss = total / seen
        validation_loss = validation_total / validation_seen
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(validation_loss, 5),
            "gap": round(validation_loss - train_loss, 5),
            "val_accuracy": round(correct / validation_seen, 5),
            "seconds": round(time.time() - started, 1),
        }
        curve.append(row)

        marker = ""
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(
            f"    epoch {epoch:>2}  train {train_loss:.4f}  val {validation_loss:.4f}  "
            f"gap {row['gap']:+.4f}  acc@1 {row['val_accuracy']:.4f}  "
            f"({row['seconds']:.0f}s){marker}"
        )

        # The gap widening while validation loss rises is memorisation, not learning.
        if epoch - best_epoch >= patience:
            print(f"    stopping: no improvement for {patience} epochs")
            break

    model.load_state_dict(best_state)
    print(f"    scoring epoch {best_epoch} (lowest validation loss)")
    return model, curve, best_epoch


def score(model, validation: Split, *, legacy_clamp: bool, popularity: np.ndarray) -> dict:
    """Rank the truth for every validation decision, overall and on the slice."""
    model.eval()
    current = validation.current_category
    truth = validation.data["target"]
    changed = current != truth
    clients = validation.data["client"]

    overall_ranks = np.zeros(len(validation), dtype="int64")
    slice_ranks = np.zeros(len(validation), dtype="int64")
    top1 = np.zeros(len(validation), dtype="int64")
    top5 = np.zeros((len(validation), 5), dtype="int64")
    score_sum = np.zeros(CATEGORIES, dtype="float64")

    with torch.no_grad():
        for start in range(0, len(validation), 2048):
            rows = np.arange(start, min(start + 2048, len(validation)))
            batch = to_batch(validation, rows).to(DEVICE)
            logits = model(batch).t1_logits.float().cpu().numpy()
            target = truth[rows]
            overall_ranks[rows] = rank_of_truth(logits, target)
            slice_ranks[rows] = rank_of_truth(
                logits, target, suppress=current[rows], legacy_clamp=legacy_clamp
            )
            top1[rows] = logits.argmax(axis=1)
            top5[rows] = np.argpartition(-logits, kth=4, axis=1)[:, :5]
            score_sum += logits.sum(axis=0)

    overall = reciprocal_rank(overall_ranks)
    sliced = reciprocal_rank(slice_ranks)
    micro_all, macro_all = micro_and_macro(overall, clients)
    micro_slice, macro_slice = micro_and_macro(sliced, clients, changed)

    train_events = pd.Series(1, index=validation.data["client"]).groupby(level=0).sum()
    return {
        "overall_micro": round(micro_all, 4),
        "overall_macro": round(macro_all, 4),
        "slice_micro": round(micro_slice, 4),
        "slice_macro": round(macro_slice, 4),
        "overall_accuracy@1": round(float((overall_ranks == 1).mean()), 4),
        "slice_accuracy@1": round(float((slice_ranks[changed] == 1).mean()), 4),
        "_overall_values": overall,
        "_slice_values": sliced,
        "_changed": changed,
        "_top1": top1,
        "_top5": top5,
        "_score_sum": score_sum,
        "_train_events": train_events,
    }
